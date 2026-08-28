from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import stat
import statistics
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/confirm_unicom_full_width_objective.py"


def _load_script():
    if not SCRIPT.is_file():
        pytest.fail("full-width confirmation producer is absent")
    spec = importlib.util.spec_from_file_location(
        "confirm_unicom_full_width_objective", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_confirmation_producer_freezes_the_registered_seed_order() -> None:
    module = _load_script()

    assert module.CONFIRMATION_SEEDS == (2, 3, 4, 5, 6)
    assert module.ARMS == ("sampled_512", "full_768")
    assert module.EPOCHS == (4, 8, 12, 16)


def test_pooled_query_bootstrap_uses_one_shared_draw_across_seeds() -> None:
    module = _load_script()
    control = np.asarray(
        [[0.0, 0.1, 0.3, 0.4], [0.2, 0.1, 0.2, 0.5]] * 2
        + [[0.1, 0.2, 0.4, 0.3]],
        dtype=np.float64,
    )
    candidate = control + np.asarray(
        [[0.1, 0.0, 0.2, -0.1], [0.0, 0.1, 0.1, 0.0]] * 2
        + [[0.2, -0.1, 0.0, 0.1]],
        dtype=np.float64,
    )

    observed = module.pooled_query_bootstrap(control, candidate)

    per_query = (candidate - control).mean(axis=0, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(768))
    draws = generator.integers(0, 4, size=(10_000, 4))
    expected = tuple(
        float(value)
        for value in np.percentile(per_query[draws].mean(axis=1), (2.5, 97.5))
    )
    assert observed == expected


def _confirmation_rows(delta: float = 0.004) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "seed": seed,
            "control_epoch16_primary": 0.91,
            "candidate_primary_by_epoch": {
                4: 0.84,
                8: 0.89,
                12: 0.91 + delta,
                16: 0.91 + delta,
            },
            "control_top1_count": 758,
            "candidate_top1_count": 758,
        }
        for seed in (2, 3, 4, 5, 6)
    )


def test_quality_summary_is_supported_without_unmeasured_operational_inputs() -> None:
    module = _load_script()
    control = np.zeros((5, 4), dtype=np.float64)
    candidate = np.full((5, 4), 0.01, dtype=np.float64)

    result = module.build_quality_summary(
        _confirmation_rows(),
        control_average_precision=control,
        candidate_average_precision=candidate,
    )

    assert tuple(result) == (
        "primary_map_deltas",
        "mean_primary_map_delta",
        "paired_t_interval",
        "positive_seed_count",
        "top1_losses",
        "epoch12_reach_count",
        "query_bootstrap_95",
        "predicates",
        "status",
    )
    assert result["primary_map_deltas"] == pytest.approx([0.004] * 5)
    assert result["paired_t_interval"] == pytest.approx([0.004, 0.004])
    assert result["query_bootstrap_95"]["pooled"] == pytest.approx([0.01, 0.01])
    assert [row["seed"] for row in result["query_bootstrap_95"]["per_seed"]] == [
        2,
        3,
        4,
        5,
        6,
    ]
    assert all(result["predicates"].values())
    assert result["status"] == "SUPPORTED_HOLDOUT_QUALITY"


def _seed_bundle(seed: int, delta: float = 0.004):
    rows = []
    inventory = []
    receipts = {
        arm: {
            "seed": seed,
            "arm": arm,
            "peak_allocated_bytes": 1000 if arm == "sampled_512" else 1001,
            "peak_reserved_bytes": 2000 if arm == "sampled_512" else 2001,
            "checkpoints": [],
        }
        for arm in ("sampled_512", "full_768")
    }
    trajectory = {4: 0.84, 8: 0.89, 12: 0.91 + delta, 16: 0.91 + delta}
    for epoch in (4, 8, 12, 16):
        arms = {}
        for arm_index, arm in enumerate(("sampled_512", "full_768")):
            value = 0.91 if arm == "sampled_512" and epoch == 16 else (
                0.85 if arm == "sampled_512" else trajectory[epoch]
            )
            digest = f"{seed + arm_index + epoch:x}"[-1] * 64
            checkpoint = {
                "epoch": epoch,
                "path": f"/seed-{seed}/{arm}/epoch-{epoch:04d}.pt",
                "sha256": digest,
                "bytes": 100 + epoch,
            }
            receipts[arm]["checkpoints"].append(checkpoint)
            inventory.append({"arm": arm, **checkpoint})
            arms[arm] = {
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_bytes": checkpoint["bytes"],
                "primary": {
                    "map_at_r": value,
                    "average_precision": [value] * 4,
                    "top1_correct": [True, True, True, True],
                },
            }
        rows.append(
            {
                "epoch": epoch,
                "query_ids_sha256": "a" * 64,
                "gallery_ids_sha256": "b" * 64,
                "arms": arms,
            }
        )
    return {"seed": seed, "rows": rows}, {"seed": seed, "inventory": inventory}, receipts


def test_seed_extraction_cross_binds_pair_rows_receipts_and_query_evidence() -> None:
    module = _load_script()
    pair_result, _inventory, receipts = _seed_bundle(2)

    extracted = module.extract_seed_evidence(
        seed=2,
        pair_result=pair_result,
        control_receipt=receipts["sampled_512"],
        candidate_receipt=receipts["full_768"],
    )

    assert extracted["row"]["seed"] == 2
    assert extracted["row"]["control_epoch16_primary"] == 0.91
    assert extracted["control_average_precision"].dtype == np.float64
    assert extracted["candidate_average_precision"].tolist() == [0.914] * 4
    assert extracted["resource"] == {
        "seed": 2,
        "peak_allocated_ratio": 1.001,
        "peak_reserved_ratio": 1.0005,
        "checkpoint_bytes_equal": True,
        "control_checkpoint_bytes": [104, 108, 112, 116],
        "candidate_checkpoint_bytes": [104, 108, 112, 116],
    }

    changed = copy.deepcopy(pair_result)
    changed["rows"][0]["arms"]["full_768"]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint authority"):
        module.extract_seed_evidence(
            seed=2,
            pair_result=changed,
            control_receipt=receipts["sampled_512"],
            candidate_receipt=receipts["full_768"],
        )


def test_seed_bundle_authentication_rejects_inventory_digest_drift() -> None:
    module = _load_script()
    pair_result, inventory, receipts = _seed_bundle(2)
    source_commit = "a" * 40
    config_sha256 = "b" * 64
    runtime = {"python": "3.12.3", "torch": "2.12.1", "cuda": "13.0"}
    config = {
        "training_receipt_authority": {
            "source_commit": "d" * 40,
            "config_commit": "e" * 40,
            "config_sha256": "f" * 64,
        },
        "confirmation_receipt_authority": {
            "source_commit": source_commit,
            "config_commit": "c" * 40,
            "config_sha256": config_sha256,
        },
        "environment": {**runtime, "numpy": "2.5.0"},
    }
    for receipt in receipts.values():
        receipt["source_commit"] = source_commit
        receipt["config_sha256"] = config_sha256
        receipt["runtime"] = runtime

    module.cross_bind_seed_bundle(
        config=config,
        seed=2,
        pair_inventory=inventory,
        pair_result=pair_result,
        control_receipt=receipts["sampled_512"],
        candidate_receipt=receipts["full_768"],
    )

    changed = copy.deepcopy(inventory)
    changed["inventory"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint authority"):
        module.cross_bind_seed_bundle(
            config=config,
            seed=2,
            pair_inventory=changed,
            pair_result=pair_result,
            control_receipt=receipts["sampled_512"],
            candidate_receipt=receipts["full_768"],
        )


def test_operational_summary_does_not_promote_single_seed_abba_to_five_seed_mean() -> None:
    module = _load_script()
    resources = []
    for seed in (2, 3, 4, 5, 6):
        pair_result, _inventory, receipts = _seed_bundle(seed)
        resources.append(
            module.extract_seed_evidence(
                seed=seed,
                pair_result=pair_result,
                control_receipt=receipts["sampled_512"],
                candidate_receipt=receipts["full_768"],
            )["resource"]
        )
    seed0_decision = {
        "status": "PROMOTE_CONFIRMATION",
        "evidence": {
            "abba_step_time_ratio": 0.995,
            "observed_cuda_step_ratio": 0.996,
        },
    }
    profile = {
        "ratios": {"step_wall": 0.995, "cuda_step": 0.996},
        "ratio_bootstrap_95": {
            "step_wall": [0.991, 0.999],
            "cuda_step": [0.992, 1.0],
        },
    }

    result = module.build_operational_summary(
        seed0_decision=seed0_decision,
        seed0_profile_comparison=profile,
        confirmation_resources=tuple(resources),
    )

    assert result["step_time"] == {
        "authority_seed": 0,
        "scope": "single_seed_terminal_checkpoint_abba_screen",
        "claim": "no_measurable_slowdown_gate_only",
        "metric": "step_wall",
        "ratio": 0.995,
        "bootstrap_95": [0.991, 0.999],
    }
    assert result["missing_evidence"] == {
        "confirmation_seed_abba_profiles": [2, 3, 4, 5, 6],
        "empirical_deployment_parameter_comparison": True,
        "empirical_inference_operation_comparison": True,
        "empirical_deployment_storage_comparison": True,
    }
    assert result["predicates"]["registered_mean_abba_available"] is False
    assert result["status"] == "INCOMPLETE_OPERATIONAL_EVIDENCE"

    changed = copy.deepcopy(profile)
    changed["ratios"]["step_wall"] = 0.9
    with pytest.raises(ValueError, match="seed-0 step-time authority"):
        module.build_operational_summary(
            seed0_decision=seed0_decision,
            seed0_profile_comparison=changed,
            confirmation_resources=tuple(resources),
        )


def _binding(name: str, index: int) -> dict[str, object]:
    return {
        "path": f"/{name}.json",
        "sha256": f"{index:x}"[-1] * 64,
        "bytes": 100 + index,
    }


def _artifact_inputs() -> dict[str, object]:
    rows = []
    for index, seed in enumerate((2, 3, 4, 5, 6), start=3):
        rows.append(
            {
                "seed": seed,
                "pair_inventory": _binding(f"seed-{seed}-inventory", index),
                "pair_result": _binding(f"seed-{seed}-result", index + 1),
                "control_receipt": _binding(f"seed-{seed}-control", index + 2),
                "candidate_receipt": _binding(f"seed-{seed}-candidate", index + 3),
            }
        )
    return {
        "run_config": _binding("run-config", 1),
        "seed0_decision": _binding("seed0-decision", 2),
        "seed0_profile_comparison": _binding("seed0-profile", 3),
        "confirmation_seeds": rows,
    }


def _artifact_sections(module):
    control = np.zeros((5, 4), dtype=np.float64)
    candidate = np.full((5, 4), 0.01, dtype=np.float64)
    quality = module.build_quality_summary(
        _confirmation_rows(),
        control_average_precision=control,
        candidate_average_precision=candidate,
    )
    resources = []
    for seed in (2, 3, 4, 5, 6):
        pair_result, _inventory, receipts = _seed_bundle(seed)
        resources.append(
            module.extract_seed_evidence(
                seed=seed,
                pair_result=pair_result,
                control_receipt=receipts["sampled_512"],
                candidate_receipt=receipts["full_768"],
            )["resource"]
        )
    operational = module.build_operational_summary(
        seed0_decision={
            "status": "PROMOTE_CONFIRMATION",
            "evidence": {
                "abba_step_time_ratio": 0.995,
                "observed_cuda_step_ratio": 0.996,
            },
        },
        seed0_profile_comparison={
            "ratios": {"step_wall": 0.995, "cuda_step": 0.996},
            "ratio_bootstrap_95": {
                "step_wall": [0.991, 0.999],
                "cuda_step": [0.992, 1.0],
            },
        },
        confirmation_resources=tuple(resources),
    )
    return quality, operational


def test_confirmation_artifact_is_exact_and_recomputed_from_bound_inputs() -> None:
    module = _load_script()
    quality, operational = _artifact_sections(module)
    result = module.build_confirmation_result(
        inputs=_artifact_inputs(), quality=quality, operational=operational
    )

    module.validate_confirmation_result(result, expected=copy.deepcopy(result))
    assert tuple(result) == (
        "schema_version",
        "inputs",
        "metric_authority",
        "quality",
        "operational",
        "decision",
        "status",
    )
    assert result["decision"] == {
        "quality": "SUPPORTED_HOLDOUT_QUALITY",
        "registered_confirmation": "INCOMPLETE_OPERATIONAL_EVIDENCE",
        "first_decisive_clause": "missing_confirmation_seed_abba_and_deployment_measurements",
    }
    assert result["status"] == "INCOMPLETE_OPERATIONAL_EVIDENCE"

    measured_failure = copy.deepcopy(operational)
    measured_failure["memory_and_checkpoint_rows"][0][
        "peak_allocated_ratio"
    ] = 1.2
    measured_failure["mean_peak_allocated_ratio"] = statistics.fmean(
        row["peak_allocated_ratio"]
        for row in measured_failure["memory_and_checkpoint_rows"]
    )
    measured_failure["predicates"]["mean_peak_allocated_ratio_at_most_1_02"] = False
    measured_failure["status"] = "CLOSE_FULL_WIDTH"
    closed = module.build_confirmation_result(
        inputs=_artifact_inputs(),
        quality=quality,
        operational=measured_failure,
    )
    assert closed["decision"] == {
        "quality": "SUPPORTED_HOLDOUT_QUALITY",
        "registered_confirmation": "CLOSE_FULL_WIDTH",
        "first_decisive_clause": "measured_confirmation_cost_predicates_failed",
    }
    assert closed["status"] == "CLOSE_FULL_WIDTH"

    for mutate in (
        lambda value: value["inputs"]["confirmation_seeds"][2]["pair_result"].__setitem__(
            "sha256", "f" * 64
        ),
        lambda value: value["quality"].__setitem__("mean_primary_map_delta", 0.5),
        lambda value: value["operational"]["step_time"].__setitem__(
            "claim", "speedup"
        ),
        lambda value: value["decision"].__setitem__(
            "registered_confirmation", "SUPPORTED_HOLDOUT"
        ),
    ):
        changed = copy.deepcopy(result)
        mutate(changed)
        with pytest.raises(ValueError):
            module.validate_confirmation_result(changed, expected=result)

    relational_drift = copy.deepcopy(result)
    relational_drift["quality"]["mean_primary_map_delta"] = 0.5
    with pytest.raises(ValueError, match="quality relation"):
        module.validate_confirmation_result(relational_drift)

    bad_interval = copy.deepcopy(result)
    bad_interval["operational"]["step_time"]["bootstrap_95"] = [1.1, 0.9]
    with pytest.raises(ValueError, match="step-time interval"):
        module.validate_confirmation_result(bad_interval)


def test_main_publishes_once_strictly_and_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    quality, operational = _artifact_sections(module)
    expected = module.build_confirmation_result(
        inputs=_artifact_inputs(), quality=quality, operational=operational
    )
    config = tmp_path / "run.json"
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    output = evidence_root / "confirmation-result-v3.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "unicom-full-width-objective-run-v3",
                "paths": {"output_root": str(evidence_root)},
                "registered_outputs": {"confirmation_result_v3": str(output)},
                "command_templates": {"confirmation_command": ["python"]},
                "confirmation_audit_inputs": {
                    "seed0_decision": _binding("seed0-decision", 2),
                    "seed0_profile_comparison": _binding("seed0-profile", 3),
                    "confirmation_seeds": _artifact_inputs()["confirmation_seeds"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def authenticate(run_config: Path, checkout: Path) -> str:
        assert run_config == config
        assert checkout == SCRIPT.parents[1]
        calls.append("authenticate")
        return "a" * 40

    def build(run_config: Path, root: Path, destination: Path):
        assert (run_config, root, destination) == (config, evidence_root, output)
        calls.append("build")
        return copy.deepcopy(expected)

    monkeypatch.setattr(module, "authenticate_confirmation_handoff", authenticate)
    monkeypatch.setattr(module, "build_from_evidence", build)
    arguments = [
        "--run-config",
        str(config),
        "--evidence-root",
        str(evidence_root),
        "--output",
        str(output),
    ]

    assert module.main(arguments) == 0
    assert calls == ["authenticate", "build"]
    persisted = module.EVALUATOR.strict_json_object(output)
    module.validate_confirmation_result(persisted, expected=expected)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    before = output.read_bytes()
    assert module.main(arguments) == 2
    assert output.read_bytes() == before


def test_main_rejects_an_unregistered_config_before_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    config = tmp_path / "run.json"
    config.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "evidence"
    root.mkdir()
    output = root / "confirmation-result-v3.json"
    calls = []
    monkeypatch.setattr(
        module,
        "authenticate_confirmation_handoff",
        lambda _config, _checkout: calls.append("authenticate") or "a" * 40,
    )
    monkeypatch.setattr(
        module,
        "build_from_evidence",
        lambda _config, _root, _output: calls.append("build"),
    )

    assert module.main(
        [
            "--run-config",
            str(config),
            "--evidence-root",
            str(root),
            "--output",
            str(output),
        ]
    ) == 2
    assert calls == ["authenticate"]
    assert not output.exists()


def test_run_config_binds_confirmation_command_and_audited_inputs() -> None:
    module = _load_script()
    arguments = [
        "--run-config",
        "/checkout/docs/run.json",
        "--evidence-root",
        "/evidence",
        "--output",
        "/evidence/confirmation-result-v3.json",
    ]
    command = ["/python", "-I", "-B", "/launcher.py", "/confirm.py", *arguments]
    config = {
        "schema_version": "unicom-full-width-objective-run-v3",
        "paths": {"output_root": "/evidence"},
        "registered_outputs": {
            "confirmation_result_v3": "/evidence/confirmation-result-v3.json"
        },
        "command_templates": {"confirmation_command": command},
        "confirmation_audit_inputs": {
            "seed0_decision": _binding("seed0-decision", 2),
            "seed0_profile_comparison": _binding("seed0-profile", 3),
            "confirmation_seeds": _artifact_inputs()["confirmation_seeds"],
        },
    }
    args = module.parse_args(arguments)

    module._validate_run_config(config, args, observed_command=command)

    changed = copy.deepcopy(config)
    changed["command_templates"]["confirmation_command"][-1] = "/other.json"
    with pytest.raises(ValueError, match="confirmation command"):
        module._validate_run_config(changed, args, observed_command=command)


def test_main_without_arguments_binds_the_launcher_original_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    quality, operational = _artifact_sections(module)
    expected = module.build_confirmation_result(
        inputs=_artifact_inputs(), quality=quality, operational=operational
    )
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    config_path = tmp_path / "run.json"
    output = evidence_root / "confirmation-result-v3.json"
    arguments = [
        "--run-config",
        str(config_path),
        "--evidence-root",
        str(evidence_root),
        "--output",
        str(output),
    ]
    command = [
        "/python",
        "-I",
        "-B",
        "/launcher.py",
        str(SCRIPT),
        *arguments,
    ]
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "unicom-full-width-objective-run-v3",
                "paths": {"output_root": str(evidence_root)},
                "registered_outputs": {"confirmation_result_v3": str(output)},
                "command_templates": {"confirmation_command": command},
                "confirmation_audit_inputs": {
                    "seed0_decision": _binding("seed0-decision", 2),
                    "seed0_profile_comparison": _binding("seed0-profile", 3),
                    "confirmation_seeds": _artifact_inputs()["confirmation_seeds"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "authenticate_confirmation_handoff",
        lambda _config, _checkout: calls.append("authenticate") or "a" * 40,
    )
    monkeypatch.setattr(
        module,
        "build_from_evidence",
        lambda _config, _root, _output: calls.append("build")
        or copy.deepcopy(expected),
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), *arguments])
    monkeypatch.setattr(sys, "orig_argv", command)

    assert module.main() == 0
    assert calls == ["authenticate", "build"]
    persisted = module.EVALUATOR.strict_json_object(output)
    module.validate_confirmation_result(persisted, expected=expected)


def _write_json(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def test_bound_json_is_hashed_and_strictly_parsed_from_one_read(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "evidence"
    path = root / "input.json"
    binding = _write_json(path, {"value": 1})

    assert module.load_bound_json(binding, root) == {"value": 1}

    path.write_text('{"value":1,"value":2}\n', encoding="utf-8")
    payload = path.read_bytes()
    changed_binding = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    with pytest.raises(ValueError, match="keys"):
        module.load_bound_json(changed_binding, root)


def test_build_from_evidence_recomputes_all_five_bound_seed_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    root = tmp_path / "evidence"
    root.mkdir()
    source_commit = "a" * 40
    training_config_sha = "b" * 64
    runtime = {"python": "3.12.3", "torch": "2.12.1", "cuda": "13.0"}
    profile = {
        "ratios": {"step_wall": 0.995, "cuda_step": 0.996},
        "ratio_bootstrap_95": {
            "step_wall": [0.991, 0.999],
            "cuda_step": [0.992, 1.0],
        },
    }
    profile_path = root / "seed-0" / "profile-comparison.json"
    profile_binding = _write_json(profile_path, profile)
    decision = {
        "status": "PROMOTE_CONFIRMATION",
        "evidence": {
            "abba_step_time_ratio": 0.995,
            "observed_cuda_step_ratio": 0.996,
        },
        "inputs": {"profile_comparison": profile_binding},
    }
    decision_binding = _write_json(root / "seed-0" / "decision.json", decision)
    audit_rows = []
    for seed in (2, 3, 4, 5, 6):
        pair_result, inventory, receipts = _seed_bundle(seed)
        for receipt in receipts.values():
            receipt["source_commit"] = source_commit
            receipt["config_sha256"] = training_config_sha
            receipt["runtime"] = runtime
        seed_root = root / f"seed-{seed}"
        audit_rows.append(
            {
                "seed": seed,
                "pair_inventory": _write_json(seed_root / "pair-inventory.json", inventory),
                "pair_result": _write_json(seed_root / "paired-result.json", pair_result),
                "control_receipt": _write_json(
                    seed_root / "sampled_512-run-receipt.json",
                    receipts["sampled_512"],
                ),
                "candidate_receipt": _write_json(
                    seed_root / "full_768-run-receipt.json", receipts["full_768"]
                ),
            }
        )
    output = root / "confirmation-result-v3.json"
    config_path = tmp_path / "run.json"
    config = {
        "schema_version": "unicom-full-width-objective-run-v3",
        "paths": {"output_root": str(root)},
        "registered_outputs": {"confirmation_result_v3": str(output)},
        "command_templates": {"confirmation_command": ["python"]},
        "training_receipt_authority": {
            "source_commit": "d" * 40,
            "config_commit": "e" * 40,
            "config_sha256": "f" * 64,
        },
        "confirmation_receipt_authority": {
            "source_commit": source_commit,
            "config_commit": "c" * 40,
            "config_sha256": training_config_sha,
        },
        "environment": {**runtime, "numpy": "2.5.0"},
        "confirmation_audit_inputs": {
            "seed0_decision": decision_binding,
            "seed0_profile_comparison": profile_binding,
            "confirmation_seeds": audit_rows,
        },
    }
    _write_json(config_path, config)
    calls = []
    monkeypatch.setattr(
        module.DECIDER,
        "validate_seed0_decision",
        lambda _value: calls.append("decision"),
    )
    monkeypatch.setattr(
        module.COMPARATOR,
        "validate_comparison_result",
        lambda _value: calls.append("profile"),
    )
    monkeypatch.setattr(
        module.EVALUATOR,
        "validate_pair_result",
        lambda _result, _inventory: calls.append("pair"),
    )
    monkeypatch.setattr(
        module.TRAINER,
        "validate_training_run_receipt",
        lambda _value: calls.append("receipt"),
    )

    result = module.build_from_evidence(config_path, root, output)

    assert calls == ["decision", "profile"] + ["pair", "receipt", "receipt"] * 5
    assert result["quality"]["status"] == "SUPPORTED_HOLDOUT_QUALITY"
    assert result["status"] == "INCOMPLETE_OPERATIONAL_EVIDENCE"
    assert result["inputs"]["run_config"] == {
        "path": str(config_path),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "bytes": config_path.stat().st_size,
    }

    config["confirmation_audit_inputs"]["confirmation_seeds"][3]["pair_result"][
        "sha256"
    ] = "f" * 64
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="audit input bytes"):
        module.build_from_evidence(config_path, root, output)


def test_independent_authenticator_recomputes_persisted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    quality, operational = _artifact_sections(module)
    expected = module.build_confirmation_result(
        inputs=_artifact_inputs(), quality=quality, operational=operational
    )
    config = tmp_path / "run.json"
    config.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "evidence"
    root.mkdir()
    output = root / "confirmation-result-v3.json"
    output.write_text(json.dumps(expected) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "build_from_evidence",
        lambda _config, _root, _output: copy.deepcopy(expected),
    )

    assert module.authenticate_persisted_result(config, root, output) == expected

    changed = copy.deepcopy(expected)
    changed["quality"]["mean_primary_map_delta"] = 0.5
    output.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="quality relation"):
        module.authenticate_persisted_result(config, root, output)
