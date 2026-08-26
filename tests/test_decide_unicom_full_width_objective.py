from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/decide_unicom_full_width_objective.py"


def _load_script():
    if not SCRIPT.is_file():
        pytest.fail("full-width seed-0 decision producer is absent")
    spec = importlib.util.spec_from_file_location(
        "decide_unicom_full_width_objective", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_step_wall_is_the_only_registered_seed0_timing_authority() -> None:
    module = _load_script()

    assert module.STEP_TIME_METRIC == "step_wall"


def test_registered_command_key_does_not_collide_with_outcome_guard() -> None:
    module = _load_script()
    arguments = [
        "--run-config",
        "/run-config.json",
        "--pair-inventory",
        "/pair-inventory.json",
        "--pair-result",
        "/pair-result.json",
        "--profile-comparison",
        "/profile.json",
        "--control-receipt",
        "/control.json",
        "--candidate-receipt",
        "/candidate.json",
        "--output",
        "/decision.json",
    ]
    parsed = module.parse_args(arguments)
    command = ["/python", "-I", "-B", "/launcher.py", "/decision-script.py", *arguments]
    config = {
        "schema_version": "unicom-full-width-objective-run-v2",
        "thresholds": {
            "operational": {
                "step_time_metric": "step_wall",
                "step_time_ratio": 1.02,
                "peak_allocated_ratio": 1.02,
                "peak_reserved_ratio": 1.02,
                "checkpoint_bytes_equal": True,
            }
        },
        "seed0_downstream": {
            "decision_path": "/decision.json",
            "pair_inventory": {"path": "/pair-inventory.json"},
            "pair_result": "/pair-result.json",
            "profile_comparison": "/profile.json",
        },
        "run_schedule": [
            {
                "seed": 0,
                "runs": [
                    {"receipt": "/control.json"},
                    {"receipt": "/candidate.json"},
                ],
            }
        ],
        "command_templates": {"decision_command": command},
    }

    module._validate_run_config(config, parsed, observed_command=command)


def test_seed0_decision_uses_wall_time_and_cross_binds_cost_evidence() -> None:
    module = _load_script()
    pair = {
        "rows": [
            {
                "epoch": epoch,
                "arms": {
                    "sampled_512": {
                        "checkpoint_bytes": 100 + epoch,
                        "primary": {
                            "map_at_r": 0.90 if epoch < 16 else 0.91,
                            "top1_correct": [True, True, True, False],
                        },
                    },
                    "full_768": {
                        "checkpoint_bytes": 100 + epoch,
                        "primary": {
                            "map_at_r": {4: 0.88, 8: 0.90, 12: 0.915, 16: 0.916}[
                                epoch
                            ],
                            "top1_correct": [True, True, True, False],
                        },
                    },
                },
            }
            for epoch in (4, 8, 12, 16)
        ]
    }
    profiles = {
        "ratios": {
            "step_wall": 1.01,
            "cuda_step": 1.50,
        },
        "checkpoint_bytes_equal": True,
    }
    control = {
        "arm": "sampled_512",
        "peak_allocated_bytes": 1000,
        "peak_reserved_bytes": 2000,
        "checkpoints": [
            {"epoch": epoch, "bytes": 100 + epoch} for epoch in (4, 8, 12, 16)
        ],
    }
    candidate = {
        "arm": "full_768",
        "peak_allocated_bytes": 1010,
        "peak_reserved_bytes": 2020,
        "checkpoints": [
            {"epoch": epoch, "bytes": 100 + epoch} for epoch in (4, 8, 12, 16)
        ],
    }

    result = module.build_seed0_decision(
        pair_result=pair,
        profile_comparison=profiles,
        control_receipt=control,
        candidate_receipt=candidate,
    )

    assert result["metric_authority"]["step_time"] == "step_wall"
    assert result["evidence"]["abba_step_time_ratio"] == 1.01
    assert result["evidence"]["observed_cuda_step_ratio"] == 1.50
    assert result["decision"]["operational_passed"] is True
    assert result["status"] == "PROMOTE_CONFIRMATION"
    module.validate_seed0_decision(result)

    for mutate in (
        lambda value: value["metric_authority"].__setitem__("step_time", "cuda_step"),
        lambda value: value["evidence"].__setitem__("abba_step_time_ratio", 1.03),
        lambda value: value["evidence"]["candidate_primary_by_epoch"][2].__setitem__(
            "map_at_r", 0.80
        ),
        lambda value: value.__setitem__("status", "CLOSE_RESOURCE"),
        lambda value: value["evidence"]["candidate_checkpoint_bytes"].__setitem__(
            0, 999
        ),
    ):
        changed = copy.deepcopy(result)
        mutate(changed)
        with pytest.raises((TypeError, ValueError)):
            module.validate_seed0_decision(changed)


def test_seed0_decision_rejects_checkpoint_or_arm_drift() -> None:
    module = _load_script()
    with pytest.raises((TypeError, ValueError)):
        module.build_seed0_decision(
            pair_result={"rows": []},
            profile_comparison={"ratios": {}},
            control_receipt={"arm": "full_768"},
            candidate_receipt={"arm": "full_768"},
        )


def test_main_publishes_once_after_invoking_all_input_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "decision.json"
    pair_inventory = tmp_path / "pair-inventory.json"
    pair_result = tmp_path / "pair-result.json"
    profile = tmp_path / "profile.json"
    control_path = tmp_path / "control.json"
    candidate_path = tmp_path / "candidate.json"
    config_path = tmp_path / "run-config.json"
    checkpoint_rows = [
        {"epoch": epoch, "bytes": 100 + epoch} for epoch in (4, 8, 12, 16)
    ]
    control = {
        "arm": "sampled_512",
        "peak_allocated_bytes": 1000,
        "peak_reserved_bytes": 2000,
        "checkpoints": checkpoint_rows,
    }
    candidate = {
        "arm": "full_768",
        "peak_allocated_bytes": 1010,
        "peak_reserved_bytes": 2020,
        "checkpoints": checkpoint_rows,
    }
    pair = {
        "rows": [
            {
                "epoch": epoch,
                "arms": {
                    "sampled_512": {
                        "checkpoint_bytes": 100 + epoch,
                        "primary": {
                            "map_at_r": 0.90 if epoch < 16 else 0.91,
                            "top1_correct": [True, True, True, False],
                        },
                    },
                    "full_768": {
                        "checkpoint_bytes": 100 + epoch,
                        "primary": {
                            "map_at_r": {4: 0.88, 8: 0.90, 12: 0.915, 16: 0.916}[
                                epoch
                            ],
                            "top1_correct": [True, True, True, False],
                        },
                    },
                },
            }
            for epoch in (4, 8, 12, 16)
        ]
    }

    def write(path: Path, value: object) -> None:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    write(control_path, control)
    write(candidate_path, candidate)
    control_sha = hashlib.sha256(control_path.read_bytes()).hexdigest()
    candidate_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    comparison = {
        "ratios": {"step_wall": 1.01, "cuda_step": 1.50},
        "checkpoint_bytes_equal": True,
        "receipt_sha256s": [control_sha, candidate_sha, candidate_sha, control_sha],
    }
    write(pair_inventory, {})
    write(pair_result, pair)
    write(profile, comparison)
    config = {
        "schema_version": "unicom-full-width-objective-run-v2",
        "environment": {
            "python": "3.13.9",
            "torch": "2.12.1+cu130",
            "cuda": "13.0",
        },
        "thresholds": {
            "operational": {
                "step_time_metric": "step_wall",
                "step_time_ratio": 1.02,
                "peak_allocated_ratio": 1.02,
                "peak_reserved_ratio": 1.02,
                "checkpoint_bytes_equal": True,
            }
        },
        "seed0_downstream": {
            "pair_inventory": {"path": str(pair_inventory)},
            "profile_comparison": str(profile),
            "pair_result": str(pair_result),
            "decision_path": str(output),
        },
        "run_schedule": [
            {
                "seed": 0,
                "runs": [
                    {"receipt": str(control_path)},
                    {"receipt": str(candidate_path)},
                ],
            }
        ],
    }
    write(config_path, config)
    calls: list[str] = []
    monkeypatch.setattr(
        module.EVALUATOR,
        "validate_pair_result",
        lambda _result, _inventory: calls.append("pair"),
    )
    monkeypatch.setattr(
        module.COMPARATOR,
        "validate_comparison_result",
        lambda _value: calls.append("comparison"),
    )
    monkeypatch.setattr(
        module.TRAINER,
        "validate_training_run_receipt",
        lambda _value: calls.append("receipt"),
    )
    monkeypatch.setattr(
        module,
        "_cross_bind_inputs",
        lambda **_kwargs: calls.append("cross"),
    )
    arguments = [
        "--run-config",
        str(config_path),
        "--pair-inventory",
        str(pair_inventory),
        "--pair-result",
        str(pair_result),
        "--profile-comparison",
        str(profile),
        "--control-receipt",
        str(control_path),
        "--candidate-receipt",
        str(candidate_path),
        "--output",
        str(output),
    ]

    assert module.main(arguments) == 0
    assert calls == ["pair", "comparison", "receipt", "receipt", "cross"]
    before = output.read_bytes()
    persisted = module.EVALUATOR.strict_json_object(output)
    module.validate_seed0_decision(persisted)
    assert module.main(arguments) == 2
    assert output.read_bytes() == before


def test_cross_binding_uses_frozen_training_authority_after_profile_repair() -> None:
    module = _load_script()
    run_config_sha = "a" * 64
    training_config_sha = "c" * 64
    training_source_commit = "b" * 40
    repaired_source_commit = "d" * 40
    runtime = {"python": "3.13.9", "torch": "2.12.1+cu130", "cuda": "13.0"}
    config = {
        "source": {"commit": repaired_source_commit},
        "training_receipt_authority": {
            "source_commit": training_source_commit,
            "config_commit": "e" * 40,
            "config_sha256": training_config_sha,
        },
        "environment": {**runtime, "numpy": "2.5.0"},
    }

    def receipt(arm: str, prefix: str) -> dict[str, object]:
        return {
            "seed": 0,
            "arm": arm,
            "source_commit": training_source_commit,
            "config_sha256": training_config_sha,
            "runtime": runtime,
            "checkpoints": [
                {
                    "epoch": epoch,
                    "path": f"/{arm}/epoch-{epoch:04d}.pt",
                    "sha256": prefix * 64,
                    "bytes": 100 + epoch,
                }
                for epoch in (4, 8, 12, 16)
            ],
        }

    control = receipt("sampled_512", "c")
    candidate = receipt("full_768", "d")
    inventory = []
    rows = []
    for index, epoch in enumerate((4, 8, 12, 16)):
        arms = {}
        for arm, source in (("sampled_512", control), ("full_768", candidate)):
            checkpoint = source["checkpoints"][index]
            inventory.append({"arm": arm, **checkpoint})
            arms[arm] = {
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_bytes": checkpoint["bytes"],
            }
        rows.append({"epoch": epoch, "arms": arms})
    profile = {
        "config_sha256": training_config_sha,
        "source_commit": training_source_commit,
    }
    arguments = {
        "config": config,
        "pair_inventory": {"seed": 0, "inventory": inventory},
        "pair_result": {"rows": rows},
        "profile_comparison": profile,
        "control_receipt": control,
        "candidate_receipt": candidate,
        "run_config_sha256": run_config_sha,
    }

    module._cross_bind_inputs(**arguments)
    for mutate in (
        lambda values: values["candidate_receipt"]["runtime"].__setitem__(
            "torch", "other"
        ),
        lambda values: values["pair_inventory"]["inventory"][0].__setitem__(
            "sha256", "e" * 64
        ),
        lambda values: values["profile_comparison"].__setitem__(
            "config_sha256", "f" * 64
        ),
        lambda values: values["config"]["training_receipt_authority"].__setitem__(
            "source_commit", "e" * 40
        ),
        lambda values: values["config"]["training_receipt_authority"].__setitem__(
            "config_sha256", "f" * 64
        ),
        lambda values: values["pair_inventory"].__setitem__("seed", 2),
    ):
        changed = copy.deepcopy(arguments)
        mutate(changed)
        with pytest.raises((TypeError, ValueError)):
            module._cross_bind_inputs(**changed)


def test_seed0_validator_rejects_a_delta_that_contradicts_its_trajectory() -> None:
    module = _load_script()
    pair = {
        "rows": [
            {
                "epoch": epoch,
                "arms": {
                    "sampled_512": {
                        "checkpoint_bytes": 100 + epoch,
                        "primary": {
                            "map_at_r": 0.90 if epoch < 16 else 0.91,
                            "top1_correct": [True, True, True, False],
                        },
                    },
                    "full_768": {
                        "checkpoint_bytes": 100 + epoch,
                        "primary": {
                            "map_at_r": {4: 0.88, 8: 0.90, 12: 0.915, 16: 0.916}[
                                epoch
                            ],
                            "top1_correct": [True, True, True, False],
                        },
                    },
                },
            }
            for epoch in (4, 8, 12, 16)
        ]
    }
    profiles = {
        "ratios": {"step_wall": 1.01, "cuda_step": 1.50},
        "checkpoint_bytes_equal": True,
    }
    control = {
        "arm": "sampled_512",
        "peak_allocated_bytes": 1000,
        "peak_reserved_bytes": 2000,
        "checkpoints": [
            {"epoch": epoch, "bytes": 100 + epoch} for epoch in (4, 8, 12, 16)
        ],
    }
    candidate = {
        "arm": "full_768",
        "peak_allocated_bytes": 1010,
        "peak_reserved_bytes": 2020,
        "checkpoints": [
            {"epoch": epoch, "bytes": 100 + epoch} for epoch in (4, 8, 12, 16)
        ],
    }
    result = module.build_seed0_decision(
        pair_result=pair,
        profile_comparison=profiles,
        control_receipt=control,
        candidate_receipt=candidate,
    )
    result["evidence"]["primary_map_delta"] += 0.001

    with pytest.raises(ValueError, match="primary mAP delta relation"):
        module.validate_seed0_decision(result)


def test_partial_epoch_checkpoint_size_drift_closes_resource() -> None:
    module = _load_script()
    rows = []
    control_checkpoints = []
    candidate_checkpoints = []
    for epoch in (4, 8, 12, 16):
        control_bytes = 100 + epoch
        candidate_bytes = control_bytes + (1 if epoch == 4 else -1 if epoch == 8 else 0)
        control_checkpoints.append({"epoch": epoch, "bytes": control_bytes})
        candidate_checkpoints.append({"epoch": epoch, "bytes": candidate_bytes})
        rows.append(
            {
                "epoch": epoch,
                "arms": {
                    "sampled_512": {
                        "checkpoint_bytes": control_bytes,
                        "primary": {
                            "map_at_r": 0.91,
                            "top1_correct": [True, True],
                        },
                    },
                    "full_768": {
                        "checkpoint_bytes": candidate_bytes,
                        "primary": {
                            "map_at_r": 0.916,
                            "top1_correct": [True, True],
                        },
                    },
                },
            }
        )

    result = module.build_seed0_decision(
        pair_result={"rows": rows},
        profile_comparison={
            "ratios": {"step_wall": 1.0, "cuda_step": 1.0},
            "checkpoint_bytes_equal": True,
        },
        control_receipt={
            "arm": "sampled_512",
            "peak_allocated_bytes": 1000,
            "peak_reserved_bytes": 2000,
            "checkpoints": control_checkpoints,
        },
        candidate_receipt={
            "arm": "full_768",
            "peak_allocated_bytes": 1000,
            "peak_reserved_bytes": 2000,
            "checkpoints": candidate_checkpoints,
        },
    )

    assert result["evidence"]["checkpoint_bytes_equal"] is False
    assert sum(result["evidence"]["control_checkpoint_bytes"]) == sum(
        result["evidence"]["candidate_checkpoint_bytes"]
    )
    assert result["status"] == "CLOSE_RESOURCE"
    module.validate_seed0_decision(result)
