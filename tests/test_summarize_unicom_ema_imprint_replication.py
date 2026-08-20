from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/summarize_unicom_ema_imprint_replication.py"
SELECTION_REPORT = (
    Path(__file__).parents[1]
    / "reports/generated/unicom_ema_imprint_factorial_88604a4_seed0.json"
)
HISTORICAL_SEED1_REPORT = (
    Path(__file__).parents[1]
    / "reports/generated/unicom_ema_imprint_replication_c83cd96_seed1.json"
)
SELECTION_AUTHORITY = {
    "path": "reports/generated/unicom_ema_imprint_factorial_88604a4_seed0.json",
    "sha256": "c0666a68e70990115d80e8dc06a9f94efe83156a3fddd50f36bdbf2b3b8cd217",
    "recording_commit": "81f3f48c374d14b5a91bbeba7a1fec2fb0a4a2d6",
    "selected_cell": "imprinted_raw",
    "decision": "PROMOTE",
}


def _load_script():
    spec = importlib.util.spec_from_file_location("summarize_unicom_replication", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_historical_seed1_authority_rejects_semantic_or_byte_drift() -> None:
    """The irreplaceable v1 input must equal its frozen Git-backed bytes."""
    module = _load_script()
    report = module.strict_json_object(HISTORICAL_SEED1_REPORT.read_bytes())

    module.authenticate_historical_seed1(report)

    report["random_raw"]["training_seconds"] += 1.0
    with pytest.raises(ValueError, match="historical seed-1"):
        module.authenticate_historical_seed1(report)


def test_historical_seed1_loader_rejects_an_exact_byte_alias(tmp_path: Path) -> None:
    """The CLI authority is the registered path, not a copied equivalent file."""
    module = _load_script()
    copied = tmp_path / HISTORICAL_SEED1_REPORT.name
    copied.write_bytes(HISTORICAL_SEED1_REPORT.read_bytes())

    with pytest.raises(ValueError, match="historical seed-1 report path"):
        module.load_historical_seed1_report(copied)


def _protocol(seed: int, classifier_init: str) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": "a" * 64,
        "unicom_revision": "b" * 40,
        "initial_checkpoint_sha256": "c" * 64,
        "partition_sha256": "d" * 64,
        "seed": seed,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 1e-5,
        "classifier_learning_rate": 1e-4,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 512,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "eval_every": 4,
        "checkpoint_every": 4,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
        "classifier_init": classifier_init,
        "ema_decay": 0.999,
        "ema_update": "optimizer-step-post-hook-trainable-parameters-only",
    }


def _pair_v1_report(seed: int, map_delta: float, recall_delta: float) -> dict[str, object]:
    if seed == 1:
        return json.loads(HISTORICAL_SEED1_REPORT.read_text(encoding="utf-8"))
    baseline_map = 0.89 + seed * 0.001
    baseline_recall = 0.97 + seed * 0.001
    epochs = [4, 8, 12, 16]

    def arm(*, imprinted: bool) -> tuple[dict[str, object], list[dict[str, object]]]:
        endpoint_map = baseline_map + (map_delta if imprinted else 0.0)
        endpoint_recall = baseline_recall + (recall_delta if imprinted else 0.0)
        start = endpoint_map - (0.005 if imprinted else 0.03)
        map_values = [
            endpoint_map if epoch == 16 else start + index * 0.01
            for index, epoch in enumerate(epochs)
        ]
        query_count = 797
        true_count = round(endpoint_recall * query_count)
        top1 = [True] * true_count + [False] * (query_count - true_count)
        evidence = [
            {"top1_correct": list(top1), "average_precision": [value] * query_count}
            for value in map_values
        ]
        return {
            "checkpoint_sha256_by_epoch": [
                f"{seed * 8 + (4 if imprinted else 0) + index:064x}" for index in range(4)
            ],
            "training_history_sha256_by_epoch": [
                f"{1000 + seed * 8 + (4 if imprinted else 0) + index:064x}"
                for index in range(4)
            ],
            "epoch_metrics": [
                {
                    "epoch": epoch,
                    "map_at_r": float(np.mean(np.asarray(evidence[index]["average_precision"]))),
                    "recall_at_1": float(np.mean(np.asarray(top1, dtype=np.float64))),
                }
                for index, epoch in enumerate(epochs)
            ],
            "training_seconds": (14000.0 if imprinted else 15000.0) + seed,
            "peak_gpu_mib": (86900 if imprinted else 87000) + seed,
            "checkpoint_storage_bytes": (14499999900 if imprinted else 14500000000) + seed,
            "deployment_storage_bytes": 1820000000 + seed,
            "measurement_receipt_sha256": (
                f"{5000 + seed * 2 + (1 if imprinted else 0):064x}"
            ),
            "profile": {
                "step_wall_seconds": 1.0,
                "fusible_non_backbone_seconds": 0.05,
            },
        }, evidence

    random_arm, random_evidence = arm(imprinted=False)
    imprinted_arm, imprinted_evidence = arm(imprinted=True)

    return {
        "schema_version": "unicom-ema-imprint-replication-pair-v1",
        "seed": seed,
        "selected_cell": "imprinted_raw",
        "registered_epochs": epochs,
        "random_training_protocol": _protocol(seed, "random"),
        "imprinted_training_protocol": _protocol(seed, "imprinted"),
        "random_raw": random_arm,
        "imprinted_raw": imprinted_arm,
        "inference_latency": {
            "warmup_repetitions": 10,
            "measured_repetitions": 50,
            "batch_size": 128,
            "milliseconds_per_image": 11.8,
        },
        "evidence": {"random_raw": random_evidence, "imprinted_raw": imprinted_evidence},
    }


def _future_report(seed: int, map_delta: float, recall_delta: float) -> dict[str, object]:
    report = _pair_v1_report(seed, map_delta, recall_delta)
    report["schema_version"] = "unicom-ema-imprint-replication-pair-v2"
    historical = json.loads(HISTORICAL_SEED1_REPORT.read_text(encoding="utf-8"))
    for cell, classifier_init in (("random_raw", "random"), ("imprinted_raw", "imprinted")):
        protocol = copy.deepcopy(historical[f"{cell.split('_')[0]}_training_protocol"])
        protocol["seed"] = seed
        protocol["classifier_init"] = classifier_init
        protocol["trainer_sha256"] = "f" * 64
        report[f"{cell.split('_')[0]}_training_protocol"] = protocol
    rng_digest = f"{7000 + seed:064x}"
    for arm, initialization_seconds in (("random_raw", 1.0), ("imprinted_raw", 10.0)):
        report[arm]["optimizer_steps_per_epoch"] = 161
        report[arm]["initialization_seconds"] = initialization_seconds
        report[arm]["initialization_receipt_sha256"] = (
            f"{8000 + seed * 2 + (arm == 'imprinted_raw'):064x}"
        )
        report[arm]["post_initialization_rng_sha256"] = rng_digest
    return report


def _report(seed: int, map_delta: float, recall_delta: float) -> dict[str, object]:
    if seed == 1:
        return _pair_v1_report(seed, map_delta, recall_delta)
    return _future_report(seed, map_delta, recall_delta)


def _summarize(module, reports: list[dict[str, object]]) -> dict[str, object]:
    return module.summarize_replications(
        reports, selection_authority=copy.deepcopy(SELECTION_AUTHORITY)
    )


def _registered_reports() -> list[dict[str, object]]:
    return [
        _pair_v1_report(1, 0.011, 0.0),
        *[_future_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(2, 7)],
    ]


def _set_map_curve(report: dict[str, object], arm: str, values: list[float]) -> None:
    for metric, evidence, value in zip(
        report[arm]["epoch_metrics"], report["evidence"][arm], values, strict=True
    ):
        evidence["average_precision"] = [value] * len(evidence["average_precision"])
        metric["map_at_r"] = float(
            np.mean(np.asarray(evidence["average_precision"], dtype=np.float64))
        )


@pytest.mark.parametrize(
    ("key", "replacement"),
    (
        ("path", "reports/generated/other.json"),
        ("sha256", "0" * 64),
        ("recording_commit", "0" * 40),
        ("selected_cell", "random_raw"),
        ("decision", "CLOSE"),
    ),
)
def test_summary_v2_binds_frozen_selection_authority_and_rejects_roundtrip_mutation(
    key: str, replacement: str
) -> None:
    """A copied authority must not become self-authenticating summary evidence."""
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]

    summary = module.summarize_replications(
        reports, selection_authority=copy.deepcopy(SELECTION_AUTHORITY)
    )

    assert summary["schema_version"] == "unicom-ema-imprint-replication-summary-v2"
    assert summary["selection_authority"] == SELECTION_AUTHORITY
    mutated = json.loads(json.dumps(summary))
    mutated["selection_authority"][key] = replacement
    with pytest.raises(ValueError, match="selection authority"):
        module.validate_summary(mutated)


def test_summary_cli_authenticates_selection_from_repo_root_not_cwd(tmp_path: Path) -> None:
    """The CLI must authenticate the frozen Git-backed report from any cwd."""
    module = _load_script()
    report_paths: list[Path] = [HISTORICAL_SEED1_REPORT]
    for seed in range(2, 7):
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(
            json.dumps(_report(seed, 0.01 + seed * 0.001, 0.0)) + "\n",
            encoding="utf-8",
        )
        report_paths.append(path)
    output = tmp_path / "summary.json"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    previous = Path.cwd()
    try:
        module.os.chdir(elsewhere)
        exit_code = module.main(
            [
                *(str(path) for path in report_paths),
                "--selection-report",
                str(SELECTION_REPORT),
                "--output",
                str(output),
            ]
        )
    finally:
        module.os.chdir(previous)

    assert exit_code == 0
    persisted = module.strict_json_object(output.read_bytes())
    assert persisted["selection_authority"] == SELECTION_AUTHORITY
    module.validate_summary(persisted)


def test_selection_authority_rejects_an_alias_or_symlink(tmp_path: Path) -> None:
    """Only the frozen repository-relative path may carry selection authority."""
    module = _load_script()
    copied = tmp_path / "selection.json"
    copied.write_bytes(SELECTION_REPORT.read_bytes())
    with pytest.raises(ValueError, match="path"):
        module.load_selection_authority(copied)

    linked = tmp_path / "selection-link.json"
    linked.symlink_to(SELECTION_REPORT)
    with pytest.raises(ValueError, match="path"):
        module.load_selection_authority(linked)


def test_summary_separates_non_monotone_epoch_quality_and_compute_operating_points() -> (
    None
):
    """Fixed-budget quality must not be mislabeled as iso-quality cost dominance."""
    module = _load_script()
    reports = _registered_reports()

    summary = module.summarize_replications(
        reports, selection_authority=copy.deepcopy(SELECTION_AUTHORITY)
    )

    assert summary["first_quality_epochs"][0] == {
        "seed": 1,
        "random_raw": 12,
        "imprinted_raw": 8,
        "speedup": 1.5,
    }
    assert summary["initialization_evidence"][0] == {
        "seed": 1,
        "status": "historical_initialization_receipt_unavailable",
        "random_raw": None,
        "imprinted_raw": None,
        "post_initialization_rng_equal": None,
    }
    assert summary["costs"]["fixed_epoch_profiled_compute_seconds"][0] == {
        "seed": 1,
        "random_raw": None,
        "imprinted_raw": None,
    }
    assert summary["costs"]["iso_quality_profiled_compute_seconds"][1] == {
        "seed": 2,
        "random_raw": 2577.0,
        "imprinted_raw": 654.0,
    }
    assert summary["costs"]["fixed_epoch_profiled_compute_seconds"][1] == {
        "seed": 2,
        "random_raw": 2577.0,
        "imprinted_raw": 2586.0,
    }
    assert summary["fixed_epoch_pareto_nondominated"] is True
    assert summary["all_first_quality_epochs_noninferior"] is True
    assert summary["costs"]["fixed_epoch_profiled_compute_overhead_seconds"][1] == {
        "seed": 2,
        "imprinted_minus_random": 9.0,
    }
    assert summary["all_future_iso_quality_profiled_compute_noninferior"] is True


def test_contaminated_wall_time_is_descriptive_but_per_seed_resources_gate() -> None:
    """Raw time cannot decide; each measured resource and exact deployment size can."""
    module = _load_script()
    reports = _registered_reports()
    baseline = module.summarize_replications(
        reports, selection_authority=copy.deepcopy(SELECTION_AUTHORITY)
    )
    reports[2]["imprinted_raw"]["training_seconds"] = 1_000_000.0
    contaminated = module.summarize_replications(
        reports, selection_authority=copy.deepcopy(SELECTION_AUTHORITY)
    )
    assert contaminated["costs"]["training_seconds"] != baseline["costs"][
        "training_seconds"
    ]
    assert contaminated["claim_supported"] is baseline["claim_supported"]

    for field, value in (
        ("peak_gpu_mib", reports[3]["random_raw"]["peak_gpu_mib"] + 1),
        (
            "checkpoint_storage_bytes",
            reports[3]["random_raw"]["checkpoint_storage_bytes"] + 1,
        ),
        (
            "deployment_storage_bytes",
            reports[3]["random_raw"]["deployment_storage_bytes"] - 1,
        ),
    ):
        mutated = copy.deepcopy(reports)
        mutated[3]["imprinted_raw"][field] = value
        result = module.summarize_replications(
            mutated, selection_authority=copy.deepcopy(SELECTION_AUTHORITY)
        )
        assert result["per_seed_resource_noninferior"] is False
        assert result["claim_supported"] is False

    seed1_resource_mutation = copy.deepcopy(reports)
    seed1_resource_mutation[0]["imprinted_raw"]["peak_gpu_mib"] = (
        seed1_resource_mutation[0]["random_raw"]["peak_gpu_mib"] + 1
    )
    with pytest.raises(ValueError, match="historical seed-1"):
        _summarize(module, seed1_resource_mutation)

    slower_future = copy.deepcopy(reports)
    _set_map_curve(slower_future[1], "random_raw", [0.89, 0.89, 0.89, 0.89])
    _set_map_curve(slower_future[1], "imprinted_raw", [0.88, 0.88, 0.88, 0.901])
    slower_result = _summarize(module, slower_future)
    assert slower_result["all_first_quality_epochs_noninferior"] is False
    assert slower_result["claim_supported"] is False


def test_summary_requires_exact_seeds_and_frozen_cell() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.002) for seed in range(1, 7)]

    summary = _summarize(module, reports)

    assert summary["training_seeds"] == [1, 2, 3, 4, 5, 6]
    assert summary["selected_cell"] == "imprinted_raw"
    assert summary["claim_supported"] is True
    reports[3]["seed"] = 7
    with pytest.raises(ValueError, match="seed|registration"):
        _summarize(module, reports)


def test_summary_allows_versioned_trainer_digest_but_not_recipe_drift() -> None:
    """Seed 1 may be historical, but all future evidence uses one reviewed trainer."""
    module = _load_script()
    reports = _registered_reports()

    summary = _summarize(module, reports)
    assert summary["reports"][0]["random_training_protocol"]["trainer_sha256"] != (
        summary["reports"][1]["random_training_protocol"]["trainer_sha256"]
    )

    reports[2]["random_training_protocol"]["partition_sha256"] = "e" * 64
    reports[2]["imprinted_training_protocol"]["partition_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="paired training protocol differs across seeds"):
        _summarize(module, reports)

    reports = _registered_reports()
    reports[2]["random_training_protocol"]["trainer_sha256"] = "e" * 64
    reports[2]["imprinted_training_protocol"]["trainer_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="prospective trainer differs across seeds"):
        _summarize(module, reports)


def test_summary_uses_paired_student_t_sign_and_recall_gates() -> None:
    module = _load_script()
    reports = [_report(seed, 0.010 + seed * 0.001, 0.0) for seed in range(1, 7)]

    summary = _summarize(module, reports)

    assert summary["map_deltas"] == pytest.approx(
        [0.01443298839283702, 0.012, 0.013, 0.014, 0.015, 0.016]
    )
    assert summary["map_delta_sample_standard_deviation"] > 0.0
    assert summary["map_delta_paired_student_t_95_interval"][0] > 0.0
    assert summary["exact_two_sided_sign_p_value"] == 0.03125
    assert summary["all_map_deltas_positive"] is True
    assert summary["all_recall_at_1_deltas_above_guard"] is True
    reports[5]["evidence"]["imprinted_raw"][-1]["top1_correct"][0] = False
    reports[5]["imprinted_raw"]["epoch_metrics"][-1]["recall_at_1"] = float(
        np.mean(reports[5]["evidence"]["imprinted_raw"][-1]["top1_correct"])
    )
    assert _summarize(module, reports)["claim_supported"] is False


def test_summary_recomputes_time_to_quality_and_cost_pareto_fields() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]

    summary = _summarize(module, reports)

    assert summary["first_quality_epochs"] == [
        {"seed": 1, "random_raw": 12, "imprinted_raw": 8, "speedup": 1.5},
        *[
            {"seed": seed, "random_raw": 16, "imprinted_raw": 4, "speedup": 4.0}
            for seed in range(2, 7)
        ],
    ]
    assert summary["costs"]["training_seconds"][0] == {
        "seed": 1,
        "random_raw": 17629.0,
        "imprinted_raw": 14252.320465842,
    }
    assert summary["costs"]["inference_latency_protocol"] == {
        "warmup_repetitions": 10,
        "measured_repetitions": 50,
        "batch_size": 128,
    }
    assert summary["costs"]["kernel_profile_threshold"] == 0.1
    assert summary["costs"]["kernel_eligible"] is False
    assert summary["fixed_epoch_pareto_nondominated"] is True
    assert summary["all_future_iso_quality_profiled_compute_noninferior"] is True
    assert summary["per_seed_resource_noninferior"] is True

    dominated = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    for report in dominated[1:]:
        report["imprinted_raw"]["training_seconds"] = 20000.0
    dominated_summary = _summarize(module, dominated)
    assert dominated_summary["quality_claim_supported"] is True
    assert dominated_summary["fixed_epoch_pareto_nondominated"] is True
    assert dominated_summary["claim_supported"] is True

    checkpoint_dominated = [
        _report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)
    ]
    for report in checkpoint_dominated[1:]:
        report["imprinted_raw"]["checkpoint_storage_bytes"] = 10**15
    assert _summarize(module, checkpoint_dominated)["per_seed_resource_noninferior"] is False

    reports[1]["evidence"]["imprinted_raw"][0]["average_precision"] = [0.0] * 797
    reports[1]["imprinted_raw"]["epoch_metrics"][0]["map_at_r"] = 0.0
    assert _summarize(module, reports)["first_quality_epochs"][1] == {
        "seed": 2,
        "random_raw": 16,
        "imprinted_raw": 8,
        "speedup": 2.0,
    }


def test_fixed_epoch_pareto_uses_quality_and_registered_costs() -> None:
    """A quality loss is still nondominated when the candidate improves a live cost."""
    module = _load_script()
    reports = _registered_reports()
    _set_map_curve(reports[1], "imprinted_raw", [0.80, 0.82, 0.84, 0.86])

    assert _summarize(module, reports)["fixed_epoch_pareto_nondominated"] is True

    reports[1]["imprinted_raw"]["peak_gpu_mib"] = (
        reports[1]["random_raw"]["peak_gpu_mib"] + 1
    )
    reports[1]["imprinted_raw"]["checkpoint_storage_bytes"] = (
        reports[1]["random_raw"]["checkpoint_storage_bytes"] + 1
    )
    reports[1]["imprinted_raw"]["deployment_storage_bytes"] = (
        reports[1]["random_raw"]["deployment_storage_bytes"] + 1
    )
    assert _summarize(module, reports)["fixed_epoch_pareto_nondominated"] is False


def test_summary_rejects_degenerate_or_reused_checkpoint_evidence() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01, 0.0) for seed in range(1, 7)]
    historical = reports[0]
    for report in reports[1:]:
        for cell in ("random_raw", "imprinted_raw"):
            values = historical["evidence"][cell][-1]["average_precision"]
            report["evidence"][cell][-1]["average_precision"] = list(values)
            report[cell]["epoch_metrics"][-1]["map_at_r"] = float(
                np.mean(report["evidence"][cell][-1]["average_precision"])
            )

    degenerate = _summarize(module, reports)

    assert degenerate["nondegenerate_training_seed_variation"] is False
    assert degenerate["claim_supported"] is False
    reports[5]["imprinted_raw"]["checkpoint_sha256_by_epoch"][3] = reports[0][
        "random_raw"
    ]["checkpoint_sha256_by_epoch"][0]
    with pytest.raises(ValueError, match="checkpoint"):
        _summarize(module, reports)


@pytest.mark.parametrize(
    "field", ("initialization_receipt_sha256", "post_initialization_rng_sha256")
)
def test_summary_rejects_reused_future_initialization_evidence(field: str) -> None:
    """A future seed may not replay another seed's initialization/process evidence."""
    module = _load_script()
    reports = _registered_reports()
    for arm in ("random_raw", "imprinted_raw"):
        reports[2][arm][field] = reports[1][arm][field]

    with pytest.raises(ValueError, match="initialization|RNG"):
        _summarize(module, reports)


@pytest.mark.parametrize(
    ("arm", "field", "value"),
    (
        ("random_raw", "training_seconds", None),
        ("imprinted_raw", "peak_gpu_mib", -1),
        ("random_raw", "checkpoint_storage_bytes", 0.5),
        ("imprinted_raw", "deployment_storage_bytes", 0),
    ),
)
def test_summary_requires_complete_registered_costs(arm: str, field: str, value: object) -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    reports[2][arm][field] = value

    with pytest.raises((TypeError, ValueError), match="cost|epoch"):
        _summarize(module, reports)


def test_atomic_publication_strict_reloads_and_never_clobbers(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    summary = _summarize(module, reports)
    output = tmp_path / "summary.json"

    module.write_summary_atomic(summary, output)

    persisted = module.strict_json_object(output.read_bytes())
    module.validate_summary(persisted)
    assert persisted == summary
    assert list(tmp_path.iterdir()) == [output]
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_summary_atomic(summary, output)
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]


def test_summary_rejects_cross_seed_protocol_or_query_count_drift() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    reports[1]["random_training_protocol"]["partition_sha256"] = "e" * 64
    reports[1]["imprinted_training_protocol"]["partition_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="paired training protocol differs across seeds"):
        _summarize(module, reports)

    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    for cell in ("random_raw", "imprinted_raw"):
        for evidence, metric in zip(
            reports[1]["evidence"][cell], reports[1][cell]["epoch_metrics"], strict=True
        ):
            evidence["top1_correct"].pop()
            evidence["average_precision"].pop()
            metric["map_at_r"] = float(np.mean(evidence["average_precision"]))
            metric["recall_at_1"] = float(np.mean(evidence["top1_correct"]))
    with pytest.raises(ValueError, match="query"):
        _summarize(module, reports)


def test_summary_atomic_rolls_back_after_post_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "summary.json"
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    calls = 0
    real_fsync = module.os.fsync

    def fail_second_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_second_fsync)
    with pytest.raises(OSError, match="directory fsync"):
        module.write_summary_atomic(_summarize(module, reports), output)
    assert not output.exists()
    rollback = list(tmp_path.glob(".*.rollback"))
    assert len(rollback) == 1
    module.validate_summary(module.strict_json_object(rollback[0].read_bytes()))


def test_summary_rollback_never_clobbers_a_foreign_quarantine_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "summary.json"
    output.write_bytes(b"owned")
    info = output.stat()
    owned = (info.st_dev, info.st_ino)
    real_rename_noreplace = module._rename_noreplace
    collision: Path | None = None
    calls = 0

    def collide_once(source: Path, destination: Path) -> None:
        nonlocal calls, collision
        calls += 1
        if calls == 1:
            collision = destination
            destination.write_bytes(b"foreign quarantine")
        real_rename_noreplace(source, destination)

    monkeypatch.setattr(module, "_rename_noreplace", collide_once)
    directory_descriptor = module.os.open(tmp_path, module.os.O_RDONLY | module.os.O_DIRECTORY)
    try:
        module._rollback_published_link(
            output, owned=owned, directory_descriptor=directory_descriptor
        )
    finally:
        module.os.close(directory_descriptor)

    assert not output.exists()
    assert collision is not None
    assert collision.read_bytes() == b"foreign quarantine"
    entries = list(tmp_path.iterdir())
    assert collision in entries
    owned_quarantine = [path for path in entries if path != collision]
    assert len(owned_quarantine) == 1
    assert owned_quarantine[0].read_bytes() == b"owned"


def test_atomic_summary_uses_an_unnamed_inode_and_ignores_foreign_temp(
    tmp_path: Path,
) -> None:
    module = _load_script()
    output = tmp_path / "summary.json"
    temporary = output.with_name(f".{output.name}.{module.os.getpid()}.tmp")
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    temporary.write_bytes(b"foreign temp")

    module.write_summary_atomic(_summarize(module, reports), output)

    module.validate_summary(module.strict_json_object(output.read_bytes()))
    assert temporary.read_bytes() == b"foreign temp"


def test_atomic_summary_reloads_the_published_inode_after_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "summary.json"
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    real_link_fd = module._link_fd_noreplace

    def corrupt_after_link(
        descriptor: int, destination: Path, directory_descriptor: int
    ) -> None:
        real_link_fd(descriptor, destination, directory_descriptor)
        module.os.lseek(descriptor, 0, module.os.SEEK_SET)
        module.os.write(descriptor, b"!")
        module.os.fsync(descriptor)

    monkeypatch.setattr(module, "_link_fd_noreplace", corrupt_after_link)
    with pytest.raises((ValueError, RuntimeError)):
        module.write_summary_atomic(_summarize(module, reports), output)

    assert not output.exists()
