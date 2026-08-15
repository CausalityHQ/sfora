from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/summarize_unicom_ema_imprint_replication.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("summarize_unicom_replication", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(seed: int, map_delta: float, recall_delta: float) -> dict[str, object]:
    baseline_map = 0.89 + seed * 0.001
    baseline_recall = 0.97 + seed * 0.001
    epochs = [4, 8, 12, 16]

    def arm(*, imprinted: bool) -> dict[str, object]:
        endpoint_map = baseline_map + (map_delta if imprinted else 0.0)
        endpoint_recall = baseline_recall + (recall_delta if imprinted else 0.0)
        start = endpoint_map - (0.005 if imprinted else 0.03)
        return {
            "checkpoint_sha256_by_epoch": [
                f"{seed * 8 + (4 if imprinted else 0) + index:064x}" for index in range(4)
            ],
            "epoch_metrics": [
                {
                    "epoch": epoch,
                    "map_at_r": endpoint_map if epoch == 16 else start + index * 0.01,
                    "recall_at_1": endpoint_recall if epoch == 16 else endpoint_recall - 0.01,
                }
                for index, epoch in enumerate(epochs)
            ],
            "training_seconds": (14000.0 if imprinted else 15000.0) + seed,
            "peak_gpu_mib": (86900 if imprinted else 87000) + seed,
            "checkpoint_storage_bytes": (14500000100 if imprinted else 14500000000) + seed,
            "deployment_storage_bytes": 1820000000 + seed,
            "profile": {
                "step_wall_seconds": 1.0,
                "fusible_non_backbone_seconds": 0.05,
            },
        }

    return {
        "schema_version": "unicom-ema-imprint-replication-pair-v1",
        "seed": seed,
        "selected_cell": "imprinted_raw",
        "registered_epochs": epochs,
        "random_raw": arm(imprinted=False),
        "imprinted_raw": arm(imprinted=True),
        "inference_latency": {
            "warmup_repetitions": 10,
            "measured_repetitions": 50,
            "batch_size": 128,
            "milliseconds_per_image": 11.8,
        },
    }


def test_summary_requires_exact_seeds_and_frozen_cell() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.002) for seed in range(1, 7)]

    summary = module.summarize_replications(reports)

    assert summary["training_seeds"] == [1, 2, 3, 4, 5, 6]
    assert summary["selected_cell"] == "imprinted_raw"
    assert summary["claim_supported"] is True
    reports[3]["seed"] = 7
    with pytest.raises(ValueError, match="seeds"):
        module.summarize_replications(reports)


def test_summary_uses_paired_student_t_sign_and_recall_gates() -> None:
    module = _load_script()
    reports = [_report(seed, 0.010 + seed * 0.001, -0.001) for seed in range(1, 7)]

    summary = module.summarize_replications(reports)

    assert summary["map_deltas"] == pytest.approx([0.011, 0.012, 0.013, 0.014, 0.015, 0.016])
    assert summary["map_delta_sample_standard_deviation"] > 0.0
    assert summary["map_delta_paired_student_t_95_interval"][0] > 0.0
    assert summary["exact_two_sided_sign_p_value"] == 0.03125
    assert summary["all_map_deltas_positive"] is True
    assert summary["all_recall_at_1_deltas_above_guard"] is True
    reports[5]["imprinted_raw"]["epoch_metrics"][-1]["recall_at_1"] -= 0.000251
    assert module.summarize_replications(reports)["claim_supported"] is False


def test_summary_recomputes_time_to_quality_and_cost_pareto_fields() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]

    summary = module.summarize_replications(reports)

    assert summary["first_quality_epochs"] == [
        {"seed": seed, "random_raw": 16, "imprinted_raw": 4, "speedup": 4.0}
        for seed in range(1, 7)
    ]
    assert summary["costs"]["training_seconds"][0] == {
        "seed": 1,
        "random_raw": 15001.0,
        "imprinted_raw": 14001.0,
    }
    assert summary["costs"]["inference_latency_protocol"] == {
        "warmup_repetitions": 10,
        "measured_repetitions": 50,
        "batch_size": 128,
    }
    assert summary["costs"]["kernel_profile_threshold"] == 0.1
    assert summary["costs"]["kernel_eligible"] is False
    assert summary["pareto_cost_noninferior"] is True
    assert summary["pareto_nondominated_against_random_raw"] is True

    dominated = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    for report in dominated:
        report["imprinted_raw"]["training_seconds"] = 20000.0
    dominated_summary = module.summarize_replications(dominated)
    assert dominated_summary["quality_claim_supported"] is True
    assert dominated_summary["pareto_cost_noninferior"] is False
    assert dominated_summary["pareto_nondominated_against_random_raw"] is False
    assert dominated_summary["claim_supported"] is False

    reports[0]["imprinted_raw"]["epoch_metrics"][0]["map_at_r"] = 0.0
    assert module.summarize_replications(reports)["first_quality_epochs"][0] == {
        "seed": 1,
        "random_raw": 16,
        "imprinted_raw": 8,
        "speedup": 2.0,
    }


def test_summary_rejects_degenerate_or_reused_checkpoint_evidence() -> None:
    module = _load_script()
    reports = [_report(seed, 0.01, 0.0) for seed in range(1, 7)]

    degenerate = module.summarize_replications(reports)

    assert degenerate["nondegenerate_training_seed_variation"] is False
    assert degenerate["claim_supported"] is False
    reports[5]["imprinted_raw"]["checkpoint_sha256_by_epoch"][3] = reports[0][
        "random_raw"
    ]["checkpoint_sha256_by_epoch"][0]
    with pytest.raises(ValueError, match="checkpoint"):
        module.summarize_replications(reports)


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
        module.summarize_replications(reports)


def test_atomic_publication_strict_reloads_and_never_clobbers(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_report(seed, 0.01 + seed * 0.001, 0.0) for seed in range(1, 7)]
    summary = module.summarize_replications(reports)
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
