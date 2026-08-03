"""PFML fixed-reference scalar analysis tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_pfml_reference",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_pfml_reference.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _report(*, final: float = 0.91) -> dict:
    recalls = [0.5 + 0.022 * index for index in range(19)] + [final]
    best = max(recalls)
    best_epoch = 10 * (recalls.index(best) + 1)
    return {
        "config": dict(_module.EXPECTED_CONFIG),
        "train_examples": _module.EXPECTED_TRAIN_EXAMPLES,
        "test_examples": _module.EXPECTED_TEST_EXAMPLES,
        "methods": {
            "pfml_end_to_end:resnet50": {
                "objective": "pfml",
                "executed_train_steps": 16_200,
                "loss_history": [304_000_000.0] * 16_200,
                "test_recall_history": recalls,
                "recall_at_1": final,
                "best_test_recall_at_1": best,
                "best_test_epoch": best_epoch,
            }
        },
    }


def test_theoretical_batch_energy_matches_fixed_pair_accounting() -> None:
    bounds = _module.theoretical_batch_energy_bounds(_report()["config"])
    assert bounds["points_per_loss"] == 1_570
    assert bounds["same_class_ordered_pairs"] == 23_880
    assert bounds["different_class_ordered_pairs"] == 2_439_450
    assert bounds["force_free_energy_minimum"] == pytest.approx(301_946_250.0)


def test_expected_steps_are_independently_derived_from_official_count_and_epochs() -> None:
    config = _module.EXPECTED_CONFIG
    steps_per_epoch = (
        _module.EXPECTED_TRAIN_EXAMPLES + int(config["batch_size"]) - 1
    ) // int(config["batch_size"])
    assert int(config["train_steps"]) == steps_per_epoch * int(config["train_epochs"])
    assert int(config["train_steps"]) == _module.EXPECTED_STEPS


def test_analysis_reports_raw_best_and_final_separately() -> None:
    result = _module.analyze_report(_report(final=0.91))
    assert result["decision"] == "passes_fixed_interpretation_metric_gates"
    assert result["final_recall_at_1_primary"] == 0.91
    assert result["raw_best_test_selected_recall_at_1"] == pytest.approx(0.91)


def test_analysis_fails_fixed_gate_without_relabeling_raw_best() -> None:
    report = _report(final=0.88)
    result = _module.analyze_report(report)
    assert result["decision"] == "fails_fixed_interpretation_metric_gates"
    assert result["gates"]["final_at_least_0_890"] is False
    assert result["raw_best_test_selected_recall_at_1"] > result["final_recall_at_1_primary"]


def test_analysis_rejects_incomplete_curve() -> None:
    report = _report()
    report["methods"]["pfml_end_to_end:resnet50"]["loss_history"].pop()
    with pytest.raises(ValueError, match="curve shapes"):
        _module.analyze_report(report)


@pytest.mark.parametrize("key", sorted(_module.EXPECTED_CONFIG))
def test_analysis_rejects_every_mutated_frozen_recipe_field(key: str) -> None:
    report = _report()
    report["config"][key] = "mutated"
    with pytest.raises(ValueError, match=f"unexpected PFML config {key}"):
        _module.analyze_report(report)


@pytest.mark.parametrize("field", ["train_examples", "test_examples"])
def test_analysis_rejects_wrong_official_partition_count(field: str) -> None:
    report = _report()
    report[field] -= 1
    with pytest.raises(ValueError, match=field):
        _module.analyze_report(report)
