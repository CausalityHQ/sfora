from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfora.mcps_study import summarize_mcps_study, write_summary_no_clobber

ARMS = (
    "proxy_anchor",
    "proxy_anchor_mcps_pg",
    "proxy_anchor_proxy_compactness",
)
INFERENCE_CONFIG_KEYS = [
    "backbone_name",
    "dataset_name",
    "dataset_root",
    "dataset_selection_policy",
    "ema_eval_momentum",
    "ema_weight_averaging",
    "embedding_dimensions",
    "embedding_layer_norm",
    "eval_batch_size",
    "head_pooling",
    "input_size",
    "pre_embedding_layer_norm",
    "pretrained_weights",
    "protocol",
    "recall_at_k_values",
    "retrieval_query_limit",
]


def _write_report(
    root: Path,
    arm: str,
    seed: int,
    *,
    r1: float,
    map_at_r: float,
    best_r1: float,
) -> None:
    directory = root / f"{arm}-seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    diagnostics = None
    if arm == "proxy_anchor_mcps_pg":
        diagnostics = {
            "conflict_rate": 0.2,
            "memory_target_rate": 0.8,
            "skip_rate": 0.0,
        }
    payload = {
        "config": {
            "backbone_name": "bn_inception",
            "batch_size": 180,
            "dataset_name": "inshop",
            "dataset_root": "/datasets/inshop",
            "dataset_selection_policy": "full_official_partition",
            "ema_eval_momentum": None,
            "ema_weight_averaging": False,
            "embedding_dimensions": 512,
            "embedding_layer_norm": False,
            "eval_batch_size": 128,
            "head_pooling": "avg_max",
            "input_size": 224,
            "objectives": [arm],
            "pre_embedding_layer_norm": False,
            "pretrained_weights": "bn_inception_52deb4733",
            "protocol": "proxy-anchor-resnet50-512",
            "recall_at_k_values": [1, 2, 4, 8, 16],
            "recipe_digest": f"digest-{arm}",
            "recipe_modified_fields": (
                {} if arm == "proxy_anchor" else {"objectives": {"after": [arm]}}
            ),
            "recipe_track": "reference" if arm == "proxy_anchor" else "modified",
            "save_model_path": str(directory / "checkpoint.pt"),
            "retrieval_query_limit": None,
            "seed": seed,
        },
        "dataset_name": "inshop",
        "methods": {
            f"{arm}_end_to_end:bn_inception": {
                "objective": arm,
                "recall_at_1": r1,
                "map_at_r": map_at_r,
                "best_test_recall_at_1": best_r1,
                "best_test_epoch": 50,
                "executed_train_steps": 8580,
                "mcps_diagnostics": diagnostics,
            }
        },
    }
    (directory / "report.json").write_text(json.dumps(payload) + "\n")


def _complete_fixture(root: Path) -> None:
    for seed in range(3):
        _write_report(
            root,
            "proxy_anchor",
            seed,
            r1=0.91 + seed / 1000,
            map_at_r=0.64,
            best_r1=0.92,
        )
        _write_report(
            root,
            "proxy_anchor_mcps_pg",
            seed,
            r1=0.913 + seed / 1000,
            map_at_r=0.642,
            best_r1=0.921,
        )
        _write_report(
            root,
            "proxy_anchor_proxy_compactness",
            seed,
            r1=0.911 + seed / 1000,
            map_at_r=0.641,
            best_r1=0.9205,
        )


def test_summary_reports_frozen_gate_and_stricter_specificity(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)

    result = summarize_mcps_study(tmp_path)

    assert result["schema_version"] == 2
    assert tuple(result["arms"]) == ARMS
    comparison = result["comparisons"]["mcps_minus_pa"]
    assert comparison["final_r1"]["per_seed"] == pytest.approx([0.003, 0.003, 0.003])
    assert comparison["final_r1"]["mean"] == pytest.approx(0.003)
    assert comparison["final_r1"]["all_positive"] is True
    assert result["comparisons"]["mcps_minus_compactness"]["final_r1"][
        "all_positive"
    ] is True
    assert result["frozen_gate"] == {
        "diagnostics_pass": True,
        "positive_seed_count": 3,
        "positive_seed_count_pass": True,
        "paired_mean": pytest.approx(0.003),
        "paired_standard_error": pytest.approx(0.0, abs=1e-15),
        "effect_size_pass": True,
        "compactness_control_pass": True,
        "inference_config_keys": INFERENCE_CONFIG_KEYS,
        "inference_config_differences": [],
        "inference_config_identical": True,
    }
    assert result["decision"] == "MCPS_PASS"
    assert result["mechanism_specificity_decision"] == "MCPS_SPECIFIC_POSITIVE"
    assert result["limitations"] == [
        "three paired seeds are descriptive and do not support an asymptotic significance claim"
    ]


def test_frozen_gate_can_pass_two_of_three_while_specificity_stays_mixed(
    tmp_path: Path,
) -> None:
    _complete_fixture(tmp_path)
    _write_report(
        tmp_path,
        "proxy_anchor_mcps_pg",
        2,
        r1=0.911,
        map_at_r=0.642,
        best_r1=0.919,
    )

    result = summarize_mcps_study(tmp_path)

    assert result["frozen_gate"]["positive_seed_count"] == 2
    assert result["frozen_gate"]["paired_mean"] == pytest.approx(0.0016666666666667)
    assert result["frozen_gate"]["effect_size_pass"] is True
    assert result["decision"] == "MCPS_PASS"
    assert result["mechanism_specificity_decision"] == "MIXED_OR_NULL"
    assert result["comparisons"]["mcps_minus_pa"]["final_r1"]["all_positive"] is False


def test_frozen_gate_closes_on_diagnostics_or_inference_config_drift(
    tmp_path: Path,
) -> None:
    _complete_fixture(tmp_path)
    report_path = tmp_path / "proxy_anchor_mcps_pg-seed1" / "report.json"
    report = json.loads(report_path.read_text())
    method = next(iter(report["methods"].values()))
    method["mcps_diagnostics"]["conflict_rate"] = 0.049
    report_path.write_text(json.dumps(report) + "\n")

    diagnostic_failure = summarize_mcps_study(tmp_path)

    assert diagnostic_failure["frozen_gate"]["diagnostics_pass"] is False
    assert diagnostic_failure["decision"] == "MCPS_CLOSE"
    assert diagnostic_failure["mechanism_specificity_decision"] == "NOT_EVALUABLE"

    method["mcps_diagnostics"]["conflict_rate"] = 0.2
    report["config"]["embedding_dimensions"] = 256
    report_path.write_text(json.dumps(report) + "\n")

    config_failure = summarize_mcps_study(tmp_path)

    assert config_failure["frozen_gate"]["inference_config_identical"] is False
    assert config_failure["frozen_gate"]["inference_config_differences"] == [
        {
            "arm": "proxy_anchor_mcps_pg",
            "keys": ["embedding_dimensions"],
            "seed": 1,
        }
    ]
    assert config_failure["decision"] == "MCPS_CLOSE"
    assert config_failure["mechanism_specificity_decision"] == "NOT_EVALUABLE"


def test_inference_comparison_ignores_output_paths_but_audits_compared_keys(
    tmp_path: Path,
) -> None:
    _complete_fixture(tmp_path)
    for arm in ARMS:
        for seed in range(3):
            path = tmp_path / f"{arm}-seed{seed}" / "report.json"
            report = json.loads(path.read_text())
            report["config"]["save_test_embeddings"] = str(
                tmp_path / f"{arm}-seed{seed}" / "test.npz"
            )
            path.write_text(json.dumps(report) + "\n")

    result = summarize_mcps_study(tmp_path)

    assert result["frozen_gate"]["inference_config_identical"] is True
    assert result["frozen_gate"]["inference_config_keys"] == INFERENCE_CONFIG_KEYS
    assert result["frozen_gate"]["inference_config_differences"] == []


@pytest.mark.parametrize(
    ("deltas", "compactness_delta", "failed_predicate"),
    [
        ((0.0014, 0.0014, 0.0014), 0.001, "effect_size_pass"),
        ((0.010, -0.004, 0.0), 0.001, "effect_size_pass"),
        ((0.004, -0.001, -0.001), 0.001, "positive_seed_count_pass"),
        ((0.003, 0.003, 0.003), -0.001, "compactness_control_pass"),
    ],
)
def test_each_frozen_effect_predicate_can_close_independently(
    tmp_path: Path,
    deltas: tuple[float, float, float],
    compactness_delta: float,
    failed_predicate: str,
) -> None:
    _complete_fixture(tmp_path)
    for seed, delta in enumerate(deltas):
        pa_r1 = 0.91 + seed / 1000
        _write_report(
            tmp_path,
            "proxy_anchor_mcps_pg",
            seed,
            r1=pa_r1 + delta,
            map_at_r=0.642,
            best_r1=0.921,
        )
        _write_report(
            tmp_path,
            "proxy_anchor_proxy_compactness",
            seed,
            r1=pa_r1 + delta - compactness_delta,
            map_at_r=0.641,
            best_r1=0.9205,
        )

    result = summarize_mcps_study(tmp_path)

    assert result["frozen_gate"][failed_predicate] is False
    assert result["decision"] == "MCPS_CLOSE"


def test_summary_rejects_missing_report_or_nonfinite_metric(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    (tmp_path / "proxy_anchor-seed2" / "report.json").unlink()
    with pytest.raises(ValueError, match="missing report"):
        summarize_mcps_study(tmp_path)

    _write_report(tmp_path, "proxy_anchor", 2, r1=float("nan"), map_at_r=0.64, best_r1=0.92)
    with pytest.raises(ValueError, match="finite builtin float"):
        summarize_mcps_study(tmp_path)


def test_summary_publication_never_clobbers(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    summary = summarize_mcps_study(tmp_path)
    output = tmp_path / "summary.json"

    write_summary_no_clobber(output, summary)

    original = output.read_bytes()
    assert json.loads(original) == summary
    with pytest.raises(FileExistsError):
        write_summary_no_clobber(output, summary)
    assert output.read_bytes() == original
