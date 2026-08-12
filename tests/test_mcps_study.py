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


def test_summary_computes_exact_paired_deltas_and_specificity(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)

    result = summarize_mcps_study(tmp_path)

    assert result["schema_version"] == 1
    assert tuple(result["arms"]) == ARMS
    comparison = result["comparisons"]["mcps_minus_pa"]
    assert comparison["final_r1"]["per_seed"] == pytest.approx([0.003, 0.003, 0.003])
    assert comparison["final_r1"]["mean"] == pytest.approx(0.003)
    assert comparison["final_r1"]["all_positive"] is True
    assert result["comparisons"]["mcps_minus_compactness"]["final_r1"][
        "all_positive"
    ] is True
    assert result["decision"] == "MCPS_SPECIFIC_POSITIVE"
    assert result["limitations"] == [
        "three paired seeds are descriptive and do not support an asymptotic significance claim"
    ]


def test_summary_marks_mixed_when_one_seed_or_control_does_not_improve(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    _write_report(
        tmp_path,
        "proxy_anchor_mcps_pg",
        2,
        r1=0.91,
        map_at_r=0.639,
        best_r1=0.919,
    )

    result = summarize_mcps_study(tmp_path)

    assert result["decision"] == "MIXED_OR_NULL"
    assert result["comparisons"]["mcps_minus_pa"]["final_r1"]["all_positive"] is False


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
