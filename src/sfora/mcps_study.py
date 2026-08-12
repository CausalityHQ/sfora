"""Paired descriptive analysis for the three-seed MCPS training study."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

ARMS = (
    "proxy_anchor",
    "proxy_anchor_mcps_pg",
    "proxy_anchor_proxy_compactness",
)
METRICS = ("final_r1", "final_map_at_r", "best_r1")


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite builtin float")
    return value


def _load_arm(root: Path, arm: str, seed: int) -> dict[str, object]:
    path = root / f"{arm}-seed{seed}" / "report.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing report: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: None)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid report JSON: {path}") from error
    if type(report) is not dict or report.get("dataset_name") != "inshop":
        raise ValueError(f"report dataset differs: {path}")
    methods = report.get("methods")
    if type(methods) is not dict or len(methods) != 1:
        raise ValueError(f"report method schema differs: {path}")
    method = next(iter(methods.values()))
    if type(method) is not dict or method.get("objective") != arm:
        raise ValueError(f"report objective differs: {path}")
    if method.get("executed_train_steps") != 8580:
        raise ValueError(f"report training steps differ: {path}")
    best_epoch = method.get("best_test_epoch")
    if type(best_epoch) is not int or not 1 <= best_epoch <= 60:
        raise ValueError(f"report best epoch differs: {path}")
    row: dict[str, object] = {
        "arm": arm,
        "seed": seed,
        "final_r1": _finite_float(method.get("recall_at_1"), name="final_r1"),
        "final_map_at_r": _finite_float(method.get("map_at_r"), name="final_map_at_r"),
        "best_r1": _finite_float(
            method.get("best_test_recall_at_1"), name="best_test_recall_at_1"
        ),
        "best_epoch": best_epoch,
    }
    diagnostics = method.get("mcps_diagnostics")
    if arm == "proxy_anchor_mcps_pg":
        if type(diagnostics) is not dict:
            raise ValueError(f"MCPS diagnostics missing: {path}")
        row["mcps_diagnostics"] = {
            key: _finite_float(diagnostics.get(key), name=key)
            for key in ("conflict_rate", "memory_target_rate", "skip_rate")
        }
    elif diagnostics is not None:
        raise ValueError(f"non-MCPS arm contains MCPS diagnostics: {path}")
    else:
        row["mcps_diagnostics"] = None
    return row


def _delta_summary(
    rows: dict[str, list[dict[str, object]]], left: str, right: str, metric: str
) -> dict[str, object]:
    deltas = [
        float(left_row[metric]) - float(right_row[metric])
        for left_row, right_row in zip(rows[left], rows[right], strict=True)
    ]
    ordered = sorted(deltas)
    return {
        "per_seed": deltas,
        "mean": float(sum(deltas) / len(deltas)),
        "median": float(ordered[len(ordered) // 2]),
        "min": float(min(deltas)),
        "max": float(max(deltas)),
        "all_positive": all(delta > 0.0 for delta in deltas),
    }


def summarize_mcps_study(root: Path) -> dict[str, object]:
    """Load exactly nine reports and compute paired descriptive comparisons."""

    root = Path(root)
    rows = {arm: [_load_arm(root, arm, seed) for seed in range(3)] for arm in ARMS}
    comparisons: dict[str, object] = {}
    for name, left, right in (
        ("mcps_minus_pa", "proxy_anchor_mcps_pg", "proxy_anchor"),
        (
            "compactness_minus_pa",
            "proxy_anchor_proxy_compactness",
            "proxy_anchor",
        ),
        (
            "mcps_minus_compactness",
            "proxy_anchor_mcps_pg",
            "proxy_anchor_proxy_compactness",
        ),
    ):
        comparisons[name] = {
            metric: _delta_summary(rows, left, right, metric) for metric in METRICS
        }
    mcps_vs_pa = comparisons["mcps_minus_pa"]
    mcps_vs_control = comparisons["mcps_minus_compactness"]
    specific = all(
        comparison[metric]["all_positive"]
        for comparison in (mcps_vs_pa, mcps_vs_control)
        for metric in ("final_r1", "final_map_at_r")
    )
    return {
        "schema_version": 1,
        "study_root": str(root.resolve()),
        "arms": list(ARMS),
        "seeds": [0, 1, 2],
        "rows": [row for arm in ARMS for row in rows[arm]],
        "comparisons": comparisons,
        "decision": "MCPS_SPECIFIC_POSITIVE" if specific else "MIXED_OR_NULL",
        "limitations": [
            "three paired seeds are descriptive and do not support an asymptotic significance claim"
        ],
    }


def write_summary_no_clobber(path: Path, summary: dict[str, object]) -> None:
    path = Path(path)
    encoded = (
        json.dumps(summary, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    if persisted != summary:
        raise ValueError("persisted MCPS summary differs")
