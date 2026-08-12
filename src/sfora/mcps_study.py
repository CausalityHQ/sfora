"""Paired descriptive analysis for the three-seed MCPS training study."""

from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path

ARMS = (
    "proxy_anchor",
    "proxy_anchor_mcps_pg",
    "proxy_anchor_proxy_compactness",
)
METRICS = ("final_r1", "final_map_at_r", "best_r1")
INFERENCE_CONFIG_KEYS = (
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
)


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
    config = report.get("config")
    if type(config) is not dict or config.get("seed") != seed:
        raise ValueError(f"report config differs: {path}")
    missing_inference_keys = [key for key in INFERENCE_CONFIG_KEYS if key not in config]
    if missing_inference_keys:
        raise ValueError(f"report inference config is incomplete: {path}")
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
        "_comparison_config": {key: config[key] for key in INFERENCE_CONFIG_KEYS},
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
    paired_final_r1 = mcps_vs_pa["final_r1"]
    paired_deltas = paired_final_r1["per_seed"]
    paired_mean = paired_final_r1["mean"]
    paired_standard_error = statistics.stdev(paired_deltas) / math.sqrt(3.0)
    positive_seed_count = sum(delta > 0.0 for delta in paired_deltas)
    diagnostics_pass = all(
        row["mcps_diagnostics"]["memory_target_rate"] >= 0.70
        and row["mcps_diagnostics"]["skip_rate"] < 0.001
        and row["mcps_diagnostics"]["conflict_rate"] >= 0.05
        for row in rows["proxy_anchor_mcps_pg"]
    )
    inference_config_differences = []
    for seed in range(3):
        reference = rows["proxy_anchor"][seed]["_comparison_config"]
        for arm in ARMS[1:]:
            candidate = rows[arm][seed]["_comparison_config"]
            differing_keys = [
                key for key in INFERENCE_CONFIG_KEYS if candidate[key] != reference[key]
            ]
            if differing_keys:
                inference_config_differences.append(
                    {"arm": arm, "keys": differing_keys, "seed": seed}
                )
    inference_config_identical = not inference_config_differences
    frozen_gate = {
        "diagnostics_pass": diagnostics_pass,
        "positive_seed_count": positive_seed_count,
        "positive_seed_count_pass": positive_seed_count >= 2,
        "paired_mean": paired_mean,
        "paired_standard_error": paired_standard_error,
        "effect_size_pass": paired_mean >= 0.0015
        and paired_mean > paired_standard_error,
        "compactness_control_pass": mcps_vs_control["final_r1"]["mean"] >= 0.0,
        "inference_config_keys": list(INFERENCE_CONFIG_KEYS),
        "inference_config_differences": inference_config_differences,
        "inference_config_identical": inference_config_identical,
    }
    frozen_pass = all(
        (
            frozen_gate["diagnostics_pass"],
            frozen_gate["positive_seed_count_pass"],
            frozen_gate["effect_size_pass"],
            frozen_gate["compactness_control_pass"],
            frozen_gate["inference_config_identical"],
        )
    )
    specific = all(
        comparison[metric]["all_positive"]
        for comparison in (mcps_vs_pa, mcps_vs_control)
        for metric in ("final_r1", "final_map_at_r")
    )
    return {
        "schema_version": 2,
        "study_root": str(root.resolve()),
        "arms": list(ARMS),
        "seeds": [0, 1, 2],
        "rows": [
            {key: value for key, value in row.items() if key != "_comparison_config"}
            for arm in ARMS
            for row in rows[arm]
        ],
        "comparisons": comparisons,
        "frozen_gate": frozen_gate,
        "decision": "MCPS_PASS" if frozen_pass else "MCPS_CLOSE",
        "mechanism_specificity_decision": (
            "NOT_EVALUABLE"
            if not diagnostics_pass or not inference_config_identical
            else "MCPS_SPECIFIC_POSITIVE"
            if specific
            else "MIXED_OR_NULL"
        ),
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
