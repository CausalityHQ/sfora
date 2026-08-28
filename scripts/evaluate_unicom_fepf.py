#!/usr/bin/env python3
"""Evaluate the frozen UniCOM FEPF scientific gates from persisted evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

EVALUATION_EPOCHS = (4, 8, 12, 16)
EPOCH4_MAP_DELTA_MIN = 0.003
EXPLORATORY_MAP_DELTA_MIN = 0.010
COMPUTE_RATIO_MAX = 1.02
CONFIRMATION_PAIRS = (
    (7, 20_260_828),
    (8, 271_828),
    (9, 314_159),
    (10, 1_618_033),
    (11, 57_721),
)
T_CRITICAL_ONE_SIDED_95_DF4 = 2.131846786326649
QUERY_BOOTSTRAP_SEED = 20_260_829
QUERY_BOOTSTRAP_REPLICATES = 10_000
RESULT_KEYS = (
    "schema",
    "phase",
    "status",
    "clause",
    "evaluator_sha256",
    "evidence_sha256",
    "config",
    "sources",
    "decision",
)


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite builtin float")
    return value


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _exploratory_observation(
    value: object, *, expected_mode: str
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("exploratory arm differs")
    if (
        value.get("mode") != expected_mode
        or type(value.get("training_seed")) is not int
        or value["training_seed"] != 0
        or type(value.get("holdout_seed")) is not int
        or value["holdout_seed"] != 0
    ):
        raise ValueError("exploratory arm identity differs")
    history = value.get("history")
    if type(history) is not list or len(history) != len(EVALUATION_EPOCHS):
        raise ValueError("exploratory history differs")
    normalized_history = []
    for expected_epoch, row in zip(EVALUATION_EPOCHS, history, strict=True):
        if (
            type(row) is not dict
            or tuple(row) != ("epoch", "metrics")
            or type(row["epoch"]) is not int
            or row["epoch"] != expected_epoch
            or type(row["metrics"]) is not dict
            or tuple(row["metrics"]) != ("map_at_r", "recall_at_1")
        ):
            raise ValueError("exploratory history differs")
        metrics = {
            "map_at_r": _finite_float(
                row["metrics"].get("map_at_r"), "exploratory mAP"
            ),
            "recall_at_1": _finite_float(
                row["metrics"].get("recall_at_1"), "exploratory Recall@1"
            ),
        }
        if any(metric < 0.0 or metric > 1.0 for metric in metrics.values()):
            raise ValueError("exploratory history metric differs")
        normalized_history.append({"epoch": expected_epoch, "metrics": metrics})
    query_evidence = value.get("query_evidence")
    if type(query_evidence) is not list or not query_evidence:
        raise ValueError("exploratory query evidence differs")
    normalized_queries = []
    paths = []
    for row in query_evidence:
        if (
            type(row) is not dict
            or type(row.get("query_path")) is not str
            or not row["query_path"]
            or type(row.get("top1_correct")) is not bool
        ):
            raise ValueError("exploratory query evidence differs")
        ap_at_r = _finite_float(row.get("ap_at_r"), "exploratory query AP@R")
        if ap_at_r < 0.0 or ap_at_r > 1.0:
            raise ValueError("exploratory query evidence differs")
        paths.append(row["query_path"])
        normalized_queries.append(
            {
                "query_path": row["query_path"],
                "top1_correct": row["top1_correct"],
                "ap_at_r": ap_at_r,
            }
        )
    if len(paths) != len(set(paths)):
        raise ValueError("exploratory query order differs")
    steps = value.get("optimizer_steps_per_epoch")
    if type(steps) is not int or steps <= 0:
        raise ValueError("exploratory optimizer step count differs")
    return {
        "history": normalized_history,
        "query_evidence": normalized_queries,
        "initialization_seconds": _positive_float(
            value.get("initialization_seconds"), "exploratory initialization duration"
        ),
        "optimizer_steps_per_epoch": steps,
        "profiled_step_wall": _positive_float(
            value.get("profiled_step_wall"), "exploratory profiled step wall"
        ),
    }


def _first_attainment(history: list[dict[str, object]], target: float) -> int | None:
    for row in history:
        if row["metrics"]["map_at_r"] >= target:
            return row["epoch"]
    return None


def _profiled_compute(observation: Mapping[str, object], epoch: int) -> float:
    return math.fsum(
        (
            observation["initialization_seconds"],
            epoch
            * observation["optimizer_steps_per_epoch"]
            * observation["profiled_step_wall"],
        )
    )


def evaluate_exploratory(
    control: object, candidate: object, *, structural_all: bool
) -> dict[str, object]:
    """Apply the frozen seed-0 kill and Pareto promotion rules."""

    if type(structural_all) is not bool:
        raise ValueError("exploratory structural predicate differs")
    control_values = _exploratory_observation(control, expected_mode="imprinted")
    candidate_values = _exploratory_observation(
        candidate, expected_mode="fepf_mean"
    )
    control_queries = control_values["query_evidence"]
    candidate_queries = candidate_values["query_evidence"]
    control_paths = tuple(row["query_path"] for row in control_queries)
    candidate_paths = tuple(row["query_path"] for row in candidate_queries)
    if control_paths != candidate_paths:
        raise ValueError("exploratory query order differs")
    gains = 0
    losses = 0
    for control_row, candidate_row in zip(
        control_queries, candidate_queries, strict=True
    ):
        control_correct = control_row["top1_correct"]
        candidate_correct = candidate_row["top1_correct"]
        gains += int(candidate_correct and not control_correct)
        losses += int(control_correct and not candidate_correct)
    control_history = control_values["history"]
    candidate_history = candidate_values["history"]
    epoch4_delta_map = math.fsum(
        (
            candidate_history[0]["metrics"]["map_at_r"],
            -control_history[0]["metrics"]["map_at_r"],
        )
    )
    endpoint_delta_map = math.fsum(
        (
            candidate_history[-1]["metrics"]["map_at_r"],
            -control_history[-1]["metrics"]["map_at_r"],
        )
    )
    endpoint_delta_r1 = math.fsum(
        (
            candidate_history[-1]["metrics"]["recall_at_1"],
            -control_history[-1]["metrics"]["recall_at_1"],
        )
    )
    target = control_history[-1]["metrics"]["map_at_r"]
    control_epoch = _first_attainment(control_history, target)
    candidate_epoch = _first_attainment(candidate_history, target)
    if control_epoch is None:
        raise ValueError("exploratory control endpoint is unattained")
    control_compute = _profiled_compute(control_values, control_epoch)
    candidate_compute = (
        None
        if candidate_epoch is None
        else _profiled_compute(candidate_values, candidate_epoch)
    )
    predicates = {
        "epoch4_delta_map_at_least_0_003": epoch4_delta_map
        >= EPOCH4_MAP_DELTA_MIN,
        "endpoint_delta_map_at_least_0_010": endpoint_delta_map
        >= EXPLORATORY_MAP_DELTA_MIN,
        "endpoint_delta_r1_positive": endpoint_delta_r1 > 0.0,
        "losses_no_more_than_one_fifth_of_gains": losses <= gains // 5,
        "candidate_attained_control_endpoint": candidate_epoch is not None,
        "candidate_attained_no_later_than_control": candidate_epoch is not None
        and candidate_epoch <= control_epoch,
        "compute_within_1_02": candidate_compute is not None
        and candidate_compute <= COMPUTE_RATIO_MAX * control_compute,
        "structural_all": structural_all,
    }
    if not predicates["epoch4_delta_map_at_least_0_003"]:
        decision = "CLOSE_EPOCH4"
    elif not predicates["endpoint_delta_map_at_least_0_010"]:
        decision = "CLOSE_MARGINAL"
    elif all(predicates.values()):
        decision = "PROMOTE"
    else:
        decision = "CLOSE_NONPARETO"
    return {
        "decision": decision,
        "clause": decision,
        "epoch4_delta_map": epoch4_delta_map,
        "epoch4_pass": predicates["epoch4_delta_map_at_least_0_003"],
        "endpoint_delta_map": endpoint_delta_map,
        "endpoint_delta_r1": endpoint_delta_r1,
        "gains": gains,
        "losses": losses,
        "control_first_attainment_epoch": control_epoch,
        "candidate_first_attainment_epoch": candidate_epoch,
        "candidate_right_censored": candidate_epoch is None,
        "control_profiled_compute": control_compute,
        "candidate_profiled_compute": candidate_compute,
        "predicates": predicates,
    }


def _five_float_values(values: object, name: str) -> tuple[float, ...]:
    if type(values) not in (tuple, list) or len(values) != 5:
        raise ValueError(f"{name} must contain exactly five values")
    return tuple(_finite_float(value, name) for value in values)


def one_sided_t_lower_bound(values: object) -> dict[str, float]:
    """Return the frozen one-sided paired-t lower-bound components."""

    observed = _five_float_values(values, "paired t values")
    mean = math.fsum(observed) / 5
    sample_std = math.sqrt(
        math.fsum((value - mean) * (value - mean) for value in observed) / 4
    )
    return {
        "mean": mean,
        "sample_std": sample_std,
        "lower_bound": mean
        - T_CRITICAL_ONE_SIDED_95_DF4 * sample_std / math.sqrt(5),
    }


def _one_sided_t_upper_bound(values: object) -> dict[str, float]:
    observed = _five_float_values(values, "paired resource log ratios")
    mean = math.fsum(observed) / 5
    sample_std = math.sqrt(
        math.fsum((value - mean) * (value - mean) for value in observed) / 4
    )
    return {
        "mean": mean,
        "sample_std": sample_std,
        "upper_bound": mean
        + T_CRITICAL_ONE_SIDED_95_DF4 * sample_std / math.sqrt(5),
    }


def query_bootstrap(pair_query_deltas: object) -> dict[str, object]:
    """Run the frozen nested paired-draw/query PCG64 sensitivity bootstrap."""

    if type(pair_query_deltas) not in (tuple, list) or len(pair_query_deltas) != 5:
        raise ValueError("query bootstrap pair inventory differs")
    normalized = []
    for pair in pair_query_deltas:
        if type(pair) not in (tuple, list) or not pair:
            raise ValueError("query bootstrap query inventory differs")
        normalized.append(tuple(_finite_float(value, "query bootstrap delta") for value in pair))
    generator = np.random.Generator(np.random.PCG64(QUERY_BOOTSTRAP_SEED))
    distribution = np.empty(QUERY_BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(QUERY_BOOTSTRAP_REPLICATES):
        selected_pairs = generator.integers(0, 5, size=5)
        selected_means = []
        for pair_index in selected_pairs:
            values = normalized[int(pair_index)]
            selected_queries = generator.integers(0, len(values), size=len(values))
            selected_means.append(
                math.fsum(values[int(index)] for index in selected_queries)
                / len(values)
            )
        distribution[replicate] = math.fsum(selected_means) / 5
    interval = np.quantile(distribution, (0.025, 0.975), method="linear")
    return {
        "bit_generator": "PCG64",
        "seed": QUERY_BOOTSTRAP_SEED,
        "replicates": QUERY_BOOTSTRAP_REPLICATES,
        "resampling_order": "paired_draw_then_queries_within_selected_draw",
        "quantile_method": "linear",
        "values": distribution.tolist(),
        "values_sha256": hashlib.sha256(distribution.tobytes(order="C")).hexdigest(),
        "interval": [float(interval[0]), float(interval[1])],
    }


def _confirmation_observation(
    value: object,
    *,
    expected_mode: str,
    training_seed: int,
    holdout_seed: int,
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("confirmation arm differs")
    if (
        value.get("mode") != expected_mode
        or type(value.get("training_seed")) is not int
        or value["training_seed"] != training_seed
        or type(value.get("holdout_seed")) is not int
        or value["holdout_seed"] != holdout_seed
    ):
        raise ValueError("confirmation arm identity differs")
    history = value.get("history")
    if type(history) is not list or len(history) != 4:
        raise ValueError("confirmation history differs")
    normalized_history = []
    for expected_epoch, row in zip(EVALUATION_EPOCHS, history, strict=True):
        if (
            type(row) is not dict
            or tuple(row) != ("epoch", "metrics")
            or type(row["epoch"]) is not int
            or row["epoch"] != expected_epoch
            or type(row["metrics"]) is not dict
            or tuple(row["metrics"]) != ("map_at_r", "recall_at_1")
        ):
            raise ValueError("confirmation history differs")
        map_at_r = _finite_float(row["metrics"].get("map_at_r"), "confirmation mAP")
        recall = _finite_float(
            row["metrics"].get("recall_at_1"), "confirmation Recall@1"
        )
        if not (0.0 <= map_at_r <= 1.0 and 0.0 <= recall <= 1.0):
            raise ValueError("confirmation metric differs")
        normalized_history.append(
            {
                "epoch": expected_epoch,
                "metrics": {"map_at_r": map_at_r, "recall_at_1": recall},
            }
        )
    queries = value.get("query_evidence")
    if type(queries) is not list or not queries:
        raise ValueError("confirmation query evidence differs")
    normalized_queries = []
    paths = []
    for row in queries:
        if (
            type(row) is not dict
            or type(row.get("query_path")) is not str
            or not row["query_path"]
            or type(row.get("top1_correct")) is not bool
        ):
            raise ValueError("confirmation query evidence differs")
        ap_at_r = _finite_float(row.get("ap_at_r"), "confirmation query AP@R")
        if not 0.0 <= ap_at_r <= 1.0:
            raise ValueError("confirmation query evidence differs")
        paths.append(row["query_path"])
        normalized_queries.append(
            {
                "query_path": row["query_path"],
                "top1_correct": row["top1_correct"],
                "ap_at_r": ap_at_r,
            }
        )
    if len(paths) != len(set(paths)):
        raise ValueError("confirmation query order differs")
    steps = value.get("optimizer_steps_per_epoch")
    if type(steps) is not int or steps <= 0:
        raise ValueError("confirmation optimizer step count differs")
    allocated = value.get("peak_allocated_bytes")
    reserved = value.get("peak_reserved_bytes")
    if (
        type(allocated) is not int
        or allocated <= 0
        or type(reserved) is not int
        or reserved <= 0
        or reserved < allocated
    ):
        raise ValueError("confirmation memory profile differs")
    return {
        "history": normalized_history,
        "query_evidence": normalized_queries,
        "initialization_seconds": _positive_float(
            value.get("initialization_seconds"), "confirmation initialization duration"
        ),
        "optimizer_steps_per_epoch": steps,
        "profiled_step_wall": _positive_float(
            value.get("profiled_step_wall"), "confirmation profiled step wall"
        ),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }


def evaluate_confirmation(pairs: object) -> dict[str, object]:
    """Apply every frozen five-pair confirmation and resource predicate."""

    if type(pairs) is not tuple or len(pairs) != len(CONFIRMATION_PAIRS):
        raise ValueError("confirmation pair order differs")
    observed_identities = tuple(
        (row.get("training_seed"), row.get("holdout_seed"))
        if type(row) is dict
        else (None, None)
        for row in pairs
    )
    if observed_identities != CONFIRMATION_PAIRS or any(
        type(value) is not int for identity in observed_identities for value in identity
    ):
        raise ValueError("confirmation pair order differs")
    pair_rows = []
    map_deltas = []
    recall_deltas = []
    pair_query_deltas = []
    step_log_ratios = []
    allocated_log_ratios = []
    reserved_log_ratios = []
    for pair, (training_seed, holdout_seed) in zip(
        pairs, CONFIRMATION_PAIRS, strict=True
    ):
        if type(pair.get("structural_equal")) is not bool:
            raise ValueError("confirmation structural predicate differs")
        control = _confirmation_observation(
            pair.get("control"),
            expected_mode="imprinted",
            training_seed=training_seed,
            holdout_seed=holdout_seed,
        )
        candidate = _confirmation_observation(
            pair.get("candidate"),
            expected_mode="fepf_mean",
            training_seed=training_seed,
            holdout_seed=holdout_seed,
        )
        control_queries = control["query_evidence"]
        candidate_queries = candidate["query_evidence"]
        if tuple(row["query_path"] for row in control_queries) != tuple(
            row["query_path"] for row in candidate_queries
        ):
            raise ValueError("confirmation query order differs")
        query_deltas = tuple(
            float(np.float64(candidate_row["ap_at_r"]) - np.float64(control_row["ap_at_r"]))
            for control_row, candidate_row in zip(
                control_queries, candidate_queries, strict=True
            )
        )
        pair_query_deltas.append(query_deltas)
        map_delta = float(
            np.float64(candidate["history"][-1]["metrics"]["map_at_r"])
            - np.float64(control["history"][-1]["metrics"]["map_at_r"])
        )
        recall_delta = float(
            np.float64(candidate["history"][-1]["metrics"]["recall_at_1"])
            - np.float64(control["history"][-1]["metrics"]["recall_at_1"])
        )
        map_deltas.append(map_delta)
        recall_deltas.append(recall_delta)
        target = control["history"][-1]["metrics"]["map_at_r"]
        control_epoch = _first_attainment(control["history"], target)
        candidate_epoch = _first_attainment(candidate["history"], target)
        if control_epoch is None:
            raise ValueError("confirmation control endpoint is unattained")
        control_compute = _profiled_compute(control, control_epoch)
        candidate_compute = (
            None
            if candidate_epoch is None
            else _profiled_compute(candidate, candidate_epoch)
        )
        compute_pass = candidate_compute is not None and (
            candidate_compute <= COMPUTE_RATIO_MAX * control_compute
        )
        step_log = math.log(
            candidate["profiled_step_wall"] / control["profiled_step_wall"]
        )
        allocated_log = math.log(
            candidate["peak_allocated_bytes"] / control["peak_allocated_bytes"]
        )
        reserved_log = math.log(
            candidate["peak_reserved_bytes"] / control["peak_reserved_bytes"]
        )
        step_log_ratios.append(step_log)
        allocated_log_ratios.append(allocated_log)
        reserved_log_ratios.append(reserved_log)
        pair_rows.append(
            {
                "training_seed": training_seed,
                "holdout_seed": holdout_seed,
                "map_delta": map_delta,
                "recall_at_1_delta": recall_delta,
                "control_first_attainment_epoch": control_epoch,
                "candidate_first_attainment_epoch": candidate_epoch,
                "control_profiled_compute": control_compute,
                "candidate_profiled_compute": candidate_compute,
                "compute_within_1_02": compute_pass,
                "step_log_ratio": step_log,
                "allocated_log_ratio": allocated_log,
                "reserved_log_ratio": reserved_log,
                "structural_equal": pair["structural_equal"],
                "query_ap_deltas": list(query_deltas),
            }
        )
    map_t = one_sided_t_lower_bound(tuple(map_deltas))
    recall_t = one_sided_t_lower_bound(tuple(recall_deltas))
    sorted_maps = sorted(map_deltas)
    map_median = sorted_maps[2]
    leave_one_out = tuple(
        math.fsum(value for other, value in enumerate(map_deltas) if other != index)
        / 4
        for index in range(5)
    )
    step_summary = _one_sided_t_upper_bound(tuple(step_log_ratios))
    allocated_summary = _one_sided_t_upper_bound(tuple(allocated_log_ratios))
    reserved_summary = _one_sided_t_upper_bound(tuple(reserved_log_ratios))
    resource_limit = math.log(COMPUTE_RATIO_MAX)
    predicates = {
        "mean_map_delta_at_least_0_010": map_t["mean"] >= 0.010,
        "mean_recall_at_1_delta_at_least_0_005": recall_t["mean"] >= 0.005,
        "all_map_deltas_positive": all(value > 0.0 for value in map_deltas),
        "all_recall_at_1_deltas_positive": all(
            value > 0.0 for value in recall_deltas
        ),
        "map_t_lower_positive": map_t["lower_bound"] > 0.0,
        "median_map_delta_at_least_0_008": map_median >= 0.008,
        "all_leave_one_out_map_means_at_least_0_008": all(
            value >= 0.008 for value in leave_one_out
        ),
        "all_profiled_compute_within_1_02": all(
            row["compute_within_1_02"] for row in pair_rows
        ),
        "step_log_upper_within_log_1_02": step_summary["upper_bound"]
        <= resource_limit,
        "allocated_log_upper_within_log_1_02": allocated_summary["upper_bound"]
        <= resource_limit,
        "reserved_log_upper_within_log_1_02": reserved_summary["upper_bound"]
        <= resource_limit,
        "cross_arm_structure_equal": all(
            row["structural_equal"] for row in pair_rows
        ),
    }
    decision = "CONFIRM" if all(predicates.values()) else "CLOSE_CONFIRMATION"
    return {
        "decision": decision,
        "clause": decision,
        "pairs": pair_rows,
        "statistics": {
            "map": {
                "mean": map_t["mean"],
                "sample_std": map_t["sample_std"],
                "t_lower_bound": map_t["lower_bound"],
                "median": map_median,
                "leave_one_out_means": list(leave_one_out),
            },
            "recall_at_1": {
                "mean": recall_t["mean"],
                "sample_std": recall_t["sample_std"],
                "t_lower_bound": recall_t["lower_bound"],
            },
            "resource_log_ratios": {
                "step_wall": step_summary,
                "peak_allocated": allocated_summary,
                "peak_reserved": reserved_summary,
                "upper_limit": resource_limit,
            },
        },
        "bootstrap": query_bootstrap(tuple(pair_query_deltas)),
        "predicates": predicates,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_typed_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return tuple(left) == tuple(right) and all(
            _strict_typed_equal(left[key], right[key]) for key in left
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            _strict_typed_equal(first, second)
            for first, second in zip(left, right, strict=True)
        )
    return left == right


def _strict_json_bytes(payload: bytes, *, canonical: bool = True) -> object:
    if type(payload) is not bytes:
        raise TypeError("strict JSON payload must be bytes")

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("strict JSON has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("strict JSON has nonfinite value")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("strict JSON differs") from error
    if canonical:
        encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        if payload != encoded:
            raise ValueError("strict JSON is noncanonical")
    return value


def _strict_json_file(path: Path, *, canonical: bool = True) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError("external JSON authority differs")
    return _strict_json_bytes(path.read_bytes(), canonical=canonical)


def _validate_evidence_root(evidence_root: Path) -> Path:
    if not isinstance(evidence_root, Path):
        raise TypeError("FEPF evidence root must be a Path")
    absolute = evidence_root.absolute()
    resolved = evidence_root.resolve()
    if absolute != resolved or resolved.is_symlink() or not resolved.is_dir():
        raise ValueError("FEPF evidence root differs")
    return resolved


def _relative_parts(value: object, name: str) -> tuple[str, ...]:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{name} path differs")
    path = Path(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{name} path differs")
    return path.parts


def _resolve_descendant(
    root: Path, relative_value: object, *, name: str, directory: bool
) -> Path:
    parts = _relative_parts(relative_value, name)
    unresolved = root
    for part in parts:
        unresolved = unresolved / part
        if unresolved.is_symlink():
            raise ValueError(f"{name} path differs")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} path differs") from error
    if directory:
        valid = resolved.is_dir() and not resolved.is_symlink()
    else:
        valid = resolved.is_file() and not resolved.is_symlink()
    if not valid:
        raise ValueError(f"{name} path differs")
    return resolved


def _validate_source_inventory(
    phase: str, sources: object
) -> list[dict[str, object]]:
    expected_pairs = ((0, 0),) if phase == "exploratory" else CONFIRMATION_PAIRS
    if type(sources) is not list or len(sources) != len(expected_pairs):
        raise ValueError("FEPF result source inventory differs")
    normalized = []
    for source, expected_pair in zip(sources, expected_pairs, strict=True):
        if (
            type(source) is not dict
            or tuple(source)
            != (
                "training_seed",
                "holdout_seed",
                "control_root",
                "candidate_root",
                "quality_profiles",
            )
            or type(source["training_seed"]) is not int
            or type(source["holdout_seed"]) is not int
            or (source["training_seed"], source["holdout_seed"]) != expected_pair
            or type(source["quality_profiles"]) is not list
            or len(source["quality_profiles"]) != 4
        ):
            raise ValueError("FEPF result source pair order differs")
        _relative_parts(source["control_root"], "control root")
        _relative_parts(source["candidate_root"], "candidate root")
        for path in source["quality_profiles"]:
            _relative_parts(path, "quality profile")
        if len(set(source["quality_profiles"])) != 4:
            raise ValueError("FEPF result quality profile order differs")
        normalized.append(dict(source))
    return normalized


def _load_script_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(specification)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
    return module


def _authority_modules():
    repository = Path(__file__).resolve().parents[1]
    trainer = _load_script_module(
        repository / "scripts" / "train_unicom_inshop.py",
        "_fepf_evaluator_trainer_authority",
    )
    profiler = _load_script_module(
        repository / "scripts" / "profile_unicom_training_step.py",
        "_fepf_evaluator_profile_authority",
    )
    return trainer, profiler


def _bound_file(
    binding: object,
    *,
    current_root: Path,
    parent_root: Path | None,
    name: str,
) -> Path:
    if (
        type(binding) is not dict
        or tuple(binding) != ("root", "path", "sha256", "bytes")
        or binding["root"] not in {"current", "parent"}
        or not _lower_sha256(binding["sha256"])
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
    ):
        raise ValueError(f"{name} binding differs")
    root = current_root if binding["root"] == "current" else parent_root
    if root is None:
        raise ValueError(f"{name} root differs")
    path = _resolve_descendant(root, binding["path"], name=name, directory=False)
    if path.stat().st_size != binding["bytes"] or _sha256_file(path) != binding["sha256"]:
        raise ValueError(f"{name} bytes differ")
    return path


def _parent_evidence_root(run_receipt: dict[str, object], current_root: Path) -> Path | None:
    parent = run_receipt.get("parent_evidence_root")
    if parent is None:
        return None
    if (
        type(parent) is not dict
        or tuple(parent) != ("kind", "path")
        or parent["kind"] != "relative"
        or type(parent["path"]) is not str
    ):
        raise ValueError("FEPF parent evidence root differs")
    relative = Path(parent["path"])
    resolved = (current_root / relative).resolve()
    if (
        relative != Path("..") / resolved.name
        or resolved.parent != current_root.parent
        or resolved == current_root
        or resolved.is_symlink()
        or not resolved.is_dir()
    ):
        raise ValueError("FEPF parent evidence root differs")
    return resolved


def _schedule_sha256(protocol: Mapping[str, object]) -> str:
    payload = (
        protocol["epochs"],
        protocol["batch_size"],
        protocol["workers"],
        protocol["learning_rate"],
        protocol["classifier_learning_rate"],
        protocol["margin"],
        protocol["scale"],
        protocol["objective"],
        protocol["selected_features"],
        protocol["evaluation_features"],
        protocol["eval_every"],
        protocol["checkpoint_every"],
        protocol["max_steps"],
        protocol["bf16"],
    )
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def _initialization_duration(
    receipt: object,
    *,
    run_receipt: dict[str, object],
    config_sha256: str,
) -> float:
    protocol = run_receipt["training_protocol"]
    if (
        type(receipt) is not dict
        or receipt.get("schema") != "initialization-receipt-v2"
        or receipt.get("mode") != run_receipt["mode"]
        or receipt.get("training_seed") != run_receipt["training_seed"]
        or receipt.get("holdout_fraction") != run_receipt["holdout_fraction"]
        or receipt.get("holdout_seed") != run_receipt["holdout_seed"]
        or receipt.get("source_sha256") != protocol["trainer_sha256"]
        or receipt.get("checkpoint_sha256") != protocol["initial_checkpoint_sha256"]
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("schedule_sha256") != _schedule_sha256(protocol)
        or receipt.get("row_norm_rtol") != 2e-6
        or receipt.get("row_norm_atol") != 2e-7
    ):
        raise ValueError("FEPF initialization provenance differs")
    return _positive_float(
        receipt.get("initialization_seconds"), "FEPF initialization duration"
    )


def _history_observation(history: object) -> tuple[list[dict[str, object]], int]:
    if type(history) is not list or len(history) != 16:
        raise ValueError("FEPF history differs")
    evaluation_history = []
    steps = []
    for expected_epoch, row in enumerate(history, start=1):
        if (
            type(row) is not dict
            or tuple(row) != ("epoch", "train", "metrics")
            or type(row["epoch"]) is not int
            or row["epoch"] != expected_epoch
            or type(row["train"]) is not dict
            or type(row["train"].get("steps")) is not int
            or row["train"]["steps"] <= 0
        ):
            raise ValueError("FEPF history differs")
        steps.append(row["train"]["steps"])
        if expected_epoch in EVALUATION_EPOCHS:
            metrics = row["metrics"]
            if type(metrics) is not dict:
                raise ValueError("FEPF evaluation history differs")
            evaluation_history.append(
                {
                    "epoch": expected_epoch,
                    "metrics": {
                        "map_at_r": _finite_float(metrics.get("map_at_r"), "FEPF mAP"),
                        "recall_at_1": _finite_float(
                            metrics.get("recall_at_1"), "FEPF Recall@1"
                        ),
                    },
                }
            )
        elif row["metrics"] is not None:
            raise ValueError("FEPF evaluation cadence differs")
    if len(set(steps)) != 1:
        raise ValueError("FEPF optimizer steps per epoch differ")
    return evaluation_history, steps[0]


def _query_observation(evaluation: object, *, expected_epoch: int) -> list[dict[str, object]]:
    if (
        type(evaluation) is not dict
        or evaluation.get("epoch") != expected_epoch
        or type(evaluation.get("query_evidence")) is not list
        or not evaluation["query_evidence"]
    ):
        raise ValueError("FEPF terminal query evidence differs")
    result = []
    for row in evaluation["query_evidence"]:
        ranked = row.get("ranked_prefix") if type(row) is dict else None
        if (
            type(row) is not dict
            or type(row.get("query_path")) is not str
            or type(row.get("ap_at_r")) is not float
            or not math.isfinite(row["ap_at_r"])
            or type(ranked) is not list
            or not ranked
            or type(ranked[0]) is not dict
            or type(ranked[0].get("correct")) is not bool
        ):
            raise ValueError("FEPF terminal query evidence differs")
        result.append(
            {
                "query_path": row["query_path"],
                "top1_correct": ranked[0]["correct"],
                "ap_at_r": row["ap_at_r"],
            }
        )
    return result


def _pooled_quality_profile(
    receipts: tuple[dict[str, object], dict[str, object]]
) -> tuple[float, int, int]:
    samples = []
    for receipt in receipts:
        timing = receipt.get("timing_samples")
        if type(timing) is not list or len(timing) != 50:
            raise ValueError("FEPF quality profile timing differs")
        for row in timing:
            samples.append(
                _positive_float(
                    row.get("step_wall_seconds") if type(row) is dict else None,
                    "FEPF quality step wall",
                )
            )
    return (
        float(np.median(np.asarray(samples, dtype=np.float64))),
        max(receipt["peak_allocated_bytes"] for receipt in receipts),
        max(receipt["peak_reserved_bytes"] for receipt in receipts),
    )


def _load_arm_observation(
    *,
    run_root: Path,
    profiles: tuple[dict[str, object], dict[str, object]],
    expected_mode: str,
    training_seed: int,
    holdout_seed: int,
    trainer,
) -> dict[str, object]:
    run_path = run_root / "run-receipt.json"
    run_receipt = _strict_json_file(run_path)
    if type(run_receipt) is not dict:
        raise ValueError("FEPF run receipt differs")
    trainer.validate_training_run_receipt_v2(run_receipt, evidence_root=run_root)
    if (
        run_receipt.get("mode") != expected_mode
        or run_receipt.get("training_seed") != training_seed
        or run_receipt.get("holdout_seed") != holdout_seed
        or run_receipt.get("stop_after_epoch") != 16
    ):
        raise ValueError("FEPF run identity differs")
    history_path = _bound_file(
        run_receipt["history"],
        current_root=run_root,
        parent_root=None,
        name="history",
    )
    history = _strict_json_file(history_path)
    trainer.validate_fepf_result(history, run_root)
    normalized_history, steps = _history_observation(history)
    parent_root = _parent_evidence_root(run_receipt, run_root)
    initialization_path = _bound_file(
        run_receipt["initialization_receipt"],
        current_root=run_root,
        parent_root=parent_root,
        name="initialization receipt",
    )
    initialization = _strict_json_file(initialization_path)
    config_authority = profiles[0]["config"]
    initialization_seconds = _initialization_duration(
        initialization,
        run_receipt=run_receipt,
        config_sha256=config_authority["sha256"],
    )
    terminal = run_receipt["evaluations"][-1]
    terminal_path = _bound_file(
        {key: terminal[key] for key in ("root", "path", "sha256", "bytes")},
        current_root=run_root,
        parent_root=parent_root,
        name="terminal evaluation",
    )
    evaluation = _strict_json_file(terminal_path)
    query_evidence = _query_observation(evaluation, expected_epoch=16)
    step_wall, allocated, reserved = _pooled_quality_profile(profiles)
    return {
        "mode": expected_mode,
        "training_seed": training_seed,
        "holdout_seed": holdout_seed,
        "history": normalized_history,
        "query_evidence": query_evidence,
        "initialization_seconds": initialization_seconds,
        "optimizer_steps_per_epoch": steps,
        "profiled_step_wall": step_wall,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
    }


def _reload_registered_pairs(
    *,
    observed_sources: list[dict[str, object]],
    evidence_root: Path,
    phase: str,
) -> tuple[tuple[dict[str, object], ...], dict[str, object], str]:
    trainer, profiler = _authority_modules()
    pairs = []
    common_config = None
    evidence_digest = hashlib.sha256(b"unicom-fepf-source-evidence-v1\0")
    for source in observed_sources:
        source_payload = json.dumps(
            source, allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode()
        evidence_digest.update(len(source_payload).to_bytes(8, "big"))
        evidence_digest.update(source_payload)
        control_root = _resolve_descendant(
            evidence_root, source["control_root"], name="control root", directory=True
        )
        candidate_root = _resolve_descendant(
            evidence_root,
            source["candidate_root"],
            name="candidate root",
            directory=True,
        )
        profile_receipts = []
        authority_paths = [
            control_root / "run-receipt.json",
            candidate_root / "run-receipt.json",
        ]
        for relative in source["quality_profiles"]:
            profile_path = _resolve_descendant(
                evidence_root, relative, name="quality profile", directory=False
            )
            authority_paths.append(profile_path)
            profile = _strict_json_file(profile_path)
            if type(profile) is not dict:
                raise ValueError("FEPF quality profile differs")
            profiler.validate_quality_profile(profile)
            profile_receipts.append(profile)
        for authority_path in authority_paths:
            payload = authority_path.read_bytes()
            evidence_digest.update(len(payload).to_bytes(8, "big"))
            evidence_digest.update(payload)
        expected_run_paths = (
            control_root / "run-receipt.json",
            candidate_root / "run-receipt.json",
            candidate_root / "run-receipt.json",
            control_root / "run-receipt.json",
        )
        for profile, expected_run in zip(
            profile_receipts, expected_run_paths, strict=True
        ):
            authority = profile.get("run_receipt")
            if (
                type(authority) is not dict
                or Path(authority.get("path", "")) != expected_run
            ):
                raise ValueError("FEPF quality profile arm order differs")
            if common_config is None:
                common_config = dict(profile["config"])
            elif not _strict_typed_equal(profile["config"], common_config):
                raise ValueError("FEPF quality profile config differs")
        control_profiles = (profile_receipts[0], profile_receipts[3])
        candidate_profiles = (profile_receipts[1], profile_receipts[2])
        training_seed = source["training_seed"]
        holdout_seed = source["holdout_seed"]
        control = _load_arm_observation(
            run_root=control_root,
            profiles=control_profiles,
            expected_mode="imprinted",
            training_seed=training_seed,
            holdout_seed=holdout_seed,
            trainer=trainer,
        )
        candidate = _load_arm_observation(
            run_root=candidate_root,
            profiles=candidate_profiles,
            expected_mode="fepf_mean",
            training_seed=training_seed,
            holdout_seed=holdout_seed,
            trainer=trainer,
        )
        structural_equal = True
        try:
            trainer.require_cross_arm_inference_equality(
                profile_receipts[0]["inference_signature"],
                profile_receipts[1]["inference_signature"],
            )
        except ValueError:
            structural_equal = False
        pairs.append(
            {
                "training_seed": training_seed,
                "holdout_seed": holdout_seed,
                "control": control,
                "candidate": candidate,
                "structural_equal": structural_equal,
            }
        )
    if common_config is None:
        raise ValueError("FEPF result config authority differs")
    return tuple(pairs), common_config, evidence_digest.hexdigest()


def _recomputed_result(
    *, phase: str, sources: object, evidence_root: Path
) -> dict[str, object]:
    if phase not in {"exploratory", "confirmation"}:
        raise ValueError("FEPF result phase differs")
    root = _validate_evidence_root(evidence_root)
    normalized_sources = _validate_source_inventory(phase, sources)
    pairs, config, evidence_sha256 = _reload_registered_pairs(
        observed_sources=normalized_sources,
        evidence_root=root,
        phase=phase,
    )
    if (
        type(config) is not dict
        or tuple(config) != ("path", "sha256", "bytes")
        or type(config["path"]) is not str
        or not _lower_sha256(config["sha256"])
        or type(config["bytes"]) is not int
        or config["bytes"] <= 0
    ):
        raise ValueError("FEPF result config authority differs")
    if phase == "exploratory":
        pair = pairs[0]
        decision = evaluate_exploratory(
            pair["control"],
            pair["candidate"],
            structural_all=pair["structural_equal"],
        )
    else:
        decision = evaluate_confirmation(pairs)
    return {
        "schema": "unicom-fepf-result-v1",
        "phase": phase,
        "status": decision["decision"],
        "clause": decision["clause"],
        "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
        "evidence_sha256": evidence_sha256,
        "config": dict(config),
        "sources": normalized_sources,
        "decision": decision,
    }


def build_fepf_result(
    *, phase: str, sources: object, evidence_root: Path
) -> dict[str, object]:
    """Build a result only after reloading all registered external evidence."""

    return _recomputed_result(phase=phase, sources=sources, evidence_root=evidence_root)


def validate_fepf_result(result: object, evidence_root: Path) -> None:
    """Strictly reload every external input and recompute the complete result."""

    if (
        type(result) is not dict
        or tuple(result) != RESULT_KEYS
        or result.get("schema") != "unicom-fepf-result-v1"
        or result.get("phase") not in {"exploratory", "confirmation"}
    ):
        raise ValueError("FEPF result schema differs")
    expected = _recomputed_result(
        phase=result["phase"],
        sources=result["sources"],
        evidence_root=evidence_root,
    )
    if not _strict_typed_equal(result, expected):
        raise ValueError("FEPF result recomputation differs")


def _path_lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def write_fepf_result_atomic(
    result: object,
    output: Path,
    temporary: Path,
    evidence_root: Path,
) -> dict[str, object]:
    """Publish canonical JSON through an exclusive temp and no-replace link."""

    if not isinstance(output, Path) or not isinstance(temporary, Path):
        raise TypeError("FEPF result publication paths must be Paths")
    root = _validate_evidence_root(evidence_root)
    if (
        output == temporary
        or output.parent.absolute() != root
        or temporary.parent.absolute() != root
        or output.parent.resolve() != root
        or temporary.parent.resolve() != root
    ):
        raise ValueError("FEPF result publication path differs")
    validate_fepf_result(result, root)
    if _path_lexists(output):
        raise FileExistsError(output)
    if _path_lexists(temporary):
        raise FileExistsError(temporary)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    directory_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    linked = False
    owned: tuple[int, int] | None = None
    completed = False
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_info = temporary.stat()
        owned = (temporary_info.st_dev, temporary_info.st_ino)
        persisted = _strict_json_file(temporary)
        if not _strict_typed_equal(persisted, result):
            raise RuntimeError("persisted FEPF result bytes differ")
        validate_fepf_result(persisted, root)
        os.link(temporary, output)
        linked = True
        os.fsync(directory_descriptor)
        temporary.unlink()
        os.fsync(directory_descriptor)
        published = _strict_json_file(output)
        output_info = output.stat()
        if (
            owned != (output_info.st_dev, output_info.st_ino)
            or output.read_bytes() != payload
        ):
            raise RuntimeError("published FEPF result bytes differ")
        validate_fepf_result(published, root)
        completed = True
        return published
    finally:
        if _path_lexists(temporary):
            temporary.unlink()
        if linked and not completed and owned is not None and _path_lexists(output):
            info = output.stat()
            if (info.st_dev, info.st_ino) == owned:
                output.unlink()
                os.fsync(directory_descriptor)
        os.close(directory_descriptor)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("exploratory", "confirmation"), required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        sources = _strict_json_file(args.sources)
        result = build_fepf_result(
            phase=args.phase,
            sources=sources,
            evidence_root=args.evidence_root,
        )
        write_fepf_result_atomic(
            result,
            args.output,
            args.temporary,
            args.evidence_root,
        )
    except Exception as error:
        print(f"FEPF evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
