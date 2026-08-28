#!/usr/bin/env python3
"""Build the audited five-seed UniCOM full-width confirmation result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

CONFIRMATION_SEEDS = (2, 3, 4, 5, 6)
ARMS = ("sampled_512", "full_768")
EPOCHS = (4, 8, 12, 16)
BOOTSTRAP_SEED = 768
BOOTSTRAP_REPLICATES = 10_000
PAIRED_T_CRITICAL = 2.7764451052
INPUT_BINDING_KEYS = ("path", "sha256", "bytes")
TOP_KEYS = (
    "schema_version",
    "inputs",
    "metric_authority",
    "quality",
    "operational",
    "decision",
    "status",
)


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py").resolve()
    spec = importlib.util.spec_from_file_location(f"_{name}_for_confirmation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError(f"loaded {name} source differs")
    return module


EVALUATOR = _load_sibling("evaluate_unicom_full_width_objective")
COMPARATOR = _load_sibling("compare_unicom_full_width_profiles")
DECIDER = _load_sibling("decide_unicom_full_width_objective")
TRAINER = _load_sibling("train_unicom_inshop")


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    return value


def _bootstrap_vector(values: np.ndarray) -> tuple[float, float]:
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 64):
        stop = min(start + 64, BOOTSTRAP_REPLICATES)
        indices = generator.integers(
            0, values.size, size=(stop - start, values.size)
        )
        replicates[start:stop] = values[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.percentile(replicates, (2.5, 97.5))
    return (float(lower), float(upper))


def pooled_query_bootstrap(
    control: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Bootstrap paired AP@R queries with one draw shared across training seeds."""

    if type(control) is not np.ndarray or type(candidate) is not np.ndarray:
        raise TypeError("paired AP evidence must be NumPy arrays")
    if control.dtype != np.float64 or candidate.dtype != np.float64:
        raise TypeError("paired AP evidence must be float64")
    if (
        control.ndim != 2
        or control.shape[0] != len(CONFIRMATION_SEEDS)
        or candidate.shape != control.shape
        or control.shape[1] == 0
        or not np.isfinite(control).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("paired AP evidence shape or values differ")
    if type(seed) is not int or seed != BOOTSTRAP_SEED:
        raise ValueError("paired bootstrap seed differs")
    if type(samples) is not int or samples != BOOTSTRAP_REPLICATES:
        raise ValueError("paired bootstrap replicate count differs")
    per_query = (candidate - control).mean(axis=0, dtype=np.float64)
    return _bootstrap_vector(per_query)


def build_quality_summary(
    rows: tuple[dict[str, object], ...],
    *,
    control_average_precision: np.ndarray,
    candidate_average_precision: np.ndarray,
) -> dict[str, object]:
    """Build the evidence-supported portion of the five-seed decision."""

    if type(rows) is not tuple or len(rows) != len(CONFIRMATION_SEEDS):
        raise ValueError("confirmation rows differ")
    if (
        type(control_average_precision) is not np.ndarray
        or type(candidate_average_precision) is not np.ndarray
        or control_average_precision.dtype != np.float64
        or candidate_average_precision.dtype != np.float64
        or control_average_precision.ndim != 2
        or control_average_precision.shape[0] != len(CONFIRMATION_SEEDS)
        or candidate_average_precision.shape != control_average_precision.shape
        or control_average_precision.shape[1] == 0
        or not np.isfinite(control_average_precision).all()
        or not np.isfinite(candidate_average_precision).all()
    ):
        raise ValueError("confirmation query evidence differs")
    deltas: list[float] = []
    top1_losses: list[int] = []
    reach_count = 0
    for expected_seed, row in zip(CONFIRMATION_SEEDS, rows, strict=True):
        if type(row) is not dict or tuple(row) != (
            "seed",
            "control_epoch16_primary",
            "candidate_primary_by_epoch",
            "control_top1_count",
            "candidate_top1_count",
        ):
            raise ValueError("confirmation row schema differs")
        if type(row["seed"]) is not int or row["seed"] != expected_seed:
            raise ValueError("confirmation seed order differs")
        control_map = _finite_float(
            row["control_epoch16_primary"], f"seed {expected_seed} control mAP"
        )
        trajectory = row["candidate_primary_by_epoch"]
        if type(trajectory) is not dict or tuple(trajectory) != EPOCHS:
            raise ValueError("confirmation trajectory differs")
        candidate_map = _finite_float(
            trajectory[16], f"seed {expected_seed} candidate mAP"
        )
        deltas.append(float(candidate_map - control_map))
        first_reached = next(
            (
                epoch
                for epoch in EPOCHS
                if _finite_float(trajectory[epoch], "trajectory mAP") >= control_map
            ),
            None,
        )
        reach_count += int(first_reached is not None and first_reached <= 12)
        control_top1 = row["control_top1_count"]
        candidate_top1 = row["candidate_top1_count"]
        if (
            type(control_top1) is not int
            or type(candidate_top1) is not int
            or control_top1 < 0
            or candidate_top1 < 0
        ):
            raise ValueError("confirmation top-1 counts differ")
        top1_losses.append(max(0, control_top1 - candidate_top1))
    mean_delta = statistics.fmean(deltas)
    half_width = PAIRED_T_CRITICAL * statistics.stdev(deltas) / math.sqrt(len(deltas))
    interval = [float(mean_delta - half_width), float(mean_delta + half_width)]
    positive_count = sum(delta > 0.0 for delta in deltas)
    query_rows = []
    for index, seed in enumerate(CONFIRMATION_SEEDS):
        per_seed = candidate_average_precision[index] - control_average_precision[index]
        query_rows.append({"seed": seed, "interval": list(_bootstrap_vector(per_seed))})
    query_bootstrap = {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "per_seed": query_rows,
        "pooled": list(
            pooled_query_bootstrap(
                control_average_precision, candidate_average_precision
            )
        ),
    }
    predicates = {
        "mean_primary_map_delta_at_least_0_003": mean_delta >= 0.003,
        "paired_t_lower_above_zero": interval[0] > 0.0,
        "at_least_four_positive_seeds": positive_count >= 4,
        "aggregate_top1_loss_at_most_5": sum(top1_losses) <= 5,
        "per_seed_top1_loss_at_most_2": max(top1_losses) <= 2,
        "control_endpoint_by_epoch_12_at_least_four": reach_count >= 4,
    }
    return {
        "primary_map_deltas": deltas,
        "mean_primary_map_delta": float(mean_delta),
        "paired_t_interval": interval,
        "positive_seed_count": positive_count,
        "top1_losses": top1_losses,
        "epoch12_reach_count": reach_count,
        "query_bootstrap_95": query_bootstrap,
        "predicates": predicates,
        "status": (
            "SUPPORTED_HOLDOUT_QUALITY"
            if all(predicates.values())
            else "CLOSE_FULL_WIDTH_QUALITY"
        ),
    }


def extract_seed_evidence(
    *,
    seed: int,
    pair_result: dict[str, object],
    control_receipt: dict[str, object],
    candidate_receipt: dict[str, object],
) -> dict[str, object]:
    """Cross-bind one confirmation pair and reduce it to registered evidence."""

    if type(seed) is not int or seed not in CONFIRMATION_SEEDS:
        raise ValueError("confirmation seed differs")
    if type(pair_result) is not dict or pair_result.get("seed") != seed:
        raise ValueError("paired result seed differs")
    receipts = {
        ARMS[0]: control_receipt,
        ARMS[1]: candidate_receipt,
    }
    allocated: list[int] = []
    reserved: list[int] = []
    checkpoint_sizes: dict[str, list[int]] = {arm: [] for arm in ARMS}
    for arm, receipt in receipts.items():
        if (
            type(receipt) is not dict
            or receipt.get("seed") != seed
            or receipt.get("arm") != arm
        ):
            raise ValueError("training receipt run binding differs")
        arm_allocated = receipt.get("peak_allocated_bytes")
        arm_reserved = receipt.get("peak_reserved_bytes")
        if (
            type(arm_allocated) is not int
            or type(arm_reserved) is not int
            or arm_allocated <= 0
            or arm_reserved < arm_allocated
        ):
            raise ValueError("training receipt memory differs")
        allocated.append(arm_allocated)
        reserved.append(arm_reserved)
        checkpoints = receipt.get("checkpoints")
        if type(checkpoints) is not list or len(checkpoints) != len(EPOCHS):
            raise ValueError("training receipt checkpoints differ")
    rows = pair_result.get("rows")
    if type(rows) is not list or len(rows) != len(EPOCHS):
        raise ValueError("paired result rows differ")
    candidate_by_epoch: dict[int, float] = {}
    control_endpoint = 0.0
    control_top1 = 0
    candidate_top1 = 0
    control_ap: np.ndarray | None = None
    candidate_ap: np.ndarray | None = None
    for epoch_index, (expected_epoch, pair_row) in enumerate(
        zip(EPOCHS, rows, strict=True)
    ):
        if type(pair_row) is not dict or pair_row.get("epoch") != expected_epoch:
            raise ValueError("paired result epoch order differs")
        arms = pair_row.get("arms")
        if type(arms) is not dict or tuple(arms) != ARMS:
            raise ValueError("paired result arms differ")
        for arm in ARMS:
            pair_arm = arms[arm]
            receipt_row = receipts[arm]["checkpoints"][epoch_index]
            if (
                type(pair_arm) is not dict
                or type(receipt_row) is not dict
                or receipt_row.get("epoch") != expected_epoch
                or tuple(
                    pair_arm.get(key)
                    for key in (
                        "checkpoint_path",
                        "checkpoint_sha256",
                        "checkpoint_bytes",
                    )
                )
                != tuple(
                    receipt_row.get(key) for key in ("path", "sha256", "bytes")
                )
            ):
                raise ValueError("paired checkpoint authority differs")
            checkpoint_bytes = receipt_row.get("bytes")
            if type(checkpoint_bytes) is not int or checkpoint_bytes <= 0:
                raise ValueError("checkpoint bytes differ")
            checkpoint_sizes[arm].append(checkpoint_bytes)
            primary = pair_arm.get("primary")
            if type(primary) is not dict:
                raise ValueError("paired primary evidence differs")
            map_at_r = _finite_float(primary.get("map_at_r"), "paired mAP")
            average_precision = primary.get("average_precision")
            top1 = primary.get("top1_correct")
            if (
                type(average_precision) is not list
                or not average_precision
                or any(
                    type(value) is not float or not math.isfinite(value)
                    for value in average_precision
                )
                or type(top1) is not list
                or len(top1) != len(average_precision)
                or any(type(value) is not bool for value in top1)
            ):
                raise ValueError("paired per-query evidence differs")
            if arm == ARMS[1]:
                candidate_by_epoch[expected_epoch] = map_at_r
            if expected_epoch == 16:
                values = np.asarray(average_precision, dtype=np.float64)
                if arm == ARMS[0]:
                    control_endpoint = map_at_r
                    control_top1 = sum(top1)
                    control_ap = values
                else:
                    candidate_top1 = sum(top1)
                    candidate_ap = values
    if (
        control_ap is None
        or candidate_ap is None
        or candidate_ap.shape != control_ap.shape
    ):
        raise ValueError("paired endpoint query evidence differs")
    control_sizes = checkpoint_sizes[ARMS[0]]
    candidate_sizes = checkpoint_sizes[ARMS[1]]
    return {
        "row": {
            "seed": seed,
            "control_epoch16_primary": control_endpoint,
            "candidate_primary_by_epoch": candidate_by_epoch,
            "control_top1_count": control_top1,
            "candidate_top1_count": candidate_top1,
        },
        "control_average_precision": control_ap,
        "candidate_average_precision": candidate_ap,
        "resource": {
            "seed": seed,
            "peak_allocated_ratio": float(allocated[1] / allocated[0]),
            "peak_reserved_ratio": float(reserved[1] / reserved[0]),
            "checkpoint_bytes_equal": control_sizes == candidate_sizes,
            "control_checkpoint_bytes": control_sizes,
            "candidate_checkpoint_bytes": candidate_sizes,
        },
    }


def cross_bind_seed_bundle(
    *,
    config: dict[str, object],
    seed: int,
    pair_inventory: dict[str, object],
    pair_result: dict[str, object],
    control_receipt: dict[str, object],
    candidate_receipt: dict[str, object],
) -> None:
    """Authenticate one confirmation bundle against frozen training authority."""

    if type(config) is not dict:
        raise ValueError("run configuration differs")
    authority = config.get("training_receipt_authority")
    if (
        type(authority) is not dict
        or tuple(authority) != ("source_commit", "config_commit", "config_sha256")
    ):
        raise ValueError("training receipt authority differs")
    source_commit = authority["source_commit"]
    config_sha256 = authority["config_sha256"]
    if (
        type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or type(config_sha256) is not str
        or len(config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        raise ValueError("training receipt authority differs")
    environment = config.get("environment")
    if type(environment) is not dict:
        raise ValueError("run environment differs")
    runtime = {
        "python": environment.get("python"),
        "torch": environment.get("torch"),
        "cuda": environment.get("cuda"),
    }
    receipts = {
        ARMS[0]: control_receipt,
        ARMS[1]: candidate_receipt,
    }
    for arm, receipt in receipts.items():
        if (
            type(receipt) is not dict
            or receipt.get("seed") != seed
            or receipt.get("arm") != arm
            or receipt.get("source_commit") != source_commit
            or receipt.get("config_sha256") != config_sha256
            or receipt.get("runtime") != runtime
        ):
            raise ValueError("training receipt run binding differs")
    inventory = pair_inventory.get("inventory")
    rows = pair_result.get("rows")
    if (
        type(seed) is not int
        or seed not in CONFIRMATION_SEEDS
        or pair_inventory.get("seed") != seed
        or pair_result.get("seed") != seed
        or type(inventory) is not list
        or len(inventory) != len(EPOCHS) * len(ARMS)
        or type(rows) is not list
        or len(rows) != len(EPOCHS)
    ):
        raise ValueError("paired bundle seed or rows differ")
    for epoch_index, expected_epoch in enumerate(EPOCHS):
        pair_row = rows[epoch_index]
        if type(pair_row) is not dict or pair_row.get("epoch") != expected_epoch:
            raise ValueError("paired row order differs")
        arms = pair_row.get("arms")
        if type(arms) is not dict or tuple(arms) != ARMS:
            raise ValueError("paired row arms differ")
        for arm_index, arm in enumerate(ARMS):
            inventory_row = inventory[epoch_index * len(ARMS) + arm_index]
            receipt_row = receipts[arm].get("checkpoints", [])[epoch_index]
            pair_arm = arms[arm]
            if (
                type(inventory_row) is not dict
                or type(receipt_row) is not dict
                or type(pair_arm) is not dict
                or inventory_row.get("arm") != arm
                or inventory_row.get("epoch") != expected_epoch
                or receipt_row.get("epoch") != expected_epoch
                or tuple(
                    inventory_row.get(key) for key in ("path", "sha256", "bytes")
                )
                != tuple(
                    receipt_row.get(key) for key in ("path", "sha256", "bytes")
                )
                or tuple(
                    pair_arm.get(key)
                    for key in (
                        "checkpoint_path",
                        "checkpoint_sha256",
                        "checkpoint_bytes",
                    )
                )
                != tuple(
                    receipt_row.get(key) for key in ("path", "sha256", "bytes")
                )
            ):
                raise ValueError("paired checkpoint authority differs")


def build_operational_summary(
    *,
    seed0_decision: dict[str, object],
    seed0_profile_comparison: dict[str, object],
    confirmation_resources: tuple[dict[str, object], ...],
) -> dict[str, object]:
    """Report observed costs and make missing confirmation evidence explicit."""

    if (
        type(seed0_decision) is not dict
        or seed0_decision.get("status") != "PROMOTE_CONFIRMATION"
    ):
        raise ValueError("seed-0 decision does not authorize confirmation")
    decision_evidence = seed0_decision.get("evidence")
    profile_ratios = seed0_profile_comparison.get("ratios")
    intervals = seed0_profile_comparison.get("ratio_bootstrap_95")
    if (
        type(decision_evidence) is not dict
        or type(profile_ratios) is not dict
        or type(intervals) is not dict
    ):
        raise ValueError("seed-0 operational evidence differs")
    wall_ratio = _finite_float(profile_ratios.get("step_wall"), "step-wall ratio")
    cuda_ratio = _finite_float(profile_ratios.get("cuda_step"), "CUDA-step ratio")
    if (
        decision_evidence.get("abba_step_time_ratio") != wall_ratio
        or decision_evidence.get("observed_cuda_step_ratio") != cuda_ratio
    ):
        raise ValueError("seed-0 step-time authority differs")
    wall_interval = intervals.get("step_wall")
    if (
        type(wall_interval) is not list
        or len(wall_interval) != 2
        or any(type(value) is not float or not math.isfinite(value) for value in wall_interval)
        or wall_interval[0] > wall_interval[1]
    ):
        raise ValueError("seed-0 step-time interval differs")
    if (
        type(confirmation_resources) is not tuple
        or len(confirmation_resources) != len(CONFIRMATION_SEEDS)
    ):
        raise ValueError("confirmation resource rows differ")
    allocated_ratios: list[float] = []
    reserved_ratios: list[float] = []
    resource_rows: list[dict[str, object]] = []
    all_checkpoint_bytes_equal = True
    for expected_seed, row in zip(
        CONFIRMATION_SEEDS, confirmation_resources, strict=True
    ):
        if type(row) is not dict or tuple(row) != (
            "seed",
            "peak_allocated_ratio",
            "peak_reserved_ratio",
            "checkpoint_bytes_equal",
            "control_checkpoint_bytes",
            "candidate_checkpoint_bytes",
        ):
            raise ValueError("confirmation resource schema differs")
        if type(row["seed"]) is not int or row["seed"] != expected_seed:
            raise ValueError("confirmation resource seed order differs")
        allocated = _finite_float(row["peak_allocated_ratio"], "allocated ratio")
        reserved = _finite_float(row["peak_reserved_ratio"], "reserved ratio")
        checkpoint_equal = row["checkpoint_bytes_equal"]
        if type(checkpoint_equal) is not bool:
            raise TypeError("checkpoint equality must be a builtin boolean")
        control_sizes = row["control_checkpoint_bytes"]
        candidate_sizes = row["candidate_checkpoint_bytes"]
        if (
            type(control_sizes) is not list
            or type(candidate_sizes) is not list
            or len(control_sizes) != len(EPOCHS)
            or len(candidate_sizes) != len(EPOCHS)
            or any(
                type(value) is not int or value <= 0
                for value in (*control_sizes, *candidate_sizes)
            )
            or checkpoint_equal != (control_sizes == candidate_sizes)
        ):
            raise ValueError("confirmation checkpoint evidence differs")
        allocated_ratios.append(allocated)
        reserved_ratios.append(reserved)
        all_checkpoint_bytes_equal &= checkpoint_equal
        resource_rows.append(row)
    mean_allocated = float(statistics.fmean(allocated_ratios))
    mean_reserved = float(statistics.fmean(reserved_ratios))
    predicates = {
        "seed0_abba_step_time_ratio_at_most_1_02": wall_ratio <= 1.02,
        "mean_peak_allocated_ratio_at_most_1_02": mean_allocated <= 1.02,
        "mean_peak_reserved_ratio_at_most_1_02": mean_reserved <= 1.02,
        "checkpoint_bytes_exactly_equal": all_checkpoint_bytes_equal,
        "registered_mean_abba_available": False,
        "empirical_deployment_measurements_available": False,
    }
    return {
        "step_time": {
            "authority_seed": 0,
            "scope": "single_seed_terminal_checkpoint_abba_screen",
            "claim": "no_measurable_slowdown_gate_only",
            "metric": "step_wall",
            "ratio": wall_ratio,
            "bootstrap_95": wall_interval,
        },
        "memory_and_checkpoint_rows": resource_rows,
        "mean_peak_allocated_ratio": mean_allocated,
        "mean_peak_reserved_ratio": mean_reserved,
        "missing_evidence": {
            "confirmation_seed_abba_profiles": list(CONFIRMATION_SEEDS),
            "empirical_deployment_parameter_comparison": True,
            "empirical_inference_operation_comparison": True,
            "empirical_deployment_storage_comparison": True,
        },
        "predicates": predicates,
        "status": "INCOMPLETE_OPERATIONAL_EVIDENCE",
    }


def _validate_binding(value: object) -> None:
    if (
        type(value) is not dict
        or tuple(value) != INPUT_BINDING_KEYS
        or type(value["path"]) is not str
        or not value["path"]
        or type(value["sha256"]) is not str
        or len(value["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["sha256"])
        or type(value["bytes"]) is not int
        or value["bytes"] <= 0
    ):
        raise ValueError("confirmation input binding differs")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("JSON object keys differ")
        result[key] = value
    return result


def load_bound_json(
    binding: dict[str, object], evidence_root: Path
) -> dict[str, object]:
    """Hash and parse one config-bound regular JSON file from the same bytes."""

    _validate_binding(binding)
    if not isinstance(evidence_root, Path):
        raise TypeError("confirmation evidence root must be a pathlib.Path")
    path = Path(binding["path"])
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve().is_relative_to(evidence_root.resolve())
    ):
        raise ValueError("confirmation audit input path differs")
    payload = path.read_bytes()
    if (
        len(payload) != binding["bytes"]
        or hashlib.sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise ValueError("confirmation audit input bytes differ")
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise ValueError("confirmation audit input root differs")
    return value


def validate_confirmation_result(
    value: object, *, expected: dict[str, object] | None = None
) -> None:
    """Validate the exact result schema and optional authenticated recomputation."""

    if type(value) is not dict or tuple(value) != TOP_KEYS:
        raise ValueError("confirmation result schema differs")
    if value["schema_version"] != "unicom-full-width-confirmation-v2":
        raise ValueError("confirmation result version differs")
    inputs = value["inputs"]
    if type(inputs) is not dict or tuple(inputs) != (
        "run_config",
        "seed0_decision",
        "seed0_profile_comparison",
        "confirmation_seeds",
    ):
        raise ValueError("confirmation inputs differ")
    for key in ("run_config", "seed0_decision", "seed0_profile_comparison"):
        _validate_binding(inputs[key])
    seed_inputs = inputs["confirmation_seeds"]
    if type(seed_inputs) is not list or len(seed_inputs) != len(CONFIRMATION_SEEDS):
        raise ValueError("confirmation seed inputs differ")
    for expected_seed, row in zip(CONFIRMATION_SEEDS, seed_inputs, strict=True):
        if type(row) is not dict or tuple(row) != (
            "seed",
            "pair_inventory",
            "pair_result",
            "control_receipt",
            "candidate_receipt",
        ):
            raise ValueError("confirmation seed input schema differs")
        if type(row["seed"]) is not int or row["seed"] != expected_seed:
            raise ValueError("confirmation seed input order differs")
        for key in (
            "pair_inventory",
            "pair_result",
            "control_receipt",
            "candidate_receipt",
        ):
            _validate_binding(row[key])
    authority = value["metric_authority"]
    if authority != {
        "quality": "five_confirmation_paired_results_primary_full_768",
        "query_bootstrap": "paired_result_per_query_average_precision",
        "step_time": "seed0_profile_comparison_step_wall_only",
        "memory": "five_confirmation_training_run_receipts",
        "checkpoint_bytes": "five_confirmation_pair_results_and_training_receipts",
        "deployment": "not_empirically_measured",
    }:
        raise ValueError("confirmation metric authority differs")
    quality = value["quality"]
    if type(quality) is not dict or tuple(quality) != (
        "primary_map_deltas",
        "mean_primary_map_delta",
        "paired_t_interval",
        "positive_seed_count",
        "top1_losses",
        "epoch12_reach_count",
        "query_bootstrap_95",
        "predicates",
        "status",
    ):
        raise ValueError("confirmation quality schema differs")
    if quality["status"] not in (
        "SUPPORTED_HOLDOUT_QUALITY",
        "CLOSE_FULL_WIDTH_QUALITY",
    ):
        raise ValueError("confirmation quality status differs")
    operational = value["operational"]
    if type(operational) is not dict or tuple(operational) != (
        "step_time",
        "memory_and_checkpoint_rows",
        "mean_peak_allocated_ratio",
        "mean_peak_reserved_ratio",
        "missing_evidence",
        "predicates",
        "status",
    ):
        raise ValueError("confirmation operational schema differs")
    step_time = operational["step_time"]
    if step_time != {
        "authority_seed": 0,
        "scope": "single_seed_terminal_checkpoint_abba_screen",
        "claim": "no_measurable_slowdown_gate_only",
        "metric": "step_wall",
        "ratio": step_time.get("ratio") if type(step_time) is dict else None,
        "bootstrap_95": step_time.get("bootstrap_95") if type(step_time) is dict else None,
    }:
        raise ValueError("confirmation step-time scope differs")
    _finite_float(step_time["ratio"], "confirmation step-time ratio")
    if operational["status"] != "INCOMPLETE_OPERATIONAL_EVIDENCE":
        raise ValueError("confirmation operational status differs")
    quality_status = quality["status"]
    expected_registered = (
        "INCOMPLETE_OPERATIONAL_EVIDENCE"
        if quality_status == "SUPPORTED_HOLDOUT_QUALITY"
        else "CLOSE_FULL_WIDTH_QUALITY"
    )
    expected_clause = (
        "missing_confirmation_seed_abba_and_deployment_measurements"
        if quality_status == "SUPPORTED_HOLDOUT_QUALITY"
        else "quality_predicates_failed"
    )
    decision = value["decision"]
    if decision != {
        "quality": quality_status,
        "registered_confirmation": expected_registered,
        "first_decisive_clause": expected_clause,
    }:
        raise ValueError("confirmation decision relation differs")
    if value["status"] != expected_registered:
        raise ValueError("confirmation status relation differs")
    if expected is not None and value != expected:
        raise ValueError("confirmation recomputation differs")


def build_confirmation_result(
    *,
    inputs: dict[str, object],
    quality: dict[str, object],
    operational: dict[str, object],
) -> dict[str, object]:
    """Assemble the versioned publication artifact from authenticated sections."""

    quality_status = quality.get("status")
    if quality_status == "SUPPORTED_HOLDOUT_QUALITY":
        registered = "INCOMPLETE_OPERATIONAL_EVIDENCE"
        clause = "missing_confirmation_seed_abba_and_deployment_measurements"
    elif quality_status == "CLOSE_FULL_WIDTH_QUALITY":
        registered = "CLOSE_FULL_WIDTH_QUALITY"
        clause = "quality_predicates_failed"
    else:
        raise ValueError("confirmation quality status differs")
    value = {
        "schema_version": "unicom-full-width-confirmation-v2",
        "inputs": inputs,
        "metric_authority": {
            "quality": "five_confirmation_paired_results_primary_full_768",
            "query_bootstrap": "paired_result_per_query_average_precision",
            "step_time": "seed0_profile_comparison_step_wall_only",
            "memory": "five_confirmation_training_run_receipts",
            "checkpoint_bytes": "five_confirmation_pair_results_and_training_receipts",
            "deployment": "not_empirically_measured",
        },
        "quality": quality,
        "operational": operational,
        "decision": {
            "quality": quality_status,
            "registered_confirmation": registered,
            "first_decisive_clause": clause,
        },
        "status": registered,
    }
    validate_confirmation_result(value)
    return value


def authenticate_confirmation_handoff(run_config: Path, checkout: Path) -> str:
    """Authenticate the detached source/config handoff for this producer."""

    return TRAINER.registered_source_commit(run_config, checkout)


def _validate_audit_inputs(value: object) -> None:
    if type(value) is not dict or tuple(value) != (
        "seed0_decision",
        "seed0_profile_comparison",
        "confirmation_seeds",
    ):
        raise ValueError("confirmation audit inputs differ")
    _validate_binding(value["seed0_decision"])
    _validate_binding(value["seed0_profile_comparison"])
    rows = value["confirmation_seeds"]
    if type(rows) is not list or len(rows) != len(CONFIRMATION_SEEDS):
        raise ValueError("confirmation audit seed inputs differ")
    for expected_seed, row in zip(CONFIRMATION_SEEDS, rows, strict=True):
        if type(row) is not dict or tuple(row) != (
            "seed",
            "pair_inventory",
            "pair_result",
            "control_receipt",
            "candidate_receipt",
        ):
            raise ValueError("confirmation audit seed schema differs")
        if type(row["seed"]) is not int or row["seed"] != expected_seed:
            raise ValueError("confirmation audit seed order differs")
        for key in (
            "pair_inventory",
            "pair_result",
            "control_receipt",
            "candidate_receipt",
        ):
            _validate_binding(row[key])


def _validate_run_config(
    config: object,
    args: argparse.Namespace,
    *,
    observed_command: list[str] | None = None,
) -> None:
    if (
        type(config) is not dict
        or config.get("schema_version") != "unicom-full-width-objective-run-v2"
    ):
        raise ValueError("run configuration version differs")
    paths = config.get("paths")
    outputs = config.get("registered_outputs")
    templates = config.get("command_templates")
    if (
        type(paths) is not dict
        or paths.get("output_root") != str(args.evidence_root)
        or type(outputs) is not dict
        or outputs.get("confirmation_result_v2") != str(args.output)
    ):
        raise ValueError("confirmation output paths differ")
    _validate_audit_inputs(config.get("confirmation_audit_inputs"))
    if observed_command is not None and (
            type(templates) is not dict
            or templates.get("confirmation_command") != observed_command
            or any(type(token) is not str or not token for token in observed_command)
    ):
        raise ValueError("confirmation command differs")


def build_from_evidence(
    run_config: Path, evidence_root: Path, output: Path
) -> dict[str, object]:
    """Load authenticated evidence and build the exact confirmation result."""

    if (
        not isinstance(run_config, Path)
        or not isinstance(evidence_root, Path)
        or not isinstance(output, Path)
        or not evidence_root.is_dir()
        or evidence_root.is_symlink()
        or output.parent.resolve() != evidence_root.resolve()
    ):
        raise ValueError("confirmation evidence paths differ")
    config = EVALUATOR.strict_json_object(run_config)
    args = argparse.Namespace(
        run_config=run_config, evidence_root=evidence_root, output=output
    )
    _validate_run_config(config, args)
    audit = config["confirmation_audit_inputs"]

    seed0_decision = load_bound_json(audit["seed0_decision"], evidence_root)
    seed0_profile = load_bound_json(
        audit["seed0_profile_comparison"], evidence_root
    )
    DECIDER.validate_seed0_decision(seed0_decision)
    COMPARATOR.validate_comparison_result(seed0_profile)
    decision_inputs = seed0_decision.get("inputs")
    if (
        type(decision_inputs) is not dict
        or decision_inputs.get("profile_comparison")
        != audit["seed0_profile_comparison"]
    ):
        raise ValueError("seed-0 decision/profile binding differs")

    rows: list[dict[str, object]] = []
    control_ap_rows: list[np.ndarray] = []
    candidate_ap_rows: list[np.ndarray] = []
    resources: list[dict[str, object]] = []
    identity_hashes: tuple[str, str] | None = None
    for seed_input in audit["confirmation_seeds"]:
        seed = seed_input["seed"]
        inventory = load_bound_json(seed_input["pair_inventory"], evidence_root)
        pair_result = load_bound_json(seed_input["pair_result"], evidence_root)
        control_receipt = load_bound_json(
            seed_input["control_receipt"], evidence_root
        )
        candidate_receipt = load_bound_json(
            seed_input["candidate_receipt"], evidence_root
        )
        EVALUATOR.validate_pair_result(pair_result, inventory)
        TRAINER.validate_training_run_receipt(control_receipt)
        TRAINER.validate_training_run_receipt(candidate_receipt)
        cross_bind_seed_bundle(
            config=config,
            seed=seed,
            pair_inventory=inventory,
            pair_result=pair_result,
            control_receipt=control_receipt,
            candidate_receipt=candidate_receipt,
        )
        pair_rows = pair_result["rows"]
        for pair_row in pair_rows:
            current = (
                pair_row.get("query_ids_sha256"),
                pair_row.get("gallery_ids_sha256"),
            )
            if (
                any(
                    type(value) is not str
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in current
                )
                or (identity_hashes is not None and current != identity_hashes)
            ):
                raise ValueError("confirmation query/gallery identities differ")
            identity_hashes = current
        extracted = extract_seed_evidence(
            seed=seed,
            pair_result=pair_result,
            control_receipt=control_receipt,
            candidate_receipt=candidate_receipt,
        )
        rows.append(extracted["row"])
        control_ap_rows.append(extracted["control_average_precision"])
        candidate_ap_rows.append(extracted["candidate_average_precision"])
        resources.append(extracted["resource"])
    quality = build_quality_summary(
        tuple(rows),
        control_average_precision=np.stack(control_ap_rows),
        candidate_average_precision=np.stack(candidate_ap_rows),
    )
    operational = build_operational_summary(
        seed0_decision=seed0_decision,
        seed0_profile_comparison=seed0_profile,
        confirmation_resources=tuple(resources),
    )
    config_payload = run_config.read_bytes()
    inputs = {
        "run_config": {
            "path": str(run_config),
            "sha256": hashlib.sha256(config_payload).hexdigest(),
            "bytes": len(config_payload),
        },
        **audit,
    }
    result = build_confirmation_result(
        inputs=inputs, quality=quality, operational=operational
    )
    validate_confirmation_result(result, expected=result)
    return result


def authenticate_persisted_result(
    run_config: Path, evidence_root: Path, result_path: Path
) -> dict[str, object]:
    """Recompute and authenticate one already-published confirmation artifact."""

    expected = build_from_evidence(run_config, evidence_root, result_path)
    persisted = EVALUATOR.strict_json_object(result_path)
    validate_confirmation_result(persisted, expected=expected)
    return persisted


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        authenticate_confirmation_handoff(
            args.run_config, Path(__file__).resolve().parents[1]
        )
        config = EVALUATOR.strict_json_object(args.run_config)
        observed_command = list(sys.orig_argv) if arguments is None else None
        _validate_run_config(config, args, observed_command=observed_command)
        result = build_from_evidence(
            args.run_config, args.evidence_root, args.output
        )
        validate_confirmation_result(result, expected=result)
        EVALUATOR.publish_result(
            result,
            args.output,
            validate=lambda persisted: validate_confirmation_result(
                persisted, expected=result
            ),
        )
    except Exception as error:
        print(f"confirmation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
