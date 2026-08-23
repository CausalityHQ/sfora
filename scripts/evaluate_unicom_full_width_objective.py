#!/usr/bin/env python3
"""Paired holdout evaluation for the UniCOM full-width objective control."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import statistics
from collections.abc import Callable
from pathlib import Path

import numpy as np

ARMS = ("sampled_512", "full_768")
EPOCHS = (4, 8, 12, 16)
SELECTION_SEEDS = (0,)
CONFIRMATION_SEEDS = (2, 3, 4, 5, 6)
PRIMARY_COORDINATES = tuple(range(768))
LEGACY_COORDINATES = tuple(range(512))
RECALL_AT_K = (1, 10, 20, 30, 40, 50)
ARM_PROTOCOLS = {
    "sampled_512": ("official-eight-mask", 512, 768),
    "full_768": ("official-eight-mask", 768, 768),
}


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a builtin float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_arm_protocol(value: object, arm: str) -> None:
    """Bind the loss and evaluation widths without weakening historical schemas."""

    if type(arm) is not str or arm not in ARM_PROTOCOLS or type(value) is not dict:
        raise ValueError("full-width arm protocol differs")
    expected = ARM_PROTOCOLS[arm]
    keys = ("objective", "selected_features", "evaluation_features")
    if any(key not in value for key in keys):
        raise ValueError("full-width arm protocol is incomplete")
    actual = tuple(value[key] for key in keys)
    if any(
        type(item) is not type(reference)
        for item, reference in zip(actual, expected, strict=True)
    ):
        raise ValueError("full-width arm protocol types differ")
    if actual != expected:
        raise ValueError("full-width arm protocol values differ")


def paired_t_interval(
    values: tuple[float, ...], *, critical: float
) -> tuple[float, float]:
    """Return a two-sided Student-t interval for paired seed deltas."""

    if type(values) is not tuple or len(values) < 2:
        raise ValueError("paired values must be a tuple with at least two entries")
    validated = tuple(
        _finite_float(value, f"paired value {index}") for index, value in enumerate(values)
    )
    critical = _finite_float(critical, "critical value", positive=True)
    mean = statistics.fmean(validated)
    half_width = critical * statistics.stdev(validated) / math.sqrt(len(validated))
    return (float(mean - half_width), float(mean + half_width))


def first_epoch_reaching(values: dict[int, float], target: float) -> int | None:
    """Return the first registered epoch that reaches a finite target."""

    if type(values) is not dict or tuple(values) != EPOCHS:
        raise ValueError("trajectory epoch order differs")
    target = _finite_float(target, "trajectory target")
    for epoch in EPOCHS:
        value = _finite_float(values[epoch], f"epoch {epoch} value")
        if value >= target:
            return epoch
    return None


def paired_query_bootstrap(
    control: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    """Bootstrap paired query evidence with one shared resampling draw."""

    if (
        type(control) is not tuple
        or type(candidate) is not tuple
        or not control
        or len(candidate) != len(control)
    ):
        raise ValueError("paired query evidence shapes differ")
    if type(seed) is not int or seed != 768 or type(samples) is not int or samples != 10_000:
        raise ValueError("paired query bootstrap configuration differs")
    control_values = np.asarray(
        [_finite_float(value, f"control query {index}") for index, value in enumerate(control)],
        dtype=np.float64,
    )
    candidate_values = np.asarray(
        [
            _finite_float(value, f"candidate query {index}")
            for index, value in enumerate(candidate)
        ],
        dtype=np.float64,
    )
    if (
        np.any(control_values < 0.0)
        or np.any(control_values > 1.0)
        or np.any(candidate_values < 0.0)
        or np.any(candidate_values > 1.0)
    ):
        raise ValueError("paired query evidence is outside [0, 1]")
    delta = candidate_values - control_values
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, delta.size, size=(samples, delta.size))
    lower, upper = np.percentile(delta[indices].mean(axis=1), (2.5, 97.5))
    return (float(lower), float(upper))


def _embedding_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _ordered_id_sha256(values: tuple[str, ...], name: str) -> str:
    if (
        type(values) is not tuple
        or not values
        or any(type(value) is not str or not value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} IDs differ")
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _validate_embeddings(values: object, rows: int, name: str) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.dtype != np.float32
        or values.shape != (rows, 768)
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} embeddings differ")
    return values


def _retrieval_payload(view: object) -> dict[str, object]:
    average_precision = getattr(view, "average_precision", None)
    top1_correct = getattr(view, "top1_correct", None)
    recall = getattr(view, "recall", None)
    if (
        type(average_precision) is not np.ndarray
        or average_precision.dtype != np.float64
        or type(top1_correct) is not np.ndarray
        or top1_correct.dtype != np.bool_
        or type(recall) is not dict
    ):
        raise ValueError("retrieval evidence differs")
    return {
        "recall": {str(key): float(recall[key]) for key in RECALL_AT_K},
        "map_at_r": float(view.map_at_r),
        "average_precision": average_precision.tolist(),
        "top1_correct": top1_correct.tolist(),
    }


def _evaluate_embedding_views(
    query: np.ndarray,
    gallery: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, object]:
    from sfora.unicom_retrieval_audit import retrieval_view

    primary = retrieval_view(
        query,
        gallery,
        query_labels,
        gallery_labels,
        coordinates=np.arange(768, dtype=np.int64),
        normalize_before=False,
        recall_at_k=RECALL_AT_K,
    )
    legacy = retrieval_view(
        query,
        gallery,
        query_labels,
        gallery_labels,
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
        recall_at_k=RECALL_AT_K,
    )
    return {
        "query_embedding_sha256": _embedding_sha256(query),
        "gallery_embedding_sha256": _embedding_sha256(gallery),
        "primary": _retrieval_payload(primary),
        "legacy": _retrieval_payload(legacy),
    }


def evaluate_pair_embeddings(
    *,
    control_query: np.ndarray,
    control_gallery: np.ndarray,
    candidate_query: np.ndarray,
    candidate_gallery: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_ids: tuple[str, ...],
    gallery_ids: tuple[str, ...],
) -> dict[str, object]:
    """Evaluate both checkpoint arms over one exact ordered holdout."""

    if (
        type(query_labels) is not np.ndarray
        or query_labels.ndim != 1
        or query_labels.size == 0
        or type(gallery_labels) is not np.ndarray
        or gallery_labels.ndim != 1
        or gallery_labels.size == 0
        or any(type(value) is not str or not value for value in query_labels.tolist())
        or any(type(value) is not str or not value for value in gallery_labels.tolist())
        or not set(query_labels.tolist()).issubset(set(gallery_labels.tolist()))
    ):
        raise ValueError("paired retrieval labels differ")
    if len(query_ids) != query_labels.size or len(gallery_ids) != gallery_labels.size:
        raise ValueError("paired retrieval ID counts differ")
    query_ids_sha256 = _ordered_id_sha256(query_ids, "query")
    gallery_ids_sha256 = _ordered_id_sha256(gallery_ids, "gallery")
    control_query = _validate_embeddings(
        control_query, query_labels.size, "control query"
    )
    control_gallery = _validate_embeddings(
        control_gallery, gallery_labels.size, "control gallery"
    )
    candidate_query = _validate_embeddings(
        candidate_query, query_labels.size, "candidate query"
    )
    candidate_gallery = _validate_embeddings(
        candidate_gallery, gallery_labels.size, "candidate gallery"
    )
    return {
        "query_ids_sha256": query_ids_sha256,
        "gallery_ids_sha256": gallery_ids_sha256,
        "arms": {
            "sampled_512": _evaluate_embedding_views(
                control_query, control_gallery, query_labels, gallery_labels
            ),
            "full_768": _evaluate_embedding_views(
                candidate_query, candidate_gallery, query_labels, gallery_labels
            ),
        },
    }


def selection_decision(
    *,
    primary_map_delta: float,
    control_top1_count: int,
    candidate_top1_count: int,
    candidate_primary_by_epoch: dict[int, float],
    control_epoch16_primary: float,
    abba_step_time_ratio: float,
    peak_allocated_ratio: float,
    peak_reserved_ratio: float,
    control_checkpoint_bytes: int,
    candidate_checkpoint_bytes: int,
) -> dict[str, object]:
    """Separate the seed-0 quality prediction from its operational gate."""

    delta = _finite_float(primary_map_delta, "primary mAP delta")
    target = _finite_float(control_epoch16_primary, "control epoch-16 primary")
    if (
        type(control_top1_count) is not int
        or type(candidate_top1_count) is not int
        or control_top1_count < 0
        or candidate_top1_count < 0
    ):
        raise ValueError("top-1 query counts differ")
    if (
        type(control_checkpoint_bytes) is not int
        or type(candidate_checkpoint_bytes) is not int
        or control_checkpoint_bytes <= 0
        or candidate_checkpoint_bytes <= 0
    ):
        raise ValueError("checkpoint byte counts differ")
    reached = first_epoch_reaching(candidate_primary_by_epoch, target)
    prediction = (
        ("primary_map_delta_at_least_0_003", delta >= 0.003),
        (
            "top1_query_loss_at_most_1",
            candidate_top1_count >= control_top1_count - 1,
        ),
        (
            "control_endpoint_reached_by_epoch_12",
            reached is not None and reached <= 12,
        ),
    )
    operational = (
        (
            "abba_step_time_ratio_at_most_1_02",
            _finite_float(
                abba_step_time_ratio, "A-B-B-A step-time ratio", positive=True
            )
            <= 1.02,
        ),
        (
            "peak_allocated_ratio_at_most_1_02",
            _finite_float(
                peak_allocated_ratio, "peak allocated ratio", positive=True
            )
            <= 1.02,
        ),
        (
            "peak_reserved_ratio_at_most_1_02",
            _finite_float(peak_reserved_ratio, "peak reserved ratio", positive=True)
            <= 1.02,
        ),
        (
            "checkpoint_bytes_exactly_equal",
            candidate_checkpoint_bytes == control_checkpoint_bytes,
        ),
    )
    prediction_matched = all(value for _name, value in prediction)
    operational_passed = all(value for _name, value in operational)
    return {
        "quality_prediction": dict(prediction),
        "prediction_matched": prediction_matched,
        "operational_predicates": dict(operational),
        "operational_passed": operational_passed,
        "decision": "PROMOTE_CONFIRMATION" if operational_passed else "CLOSE_RESOURCE",
    }


def confirmation_decision(
    rows: tuple[dict[str, object], ...],
    *,
    mean_abba_step_time_ratio: float,
    mean_peak_allocated_ratio: float,
    mean_peak_reserved_ratio: float,
    checkpoint_bytes_equal: bool,
    deployed_parameters_equal: bool,
    inference_operations_equal: bool,
    deployment_storage_equal: bool,
) -> dict[str, object]:
    """Apply the frozen five-seed quality and resource decision."""

    if type(rows) is not tuple or len(rows) != len(CONFIRMATION_SEEDS):
        raise ValueError("confirmation rows differ")
    deltas: list[float] = []
    top1_losses: list[int] = []
    epoch12_reach_count = 0
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
            row["control_epoch16_primary"], f"seed {expected_seed} control primary"
        )
        trajectory = row["candidate_primary_by_epoch"]
        if type(trajectory) is not dict:
            raise ValueError("confirmation trajectory differs")
        candidate_map = _finite_float(
            trajectory.get(16), f"seed {expected_seed} candidate primary"
        )
        deltas.append(float(candidate_map - control_map))
        reached = first_epoch_reaching(trajectory, control_map)
        epoch12_reach_count += int(reached is not None and reached <= 12)
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
    exact_flags = (
        checkpoint_bytes_equal,
        deployed_parameters_equal,
        inference_operations_equal,
        deployment_storage_equal,
    )
    if any(type(value) is not bool for value in exact_flags):
        raise TypeError("confirmation exact-equality flags must be builtin booleans")
    delta_tuple = tuple(deltas)
    interval = paired_t_interval(delta_tuple, critical=2.7764451052)
    mean_delta = statistics.fmean(delta_tuple)
    positive_seed_count = sum(value > 0.0 for value in delta_tuple)
    predicates = {
        "mean_primary_map_delta_at_least_0_003": mean_delta >= 0.003,
        "paired_t_lower_above_zero": interval[0] > 0.0,
        "at_least_four_positive_seeds": positive_seed_count >= 4,
        "aggregate_top1_loss_at_most_5": sum(top1_losses) <= 5,
        "per_seed_top1_loss_at_most_2": max(top1_losses) <= 2,
        "control_endpoint_by_epoch_12_at_least_four": epoch12_reach_count >= 4,
        "mean_abba_step_time_ratio_at_most_1_02": _finite_float(
            mean_abba_step_time_ratio, "mean A-B-B-A step-time ratio", positive=True
        )
        <= 1.02,
        "mean_peak_allocated_ratio_at_most_1_02": _finite_float(
            mean_peak_allocated_ratio, "mean peak allocated ratio", positive=True
        )
        <= 1.02,
        "mean_peak_reserved_ratio_at_most_1_02": _finite_float(
            mean_peak_reserved_ratio, "mean peak reserved ratio", positive=True
        )
        <= 1.02,
        "checkpoint_bytes_exactly_equal": checkpoint_bytes_equal,
        "deployed_parameters_exactly_equal": deployed_parameters_equal,
        "inference_operations_exactly_equal": inference_operations_equal,
        "deployment_storage_exactly_equal": deployment_storage_equal,
    }
    supported = all(predicates.values())
    return {
        "primary_map_deltas": deltas,
        "mean_primary_map_delta": float(mean_delta),
        "paired_t_interval": list(interval),
        "positive_seed_count": positive_seed_count,
        "top1_losses": top1_losses,
        "epoch12_reach_count": epoch12_reach_count,
        "predicates": predicates,
        "decision": "SUPPORTED_HOLDOUT" if supported else "CLOSE_FULL_WIDTH",
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("JSON object keys differ")
        result[key] = value
    return result


def strict_json_object(path: Path) -> dict[str, object]:
    """Load one bounded JSON object while rejecting duplicates and nonfinite values."""

    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("strict JSON path differs")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("strict JSON exceeds 64 MiB")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise ValueError("strict JSON root differs")
    return value


def _same_inode(path: Path, owned: tuple[int, int]) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return not path.is_symlink() and (info.st_dev, info.st_ino) == owned


def publish_result(
    payload: dict[str, object],
    output: Path,
    *,
    validate: Callable[[dict[str, object]], None],
) -> None:
    """Exclusively publish one strict-reloaded result with inode-owned rollback."""

    if type(payload) is not dict or not isinstance(output, Path):
        raise TypeError("result publication inputs differ")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent_info = output.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or output.parent.is_symlink():
        raise ValueError("result output parent must be a real directory")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(encoded) > 64 * 1024 * 1024:
        raise ValueError("result exceeds 64 MiB")
    descriptor: int | None = None
    owned: tuple[int, int] | None = None
    linked = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = strict_json_object(temporary)
        validate(reloaded)
        os.link(temporary, output)
        linked = True
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        temporary.unlink()
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if owned is not None and linked and _same_inode(output, owned):
            output.unlink()
        if owned is not None and _same_inode(temporary, owned):
            temporary.unlink()
        raise
