#!/usr/bin/env python3
"""Paired holdout evaluation for the UniCOM full-width objective control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

ARMS = ("sampled_512", "full_768")
EPOCHS = (4, 8, 12, 16)
SELECTION_SEEDS = (0,)
CONFIRMATION_SEEDS = (2, 3, 4, 5, 6)
PRIMARY_COORDINATES = tuple(range(768))
LEGACY_COORDINATES = tuple(range(512))
RECALL_AT_K = (1, 10, 20, 30, 40, 50)
UNICOM_INITIAL_CHECKPOINT_NAME = "FP16-ViT-L-14-336px.pt"
ARM_PROTOCOLS = {
    "sampled_512": ("official-eight-mask", 512, 768),
    "full_768": ("official-eight-mask", 768, 768),
}
PAIR_CONFIG_KEYS = ("schema_version", "seed", "inventory")
PAIR_INVENTORY_KEYS = ("arm", "epoch", "path", "sha256", "bytes")
PAIR_RESULT_KEYS = ("schema_version", "seed", "rows")
PAIR_ROW_KEYS = ("epoch", "query_ids_sha256", "gallery_ids_sha256", "arms")
PAIR_ARM_KEYS = (
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "query_embedding_sha256",
    "gallery_embedding_sha256",
    "primary",
    "legacy",
    "elapsed_seconds",
    "peak_allocated_bytes",
)
RETRIEVAL_VIEW_KEYS = (
    "recall",
    "recall_counts",
    "map_at_r",
    "average_precision",
    "top1_correct",
)
TRAINING_CHECKPOINT_KEYS = (
    "epoch",
    "model",
    "classifier",
    "ema",
    "optimizer",
    "scheduler",
    "scaler",
    "mask_generator",
    "torch_rng_state",
    "cuda_rng_states",
    "selection_holdout",
    "training_protocol",
    "history",
)


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
        "recall_counts": {
            str(key): int(round(float(recall[key]) * top1_correct.size))
            for key in RECALL_AT_K
        },
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


def _lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_registered_checkpoint(
    inventory: dict[str, object], *, torch_load: Callable[..., object]
) -> dict[str, object]:
    """Authenticate one registered checkpoint before deserializing it."""

    if type(inventory) is not dict or tuple(inventory) != PAIR_INVENTORY_KEYS:
        raise ValueError("checkpoint inventory schema differs")
    path = Path(inventory["path"])
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != inventory["bytes"]
        or _sha256_file(path) != inventory["sha256"]
    ):
        raise ValueError("checkpoint file binding differs")
    checkpoint = torch_load(path, map_location="cpu", weights_only=False, mmap=True)
    if type(checkpoint) is not dict or tuple(checkpoint) != TRAINING_CHECKPOINT_KEYS:
        raise ValueError("training checkpoint schema differs")
    return checkpoint


def _validate_pair_config(config: object) -> tuple[dict[str, object], ...]:
    if type(config) is not dict or tuple(config) != PAIR_CONFIG_KEYS:
        raise ValueError("paired evaluator configuration schema differs")
    if (
        config["schema_version"] != "unicom-full-width-pair-config-v1"
        or type(config["seed"]) is not int
        or config["seed"] not in (*SELECTION_SEEDS, *CONFIRMATION_SEEDS)
        or type(config["inventory"]) is not list
        or len(config["inventory"]) != len(EPOCHS) * len(ARMS)
    ):
        raise ValueError("paired evaluator configuration binding differs")
    inventory = tuple(config["inventory"])
    paths: set[str] = set()
    hashes: set[str] = set()
    for expected, row in zip(
        ((arm, epoch) for epoch in EPOCHS for arm in ARMS), inventory, strict=True
    ):
        if (
            type(row) is not dict
            or tuple(row) != PAIR_INVENTORY_KEYS
            or (row["arm"], row["epoch"]) != expected
            or type(row["path"]) is not str
            or not row["path"]
            or row["path"] in paths
            or not _lower_hex(row["sha256"], 64)
            or row["sha256"] in hashes
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
        ):
            raise ValueError("paired evaluator inventory differs")
        paths.add(row["path"])
        hashes.add(row["sha256"])
    return inventory


def _validate_encoded(value: object) -> dict[str, object]:
    keys = (
        "query",
        "gallery",
        "query_labels",
        "gallery_labels",
        "query_ids",
        "gallery_ids",
        "elapsed_seconds",
        "peak_allocated_bytes",
    )
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError("paired checkpoint encoder schema differs")
    if (
        type(value["elapsed_seconds"]) is not float
        or not math.isfinite(value["elapsed_seconds"])
        or value["elapsed_seconds"] < 0.0
        or type(value["peak_allocated_bytes"]) is not int
        or value["peak_allocated_bytes"] < 0
    ):
        raise ValueError("paired checkpoint encoder costs differ")
    return value


def evaluate_pair(
    config: dict[str, object],
    load_checkpoint: Callable[[dict[str, object]], object],
    encode: Callable[[Mapping[str, object], dict[str, object]], object],
) -> dict[str, object]:
    """Load and evaluate one raw-model checkpoint pair at every frozen epoch."""

    inventory = _validate_pair_config(config)
    if not callable(load_checkpoint) or not callable(encode):
        raise TypeError("paired evaluator callbacks differ")
    rows: list[dict[str, object]] = []
    for epoch_index, epoch in enumerate(EPOCHS):
        encoded_by_arm: dict[str, dict[str, object]] = {}
        inventory_by_arm: dict[str, dict[str, object]] = {}
        for arm_index, arm in enumerate(ARMS):
            inventory_row = inventory[epoch_index * len(ARMS) + arm_index]
            checkpoint = load_checkpoint(inventory_row)
            if (
                type(checkpoint) is not dict
                or checkpoint.get("epoch") != epoch
                or not isinstance(checkpoint.get("model"), Mapping)
                or checkpoint.get("selection_holdout")
                != {"seed": 0, "fraction": 0.2}
            ):
                raise ValueError("paired raw checkpoint binding differs")
            validate_arm_protocol(checkpoint.get("training_protocol"), arm)
            encoded_by_arm[arm] = _validate_encoded(
                encode(checkpoint["model"], inventory_row)
            )
            inventory_by_arm[arm] = inventory_row
        control = encoded_by_arm[ARMS[0]]
        candidate = encoded_by_arm[ARMS[1]]
        if (
            type(control["query_ids"]) is not tuple
            or type(control["gallery_ids"]) is not tuple
            or candidate["query_ids"] != control["query_ids"]
            or candidate["gallery_ids"] != control["gallery_ids"]
            or not np.array_equal(candidate["query_labels"], control["query_labels"])
            or not np.array_equal(candidate["gallery_labels"], control["gallery_labels"])
        ):
            raise ValueError("paired checkpoint evaluation records differ")
        measured = evaluate_pair_embeddings(
            control_query=control["query"],
            control_gallery=control["gallery"],
            candidate_query=candidate["query"],
            candidate_gallery=candidate["gallery"],
            query_labels=control["query_labels"],
            gallery_labels=control["gallery_labels"],
            query_ids=control["query_ids"],
            gallery_ids=control["gallery_ids"],
        )
        arm_values = {}
        for arm in ARMS:
            inventory_row = inventory_by_arm[arm]
            encoded = encoded_by_arm[arm]
            arm_values[arm] = {
                "checkpoint_path": inventory_row["path"],
                "checkpoint_sha256": inventory_row["sha256"],
                "checkpoint_bytes": inventory_row["bytes"],
                **measured["arms"][arm],
                "elapsed_seconds": encoded["elapsed_seconds"],
                "peak_allocated_bytes": encoded["peak_allocated_bytes"],
            }
        rows.append(
            {
                "epoch": epoch,
                "query_ids_sha256": measured["query_ids_sha256"],
                "gallery_ids_sha256": measured["gallery_ids_sha256"],
                "arms": arm_values,
            }
        )
    result = {
        "schema_version": "unicom-full-width-pair-result-v1",
        "seed": config["seed"],
        "rows": rows,
    }
    validate_pair_result(result, config)
    return result


def _validate_retrieval_view(value: object) -> int:
    if type(value) is not dict or tuple(value) != RETRIEVAL_VIEW_KEYS:
        raise ValueError("paired retrieval view schema differs")
    recall = value["recall"]
    recall_counts = value["recall_counts"]
    average_precision = value["average_precision"]
    top1_correct = value["top1_correct"]
    if (
        type(recall) is not dict
        or tuple(recall) != tuple(str(key) for key in RECALL_AT_K)
        or any(
            type(item) is not float or not math.isfinite(item) or not 0.0 <= item <= 1.0
            for item in recall.values()
        )
        or type(recall_counts) is not dict
        or tuple(recall_counts) != tuple(str(key) for key in RECALL_AT_K)
        or type(average_precision) is not list
        or not average_precision
        or any(
            type(item) is not float or not math.isfinite(item) or not 0.0 <= item <= 1.0
            for item in average_precision
        )
        or type(top1_correct) is not list
        or len(top1_correct) != len(average_precision)
        or any(type(item) is not bool for item in top1_correct)
        or type(value["map_at_r"]) is not float
        or not math.isfinite(value["map_at_r"])
    ):
        raise ValueError("paired retrieval view values differ")
    expected_map = float(np.asarray(average_precision, dtype=np.float64).mean())
    expected_r1 = float(np.asarray(top1_correct, dtype=np.bool_).mean())
    if (
        any(
            type(recall_counts[key]) is not int
            or recall_counts[key] < 0
            or recall_counts[key] > len(average_precision)
            or recall[key] != float(recall_counts[key] / len(average_precision))
            for key in recall_counts
        )
        or recall_counts["1"] != sum(top1_correct)
        or value["map_at_r"] != expected_map
        or recall["1"] != expected_r1
    ):
        raise ValueError("paired retrieval view aggregates differ")
    return len(average_precision)


def validate_pair_result(value: object, config: object) -> None:
    inventory = _validate_pair_config(config)
    if (
        type(value) is not dict
        or tuple(value) != PAIR_RESULT_KEYS
        or value["schema_version"] != "unicom-full-width-pair-result-v1"
        or value["seed"] != config["seed"]
        or type(value["rows"]) is not list
        or len(value["rows"]) != len(EPOCHS)
    ):
        raise ValueError("paired evaluator result schema differs")
    record_hashes: tuple[str, str] | None = None
    query_count: int | None = None
    for epoch_index, (epoch, row) in enumerate(zip(EPOCHS, value["rows"], strict=True)):
        if (
            type(row) is not dict
            or tuple(row) != PAIR_ROW_KEYS
            or row["epoch"] != epoch
            or not _lower_hex(row["query_ids_sha256"], 64)
            or not _lower_hex(row["gallery_ids_sha256"], 64)
            or type(row["arms"]) is not dict
            or tuple(row["arms"]) != ARMS
        ):
            raise ValueError("paired evaluator result row differs")
        current_hashes = (row["query_ids_sha256"], row["gallery_ids_sha256"])
        if record_hashes is None:
            record_hashes = current_hashes
        elif current_hashes != record_hashes:
            raise ValueError("paired evaluator record order differs")
        for arm_index, arm in enumerate(ARMS):
            arm_value = row["arms"][arm]
            expected = inventory[epoch_index * len(ARMS) + arm_index]
            if (
                type(arm_value) is not dict
                or tuple(arm_value) != PAIR_ARM_KEYS
                or arm_value["checkpoint_path"] != expected["path"]
                or arm_value["checkpoint_sha256"] != expected["sha256"]
                or arm_value["checkpoint_bytes"] != expected["bytes"]
                or not _lower_hex(arm_value["query_embedding_sha256"], 64)
                or not _lower_hex(arm_value["gallery_embedding_sha256"], 64)
                or type(arm_value["elapsed_seconds"]) is not float
                or not math.isfinite(arm_value["elapsed_seconds"])
                or arm_value["elapsed_seconds"] < 0.0
                or type(arm_value["peak_allocated_bytes"]) is not int
                or arm_value["peak_allocated_bytes"] < 0
            ):
                raise ValueError("paired evaluator arm binding differs")
            primary_count = _validate_retrieval_view(arm_value["primary"])
            legacy_count = _validate_retrieval_view(arm_value["legacy"])
            if primary_count != legacy_count:
                raise ValueError("paired evaluator view counts differ")
            if query_count is None:
                query_count = primary_count
            elif primary_count != query_count:
                raise ValueError("paired evaluator query count differs")


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
        if owned is not None and _same_inode(temporary, owned):
            temporary.unlink()
        raise


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(arguments)


def run_registered_pair(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    config = strict_json_object(args.config)
    _validate_pair_config(config)
    load_checkpoint, encode = build_real_pair_callbacks(args, config)
    return evaluate_pair(config, load_checkpoint, encode), config


def _git_revision(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_trainer_module():
    path = Path(__file__).resolve().with_name("train_unicom_inshop.py")
    name = "_unicom_full_width_trainer"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load UniCOM trainer source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def build_real_pair_callbacks(
    args: argparse.Namespace,
    config: dict[str, object],
    *,
    torch_module=None,
    trainer=None,
    parse_partition=None,
    split_holdout=None,
    git_revision: Callable[[Path], str] | None = None,
):
    """Build the authenticated raw-checkpoint loader and train-only encoder."""

    inventory = _validate_pair_config(config)
    if torch_module is None:
        import torch as torch_module
    if trainer is None:
        trainer = _load_trainer_module()
    if parse_partition is None:
        from sfora.unicom_inshop import parse_inshop_partition as parse_partition
    if split_holdout is None:
        from sfora.unicom_training import identity_holdout as split_holdout
    revision = _git_revision if git_revision is None else git_revision
    if revision(args.unicom_checkout) != trainer.UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if args.initial_checkpoint.name != UNICOM_INITIAL_CHECKPOINT_NAME:
        raise ValueError("initial checkpoint filename differs")
    if (
        not args.initial_checkpoint.is_file()
        or args.initial_checkpoint.is_symlink()
        or _sha256_file(args.initial_checkpoint) != trainer.UNICOM_L14_336_SHA256
    ):
        raise ValueError("UNICOM initial checkpoint differs")
    trainer_path = Path(trainer.__file__).resolve()
    partition_path = args.dataset_root / "Eval" / "list_eval_partition.txt"
    if (
        not trainer_path.is_file()
        or trainer_path.is_symlink()
        or not partition_path.is_file()
        or partition_path.is_symlink()
    ):
        raise ValueError("paired evaluator provenance path differs")
    expected_checkpoint_provenance = {
        "trainer_sha256": _sha256_file(trainer_path),
        "unicom_revision": trainer.UNICOM_REVISION,
        "initial_checkpoint_sha256": trainer.UNICOM_L14_336_SHA256,
        "partition_sha256": _sha256_file(partition_path),
    }
    if (
        type(args.batch_size) is not int
        or args.batch_size <= 0
        or type(args.workers) is not int
        or args.workers < 0
    ):
        raise ValueError("paired evaluator batch configuration differs")
    for row in inventory:
        path = Path(row["path"])
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or _sha256_file(path) != row["sha256"]
        ):
            raise ValueError("checkpoint file binding differs")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for full-width evaluation")
    records = parse_partition(args.dataset_root)
    train_records = tuple(record for record in records if record.split == "train")
    _optimization, query, gallery, _labels = split_holdout(
        train_records, fraction=0.2, seed=0
    )
    if not query or not gallery:
        raise ValueError("paired train-only holdout differs")
    model, transform = trainer._load_official_model(
        args.unicom_checkout, args.initial_checkpoint
    )
    device = torch_module.device("cuda")
    model = model.to(device)
    expected_query_labels = np.asarray([record.label for record in query])
    expected_gallery_labels = np.asarray([record.label for record in gallery])
    query_ids = tuple(str(record.image_path) for record in query)
    gallery_ids = tuple(str(record.image_path) for record in gallery)

    def load_checkpoint(row: dict[str, object]) -> dict[str, object]:
        checkpoint = load_registered_checkpoint(row, torch_load=torch_module.load)
        protocol = checkpoint["training_protocol"]
        if type(protocol) is not dict or protocol.get("seed") != config["seed"]:
            raise ValueError("checkpoint seed binding differs")
        if any(
            type(protocol.get(key)) is not str
            or protocol.get(key) != expected_value
            for key, expected_value in expected_checkpoint_provenance.items()
        ):
            raise ValueError("checkpoint provenance differs")
        return checkpoint

    def encode(
        model_state: Mapping[str, object], _row: dict[str, object]
    ) -> dict[str, object]:
        model.load_state_dict(model_state, strict=True)
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        query_values, query_labels = trainer._encode_records(
            model,
            query,
            transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        gallery_values, gallery_labels = trainer._encode_records(
            model,
            gallery,
            transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        torch_module.cuda.synchronize()
        elapsed = float(time.perf_counter() - started)
        if (
            not np.array_equal(query_labels, expected_query_labels)
            or not np.array_equal(gallery_labels, expected_gallery_labels)
        ):
            raise ValueError("paired encoder labels differ")
        return {
            "query": np.ascontiguousarray(query_values, dtype=np.float32),
            "gallery": np.ascontiguousarray(gallery_values, dtype=np.float32),
            "query_labels": query_labels,
            "gallery_labels": gallery_labels,
            "query_ids": query_ids,
            "gallery_ids": gallery_ids,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
        }

    return load_checkpoint, encode


def _validate_output_preflight(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent_info = output.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or output.parent.is_symlink():
        raise ValueError("result output parent must be a real directory")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        _validate_output_preflight(args.output)
        result, config = run_registered_pair(args)
        publish_result(
            result,
            args.output,
            validate=lambda value: validate_pair_result(value, config),
        )
    except Exception as error:
        print(f"full-width evaluation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
