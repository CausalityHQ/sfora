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
import torch

from sfora.atomic_publication import BudgetedPublisher, publish_bytes_noreplace

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
INFERENCE_OPERATIONS = (
    "official_forward",
    "full768_l2",
    "prefix512",
    "squared_euclidean",
)
RESULT_KEYS = (
    "schema",
    "phase",
    "status",
    "clause",
    "evaluator_sha256",
    "evidence_manifest",
    "evidence_sha256",
    "config",
    "sources_authority",
    "sources",
    "decision",
)


def _canonical_tensor_bytes(value: torch.Tensor) -> bytes:
    if not isinstance(value, torch.Tensor):
        raise ValueError("checkpoint inference tensor differs")
    return (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )


def checkpoint_inference_signature(
    checkpoint: object,
    *,
    structural_inventory: object,
    descriptor_sha256: object,
) -> dict[str, object]:
    """Rebuild raw-backbone value evidence directly from a terminal checkpoint."""

    if (
        type(checkpoint) is not dict
        or not isinstance(checkpoint.get("model"), dict)
        or not checkpoint["model"]
        or not _lower_sha256(descriptor_sha256)
    ):
        raise ValueError("checkpoint inference evidence differs")
    if (
        type(structural_inventory) is not dict
        or tuple(structural_inventory)
        != ("schema", "tensors", "classifier", "operations")
        or structural_inventory["schema"] != "unicom-fepf-structure-v1"
        or structural_inventory["operations"] != list(INFERENCE_OPERATIONS)
        or type(structural_inventory["tensors"]) is not list
        or not structural_inventory["tensors"]
    ):
        raise ValueError("checkpoint inference structural inventory differs")
    state = checkpoint["model"]
    structural_rows = structural_inventory["tensors"]
    row_keys = (
        "name",
        "kind",
        "shape",
        "dtype",
        "numel",
        "element_size",
        "bytes",
    )
    if any(type(row) is not dict or tuple(row) != row_keys for row in structural_rows):
        raise ValueError("checkpoint inference structural inventory differs")
    names = [row["name"] for row in structural_rows]
    if (
        names != sorted(names)
        or len(names) != len(set(names))
        or set(state) != set(names)
    ):
        raise ValueError("checkpoint inference state inventory differs")
    parameter_names = {
        row["name"] for row in structural_rows if row["kind"] == "parameter"
    }
    if not parameter_names or any(
        row["kind"] not in {"parameter", "buffer"} for row in structural_rows
    ):
        raise ValueError("checkpoint inference parameter inventory differs")
    ema = checkpoint.get("ema")
    if ema is not None:
        if type(ema) is not dict or type(ema.get("backbone")) is not dict:
            raise ValueError("checkpoint inference parameter inventory differs")
        intrinsic_parameter_names = set(ema["backbone"])
        if intrinsic_parameter_names != parameter_names:
            raise ValueError("checkpoint inference parameter inventory differs")
    for row in structural_rows:
        value = state[row["name"]]
        if (
            not isinstance(value, torch.Tensor)
            or row["shape"] != list(value.shape)
            or row["dtype"] != str(value.dtype)
            or row["numel"] != value.numel()
            or row["element_size"] != value.element_size()
            or row["bytes"] != value.numel() * value.element_size()
        ):
            raise ValueError("checkpoint inference structural inventory differs")
    classifier_row = structural_inventory["classifier"]
    classifier = checkpoint.get("classifier")
    if (
        type(classifier_row) is not dict
        or tuple(classifier_row)
        != ("shape", "dtype", "numel", "element_size", "bytes")
        or not isinstance(classifier, torch.Tensor)
        or classifier_row["shape"] != list(classifier.shape)
        or classifier_row["dtype"] != str(classifier.dtype)
        or classifier_row["numel"] != classifier.numel()
        or classifier_row["element_size"] != classifier.element_size()
        or classifier_row["bytes"]
        != classifier.numel() * classifier.element_size()
    ):
        raise ValueError("checkpoint inference classifier schema differs")
    aggregate = hashlib.sha256()
    tensors = []
    total_bytes = 0
    for structural_row in structural_rows:
        name = structural_row["name"]
        value = state[name]
        payload = _canonical_tensor_bytes(value)
        row = {
            "name": name,
            "kind": structural_row["kind"],
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "element_size": value.element_size(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        metadata = json.dumps(
            {key: row[key] for key in tuple(row)[:-1]}, separators=(",", ":")
        ).encode()
        aggregate.update(len(metadata).to_bytes(8, "big"))
        aggregate.update(metadata)
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
        total_bytes += len(payload)
        tensors.append(row)
    return {
        "schema": "unicom-inference-signature-v1",
        "tensors": tensors,
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "descriptor_dtype": "torch.float32",
        "descriptor_dimension": 512,
        "descriptor_sha256": descriptor_sha256,
        "operations": list(structural_inventory["operations"]),
    }


def require_same_arm_checkpoint_signature(
    checkpoint: object,
    *,
    recorded: object,
    structural_inventory: object,
    descriptor_sha256: object,
) -> None:
    expected = checkpoint_inference_signature(
        checkpoint,
        structural_inventory=structural_inventory,
        descriptor_sha256=descriptor_sha256,
    )
    if not _strict_typed_equal(recorded, expected):
        raise ValueError("checkpoint inference authenticity differs")


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
        normalized = {
            "query_path": row["query_path"],
            "top1_correct": row["top1_correct"],
            "ap_at_r": ap_at_r,
        }
        if "query_label" in row or "relevant_gallery_count" in row:
            if (
                type(row.get("query_label")) is not str
                or not row["query_label"]
                or type(row.get("relevant_gallery_count")) is not int
                or row["relevant_gallery_count"] <= 0
            ):
                raise ValueError("exploratory query evidence differs")
            normalized["query_label"] = row["query_label"]
            normalized["relevant_gallery_count"] = row["relevant_gallery_count"]
        normalized_queries.append(normalized)
    if len(paths) != len(set(paths)):
        raise ValueError("exploratory query order differs")
    steps = value.get("optimizer_steps_per_epoch")
    if type(steps) is not int or steps <= 0:
        raise ValueError("exploratory optimizer step count differs")
    result = {
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
    paired_keys = (
        "query_inventory",
        "gallery_inventory",
        "gallery_inventory_sha256",
        "geometry",
    )
    if any(key in value for key in paired_keys):
        if not all(key in value for key in paired_keys):
            raise ValueError("paired query unit differs")
        result.update({key: value[key] for key in paired_keys})
    return result


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


def evaluate_epoch4(
    control: object, candidate: object, *, structural_all: bool
) -> dict[str, object]:
    """Apply only the controller-visible stop-4 gate before any continuation."""

    if type(structural_all) is not bool:
        raise ValueError("epoch4 structural predicate differs")

    def arm(value: object, mode: str) -> float:
        if (
            type(value) is not dict
            or value.get("mode") != mode
            or value.get("training_seed") != 0
            or value.get("holdout_seed") != 0
            or type(value.get("history")) is not list
            or len(value["history"]) != 1
        ):
            raise ValueError("epoch4 history differs")
        row = value["history"][0]
        if (
            type(row) is not dict
            or row.get("epoch") != 4
            or type(row.get("metrics")) is not dict
        ):
            raise ValueError("epoch4 history differs")
        return _finite_float(row["metrics"].get("map_at_r"), "epoch4 mAP")

    delta = math.fsum((arm(candidate, "fepf_mean"), -arm(control, "imprinted")))
    passes = structural_all and delta >= EPOCH4_MAP_DELTA_MIN
    decision = "PASS_TO_RESUME" if passes else "CLOSE_EPOCH4"
    return {
        "decision": decision,
        "clause": decision,
        "epoch4_delta_map": delta,
        "epoch4_pass": passes,
        "structural_all": structural_all,
    }


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
    if any(key in control_values or key in candidate_values for key in (
        "query_inventory",
        "gallery_inventory",
        "gallery_inventory_sha256",
        "geometry",
    )):
        paired_query_deltas(control_values, candidate_values)
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


def paired_query_deltas(
    control: Mapping[str, object], candidate: Mapping[str, object]
) -> tuple[float, ...]:
    """Require one exact paired query/gallery/geometry unit before subtraction."""

    comparison_keys = (
        "query_inventory",
        "gallery_inventory",
        "gallery_inventory_sha256",
        "geometry",
    )
    if any(
        key not in control
        or key not in candidate
        or not _strict_typed_equal(control[key], candidate[key])
        for key in comparison_keys
    ):
        raise ValueError("paired query unit differs")
    control_rows = control.get("query_evidence")
    candidate_rows = candidate.get("query_evidence")
    if type(control_rows) is not list or type(candidate_rows) is not list:
        raise ValueError("paired query unit differs")
    control_units = [
        [row.get("query_path"), row.get("query_label"), row.get("relevant_gallery_count")]
        for row in control_rows
        if type(row) is dict
    ]
    candidate_units = [
        [row.get("query_path"), row.get("query_label"), row.get("relevant_gallery_count")]
        for row in candidate_rows
        if type(row) is dict
    ]
    if (
        control_units != control["query_inventory"]
        or candidate_units != candidate["query_inventory"]
        or not _strict_typed_equal(control_units, candidate_units)
    ):
        raise ValueError("paired query unit differs")
    return tuple(
        float(np.float64(right["ap_at_r"]) - np.float64(left["ap_at_r"]))
        for left, right in zip(control_rows, candidate_rows, strict=True)
    )


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
        normalized = {
            "query_path": row["query_path"],
            "top1_correct": row["top1_correct"],
            "ap_at_r": ap_at_r,
        }
        if "query_label" in row or "relevant_gallery_count" in row:
            if (
                type(row.get("query_label")) is not str
                or not row["query_label"]
                or type(row.get("relevant_gallery_count")) is not int
                or row["relevant_gallery_count"] <= 0
            ):
                raise ValueError("confirmation query evidence differs")
            normalized["query_label"] = row["query_label"]
            normalized["relevant_gallery_count"] = row["relevant_gallery_count"]
        normalized_queries.append(normalized)
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
    result = {
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
    paired_keys = (
        "query_inventory",
        "gallery_inventory",
        "gallery_inventory_sha256",
        "geometry",
    )
    if any(key in value for key in paired_keys):
        if not all(key in value for key in paired_keys):
            raise ValueError("paired query unit differs")
        result.update({key: value[key] for key in paired_keys})
    return result


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
        if all(key in control for key in ("query_inventory", "gallery_inventory", "geometry")):
            query_deltas = paired_query_deltas(control, candidate)
        else:
            query_deltas = tuple(
                float(
                    np.float64(candidate_row["ap_at_r"])
                    - np.float64(control_row["ap_at_r"])
                )
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


def build_evidence_manifest(entries: object) -> dict[str, object]:
    """Hash an exact ordered inventory of transitive external authorities."""

    if type(entries) is not list or not entries:
        raise ValueError("evidence manifest differs")
    normalized = []
    paths: set[Path] = set()
    identities: set[tuple[str, str]] = set()
    digest = hashlib.sha256(b"unicom-fepf-evidence-manifest-v1\0")
    for entry in entries:
        if (
            type(entry) is not dict
            or tuple(entry) != ("role", "identity", "path")
            or type(entry["role"]) is not str
            or not entry["role"]
            or type(entry["identity"]) is not str
            or not entry["identity"]
            or not isinstance(entry["path"], Path)
        ):
            raise ValueError("evidence manifest entry differs")
        path = entry["path"]
        if (
            path.absolute() != path.resolve()
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ValueError("evidence manifest path differs")
        identity = (entry["role"], entry["identity"])
        if path in paths or identity in identities:
            raise ValueError("duplicate evidence authority")
        paths.add(path)
        identities.add(identity)
        row = {
            "role": entry["role"],
            "identity": entry["identity"],
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        encoded = json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        normalized.append(row)
    return {
        "schema": "unicom-fepf-evidence-manifest-v1",
        "entries": normalized,
        "sha256": digest.hexdigest(),
    }


def validate_profile_process_order(receipts: object) -> None:
    """Require four fresh nonoverlapping C-FEPF-FEPF-C profile processes."""

    if type(receipts) is not tuple or len(receipts) != 4:
        raise ValueError("fresh profile inventory differs")
    process_evidence: set[tuple[object, ...]] = set()
    previous_finished: int | None = None
    for receipt in receipts:
        if type(receipt) is not dict:
            raise ValueError("fresh profile evidence differs")
        started = receipt.get("started_unix_ns")
        finished = receipt.get("finished_unix_ns")
        checkpoint = receipt.get("checkpoint")
        run = receipt.get("run_receipt")
        if (
            type(started) is not int
            or type(finished) is not int
            or started < 0
            or finished <= started
            or type(checkpoint) is not dict
            or not _lower_sha256(checkpoint.get("sha256"))
            or type(run) is not dict
            or not _lower_sha256(run.get("sha256"))
        ):
            raise ValueError("fresh profile evidence differs")
        token = (started, finished, checkpoint["sha256"], run["sha256"])
        if token in process_evidence:
            raise ValueError("fresh profile evidence differs")
        if previous_finished is not None and started <= previous_finished:
            raise ValueError("quality profile chronology differs")
        process_evidence.add(token)
        previous_finished = finished


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
    if phase not in {"epoch4", "exploratory", "confirmation"}:
        raise ValueError("FEPF result phase differs")
    expected_pairs = (
        ((0, 0),) if phase in {"epoch4", "exploratory"} else CONFIRMATION_PAIRS
    )
    if type(sources) is not list or len(sources) != len(expected_pairs):
        raise ValueError("FEPF result source inventory differs")
    normalized = []
    for source, expected_pair in zip(sources, expected_pairs, strict=True):
        expected_keys = (
            "training_seed",
            "holdout_seed",
            "control_root",
            "candidate_root",
            "quality_profiles",
        )
        if phase == "epoch4" and type(source) is dict:
            config = source.get("config")
            if (
                type(config) is not dict
                or tuple(config) != ("path", "sha256", "bytes")
                or type(config["path"]) is not str
                or Path(config["path"]) != Path(config["path"]).resolve()
                or not _lower_sha256(config["sha256"])
                or type(config["bytes"]) is not int
                or config["bytes"] <= 0
            ):
                raise ValueError("FEPF epoch4 config authority differs")
        if (
            type(source) is not dict
            or tuple(source)
            != (expected_keys + ("config",) if phase == "epoch4" else expected_keys)
            or type(source["training_seed"]) is not int
            or type(source["holdout_seed"]) is not int
            or (source["training_seed"], source["holdout_seed"]) != expected_pair
            or type(source["quality_profiles"]) is not list
            or len(source["quality_profiles"]) != (0 if phase == "epoch4" else 4)
        ):
            raise ValueError("FEPF result source pair order differs")
        _relative_parts(source["control_root"], "control root")
        _relative_parts(source["candidate_root"], "candidate root")
        for path in source["quality_profiles"]:
            _relative_parts(path, "quality profile")
        if len(set(source["quality_profiles"])) != len(source["quality_profiles"]):
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


def _history_observation(
    history: object, *, stop_after_epoch: int = 16
) -> tuple[list[dict[str, object]], int]:
    if stop_after_epoch not in {4, 16}:
        raise ValueError("FEPF history stop differs")
    if type(history) is not list or len(history) != stop_after_epoch:
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


def _inventory_sha256(domain: bytes, inventory: object) -> str:
    payload = json.dumps(
        inventory, allow_nan=False, ensure_ascii=False, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256(domain + b"\0")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _query_observation_from_rows(
    evaluation: object, rows: object, *, expected_epoch: int
) -> dict[str, object]:
    if (
        type(evaluation) is not dict
        or evaluation.get("epoch") != expected_epoch
        or type(rows) is not list
        or not rows
    ):
        raise ValueError("FEPF terminal query evidence differs")
    result = []
    for row in rows:
        ranked = row.get("ranked_prefix") if type(row) is dict else None
        if (
            type(row) is not dict
            or type(row.get("query_path")) is not str
            or type(row.get("query_label")) is not str
            or not row["query_label"]
            or type(row.get("relevant_gallery_count")) is not int
            or row["relevant_gallery_count"] <= 0
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
                "query_label": row["query_label"],
                "relevant_gallery_count": row["relevant_gallery_count"],
                "top1_correct": ranked[0]["correct"],
                "ap_at_r": row["ap_at_r"],
            }
        )
    query_records = evaluation.get("query_records")
    gallery_records = evaluation.get("gallery_records")
    geometry = evaluation.get("geometry")
    if (
        type(query_records) is not list
        or type(gallery_records) is not list
        or not gallery_records
        or type(geometry) is not dict
    ):
        raise ValueError("FEPF paired query unit differs")
    query_inventory = [
        [row["image_name"], row["label"], evidence["relevant_gallery_count"]]
        for row, evidence in zip(query_records, result, strict=True)
        if type(row) is dict
        and tuple(row) == ("image_name", "label")
        and row["image_name"] == evidence["query_path"]
        and row["label"] == evidence["query_label"]
    ]
    gallery_inventory = [
        [row["image_name"], row["label"]]
        for row in gallery_records
        if type(row) is dict
        and tuple(row) == ("image_name", "label")
        and type(row["image_name"]) is str
        and type(row["label"]) is str
    ]
    if len(query_inventory) != len(result) or len(gallery_inventory) != len(gallery_records):
        raise ValueError("FEPF paired query unit differs")
    return {
        "query_evidence": result,
        "query_inventory": query_inventory,
        "gallery_inventory": gallery_inventory,
        "gallery_inventory_sha256": _inventory_sha256(
            b"unicom-fepf-gallery-inventory-v1", gallery_inventory
        ),
        "geometry": dict(geometry),
    }


def load_ranked_query_observation(
    evaluation: object, *, evidence_root: Path, expected_epoch: int
) -> dict[str, object]:
    """Strict-load the separate ranked-prefix authority for one evaluation."""

    if (
        type(evaluation) is not dict
        or evaluation.get("epoch") != expected_epoch
        or not isinstance(evidence_root, Path)
    ):
        raise ValueError("FEPF terminal query evidence differs")
    root = evidence_root.resolve()
    if evidence_root.absolute() != root or not root.is_dir() or root.is_symlink():
        raise ValueError("FEPF ranked query evidence root differs")
    binding = evaluation.get("ranked_prefix_evidence")
    expected_name = f"evaluation-epoch-{expected_epoch:04d}-ranked-prefix.json"
    path = root / expected_name
    if (
        type(binding) is not dict
        or tuple(binding) != ("path", "sha256", "bytes")
        or binding["path"] != expected_name
        or type(binding["sha256"]) is not str
        or len(binding["sha256"]) != 64
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
        or path.is_symlink()
        or not path.is_file()
        or path.resolve().parent != root
    ):
        raise ValueError("FEPF ranked query evidence authority differs")
    payload = path.read_bytes()
    if (
        len(payload) != binding["bytes"]
        or hashlib.sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise ValueError("FEPF ranked query evidence authority differs")
    try:
        rows = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("FEPF ranked query evidence authority differs") from error
    if payload != (json.dumps(rows, indent=2, allow_nan=False) + "\n").encode():
        raise ValueError("FEPF ranked query evidence is noncanonical")
    row_keys = (
        "query_path", "query_label", "relevant_gallery_count", "ap_at_r",
        "query_sha256", "complete_ranking_sha256", "ranked_prefix",
    )
    ranked_keys = (
        "gallery_index", "gallery_path", "gallery_label", "score", "correct"
    )
    if type(rows) is not list or not rows:
        raise ValueError("FEPF ranked query evidence schema differs")
    for row in rows:
        ranked = row.get("ranked_prefix") if type(row) is dict else None
        if (
            type(row) is not dict
            or tuple(row) != row_keys
            or type(row["query_path"]) is not str
            or type(row["query_label"]) is not str
            or type(row["relevant_gallery_count"]) is not int
            or row["relevant_gallery_count"] <= 0
            or type(row["ap_at_r"]) is not float
            or not math.isfinite(row["ap_at_r"])
            or not _lower_sha256(row["query_sha256"])
            or not _lower_sha256(row["complete_ranking_sha256"])
            or type(ranked) is not list
            or not ranked
        ):
            raise ValueError("FEPF ranked query evidence schema differs")
        if any(
            type(item) is not dict
            or tuple(item) != ranked_keys
            or type(item["gallery_index"]) is not int
            or item["gallery_index"] < 0
            or type(item["gallery_path"]) is not str
            or type(item["gallery_label"]) is not str
            or type(item["score"]) is not float
            or not math.isfinite(item["score"])
            or type(item["correct"]) is not bool
            for item in ranked
        ):
            raise ValueError("FEPF ranked query evidence schema differs")
    return _query_observation_from_rows(
        evaluation, rows, expected_epoch=expected_epoch
    )


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
    profiles: tuple[dict[str, object], ...],
    expected_mode: str,
    training_seed: int,
    holdout_seed: int,
    stop_after_epoch: int,
    config_authority: Mapping[str, object],
    structural_inventory: Mapping[str, object],
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
        or run_receipt.get("stop_after_epoch") != stop_after_epoch
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
    normalized_history, steps = _history_observation(
        history, stop_after_epoch=stop_after_epoch
    )
    parent_root = _parent_evidence_root(run_receipt, run_root)
    initialization_path = _bound_file(
        run_receipt["initialization_receipt"],
        current_root=run_root,
        parent_root=parent_root,
        name="initialization receipt",
    )
    initialization = _strict_json_file(initialization_path)
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
    query_unit = load_ranked_query_observation(
        evaluation,
        evidence_root=terminal_path.parent,
        expected_epoch=stop_after_epoch,
    )
    checkpoint_binding = run_receipt["checkpoints"][-1]
    checkpoint_path = _bound_file(
        {key: checkpoint_binding[key] for key in ("root", "path", "sha256", "bytes")},
        current_root=run_root,
        parent_root=parent_root,
        name="terminal checkpoint",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if profiles:
        parameter_schema = [
            {
                "name": f"raw_model.{row['name']}",
                "shape": row["shape"],
                "dtype": row["dtype"],
            }
            for row in structural_inventory["tensors"]
            if row["kind"] == "parameter"
        ]
        parameter_schema.append(
            {
                "name": "classifier",
                "shape": structural_inventory["classifier"]["shape"],
                "dtype": structural_inventory["classifier"]["dtype"],
            }
        )
        if any(
            not _strict_typed_equal(profile["parameter_schema"], parameter_schema)
            for profile in profiles
        ):
            raise ValueError("checkpoint inference parameter schema differs")
    descriptor_sha256 = evaluation["evaluation_signature"]["descriptor_sha256"]
    require_same_arm_checkpoint_signature(
        checkpoint,
        recorded=run_receipt["inference_signature"],
        structural_inventory=structural_inventory,
        descriptor_sha256=descriptor_sha256,
    )
    for profile in profiles:
        require_same_arm_checkpoint_signature(
            checkpoint,
            recorded=profile["inference_signature"],
            structural_inventory=structural_inventory,
            descriptor_sha256=descriptor_sha256,
        )
    result = {
        "mode": expected_mode,
        "training_seed": training_seed,
        "holdout_seed": holdout_seed,
        "history": normalized_history,
        **query_unit,
        "initialization_seconds": initialization_seconds,
        "optimizer_steps_per_epoch": steps,
    }
    if profiles:
        step_wall, allocated, reserved = _pooled_quality_profile(
            (profiles[0], profiles[1])
        )
        result.update(
            {
                "profiled_step_wall": step_wall,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
            }
        )
    return result


def _record_evidence(
    entries: list[dict[str, object]],
    seen: set[Path],
    *,
    role: str,
    identity: str,
    path: Path,
) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    entries.append({"role": role, "identity": identity, "path": resolved})


def _record_run_transitive_evidence(
    *,
    run_root: Path,
    pair_identity: str,
    arm: str,
    trainer,
    entries: list[dict[str, object]],
    seen: set[Path],
) -> None:
    """Enumerate every validated run, parent, artifact, and descriptor preimage."""

    run_path = run_root / "run-receipt.json"
    run_receipt = _strict_json_file(run_path)
    if type(run_receipt) is not dict:
        raise ValueError("FEPF run evidence differs")
    trainer.validate_training_run_receipt_v2(run_receipt, evidence_root=run_root)
    prefix = f"{pair_identity}.{arm}"
    _record_evidence(
        entries,
        seen,
        role=f"{prefix}.run_receipt",
        identity=str(run_path),
        path=run_path,
    )
    parent_root = _parent_evidence_root(run_receipt, run_root)
    parent_binding = run_receipt.get("parent_run_receipt")
    if parent_root is not None:
        parent_path = _bound_file(
            parent_binding,
            current_root=run_root,
            parent_root=parent_root,
            name="parent run receipt",
        )
        if parent_path != parent_root / "run-receipt.json":
            raise ValueError("FEPF parent run authority differs")
        _record_run_transitive_evidence(
            run_root=parent_root,
            pair_identity=pair_identity,
            arm=f"{arm}.parent",
            trainer=trainer,
            entries=entries,
            seen=seen,
        )
    for role, binding in (
        ("initialization_receipt", run_receipt["initialization_receipt"]),
        ("history", run_receipt["history"]),
    ):
        path = _bound_file(
            binding,
            current_root=run_root,
            parent_root=parent_root,
            name=role.replace("_", " "),
        )
        _record_evidence(
            entries,
            seen,
            role=f"{prefix}.{role}",
            identity=f"{binding['root']}:{binding['path']}",
            path=path,
        )
    for collection, singular in (("checkpoints", "checkpoint"), ("evaluations", "evaluation")):
        for binding in run_receipt[collection]:
            path = _bound_file(
                {key: binding[key] for key in ("root", "path", "sha256", "bytes")},
                current_root=run_root,
                parent_root=parent_root,
                name=singular,
            )
            epoch = binding["epoch"]
            _record_evidence(
                entries,
                seen,
                role=f"{prefix}.{singular}.epoch{epoch}",
                identity=f"{binding['root']}:{binding['path']}",
                path=path,
            )
            if singular != "evaluation":
                continue
            evaluation = _strict_json_file(path)
            for descriptor_name in ("query_descriptors", "gallery_descriptors"):
                descriptor = evaluation.get(descriptor_name)
                if (
                    type(descriptor) is not dict
                    or type(descriptor.get("path")) is not str
                    or not _lower_sha256(descriptor.get("sha256"))
                    or type(descriptor.get("bytes")) is not int
                    or descriptor["bytes"] <= 0
                ):
                    raise ValueError("FEPF descriptor authority differs")
                descriptor_path = _resolve_descendant(
                    path.parent,
                    descriptor["path"],
                    name=descriptor_name.replace("_", " "),
                    directory=False,
                )
                if (
                    descriptor_path.stat().st_size != descriptor["bytes"]
                    or _sha256_file(descriptor_path) != descriptor["sha256"]
                ):
                    raise ValueError("FEPF descriptor authority differs")
                _record_evidence(
                    entries,
                    seen,
                    role=f"{prefix}.{descriptor_name}.epoch{epoch}",
                    identity=descriptor["path"],
                    path=descriptor_path,
                )


def _absolute_file_authority(value: object, *, name: str) -> Path:
    if (
        type(value) is not dict
        or tuple(value) != ("path", "sha256", "bytes")
        or type(value["path"]) is not str
        or not _lower_sha256(value["sha256"])
        or type(value["bytes"]) is not int
        or value["bytes"] <= 0
    ):
        raise ValueError(f"{name} authority differs")
    path = Path(value["path"])
    if (
        path != path.resolve()
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != value["bytes"]
        or _sha256_file(path) != value["sha256"]
    ):
        raise ValueError(f"{name} authority differs")
    return path


def _config_structural_inventory(config_authority: object) -> dict[str, object]:
    path = _absolute_file_authority(config_authority, name="FEPF config")
    config = _strict_json_file(path)
    if type(config) is not dict or type(config.get("fepf_inference_structure")) is not dict:
        raise ValueError("FEPF config structural inventory differs")
    return config["fepf_inference_structure"]


def _reload_registered_pairs(
    *,
    observed_sources: list[dict[str, object]],
    evidence_root: Path,
    phase: str,
) -> tuple[tuple[dict[str, object], ...], dict[str, object], dict[str, object]]:
    trainer, profiler = _authority_modules()
    pairs = []
    common_config = None
    evidence_entries: list[dict[str, object]] = []
    seen_evidence: set[Path] = set()
    for source in observed_sources:
        pair_identity = f"seed{source['training_seed']}-holdout{source['holdout_seed']}"
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
        for relative in source["quality_profiles"]:
            profile_path = _resolve_descendant(
                evidence_root, relative, name="quality profile", directory=False
            )
            profile = _strict_json_file(profile_path)
            if type(profile) is not dict:
                raise ValueError("FEPF quality profile differs")
            profiler.validate_quality_profile(profile)
            profile_receipts.append(profile)
            _record_evidence(
                evidence_entries,
                seen_evidence,
                role=f"{pair_identity}.quality_profile.{len(profile_receipts) - 1}",
                identity=relative,
                path=profile_path,
            )
        if phase == "epoch4":
            config_authority = source.get("config")
            if type(config_authority) is not dict:
                raise ValueError("FEPF epoch4 config authority differs")
            config_path = _absolute_file_authority(
                config_authority, name="FEPF epoch4 config"
            )
            if common_config is None:
                common_config = dict(config_authority)
            elif not _strict_typed_equal(config_authority, common_config):
                raise ValueError("FEPF epoch4 config authority differs")
            _record_evidence(
                evidence_entries,
                seen_evidence,
                role=f"{pair_identity}.config",
                identity=str(config_path),
                path=config_path,
            )
            control_profiles = ()
            candidate_profiles = ()
        else:
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
                for role in ("checkpoint", "run_receipt", "config"):
                    authority_path = Path(profile[role]["path"])
                    _record_evidence(
                        evidence_entries,
                        seen_evidence,
                        role=f"{pair_identity}.profile.{role}",
                        identity=str(authority_path),
                        path=authority_path,
                    )
            validate_profile_process_order(tuple(profile_receipts))
            control_profiles = (profile_receipts[0], profile_receipts[3])
            candidate_profiles = (profile_receipts[1], profile_receipts[2])
            config_authority = common_config
        structural_inventory = _config_structural_inventory(config_authority)
        training_seed = source["training_seed"]
        holdout_seed = source["holdout_seed"]
        control = _load_arm_observation(
            run_root=control_root,
            profiles=control_profiles,
            expected_mode="imprinted",
            training_seed=training_seed,
            holdout_seed=holdout_seed,
            stop_after_epoch=4 if phase == "epoch4" else 16,
            config_authority=config_authority,
            structural_inventory=structural_inventory,
            trainer=trainer,
        )
        candidate = _load_arm_observation(
            run_root=candidate_root,
            profiles=candidate_profiles,
            expected_mode="fepf_mean",
            training_seed=training_seed,
            holdout_seed=holdout_seed,
            stop_after_epoch=4 if phase == "epoch4" else 16,
            config_authority=config_authority,
            structural_inventory=structural_inventory,
            trainer=trainer,
        )
        _record_run_transitive_evidence(
            run_root=control_root,
            pair_identity=pair_identity,
            arm="control",
            trainer=trainer,
            entries=evidence_entries,
            seen=seen_evidence,
        )
        _record_run_transitive_evidence(
            run_root=candidate_root,
            pair_identity=pair_identity,
            arm="candidate",
            trainer=trainer,
            entries=evidence_entries,
            seen=seen_evidence,
        )
        structural_equal = True
        try:
            trainer.require_cross_arm_inference_equality(
                (
                    profile_receipts[0]["inference_signature"]
                    if profile_receipts
                    else _strict_json_file(control_root / "run-receipt.json")[
                        "inference_signature"
                    ]
                ),
                (
                    profile_receipts[1]["inference_signature"]
                    if profile_receipts
                    else _strict_json_file(candidate_root / "run-receipt.json")[
                        "inference_signature"
                    ]
                ),
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
    return tuple(pairs), common_config, build_evidence_manifest(evidence_entries)


def _recomputed_result(
    *,
    phase: str,
    sources: object | None,
    sources_authority: object,
    evidence_root: Path,
) -> dict[str, object]:
    if phase not in {"epoch4", "exploratory", "confirmation"}:
        raise ValueError("FEPF result phase differs")
    root = _validate_evidence_root(evidence_root)
    sources_path = _absolute_file_authority(
        sources_authority, name="FEPF sources"
    )
    external_sources = _strict_json_file(sources_path)
    if sources is not None and not _strict_typed_equal(sources, external_sources):
        raise ValueError("FEPF sources authority differs")
    normalized_sources = _validate_source_inventory(phase, external_sources)
    pairs, config, evidence_manifest = _reload_registered_pairs(
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
    if (
        type(evidence_manifest) is not dict
        or evidence_manifest.get("schema") != "unicom-fepf-evidence-manifest-v1"
        or type(evidence_manifest.get("entries")) is not list
        or not _lower_sha256(evidence_manifest.get("sha256"))
    ):
        raise ValueError("FEPF evidence manifest differs")
    manifest_inputs = [
        {
            "role": "source_selectors",
            "identity": str(sources_path),
            "path": sources_path,
        },
        *[
            {
                "role": row["role"],
                "identity": row["identity"],
                "path": Path(row["path"]),
            }
            for row in evidence_manifest["entries"]
        ],
    ]
    evidence_manifest = build_evidence_manifest(manifest_inputs)
    if phase == "epoch4":
        pair = pairs[0]
        decision = evaluate_epoch4(
            pair["control"],
            pair["candidate"],
            structural_all=pair["structural_equal"],
        )
    elif phase == "exploratory":
        pair = pairs[0]
        decision = evaluate_exploratory(
            pair["control"],
            pair["candidate"],
            structural_all=pair["structural_equal"],
        )
    else:
        decision = evaluate_confirmation(pairs)
    structural_valid = all(pair["structural_equal"] for pair in pairs)
    status = decision["decision"] if structural_valid else "INVALID"
    clause = decision["clause"] if structural_valid else "INVALID_STRUCTURAL_PANEL"
    return {
        "schema": "unicom-fepf-result-v1",
        "phase": phase,
        "status": status,
        "clause": clause,
        "evaluator_sha256": _sha256_file(Path(__file__).resolve()),
        "evidence_manifest": evidence_manifest,
        "evidence_sha256": evidence_manifest["sha256"],
        "config": dict(config),
        "sources_authority": dict(sources_authority),
        "sources": normalized_sources,
        "decision": decision,
    }


def build_fepf_result(
    *,
    phase: str,
    sources: object,
    sources_authority: object,
    evidence_root: Path,
) -> dict[str, object]:
    """Build a result only after reloading all registered external evidence."""

    return _recomputed_result(
        phase=phase,
        sources=sources,
        sources_authority=sources_authority,
        evidence_root=evidence_root,
    )


def validate_fepf_result(
    result: object, evidence_root: Path, *, sources_authority: object
) -> None:
    """Strictly reload every external input and recompute the complete result."""

    if (
        type(result) is not dict
        or tuple(result) != RESULT_KEYS
        or result.get("schema") != "unicom-fepf-result-v1"
        or result.get("phase") not in {"epoch4", "exploratory", "confirmation"}
    ):
        raise ValueError("FEPF result schema differs")
    expected = _recomputed_result(
        phase=result["phase"],
        sources=None,
        sources_authority=sources_authority,
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


def _inode_identity(path: Path) -> tuple[int, int] | None:
    try:
        information = path.lstat()
    except FileNotFoundError:
        return None
    return information.st_dev, information.st_ino


def _unlink_owned(path: Path, owned: tuple[int, int], directory_descriptor: int) -> bool:
    if _inode_identity(path) != owned:
        return False
    path.unlink()
    os.fsync(directory_descriptor)
    return True


def write_fepf_result_atomic(
    result: object,
    output: Path,
    temporary: Path,
    evidence_root: Path,
    *,
    sources_authority: object,
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
    validate_fepf_result(result, root, sources_authority=sources_authority)
    if _path_lexists(output):
        raise FileExistsError(output)
    if _path_lexists(temporary):
        raise FileExistsError(temporary)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    def validate(persisted_payload: bytes) -> None:
        persisted = json.loads(persisted_payload)
        if persisted_payload != payload or not _strict_typed_equal(persisted, result):
            raise RuntimeError("persisted FEPF result bytes differ")
        validate_fepf_result(persisted, root, sources_authority=sources_authority)

    published = publish_bytes_noreplace(output, payload, validator=validate)
    persisted = json.loads(published.payload)
    published.close()
    return persisted


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("epoch4", "exploratory", "confirmation"), required=True
    )
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--sources-bytes", type=int, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--publication-stage")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--authority-preflight-only", action="store_true")
    return parser.parse_args(arguments)


def validate_publication_payload_bound(
    row: object, *, destination: Path, payload: bytes, campaign_root: Path
) -> None:
    if type(row) is not dict:
        raise ValueError("publication budget row differs")
    try:
        relative = destination.resolve().relative_to(campaign_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("publication budget path differs") from error
    if row.get("path") != relative:
        raise ValueError("publication budget path differs")
    if len(payload) > row.get("persistent_bytes", -1):
        raise OSError("publication payload bytes exceed budget")


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        if args.publication_stage is None or args.campaign_root is None:
            raise ValueError(
                "evaluation publication stage and campaign root are required"
            )
        config = _strict_json_file(args.config)
        builder_path = Path(__file__).with_name("build_unicom_fepf_run_config.py")
        specification = importlib.util.spec_from_file_location(
            "evaluation_exact_budget_builder", builder_path
        )
        if specification is None or specification.loader is None:
            raise ValueError("evaluation publication budget validator differs")
        builder = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(builder)
        budget_validator = (
            builder.validate_exact_publication_budget
            if args.authority_preflight_only
            else builder.validate_external_exact_publication_budget
        )
        budget_validator(config, config.get("publication_budget"))
        budget = config.get("publication_budget")
        budget_payload = (json.dumps(budget, indent=2, allow_nan=False) + "\n").encode()
        root = Path(config.get("artifact_root", ""))
        try:
            relative_output = args.output.resolve().relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("evaluation publication budget path differs") from error
        rows = budget.get("publications") if type(budget) is dict else None
        matching = (
            [row for row in rows if row.get("path") == relative_output]
            if type(rows) is list
            else []
        )
        if args.publication_stage is not None:
            if args.campaign_root is None or args.campaign_root.resolve() != root.resolve():
                raise ValueError("evaluation campaign root differs")
            expected_name = f"{args.publication_stage}:result"
            matching = [
                row for row in matching if row.get("name") == expected_name
            ]
        if (
            hashlib.sha256(budget_payload).hexdigest()
            != config.get("publication_budget_sha256")
            or len(matching) != 1
        ):
            raise ValueError("evaluation publication budget authority differs")
        publisher = BudgetedPublisher(
            campaign_root=root,
            budget_path=root / config["publication_budget_path"],
            budget_sha256=config["publication_budget_sha256"],
            exact_budget=config["publication_budget"],
            physical_admission=not args.authority_preflight_only,
        )
        publisher.validate_payload(
            name=f"{args.publication_stage}:result",
            destination=args.output,
            payload=b"",
        )
        if args.authority_preflight_only:
            return 0
        available = os.statvfs(root)
        row = matching[0]
        if (
            available.f_bavail * available.f_frsize
            < row["persistent_bytes"] + row["temporary_bytes"]
            or available.f_favail
            < row["persistent_inodes"] + row["temporary_inodes"]
        ):
            raise OSError("evaluation publication capacity is insufficient")
        sources = _strict_json_file(args.sources)
        sources_authority = {
            "path": str(args.sources.resolve()),
            "sha256": args.sources_sha256,
            "bytes": args.sources_bytes,
        }
        result = build_fepf_result(
            phase=args.phase,
            sources=sources,
            sources_authority=sources_authority,
            evidence_root=args.evidence_root,
        )
        result_payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        validate_publication_payload_bound(
            row,
            destination=args.output,
            payload=result_payload,
            campaign_root=root,
        )
        publisher.validate_payload(
            name=f"{args.publication_stage}:result",
            destination=args.output,
            payload=result_payload,
        )
        write_fepf_result_atomic(
            result,
            args.output,
            args.temporary,
            args.evidence_root,
            sources_authority=sources_authority,
        )
    except Exception as error:
        print(f"FEPF evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
