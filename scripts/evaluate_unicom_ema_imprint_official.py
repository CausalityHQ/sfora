#!/usr/bin/env python3
"""One-shot official In-Shop readout for retained UniCOM imprinting checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import stat
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

GATING_SEEDS = (2, 3, 4, 5, 6)
SENSITIVITY_SEEDS = (1,)
ARMS = ("random_raw", "imprinted_raw")
EPOCHS = (4, 8, 12, 16)
SUMMARY_SHA256 = "cb15ffd66a45b26f65cdae99c411767f773abecdf625b6703c34fb6b7399bf23"
RUN_DIRECTORIES = {
    (1, "random_raw"): "unicom-ema-random-81f3f48-seed1-e16",
    (1, "imprinted_raw"): "unicom-ema-imprinted-81f3f48-seed1-e16-retry1",
    **{
        (seed, arm): f"unicom-ema-{arm.removesuffix('_raw')}-387d697-seed{seed}-e16"
        for seed in GATING_SEEDS
        for arm in ARMS
    },
}


def registered_rows() -> tuple[dict[str, object], ...]:
    """Return the immutable evaluation order without opening outcome data."""

    rows: list[dict[str, object]] = []
    for seed in (*GATING_SEEDS, *SENSITIVITY_SEEDS):
        for arm in ARMS:
            for epoch in EPOCHS:
                rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "epoch": epoch,
                        "gating": seed in GATING_SEEDS,
                    }
                )
    return tuple(rows)


def load_registered_summary(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("registered summary path differs")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SUMMARY_SHA256:
        raise ValueError("registered summary SHA-256 differs")
    value = json.loads(payload)
    if type(value) is not dict:
        raise ValueError("registered summary schema differs")
    return value


def _sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("registered checkpoint SHA-256 differs")
    return value


def registered_checkpoint_inventory(
    summary: dict[str, Any], root: Path
) -> tuple[dict[str, object], ...]:
    """Bind the registered summary rows to the retained DGX checkpoint paths."""

    if type(summary) is not dict or summary.get("selected_cell") != "imprinted_raw":
        raise ValueError("registered summary differs")
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("checkpoint root differs")
    reports = summary.get("reports")
    if type(reports) is not list or [report.get("seed") for report in reports] != list(range(1, 7)):
        raise ValueError("registered report seeds differ")
    by_seed = {report["seed"]: report for report in reports}
    inventory: list[dict[str, object]] = []
    hashes: set[str] = set()
    for row in registered_rows():
        seed = row["seed"]
        arm = row["arm"]
        epoch = row["epoch"]
        report = by_seed[seed]
        arm_report = report.get(arm)
        if type(arm_report) is not dict:
            raise ValueError("registered arm report differs")
        arm_hashes = arm_report.get("checkpoint_sha256_by_epoch")
        if type(arm_hashes) is not list or len(arm_hashes) != len(EPOCHS):
            raise ValueError("registered checkpoint hashes differ")
        digest = _sha256(arm_hashes[EPOCHS.index(epoch)])
        if digest in hashes:
            raise ValueError("registered checkpoint hashes are not unique")
        hashes.add(digest)
        path = root / RUN_DIRECTORIES[(seed, arm)] / f"epoch-{epoch:04d}.pt"
        inventory.append({**row, "path": str(path), "sha256": digest})
    return tuple(inventory)


def paired_t_interval(values: tuple[float, ...], *, critical: float) -> tuple[float, float]:
    """Return a two-sided paired Student-t interval for exact builtin floats."""

    if type(values) is not tuple or len(values) < 2:
        raise ValueError("paired values must contain at least two builtin floats")
    if any(type(value) is not float for value in values) or type(critical) is not float:
        raise TypeError("paired values and critical must be builtin floats")
    if not all(math.isfinite(value) for value in (*values, critical)) or critical <= 0.0:
        raise ValueError("paired values must be finite and critical positive")
    mean = statistics.fmean(values)
    half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
    return (float(mean - half_width), float(mean + half_width))


def paired_query_bootstrap_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 205,
) -> tuple[float, float]:
    """Bootstrap paired queries with one shared query draw across gating seeds."""

    if type(baseline) is not np.ndarray or type(candidate) is not np.ndarray:
        raise TypeError("paired correctness must be NumPy arrays")
    if baseline.dtype != np.bool_ or candidate.dtype != np.bool_:
        raise TypeError("paired correctness must be boolean")
    if baseline.ndim != 2 or baseline.shape[0] != len(GATING_SEEDS):
        raise ValueError("paired correctness must have one row per gating seed")
    if candidate.shape != baseline.shape or baseline.shape[1] == 0:
        raise ValueError("paired correctness shapes differ or are empty")
    if type(samples) is not int or samples != 10_000 or type(seed) is not int or seed != 205:
        raise ValueError("paired bootstrap configuration differs")
    per_query = (candidate.astype(np.float64) - baseline.astype(np.float64)).mean(
        axis=0, dtype=np.float64
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    replicates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 64):
        stop = min(start + 64, samples)
        indices = generator.integers(0, per_query.size, size=(stop - start, per_query.size))
        replicates[start:stop] = per_query[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.percentile(replicates, (2.5, 97.5))
    return (float(lower), float(upper))


def _first_epoch_reaching(metrics: dict[int, float], target: float) -> int | None:
    for epoch in EPOCHS:
        if metrics[epoch] >= target:
            return epoch
    return None


def trajectory_decision(
    by_seed: dict[int, dict[str, dict[int, float]]],
) -> dict[str, object]:
    """Apply the grid-native shared-random-endpoint trajectory decision."""

    if type(by_seed) is not dict or tuple(by_seed) != GATING_SEEDS:
        raise ValueError("trajectory seeds differ")
    rows: list[dict[str, object]] = []
    no_later = True
    by_epoch_eight = 0
    for seed in GATING_SEEDS:
        arms = by_seed[seed]
        if type(arms) is not dict or tuple(arms) != ARMS:
            raise ValueError("trajectory arms differ")
        for metrics in arms.values():
            if type(metrics) is not dict or tuple(metrics) != EPOCHS:
                raise ValueError("trajectory epochs differ")
            if any(
                type(value) is not float or not math.isfinite(value) for value in metrics.values()
            ):
                raise ValueError("trajectory metric differs")
        target = arms["random_raw"][16]
        random_epoch = _first_epoch_reaching(arms["random_raw"], target)
        imprinted_epoch = _first_epoch_reaching(arms["imprinted_raw"], target)
        if random_epoch is None or imprinted_epoch is None:
            speedup = None
            no_later = False
        else:
            speedup = float(random_epoch / imprinted_epoch)
            no_later = no_later and imprinted_epoch <= random_epoch
            by_epoch_eight += int(imprinted_epoch <= 8)
        rows.append(
            {
                "seed": seed,
                "random_epoch": random_epoch,
                "imprinted_epoch": imprinted_epoch,
                "speedup": speedup,
            }
        )
    return {"per_seed": rows, "supported": no_later and by_epoch_eight >= 3}


def _gating_deltas(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    if type(values) is not tuple or len(values) != len(GATING_SEEDS):
        raise ValueError(f"{name} must contain five values")
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise TypeError(f"{name} must contain finite builtin floats")
    return values


def quality_decision(
    *,
    primary_map_deltas: tuple[float, ...],
    legacy_map_deltas: tuple[float, ...],
    recall_at_1_deltas: tuple[float, ...],
    baseline_top1: np.ndarray,
    candidate_top1: np.ndarray,
    sensitivity_map_delta: float,
) -> dict[str, object]:
    """Recompute the frozen five-seed quality gate and six-seed sensitivity."""

    primary = _gating_deltas(primary_map_deltas, "primary mAP deltas")
    legacy = _gating_deltas(legacy_map_deltas, "legacy mAP deltas")
    recall = _gating_deltas(recall_at_1_deltas, "Recall@1 deltas")
    if type(sensitivity_map_delta) is not float or not math.isfinite(sensitivity_map_delta):
        raise TypeError("sensitivity mAP delta must be a finite builtin float")
    primary_interval = paired_t_interval(primary, critical=2.7764451052)
    recall_interval = paired_t_interval(recall, critical=2.7764451052)
    query_interval = paired_query_bootstrap_interval(baseline_top1, candidate_top1)
    six_seed_interval = paired_t_interval((sensitivity_map_delta, *primary), critical=2.5705818366)
    positive_count = sum(value > 0.0 for value in primary)
    primary_mean = statistics.fmean(primary)
    legacy_mean = statistics.fmean(legacy)
    sign_agrees = primary_mean * legacy_mean > 0.0
    per_seed_guard = min(recall) >= -0.003
    supported = (
        primary_interval[0] > 0.002
        and positive_count >= 4
        and recall_interval[0] > -0.001
        and query_interval[0] > -0.001
        and per_seed_guard
        and sign_agrees
    )
    return {
        "primary_map_interval": list(primary_interval),
        "positive_primary_seed_count": positive_count,
        "recall_at_1_seed_interval": list(recall_interval),
        "recall_at_1_query_interval": list(query_interval),
        "r1_per_seed_guard": per_seed_guard,
        "legacy_sign_agrees": sign_agrees,
        "six_seed_map_interval": list(six_seed_interval),
        "anomaly_reaudit_required": primary_mean > 0.0305,
        "exclusion_sensitive": supported and six_seed_interval[0] <= 0.002,
        "supported": supported,
    }


def _embedding_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _retrieval_payload(view: Any) -> dict[str, object]:
    if view.average_precision is None:
        raise ValueError("retrieval view omitted per-query average precision")
    recall = {str(key): float(value) for key, value in view.recall.items()}
    query_count = int(view.top1_correct.size)
    return {
        "recall": recall,
        "topk_correct_counts": {
            key: int(round(value * query_count)) for key, value in recall.items()
        },
        "map_at_r": float(view.map_at_r),
        "average_precision": view.average_precision.tolist(),
        "top1_correct": view.top1_correct.tolist(),
    }


def evaluate_embedding_views(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, object]:
    """Evaluate the registered full-unit and legacy prefix retrieval views."""

    from sfora.unicom_retrieval_audit import l2_normalize, retrieval_view

    for values, rows in (
        (query_embeddings, len(query_labels)),
        (gallery_embeddings, len(gallery_labels)),
    ):
        if (
            type(values) is not np.ndarray
            or values.dtype != np.float32
            or values.ndim != 2
            or values.shape != (rows, 768)
            or rows == 0
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
        ):
            raise ValueError("official embedding matrix differs")
    primary = retrieval_view(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        coordinates=np.arange(768, dtype=np.int64),
        normalize_before=False,
        recall_at_k=(1, 10, 20, 30, 40, 50),
    )
    legacy = retrieval_view(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
        recall_at_k=(1, 10, 20, 30, 40, 50),
    )
    query_unit = l2_normalize(query_embeddings).astype(np.float64)
    gallery_unit = l2_normalize(gallery_embeddings).astype(np.float64)
    return {
        "query_embedding_sha256": _embedding_sha256(query_embeddings),
        "gallery_embedding_sha256": _embedding_sha256(gallery_embeddings),
        "query_prefix_energy_mean": float(
            np.mean(np.sum(query_unit[:, :512] ** 2, axis=1, dtype=np.float64))
        ),
        "gallery_prefix_energy_mean": float(
            np.mean(np.sum(gallery_unit[:, :512] ** 2, axis=1, dtype=np.float64))
        ),
        "primary": _retrieval_payload(primary),
        "legacy": _retrieval_payload(legacy),
    }


def identity_uniform_map(
    average_precision: tuple[float, ...],
    labels: tuple[str, ...],
    paths: tuple[str, ...],
) -> float:
    """Average AP@R over one lexicographically first query per identity."""

    if (
        type(average_precision) is not tuple
        or type(labels) is not tuple
        or type(paths) is not tuple
        or not average_precision
        or len(average_precision) != len(labels)
        or len(labels) != len(paths)
    ):
        raise ValueError("identity-uniform query evidence shape differs")
    if any(
        type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in average_precision
    ):
        raise TypeError("identity-uniform AP@R differs")
    if any(type(value) is not str or not value for value in (*labels, *paths)):
        raise TypeError("identity-uniform labels/paths differ")
    if len(set(paths)) != len(paths):
        raise ValueError("identity-uniform query paths are not unique")
    selected: dict[str, tuple[str, float]] = {}
    for value, label, path in zip(average_precision, labels, paths, strict=True):
        previous = selected.get(label)
        if previous is None or path < previous[0]:
            selected[label] = (path, value)
    return float(statistics.fmean(value for _path, value in selected.values()))


def validate_evaluation_model(model: Any) -> None:
    """Require the frozen eval-only FP32 model contract."""

    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError("evaluation model must be a Torch module")
    training_modules = tuple(
        name or "<root>" for name, module in model.named_modules() if module.training
    )
    if training_modules:
        raise ValueError(
            f"evaluation model module {training_modules[0]} must be in eval mode"
        )
    batch_norms = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )
    expected_batch_norms = (("feature.1", 1024), ("feature.3", 768))
    if tuple(
        (name, module.num_features) for name, module in batch_norms
    ) != expected_batch_norms or any(
        type(module) is not torch.nn.BatchNorm1d for _name, module in batch_norms
    ):
        raise ValueError("evaluation model BatchNorm layout differs")
    for _name, module in batch_norms:
        if (
            module.track_running_stats is not True
            or module.affine is not True
            or type(module.eps) is not float
            or module.eps != 2e-5
            or module.weight is None
            or module.bias is None
            or module.running_mean is None
            or module.running_var is None
            or module.num_batches_tracked is None
            or module.running_mean.dtype != torch.float32
            or module.running_var.dtype != torch.float32
            or not bool(torch.isfinite(module.running_mean).all())
            or not bool(torch.isfinite(module.running_var).all())
            or bool((module.running_var < 0).any())
            or module.num_batches_tracked.dtype != torch.int64
            or module.num_batches_tracked.ndim != 0
            or int(module.num_batches_tracked) < 0
        ):
            raise ValueError("evaluation model BatchNorm running statistics differ")
    parameters = tuple(model.named_parameters())
    if not parameters or any(
        tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all())
        for _name, tensor in parameters
    ):
        raise ValueError("evaluation model parameters and buffers must be FP32")
    expected_integer_buffers = {
        "feature.1.num_batches_tracked",
        "feature.3.num_batches_tracked",
    }
    for name, tensor in model.named_buffers():
        if tensor.is_floating_point():
            if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                raise ValueError("evaluation model parameters and buffers must be FP32")
        elif (
            name not in expected_integer_buffers
            or tensor.dtype != torch.int64
            or tensor.ndim != 0
            or int(tensor) < 0
        ):
            raise ValueError("evaluation model non-floating buffer differs")


ROW_KEYS = (
    "seed",
    "arm",
    "epoch",
    "gating",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_model_key",
    "query_embedding_sha256",
    "gallery_embedding_sha256",
    "query_prefix_energy_mean",
    "gallery_prefix_energy_mean",
    "primary",
    "legacy",
    "identity_uniform_map",
    "elapsed_seconds",
    "peak_gpu_mib",
)
VIEW_KEYS = (
    "recall",
    "topk_correct_counts",
    "map_at_r",
    "average_precision",
    "top1_correct",
)
RECALL_KEYS = ("1", "10", "20", "30", "40", "50")


def _validate_view(view: object, *, query_count: int) -> None:
    if type(view) is not dict or tuple(view) != VIEW_KEYS:
        raise ValueError("evaluation view schema differs")
    recall = view["recall"]
    counts = view["topk_correct_counts"]
    if (
        type(recall) is not dict
        or tuple(recall) != RECALL_KEYS
        or type(counts) is not dict
        or tuple(counts) != RECALL_KEYS
    ):
        raise ValueError("evaluation recall schema differs")
    for key in RECALL_KEYS:
        value = recall[key]
        count = counts[key]
        if (
            type(value) is not float
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            or type(count) is not int
            or not 0 <= count <= query_count
            or value != count / query_count
        ):
            raise ValueError("evaluation recall relation differs")
    average_precision = view["average_precision"]
    top1 = view["top1_correct"]
    if (
        type(average_precision) is not list
        or len(average_precision) != query_count
        or any(
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in average_precision
        )
        or type(top1) is not list
        or len(top1) != query_count
        or any(type(value) is not bool for value in top1)
    ):
        raise ValueError("evaluation query evidence differs")
    map_at_r = view["map_at_r"]
    if (
        type(map_at_r) is not float
        or not math.isfinite(map_at_r)
        or map_at_r != float(np.mean(np.asarray(average_precision, dtype=np.float64)))
        or counts["1"] != sum(top1)
    ):
        raise ValueError("evaluation query aggregate differs")


def validate_evaluation_row(row: object, expected: object, *, query_count: int) -> None:
    """Strictly validate one row and every locally recomputable relation."""

    if type(query_count) is not int or query_count <= 0:
        raise ValueError("evaluation query count differs")
    if type(row) is not dict or tuple(row) != ROW_KEYS or type(expected) is not dict:
        raise ValueError("evaluation row schema differs")
    bindings = {
        "seed": expected.get("seed"),
        "arm": expected.get("arm"),
        "epoch": expected.get("epoch"),
        "gating": expected.get("gating"),
        "checkpoint_path": expected.get("path"),
        "checkpoint_sha256": expected.get("sha256"),
    }
    if any(
        row[key] != value or type(row[key]) is not type(value) for key, value in bindings.items()
    ):
        raise ValueError("evaluation row binding differs")
    if row["checkpoint_model_key"] != "model":
        raise ValueError("evaluation checkpoint model key differs")
    for key in (
        "checkpoint_sha256",
        "query_embedding_sha256",
        "gallery_embedding_sha256",
    ):
        _sha256(row[key])
    for key in ("query_prefix_energy_mean", "gallery_prefix_energy_mean"):
        value = row[key]
        if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("evaluation prefix energy differs")
    _validate_view(row["primary"], query_count=query_count)
    _validate_view(row["legacy"], query_count=query_count)
    identity = row["identity_uniform_map"]
    if (
        type(identity) is not dict
        or tuple(identity) != ("primary", "legacy")
        or any(
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in identity.values()
        )
    ):
        raise ValueError("evaluation identity-uniform mAP@R differs")
    elapsed = row["elapsed_seconds"]
    peak = row["peak_gpu_mib"]
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("evaluation elapsed time differs")
    if type(peak) is not int or peak < 0:
        raise ValueError("evaluation peak GPU allocation differs")


def evaluate_loaded_checkpoint(
    model: Any,
    checkpoint: object,
    encode: Callable[[Any], object],
) -> dict[str, object]:
    """Strict-load one raw checkpoint and evaluate it without touching EMA state."""

    import torch

    if type(checkpoint) is not dict or type(checkpoint.get("model")) not in (
        dict,
        OrderedDict,
    ):
        raise ValueError("checkpoint raw model state differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    validate_evaluation_model(model)
    with torch.inference_mode():
        encoded = encode(model)
    if type(encoded) is not tuple or len(encoded) != 5:
        raise ValueError("checkpoint encoder result differs")
    query, gallery, query_labels, gallery_labels, query_paths = encoded
    views = evaluate_embedding_views(query, gallery, query_labels, gallery_labels)
    if type(query_paths) is not tuple or len(query_paths) != query.shape[0]:
        raise ValueError("checkpoint query paths differ")
    labels = tuple(query_labels.tolist())
    identity = {
        name: identity_uniform_map(tuple(views[name]["average_precision"]), labels, query_paths)
        for name in ("primary", "legacy")
    }
    return {**views, "identity_uniform_map": identity}


def progress_line(
    *,
    row_index: int,
    inventory_row: dict[str, object],
    elapsed_seconds: float,
    peak_gpu_mib: int,
) -> str:
    """Return the exact outcome-free progress record."""

    if type(row_index) is not int or row_index <= 0:
        raise ValueError("progress row index differs")
    if type(inventory_row) is not dict:
        raise TypeError("progress inventory row differs")
    if type(elapsed_seconds) is not float or not math.isfinite(elapsed_seconds):
        raise ValueError("progress elapsed time differs")
    if type(peak_gpu_mib) is not int or peak_gpu_mib < 0:
        raise ValueError("progress peak allocation differs")
    payload = {
        "row_index": row_index,
        "seed": inventory_row.get("seed"),
        "arm": inventory_row.get("arm"),
        "epoch": inventory_row.get("epoch"),
        "checkpoint_sha256": _sha256(inventory_row.get("sha256")),
        "elapsed_seconds": elapsed_seconds,
        "peak_gpu_mib": peak_gpu_mib,
    }
    if (
        type(payload["seed"]) is not int
        or payload["arm"] not in ARMS
        or type(payload["epoch"]) is not int
        or payload["epoch"] not in EPOCHS
    ):
        raise ValueError("progress row binding differs")
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


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
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("strict JSON path differs")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("strict JSON exceeds 64 MiB")
    value = json.loads(
        path.read_text(),
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


RESULT_KEYS = (
    "schema_version",
    "design_commit",
    "source_commit",
    "dataset",
    "environment",
    "evaluation",
    "attempts",
    "inventory",
    "rows",
    "decisions",
    "status",
)

RUN_CONFIG_KEYS = (
    "schema_version",
    "design_commit",
    "source_commit",
    "dataset_root",
    "partition_sha256",
    "checkpoint_root",
    "summary_path",
    "unicom_checkout",
    "unicom_revision",
    "trainer_sha256_by_seed",
    "initial_checkpoint",
    "initial_checkpoint_sha256",
    "output",
    "batch_size",
    "workers",
    "attempt",
    "prior_attempt_exit_statuses",
    "inventory",
)


def validate_run_config(config: object) -> None:
    """Validate the exact ordinary-Git run configuration without opening inputs."""

    if type(config) is not dict or tuple(config) != RUN_CONFIG_KEYS:
        raise ValueError("official run configuration schema differs")
    if config["schema_version"] != "unicom-ema-imprint-official-run-v1":
        raise ValueError("official run configuration version differs")
    _commit(config["design_commit"])
    _commit(config["source_commit"])
    _sha256(config["partition_sha256"])
    _commit(config["unicom_revision"])
    trainer_hashes = config["trainer_sha256_by_seed"]
    if (
        type(trainer_hashes) is not list
        or len(trainer_hashes) != 6
        or any(
            type(row) is not dict
            or tuple(row) != ("seed", "sha256")
            or row["seed"] != seed
            for seed, row in enumerate(trainer_hashes, start=1)
        )
    ):
        raise ValueError("official run trainer authority differs")
    for row in trainer_hashes:
        _sha256(row["sha256"])
    _sha256(config["initial_checkpoint_sha256"])
    for key in ("dataset_root", "checkpoint_root", "unicom_checkout", "initial_checkpoint"):
        value = config[key]
        if type(value) is not str or not Path(value).is_absolute():
            raise ValueError(f"official run {key} differs")
    if Path(config["initial_checkpoint"]).name != "FP16-ViT-L-14-336px.pt":
        raise ValueError("official run initial checkpoint basename differs")
    for key in ("summary_path", "output"):
        value = config[key]
        if (
            type(value) is not str
            or Path(value).is_absolute()
            or ".." in Path(value).parts
            or not value
        ):
            raise ValueError(f"official run {key} differs")
    if config["batch_size"] != 128 or config["workers"] != 4:
        raise ValueError("official run loader configuration differs")
    attempt = config["attempt"]
    prior_statuses = config["prior_attempt_exit_statuses"]
    if (
        type(attempt) is not int
        or attempt not in (1, 2)
        or type(prior_statuses) is not list
        or len(prior_statuses) != attempt - 1
        or any(type(value) is not int or value == 0 for value in prior_statuses)
    ):
        raise ValueError("official run attempt history differs")
    inventory = config["inventory"]
    if type(inventory) is not list or len(inventory) != 48:
        raise ValueError("official run inventory count differs")
    root = Path(config["checkpoint_root"])
    hashes: set[str] = set()
    for registered, row in zip(registered_rows(), inventory, strict=True):
        if (
            type(row) is not dict
            or tuple(row) != ("seed", "arm", "epoch", "gating", "path", "sha256")
            or any(row[key] != value for key, value in registered.items())
            or type(row["path"]) is not str
            or not Path(row["path"]).is_absolute()
            or not Path(row["path"]).is_relative_to(root)
        ):
            raise ValueError("official run inventory order differs")
        digest = _sha256(row["sha256"])
        if digest in hashes:
            raise ValueError("official run inventory hashes are not unique")
        hashes.add(digest)


def validate_summary_authority(
    summary: dict[str, object], config: dict[str, object]
) -> None:
    """Cross-bind dataset and UniCOM authorities to every registered training arm."""

    validate_run_config(config)
    reports = summary.get("reports") if type(summary) is dict else None
    if type(reports) is not list or len(reports) != 6:
        raise ValueError("registered summary authority differs")
    shared_expected = {
        "partition_sha256": config["partition_sha256"],
        "unicom_revision": config["unicom_revision"],
        "initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
    }
    trainer_by_seed = {
        row["seed"]: row["sha256"] for row in config["trainer_sha256_by_seed"]
    }
    for expected_seed, report in enumerate(reports, start=1):
        if type(report) is not dict:
            raise ValueError("registered summary authority differs")
        for key in ("random_training_protocol", "imprinted_training_protocol"):
            protocol = report.get(key)
            if type(protocol) is not dict or any(
                protocol.get(name) != value for name, value in shared_expected.items()
            ) or report.get("seed") != expected_seed or protocol.get(
                "trainer_sha256"
            ) != trainer_by_seed[expected_seed]:
                raise ValueError("registered summary authority differs")


def attempt_history(config: dict[str, object]) -> list[dict[str, int]]:
    validate_run_config(config)
    return [
        {"attempt": index, "exit_status": status}
        for index, status in enumerate(config["prior_attempt_exit_statuses"], start=1)
    ] + [{"attempt": config["attempt"], "exit_status": 0}]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_registered_inputs(
    config: dict[str, object],
    *,
    hash_file: Callable[[Path], str] = sha256_file,
    read_partition: Callable[[Path], bytes] = Path.read_bytes,
) -> bytes:
    """Authenticate all retained checkpoints before opening official split bytes."""

    validate_run_config(config)
    for row in config["inventory"]:
        if hash_file(Path(row["path"])) != row["sha256"]:
            raise ValueError("registered checkpoint bytes differ")
    if hash_file(Path(config["initial_checkpoint"])) != config["initial_checkpoint_sha256"]:
        raise ValueError("registered initial checkpoint bytes differ")
    partition = Path(config["dataset_root"]) / "Eval" / "list_eval_partition.txt"
    payload = read_partition(partition)
    if (
        type(payload) is not bytes
        or hashlib.sha256(payload).hexdigest() != config["partition_sha256"]
    ):
        raise ValueError("official partition bytes differ")
    return payload


def configure_deterministic_runtime(torch_module) -> dict[str, str]:
    """Freeze strict FP32 deterministic inference settings before any model work."""

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG differs")
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    return {
        "python": platform.python_version(),
        "torch": str(torch_module.__version__),
        "numpy": str(np.__version__),
        "cuda": str(torch_module.version.cuda or "none"),
        "cublas_workspace_config": ":4096:8",
    }


def evaluate_registered_rows(
    inventory: list[dict[str, object]],
    *,
    query_count: int,
    evaluate: Callable[[dict[str, object]], dict[str, object]],
    emit_progress: Callable[[str], None],
) -> list[dict[str, object]]:
    """Evaluate every registered row serially and return only a complete panel."""

    if type(inventory) is not list or len(inventory) != 48:
        raise ValueError("registered evaluation inventory differs")
    if type(query_count) is not int or query_count <= 0:
        raise ValueError("registered evaluation query count differs")
    completed: list[dict[str, object]] = []
    for index, expected in enumerate(inventory, start=1):
        row = evaluate(expected)
        validate_evaluation_row(row, expected, query_count=query_count)
        completed.append(row)
        emit_progress(
            progress_line(
                row_index=index,
                inventory_row=expected,
                elapsed_seconds=row["elapsed_seconds"],
                peak_gpu_mib=row["peak_gpu_mib"],
            )
        )
    return completed


def _git_output(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def authenticate_source(config: dict[str, object], project_root: Path) -> None:
    """Bind executed source bytes to the reviewed source commit in a clean checkout."""

    validate_run_config(config)
    if not project_root.is_absolute() or not project_root.is_dir() or project_root.is_symlink():
        raise ValueError("official project root differs")
    head = _git_output(project_root, "rev-parse", "HEAD")
    _commit(head)
    source_commit = config["source_commit"]
    subprocess.run(
        ["git", "-C", str(project_root), "merge-base", "--is-ancestor", source_commit, head],
        check=True,
        capture_output=True,
    )
    symbolic = subprocess.run(
        ["git", "-C", str(project_root), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
    )
    if symbolic.returncode == 0:
        raise ValueError("official checkout must be detached")
    if _git_output(project_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("official checkout must be clean")
    for relative in (
        "scripts/evaluate_unicom_ema_imprint_official.py",
        "scripts/train_unicom_inshop.py",
        "src/sfora/unicom_retrieval_audit.py",
        "src/sfora/unicom_inshop.py",
    ):
        worktree = (project_root / relative).read_bytes()
        blob = subprocess.run(
            ["git", "-C", str(project_root), "show", f"{source_commit}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if worktree != blob:
            raise ValueError(f"official source bytes differ: {relative}")


def _load_local_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load registered module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise ValueError(f"registered module origin differs: {path}")
    return module


def activate_project_source(project_root: Path) -> None:
    """Force every project import to originate from the executing checkout."""

    source_root = (project_root / "src").resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name != "sfora" and not name.startswith("sfora."):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(source_root):
            raise ValueError(f"preloaded project module origin differs: {name}")
    source_text = str(source_root)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)
    importlib.invalidate_caches()
    import sfora.unicom_inshop as inshop

    if not Path(inshop.__file__).resolve().is_relative_to(source_root):
        raise ValueError("executing In-Shop module origin differs")


def run_official(config: dict[str, object], *, project_root: Path) -> dict[str, object]:
    """Run the complete retained-checkpoint official readout once in frozen order."""

    authenticate_source(config, project_root)
    summary = load_registered_summary(project_root / config["summary_path"])
    validate_summary_authority(summary, config)
    expected_inventory = list(
        registered_checkpoint_inventory(summary, Path(config["checkpoint_root"]))
    )
    if config["inventory"] != expected_inventory:
        raise ValueError("official configuration inventory differs from registered summary")
    partition_payload = preflight_registered_inputs(config)

    import torch

    environment = configure_deterministic_runtime(torch)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official checkpoint evaluation")
    activate_project_source(project_root)
    trainer = _load_local_module(
        project_root / "scripts" / "train_unicom_inshop.py",
        "_unicom_official_retained_trainer",
    )
    if trainer._git_revision(Path(config["unicom_checkout"])) != config["unicom_revision"]:
        raise ValueError("registered UniCOM revision differs")
    if config["initial_checkpoint_sha256"] != trainer.UNICOM_L14_336_SHA256:
        raise ValueError("registered initial model authority differs")

    from sfora.unicom_inshop import parse_inshop_partition

    records = parse_inshop_partition(Path(config["dataset_root"]))
    if hashlib.sha256(partition_payload).hexdigest() != config["partition_sha256"]:
        raise AssertionError("official partition binding changed")
    query_records = tuple(record for record in records if record.split == "query")
    gallery_records = tuple(record for record in records if record.split == "gallery")
    if len(query_records) != 14_218 or len(gallery_records) != 12_612:
        raise ValueError("official query/gallery counts differ")
    test_identities = {record.label for record in query_records}
    if test_identities != {record.label for record in gallery_records}:
        raise ValueError("official test identity membership differs")
    if len(test_identities) != 3_985:
        raise ValueError("official test identity count differs")

    model, transform = trainer._load_official_model(
        Path(config["unicom_checkout"]), Path(config["initial_checkpoint"])
    )
    device = torch.device("cuda")
    model = model.float().to(device).eval()
    transform_repr = repr(transform)
    transform_sha256 = hashlib.sha256(transform_repr.encode()).hexdigest()
    query_paths = tuple(str(record.image_path) for record in query_records)

    def evaluate_one(expected: dict[str, object]) -> dict[str, object]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        checkpoint = torch.load(
            Path(expected["path"]),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )

        def encode(active_model):
            query, query_labels = trainer._encode_records(
                active_model,
                query_records,
                transform,
                device=device,
                batch_size=config["batch_size"],
                workers=config["workers"],
            )
            gallery, gallery_labels = trainer._encode_records(
                active_model,
                gallery_records,
                transform,
                device=device,
                batch_size=config["batch_size"],
                workers=config["workers"],
            )
            return query, gallery, query_labels, gallery_labels, query_paths

        measured = evaluate_loaded_checkpoint(model, checkpoint, encode)
        torch.cuda.synchronize(device)
        elapsed = float(time.perf_counter() - started)
        peak = int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
        return {
            "seed": expected["seed"],
            "arm": expected["arm"],
            "epoch": expected["epoch"],
            "gating": expected["gating"],
            "checkpoint_path": expected["path"],
            "checkpoint_sha256": expected["sha256"],
            "checkpoint_model_key": "model",
            **measured,
            "elapsed_seconds": elapsed,
            "peak_gpu_mib": peak,
        }

    rows = evaluate_registered_rows(
        expected_inventory,
        query_count=len(query_records),
        evaluate=evaluate_one,
        emit_progress=lambda line: print(line, flush=True),
    )
    return build_result(
        design_commit=config["design_commit"],
        source_commit=config["source_commit"],
        dataset={
            "partition_sha256": config["partition_sha256"],
            "query_count": len(query_records),
            "gallery_count": len(gallery_records),
            "test_identity_count": len(test_identities),
        },
        environment=environment,
        evaluation={
            "transform_repr": transform_repr,
            "transform_sha256": transform_sha256,
            "batch_size": config["batch_size"],
            "workers": config["workers"],
            "dtype": "float32",
        },
        attempts=attempt_history(config),
        inventory=expected_inventory,
        rows=rows,
    )


def _derive_decisions(rows: list[dict[str, object]]) -> dict[str, object]:
    lookup = {
        (row["seed"], row["arm"], row["epoch"]): row
        for row in rows
    }
    primary: list[float] = []
    legacy: list[float] = []
    recall: list[float] = []
    baseline_top1: list[list[bool]] = []
    candidate_top1: list[list[bool]] = []
    trajectory: dict[int, dict[str, dict[int, float]]] = {}
    for seed in GATING_SEEDS:
        random_endpoint = lookup[(seed, "random_raw", 16)]
        imprinted_endpoint = lookup[(seed, "imprinted_raw", 16)]
        primary.append(
            imprinted_endpoint["primary"]["map_at_r"]
            - random_endpoint["primary"]["map_at_r"]
        )
        legacy.append(
            imprinted_endpoint["legacy"]["map_at_r"]
            - random_endpoint["legacy"]["map_at_r"]
        )
        recall.append(
            imprinted_endpoint["primary"]["recall"]["1"]
            - random_endpoint["primary"]["recall"]["1"]
        )
        baseline_top1.append(random_endpoint["primary"]["top1_correct"])
        candidate_top1.append(imprinted_endpoint["primary"]["top1_correct"])
        trajectory[seed] = {
            arm: {
                epoch: lookup[(seed, arm, epoch)]["primary"]["map_at_r"]
                for epoch in EPOCHS
            }
            for arm in ARMS
        }
    sensitivity = (
        lookup[(1, "imprinted_raw", 16)]["primary"]["map_at_r"]
        - lookup[(1, "random_raw", 16)]["primary"]["map_at_r"]
    )
    quality = quality_decision(
        primary_map_deltas=tuple(primary),
        legacy_map_deltas=tuple(legacy),
        recall_at_1_deltas=tuple(recall),
        baseline_top1=np.asarray(baseline_top1, dtype=np.bool_),
        candidate_top1=np.asarray(candidate_top1, dtype=np.bool_),
        sensitivity_map_delta=sensitivity,
    )
    trajectory_result = trajectory_decision(trajectory)
    return {
        "quality": quality,
        "trajectory": trajectory_result,
        "retained_checkpoint_gate_passed": (
            quality["supported"] is True and trajectory_result["supported"] is True
        ),
    }


def build_result(
    *,
    design_commit: str,
    source_commit: str,
    dataset: dict[str, object],
    environment: dict[str, object],
    evaluation: dict[str, object],
    attempts: list[dict[str, object]],
    inventory: list[dict[str, object]],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    decisions = _derive_decisions(rows)
    return {
        "schema_version": "unicom-ema-imprint-official-v1",
        "design_commit": design_commit,
        "source_commit": source_commit,
        "dataset": dataset,
        "environment": environment,
        "evaluation": evaluation,
        "attempts": attempts,
        "inventory": inventory,
        "rows": rows,
        "decisions": decisions,
        "status": "PASS" if decisions["retained_checkpoint_gate_passed"] else "FAIL",
    }


def _commit(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("result Git commit differs")
    return value


def validate_result(result: object) -> None:
    """Strictly validate the complete result and recompute every decision."""

    if type(result) is not dict or tuple(result) != RESULT_KEYS:
        raise ValueError("official result schema differs")
    if result["schema_version"] != "unicom-ema-imprint-official-v1":
        raise ValueError("official result schema version differs")
    _commit(result["design_commit"])
    _commit(result["source_commit"])
    dataset = result["dataset"]
    if (
        type(dataset) is not dict
        or tuple(dataset)
        != ("partition_sha256", "query_count", "gallery_count", "test_identity_count")
    ):
        raise ValueError("official result dataset schema differs")
    _sha256(dataset["partition_sha256"])
    if any(
        type(dataset[key]) is not int or dataset[key] <= 0
        for key in ("query_count", "gallery_count", "test_identity_count")
    ):
        raise ValueError("official result dataset counts differ")
    environment = result["environment"]
    if (
        type(environment) is not dict
        or tuple(environment) != ("python", "torch", "numpy", "cuda", "cublas_workspace_config")
        or any(type(value) is not str or not value for value in environment.values())
        or environment["cublas_workspace_config"] != ":4096:8"
    ):
        raise ValueError("official result environment differs")
    evaluation = result["evaluation"]
    if (
        type(evaluation) is not dict
        or tuple(evaluation)
        != ("transform_repr", "transform_sha256", "batch_size", "workers", "dtype")
        or type(evaluation["transform_repr"]) is not str
        or not evaluation["transform_repr"]
        or evaluation["dtype"] != "float32"
        or type(evaluation["batch_size"]) is not int
        or evaluation["batch_size"] <= 0
        or type(evaluation["workers"]) is not int
        or evaluation["workers"] < 0
    ):
        raise ValueError("official result evaluation schema differs")
    _sha256(evaluation["transform_sha256"])
    attempts = result["attempts"]
    if (
        type(attempts) is not list
        or not 1 <= len(attempts) <= 2
        or any(
            type(value) is not dict
            or tuple(value) != ("attempt", "exit_status")
            or value["attempt"] != index
            or type(value["exit_status"]) is not int
            for index, value in enumerate(attempts, start=1)
        )
        or attempts[-1]["exit_status"] != 0
        or any(value["exit_status"] == 0 for value in attempts[:-1])
    ):
        raise ValueError("official result attempts differ")
    inventory = result["inventory"]
    rows = result["rows"]
    if (
        type(inventory) is not list
        or type(rows) is not list
        or len(inventory) != 48
        or len(rows) != 48
    ):
        raise ValueError("official result row count differs")
    for registered, expected, row in zip(registered_rows(), inventory, rows, strict=True):
        if (
            type(expected) is not dict
            or tuple(expected) != ("seed", "arm", "epoch", "gating", "path", "sha256")
            or any(expected[key] != value for key, value in registered.items())
        ):
            raise ValueError("official result inventory order differs")
        _sha256(expected["sha256"])
        validate_evaluation_row(row, expected, query_count=dataset["query_count"])
    expected_decisions = _derive_decisions(rows)
    if result["decisions"] != expected_decisions:
        raise ValueError("official result decisions differ")
    expected_status = (
        "PASS" if expected_decisions["retained_checkpoint_gate_passed"] else "FAIL"
    )
    if type(result["status"]) is not str or result["status"] != expected_status:
        raise ValueError("official result status differs")


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    try:
        if sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1:
            raise ValueError("official evaluator requires isolated Python with -I -B")
        args = parse_args(arguments)
        config = strict_json_object(args.config)
        validate_run_config(config)
        result = run_official(config, project_root=Path(__file__).resolve().parents[1])
        output = Path(__file__).resolve().parents[1] / config["output"]
        publish_result(result, output, validate=validate_result)
    except Exception as error:
        print(f"official evaluation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
