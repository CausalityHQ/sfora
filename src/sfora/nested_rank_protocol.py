"""Deterministic class-disjoint evaluation and identity-balanced schedules."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np
from numpy.typing import NDArray

from sfora.representation_ceiling import deterministic_class_partition

type SampleId = str | int


def ordered_training_records_sha256(
    sample_ids: NDArray[np.int64],
    labels: NDArray[np.int64],
    relative_paths: tuple[str, ...],
) -> str:
    """Hash an ordered train-only identity/label/path table."""

    if (
        type(sample_ids) is not np.ndarray
        or sample_ids.dtype != np.int64
        or sample_ids.ndim != 1
        or sample_ids.size == 0
        or len(set(sample_ids.tolist())) != sample_ids.size
        or type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.shape != sample_ids.shape
        or np.any(labels < 0)
        or type(relative_paths) is not tuple
        or len(relative_paths) != sample_ids.size
        or len(set(relative_paths)) != len(relative_paths)
    ):
        raise ValueError("ordered training record authority differs")
    digest = hashlib.sha256()
    for sample_id, label, path_text in zip(sample_ids, labels, relative_paths, strict=True):
        if type(path_text) is not str or not path_text:
            raise ValueError("ordered training record authority differs")
        path = PurePosixPath(path_text)
        if path.is_absolute() or ".." in path.parts or str(path) != path_text:
            raise ValueError("ordered training record authority differs")
        digest.update(str(int(sample_id)).encode())
        digest.update(b"\0")
        digest.update(str(int(label)).encode())
        digest.update(b"\0")
        digest.update(path_text.encode())
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ClassDisjointFold:
    """Row memberships for an optimization/validation class split."""

    optimization: tuple[int, ...]
    validation: tuple[int, ...]
    validation_queries: tuple[int, ...]


def _labels_array(labels: NDArray[np.int64]) -> NDArray[np.int64]:
    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.ndim != 1
        or labels.size == 0
        or np.any(labels < 0)
    ):
        raise ValueError("class-disjoint protocol authority differs")
    return labels


def class_disjoint_fold(
    sample_ids: tuple[SampleId, ...], labels: NDArray[np.int64], *, seed: int
) -> ClassDisjointFold:
    """Reuse the sealed 80/20 class partition and query every validation row."""

    labels = _labels_array(labels)
    if (
        type(sample_ids) is not tuple
        or len(sample_ids) != labels.size
        or len(set(sample_ids)) != len(sample_ids)
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("class-disjoint protocol authority differs")
    _, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 2):
        raise ValueError("class-disjoint protocol authority differs")
    concrete_labels = tuple(int(label) for label in labels)
    partition = deterministic_class_partition(concrete_labels, fit_fraction=0.8, seed=seed)
    return ClassDisjointFold(
        optimization=partition.fit_row_indexes,
        validation=partition.validation_row_indexes,
        validation_queries=partition.validation_row_indexes,
    )


def shared_optimization_rows(
    sample_ids: tuple[SampleId, ...],
    labels: NDArray[np.int64],
    *,
    seeds: tuple[int, ...],
) -> tuple[int, ...]:
    """Return rows whose classes are outside every confirmation validation fold."""

    labels = _labels_array(labels)
    if (
        type(seeds) is not tuple
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds)
    ):
        raise ValueError("shared optimization authority differs")
    allowed = set(range(labels.size))
    for seed in seeds:
        allowed.intersection_update(class_disjoint_fold(sample_ids, labels, seed=seed).optimization)
    rows = tuple(row for row in range(labels.size) if row in allowed)
    if not rows:
        raise ValueError("shared optimization authority differs")
    return rows


def _domain_seed(seed: int, domain: bytes) -> int:
    digest = hashlib.sha256(seed.to_bytes(8, "little", signed=False) + domain).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _nearest_class_neighbors(
    classes: tuple[int, ...],
    centroids: NDArray[np.float64],
    *,
    retained: int = 31,
    block_rows: int = 256,
) -> dict[int, tuple[int, ...]]:
    """Retain the exact nearest class prefix using bounded GEMM blocks."""

    if (
        type(classes) is not tuple
        or len(classes) < retained + 1
        or len(set(classes)) != len(classes)
        or type(centroids) is not np.ndarray
        or centroids.dtype != np.float64
        or centroids.ndim != 2
        or centroids.shape[0] != len(classes)
        or not np.isfinite(centroids).all()
        or type(retained) is not int
        or not 1 <= retained < len(classes)
        or type(block_rows) is not int
        or block_rows <= 0
    ):
        raise ValueError("nearest class neighbor authority differs")
    result: dict[int, tuple[int, ...]] = {}
    for start in range(0, len(classes), block_rows):
        stop = min(start + block_rows, len(classes))
        similarities = centroids[start:stop] @ centroids.T
        for local_position, source_position in enumerate(range(start, stop)):
            row = similarities[local_position]
            row[source_position] = -np.inf
            boundary = float(np.partition(row, len(row) - retained)[len(row) - retained])
            candidates = np.flatnonzero(row >= boundary)
            ordered = sorted(
                (int(candidate) for candidate in candidates),
                key=lambda candidate: (-float(row[candidate]), classes[candidate]),
            )[:retained]
            if len(ordered) != retained:
                raise ValueError("nearest class neighbor inventory differs")
            result[classes[source_position]] = tuple(classes[candidate] for candidate in ordered)
    return result


def identity_balanced_schedule(
    labels: NDArray[np.int64],
    teacher_rows: NDArray[np.float32],
    optimization_rows: tuple[int, ...],
    *,
    seed: int,
    steps: int,
) -> tuple[tuple[int, ...], ...]:
    """Build collision-safe 32-by-4 batches around frozen teacher neighbors."""

    labels = _labels_array(labels)
    if (
        type(teacher_rows) is not np.ndarray
        or teacher_rows.dtype != np.float32
        or teacher_rows.ndim != 2
        or teacher_rows.shape[0] != labels.size
        or teacher_rows.shape[1] == 0
        or not np.isfinite(teacher_rows).all()
        or np.any(np.linalg.norm(teacher_rows, axis=1) == 0.0)
        or type(optimization_rows) is not tuple
        or len(set(optimization_rows)) != len(optimization_rows)
        or any(type(row) is not int or not 0 <= row < labels.size for row in optimization_rows)
        or type(seed) is not int
        or not 0 <= seed < 2**63
        or type(steps) is not int
        or steps <= 0
    ):
        raise ValueError("identity schedule authority differs")
    class_rows: dict[int, tuple[int, ...]] = {}
    for row in optimization_rows:
        label = int(labels[row])
        class_rows.setdefault(label, ())
        class_rows[label] += (row,)
    classes = tuple(sorted(class_rows))
    if len(classes) < 32:
        raise ValueError("identity schedule authority differs")

    centroids = []
    for label in classes:
        centroid = teacher_rows[list(class_rows[label])].astype(np.float64).mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("identity schedule authority differs")
        centroids.append(centroid / norm)
    centroid_matrix = np.stack(centroids)
    neighbor_order = _nearest_class_neighbors(classes, centroid_matrix)

    anchor_rng = np.random.Generator(np.random.PCG64(_domain_seed(seed, b"anchors")))
    anchor_order: list[int] = []
    anchor_cursor = 0
    image_rng = {
        label: np.random.Generator(
            np.random.PCG64(
                _domain_seed(seed, b"images" + label.to_bytes(8, "little", signed=True))
            )
        )
        for label in classes
    }
    image_orders: dict[int, list[int]] = {label: [] for label in classes}
    image_cursors: dict[int, int] = {label: 0 for label in classes}

    def next_anchor(selected: set[int]) -> int:
        nonlocal anchor_cursor, anchor_order
        while True:
            if anchor_cursor == len(anchor_order):
                anchor_order = [classes[index] for index in anchor_rng.permutation(len(classes))]
                anchor_cursor = 0
            candidate = anchor_order[anchor_cursor]
            anchor_cursor += 1
            if candidate not in selected:
                return candidate

    def next_image(label: int) -> int:
        if image_cursors[label] == len(image_orders[label]):
            source = class_rows[label]
            image_orders[label] = [
                source[index] for index in image_rng[label].permutation(len(source))
            ]
            image_cursors[label] = 0
        row = image_orders[label][image_cursors[label]]
        image_cursors[label] += 1
        return row

    batches = []
    for _ in range(steps):
        selected: set[int] = set()
        anchors: list[int] = []
        while len(anchors) < 8:
            anchor = next_anchor(selected)
            selected.add(anchor)
            anchors.append(anchor)
        batch_classes = list(anchors)
        for anchor in anchors:
            added = 0
            for candidate in neighbor_order[anchor]:
                if candidate in selected:
                    continue
                selected.add(candidate)
                batch_classes.append(candidate)
                added += 1
                if added == 3:
                    break
            if added != 3:
                raise ValueError("identity schedule authority differs")
        batch = tuple(next_image(label) for label in batch_classes for _ in range(4))
        batches.append(batch)
    return tuple(batches)


def cluster_bootstrap_lower_bound(
    differences: NDArray[np.float64],
    class_ids: NDArray[np.int64],
    *,
    draws: int = 10_000,
    seed: int = 17,
    quantile: float = 0.05,
) -> float:
    """Bootstrap a query-weighted paired mean by resampling class clusters."""

    if (
        type(differences) is not np.ndarray
        or differences.dtype != np.float64
        or differences.ndim != 1
        or differences.size == 0
        or not np.isfinite(differences).all()
        or type(class_ids) is not np.ndarray
        or class_ids.dtype != np.int64
        or class_ids.shape != differences.shape
        or type(draws) is not int
        or draws <= 0
        or type(seed) is not int
        or not 0 <= seed < 2**63
        or type(quantile) is not float
        or not 0.0 < quantile < 1.0
    ):
        raise ValueError("cluster bootstrap authority differs")
    classes, inverse = np.unique(class_ids, return_inverse=True)
    sums = np.zeros(classes.size, dtype=np.float64)
    counts = np.zeros(classes.size, dtype=np.int64)
    np.add.at(sums, inverse, differences)
    np.add.at(counts, inverse, 1)
    rng = np.random.Generator(np.random.PCG64(seed))
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, classes.size, size=classes.size)
        estimates[draw] = sums[selected].sum() / counts[selected].sum()
    estimates.sort()
    return float(estimates[math.ceil(quantile * draws) - 1])
