#!/usr/bin/env python3
"""Evaluate progressive residual retrieval against authenticated ANN truth."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

Metric = Literal["angular", "squared_l2"]
_HDF5_DATASETS = {"train", "test", "neighbors", "distances"}
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class AnnBenchmarkDataset:
    """Authenticated in-memory view of one official ANN-Benchmarks dataset."""

    source_sha256: str
    metric: Metric
    train: np.ndarray
    test: np.ndarray
    truth_ordinals: np.ndarray
    truth_distances: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_distances(train: np.ndarray, query: np.ndarray, metric: Metric) -> np.ndarray:
    distances = np.empty(train.shape[0], dtype=np.float64)
    query64 = query.astype(np.float64)
    query_norm = float(np.linalg.norm(query64))
    for start in range(0, train.shape[0], 65_536):
        stop = min(start + 65_536, train.shape[0])
        block = train[start:stop].astype(np.float64)
        if metric == "angular":
            norms = np.linalg.norm(block, axis=1)
            distances[start:stop] = 1.0 - (block @ query64) / (norms * query_norm)
        else:
            delta = block - query64
            distances[start:stop] = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return distances


def _validate_truth_boundaries(
    train: np.ndarray,
    test: np.ndarray,
    truth_ordinals: np.ndarray,
    truth_distances: np.ndarray,
    metric: Metric,
) -> None:
    query_count = test.shape[0]
    checks = sorted({0, query_count // 2, query_count - 1})
    truth_width = truth_ordinals.shape[1]
    for query_index in checks:
        distances = _metric_distances(train, test[query_index], metric)
        selected = truth_ordinals[query_index]
        selected_distances = distances[selected]
        if not np.allclose(
            selected_distances,
            truth_distances[query_index],
            rtol=1e-4,
            atol=1e-5,
        ):
            raise ValueError("ANN-Benchmarks authority differs")
        boundary = float(np.partition(distances, truth_width - 1)[truth_width - 1])
        tolerance = max(1e-7, abs(boundary) * 1e-6)
        selected_set = set(int(value) for value in selected)
        required = np.flatnonzero(distances < boundary - tolerance)
        if (
            any(int(value) not in selected_set for value in required)
            or bool((selected_distances > boundary + tolerance).any())
        ):
            raise ValueError("ANN-Benchmarks authority differs")


def load_ann_benchmark(
    path: Path,
    *,
    expected_sha256: str,
    expected_metric: Metric,
) -> AnnBenchmarkDataset:
    """Authenticate and load one native-metric ANN-Benchmarks HDF5 artifact."""

    if (
        not isinstance(path, Path)
        or path.is_symlink()
        or not path.is_file()
        or type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(character not in _HEX for character in expected_sha256)
        or expected_metric not in ("angular", "squared_l2")
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError("ANN-Benchmarks authority differs")

    try:
        with h5py.File(path, "r") as handle:
            stored_metric = handle.attrs.get("distance")
            mapped_metric = {
                "angular": "angular",
                "euclidean": "squared_l2",
            }.get(stored_metric)
            if set(handle.keys()) != _HDF5_DATASETS or mapped_metric != expected_metric:
                raise ValueError("ANN-Benchmarks authority differs")
            train = np.asarray(handle["train"][...])
            test = np.asarray(handle["test"][...])
            neighbors = np.asarray(handle["neighbors"][...])
            distances = np.asarray(handle["distances"][...])
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("ANN-Benchmarks authority differs") from error

    if (
        train.dtype != np.float32
        or train.ndim != 2
        or train.shape[0] < 1
        or train.shape[1] < 1
        or test.dtype != np.float32
        or test.ndim != 2
        or test.shape[0] < 1
        or test.shape[1] != train.shape[1]
        or neighbors.dtype != np.int32
        or neighbors.ndim != 2
        or neighbors.shape[0] != test.shape[0]
        or not 1 <= neighbors.shape[1] <= train.shape[0]
        or distances.dtype != np.float32
        or distances.shape != neighbors.shape
        or not bool(np.isfinite(train).all())
        or not bool(np.isfinite(test).all())
        or not bool(np.isfinite(distances).all())
        or bool((distances < 0).any())
        or bool((neighbors < 0).any())
        or bool((neighbors >= train.shape[0]).any())
        or bool((np.diff(distances, axis=1) < 0).any())
        or any(len(np.unique(row)) != len(row) for row in neighbors)
        or (
            expected_metric == "angular"
            and (
                bool((np.linalg.norm(train, axis=1) == 0).any())
                or bool((np.linalg.norm(test, axis=1) == 0).any())
            )
        )
    ):
        raise ValueError("ANN-Benchmarks authority differs")

    truth_ordinals = np.ascontiguousarray(neighbors, dtype=np.int64)
    train = np.ascontiguousarray(train)
    test = np.ascontiguousarray(test)
    truth_distances = np.ascontiguousarray(distances)
    _validate_truth_boundaries(
        train,
        test,
        truth_ordinals,
        truth_distances,
        expected_metric,
    )
    for value in (train, test, truth_ordinals, truth_distances):
        value.flags.writeable = False
    return AnnBenchmarkDataset(
        source_sha256=expected_sha256,
        metric=expected_metric,
        train=train,
        test=test,
        truth_ordinals=truth_ordinals,
        truth_distances=truth_distances,
    )


def ann_benchmark_recall_hits(
    dataset: AnnBenchmarkDataset,
    returned_ordinals: np.ndarray,
    *,
    query_ordinals: np.ndarray,
    count: int,
) -> np.ndarray:
    """Return official additive-epsilon distance-threshold recall hits."""

    if (
        type(dataset) is not AnnBenchmarkDataset
        or type(returned_ordinals) is not np.ndarray
        or returned_ordinals.dtype != np.int64
        or returned_ordinals.ndim != 2
        or returned_ordinals.shape[0] < 1
        or returned_ordinals.shape[1] < 1
        or not returned_ordinals.flags.c_contiguous
        or type(query_ordinals) is not np.ndarray
        or query_ordinals.dtype != np.int64
        or query_ordinals.ndim != 1
        or query_ordinals.shape != (returned_ordinals.shape[0],)
        or not query_ordinals.flags.c_contiguous
        or bool((query_ordinals < 0).any())
        or bool((query_ordinals >= dataset.test.shape[0]).any())
        or len(np.unique(query_ordinals)) != len(query_ordinals)
        or type(count) is not int
        or not 1 <= count <= returned_ordinals.shape[1]
        or count > dataset.truth_distances.shape[1]
        or bool((returned_ordinals < 0).any())
        or bool((returned_ordinals >= dataset.train.shape[0]).any())
    ):
        raise ValueError("ANN-Benchmarks result differs")
    ordered = np.sort(returned_ordinals, axis=1)
    if returned_ordinals.shape[1] > 1 and bool(
        (ordered[:, 1:] == ordered[:, :-1]).any()
    ):
        raise ValueError("ANN-Benchmarks result differs")

    query_count = returned_ordinals.shape[0]
    hits = np.empty(query_count, dtype=np.int64)
    thresholds = (
        dataset.truth_distances[query_ordinals, count - 1].astype(np.float64) + 1e-3
    )
    for start in range(0, query_count, 256):
        stop = min(start + 256, query_count)
        selected = dataset.train[returned_ordinals[start:stop, :count]].astype(np.float64)
        queries = dataset.test[query_ordinals[start:stop]].astype(np.float64)
        if dataset.metric == "angular":
            query_norms = np.linalg.norm(queries, axis=1)
            selected_norms = np.linalg.norm(selected, axis=2)
            distances = 1.0 - np.einsum("qkd,qd->qk", selected, queries) / (
                selected_norms * query_norms[:, None]
            )
        else:
            delta = selected - queries[:, None, :]
            distances = np.sqrt(np.einsum("qkd,qkd->qk", delta, delta))
        hits[start:stop] = (distances <= thresholds[start:stop, None]).sum(axis=1)
    return hits


def ann_benchmark_containment_hits(
    dataset: AnnBenchmarkDataset,
    candidate_ordinals: np.ndarray,
    *,
    query_ordinals: np.ndarray,
    candidate_width: int,
    truth_width: int,
) -> np.ndarray:
    """Count native-distance truth hits anywhere in each candidate prefix."""

    if (
        type(dataset) is not AnnBenchmarkDataset
        or type(candidate_ordinals) is not np.ndarray
        or candidate_ordinals.dtype != np.int64
        or candidate_ordinals.ndim != 2
        or candidate_ordinals.shape[0] < 1
        or candidate_ordinals.shape[1] < 1
        or not candidate_ordinals.flags.c_contiguous
        or type(query_ordinals) is not np.ndarray
        or query_ordinals.dtype != np.int64
        or query_ordinals.ndim != 1
        or query_ordinals.shape != (candidate_ordinals.shape[0],)
        or not query_ordinals.flags.c_contiguous
        or len(np.unique(query_ordinals)) != len(query_ordinals)
        or bool((query_ordinals < 0).any())
        or bool((query_ordinals >= dataset.test.shape[0]).any())
        or type(candidate_width) is not int
        or not 1 <= candidate_width <= candidate_ordinals.shape[1]
        or type(truth_width) is not int
        or not 1 <= truth_width <= dataset.truth_distances.shape[1]
        or bool((candidate_ordinals < 0).any())
        or bool((candidate_ordinals >= dataset.train.shape[0]).any())
    ):
        raise ValueError("ANN-Benchmarks containment differs")
    ordered = np.sort(candidate_ordinals, axis=1)
    if candidate_ordinals.shape[1] > 1 and bool(
        (ordered[:, 1:] == ordered[:, :-1]).any()
    ):
        raise ValueError("ANN-Benchmarks containment differs")

    hits = np.empty(candidate_ordinals.shape[0], dtype=np.int64)
    thresholds = (
        dataset.truth_distances[query_ordinals, truth_width - 1].astype(np.float64) + 1e-3
    )
    for start in range(0, candidate_ordinals.shape[0], 64):
        stop = min(start + 64, candidate_ordinals.shape[0])
        selected = dataset.train[
            candidate_ordinals[start:stop, :candidate_width]
        ].astype(np.float64)
        queries = dataset.test[query_ordinals[start:stop]].astype(np.float64)
        if dataset.metric == "angular":
            query_norms = np.linalg.norm(queries, axis=1)
            selected_norms = np.linalg.norm(selected, axis=2)
            distances = 1.0 - np.einsum("qkd,qd->qk", selected, queries) / (
                selected_norms * query_norms[:, None]
            )
        else:
            delta = selected - queries[:, None, :]
            distances = np.sqrt(np.einsum("qkd,qkd->qk", delta, delta))
        raw_hits = (distances <= thresholds[start:stop, None]).sum(axis=1)
        hits[start:stop] = np.minimum(raw_hits, truth_width)
    return hits


def candidate_containment_hits(
    candidate_ordinals: np.ndarray,
    truth_ordinals: np.ndarray,
    *,
    candidate_width: int,
    truth_width: int,
) -> np.ndarray:
    """Count truth IDs present anywhere in each bounded candidate prefix."""

    if (
        type(candidate_ordinals) is not np.ndarray
        or candidate_ordinals.dtype != np.int64
        or candidate_ordinals.ndim != 2
        or candidate_ordinals.shape[0] < 1
        or candidate_ordinals.shape[1] < 1
        or not candidate_ordinals.flags.c_contiguous
        or type(truth_ordinals) is not np.ndarray
        or truth_ordinals.dtype != np.int64
        or truth_ordinals.ndim != 2
        or truth_ordinals.shape[0] != candidate_ordinals.shape[0]
        or truth_ordinals.shape[1] < 1
        or not truth_ordinals.flags.c_contiguous
        or type(candidate_width) is not int
        or not 1 <= candidate_width <= candidate_ordinals.shape[1]
        or type(truth_width) is not int
        or not 1 <= truth_width <= truth_ordinals.shape[1]
        or bool((candidate_ordinals < 0).any())
        or bool((truth_ordinals < 0).any())
        or (
            candidate_ordinals.shape[1] > 1
            and bool(
                (
                    np.diff(np.sort(candidate_ordinals, axis=1), axis=1)
                    == 0
                ).any()
            )
        )
        or (
            truth_ordinals.shape[1] > 1
            and bool(
                (np.diff(np.sort(truth_ordinals, axis=1), axis=1) == 0).any()
            )
        )
    ):
        raise ValueError("candidate containment authority differs")
    candidates = np.sort(candidate_ordinals[:, :candidate_width], axis=1)
    truth = truth_ordinals[:, :truth_width]
    lower = np.zeros(truth.shape, dtype=np.int64)
    upper = np.full(truth.shape, candidate_width, dtype=np.int64)
    while bool((lower < upper).any()):
        active = lower < upper
        middle = (lower + upper) // 2
        probes = np.take_along_axis(
            candidates,
            np.minimum(middle, candidate_width - 1),
            axis=1,
        )
        below = probes < truth
        lower = np.where(active & below, middle + 1, lower)
        upper = np.where(active & ~below, middle, upper)
    probes = np.take_along_axis(
        candidates,
        np.minimum(lower, candidate_width - 1),
        axis=1,
    )
    hits = ((lower < candidate_width) & (probes == truth)).sum(axis=1, dtype=np.int64)
    return np.asarray(hits, dtype=np.int64)
