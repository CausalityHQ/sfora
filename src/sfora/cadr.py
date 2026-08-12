"""Cosine-Anchored Diagonal Boundary Reweighting primitives."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class LabelSplit:
    fit_labels: np.ndarray
    validation_labels: np.ndarray


@dataclass(frozen=True)
class PairSet:
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    features: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class CadrModel:
    weights: np.ndarray
    intercept: float
    objective: float
    iterations: int
    converged: bool


def split_labels(labels: np.ndarray) -> LabelSplit:
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int64 or labels.ndim != 1:
        raise ValueError("labels must be a rank-1 int64 array")
    values, counts = np.unique(labels, return_counts=True)
    eligible = values[counts >= 2]
    if eligible.size < 5:
        raise ValueError("at least five eligible labels are required")
    ordered = sorted(
        (int(v) for v in eligible),
        key=lambda v: (
            hashlib.sha256(
                b"CADR-split-v1:" + np.asarray(v, dtype="<i8").tobytes()
            ).digest(),
            v,
        ),
    )
    boundary = int(np.floor(0.8 * len(ordered)))
    if boundary == 0 or boundary == len(ordered):
        raise ValueError("label split is empty")
    return LabelSplit(
        np.asarray(ordered[:boundary], dtype=np.int64),
        np.asarray(ordered[boundary:], dtype=np.int64),
    )


def _validate_arrays(embeddings: np.ndarray, labels: np.ndarray, example_ids: np.ndarray) -> None:
    if embeddings.dtype != np.float32 or embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2 float32")
    if labels.dtype != np.int64 or labels.shape != (embeddings.shape[0],):
        raise ValueError("labels must be row-aligned int64")
    if example_ids.shape != (embeddings.shape[0],) or example_ids.dtype.kind != "U":
        raise ValueError("example IDs must be row-aligned Unicode")
    if (
        any(not value for value in example_ids.tolist())
        or np.unique(example_ids).size != example_ids.size
    ):
        raise ValueError("example IDs must be nonempty and unique")
    if not np.isfinite(embeddings).all() or not np.allclose(
        np.linalg.norm(embeddings.astype(np.float64), axis=1), 1.0, atol=2e-5, rtol=0
    ):
        raise ValueError("embeddings must be finite and unit normalized")


def _positive_key(i: int, j: int, ids: np.ndarray) -> tuple[bytes, str, str]:
    left, right = sorted((str(ids[i]), str(ids[j])))
    digest = hashlib.sha256(
        b"CADR-positive-v1:" + left.encode() + b"\0" + right.encode()
    ).digest()
    return digest, left, right


def build_pairs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    example_ids: np.ndarray,
    allowed_labels: np.ndarray,
    *,
    k: int = 10,
) -> PairSet:
    _validate_arrays(embeddings, labels, example_ids)
    if allowed_labels.dtype != np.int64 or allowed_labels.ndim != 1 or k < 1:
        raise ValueError("allowed labels and k differ")
    pool = np.flatnonzero(np.isin(labels, allowed_labels))
    positives: list[tuple[int, int]] = []
    for label in allowed_labels.tolist():
        rows = np.flatnonzero(labels == label)
        candidates = [
            (int(rows[a]), int(rows[b]))
            for a in range(rows.size)
            for b in range(a + 1, rows.size)
        ]
        candidates.sort(key=lambda pair: _positive_key(pair[0], pair[1], example_ids))
        positives.extend(candidates[:30])
    negatives: set[tuple[int, int]] = set()
    if pool.size:
        matrix = embeddings[pool] @ embeddings[pool].T
        for local_i, global_i in enumerate(pool.tolist()):
            candidates = [
                (float(matrix[local_i, local_j]), int(global_j))
                for local_j, global_j in enumerate(pool.tolist())
                if labels[global_j] != labels[global_i]
            ]
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _, global_j in candidates[:k]:
                negatives.add(tuple(sorted((global_i, global_j))))
    positive_array = np.asarray(positives, dtype=np.int64).reshape(-1, 2)
    negative_array = np.asarray(sorted(negatives), dtype=np.int64).reshape(-1, 2)
    all_pairs = np.vstack([positive_array, negative_array])
    features = (
        embeddings[all_pairs[:, 0]].astype(np.float64)
        * embeddings[all_pairs[:, 1]].astype(np.float64)
    )
    targets = np.concatenate(
        [
            np.ones(len(positive_array), dtype=np.float64),
            -np.ones(len(negative_array), dtype=np.float64),
        ]
    )
    return PairSet(positive_array, negative_array, features, targets)


def fit_cadr(features: np.ndarray, targets: np.ndarray, *, lambda_value: float) -> CadrModel:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or set(np.unique(y)) != {-1.0, 1.0}:
        raise ValueError("pair matrix or targets differ")
    if not np.isfinite(x).all() or lambda_value <= 0:
        raise ValueError("fit inputs differ")
    pos = y == 1
    sample_weight = np.where(pos, 0.5 / pos.sum(), 0.5 / (~pos).sum())
    dimension = x.shape[1]

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        weights, intercept = value[:-1], value[-1]
        signed = y * (x @ weights + intercept)
        loss = np.logaddexp(0.0, -signed)
        probability = -y / (1.0 + np.exp(np.clip(signed, -700, 700)))
        delta = weights - 1.0
        total = float(sample_weight @ loss + lambda_value * np.mean(delta * delta))
        gradient = np.empty_like(value)
        gradient[:-1] = x.T @ (sample_weight * probability) + 2 * lambda_value * delta / dimension
        gradient[-1] = sample_weight @ probability
        return total, gradient

    initial = np.concatenate([np.ones(dimension, dtype=np.float64), [0.0]])
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise ValueError(f"CADR fit did not converge: {result.message}")
    return CadrModel(
        result.x[:-1].copy(), float(result.x[-1]), float(result.fun), int(result.nit), True
    )
