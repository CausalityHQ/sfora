"""Training-only diagnostics for orbit-adaptive potential fields.

This module does not implement an OAPF training loss.  It implements the
prospectively fixed falsification tests in ``docs/oapf_candidate.md`` so the
candidate can die before a retrieval run if augmentation-orbit radius is only
ordinary uncertainty or density under a different name.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]
from sklearn.metrics import roc_auc_score

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def deterministic_view_seed(pack: str, example_id: str, view_index: int) -> int:
    """Return the pinned 63-bit seed for one OAPF augmentation draw."""

    if pack not in {"A", "B"}:
        raise ValueError("pack must be 'A' or 'B'")
    if view_index < 0 or view_index >= 6:
        raise ValueError("view_index must be in [0, 6)")
    payload = f"oapf-v1|{pack}|{example_id}|{view_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def orbit_radius_and_rms(
    canonical: FloatArray,
    views: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute q90 Euclidean orbit radius and ordinary RMS dispersion."""

    if canonical.ndim != 2 or views.ndim != 3:
        raise ValueError("canonical must be [N,D] and views [N,V,D]")
    if views.shape[0] != canonical.shape[0] or views.shape[2] != canonical.shape[1]:
        raise ValueError("canonical and views shapes are incompatible")
    if views.shape[1] != 6:
        raise ValueError("views must contain exactly six registered draws")
    if not np.allclose(np.linalg.norm(canonical, axis=1), 1.0, atol=1.0e-5):
        raise ValueError("canonical embeddings must be L2-normalized")
    if not np.allclose(np.linalg.norm(views, axis=2), 1.0, atol=1.0e-5):
        raise ValueError("view embeddings must be L2-normalized")
    displacement = np.linalg.norm(views - canonical[:, None, :], axis=2)
    radius = np.quantile(displacement, 0.9, axis=1, method="linear")
    centered = views - views.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(centered * centered, axis=2), axis=1))
    return np.maximum(radius, 1.0e-6), np.maximum(rms, 1.0e-6)


def same_class_pairs(labels: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Return unordered endpoint indices and the label of every same-class pair."""

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        grouped[int(label)].append(index)
    left: list[int] = []
    right: list[int] = []
    pair_labels: list[int] = []
    for label in sorted(grouped):
        indices = grouped[label]
        for position, first in enumerate(indices):
            for second in indices[position + 1 :]:
                left.append(first)
                right.append(second)
                pair_labels.append(label)
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(pair_labels, dtype=np.int64),
    )


def heldout_pair_compatibility(
    canonical: FloatArray,
    heldout_views: FloatArray,
    nearest_negative_distances: FloatArray,
    left: IntArray,
    right: IntArray,
) -> tuple[FloatArray, IntArray]:
    """Return the registered continuous and 10-of-12 binary stability outcomes."""

    if heldout_views.ndim != 3 or heldout_views.shape[1] != 6:
        raise ValueError("heldout_views must contain exactly six views")
    if nearest_negative_distances.shape != heldout_views.shape[:2]:
        raise ValueError("nearest-negative distances must have shape [N,6]")
    if left.shape != right.shape:
        raise ValueError("pair endpoint arrays must match")

    count = np.empty(left.size, dtype=np.int64)
    # In-Shop has enough within-class pairs that materializing all indexed
    # [pair, view, embedding] tensors at once can consume several GiB.
    for start in range(0, left.size, 4096):
        stop = min(start + 4096, left.size)
        chunk_left = left[start:stop]
        chunk_right = right[start:stop]
        forward = np.linalg.norm(
            heldout_views[chunk_left] - canonical[chunk_right, None, :], axis=2
        )
        reverse = np.linalg.norm(
            heldout_views[chunk_right] - canonical[chunk_left, None, :], axis=2
        )
        forward_success = nearest_negative_distances[chunk_left] - forward > 0.0
        reverse_success = nearest_negative_distances[chunk_right] - reverse > 0.0
        count[start:stop] = forward_success.sum(axis=1) + reverse_success.sum(axis=1)
    return count.astype(np.float64) / 12.0, (count >= 10).astype(np.int64)


def class_balanced_pair_weights(pair_labels: IntArray) -> FloatArray:
    """Give every class equal total weight and every pair within it equal weight."""

    labels, counts = np.unique(pair_labels, return_counts=True)
    count_by_label = dict(zip(labels.tolist(), counts.tolist(), strict=True))
    weights = np.asarray(
        [1.0 / count_by_label[int(label)] for label in pair_labels], dtype=np.float64
    )
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def fixed_class_folds(labels: IntArray, folds: int = 5) -> dict[int, int]:
    """Apply PCG64(174) to sorted labels and assign round-robin folds."""

    unique = np.unique(labels)
    rng = np.random.Generator(np.random.PCG64(174))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    return {int(label): index % folds for index, label in enumerate(shuffled)}


def within_class_derangement(
    values: FloatArray,
    labels: IntArray,
    seed: int,
) -> FloatArray:
    """Apply a seeded uniform random derangement inside every nonsingleton class."""

    result = values.copy()
    rng = np.random.Generator(np.random.PCG64(seed))
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if indices.size < 2:
            continue
        permutation = rng.permutation(indices.size)
        while np.any(permutation == np.arange(indices.size)):
            permutation = rng.permutation(indices.size)
        result[indices] = values[indices[permutation]]
    return result


@dataclass(frozen=True)
class Standardizer:
    mean: FloatArray
    scale: FloatArray

    def apply(self, values: FloatArray) -> FloatArray:
        return (values - self.mean) / self.scale


def fit_standardizer(values: FloatArray, weights: FloatArray) -> Standardizer:
    normalized = weights / weights.sum()
    mean = np.sum(values * normalized[:, None], axis=0)
    variance = np.sum((values - mean) ** 2 * normalized[:, None], axis=0)
    return Standardizer(mean=mean, scale=np.maximum(np.sqrt(variance), 1.0e-8))


def fit_weighted_logistic(
    features: FloatArray,
    targets: IntArray,
    weights: FloatArray,
    *,
    nonnegative_from: int | None = None,
) -> FloatArray:
    """Fit fixed-C=1 weighted logistic regression with optional coefficient bounds."""

    design = np.column_stack([np.ones(features.shape[0]), features])
    normalized_weights = weights * (weights.shape[0] / weights.sum())

    def objective(beta: FloatArray) -> tuple[float, FloatArray]:
        logits = design @ beta
        loss = np.logaddexp(0.0, logits) - targets * logits
        penalty = 0.5 * np.sum(beta[1:] ** 2)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        gradient = design.T @ (normalized_weights * (probability - targets))
        gradient[1:] += beta[1:]
        return float(np.sum(normalized_weights * loss) + penalty), gradient

    bounds = None
    if nonnegative_from is not None:
        first_beta = 1 + nonnegative_from
        bounds = [(None, None)] * first_beta + [(0.0, None)] * (design.shape[1] - first_beta)
    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1.0e-10},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def logistic_probabilities(features: FloatArray, coefficients: FloatArray) -> FloatArray:
    design = np.column_stack([np.ones(features.shape[0]), features])
    logits = design @ coefficients
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def weighted_auc(targets: IntArray, scores: FloatArray, weights: FloatArray) -> float:
    if np.unique(targets).size != 2:
        raise ValueError("AUC requires both outcome classes")
    return float(roc_auc_score(targets, scores, sample_weight=weights))
