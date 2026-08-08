#!/usr/bin/env python3
"""Frozen Stage-A necessary-condition diagnostic for Pass181 CIEB.

The command deliberately runs the entropy-stability/CV screen before it
constructs any matched masks.  It reuses Pass159's fail-closed artifact binding;
query and gallery data are therefore used only to bind the checkpoint's official
R@1 and never enter a candidate statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from diagnose_pass159_cotangent_stage_a import (  # noqa: E402
    BoundSeed,
    load_bound_seed,
    sha256_file,
)

DOMAIN = "pass181-cieb-stage-a-v1"
EPSILON = 1.0e-6
TAU = 0.05
TARGET_COUNT = 52
MASK_COUNT = 1_000
FOREIGN_K = 32
BOOTSTRAP_SEED = 181
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_DIMENSION = 512
_ZERO_EPS = 1.0e-12


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _aligned_labels(labels: np.ndarray, row_count: int, *, name: str) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64)
    if array.shape != (row_count,):
        raise ValueError(f"{name} must align with descriptor rows")
    return array


def _class_statistics(
    descriptors: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = _finite_matrix(descriptors, name="descriptors")
    y = _aligned_labels(labels, len(z), name="labels")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("ownership entropy requires at least two estimator classes")
    means = np.empty((len(classes), z.shape[1]), dtype=np.float64)
    within_variances = np.empty_like(means)
    for position, label in enumerate(classes.tolist()):
        rows = z[y == int(label)]
        means[position] = rows.mean(axis=0)
        centered = rows - means[position]
        within_variances[position] = np.mean(centered * centered, axis=0)
    class_balanced_mean = means.mean(axis=0)
    return classes, means, within_variances, class_balanced_mean


def ownership_entropy(descriptors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the frozen class-balanced ownership entropy for every coordinate."""
    classes, means, within_variances, global_mean = _class_statistics(descriptors, labels)
    ownership = (means - global_mean) ** 2 / (within_variances + EPSILON)
    totals = ownership.sum(axis=0)
    q = np.empty_like(ownership)
    nonconstant = totals > 0.0
    q[:, nonconstant] = ownership[:, nonconstant] / totals[nonconstant]
    q[:, ~nonconstant] = 1.0 / len(classes)
    terms = np.zeros_like(q)
    positive = q > 0.0
    terms[positive] = q[positive] * np.log(q[positive])
    entropy = -terms.sum(axis=0) / np.log(float(len(classes)))
    entropy[~nonconstant] = 1.0
    if not np.isfinite(entropy).all():
        raise ValueError("ownership entropy is nonfinite")
    return entropy


def entropy_weights(entropy: np.ndarray) -> tuple[np.ndarray, float]:
    h = np.asarray(entropy, dtype=np.float64)
    if h.ndim != 1 or len(h) == 0 or not np.isfinite(h).all():
        raise ValueError("entropy must be a nonempty finite vector")
    shifted = h + EPSILON
    weights = shifted / shifted.mean()
    cv = float(np.std(weights, ddof=0) / np.mean(weights))
    return weights, cv


def class_fold(label: int) -> int:
    canonical = str(int(label))
    digest = hashlib.sha256(f"{DOMAIN}|fold|{canonical}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % 5


def _split_digest(example_id: str) -> bytes:
    return hashlib.sha256(f"{DOMAIN}|split|{example_id}".encode()).digest()


def split_identity(example_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(example_ids).astype(str)
    if ids.ndim != 1 or len(ids) < 2:
        raise ValueError("held-out identity requires at least two images")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("held-out identity has duplicate example IDs")
    order = sorted(
        range(len(ids)),
        key=lambda index: (_split_digest(ids[index]), ids[index]),
    )
    gallery = np.asarray(order[0::2], dtype=np.int64)
    query = np.asarray(order[1::2], dtype=np.int64)
    if len(gallery) == 0 or len(query) == 0:
        raise ValueError("held-out identity split produced an empty side")
    return gallery, query


def _average_ranks(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("rank input must be a finite vector")
    order = np.argsort(vector, kind="stable")
    ranks = np.empty(len(vector), dtype=np.float64)
    start = 0
    while start < len(vector):
        stop = start + 1
        while stop < len(vector) and vector[order[stop]] == vector[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = _average_ranks(left)
    y = _average_ranks(right)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= _ZERO_EPS:
        return 0.0
    return float(np.dot(x, y) / denominator)


def _fold_partitions(estimator_folds: list[int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(estimator_folds) != 4:
        raise ValueError("a held-out fold must leave exactly four estimator folds")
    ordered = sorted(estimator_folds)
    return [
        ((ordered[0], ordered[1]), (ordered[2], ordered[3])),
        ((ordered[0], ordered[2]), (ordered[1], ordered[3])),
        ((ordered[0], ordered[3]), (ordered[1], ordered[2])),
    ]


def compute_entropy_screen(
    bound: BoundSeed, *, expected_dimension: int = EXPECTED_DIMENSION
) -> dict[str, Any]:
    z = _finite_matrix(bound.train_embeddings, name="training descriptors")
    y = _aligned_labels(bound.train_labels, len(z), name="training labels")
    if z.shape[1] != expected_dimension:
        raise ValueError(f"CIEB requires a {expected_dimension}-D head")
    classes = np.unique(y)
    folds_by_class = {int(label): class_fold(int(label)) for label in classes.tolist()}
    fold_rows = np.asarray([folds_by_class[int(label)] for label in y], dtype=np.int64)
    present_folds = set(fold_rows.tolist())
    if present_folds != set(range(5)):
        raise ValueError(f"CIEB requires all five class folds, observed {sorted(present_folds)}")

    full_entropy = ownership_entropy(z, y)
    weights, cv = entropy_weights(full_entropy)
    fold_results: list[dict[str, Any]] = []
    correlations: list[float] = []
    for held_fold in range(5):
        estimator_folds = [fold for fold in range(5) if fold != held_fold]
        fold_correlations: list[float] = []
        for left_folds, right_folds in _fold_partitions(estimator_folds):
            left = np.isin(fold_rows, left_folds)
            right = np.isin(fold_rows, right_folds)
            correlation = spearman(
                ownership_entropy(z[left], y[left]),
                ownership_entropy(z[right], y[right]),
            )
            fold_correlations.append(correlation)
            correlations.append(correlation)
        fold_results.append(
            {
                "held_fold": held_fold,
                "estimator_class_count": int(
                    np.sum([folds_by_class[int(c)] != held_fold for c in classes])
                ),
                "held_class_count": int(
                    np.sum([folds_by_class[int(c)] == held_fold for c in classes])
                ),
                "split_half_correlations": fold_correlations,
            }
        )
    stability = float(np.median(np.asarray(correlations, dtype=np.float64)))
    return {
        "seed": int(bound.seed),
        "stability": stability,
        "cv": cv,
        "full_entropy_min": float(full_entropy.min()),
        "full_entropy_median": float(np.median(full_entropy)),
        "full_entropy_max": float(full_entropy.max()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "folds": fold_results,
    }


def entropy_stage_verdict(*, stabilities: list[float], cvs: list[float]) -> dict[str, Any]:
    if len(stabilities) != 4 or len(cvs) != 4:
        raise ValueError("entropy-stage verdict requires four seeds")
    values = np.asarray([stabilities, cvs], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("entropy-stage verdict received nonfinite values")
    low_stability = int(np.sum(values[0] < 0.30))
    low_cv = int(np.sum(values[1] < 0.05))
    reasons: list[str] = []
    if low_stability >= 3:
        reasons.append("stability is below 0.30 in at least three seeds")
    if low_cv >= 3:
        reasons.append("CV is below 0.05 in at least three seeds")
    return {
        "stage_a": "FAIL" if reasons else "CONTINUE_TO_MATCHED_MASKS",
        "stop_before_matched_masks": bool(reasons),
        "reasons": reasons,
        "stabilities": [float(value) for value in stabilities],
        "cvs": [float(value) for value in cvs],
        "seeds_stability_below_0_30": low_stability,
        "seeds_cv_below_0_05": low_cv,
    }


def nuisance_strata(
    descriptors: np.ndarray,
    labels: np.ndarray,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    z = _finite_matrix(descriptors, name="estimator descriptors")
    y = _aligned_labels(labels, len(z), name="estimator labels")
    classes, means, _, global_mean = _class_statistics(z, y)
    proxy_matrix = _finite_matrix(proxies, name="proxies")
    p_labels = _aligned_labels(proxy_labels, len(proxy_matrix), name="proxy labels")
    if proxy_matrix.shape[1] != z.shape[1]:
        raise ValueError("proxies and descriptors must share a dimension")
    proxy_index = {int(label): index for index, label in enumerate(p_labels.tolist())}
    if len(proxy_index) != len(p_labels):
        raise ValueError("nuisance matching requires one proxy per class")
    try:
        own_proxies = np.asarray([proxy_matrix[proxy_index[int(label)]] for label in classes])
    except KeyError as error:
        raise ValueError(f"estimator class lacks own proxy: {error.args[0]}") from error
    proxy_norms = np.linalg.norm(own_proxies, axis=1)
    if np.any(proxy_norms <= _ZERO_EPS):
        raise ValueError("own-class proxy has zero norm")
    own_proxies = own_proxies / proxy_norms[:, None]

    class_second_moments = np.empty_like(means)
    for position, label in enumerate(classes.tolist()):
        centered = z[y == int(label)] - global_mean
        class_second_moments[position] = np.mean(centered * centered, axis=0)
    v = class_second_moments.mean(axis=0)
    r = np.mean(means * own_proxies, axis=0)
    log_v = np.log(v + EPSILON)
    coordinate = np.arange(z.shape[1], dtype=np.int64)

    def ordinal_bins(values: np.ndarray) -> np.ndarray:
        order = np.lexsort((coordinate, values))
        ranks = np.empty(len(values), dtype=np.int64)
        ranks[order] = np.arange(len(values), dtype=np.int64)
        return np.minimum(7, (8 * ranks) // len(values))

    v_bin = ordinal_bins(log_v)
    r_bin = ordinal_bins(r)
    return {
        "v": v,
        "r": r,
        "log_v": log_v,
        "v_bin": v_bin,
        "r_bin": r_bin,
        "strata": 8 * v_bin + r_bin,
    }


def _mask_hash(seed: int, fold: int, replicate: int, cell: int, coordinate: int) -> bytes:
    text = f"{DOMAIN}|mask|{seed}|{fold}|{replicate}|{cell}|{coordinate}"
    return hashlib.sha256(text.encode("utf-8")).digest()


def build_matched_masks(
    strata: np.ndarray,
    target_coordinates: np.ndarray,
    *,
    seed: int,
    fold: int,
    mask_count: int = MASK_COUNT,
) -> np.ndarray:
    cells = np.asarray(strata, dtype=np.int64)
    target = np.asarray(target_coordinates, dtype=np.int64)
    if cells.ndim != 1 or len(cells) == 0 or np.any((cells < 0) | (cells >= 64)):
        raise ValueError("strata must be a nonempty vector of 8-by-8 cell indices")
    if target.ndim != 1 or len(target) == 0 or len(np.unique(target)) != len(target):
        raise ValueError("target coordinates must be a nonempty unique vector")
    if np.any((target < 0) | (target >= len(cells))) or len(cells) > np.iinfo(np.uint16).max:
        raise ValueError("target coordinates fall outside the uint16 coordinate domain")
    if mask_count <= 0:
        raise ValueError("mask_count must be positive")
    target = np.sort(target)
    needed = np.bincount(cells[target], minlength=64)
    possible = 1
    for cell, count in enumerate(needed.tolist()):
        if count:
            available = int(np.sum(cells == cell))
            if available < count:
                raise ValueError("a nuisance stratum cannot supply its target count")
            possible *= math.comb(available, count)
            if possible > mask_count:
                break
    if possible - 1 < mask_count:
        raise ValueError("fewer unique nuisance-matched controls exist than requested")

    target_key = tuple(int(value) for value in target.tolist())
    seen = {target_key}
    masks: list[tuple[int, ...]] = []
    replicate = 0
    while len(masks) < mask_count:
        selected: list[int] = []
        for cell, count in enumerate(needed.tolist()):
            if count == 0:
                continue
            candidates = np.flatnonzero(cells == cell).tolist()
            candidates.sort(
                key=lambda coordinate: _mask_hash(seed, fold, replicate, cell, coordinate)
            )
            selected.extend(candidates[:count])
        key = tuple(sorted(selected))
        if key not in seen:
            seen.add(key)
            masks.append(key)
        replicate += 1
    return np.asarray(masks, dtype=np.uint16)


def canonical_mask_sha256(masks: np.ndarray) -> str:
    matrix = np.asarray(masks)
    if matrix.ndim != 2 or matrix.dtype != np.uint16:
        raise ValueError("canonical mask matrix must be uint16 and two-dimensional")
    canonical = np.ascontiguousarray(matrix.astype("<u2", copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _logmeanexp(values: np.ndarray, *, tau: float) -> float:
    scaled = np.asarray(values, dtype=np.float64) / float(tau)
    maximum = float(np.max(scaled))
    return float(tau * (maximum + np.log(np.mean(np.exp(scaled - maximum)))))


def _masked_rows(rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(rows, dtype=np.float64).copy()
    result[:, np.asarray(mask, dtype=np.int64)] = 0.0
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= _ZERO_EPS) or not np.isfinite(norms).all():
        raise ValueError("coordinate ablation produced a zero or nonfinite norm")
    return result / norms[:, None]


def masked_similarity_matrix(
    query: np.ndarray,
    supports: np.ndarray,
    masks: np.ndarray,
    *,
    chunk_size: int = 64,
) -> np.ndarray:
    """Return exact zero-mask-and-renormalize cosine scores, chunked over masks."""
    q = np.asarray(query, dtype=np.float64)
    s = _finite_matrix(supports, name="masked-similarity supports")
    mask_matrix = np.asarray(masks, dtype=np.int64)
    if q.ndim != 1 or not np.isfinite(q).all() or s.shape[1] != len(q):
        raise ValueError("masked-similarity query and supports must align and be finite")
    if (
        mask_matrix.ndim != 2
        or len(mask_matrix) == 0
        or np.any(mask_matrix < 0)
        or np.any(mask_matrix >= len(q))
        or chunk_size <= 0
    ):
        raise ValueError("masked-similarity masks or chunk size are invalid")
    if any(len(np.unique(mask)) != len(mask) for mask in mask_matrix):
        raise ValueError("masked-similarity masks must not repeat coordinates")
    query_total = float(np.dot(q, q))
    support_totals = np.sum(s * s, axis=1)
    if query_total <= _ZERO_EPS or np.any(support_totals <= _ZERO_EPS):
        raise ValueError("masked-similarity inputs contain a zero vector")
    base_dot = s @ q
    result = np.empty((len(mask_matrix), len(s)), dtype=np.float64)
    for start in range(0, len(mask_matrix), chunk_size):
        stop = min(start + chunk_size, len(mask_matrix))
        chunk = mask_matrix[start:stop]
        query_removed = q[chunk]
        support_removed = s[:, chunk]
        query_remaining = query_total - np.sum(query_removed * query_removed, axis=1)
        support_remaining = support_totals[:, None] - np.sum(
            support_removed * support_removed, axis=2
        )
        if np.any(query_remaining <= _ZERO_EPS) or np.any(support_remaining <= _ZERO_EPS):
            raise ValueError("coordinate ablation produced a zero or nonfinite norm")
        removed_dot = np.einsum("mk,smk->ms", query_removed, support_removed)
        denominator = np.sqrt(query_remaining[:, None] * support_remaining.T)
        chunk_scores = (base_dot[None, :] - removed_dot) / denominator
        if not np.isfinite(chunk_scores).all():
            raise ValueError("masked similarity is nonfinite")
        result[start:stop] = chunk_scores
    return result


def _logmeanexp_rows(values: np.ndarray, *, tau: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64) / float(tau)
    maxima = np.max(matrix, axis=1)
    return float(tau) * (
        maxima + np.log(np.mean(np.exp(matrix - maxima[:, None]), axis=1))
    )


def _recall_at_1(
    descriptors: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    gallery_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    mask: np.ndarray | None,
) -> float:
    z = np.asarray(descriptors, dtype=np.float64)
    gallery = z[gallery_indices] if mask is None else _masked_rows(z[gallery_indices], mask)
    queries = z[query_indices] if mask is None else _masked_rows(z[query_indices], mask)
    correct = 0
    gallery_ids = np.asarray(ids).astype(str)[gallery_indices]
    gallery_hashes = [_split_digest(value) for value in gallery_ids]
    for local_query, query_index in enumerate(query_indices.tolist()):
        scores = gallery @ queries[local_query]
        nearest = min(
            range(len(gallery_indices)),
            key=lambda index: (-float(scores[index]), gallery_hashes[index], gallery_ids[index]),
        )
        correct += int(int(labels[gallery_indices[nearest]]) == int(labels[query_index]))
    return correct / len(query_indices)


def score_mask_effects(
    descriptors: np.ndarray,
    labels: np.ndarray,
    example_ids: np.ndarray,
    gallery_indices: np.ndarray,
    query_indices: np.ndarray,
    masks: np.ndarray,
    *,
    foreign_k: int = FOREIGN_K,
    tau: float = TAU,
) -> dict[str, Any]:
    """Score target-plus-control masks using one unablated frozen foreign set."""
    z = _finite_matrix(descriptors, name="held descriptors")
    norms = np.linalg.norm(z, axis=1)
    if not np.allclose(norms, 1.0, atol=2.0e-5, rtol=2.0e-5):
        raise ValueError("held descriptors must contain unit rows")
    z = z / norms[:, None]
    y = _aligned_labels(labels, len(z), name="held labels")
    ids = np.asarray(example_ids).astype(str)
    if ids.shape != (len(z),) or len(set(ids.tolist())) != len(ids):
        raise ValueError("held example IDs must be aligned and unique")
    gallery = np.asarray(gallery_indices, dtype=np.int64)
    queries = np.asarray(query_indices, dtype=np.int64)
    mask_matrix = np.asarray(masks, dtype=np.uint16)
    if gallery.ndim != 1 or queries.ndim != 1 or not len(gallery) or not len(queries):
        raise ValueError("gallery and query index vectors must be nonempty")
    if set(gallery.tolist()) & set(queries.tolist()):
        raise ValueError("gallery and query rows must be disjoint")
    if mask_matrix.ndim != 2 or mask_matrix.shape[0] < 2:
        raise ValueError("masks must contain the target followed by controls")
    if foreign_k <= 0 or tau <= 0.0:
        raise ValueError("foreign_k and tau must be positive")

    query_effects: list[np.ndarray] = []
    query_labels: list[int] = []
    frozen_foreign_ids: list[list[str]] = []
    gallery_hashes = {int(index): _split_digest(ids[int(index)]) for index in gallery}
    for query_index in queries.tolist():
        label = int(y[query_index])
        positives = gallery[y[gallery] == label]
        foreign = gallery[y[gallery] != label]
        if len(positives) == 0 or len(foreign) < foreign_k:
            raise ValueError("held query lacks positive or foreign gallery support")
        scores = z[foreign] @ z[query_index]
        order = sorted(
            range(len(foreign)),
            key=lambda position: (
                -float(scores[position]),
                gallery_hashes[int(foreign[position])],
                ids[int(foreign[position])],
            ),
        )
        frozen_foreign = foreign[np.asarray(order[:foreign_k], dtype=np.int64)]
        frozen_foreign_ids.append(ids[frozen_foreign].tolist())
        unablated_positive = z[positives] @ z[query_index]
        unablated_foreign = z[frozen_foreign] @ z[query_index]
        unablated_margin = _logmeanexp(unablated_positive, tau=tau) - _logmeanexp(
            unablated_foreign, tau=tau
        )
        support_indices = np.concatenate((positives, frozen_foreign))
        masked_scores = masked_similarity_matrix(
            z[query_index], z[support_indices], mask_matrix, chunk_size=64
        )
        positive_count = len(positives)
        effects = (
            _logmeanexp_rows(masked_scores[:, :positive_count], tau=tau)
            - _logmeanexp_rows(masked_scores[:, positive_count:], tau=tau)
            - unablated_margin
        )
        query_effects.append(effects)
        query_labels.append(label)

    effect_matrix = np.asarray(query_effects, dtype=np.float64)
    query_label_array = np.asarray(query_labels, dtype=np.int64)
    eligible = np.unique(query_label_array)
    identity_effects = np.asarray(
        [effect_matrix[query_label_array == label].mean(axis=0) for label in eligible],
        dtype=np.float64,
    )
    return {
        "eligible_labels": eligible,
        "identity_effects": identity_effects,
        "frozen_foreign_ids": frozen_foreign_ids,
        "unablated_recall_at_1": _recall_at_1(z, y, ids, gallery, queries, mask=None),
        "target_recall_at_1": _recall_at_1(z, y, ids, gallery, queries, mask=mask_matrix[0]),
    }


def _target_coordinates(entropy: np.ndarray, *, count: int = TARGET_COUNT) -> np.ndarray:
    h = np.asarray(entropy, dtype=np.float64)
    if count <= 0 or count > len(h) or not np.isfinite(h).all():
        raise ValueError("invalid target coordinate count or entropy")
    coordinate = np.arange(len(h), dtype=np.int64)
    return np.lexsort((coordinate, h))[:count]


def _covariate_balance(
    nuisance: dict[str, np.ndarray],
    target: np.ndarray,
    controls: np.ndarray,
) -> dict[str, Any]:
    target_log_v = float(np.mean(nuisance["log_v"][target]))
    target_r = float(np.mean(nuisance["r"][target]))
    control_log_v = np.mean(nuisance["log_v"][controls], axis=1)
    control_r = np.mean(nuisance["r"][controls], axis=1)
    return {
        "target_mean_log_v": target_log_v,
        "target_mean_r": target_r,
        "control_mean_log_v_mean": float(control_log_v.mean()),
        "control_mean_r_mean": float(control_r.mean()),
        "max_abs_mean_log_v_difference": float(np.max(np.abs(control_log_v - target_log_v))),
        "max_abs_mean_r_difference": float(np.max(np.abs(control_r - target_r))),
    }


def compute_seed_full(
    bound: BoundSeed,
    *,
    target_count: int = TARGET_COUNT,
    mask_count: int = MASK_COUNT,
    foreign_k: int = FOREIGN_K,
) -> dict[str, Any]:
    z = _finite_matrix(bound.train_embeddings, name="training descriptors")
    y = _aligned_labels(bound.train_labels, len(z), name="training labels")
    ids = np.asarray(bound.train_example_ids).astype(str)
    folds = np.asarray([class_fold(int(label)) for label in y], dtype=np.int64)
    all_labels: list[np.ndarray] = []
    all_effects: list[np.ndarray] = []
    fold_results: list[dict[str, Any]] = []
    excluded_labels: list[int] = []
    for held_fold in range(5):
        estimator = folds != held_fold
        held = folds == held_fold
        entropy = ownership_entropy(z[estimator], y[estimator])
        target = _target_coordinates(entropy, count=target_count)
        nuisance = nuisance_strata(
            z[estimator],
            y[estimator],
            bound.proxies,
            bound.proxy_labels,
        )
        controls = build_matched_masks(
            nuisance["strata"],
            target,
            seed=int(bound.seed),
            fold=held_fold,
            mask_count=mask_count,
        )
        masks = np.concatenate((target.astype(np.uint16)[None, :], controls), axis=0)

        held_indices = np.flatnonzero(held)
        gallery_parts: list[np.ndarray] = []
        query_parts: list[np.ndarray] = []
        for label in sorted(np.unique(y[held]).tolist()):
            identity = held_indices[y[held_indices] == int(label)]
            if len(identity) < 2:
                excluded_labels.append(int(label))
                continue
            local_gallery, local_query = split_identity(ids[identity])
            gallery_parts.append(identity[local_gallery])
            query_parts.append(identity[local_query])
        if not gallery_parts or not query_parts:
            raise ValueError(f"held fold {held_fold} has no eligible identities")
        gallery_indices = np.concatenate(gallery_parts)
        query_indices = np.concatenate(query_parts)
        scored = score_mask_effects(
            z,
            y,
            ids,
            gallery_indices,
            query_indices,
            masks,
            foreign_k=foreign_k,
            tau=TAU,
        )
        all_labels.append(scored["eligible_labels"])
        all_effects.append(scored["identity_effects"])
        fold_results.append(
            {
                "held_fold": held_fold,
                "eligible_identity_count": int(len(scored["eligible_labels"])),
                "target_coordinates": target.tolist(),
                "control_mask_sha256": canonical_mask_sha256(controls),
                "control_mask_shape": list(controls.shape),
                "covariate_balance": _covariate_balance(nuisance, target, controls),
                "unablated_recall_at_1": float(scored["unablated_recall_at_1"]),
                "target_recall_at_1": float(scored["target_recall_at_1"]),
            }
        )
    labels = np.concatenate(all_labels)
    effects = np.concatenate(all_effects, axis=0)
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    effects = effects[order]
    if len(np.unique(labels)) != len(labels):
        raise ValueError("an eligible identity appeared in more than one held fold")
    control_identity_effects = effects[:, 1:]
    target_identity_effects = effects[:, 0]
    control_means = control_identity_effects.mean(axis=0)
    standardizer = float(np.std(control_means, ddof=1))
    if standardizer <= _ZERO_EPS or not np.isfinite(standardizer):
        raise ValueError("control-mask nuisance standardizer is at most 1e-12")
    target_mean = float(target_identity_effects.mean())
    mean_control = float(control_means.mean())
    return {
        "seed": int(bound.seed),
        "labels": labels,
        "target_identity_effects": target_identity_effects,
        "control_identity_effects": control_identity_effects,
        "T": target_mean,
        "mean_R": mean_control,
        "D": target_mean - mean_control,
        "S": standardizer,
        "Z": (target_mean - mean_control) / standardizer,
        "excluded_identity_labels": sorted(excluded_labels),
        "folds": fold_results,
    }


def full_stage_a_verdict(
    seed_effects: list[dict[str, object]],
    *,
    stabilities: list[float],
    cvs: list[float],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    if len(seed_effects) != 4 or {int(item["seed"]) for item in seed_effects} != set(range(4)):
        raise ValueError("full verdict requires exactly seeds zero through three")
    if len(stabilities) != 4 or len(cvs) != 4 or bootstrap_replicates <= 0:
        raise ValueError("full verdict requires four entropy metrics and positive bootstrap count")
    ordered = sorted(seed_effects, key=lambda item: int(item["seed"]))
    reference_labels = np.asarray(ordered[0]["labels"], dtype=np.int64)
    if len(reference_labels) == 0 or len(np.unique(reference_labels)) != len(reference_labels):
        raise ValueError("bootstrap requires unique eligible identity labels")

    advantages: list[np.ndarray] = []
    standardizers: list[float] = []
    d_values: list[float] = []
    z_values: list[float] = []
    seed_summaries: dict[str, dict[str, float]] = {}
    for item in ordered:
        labels = np.asarray(item["labels"], dtype=np.int64)
        if not np.array_equal(labels, reference_labels):
            raise ValueError("eligible identity labels differ across seeds")
        target = np.asarray(item["target_identity_effects"], dtype=np.float64)
        controls = np.asarray(item["control_identity_effects"], dtype=np.float64)
        if target.shape != (len(labels),) or controls.ndim != 2 or controls.shape[0] != len(labels):
            raise ValueError("identity effects do not align")
        if (
            controls.shape[1] < 2
            or not np.isfinite(target).all()
            or not np.isfinite(controls).all()
        ):
            raise ValueError("identity effects require finite target and multiple controls")
        control_means = controls.mean(axis=0)
        standardizer = float(np.std(control_means, ddof=1))
        if standardizer <= _ZERO_EPS or not np.isfinite(standardizer):
            raise ValueError("control-mask nuisance standardizer is at most 1e-12")
        advantage = target - controls.mean(axis=1)
        d_value = float(advantage.mean())
        z_value = d_value / standardizer
        advantages.append(advantage)
        standardizers.append(standardizer)
        d_values.append(d_value)
        z_values.append(z_value)
        seed_summaries[str(int(item["seed"]))] = {
            "T": float(target.mean()),
            "mean_R": float(control_means.mean()),
            "D": d_value,
            "S": standardizer,
            "Z": z_value,
        }

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
    identity_count = len(reference_labels)
    for replicate in range(bootstrap_replicates):
        sampled = rng.integers(0, identity_count, size=identity_count)
        bootstrap[replicate] = float(
            np.mean(
                [
                    advantage[sampled].mean() / standardizer
                    for advantage, standardizer in zip(advantages, standardizers, strict=True)
                ]
            )
        )
    lower_bound = float(np.percentile(bootstrap, 2.5))
    d_pooled = float(np.mean(d_values))
    z_pooled = float(np.mean(z_values))
    criteria = {
        "stability_at_least_0_60_every_seed": all(value >= 0.60 for value in stabilities),
        "cv_at_least_0_10_every_seed": all(value >= 0.10 for value in cvs),
        "Z_pooled_at_least_0_05": z_pooled >= 0.05,
        "bootstrap_lower_bound_positive": lower_bound > 0.0,
        "D_positive_every_seed": all(value > 0.0 for value in d_values),
    }
    fail_reasons: list[str] = []
    if sum(value < 0.30 for value in stabilities) >= 3:
        fail_reasons.append("stability is below 0.30 in at least three seeds")
    if sum(value < 0.05 for value in cvs) >= 3:
        fail_reasons.append("CV is below 0.05 in at least three seeds")
    if d_pooled <= 0.0:
        fail_reasons.append("D_pooled is nonpositive")
    if fail_reasons:
        stage_a = "FAIL"
    elif all(criteria.values()):
        stage_a = "PASS_ONWARD"
    else:
        stage_a = "UNRESOLVED"
    return {
        "stage_a": stage_a,
        "reasons": fail_reasons,
        "criteria": criteria,
        "identity_count": identity_count,
        "D_pooled": d_pooled,
        "Z_pooled": z_pooled,
        "D_by_seed": {str(seed): d_values[seed] for seed in range(4)},
        "seed_summaries": seed_summaries,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_95_lower_bound": lower_bound,
        "numpy_version": np.__version__,
    }


def _json_seed_full(result: dict[str, Any]) -> dict[str, Any]:
    target = np.asarray(result["target_identity_effects"], dtype=np.float64)
    controls = np.asarray(result["control_identity_effects"], dtype=np.float64)
    matched_mean = controls.mean(axis=1)
    canonical_controls = np.ascontiguousarray(controls.astype("<f8", copy=False))
    return {
        key: value
        for key, value in result.items()
        if key not in {"labels", "target_identity_effects", "control_identity_effects"}
    } | {
        "labels": np.asarray(result["labels"]).tolist(),
        "target_identity_effects": target.tolist(),
        "matched_mean_identity_effects": matched_mean.tolist(),
        "identity_advantages": (target - matched_mean).tolist(),
        "control_means": controls.mean(axis=0).tolist(),
        "control_identity_effects_shape": list(controls.shape),
        "control_identity_effects_sha256": hashlib.sha256(
            canonical_controls.tobytes(order="C")
        ).hexdigest(),
    }


def validate_cross_seed_training_binding(bounds: list[BoundSeed]) -> None:
    if not bounds:
        raise ValueError("cross-seed binding requires at least one seed")
    reference_ids = np.asarray(bounds[0].train_example_ids).astype(str)
    reference_labels = np.asarray(bounds[0].train_labels, dtype=np.int64)
    for bound in bounds[1:]:
        if not np.array_equal(np.asarray(bound.train_example_ids).astype(str), reference_ids):
            raise ValueError("training example-ID order differs across seeds")
        if not np.array_equal(np.asarray(bound.train_labels, dtype=np.int64), reference_labels):
            raise ValueError("training labels differ across seeds")


_OUTPUT_SCHEMA = {
    "schema_version": "pass181-cieb-stage-a-v1",
    "stages": ["artifact_binding", "entropy_stability_cv", "matched_masks_margin_bootstrap"],
    "early_stop": "registered entropy/CV fail",
    "candidate_statistics_use": "training split only",
}


def run_manifest(
    manifest_path: Path,
    preregistration_path: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or set(manifest.get("seeds", {})) != {
        "0",
        "1",
        "2",
        "3",
    }:
        raise ValueError("Pass181 manifest must bind exactly seeds zero through three")
    bounds: list[BoundSeed] = []
    for seed in range(4):
        bound = load_bound_seed(manifest["seeds"][str(seed)], seed=seed)
        if bound.train_embeddings.shape[1] != EXPECTED_DIMENSION:
            raise ValueError("CIEB requires a 512-D head")
        if set(np.unique(bound.train_labels).tolist()) != set(bound.proxy_labels.tolist()):
            raise ValueError("checkpoint must have one proxy for every and only training label")
        bounds.append(bound)
    validate_cross_seed_training_binding(bounds)
    entropy_results = [compute_entropy_screen(bound) for bound in bounds]

    stabilities = [float(result["stability"]) for result in entropy_results]
    cvs = [float(result["cv"]) for result in entropy_results]
    entropy_verdict = entropy_stage_verdict(stabilities=stabilities, cvs=cvs)
    preregistration = {
        "document_path": str(preregistration_path),
        "document_sha256": sha256_file(preregistration_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "diagnostic_source_sha256": sha256_file(Path(__file__)),
        "output_schema_sha256": hashlib.sha256(
            json.dumps(_OUTPUT_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "target_count": TARGET_COUNT,
        "matched_control_count": MASK_COUNT,
        "foreign_gallery_count": FOREIGN_K,
        "tau": TAU,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": int(bootstrap_replicates),
        "uses_test_data": "artifact_binding_only",
        "stage_a_cannot_clear_gate_1": True,
    }
    base = {
        **_OUTPUT_SCHEMA,
        "preregistration": preregistration,
        "artifact_binding": [bound.artifact_binding for bound in bounds],
        "official_recall_at_1": [bound.official_recall_at_1 for bound in bounds],
        "entropy_screen": entropy_results,
    }
    if entropy_verdict["stop_before_matched_masks"]:
        return {**base, "matched_mask_stage_executed": False, "verdict": entropy_verdict}

    full_results = [compute_seed_full(bound) for bound in bounds]
    verdict = full_stage_a_verdict(
        full_results,
        stabilities=stabilities,
        cvs=cvs,
        bootstrap_replicates=bootstrap_replicates,
    )
    return {
        **base,
        "matched_mask_stage_executed": True,
        "per_seed": [_json_seed_full(result) for result in full_results],
        "verdict": verdict,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("docs/pass181_cieb_proposal_2026-08-08.md"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    result = run_manifest(
        args.manifest,
        args.preregistration,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    write_json_atomic(args.output, result)
    print(json.dumps(result["verdict"], sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
