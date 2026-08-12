"""Deterministic train-cohort residual similarity primitives."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class HeldoutSplit:
    """Class-disjoint cohort and evaluation label sets."""

    cohort_labels: np.ndarray
    evaluation_labels: np.ndarray


@dataclass(frozen=True)
class RetrievalSplit:
    """One deterministic query and the remaining gallery rows per identity."""

    query_indices: np.ndarray
    gallery_indices: np.ndarray
    query_labels: np.ndarray


@dataclass(frozen=True)
class ArmComparison:
    """Recall and paired transition evidence relative to raw cosine."""

    raw_recall: float
    recall: float
    gain: float
    wrong_to_right: int
    right_to_wrong: int
    p_value: float
    shard_gains: tuple[float, float, float, float]
    raw_correct: int = 0
    correct: int = 0
    shard_counts: tuple[int, int, int, int] = (0, 0, 0, 0)
    shard_raw_correct: tuple[int, int, int, int] = (0, 0, 0, 0)
    shard_correct: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True)
class Evaluation:
    """All frozen arms and the centroid-assignment null distribution."""

    arms: dict[str, ArmComparison]
    shuffled_gains: tuple[float, ...]
    shuffled_p95: float


def _labels(value: np.ndarray, *, name: str = "labels") -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.dtype != np.int64
        or value.size == 0
    ):
        raise ValueError(f"{name} must be a nonempty int64 vector")
    return value


def _label_key(value: np.int64) -> tuple[bytes, int]:
    encoded = np.asarray(value, dtype="<i8").tobytes()
    return hashlib.sha256(encoded).digest(), int(value)


def split_heldout_labels(labels: np.ndarray) -> HeldoutSplit:
    """Split eligible labels 80/20 by a frozen content hash."""

    values = _labels(labels)
    unique, counts = np.unique(values, return_counts=True)
    eligible = sorted(unique[counts >= 2], key=_label_key)
    if len(eligible) < 2:
        raise ValueError("at least two evaluation-eligible labels are required")
    boundary = int(np.floor(0.8 * len(eligible)))
    if boundary == 0 or boundary == len(eligible):
        raise ValueError("eligible labels produced an empty partition")
    singletons = sorted(unique[counts == 1], key=_label_key)
    return HeldoutSplit(
        cohort_labels=np.asarray(eligible[:boundary] + singletons, dtype=np.int64),
        evaluation_labels=np.asarray(eligible[boundary:], dtype=np.int64),
    )


def _ids(value: np.ndarray, row_count: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.shape[0] != row_count
        or value.dtype.kind != "U"
        or any(not item for item in value.tolist())
        or len(set(value.tolist())) != row_count
    ):
        raise ValueError("example_ids must be unique nonempty Unicode strings")
    return value


def select_query_and_gallery(
    labels: np.ndarray, example_ids: np.ndarray, evaluation_labels: np.ndarray
) -> RetrievalSplit:
    """Choose one content-hash-minimal query row per evaluation label."""

    values = _labels(labels)
    ids = _ids(example_ids, values.size)
    selected_labels = _labels(evaluation_labels, name="evaluation_labels")
    if len(set(selected_labels.tolist())) != selected_labels.size:
        raise ValueError("evaluation_labels must be unique")
    query_indices: list[int] = []
    selected_rows: list[int] = []
    for label in selected_labels:
        rows = np.flatnonzero(values == label).tolist()
        if len(rows) < 2:
            raise ValueError("every evaluation label must contain at least two rows")
        query_index = min(
            rows,
            key=lambda index: (
                hashlib.sha256(str(ids[index]).encode("utf-8")).digest(),
                str(ids[index]),
            ),
        )
        query_indices.append(query_index)
        selected_rows.extend(rows)
    query_set = set(query_indices)
    gallery_indices = [index for index in selected_rows if index not in query_set]
    return RetrievalSplit(
        query_indices=np.asarray(query_indices, dtype=np.int64),
        gallery_indices=np.asarray(gallery_indices, dtype=np.int64),
        query_labels=np.asarray(selected_labels, dtype=np.int64),
    )


def reporting_shards(evaluation_labels: np.ndarray) -> np.ndarray:
    """Map labels to four domain-separated reporting shards."""

    labels = _labels(evaluation_labels, name="evaluation_labels")
    shards = [
        hashlib.sha256(b"AHNCR-shard-v1:" + np.asarray(label, dtype="<i8").tobytes()).digest()[0]
        >> 6
        for label in labels
    ]
    return np.asarray(shards, dtype=np.int64)


def _unit_embeddings(value: np.ndarray, *, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.dtype != np.float32
        or min(value.shape) == 0
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite rank-2 float32 array")
    norms = np.linalg.norm(value.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=5e-5):
        raise ValueError(f"{name} rows must be unit normalized")
    return value


def _validate_operator(
    values: np.ndarray, cohort: np.ndarray, k: int, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _unit_embeddings(values, name="values")
    cohort_matrix = _unit_embeddings(cohort, name="cohort")
    if matrix.shape[1] != cohort_matrix.shape[1]:
        raise ValueError("embedding dimensions differ")
    if type(k) is not int or k <= 0 or k > cohort_matrix.shape[0]:
        raise ValueError("k must be a positive integer no larger than the cohort")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    return matrix, cohort_matrix


def _stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    indices = np.arange(scores.shape[1], dtype=np.int64)
    return np.stack([np.lexsort((indices, -row))[:k] for row in scores], axis=0)


def hard_negative_centroids(
    queries: np.ndarray,
    cohort: np.ndarray,
    *,
    k: int = 50,
    block_size: int = 256,
) -> np.ndarray:
    """Mean the stable top-k class-disjoint cohort rows for each query."""

    query_matrix, cohort_matrix = _validate_operator(queries, cohort, k, block_size)
    result = np.empty_like(query_matrix)
    cohort_t = np.asarray(cohort_matrix.T, dtype=np.float32)
    for start in range(0, query_matrix.shape[0], block_size):
        stop = min(start + block_size, query_matrix.shape[0])
        scores = np.asarray(query_matrix[start:stop] @ cohort_t, dtype=np.float32)
        selected = _stable_topk(scores, k)
        means = np.mean(cohort_matrix[selected].astype(np.float64), axis=1)
        result[start:stop] = np.asarray(means, dtype=np.float32)
    return result


def unary_cohort_density(
    gallery: np.ndarray,
    cohort: np.ndarray,
    *,
    k: int = 50,
    block_size: int = 256,
) -> np.ndarray:
    """Mean stable top-k cohort similarity for each gallery row."""

    gallery_matrix, cohort_matrix = _validate_operator(gallery, cohort, k, block_size)
    result = np.empty(gallery_matrix.shape[0], dtype=np.float64)
    cohort_t = np.asarray(cohort_matrix.T, dtype=np.float32)
    for start in range(0, gallery_matrix.shape[0], block_size):
        stop = min(start + block_size, gallery_matrix.shape[0])
        scores = np.asarray(gallery_matrix[start:stop] @ cohort_t, dtype=np.float32)
        selected = _stable_topk(scores, k)
        result[start:stop] = np.mean(
            np.take_along_axis(scores, selected, axis=1).astype(np.float64),
            axis=1,
            dtype=np.float64,
        )
    return result


def exact_mcnemar(wrong_to_right: int, right_to_wrong: int) -> float:
    """Exact two-sided binomial McNemar p-value."""

    if (
        type(wrong_to_right) is not int
        or type(right_to_wrong) is not int
        or wrong_to_right < 0
        or right_to_wrong < 0
    ):
        raise ValueError("transition counts must be nonnegative concrete integers")
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(wrong_to_right, right_to_wrong) + 1)
    )
    return min(1.0, float(Fraction(2 * tail, 2**discordant)))


def _comparison(
    raw_indices: np.ndarray,
    corrected_indices: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    shard_ids: np.ndarray,
) -> ArmComparison:
    raw_correct = gallery_labels[raw_indices] == query_labels
    corrected = gallery_labels[corrected_indices] == query_labels
    raw_recall = float(np.mean(raw_correct, dtype=np.float64))
    recall = float(np.mean(corrected, dtype=np.float64))
    wrong_to_right = int(np.sum(~raw_correct & corrected))
    right_to_wrong = int(np.sum(raw_correct & ~corrected))
    shard_gains = tuple(
        float(
            np.mean(corrected[shard_ids == shard], dtype=np.float64)
            - np.mean(raw_correct[shard_ids == shard], dtype=np.float64)
        )
        for shard in range(4)
    )
    shard_counts = tuple(int(np.sum(shard_ids == shard)) for shard in range(4))
    shard_raw_correct = tuple(int(np.sum(raw_correct[shard_ids == shard])) for shard in range(4))
    shard_correct = tuple(int(np.sum(corrected[shard_ids == shard])) for shard in range(4))
    return ArmComparison(
        raw_recall=raw_recall,
        recall=recall,
        gain=recall - raw_recall,
        wrong_to_right=wrong_to_right,
        right_to_wrong=right_to_wrong,
        p_value=exact_mcnemar(wrong_to_right, right_to_wrong),
        shard_gains=shard_gains,  # type: ignore[arg-type]
        raw_correct=int(np.sum(raw_correct)),
        correct=int(np.sum(corrected)),
        shard_counts=shard_counts,  # type: ignore[arg-type]
        shard_raw_correct=shard_raw_correct,  # type: ignore[arg-type]
        shard_correct=shard_correct,  # type: ignore[arg-type]
    )


def _normalized_residual(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    residual = np.asarray(values - centroids, dtype=np.float32)
    norms = np.linalg.norm(residual.astype(np.float64), axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise ValueError("symmetric residual norms must be finite and nonzero")
    return np.asarray(residual / norms[:, None], dtype=np.float32)


def evaluate_ahncr(
    queries: np.ndarray,
    query_labels: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    cohort: np.ndarray,
    shard_ids: np.ndarray,
    *,
    k: int = 50,
    block_size: int = 256,
    permutation_seed: int = 20260814,
) -> Evaluation:
    """Evaluate AHNCR and all frozen controls from shared raw products."""

    query_matrix = _unit_embeddings(queries, name="queries")
    gallery_matrix = _unit_embeddings(gallery, name="gallery")
    cohort_matrix = _unit_embeddings(cohort, name="cohort")
    if not (query_matrix.shape[1] == gallery_matrix.shape[1] == cohort_matrix.shape[1]):
        raise ValueError("embedding dimensions differ")
    query_targets = _labels(query_labels, name="query_labels")
    gallery_targets = _labels(gallery_labels, name="gallery_labels")
    shards = _labels(shard_ids, name="shard_ids")
    if query_targets.size != query_matrix.shape[0] or shards.size != query_matrix.shape[0]:
        raise ValueError("query labels and shards must align with queries")
    if gallery_targets.size != gallery_matrix.shape[0]:
        raise ValueError("gallery labels must align with gallery")
    if set(shards.tolist()) != {0, 1, 2, 3}:
        raise ValueError("all four reporting shards must be nonempty")
    if type(permutation_seed) is not int or permutation_seed < 0:
        raise ValueError("permutation_seed must be a nonnegative concrete integer")

    query_centroids = hard_negative_centroids(
        query_matrix, cohort_matrix, k=k, block_size=block_size
    )
    gallery_centroids = hard_negative_centroids(
        gallery_matrix, cohort_matrix, k=k, block_size=block_size
    )
    gallery_density = unary_cohort_density(
        gallery_matrix, cohort_matrix, k=k, block_size=block_size
    )
    raw_scores = np.asarray(query_matrix @ gallery_matrix.T, dtype=np.float32)
    centroid_scores = np.asarray(query_centroids @ gallery_matrix.T, dtype=np.float32)
    global_mean = np.asarray(np.mean(cohort_matrix.astype(np.float64), axis=0), dtype=np.float32)
    global_scores = np.asarray(global_mean @ gallery_matrix.T, dtype=np.float32)
    symmetric_scores = np.asarray(
        _normalized_residual(query_matrix, query_centroids)
        @ _normalized_residual(gallery_matrix, gallery_centroids).T,
        dtype=np.float32,
    )
    scores = {
        "raw": raw_scores,
        "ahncr": np.asarray(2.0 * raw_scores - centroid_scores, dtype=np.float32),
        "global_mean": np.asarray(2.0 * raw_scores - global_scores[None, :], dtype=np.float32),
        "positive_expansion": np.asarray(2.0 * raw_scores + centroid_scores, dtype=np.float32),
        "unary_cohort_density": np.asarray(
            2.0 * raw_scores - gallery_density.astype(np.float32)[None, :],
            dtype=np.float32,
        ),
        "symmetric_ad_norm": symmetric_scores,
    }
    raw_indices = np.argmax(raw_scores, axis=1)
    arms = {
        name: _comparison(
            raw_indices,
            np.argmax(arm_scores, axis=1),
            query_targets,
            gallery_targets,
            shards,
        )
        for name, arm_scores in scores.items()
    }
    generator = np.random.Generator(np.random.PCG64(permutation_seed))
    shuffled_gains: list[float] = []
    for _ in range(20):
        permutation = generator.permutation(query_matrix.shape[0])
        shuffled_scores = np.asarray(
            2.0 * raw_scores - query_centroids[permutation] @ gallery_matrix.T,
            dtype=np.float32,
        )
        shuffled_gains.append(
            _comparison(
                raw_indices,
                np.argmax(shuffled_scores, axis=1),
                query_targets,
                gallery_targets,
                shards,
            ).gain
        )
    gains = tuple(shuffled_gains)
    return Evaluation(
        arms=arms,
        shuffled_gains=gains,
        shuffled_p95=float(np.quantile(gains, 0.95, method="linear")),
    )


def decide_ahncr(evaluation: Evaluation) -> tuple[bool, dict[str, bool]]:
    """Apply the prospectively frozen AHNCR mechanism gate."""

    if not isinstance(evaluation, Evaluation):
        raise ValueError("evaluation must be an Evaluation")
    required = (
        "raw",
        "ahncr",
        "global_mean",
        "positive_expansion",
        "unary_cohort_density",
        "symmetric_ad_norm",
    )
    if tuple(evaluation.arms) != required:
        raise ValueError("evaluation arms differ from the frozen order")
    candidate = evaluation.arms["ahncr"]
    predicates = {
        "pooled_gain": candidate.gain >= 0.003,
        "paired_significance": candidate.p_value < 0.01,
        "shard_consistency": sum(gain > 0.0 for gain in candidate.shard_gains) >= 3,
        "global_mean_control": candidate.gain - evaluation.arms["global_mean"].gain >= 0.001,
        "positive_expansion_control": (
            candidate.gain - evaluation.arms["positive_expansion"].gain >= 0.001
        ),
        "unary_density_control": (
            candidate.gain - evaluation.arms["unary_cohort_density"].gain >= 0.001
        ),
        "symmetric_normalization_control": (
            candidate.gain - evaluation.arms["symmetric_ad_norm"].gain >= 0.001
        ),
        "assignment_null": candidate.gain > evaluation.shuffled_p95,
        "transition_direction": candidate.wrong_to_right > candidate.right_to_wrong,
    }
    return all(predicates.values()), predicates
