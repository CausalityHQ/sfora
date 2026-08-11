"""Train-only amortization of a gallery local-scale potential."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class RidgePotential:
    """A scalar affine potential fitted with an unregularized intercept."""

    weights: np.ndarray
    intercept: float
    ridge_lambda: float


@dataclass(frozen=True)
class RetrievalComparison:
    """Raw/corrected Recall@1 and their exact paired transition statistics."""

    raw_recall: float
    corrected_recall: float
    gain: float
    wrong_to_right: int
    right_to_wrong: int
    p_value: float


def _validate_design(value: np.ndarray, *, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.dtype != np.float32
        or value.shape[0] == 0
        or value.shape[1] == 0
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a nonempty finite rank-2 float32 array")
    return value


def _validate_targets(value: np.ndarray, row_count: int, *, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (row_count,)
        or value.dtype != np.float64
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite row-aligned float64 vector")
    return value


def fit_ridge_potential(
    embeddings: np.ndarray, targets: np.ndarray, ridge_lambda: float
) -> RidgePotential:
    """Fit a centered ridge model while leaving its intercept unregularized."""

    design32 = _validate_design(embeddings, name="embeddings")
    response = _validate_targets(targets, design32.shape[0], name="targets")
    if type(ridge_lambda) is not float or not np.isfinite(ridge_lambda):
        raise ValueError("ridge_lambda must be a finite float")
    if ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda must be positive")

    design = design32.astype(np.float64)
    design_mean = np.mean(design, axis=0, dtype=np.float64)
    target_mean = float(np.mean(response, dtype=np.float64))
    centered_design = design - design_mean
    target_scale = float(np.std(response, dtype=np.float64))
    if target_scale == 0.0:
        weights = np.zeros(design.shape[1], dtype=np.float64)
    else:
        standardized_target = (response - target_mean) / target_scale
        gram = centered_design.T @ centered_design
        weights_standardized = np.linalg.solve(
            gram + ridge_lambda * np.eye(design.shape[1], dtype=np.float64),
            centered_design.T @ standardized_target,
        )
        weights = np.asarray(weights_standardized * target_scale, dtype=np.float64)
    intercept = float(target_mean - design_mean @ weights)
    return RidgePotential(
        weights=weights, intercept=intercept, ridge_lambda=ridge_lambda
    )


def predict_potential(model: RidgePotential, embeddings: np.ndarray) -> np.ndarray:
    """Predict a raw cosine-unit local-scale potential."""

    if not isinstance(model, RidgePotential):
        raise ValueError("model must be a RidgePotential")
    design = _validate_design(embeddings, name="embeddings")
    if model.weights.dtype != np.float64 or model.weights.shape != (design.shape[1],):
        raise ValueError("model weights do not match embedding dimensions")
    if not np.isfinite(model.weights).all() or not np.isfinite(model.intercept):
        raise ValueError("model parameters must be finite")
    return np.asarray(
        design.astype(np.float64) @ model.weights + model.intercept,
        dtype=np.float64,
    )


def select_ridge_lambda(
    fit_embeddings: np.ndarray,
    fit_targets: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_targets: np.ndarray,
    ridge_grid: tuple[float, ...],
) -> tuple[RidgePotential, list[dict[str, float]]]:
    """Select the first ridge lambda attaining the lowest validation MSE."""

    fit_design = _validate_design(fit_embeddings, name="fit_embeddings")
    validation_design = _validate_design(
        validation_embeddings, name="validation_embeddings"
    )
    if fit_design.shape[1] != validation_design.shape[1]:
        raise ValueError("fit and validation embedding dimensions differ")
    fit_response = _validate_targets(
        fit_targets, fit_design.shape[0], name="fit_targets"
    )
    validation_response = _validate_targets(
        validation_targets,
        validation_design.shape[0],
        name="validation_targets",
    )
    if not isinstance(ridge_grid, tuple) or not ridge_grid:
        raise ValueError("ridge_grid must be a nonempty tuple")

    selected: RidgePotential | None = None
    best_mse = np.inf
    rows: list[dict[str, float]] = []
    for ridge_lambda in ridge_grid:
        model = fit_ridge_potential(fit_design, fit_response, ridge_lambda)
        errors = predict_potential(model, validation_design) - validation_response
        mse = float(np.mean(errors * errors, dtype=np.float64))
        rows.append({"ridge_lambda": ridge_lambda, "validation_mse": mse})
        if mse < best_mse:
            best_mse = mse
            selected = model
    if selected is None:
        raise ValueError("ridge_grid did not contain a valid model")
    return selected, rows


def _validate_labels(value: np.ndarray, row_count: int, *, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.shape != (row_count,)
        or value.dtype != np.int64
    ):
        raise ValueError(f"{name} must be a row-aligned int64 vector")
    return value


def _exact_mcnemar(wrong_to_right: int, right_to_wrong: int) -> float:
    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(wrong_to_right, right_to_wrong) + 1)
    )
    return min(1.0, float(Fraction(2 * tail, 2**discordant)))


def compare_potential(
    queries: np.ndarray,
    query_labels: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    potential: np.ndarray,
    block_size: int = 256,
) -> RetrievalComparison:
    """Compare raw cosine with ``2*cosine - potential`` using paired queries."""

    return compare_potentials(
        queries,
        query_labels,
        gallery,
        gallery_labels,
        {"potential": potential},
        block_size,
    )["potential"]


def compare_potentials(
    queries: np.ndarray,
    query_labels: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    potentials: dict[str, np.ndarray],
    block_size: int = 256,
) -> dict[str, RetrievalComparison]:
    """Compare many unary potentials while sharing each query-gallery product."""

    query_matrix = _validate_embeddings(queries)
    gallery_matrix = _validate_embeddings(gallery)
    if query_matrix.shape[1] != gallery_matrix.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    query_targets = _validate_labels(
        query_labels, query_matrix.shape[0], name="query_labels"
    )
    gallery_targets = _validate_labels(
        gallery_labels, gallery_matrix.shape[0], name="gallery_labels"
    )
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if type(potentials) is not dict or not potentials:
        raise ValueError("potentials must be a nonempty concrete dict")
    potential32: dict[str, np.ndarray] = {}
    for name, potential in potentials.items():
        if type(name) is not str or not name:
            raise ValueError("potential names must be nonempty concrete strings")
        if (
            not isinstance(potential, np.ndarray)
            or potential.shape != (gallery_matrix.shape[0],)
            or potential.dtype != np.float64
            or not np.isfinite(potential).all()
        ):
            raise ValueError(
                "each potential must be a finite gallery-aligned float64 vector"
            )
        potential32[name] = np.asarray(potential, dtype=np.float32)

    raw_indices = np.empty(query_matrix.shape[0], dtype=np.int64)
    corrected_indices = {
        name: np.empty(query_matrix.shape[0], dtype=np.int64) for name in potentials
    }
    gallery_t = np.asarray(gallery_matrix.T, dtype=np.float32)
    for start in range(0, query_matrix.shape[0], block_size):
        stop = min(start + block_size, query_matrix.shape[0])
        scores = np.asarray(
            2.0 * (np.asarray(query_matrix[start:stop]) @ gallery_t),
            dtype=np.float32,
        )
        raw_indices[start:stop] = np.argmax(scores, axis=1)
        for name, potential in potential32.items():
            corrected_indices[name][start:stop] = np.argmax(
                scores - potential, axis=1
            )
    raw_correct = gallery_targets[raw_indices] == query_targets
    raw_recall = float(np.mean(raw_correct, dtype=np.float64))
    results: dict[str, RetrievalComparison] = {}
    for name, indices in corrected_indices.items():
        corrected_correct = gallery_targets[indices] == query_targets
        wrong_to_right = int(np.sum(~raw_correct & corrected_correct))
        right_to_wrong = int(np.sum(raw_correct & ~corrected_correct))
        corrected_recall = float(np.mean(corrected_correct, dtype=np.float64))
        results[name] = RetrievalComparison(
            raw_recall=raw_recall,
            corrected_recall=corrected_recall,
            gain=corrected_recall - raw_recall,
            wrong_to_right=wrong_to_right,
            right_to_wrong=right_to_wrong,
            p_value=_exact_mcnemar(wrong_to_right, right_to_wrong),
        )
    return results


def _average_ranks(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="stable")
    ranks = np.empty(value.size, dtype=np.float64)
    start = 0
    while start < value.size:
        stop = start + 1
        while stop < value.size and value[order[stop]] == value[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left, dtype=np.float64)
    right_centered = right - np.mean(right, dtype=np.float64)
    denominator = float(
        np.sqrt(
            np.sum(left_centered * left_centered, dtype=np.float64)
            * np.sum(right_centered * right_centered, dtype=np.float64)
        )
    )
    if denominator == 0.0:
        raise ValueError("correlation is undefined for a constant vector")
    return float(
        np.sum(left_centered * right_centered, dtype=np.float64) / denominator
    )


def density_diagnostics(
    predicted: np.ndarray, observed: np.ndarray
) -> dict[str, float]:
    """Return Pearson, average-rank Spearman, and MSE diagnostics."""

    if (
        not isinstance(predicted, np.ndarray)
        or not isinstance(observed, np.ndarray)
        or predicted.ndim != 1
        or observed.shape != predicted.shape
        or predicted.dtype != np.float64
        or observed.dtype != np.float64
        or predicted.size < 2
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("diagnostic inputs must be finite aligned float64 vectors")
    errors = predicted - observed
    return {
        "pearson": _correlation(predicted, observed),
        "spearman": _correlation(_average_ranks(predicted), _average_ranks(observed)),
        "mse": float(np.mean(errors * errors, dtype=np.float64)),
    }


def decide_alsp(
    *,
    correlation: float,
    alsp_gain: float,
    oracle_gain: float,
    alsp_p_value: float,
    permuted_gain: float,
    random_null_p95: float,
) -> tuple[bool, dict[str, bool]]:
    """Apply the prospectively frozen ALSP pass predicates."""

    values = (
        correlation,
        alsp_gain,
        oracle_gain,
        alsp_p_value,
        permuted_gain,
        random_null_p95,
    )
    if any(type(value) is not float or not np.isfinite(value) for value in values):
        raise ValueError("decision inputs must be finite concrete floats")
    predicates = {
        "correlation": correlation >= 0.20,
        "absolute_gain": alsp_gain >= 0.001,
        "oracle_recovery": oracle_gain > 0.0 and alsp_gain >= 0.30 * oracle_gain,
        "paired_significance": 0.0 <= alsp_p_value < 0.05,
        "permuted_control": permuted_gain < alsp_gain - 0.00025,
        "random_direction_null": alsp_gain > random_null_p95,
    }
    return all(predicates.values()), predicates


def _validate_embeddings(value: np.ndarray) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.dtype != np.float32
        or value.shape[0] < 2
        or value.shape[1] == 0
    ):
        raise ValueError("embeddings must be a nonempty rank-2 float32 array")
    if not np.isfinite(value).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(value.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-4):
        raise ValueError("embeddings must be unit-normalized")
    return value


def _validate_row_ids(value: np.ndarray, row_count: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.shape[0] != row_count
        or value.dtype.kind != "U"
        or np.any(value == "")
        or np.unique(value).size != value.size
    ):
        raise ValueError("row_ids must be unique nonempty Unicode row identifiers")
    return value


def nonself_density(
    embeddings: np.ndarray,
    row_ids: np.ndarray,
    k: int = 50,
    block_size: int = 256,
) -> np.ndarray:
    """Return each row's mean top-k cosine to explicitly different row IDs."""

    matrix = _validate_embeddings(embeddings)
    identifiers = _validate_row_ids(row_ids, matrix.shape[0])
    if type(k) is not int or k <= 0 or k >= matrix.shape[0]:
        raise ValueError("k must be a positive integer below the row count")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    result = np.empty(matrix.shape[0], dtype=np.float64)
    matrix_t = np.asarray(matrix.T, dtype=np.float32)
    index_by_id = {str(row_id): index for index, row_id in enumerate(identifiers)}
    for start in range(0, matrix.shape[0], block_size):
        stop = min(start + block_size, matrix.shape[0])
        scores = np.asarray(matrix[start:stop] @ matrix_t, dtype=np.float32)
        for local_index, row_id in enumerate(identifiers[start:stop]):
            scores[local_index, index_by_id[str(row_id)]] = -np.inf
        selected = np.partition(scores, scores.shape[1] - k, axis=1)[:, -k:]
        result[start:stop] = np.mean(selected, axis=1, dtype=np.float64)
    return result


def split_labels(
    labels: np.ndarray, fit_fraction: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic hash-ordered class-disjoint fit/validation split."""

    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or labels.dtype != np.int64
        or labels.size < 2
    ):
        raise ValueError("labels must be a rank-1 int64 array with at least two rows")
    if type(fit_fraction) is not float or not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be a float strictly between zero and one")
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("at least two distinct labels are required")
    ordered = sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(np.asarray(value, dtype=np.int64).tobytes()).digest(),
            int(value),
        ),
    )
    boundary = int(np.floor(len(ordered) * fit_fraction))
    if boundary == 0 or boundary == len(ordered):
        raise ValueError("fit_fraction produces an empty label partition")
    return (
        np.asarray(ordered[:boundary], dtype=np.int64),
        np.asarray(ordered[boundary:], dtype=np.int64),
    )
