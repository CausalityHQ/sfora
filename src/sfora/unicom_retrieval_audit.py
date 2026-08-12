"""Pure NumPy diagnostics for frozen UNICOM retrieval embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalView:
    recall: dict[int, float]
    map_at_r: float
    top1_indices: np.ndarray
    top1_correct: np.ndarray


@dataclass(frozen=True)
class GeometryDecision:
    primary: str
    full_dimension_control: bool
    evaluator_repair: bool
    coordinate_nonexchangeability: bool


@dataclass(frozen=True, eq=False)
class GeometryAudit:
    official: RetrievalView
    prefix_unit: RetrievalView
    full_unit: RetrievalView
    random_units: tuple[RetrievalView, ...]
    reproduction_passed: bool
    delta_norm: float
    norm_interval: tuple[float, float]
    delta_full: float
    full_interval: tuple[float, float]
    delta_mask: float
    mask_wins: int
    disagree: float
    energy_disagreement_count: int
    energy_gap_mean: float | None
    energy_gap_median: float | None
    energy_gap_negative_fraction: float | None
    energy_gap_interval: tuple[float, float] | None
    error_energy_point_biserial: float | None
    decision: GeometryDecision

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeometryAudit):
            return NotImplemented

        def view_equal(left: RetrievalView, right: RetrievalView) -> bool:
            return (
                left.recall == right.recall
                and left.map_at_r == right.map_at_r
                and np.array_equal(left.top1_indices, right.top1_indices)
                and np.array_equal(left.top1_correct, right.top1_correct)
            )

        return (
            view_equal(self.official, other.official)
            and view_equal(self.prefix_unit, other.prefix_unit)
            and view_equal(self.full_unit, other.full_unit)
            and len(self.random_units) == len(other.random_units)
            and all(
                view_equal(left, right)
                for left, right in zip(self.random_units, other.random_units, strict=True)
            )
            and self.reproduction_passed == other.reproduction_passed
            and self.delta_norm == other.delta_norm
            and self.norm_interval == other.norm_interval
            and self.delta_full == other.delta_full
            and self.full_interval == other.full_interval
            and self.delta_mask == other.delta_mask
            and self.mask_wins == other.mask_wins
            and self.disagree == other.disagree
            and self.energy_disagreement_count == other.energy_disagreement_count
            and self.energy_gap_mean == other.energy_gap_mean
            and self.energy_gap_median == other.energy_gap_median
            and self.energy_gap_negative_fraction == other.energy_gap_negative_fraction
            and self.energy_gap_interval == other.energy_gap_interval
            and self.error_energy_point_biserial == other.error_energy_point_biserial
            and self.decision == other.decision
        )


def _require_fp32_matrix(values: np.ndarray, *, name: str = "embeddings") -> None:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be a NumPy array")
    if values.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize FP32 rows with ordered FP64 reductions."""

    _require_fp32_matrix(values)
    squared = values.astype(np.float64) ** 2
    norms = np.sqrt(np.sum(squared, axis=1, keepdims=True, dtype=np.float64))
    if np.any(norms == 0.0):
        raise ValueError("embedding row has zero L2 norm")
    return np.ascontiguousarray((values.astype(np.float64) / norms).astype(np.float32))


def random_masks(*, dimension: int, selected: int, count: int) -> tuple[np.ndarray, ...]:
    if type(dimension) is not int or type(selected) is not int or type(count) is not int:
        raise TypeError("dimension, selected, and count must be builtin integers")
    if dimension <= 0 or selected <= 0 or selected > dimension or count <= 0:
        raise ValueError("invalid random-mask dimensions")
    return tuple(
        np.sort(
            np.random.Generator(np.random.PCG64(seed)).choice(
                dimension, selected, replace=False
            )
        ).astype(np.int64)
        for seed in range(count)
    )


def _validate_labels(labels: np.ndarray, rows: int, *, name: str) -> None:
    if type(labels) is not np.ndarray or labels.ndim != 1 or labels.shape[0] != rows:
        raise ValueError(f"{name} must be a one-dimensional array matching embedding rows")
    if labels.dtype.kind not in {"U", "S", "O"}:
        raise TypeError(f"{name} must contain identity strings")
    if any(type(value) is not str or not value for value in labels.tolist()):
        raise TypeError(f"{name} must contain nonempty builtin strings")


def _validate_coordinates(coordinates: np.ndarray, dimension: int) -> np.ndarray:
    if type(coordinates) is not np.ndarray or coordinates.dtype != np.int64:
        raise TypeError("coordinates must be an int64 NumPy array")
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("coordinates must be a nonempty vector")
    if np.any(coordinates < 0) or np.any(coordinates >= dimension):
        raise ValueError("coordinate is outside the embedding dimension")
    if not np.array_equal(coordinates, np.unique(coordinates)):
        raise ValueError("coordinates must be unique and sorted")
    return coordinates


def _selected_view(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    normalize_before: bool,
) -> np.ndarray:
    if type(normalize_before) is not bool:
        raise TypeError("normalize_before must be a builtin bool")
    if normalize_before:
        return np.ascontiguousarray(l2_normalize(values)[:, coordinates])
    selected = np.ascontiguousarray(values[:, coordinates])
    return l2_normalize(selected)


def retrieval_view(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    coordinates: np.ndarray,
    normalize_before: bool,
    chunk_size: int = 256,
) -> RetrievalView:
    """Evaluate one deterministic selected-coordinate retrieval view."""

    _require_fp32_matrix(query_embeddings, name="query_embeddings")
    _require_fp32_matrix(gallery_embeddings, name="gallery_embeddings")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    _validate_labels(query_labels, query_embeddings.shape[0], name="query_labels")
    _validate_labels(gallery_labels, gallery_embeddings.shape[0], name="gallery_labels")
    coordinates = _validate_coordinates(coordinates, query_embeddings.shape[1])
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive builtin integer")

    query = _selected_view(query_embeddings, coordinates, normalize_before=normalize_before)
    gallery = _selected_view(gallery_embeddings, coordinates, normalize_before=normalize_before)
    gallery64 = gallery.astype(np.float64)
    gallery_norms = np.sum(gallery64 * gallery64, axis=1, dtype=np.float64)
    gallery_order = np.arange(gallery.shape[0], dtype=np.int64)
    recall_hits = {key: [] for key in (1, 10, 20, 30)}
    average_precisions: list[float] = []
    top1_indices: list[int] = []

    for start in range(0, query.shape[0], chunk_size):
        query_chunk = query[start : start + chunk_size].astype(np.float64)
        query_norms = np.sum(query_chunk * query_chunk, axis=1, dtype=np.float64)
        distances = query_norms[:, None] + gallery_norms[None, :] - 2.0 * (
            query_chunk @ gallery64.T
        )
        for offset, row in enumerate(distances):
            query_index = start + offset
            order = np.lexsort((gallery_order, row))
            matches = gallery_labels[order] == query_labels[query_index]
            top1_indices.append(int(order[0]))
            for key in recall_hits:
                recall_hits[key].append(bool(np.any(matches[: min(key, matches.size)])))
            relevant = int(np.count_nonzero(gallery_labels == query_labels[query_index]))
            if relevant == 0:
                raise ValueError("query identity has no relevant gallery item")
            truncated = matches[:relevant]
            precision = np.cumsum(truncated, dtype=np.int64) / np.arange(1, relevant + 1)
            average_precisions.append(float(np.sum(precision * truncated) / relevant))

    top1 = np.asarray(top1_indices, dtype=np.int64)
    top1_correct = np.asarray(gallery_labels[top1] == query_labels, dtype=np.bool_)
    return RetrievalView(
        recall={key: float(np.mean(values)) for key, values in recall_hits.items()},
        map_at_r=float(np.mean(average_precisions)),
        top1_indices=top1,
        top1_correct=top1_correct,
    )


def paired_r1_interval(
    baseline_correct: np.ndarray,
    candidate_correct: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 205,
) -> tuple[float, float]:
    if (
        type(baseline_correct) is not np.ndarray
        or type(candidate_correct) is not np.ndarray
        or baseline_correct.dtype != np.bool_
        or candidate_correct.dtype != np.bool_
        or baseline_correct.ndim != 1
        or candidate_correct.shape != baseline_correct.shape
        or baseline_correct.size == 0
    ):
        raise TypeError("paired correctness arrays must be matching nonempty boolean vectors")
    if type(samples) is not int or samples <= 0 or type(seed) is not int:
        raise ValueError("samples and seed must be builtin integers with samples positive")
    differences = candidate_correct.astype(np.float64) - baseline_correct.astype(np.float64)
    return _bootstrap_mean_interval(differences, samples=samples, seed=seed)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 64,
) -> tuple[float, float]:
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1, dtype=np.float64)
    bounds = np.percentile(means, [2.5, 97.5])
    return float(bounds[0]), float(bounds[1])


def geometry_decision(
    *,
    delta_norm: float,
    norm_lower_bound: float,
    delta_full: float,
    full_lower_bound: float,
    delta_mask: float,
    mask_wins: int,
    disagree: float,
) -> GeometryDecision:
    values = (
        delta_norm,
        norm_lower_bound,
        delta_full,
        full_lower_bound,
        delta_mask,
        disagree,
    )
    if any(type(value) is not float or not np.isfinite(value) for value in values):
        raise TypeError("decision metrics must be finite builtin floats")
    if type(mask_wins) is not int or not 0 <= mask_wins <= 32:
        raise TypeError("mask_wins must be a builtin integer in [0, 32]")
    full_dimension_control = delta_full >= 0.002 and full_lower_bound > 0.0
    evaluator_repair = (
        not full_dimension_control and delta_norm >= 0.002 and norm_lower_bound > 0.0
    )
    coordinate = delta_mask >= 0.002 and mask_wins >= 24 and disagree >= 0.10
    if full_dimension_control:
        primary = "FULL_DIMENSION_CONTROL"
    elif evaluator_repair:
        primary = "EVALUATOR_REPAIR"
    elif coordinate:
        primary = "COORDINATE_NONEXCHANGEABILITY"
    else:
        primary = "GEOMETRY_NULL"
    return GeometryDecision(
        primary=primary,
        full_dimension_control=full_dimension_control,
        evaluator_repair=evaluator_repair,
        coordinate_nonexchangeability=coordinate,
    )


def audit_deployment_geometry(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    selected: int = 512,
    random_count: int = 32,
    bootstrap_samples: int = 10_000,
    expected_official_r1: float = 0.746,
    reproduction_tolerance: float = 0.002,
) -> GeometryAudit:
    """Run all registered retrieval views and reduce them to one audit result."""

    if type(selected) is not int or selected <= 0 or selected > query_embeddings.shape[1]:
        raise ValueError("selected must be within the embedding dimension")
    if type(random_count) is not int or random_count <= 0:
        raise ValueError("random_count must be positive")
    if type(expected_official_r1) is not float or type(reproduction_tolerance) is not float:
        raise TypeError("reproduction values must be builtin floats")
    if not np.isfinite(expected_official_r1) or not np.isfinite(reproduction_tolerance):
        raise ValueError("reproduction values must be finite")
    if reproduction_tolerance < 0.0:
        raise ValueError("reproduction_tolerance must be nonnegative")

    prefix = np.arange(selected, dtype=np.int64)
    full = np.arange(query_embeddings.shape[1], dtype=np.int64)
    official = retrieval_view(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        coordinates=prefix,
        normalize_before=True,
    )
    prefix_unit = retrieval_view(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        coordinates=prefix,
        normalize_before=False,
    )
    full_unit = retrieval_view(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        coordinates=full,
        normalize_before=False,
    )
    random_units = tuple(
        retrieval_view(
            query_embeddings,
            gallery_embeddings,
            query_labels,
            gallery_labels,
            coordinates=mask,
            normalize_before=False,
        )
        for mask in random_masks(
            dimension=query_embeddings.shape[1], selected=selected, count=random_count
        )
    )

    delta_norm = float(prefix_unit.recall[1] - official.recall[1])
    norm_interval = paired_r1_interval(
        official.top1_correct,
        prefix_unit.top1_correct,
        samples=bootstrap_samples,
        seed=205,
    )
    delta_full = float(full_unit.recall[1] - prefix_unit.recall[1])
    full_interval = paired_r1_interval(
        prefix_unit.top1_correct,
        full_unit.top1_correct,
        samples=bootstrap_samples,
        seed=206,
    )
    random_r1 = np.asarray([view.recall[1] for view in random_units], dtype=np.float64)
    delta_mask = float(np.median(random_r1) - prefix_unit.recall[1])
    mask_wins = int(np.count_nonzero(random_r1 > prefix_unit.recall[1]))
    disagreement_rates = np.asarray(
        [np.mean(view.top1_indices != prefix_unit.top1_indices) for view in random_units],
        dtype=np.float64,
    )
    disagree = float(np.median(disagreement_rates))

    gallery_full_unit = l2_normalize(gallery_embeddings)
    prefix_energy = np.sum(
        gallery_full_unit[:, :selected].astype(np.float64) ** 2,
        axis=1,
        dtype=np.float64,
    )
    disagreement = official.top1_indices != prefix_unit.top1_indices
    energy_gaps = (
        prefix_energy[official.top1_indices[disagreement]]
        - prefix_energy[prefix_unit.top1_indices[disagreement]]
    )
    if energy_gaps.size:
        energy_gap_mean: float | None = float(np.mean(energy_gaps, dtype=np.float64))
        energy_gap_median: float | None = float(np.median(energy_gaps))
        energy_gap_negative_fraction: float | None = float(np.mean(energy_gaps < 0.0))
        energy_gap_interval: tuple[float, float] | None = _bootstrap_mean_interval(
            energy_gaps,
            samples=bootstrap_samples,
            seed=205,
        )
    else:
        energy_gap_mean = None
        energy_gap_median = None
        energy_gap_negative_fraction = None
        energy_gap_interval = None

    errors = (~official.top1_correct).astype(np.float64)
    selected_energy = prefix_energy[official.top1_indices]
    if np.std(errors) == 0.0 or np.std(selected_energy) == 0.0:
        association: float | None = None
    else:
        association = float(np.corrcoef(errors, selected_energy)[0, 1])

    reproduction_passed = (
        abs(full_unit.recall[1] - expected_official_r1) <= reproduction_tolerance
    )
    if reproduction_passed:
        decision = geometry_decision(
            delta_norm=delta_norm,
            norm_lower_bound=norm_interval[0],
            delta_full=delta_full,
            full_lower_bound=full_interval[0],
            delta_mask=delta_mask,
            mask_wins=mask_wins,
            disagree=disagree,
        )
    else:
        decision = GeometryDecision(
            primary="REPRODUCTION_FAILED",
            full_dimension_control=False,
            evaluator_repair=False,
            coordinate_nonexchangeability=False,
        )
    return GeometryAudit(
        official=official,
        prefix_unit=prefix_unit,
        full_unit=full_unit,
        random_units=random_units,
        reproduction_passed=reproduction_passed,
        delta_norm=delta_norm,
        norm_interval=norm_interval,
        delta_full=delta_full,
        full_interval=full_interval,
        delta_mask=delta_mask,
        mask_wins=mask_wins,
        disagree=disagree,
        energy_disagreement_count=int(energy_gaps.size),
        energy_gap_mean=energy_gap_mean,
        energy_gap_median=energy_gap_median,
        energy_gap_negative_fraction=energy_gap_negative_fraction,
        energy_gap_interval=energy_gap_interval,
        error_energy_point_biserial=association,
        decision=decision,
    )
