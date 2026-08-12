"""Pure NumPy calibrated-tail-moment retrieval arithmetic."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from sfora.unicom_retrieval_audit import (
    l2_normalize,
    retrieval_metrics_from_score_chunks,
)


def _require_unit_fp32(values: np.ndarray, *, name: str = "embeddings") -> None:
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
    values64 = values.astype(np.float64)
    norms = np.sqrt(np.sum(values64 * values64, axis=1, dtype=np.float64))
    if np.any(norms == 0.0):
        raise ValueError(f"{name} contains a zero-norm row")
    if np.any(np.abs(norms - 1.0) > 2e-6):
        raise ValueError(f"{name} rows must be unit norm")


def _require_basis_kind(value: object) -> str:
    if type(value) is not str or value not in {"native", "pca"}:
        raise ValueError("basis_kind must be native or pca")
    return value


@dataclass(frozen=True, eq=False)
class TailMomentFit:
    dimension: int
    basis_kind: str
    width: int
    neighbors: int
    pairs: np.ndarray
    lambda_raw: float
    lambda_value: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TailMomentFit):
            return NotImplemented
        return (
            self.dimension == other.dimension
            and self.basis_kind == other.basis_kind
            and self.width == other.width
            and self.neighbors == other.neighbors
            and np.array_equal(self.pairs, other.pairs)
            and self.lambda_raw == other.lambda_raw
            and self.lambda_value == other.lambda_value
        )


@dataclass(frozen=True, eq=False)
class ProjectionBasis:
    kind: str
    mean: np.ndarray
    matrix: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProjectionBasis):
            return NotImplemented
        return (
            self.kind == other.kind
            and np.array_equal(self.mean, other.mean)
            and np.array_equal(self.matrix, other.matrix)
        )


@dataclass(frozen=True)
class LambdaInterval:
    point: float
    lower: float
    upper: float
    samples: int
    seed: int


@dataclass(frozen=True, eq=False)
class TailNull:
    seeds: tuple[int, ...]
    pairs: np.ndarray
    lambda_raw_values: tuple[float, ...]
    p_value: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TailNull):
            return NotImplemented
        return (
            self.seeds == other.seeds
            and np.array_equal(self.pairs, other.pairs)
            and self.lambda_raw_values == other.lambda_raw_values
            and self.p_value == other.p_value
        )


@dataclass(frozen=True, eq=False)
class ScoreView:
    recall: dict[int, float]
    map_at_r: float
    top1_indices: np.ndarray
    top1_correct: np.ndarray
    values_per_row: int
    fixed_bytes: int
    total_bytes: int
    query_projection_seconds: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoreView):
            return NotImplemented
        return (
            self.recall == other.recall
            and self.map_at_r == other.map_at_r
            and np.array_equal(self.top1_indices, other.top1_indices)
            and np.array_equal(self.top1_correct, other.top1_correct)
            and self.values_per_row == other.values_per_row
            and self.fixed_bytes == other.fixed_bytes
            and self.total_bytes == other.total_bytes
            and self.query_projection_seconds == other.query_projection_seconds
        )


@dataclass(frozen=True)
class PairedInterval:
    point: float
    lower: float
    upper: float
    samples: int
    seed: int


@dataclass(frozen=True)
class CTMDecision:
    status: str
    reason: str
    r1_gain_passed: bool
    r1_lower_passed: bool
    map_passed: bool
    gap_recovery_passed: bool
    quality_controls_passed: bool
    equal_storage_passed: bool
    coefficient_passed: bool
    some_width_signal_passed: bool
    replication_passed: bool


CONTROL_ORDER = (
    "native_ctm",
    "pca_ctm",
    "renormalized_prefix_plus_zero",
    "plain_prefix_plus_zero",
    "lambda_zero",
    "unicom_tail_energy",
    "pca_renormalized_prefix_plus_zero",
    "tail_sign_control",
    "tail_permuted_control",
    "official_512",
    "full_width_768",
)
MATCHED_ROW_WIDTH_CONTROLS = CONTROL_ORDER[2:9]
EQUAL_TOTAL_STORAGE_CONTROLS = (
    "renormalized_prefix_plus_zero",
    "plain_prefix_plus_zero",
    "lambda_zero",
    "unicom_tail_energy",
    "tail_sign_control",
    "tail_permuted_control",
)


def head_neighbor_pairs(
    unit: np.ndarray,
    *,
    width: int,
    neighbors: int = 50,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return stable top-head-inner-product directed neighbor pairs."""

    _require_unit_fp32(unit)
    if type(width) is not int or not 0 < width < unit.shape[1]:
        raise ValueError("width must be a builtin integer strictly below dimension")
    if type(neighbors) is not int or not 0 < neighbors < unit.shape[0]:
        raise ValueError("neighbors must be a positive builtin integer below row count")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive builtin integer")

    head = unit[:, :width].astype(np.float64)
    pairs = np.empty((head.shape[0], neighbors, 2), dtype=np.int64)
    gallery_indices = np.arange(head.shape[0], dtype=np.int64)
    for start in range(0, head.shape[0], chunk_size):
        scores = head[start : start + chunk_size] @ head.T
        for offset, row in enumerate(scores):
            query_index = start + offset
            row[query_index] = -np.inf
            partition = np.argpartition(row, row.size - neighbors)[-neighbors:]
            boundary = np.min(row[partition])
            above = np.flatnonzero(row > boundary)
            tied = np.flatnonzero(row == boundary)[: neighbors - above.size]
            candidates = np.concatenate((above, tied))
            order = candidates[
                np.lexsort((gallery_indices[candidates], -row[candidates]))
            ]
            pairs[query_index, :, 0] = query_index
            pairs[query_index, :, 1] = order
    result = np.ascontiguousarray(pairs.reshape(-1, 2))
    result.flags.writeable = False
    return result


def _row_sufficient_statistics(
    tail: np.ndarray,
    radius: np.ndarray,
    pairs: np.ndarray,
    *,
    neighbors: int,
    query_chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = tail.shape[0]
    shaped = pairs.reshape(row_count, neighbors, 2)
    numerator = np.empty(row_count, dtype=np.float64)
    denominator = np.empty(row_count, dtype=np.float64)
    for start in range(0, row_count, query_chunk_size):
        stop = min(start + query_chunk_size, row_count)
        selected = shaped[start:stop]
        query_indices = selected[:, :, 0]
        gallery_indices = selected[:, :, 1]
        x = radius[query_indices] * radius[gallery_indices]
        y = np.sum(
            tail[query_indices] * tail[gallery_indices], axis=2, dtype=np.float64
        )
        numerator[start:stop] = np.sum(x * y, axis=1, dtype=np.float64)
        denominator[start:stop] = np.sum(x * x, axis=1, dtype=np.float64)
    return numerator, denominator


def _pair_sufficient_statistics(
    tail: np.ndarray,
    radius: np.ndarray,
    pairs: np.ndarray,
    *,
    neighbors: int,
    query_chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = tail.shape[0]
    shaped = pairs.reshape(row_count, neighbors, 2)
    numerator = np.empty(pairs.shape[0], dtype=np.float64)
    denominator = np.empty(pairs.shape[0], dtype=np.float64)
    for start in range(0, row_count, query_chunk_size):
        stop = min(start + query_chunk_size, row_count)
        selected = shaped[start:stop]
        query_indices = selected[:, :, 0]
        gallery_indices = selected[:, :, 1]
        x = radius[query_indices] * radius[gallery_indices]
        y = np.sum(
            tail[query_indices] * tail[gallery_indices], axis=2, dtype=np.float64
        )
        destination = slice(start * neighbors, stop * neighbors)
        numerator[destination] = (x * y).reshape(-1)
        denominator[destination] = (x * x).reshape(-1)
    return numerator, denominator


def fit_tail_moment(
    unit: np.ndarray,
    *,
    width: int,
    basis_kind: str,
    neighbors: int = 50,
) -> TailMomentFit:
    """Fit the registered scalar tail-product coefficient."""

    basis_kind = _require_basis_kind(basis_kind)
    pairs = head_neighbor_pairs(unit, width=width, neighbors=neighbors)
    tail = unit[:, width:].astype(np.float64)
    radius = np.sqrt(np.sum(tail * tail, axis=1, dtype=np.float64))
    numerator_by_row, denominator_by_row = _row_sufficient_statistics(
        tail, radius, pairs, neighbors=neighbors
    )
    numerator = float(np.sum(numerator_by_row, dtype=np.float64))
    denominator = float(np.sum(denominator_by_row, dtype=np.float64))
    lambda_raw = 0.0 if denominator == 0.0 else numerator / denominator
    return TailMomentFit(
        dimension=unit.shape[1],
        basis_kind=basis_kind,
        width=width,
        neighbors=neighbors,
        pairs=pairs,
        lambda_raw=lambda_raw,
        lambda_value=float(np.clip(lambda_raw, 0.0, 1.0)),
    )


def encode_tail_moment(
    unit: np.ndarray,
    fit: TailMomentFit,
    *,
    basis_kind: str,
) -> np.ndarray:
    """Encode unit rows as the registered head plus calibrated tail radius."""

    _require_unit_fp32(unit)
    basis_kind = _require_basis_kind(basis_kind)
    if type(fit) is not TailMomentFit:
        raise TypeError("fit must be a TailMomentFit")
    if unit.shape[1] != fit.dimension or basis_kind != fit.basis_kind:
        raise ValueError("embedding basis/dimension differs from fit")
    tail = unit[:, fit.width :].astype(np.float64)
    radius = np.sqrt(np.sum(tail * tail, axis=1, dtype=np.float64))
    scalar = np.sqrt(fit.lambda_value) * radius
    return np.ascontiguousarray(
        np.column_stack((unit[:, : fit.width], scalar.astype(np.float32))),
        dtype=np.float32,
    )


def fit_projection_basis(train_unit: np.ndarray, *, kind: str) -> ProjectionBasis:
    """Fit a native or train-only PCA basis with canonical column signs."""

    _require_unit_fp32(train_unit, name="train_unit")
    if type(kind) is not str or kind not in {"native", "pca"}:
        raise ValueError("basis kind must be native or pca")
    dimension = train_unit.shape[1]
    if kind == "native":
        mean = np.zeros(dimension, dtype=np.float32)
        matrix = np.eye(dimension, dtype=np.float32)
    else:
        if train_unit.shape[0] < dimension:
            raise ValueError("PCA needs at least as many train rows as dimensions")
        train64 = train_unit.astype(np.float64)
        mean64 = np.mean(train64, axis=0, dtype=np.float64)
        _, _, vt = np.linalg.svd(train64 - mean64, full_matrices=False)
        matrix64 = vt.T
        for column in matrix64.T:
            pivot = int(np.argmax(np.abs(column)))
            if column[pivot] < 0.0:
                column *= -1.0
        mean = np.ascontiguousarray(mean64.astype(np.float32))
        matrix = np.ascontiguousarray(matrix64.astype(np.float32))
    mean.flags.writeable = False
    matrix.flags.writeable = False
    return ProjectionBasis(kind=kind, mean=mean, matrix=matrix)


def project_unit(unit: np.ndarray, basis: ProjectionBasis) -> np.ndarray:
    """Apply one frozen basis without refitting on evaluation rows."""

    _require_unit_fp32(unit)
    if type(basis) is not ProjectionBasis:
        raise TypeError("basis must be a ProjectionBasis")
    if basis.mean.shape != (unit.shape[1],) or basis.matrix.shape != (
        unit.shape[1],
        unit.shape[1],
    ):
        raise ValueError("basis dimension differs")
    if basis.kind == "native":
        return unit
    projected64 = (unit.astype(np.float64) - basis.mean.astype(np.float64)) @ basis.matrix
    return l2_normalize(np.ascontiguousarray(projected64.astype(np.float32)))


def _require_train_labels(labels: np.ndarray, *, rows: int) -> tuple[str, ...]:
    if type(labels) is not np.ndarray or labels.ndim != 1 or labels.shape != (rows,):
        raise ValueError("train labels must match embedding rows")
    if labels.dtype.kind not in {"U", "S", "O"}:
        raise TypeError("train labels must contain identity strings")
    values = labels.tolist()
    if any(type(value) is not str or not value for value in values):
        raise TypeError("train labels must contain nonempty builtin strings")
    return tuple(values)


def cluster_lambda_interval(
    unit: np.ndarray,
    labels: np.ndarray,
    *,
    width: int,
    basis_kind: str,
    samples: int = 10_000,
    seed: int = 205,
    neighbors: int = 50,
) -> LambdaInterval:
    """Return a two-way train-identity cluster interval for lambda_raw."""

    if type(samples) is not int or samples <= 0 or type(seed) is not int:
        raise ValueError("samples and seed must be builtin integers with samples positive")
    label_values = _require_train_labels(labels, rows=unit.shape[0])
    fit = fit_tail_moment(
        unit, width=width, basis_kind=basis_kind, neighbors=neighbors
    )
    tail = unit[:, width:].astype(np.float64)
    radius = np.sqrt(np.sum(tail * tail, axis=1, dtype=np.float64))
    numerators, denominators = _pair_sufficient_statistics(
        tail, radius, fit.pairs, neighbors=neighbors
    )
    identities = tuple(dict.fromkeys(label_values))
    identity_index = {identity: index for index, identity in enumerate(identities)}
    row_identity = np.asarray([identity_index[value] for value in label_values], dtype=np.int64)
    query_identity = row_identity[fit.pairs[:, 0]]
    gallery_identity = row_identity[fit.pairs[:, 1]]
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(samples, dtype=np.float64)
    for draw_index in range(samples):
        sampled = generator.integers(0, len(identities), size=len(identities))
        counts = np.bincount(sampled, minlength=len(identities)).astype(np.float64)
        weights = counts[query_identity] * counts[gallery_identity]
        denominator = float(np.sum(denominators * weights, dtype=np.float64))
        draws[draw_index] = (
            0.0
            if denominator == 0.0
            else float(np.sum(numerators * weights, dtype=np.float64)) / denominator
        )
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return LambdaInterval(
        point=fit.lambda_raw,
        lower=float(lower),
        upper=float(upper),
        samples=samples,
        seed=seed,
    )


def permuted_tail_null(
    unit: np.ndarray,
    *,
    width: int,
    basis_kind: str,
    neighbors: int = 50,
    seeds: range = range(206, 238),
) -> TailNull:
    """Permute tail directions while preserving radii and observed head pairs."""

    if type(seeds) is not range or len(seeds) == 0:
        raise TypeError("seeds must be a nonempty range")
    seed_values = tuple(seeds)
    if any(type(seed) is not int or seed < 0 for seed in seed_values):
        raise ValueError("null seeds must be nonnegative builtin integers")
    fit = fit_tail_moment(
        unit, width=width, basis_kind=basis_kind, neighbors=neighbors
    )
    tail = unit[:, width:].astype(np.float64)
    radius = np.sqrt(np.sum(tail * tail, axis=1, dtype=np.float64))
    direction = np.zeros_like(tail)
    nonzero = np.flatnonzero(radius > 0.0)
    direction[nonzero] = tail[nonzero] / radius[nonzero, None]
    denominator_by_row = _row_sufficient_statistics(
        tail, radius, fit.pairs, neighbors=neighbors
    )[1]
    denominator = float(np.sum(denominator_by_row, dtype=np.float64))
    raw_values: list[float] = []
    for seed in seed_values:
        generator = np.random.Generator(np.random.PCG64(seed))
        permuted_direction = np.zeros_like(direction)
        permuted_direction[nonzero] = direction[generator.permutation(nonzero)]
        permuted_tail = radius[:, None] * permuted_direction
        numerator_by_row = _row_sufficient_statistics(
            permuted_tail, radius, fit.pairs, neighbors=neighbors
        )[0]
        numerator = float(np.sum(numerator_by_row, dtype=np.float64))
        raw_values.append(0.0 if denominator == 0.0 else numerator / denominator)
    p_value = (
        1 + sum(value >= fit.lambda_raw for value in raw_values)
    ) / (len(raw_values) + 1)
    return TailNull(
        seeds=seed_values,
        pairs=fit.pairs,
        lambda_raw_values=tuple(raw_values),
        p_value=p_value,
    )


def _require_fp32_matrix(values: np.ndarray, *, name: str) -> None:
    if type(values) is not np.ndarray or values.dtype != np.float32:
        raise TypeError(f"{name} must be a float32 NumPy array")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def evaluate_inner_product(
    query: np.ndarray,
    gallery: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    values_per_row: int,
    fixed_bytes: int = 0,
    query_projection_seconds: float = 0.0,
    chunk_size: int = 256,
) -> ScoreView:
    """Evaluate stable inner-product retrieval through the shared metric reducer."""

    _require_fp32_matrix(query, name="query")
    _require_fp32_matrix(gallery, name="gallery")
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery dimensions differ")
    if type(values_per_row) is not int or values_per_row != gallery.shape[1]:
        raise ValueError("values_per_row must equal the gallery width")
    if type(fixed_bytes) is not int or fixed_bytes < 0:
        raise ValueError("fixed_bytes must be a nonnegative builtin integer")
    if (
        type(query_projection_seconds) is not float
        or not np.isfinite(query_projection_seconds)
        or query_projection_seconds < 0.0
    ):
        raise ValueError("query_projection_seconds must be a nonnegative builtin float")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive builtin integer")
    gallery64 = gallery.astype(np.float64)

    def score_chunks():
        for start in range(0, query.shape[0], chunk_size):
            query64 = query[start : start + chunk_size].astype(np.float64)
            yield np.ascontiguousarray(query64 @ gallery64.T)

    metrics = retrieval_metrics_from_score_chunks(
        score_chunks(), query_labels, gallery_labels
    )
    return ScoreView(
        recall=metrics.recall,
        map_at_r=metrics.map_at_r,
        top1_indices=metrics.top1_indices,
        top1_correct=metrics.top1_correct,
        values_per_row=values_per_row,
        fixed_bytes=fixed_bytes,
        total_bytes=gallery.shape[0] * values_per_row * 4 + fixed_bytes,
        query_projection_seconds=query_projection_seconds,
    )


def _evaluate_score_chunks(
    chunks: Iterable[np.ndarray],
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    values_per_row: int,
    fixed_bytes: int = 0,
    query_projection_seconds: float = 0.0,
) -> ScoreView:
    metrics = retrieval_metrics_from_score_chunks(chunks, query_labels, gallery_labels)
    return ScoreView(
        recall=metrics.recall,
        map_at_r=metrics.map_at_r,
        top1_indices=metrics.top1_indices,
        top1_correct=metrics.top1_correct,
        values_per_row=values_per_row,
        fixed_bytes=fixed_bytes,
        total_bytes=gallery_labels.shape[0] * values_per_row * 4 + fixed_bytes,
        query_projection_seconds=query_projection_seconds,
    )


def evaluate_width(
    query_native: np.ndarray,
    gallery_native: np.ndarray,
    query_pca: np.ndarray,
    gallery_pca: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    width: int,
    native_fit: TailMomentFit,
    pca_fit: TailMomentFit,
    pca_fixed_bytes: int,
    official_width: int = 512,
    pca_query_projection_seconds: float = 0.0,
) -> dict[str, ScoreView]:
    """Evaluate one registered width under every matched control."""

    for name, values in (
        ("query_native", query_native),
        ("gallery_native", gallery_native),
        ("query_pca", query_pca),
        ("gallery_pca", gallery_pca),
    ):
        _require_unit_fp32(values, name=name)
    dimension = query_native.shape[1]
    if any(values.shape[1] != dimension for values in (gallery_native, query_pca, gallery_pca)):
        raise ValueError("evaluation dimensions differ")
    if type(width) is not int or not 0 < width + 1 < dimension:
        raise ValueError("evaluation width must leave at least one tail coordinate")
    if type(official_width) is not int or not width < official_width <= dimension:
        raise ValueError("official_width must exceed width and fit the dimension")
    if type(pca_fixed_bytes) is not int or pca_fixed_bytes < 0:
        raise ValueError("pca_fixed_bytes must be a nonnegative builtin integer")

    native_query = encode_tail_moment(query_native, native_fit, basis_kind="native")
    native_gallery = encode_tail_moment(gallery_native, native_fit, basis_kind="native")
    pca_query = encode_tail_moment(query_pca, pca_fit, basis_kind="pca")
    pca_gallery = encode_tail_moment(gallery_pca, pca_fit, basis_kind="pca")
    native_prefix_query = np.ascontiguousarray(query_native[:, : width + 1])
    native_prefix_gallery = np.ascontiguousarray(gallery_native[:, : width + 1])
    renormalized_query = l2_normalize(native_prefix_query)
    renormalized_gallery = l2_normalize(native_prefix_gallery)
    zero_query = np.ascontiguousarray(
        np.column_stack((query_native[:, :width], np.zeros(query_native.shape[0], np.float32)))
    )
    zero_gallery = np.ascontiguousarray(
        np.column_stack(
            (gallery_native[:, :width], np.zeros(gallery_native.shape[0], np.float32))
        )
    )
    pca_prefix_query = l2_normalize(np.ascontiguousarray(query_pca[:, : width + 1]))
    pca_prefix_gallery = l2_normalize(np.ascontiguousarray(gallery_pca[:, : width + 1]))

    def unicom_chunks() -> Iterable[np.ndarray]:
        gallery_head = gallery_native[:, :width].astype(np.float64)
        gallery_tail = gallery_native[:, width:].astype(np.float64)
        reward = np.sum(gallery_tail * gallery_tail, axis=1, dtype=np.float64) / 2.0
        for start in range(0, query_native.shape[0], 256):
            query_head = query_native[start : start + 256, :width].astype(np.float64)
            yield np.ascontiguousarray(query_head @ gallery_head.T + reward[None, :])

    sign_gallery = native_gallery.copy()
    sign_gallery[1::2, -1] *= np.float32(-1.0)
    permuted_gallery = native_gallery.copy()
    permutation = np.random.Generator(np.random.PCG64(238)).permutation(gallery_native.shape[0])
    permuted_gallery[:, -1] = native_gallery[permutation, -1]

    def official_chunks() -> Iterable[np.ndarray]:
        query64 = query_native[:, :official_width].astype(np.float64)
        gallery64 = gallery_native[:, :official_width].astype(np.float64)
        gallery_norm = np.sum(gallery64 * gallery64, axis=1, dtype=np.float64)
        for start in range(0, query64.shape[0], 256):
            current = query64[start : start + 256]
            query_norm = np.sum(current * current, axis=1, dtype=np.float64)
            distance = query_norm[:, None] + gallery_norm[None, :] - 2.0 * (current @ gallery64.T)
            yield np.ascontiguousarray(-distance)

    return {
        "native_ctm": evaluate_inner_product(
            native_query, native_gallery, query_labels, gallery_labels,
            values_per_row=width + 1,
        ),
        "pca_ctm": evaluate_inner_product(
            pca_query, pca_gallery, query_labels, gallery_labels,
            values_per_row=width + 1, fixed_bytes=pca_fixed_bytes,
            query_projection_seconds=pca_query_projection_seconds,
        ),
        "renormalized_prefix_plus_zero": evaluate_inner_product(
            renormalized_query, renormalized_gallery, query_labels, gallery_labels,
            values_per_row=width + 1,
        ),
        "plain_prefix_plus_zero": evaluate_inner_product(
            native_prefix_query, native_prefix_gallery, query_labels, gallery_labels,
            values_per_row=width + 1,
        ),
        "lambda_zero": evaluate_inner_product(
            zero_query, zero_gallery, query_labels, gallery_labels,
            values_per_row=width + 1,
        ),
        "unicom_tail_energy": _evaluate_score_chunks(
            unicom_chunks(), query_labels, gallery_labels, values_per_row=width + 1
        ),
        "pca_renormalized_prefix_plus_zero": evaluate_inner_product(
            pca_prefix_query, pca_prefix_gallery, query_labels, gallery_labels,
            values_per_row=width + 1, fixed_bytes=pca_fixed_bytes,
            query_projection_seconds=pca_query_projection_seconds,
        ),
        "tail_sign_control": evaluate_inner_product(
            native_query, sign_gallery, query_labels, gallery_labels,
            values_per_row=width + 1,
        ),
        "tail_permuted_control": evaluate_inner_product(
            native_query, permuted_gallery, query_labels, gallery_labels,
            values_per_row=width + 1,
        ),
        "official_512": _evaluate_score_chunks(
            official_chunks(), query_labels, gallery_labels, values_per_row=official_width
        ),
        "full_width_768": evaluate_inner_product(
            query_native, gallery_native, query_labels, gallery_labels,
            values_per_row=dimension,
        ),
    }


def query_identity_interval(
    baseline_correct: np.ndarray,
    candidate_correct: np.ndarray,
    query_labels: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 205,
) -> PairedInterval:
    """Bootstrap paired R@1 differences by query identity with gallery fixed."""

    if (
        type(baseline_correct) is not np.ndarray
        or type(candidate_correct) is not np.ndarray
        or baseline_correct.dtype != np.bool_
        or candidate_correct.dtype != np.bool_
        or baseline_correct.ndim != 1
        or candidate_correct.shape != baseline_correct.shape
        or baseline_correct.size == 0
    ):
        raise TypeError("correctness arrays must be matching nonempty boolean vectors")
    labels = _require_train_labels(query_labels, rows=baseline_correct.size)
    if type(samples) is not int or samples <= 0 or type(seed) is not int:
        raise ValueError("samples and seed must be builtin integers with samples positive")
    differences = candidate_correct.astype(np.float64) - baseline_correct.astype(np.float64)
    identities = tuple(dict.fromkeys(labels))
    sums = np.asarray(
        [sum(differences[index] for index, value in enumerate(labels) if value == identity)
         for identity in identities],
        dtype=np.float64,
    )
    counts = np.asarray(
        [sum(value == identity for value in labels) for identity in identities], dtype=np.int64
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(samples, dtype=np.float64)
    for draw_index in range(samples):
        selected = generator.integers(0, len(identities), size=len(identities))
        draws[draw_index] = float(np.sum(sums[selected])) / int(np.sum(counts[selected]))
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return PairedInterval(
        point=float(np.mean(differences)),
        lower=float(lower),
        upper=float(upper),
        samples=samples,
        seed=seed,
    )


def ctm_decision(
    *,
    ctm_r1: float,
    ctm_map_at_r: float,
    renormalized_r1: float,
    renormalized_map_at_r: float,
    official_512_r1: float,
    paired_lower: float,
    control_r1: Mapping[str, float],
    ctm_total_bytes: int,
    control_total_bytes: Mapping[str, int],
    lambda_raw: float,
    lambda_lower: float,
    null_p_value: float,
    width_gains: tuple[tuple[int, float], ...],
    replication_status: str,
) -> CTMDecision:
    """Apply the frozen CTM quality, uncertainty, and storage gates."""

    scalar_values = (
        ctm_r1, ctm_map_at_r, renormalized_r1, renormalized_map_at_r,
        official_512_r1, paired_lower, lambda_raw, lambda_lower, null_p_value,
    )
    if any(type(value) is not float or not np.isfinite(value) for value in scalar_values):
        raise TypeError("decision scalars must be finite builtin floats")
    if type(control_r1) is not dict or tuple(control_r1) != MATCHED_ROW_WIDTH_CONTROLS:
        raise ValueError("quality-control order differs")
    if any(type(value) is not float or not np.isfinite(value) for value in control_r1.values()):
        raise TypeError("quality-control values must be finite builtin floats")
    if (
        type(control_total_bytes) is not dict
        or tuple(control_total_bytes) != EQUAL_TOTAL_STORAGE_CONTROLS
    ):
        raise ValueError("storage-control order differs")
    if type(ctm_total_bytes) is not int or ctm_total_bytes <= 0 or any(
        type(value) is not int or value <= 0 for value in control_total_bytes.values()
    ):
        raise TypeError("storage values must be positive builtin integers")
    if type(width_gains) is not tuple or not width_gains or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not int
        or type(item[1]) is not float
        or not np.isfinite(item[1])
        for item in width_gains
    ):
        raise TypeError("width gains must be ordered builtin int/float pairs")
    if replication_status not in {"PENDING", "PASSED", "FAILED"}:
        raise ValueError("replication status differs")

    simpler = renormalized_r1 >= official_512_r1
    r1_gain = ctm_r1 - renormalized_r1 >= 0.003 - 1e-15
    r1_lower = paired_lower > 0.0
    map_passed = renormalized_map_at_r - ctm_map_at_r <= 0.001 + 1e-15
    gap = official_512_r1 - renormalized_r1
    gap_recovery = gap > 0.0 and (ctm_r1 - renormalized_r1) / gap >= 0.5 - 1e-15
    controls_passed = all(ctm_r1 > value for value in control_r1.values())
    storage_passed = all(value == ctm_total_bytes for value in control_total_bytes.values())
    coefficient_passed = lambda_raw > 0.0 and lambda_lower > 0.0 and null_p_value <= 0.05
    width_signal = any(gain >= 0.001 - 1e-15 for _, gain in width_gains)
    replication_passed = replication_status == "PASSED"
    gates = (
        r1_gain, r1_lower, map_passed, gap_recovery, controls_passed,
        storage_passed, coefficient_passed, width_signal,
    )
    if simpler:
        status, reason = "USE_RENORMALIZED", "renormalized descriptor matches official anchor"
    elif not all(gates) or replication_status == "FAILED":
        status, reason = "CLOSE", "one or more registered falsifiers failed"
    elif replication_status == "PENDING":
        status, reason = "REPLICATE", "In-Shop gates passed; Cars196 is pending"
    else:
        status, reason = "GENERAL_CLAIM_READY", "all local and replication gates passed"
    return CTMDecision(
        status=status,
        reason=reason,
        r1_gain_passed=r1_gain,
        r1_lower_passed=r1_lower,
        map_passed=map_passed,
        gap_recovery_passed=gap_recovery,
        quality_controls_passed=controls_passed,
        equal_storage_passed=storage_passed,
        coefficient_passed=coefficient_passed,
        some_width_signal_passed=width_signal,
        replication_passed=replication_passed,
    )
