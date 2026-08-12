"""Pure NumPy calibrated-tail-moment retrieval arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfora.unicom_retrieval_audit import l2_normalize


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
