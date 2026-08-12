"""Pure NumPy calibrated-tail-moment retrieval arithmetic."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
