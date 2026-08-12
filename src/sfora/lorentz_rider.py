"""Pure NumPy arithmetic for the Lorentz compression falsifier ladder."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeltaEstimate:
    delta: float
    diameter: float
    relative: float
    base: int


def _require_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name} must be a float32 or float64 NumPy array")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix with at least two rows")
    if not values.flags.c_contiguous or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite and C-contiguous")
    return values


def l0_subsample_indices(train_count: int, dataset_index: int, replicate: int) -> np.ndarray:
    """Return one registered sorted 2,000-row L0 train subsample."""

    if type(train_count) is not int or train_count < 2_000:
        raise ValueError("train_count must be a builtin integer at least 2000")
    if type(dataset_index) is not int or dataset_index not in {0, 1, 2}:
        raise ValueError("dataset_index must be 0, 1, or 2")
    if type(replicate) is not int or not 0 <= replicate < 10:
        raise ValueError("replicate must be a builtin integer in [0, 10)")
    seed = 7_000 + 100 * dataset_index + replicate
    generator = np.random.Generator(np.random.PCG64(seed))
    return np.sort(generator.choice(train_count, size=2_000, replace=False)).astype(
        np.int64, copy=False
    )


def pairwise_chord_distances(values: np.ndarray) -> np.ndarray:
    """Compute ordered float64 Euclidean distances for one row matrix."""

    matrix = _require_matrix(values, name="values").astype(np.float64)
    squared_norm = np.sum(matrix * matrix, axis=1, dtype=np.float64)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (matrix @ matrix.T)
    np.maximum(squared, 0.0, out=squared)
    result = np.sqrt(squared, out=squared)
    np.fill_diagonal(result, 0.0)
    return np.ascontiguousarray(result)


def _require_distances(distances: np.ndarray) -> np.ndarray:
    value = _require_matrix(distances, name="distances")
    if value.shape[0] != value.shape[1]:
        raise ValueError("distances must be square")
    matrix = value.astype(np.float64, copy=False)
    if np.any(matrix < 0.0) or not np.array_equal(np.diag(matrix), np.zeros(matrix.shape[0])):
        raise ValueError("distances must be nonnegative with an exact zero diagonal")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("distances must be symmetric")
    return matrix


def medoid_index(distances: np.ndarray) -> int:
    """Choose the minimum row-sum point, preserving lowest-index ties."""

    matrix = _require_distances(distances)
    return int(np.argmin(np.sum(matrix, axis=1, dtype=np.float64)))


def delta_bruteforce(distances: np.ndarray) -> float:
    """Compute exact four-point delta for at most 32 rows."""

    matrix = _require_distances(distances)
    if matrix.shape[0] > 32:
        raise ValueError("bruteforce delta is restricted to at most 32 rows")
    maximum = 0.0
    for a, b, c, d in itertools.combinations(range(matrix.shape[0]), 4):
        sums = sorted(
            (
                matrix[a, b] + matrix[c, d],
                matrix[a, c] + matrix[b, d],
                matrix[a, d] + matrix[b, c],
            )
        )
        maximum = max(maximum, float(sums[2] - sums[1]) / 2.0)
    return maximum


def gromov_delta_rel(distances: np.ndarray, base: int) -> DeltaEstimate:
    """Compute the exact fixed-base four-point estimate and diameter scaling."""

    matrix = _require_distances(distances)
    if type(base) is not int or not 0 <= base < matrix.shape[0]:
        raise ValueError("base must be a valid builtin row index")
    diameter = float(np.max(matrix))
    if diameter == 0.0:
        raise ValueError("distance diameter must be positive")
    product = (matrix[:, base, None] + matrix[base, None, :] - matrix) / 2.0
    closure = np.full_like(product, -np.inf)
    for k in range(matrix.shape[0]):
        np.maximum(closure, np.minimum(product[:, k, None], product[k, None, :]), out=closure)
    delta = float(np.max(closure - product))
    return DeltaEstimate(delta=delta, diameter=diameter, relative=2.0 * delta / diameter, base=base)


def column_permutation_null(values: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute every coordinate while preserving its values."""

    matrix = _require_matrix(values, name="values").astype(np.float32)
    if type(seed) is not int:
        raise TypeError("seed must be a builtin integer")
    generator = np.random.Generator(np.random.PCG64(seed))
    result = np.empty_like(matrix)
    for column in range(matrix.shape[1]):
        result[:, column] = matrix[generator.permutation(matrix.shape[0]), column]
    return np.ascontiguousarray(result)


def spectrum_gaussian_null(values: np.ndarray, seed: int) -> np.ndarray:
    """Draw a Gaussian null with the observed mean and covariance spectrum."""

    matrix = _require_matrix(values, name="values")
    if matrix.shape[0] <= matrix.shape[1]:
        raise ValueError("Gaussian null requires more rows than coordinates")
    if type(seed) is not int:
        raise TypeError("seed must be a builtin integer")
    values64 = matrix.astype(np.float64)
    mean = np.mean(values64, axis=0, dtype=np.float64)
    centered = values64 - mean
    covariance = (centered.T @ centered) / matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    for column in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[pivot, column] < 0.0:
            eigenvectors[:, column] *= -1.0
    generator = np.random.Generator(np.random.PCG64(seed))
    scores = generator.standard_normal(size=matrix.shape)
    scores -= np.mean(scores, axis=0, dtype=np.float64)
    orthonormal, _ = np.linalg.qr(scores, mode="reduced")
    result = mean + (np.sqrt(matrix.shape[0]) * orthonormal * np.sqrt(eigenvalues)) @ eigenvectors.T
    return np.ascontiguousarray(result.astype(np.float32))
