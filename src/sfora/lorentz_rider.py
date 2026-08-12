"""Pure NumPy arithmetic for the Lorentz compression falsifier ladder."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeltaEstimate:
    delta: float
    diameter: float
    relative: float
    base: int


@dataclass(frozen=True, eq=False)
class FrozenPCA:
    mean: np.ndarray
    components: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrozenPCA):
            return NotImplemented
        return np.array_equal(self.mean, other.mean) and np.array_equal(
            self.components, other.components
        )


@dataclass(frozen=True)
class MaxInterval:
    point: float
    lower: float
    upper: float
    standard_errors: float
    samples: int
    seed: int
    replicates: tuple[float, ...]
    replicates_sha256: str


@dataclass(frozen=True)
class L1Decision:
    status: str
    endpoint_passed: bool
    spatial_passed: bool
    power_passed: bool


def _require_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name} must be a float32 or float64 NumPy array")
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix with at least two rows")
    if not values.flags.c_contiguous or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite and C-contiguous")
    return values


def _require_nonempty_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.dtype not in (np.float32, np.float64):
        raise TypeError(f"{name} must be a float32 or float64 NumPy array")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
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


def _canonicalize_component_signs(components: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(components.copy())
    for row in range(result.shape[0]):
        pivot = int(np.argmax(np.abs(result[row])))
        if result[row, pivot] < 0.0:
            result[row] *= -1.0
    return result


def fit_frozen_pca(train: np.ndarray, dimension: int) -> FrozenPCA:
    """Fit one deterministic PCA basis using train rows only."""

    matrix = _require_matrix(train, name="train")
    if type(dimension) is not int or not 0 < dimension <= matrix.shape[1]:
        raise ValueError("dimension must be a positive builtin integer within input width")
    values64 = matrix.astype(np.float64)
    mean64 = np.mean(values64, axis=0, dtype=np.float64)
    centered = values64 - mean64
    covariance = (centered.T @ centered) / matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")[:dimension]
    components64 = _canonicalize_component_signs(eigenvectors[:, order].T)
    return FrozenPCA(
        mean=np.ascontiguousarray(mean64.astype(np.float32)),
        components=np.ascontiguousarray(components64.astype(np.float32)),
    )


def apply_frozen_pca(values: np.ndarray, fit: FrozenPCA) -> np.ndarray:
    """Apply a frozen PCA mean and component matrix without refitting."""

    matrix = _require_nonempty_matrix(values, name="values")
    if type(fit) is not FrozenPCA:
        raise TypeError("fit must be a FrozenPCA")
    if (
        type(fit.mean) is not np.ndarray
        or fit.mean.dtype != np.float32
        or fit.mean.shape != (matrix.shape[1],)
        or type(fit.components) is not np.ndarray
        or fit.components.dtype != np.float32
        or fit.components.ndim != 2
        or fit.components.shape[1] != matrix.shape[1]
        or not fit.mean.flags.c_contiguous
        or not fit.components.flags.c_contiguous
        or not np.isfinite(fit.mean).all()
        or not np.isfinite(fit.components).all()
    ):
        raise ValueError("frozen PCA schema differs")
    centered = matrix.astype(np.float32) - fit.mean
    return np.ascontiguousarray(centered @ fit.components.T)


def scale_for_target_median(train_projected: np.ndarray, target: float) -> float:
    """Choose the scale that maps the median train radius to one fixed target."""

    matrix = _require_matrix(train_projected, name="train_projected")
    if type(target) is not float or not np.isfinite(target) or target <= 0.0:
        raise ValueError("target must be a positive finite builtin float")
    radii = np.sqrt(np.sum(matrix.astype(np.float64) ** 2, axis=1, dtype=np.float64))
    median = float(np.median(radii))
    if median <= 0.0:
        raise ValueError("train projected median radius must be positive")
    return target / median


def lorentz_lift(values: np.ndarray, scale: float, clip: float = 2.5) -> np.ndarray:
    """Lift one Euclidean matrix to the unit Lorentz hyperboloid in FP32."""

    matrix = _require_nonempty_matrix(values, name="values").astype(np.float32)
    if type(scale) is not float or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive finite builtin float")
    if type(clip) is not float or not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("clip must be a positive finite builtin float")
    radius64 = np.sqrt(np.sum(matrix.astype(np.float64) ** 2, axis=1, dtype=np.float64))
    lifted_radius = np.minimum(scale * radius64, clip).astype(np.float32)
    direction = np.zeros_like(matrix)
    nonzero = np.flatnonzero(radius64 > 0.0)
    direction[nonzero] = matrix[nonzero] / radius64[nonzero, None].astype(np.float32)
    result = np.column_stack(
        (
            np.cosh(lifted_radius),
            np.sinh(lifted_radius)[:, None] * direction,
        )
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Lorentz lift produced nonfinite values")
    return np.ascontiguousarray(result)


def _require_lorentz(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = _require_nonempty_matrix(values, name=name)
    if matrix.dtype != np.float32 or matrix.shape[1] < 2:
        raise TypeError(f"{name} must be a float32 Lorentz matrix")
    if np.any(matrix[:, 0] < 1.0):
        raise ValueError(f"{name} time coordinates differ")
    constraint = -(matrix[:, 0].astype(np.float64) ** 2) + np.sum(
        matrix[:, 1:].astype(np.float64) ** 2, axis=1, dtype=np.float64
    )
    if np.any(np.abs(constraint + 1.0) > 3e-5):
        raise ValueError(f"{name} hyperboloid constraint differs")
    return matrix


def lorentz_mips_scores(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Score nearest Lorentz neighbors with one ordinary FP32 inner product."""

    query_value = _require_lorentz(query, name="query")
    gallery_value = _require_lorentz(gallery, name="gallery")
    if query_value.shape[1] != gallery_value.shape[1]:
        raise ValueError("Lorentz dimensions differ")
    transformed = query_value.copy()
    transformed[:, 0] *= np.float32(-1.0)
    return np.ascontiguousarray(transformed @ gallery_value.T)


def lorentz_distance_block(
    query: np.ndarray, gallery: np.ndarray, *, gallery_chunk_size: int = 256
) -> np.ndarray:
    """Compute stable unit-curvature Lorentz distances in FP32."""

    query_value = _require_lorentz(query, name="query")
    gallery_value = _require_lorentz(gallery, name="gallery")
    if query_value.shape[1] != gallery_value.shape[1]:
        raise ValueError("Lorentz dimensions differ")
    if type(gallery_chunk_size) is not int or gallery_chunk_size <= 0:
        raise ValueError("gallery_chunk_size must be a positive builtin integer")
    result = np.empty((query_value.shape[0], gallery_value.shape[0]), dtype=np.float32)
    for start in range(0, gallery_value.shape[0], gallery_chunk_size):
        stop = min(start + gallery_chunk_size, gallery_value.shape[0])
        gallery_chunk = gallery_value[start:stop]
        spatial_difference = query_value[:, None, 1:] - gallery_chunk[None, :, 1:]
        time_difference = query_value[:, None, 0] - gallery_chunk[None, :, 0]
        value = (
            np.sum(
                spatial_difference * spatial_difference,
                axis=2,
                dtype=np.float32,
            )
            - time_difference * time_difference
        ) / np.float32(2.0)
        if float(np.min(value)) < -2e-5:
            raise ValueError("Lorentz distance intermediate is materially negative")
        np.maximum(value, np.float32(0.0), out=value)
        result[:, start:stop] = np.log1p(value + np.sqrt(value * (value + np.float32(2.0))))
    return np.ascontiguousarray(result)


def score_control_blocks(
    query_projected: np.ndarray,
    gallery_projected: np.ndarray,
    scale: float,
    clip: float,
) -> dict[str, Iterable[np.ndarray]]:
    """Return the exact ordered L1 scorer and matched radial controls."""

    query = _require_nonempty_matrix(query_projected, name="query_projected").astype(np.float32)
    gallery = _require_nonempty_matrix(gallery_projected, name="gallery_projected").astype(
        np.float32
    )
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("query and gallery projected dimensions differ")
    if type(scale) is not float or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive finite builtin float")
    if type(clip) is not float or not np.isfinite(clip) or clip <= 0.0:
        raise ValueError("clip must be a positive finite builtin float")

    gallery64 = gallery.astype(np.float64)
    gallery_norm64 = np.sqrt(np.sum(gallery64 * gallery64, axis=1, dtype=np.float64))
    gallery_norm = gallery_norm64.astype(np.float32)
    gallery_direction = np.zeros_like(gallery)
    gallery_nonzero = gallery_norm > 0.0
    gallery_direction[gallery_nonzero] = (
        gallery[gallery_nonzero] / gallery_norm[gallery_nonzero, None]
    )
    gallery_radius = np.minimum(scale * gallery_norm64, clip).astype(np.float32)
    gallery_sinh = np.sinh(gallery_radius)
    gallery_lifted = lorentz_lift(gallery, scale, clip)
    gallery_squared = np.sum(gallery * gallery, axis=1, dtype=np.float32)

    def chunks(control: str) -> Iterable[np.ndarray]:
        for start in range(0, query.shape[0], 256):
            query_chunk = query[start : start + 256]
            query64 = query_chunk.astype(np.float64)
            query_norm64 = np.sqrt(np.sum(query64 * query64, axis=1, dtype=np.float64))
            query_norm = query_norm64.astype(np.float32)
            query_direction = np.zeros_like(query_chunk)
            nonzero = query_norm > 0.0
            query_direction[nonzero] = query_chunk[nonzero] / query_norm[nonzero, None]
            cosine = query_direction @ gallery_direction.T
            query_radius = np.minimum(scale * query_norm64, clip).astype(np.float32)
            if control == "lorentz":
                query_lifted = lorentz_lift(query_chunk, scale, clip)
                score = lorentz_mips_scores(query_lifted, gallery_lifted) / query_lifted[:, :1]
            elif control == "pca_euclidean":
                score = -(
                    np.sum(
                        query_chunk * query_chunk,
                        axis=1,
                        dtype=np.float32,
                    )[:, None]
                    + gallery_squared[None, :]
                    - np.float32(2.0) * (query_chunk @ gallery.T)
                )
            elif control == "pca_cosine":
                score = cosine
            elif control == "spatial_only":
                score = (
                    np.sinh(query_radius)[:, None] * gallery_sinh[None, :] * cosine
                    - np.float32(0.5) * gallery_sinh[None, :] ** 2
                )
            else:
                power = 1 if control == "power_1" else 3
                query_h = query_radius**power
                gallery_h = gallery_radius**power
                score = query_h[:, None] / np.sqrt(
                    np.float32(1.0) + query_h[:, None] ** 2
                ) * gallery_h[None, :] * cosine - np.sqrt(np.float32(1.0) + gallery_h[None, :] ** 2)
            yield np.ascontiguousarray(score.astype(np.float32, copy=False))

    return {
        name: chunks(name)
        for name in (
            "lorentz",
            "pca_euclidean",
            "pca_cosine",
            "spatial_only",
            "power_1",
            "power_3",
        )
    }


def paired_identity_max_interval(
    baseline_correct: np.ndarray,
    interior_correct: np.ndarray,
    labels: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 205,
) -> MaxInterval:
    """Bootstrap the selected interior gain by resampling whole identities."""

    if (
        type(baseline_correct) is not np.ndarray
        or baseline_correct.dtype != np.bool_
        or baseline_correct.ndim != 1
        or baseline_correct.size == 0
        or type(interior_correct) is not np.ndarray
        or interior_correct.dtype != np.bool_
        or interior_correct.ndim != 2
        or interior_correct.shape[1] != baseline_correct.size
        or interior_correct.shape[0] == 0
    ):
        raise TypeError("correctness arrays must be matching nonempty boolean arrays")
    if (
        type(labels) is not np.ndarray
        or labels.ndim != 1
        or labels.shape != baseline_correct.shape
        or labels.dtype.kind not in {"U", "S"}
    ):
        raise TypeError("labels must be a matching string vector")
    if type(samples) is not int or samples <= 0 or type(seed) is not int:
        raise ValueError("samples and seed must be builtin integers with samples positive")

    label_rows: dict[str, list[int]] = {}
    for row, label in enumerate(labels.tolist()):
        label_rows.setdefault(label, []).append(row)
    groups = tuple(np.asarray(rows, dtype=np.int64) for rows in label_rows.values())
    counts = np.asarray([group.size for group in groups], dtype=np.float64)
    baseline_sums = np.asarray(
        [np.sum(baseline_correct[group], dtype=np.int64) for group in groups],
        dtype=np.float64,
    )
    interior_sums = np.asarray(
        [[np.sum(row[group], dtype=np.int64) for group in groups] for row in interior_correct],
        dtype=np.float64,
    )
    point = float(
        np.max(
            np.sum(interior_sums, axis=1, dtype=np.float64) / np.sum(counts)
            - np.sum(baseline_sums, dtype=np.float64) / np.sum(counts)
        )
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    replicates = np.empty(samples, dtype=np.float64)
    chunk_size = 64
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        selected = generator.integers(0, len(groups), size=(stop - start, len(groups)))
        denominator = np.sum(counts[selected], axis=1, dtype=np.float64)
        baseline_mean = np.sum(baseline_sums[selected], axis=1, dtype=np.float64) / denominator
        interior_mean = (
            np.sum(interior_sums[:, selected], axis=2, dtype=np.float64) / denominator[None, :]
        )
        replicates[start:stop] = np.max(interior_mean - baseline_mean[None, :], axis=0)
    lower, upper = np.percentile(replicates, [2.5, 97.5])
    deviation = float(np.std(replicates, ddof=1)) if samples > 1 else 0.0
    standard_errors = point / deviation if deviation > 0.0 else 0.0
    return MaxInterval(
        point=point,
        lower=float(lower),
        upper=float(upper),
        standard_errors=standard_errors,
        samples=samples,
        seed=seed,
        replicates=tuple(float(value) for value in replicates),
        replicates_sha256=hashlib.sha256(replicates.tobytes(order="C")).hexdigest(),
    )


def l1_decision(
    *,
    endpoint_gain: float,
    endpoint_lower: float,
    standard_errors: float,
    spatial_gain: float,
    power_gain: float,
) -> L1Decision:
    """Apply the one-dataset attribution gate without cross-dataset promotion."""

    values = (
        endpoint_gain,
        endpoint_lower,
        standard_errors,
        spatial_gain,
        power_gain,
    )
    if any(type(value) is not float or not np.isfinite(value) for value in values):
        raise TypeError("L1 decision inputs must be finite builtin floats")
    endpoint_passed = endpoint_gain > 0.0 and endpoint_lower > 0.0 and standard_errors >= 3.0
    spatial_passed = spatial_gain > 0.0
    power_passed = power_gain > 0.0
    return L1Decision(
        status=(
            "GEOMETRY_SURVIVES"
            if endpoint_passed and spatial_passed and power_passed
            else "CLOSE_GEOMETRY"
        ),
        endpoint_passed=endpoint_passed,
        spatial_passed=spatial_passed,
        power_passed=power_passed,
    )
