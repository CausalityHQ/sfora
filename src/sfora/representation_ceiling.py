"""Deterministic train-only primitives for representation-ceiling diagnosis."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import cast

import torch


def _float32_matrix(value: object, *, rows: int = 2) -> bool:
    return bool(
        type(value) is torch.Tensor
        and value.device.type == "cpu"
        and value.dtype == torch.float32
        and value.ndim == 2
        and value.shape[0] >= rows
        and value.shape[1] >= 2
        and value.is_contiguous()
        and torch.isfinite(value).all()
    )


@dataclass(frozen=True)
class ClassDisjointPartition:
    """Ordered row and class identities for one deterministic class split."""

    fit_row_indexes: tuple[int, ...]
    validation_row_indexes: tuple[int, ...]
    fit_class_ids: tuple[int, ...]
    validation_class_ids: tuple[int, ...]


def deterministic_class_partition(
    labels: tuple[int, ...], *, fit_fraction: float, seed: int
) -> ClassDisjointPartition:
    """Split complete concrete integer classes using a stable SHA-256 ordering."""

    if (
        type(labels) is not tuple
        or len(labels) < 3
        or any(type(label) is not int or not -(2**63) <= label < 2**63 for label in labels)
        or type(fit_fraction) is not float
        or not math.isfinite(fit_fraction)
        or not 0.0 < fit_fraction < 1.0
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("class partition authority differs")
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if len(counts) < 2:
        raise ValueError("class partition authority differs")

    seed_bytes = seed.to_bytes(8, byteorder="little", signed=False)
    ordered_classes = sorted(
        counts,
        key=lambda label: (
            hashlib.sha256(seed_bytes + label.to_bytes(8, "little", signed=True)).digest(),
            label,
        ),
    )
    fit_class_count = int(math.floor(len(ordered_classes) * fit_fraction + 0.5))
    if not 0 < fit_class_count < len(ordered_classes):
        raise ValueError("class partition authority differs")
    fit_set = frozenset(ordered_classes[:fit_class_count])
    validation_set = set(ordered_classes) - fit_set
    if any(counts[label] < 2 for label in validation_set):
        raise ValueError("class partition authority differs")
    fit_rows = tuple(index for index, label in enumerate(labels) if label in fit_set)
    validation_rows = tuple(index for index, label in enumerate(labels) if label not in fit_set)
    return ClassDisjointPartition(
        fit_row_indexes=fit_rows,
        validation_row_indexes=validation_rows,
        fit_class_ids=tuple(sorted(fit_set)),
        validation_class_ids=tuple(sorted(validation_set)),
    )


@dataclass(frozen=True, init=False)
class CenteredPcaTransform:
    """A train-fitted centered PCA projection with deterministic component signs."""

    _mean: torch.Tensor
    _components: torch.Tensor

    def __init__(self, *, mean: torch.Tensor, components: torch.Tensor) -> None:
        if (
            type(mean) is not torch.Tensor
            or mean.device.type != "cpu"
            or mean.dtype != torch.float32
            or mean.ndim != 1
            or not mean.is_contiguous()
            or not bool(torch.isfinite(mean).all())
            or not _float32_matrix(components, rows=1)
            or components.shape[1] != len(mean)
        ):
            raise ValueError("PCA authority differs")
        object.__setattr__(self, "_mean", mean.detach().clone())
        object.__setattr__(self, "_components", components.detach().clone())

    @property
    def mean(self) -> torch.Tensor:
        """Return a defensive copy of the fitted mean."""

        return self._mean.clone()

    @property
    def components(self) -> torch.Tensor:
        """Return a defensive copy of the fitted components."""

        return self._components.clone()

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        """Center, project, and row-normalize a compatible CPU float32 matrix."""

        if not _float32_matrix(value, rows=1) or value.shape[1] != len(self._mean):
            raise ValueError("PCA authority differs")
        projected = (value.double() - self._mean.double()) @ self._components.double().T
        return _normalize_float64_rows(projected, message="PCA authority differs")


def fit_centered_pca(value: torch.Tensor, *, dimensions: int) -> CenteredPcaTransform:
    """Fit deterministic centered PCA using a float64 full-matrix SVD."""

    if (
        not _float32_matrix(value)
        or type(dimensions) is not int
        or not 1 <= dimensions <= min(value.shape[0] - 1, value.shape[1])
    ):
        raise ValueError("PCA authority differs")
    double = value.double()
    mean64 = double.mean(dim=0)
    centered = double - mean64
    _left, singular, right = torch.linalg.svd(centered, full_matrices=False)
    rank_tolerance = (
        max(value.shape) * torch.finfo(torch.float64).eps * max(float(singular[0]), 1.0)
    )
    if (
        len(singular) < dimensions
        or not bool(torch.isfinite(singular).all())
        or float(singular[dimensions - 1]) <= rank_tolerance
    ):
        raise ValueError("PCA authority differs")
    retained_boundary = min(dimensions + 1, len(singular))
    degeneracy_tolerance = rank_tolerance
    if any(
        abs(float(singular[index] - singular[index + 1])) <= degeneracy_tolerance
        for index in range(retained_boundary - 1)
    ):
        raise ValueError("PCA authority differs")
    components64 = right[:dimensions].clone()
    for component in components64:
        pivot = int(torch.argmax(torch.abs(component)))
        if float(component[pivot]) < 0.0:
            component.neg_()
    return CenteredPcaTransform(
        mean=mean64.float().contiguous(),
        components=components64.float().contiguous(),
    )


@dataclass(frozen=True, init=False)
class AffineMap:
    """A strict CPU float32 affine map using output-by-input weights."""

    _weight: torch.Tensor
    _bias: torch.Tensor

    def __init__(self, *, weight: torch.Tensor, bias: torch.Tensor) -> None:
        if (
            not _float32_matrix(weight, rows=1)
            or type(bias) is not torch.Tensor
            or bias.device.type != "cpu"
            or bias.dtype != torch.float32
            or bias.ndim != 1
            or len(bias) != weight.shape[0]
            or not bias.is_contiguous()
            or not bool(torch.isfinite(bias).all())
        ):
            raise ValueError("affine authority differs")
        object.__setattr__(self, "_weight", weight.detach().clone())
        object.__setattr__(self, "_bias", bias.detach().clone())

    @property
    def weight(self) -> torch.Tensor:
        """Return a defensive copy of the fitted weight."""

        return self._weight.clone()

    @property
    def bias(self) -> torch.Tensor:
        """Return a defensive copy of the fitted bias."""

        return self._bias.clone()

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the affine map without normalizing its output."""

        if not _float32_matrix(value, rows=1) or value.shape[1] != self._weight.shape[1]:
            raise ValueError("affine authority differs")
        output = value.double() @ self._weight.double().T + self._bias.double()
        if not bool(torch.isfinite(output).all()):
            raise ValueError("affine authority differs")
        output32 = output.float()
        if not bool(torch.isfinite(output32).all()):
            raise ValueError("affine authority differs")
        return output32


def fit_ridge_affine(source: torch.Tensor, target: torch.Tensor, *, penalty: float) -> AffineMap:
    """Fit a float64 ridge affine map while leaving the intercept unpenalized."""

    if (
        not _float32_matrix(source)
        or not _float32_matrix(target)
        or source.shape[0] != target.shape[0]
        or type(penalty) is not float
        or not math.isfinite(penalty)
        or penalty <= 0.0
    ):
        raise ValueError("ridge authority differs")
    source64 = source.double()
    target64 = target.double()
    augmented = torch.cat((source64, torch.ones((len(source64), 1), dtype=torch.float64)), dim=1)
    centered_source = source64 - source64.mean(dim=0)
    feature_energy = float(torch.sum(centered_source * centered_source)) / source.shape[1]
    if not math.isfinite(feature_energy) or feature_energy <= torch.finfo(torch.float64).eps:
        raise ValueError("ridge authority differs")
    regularizer = torch.eye(augmented.shape[1], dtype=torch.float64) * penalty * feature_energy
    regularizer[-1, -1] = 0.0
    normal = augmented.T @ augmented + regularizer
    rhs = augmented.T @ target64
    try:
        coefficients = torch.linalg.solve(normal, rhs)
    except RuntimeError as error:
        raise ValueError("ridge authority differs") from error
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("ridge authority differs")
    return AffineMap(
        weight=coefficients[:-1].T.float().contiguous(),
        bias=coefficients[-1].float().contiguous(),
    )


def apply_normalized_affine(value: torch.Tensor, affine: AffineMap) -> torch.Tensor:
    """Apply a strict affine map and normalize each finite nonzero output row."""

    if type(affine) is not AffineMap:
        raise ValueError("affine authority differs")
    if not _float32_matrix(value, rows=1) or value.shape[1] != affine._weight.shape[1]:
        raise ValueError("affine authority differs")
    output = value.double() @ affine._weight.double().T + affine._bias.double()
    return _normalize_float64_rows(output, message="affine authority differs")


def _normalize_float64_rows(value: torch.Tensor, *, message: str) -> torch.Tensor:
    norms = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
        raise ValueError(message)
    normalized = (value / norms).float()
    output_norms = torch.linalg.vector_norm(normalized.double(), dim=1)
    if (
        not bool(torch.isfinite(normalized).all())
        or bool((output_norms < 1.0 - 2e-6).any())
        or bool((output_norms > 1.0 + 2e-6).any())
    ):
        raise ValueError(message)
    return cast(torch.Tensor, normalized.contiguous())
