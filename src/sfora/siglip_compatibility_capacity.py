"""Descriptor-only capacity diagnostics for SigLIP gallery compatibility."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _validated_pair(
    student: torch.Tensor, teacher: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        type(student) is not torch.Tensor
        or type(teacher) is not torch.Tensor
        or student.device.type != "cpu"
        or teacher.device.type != "cpu"
        or student.dtype != torch.float32
        or teacher.dtype != torch.float32
        or student.ndim != 2
        or teacher.ndim != 2
        or student.shape != teacher.shape
        or student.shape[0] < 2
        or student.shape[1] < 2
        or not bool(torch.isfinite(student).all())
        or not bool(torch.isfinite(teacher).all())
        or bool((torch.linalg.vector_norm(student, dim=1) <= 0).any())
        or bool((torch.linalg.vector_norm(teacher, dim=1) <= 0).any())
    ):
        raise ValueError("compatibility descriptor authority differs")
    return student, teacher


def compatibility_folds(labels: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return the registered three-way class folds."""

    if (
        type(labels) is not tuple
        or len(labels) != 39
        or any(type(label) is not int for label in labels)
        or set(labels) != set(range(39))
    ):
        raise ValueError("compatibility class authority differs")
    ranked = sorted(
        (
            hashlib.sha256(
                b"sfora-compatibility-capacity-fold-v1\0" + str(label).encode("ascii")
            ).digest(),
            label,
        )
        for label in labels
    )
    return tuple(
        tuple(label for _, label in ranked[start : start + 13])
        for start in range(0, 39, 13)
    )


@dataclass(frozen=True, slots=True)
class AffineMap:
    """A validated FP64 affine descriptor map with normalized FP32 output."""

    weight: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.weight) is not torch.Tensor
            or type(self.bias) is not torch.Tensor
            or self.weight.device.type != "cpu"
            or self.bias.device.type != "cpu"
            or self.weight.dtype != torch.float64
            or self.bias.dtype != torch.float64
            or self.weight.ndim != 2
            or self.weight.shape[0] != self.weight.shape[1]
            or self.bias.shape != (self.weight.shape[0],)
            or not bool(torch.isfinite(self.weight).all())
            or not bool(torch.isfinite(self.bias).all())
        ):
            raise ValueError("compatibility affine authority differs")

    def apply(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Apply this map and return normalized CPU FP32 descriptors."""

        if (
            type(descriptors) is not torch.Tensor
            or descriptors.device.type != "cpu"
            or descriptors.dtype != torch.float32
            or descriptors.ndim != 2
            or descriptors.shape[0] < 2
            or descriptors.shape[1] != self.weight.shape[0]
            or not bool(torch.isfinite(descriptors).all())
            or bool((torch.linalg.vector_norm(descriptors, dim=1) <= 0).any())
        ):
            raise ValueError("compatibility descriptor authority differs")
        mapped = F.normalize(descriptors.double() @ self.weight + self.bias, dim=1)
        if not bool(torch.isfinite(mapped).all()):
            raise ValueError("compatibility affine authority differs")
        return mapped.float().contiguous()


def fit_centered_similarity(student: torch.Tensor, teacher: torch.Tensor) -> AffineMap:
    """Fit the registered centered orthogonal descriptor map."""

    student, teacher = _validated_pair(student, teacher)
    source = student.double()
    target = teacher.double()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    left, _, right_t = torch.linalg.svd(
        (source - source_mean).T @ (target - target_mean), full_matrices=False
    )
    weight = (left @ right_t).contiguous()
    bias = (target_mean - source_mean @ weight).contiguous()
    return AffineMap(weight, bias)


def fit_regularized_affine(
    student: torch.Tensor, teacher: torch.Tensor, regularization: float
) -> AffineMap:
    """Fit the registered identity-regularized affine ridge map."""

    student, teacher = _validated_pair(student, teacher)
    if type(regularization) is not float or regularization not in {1e-4, 1e-2, 1.0}:
        raise ValueError("compatibility regularization differs")
    source = student.double()
    target = teacher.double()
    rows, dimensions = source.shape
    augmented = torch.cat((source, torch.ones(rows, 1, dtype=torch.float64)), dim=1)
    identity_target = torch.cat(
        (torch.eye(dimensions, dtype=torch.float64), torch.zeros(1, dimensions)), dim=0
    )
    system = augmented.T @ augmented / rows
    system += regularization * torch.eye(dimensions + 1, dtype=torch.float64)
    right = augmented.T @ target / rows + regularization * identity_target
    solution = torch.linalg.solve(system, right)
    return AffineMap(solution[:-1].contiguous(), solution[-1].contiguous())
