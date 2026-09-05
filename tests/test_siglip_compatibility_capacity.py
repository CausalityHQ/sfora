from __future__ import annotations

import pytest
import torch

from sfora.siglip_compatibility_capacity import (
    AffineMap,
    compatibility_folds,
    fit_centered_similarity,
    fit_regularized_affine,
)


def _paired_descriptors(rows: int = 96, dimensions: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(71)
    student = torch.nn.functional.normalize(
        torch.randn(rows, dimensions, generator=generator), dim=1
    )
    weight = torch.eye(dimensions, dtype=torch.float64)
    weight[0, 1] = 0.2
    bias = torch.linspace(-0.03, 0.03, dimensions, dtype=torch.float64)
    teacher = torch.nn.functional.normalize(student.double() @ weight + bias, dim=1).float()
    return student, teacher


def test_compatibility_folds_are_exact_deterministic_class_partitions() -> None:
    folds = compatibility_folds(tuple(range(39)))
    assert folds == (
        (19, 13, 4, 17, 2, 9, 35, 18, 33, 0, 29, 7, 23),
        (28, 20, 16, 8, 1, 27, 10, 12, 30, 31, 11, 37, 36),
        (22, 24, 21, 15, 34, 3, 32, 25, 26, 38, 6, 14, 5),
    )
    assert set().union(*(set(fold) for fold in folds)) == set(range(39))


@pytest.mark.parametrize(
    "labels",
    [tuple(range(38)), tuple(range(1, 40)), tuple([0] * 39), list(range(39))],
)
def test_compatibility_folds_reject_authority_drift(labels: object) -> None:
    with pytest.raises(ValueError, match="compatibility class authority differs"):
        compatibility_folds(labels)  # type: ignore[arg-type]


def test_centered_similarity_recovers_rotation_and_finite_centering() -> None:
    generator = torch.Generator().manual_seed(31)
    student = torch.nn.functional.normalize(torch.randn(128, 10, generator=generator), dim=1)
    rotation, _ = torch.linalg.qr(torch.randn(10, 10, generator=generator).double())
    teacher = torch.nn.functional.normalize(student.double() @ rotation, dim=1).float()

    fitted = fit_centered_similarity(student, teacher)
    mapped = fitted.apply(student)

    assert isinstance(fitted, AffineMap)
    assert fitted.weight.dtype == torch.float64
    assert fitted.bias.dtype == torch.float64
    assert mapped.dtype == torch.float32 and mapped.device.type == "cpu"
    assert torch.allclose(mapped, teacher, atol=2e-3, rtol=0)
    assert torch.allclose(torch.linalg.vector_norm(mapped, dim=1), torch.ones(128), atol=1e-6)


@pytest.mark.parametrize("regularization", [1e-4, 1e-2, 1.0])
def test_regularized_affine_is_deterministic_and_improves_pairing(regularization: float) -> None:
    student, teacher = _paired_descriptors()
    first = fit_regularized_affine(student, teacher, regularization)
    second = fit_regularized_affine(student, teacher, regularization)
    identity_error = torch.mean((student - teacher) ** 2)
    fitted_error = torch.mean((first.apply(student) - teacher) ** 2)

    assert torch.equal(first.weight, second.weight)
    assert torch.equal(first.bias, second.bias)
    assert float(fitted_error) < float(identity_error)


@pytest.mark.parametrize("regularization", [0.0, -1.0, 1e-3, 1e-1, 10.0, True])
def test_regularized_affine_rejects_unregistered_strength(regularization: object) -> None:
    student, teacher = _paired_descriptors()
    with pytest.raises(ValueError, match="compatibility regularization differs"):
        fit_regularized_affine(student, teacher, regularization)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("student", "teacher"),
    [
        (torch.ones(1, 4), torch.ones(1, 4)),
        (torch.ones(4, 3), torch.ones(4, 4)),
        (torch.ones(4, 4, dtype=torch.float64), torch.ones(4, 4)),
        (torch.full((4, 4), float("nan")), torch.ones(4, 4)),
        (torch.zeros(4, 4), torch.ones(4, 4)),
    ],
)
def test_affine_fit_rejects_invalid_descriptor_authority(
    student: torch.Tensor, teacher: torch.Tensor
) -> None:
    for fit in (fit_centered_similarity,):
        with pytest.raises(ValueError, match="compatibility descriptor authority differs"):
            fit(student, teacher)


def test_affine_apply_rejects_shape_dtype_and_nonfinite_drift() -> None:
    affine = AffineMap(torch.eye(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64))
    invalid = (
        torch.ones(3, 3),
        torch.ones(3, 4, dtype=torch.float64),
        torch.full((3, 4), float("nan")),
        torch.zeros(3, 4),
    )
    for descriptors in invalid:
        with pytest.raises(ValueError, match="compatibility descriptor authority differs"):
            affine.apply(descriptors)
