from __future__ import annotations

import pytest
import torch

from sfora.siglip_compatibility_capacity import (
    AffineMap,
    CompatibilityResidual,
    compatibility_folds,
    fit_centered_similarity,
    fit_regularized_affine,
    fit_teacher_anchored_residual,
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


def _residual_bank() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    student, teacher = _paired_descriptors(rows=256, dimensions=4)
    return student, teacher, tuple(f"example-{index:03d}" for index in range(256))


def test_compatibility_residual_starts_as_exact_identity() -> None:
    model = CompatibilityResidual(4, rank=32, seed=17)
    student, _, _ = _residual_bank()
    assert torch.equal(model(student.double()), student.double())
    assert sum(parameter.numel() for parameter in model.parameters()) == 260


def test_teacher_anchored_residual_is_deterministic_and_records_all_losses() -> None:
    student, teacher, ids = _residual_bank()
    first = fit_teacher_anchored_residual(student, teacher, ids, relational=True, seed=17)
    second = fit_teacher_anchored_residual(student, teacher, ids, relational=True, seed=17)

    assert first.anchor_ids == second.anchor_ids
    assert first.anchor_ids == tuple(
        sorted(
            ids,
            key=lambda value: __import__("hashlib").sha256(
                b"sfora-compatibility-anchor-v1\0" + value.encode()
            ).digest(),
        )[:256]
    )
    assert set(first.losses) == {"paired", "forward", "reverse", "self"}
    assert all(len(values) == 2_000 for values in first.losses.values())
    assert first.losses == second.losses
    assert all(
        torch.equal(first.state_dict[key], second.state_dict[key]) for key in first.state_dict
    )
    assert float((first.apply(student) - teacher).square().mean()) < float(
        (student - teacher).square().mean()
    )


def test_paired_only_residual_records_zero_relational_losses() -> None:
    student, teacher, ids = _residual_bank()
    fitted = fit_teacher_anchored_residual(student, teacher, ids, relational=False, seed=17)
    assert fitted.relational is False
    assert all(
        value == 0.0
        for name in ("forward", "reverse", "self")
        for value in fitted.losses[name]
    )


@pytest.mark.parametrize(
    ("ids", "relational", "seed"),
    [
        (tuple(f"id-{index}" for index in range(255)), True, 17),
        (tuple("duplicate" for _ in range(256)), True, 17),
        (tuple(f"id-{index}" for index in range(256)), 1, 17),
        (tuple(f"id-{index}" for index in range(256)), True, -1),
    ],
)
def test_teacher_anchored_residual_rejects_authority_drift(
    ids: tuple[str, ...], relational: object, seed: int
) -> None:
    student, teacher, _ = _residual_bank()
    with pytest.raises(ValueError, match="compatibility residual authority differs"):
        fit_teacher_anchored_residual(
            student, teacher, ids, relational=relational, seed=seed  # type: ignore[arg-type]
        )
