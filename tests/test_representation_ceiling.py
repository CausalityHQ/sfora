import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from sfora.representation_ceiling import (
    AffineMap,
    CenteredPcaTransform,
    TeacherGuidedProjection,
    apply_normalized_affine,
    deterministic_class_partition,
    fit_centered_pca,
    fit_ridge_affine,
    fit_teacher_guided_projection,
)


def test_module_has_no_experimental_script_dependency() -> None:
    code = """
import importlib.abc
import sys

class BlockScripts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "scripts" or fullname.startswith("probe_"):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockScripts())
import sfora.representation_ceiling
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_public_api_lazily_exposes_representation_ceiling_primitives() -> None:
    from sfora import (
        AffineMap as PublicAffineMap,
    )
    from sfora import (
        CenteredPcaTransform as PublicCenteredPcaTransform,
    )
    from sfora import (
        ClassDisjointPartition as PublicClassDisjointPartition,
    )
    from sfora import (
        TeacherGuidedProjection as PublicTeacherGuidedProjection,
    )
    from sfora import (
        apply_normalized_affine as public_apply_normalized_affine,
    )
    from sfora import (
        deterministic_class_partition as public_deterministic_class_partition,
    )
    from sfora import (
        fit_centered_pca as public_fit_centered_pca,
    )
    from sfora import (
        fit_ridge_affine as public_fit_ridge_affine,
    )
    from sfora import (
        fit_teacher_guided_projection as public_fit_teacher_guided_projection,
    )

    assert PublicAffineMap is AffineMap
    assert PublicCenteredPcaTransform is CenteredPcaTransform
    assert PublicTeacherGuidedProjection is TeacherGuidedProjection
    assert PublicClassDisjointPartition.__name__ == "ClassDisjointPartition"
    assert public_apply_normalized_affine is apply_normalized_affine
    assert public_deterministic_class_partition is deterministic_class_partition
    assert public_fit_centered_pca is fit_centered_pca
    assert public_fit_ridge_affine is fit_ridge_affine
    assert public_fit_teacher_guided_projection is fit_teacher_guided_projection


def test_class_partition_is_deterministic_ordered_and_disjoint() -> None:
    labels = (11, 11, 22, 22, 33, 33, 44, 44, 55, 55)

    first = deterministic_class_partition(labels, fit_fraction=0.6, seed=17)
    second = deterministic_class_partition(labels, fit_fraction=0.6, seed=17)
    changed = deterministic_class_partition(labels, fit_fraction=0.6, seed=1729)

    assert first == second
    assert first.fit_row_indexes == tuple(sorted(first.fit_row_indexes))
    assert first.validation_row_indexes == tuple(sorted(first.validation_row_indexes))
    assert set(first.fit_row_indexes).isdisjoint(first.validation_row_indexes)
    assert sorted(first.fit_row_indexes + first.validation_row_indexes) == list(range(len(labels)))
    assert set(first.fit_class_ids).isdisjoint(first.validation_class_ids)
    assert len(first.fit_class_ids) == 3
    assert len(first.validation_class_ids) == 2
    assert first != changed


@pytest.mark.parametrize(
    ("labels", "fit_fraction", "seed"),
    [
        ((1, 1), 0.8, 17),
        ((1, 1, 2), 0.5, 17),
        ((1, 1, True, True), 0.5, 17),
        ((1, 1, 2, 2), 0.0, 17),
        ((1, 1, 2, 2), 1.0, 17),
        ((1, 1, 2, 2), 0.5, True),
    ],
)
def test_class_partition_rejects_invalid_or_degenerate_authority(
    labels: tuple[int, ...], fit_fraction: float, seed: int
) -> None:
    with pytest.raises(ValueError, match="class partition authority"):
        deterministic_class_partition(labels, fit_fraction=fit_fraction, seed=seed)


def test_class_partition_allows_fit_singletons_but_rejects_validation_singletons() -> None:
    accepted = deterministic_class_partition((1, 2, 2, 3, 4, 4), fit_fraction=0.5, seed=17)
    assert 1 in accepted.fit_class_ids
    assert 3 in accepted.fit_class_ids

    with pytest.raises(ValueError, match="class partition authority"):
        deterministic_class_partition((1, 1, 2, 3, 3, 4), fit_fraction=0.5, seed=17)


def test_centered_pca_is_fit_only_deterministic_and_sign_oriented() -> None:
    fit = torch.tensor(
        [[3.0, 0.0, 1.0], [2.0, 1.0, 0.0], [-2.0, -1.0, 0.0], [-3.0, 0.0, -1.0]],
        dtype=torch.float32,
    )
    validation = torch.tensor([[9.0, 2.0, -4.0], [-7.0, 3.0, 1.0]], dtype=torch.float32)

    first = fit_centered_pca(fit, dimensions=2)
    second = fit_centered_pca(fit.clone(), dimensions=2)

    assert isinstance(first, CenteredPcaTransform)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.components, second.components)
    assert torch.equal(first.mean, fit.double().mean(dim=0).float())
    for component in first.components:
        pivot = int(torch.argmax(torch.abs(component)))
        assert float(component[pivot]) > 0.0
    encoded = first.apply(validation)
    projected = (validation.double() - first.mean.double()) @ first.components.double().T
    expected = (projected / torch.linalg.vector_norm(projected, dim=1, keepdim=True)).float()
    torch.testing.assert_close(encoded, expected, rtol=0.0, atol=0.0)
    assert torch.allclose(torch.linalg.vector_norm(encoded, dim=1), torch.ones(2))


def test_ridge_affine_recovers_bias_and_uses_unpenalized_intercept() -> None:
    source = torch.tensor(
        [[-2.0, -1.0], [-1.0, 2.0], [0.0, -2.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=torch.float32,
    )
    weight = torch.tensor([[2.0, -0.5], [0.25, 1.5]], dtype=torch.float32)
    bias = torch.tensor([3.0, -4.0], dtype=torch.float32)
    target = source @ weight.T + bias

    fitted = fit_ridge_affine(source, target, penalty=1e-12)

    assert isinstance(fitted, AffineMap)
    torch.testing.assert_close(fitted.weight, weight, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fitted.bias, bias, rtol=1e-5, atol=1e-5)
    raw = fitted.apply(source)
    torch.testing.assert_close(raw, target, rtol=1e-5, atol=1e-5)


def test_teacher_guided_projection_transfers_compact_teacher_geometry() -> None:
    latent = torch.tensor(
        [
            [3.0, 0.0],
            [2.0, 0.0],
            [-3.0, 0.0],
            [-2.0, 0.0],
            [0.0, 2.0],
            [0.0, 1.0],
            [0.0, -2.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    teacher_basis = torch.tensor([[2.0, 1.0, 0.0], [-1.0, 2.0, 1.0]], dtype=torch.float32)
    teacher_offset = torch.tensor([5.0, -3.0, 2.0], dtype=torch.float32)
    teacher = latent @ teacher_basis + teacher_offset
    source_rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)
    source = torch.cat(
        (
            latent @ source_rotation,
            torch.tensor(
                [[10.0], [-10.0], [10.0], [-10.0], [9.0], [-9.0], [9.0], [-9.0]],
                dtype=torch.float32,
            ),
        ),
        dim=1,
    )

    fitted = fit_teacher_guided_projection(
        source,
        teacher,
        dimensions=2,
        penalty=1e-12,
    )
    encoded_source = fitted.apply_source(source)
    encoded_teacher = fitted.apply_teacher(teacher)
    held_out_latent = torch.tensor([[2.5, 0.0], [0.0, 1.5]], dtype=torch.float32)
    held_out_source = torch.cat(
        (held_out_latent @ source_rotation, torch.zeros((2, 1), dtype=torch.float32)), dim=1
    )
    held_out_teacher = held_out_latent @ teacher_basis + teacher_offset

    assert isinstance(fitted, TeacherGuidedProjection)
    assert fitted.source_projection.weight.shape == (2, 3)
    assert fitted.teacher_projection.components.shape == (2, 3)
    assert torch.equal(
        torch.argmax(encoded_source @ encoded_source.T - torch.eye(len(encoded_source)), dim=1),
        torch.tensor([1, 0, 3, 2, 5, 4, 7, 6]),
    )
    assert torch.equal(
        torch.argmax(encoded_teacher @ encoded_teacher.T - torch.eye(len(encoded_teacher)), dim=1),
        torch.tensor([1, 0, 3, 2, 5, 4, 7, 6]),
    )
    torch.testing.assert_close(encoded_source, encoded_teacher, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(
        fitted.apply_source(held_out_source),
        fitted.apply_teacher(held_out_teacher),
        rtol=0.0,
        atol=1e-5,
    )

    changed = fit_teacher_guided_projection(
        source,
        -teacher,
        dimensions=2,
        penalty=1e-12,
    )
    changed_source = changed.apply_source(source)
    changed_teacher = changed.apply_teacher(-teacher)
    torch.testing.assert_close(changed_source, changed_teacher, rtol=0.0, atol=1e-5)
    assert not torch.allclose(encoded_source, changed_source)

    regularized = fit_teacher_guided_projection(
        source,
        teacher,
        dimensions=2,
        penalty=1.0,
    )
    regularized_weight_norm = torch.linalg.vector_norm(regularized.source_projection.weight)
    fitted_weight_norm = torch.linalg.vector_norm(fitted.source_projection.weight)
    assert regularized_weight_norm < fitted_weight_norm


def test_teacher_guided_projection_rejects_misaligned_component_widths() -> None:
    source_projection = AffineMap(
        weight=torch.ones((3, 2), dtype=torch.float32),
        bias=torch.zeros(3, dtype=torch.float32),
    )
    teacher_projection = CenteredPcaTransform(
        mean=torch.zeros(2, dtype=torch.float32),
        components=torch.eye(2, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="teacher-guided projection authority"):
        TeacherGuidedProjection(
            source_projection=source_projection,
            teacher_projection=teacher_projection,
        )


@pytest.mark.parametrize(
    ("source", "teacher", "dimensions"),
    [
        (torch.ones((4, 2), dtype=torch.float32), torch.ones((4, 3)), 1),
        (torch.ones((4, 2), dtype=torch.float32), torch.ones((3, 3)), 2),
        (torch.ones((4, 2), dtype=torch.float32), torch.ones((4, 3)), 3),
    ],
)
def test_teacher_guided_projection_rejects_incompatible_authority_before_fitting(
    source: torch.Tensor, teacher: torch.Tensor, dimensions: int
) -> None:
    with pytest.raises(ValueError, match="teacher-guided projection authority"):
        fit_teacher_guided_projection(
            source,
            teacher,
            dimensions=dimensions,
            penalty=1e-4,
        )


def test_ridge_penalty_is_relative_to_feature_energy() -> None:
    source = torch.tensor(
        [[-2.0, -1.0], [-1.0, 2.0], [0.0, -2.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=torch.float32,
    )
    target = F.normalize(
        source @ torch.tensor([[1.0, -0.4], [0.2, 1.3]], dtype=torch.float32).T
        + torch.tensor([0.3, -0.7]),
        dim=1,
    )

    original = fit_ridge_affine(source, target, penalty=1e-2)
    rescaled = fit_ridge_affine(source * 10.0, target, penalty=1e-2)

    torch.testing.assert_close(
        original.apply(source), rescaled.apply(source * 10.0), rtol=1e-5, atol=1e-5
    )
    repeated = fit_ridge_affine(source.repeat((7, 1)), target.repeat((7, 1)), penalty=1e-2)
    torch.testing.assert_close(original.apply(source), repeated.apply(source), rtol=1e-5, atol=1e-5)


def test_normalized_affine_matches_explicit_application_without_mutation() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float32)
    original = values.clone()
    affine = AffineMap(
        weight=torch.tensor([[1.0, 2.0], [-2.0, 1.0]], dtype=torch.float32),
        bias=torch.tensor([0.5, -0.25], dtype=torch.float32),
    )

    actual = apply_normalized_affine(values, affine)
    raw = values.double() @ affine.weight.double().T + affine.bias.double()
    expected = (raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)).float()

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.equal(values, original)


def test_transforms_clone_authority_and_normalize_large_finite_rows() -> None:
    mean = torch.tensor([0.0, 0.0], dtype=torch.float32)
    components = torch.eye(2, dtype=torch.float32)
    pca = CenteredPcaTransform(mean=mean, components=components)
    weight = torch.eye(2, dtype=torch.float32)
    bias = torch.zeros(2, dtype=torch.float32)
    affine = AffineMap(weight=weight, bias=bias)
    mean.fill_(9.0)
    components.zero_()
    weight.zero_()
    bias.fill_(9.0)
    large = torch.tensor([[1e20, 1e20]], dtype=torch.float32)

    pca_output = pca.apply(large)
    affine_output = apply_normalized_affine(large, affine)

    expected = torch.full((1, 2), 2.0**-0.5, dtype=torch.float32)
    torch.testing.assert_close(pca_output, expected)
    torch.testing.assert_close(affine_output, expected)
    assert bool((torch.linalg.vector_norm(pca_output, dim=1) > 0.999).all())
    assert bool((torch.linalg.vector_norm(affine_output, dim=1) > 0.999).all())

    exposed_weight = affine.weight
    exposed_weight.zero_()
    exposed_components = pca.components
    exposed_components.zero_()
    torch.testing.assert_close(apply_normalized_affine(large, affine), expected)
    torch.testing.assert_close(pca.apply(large), expected)


@pytest.mark.parametrize(
    "case",
    [
        "pca-nonfinite",
        "pca-rank",
        "pca-degenerate",
        "pca-nonconstant-rank",
        "ridge-penalty",
        "ridge-shape",
        "affine-zero",
    ],
)
def test_transforms_fail_closed_on_invalid_authority(case: str) -> None:
    if case == "pca-nonfinite":
        value = torch.tensor([[1.0, 2.0], [float("nan"), 1.0]], dtype=torch.float32)
        with pytest.raises(ValueError, match="PCA authority"):
            fit_centered_pca(value, dimensions=1)
    elif case == "pca-rank":
        value = torch.ones((4, 3), dtype=torch.float32)
        with pytest.raises(ValueError, match="PCA authority"):
            fit_centered_pca(value, dimensions=2)
    elif case == "pca-degenerate":
        value = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=torch.float32
        )
        with pytest.raises(ValueError, match="PCA authority"):
            fit_centered_pca(value, dimensions=1)
    elif case == "pca-nonconstant-rank":
        column = torch.arange(1, 21, dtype=torch.float32)
        value = torch.stack((column, column, column), dim=1)
        with pytest.raises(ValueError, match="PCA authority"):
            fit_centered_pca(value, dimensions=2)
    elif case == "ridge-penalty":
        value = torch.eye(3, dtype=torch.float32)
        with pytest.raises(ValueError, match="ridge authority"):
            fit_ridge_affine(value, value, penalty=0.0)
    elif case == "ridge-shape":
        with pytest.raises(ValueError, match="ridge authority"):
            fit_ridge_affine(
                torch.ones((4, 2), dtype=torch.float32),
                torch.ones((3, 2), dtype=torch.float32),
                penalty=1e-4,
            )
    else:
        affine = AffineMap(
            weight=torch.zeros((2, 2), dtype=torch.float32),
            bias=torch.zeros(2, dtype=torch.float32),
        )
        with pytest.raises(ValueError, match="affine authority"):
            apply_normalized_affine(torch.ones((2, 2), dtype=torch.float32), affine)
