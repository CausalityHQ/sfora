from __future__ import annotations

import hashlib

import pytest
import torch

from sfora.siglip_coverage_calibration import (
    CoverageAffine,
    CoverageMaps,
    coverage_calibration_classification,
    coverage_retrieval_cells,
    coverage_support_indexes,
    fit_coverage_affine,
)


def _support_ids(ids: tuple[str, ...], indexes: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(ids[index] for index in indexes)


def test_coverage_support_indexes_are_exact_balanced_and_permutation_stable() -> None:
    ids = tuple(f"image-{index:02d}" for index in range(20))
    labels = (3,) * 10 + (7,) * 10
    expected = tuple(
        sorted(
            (
                image_id
                for label in (3, 7)
                for _digest, image_id in sorted(
                    (
                        hashlib.sha256(
                            b"sfora-coverage-support-v1\0"
                            + str(label).encode("ascii")
                            + b"\0"
                            + ids[index].encode("utf-8")
                        ).digest(),
                        ids[index],
                    )
                    for index, value in enumerate(labels)
                    if value == label
                )[:8]
            )
        )
    )
    selected = coverage_support_indexes(ids, labels)
    assert _support_ids(ids, selected) == expected
    assert {label: sum(labels[index] == label for index in selected) for label in (3, 7)} == {
        3: 8,
        7: 8,
    }

    permutation = tuple(reversed(range(len(ids))))
    permuted_ids = tuple(ids[index] for index in permutation)
    permuted_labels = tuple(labels[index] for index in permutation)
    permuted = coverage_support_indexes(permuted_ids, permuted_labels)
    assert _support_ids(permuted_ids, permuted) == expected


@pytest.mark.parametrize(
    ("ids", "labels", "support"),
    [
        (("x",) * 20, (0,) * 10 + (1,) * 10, 8),
        (tuple(f"x{i}" for i in range(19)), (0,) * 9 + (1,) * 10, 8),
        (tuple(f"x{i}" for i in range(20)), (0,) * 10 + (1,) * 10, True),
        (tuple(f"x{i}" for i in range(20)), (0,) * 10 + (1,) * 10, 4),
    ],
)
def test_coverage_support_indexes_reject_authority_drift(
    ids: tuple[str, ...], labels: tuple[int, ...], support: object
) -> None:
    with pytest.raises(ValueError, match="support authority"):
        coverage_support_indexes(ids, labels, support_per_class=support)  # type: ignore[arg-type]


def test_fit_coverage_affine_recovers_full_rank_map_and_is_order_invariant() -> None:
    source = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
            [0.25, -0.75],
            [2.0, -1.0],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor([[1.5, -0.25], [0.5, 2.0]], dtype=torch.float64)
    bias = torch.tensor([0.3, -0.7], dtype=torch.float64)
    target = (source.double() @ weight + bias).float()
    ids = tuple(f"id-{index}" for index in range(source.shape[0]))

    fitted = fit_coverage_affine(source, target, ids)
    assert isinstance(fitted, CoverageAffine)
    assert fitted.rank == 3
    assert fitted.rcond == 1e-12
    assert fitted.driver == "gelsd"
    # The exact affine targets crossed the registered FP32 descriptor boundary.
    assert torch.allclose(fitted.weight, weight, atol=1e-6, rtol=0)
    assert torch.allclose(fitted.bias, bias, atol=1e-6, rtol=0)
    expected = torch.nn.functional.normalize(target, dim=1)
    assert fitted.apply(source).dtype == torch.float32
    assert fitted.apply(source).is_contiguous()
    assert torch.allclose(fitted.apply(source), expected, atol=1e-6, rtol=0)

    permutation = torch.tensor([5, 1, 3, 0, 4, 2])
    reordered = fit_coverage_affine(
        source[permutation],
        target[permutation],
        tuple(ids[index] for index in permutation.tolist()),
    )
    assert torch.equal(fitted.weight, reordered.weight)
    assert torch.equal(fitted.bias, reordered.bias)


@pytest.mark.parametrize("mutation", ("rank", "duplicate", "nonfinite", "dtype"))
def test_fit_coverage_affine_rejects_invalid_inputs(mutation: str) -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]], dtype=torch.float32)
    target = source.clone()
    ids = ("a", "b", "c", "d")
    if mutation == "rank":
        source[:, 1] = source[:, 0]
    elif mutation == "duplicate":
        ids = ("a", "a", "c", "d")
    elif mutation == "nonfinite":
        target[0, 0] = torch.nan
    elif mutation == "dtype":
        source = source.double()
    with pytest.raises(ValueError, match="affine authority"):
        fit_coverage_affine(source, target, ids)


def test_coverage_affine_rejects_zero_mapped_output() -> None:
    mapping = CoverageAffine(
        weight=torch.zeros((2, 2), dtype=torch.float64),
        bias=torch.zeros(2, dtype=torch.float64),
        rank=3,
        singular_values=torch.ones(3, dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )
    with pytest.raises(ValueError, match="affine output"):
        mapping.apply(torch.eye(2, dtype=torch.float32))


def _identity_affine() -> CoverageAffine:
    return CoverageAffine(
        weight=torch.eye(2, dtype=torch.float64),
        bias=torch.zeros(2, dtype=torch.float64),
        rank=3,
        singular_values=torch.ones(3, dtype=torch.float64),
        rcond=1e-12,
        driver="gelsd",
    )


def test_coverage_retrieval_cells_cover_both_deployment_directions() -> None:
    descriptors = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32
    )
    ids = ("a", "b", "c", "d")
    labels = (0, 0, 1, 1)
    cells = coverage_retrieval_cells(
        descriptors,
        descriptors,
        ids,
        labels,
        CoverageMaps(_identity_affine(), _identity_affine()),
    )
    assert set(cells) == {
        "identity-forward",
        "identity-reverse",
        "student-self",
        "teacher-self",
        "forward-student-query",
        "forward-teacher-query",
        "offline-student-query",
        "offline-teacher-query",
    }
    assert all(value.micro_r1 == 1.0 for value in cells.values())
    assert all(value.micro_map_at_r == 1.0 for value in cells.values())
    assert coverage_calibration_classification(cells) == "coverage-calibration-qualified"


def test_coverage_classification_distinguishes_offline_and_rejection() -> None:
    descriptors = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32
    )
    cells = coverage_retrieval_cells(
        descriptors,
        descriptors,
        ("a", "b", "c", "d"),
        (0, 0, 1, 1),
        CoverageMaps(_identity_affine(), _identity_affine()),
    )
    failed = cells["identity-forward"]
    object.__setattr__(failed, "hits", (False,) * 4)
    object.__setattr__(failed, "average_precisions", (0.0,) * 4)
    cells["forward-student-query"] = failed
    assert coverage_calibration_classification(cells) == "offline-gallery-qualified"
    cells["offline-student-query"] = failed
    assert coverage_calibration_classification(cells) == "coverage-calibration-rejected"
