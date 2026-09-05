from __future__ import annotations

import json

import pytest
import torch

from sfora.siglip_compatibility_capacity import (
    AffineMap,
    CompatibilityDecisionMetrics,
    CompatibilityResidual,
    build_compatibility_capacity_result,
    classify_compatibility_capacity,
    compatibility_folds,
    compatibility_retrieval_evidence,
    csls_scores,
    fit_centered_similarity,
    fit_regularized_affine,
    fit_teacher_anchored_residual,
    hubness_present,
    select_compatibility_finalist,
    validate_compatibility_capacity_result_bytes,
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


def _retrieval_bank() -> tuple[torch.Tensor, tuple[str, ...], tuple[int, ...]]:
    descriptors = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 1.0],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    return descriptors, tuple(f"image-{index}" for index in range(6)), (0, 0, 1, 1, 2, 2)


def test_compatibility_retrieval_recomputes_exact_micro_macro_and_diagnostics() -> None:
    descriptors, ids, labels = _retrieval_bank()
    evidence = compatibility_retrieval_evidence(
        descriptors,
        descriptors,
        query_ids=ids,
        gallery_ids=ids,
        query_labels=labels,
        gallery_labels=labels,
        reference_query=descriptors,
        reference_gallery=descriptors,
    )

    assert evidence.hits == (True,) * 6
    assert evidence.average_precisions == (1.0,) * 6
    assert evidence.micro_r1 == 1.0
    assert evidence.micro_map_at_r == 1.0
    assert evidence.class_macro_r1 == 1.0
    assert evidence.class_macro_map_at_r == 1.0
    assert evidence.paired_cosines == (1.0,) * 6
    assert evidence.cross_score_mse == 0.0
    assert evidence.top10_overlaps == (1.0,) * 6
    assert evidence.hub_counts == (5, 5, 5, 5, 5, 5)


def test_compatibility_retrieval_rejects_label_and_reference_drift() -> None:
    descriptors, ids, labels = _retrieval_bank()
    invalid = (
        {"gallery_labels": (1, 0, 1, 1, 2, 2)},
        {"reference_query": descriptors.double()},
        {"reference_gallery": torch.full_like(descriptors, float("nan"))},
    )
    baseline = {
        "query_ids": ids,
        "gallery_ids": ids,
        "query_labels": labels,
        "gallery_labels": labels,
        "reference_query": descriptors,
        "reference_gallery": descriptors,
    }
    for mutation in invalid:
        with pytest.raises(ValueError, match="compatibility retrieval authority differs"):
            compatibility_retrieval_evidence(
                descriptors, descriptors, **(baseline | mutation)  # type: ignore[arg-type]
            )


def test_csls_uses_registered_cross_domain_local_scaling() -> None:
    query = torch.eye(2, dtype=torch.float32)
    gallery = torch.eye(2, dtype=torch.float32)
    assert torch.equal(
        csls_scores(query, gallery, neighbors=1),
        torch.tensor([[0.0, -2.0], [-2.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="compatibility CSLS authority differs"):
        csls_scores(query, gallery, neighbors=3)


def test_finalist_selection_uses_worst_fold_direction_and_registered_ties() -> None:
    results = {
        "affine-0.0001": ((0.80, 0.82), (0.83, 0.81), (0.84, 0.80)),
        "affine-0.01": ((0.80, 0.80), (0.82, 0.81), (0.83, 0.84)),
        "affine-1": ((0.80, 0.80), (0.81, 0.82), (0.83, 0.84)),
        "teacher-anchored-residual": ((0.79, 0.90), (0.91, 0.92), (0.93, 0.94)),
    }
    assert select_compatibility_finalist(results) == "affine-1"
    results["teacher-anchored-residual"] = ((0.81, 0.81),) * 3
    assert select_compatibility_finalist(results) == "teacher-anchored-residual"


def _decision(r1: float, map_at_r: float, self_r1: float = 0.995) -> CompatibilityDecisionMetrics:
    return CompatibilityDecisionMetrics(
        forward_r1=r1,
        forward_map_at_r=map_at_r,
        reverse_r1=r1,
        reverse_map_at_r=map_at_r,
        self_r1=self_r1,
        self_map_at_r=0.97,
    )


@pytest.mark.parametrize(
    ("finalist", "oracle", "expected"),
    [
        (_decision(0.98, 0.96), _decision(0.98, 0.96), "posthoc-passed"),
        (_decision(0.85, 0.84), _decision(0.91, 0.91), "coverage-failure"),
        (_decision(0.85, 0.84), _decision(0.79, 0.90), "information-failure"),
        (_decision(0.85, 0.84), _decision(0.85, 0.85), "ambiguous-capacity"),
    ],
)
def test_capacity_classification_is_exhaustive_and_plain_cosine_only(
    finalist: CompatibilityDecisionMetrics,
    oracle: CompatibilityDecisionMetrics,
    expected: str,
) -> None:
    assert classify_compatibility_capacity(finalist, oracle) == expected


def test_hubness_is_independent_five_point_csls_lift() -> None:
    assert hubness_present((0.70, 0.80), (0.75, 0.81)) is True
    assert hubness_present((0.70, 0.80), (0.749, 0.849)) is False


def _capacity_result() -> bytes:
    descriptors, ids, labels = _retrieval_bank()
    evidence = compatibility_retrieval_evidence(
        descriptors,
        descriptors,
        query_ids=ids,
        gallery_ids=ids,
        query_labels=labels,
        gallery_labels=labels,
        reference_query=descriptors,
        reference_gallery=descriptors,
    )
    cells = {
        name: evidence
        for name in (
            "finalist-forward",
            "finalist-reverse",
            "finalist-self",
            "oracle-forward",
            "oracle-reverse",
            "oracle-self",
        )
    }
    folds = {
        "affine-0.0001": ((0.80, 0.82), (0.83, 0.81), (0.84, 0.80)),
        "affine-0.01": ((0.80, 0.80), (0.82, 0.81), (0.83, 0.84)),
        "affine-1": ((0.80, 0.80), (0.81, 0.82), (0.83, 0.84)),
        "teacher-anchored-residual": ((0.81, 0.81),) * 3,
    }
    return build_compatibility_capacity_result(
        checkpoint_sha256="11" * 32,
        descriptor_artifact_sha256="22" * 32,
        fold_results=folds,
        cells=cells,
        cosine_r1=(0.70, 0.80),
        csls_r1=(0.75, 0.81),
    )


def test_capacity_result_is_canonical_and_recomputes_selection_and_decisions() -> None:
    raw = _capacity_result()
    value = validate_compatibility_capacity_result_bytes(raw)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert value["claim_eligible"] is False
    assert value["finalist"] == "teacher-anchored-residual"
    assert value["classification"] == "posthoc-passed"
    assert value["hubness_present"] is True
    assert json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n" == raw


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("classification",), "coverage-failure"),
        (("finalist",), "affine-1"),
        (("claim_eligible",), 0),
        (("cells", "finalist-forward", "micro_r1"), 0.5),
        (("cells", "oracle-forward", "hits"), [False] * 6),
        (("hubness_present",), False),
    ],
)
def test_capacity_result_rejects_summary_decision_and_concrete_type_drift(
    path: tuple[str, ...], replacement: object
) -> None:
    value = json.loads(_capacity_result())
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    mutated = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="compatibility capacity result"):
        validate_compatibility_capacity_result_bytes(mutated)
