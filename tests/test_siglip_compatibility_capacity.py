from __future__ import annotations

import json
from typing import cast

import pytest
import torch

from sfora.siglip_compatibility_capacity import (
    AffineMap,
    CompatibilityDecisionMetrics,
    CompatibilityResidual,
    CompatibilityRetrievalEvidence,
    _paired_cosine_loss,
    build_compatibility_capacity_result,
    classify_compatibility_capacity,
    combine_compatibility_retrieval_evidence,
    compatibility_folds,
    compatibility_oracle_halves,
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
    labels = (
        0,
        1,
        2,
        3,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        16,
        17,
        18,
        20,
        21,
        22,
        23,
        25,
        27,
        28,
        29,
        30,
        31,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
    )
    folds = compatibility_folds(labels)
    assert folds == (
        (13, 46, 17, 42, 2, 9, 35, 18, 44, 33, 0, 29, 7),
        (23, 28, 20, 16, 8, 1, 27, 39, 10, 30, 41, 31, 11),
        (37, 36, 22, 43, 21, 34, 3, 25, 38, 48, 47, 6, 14),
    )
    assert set().union(*(set(fold) for fold in folds)) == set(labels)


@pytest.mark.parametrize(
    "labels",
    [tuple(range(38)), tuple(range(39)), tuple([0] * 39), list(range(39))],
)
def test_compatibility_folds_reject_authority_drift(labels: object) -> None:
    with pytest.raises(ValueError, match="compatibility class authority differs"):
        compatibility_folds(labels)  # type: ignore[arg-type]


def test_compatibility_oracle_halves_are_exact_and_reversible() -> None:
    assert compatibility_oracle_halves((4, 5, 12, 15, 19, 24, 26, 32, 40, 45)) == (
        (19, 24, 40, 4, 32),
        (45, 12, 5, 26, 15),
    )


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
            key=lambda value: (
                __import__("hashlib")
                .sha256(b"sfora-compatibility-anchor-v1\0" + value.encode())
                .digest()
            ),
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
        value == 0.0 for name in ("forward", "reverse", "self") for value in fitted.losses[name]
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
            student,
            teacher,
            ids,
            relational=relational,  # type: ignore[arg-type]
            seed=seed,
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
    assert evidence.ids == ids
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
                descriptors,
                descriptors,
                **(baseline | mutation),
            )


def test_compatibility_retrieval_clamps_fp32_cosine_roundoff() -> None:
    generator = torch.Generator().manual_seed(1)
    descriptors = torch.nn.functional.normalize(torch.randn(64, 512, generator=generator), dim=1)
    ids = tuple(f"roundoff-{index}" for index in range(64))
    labels = tuple(index // 2 for index in range(64))
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
    assert max(evidence.paired_cosines) <= 1.0
    assert min(evidence.paired_cosines) >= -1.0


def test_paired_cosine_loss_clamps_normalization_roundoff() -> None:
    generator = torch.Generator().manual_seed(1)
    descriptors = torch.nn.functional.normalize(torch.randn(64, 512, generator=generator), dim=1)
    loss = _paired_cosine_loss(
        torch.nn.functional.normalize(descriptors.double(), dim=1),
        descriptors.double(),
    )
    assert float(loss) >= 0.0


def test_compatibility_retrieval_combines_disjoint_held_out_panels_by_identity() -> None:
    descriptors, first_ids, first_labels = _retrieval_bank()
    second_ids = tuple(f"held-out-{index}" for index in range(6))
    second_labels = tuple(label + 3 for label in first_labels)
    first = compatibility_retrieval_evidence(
        descriptors,
        descriptors,
        query_ids=first_ids,
        gallery_ids=first_ids,
        query_labels=first_labels,
        gallery_labels=first_labels,
        reference_query=descriptors,
        reference_gallery=descriptors,
    )
    second = compatibility_retrieval_evidence(
        descriptors,
        descriptors,
        query_ids=second_ids,
        gallery_ids=second_ids,
        query_labels=second_labels,
        gallery_labels=second_labels,
        reference_query=descriptors,
        reference_gallery=descriptors,
    )
    expected_ids = tuple(
        value for pair in zip(first_ids, second_ids, strict=True) for value in pair
    )
    combined = combine_compatibility_retrieval_evidence((first, second), expected_ids=expected_ids)

    assert combined.ids == expected_ids
    assert combined.labels == tuple(
        value for pair in zip(first_labels, second_labels, strict=True) for value in pair
    )
    assert combined.hits == (True,) * 12
    assert combined.cross_score_mse == 0.0
    with pytest.raises(ValueError, match="compatibility retrieval authority differs"):
        combine_compatibility_retrieval_evidence((first, first), expected_ids=first_ids + first_ids)


def test_csls_uses_registered_cross_domain_local_scaling() -> None:
    query = torch.eye(2, dtype=torch.float32)
    gallery = torch.eye(2, dtype=torch.float32)
    assert torch.equal(
        csls_scores(query, gallery, neighbors=1),
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
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
    ("finalist", "oracle", "matched_finalist", "expected"),
    [
        (_decision(0.98, 0.96), _decision(0.98, 0.96), _decision(0.98, 0.96), "posthoc-passed"),
        (_decision(0.85, 0.84), _decision(0.91, 0.91), _decision(0.85, 0.84), "coverage-failure"),
        (_decision(0.85, 0.84), _decision(0.91, 0.91), _decision(0.98, 0.96), "ambiguous-capacity"),
        (_decision(0.85, 0.84), _decision(0.91, 0.91), _decision(0.91, 0.91), "ambiguous-capacity"),
        (_decision(0.85, 0.84), _decision(0.91, 0.91), _decision(0.96, 0.94), "ambiguous-capacity"),
        (
            _decision(0.85, 0.84),
            _decision(0.79, 0.90),
            _decision(0.70, 0.70),
            "registered-map-failure",
        ),
        (_decision(0.85, 0.84), _decision(0.85, 0.85), _decision(0.85, 0.85), "ambiguous-capacity"),
    ],
)
def test_capacity_classification_is_exhaustive_and_plain_cosine_only(
    finalist: CompatibilityDecisionMetrics,
    oracle: CompatibilityDecisionMetrics,
    matched_finalist: CompatibilityDecisionMetrics,
    expected: str,
) -> None:
    assert classify_compatibility_capacity(finalist, oracle, matched_finalist) == expected


def test_hubness_is_independent_five_point_csls_lift() -> None:
    assert hubness_present((0.70, 0.80), (0.75, 0.81)) is True
    assert hubness_present((0.70, 0.80), (0.749, 0.849)) is False


def _capacity_result() -> bytes:
    labels = tuple(label for label in (4, 5, 12, 15, 19, 24, 26, 32, 40, 45) for _ in range(52))
    descriptors = torch.nn.functional.pad(
        torch.eye(10, dtype=torch.float32), (39, 0)
    ).repeat_interleave(52, dim=0)
    ids = tuple(f"development-{index}" for index in range(520))
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
            "identity-forward",
            "identity-reverse",
            "identity-self",
            "identity-oracle-panel-forward",
            "identity-oracle-panel-reverse",
            "identity-oracle-panel-self",
            "finalist-forward",
            "finalist-reverse",
            "finalist-self",
            "finalist-oracle-panel-forward",
            "finalist-oracle-panel-reverse",
            "finalist-oracle-panel-self",
            "oracle-forward",
            "oracle-reverse",
            "oracle-self",
        )
    }
    fitting_labels = tuple(
        label for label in sorted(set(range(49)) - set(labels)) for _ in range(10)
    )
    fitting_descriptors = torch.nn.functional.pad(
        torch.eye(39, dtype=torch.float32), (0, 10)
    ).repeat_interleave(10, dim=0)
    fitting_ids = tuple(f"fitting-{index}" for index in range(390))
    fold_evidence: dict[
        str, tuple[tuple[CompatibilityRetrievalEvidence, CompatibilityRetrievalEvidence], ...]
    ] = {}
    fold_pairs: list[tuple[CompatibilityRetrievalEvidence, CompatibilityRetrievalEvidence]] = []
    fitting_fold_labels = tuple(sorted(set(range(49)) - set(labels)))
    for fold in compatibility_folds(fitting_fold_labels):
        retained = torch.tensor([label in set(fold) for label in fitting_labels])
        fold_labels = tuple(label for label in fitting_labels if label in set(fold))
        fold_descriptors = fitting_descriptors[retained]
        fold_ids = tuple(
            identity
            for identity, label in zip(fitting_ids, fitting_labels, strict=True)
            if label in set(fold)
        )
        fold_cell = compatibility_retrieval_evidence(
            fold_descriptors,
            fold_descriptors,
            query_ids=fold_ids,
            gallery_ids=fold_ids,
            query_labels=fold_labels,
            gallery_labels=fold_labels,
            reference_query=fold_descriptors,
            reference_gallery=fold_descriptors,
        )
        fold_pairs.append((fold_cell, fold_cell))
    for name in (
        "affine-0.0001",
        "affine-0.01",
        "affine-1",
        "teacher-anchored-residual",
        "centered-similarity",
        "paired-only-residual",
    ):
        fold_evidence[name] = tuple(fold_pairs)
    fitting_identity = compatibility_retrieval_evidence(
        fitting_descriptors,
        fitting_descriptors,
        query_ids=fitting_ids,
        gallery_ids=fitting_ids,
        query_labels=fitting_labels,
        gallery_labels=fitting_labels,
        reference_query=fitting_descriptors,
        reference_gallery=fitting_descriptors,
    )
    optimization_records: list[dict[str, object]] = []
    for fold_index, fold in enumerate(compatibility_folds(fitting_fold_labels)):
        training_ids = tuple(
            identity
            for identity, label in zip(fitting_ids, fitting_labels, strict=True)
            if label not in set(fold)
        )
        anchor_ids = sorted(
            training_ids,
            key=lambda identity: (
                __import__("hashlib")
                .sha256(b"sfora-compatibility-anchor-v1\0" + identity.encode())
                .digest()
            ),
        )[:256]
        optimization_records.append(
            {
                "relational": True,
                "seed": 20260905 + fold_index,
                "dimensions": 49,
                "parameter_count": 3_185,
                "anchor_ids": anchor_ids,
                "losses": {
                    name: [0.0] * 2_000 for name in ("paired", "forward", "reverse", "self")
                },
                "device": "cpu",
                "torch_version": str(torch.__version__),
                "optimizer": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            }
        )
    oracle_records: list[dict[str, object]] = []
    oracle_halves = compatibility_oracle_halves(tuple(sorted(set(labels))))
    for fold_index, training_labels in enumerate(oracle_halves):
        training_ids = tuple(
            identity
            for identity, label in zip(ids, labels, strict=True)
            if label in set(training_labels)
        )
        oracle_records.append(
            {
                "relational": True,
                "seed": 20260909 + fold_index,
                "dimensions": 49,
                "parameter_count": 3_185,
                "anchor_ids": sorted(
                    training_ids,
                    key=lambda identity: (
                        __import__("hashlib")
                        .sha256(b"sfora-compatibility-anchor-v1\0" + identity.encode())
                        .digest()
                    ),
                )[:256],
                "losses": {
                    name: [0.0] * 2_000 for name in ("paired", "forward", "reverse", "self")
                },
                "device": "cpu",
                "torch_version": str(torch.__version__),
                "optimizer": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
            }
        )
    return build_compatibility_capacity_result(
        checkpoint_sha256="11" * 32,
        descriptor_artifact_sha256="22" * 32,
        control_binding_sha256="33" * 32,
        optimization_manifest_sha256="44" * 32,
        spatial_artifact_sha256="55" * 32,
        image_manifest_sha256="66" * 32,
        preprocessing="siglip-evaluation-transform-v1",
        fold_evidence=fold_evidence,
        fitting_identity=fitting_identity,
        residual_optimization={
            "teacher-anchored-residual": tuple(optimization_records),
            "paired-only-residual": tuple(
                record | {"relational": False} for record in optimization_records
            ),
        },
        decision_optimization={
            "oracle-halves": tuple(oracle_records),
            "finalist-refit": None,
        },
        cells=cells,
        identity_cosine_r1=(1.0, 1.0),
        identity_csls_hits=((True,) * 520, (True,) * 520),
        finalist_cosine_r1=(1.0, 1.0),
        finalist_csls_hits=((True,) * 520, (True,) * 520),
    )


def test_capacity_result_is_canonical_and_recomputes_selection_and_decisions() -> None:
    raw = _capacity_result()
    value = validate_compatibility_capacity_result_bytes(raw)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert value["claim_eligible"] is False
    assert value["finalist"] == "affine-1"
    assert set(cast(dict[str, object], value["control_fold_results"])) == {
        "centered-similarity",
        "paired-only-residual",
    }
    assert value["classification"] == "posthoc-passed"
    assert value["paired_cosine_gap"] == {
        "development_mean": 1.0,
        "fitting_mean": 1.0,
        "gap": 0.0,
    }
    assert value["hubness_present"] is False
    assert json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n" == raw


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("classification",), "coverage-failure"),
        (("finalist",), "teacher-anchored-residual"),
        (("fold_results", "affine-1", 0, 0), 0.5),
        (("control_fold_results", "centered-similarity", 0, 0), 0.5),
        (("fold_evidence", "affine-1", 0, "forward", "hits", 0), False),
        (("paired_cosine_gap", "gap"), 0.5),
        (("residual_optimization", "teacher-anchored-residual", 0, "parameter_count"), 1),
        (("residual_optimization", "teacher-anchored-residual", 0, "learning_rate"), 0.01),
        (("residual_optimization", "teacher-anchored-residual", 0, "device"), "tpu"),
        (
            ("residual_optimization", "teacher-anchored-residual", 0, "anchor_ids", 0),
            "forged-anchor",
        ),
        (("residual_optimization", "paired-only-residual", 0, "losses", "paired"), []),
        (("decision_optimization", "oracle-halves"), []),
        (("decision_optimization", "finalist-refit"), {}),
        (("claim_eligible",), 0),
        (("spatial_artifact_sha256",), "0" * 63),
        (("preprocessing",), "different-transform"),
        (("cells", "finalist-forward", "micro_r1"), 0.5),
        (("cells", "oracle-forward", "hits"), [False] * 20),
        (("cells", "oracle-forward", "labels"), [49] + list(range(1, 20))),
        (("cells", "oracle-forward", "ids", 0), "external-image"),
        (("csls_diagnostic", "finalist", "cosine_r1"), [0.5, 1.0]),
        (("csls_diagnostic", "identity", "directions", "forward", "r1"), 0.75),
        (("csls_diagnostic", "finalist", "directions", "reverse", "hits"), []),
        (("hubness_present",), True),
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


def test_capacity_result_rejects_coherent_fold_identity_rewrite() -> None:
    value = json.loads(_capacity_result())
    for entries in value["fold_evidence"].values():
        entries[0]["forward"]["ids"][0] = "forged-fold-id"
        entries[0]["reverse"]["ids"][0] = "forged-fold-id"
    mutated = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="compatibility capacity result"):
        validate_compatibility_capacity_result_bytes(mutated)


def test_capacity_result_rejects_coherent_residual_dimension_drift() -> None:
    value = json.loads(_capacity_result())
    record = value["residual_optimization"]["teacher-anchored-residual"][1]
    record["dimensions"] = 2
    record["parameter_count"] = 130
    mutated = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="compatibility capacity result optimization"):
        validate_compatibility_capacity_result_bytes(mutated)


def test_capacity_result_rejects_fitting_development_identity_overlap() -> None:
    value = json.loads(_capacity_result())
    fitting_id = value["fitting_identity"]["ids"][0]
    for cell in value["cells"].values():
        cell["ids"][0] = fitting_id
    mutated = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="compatibility capacity result"):
        validate_compatibility_capacity_result_bytes(mutated)


def test_capacity_result_requires_selected_residual_refit_optimization() -> None:
    value = json.loads(_capacity_result())
    for arm in ("affine-0.0001", "affine-0.01", "affine-1"):
        for fold in value["fold_evidence"][arm]:
            for direction in ("forward", "reverse"):
                evidence = fold[direction]
                evidence["average_precisions"] = [0.5] * len(evidence["average_precisions"])
                evidence["micro_map_at_r"] = 0.5
                evidence["class_macro_map_at_r"] = 0.5
        value["fold_results"][arm] = [[0.5, 0.5]] * 3
    fitting_ids = value["fitting_identity"]["ids"]
    refit = json.loads(json.dumps(value["residual_optimization"]["teacher-anchored-residual"][0]))
    refit["seed"] = 20260908
    refit["anchor_ids"] = sorted(
        fitting_ids,
        key=lambda identity: (
            __import__("hashlib")
            .sha256(b"sfora-compatibility-anchor-v1\0" + identity.encode())
            .digest()
        ),
    )[:256]
    value["decision_optimization"]["finalist-refit"] = refit
    value["finalist"] = "teacher-anchored-residual"
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    validate_compatibility_capacity_result_bytes(raw)

    value["decision_optimization"]["finalist-refit"] = None
    mutated = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="compatibility capacity result optimization"):
        validate_compatibility_capacity_result_bytes(mutated)
