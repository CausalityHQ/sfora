from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
import torch

from sfora.siglip_gallery_compatibility_alignment import (
    apply_orthogonal_alignment,
    build_alignment_result,
    fit_orthogonal_alignment,
    validate_alignment_result_bytes,
)
from sfora.siglip_spatial_tail_recovery import SpatialRetrievalEvidence


def test_procrustes_recovers_reflected_coordinates_and_preserves_geometry() -> None:
    generator = torch.Generator().manual_seed(17)
    student = torch.nn.functional.normalize(torch.randn(96, 12, generator=generator), dim=1)
    orthogonal, _ = torch.linalg.qr(torch.randn(12, 12, generator=generator).double())
    orthogonal[:, 0].neg_()
    teacher = torch.nn.functional.normalize(student.double() @ orthogonal, dim=1).float()

    fitted = fit_orthogonal_alignment(student, teacher)
    aligned = apply_orthogonal_alignment(student, fitted)

    assert torch.allclose(fitted, orthogonal, atol=1e-6, rtol=0)
    assert torch.allclose(aligned, teacher, atol=1e-6, rtol=0)
    assert torch.allclose(aligned @ aligned.T, student @ student.T, atol=1e-5, rtol=0)
    assert float((fitted.T @ fitted - torch.eye(12, dtype=torch.float64)).abs().max()) <= 1e-8


@pytest.mark.parametrize(
    ("student", "teacher"),
    [
        (torch.ones(1, 4), torch.ones(1, 4)),
        (torch.ones(4, 3), torch.ones(4, 4)),
        (torch.ones(4, 4, dtype=torch.float16), torch.ones(4, 4)),
        (torch.full((4, 4), float("nan")), torch.ones(4, 4)),
        (torch.zeros(4, 4), torch.ones(4, 4)),
    ],
)
def test_procrustes_rejects_invalid_descriptor_authority(
    student: torch.Tensor, teacher: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="alignment descriptor authority differs"):
        fit_orthogonal_alignment(student, teacher)


def test_apply_rejects_nonorthogonal_or_wrong_shape_alignment() -> None:
    descriptors = torch.nn.functional.normalize(torch.eye(4), dim=1)
    for matrix in (torch.eye(3, dtype=torch.float64), torch.ones(4, 4, dtype=torch.float64)):
        with pytest.raises(ValueError, match="alignment matrix authority differs"):
            apply_orthogonal_alignment(descriptors, matrix)


def _evidence(correct: int, average_precision: float) -> SpatialRetrievalEvidence:
    return SpatialRetrievalEvidence(
        correct=tuple(index < correct for index in range(100)),
        average_precisions=tuple(average_precision for _ in range(100)),
    )


def _result_bytes() -> bytes:
    self_evidence = _evidence(100, 0.97)
    return build_alignment_result(
        checkpoint_sha256="11" * 32,
        spatial_artifact_sha256="22" * 32,
        alignment_artifact_sha256="33" * 32,
        fit_labels=tuple(sorted(set(range(49)) - {4, 5, 12, 15, 19, 24, 26, 32, 40, 45})),
        development_labels=(4, 5, 12, 15, 19, 24, 26, 32, 40, 45),
        cells={
            "teacher-teacher": _evidence(100, 1.0),
            "student-student": self_evidence,
            "student-teacher": _evidence(70, 0.66),
            "teacher-student": _evidence(71, 0.67),
            "aligned-aligned": self_evidence,
            "aligned-teacher": _evidence(98, 0.96),
            "teacher-aligned": _evidence(97, 0.95),
        },
        fidelity={
            "fit-before": tuple(0.60 for _ in range(80)),
            "fit-after": tuple(0.90 for _ in range(80)),
            "development-before": tuple(0.55 for _ in range(100)),
            "development-after": tuple(0.88 for _ in range(100)),
        },
        orthogonality_error=1e-12,
        self_geometry_max_score_delta=1e-7,
    )


def test_alignment_result_is_canonical_and_recomputes_pass() -> None:
    raw = _result_bytes()
    value = validate_alignment_result_bytes(raw)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert value["classification"] == "alignment-passed"
    assert value["claim_eligible"] is False
    assert value["external_evaluation_access"] is False


def _replace_student_self_with_valid_failure(value: dict[str, Any]) -> None:
    cell = value["cells"]["student-student"]
    cell["hits"] = [False] * 100
    cell["average_precision"] = [0.0] * 100
    cell["correct"] = 0
    cell["map_at_r"] = 0.0


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update({"claim_eligible": 0}), "authority differs"),
        (lambda value: value.update({"alignment_artifact_sha256": "x" * 64}), "digest differs"),
        (
            lambda value: value["cells"]["aligned-teacher"].update({"correct": 96}),
            "retrieval relation differs",
        ),
        (
            lambda value: value["cells"]["aligned-aligned"]["hits"].__setitem__(0, False),
            "retrieval relation differs|self geometry differs",
        ),
        (_replace_student_self_with_valid_failure, "self retrieval differs"),
        (
            lambda value: value["fidelity"]["fit-after"].update({"mean": 0.1}),
            "fidelity relation differs",
        ),
        (
            lambda value: value["fidelity"]["development-after"]["cosines"].pop(),
            "fidelity cardinality differs",
        ),
        (
            lambda value: value.update({"self_geometry_max_score_delta": 1.1e-5}),
            "decision differs",
        ),
        (lambda value: value.update({"classification": "alignment-rejected"}), "decision differs"),
    ],
)
def test_alignment_result_rejects_schema_metric_fidelity_and_decision_drift(
    mutation: Callable[[dict[str, Any]], None], match: str
) -> None:
    value = json.loads(_result_bytes())
    mutation(value)
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError, match=match):
        validate_alignment_result_bytes(raw)
