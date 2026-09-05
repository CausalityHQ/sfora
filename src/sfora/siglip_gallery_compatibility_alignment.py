"""Orthogonal descriptor alignment for backward-compatible SigLIP retrieval."""

from __future__ import annotations

import json
import math
from typing import cast

import torch
from torch.nn import functional as F

from sfora.siglip_spatial_tail_recovery import (
    SpatialRetrievalEvidence,
    spatial_tail_class_split,
)


def _validated_descriptors(
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
        or student.shape != teacher.shape
        or student.shape[0] < 2
        or student.shape[1] < 2
        or not bool(torch.isfinite(student).all())
        or not bool(torch.isfinite(teacher).all())
        or not torch.allclose(
            torch.linalg.vector_norm(student, dim=1),
            torch.ones(student.shape[0]),
            atol=1e-5,
            rtol=0,
        )
        or not torch.allclose(
            torch.linalg.vector_norm(teacher, dim=1),
            torch.ones(teacher.shape[0]),
            atol=1e-5,
            rtol=0,
        )
    ):
        raise ValueError("alignment descriptor authority differs")
    return student, teacher


def fit_orthogonal_alignment(
    student_descriptors: torch.Tensor, teacher_descriptors: torch.Tensor
) -> torch.Tensor:
    """Fit the deterministic FP64 orthogonal Procrustes map student→teacher."""

    student, teacher = _validated_descriptors(student_descriptors, teacher_descriptors)
    covariance = student.double().T @ teacher.double()
    left, _singular, right = torch.linalg.svd(covariance, full_matrices=False)
    matrix = (left @ right).contiguous()
    error = float(
        (matrix.T @ matrix - torch.eye(matrix.shape[0], dtype=torch.float64)).abs().max()
    )
    if not bool(torch.isfinite(matrix).all()) or error > 1e-8:
        raise ValueError("alignment matrix authority differs")
    return cast(torch.Tensor, matrix)


def apply_orthogonal_alignment(
    descriptors: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """Apply an authenticated orthogonal map and return normalized FP32 descriptors."""

    if (
        type(descriptors) is not torch.Tensor
        or descriptors.device.type != "cpu"
        or descriptors.dtype != torch.float32
        or descriptors.ndim != 2
        or descriptors.shape[0] < 1
        or not bool(torch.isfinite(descriptors).all())
        or type(matrix) is not torch.Tensor
        or matrix.device.type != "cpu"
        or matrix.dtype != torch.float64
        or matrix.ndim != 2
        or matrix.shape != (descriptors.shape[1], descriptors.shape[1])
        or not bool(torch.isfinite(matrix).all())
        or float(
            (
                matrix.T @ matrix
                - torch.eye(matrix.shape[0], dtype=torch.float64)
            ).abs().max()
        )
        > 1e-8
    ):
        raise ValueError("alignment matrix authority differs")
    aligned = F.normalize(descriptors.double() @ matrix, dim=1).float().contiguous()
    if not bool(torch.isfinite(aligned).all()):
        raise ValueError("alignment descriptor authority differs")
    return aligned


def _hex_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("alignment digest differs")
    return value


def _retrieval(value: object, *, summarized: bool) -> dict[str, object]:
    keys = {"hits", "average_precision"}
    if summarized:
        keys |= {"queries", "correct", "map_at_r"}
    if type(value) is SpatialRetrievalEvidence:
        raw_hits: object = list(value.correct)
        raw_scores: object = list(value.average_precisions)
    elif type(value) is dict and set(value) == keys:
        item = cast(dict[str, object], value)
        raw_hits = item["hits"]
        raw_scores = item["average_precision"]
    else:
        raise ValueError("alignment retrieval evidence differs")
    if (
        type(raw_hits) is not list
        or type(raw_scores) is not list
        or len(raw_hits) < 2
        or len(raw_scores) != len(raw_hits)
        or any(type(hit) is not bool for hit in raw_hits)
        or any(
            type(score) is not float
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            for score in raw_scores
        )
    ):
        raise ValueError("alignment retrieval evidence differs")
    hits = cast(list[bool], raw_hits)
    scores = cast(list[float], raw_scores)
    correct = sum(hits)
    mean = math.fsum(scores) / len(scores)
    if summarized:
        item = cast(dict[str, object], value)
        if (
            type(item["queries"]) is not int
            or item["queries"] != len(hits)
            or type(item["correct"]) is not int
            or item["correct"] != correct
            or type(item["map_at_r"]) is not float
            or item["map_at_r"] != mean
        ):
            raise ValueError("alignment retrieval relation differs")
    return {
        "queries": len(hits),
        "correct": correct,
        "map_at_r": mean,
        "hits": hits.copy(),
        "average_precision": scores.copy(),
    }


def _fidelity(value: object) -> dict[str, object]:
    names = {"fit-before", "fit-after", "development-before", "development-after"}
    if type(value) is not dict or set(value) != names:
        raise ValueError("alignment fidelity evidence differs")
    output: dict[str, object] = {}
    for name in sorted(names):
        scores = cast(dict[str, object], value)[name]
        if type(scores) is tuple:
            scores = list(scores)
        if (
            type(scores) is not list
            or len(scores) < 2
            or any(
                type(score) is not float
                or not math.isfinite(score)
                or not -1.0 <= score <= 1.0
                for score in scores
            )
        ):
            raise ValueError("alignment fidelity evidence differs")
        output[name] = {
            "cosines": scores.copy(),
            "mean": math.fsum(cast(list[float], scores)) / len(scores),
        }
    return output


def _parsed_fidelity(value: object) -> dict[str, object]:
    names = {"fit-before", "fit-after", "development-before", "development-after"}
    if type(value) is not dict or set(value) != names:
        raise ValueError("alignment fidelity evidence differs")
    vectors: dict[str, object] = {}
    for name in names:
        cell = cast(dict[str, object], value)[name]
        if type(cell) is not dict or set(cell) != {"cosines", "mean"}:
            raise ValueError("alignment fidelity evidence differs")
        scores = cast(dict[str, object], cell)["cosines"]
        mean = cast(dict[str, object], cell)["mean"]
        if (
            type(scores) is not list
            or len(scores) < 2
            or any(
                type(score) is not float
                or not math.isfinite(score)
                or not -1.0 <= score <= 1.0
                for score in scores
            )
            or type(mean) is not float
            or mean != math.fsum(cast(list[float], scores)) / len(scores)
        ):
            raise ValueError("alignment fidelity relation differs")
        vectors[name] = scores
    return _fidelity(vectors)


def _validate_fidelity_cardinality(
    fidelity: dict[str, object], *, query_count: int
) -> None:
    lengths = {
        name: len(cast(list[float], cast(dict[str, object], cell)["cosines"]))
        for name, cell in fidelity.items()
    }
    if (
        lengths["fit-before"] != lengths["fit-after"]
        or lengths["development-before"] != lengths["development-after"]
        or lengths["development-before"] != query_count
    ):
        raise ValueError("alignment fidelity cardinality differs")


def _alignment_decision(
    cells: dict[str, dict[str, object]],
    fidelity: dict[str, object],
    self_geometry_max_score_delta: float,
) -> str:
    aligned_student = cells["aligned-aligned"]
    if aligned_student != cells["student-student"]:
        raise ValueError("alignment self retrieval differs")
    if (
        type(self_geometry_max_score_delta) is not float
        or not math.isfinite(self_geometry_max_score_delta)
        or self_geometry_max_score_delta < 0.0
    ):
        raise ValueError("alignment self geometry differs")
    aligned_teacher = cells["aligned-teacher"]
    teacher_aligned = cells["teacher-aligned"]
    fit_before = cast(dict[str, object], fidelity["fit-before"])["mean"]
    fit_after = cast(dict[str, object], fidelity["fit-after"])["mean"]
    dev_before = cast(dict[str, object], fidelity["development-before"])["mean"]
    dev_after = cast(dict[str, object], fidelity["development-after"])["mean"]
    passed = (
        cast(int, aligned_teacher["correct"]) * 100
        >= cast(int, aligned_teacher["queries"]) * 97
        and cast(float, aligned_teacher["map_at_r"]) >= 0.95
        and cast(int, teacher_aligned["correct"]) * 100
        >= cast(int, teacher_aligned["queries"]) * 97
        and cast(float, teacher_aligned["map_at_r"]) >= 0.95
        and cast(int, aligned_student["correct"]) * 100
        >= cast(int, aligned_student["queries"]) * 99
        and cast(float, aligned_student["map_at_r"]) >= 0.96
        and self_geometry_max_score_delta <= 1e-5
        and cast(float, fit_after) >= cast(float, fit_before)
        and cast(float, dev_after) >= cast(float, dev_before)
    )
    return "alignment-passed" if passed else "alignment-rejected"


def build_alignment_result(
    *,
    checkpoint_sha256: str,
    spatial_artifact_sha256: str,
    alignment_artifact_sha256: str,
    fit_labels: tuple[int, ...],
    development_labels: tuple[int, ...],
    cells: dict[str, SpatialRetrievalEvidence],
    fidelity: dict[str, tuple[float, ...]],
    orthogonality_error: float,
    self_geometry_max_score_delta: float,
) -> bytes:
    """Build canonical fitting-only alignment evidence."""

    expected_fit, expected_development = spatial_tail_class_split(tuple(range(49)))
    if fit_labels != tuple(sorted(expected_fit)) or development_labels != tuple(
        sorted(expected_development)
    ):
        raise ValueError("alignment class authority differs")
    names = {
        "teacher-teacher",
        "student-student",
        "student-teacher",
        "teacher-student",
        "aligned-aligned",
        "aligned-teacher",
        "teacher-aligned",
    }
    if type(cells) is not dict or set(cells) != names:
        raise ValueError("alignment cell inventory differs")
    parsed = {name: _retrieval(cells[name], summarized=False) for name in sorted(names)}
    if len({cast(int, cell["queries"]) for cell in parsed.values()}) != 1:
        raise ValueError("alignment query count differs")
    parsed_fidelity = _fidelity(fidelity)
    query_count = cast(int, next(iter(parsed.values()))["queries"])
    _validate_fidelity_cardinality(parsed_fidelity, query_count=query_count)
    if (
        type(orthogonality_error) is not float
        or not math.isfinite(orthogonality_error)
        or not 0.0 <= orthogonality_error <= 1e-8
    ):
        raise ValueError("alignment matrix authority differs")
    result = {
        "schema": "sfora-siglip-gallery-compatibility-alignment-v1",
        "claim_eligible": False,
        "external_evaluation_access": False,
        "checkpoint_sha256": _hex_digest(checkpoint_sha256),
        "spatial_artifact_sha256": _hex_digest(spatial_artifact_sha256),
        "alignment_artifact_sha256": _hex_digest(alignment_artifact_sha256),
        "fit_labels": list(fit_labels),
        "development_labels": list(development_labels),
        "cells": parsed,
        "fidelity": parsed_fidelity,
        "orthogonality_error": orthogonality_error,
        "self_geometry_max_score_delta": self_geometry_max_score_delta,
        "classification": _alignment_decision(
            parsed, parsed_fidelity, self_geometry_max_score_delta
        ),
    }
    raw = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    validate_alignment_result_bytes(raw)
    return raw


def validate_alignment_result_bytes(raw: bytes) -> dict[str, object]:
    """Validate canonical alignment evidence and independently recompute its decision."""

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as parse_error:
        raise ValueError("alignment result is not JSON") from parse_error
    keys = {
        "schema",
        "claim_eligible",
        "external_evaluation_access",
        "checkpoint_sha256",
        "spatial_artifact_sha256",
        "alignment_artifact_sha256",
        "fit_labels",
        "development_labels",
        "cells",
        "fidelity",
        "orthogonality_error",
        "self_geometry_max_score_delta",
        "classification",
    }
    if type(raw) is not bytes or type(value) is not dict or set(value) != keys:
        raise ValueError("alignment result schema differs")
    result = cast(dict[str, object], value)
    if (
        result["schema"] != "sfora-siglip-gallery-compatibility-alignment-v1"
        or result["claim_eligible"] is not False
        or result["external_evaluation_access"] is not False
    ):
        raise ValueError("alignment result authority differs")
    for name in ("checkpoint_sha256", "spatial_artifact_sha256", "alignment_artifact_sha256"):
        _hex_digest(result[name])
    expected_fit, expected_development = spatial_tail_class_split(tuple(range(49)))
    if result["fit_labels"] != sorted(expected_fit) or result["development_labels"] != sorted(
        expected_development
    ):
        raise ValueError("alignment class authority differs")
    names = {
        "teacher-teacher",
        "student-student",
        "student-teacher",
        "teacher-student",
        "aligned-aligned",
        "aligned-teacher",
        "teacher-aligned",
    }
    if type(result["cells"]) is not dict or set(cast(dict[str, object], result["cells"])) != names:
        raise ValueError("alignment cell inventory differs")
    cells = {
        name: _retrieval(cast(dict[str, object], result["cells"])[name], summarized=True)
        for name in names
    }
    if len({cast(int, cell["queries"]) for cell in cells.values()}) != 1:
        raise ValueError("alignment query count differs")
    fidelity = _parsed_fidelity(result["fidelity"])
    query_count = cast(int, next(iter(cells.values()))["queries"])
    _validate_fidelity_cardinality(fidelity, query_count=query_count)
    error = result["orthogonality_error"]
    if type(error) is not float or not math.isfinite(error) or not 0.0 <= error <= 1e-8:
        raise ValueError("alignment matrix authority differs")
    decision = _alignment_decision(
        cells,
        fidelity,
        cast(float, result["self_geometry_max_score_delta"]),
    )
    if result["classification"] != decision:
        raise ValueError("alignment decision differs")
    try:
        canonical = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as error_value:
        raise ValueError("alignment result is not canonical") from error_value
    if raw != canonical:
        raise ValueError("alignment result is not canonical")
    return result
