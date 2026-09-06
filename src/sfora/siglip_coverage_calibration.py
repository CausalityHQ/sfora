"""Coverage-calibrated affine compatibility for SigLIP descriptors."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import cast

import torch

from sfora.siglip_compatibility_capacity import (
    CompatibilityRetrievalEvidence,
    _evidence_mapping,
    _parse_evidence,
    compatibility_retrieval_evidence,
)


def _has_unit_rows(value: torch.Tensor) -> bool:
    norms = torch.linalg.vector_norm(value.double(), dim=1)
    return bool(torch.isfinite(norms).all()) and bool((torch.abs(norms - 1.0) <= 2e-5).all())


def coverage_support_indexes(
    ids: tuple[str, ...],
    labels: tuple[int, ...],
    *,
    support_per_class: int = 8,
) -> tuple[int, ...]:
    """Select the frozen balanced support set without reading descriptors."""

    if (
        type(ids) is not tuple
        or type(labels) is not tuple
        or len(ids) != len(labels)
        or not ids
        or len(set(ids)) != len(ids)
        or any(type(value) is not str or not value for value in ids)
        or any(type(value) is not int or value < 0 for value in labels)
        or type(support_per_class) is not int
        or support_per_class != 8
    ):
        raise ValueError("coverage support authority differs")
    selected: list[int] = []
    for label in sorted(set(labels)):
        candidates = [index for index, value in enumerate(labels) if value == label]
        if len(candidates) < 10:
            raise ValueError("coverage support authority differs")
        ranked = sorted(
            (
                hashlib.sha256(
                    b"sfora-coverage-support-v1\0"
                    + str(label).encode("ascii")
                    + b"\0"
                    + ids[index].encode("utf-8")
                ).digest(),
                ids[index],
                index,
            )
            for index in candidates
        )
        selected.extend(index for _digest, _id, index in ranked[:support_per_class])
    return tuple(sorted(selected, key=ids.__getitem__))


@dataclass(frozen=True, slots=True)
class CoverageAffine:
    """A full-rank FP64 affine with a normalized FP32 application boundary."""

    weight: torch.Tensor
    bias: torch.Tensor
    rank: int
    singular_values: torch.Tensor
    rcond: float
    driver: str

    def __post_init__(self) -> None:
        dimensions = self.weight.shape[0] if type(self.weight) is torch.Tensor else -1
        if (
            type(self.weight) is not torch.Tensor
            or type(self.bias) is not torch.Tensor
            or type(self.singular_values) is not torch.Tensor
            or self.weight.device.type != "cpu"
            or self.bias.device.type != "cpu"
            or self.singular_values.device.type != "cpu"
            or self.weight.dtype != torch.float64
            or self.bias.dtype != torch.float64
            or self.singular_values.dtype != torch.float64
            or self.weight.ndim != 2
            or dimensions < 2
            or self.weight.shape != (dimensions, dimensions)
            or self.bias.shape != (dimensions,)
            or self.singular_values.shape != (dimensions + 1,)
            or not bool(torch.isfinite(self.weight).all())
            or not bool(torch.isfinite(self.bias).all())
            or not bool(torch.isfinite(self.singular_values).all())
            or bool((self.singular_values <= 0).any())
            or type(self.rank) is not int
            or self.rank != dimensions + 1
            or type(self.rcond) is not float
            or self.rcond != 1e-12
            or type(self.driver) is not str
            or self.driver != "gelsd"
        ):
            raise ValueError("coverage affine authority differs")

    def apply(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Apply the affine in FP64 and return finite normalized FP32 rows."""

        if (
            type(descriptors) is not torch.Tensor
            or descriptors.device.type != "cpu"
            or descriptors.dtype != torch.float32
            or descriptors.ndim != 2
            or descriptors.shape[0] < 1
            or descriptors.shape[1] != self.weight.shape[0]
            or not bool(torch.isfinite(descriptors).all())
            or not _has_unit_rows(descriptors)
        ):
            raise ValueError("coverage affine authority differs")
        mapped = descriptors.double() @ self.weight + self.bias
        scale = torch.amax(torch.abs(mapped), dim=1, keepdim=True)
        if (
            not bool(torch.isfinite(mapped).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any())
        ):
            raise ValueError("coverage affine output differs")
        scaled = mapped / scale
        norms = torch.linalg.vector_norm(scaled, dim=1, keepdim=True)
        result: torch.Tensor = (scaled / norms).float().contiguous()
        if not bool(torch.isfinite(result).all()) or not _has_unit_rows(result):
            raise ValueError("coverage affine output differs")
        return result


def fit_coverage_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    ids: tuple[str, ...],
) -> CoverageAffine:
    """Fit the frozen minimum-norm ordinary least-squares affine."""

    if (
        type(source) is not torch.Tensor
        or type(target) is not torch.Tensor
        or source.device.type != "cpu"
        or target.device.type != "cpu"
        or source.dtype != torch.float32
        or target.dtype != torch.float32
        or source.ndim != 2
        or target.shape != source.shape
        or source.shape[1] < 2
        or source.shape[0] < source.shape[1] + 1
        or not bool(torch.isfinite(source).all())
        or not bool(torch.isfinite(target).all())
        or not _has_unit_rows(source)
        or not _has_unit_rows(target)
        or type(ids) is not tuple
        or len(ids) != source.shape[0]
        or len(set(ids)) != len(ids)
        or any(type(value) is not str or not value for value in ids)
    ):
        raise ValueError("coverage affine authority differs")
    order = torch.tensor(sorted(range(len(ids)), key=ids.__getitem__), dtype=torch.int64)
    ordered_source = source[order].double()
    ordered_target = target[order].double()
    augmented = torch.cat(
        (ordered_source, torch.ones(source.shape[0], 1, dtype=torch.float64)), dim=1
    )
    solved = torch.linalg.lstsq(
        augmented,
        ordered_target,
        rcond=1e-12,
        driver="gelsd",
    )
    rank = int(solved.rank.item())
    dimensions = source.shape[1]
    if rank != dimensions + 1 or solved.singular_values.shape != (dimensions + 1,):
        raise ValueError("coverage affine authority differs")
    solution = solved.solution
    return CoverageAffine(
        weight=solution[:-1].contiguous(),
        bias=solution[-1].contiguous(),
        rank=rank,
        singular_values=solved.singular_values.contiguous(),
        rcond=1e-12,
        driver="gelsd",
    )


@dataclass(frozen=True, slots=True)
class CoverageMaps:
    """The separately fitted online and offline compatibility directions."""

    student_to_teacher: CoverageAffine
    teacher_to_student: CoverageAffine

    def __post_init__(self) -> None:
        if (
            type(self.student_to_teacher) is not CoverageAffine
            or type(self.teacher_to_student) is not CoverageAffine
            or self.student_to_teacher.weight.shape != self.teacher_to_student.weight.shape
        ):
            raise ValueError("coverage maps authority differs")


_COVERAGE_CELL_NAMES = {
    "identity-forward",
    "identity-reverse",
    "student-self",
    "teacher-self",
    "forward-student-query",
    "forward-teacher-query",
    "offline-student-query",
    "offline-teacher-query",
}


def coverage_retrieval_cells(
    student: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    labels: tuple[int, ...],
    maps: CoverageMaps,
) -> dict[str, CompatibilityRetrievalEvidence]:
    """Evaluate native, forward-map, and offline-gallery compatibility cells."""

    if type(maps) is not CoverageMaps:
        raise ValueError("coverage maps authority differs")
    mapped_student = maps.student_to_teacher.apply(student)
    mapped_teacher = maps.teacher_to_student.apply(teacher)

    def evidence(query: torch.Tensor, gallery: torch.Tensor) -> CompatibilityRetrievalEvidence:
        return compatibility_retrieval_evidence(
            query,
            gallery,
            query_ids=ids,
            gallery_ids=ids,
            query_labels=labels,
            gallery_labels=labels,
            reference_query=teacher,
            reference_gallery=teacher,
        )

    return {
        "identity-forward": evidence(student, teacher),
        "identity-reverse": evidence(teacher, student),
        "student-self": evidence(student, student),
        "teacher-self": evidence(teacher, teacher),
        "forward-student-query": evidence(mapped_student, teacher),
        "forward-teacher-query": evidence(teacher, mapped_student),
        "offline-student-query": evidence(student, mapped_teacher),
        "offline-teacher-query": evidence(mapped_teacher, student),
    }


def _coverage_cell_passes(value: CompatibilityRetrievalEvidence) -> bool:
    return value.micro_r1 >= 0.97 and value.micro_map_at_r >= 0.95


def coverage_calibration_classification(
    cells: dict[str, CompatibilityRetrievalEvidence],
) -> str:
    """Classify the frozen four-direction compatibility gate."""

    if (
        type(cells) is not dict
        or set(cells) != _COVERAGE_CELL_NAMES
        or any(type(value) is not CompatibilityRetrievalEvidence for value in cells.values())
    ):
        raise ValueError("coverage retrieval authority differs")
    if not _coverage_cell_passes(cells["student-self"]):
        return "student-quality-rejected"
    forward = all(
        _coverage_cell_passes(cells[name])
        for name in ("forward-student-query", "forward-teacher-query")
    )
    offline = all(
        _coverage_cell_passes(cells[name])
        for name in ("offline-student-query", "offline-teacher-query")
    )
    if forward and offline:
        return "coverage-calibration-qualified"
    if offline:
        return "offline-gallery-qualified"
    if forward:
        return "forward-only-qualified"
    return "coverage-calibration-rejected"


@dataclass(frozen=True, slots=True)
class CoverageCalibrationRun:
    """Sealed support split, maps, and support-disjoint retrieval evidence."""

    maps: CoverageMaps
    support_ids: tuple[str, ...]
    support_labels: tuple[int, ...]
    evaluation_ids: tuple[str, ...]
    cells: dict[str, CompatibilityRetrievalEvidence]
    classification: str

    def __post_init__(self) -> None:
        if (
            type(self.maps) is not CoverageMaps
            or type(self.support_ids) is not tuple
            or type(self.evaluation_ids) is not tuple
            or not self.support_ids
            or type(self.support_labels) is not tuple
            or len(self.support_labels) != len(self.support_ids)
            or any(type(value) is not int or value < 0 for value in self.support_labels)
            or any(self.support_labels.count(label) != 8 for label in set(self.support_labels))
            or not self.evaluation_ids
            or set(self.support_ids) & set(self.evaluation_ids)
            or tuple(sorted(self.support_ids)) != self.support_ids
            or tuple(sorted(self.evaluation_ids)) != self.evaluation_ids
            or type(self.cells) is not dict
            or self.classification != coverage_calibration_classification(self.cells)
            or any(value.ids != self.evaluation_ids for value in self.cells.values())
        ):
            raise ValueError("coverage calibration run authority differs")


def analyze_coverage_calibration(
    base_student: torch.Tensor,
    base_teacher: torch.Tensor,
    base_ids: tuple[str, ...],
    candidate_student: torch.Tensor,
    candidate_teacher: torch.Tensor,
    candidate_ids: tuple[str, ...],
    candidate_labels: tuple[int, ...],
) -> CoverageCalibrationRun:
    """Fit on base plus frozen support and score only the remaining candidates."""

    if (
        type(base_student) is not torch.Tensor
        or type(base_teacher) is not torch.Tensor
        or base_student.device.type != "cpu"
        or base_teacher.device.type != "cpu"
        or base_student.dtype != torch.float32
        or base_teacher.dtype != torch.float32
        or base_student.ndim != 2
        or base_teacher.shape != base_student.shape
        or base_student.shape[1] < 2
        or base_student.shape[0] < base_student.shape[1] + 1
        or type(base_ids) is not tuple
        or len(base_ids) != base_student.shape[0]
        or len(set(base_ids)) != len(base_ids)
        or any(type(value) is not str or not value for value in base_ids)
        or not bool(torch.isfinite(base_student).all())
        or not bool(torch.isfinite(base_teacher).all())
        or bool((torch.linalg.vector_norm(base_student, dim=1) <= 0).any())
        or bool((torch.linalg.vector_norm(base_teacher, dim=1) <= 0).any())
        or type(candidate_student) is not torch.Tensor
        or type(candidate_teacher) is not torch.Tensor
        or candidate_student.device.type != "cpu"
        or candidate_teacher.device.type != "cpu"
        or candidate_student.dtype != torch.float32
        or candidate_teacher.dtype != torch.float32
        or candidate_student.ndim != 2
        or candidate_teacher.shape != candidate_student.shape
        or candidate_student.shape[1] != base_student.shape[1]
        or candidate_student.shape[0] != len(candidate_ids)
        or type(candidate_ids) is not tuple
        or type(candidate_labels) is not tuple
        or len(candidate_labels) != len(candidate_ids)
        or set(base_ids) & set(candidate_ids)
        or not bool(torch.isfinite(candidate_student).all())
        or not bool(torch.isfinite(candidate_teacher).all())
        or bool((torch.linalg.vector_norm(candidate_student, dim=1) <= 0).any())
        or bool((torch.linalg.vector_norm(candidate_teacher, dim=1) <= 0).any())
    ):
        raise ValueError("coverage calibration input authority differs")
    support_indexes = coverage_support_indexes(candidate_ids, candidate_labels)
    support = set(support_indexes)
    evaluation_indexes = tuple(
        sorted(
            (index for index in range(len(candidate_ids)) if index not in support),
            key=candidate_ids.__getitem__,
        )
    )
    support_student = candidate_student[list(support_indexes)]
    support_teacher = candidate_teacher[list(support_indexes)]
    fitting_student = torch.cat((base_student, support_student), dim=0).contiguous()
    fitting_teacher = torch.cat((base_teacher, support_teacher), dim=0).contiguous()
    fitting_ids = (*base_ids, *(candidate_ids[index] for index in support_indexes))
    maps = CoverageMaps(
        student_to_teacher=fit_coverage_affine(fitting_student, fitting_teacher, fitting_ids),
        teacher_to_student=fit_coverage_affine(fitting_teacher, fitting_student, fitting_ids),
    )
    evaluation_ids = tuple(candidate_ids[index] for index in evaluation_indexes)
    cells = coverage_retrieval_cells(
        candidate_student[list(evaluation_indexes)].contiguous(),
        candidate_teacher[list(evaluation_indexes)].contiguous(),
        evaluation_ids,
        tuple(candidate_labels[index] for index in evaluation_indexes),
        maps,
    )
    return CoverageCalibrationRun(
        maps=maps,
        support_ids=tuple(sorted(candidate_ids[index] for index in support_indexes)),
        support_labels=tuple(
            candidate_labels[index]
            for index in sorted(support_indexes, key=candidate_ids.__getitem__)
        ),
        evaluation_ids=evaluation_ids,
        cells=cells,
        classification=coverage_calibration_classification(cells),
    )


_RESULT_INPUT_ROLES = {
    "checkpoint",
    "control_binding",
    "evaluation_manifest",
    "map_artifact",
    "optimization_manifest",
    "spatial_artifact",
}


def _coverage_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("coverage calibration result digest differs")
    return value


@dataclass(frozen=True, slots=True)
class CoverageArtifactIdentity:
    """One authenticated local regular-file input or output."""

    path: str
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or not self.path.startswith("/")
            or type(self.byte_length) is not int
            or self.byte_length < 1
        ):
            raise ValueError("coverage calibration result input differs")
        _coverage_digest(self.sha256)

    def to_mapping(self) -> dict[str, object]:
        """Return the exact JSON representation."""

        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }


def _solver_mapping(value: CoverageAffine) -> dict[str, object]:
    return {
        "dimensions": value.weight.shape[0],
        "rank": value.rank,
        "singular_values": [float(entry) for entry in value.singular_values],
        "rcond": value.rcond,
        "driver": value.driver,
    }


def build_coverage_calibration_result(
    run: CoverageCalibrationRun,
    *,
    artifact_identities: dict[str, CoverageArtifactIdentity],
    model_source_commit: str,
    execution_source_commit: str,
    dataset_id: str,
    dataset_revision: str,
    optimization_image_root: str,
    evaluation_image_root: str,
    optimization_images_sha256: str,
    support_images_sha256: str,
    evaluation_images_sha256: str,
    optimization_rows: int,
    torch_version: str,
    torch_num_threads: int,
    blas_config: str,
    cpu_identity: str,
) -> bytes:
    """Build canonical claim-ineligible evidence and validate it before return."""

    if type(run) is not CoverageCalibrationRun:
        raise ValueError("coverage calibration result authority differs")
    if (
        type(artifact_identities) is not dict
        or set(artifact_identities) != _RESULT_INPUT_ROLES
        or any(
            type(value) is not CoverageArtifactIdentity for value in artifact_identities.values()
        )
    ):
        raise ValueError("coverage calibration result inputs differ")
    value: dict[str, object] = {
        "schema": "sfora-siglip-coverage-calibration-v1",
        "claim_eligible": False,
        "preprocessing": "siglip-evaluation-transform-v1",
        "support_per_class": 8,
        "thresholds": {"micro_map_at_r": 0.95, "micro_r1": 0.97},
        "inputs": {
            role: artifact_identities[role].to_mapping() for role in sorted(artifact_identities)
        },
        "model_source_commit": model_source_commit,
        "execution_source_commit": execution_source_commit,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "optimization_image_root": optimization_image_root,
        "evaluation_image_root": evaluation_image_root,
        "image_namespaces": {
            "optimization": {
                "sha256": _coverage_digest(optimization_images_sha256),
                "rows": optimization_rows,
            },
            "support": {
                "sha256": _coverage_digest(support_images_sha256),
                "rows": len(run.support_ids),
            },
            "evaluation": {
                "sha256": _coverage_digest(evaluation_images_sha256),
                "rows": len(run.evaluation_ids),
            },
        },
        "support_ids": list(run.support_ids),
        "support_labels": list(run.support_labels),
        "evaluation_ids": list(run.evaluation_ids),
        "solver": {
            "student_to_teacher": _solver_mapping(run.maps.student_to_teacher),
            "teacher_to_student": _solver_mapping(run.maps.teacher_to_student),
        },
        "environment": {
            "torch_version": torch_version,
            "torch_num_threads": torch_num_threads,
            "blas_config": blas_config,
            "cpu_identity": cpu_identity,
        },
        "cells": {name: _evidence_mapping(run.cells[name]) for name in sorted(run.cells)},
        "classification": run.classification,
        "offline_query_encoder_extra_ops": 0,
    }
    try:
        raw = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("coverage calibration result authority differs") from error
    validate_coverage_calibration_result_bytes(raw)
    return raw


def _validate_solver_evidence(value: object) -> int:
    keys = {"dimensions", "rank", "singular_values", "rcond", "driver"}
    if type(value) is not dict or set(value) != keys:
        raise ValueError("coverage calibration result solver differs")
    item = cast(dict[str, object], value)
    dimensions = item["dimensions"]
    rank = item["rank"]
    singular = item["singular_values"]
    if (
        type(dimensions) is not int
        or dimensions < 2
        or type(rank) is not int
        or rank != dimensions + 1
        or type(singular) is not list
        or len(singular) != dimensions + 1
        or any(
            type(entry) is not float or not math.isfinite(entry) or entry <= 0 for entry in singular
        )
        or type(item["rcond"]) is not float
        or item["rcond"] != 1e-12
        or type(item["driver"]) is not str
        or item["driver"] != "gelsd"
    ):
        raise ValueError("coverage calibration result solver differs")
    return dimensions


def validate_coverage_calibration_result_bytes(raw: bytes) -> dict[str, object]:
    """Validate canonical bytes and independently reconstruct all result decisions."""

    try:
        value = json.loads(raw)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("coverage calibration result is not JSON") from error
    keys = {
        "schema",
        "claim_eligible",
        "preprocessing",
        "support_per_class",
        "thresholds",
        "inputs",
        "support_ids",
        "support_labels",
        "evaluation_ids",
        "solver",
        "environment",
        "cells",
        "classification",
        "offline_query_encoder_extra_ops",
        "model_source_commit",
        "execution_source_commit",
        "dataset_id",
        "dataset_revision",
        "optimization_image_root",
        "evaluation_image_root",
        "image_namespaces",
    }
    if type(raw) is not bytes or type(value) is not dict or set(value) != keys:
        raise ValueError("coverage calibration result schema differs")
    result = cast(dict[str, object], value)
    try:
        canonical = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("coverage calibration result bytes differ") from error
    if raw != canonical:
        raise ValueError("coverage calibration result bytes differ")
    if (
        result["schema"] != "sfora-siglip-coverage-calibration-v1"
        or type(result["claim_eligible"]) is not bool
        or result["claim_eligible"] is not False
        or result["preprocessing"] != "siglip-evaluation-transform-v1"
        or type(result["support_per_class"]) is not int
        or result["support_per_class"] != 8
        or type(result["offline_query_encoder_extra_ops"]) is not int
        or result["offline_query_encoder_extra_ops"] != 0
    ):
        raise ValueError("coverage calibration result authority differs")
    thresholds = result["thresholds"]
    if type(thresholds) is not dict or thresholds != {
        "micro_map_at_r": 0.95,
        "micro_r1": 0.97,
    }:
        raise ValueError("coverage calibration result thresholds differ")
    if (
        type(result["model_source_commit"]) is not str
        or len(result["model_source_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in result["model_source_commit"])
        or type(result["execution_source_commit"]) is not str
        or len(result["execution_source_commit"]) != 40
        or any(
            character not in "0123456789abcdef" for character in result["execution_source_commit"]
        )
        or type(result["dataset_id"]) is not str
        or not result["dataset_id"]
        or type(result["dataset_revision"]) is not str
        or len(result["dataset_revision"]) != 40
        or any(character not in "0123456789abcdef" for character in result["dataset_revision"])
        or type(result["optimization_image_root"]) is not str
        or not result["optimization_image_root"].startswith("/")
        or type(result["evaluation_image_root"]) is not str
        or not result["evaluation_image_root"].startswith("/")
    ):
        raise ValueError("coverage calibration result identity differs")
    inputs = result["inputs"]
    if type(inputs) is not dict or set(inputs) != _RESULT_INPUT_ROLES:
        raise ValueError("coverage calibration result inputs differ")
    try:
        for identity in inputs.values():
            if type(identity) is not dict or set(identity) != {
                "path",
                "sha256",
                "byte_length",
            }:
                raise ValueError("coverage calibration result input differs")
            CoverageArtifactIdentity(
                path=identity["path"],
                sha256=identity["sha256"],
                byte_length=identity["byte_length"],
            )
    except (TypeError, ValueError) as error:
        raise ValueError("coverage calibration result inputs differ") from error
    namespaces = result["image_namespaces"]
    if type(namespaces) is not dict or set(namespaces) != {
        "optimization",
        "support",
        "evaluation",
    }:
        raise ValueError("coverage calibration result image namespaces differ")
    expected_rows = {
        "support": len(cast(list[object], result["support_ids"])),
        "evaluation": len(cast(list[object], result["evaluation_ids"])),
    }
    for role, namespace in namespaces.items():
        if (
            type(namespace) is not dict
            or set(namespace) != {"sha256", "rows"}
            or type(namespace["rows"]) is not int
            or namespace["rows"] < 1
            or (role in expected_rows and namespace["rows"] != expected_rows[role])
        ):
            raise ValueError("coverage calibration result image namespaces differ")
        _coverage_digest(namespace["sha256"])
    support_ids = result["support_ids"]
    support_labels = result["support_labels"]
    evaluation_ids = result["evaluation_ids"]
    if (
        type(support_ids) is not list
        or not support_ids
        or len(support_ids) % 8 != 0
        or any(type(entry) is not str or not entry for entry in support_ids)
        or support_ids != sorted(support_ids)
        or len(set(support_ids)) != len(support_ids)
        or type(support_labels) is not list
        or len(support_labels) != len(support_ids)
        or any(type(entry) is not int or entry < 0 for entry in support_labels)
        or any(support_labels.count(label) != 8 for label in set(support_labels))
        or type(evaluation_ids) is not list
        or len(evaluation_ids) < 2
        or any(type(entry) is not str or not entry for entry in evaluation_ids)
        or evaluation_ids != sorted(evaluation_ids)
        or len(set(evaluation_ids)) != len(evaluation_ids)
        or set(support_ids) & set(evaluation_ids)
    ):
        raise ValueError("coverage calibration result identities differ")
    solver = result["solver"]
    if type(solver) is not dict or set(solver) != {
        "student_to_teacher",
        "teacher_to_student",
    }:
        raise ValueError("coverage calibration result solver differs")
    dimensions = {
        _validate_solver_evidence(solver[direction])
        for direction in ("student_to_teacher", "teacher_to_student")
    }
    if len(dimensions) != 1:
        raise ValueError("coverage calibration result solver differs")
    environment = result["environment"]
    if (
        type(environment) is not dict
        or set(environment) != {"torch_version", "torch_num_threads", "blas_config", "cpu_identity"}
        or type(environment["torch_version"]) is not str
        or not environment["torch_version"]
        or type(environment["torch_num_threads"]) is not int
        or environment["torch_num_threads"] < 1
        or type(environment["blas_config"]) is not str
        or not environment["blas_config"]
        or type(environment["cpu_identity"]) is not str
        or not environment["cpu_identity"]
    ):
        raise ValueError("coverage calibration result environment differs")
    encoded_cells = result["cells"]
    if type(encoded_cells) is not dict or set(encoded_cells) != _COVERAGE_CELL_NAMES:
        raise ValueError("coverage calibration result cells differ")
    try:
        cells = {name: _parse_evidence(encoded_cells[name]) for name in _COVERAGE_CELL_NAMES}
    except ValueError as error:
        raise ValueError("coverage calibration result cells differ") from error
    expected_ids = tuple(cast(list[str], evaluation_ids))
    expected_labels = cells["student-self"].labels
    if any(
        evidence.ids != expected_ids or evidence.labels != expected_labels
        for evidence in cells.values()
    ):
        raise ValueError("coverage calibration result cells differ")
    classification = coverage_calibration_classification(cells)
    if type(result["classification"]) is not str or result["classification"] != classification:
        raise ValueError("coverage calibration result classification differs")
    return result
