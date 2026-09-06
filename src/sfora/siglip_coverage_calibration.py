"""Coverage-calibrated affine compatibility for SigLIP descriptors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sfora.siglip_compatibility_capacity import (
    CompatibilityRetrievalEvidence,
    compatibility_retrieval_evidence,
)


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
        ):
            raise ValueError("coverage affine authority differs")
        mapped = descriptors.double() @ self.weight + self.bias
        if not bool(torch.isfinite(mapped).all()) or bool(
            (torch.linalg.vector_norm(mapped, dim=1) <= 0).any()
        ):
            raise ValueError("coverage affine output differs")
        result = F.normalize(mapped, dim=1).float().contiguous()
        if not bool(torch.isfinite(result).all()):
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
    return "coverage-calibration-rejected"
