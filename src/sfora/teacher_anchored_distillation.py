"""Dataset-agnostic schedules and objectives for teacher-anchored distillation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

type SampleId = str | int

_DIMENSIONS = 128
_NEAREST_ANCHORS = 64
_MIDDLE_ANCHORS = 64
_UNIFORM_ANCHORS = 384
_ANCHORS = _NEAREST_ANCHORS + _MIDDLE_ANCHORS + _UNIFORM_ANCHORS
_SEEDS_PER_BATCH = 128


@dataclass(frozen=True, slots=True)
class TeacherAnchoredConfig:
    """The frozen dataset-independent teacher-anchored recipe."""

    dimensions: int = _DIMENSIONS
    temperatures: tuple[float, float] = (0.05, 0.20)
    nearest_anchor_count: int = _NEAREST_ANCHORS
    middle_anchor_count: int = _MIDDLE_ANCHORS
    uniform_anchor_count: int = _UNIFORM_ANCHORS
    seed_rows_per_batch: int = _SEEDS_PER_BATCH
    anchor_weight: float = 1.0
    point_weight: float = 0.1
    symmetric_weight: float = 0.5
    drift_weight: float = 0.05
    covariance_weight: float = 0.01

    def __post_init__(self) -> None:
        expected: tuple[tuple[object, object, type[object]], ...] = (
            (self.dimensions, _DIMENSIONS, int),
            (self.temperatures, (0.05, 0.20), tuple),
            (self.nearest_anchor_count, _NEAREST_ANCHORS, int),
            (self.middle_anchor_count, _MIDDLE_ANCHORS, int),
            (self.uniform_anchor_count, _UNIFORM_ANCHORS, int),
            (self.seed_rows_per_batch, _SEEDS_PER_BATCH, int),
            (self.anchor_weight, 1.0, float),
            (self.point_weight, 0.1, float),
            (self.symmetric_weight, 0.5, float),
            (self.drift_weight, 0.05, float),
            (self.covariance_weight, 0.01, float),
        )
        if any(type(value) is not kind or value != wanted for value, wanted, kind in expected):
            raise ValueError("teacher-anchored configuration differs")
        if any(type(value) is not float or not math.isfinite(value) for value in self.temperatures):
            raise ValueError("teacher-anchored configuration differs")

    @property
    def anchor_count(self) -> int:
        """Return the complete anchor width."""

        return self.nearest_anchor_count + self.middle_anchor_count + self.uniform_anchor_count

    @property
    def batch_rows(self) -> int:
        """Return the seed-plus-partner batch width."""

        return self.seed_rows_per_batch * 2


@dataclass(frozen=True, init=False, slots=True)
class TeacherAnchorSchedule:
    """Immutable row indexes and authority digest for teacher anchors."""

    _row_indexes: NDArray[np.int64]
    sha256: str

    def __init__(self, *, row_indexes: NDArray[np.int64], sha256: str) -> None:
        frozen = np.ascontiguousarray(row_indexes, dtype=np.int64).copy()
        frozen.flags.writeable = False
        object.__setattr__(self, "_row_indexes", frozen)
        object.__setattr__(self, "sha256", sha256)

    @property
    def row_indexes(self) -> NDArray[np.int64]:
        """Return a read-only defensive copy of the anchor indexes."""

        result = self._row_indexes.copy()
        result.flags.writeable = False
        return result


@dataclass(frozen=True, init=False, slots=True)
class TeacherNeighborBatches:
    """Immutable structured batches for one epoch."""

    _row_indexes: NDArray[np.int64]
    dropped_seed_row_indexes: tuple[int, ...]
    repeated_identity_count: int
    sha256: str

    def __init__(
        self,
        *,
        row_indexes: NDArray[np.int64],
        dropped_seed_row_indexes: tuple[int, ...],
        repeated_identity_count: int,
        sha256: str,
    ) -> None:
        frozen = np.ascontiguousarray(row_indexes, dtype=np.int64).copy()
        frozen.flags.writeable = False
        object.__setattr__(self, "_row_indexes", frozen)
        object.__setattr__(self, "dropped_seed_row_indexes", dropped_seed_row_indexes)
        object.__setattr__(self, "repeated_identity_count", repeated_identity_count)
        object.__setattr__(self, "sha256", sha256)

    @property
    def row_indexes(self) -> NDArray[np.int64]:
        """Return a read-only defensive copy of the batch indexes."""

        result = self._row_indexes.copy()
        result.flags.writeable = False
        return result


@dataclass(frozen=True, init=False, slots=True)
class TeacherNeighborRanking:
    """Reusable exact nearest rows for collision-safe epoch batching."""

    _row_indexes: NDArray[np.int64]
    input_sha256: str
    sha256: str

    def __init__(
        self,
        *,
        row_indexes: NDArray[np.int64],
        input_sha256: str,
        sha256: str,
    ) -> None:
        frozen = np.ascontiguousarray(row_indexes, dtype=np.int64).copy()
        frozen.flags.writeable = False
        object.__setattr__(self, "_row_indexes", frozen)
        object.__setattr__(self, "input_sha256", input_sha256)
        object.__setattr__(self, "sha256", sha256)

    @property
    def row_indexes(self) -> NDArray[np.int64]:
        """Return a read-only defensive copy of the nearest-row ranks."""

        result = self._row_indexes.copy()
        result.flags.writeable = False
        return result


@dataclass(frozen=True, slots=True)
class TeacherAnchoredLoss:
    """Named differentiable terms of the frozen objective."""

    total: torch.Tensor
    anchor: torch.Tensor
    point: torch.Tensor
    symmetric: torch.Tensor
    drift: torch.Tensor
    covariance: torch.Tensor


class TeacherAnchoredNumericalError(ValueError):
    """A nonfinite objective input or result, distinct from authority drift."""


@dataclass(frozen=True, slots=True)
class EmbeddingGeometryDiagnostics:
    """Finite collapse diagnostics computed from normalized codes."""

    effective_rank: float
    leading_eigenvalue_share: float
    top_eight_eigenvalue_share: float


def teacher_anchor_schedule(
    teacher_codes: NDArray[np.float32], sample_ids: tuple[SampleId, ...], *, seed: int
) -> TeacherAnchorSchedule:
    """Build the frozen nearest, middle-rank, and uniform anchor schedule."""

    codes, identities = _validated_inputs(
        teacher_codes,
        sample_ids,
        seed=seed,
        minimum_rows=_ANCHORS + _UNIFORM_ANCHORS + 1,
        message="teacher anchor authority differs",
    )
    row_count = codes.shape[0]
    result = np.empty((row_count, _ANCHORS), dtype=np.int64)
    double_codes = codes.astype(np.float64)
    for block_start in range(0, row_count, 256):
        block_stop = min(block_start + 256, row_count)
        similarities = double_codes[block_start:block_stop] @ double_codes.T
        for local, query_row in enumerate(range(block_start, block_stop)):
            row = similarities[local]
            ranked = _bounded_top_rows(row, query_row=query_row, identities=identities, count=512)
            nearest = ranked[:_NEAREST_ANCHORS]
            middle_inventory = np.asarray(ranked[_NEAREST_ANCHORS:], dtype=np.int64)
            excluded = np.zeros(row_count, dtype=np.bool_)
            excluded[query_row] = True
            excluded[np.asarray(ranked, dtype=np.int64)] = True
            uniform_inventory = np.flatnonzero(~excluded).astype(np.int64, copy=False)
            generator = np.random.Generator(
                np.random.PCG64(
                    _domain_seed(
                        seed,
                        b"anchors",
                        _identity_bytes(
                            identities[query_row], message="teacher anchor authority differs"
                        ),
                    )
                )
            )
            middle = middle_inventory[
                generator.choice(len(middle_inventory), size=_MIDDLE_ANCHORS, replace=False)
            ]
            uniform = uniform_inventory[
                generator.choice(len(uniform_inventory), size=_UNIFORM_ANCHORS, replace=False)
            ]
            result[query_row] = np.concatenate(
                (np.asarray(nearest, dtype=np.int64), middle, uniform)
            )
    digest = _schedule_sha256(b"teacher-anchors-v1", codes, sample_ids, seed, result)
    return TeacherAnchorSchedule(row_indexes=result, sha256=digest)


def teacher_neighbor_ranking(
    teacher_codes: NDArray[np.float32], sample_ids: tuple[SampleId, ...]
) -> TeacherNeighborRanking:
    """Build one reusable bounded nearest-row authority for every epoch and arm."""

    codes, identities = _validated_inputs(
        teacher_codes,
        sample_ids,
        seed=0,
        minimum_rows=_SEEDS_PER_BATCH * 2,
        message="teacher neighbor authority differs",
    )
    row_count = codes.shape[0]
    retained = min(256, row_count - 1)
    result = np.empty((row_count, retained), dtype=np.int64)
    double_codes = codes.astype(np.float64)
    for block_start in range(0, row_count, 256):
        block_stop = min(block_start + 256, row_count)
        similarities = double_codes[block_start:block_stop] @ double_codes.T
        for local, query_row in enumerate(range(block_start, block_stop)):
            result[query_row] = np.asarray(
                _bounded_top_rows(
                    similarities[local],
                    query_row=query_row,
                    identities=identities,
                    count=retained,
                ),
                dtype=np.int64,
            )
    input_sha256 = _input_sha256(codes, sample_ids)
    digest = hashlib.sha256(b"teacher-neighbors-v1" + bytes.fromhex(input_sha256))
    digest.update(result.tobytes(order="C"))
    return TeacherNeighborRanking(
        row_indexes=result,
        input_sha256=input_sha256,
        sha256=digest.hexdigest(),
    )


def teacher_neighbor_batches(
    teacher_codes: NDArray[np.float32],
    sample_ids: tuple[SampleId, ...],
    *,
    seed: int,
    epoch: int,
    ranking: TeacherNeighborRanking | None = None,
) -> TeacherNeighborBatches:
    """Build deterministic seed-plus-nearest-partner batches for one epoch."""

    if type(epoch) is not int or epoch < 1:
        raise ValueError("teacher batch authority differs")
    codes, identities = _validated_inputs(
        teacher_codes,
        sample_ids,
        seed=seed,
        minimum_rows=_SEEDS_PER_BATCH * 2,
        message="teacher batch authority differs",
    )
    row_count = codes.shape[0]
    if ranking is None:
        ranking = teacher_neighbor_ranking(codes, sample_ids)
    if (
        type(ranking) is not TeacherNeighborRanking
        or ranking._row_indexes.shape != (row_count, min(256, row_count - 1))
        or ranking.input_sha256 != _input_sha256(codes, sample_ids)
    ):
        raise ValueError("teacher batch authority differs")
    permutation_generator = np.random.Generator(
        np.random.PCG64(_domain_seed(seed, b"batches", epoch.to_bytes(8, "little")))
    )
    permutation = permutation_generator.permutation(row_count)
    usable = row_count - row_count % _SEEDS_PER_BATCH
    seed_order = permutation[:usable]
    dropped = tuple(int(row) for row in permutation[usable:])
    result = np.empty((usable // _SEEDS_PER_BATCH, _SEEDS_PER_BATCH * 2), dtype=np.int64)
    for batch_index, start in enumerate(range(0, usable, _SEEDS_PER_BATCH)):
        seeds = [int(row) for row in seed_order[start : start + _SEEDS_PER_BATCH]]
        selected = set(seeds)
        partners: list[int] = []
        for source in seeds:
            try:
                partner = next(
                    int(row) for row in ranking._row_indexes[source] if int(row) not in selected
                )
            except StopIteration as error:
                raise ValueError("teacher batch authority differs") from error
            selected.add(partner)
            partners.append(partner)
        result[batch_index] = np.asarray((*seeds, *partners), dtype=np.int64)
    counts = np.bincount(result.reshape(-1), minlength=row_count)
    repeated = int(np.count_nonzero(counts > 1))
    digest = _schedule_sha256(
        b"teacher-batches-v1" + epoch.to_bytes(8, "little"),
        codes,
        sample_ids,
        seed,
        result,
    )
    return TeacherNeighborBatches(
        row_indexes=result,
        dropped_seed_row_indexes=dropped,
        repeated_identity_count=repeated,
        sha256=digest,
    )


def verify_teacher_anchor_schedule(
    teacher_codes: NDArray[np.float32],
    sample_ids: tuple[SampleId, ...],
    *,
    seed: int,
    schedule: TeacherAnchorSchedule,
) -> bool:
    """Verify a prebuilt anchor schedule without repeating neighbor selection."""

    codes, identities = _validated_inputs(
        teacher_codes,
        sample_ids,
        seed=seed,
        minimum_rows=_ANCHORS + _UNIFORM_ANCHORS + 1,
        message="teacher anchor authority differs",
    )
    if (
        type(schedule) is not TeacherAnchorSchedule
        or schedule._row_indexes.shape != (codes.shape[0], _ANCHORS)
        or np.any(schedule._row_indexes < 0)
        or np.any(schedule._row_indexes >= codes.shape[0])
        or schedule.sha256
        != _schedule_sha256(
            b"teacher-anchors-v1", codes, identities, seed, schedule._row_indexes
        )
    ):
        raise ValueError("teacher anchor authority differs")
    return True


def teacher_anchored_input_sha256(
    teacher_codes: NDArray[np.float32], sample_ids: tuple[SampleId, ...]
) -> str:
    """Hash one validated teacher-code row namespace without selecting neighbors."""

    codes, identities = _validated_inputs(
        teacher_codes,
        sample_ids,
        seed=0,
        minimum_rows=1,
        message="teacher input authority differs",
    )
    return _input_sha256(codes, identities)


def verify_teacher_neighbor_batches(
    teacher_codes: NDArray[np.float32],
    sample_ids: tuple[SampleId, ...],
    *,
    seed: int,
    epoch: int,
    schedule: TeacherNeighborBatches,
) -> bool:
    """Verify prebuilt epoch rows without repeating nearest-neighbor ranking."""

    if type(epoch) is not int or epoch < 1:
        raise ValueError("teacher batch authority differs")
    codes, identities = _validated_inputs(
        teacher_codes,
        sample_ids,
        seed=seed,
        minimum_rows=_SEEDS_PER_BATCH * 2,
        message="teacher batch authority differs",
    )
    expected_batches = codes.shape[0] // _SEEDS_PER_BATCH
    expected_usable = expected_batches * _SEEDS_PER_BATCH
    rows = schedule._row_indexes if type(schedule) is TeacherNeighborBatches else None
    dropped = schedule.dropped_seed_row_indexes if type(schedule) is TeacherNeighborBatches else ()
    permutation = np.random.Generator(
        np.random.PCG64(_domain_seed(seed, b"batches", epoch.to_bytes(8, "little")))
    ).permutation(codes.shape[0])
    if (
        type(schedule) is not TeacherNeighborBatches
        or rows is None
        or rows.shape != (expected_batches, _SEEDS_PER_BATCH * 2)
        or np.any(rows < 0)
        or np.any(rows >= codes.shape[0])
        or type(dropped) is not tuple
        or len(dropped) != codes.shape[0] - expected_usable
        or any(type(row) is not int or not 0 <= row < codes.shape[0] for row in dropped)
        or len(set(dropped)) != len(dropped)
        or any(len(np.unique(batch)) != _SEEDS_PER_BATCH * 2 for batch in rows)
        or len(np.unique(rows[:, :_SEEDS_PER_BATCH])) != expected_usable
        or not np.array_equal(
            rows[:, :_SEEDS_PER_BATCH].reshape(-1), permutation[:expected_usable]
        )
        or dropped != tuple(int(row) for row in permutation[expected_usable:])
        or set(rows[:, :_SEEDS_PER_BATCH].reshape(-1).tolist()).union(dropped)
        != set(range(codes.shape[0]))
        or type(schedule.repeated_identity_count) is not int
        or schedule.repeated_identity_count
        != int(np.count_nonzero(np.bincount(rows.reshape(-1), minlength=codes.shape[0]) > 1))
        or schedule.sha256
        != _schedule_sha256(
            b"teacher-batches-v1" + epoch.to_bytes(8, "little"),
            codes,
            identities,
            seed,
            rows,
        )
    ):
        raise ValueError("teacher batch authority differs")
    return True


def cross_dimensional_anchor_distillation_loss(
    student: torch.Tensor,
    student_anchors: torch.Tensor,
    teacher: torch.Tensor,
    teacher_anchors: torch.Tensor,
    *,
    temperatures: tuple[float, ...],
) -> torch.Tensor:
    """Match query-to-anchor relations across different embedding widths."""

    tensors = (student, student_anchors, teacher, teacher_anchors)
    if (
        any(type(value) is not torch.Tensor for value in tensors)
        or student.ndim != 2
        or teacher.ndim != 2
        or student_anchors.ndim != 3
        or teacher_anchors.ndim != 3
        or student.shape[0] < 1
        or student.shape[1] < 2
        or teacher.shape[1] < 2
        or student_anchors.shape[:2] != teacher_anchors.shape[:2]
        or student_anchors.shape[0] != student.shape[0]
        or teacher.shape[0] != student.shape[0]
        or student_anchors.shape[2] != student.shape[1]
        or teacher_anchors.shape[2] != teacher.shape[1]
        or student_anchors.shape[1] < 2
        or any(value.dtype != torch.float32 for value in tensors)
        or any(value.device != student.device for value in tensors)
        or any(not value.is_contiguous() for value in tensors)
        or not student.requires_grad
        or not student_anchors.requires_grad
        or teacher.requires_grad
        or teacher_anchors.requires_grad
        or type(temperatures) is not tuple
        or not temperatures
        or any(
            type(temperature) is not float
            or not math.isfinite(temperature)
            or temperature <= 0.0
            for temperature in temperatures
        )
    ):
        raise ValueError("cross-dimensional anchor authority differs")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise TeacherAnchoredNumericalError("cross-dimensional anchor numerical failure")
    for value in tensors:
        norms = torch.linalg.vector_norm(value.detach().double(), dim=-1)
        if not bool((torch.abs(norms - 1.0) <= 2e-5).all()):
            raise ValueError("cross-dimensional anchor authority differs")

    with torch.autocast(device_type=student.device.type, enabled=False):
        teacher_logits = torch.einsum("bd,bad->ba", teacher, teacher_anchors)
        student_logits = torch.einsum("bd,bad->ba", student, student_anchors)
        terms = [
            _forward_kl(teacher_logits / temperature, student_logits / temperature)
            for temperature in temperatures
        ]
        result = torch.stack(terms).mean()
    if result.ndim != 0 or not bool(torch.isfinite(result)):
        raise TeacherAnchoredNumericalError("cross-dimensional anchor numerical failure")
    return result


def cross_dimensional_relational_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperatures: tuple[float, ...],
) -> torch.Tensor:
    """Match leave-self-out pair geometry across different embedding widths."""

    if (
        type(student) is not torch.Tensor
        or type(teacher) is not torch.Tensor
        or student.ndim != 2
        or teacher.ndim != 2
        or student.shape[0] < 3
        or teacher.shape[0] != student.shape[0]
        or student.shape[1] < 2
        or teacher.shape[1] < 2
        or student.dtype != torch.float32
        or teacher.dtype != torch.float32
        or student.device != teacher.device
        or not student.is_contiguous()
        or not teacher.is_contiguous()
        or not student.requires_grad
        or teacher.requires_grad
        or type(temperatures) is not tuple
        or not temperatures
        or any(
            type(temperature) is not float
            or not math.isfinite(temperature)
            or temperature <= 0.0
            for temperature in temperatures
        )
    ):
        raise ValueError("cross-dimensional relational authority differs")
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise TeacherAnchoredNumericalError("cross-dimensional relational numerical failure")
    for value in (student, teacher):
        norms = torch.linalg.vector_norm(value.detach().double(), dim=1)
        if not bool((torch.abs(norms - 1.0) <= 2e-5).all()):
            raise ValueError("cross-dimensional relational authority differs")

    with torch.autocast(device_type=student.device.type, enabled=False):
        mask = ~torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
        teacher_relations = (teacher @ teacher.T)[mask].reshape(
            student.shape[0], student.shape[0] - 1
        )
        student_relations = (student @ student.T)[mask].reshape(
            student.shape[0], student.shape[0] - 1
        )
        result = torch.stack(
            [
                _forward_kl(
                    teacher_relations / temperature,
                    student_relations / temperature,
                )
                for temperature in temperatures
            ]
        ).mean()
    if result.ndim != 0 or not bool(torch.isfinite(result)):
        raise TeacherAnchoredNumericalError("cross-dimensional relational numerical failure")
    return result


def cross_dimensional_similarity_distillation_loss(
    student_similarities: torch.Tensor,
    teacher_similarities: torch.Tensor,
    *,
    temperatures: tuple[float, ...],
) -> torch.Tensor:
    """Match precomputed cosine-similarity distributions across embedding widths."""

    if (
        type(student_similarities) is not torch.Tensor
        or type(teacher_similarities) is not torch.Tensor
        or student_similarities.ndim != 2
        or teacher_similarities.shape != student_similarities.shape
        or student_similarities.shape[0] < 1
        or student_similarities.shape[1] < 2
        or student_similarities.dtype != torch.float32
        or teacher_similarities.dtype != torch.float32
        or student_similarities.device != teacher_similarities.device
        or not student_similarities.is_contiguous()
        or not teacher_similarities.is_contiguous()
        or not student_similarities.requires_grad
        or teacher_similarities.requires_grad
        or type(temperatures) is not tuple
        or not temperatures
        or any(
            type(temperature) is not float
            or not math.isfinite(temperature)
            or temperature <= 0.0
            for temperature in temperatures
        )
    ):
        raise ValueError("cross-dimensional similarity authority differs")
    if not bool(torch.isfinite(student_similarities).all()) or not bool(
        torch.isfinite(teacher_similarities).all()
    ):
        raise TeacherAnchoredNumericalError("cross-dimensional similarity numerical failure")
    if bool((student_similarities.detach().abs() > 1.00002).any()) or bool(
        (teacher_similarities.abs() > 1.00002).any()
    ):
        raise ValueError("cross-dimensional similarity authority differs")

    with torch.autocast(device_type=student_similarities.device.type, enabled=False):
        result = torch.stack(
            [
                _forward_kl(
                    teacher_similarities / temperature,
                    student_similarities / temperature,
                )
                for temperature in temperatures
            ]
        ).mean()
    if result.ndim != 0 or not bool(torch.isfinite(result)):
        raise TeacherAnchoredNumericalError("cross-dimensional similarity numerical failure")
    return result


def teacher_anchored_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    anchors: torch.Tensor,
    adapted_features: torch.Tensor,
    original_features: torch.Tensor,
    config: TeacherAnchoredConfig,
) -> TeacherAnchoredLoss:
    """Compute the label-free five-term teacher-anchored objective."""

    _validate_loss_inputs(student, teacher, anchors, adapted_features, original_features, config)
    with torch.autocast(device_type=student.device.type, enabled=False):
        anchor_terms: list[torch.Tensor] = []
        symmetric_terms: list[torch.Tensor] = []
        mask = ~torch.eye(student.shape[0], dtype=torch.bool, device=student.device)
        for temperature in config.temperatures:
            teacher_anchor_logits = torch.einsum("bd,bad->ba", teacher, anchors) / temperature
            student_anchor_logits = torch.einsum("bd,bad->ba", student, anchors) / temperature
            anchor_terms.append(_forward_kl(teacher_anchor_logits, student_anchor_logits))

            teacher_pair_logits = ((teacher @ teacher.T) / temperature)[mask].reshape(
                student.shape[0], student.shape[0] - 1
            )
            student_pair_logits = ((student @ student.T) / temperature)[mask].reshape(
                student.shape[0], student.shape[0] - 1
            )
            symmetric_terms.append(_forward_kl(teacher_pair_logits, student_pair_logits))

        anchor_loss = torch.stack(anchor_terms).mean()
        symmetric_loss = torch.stack(symmetric_terms).mean()
        point_loss = (1.0 - torch.sum(student * teacher, dim=-1)).mean()
        drift_loss = torch.square(
            adapted_features @ adapted_features.T - original_features @ original_features.T
        ).mean()

        centered_student = student - student.mean(dim=0)
        centered_teacher = teacher - teacher.mean(dim=0)
        denominator = student.shape[0] - 1
        student_covariance = centered_student.T @ centered_student / denominator
        teacher_covariance = centered_teacher.T @ centered_teacher / denominator
        covariance_loss = torch.square(student_covariance - teacher_covariance).sum() / (
            torch.square(teacher_covariance).sum() + 1e-12
        )
        student_sigma = torch.sqrt(torch.diag(student_covariance).clamp_min(0) + 1e-4)
        teacher_sigma = torch.sqrt(torch.diag(teacher_covariance).clamp_min(0) + 1e-4)
        covariance_loss = (
            covariance_loss
            + torch.square(torch.relu(0.5 - student_sigma / (teacher_sigma + 1e-4))).mean()
        )
        total = (
            config.anchor_weight * anchor_loss
            + config.point_weight * point_loss
            + config.symmetric_weight * symmetric_loss
            + config.drift_weight * drift_loss
            + config.covariance_weight * covariance_loss
        )
    components = (total, anchor_loss, point_loss, symmetric_loss, drift_loss, covariance_loss)
    if any(value.ndim != 0 for value in components):
        raise ValueError("teacher-anchored loss authority differs")
    if any(not bool(torch.isfinite(value)) for value in components):
        raise TeacherAnchoredNumericalError("teacher-anchored loss numerical failure")
    return TeacherAnchoredLoss(
        total=total,
        anchor=anchor_loss,
        point=point_loss,
        symmetric=symmetric_loss,
        drift=drift_loss,
        covariance=covariance_loss,
    )


def teacher_anchored_forward(
    encoder: nn.Module,
    head: nn.Linear,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode images through the registered normalized feature/head boundary."""

    if (
        not isinstance(encoder, nn.Module)
        or type(head) is not nn.Linear
        or type(images) is not torch.Tensor
        or images.ndim < 2
        or images.shape[0] < 1
        or not bool(torch.isfinite(images).all())
    ):
        raise ValueError("teacher-anchored forward authority differs")
    with torch.autocast(device_type=images.device.type, enabled=False):
        features = encoder(images).float()
        if (
            features.ndim != 2
            or features.shape[0] != images.shape[0]
            or features.shape[1] != head.in_features
        ):
            raise ValueError("teacher-anchored forward authority differs")
        if not bool(torch.isfinite(features).all()):
            raise TeacherAnchoredNumericalError("teacher-anchored forward numerical failure")
        feature_norms = torch.linalg.vector_norm(features.double(), dim=1)
        if not bool(torch.isfinite(feature_norms).all()) or bool((feature_norms <= 1e-12).any()):
            raise TeacherAnchoredNumericalError("teacher-anchored forward numerical failure")
        normalized_features = torch.nn.functional.normalize(features, dim=1)
        normalized_feature_norms = torch.linalg.vector_norm(
            normalized_features.double(), dim=1
        )
        if (
            not bool(torch.isfinite(normalized_features).all())
            or bool((normalized_feature_norms <= 1e-12).any())
        ):
            raise TeacherAnchoredNumericalError("teacher-anchored forward numerical failure")
        raw_codes = head(normalized_features).float()
        if raw_codes.ndim != 2:
            raise ValueError("teacher-anchored forward authority differs")
        if not bool(torch.isfinite(raw_codes).all()):
            raise TeacherAnchoredNumericalError("teacher-anchored forward numerical failure")
        code_norms = torch.linalg.vector_norm(raw_codes.double(), dim=1)
        if not bool(torch.isfinite(code_norms).all()) or bool((code_norms <= 1e-12).any()):
            raise TeacherAnchoredNumericalError("teacher-anchored forward numerical failure")
        normalized_codes = torch.nn.functional.normalize(raw_codes, dim=1)
        normalized_code_norms = torch.linalg.vector_norm(normalized_codes.double(), dim=1)
        if (
            not bool(torch.isfinite(normalized_codes).all())
            or bool((normalized_code_norms <= 1e-12).any())
        ):
            raise TeacherAnchoredNumericalError("teacher-anchored forward numerical failure")
        return normalized_features, normalized_codes


def embedding_geometry_diagnostics(codes: torch.Tensor) -> EmbeddingGeometryDiagnostics:
    """Return effective rank and leading covariance shares for normalized codes."""

    if not _valid_normalized_tensor(codes, dimensions=None) or codes.shape[0] < 2:
        raise ValueError("embedding geometry authority differs")
    with torch.no_grad(), torch.autocast(device_type=codes.device.type, enabled=False):
        centered = codes - codes.mean(dim=0)
        covariance = centered.T @ centered / (codes.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        total = eigenvalues.sum()
        if float(total) <= 1e-12:
            return EmbeddingGeometryDiagnostics(
                effective_rank=0.0,
                leading_eigenvalue_share=1.0,
                top_eight_eigenvalue_share=1.0,
            )
        probabilities = eigenvalues / total
        positive = probabilities > 0
        entropy = -(probabilities[positive] * probabilities[positive].log()).sum()
        effective_rank = float(torch.exp(entropy))
        leading = float(probabilities[-1])
        top_eight = float(probabilities[-min(8, probabilities.numel()) :].sum())
    if not all(math.isfinite(value) for value in (effective_rank, leading, top_eight)):
        raise ValueError("embedding geometry authority differs")
    return EmbeddingGeometryDiagnostics(
        effective_rank=effective_rank,
        leading_eigenvalue_share=leading,
        top_eight_eigenvalue_share=top_eight,
    )


def _forward_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(reference_logits, dim=-1)
    return torch.sum(
        probabilities
        * (
            torch.log_softmax(reference_logits, dim=-1)
            - torch.log_softmax(candidate_logits, dim=-1)
        ),
        dim=-1,
    ).mean()


def _validate_loss_inputs(
    student: torch.Tensor,
    teacher: torch.Tensor,
    anchors: torch.Tensor,
    adapted_features: torch.Tensor,
    original_features: torch.Tensor,
    config: TeacherAnchoredConfig,
) -> None:
    tensors: tuple[torch.Tensor, ...] = (
        student,
        teacher,
        anchors,
        adapted_features,
        original_features,
    )
    if (
        type(config) is not TeacherAnchoredConfig
        or any(type(value) is not torch.Tensor for value in tensors)
        or student.ndim != 2
        or student.shape[0] < 2
        or teacher.shape != student.shape
        or anchors.ndim != 3
        or anchors.shape[0] != student.shape[0]
        or anchors.shape[1] < 1
        or anchors.shape[2] != student.shape[1]
        or adapted_features.ndim != 2
        or adapted_features.shape[0] != student.shape[0]
        or original_features.shape != adapted_features.shape
        or teacher.requires_grad
        or anchors.requires_grad
        or original_features.requires_grad
        or any(value.device != student.device for value in tensors)
        or any(value.dtype != torch.float32 for value in tensors)
    ):
        raise ValueError("teacher-anchored loss authority differs")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise TeacherAnchoredNumericalError("teacher-anchored loss numerical failure")
    if any(not _valid_normalized_tensor(value, dimensions=None) for value in tensors):
        raise ValueError("teacher-anchored loss authority differs")


def _valid_normalized_tensor(value: object, *, dimensions: int | None) -> bool:
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.ndim not in (2, 3)
        or (dimensions is not None and value.shape[-1] != dimensions)
        or not bool(torch.isfinite(value).all())
    ):
        return False
    norms = torch.linalg.vector_norm(value, dim=-1)
    return bool(torch.all(torch.abs(norms - 1.0) <= 2e-5))


def _validated_inputs(
    teacher_codes: object,
    sample_ids: object,
    *,
    seed: object,
    minimum_rows: int,
    message: str,
) -> tuple[NDArray[np.float32], tuple[SampleId, ...]]:
    if (
        type(teacher_codes) is not np.ndarray
        or teacher_codes.dtype != np.float32
        or teacher_codes.ndim != 2
        or teacher_codes.shape[0] < minimum_rows
        or teacher_codes.shape[1] != _DIMENSIONS
        or not teacher_codes.flags.c_contiguous
        or not np.isfinite(teacher_codes).all()
        or type(sample_ids) is not tuple
        or len(sample_ids) != teacher_codes.shape[0]
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError(message)
    encoded_identities = tuple(_identity_bytes(value, message=message) for value in sample_ids)
    if len(set(encoded_identities)) != len(encoded_identities):
        raise ValueError(message)
    norms = np.linalg.vector_norm(teacher_codes.astype(np.float64), axis=1)
    if np.any(np.abs(norms - 1.0) > 2e-6):
        raise ValueError(message)
    return teacher_codes, sample_ids


def _identity_bytes(value: object, *, message: str) -> bytes:
    if type(value) is int and -(2**63) <= value < 2**63:
        return b"i" + value.to_bytes(8, "little", signed=True)
    if type(value) is str and value:
        encoded = value.encode("utf-8")
        return b"s" + len(encoded).to_bytes(8, "little") + encoded
    raise ValueError(message)


def _identity_order_key(value: SampleId) -> tuple[int, int | str]:
    if type(value) is int:
        return (0, value)
    return (1, value)


def _bounded_top_rows(
    similarities: NDArray[np.float64],
    *,
    query_row: int,
    identities: tuple[SampleId, ...],
    count: int,
) -> list[int]:
    """Return an exact bounded prefix without sorting every ranked pair."""

    scores = similarities.copy()
    scores[query_row] = -np.inf
    partition = np.argpartition(scores, -count)[-count:]
    cutoff = float(np.min(scores[partition]))
    above = np.flatnonzero(scores > cutoff).tolist()
    tied = [int(row) for row in np.flatnonzero(scores == cutoff) if int(row) != query_row]
    tied.sort(key=lambda row: _identity_order_key(identities[row]))
    selected = above + tied[: count - len(above)]
    selected.sort(key=lambda row: (-float(scores[row]), _identity_order_key(identities[row])))
    if len(selected) != count:
        raise ValueError("teacher anchor authority differs")
    return selected


def _input_sha256(codes: NDArray[np.float32], sample_ids: tuple[SampleId, ...]) -> str:
    digest = hashlib.sha256(b"teacher-input-v1")
    digest.update(codes.tobytes(order="C"))
    for sample_id in sample_ids:
        encoded = _identity_bytes(sample_id, message="teacher input authority differs")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _domain_seed(seed: int, domain: bytes, identity: bytes) -> int:
    digest = hashlib.sha256(seed.to_bytes(8, "little") + domain + identity).digest()
    return int.from_bytes(digest[:8], "little")


def _schedule_sha256(
    domain: bytes,
    codes: NDArray[np.float32],
    sample_ids: tuple[SampleId, ...],
    seed: int,
    rows: NDArray[np.int64],
) -> str:
    digest = hashlib.sha256(domain + seed.to_bytes(8, "little"))
    digest.update(codes.tobytes(order="C"))
    for sample_id in sample_ids:
        encoded = _identity_bytes(sample_id, message="schedule authority differs")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update(rows.tobytes(order="C"))
    return digest.hexdigest()
