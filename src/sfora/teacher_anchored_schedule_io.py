"""Strict binary authority for reusable teacher-anchored schedules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from sfora.teacher_anchored_distillation import (
    TeacherAnchorSchedule,
    TeacherNeighborBatches,
    teacher_anchor_schedule,
    teacher_anchored_input_sha256,
    teacher_neighbor_batches,
    teacher_neighbor_ranking,
    verify_teacher_anchor_schedule,
    verify_teacher_neighbor_batches,
)

_SCHEMA = "sfora-teacher-anchored-schedule-v1"
_EPOCHS = 10
_ANCHORS = 512
_SEEDS_PER_BATCH = 128
_BATCH_ROWS = 256
_MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_HEADER_KEYS = (
    "anchor_bytes",
    "anchor_schedule_sha256",
    "epoch_schedules",
    "fitting_row_count",
    "ranking_sha256",
    "schema",
    "seed",
    "source_revision",
    "source_snapshot_sha256",
    "split_sha256",
    "teacher_input_sha256",
    "teacher_snapshot_sha256",
)
_EPOCH_KEYS = (
    "dropped_seed_row_indexes",
    "epoch",
    "repeated_identity_count",
    "row_bytes",
    "row_count",
    "sha256",
)


def _hex(value: object, width: int) -> bool:
    return type(value) is str and len(value) == width and set(value) <= _HEX


@dataclass(frozen=True, slots=True)
class TeacherAnchoredScheduleBinding:
    """Arm-independent identities that bind one reusable schedule artifact."""

    source_revision: str
    source_snapshot_sha256: str
    teacher_snapshot_sha256: str
    split_sha256: str
    teacher_input_sha256: str
    ranking_sha256: str
    seed: int
    fitting_row_count: int

    def __post_init__(self) -> None:
        if (
            not _hex(self.source_revision, 40)
            or any(
                not _hex(value, 64)
                for value in (
                    self.source_snapshot_sha256,
                    self.teacher_snapshot_sha256,
                    self.split_sha256,
                    self.teacher_input_sha256,
                    self.ranking_sha256,
                )
            )
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or type(self.fitting_row_count) is not int
            or self.fitting_row_count < 897
        ):
            raise ValueError("teacher-anchored schedule binding differs")


@dataclass(frozen=True, slots=True)
class SealedTeacherAnchoredSchedule:
    """Authenticated immutable schedules shared by every training arm."""

    binding: TeacherAnchoredScheduleBinding
    anchor_schedule: TeacherAnchorSchedule
    epoch_schedules: tuple[TeacherNeighborBatches, ...]
    sha256: str

    def __post_init__(self) -> None:
        if not _hex(self.sha256, 64):
            raise ValueError("teacher-anchored schedule seal differs")
        _validate_schedules(self.binding, self.anchor_schedule, self.epoch_schedules)


def teacher_anchored_schedule_rows(
    schedule: SealedTeacherAnchoredSchedule,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return the exact nested row tuples consumed by the training loop."""

    if type(schedule) is not SealedTeacherAnchoredSchedule:
        raise ValueError("teacher-anchored schedule seal differs")
    return tuple(
        tuple(tuple(int(value) for value in batch) for batch in epoch._row_indexes)
        for epoch in schedule.epoch_schedules
    )


def build_teacher_anchored_schedule(
    teacher_codes: np.ndarray,
    sample_ids: tuple[str | int, ...],
    *,
    seed: int,
    source_revision: str,
    source_snapshot_sha256: str,
    teacher_snapshot_sha256: str,
    split_sha256: str,
) -> SealedTeacherAnchoredSchedule:
    """Build every arm-independent schedule once and bind its exact bytes."""

    ranking = teacher_neighbor_ranking(teacher_codes, sample_ids)
    anchor = teacher_anchor_schedule(teacher_codes, sample_ids, seed=seed)
    epochs = tuple(
        teacher_neighbor_batches(
            teacher_codes,
            sample_ids,
            seed=seed,
            epoch=epoch,
            ranking=ranking,
        )
        for epoch in range(1, _EPOCHS + 1)
    )
    binding = TeacherAnchoredScheduleBinding(
        source_revision=source_revision,
        source_snapshot_sha256=source_snapshot_sha256,
        teacher_snapshot_sha256=teacher_snapshot_sha256,
        split_sha256=split_sha256,
        teacher_input_sha256=ranking.input_sha256,
        ranking_sha256=ranking.sha256,
        seed=seed,
        fitting_row_count=teacher_codes.shape[0],
    )
    payload = canonical_teacher_anchored_schedule_bytes(
        binding=binding,
        anchor_schedule=anchor,
        epoch_schedules=epochs,
    )
    return SealedTeacherAnchoredSchedule(
        binding=binding,
        anchor_schedule=anchor,
        epoch_schedules=epochs,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_schedules(
    binding: TeacherAnchoredScheduleBinding,
    anchor_schedule: TeacherAnchorSchedule,
    epoch_schedules: tuple[TeacherNeighborBatches, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if (
        type(binding) is not TeacherAnchoredScheduleBinding
        or type(anchor_schedule) is not TeacherAnchorSchedule
        or type(epoch_schedules) is not tuple
        or len(epoch_schedules) != _EPOCHS
        or any(type(item) is not TeacherNeighborBatches for item in epoch_schedules)
        or not _hex(anchor_schedule.sha256, 64)
    ):
        raise ValueError("teacher-anchored schedule seal differs")
    row_count = binding.fitting_row_count
    anchors = anchor_schedule._row_indexes
    if (
        anchors.dtype != np.int64
        or anchors.shape != (row_count, _ANCHORS)
        or not anchors.flags.c_contiguous
        or np.any(anchors < 0)
        or np.any(anchors >= row_count)
    ):
        raise ValueError("teacher-anchored schedule seal differs")
    for row in range(row_count):
        values = anchors[row]
        if row in values or len(np.unique(values)) != _ANCHORS:
            raise ValueError("teacher-anchored schedule seal differs")

    batch_count = row_count // _SEEDS_PER_BATCH
    usable = batch_count * _SEEDS_PER_BATCH
    expected_dropped = row_count - usable
    rows_by_epoch: list[np.ndarray] = []
    for schedule in epoch_schedules:
        rows = schedule._row_indexes
        dropped = schedule.dropped_seed_row_indexes
        if (
            rows.dtype != np.int64
            or rows.shape != (batch_count, _BATCH_ROWS)
            or not rows.flags.c_contiguous
            or np.any(rows < 0)
            or np.any(rows >= row_count)
            or type(dropped) is not tuple
            or len(dropped) != expected_dropped
            or any(type(value) is not int or not 0 <= value < row_count for value in dropped)
            or len(set(dropped)) != len(dropped)
            or type(schedule.repeated_identity_count) is not int
            or not _hex(schedule.sha256, 64)
        ):
            raise ValueError("teacher-anchored schedule seal differs")
        if any(len(np.unique(batch)) != _BATCH_ROWS for batch in rows):
            raise ValueError("teacher-anchored schedule seal differs")
        seeds = rows[:, :_SEEDS_PER_BATCH].reshape(-1)
        if len(np.unique(seeds)) != usable or set(seeds.tolist()).intersection(dropped):
            raise ValueError("teacher-anchored schedule seal differs")
        if set(seeds.tolist()).union(dropped) != set(range(row_count)):
            raise ValueError("teacher-anchored schedule seal differs")
        counts = np.bincount(rows.reshape(-1), minlength=row_count)
        if schedule.repeated_identity_count != int(np.count_nonzero(counts > 1)):
            raise ValueError("teacher-anchored schedule seal differs")
        rows_by_epoch.append(rows)
    return anchors, tuple(rows_by_epoch)


def canonical_teacher_anchored_schedule_bytes(
    *,
    binding: TeacherAnchoredScheduleBinding,
    anchor_schedule: TeacherAnchorSchedule,
    epoch_schedules: tuple[TeacherNeighborBatches, ...],
) -> bytes:
    """Encode one strict canonical header followed by little-endian int64 blocks."""

    anchors, rows_by_epoch = _validate_schedules(binding, anchor_schedule, epoch_schedules)
    anchor_bytes = anchors.astype("<i8", copy=False).tobytes(order="C")
    epoch_headers = []
    epoch_blocks = []
    for epoch, (schedule, rows) in enumerate(
        zip(epoch_schedules, rows_by_epoch, strict=True), start=1
    ):
        block = rows.astype("<i8", copy=False).tobytes(order="C")
        epoch_blocks.append(block)
        epoch_headers.append(
            {
                "dropped_seed_row_indexes": list(schedule.dropped_seed_row_indexes),
                "epoch": epoch,
                "repeated_identity_count": schedule.repeated_identity_count,
                "row_bytes": len(block),
                "row_count": rows.shape[0],
                "sha256": schedule.sha256,
            }
        )
    header = {
        "anchor_bytes": len(anchor_bytes),
        "anchor_schedule_sha256": anchor_schedule.sha256,
        "epoch_schedules": epoch_headers,
        "fitting_row_count": binding.fitting_row_count,
        "ranking_sha256": binding.ranking_sha256,
        "schema": _SCHEMA,
        "seed": binding.seed,
        "source_revision": binding.source_revision,
        "source_snapshot_sha256": binding.source_snapshot_sha256,
        "split_sha256": binding.split_sha256,
        "teacher_input_sha256": binding.teacher_input_sha256,
        "teacher_snapshot_sha256": binding.teacher_snapshot_sha256,
    }
    payload = _canonical_json(header) + b"\n" + anchor_bytes + b"".join(epoch_blocks)
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise ValueError("teacher-anchored schedule seal differs")
    return payload


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("teacher-anchored schedule seal differs")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ValueError("teacher-anchored schedule seal differs")


def parse_teacher_anchored_schedule_bytes(
    payload: bytes,
    *,
    expected_sha256: str,
    expected_binding: TeacherAnchoredScheduleBinding,
) -> SealedTeacherAnchoredSchedule:
    """Authenticate exact bytes, then decode and validate the sealed schedules."""

    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > _MAX_PAYLOAD_BYTES
        or not _hex(expected_sha256, 64)
        or hashlib.sha256(payload).hexdigest() != expected_sha256
        or type(expected_binding) is not TeacherAnchoredScheduleBinding
    ):
        raise ValueError("teacher-anchored schedule seal differs")
    newline = payload.find(b"\n", 0, 65537)
    if newline < 1:
        raise ValueError("teacher-anchored schedule seal differs")
    header_bytes = payload[:newline]
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("teacher-anchored schedule seal differs") from error
    if (
        type(header) is not dict
        or tuple(sorted(header)) != _HEADER_KEYS
        or _canonical_json(header) != header_bytes
        or not _hex(header["source_revision"], 40)
        or any(
            not _hex(header[key], 64)
            for key in (
                "source_snapshot_sha256",
                "teacher_snapshot_sha256",
                "split_sha256",
                "teacher_input_sha256",
                "ranking_sha256",
            )
        )
        or type(header["seed"]) is not int
        or type(header["fitting_row_count"]) is not int
    ):
        raise ValueError("teacher-anchored schedule seal differs")
    binding = TeacherAnchoredScheduleBinding(
        source_revision=header["source_revision"],
        source_snapshot_sha256=header["source_snapshot_sha256"],
        teacher_snapshot_sha256=header["teacher_snapshot_sha256"],
        split_sha256=header["split_sha256"],
        teacher_input_sha256=header["teacher_input_sha256"],
        ranking_sha256=header["ranking_sha256"],
        seed=header["seed"],
        fitting_row_count=header["fitting_row_count"],
    )
    if binding != expected_binding or header["schema"] != _SCHEMA:
        raise ValueError("teacher-anchored schedule binding differs")
    anchor_bytes = header["anchor_bytes"]
    epoch_headers = header["epoch_schedules"]
    if (
        type(anchor_bytes) is not int
        or anchor_bytes != binding.fitting_row_count * _ANCHORS * 8
        or not _hex(header["anchor_schedule_sha256"], 64)
        or type(epoch_headers) is not list
        or len(epoch_headers) != _EPOCHS
    ):
        raise ValueError("teacher-anchored schedule seal differs")
    cursor = newline + 1
    anchor_stop = cursor + anchor_bytes
    if anchor_stop > len(payload):
        raise ValueError("teacher-anchored schedule seal differs")
    anchors = np.frombuffer(memoryview(payload)[cursor:anchor_stop], dtype="<i8").reshape(
        binding.fitting_row_count, _ANCHORS
    )
    cursor = anchor_stop
    schedules = []
    for epoch, raw in enumerate(epoch_headers, start=1):
        if type(raw) is not dict or tuple(sorted(raw)) != _EPOCH_KEYS:
            raise ValueError("teacher-anchored schedule seal differs")
        row_count = raw["row_count"]
        row_bytes = raw["row_bytes"]
        if (
            raw["epoch"] != epoch
            or type(raw["epoch"]) is not int
            or type(row_count) is not int
            or row_count != binding.fitting_row_count // _SEEDS_PER_BATCH
            or type(row_bytes) is not int
            or row_bytes != row_count * _BATCH_ROWS * 8
            or type(raw["dropped_seed_row_indexes"]) is not list
            or type(raw["repeated_identity_count"]) is not int
            or not _hex(raw["sha256"], 64)
            or cursor + row_bytes > len(payload)
        ):
            raise ValueError("teacher-anchored schedule seal differs")
        rows = np.frombuffer(
            memoryview(payload)[cursor : cursor + row_bytes], dtype="<i8"
        ).reshape(row_count, _BATCH_ROWS)
        cursor += row_bytes
        schedules.append(
            TeacherNeighborBatches(
                row_indexes=rows,
                dropped_seed_row_indexes=tuple(raw["dropped_seed_row_indexes"]),
                repeated_identity_count=raw["repeated_identity_count"],
                sha256=raw["sha256"],
            )
        )
    if cursor != len(payload):
        raise ValueError("teacher-anchored schedule seal differs")
    anchor = TeacherAnchorSchedule(
        row_indexes=anchors, sha256=header["anchor_schedule_sha256"]
    )
    _validate_schedules(binding, anchor, tuple(schedules))
    return SealedTeacherAnchoredSchedule(
        binding=binding,
        anchor_schedule=anchor,
        epoch_schedules=tuple(schedules),
        sha256=expected_sha256,
    )


def parse_teacher_anchored_schedule_for_inputs(
    payload: bytes,
    *,
    expected_sha256: str,
    teacher_codes: np.ndarray,
    sample_ids: tuple[str | int, ...],
    source_revision: str,
    source_snapshot_sha256: str,
    teacher_snapshot_sha256: str,
    split_sha256: str,
    seed: int,
) -> SealedTeacherAnchoredSchedule:
    """Load a seal and linearly bind schedules to inputs without rebuilding ranking.

    The ranking digest is audit metadata covered by ``expected_sha256``. Its rows
    are deliberately not regenerated here because that would repeat the quadratic
    preparation step in every training arm.
    """

    if (
        type(payload) is not bytes
        or len(payload) > _MAX_PAYLOAD_BYTES
        or not _hex(expected_sha256, 64)
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("teacher-anchored schedule seal differs")
    newline = payload.find(b"\n", 0, 65537)
    if newline < 1:
        raise ValueError("teacher-anchored schedule seal differs")
    try:
        raw = json.loads(
            payload[:newline],
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
        binding = TeacherAnchoredScheduleBinding(
            source_revision=raw["source_revision"],
            source_snapshot_sha256=raw["source_snapshot_sha256"],
            teacher_snapshot_sha256=raw["teacher_snapshot_sha256"],
            split_sha256=raw["split_sha256"],
            teacher_input_sha256=raw["teacher_input_sha256"],
            ranking_sha256=raw["ranking_sha256"],
            seed=raw["seed"],
            fitting_row_count=raw["fitting_row_count"],
        )
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("teacher-anchored schedule seal differs") from error
    loaded = parse_teacher_anchored_schedule_bytes(
        payload, expected_sha256=expected_sha256, expected_binding=binding
    )
    if type(teacher_codes) is not np.ndarray or teacher_codes.ndim != 2:
        raise ValueError("teacher-anchored schedule seal differs")
    expected_values = (
        (binding.source_revision, source_revision),
        (binding.source_snapshot_sha256, source_snapshot_sha256),
        (binding.teacher_snapshot_sha256, teacher_snapshot_sha256),
        (binding.split_sha256, split_sha256),
        (binding.seed, seed),
        (binding.fitting_row_count, teacher_codes.shape[0]),
        (
            binding.teacher_input_sha256,
            teacher_anchored_input_sha256(teacher_codes, sample_ids),
        ),
    )
    if any(
        type(actual) is not type(expected) or actual != expected
        for actual, expected in expected_values
    ):
        raise ValueError("teacher-anchored schedule binding differs")
    verify_teacher_anchor_schedule(
        teacher_codes, sample_ids, seed=seed, schedule=loaded.anchor_schedule
    )
    for epoch, schedule in enumerate(loaded.epoch_schedules, start=1):
        verify_teacher_neighbor_batches(
            teacher_codes,
            sample_ids,
            seed=seed,
            epoch=epoch,
            schedule=schedule,
        )
    return loaded
