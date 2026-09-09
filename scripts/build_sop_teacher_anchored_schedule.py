#!/usr/bin/env python3
"""Build one authenticated arm-independent teacher-anchored schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from threadpoolctl import threadpool_info  # type: ignore[import-untyped]

from sfora.atomic_publication import publish_large_writer_noreplace
from sfora.representation_ceiling import fit_teacher_guided_projection
from sfora.teacher_anchored_distillation import teacher_anchored_input_sha256
from sfora.teacher_anchored_schedule_io import (
    SealedTeacherAnchoredSchedule,
    build_teacher_anchored_schedule,
    canonical_teacher_anchored_schedule_bytes,
)

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from sop_teacher_anchored_runtime import (  # noqa: E402
    TeacherAnchoredTrainPair,
    load_authenticated_train_pair,
)
from train_sop_teacher_anchored_distillation import (  # noqa: E402
    TeacherAnchoredRuntimeReceipt,
    TeacherAnchoredSplit,
    _normalize_snapshot_rows,
    _parameter_sha256,
    build_teacher_anchored_split,
    configure_teacher_anchored_runtime,
)

_SCHEMA = "sfora-teacher-anchored-schedule-build-result-v1"
_HEX = frozenset("0123456789abcdef")


class ScheduleBuildArguments(NamedTuple):
    """Strict local-only schedule construction arguments."""

    source_snapshot: Path
    source_snapshot_sha256: str
    teacher_snapshot: Path
    teacher_snapshot_sha256: str
    source_revision: str
    seed: int
    output: Path


class BuiltTeacherAnchoredSchedule(NamedTuple):
    """Exact schedule bytes and the fitting-only evidence used to build them."""

    payload: bytes
    receipt_bytes: bytes
    sealed: SealedTeacherAnchoredSchedule
    split: TeacherAnchoredSplit
    teacher_codes: NDArray[np.float32]
    teacher_pca_mean: torch.Tensor
    teacher_pca_components: torch.Tensor
    teacher_pca_sha256: str
    runtime: TeacherAnchoredRuntimeReceipt


def _hex(value: object, width: int) -> bool:
    return type(value) is str and len(value) == width and set(value) <= _HEX


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_schedule_build_args(
    arguments: Sequence[str] | None = None,
) -> ScheduleBuildArguments:
    """Parse the explicit offline builder capability."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-snapshot", required=True, type=_absolute_path)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--teacher-snapshot", required=True, type=_absolute_path)
    parser.add_argument("--teacher-snapshot-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(17, 1729, 65537))
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--execute-build-schedule", required=True, action="store_true")
    parsed = parser.parse_args(arguments)
    if (
        not _hex(parsed.source_snapshot_sha256, 64)
        or not _hex(parsed.teacher_snapshot_sha256, 64)
        or not _hex(parsed.source_revision, 40)
        or parsed.source_snapshot == parsed.teacher_snapshot
        or parsed.output in (parsed.source_snapshot, parsed.teacher_snapshot)
    ):
        parser.error("teacher-anchored schedule build authority differs")
    return ScheduleBuildArguments(
        source_snapshot=parsed.source_snapshot,
        source_snapshot_sha256=parsed.source_snapshot_sha256,
        teacher_snapshot=parsed.teacher_snapshot,
        teacher_snapshot_sha256=parsed.teacher_snapshot_sha256,
        source_revision=parsed.source_revision,
        seed=parsed.seed,
        output=parsed.output,
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _receipt_value(artifact: BuiltTeacherAnchoredSchedule) -> dict[str, object]:
    if type(artifact) is not BuiltTeacherAnchoredSchedule:
        raise ValueError("teacher-anchored schedule build result differs")
    sealed = artifact.sealed
    return {
        "anchor_schedule_sha256": sealed.anchor_schedule.sha256,
        "claim_eligible": False,
        "blas_threads": artifact.runtime.blas_threads,
        "cpu_threads": artifact.runtime.cpu_threads,
        "epoch_schedule_sha256": [epoch.sha256 for epoch in sealed.epoch_schedules],
        "fitting_row_count": sealed.binding.fitting_row_count,
        "ranking_sha256": sealed.binding.ranking_sha256,
        "numpy_version": np.__version__,
        "schedule_bytes": len(artifact.payload),
        "schedule_sha256": sealed.sha256,
        "schema": _SCHEMA,
        "seed": sealed.binding.seed,
        "source_revision": sealed.binding.source_revision,
        "source_snapshot_sha256": sealed.binding.source_snapshot_sha256,
        "split_sha256": sealed.binding.split_sha256,
        "teacher_input_sha256": sealed.binding.teacher_input_sha256,
        "teacher_pca_sha256": artifact.teacher_pca_sha256,
        "teacher_snapshot_sha256": sealed.binding.teacher_snapshot_sha256,
        "torch_version": str(torch.__version__),
    }


def canonical_schedule_build_receipt_bytes(
    artifact: BuiltTeacherAnchoredSchedule,
) -> bytes:
    """Serialize the complete claim-ineligible build receipt."""

    value = _receipt_value(artifact)
    if (
        hashlib.sha256(artifact.payload).hexdigest() != value["schedule_sha256"]
        or not _hex(artifact.teacher_pca_sha256, 64)
        or artifact.split.sha256 != value["split_sha256"]
        or artifact.teacher_codes.shape != (value["fitting_row_count"], 128)
        or artifact.teacher_codes.dtype != np.float32
        or not artifact.teacher_codes.flags.c_contiguous
        or not bool(np.isfinite(artifact.teacher_codes).all())
        or not bool(
            np.allclose(
                np.linalg.norm(artifact.teacher_codes.astype(np.float64), axis=1),
                1.0,
                rtol=0.0,
                atol=1e-5,
            )
        )
        or teacher_anchored_input_sha256(artifact.teacher_codes, artifact.split.fitting_image_ids)
        != artifact.sealed.binding.teacher_input_sha256
        or _parameter_sha256(artifact.teacher_pca_mean, artifact.teacher_pca_components)
        != artifact.teacher_pca_sha256
        or artifact.runtime.seed != artifact.sealed.binding.seed
        or artifact.runtime.cpu_threads != 2
        or artifact.runtime.blas_threads != 2
        or torch.get_num_threads() != artifact.runtime.cpu_threads
        or {int(pool["num_threads"]) for pool in threadpool_info() if pool["user_api"] == "blas"}
        != {artifact.runtime.blas_threads}
    ):
        raise ValueError("teacher-anchored schedule build result differs")
    return _canonical_json(value)


def build_sop_teacher_anchored_schedule(
    pair: TeacherAnchoredTrainPair,
    *,
    seed: int,
    source_revision: str,
) -> BuiltTeacherAnchoredSchedule:
    """Fit the teacher PCA on fitting rows and build the shared schedule once."""

    if (
        type(pair) is not TeacherAnchoredTrainPair
        or type(seed) is not int
        or seed not in (17, 1729, 65537)
        or not _hex(source_revision, 40)
        or pair.source_metadata.get("model_revision") != source_revision
        or pair.teacher_metadata.get("model_revision") != source_revision
    ):
        raise ValueError("teacher-anchored schedule build authority differs")
    runtime = configure_teacher_anchored_runtime(seed)
    split = build_teacher_anchored_split(pair.labels, pair.image_ids, seed=seed)
    rows = np.asarray(split.fitting_rows, dtype=np.int64)
    source_fit = _normalize_snapshot_rows(
        torch.from_numpy(np.ascontiguousarray(pair.source_embeddings[rows]))
    )
    teacher_fit = _normalize_snapshot_rows(
        torch.from_numpy(np.ascontiguousarray(pair.teacher_embeddings[rows]))
    )
    projection = fit_teacher_guided_projection(
        source_fit,
        teacher_fit,
        dimensions=128,
        penalty=1e-6,
    )
    teacher_codes = np.ascontiguousarray(
        projection.apply_teacher(teacher_fit).detach().cpu().numpy(), dtype=np.float32
    )
    teacher_pca_sha256 = _parameter_sha256(
        projection.teacher_projection.mean,
        projection.teacher_projection.components,
    )
    sealed = build_teacher_anchored_schedule(
        teacher_codes,
        split.fitting_image_ids,
        seed=seed,
        source_revision=source_revision,
        source_snapshot_sha256=pair.source_snapshot_sha256,
        teacher_snapshot_sha256=pair.teacher_snapshot_sha256,
        split_sha256=split.sha256,
    )
    payload = canonical_teacher_anchored_schedule_bytes(
        binding=sealed.binding,
        anchor_schedule=sealed.anchor_schedule,
        epoch_schedules=sealed.epoch_schedules,
    )
    incomplete = BuiltTeacherAnchoredSchedule(
        payload=payload,
        receipt_bytes=b"",
        sealed=sealed,
        split=split,
        teacher_codes=teacher_codes,
        teacher_pca_mean=projection.teacher_projection.mean.detach().clone(),
        teacher_pca_components=projection.teacher_projection.components.detach().clone(),
        teacher_pca_sha256=teacher_pca_sha256,
        runtime=runtime,
    )
    return incomplete._replace(receipt_bytes=canonical_schedule_build_receipt_bytes(incomplete))


def publish_sop_teacher_anchored_schedule(
    output: Path, artifact: BuiltTeacherAnchoredSchedule
) -> None:
    """Publish the exact schedule bytes once without replacing any path."""

    if not isinstance(output, Path) or type(artifact) is not BuiltTeacherAnchoredSchedule:
        raise ValueError("teacher-anchored schedule publication differs")
    try:
        if canonical_schedule_build_receipt_bytes(artifact) != artifact.receipt_bytes:
            raise ValueError("teacher-anchored schedule publication differs")
        if artifact.payload != canonical_teacher_anchored_schedule_bytes(
            binding=artifact.sealed.binding,
            anchor_schedule=artifact.sealed.anchor_schedule,
            epoch_schedules=artifact.sealed.epoch_schedules,
        ):
            raise ValueError("teacher-anchored schedule publication differs")
    except ValueError as error:
        raise ValueError("teacher-anchored schedule publication differs") from error

    def write(descriptor: int) -> None:
        view = memoryview(artifact.payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise RuntimeError("teacher-anchored schedule publication stalled")
            offset += written

    def validate(descriptor: int, size: int) -> None:
        if size != len(artifact.payload):
            raise ValueError("teacher-anchored schedule publication differs")
        digest = hashlib.sha256()
        offset = 0
        while offset < size:
            chunk = os.pread(descriptor, min(size - offset, 1024 * 1024), offset)
            if not chunk:
                raise ValueError("teacher-anchored schedule publication differs")
            digest.update(chunk)
            offset += len(chunk)
        if digest.hexdigest() != artifact.sealed.sha256:
            raise ValueError("teacher-anchored schedule publication differs")

    with publish_large_writer_noreplace(
        output,
        write,
        validator=validate,
    ):
        pass


def main(arguments: Sequence[str] | None = None) -> int:
    """Authenticate inputs, construct once, publish once, and emit the receipt."""

    try:
        parsed = parse_schedule_build_args(arguments)
        if os.path.lexists(parsed.output):
            raise FileExistsError(parsed.output)
        pair = load_authenticated_train_pair(
            parsed.source_snapshot,
            parsed.source_snapshot_sha256,
            parsed.teacher_snapshot,
            parsed.teacher_snapshot_sha256,
        )
        artifact = build_sop_teacher_anchored_schedule(
            pair,
            seed=parsed.seed,
            source_revision=parsed.source_revision,
        )
        publish_sop_teacher_anchored_schedule(parsed.output, artifact)
    except Exception as error:
        print(f"SOP teacher-anchored schedule build failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(artifact.receipt_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
