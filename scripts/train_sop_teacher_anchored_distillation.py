#!/usr/bin/env python3
"""Strict local trainer boundary for teacher-anchored SOP experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.amp.grad_scaler import GradScaler

from sfora.deterministic_similarity_runtime import (
    DeterministicSimilarityRuntimeReceipt as TeacherAnchoredRuntimeReceipt,
)
from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import (
    TeacherGuidedProjection,
    deterministic_class_partition,
    fit_teacher_guided_projection,
)
from sfora.teacher_anchored_distillation import (
    TeacherAnchoredConfig,
    TeacherAnchoredLoss,
    TeacherAnchoredNumericalError,
    embedding_geometry_diagnostics,
    teacher_anchored_forward,
    teacher_anchored_loss,
)
from sfora.teacher_anchored_schedule_io import (
    SealedTeacherAnchoredSchedule,
    parse_teacher_anchored_schedule_for_inputs,
    teacher_anchored_schedule_rows,
)

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from sop_teacher_anchored_runtime import (  # noqa: E402
    TeacherAnchoredImageManifest,
    TeacherAnchoredSourceModel,
    TeacherAnchoredTrainPair,
    bind_authenticated_train_images,
    load_authenticated_source_model,
    load_authenticated_train_pair,
)

_UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"


class TeacherAnchoredInitialization(NamedTuple):
    """Exact split-local projection state and initialized serving head."""

    projection: TeacherGuidedProjection
    head: nn.Linear
    teacher_pca_sha256: str


class TeacherAnchoredArguments(NamedTuple):
    """Authenticated local-only training invocation."""

    source_checkpoint: Path
    source_checkpoint_sha256: str
    teacher_checkpoint: Path
    teacher_checkpoint_sha256: str
    unicom_checkout: Path
    source_snapshot: Path
    source_snapshot_sha256: str
    teacher_snapshot: Path
    teacher_snapshot_sha256: str
    schedule: Path
    schedule_sha256: str
    image_root: Path
    image_tree_sha256: str
    teacher_pca_sha256: str
    launch_receipt: Path
    launch_receipt_sha256: str
    ceiling_receipt: Path
    ceiling_receipt_sha256: str
    source_revision: str
    seed: int
    arm: str
    output: Path
    progress: Path
    execute_teacher_anchored: bool


class TeacherAnchoredPreparedArrays(NamedTuple):
    """Fitting-local tensors and initialized projection for one sealed split."""

    split: TeacherAnchoredSplit
    initialization: TeacherAnchoredInitialization
    source_features: torch.Tensor
    teacher_codes: torch.Tensor


class TeacherAnchoredBoundSchedule(NamedTuple):
    """Authenticated schedule materialized in the fitting-local row space."""

    sealed: SealedTeacherAnchoredSchedule
    schedules: tuple[tuple[tuple[int, ...], ...], ...]
    anchor_row_indexes: torch.Tensor
    fitting_image_paths: tuple[Path, ...]


class TeacherAnchoredHeadReplayReceipt(NamedTuple):
    """Portable initializer replay evidence."""

    rows: int
    chunk_rows: int
    device_type: str
    device_name: str
    device_capability: str
    input_sha256: str
    head_state_sha256: str
    runtime_codes_sha256: str
    reference_codes_sha256: str
    maximum_absolute_error: float
    minimum_cosine: float


class TeacherAnchoredSnapshotReplayReceipt(NamedTuple):
    """Exact live-image replay evidence for pristine and training-shaped encodes."""

    rows: int
    device_type: str
    device_name: str
    image_ids_sha256: str
    images_sha256: str
    snapshot_codes_sha256: str
    pristine_state_sha256: str
    training_state_sha256: str
    single_codes_sha256: str
    batch_codes_sha256: str
    maximum_single_error: float
    maximum_batch_error: float
    maximum_shape_error: float
    minimum_single_cosine: float
    minimum_batch_cosine: float
    minimum_shape_cosine: float


class TeacherAnchoredEpochDiagnostic(NamedTuple):
    """Binding fitting and non-binding validation evidence for one epoch."""

    epoch: int
    fitting_map_at_r: float
    fitting_effective_rank: float
    fitting_leading_eigenvalue_share: float
    validation_map_at_r: float


class TeacherAnchoredTrainingReceipt(NamedTuple):
    """Exact endpoint or fitting-stop evidence for one training arm."""

    arm: str
    schedule_sha256: str
    initial_encoder_sha256: str
    final_encoder_sha256: str
    initial_frozen_sha256: str
    final_frozen_sha256: str
    initial_head_sha256: str
    final_head_sha256: str
    optimizer_reset_epochs: tuple[int, ...]
    attempted_updates: int
    successful_updates: int
    completed_epochs: tuple[int, ...]
    diagnostics: tuple[TeacherAnchoredEpochDiagnostic, ...]
    stopped_reason: str | None
    candidate_epoch: int | None


class TeacherAnchoredFittingProbeSelection(NamedTuple):
    """All rows belonging to the first 512 seed-hashed fitting classes."""

    class_ids: tuple[int, ...]
    row_indexes: tuple[int, ...]
    sha256: str


class TeacherAnchoredSplit(NamedTuple):
    """Class-disjoint fitting and validation rows in their original order."""

    fitting_rows: tuple[int, ...]
    validation_rows: tuple[int, ...]
    fitting_class_ids: tuple[int, ...]
    validation_class_ids: tuple[int, ...]
    fitting_labels: tuple[int, ...]
    validation_labels: tuple[int, ...]
    fitting_image_ids: tuple[int, ...]
    validation_image_ids: tuple[int, ...]
    sha256: str


class TeacherAnchoredProbeScore(NamedTuple):
    """Exact float32 and deployed symmetric-int8 fitting/validation score."""

    candidate_width: int
    float_map_at_r: float
    float_r1: float
    packed_map_at_r: float
    packed_r1: float


class TeacherAnchoredBootstrapSplit(NamedTuple):
    """One ordered historical validation split for paired replay."""

    seed: int
    row_indexes: tuple[int, ...]
    treatment: tuple[float, ...]
    baseline: tuple[float, ...]


class TeacherAnchoredProgressState(NamedTuple):
    """Last authenticated progress event in a launch-bound chain."""

    sequence: int
    arm: str
    epoch: int
    update: int
    line_sha256: str


class TeacherAnchoredPublishedArtifacts(NamedTuple):
    """No-clobber receipt and complete merged checkpoint paths."""

    receipt: Path
    checkpoint: Path
    receipt_sha256: str
    checkpoint_sha256: str


class TeacherAnchoredNonfiniteUpdate(ValueError):
    """A numerically invalid optimizer attempt, distinct from authority/resource failures."""


TeacherAnchoredComputeLoss = Callable[
    [int, int, tuple[int, ...], tuple[str, ...]], TeacherAnchoredLoss
]
TeacherAnchoredDiagnose = Callable[[int], TeacherAnchoredEpochDiagnostic]
TeacherAnchoredProgress = Callable[[str, int, int], None]


def parse_teacher_anchored_args(arguments: list[str]) -> TeacherAnchoredArguments:
    """Parse the fail-closed local capability boundary without executing science."""

    if type(arguments) is not list or any(type(argument) is not str for argument in arguments):
        raise ValueError("unsupported argument")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    for name in (
        "source-checkpoint",
        "teacher-checkpoint",
        "source-snapshot",
        "teacher-snapshot",
        "schedule",
        "unicom-checkout",
        "image-root",
        "launch-receipt",
        "ceiling-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}")
    for name in (
        "source-checkpoint-sha256",
        "teacher-checkpoint-sha256",
        "source-snapshot-sha256",
        "teacher-snapshot-sha256",
        "schedule-sha256",
        "image-tree-sha256",
        "teacher-pca-sha256",
        "launch-receipt-sha256",
        "ceiling-receipt-sha256",
        "source-revision",
        "seed",
        "arm",
    ):
        parser.add_argument(f"--{name}")
    parser.add_argument("--execute-teacher-anchored", action="store_true")
    known_flags = {action.option_strings[0] for action in parser._actions if action.option_strings}
    presented_flags = [
        argument.split("=", 1)[0] for argument in arguments if argument.startswith("--")
    ]
    for flag in known_flags:
        if presented_flags.count(flag) > 1:
            raise ValueError("unsupported argument")
    try:
        parsed, unknown = parser.parse_known_args(arguments)
    except (argparse.ArgumentError, SystemExit) as error:
        raise ValueError("unsupported argument") from error
    if unknown:
        raise ValueError("unsupported argument")

    path_values: dict[str, Path] = {}
    for name in (
        "source_checkpoint",
        "teacher_checkpoint",
        "source_snapshot",
        "teacher_snapshot",
        "schedule",
        "unicom_checkout",
        "image_root",
        "launch_receipt",
        "ceiling_receipt",
    ):
        raw = getattr(parsed, name)
        path = Path(raw) if type(raw) is str else Path()
        if type(raw) is not str or not path.is_absolute() or not path.exists():
            raise ValueError("absolute local input authority differs")
        if name in ("image_root", "unicom_checkout") and not path.is_dir():
            raise ValueError("absolute local input authority differs")
        if name not in ("image_root", "unicom_checkout") and not path.is_file():
            raise ValueError("absolute local input authority differs")
        if name in ("launch_receipt", "ceiling_receipt") and path.is_symlink():
            raise ValueError("absolute local input authority differs")
        path_values[name] = path

    raw_output = parsed.output
    output = Path(raw_output) if type(raw_output) is str else Path()
    if type(raw_output) is not str or not output.is_absolute():
        raise ValueError("absolute local output authority differs")
    if output.exists():
        raise ValueError("output already exists")
    if not output.parent.is_dir():
        raise ValueError("absolute local output authority differs")
    progress = output.with_suffix(".progress.jsonl")
    if progress.exists():
        raise ValueError("progress already exists")

    digests: dict[str, str] = {}
    for name in (
        "source_checkpoint_sha256",
        "teacher_checkpoint_sha256",
        "source_snapshot_sha256",
        "teacher_snapshot_sha256",
        "schedule_sha256",
        "image_tree_sha256",
        "teacher_pca_sha256",
        "launch_receipt_sha256",
        "ceiling_receipt_sha256",
    ):
        value = getattr(parsed, name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("SHA-256 authority differs")
        digests[name] = value
    revision = parsed.source_revision
    if type(revision) is not str or len(revision) != 40 or revision != _UNICOM_REVISION:
        raise ValueError("revision authority differs")
    try:
        seed = int(parsed.seed)
    except (TypeError, ValueError) as error:
        raise ValueError("seed authority differs") from error
    if type(parsed.seed) is not str or str(seed) != parsed.seed or seed not in (17, 1729, 65537):
        raise ValueError("seed authority differs")
    if parsed.arm not in ("head-only", "base", "anchor", "symmetric", "complete"):
        raise ValueError("arm authority differs")
    if parsed.execute_teacher_anchored is not True:
        raise ValueError("execution flag is required")

    return TeacherAnchoredArguments(
        source_checkpoint=path_values["source_checkpoint"],
        source_checkpoint_sha256=digests["source_checkpoint_sha256"],
        teacher_checkpoint=path_values["teacher_checkpoint"],
        teacher_checkpoint_sha256=digests["teacher_checkpoint_sha256"],
        unicom_checkout=path_values["unicom_checkout"],
        source_snapshot=path_values["source_snapshot"],
        source_snapshot_sha256=digests["source_snapshot_sha256"],
        teacher_snapshot=path_values["teacher_snapshot"],
        teacher_snapshot_sha256=digests["teacher_snapshot_sha256"],
        schedule=path_values["schedule"],
        schedule_sha256=digests["schedule_sha256"],
        image_root=path_values["image_root"],
        image_tree_sha256=digests["image_tree_sha256"],
        teacher_pca_sha256=digests["teacher_pca_sha256"],
        launch_receipt=path_values["launch_receipt"],
        launch_receipt_sha256=digests["launch_receipt_sha256"],
        ceiling_receipt=path_values["ceiling_receipt"],
        ceiling_receipt_sha256=digests["ceiling_receipt_sha256"],
        source_revision=revision,
        seed=seed,
        arm=parsed.arm,
        output=output,
        progress=progress,
        execute_teacher_anchored=True,
    )


def configure_teacher_anchored_runtime(seed: int) -> TeacherAnchoredRuntimeReceipt:
    """Pin RNG and Torch arithmetic state before any experiment computation."""

    if type(seed) is not int or seed not in (17, 1729, 65537):
        raise ValueError("teacher-anchored runtime authority differs")
    try:
        return configure_deterministic_similarity_runtime(seed, cpu_threads=2)
    except ValueError as error:
        raise ValueError("teacher-anchored runtime authority differs") from error


def authenticate_teacher_anchored_files(arguments: TeacherAnchoredArguments) -> tuple[str, ...]:
    """Authenticate model/snapshot files before the schedule's retained read."""

    if type(arguments) is not TeacherAnchoredArguments:
        raise ValueError("teacher-anchored input authority differs")
    authenticate_teacher_anchored_checkout(arguments.unicom_checkout, arguments.source_revision)
    observed: list[str] = []
    for path, expected in (
        (arguments.source_checkpoint, arguments.source_checkpoint_sha256),
        (arguments.teacher_checkpoint, arguments.teacher_checkpoint_sha256),
        (arguments.source_snapshot, arguments.source_snapshot_sha256),
        (arguments.teacher_snapshot, arguments.teacher_snapshot_sha256),
    ):
        info = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise ValueError("teacher-anchored input must be a regular local file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        value = digest.hexdigest()
        if value != expected:
            raise ValueError("teacher-anchored input digest differs")
        observed.append(value)
    return tuple(observed)


def load_teacher_anchored_training_image(
    path: Path, transform: Callable[[Image.Image], torch.Tensor]
) -> torch.Tensor:
    """Decode one retained regular image with the snapshot builder's RGB semantics."""

    if not isinstance(path, Path) or not path.is_absolute() or not callable(transform):
        raise ValueError("training image authority differs")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_NONBLOCK
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ValueError("training image authority differs") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 64 * 1024 * 1024:
            raise ValueError("training image authority differs")
        payload = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, before.st_size - len(payload))):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or os.read(descriptor, 1)
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
        ):
            raise ValueError("training image authority differs")
    finally:
        os.close(descriptor)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            tensor = transform(image.convert("RGB"))
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("training image authority differs") from error
    if (
        type(tensor) is not torch.Tensor
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.float32
        or tensor.ndim != 3
        or not tensor.is_contiguous()
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError("training image transform differs")
    return tensor


def load_teacher_anchored_schedule_file(
    path: Path,
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
    """Read one retained regular-file identity and bind its schedules to inputs."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("teacher-anchored schedule seal differs")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        raise ValueError("teacher-anchored schedule seal differs") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 512 * 1024 * 1024:
            raise ValueError("teacher-anchored schedule seal differs")
        payload = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, info.st_size - len(payload))):
            payload.extend(chunk)
        final_info = os.fstat(descriptor)
        if (
            len(payload) != info.st_size
            or os.read(descriptor, 1)
            or (
                final_info.st_dev,
                final_info.st_ino,
                final_info.st_size,
                final_info.st_mtime_ns,
                final_info.st_ctime_ns,
            )
            != (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
        ):
            raise ValueError("teacher-anchored schedule seal differs")
    finally:
        os.close(descriptor)
    data = bytes(payload)
    del payload
    return parse_teacher_anchored_schedule_for_inputs(
        data,
        expected_sha256=expected_sha256,
        teacher_codes=teacher_codes,
        sample_ids=sample_ids,
        source_revision=source_revision,
        source_snapshot_sha256=source_snapshot_sha256,
        teacher_snapshot_sha256=teacher_snapshot_sha256,
        split_sha256=split_sha256,
        seed=seed,
    )


def load_teacher_anchored_ceiling_pca_sha256(path: Path, *, expected_sha256: str, seed: int) -> str:
    """Authenticate the sealed ceiling receipt and select one split's PCA identity."""

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not _is_sha256(expected_sha256)
        or type(seed) is not int
        or seed not in (17, 1729, 65537)
    ):
        raise ValueError("ceiling receipt authority differs")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | os.O_NONBLOCK
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise ValueError("ceiling receipt authority differs") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 16 * 1024 * 1024:
            raise ValueError("ceiling receipt authority differs")
        payload = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, before.st_size - len(payload))):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or os.read(descriptor, 1)
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
        ):
            raise ValueError("ceiling receipt authority differs")
    finally:
        os.close(descriptor)
    data = bytes(payload)
    del payload
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("ceiling receipt authority differs")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ceiling receipt authority differs") from error
    if (
        type(value) is not dict
        or _canonical_json_bytes(value) != data
        or value.get("schema") != "sfora-representation-ceiling-v1"
        or value.get("claim_eligible") is not False
        or value.get("outer_split_seeds") != [17, 1729, 65537]
        or type(value.get("splits")) is not list
        or len(value["splits"]) != 3
    ):
        raise ValueError("ceiling receipt authority differs")
    selected: str | None = None
    observed_seeds: list[int] = []
    for split in value["splits"]:
        if type(split) is not dict or type(split.get("seed")) is not int:
            raise ValueError("ceiling receipt authority differs")
        split_seed = split["seed"]
        transforms = split.get("transform_sha256")
        if (
            type(transforms) is not dict
            or set(transforms) != {"ridge", "ridge-pca", "source-pca", "teacher-pca"}
            or any(not _is_sha256(digest) for digest in transforms.values())
        ):
            raise ValueError("ceiling receipt authority differs")
        observed_seeds.append(split_seed)
        if split_seed == seed:
            selected = transforms["teacher-pca"]
    if observed_seeds != [17, 1729, 65537] or selected is None:
        raise ValueError("ceiling receipt authority differs")
    return selected


def authenticate_teacher_anchored_launch_receipt(arguments: TeacherAnchoredArguments) -> str:
    """Authenticate a canonical launch receipt against this exact arm invocation."""

    if type(arguments) is not TeacherAnchoredArguments:
        raise ValueError("teacher-anchored launch receipt authority differs")
    try:
        descriptor = os.open(
            arguments.launch_receipt,
            os.O_RDONLY
            | os.O_NONBLOCK
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as error:
        raise ValueError("teacher-anchored launch receipt authority differs") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 1024 * 1024:
            raise ValueError("teacher-anchored launch receipt authority differs")
        payload = bytearray()
        while chunk := os.read(descriptor, min(1024 * 1024, before.st_size - len(payload))):
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if (
            len(payload) != before.st_size
            or os.read(descriptor, 1)
            or (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
        ):
            raise ValueError("teacher-anchored launch receipt authority differs")
    finally:
        os.close(descriptor)
    data = bytes(payload)
    del payload
    if hashlib.sha256(data).hexdigest() != arguments.launch_receipt_sha256:
        raise ValueError("teacher-anchored launch receipt authority differs")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("teacher-anchored launch receipt authority differs") from error
    expected = {
        "arm": arguments.arm,
        "claim_eligible": False,
        "inputs_sha256": {
            "ceiling_receipt": arguments.ceiling_receipt_sha256,
            "image_tree": arguments.image_tree_sha256,
            "schedule": arguments.schedule_sha256,
            "source_checkpoint": arguments.source_checkpoint_sha256,
            "source_snapshot": arguments.source_snapshot_sha256,
            "teacher_checkpoint": arguments.teacher_checkpoint_sha256,
            "teacher_snapshot": arguments.teacher_snapshot_sha256,
        },
        "output": str(arguments.output),
        "schema": "sfora-teacher-anchored-launch-v1",
        "seed": arguments.seed,
        "source_revision": arguments.source_revision,
    }
    if value != expected or data != _canonical_json_bytes(expected):
        raise ValueError("teacher-anchored launch receipt authority differs")
    return arguments.launch_receipt_sha256


def authenticate_teacher_anchored_checkout(checkout: Path, expected_revision: str) -> str:
    """Require the executable UNICOM checkout to be the exact clean revision."""

    if (
        not isinstance(checkout, Path)
        or checkout.is_symlink()
        or not checkout.is_dir()
        or type(expected_revision) is not str
        or len(expected_revision) != 40
        or any(character not in "0123456789abcdef" for character in expected_revision)
    ):
        raise ValueError("teacher-anchored checkout authority differs")
    checkout = checkout.resolve()
    git_environment = {
        key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")
    }

    def git_output(*arguments: str) -> str:
        return subprocess.run(
            ["git", "--no-replace-objects", "-C", str(checkout), *arguments],
            check=True,
            capture_output=True,
            text=True,
            env=git_environment,
        ).stdout

    try:
        root = Path(git_output("rev-parse", "--show-toplevel").strip()).resolve()
        revision = git_output("rev-parse", "HEAD").strip()
        status = git_output("status", "--porcelain", "--untracked-files=all")
        ignored_python = tuple(
            entry
            for entry in git_output(
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            ).split("\0")
            if entry.endswith(".py")
        )
        index_entries = tuple(
            entry for entry in git_output("ls-files", "-v", "-z").split("\0") if entry
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("teacher-anchored checkout authority differs") from error
    if (
        root != checkout
        or revision != expected_revision
        or status
        or ignored_python
        or not index_entries
        or any(not entry.startswith("H ") for entry in index_entries)
    ):
        raise ValueError("teacher-anchored checkout authority differs")
    return revision


def initialize_teacher_anchored_student(
    source_fit: torch.Tensor,
    teacher_fit: torch.Tensor,
    *,
    expected_teacher_pca_sha256: str,
) -> TeacherAnchoredInitialization:
    """Fit the frozen 128D teacher-PCA/ridge initializer and authenticate it."""

    if (
        type(expected_teacher_pca_sha256) is not str
        or len(expected_teacher_pca_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_teacher_pca_sha256)
    ):
        raise ValueError("teacher PCA authority differs")
    source_unit = _normalize_snapshot_rows(source_fit)
    teacher_unit = _normalize_snapshot_rows(teacher_fit)
    projection = fit_teacher_guided_projection(
        source_unit,
        teacher_unit,
        dimensions=128,
        penalty=1e-6,
    )
    digest = _parameter_sha256(
        projection.teacher_projection.mean,
        projection.teacher_projection.components,
    )
    if digest != expected_teacher_pca_sha256:
        raise ValueError("teacher PCA authority differs")
    weight = projection.source_projection.weight
    bias = projection.source_projection.bias
    head = nn.Linear(weight.shape[1], weight.shape[0], bias=True, dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    return TeacherAnchoredInitialization(
        projection=projection,
        head=head,
        teacher_pca_sha256=digest,
    )


def prepare_teacher_anchored_arrays(
    pair: TeacherAnchoredTrainPair,
    *,
    seed: int,
    expected_teacher_pca_sha256: str,
) -> TeacherAnchoredPreparedArrays:
    """Create only fitting-local optimization tensors from an authenticated pair."""

    if (
        type(pair) is not TeacherAnchoredTrainPair
        or type(seed) is not int
        or seed not in (17, 1729, 65537)
        or pair.source_embeddings.shape != pair.teacher_embeddings.shape
        or pair.source_embeddings.shape[0] != len(pair.labels)
        or pair.source_embeddings.dtype != np.float32
        or pair.teacher_embeddings.dtype != np.float32
        or not pair.source_embeddings.flags.c_contiguous
        or not pair.teacher_embeddings.flags.c_contiguous
    ):
        raise ValueError("teacher-anchored prepared array authority differs")
    split = build_teacher_anchored_split(pair.labels, pair.image_ids, seed=seed)
    fitting_rows = np.asarray(split.fitting_rows, dtype=np.int64)
    source_features = _normalize_snapshot_rows(
        torch.from_numpy(np.ascontiguousarray(pair.source_embeddings[fitting_rows]))
    )
    teacher_features = _normalize_snapshot_rows(
        torch.from_numpy(np.ascontiguousarray(pair.teacher_embeddings[fitting_rows]))
    )
    initialization = initialize_teacher_anchored_student(
        source_features,
        teacher_features,
        expected_teacher_pca_sha256=expected_teacher_pca_sha256,
    )
    teacher_codes = (
        initialization.projection.apply_teacher(teacher_features).float().cpu().contiguous()
    )
    if teacher_codes.shape != (len(split.fitting_rows), 128) or not bool(
        torch.isfinite(teacher_codes).all()
    ):
        raise ValueError("teacher-anchored prepared array authority differs")
    return TeacherAnchoredPreparedArrays(
        split=split,
        initialization=initialization,
        source_features=source_features,
        teacher_codes=teacher_codes,
    )


def bind_teacher_anchored_schedule(
    arguments: TeacherAnchoredArguments,
    prepared: TeacherAnchoredPreparedArrays,
    manifest: TeacherAnchoredImageManifest,
) -> TeacherAnchoredBoundSchedule:
    """Authenticate the shared schedule and map paths to its local row indexes."""

    if (
        type(arguments) is not TeacherAnchoredArguments
        or type(prepared) is not TeacherAnchoredPreparedArrays
        or type(manifest) is not TeacherAnchoredImageManifest
        or manifest.sha256 != arguments.image_tree_sha256
        or len(manifest.image_paths)
        != len(prepared.split.fitting_rows) + len(prepared.split.validation_rows)
        or prepared.source_features.shape[0] != len(prepared.split.fitting_rows)
        or prepared.teacher_codes.shape != (len(prepared.split.fitting_rows), 128)
    ):
        raise ValueError("teacher-anchored schedule binding differs")
    sealed = load_teacher_anchored_schedule_file(
        arguments.schedule,
        expected_sha256=arguments.schedule_sha256,
        teacher_codes=np.ascontiguousarray(prepared.teacher_codes.numpy(), dtype=np.float32),
        sample_ids=prepared.split.fitting_image_ids,
        source_revision=arguments.source_revision,
        source_snapshot_sha256=arguments.source_snapshot_sha256,
        teacher_snapshot_sha256=arguments.teacher_snapshot_sha256,
        split_sha256=prepared.split.sha256,
        seed=arguments.seed,
    )
    fitting_image_paths = tuple(manifest.image_paths[row] for row in prepared.split.fitting_rows)
    anchor_row_indexes = torch.from_numpy(sealed.anchor_schedule.row_indexes.copy()).contiguous()
    if anchor_row_indexes.dtype != torch.int64 or anchor_row_indexes.shape != (
        len(fitting_image_paths),
        512,
    ):
        raise ValueError("teacher-anchored schedule binding differs")
    return TeacherAnchoredBoundSchedule(
        sealed=sealed,
        schedules=teacher_anchored_schedule_rows(sealed),
        anchor_row_indexes=anchor_row_indexes,
        fitting_image_paths=fitting_image_paths,
    )


def validate_teacher_anchored_head_replay(
    initialization: TeacherAnchoredInitialization,
    source_rows: torch.Tensor,
    *,
    device: str,
    expected_receipt: TeacherAnchoredHeadReplayReceipt | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> TeacherAnchoredHeadReplayReceipt:
    """Replay the initialized float32 head against the fitted float64 affine map."""

    if (
        type(initialization) is not TeacherAnchoredInitialization
        or type(source_rows) is not torch.Tensor
        or source_rows.device.type != "cpu"
        or source_rows.dtype != torch.float32
        or source_rows.ndim != 2
        or source_rows.shape[0] < 1
        or not source_rows.is_contiguous()
        or not bool(torch.isfinite(source_rows).all())
        or device not in ("cpu", "cuda")
        or (device == "cuda" and not torch.cuda.is_available())
        or (heartbeat is not None and not callable(heartbeat))
    ):
        raise ValueError("teacher-anchored head replay authority differs")
    source_unit = _normalize_snapshot_rows(source_rows)
    reference_float = initialization.projection.apply_source(source_unit)
    reference = reference_float.double()
    head = initialization.head.to(device=device, dtype=torch.float32)
    head.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, source_rows.shape[0], 256):
            raw = head(source_unit[start : start + 256].to(device=device, dtype=torch.float32))
            chunks.append(torch.nn.functional.normalize(raw.float(), dim=-1).cpu())
            if heartbeat is not None:
                heartbeat()
    runtime = torch.cat(chunks).double()
    maximum_absolute_error = float(torch.max(torch.abs(runtime - reference)))
    cosines = torch.sum(runtime * reference, dim=1) / (
        torch.linalg.vector_norm(runtime, dim=1) * torch.linalg.vector_norm(reference, dim=1)
    )
    minimum_cosine = float(torch.min(cosines))
    if (
        not math.isfinite(maximum_absolute_error)
        or not math.isfinite(minimum_cosine)
        or maximum_absolute_error > 1e-5
        or minimum_cosine < 1.0 - 1e-7
    ):
        raise ValueError("teacher-anchored head replay differs")
    capability = (
        torch.cuda.get_device_capability(torch.device(device)) if device == "cuda" else None
    )
    receipt = TeacherAnchoredHeadReplayReceipt(
        rows=source_rows.shape[0],
        chunk_rows=256,
        device_type=device,
        device_name=torch.cuda.get_device_name(torch.device(device)) if device == "cuda" else "cpu",
        device_capability=(f"{capability[0]}.{capability[1]}" if capability is not None else "cpu"),
        input_sha256=_tensor_sha256(source_rows),
        head_state_sha256=module_state_sha256(initialization.head),
        runtime_codes_sha256=_tensor_sha256(runtime.float().contiguous()),
        reference_codes_sha256=_tensor_sha256(reference_float),
        maximum_absolute_error=maximum_absolute_error,
        minimum_cosine=minimum_cosine,
    )
    if expected_receipt is not None and receipt != expected_receipt:
        raise ValueError("teacher-anchored head replay receipt differs")
    return receipt


def validate_teacher_anchored_snapshot_feature_replay(
    pristine_encoder: nn.Module,
    training_encoder: nn.Module,
    images: torch.Tensor,
    expected_snapshot_codes: torch.Tensor,
    *,
    image_ids: tuple[int, ...],
    device: str,
) -> TeacherAnchoredSnapshotReplayReceipt:
    """Compare pristine single-image and training-batch encodes to one snapshot."""

    if (
        not isinstance(pristine_encoder, nn.Module)
        or not isinstance(training_encoder, nn.Module)
        or type(images) is not torch.Tensor
        or images.device.type != "cpu"
        or images.dtype != torch.float32
        or images.ndim < 2
        or not images.is_contiguous()
        or not bool(torch.isfinite(images).all())
        or type(expected_snapshot_codes) is not torch.Tensor
        or expected_snapshot_codes.device.type != "cpu"
        or expected_snapshot_codes.dtype != torch.float32
        or expected_snapshot_codes.ndim != 2
        or expected_snapshot_codes.shape[0] != images.shape[0]
        or not expected_snapshot_codes.is_contiguous()
        or not bool(torch.isfinite(expected_snapshot_codes).all())
        or type(image_ids) is not tuple
        or len(image_ids) != images.shape[0]
        or any(
            type(identity) is not int or not -(2**63) <= identity < 2**63 for identity in image_ids
        )
        or device not in ("cpu", "cuda")
        or (device == "cuda" and not torch.cuda.is_available())
    ):
        raise ValueError("teacher-anchored snapshot feature replay authority differs")
    expected = _normalize_snapshot_rows(expected_snapshot_codes).double()
    pristine_state = module_state_sha256(pristine_encoder)
    training_state = module_state_sha256(training_encoder)
    pristine_encoder.to(device).eval()
    training_encoder.to(device).train()
    for module in training_encoder.modules():
        if (
            isinstance(
                module,
                (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
            )
            or type(module).__name__ == "DropPath"
        ):
            module.eval()
    with torch.no_grad():
        single = torch.cat(
            [
                pristine_encoder(images[row : row + 1].to(device)).float().cpu()
                for row in range(len(images))
            ]
        )
        batch = training_encoder(images.to(device)).float().cpu()
    single = torch.nn.functional.normalize(single, dim=1).contiguous()
    batch = torch.nn.functional.normalize(batch, dim=1).contiguous()
    if (
        single.shape != expected.shape
        or batch.shape != expected.shape
        or module_state_sha256(pristine_encoder) != pristine_state
        or module_state_sha256(training_encoder) != training_state
    ):
        raise ValueError("teacher-anchored snapshot feature replay differs")
    single64 = single.double()
    batch64 = batch.double()
    maximum_single_error = float(torch.max(torch.abs(single64 - expected)))
    maximum_batch_error = float(torch.max(torch.abs(batch64 - expected)))
    maximum_shape_error = float(torch.max(torch.abs(single64 - batch64)))
    minimum_single_cosine = float(torch.min(torch.sum(single64 * expected, dim=1)))
    minimum_batch_cosine = float(torch.min(torch.sum(batch64 * expected, dim=1)))
    minimum_shape_cosine = float(torch.min(torch.sum(single64 * batch64, dim=1)))
    metrics = (
        maximum_single_error,
        maximum_batch_error,
        maximum_shape_error,
        minimum_single_cosine,
        minimum_batch_cosine,
        minimum_shape_cosine,
    )
    if (
        not all(math.isfinite(value) for value in metrics)
        or maximum_single_error > 0.002
        or maximum_batch_error > 0.002
        or maximum_shape_error > 0.002
        or minimum_single_cosine < 1.0 - 1e-5
        or minimum_batch_cosine < 1.0 - 1e-5
        or minimum_shape_cosine < 1.0 - 1e-5
        or pristine_state != training_state
    ):
        raise ValueError("teacher-anchored snapshot feature replay differs")
    identity_bytes = struct.pack(f"<{len(image_ids)}q", *image_ids)
    return TeacherAnchoredSnapshotReplayReceipt(
        rows=len(images),
        device_type=device,
        device_name=torch.cuda.get_device_name(torch.device(device)) if device == "cuda" else "cpu",
        image_ids_sha256=hashlib.sha256(identity_bytes).hexdigest(),
        images_sha256=_tensor_sha256(images),
        snapshot_codes_sha256=_tensor_sha256(expected_snapshot_codes),
        pristine_state_sha256=pristine_state,
        training_state_sha256=training_state,
        single_codes_sha256=_tensor_sha256(single),
        batch_codes_sha256=_tensor_sha256(batch),
        maximum_single_error=maximum_single_error,
        maximum_batch_error=maximum_batch_error,
        maximum_shape_error=maximum_shape_error,
        minimum_single_cosine=minimum_single_cosine,
        minimum_batch_cosine=minimum_batch_cosine,
        minimum_shape_cosine=minimum_shape_cosine,
    )


def configure_teacher_anchored_epoch(
    encoder: nn.Module,
    head: nn.Linear,
    *,
    epoch: int,
    head_only: bool,
) -> tuple[str, ...]:
    """Apply exact trainability and deterministic model-mode authority."""

    blocks = getattr(encoder, "blocks", None)
    norm = getattr(encoder, "norm", None)
    if (
        type(epoch) is not int
        or not 1 <= epoch <= 10
        or type(head_only) is not bool
        or not isinstance(blocks, nn.ModuleList)
        or len(blocks) != 12
        or not isinstance(norm, nn.Module)
        or type(head) is not nn.Linear
    ):
        raise ValueError("teacher-anchored epoch authority differs")

    encoder.train()
    head.train()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    active: list[str] = []
    if epoch >= 2 and not head_only:
        for block_index in (10, 11):
            for name, parameter in blocks[block_index].named_parameters():
                parameter.requires_grad_(True)
                active.append(f"blocks.{block_index}.{name}")
        for name, parameter in norm.named_parameters():
            parameter.requires_grad_(True)
            active.append(f"norm.{name}")
    active.extend(f"head.{name}" for name, _parameter in head.named_parameters())

    batch_norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for module in encoder.modules():
        if isinstance(module, batch_norm_types) or type(module).__name__ == "DropPath":
            module.eval()
    return tuple(active)


def build_teacher_anchored_optimizer(
    encoder: nn.Module,
    head: nn.Linear,
    *,
    epoch: int,
    head_only: bool = False,
) -> torch.optim.AdamW:
    """Construct fresh exact AdamW groups for epoch one or epoch two onward."""

    if type(epoch) is not int or epoch not in (1, 2) or type(head_only) is not bool:
        raise ValueError("teacher-anchored optimizer authority differs")
    encoder_trainable = any(parameter.requires_grad for parameter in encoder.parameters())
    if (epoch == 1 or head_only) == encoder_trainable:
        raise ValueError("teacher-anchored optimizer authority differs")
    groups: list[dict[str, object]] = []
    for prefix, module, learning_rate in (
        ("head", head, 1e-4),
        ("backbone", encoder, 1e-6),
    ):
        for decay in (True, False):
            parameters = [
                parameter
                for name, parameter in module.named_parameters()
                if parameter.requires_grad and ((not name.endswith("bias")) is decay)
            ]
            if parameters:
                groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": 0.01 if decay else 0.0,
                        "group_name": f"{prefix}-{'decay' if decay else 'no-decay'}",
                    }
                )
    if not groups or (
        epoch == 2
        and not head_only
        and not any(group["group_name"] == "backbone-decay" for group in groups)
    ):
        raise ValueError("teacher-anchored optimizer authority differs")
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def build_teacher_anchored_grad_scaler(device_type: str) -> GradScaler:
    """Construct the registered fixed-growth mixed-precision scaler."""

    if device_type not in ("cpu", "cuda"):
        raise ValueError("teacher-anchored scaler authority differs")
    if device_type == "cuda" and not torch.cuda.is_available():
        raise ValueError("teacher-anchored scaler authority differs")
    return GradScaler(
        device_type,
        init_scale=1024.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2**31 - 1,
        enabled=True,
    )


def teacher_anchored_optimizer_step(
    loss: torch.Tensor,
    optimizer: torch.optim.AdamW,
    scaler: GradScaler,
    parameters: tuple[nn.Parameter, ...],
) -> float:
    """Perform one clipped update and reject nonfinite or skipped updates."""

    if (
        type(loss) is not torch.Tensor
        or loss.ndim != 0
        or loss.dtype != torch.float32
        or type(optimizer) is not torch.optim.AdamW
        or type(scaler) is not GradScaler
        or not parameters
        or any(type(parameter) is not nn.Parameter for parameter in parameters)
    ):
        raise ValueError("teacher-anchored update authority differs")
    if not bool(torch.isfinite(loss)):
        raise TeacherAnchoredNonfiniteUpdate("teacher-anchored nonfinite update")
    optimizer.zero_grad(set_to_none=True)
    scale_before = scaler.get_scale()
    scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
    scaler.unscale_(optimizer)
    if any(parameter.grad is None for parameter in parameters):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("teacher-anchored update authority differs")
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=1.0,
        error_if_nonfinite=False,
    )
    if not bool(torch.isfinite(gradient_norm)):
        optimizer.zero_grad(set_to_none=True)
        raise TeacherAnchoredNonfiniteUpdate("teacher-anchored nonfinite update")
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < scale_before or any(
        not bool(torch.isfinite(parameter).all()) for parameter in parameters
    ):
        raise TeacherAnchoredNonfiniteUpdate("teacher-anchored nonfinite update")
    return float(gradient_norm)


def teacher_anchored_lr_factor(update: int, *, total_updates: int) -> float:
    """Return the exact warmup-plus-cosine epoch-two-to-ten LR factor."""

    if (
        type(update) is not int
        or type(total_updates) is not int
        or total_updates <= 100
        or not 0 <= update < total_updates
    ):
        raise ValueError("teacher-anchored schedule authority differs")
    if update < 100:
        return (update + 1) / 100
    return 0.5 * (1.0 + math.cos(math.pi * (update - 99) / (total_updates - 100)))


def teacher_anchored_arm_objective(loss: TeacherAnchoredLoss, *, arm: str) -> torch.Tensor:
    """Select one exact nested objective without changing its component reductions."""

    if type(loss) is not TeacherAnchoredLoss or arm not in (
        "head-only",
        "base",
        "anchor",
        "symmetric",
        "complete",
    ):
        raise ValueError("teacher-anchored arm objective differs")
    components = (
        loss.total,
        loss.anchor,
        loss.point,
        loss.symmetric,
        loss.drift,
        loss.covariance,
    )
    if any(
        type(component) is not torch.Tensor
        or component.ndim != 0
        or component.dtype != torch.float32
        for component in components
    ):
        raise ValueError("teacher-anchored arm objective authority differs")
    if not all(bool(torch.isfinite(component)) for component in components):
        raise TeacherAnchoredNonfiniteUpdate("teacher-anchored nonfinite update")
    base = 0.1 * loss.point + 0.05 * loss.drift + 0.01 * loss.covariance
    if arm in ("anchor", "complete", "head-only"):
        base = base + loss.anchor
    if arm in ("symmetric", "complete", "head-only"):
        base = base + 0.5 * loss.symmetric
    if not bool(torch.isfinite(base)):
        raise TeacherAnchoredNonfiniteUpdate("teacher-anchored nonfinite update")
    return base


def teacher_anchored_objective_sha256(arm: str) -> str:
    """Bind one nested arm to the complete frozen objective recipe."""

    if arm not in ("head-only", "base", "anchor", "symmetric", "complete"):
        raise ValueError("teacher-anchored artifact authority differs")
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "anchor_weight": 1.0 if arm in ("head-only", "anchor", "complete") else 0.0,
                "arm": arm,
                "covariance_weight": 0.01,
                "drift_weight": 0.05,
                "point_weight": 0.1,
                "symmetric_weight": 0.5 if arm in ("head-only", "symmetric", "complete") else 0.0,
                "temperatures": [0.05, 0.2],
            }
        )
    ).hexdigest()


def teacher_anchored_runtime_sha256(receipt: TeacherAnchoredRuntimeReceipt) -> str:
    """Bind every deterministic runtime setting to one canonical digest."""

    if type(receipt) is not TeacherAnchoredRuntimeReceipt:
        raise ValueError("teacher-anchored runtime receipt differs")
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "runtime": receipt._asdict(),
                "schema": "sfora-teacher-anchored-runtime-v1",
            }
        )
    ).hexdigest()


def teacher_anchored_model_mode_sha256(arm: str) -> str:
    """Bind the fixed per-epoch train/eval and trainability recipe."""

    if arm not in ("head-only", "base", "anchor", "symmetric", "complete"):
        raise ValueError("teacher-anchored model mode differs")
    later = ["head"] if arm == "head-only" else ["blocks.10", "blocks.11", "norm", "head"]
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "batch_norm": "eval",
                "drop_path": "eval",
                "epoch_1_trainable": ["head"],
                "epochs_2_to_10_trainable": later,
                "schema": "sfora-teacher-anchored-model-mode-v1",
            }
        )
    ).hexdigest()


def teacher_anchored_trainable_inventory_sha256(encoder: nn.Module, head: nn.Linear) -> str:
    """Bind names, shapes, dtypes, and trainability of every live parameter."""

    if not isinstance(encoder, nn.Module) or type(head) is not nn.Linear:
        raise ValueError("teacher-anchored trainable inventory differs")
    parameters = tuple(encoder.named_parameters()) + tuple(
        (f"head.{name}", parameter) for name, parameter in head.named_parameters()
    )
    if not parameters:
        raise ValueError("teacher-anchored trainable inventory differs")
    rows = [
        {
            "dtype": str(parameter.dtype),
            "name": name,
            "shape": list(parameter.shape),
            "trainable": parameter.requires_grad,
        }
        for name, parameter in parameters
    ]
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "parameters": rows,
                "schema": "sfora-teacher-anchored-trainable-inventory-v1",
            }
        )
    ).hexdigest()


def teacher_anchored_ridge_sha256(initialization: TeacherAnchoredInitialization) -> str:
    """Bind the exact fitted source-to-teacher-PCA affine state."""

    if type(initialization) is not TeacherAnchoredInitialization:
        raise ValueError("teacher-anchored ridge authority differs")
    projection = initialization.projection.source_projection
    weight = projection.weight.detach().cpu().float().contiguous()
    bias = projection.bias.detach().cpu().float().contiguous()
    parameter_sha256 = _parameter_sha256(weight, bias)
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "dimensions": int(weight.shape[0]),
                "input_dimensions": int(weight.shape[1]),
                "parameter_sha256": parameter_sha256,
                "penalty": 1e-6,
                "schema": "sfora-teacher-anchored-ridge-v1",
            }
        )
    ).hexdigest()


def teacher_anchored_schedule_sha256(
    schedules: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    fitting_row_count: int,
) -> str:
    """Hash the exact ten-epoch row schedule without JSON-number ambiguity."""

    _validate_teacher_anchored_schedules(schedules, fitting_row_count=fitting_row_count)
    digest = hashlib.sha256(b"sfora-teacher-anchored-schedules-v1\x00")
    digest.update(struct.pack("<Q", fitting_row_count))
    digest.update(struct.pack("<I", len(schedules)))
    for epoch_batches in schedules:
        digest.update(struct.pack("<Q", len(epoch_batches)))
        for rows in epoch_batches:
            digest.update(struct.pack("<Q", len(rows)))
            digest.update(struct.pack(f"<{len(rows)}q", *rows))
    return digest.hexdigest()


def load_teacher_anchored_probe_scorer() -> tuple[Callable[..., object], Path]:
    """Load the exact repository probe scorer after enabling sibling imports."""

    path = Path(__file__).resolve().parent / "probe_sop_relational_linear.py"
    scripts_directory = str(path.parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    module_name = "sfora_teacher_anchored_probe_scorer"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("teacher-anchored probe scorer authority differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    scorer = getattr(module, "score_symmetric", None)
    if not callable(scorer):
        raise ValueError("teacher-anchored probe scorer authority differs")
    return cast(Callable[..., object], scorer), path


def score_teacher_anchored_probe(
    codes: torch.Tensor, labels: tuple[int, ...], *, device: torch.device
) -> TeacherAnchoredProbeScore:
    """Score one live code matrix through the exact float and serving paths."""

    if (
        type(codes) is not torch.Tensor
        or codes.device.type != "cpu"
        or codes.dtype != torch.float32
        or codes.ndim != 2
        or codes.shape[0] < 2
        or not codes.is_contiguous()
        or not bool(torch.isfinite(codes).all())
        or type(labels) is not tuple
        or len(labels) != len(codes)
        or any(type(label) is not int or label < 1 for label in labels)
        or type(device) is not torch.device
    ):
        raise ValueError("teacher-anchored probe authority differs")
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    if any(count < 2 for count in counts.values()):
        raise ValueError("teacher-anchored probe authority differs")
    candidate_width = max(counts.values()) - 1
    scorer, _path = load_teacher_anchored_probe_scorer()
    float_score = scorer(codes, labels, candidate_width=candidate_width, device=device)
    packed_score = scorer(
        pack_int8_unit_embeddings(codes),
        labels,
        candidate_width=candidate_width,
        device=device,
    )
    if type(float_score) is not dict or type(packed_score) is not dict:
        raise ValueError("teacher-anchored probe authority differs")
    values = (
        float_score.get("map_at_r"),
        float_score.get("r1"),
        packed_score.get("map_at_r"),
        packed_score.get("r1"),
    )
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("teacher-anchored probe authority differs")
    return TeacherAnchoredProbeScore(
        candidate_width, *cast(tuple[float, float, float, float], values)
    )


def make_teacher_anchored_diagnose(
    encoder: nn.Module,
    head: nn.Linear,
    *,
    fitting_image_paths: tuple[Path, ...],
    fitting_labels: tuple[int, ...],
    validation_image_paths: tuple[Path, ...],
    validation_labels: tuple[int, ...],
    transform_image: Callable[[Path], torch.Tensor],
    device: torch.device,
    heartbeat: Callable[[], None] | None = None,
) -> TeacherAnchoredDiagnose:
    """Build the fixed-chunk live scorer while preserving training state exactly."""

    def valid_population(paths: tuple[Path, ...], labels: tuple[int, ...]) -> bool:
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return (
            type(paths) is tuple
            and len(paths) >= 2
            and all(isinstance(path, Path) and path.is_absolute() for path in paths)
            and type(labels) is tuple
            and len(labels) == len(paths)
            and all(type(label) is int and label >= 1 for label in labels)
            and all(count >= 2 for count in counts.values())
        )

    if (
        not isinstance(encoder, nn.Module)
        or type(head) is not nn.Linear
        or not valid_population(fitting_image_paths, fitting_labels)
        or not valid_population(validation_image_paths, validation_labels)
        or not callable(transform_image)
        or (heartbeat is not None and not callable(heartbeat))
        or type(device) is not torch.device
        or any(
            parameter.device != device for parameter in (*encoder.parameters(), *head.parameters())
        )
    ):
        raise ValueError("teacher-anchored diagnostic labels differ")

    def encode(paths: tuple[Path, ...]) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(paths), 64):
            transformed = tuple(transform_image(path) for path in paths[start : start + 64])
            if (
                not transformed
                or any(type(image) is not torch.Tensor for image in transformed)
                or any(
                    image.device.type != "cpu"
                    or image.dtype != torch.float32
                    or not image.is_contiguous()
                    or not bool(torch.isfinite(image).all())
                    for image in transformed
                )
                or any(image.shape != transformed[0].shape for image in transformed)
            ):
                raise ValueError("teacher-anchored diagnostic image differs")
            retained = len(transformed)
            padded = (*transformed, *(transformed[-1] for _ in range(64 - retained)))
            images = torch.stack(padded).to(device=device, non_blocking=False)
            with torch.inference_mode():
                _features, codes = teacher_anchored_forward(encoder, head, images)
            chunks.append(codes[:retained].float().cpu().contiguous())
            if heartbeat is not None:
                heartbeat()
        return torch.cat(chunks).contiguous()

    step_zero: TeacherAnchoredEpochDiagnostic | None = None

    def diagnose(epoch: int) -> TeacherAnchoredEpochDiagnostic:
        nonlocal step_zero
        if type(epoch) is not int or not 0 <= epoch <= 10:
            raise ValueError("teacher-anchored diagnostic authority differs")
        if epoch == 0 and step_zero is not None:
            return step_zero
        modules = tuple(encoder.modules()) + tuple(head.modules())
        training_states = tuple(module.training for module in modules)
        parameters = tuple(encoder.parameters()) + tuple(head.parameters())
        gradient_states = tuple(parameter.requires_grad for parameter in parameters)
        try:
            encoder.eval()
            head.eval()
            fitting_codes = encode(fitting_image_paths)
            validation_codes = encode(validation_image_paths)
            fitting_score = score_teacher_anchored_probe(
                fitting_codes, fitting_labels, device=device
            )
            validation_score = score_teacher_anchored_probe(
                validation_codes, validation_labels, device=device
            )
            geometry = embedding_geometry_diagnostics(fitting_codes)
            result = TeacherAnchoredEpochDiagnostic(
                epoch=epoch,
                fitting_map_at_r=fitting_score.packed_map_at_r,
                fitting_effective_rank=geometry.effective_rank,
                fitting_leading_eigenvalue_share=geometry.leading_eigenvalue_share,
                validation_map_at_r=validation_score.packed_map_at_r,
            )
        finally:
            for module, training in zip(modules, training_states, strict=True):
                module.training = training
            for parameter, requires_grad in zip(parameters, gradient_states, strict=True):
                parameter.requires_grad_(requires_grad)
        if epoch == 0:
            step_zero = result
        return result

    return diagnose


def compute_teacher_anchored_snapshot_diagnostic(
    head: nn.Linear,
    *,
    fitting_features: torch.Tensor,
    fitting_labels: tuple[int, ...],
    fitting_probe_rows: tuple[int, ...],
    validation_features: torch.Tensor,
    validation_labels: tuple[int, ...],
    device: torch.device,
) -> TeacherAnchoredEpochDiagnostic:
    """Compute the independent step-zero reference from authenticated snapshots."""

    if (
        type(head) is not nn.Linear
        or type(device) is not torch.device
        or type(fitting_features) is not torch.Tensor
        or type(validation_features) is not torch.Tensor
        or fitting_features.device.type != "cpu"
        or validation_features.device.type != "cpu"
        or fitting_features.dtype != torch.float32
        or validation_features.dtype != torch.float32
        or fitting_features.ndim != 2
        or validation_features.ndim != 2
        or fitting_features.shape[1] != head.in_features
        or validation_features.shape[1] != head.in_features
        or not fitting_features.is_contiguous()
        or not validation_features.is_contiguous()
        or type(fitting_labels) is not tuple
        or len(fitting_labels) != len(fitting_features)
        or type(validation_labels) is not tuple
        or len(validation_labels) != len(validation_features)
        or type(fitting_probe_rows) is not tuple
        or not fitting_probe_rows
        or len(set(fitting_probe_rows)) != len(fitting_probe_rows)
        or any(
            type(row) is not int or not 0 <= row < len(fitting_features)
            for row in fitting_probe_rows
        )
    ):
        raise ValueError("teacher-anchored snapshot diagnostic authority differs")

    def encode(features: torch.Tensor) -> torch.Tensor:
        _normalize_snapshot_rows(features)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(features), 64):
            retained = features[start : start + 64]
            padded = torch.cat(
                (retained, retained[-1:].expand(64 - len(retained), -1)),
                dim=0,
            )
            with torch.inference_mode():
                normalized = torch.nn.functional.normalize(padded.to(device), dim=1)
                raw_codes = head(normalized).float()
                code_norms = torch.linalg.vector_norm(raw_codes.double(), dim=1)
                if not bool(torch.isfinite(raw_codes).all()) or bool((code_norms <= 1e-12).any()):
                    raise TeacherAnchoredNumericalError(
                        "teacher-anchored snapshot diagnostic numerical failure"
                    )
                codes = torch.nn.functional.normalize(raw_codes, dim=1)
            chunks.append(codes[: len(retained)].cpu().contiguous())
        return torch.cat(chunks).contiguous()

    fitting_codes = encode(fitting_features)
    validation_codes = encode(validation_features)
    probe_indexes = torch.tensor(
        fitting_probe_rows,
        dtype=torch.int64,
        device=fitting_codes.device,
    )
    probe_codes = fitting_codes[probe_indexes].cpu().contiguous()
    probe_labels = tuple(fitting_labels[row] for row in fitting_probe_rows)
    validation_codes = validation_codes.cpu().contiguous()
    fitting_score = score_teacher_anchored_probe(probe_codes, probe_labels, device=device)
    validation_score = score_teacher_anchored_probe(
        validation_codes, validation_labels, device=device
    )
    geometry = embedding_geometry_diagnostics(probe_codes)
    return TeacherAnchoredEpochDiagnostic(
        epoch=0,
        fitting_map_at_r=fitting_score.packed_map_at_r,
        fitting_effective_rank=geometry.effective_rank,
        fitting_leading_eigenvalue_share=geometry.leading_eigenvalue_share,
        validation_map_at_r=validation_score.packed_map_at_r,
    )


def build_teacher_anchored_split(
    labels: np.ndarray, image_ids: np.ndarray, *, seed: int
) -> TeacherAnchoredSplit:
    """Seal the class-disjoint split and its fitting-local index space."""

    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.ndim != 1
        or labels.size < 3
        or not labels.flags.c_contiguous
        or type(image_ids) is not np.ndarray
        or image_ids.dtype != np.int64
        or image_ids.shape != labels.shape
        or not image_ids.flags.c_contiguous
        or np.unique(image_ids).size != image_ids.size
        or type(seed) is not int
        or seed not in (17, 1729, 65537)
    ):
        raise ValueError("teacher-anchored split authority differs")
    concrete_labels = tuple(int(value) for value in labels)
    partition = deterministic_class_partition(concrete_labels, fit_fraction=0.8, seed=seed)
    fitting_rows = partition.fit_row_indexes
    validation_rows = partition.validation_row_indexes
    fitting_labels = tuple(concrete_labels[row] for row in fitting_rows)
    validation_labels = tuple(concrete_labels[row] for row in validation_rows)
    concrete_image_ids = tuple(int(value) for value in image_ids)
    fitting_image_ids = tuple(concrete_image_ids[row] for row in fitting_rows)
    validation_image_ids = tuple(concrete_image_ids[row] for row in validation_rows)
    digest = hashlib.sha256(b"sfora-teacher-anchored-split-v1\x00")
    digest.update(labels.astype("<i8", copy=False).tobytes(order="C"))
    digest.update(image_ids.astype("<i8", copy=False).tobytes(order="C"))
    for values in (
        fitting_rows,
        validation_rows,
        partition.fit_class_ids,
        partition.validation_class_ids,
    ):
        digest.update(struct.pack("<Q", len(values)))
        digest.update(struct.pack(f"<{len(values)}q", *values))
    return TeacherAnchoredSplit(
        fitting_rows,
        validation_rows,
        partition.fit_class_ids,
        partition.validation_class_ids,
        fitting_labels,
        validation_labels,
        fitting_image_ids,
        validation_image_ids,
        digest.hexdigest(),
    )


def select_teacher_anchored_fitting_probe(
    labels: np.ndarray, *, seed: int
) -> TeacherAnchoredFittingProbeSelection:
    """Select complete positive sets for the fixed fitting-only stop diagnostic."""

    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.ndim != 1
        or labels.size < 1024
        or not labels.flags.c_contiguous
        or type(seed) is not int
        or seed not in (17, 1729, 65537)
    ):
        raise ValueError("teacher-anchored fitting probe authority differs")
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 512 or bool((counts < 2).any()):
        raise ValueError("teacher-anchored fitting probe authority differs")
    ordered = sorted(
        (int(value) for value in classes),
        key=lambda value: (hashlib.sha256(struct.pack("<Qq", seed, value)).digest(), value),
    )
    selected = tuple(ordered[:512])
    selected_set = set(selected)
    rows = tuple(row for row, label in enumerate(labels) if int(label) in selected_set)
    digest = hashlib.sha256(b"sfora-teacher-anchored-fitting-probe-v1\x00")
    digest.update(labels.astype("<i8", copy=False).tobytes(order="C"))
    digest.update(struct.pack(f"<{len(selected)}q", *selected))
    digest.update(struct.pack(f"<{len(rows)}q", *rows))
    return TeacherAnchoredFittingProbeSelection(selected, rows, digest.hexdigest())


def teacher_anchored_bootstrap_lower_bound(
    treatment: tuple[float, ...],
    baseline: tuple[float, ...],
    identities: tuple[int, ...],
    *,
    expected_lower_bound: float,
) -> float:
    """Replay the frozen Torch class-cluster lower-bound estimator exactly."""

    path = Path(__file__).resolve().parent / "probe_representation_ceiling.py"
    scripts_directory = str(path.parent)
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    module_name = "sfora_teacher_anchored_bootstrap"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("teacher-anchored bootstrap authority differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    estimator = getattr(module, "class_cluster_lower_bound", None)
    if not callable(estimator):
        raise ValueError("teacher-anchored bootstrap authority differs")
    lower = estimator(treatment, baseline, identities, seed=17, samples=10_000)
    if type(lower) is not float or not math.isfinite(lower):
        raise ValueError("teacher-anchored bootstrap authority differs")
    if (
        type(expected_lower_bound) is not float
        or not math.isfinite(expected_lower_bound)
        or lower != expected_lower_bound
    ):
        raise ValueError("teacher-anchored bootstrap replay differs")
    return lower


def replay_teacher_anchored_bootstrap(
    train_labels: np.ndarray,
    *,
    labels_sha256: str,
    splits: tuple[TeacherAnchoredBootstrapSplit, ...],
    expected_lower_bound: float,
) -> float:
    """Authenticate labels and replay the exact ordered three-split estimator."""

    if (
        type(train_labels) is not np.ndarray
        or train_labels.dtype != np.dtype("<i8")
        or train_labels.ndim != 1
        or not train_labels.flags.c_contiguous
        or hashlib.sha256(train_labels.tobytes(order="C")).hexdigest() != labels_sha256
        or type(splits) is not tuple
        or len(splits) != 3
        or type(expected_lower_bound) is not float
        or not math.isfinite(expected_lower_bound)
    ):
        raise ValueError("teacher-anchored bootstrap authority differs")
    treatment: list[float] = []
    baseline: list[float] = []
    identities: list[int] = []
    for split in splits:
        if (
            type(split) is not TeacherAnchoredBootstrapSplit
            or not split.row_indexes
            or len(split.row_indexes) != len(split.treatment)
            or len(split.row_indexes) != len(split.baseline)
            or len(set(split.row_indexes)) != len(split.row_indexes)
            or any(
                type(row) is not int or not 0 <= row < len(train_labels)
                for row in split.row_indexes
            )
        ):
            raise ValueError("teacher-anchored bootstrap authority differs")
        treatment.extend(split.treatment)
        baseline.extend(split.baseline)
        identities.extend(int(train_labels[row]) for row in split.row_indexes)
    if tuple(split.seed for split in splits) != (17, 1729, 65537):
        raise ValueError("teacher-anchored bootstrap authority differs")
    concrete_labels = tuple(int(value) for value in train_labels)
    if any(
        split.row_indexes
        != deterministic_class_partition(
            concrete_labels, fit_fraction=0.8, seed=split.seed
        ).validation_row_indexes
        for split in splits
    ):
        raise ValueError("teacher-anchored bootstrap authority differs")
    return teacher_anchored_bootstrap_lower_bound(
        tuple(treatment),
        tuple(baseline),
        tuple(identities),
        expected_lower_bound=expected_lower_bound,
    )


def compute_teacher_anchored_batch_loss(
    encoder: nn.Module,
    head: nn.Linear,
    *,
    row_indexes: tuple[int, ...],
    active_parameter_names: tuple[str, ...],
    image_paths: tuple[Path, ...],
    transform_image: Callable[[Path], torch.Tensor],
    source_features: torch.Tensor,
    teacher_codes: torch.Tensor,
    anchor_row_indexes: torch.Tensor,
    config: TeacherAnchoredConfig,
    device: torch.device,
) -> TeacherAnchoredLoss:
    """Map one fitting-local schedule batch into the generic objective."""

    actual_active = tuple(
        name for name, parameter in encoder.named_parameters() if parameter.requires_grad
    ) + tuple(
        f"head.{name}" for name, parameter in head.named_parameters() if parameter.requires_grad
    )
    row_count = len(image_paths) if type(image_paths) is tuple else -1
    if (
        not isinstance(encoder, nn.Module)
        or type(head) is not nn.Linear
        or type(config) is not TeacherAnchoredConfig
        or type(device) is not torch.device
        or type(row_indexes) is not tuple
        or len(row_indexes) != config.batch_rows
        or any(type(row) is not int or not 0 <= row < row_count for row in row_indexes)
        or len(set(row_indexes)) != len(row_indexes)
        or type(active_parameter_names) is not tuple
        or active_parameter_names != actual_active
        or any(not isinstance(path, Path) or not path.is_absolute() for path in image_paths)
        or not callable(transform_image)
        or type(source_features) is not torch.Tensor
        or source_features.device.type != "cpu"
        or source_features.dtype != torch.float32
        or source_features.shape != (row_count, head.in_features)
        or not source_features.is_contiguous()
        or type(teacher_codes) is not torch.Tensor
        or teacher_codes.device.type != "cpu"
        or teacher_codes.dtype != torch.float32
        or teacher_codes.shape != (row_count, config.dimensions)
        or not teacher_codes.is_contiguous()
        or type(anchor_row_indexes) is not torch.Tensor
        or anchor_row_indexes.device.type != "cpu"
        or anchor_row_indexes.dtype != torch.int64
        or anchor_row_indexes.shape != (row_count, config.anchor_count)
        or not anchor_row_indexes.is_contiguous()
        or bool((anchor_row_indexes < 0).any())
        or bool((anchor_row_indexes >= row_count).any())
        or not bool(torch.isfinite(source_features).all())
        or not bool(torch.isfinite(teacher_codes).all())
    ):
        raise ValueError("teacher-anchored batch authority differs")
    transformed = tuple(transform_image(image_paths[row]) for row in row_indexes)
    if (
        any(type(image) is not torch.Tensor for image in transformed)
        or any(image.device.type != "cpu" or image.dtype != torch.float32 for image in transformed)
        or any(image.shape != transformed[0].shape for image in transformed)
        or any(not bool(torch.isfinite(image).all()) for image in transformed)
    ):
        raise ValueError("teacher-anchored batch authority differs")
    images = torch.stack(transformed).to(device)
    adapted_features, student_codes = teacher_anchored_forward(encoder, head, images)
    indexes = torch.tensor(row_indexes, dtype=torch.int64)
    batch_teacher = teacher_codes[indexes].to(device)
    batch_anchors = teacher_codes[anchor_row_indexes[indexes]].to(device)
    batch_source = source_features[indexes].to(device)
    return teacher_anchored_loss(
        student_codes,
        batch_teacher,
        batch_anchors,
        adapted_features,
        batch_source,
        config,
    )


def compute_teacher_anchored_snapshot_batch_loss(
    head: nn.Linear,
    *,
    row_indexes: tuple[int, ...],
    active_parameter_names: tuple[str, ...],
    source_features: torch.Tensor,
    teacher_codes: torch.Tensor,
    anchor_row_indexes: torch.Tensor,
    config: TeacherAnchoredConfig,
    device: torch.device,
) -> TeacherAnchoredLoss:
    """Compute the head-only control from authenticated frozen source features."""

    row_count = source_features.shape[0] if type(source_features) is torch.Tensor else -1
    if (
        type(head) is not nn.Linear
        or type(config) is not TeacherAnchoredConfig
        or type(device) is not torch.device
        or active_parameter_names != ("head.weight", "head.bias")
        or type(row_indexes) is not tuple
        or len(row_indexes) != config.batch_rows
        or len(set(row_indexes)) != len(row_indexes)
        or any(type(row) is not int or not 0 <= row < row_count for row in row_indexes)
        or type(source_features) is not torch.Tensor
        or source_features.device.type != "cpu"
        or source_features.dtype != torch.float32
        or source_features.ndim != 2
        or source_features.shape[1] != head.in_features
        or not source_features.is_contiguous()
        or type(teacher_codes) is not torch.Tensor
        or teacher_codes.device.type != "cpu"
        or teacher_codes.dtype != torch.float32
        or teacher_codes.shape != (row_count, config.dimensions)
        or not teacher_codes.is_contiguous()
        or type(anchor_row_indexes) is not torch.Tensor
        or anchor_row_indexes.device.type != "cpu"
        or anchor_row_indexes.dtype != torch.int64
        or anchor_row_indexes.shape != (row_count, config.anchor_count)
        or not anchor_row_indexes.is_contiguous()
        or bool((anchor_row_indexes < 0).any())
        or bool((anchor_row_indexes >= row_count).any())
        or not bool(torch.isfinite(source_features).all())
        or not bool(torch.isfinite(teacher_codes).all())
    ):
        raise ValueError("teacher-anchored snapshot batch authority differs")
    indexes = torch.tensor(row_indexes, dtype=torch.int64)
    original = source_features[indexes].to(device)
    student = torch.nn.functional.normalize(head(original).float(), dim=1)
    batch_teacher = teacher_codes[indexes].to(device)
    batch_anchors = teacher_codes[anchor_row_indexes[indexes]].to(device)
    return teacher_anchored_loss(
        student,
        batch_teacher,
        batch_anchors,
        original,
        original,
        config,
    )


def run_teacher_anchored_training(
    encoder: nn.Module,
    head: nn.Linear,
    *,
    arm: str,
    schedules: tuple[tuple[tuple[int, ...], ...], ...],
    fitting_row_count: int,
    expected_schedule_sha256: str,
    initialization_fitting_map_at_r: float,
    initialization_effective_rank: float,
    initialization_leading_eigenvalue_share: float,
    compute_loss: TeacherAnchoredComputeLoss,
    diagnose: TeacherAnchoredDiagnose,
    progress: TeacherAnchoredProgress,
    device_type: str,
) -> TeacherAnchoredTrainingReceipt:
    """Execute the fixed loop while owning optimization and state integrity."""

    if (
        not isinstance(encoder, nn.Module)
        or type(head) is not nn.Linear
        or arm not in ("head-only", "base", "anchor", "symmetric", "complete")
        or type(initialization_fitting_map_at_r) is not float
        or type(initialization_effective_rank) is not float
        or type(initialization_leading_eigenvalue_share) is not float
        or not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                initialization_fitting_map_at_r,
                initialization_effective_rank,
                initialization_leading_eigenvalue_share,
            )
        )
        or type(fitting_row_count) is not int
        or fitting_row_count < 256
        or not _is_sha256(expected_schedule_sha256)
        or not callable(compute_loss)
        or not callable(diagnose)
        or not callable(progress)
        or device_type not in ("cpu", "cuda")
        or head.weight.device.type != device_type
        or any(parameter.device.type != device_type for parameter in encoder.parameters())
    ):
        raise ValueError("teacher-anchored training authority differs")
    schedule_sha256 = teacher_anchored_schedule_sha256(
        schedules, fitting_row_count=fitting_row_count
    )
    if schedule_sha256 != expected_schedule_sha256:
        raise ValueError("teacher-anchored training schedule differs")
    head_only = arm == "head-only"
    if not _module_state_is_finite(encoder) or not _module_state_is_finite(head):
        raise ValueError("teacher-anchored training authority differs")
    initial_encoder_sha256 = module_state_sha256(encoder)
    initial_head_sha256 = module_state_sha256(head)
    initial_frozen_sha256 = encoder_frozen_state_sha256(encoder, head_only=head_only)
    total_later_updates = sum(len(epoch_batches) for epoch_batches in schedules[1:])
    if total_later_updates <= 100:
        raise ValueError("teacher-anchored training schedule differs")
    initialization_diagnostic = diagnose(0)
    if (
        type(initialization_diagnostic) is not TeacherAnchoredEpochDiagnostic
        or initialization_diagnostic.epoch != 0
        or not all(math.isfinite(value) for value in initialization_diagnostic[1:])
        or initialization_diagnostic.fitting_map_at_r != initialization_fitting_map_at_r
        or initialization_diagnostic.fitting_effective_rank != initialization_effective_rank
        or initialization_diagnostic.fitting_leading_eigenvalue_share
        != initialization_leading_eigenvalue_share
    ):
        raise ValueError("teacher-anchored live step-zero differs")

    optimizer: torch.optim.AdamW | None = None
    scaler: GradScaler | None = None
    later_update = 0
    attempted_updates = 0
    successful_updates = 0
    completed_epochs: list[int] = []
    diagnostics: list[TeacherAnchoredEpochDiagnostic] = [initialization_diagnostic]
    optimizer_reset_epochs: list[int] = []
    stopped_reason: str | None = None
    progress("initialized", 0, 0)
    for epoch, epoch_batches in enumerate(schedules, start=1):
        active = configure_teacher_anchored_epoch(
            encoder,
            head,
            epoch=epoch,
            head_only=head_only,
        )
        if epoch in (1, 2):
            optimizer = build_teacher_anchored_optimizer(
                encoder,
                head,
                epoch=epoch,
                head_only=head_only,
            )
            scaler = build_teacher_anchored_grad_scaler(device_type)
            optimizer_reset_epochs.append(epoch)
        if optimizer is None or scaler is None:
            raise ValueError("teacher-anchored optimizer authority differs")
        for update_index, rows in enumerate(epoch_batches):
            factor = (
                1.0
                if epoch == 1
                else teacher_anchored_lr_factor(later_update, total_updates=total_later_updates)
            )
            for group in optimizer.param_groups:
                group_name = group.get("group_name")
                if type(group_name) is not str:
                    raise ValueError("teacher-anchored optimizer authority differs")
                group["lr"] = (1e-4 if group_name.startswith("head-") else 1e-6) * factor
            attempted_updates += 1
            parameters = tuple(
                parameter
                for parameter in (*encoder.parameters(), *head.parameters())
                if parameter.requires_grad
            )
            finite_state = tuple(parameter.detach().clone() for parameter in parameters)
            try:
                loss = compute_loss(epoch, update_index, rows, active)
                objective = teacher_anchored_arm_objective(loss, arm=arm)
                teacher_anchored_optimizer_step(objective, optimizer, scaler, parameters)
            except (TeacherAnchoredNonfiniteUpdate, TeacherAnchoredNumericalError):
                with torch.no_grad():
                    for parameter, value in zip(parameters, finite_state, strict=True):
                        parameter.copy_(value)
                optimizer.zero_grad(set_to_none=True)
                stopped_reason = "nonfinite-update"
                progress("stopped", epoch, successful_updates)
                break
            successful_updates += 1
            progress("update", epoch, successful_updates)
            if epoch >= 2:
                later_update += 1
        if stopped_reason is not None:
            break
        if epoch == 1 and module_state_sha256(encoder) != initial_encoder_sha256:
            raise ValueError("teacher-anchored frozen state differs")
        diagnostic = diagnose(epoch)
        if (
            type(diagnostic) is not TeacherAnchoredEpochDiagnostic
            or diagnostic.epoch != epoch
            or not all(math.isfinite(value) for value in diagnostic[1:])
        ):
            raise ValueError("teacher-anchored diagnostic authority differs")
        diagnostics.append(diagnostic)
        completed_epochs.append(epoch)
        progress("epoch-complete", epoch, successful_updates)
        if epoch == 1:
            if diagnostic.fitting_map_at_r < initialization_fitting_map_at_r - 0.002:
                stopped_reason = "epoch-one-fitting-map-regression"
            elif diagnostic.fitting_effective_rank < 0.7 * initialization_effective_rank:
                stopped_reason = "epoch-one-effective-rank-collapse"
            elif (
                diagnostic.fitting_leading_eigenvalue_share
                > 2.0 * initialization_leading_eigenvalue_share
            ):
                stopped_reason = "epoch-one-leading-eigenvalue-collapse"
            if stopped_reason is not None:
                progress("stopped", epoch, successful_updates)
                break

    final_encoder_sha256 = module_state_sha256(encoder)
    final_frozen_sha256 = encoder_frozen_state_sha256(encoder, head_only=head_only)
    if head_only and final_encoder_sha256 != initial_encoder_sha256:
        raise ValueError("teacher-anchored head-only encoder differs")
    if final_frozen_sha256 != initial_frozen_sha256:
        raise ValueError("teacher-anchored frozen state differs")
    if stopped_reason is None:
        progress("complete", 10, successful_updates)
    return TeacherAnchoredTrainingReceipt(
        arm=arm,
        schedule_sha256=schedule_sha256,
        initial_encoder_sha256=initial_encoder_sha256,
        final_encoder_sha256=final_encoder_sha256,
        initial_frozen_sha256=initial_frozen_sha256,
        final_frozen_sha256=final_frozen_sha256,
        initial_head_sha256=initial_head_sha256,
        final_head_sha256=module_state_sha256(head),
        optimizer_reset_epochs=tuple(optimizer_reset_epochs),
        attempted_updates=attempted_updates,
        successful_updates=successful_updates,
        completed_epochs=tuple(completed_epochs),
        diagnostics=tuple(diagnostics),
        stopped_reason=stopped_reason,
        candidate_epoch=10
        if stopped_reason is None and completed_epochs == list(range(1, 11))
        else None,
    )


def _validate_teacher_anchored_schedules(
    schedules: tuple[tuple[tuple[int, ...], ...], ...],
    *,
    fitting_row_count: int,
) -> None:
    if (
        type(schedules) is not tuple
        or len(schedules) != 10
        or type(fitting_row_count) is not int
        or fitting_row_count < 256
        or any(
            type(epoch_batches) is not tuple or len(epoch_batches) != fitting_row_count // 128
            for epoch_batches in schedules
        )
    ):
        raise ValueError("teacher-anchored training schedule differs")
    for epoch_batches in schedules:
        seed_rows: list[int] = []
        for rows in epoch_batches:
            if (
                type(rows) is not tuple
                or len(rows) != 256
                or any(type(row) is not int or not 0 <= row < fitting_row_count for row in rows)
                or len(set(rows)) != len(rows)
            ):
                raise ValueError("teacher-anchored training schedule differs")
            seed_rows.extend(rows[:128])
        expected_seed_count = fitting_row_count // 128 * 128
        if len(seed_rows) != expected_seed_count or len(set(seed_rows)) != expected_seed_count:
            raise ValueError("teacher-anchored training schedule differs")


def teacher_anchored_progress_bytes(
    *,
    launch_receipt_sha256: str,
    sequence: int,
    arm: str,
    epoch: int,
    update: int,
    previous_line_sha256: str,
    stage: str,
) -> bytes:
    """Encode one launch-bound progress event in the authenticated chain."""

    if (
        not _is_sha256(launch_receipt_sha256)
        or not _is_sha256(previous_line_sha256)
        or type(sequence) is not int
        or sequence < 1
        or arm not in ("head-only", "base", "anchor", "symmetric", "complete")
        or type(epoch) is not int
        or not 0 <= epoch <= 10
        or type(update) is not int
        or update < 0
        or stage
        not in ("heartbeat", "initialized", "update", "epoch-complete", "stopped", "complete")
    ):
        raise ValueError("teacher-anchored progress authority differs")
    return _canonical_json_bytes(
        {
            "arm": arm,
            "epoch": epoch,
            "launch_receipt_sha256": launch_receipt_sha256,
            "previous_line_sha256": previous_line_sha256,
            "schema": "sfora-teacher-anchored-progress-v1",
            "sequence": sequence,
            "stage": stage,
            "update": update,
        }
    )


def validate_teacher_anchored_progress_chain(
    lines: tuple[bytes, ...], launch_receipt_sha256: str
) -> TeacherAnchoredProgressState:
    """Validate exact canonical continuation before progress can reset a watchdog."""

    if type(lines) is not tuple or not lines or not _is_sha256(launch_receipt_sha256):
        raise ValueError("teacher-anchored progress chain differs")
    previous = "0" * 64
    last: dict[str, object] | None = None
    last_scientific: dict[str, object] | None = None
    for expected_sequence, line in enumerate(lines, start=1):
        if type(line) is not bytes:
            raise ValueError("teacher-anchored progress chain differs")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("teacher-anchored progress chain differs") from error
        if (
            type(value) is not dict
            or set(value)
            != {
                "arm",
                "epoch",
                "launch_receipt_sha256",
                "previous_line_sha256",
                "schema",
                "sequence",
                "stage",
                "update",
            }
            or line != _canonical_json_bytes(value)
            or value["schema"] != "sfora-teacher-anchored-progress-v1"
            or value["launch_receipt_sha256"] != launch_receipt_sha256
            or value["previous_line_sha256"] != previous
            or type(value["sequence"]) is not int
            or value["sequence"] != expected_sequence
            or type(value["epoch"]) is not int
            or type(value["update"]) is not int
        ):
            raise ValueError("teacher-anchored progress chain differs")
        try:
            reconstructed = teacher_anchored_progress_bytes(
                launch_receipt_sha256=cast(str, value["launch_receipt_sha256"]),
                sequence=value["sequence"],
                arm=cast(str, value["arm"]),
                epoch=value["epoch"],
                update=value["update"],
                previous_line_sha256=cast(str, value["previous_line_sha256"]),
                stage=cast(str, value["stage"]),
            )
        except ValueError as error:
            raise ValueError("teacher-anchored progress chain differs") from error
        if reconstructed != line:
            raise ValueError("teacher-anchored progress chain differs")
        stage = value["stage"]
        epoch = value["epoch"]
        update = value["update"]
        if stage == "heartbeat":
            valid_transition = (
                epoch == 0 and update == 0
                if last is None
                else value["arm"] == last["arm"]
                and epoch == last["epoch"]
                and update == last["update"]
            )
        elif last_scientific is None:
            valid_transition = stage == "initialized" and epoch == 0 and update == 0
        else:
            previous_stage = last_scientific["stage"]
            previous_epoch = cast(int, last_scientific["epoch"])
            previous_update = cast(int, last_scientific["update"])
            same_arm = value["arm"] == last_scientific["arm"]
            if stage == "update":
                valid_transition = (
                    same_arm
                    and update == previous_update + 1
                    and (
                        (previous_stage == "initialized" and epoch == 1)
                        or (previous_stage == "update" and epoch == previous_epoch)
                        or (previous_stage == "epoch-complete" and epoch == previous_epoch + 1)
                    )
                )
            elif stage == "epoch-complete":
                valid_transition = (
                    same_arm
                    and previous_stage == "update"
                    and epoch == previous_epoch
                    and update == previous_update
                )
            elif stage == "stopped":
                valid_transition = same_arm and (
                    (previous_stage == "initialized" and epoch == 1 and update == 0)
                    or (
                        previous_stage in ("update", "epoch-complete")
                        and epoch == previous_epoch
                        and update == previous_update
                    )
                    or (
                        previous_stage == "epoch-complete"
                        and epoch == previous_epoch + 1
                        and update == previous_update
                    )
                )
            elif stage == "complete":
                valid_transition = (
                    same_arm
                    and previous_stage == "epoch-complete"
                    and previous_epoch == 10
                    and epoch == 10
                    and update == previous_update
                )
            else:
                valid_transition = False
        if not valid_transition:
            raise ValueError("teacher-anchored progress chain differs")
        previous = hashlib.sha256(line).hexdigest()
        last = value
        if stage != "heartbeat":
            last_scientific = value
    if last is None or type(last["arm"]) is not str:
        raise ValueError("teacher-anchored progress chain differs")
    return TeacherAnchoredProgressState(
        sequence=cast(int, last["sequence"]),
        arm=last["arm"],
        epoch=cast(int, last["epoch"]),
        update=cast(int, last["update"]),
        line_sha256=previous,
    )


class TeacherAnchoredProgressWriter:
    """Append and fsync one authenticated progress chain without clobbering."""

    def __init__(self, path: Path, launch_receipt_sha256: str, arm: str) -> None:
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or not path.parent.is_dir()
            or not _is_sha256(launch_receipt_sha256)
            or arm not in ("head-only", "base", "anchor", "symmetric", "complete")
        ):
            raise ValueError("teacher-anchored progress writer authority differs")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as error:
            raise ValueError("progress output already exists") from error
        except OSError as error:
            raise ValueError("teacher-anchored progress writer authority differs") from error
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError("teacher-anchored progress writer authority differs")
        self._descriptor: int | None = descriptor
        self._launch_receipt_sha256 = launch_receipt_sha256
        self._arm = arm
        self._lines: list[bytes] = []
        self._epoch = 0
        self._update = 0

    def __call__(self, stage: str, epoch: int, update: int) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            raise ValueError("progress writer is closed")
        previous = hashlib.sha256(self._lines[-1]).hexdigest() if self._lines else "0" * 64
        line = teacher_anchored_progress_bytes(
            launch_receipt_sha256=self._launch_receipt_sha256,
            sequence=len(self._lines) + 1,
            arm=self._arm,
            epoch=epoch,
            update=update,
            previous_line_sha256=previous,
            stage=stage,
        )
        validate_teacher_anchored_progress_chain((*self._lines, line), self._launch_receipt_sha256)
        view = memoryview(line)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("teacher-anchored progress append failed")
            view = view[written:]
        os.fsync(descriptor)
        self._lines.append(line)
        self._epoch = epoch
        self._update = update

    def heartbeat(self) -> None:
        """Persist an authenticated liveness event without advancing science state."""

        self("heartbeat", self._epoch, self._update)

    def close(self) -> None:
        descriptor = self._descriptor
        if descriptor is not None:
            os.close(descriptor)
            self._descriptor = None


def publish_teacher_anchored_artifacts(
    output: Path,
    encoder: nn.Module,
    head: nn.Linear,
    receipt: TeacherAnchoredTrainingReceipt,
    *,
    authority: dict[str, object],
) -> TeacherAnchoredPublishedArtifacts:
    """Publish a complete merged state and canonical claim-ineligible receipt."""

    checkpoint = output.with_suffix(".pt")
    if (
        not isinstance(output, Path)
        or not output.is_absolute()
        or not output.parent.is_dir()
        or checkpoint == output
    ):
        raise ValueError("teacher-anchored artifact path alias differs")
    if (
        output.exists()
        or checkpoint.exists()
        or not isinstance(encoder, nn.Module)
        or type(head) is not nn.Linear
        or type(receipt) is not TeacherAnchoredTrainingReceipt
    ):
        raise ValueError("teacher-anchored artifact already exists")
    authority_keys = {
        "anchor_schedule_sha256",
        "arm",
        "batch_schedule_sha256",
        "fitting_probe_sha256",
        "head_replay_sha256",
        "inputs_sha256",
        "model_mode_sha256",
        "objective_sha256",
        "ridge_sha256",
        "runtime_sha256",
        "snapshot_replay_sha256",
        "seed",
        "source_revision",
        "split_sha256",
        "teacher_pca_sha256",
        "trainable_inventory_sha256",
    }
    digest_keys = authority_keys - {"arm", "inputs_sha256", "seed", "source_revision"}
    inputs = authority.get("inputs_sha256") if type(authority) is dict else None
    if (
        type(authority) is not dict
        or set(authority) != authority_keys
        or authority["arm"] != receipt.arm
        or authority["batch_schedule_sha256"] != receipt.schedule_sha256
        or authority["objective_sha256"] != teacher_anchored_objective_sha256(receipt.arm)
        or any(not _is_sha256(authority[key]) for key in digest_keys)
        or type(authority["seed"]) is not int
        or authority["seed"] not in (17, 1729, 65537)
        or type(authority["source_revision"]) is not str
        or len(authority["source_revision"]) != 40
        or any(character not in "0123456789abcdef" for character in authority["source_revision"])
        or type(inputs) is not dict
        or set(inputs)
        != {
            "ceiling_receipt",
            "image_tree",
            "launch_receipt",
            "schedule",
            "source_checkpoint",
            "source_snapshot",
            "teacher_checkpoint",
            "teacher_snapshot",
        }
        or any(not _is_sha256(value) for value in inputs.values())
    ):
        raise ValueError("teacher-anchored artifact authority differs")
    head_only = receipt.arm == "head-only"
    if (
        module_state_sha256(encoder) != receipt.final_encoder_sha256
        or module_state_sha256(head) != receipt.final_head_sha256
        or not _module_state_is_finite(encoder)
        or not _module_state_is_finite(head)
        or encoder_frozen_state_sha256(encoder, head_only=head_only) != receipt.final_frozen_sha256
    ):
        raise ValueError("teacher-anchored model state differs")
    state = {
        **{f"encoder.{name}": value.detach().cpu() for name, value in encoder.state_dict().items()},
        **{f"head.{name}": value.detach().cpu() for name, value in head.state_dict().items()},
    }
    checkpoint_fd, checkpoint_name = tempfile.mkstemp(
        dir=output.parent, prefix="teacher-anchored-checkpoint-"
    )
    receipt_fd, receipt_name = tempfile.mkstemp(
        dir=output.parent, prefix="teacher-anchored-receipt-"
    )
    checkpoint_temporary = Path(checkpoint_name)
    receipt_temporary = Path(receipt_name)
    checkpoint_linked = False
    receipt_linked = False
    try:
        os.close(checkpoint_fd)
        checkpoint_fd = -1
        torch.save(state, checkpoint_temporary)
        with checkpoint_temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        checkpoint_sha256 = _file_sha256(checkpoint_temporary)
        value: dict[str, object] = {
            "arm": receipt.arm,
            "attempted_updates": receipt.attempted_updates,
            "authority": authority,
            "candidate_epoch": receipt.candidate_epoch,
            "checkpoint_path": checkpoint.name,
            "checkpoint_sha256": checkpoint_sha256,
            "claim_eligible": False,
            "completed_epochs": list(receipt.completed_epochs),
            "diagnostics": [diagnostic._asdict() for diagnostic in receipt.diagnostics],
            "final_encoder_sha256": receipt.final_encoder_sha256,
            "final_frozen_sha256": receipt.final_frozen_sha256,
            "final_head_sha256": receipt.final_head_sha256,
            "initial_encoder_sha256": receipt.initial_encoder_sha256,
            "initial_frozen_sha256": receipt.initial_frozen_sha256,
            "initial_head_sha256": receipt.initial_head_sha256,
            "optimizer_reset_epochs": list(receipt.optimizer_reset_epochs),
            "schedule_sha256": receipt.schedule_sha256,
            "schema": "sfora-teacher-anchored-arm-v1",
            "stopped_reason": receipt.stopped_reason,
            "successful_updates": receipt.successful_updates,
        }
        payload = _canonical_json_bytes(value)
        with os.fdopen(receipt_fd, "wb") as stream:
            receipt_fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(checkpoint_temporary, checkpoint)
        checkpoint_linked = True
        os.link(receipt_temporary, output)
        receipt_linked = True
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for linked, temporary, destination in (
            (receipt_linked, receipt_temporary, output),
            (checkpoint_linked, checkpoint_temporary, checkpoint),
        ):
            if linked and temporary.exists() and destination.exists():
                temporary_stat = temporary.stat()
                destination_stat = destination.stat()
                if (temporary_stat.st_dev, temporary_stat.st_ino) == (
                    destination_stat.st_dev,
                    destination_stat.st_ino,
                ):
                    destination.unlink()
        raise
    finally:
        if checkpoint_fd >= 0:
            os.close(checkpoint_fd)
        if receipt_fd >= 0:
            os.close(receipt_fd)
        for path in (receipt_temporary, checkpoint_temporary):
            if path.exists():
                path.unlink()
    return TeacherAnchoredPublishedArtifacts(
        receipt=output,
        checkpoint=checkpoint,
        receipt_sha256=hashlib.sha256(payload).hexdigest(),
        checkpoint_sha256=checkpoint_sha256,
    )


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("teacher-anchored canonical JSON authority differs") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parameter_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("teacher PCA authority differs")
        digest.update(struct.pack("<I", value.ndim))
        digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
        digest.update(value.detach().numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _normalize_snapshot_rows(value: torch.Tensor) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] < 2
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("teacher-anchored normalization authority differs")
    value64 = value.double()
    norms = torch.linalg.vector_norm(value64, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
        raise ValueError("teacher-anchored normalization authority differs")
    return cast(torch.Tensor, (value64 / norms).float().contiguous())


def _tensor_sha256(value: torch.Tensor) -> str:
    if (
        type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("tensor digest authority differs")
    digest = hashlib.sha256()
    digest.update(struct.pack("<I", value.ndim))
    digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
    digest.update(value.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash the complete named parameter and buffer inventory deterministically."""

    if not isinstance(module, nn.Module):
        raise ValueError("module state authority differs")
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<I", value.ndim))
        digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _module_state_is_finite(module: nn.Module) -> bool:
    if not isinstance(module, nn.Module):
        return False
    return all(bool(torch.isfinite(value).all()) for value in module.state_dict().values())


def encoder_frozen_state_sha256(encoder: nn.Module, *, head_only: bool) -> str:
    """Hash every encoder state entry that must remain immutable for an arm."""

    if not isinstance(encoder, nn.Module) or type(head_only) is not bool:
        raise ValueError("teacher-anchored frozen state differs")
    excluded = () if head_only else ("blocks.10.", "blocks.11.", "norm.")
    digest = hashlib.sha256(b"sfora-teacher-anchored-frozen-state-v1\x00")
    retained = 0
    state = encoder.state_dict()
    for name in sorted(state):
        if any(name.startswith(prefix) for prefix in excluded):
            continue
        value = state[name].detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError("teacher-anchored frozen state differs")
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(value.dtype).encode("ascii")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack("<I", value.ndim))
        digest.update(struct.pack(f"<{value.ndim}Q", *value.shape))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        retained += 1
    if retained == 0:
        raise ValueError("teacher-anchored frozen state differs")
    return digest.hexdigest()


def resolve_teacher_anchored_cuda_device() -> torch.device:
    """Resolve the indexed CUDA device used by model parameters and diagnostics."""

    if not torch.cuda.is_available():
        raise ValueError("teacher-anchored CUDA authority differs")
    index = torch.cuda.current_device()
    if type(index) is not int or index < 0:
        raise ValueError("teacher-anchored CUDA authority differs")
    return torch.device("cuda", index)


def execute_teacher_anchored_arm(
    arguments: TeacherAnchoredArguments,
) -> TeacherAnchoredPublishedArtifacts:
    """Authenticate, execute, and publish one registered teacher-anchored arm."""

    if type(arguments) is not TeacherAnchoredArguments:
        raise ValueError("teacher-anchored execution authority differs")
    launch_receipt_sha256 = authenticate_teacher_anchored_launch_receipt(arguments)
    progress = TeacherAnchoredProgressWriter(
        arguments.progress, launch_receipt_sha256, arguments.arm
    )
    try:
        progress.heartbeat()
        return _execute_teacher_anchored_arm(arguments, progress, launch_receipt_sha256)
    finally:
        progress.close()


def _execute_teacher_anchored_arm(
    arguments: TeacherAnchoredArguments,
    progress: TeacherAnchoredProgressWriter,
    launch_receipt_sha256: str,
) -> TeacherAnchoredPublishedArtifacts:
    """Run one arm after its progress chain has been opened."""

    authenticate_teacher_anchored_files(arguments)
    authenticate_teacher_anchored_checkout(arguments.unicom_checkout, arguments.source_revision)
    runtime = configure_teacher_anchored_runtime(arguments.seed)
    pair = load_authenticated_train_pair(
        arguments.source_snapshot,
        arguments.source_snapshot_sha256,
        arguments.teacher_snapshot,
        arguments.teacher_snapshot_sha256,
    )
    manifest = bind_authenticated_train_images(
        pair,
        arguments.image_root,
        arguments.image_tree_sha256,
        heartbeat=progress.heartbeat,
    )
    prepared = prepare_teacher_anchored_arrays(
        pair,
        seed=arguments.seed,
        expected_teacher_pca_sha256=arguments.teacher_pca_sha256,
    )
    registered_pca = load_teacher_anchored_ceiling_pca_sha256(
        arguments.ceiling_receipt,
        expected_sha256=arguments.ceiling_receipt_sha256,
        seed=arguments.seed,
    )
    if (
        registered_pca != arguments.teacher_pca_sha256
        or prepared.initialization.teacher_pca_sha256 != registered_pca
    ):
        raise ValueError("teacher-anchored ceiling projection differs")

    bound = bind_teacher_anchored_schedule(arguments, prepared, manifest)
    pristine_model = load_authenticated_source_model(
        arguments.unicom_checkout, arguments.source_checkpoint
    )
    training_model = load_authenticated_source_model(
        arguments.unicom_checkout, arguments.source_checkpoint
    )
    if (
        type(pristine_model) is not TeacherAnchoredSourceModel
        or type(training_model) is not TeacherAnchoredSourceModel
        or pristine_model.revision != arguments.source_revision
        or training_model.revision != arguments.source_revision
        or pristine_model.checkpoint_sha256 != arguments.source_checkpoint_sha256
        or training_model.checkpoint_sha256 != arguments.source_checkpoint_sha256
    ):
        raise ValueError("teacher-anchored source model authority differs")
    device = resolve_teacher_anchored_cuda_device()

    validation_rows = torch.tensor(prepared.split.validation_rows, dtype=torch.int64)
    validation_source = torch.from_numpy(
        np.ascontiguousarray(pair.source_embeddings[validation_rows.numpy()], dtype=np.float32)
    )
    head_replay = validate_teacher_anchored_head_replay(
        prepared.initialization,
        validation_source,
        device="cuda",
        heartbeat=progress.heartbeat,
    )

    replay_rows = tuple(
        sorted(
            prepared.split.fitting_rows,
            key=lambda row: (
                hashlib.sha256(
                    struct.pack("<Qq", arguments.seed, int(pair.image_ids[row]))
                ).digest(),
                int(pair.image_ids[row]),
            ),
        )[:32]
    )
    replay_paths = tuple(manifest.image_paths[row] for row in replay_rows)
    replay_images = torch.stack(
        tuple(
            load_teacher_anchored_training_image(path, pristine_model.transform)
            for path in replay_paths
        )
    ).contiguous()
    replay_expected = torch.from_numpy(
        np.ascontiguousarray(pair.source_embeddings[np.asarray(replay_rows)], dtype=np.float32)
    )
    snapshot_replay = validate_teacher_anchored_snapshot_feature_replay(
        pristine_model.encoder,
        training_model.encoder,
        replay_images,
        replay_expected,
        image_ids=tuple(int(pair.image_ids[row]) for row in replay_rows),
        device="cuda",
    )
    pristine_model.encoder.to("cpu")
    del pristine_model, replay_images, replay_expected
    torch.cuda.empty_cache()

    encoder = training_model.encoder.to(device)
    head = prepared.initialization.head.to(device)
    configure_teacher_anchored_epoch(encoder, head, epoch=2, head_only=arguments.arm == "head-only")
    planned_trainable_inventory_sha256 = teacher_anchored_trainable_inventory_sha256(encoder, head)

    def transform_image(path: Path) -> torch.Tensor:
        return load_teacher_anchored_training_image(path, training_model.transform)

    probe = select_teacher_anchored_fitting_probe(
        np.ascontiguousarray(prepared.split.fitting_labels, dtype=np.int64),
        seed=arguments.seed,
    )
    fitting_probe_paths = tuple(bound.fitting_image_paths[row] for row in probe.row_indexes)
    fitting_probe_labels = tuple(prepared.split.fitting_labels[row] for row in probe.row_indexes)
    validation_paths = tuple(manifest.image_paths[row] for row in prepared.split.validation_rows)
    diagnose = make_teacher_anchored_diagnose(
        encoder,
        head,
        fitting_image_paths=fitting_probe_paths,
        fitting_labels=fitting_probe_labels,
        validation_image_paths=validation_paths,
        validation_labels=prepared.split.validation_labels,
        transform_image=transform_image,
        device=device,
        heartbeat=progress.heartbeat,
    )
    initialization_diagnostic = compute_teacher_anchored_snapshot_diagnostic(
        head,
        fitting_features=prepared.source_features,
        fitting_labels=prepared.split.fitting_labels,
        fitting_probe_rows=probe.row_indexes,
        validation_features=validation_source,
        validation_labels=prepared.split.validation_labels,
        device=device,
    )
    config = TeacherAnchoredConfig()

    def compute_loss(
        _epoch: int,
        _update: int,
        rows: tuple[int, ...],
        active: tuple[str, ...],
    ) -> TeacherAnchoredLoss:
        if arguments.arm == "head-only":
            return compute_teacher_anchored_snapshot_batch_loss(
                head,
                row_indexes=rows,
                active_parameter_names=active,
                source_features=prepared.source_features,
                teacher_codes=prepared.teacher_codes,
                anchor_row_indexes=bound.anchor_row_indexes,
                config=config,
                device=device,
            )
        return compute_teacher_anchored_batch_loss(
            encoder,
            head,
            row_indexes=rows,
            active_parameter_names=active,
            image_paths=bound.fitting_image_paths,
            transform_image=transform_image,
            source_features=prepared.source_features,
            teacher_codes=prepared.teacher_codes,
            anchor_row_indexes=bound.anchor_row_indexes,
            config=config,
            device=device,
        )

    receipt = run_teacher_anchored_training(
        encoder,
        head,
        arm=arguments.arm,
        schedules=bound.schedules,
        fitting_row_count=len(bound.fitting_image_paths),
        expected_schedule_sha256=teacher_anchored_schedule_sha256(
            bound.schedules, fitting_row_count=len(bound.fitting_image_paths)
        ),
        initialization_fitting_map_at_r=initialization_diagnostic.fitting_map_at_r,
        initialization_effective_rank=initialization_diagnostic.fitting_effective_rank,
        initialization_leading_eigenvalue_share=(
            initialization_diagnostic.fitting_leading_eigenvalue_share
        ),
        compute_loss=compute_loss,
        diagnose=diagnose,
        progress=progress,
        device_type="cuda",
    )

    authority: dict[str, object] = {
        "anchor_schedule_sha256": bound.sealed.anchor_schedule.sha256,
        "arm": arguments.arm,
        "batch_schedule_sha256": receipt.schedule_sha256,
        "fitting_probe_sha256": probe.sha256,
        "head_replay_sha256": hashlib.sha256(
            _canonical_json_bytes(head_replay._asdict())
        ).hexdigest(),
        "inputs_sha256": {
            "ceiling_receipt": arguments.ceiling_receipt_sha256,
            "image_tree": arguments.image_tree_sha256,
            "launch_receipt": launch_receipt_sha256,
            "schedule": arguments.schedule_sha256,
            "source_checkpoint": arguments.source_checkpoint_sha256,
            "source_snapshot": arguments.source_snapshot_sha256,
            "teacher_checkpoint": arguments.teacher_checkpoint_sha256,
            "teacher_snapshot": arguments.teacher_snapshot_sha256,
        },
        "model_mode_sha256": teacher_anchored_model_mode_sha256(arguments.arm),
        "objective_sha256": teacher_anchored_objective_sha256(arguments.arm),
        "ridge_sha256": teacher_anchored_ridge_sha256(prepared.initialization),
        "runtime_sha256": teacher_anchored_runtime_sha256(runtime),
        "seed": arguments.seed,
        "snapshot_replay_sha256": hashlib.sha256(
            _canonical_json_bytes(snapshot_replay._asdict())
        ).hexdigest(),
        "source_revision": arguments.source_revision,
        "split_sha256": prepared.split.sha256,
        "teacher_pca_sha256": registered_pca,
        "trainable_inventory_sha256": planned_trainable_inventory_sha256,
    }
    return publish_teacher_anchored_artifacts(
        arguments.output, encoder, head, receipt, authority=authority
    )


def main(arguments: list[str] | None = None) -> int:
    """Execute one authenticated local arm and emit its canonical receipt."""

    parsed = parse_teacher_anchored_args(sys.argv[1:] if arguments is None else arguments)
    published = execute_teacher_anchored_arm(parsed)
    payload = published.receipt.read_bytes()
    if not payload or payload[-1:] != b"\n":
        raise ValueError("teacher-anchored published receipt differs")
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    main()
