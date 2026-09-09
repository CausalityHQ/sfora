#!/usr/bin/env python3
"""Strict local trainer boundary for teacher-anchored SOP experiments."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
import stat
import struct
from pathlib import Path
from typing import NamedTuple, cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler

from sfora.representation_ceiling import (
    TeacherGuidedProjection,
    fit_teacher_guided_projection,
)


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
    source_snapshot: Path
    source_snapshot_sha256: str
    teacher_snapshot: Path
    teacher_snapshot_sha256: str
    image_root: Path
    image_tree_sha256: str
    source_revision: str
    seed: int
    arm: str
    output: Path
    execute_teacher_anchored: bool


class TeacherAnchoredRuntimeReceipt(NamedTuple):
    """Exact deterministic arithmetic state bound into experiment receipts."""

    seed: int
    cublas_workspace_config: str
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_matmul_tf32: bool
    cudnn_tf32: bool
    float32_matmul_precision: str
    math_sdp_enabled: bool
    flash_sdp_enabled: bool
    memory_efficient_sdp_enabled: bool
    cudnn_sdp_enabled: bool


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
        "image-root",
        "output",
    ):
        parser.add_argument(f"--{name}")
    for name in (
        "source-checkpoint-sha256",
        "teacher-checkpoint-sha256",
        "source-snapshot-sha256",
        "teacher-snapshot-sha256",
        "image-tree-sha256",
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
        "image_root",
    ):
        raw = getattr(parsed, name)
        path = Path(raw) if type(raw) is str else Path()
        if type(raw) is not str or not path.is_absolute() or not path.exists():
            raise ValueError("absolute local input authority differs")
        if name == "image_root" and not path.is_dir():
            raise ValueError("absolute local input authority differs")
        if name != "image_root" and not path.is_file():
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

    digests: dict[str, str] = {}
    for name in (
        "source_checkpoint_sha256",
        "teacher_checkpoint_sha256",
        "source_snapshot_sha256",
        "teacher_snapshot_sha256",
        "image_tree_sha256",
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
    if (
        type(revision) is not str
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
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
        source_snapshot=path_values["source_snapshot"],
        source_snapshot_sha256=digests["source_snapshot_sha256"],
        teacher_snapshot=path_values["teacher_snapshot"],
        teacher_snapshot_sha256=digests["teacher_snapshot_sha256"],
        image_root=path_values["image_root"],
        image_tree_sha256=digests["image_tree_sha256"],
        source_revision=revision,
        seed=seed,
        arm=parsed.arm,
        output=output,
        execute_teacher_anchored=True,
    )


def configure_teacher_anchored_runtime(seed: int) -> TeacherAnchoredRuntimeReceipt:
    """Pin RNG and Torch arithmetic state before any experiment computation."""

    if type(seed) is not int or seed not in (17, 1729, 65537):
        raise ValueError("teacher-anchored runtime authority differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("teacher-anchored runtime authority differs")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    return TeacherAnchoredRuntimeReceipt(
        seed=seed,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        math_sdp_enabled=cast(bool, torch.backends.cuda.math_sdp_enabled()),  # type: ignore[no-untyped-call]
        flash_sdp_enabled=cast(bool, torch.backends.cuda.flash_sdp_enabled()),  # type: ignore[no-untyped-call]
        memory_efficient_sdp_enabled=cast(
            bool,
            torch.backends.cuda.mem_efficient_sdp_enabled(),  # type: ignore[no-untyped-call]
        ),
        cudnn_sdp_enabled=cast(
            bool,
            torch.backends.cuda.cudnn_sdp_enabled(),  # type: ignore[no-untyped-call]
        ),
    )


def authenticate_teacher_anchored_files(arguments: TeacherAnchoredArguments) -> tuple[str, ...]:
    """Authenticate all four immutable local file capabilities before loading them."""

    if type(arguments) is not TeacherAnchoredArguments:
        raise ValueError("teacher-anchored input authority differs")
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


def validate_teacher_anchored_head_replay(
    initialization: TeacherAnchoredInitialization,
    source_rows: torch.Tensor,
    *,
    device: str,
    expected_receipt: TeacherAnchoredHeadReplayReceipt | None = None,
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
        or not bool(torch.isfinite(loss))
        or type(optimizer) is not torch.optim.AdamW
        or type(scaler) is not GradScaler
        or not parameters
        or any(type(parameter) is not nn.Parameter for parameter in parameters)
    ):
        raise ValueError("teacher-anchored nonfinite update")
    optimizer.zero_grad(set_to_none=True)
    scale_before = scaler.get_scale()
    scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
    scaler.unscale_(optimizer)
    try:
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=1.0,
            error_if_nonfinite=True,
        )
    except RuntimeError as error:
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("teacher-anchored nonfinite update") from error
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < scale_before or any(
        not bool(torch.isfinite(parameter).all()) for parameter in parameters
    ):
        raise ValueError("teacher-anchored nonfinite update")
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
        if not bool(torch.isfinite(value).all()):
            raise ValueError("module state authority differs")
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


def main() -> int:
    """Refuse execution until the strict CLI and complete loop are implemented."""

    raise SystemExit("teacher-anchored training CLI is not implemented")


if __name__ == "__main__":
    main()
