#!/usr/bin/env python3
"""Run the authenticated SigLIP-so400m pooled Proxy Anchor control."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig
from sfora.token_set_screen import F1_TRAIN_CLASSES


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON refuses nonfinite floats")
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError("canonical JSON object keys must be concrete strings")
            _validate_json_value(item)
        return
    raise TypeError(f"canonical JSON refuses {type(value).__name__}")


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Return strict sorted compact JSON with exactly one trailing newline."""

    if type(payload) is not dict:
        raise TypeError("canonical payload must be a concrete object")
    _validate_json_value(payload)
    if "claim_eligible" in payload and payload["claim_eligible"] is not False:
        raise ValueError("control receipts must be concretely claim-ineligible")
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _write_new(path: Path, payload: bytes) -> None:
    """Publish bytes without overwriting an existing authority path."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, path, follow_symlinks=False)
    finally:
        partial.unlink(missing_ok=True)


def _learning_rate_multiplier(
    config: SiglipProxyControlConfig,
    *,
    step: int,
    steps_per_epoch: int,
) -> float:
    """Resolve the exact warmup-inclusive, epoch-bound schedule multiplier."""

    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative concrete integer")
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be a positive concrete integer")
    warmup_steps = config.warmup_epochs * steps_per_epoch
    warmup = min(1.0, (step + 1) / warmup_steps)
    completed_epoch = step // steps_per_epoch
    decays = sum(completed_epoch >= epoch for epoch in config.decay_epochs)
    return warmup * config.decay_gamma**decays


def _seed64(*parts: object) -> int:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")


@dataclass(frozen=True)
class SamplerState:
    """Persistent per-class permutation cycles and cursors."""

    cycles: tuple[int, ...]
    positions: tuple[int, ...]

    @classmethod
    def initial(cls) -> SamplerState:
        return cls(cycles=(0,) * 49, positions=(0,) * 49)


@dataclass(frozen=True)
class SmokeObservation:
    """One isolated three-step smoke-rung observation."""

    microbatch_size: int
    steps_completed: int
    peak_process_rss_bytes: int
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    memory_psi_growth: float
    swap_growth_bytes: int
    examples_per_second: float
    final_loss: float
    complete_tower_gradient_coverage: bool
    maximum_score_disagreement: float


@dataclass(frozen=True)
class SmokeReceipt:
    """Ordered rung evidence and the first passing microbatch."""

    observations: tuple[SmokeObservation, ...]
    selected_microbatch_size: int
    projected_seed_seconds: float


@dataclass(frozen=True)
class CheckpointAuthority:
    """Authenticated local checkpoint and its canonical receipt."""

    seed: int
    epoch: int
    path: Path
    receipt_path: Path
    sha256: str
    bytes: int


_CHECKPOINT_SCHEMA = "sfora-siglip-proxy-checkpoint-v1"
_CONTROL_SEEDS = (17, 29, 43)


def _checkpoint_basename(*, seed: int, epoch: int) -> str:
    return f"seed-{seed:03d}-epoch-{epoch:03d}.pt"


def _checkpoint_receipt_basename(*, seed: int, epoch: int) -> str:
    return f"seed-{seed:03d}-epoch-{epoch:03d}.checkpoint.json"


def _validate_checkpoint_coordinates(*, seed: int, epoch: int) -> None:
    if type(seed) is not int or seed not in _CONTROL_SEEDS:
        raise ValueError("checkpoint seed differs from the registered seeds")
    if type(epoch) is not int or not 1 <= epoch <= 60:
        raise ValueError("checkpoint epoch must be in [1, 60]")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checkpoint_authority_from_receipt(
    receipt_path: Path,
    *,
    directory: Path,
    expected_seed: int,
) -> CheckpointAuthority:
    if receipt_path.is_symlink() or not stat.S_ISREG(receipt_path.lstat().st_mode):
        raise ValueError("checkpoint receipt must be a regular file")
    raw = receipt_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("checkpoint receipt is not valid JSON") from error
    if type(payload) is not dict or raw != _canonical_bytes(cast(dict[str, Any], payload)):
        raise ValueError("checkpoint receipt is not canonical")
    expected_keys = {
        "bytes",
        "checkpoint",
        "claim_eligible",
        "epoch",
        "schema",
        "seed",
        "sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError("checkpoint receipt schema differs")
    seed = payload["seed"]
    epoch = payload["epoch"]
    size = payload["bytes"]
    checksum = payload["sha256"]
    checkpoint = payload["checkpoint"]
    if (
        type(seed) is not int
        or type(epoch) is not int
        or type(size) is not int
        or type(checksum) is not str
        or type(checkpoint) is not str
        or payload["schema"] != _CHECKPOINT_SCHEMA
        or payload["claim_eligible"] is not False
    ):
        raise ValueError("checkpoint receipt types differ")
    _validate_checkpoint_coordinates(seed=seed, epoch=epoch)
    if seed != expected_seed or size < 1:
        raise ValueError("checkpoint receipt authority differs")
    expected_checkpoint = _checkpoint_basename(seed=seed, epoch=epoch)
    expected_receipt = _checkpoint_receipt_basename(seed=seed, epoch=epoch)
    if checkpoint != expected_checkpoint or receipt_path.name != expected_receipt:
        raise ValueError("checkpoint path binding differs")
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise ValueError("checkpoint digest encoding differs")
    path = directory / checkpoint
    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("checkpoint must be a regular file")
    observed_checksum, observed_size = _sha256_file(path)
    if observed_size != size or observed_checksum != checksum:
        raise ValueError("checkpoint digest or length differs")
    return CheckpointAuthority(
        seed=seed,
        epoch=epoch,
        path=path,
        receipt_path=receipt_path,
        sha256=checksum,
        bytes=size,
    )


def latest_authenticated_checkpoint(
    directory: Path,
    *,
    seed: int,
) -> CheckpointAuthority | None:
    """Return the newest fully authenticated rolling checkpoint for one seed."""

    _validate_checkpoint_coordinates(seed=seed, epoch=1)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint directory must be a real directory")
    authorities = [
        _checkpoint_authority_from_receipt(
            receipt,
            directory=directory,
            expected_seed=seed,
        )
        for receipt in sorted(directory.glob(f"seed-{seed:03d}-epoch-*.checkpoint.json"))
    ]
    if not authorities:
        return None
    epochs = [authority.epoch for authority in authorities]
    if len(epochs) != len(set(epochs)):
        raise ValueError("checkpoint epochs are duplicated")
    return max(authorities, key=lambda authority: authority.epoch)


def publish_epoch_checkpoint(
    *,
    directory: Path,
    seed: int,
    epoch: int,
    write_checkpoint: Callable[[Path], object],
    maximum_checkpoint_bytes: int,
) -> CheckpointAuthority:
    """Stream, authenticate, and rotate one rolling epoch checkpoint."""

    _validate_checkpoint_coordinates(seed=seed, epoch=epoch)
    if type(maximum_checkpoint_bytes) is not int or maximum_checkpoint_bytes < 1:
        raise ValueError("maximum checkpoint bytes must be positive")
    if not callable(write_checkpoint):
        raise TypeError("checkpoint writer must be callable")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint directory must be a real directory")
    previous = latest_authenticated_checkpoint(directory, seed=seed)
    if previous is not None and epoch <= previous.epoch:
        raise ValueError("checkpoint epoch must advance")
    retained_bytes = 0 if previous is None else previous.bytes
    required_free = retained_bytes + math.ceil(maximum_checkpoint_bytes * 1.2)
    if shutil.disk_usage(directory).free < required_free:
        raise OSError("insufficient free space for checkpoint publication")

    basename = _checkpoint_basename(seed=seed, epoch=epoch)
    receipt_basename = _checkpoint_receipt_basename(seed=seed, epoch=epoch)
    path = directory / basename
    receipt_path = directory / receipt_basename
    partial = directory / f"{basename}.partial"
    publication_paths = (path, receipt_path, partial)
    if any(candidate.exists() or candidate.is_symlink() for candidate in publication_paths):
        raise FileExistsError(path)
    published_checkpoint = False
    try:
        write_checkpoint(partial)
        if (
            partial.is_symlink()
            or not partial.exists()
            or not stat.S_ISREG(partial.lstat().st_mode)
        ):
            raise ValueError("checkpoint writer must create one regular file")
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        checksum, size = _sha256_file(partial)
        if size < 1 or size > maximum_checkpoint_bytes:
            raise ValueError("checkpoint length differs from the registered envelope")
        os.link(partial, path, follow_symlinks=False)
        published_checkpoint = True
        _fsync_directory(directory)
        receipt_payload = _canonical_bytes(
            {
                "bytes": size,
                "checkpoint": basename,
                "claim_eligible": False,
                "epoch": epoch,
                "schema": _CHECKPOINT_SCHEMA,
                "seed": seed,
                "sha256": checksum,
            }
        )
        _write_new(receipt_path, receipt_payload)
        _fsync_directory(directory)
        authority = _checkpoint_authority_from_receipt(
            receipt_path,
            directory=directory,
            expected_seed=seed,
        )
    except BaseException:
        receipt_path.unlink(missing_ok=True)
        if published_checkpoint:
            path.unlink(missing_ok=True)
        raise
    finally:
        partial.unlink(missing_ok=True)

    if previous is not None:
        previous.receipt_path.unlink()
        previous.path.unlink()
        _fsync_directory(directory)
    return authority


def _smoke_projection_seconds(
    observation: SmokeObservation,
    *,
    config: SiglipProxyControlConfig,
    steps_per_epoch: int,
) -> float:
    if observation.examples_per_second <= 0.0 or not math.isfinite(observation.examples_per_second):
        return math.inf
    examples = config.train_epochs * steps_per_epoch * config.logical_batch_size
    return examples / observation.examples_per_second


def _smoke_observation_passes(
    observation: SmokeObservation,
    *,
    config: SiglipProxyControlConfig,
    steps_per_epoch: int,
) -> bool:
    integer_values = (
        observation.microbatch_size,
        observation.steps_completed,
        observation.peak_process_rss_bytes,
        observation.peak_cuda_allocated_bytes,
        observation.peak_cuda_reserved_bytes,
        observation.swap_growth_bytes,
    )
    if any(type(value) is not int or value < 0 for value in integer_values):
        return False
    float_values = (
        observation.memory_psi_growth,
        observation.examples_per_second,
        observation.final_loss,
        observation.maximum_score_disagreement,
    )
    if any(type(value) is not float or not math.isfinite(value) for value in float_values):
        return False
    combined_memory = observation.peak_process_rss_bytes + observation.peak_cuda_reserved_bytes
    projected_seconds = _smoke_projection_seconds(
        observation,
        config=config,
        steps_per_epoch=steps_per_epoch,
    )
    return (
        observation.steps_completed == 3
        and combined_memory < config.combined_memory_limit_bytes
        and observation.memory_psi_growth <= 0.0
        and observation.swap_growth_bytes <= 0
        and observation.examples_per_second > 0.0
        and projected_seconds <= config.maximum_projected_seed_hours * 3600.0
        and observation.complete_tower_gradient_coverage is True
        and observation.maximum_score_disagreement <= config.replay_score_tolerance
    )


def run_memory_smoke(
    *,
    config: SiglipProxyControlConfig,
    steps_per_epoch: int,
    run_rung: Callable[[int], SmokeObservation],
) -> SmokeReceipt:
    """Select the first isolated rung satisfying every preregistered smoke gate."""

    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be a positive concrete integer")
    observations: list[SmokeObservation] = []
    for microbatch_size in config.smoke_microbatch_ladder:
        observation = run_rung(microbatch_size)
        if observation.microbatch_size != microbatch_size:
            raise ValueError("smoke observation microbatch differs from the requested rung")
        observations.append(observation)
        if _smoke_observation_passes(
            observation,
            config=config,
            steps_per_epoch=steps_per_epoch,
        ):
            return SmokeReceipt(
                observations=tuple(observations),
                selected_microbatch_size=microbatch_size,
                projected_seed_seconds=_smoke_projection_seconds(
                    observation,
                    config=config,
                    steps_per_epoch=steps_per_epoch,
                ),
            )
    raise RuntimeError("no smoke microbatch passed every registered gate")


def _permutation(values: list[int], *, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[int(position)] for position in order]


def _build_epoch_batches(
    *,
    example_ids: tuple[str, ...],
    labels: torch.Tensor,
    seed: int,
    epoch: int,
    steps_per_epoch: int,
    state: SamplerState,
) -> tuple[tuple[tuple[int, ...], ...], SamplerState]:
    """Build exact 30-class/4-image logical batches and advance sampler cursors."""

    if len(example_ids) != labels.numel() or labels.ndim != 1:
        raise ValueError("example IDs and labels differ")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("sampler labels must use an integer dtype")
    if type(seed) is not int or type(epoch) is not int or epoch < 0:
        raise ValueError("sampler seed and epoch must be concrete integers")
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("sampler example IDs must be unique")
    if len(state.cycles) != 49 or len(state.positions) != 49:
        raise ValueError("sampler state must bind all 49 optimization classes")
    grouped: dict[int, list[int]] = {label: [] for label in sorted(F1_TRAIN_CLASSES)}
    for index, label_tensor in enumerate(labels.detach().cpu().to(torch.int64)):
        label = int(label_tensor)
        if label not in grouped:
            raise ValueError("sampler labels differ from the optimization band")
        grouped[label].append(index)
    for label in grouped:
        grouped[label].sort(key=example_ids.__getitem__)
        if len(grouped[label]) < 4:
            raise ValueError("every optimization class requires at least four examples")

    cycles = list(state.cycles)
    positions = list(state.positions)
    batches: list[tuple[int, ...]] = []
    classes = sorted(F1_TRAIN_CLASSES)
    for step in range(steps_per_epoch):
        selected_classes = _permutation(
            classes,
            seed=_seed64("classes", seed, epoch, step),
        )[:30]
        batch: list[int] = []
        for label in selected_classes:
            candidates = grouped[label]
            if positions[label] + 4 > len(candidates):
                cycles[label] += 1
                positions[label] = 0
            ordered = _permutation(
                candidates,
                seed=_seed64("examples", seed, label, cycles[label]),
            )
            start = positions[label]
            batch.extend(ordered[start : start + 4])
            positions[label] += 4
        if len(batch) != 120 or len(set(batch)) != 120:
            raise RuntimeError("sampler failed to construct a unique logical batch")
        batches.append(tuple(batch))
    return tuple(batches), SamplerState(cycles=tuple(cycles), positions=tuple(positions))


def _optimizer_groups(
    model: PooledProxyAnchorModel,
    config: SiglipProxyControlConfig,
) -> list[dict[str, Any]]:
    """Partition every trainable parameter once by family and decay authority."""

    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "proxies":
            learning_rate = config.proxy_learning_rate
            decay = 0.0
        elif name.startswith("projection."):
            learning_rate = config.projection_learning_rate
            decay = 0.0 if name.endswith(".bias") else config.weight_decay
        elif name.startswith("tower."):
            learning_rate = config.tower_learning_rate
            decay = (
                0.0 if name.endswith(".bias") or ".norm." in name.lower() else config.weight_decay
            )
        else:
            raise ValueError(f"unclassified trainable parameter: {name}")
        grouped.setdefault((learning_rate, decay), []).append(parameter)
    groups: list[dict[str, Any]] = [
        {"params": parameters, "lr": learning_rate, "weight_decay": decay}
        for (learning_rate, decay), parameters in sorted(grouped.items())
    ]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    duplicated = len(parameter_ids) != len(set(parameter_ids))
    incomplete = sorted(parameter_ids) != sorted(expected_ids)
    if duplicated or incomplete:
        raise RuntimeError("optimizer groups do not partition trainable parameters exactly once")
    return groups
