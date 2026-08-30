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
from torch.nn import functional as F

from sfora.data import ImageExample, load_image_retrieval_examples, materialize_image
from sfora.siglip_proxy_control import (
    PooledProxyAnchorModel,
    SiglipProxyControlConfig,
    recomputed_proxy_anchor_backward,
    validate_control_partition,
)
from sfora.substrate_screen import SUBSTRATE_F0_CLASSES
from sfora.token_set_screen import F1_TRAIN_CLASSES, F1_VALIDATION_CLASSES


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


@dataclass(frozen=True)
class ControlExampleBands:
    """The three Cars-train evidence bands and their ordered manifest."""

    optimization: tuple[ImageExample, ...]
    clean_validation: tuple[ImageExample, ...]
    burned_diagnostic: tuple[ImageExample, ...]
    ordered_manifest: tuple[ImageExample, ...]


@dataclass(frozen=True)
class EpochTrainingEvidence:
    """One epoch's exact optimizer and replay evidence."""

    epoch: int
    optimizer_steps: int
    losses: tuple[float, ...]
    maximum_score_disagreement: float
    sampler_state: SamplerState


@dataclass(frozen=True)
class RestoredControlCheckpoint:
    """Resume coordinates recovered from an authenticated checkpoint payload."""

    seed: int
    completed_epoch: int
    sampler_state: SamplerState


class SiglipPooledTower(nn.Module):
    """Expose only the pinned SigLIP vision pooler output."""

    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=pixel_values.device.type == "cuda",
        ):
            output = self.vision_model(pixel_values=pixel_values, return_dict=True)
        pooled = getattr(output, "pooler_output", None)
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise ValueError("SigLIP vision output lacks a rank-two pooler output")
        return pooled.float()


_CHECKPOINT_SCHEMA = "sfora-siglip-proxy-checkpoint-v1"
_CHECKPOINT_PAYLOAD_SCHEMA = "sfora-siglip-proxy-checkpoint-payload-v1"
_CONTROL_SEEDS = (17, 29, 43)


def _json_compatible(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    return value


def _config_sha256(config: SiglipProxyControlConfig) -> str:
    payload = _json_compatible(vars(config))
    if type(payload) is not dict:
        raise TypeError("control config did not produce an object authority")
    return hashlib.sha256(_canonical_bytes(cast(dict[str, Any], payload))).hexdigest()


def write_control_checkpoint(
    path: Path,
    *,
    model: PooledProxyAnchorModel,
    optimizer: torch.optim.Optimizer,
    config: SiglipProxyControlConfig,
    seed: int,
    completed_epoch: int,
    sampler_state: SamplerState,
) -> None:
    """Write one create-new checkpoint payload for authenticated publication."""

    _validate_checkpoint_coordinates(seed=seed, epoch=completed_epoch)
    if len(sampler_state.cycles) != 49 or len(sampler_state.positions) != 49:
        raise ValueError("checkpoint sampler state must bind all optimization classes")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "claim_eligible": False,
        "completed_epoch": completed_epoch,
        "config_sha256": _config_sha256(config),
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_states": tuple(torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sampler_cycles": sampler_state.cycles,
        "sampler_positions": sampler_state.positions,
        "schema": _CHECKPOINT_PAYLOAD_SCHEMA,
        "seed": seed,
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())


def restore_control_checkpoint(
    path: Path,
    *,
    model: PooledProxyAnchorModel,
    optimizer: torch.optim.Optimizer,
    config: SiglipProxyControlConfig,
    expected_seed: int,
) -> RestoredControlCheckpoint:
    """Restore only a strict same-config, same-environment checkpoint payload."""

    if path.is_symlink() or not path.exists() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("control checkpoint must be a regular file")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = {
        "claim_eligible",
        "completed_epoch",
        "config_sha256",
        "cpu_rng_state",
        "cuda_rng_states",
        "model_state",
        "optimizer_state",
        "sampler_cycles",
        "sampler_positions",
        "schema",
        "seed",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("control checkpoint payload schema differs")
    seed = payload["seed"]
    completed_epoch = payload["completed_epoch"]
    cycles = payload["sampler_cycles"]
    positions = payload["sampler_positions"]
    if (
        type(seed) is not int
        or type(completed_epoch) is not int
        or type(cycles) is not tuple
        or type(positions) is not tuple
        or payload["claim_eligible"] is not False
        or payload["schema"] != _CHECKPOINT_PAYLOAD_SCHEMA
        or payload["config_sha256"] != _config_sha256(config)
    ):
        raise ValueError("control checkpoint authority differs")
    _validate_checkpoint_coordinates(seed=seed, epoch=completed_epoch)
    if seed != expected_seed:
        raise ValueError("control checkpoint seed differs")
    if (
        len(cycles) != 49
        or len(positions) != 49
        or any(type(value) is not int or value < 0 for value in cycles + positions)
    ):
        raise ValueError("control checkpoint sampler state differs")
    cpu_rng_state = payload["cpu_rng_state"]
    cuda_rng_states = payload["cuda_rng_states"]
    if (
        not isinstance(cpu_rng_state, torch.Tensor)
        or cpu_rng_state.dtype != torch.uint8
        or type(cuda_rng_states) is not tuple
        or any(not isinstance(state, torch.Tensor) for state in cuda_rng_states)
    ):
        raise ValueError("control checkpoint RNG state differs")
    if torch.cuda.is_available() != bool(cuda_rng_states):
        raise ValueError("control checkpoint CUDA environment differs")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_states:
        torch.cuda.set_rng_state_all(list(cuda_rng_states))
    sampler_state = SamplerState(cycles=cycles, positions=positions)
    return RestoredControlCheckpoint(
        seed=seed,
        completed_epoch=completed_epoch,
        sampler_state=sampler_state,
    )


def load_control_examples(
    *,
    loader: Callable[..., list[ImageExample]] = load_image_retrieval_examples,
) -> ControlExampleBands:
    """Load Cars train once and partition it without exposing official test classes."""

    examples = loader(dataset_name="cars", split="train")
    if type(examples) is not list or not examples:
        raise ValueError("Cars train loader must return a nonempty concrete list")
    if any(type(example) is not ImageExample for example in examples):
        raise TypeError("Cars train loader returned a non-ImageExample value")
    labels = [example.label for example in examples]
    if any(type(label) is not int for label in labels):
        raise TypeError("Cars labels must be concrete integers")
    if any(label >= 98 or label < 0 for label in labels):
        raise ValueError("official test classes must never enter the control boundary")
    example_ids = [example.example_id for example in examples]
    if any(type(example_id) is not str or not example_id for example_id in example_ids):
        raise ValueError("Cars example IDs must be nonempty concrete strings")
    if len(example_ids) != len(set(example_ids)):
        raise ValueError("Cars example IDs must be unique")
    ordered = tuple(sorted(examples, key=lambda example: example.example_id))
    optimization = tuple(example for example in ordered if example.label in F1_TRAIN_CLASSES)
    clean = tuple(example for example in ordered if example.label in F1_VALIDATION_CLASSES)
    burned = tuple(example for example in ordered if example.label in SUBSTRATE_F0_CLASSES)
    validate_control_partition(
        optimization_labels=torch.tensor(
            [example.label for example in optimization], dtype=torch.int64
        ),
        clean_validation_labels=torch.tensor(
            [example.label for example in clean], dtype=torch.int64
        ),
        burned_diagnostic_labels=torch.tensor(
            [example.label for example in burned], dtype=torch.int64
        ),
    )
    return ControlExampleBands(
        optimization=optimization,
        clean_validation=clean,
        burned_diagnostic=burned,
        ordered_manifest=ordered,
    )


def load_siglip_control_components(
    *,
    config: SiglipProxyControlConfig,
    vision_model_cls: Any | None = None,
    processor_cls: Any | None = None,
) -> tuple[SiglipPooledTower, Any]:
    """Load the pinned local-only eager SigLIP vision tower and processor."""

    if vision_model_cls is None or processor_cls is None:
        from transformers import AutoImageProcessor, SiglipVisionModel

        vision_model_cls = SiglipVisionModel if vision_model_cls is None else vision_model_cls
        processor_cls = AutoImageProcessor if processor_cls is None else processor_cls
    vision_model = vision_model_cls.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        local_files_only=True,
        attn_implementation="eager",
    )
    if not isinstance(vision_model, nn.Module):
        raise TypeError("SigLIP vision loader must return a torch module")
    attention = getattr(getattr(vision_model, "config", None), "_attn_implementation", None)
    if attention != "eager":
        raise ValueError("SigLIP attention implementation differs from eager authority")
    checkpointing = getattr(vision_model, "gradient_checkpointing_enable", None)
    if not callable(checkpointing):
        raise TypeError("SigLIP vision tower lacks gradient checkpointing")
    checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
    processor = processor_cls.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        local_files_only=True,
    )
    return SiglipPooledTower(vision_model), processor


def build_control_train_transform() -> Any:
    """Construct the prospectively frozen Cars optimization augmentation."""

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                384,
                scale=(0.16, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def preprocess_control_evaluation(processor: Any, images: list[object]) -> torch.Tensor:
    """Use only the pinned processor's deterministic evaluation preprocessing."""

    if type(images) is not list or not images:
        raise ValueError("evaluation images must be a nonempty concrete list")
    encoded = processor(images=images, return_tensors="pt")
    if type(encoded) is not dict or "pixel_values" not in encoded:
        raise ValueError("SigLIP processor did not return pixel values")
    pixel_values = encoded["pixel_values"]
    if (
        not isinstance(pixel_values, torch.Tensor)
        or pixel_values.ndim != 4
        or pixel_values.shape != (len(images), 3, 384, 384)
        or not pixel_values.is_floating_point()
        or not bool(torch.isfinite(pixel_values).all())
    ):
        raise ValueError("SigLIP evaluation pixels differ from the frozen input contract")
    return pixel_values.float()


def materialize_control_training_batch(
    *,
    examples: tuple[ImageExample, ...],
    positions: tuple[int, ...],
    transform: Callable[[object], torch.Tensor],
    materialize: Callable[[object], object] = materialize_image,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize and augment every selected image exactly once in fixed order."""

    if type(examples) is not tuple or not examples:
        raise ValueError("training examples must be a nonempty concrete tuple")
    if type(positions) is not tuple or not positions:
        raise ValueError("training positions must be a nonempty concrete tuple")
    if any(
        type(position) is not int or not 0 <= position < len(examples) for position in positions
    ):
        raise ValueError("training positions exceed the optimization manifest")
    if len(set(positions)) != len(positions):
        raise ValueError("a logical batch may not repeat an image")
    tensors: list[torch.Tensor] = []
    labels: list[int] = []
    expected_shape: tuple[int, ...] | None = None
    for position in positions:
        example = examples[position]
        tensor = transform(materialize(example.image))
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or not tensor.is_floating_point()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError("training transform must return one finite floating image tensor")
        shape = tuple(int(dimension) for dimension in tensor.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError("training transform returned inconsistent image shapes")
        tensors.append(tensor)
        labels.append(example.label)
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.int64)


def embed_control_examples(
    *,
    model: PooledProxyAnchorModel,
    examples: tuple[ImageExample, ...],
    processor: Any,
    device: torch.device,
    batch_size: int,
    materialize: Callable[[object], object] = materialize_image,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed one isolated band through raw pooler and projected descriptor paths."""

    if type(examples) is not tuple or not examples:
        raise ValueError("embedding examples must be a nonempty concrete tuple")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("embedding batch size must be positive")
    model.eval()
    raw_batches: list[torch.Tensor] = []
    projected_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            images = [materialize(example.image) for example in batch]
            pixels = preprocess_control_evaluation(processor, images).to(device)
            pooled = model.tower(pixels).float()
            if (
                pooled.ndim != 2
                or pooled.shape[1] != model.projection.in_features
                or not bool(torch.isfinite(pooled).all())
                or bool((torch.linalg.vector_norm(pooled, dim=1) <= 0).any())
            ):
                raise ValueError("raw SigLIP pooler descriptors differ from authority")
            projected = model.projection(pooled).float()
            if not bool(torch.isfinite(projected).all()) or bool(
                (torch.linalg.vector_norm(projected, dim=1) <= 0).any()
            ):
                raise ValueError("projected control descriptors differ from authority")
            raw_batches.append(F.normalize(pooled, dim=1).cpu())
            projected_batches.append(F.normalize(projected, dim=1).cpu())
    labels = torch.tensor([example.label for example in examples], dtype=torch.int64)
    return torch.cat(raw_batches), torch.cat(projected_batches), labels


def train_control_epoch(
    *,
    model: PooledProxyAnchorModel,
    optimizer: torch.optim.Optimizer,
    examples: tuple[ImageExample, ...],
    transform: Callable[[object], torch.Tensor],
    seed: int,
    epoch: int,
    steps_per_epoch: int,
    sampler_state: SamplerState,
    microbatch_size: int,
    config: SiglipProxyControlConfig,
    device: torch.device,
    materialize: Callable[[object], object] = materialize_image,
) -> EpochTrainingEvidence:
    """Execute one frozen epoch using exact logical-batch score replay."""

    if type(seed) is not int or seed not in config.seeds:
        raise ValueError("training seed differs from the frozen control seeds")
    if type(epoch) is not int or not 0 <= epoch < config.train_epochs:
        raise ValueError("training epoch differs from the frozen epoch range")
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("steps per epoch must be positive")
    example_ids = tuple(example.example_id for example in examples)
    labels = torch.tensor([example.label for example in examples], dtype=torch.int64)
    batches, next_sampler_state = _build_epoch_batches(
        example_ids=example_ids,
        labels=labels,
        seed=seed,
        epoch=epoch,
        steps_per_epoch=steps_per_epoch,
        state=sampler_state,
    )
    model.train()
    losses: list[float] = []
    maximum_disagreement = 0.0
    for step, positions in enumerate(batches):
        multiplier = _learning_rate_multiplier(
            config,
            step=epoch * steps_per_epoch + step,
            steps_per_epoch=steps_per_epoch,
        )
        for group in optimizer.param_groups:
            base_learning_rate = group.get("initial_lr")
            if type(base_learning_rate) is not float or base_learning_rate <= 0.0:
                raise ValueError("optimizer group lacks its frozen initial learning rate")
            group["lr"] = base_learning_rate * multiplier
        pixels, batch_labels = materialize_control_training_batch(
            examples=examples,
            positions=positions,
            transform=transform,
            materialize=materialize,
        )
        optimizer.zero_grad(set_to_none=True)
        replay = recomputed_proxy_anchor_backward(
            model,
            pixels.to(device),
            batch_labels.to(device),
            microbatch_size=microbatch_size,
            alpha=config.proxy_anchor_alpha,
            delta=config.proxy_anchor_delta,
            score_tolerance=config.replay_score_tolerance,
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("control gradient norm must remain finite")
        optimizer.step()
        loss = float(replay.loss)
        if not math.isfinite(loss):
            raise RuntimeError("control loss must remain finite")
        losses.append(loss)
        maximum_disagreement = max(
            maximum_disagreement,
            replay.maximum_score_disagreement,
        )
    return EpochTrainingEvidence(
        epoch=epoch,
        optimizer_steps=len(batches),
        losses=tuple(losses),
        maximum_score_disagreement=maximum_disagreement,
        sampler_state=next_sampler_state,
    )


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
    modules = dict(model.named_modules())
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        owner_name, _, parameter_name = name.rpartition(".")
        owner = modules.get(owner_name)
        exclude_decay = parameter_name == "bias" or isinstance(owner, nn.LayerNorm)
        if name == "proxies":
            learning_rate = config.proxy_learning_rate
            decay = 0.0
        elif name.startswith("projection."):
            learning_rate = config.projection_learning_rate
            decay = 0.0 if exclude_decay else config.weight_decay
        elif name.startswith("tower."):
            learning_rate = config.tower_learning_rate
            decay = 0.0 if exclude_decay else config.weight_decay
        else:
            raise ValueError(f"unclassified trainable parameter: {name}")
        grouped.setdefault((learning_rate, decay), []).append(parameter)
    groups: list[dict[str, Any]] = [
        {
            "params": parameters,
            "lr": learning_rate,
            "initial_lr": learning_rate,
            "weight_decay": decay,
        }
        for (learning_rate, decay), parameters in sorted(grouped.items())
    ]
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    expected_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    duplicated = len(parameter_ids) != len(set(parameter_ids))
    incomplete = sorted(parameter_ids) != sorted(expected_ids)
    if duplicated or incomplete:
        raise RuntimeError("optimizer groups do not partition trainable parameters exactly once")
    return groups
