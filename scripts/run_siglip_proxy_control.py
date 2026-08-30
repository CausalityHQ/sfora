#!/usr/bin/env python3
"""Run the authenticated SigLIP-so400m pooled Proxy Anchor control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.data import (
    _HF_DATASET_REVISIONS,
    ImageExample,
    load_image_retrieval_examples,
    materialize_image,
)
from sfora.siglip_proxy_control import (
    ControlBandEvidence,
    PooledProxyAnchorModel,
    SeedControlEvidence,
    SiglipProxyControlConfig,
    evaluate_control_band,
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
        _fsync_directory(path.parent)
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
    current_epoch = step // steps_per_epoch + 1
    decays = sum(current_epoch >= epoch for epoch in config.decay_epochs)
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
    failure_reason: str | None = None


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
    final_objective: float
    maximum_score_disagreement: float
    initial_snapshot_sha256: str


@dataclass(frozen=True)
class BandRepresentationEvidence:
    """Raw-pooler and projected evidence for one isolated class band."""

    raw: ControlBandEvidence
    projected: ControlBandEvidence


@dataclass(frozen=True)
class ControlEvaluationSnapshot:
    """One permitted initial or final evaluation over all three class bands."""

    optimization: BandRepresentationEvidence
    clean_validation: BandRepresentationEvidence
    burned_diagnostic: BandRepresentationEvidence


@dataclass(frozen=True)
class ControlSeedRunResult:
    """Authenticated in-memory result of one complete scientific seed."""

    seed: int
    initial: ControlEvaluationSnapshot
    final: ControlEvaluationSnapshot
    seed_evidence: SeedControlEvidence
    final_objective: float
    optimizer_steps: int
    maximum_score_disagreement: float
    final_checkpoint: CheckpointAuthority
    initial_model_sha256: str
    wall_seconds: float
    examples_per_second: float
    peak_process_rss_bytes: int
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int


@dataclass(frozen=True)
class ControlRunAuthority:
    """Source, data ordering, environment, and resolved execution authority."""

    source_revision: str
    source_tree_digest: str
    manifest_sha256: str
    torch_version: str
    transformers_version: str
    torchvision_version: str
    cuda_runtime: str | None
    device_name: str
    microbatch_size: int
    steps_per_epoch: int

    def __post_init__(self) -> None:
        _require_lower_hex(self.source_revision, length=40, name="source revision")
        _require_lower_hex(self.source_tree_digest, length=64, name="source tree digest")
        _require_lower_hex(self.manifest_sha256, length=64, name="manifest SHA-256")
        for name in (
            "torch_version",
            "transformers_version",
            "torchvision_version",
            "device_name",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a nonempty concrete string")
        if self.cuda_runtime is not None and (
            type(self.cuda_runtime) is not str or not self.cuda_runtime
        ):
            raise ValueError("CUDA runtime must be null or a nonempty concrete string")
        if (
            type(self.microbatch_size) is not int
            or self.microbatch_size < 1
            or 120 % self.microbatch_size != 0
        ):
            raise ValueError("authority microbatch must divide 120")
        if type(self.steps_per_epoch) is not int or self.steps_per_epoch < 1:
            raise ValueError("authority steps per epoch must be positive")


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


def _require_lower_hex(value: object, *, length: int, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be exactly {length} lowercase hexadecimal characters")


def require_control_determinism(device: torch.device) -> None:
    """Enable and verify the exact deterministic torch execution envelope."""

    if device.type not in {"cpu", "cuda"}:
        raise ValueError("control device must be CPU or CUDA")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("torch refused the frozen deterministic execution envelope")


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


def _run_authority_sha256(authority: ControlRunAuthority) -> str:
    payload = _json_compatible(vars(authority))
    if type(payload) is not dict:
        raise TypeError("run authority did not produce an object")
    return hashlib.sha256(_canonical_bytes(cast(dict[str, Any], payload))).hexdigest()


def control_manifest_sha256(examples: tuple[ImageExample, ...]) -> str:
    """Hash the exact ordered example-ID/label manifest used by the sampler."""

    if type(examples) is not tuple or not examples:
        raise ValueError("control manifest must be a nonempty concrete tuple")
    rows = [{"example_id": example.example_id, "label": example.label} for example in examples]
    return hashlib.sha256(_canonical_bytes({"examples": rows})).hexdigest()


def _band_scalar_payload(evidence: ControlBandEvidence) -> dict[str, object]:
    return {
        "correct": evidence.retrieval.correct,
        "queries": evidence.retrieval.queries,
        "recall_at_1": evidence.retrieval.recall_at_1,
        "mean_nearest_positive_cosine": evidence.margins.mean_nearest_positive_cosine,
        "mean_nearest_negative_cosine": evidence.margins.mean_nearest_negative_cosine,
        "mean_margin": evidence.margins.mean_margin,
    }


def _snapshot_scalar_payload(snapshot: ControlEvaluationSnapshot) -> dict[str, object]:
    bands = {
        "optimization": snapshot.optimization,
        "clean_validation": snapshot.clean_validation,
        "burned_diagnostic": snapshot.burned_diagnostic,
    }
    return {
        role: {
            "raw": _band_scalar_payload(evidence.raw),
            "projected": _band_scalar_payload(evidence.projected),
        }
        for role, evidence in bands.items()
    }


def _snapshot_sha256(snapshot: ControlEvaluationSnapshot) -> str:
    return hashlib.sha256(_canonical_bytes(_snapshot_scalar_payload(snapshot))).hexdigest()


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        metadata = _canonical_bytes(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            }
        )
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def write_control_checkpoint(
    path: Path,
    *,
    model: PooledProxyAnchorModel,
    optimizer: torch.optim.Optimizer,
    config: SiglipProxyControlConfig,
    seed: int,
    completed_epoch: int,
    sampler_state: SamplerState,
    final_objective: float,
    maximum_score_disagreement: float,
    run_authority: ControlRunAuthority,
    initial_snapshot_sha256: str,
) -> None:
    """Write one create-new checkpoint payload for authenticated publication."""

    _validate_checkpoint_coordinates(seed=seed, epoch=completed_epoch)
    if len(sampler_state.cycles) != 49 or len(sampler_state.positions) != 49:
        raise ValueError("checkpoint sampler state must bind all optimization classes")
    if type(final_objective) is not float or not math.isfinite(final_objective):
        raise ValueError("checkpoint objective must be a concrete finite float")
    if (
        type(maximum_score_disagreement) is not float
        or not math.isfinite(maximum_score_disagreement)
        or not 0.0 <= maximum_score_disagreement <= config.replay_score_tolerance
    ):
        raise ValueError("checkpoint replay disagreement differs from authority")
    _require_lower_hex(
        initial_snapshot_sha256,
        length=64,
        name="initial snapshot SHA-256",
    )
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
        "final_objective": final_objective,
        "initial_snapshot_sha256": initial_snapshot_sha256,
        "maximum_score_disagreement": maximum_score_disagreement,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "run_authority_sha256": _run_authority_sha256(run_authority),
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
    expected_run_authority: ControlRunAuthority,
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
        "final_objective",
        "initial_snapshot_sha256",
        "maximum_score_disagreement",
        "model_state",
        "optimizer_state",
        "run_authority_sha256",
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
    final_objective = payload["final_objective"]
    initial_snapshot_sha256 = payload["initial_snapshot_sha256"]
    maximum_score_disagreement = payload["maximum_score_disagreement"]
    if (
        type(seed) is not int
        or type(completed_epoch) is not int
        or type(cycles) is not tuple
        or type(positions) is not tuple
        or type(final_objective) is not float
        or not math.isfinite(final_objective)
        or type(maximum_score_disagreement) is not float
        or not math.isfinite(maximum_score_disagreement)
        or not 0.0 <= maximum_score_disagreement <= config.replay_score_tolerance
        or type(initial_snapshot_sha256) is not str
        or payload["claim_eligible"] is not False
        or payload["schema"] != _CHECKPOINT_PAYLOAD_SCHEMA
        or payload["config_sha256"] != _config_sha256(config)
    ):
        raise ValueError("control checkpoint authority differs")
    _require_lower_hex(
        initial_snapshot_sha256,
        length=64,
        name="initial snapshot SHA-256",
    )
    if payload["run_authority_sha256"] != _run_authority_sha256(expected_run_authority):
        raise ValueError("control checkpoint run authority differs")
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
        final_objective=final_objective,
        maximum_score_disagreement=maximum_score_disagreement,
        initial_snapshot_sha256=initial_snapshot_sha256,
    )


def load_control_examples(
    *,
    loader: Callable[..., list[ImageExample]] = load_image_retrieval_examples,
) -> ControlExampleBands:
    """Load Cars train once and partition it without exposing official test classes."""

    config = SiglipProxyControlConfig()
    if _HF_DATASET_REVISIONS.get(config.dataset_name) != config.dataset_revision:
        raise RuntimeError("Cars dataset revision differs from the frozen control authority")
    examples = loader(dataset_name="cars", split="train")
    if type(examples) is not list or not examples:
        raise ValueError("Cars train loader must return a nonempty concrete list")
    if any(type(example) is not ImageExample for example in examples):
        raise TypeError("Cars train loader returned a non-ImageExample value")
    labels = [example.label for example in examples]
    if any(type(label) is not int for label in labels):
        raise TypeError("Cars labels must be concrete integers")
    allowed_classes = F1_TRAIN_CLASSES | F1_VALIDATION_CLASSES | SUBSTRATE_F0_CLASSES
    if any(label not in allowed_classes for label in labels):
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
    resolved_revision = getattr(getattr(vision_model, "config", None), "_commit_hash", None)
    if resolved_revision != config.model_revision:
        raise ValueError("SigLIP resolved model revision differs from authority")
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
    if not isinstance(encoded, Mapping) or "pixel_values" not in encoded:
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
    maximum_steps: int | None = None,
) -> EpochTrainingEvidence:
    """Execute one frozen epoch using exact logical-batch score replay."""

    if type(seed) is not int or seed not in config.seeds:
        raise ValueError("training seed differs from the frozen control seeds")
    if type(epoch) is not int or not 0 <= epoch < config.train_epochs:
        raise ValueError("training epoch differs from the frozen epoch range")
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("steps per epoch must be positive")
    if maximum_steps is not None and (
        type(maximum_steps) is not int or not 1 <= maximum_steps <= steps_per_epoch
    ):
        raise ValueError("maximum steps must lie within the resolved epoch")
    executed_steps = steps_per_epoch if maximum_steps is None else maximum_steps
    example_ids = tuple(example.example_id for example in examples)
    labels = torch.tensor([example.label for example in examples], dtype=torch.int64)
    batches, next_sampler_state = _build_epoch_batches(
        example_ids=example_ids,
        labels=labels,
        seed=seed,
        epoch=epoch,
        steps_per_epoch=executed_steps,
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


def _memory_psi_full_avg10() -> float:
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "full":
            for field in fields[1:]:
                if field.startswith("avg10="):
                    return float(field.removeprefix("avg10="))
    raise RuntimeError("memory PSI full avg10 is unavailable")


def _swap_used_bytes() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"SwapTotal", "SwapFree"}:
            values[key] = int(value.split()[0]) * 1024
    if set(values) != {"SwapTotal", "SwapFree"}:
        raise RuntimeError("swap authority is unavailable")
    return values["SwapTotal"] - values["SwapFree"]


def run_control_smoke_rung(
    *,
    config: SiglipProxyControlConfig,
    optimization_examples: tuple[ImageExample, ...],
    microbatch_size: int,
    steps_per_epoch: int,
    device: torch.device,
) -> SmokeObservation:
    """Run one fresh-process, optimization-only three-step smoke rung."""

    require_control_determinism(device)
    if microbatch_size not in config.smoke_microbatch_ladder:
        raise ValueError("smoke microbatch is outside the frozen ladder")
    labels = torch.tensor([example.label for example in optimization_examples])
    validate_control_partition(
        optimization_labels=labels,
        clean_validation_labels=torch.tensor(
            [label for label in sorted(F1_VALIDATION_CLASSES) for _ in range(2)]
        ),
        burned_diagnostic_labels=torch.tensor(
            [label for label in sorted(SUBSTRATE_F0_CLASSES) for _ in range(2)]
        ),
    )
    psi_before = _memory_psi_full_avg10()
    swap_before = _swap_used_bytes()
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("the pooled-control smoke requires CUDA bf16")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    tower, _processor = load_siglip_control_components(config=config)
    torch.manual_seed(config.seeds[0])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seeds[0])
    model = PooledProxyAnchorModel(
        tower=tower,
        input_dimensions=config.input_dimensions,
        embedding_dimensions=config.embedding_dimensions,
        class_count=len(F1_TRAIN_CLASSES),
    ).to(device)
    optimizer = torch.optim.AdamW(_optimizer_groups(model, config))
    started_at = time.monotonic()
    evidence = train_control_epoch(
        model=model,
        optimizer=optimizer,
        examples=optimization_examples,
        transform=build_control_train_transform(),
        seed=config.seeds[0],
        epoch=0,
        steps_per_epoch=steps_per_epoch,
        sampler_state=SamplerState.initial(),
        microbatch_size=microbatch_size,
        config=config,
        device=device,
        maximum_steps=3,
    )
    elapsed = max(time.monotonic() - started_at, 1.0e-12)
    complete_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.tower.parameters()
        if parameter.requires_grad
    )
    observation = SmokeObservation(
        microbatch_size=microbatch_size,
        steps_completed=evidence.optimizer_steps,
        peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        peak_cuda_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        peak_cuda_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        memory_psi_growth=max(0.0, _memory_psi_full_avg10() - psi_before),
        swap_growth_bytes=max(0, _swap_used_bytes() - swap_before),
        examples_per_second=(evidence.optimizer_steps * config.logical_batch_size) / elapsed,
        final_loss=evidence.losses[-1],
        complete_tower_gradient_coverage=complete_gradients,
        maximum_score_disagreement=evidence.maximum_score_disagreement,
    )
    del optimizer, model, tower
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return observation


def _evaluate_control_snapshot(
    *,
    model: PooledProxyAnchorModel,
    bands: ControlExampleBands,
    processor: Any,
    device: torch.device,
    batch_size: int,
    query_block: int,
) -> ControlEvaluationSnapshot:
    evidence: list[BandRepresentationEvidence] = []
    for examples in (
        bands.optimization,
        bands.clean_validation,
        bands.burned_diagnostic,
    ):
        raw, projected, labels = embed_control_examples(
            model=model,
            examples=examples,
            processor=processor,
            device=device,
            batch_size=batch_size,
        )
        evidence.append(
            BandRepresentationEvidence(
                raw=evaluate_control_band(raw, labels, query_block=query_block),
                projected=evaluate_control_band(projected, labels, query_block=query_block),
            )
        )
    return ControlEvaluationSnapshot(
        optimization=evidence[0],
        clean_validation=evidence[1],
        burned_diagnostic=evidence[2],
    )


def run_control_seed(
    *,
    config: SiglipProxyControlConfig,
    seed: int,
    bands: ControlExampleBands,
    checkpoint_directory: Path,
    maximum_checkpoint_bytes: int,
    microbatch_size: int,
    evaluation_batch_size: int,
    query_block: int,
    device: torch.device,
    smoke_receipt: SmokeReceipt,
    run_authority: ControlRunAuthority,
) -> ControlSeedRunResult:
    """Run or resume one frozen seed without intermediate evaluation."""

    started_at = time.monotonic()
    require_control_determinism(device)
    if type(seed) is not int or seed not in config.seeds:
        raise ValueError("seed differs from the frozen control seeds")
    if type(microbatch_size) is not int or config.logical_batch_size % microbatch_size != 0:
        raise ValueError("microbatch size must divide the frozen logical batch")
    if type(evaluation_batch_size) is not int or evaluation_batch_size < 1:
        raise ValueError("evaluation batch size must be positive")
    if type(query_block) is not int or query_block < 1:
        raise ValueError("query block must be positive")
    if (
        smoke_receipt.selected_microbatch_size != microbatch_size
        or not smoke_receipt.observations
        or smoke_receipt.observations[-1].microbatch_size != microbatch_size
        or not _smoke_observation_passes(
            smoke_receipt.observations[-1],
            config=config,
            steps_per_epoch=run_authority.steps_per_epoch,
        )
    ):
        raise ValueError("scientific microbatch lacks a passing smoke authority")
    validate_control_partition(
        optimization_labels=torch.tensor(
            [example.label for example in bands.optimization], dtype=torch.int64
        ),
        clean_validation_labels=torch.tensor(
            [example.label for example in bands.clean_validation], dtype=torch.int64
        ),
        burned_diagnostic_labels=torch.tensor(
            [example.label for example in bands.burned_diagnostic], dtype=torch.int64
        ),
    )
    steps_per_epoch = config.steps_per_epoch(len(bands.optimization))
    if steps_per_epoch < 1:
        raise ValueError("optimization split does not resolve one logical step")
    if (
        run_authority.microbatch_size != microbatch_size
        or run_authority.steps_per_epoch != steps_per_epoch
        or run_authority.manifest_sha256 != control_manifest_sha256(bands.ordered_manifest)
        or run_authority.torch_version != torch.__version__
        or run_authority.cuda_runtime != torch.version.cuda
        or run_authority.device_name
        != (torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    ):
        raise ValueError("run authority differs from the resolved execution")
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("the pooled control requires CUDA bf16 support")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
        torch.cuda.reset_peak_memory_stats(device)
    tower, processor = load_siglip_control_components(config=config)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = PooledProxyAnchorModel(
        tower=tower,
        input_dimensions=config.input_dimensions,
        embedding_dimensions=config.embedding_dimensions,
        class_count=len(F1_TRAIN_CLASSES),
    ).to(device)
    initial_model_sha256 = _model_state_sha256(model)
    optimizer = torch.optim.AdamW(_optimizer_groups(model, config))

    initial = _evaluate_control_snapshot(
        model=model,
        bands=bands,
        processor=processor,
        device=device,
        batch_size=evaluation_batch_size,
        query_block=query_block,
    )
    initial_snapshot_sha256 = _snapshot_sha256(initial)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    _reap_orphan_checkpoints(checkpoint_directory, seed=seed)
    previous = latest_authenticated_checkpoint(checkpoint_directory, seed=seed)
    if previous is None:
        start_epoch = 0
        sampler_state = SamplerState.initial()
        final_objective = math.nan
        maximum_disagreement = 0.0
        final_checkpoint: CheckpointAuthority | None = None
    else:
        restored = restore_control_checkpoint(
            previous.path,
            model=model,
            optimizer=optimizer,
            config=config,
            expected_seed=seed,
            expected_run_authority=run_authority,
        )
        if restored.completed_epoch != previous.epoch:
            raise ValueError("checkpoint receipt and payload epochs differ")
        if restored.initial_snapshot_sha256 != initial_snapshot_sha256:
            raise ValueError("checkpoint initial snapshot authority differs")
        start_epoch = restored.completed_epoch
        sampler_state = restored.sampler_state
        final_objective = restored.final_objective
        maximum_disagreement = restored.maximum_score_disagreement
        final_checkpoint = previous

    transform = build_control_train_transform()
    training_started_at = time.monotonic()
    for epoch in range(start_epoch, config.train_epochs):
        epoch_evidence = train_control_epoch(
            model=model,
            optimizer=optimizer,
            examples=bands.optimization,
            transform=transform,
            seed=seed,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            sampler_state=sampler_state,
            microbatch_size=microbatch_size,
            config=config,
            device=device,
        )
        sampler_state = epoch_evidence.sampler_state
        final_objective = epoch_evidence.losses[-1]
        maximum_disagreement = max(
            maximum_disagreement,
            epoch_evidence.maximum_score_disagreement,
        )

        write_checkpoint = partial(
            write_control_checkpoint,
            model=model,
            optimizer=optimizer,
            config=config,
            seed=seed,
            completed_epoch=epoch + 1,
            sampler_state=sampler_state,
            final_objective=final_objective,
            maximum_score_disagreement=maximum_disagreement,
            run_authority=run_authority,
            initial_snapshot_sha256=initial_snapshot_sha256,
        )
        final_checkpoint = publish_epoch_checkpoint(
            directory=checkpoint_directory,
            seed=seed,
            epoch=epoch + 1,
            write_checkpoint=write_checkpoint,
            maximum_checkpoint_bytes=maximum_checkpoint_bytes,
        )
    if final_checkpoint is None or final_checkpoint.epoch != config.train_epochs:
        raise RuntimeError("control seed lacks its final authenticated checkpoint")

    final = _evaluate_control_snapshot(
        model=model,
        bands=bands,
        processor=processor,
        device=device,
        batch_size=evaluation_batch_size,
        query_block=query_block,
    )
    seed_evidence = SeedControlEvidence(
        seed=seed,
        train_initial_margin=initial.optimization.projected.margins.mean_margin,
        train_final_margin=final.optimization.projected.margins.mean_margin,
        clean_initial_recall_at_1=initial.clean_validation.projected.retrieval.recall_at_1,
        clean_final_recall_at_1=final.clean_validation.projected.retrieval.recall_at_1,
        clean_initial_margin=initial.clean_validation.projected.margins.mean_margin,
        clean_final_margin=final.clean_validation.projected.margins.mean_margin,
        burned_initial_margin=initial.burned_diagnostic.projected.margins.mean_margin,
        burned_final_margin=final.burned_diagnostic.projected.margins.mean_margin,
    )
    wall_seconds = time.monotonic() - started_at
    training_seconds = max(time.monotonic() - training_started_at, 1.0e-12)
    trained_examples = (
        (config.train_epochs - start_epoch) * steps_per_epoch * config.logical_batch_size
    )
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return ControlSeedRunResult(
        seed=seed,
        initial=initial,
        final=final,
        seed_evidence=seed_evidence,
        final_objective=final_objective,
        optimizer_steps=config.train_epochs * steps_per_epoch,
        maximum_score_disagreement=maximum_disagreement,
        final_checkpoint=final_checkpoint,
        initial_model_sha256=initial_model_sha256,
        wall_seconds=wall_seconds,
        examples_per_second=trained_examples / training_seconds,
        peak_process_rss_bytes=peak_rss_bytes,
        peak_cuda_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        peak_cuda_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    )


def control_seed_receipt_bytes(
    *,
    result: ControlSeedRunResult,
    config: SiglipProxyControlConfig,
    bands: ControlExampleBands,
    smoke_receipt: SmokeReceipt,
    smoke_sha256: str,
    run_authority: ControlRunAuthority,
) -> bytes:
    """Serialize one strict claim-ineligible scientific seed receipt."""

    _require_lower_hex(smoke_sha256, length=64, name="smoke SHA-256")
    if result.seed not in config.seeds or result.final_checkpoint.epoch != config.train_epochs:
        raise ValueError("seed result differs from the frozen terminal authority")
    if result.final_checkpoint.seed != result.seed:
        raise ValueError("seed result and checkpoint identities differ")
    if run_authority.manifest_sha256 != control_manifest_sha256(bands.ordered_manifest):
        raise ValueError("receipt manifest authority differs")
    seed_evidence = result.seed_evidence
    if seed_evidence.seed != result.seed:
        raise ValueError("seed evidence identity differs")
    train_change = seed_evidence.train_final_margin - seed_evidence.train_initial_margin
    clean_recall_change = (
        seed_evidence.clean_final_recall_at_1 - seed_evidence.clean_initial_recall_at_1
    )
    clean_margin_change = seed_evidence.clean_final_margin - seed_evidence.clean_initial_margin
    burned_change = seed_evidence.burned_final_margin - seed_evidence.burned_initial_margin
    ratio = burned_change / train_change if train_change > 0.0 else None
    config_payload = _json_compatible(vars(config))
    if type(config_payload) is not dict:
        raise TypeError("control config did not produce an object")
    smoke_payload = {
        "observations": [
            cast(dict[str, object], _json_compatible(vars(observation)))
            for observation in smoke_receipt.observations
        ],
        "projected_seed_seconds": smoke_receipt.projected_seed_seconds,
        "selected_microbatch_size": smoke_receipt.selected_microbatch_size,
        "sha256": smoke_sha256,
    }
    payload: dict[str, Any] = {
        "schema": "sfora-siglip-proxy-control-seed-v1",
        "claim_eligible": False,
        "seed": result.seed,
        "source": {
            "revision": run_authority.source_revision,
            "tree_digest": run_authority.source_tree_digest,
            "dirty": False,
        },
        "dataset": {
            "name": config.dataset_name,
            "revision": config.dataset_revision,
            "manifest_sha256": run_authority.manifest_sha256,
            "optimization_examples": len(bands.optimization),
            "clean_validation_examples": len(bands.clean_validation),
            "burned_diagnostic_examples": len(bands.burned_diagnostic),
        },
        "model": {
            "name": config.model_name,
            "revision": config.model_revision,
            "resolved_revision": config.model_revision,
            "initial_state_sha256": result.initial_model_sha256,
        },
        "config": config_payload,
        "config_sha256": _config_sha256(config),
        "smoke": smoke_payload,
        "evaluation": {
            "initial": _snapshot_scalar_payload(result.initial),
            "final": _snapshot_scalar_payload(result.final),
        },
        "changes": {
            "train_margin_change": train_change,
            "clean_recall_change": clean_recall_change,
            "clean_margin_change": clean_margin_change,
            "burned_margin_change": burned_change,
            "memorization_to_transfer_ratio": ratio,
            "transfer_mechanism_conclusion_supported": ratio is not None,
        },
        "training": {
            "optimizer_steps": result.optimizer_steps,
            "steps_per_epoch": run_authority.steps_per_epoch,
            "microbatch_size": run_authority.microbatch_size,
            "final_objective": result.final_objective,
            "maximum_score_disagreement": result.maximum_score_disagreement,
        },
        "checkpoint": {
            "basename": result.final_checkpoint.path.name,
            "receipt_basename": result.final_checkpoint.receipt_path.name,
            "sha256": result.final_checkpoint.sha256,
            "bytes": result.final_checkpoint.bytes,
            "epoch": result.final_checkpoint.epoch,
        },
        "resources": {
            "wall_seconds": result.wall_seconds,
            "examples_per_second": result.examples_per_second,
            "peak_process_rss_bytes": result.peak_process_rss_bytes,
            "peak_cuda_allocated_bytes": result.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": result.peak_cuda_reserved_bytes,
        },
        "environment": cast(dict[str, object], _json_compatible(vars(run_authority))),
    }
    return _canonical_bytes(payload)


def control_aggregate_receipt_bytes(seed_receipts: tuple[bytes, ...]) -> bytes:
    """Authenticate and aggregate the exact three terminal seed receipts."""

    if type(seed_receipts) is not tuple or len(seed_receipts) != 3:
        raise ValueError("aggregate requires the exact seeds 17, 29, and 43")
    expected_keys = {
        "schema",
        "claim_eligible",
        "seed",
        "source",
        "dataset",
        "model",
        "config",
        "config_sha256",
        "smoke",
        "evaluation",
        "changes",
        "training",
        "checkpoint",
        "resources",
        "environment",
    }
    parsed: list[dict[str, Any]] = []
    for raw in seed_receipts:
        if type(raw) is not bytes:
            raise TypeError("seed receipts must be concrete bytes")
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("seed receipt is not valid JSON") from error
        if (
            type(value) is not dict
            or set(value) != expected_keys
            or value.get("schema") != "sfora-siglip-proxy-control-seed-v1"
            or value.get("claim_eligible") is not False
            or raw != _canonical_bytes(cast(dict[str, Any], value))
        ):
            raise ValueError("seed receipt authority differs")
        parsed.append(cast(dict[str, Any], value))
    seeds = tuple(value["seed"] for value in parsed)
    if seeds != (17, 29, 43) or any(type(seed) is not int for seed in seeds):
        raise ValueError("aggregate requires the exact seeds 17, 29, and 43")
    initial_recalls: list[float] = []
    final_recalls: list[float] = []
    ratios: list[float | None] = []
    for value in parsed:
        try:
            initial = value["evaluation"]["initial"]["clean_validation"]["projected"]["recall_at_1"]
            final = value["evaluation"]["final"]["clean_validation"]["projected"]["recall_at_1"]
            ratio = value["changes"]["memorization_to_transfer_ratio"]
        except (KeyError, TypeError) as error:
            raise ValueError("seed receipt evidence schema differs") from error
        if (
            type(initial) is not float
            or type(final) is not float
            or not math.isfinite(initial)
            or not math.isfinite(final)
            or not 0.0 <= initial <= 1.0
            or not 0.0 <= final <= 1.0
            or (ratio is not None and (type(ratio) is not float or not math.isfinite(ratio)))
        ):
            raise ValueError("seed receipt aggregate evidence differs")
        initial_recalls.append(initial)
        final_recalls.append(final)
        ratios.append(ratio)
    all_ratios_defined = all(ratio is not None for ratio in ratios)
    mean_ratio = sum(cast(float, ratio) for ratio in ratios) / 3.0 if all_ratios_defined else None
    return _canonical_bytes(
        {
            "schema": "sfora-siglip-proxy-control-aggregate-v1",
            "claim_eligible": False,
            "seeds": list(seeds),
            "seed_receipts": [
                {"seed": seed, "sha256": hashlib.sha256(raw).hexdigest()}
                for seed, raw in zip(seeds, seed_receipts, strict=True)
            ],
            "mean_clean_initial_recall_at_1": sum(initial_recalls) / 3.0,
            "mean_clean_final_recall_at_1": sum(final_recalls) / 3.0,
            "mean_clean_recall_change": sum(
                final - initial
                for initial, final in zip(initial_recalls, final_recalls, strict=True)
            )
            / 3.0,
            "memorization_to_transfer_ratios": ratios,
            "mean_memorization_to_transfer_ratio": mean_ratio,
        }
    )


def _smoke_observation_payload(observation: SmokeObservation) -> dict[str, object]:
    payload = _json_compatible(vars(observation))
    if type(payload) is not dict:
        raise TypeError("smoke observation did not produce an object")
    return cast(dict[str, object], payload)


def _smoke_receipt_bytes(
    receipt: SmokeReceipt,
    *,
    config: SiglipProxyControlConfig,
    source_revision: str,
    source_tree_digest: str,
    manifest_sha256: str,
    steps_per_epoch: int,
) -> bytes:
    _require_lower_hex(source_revision, length=40, name="source revision")
    _require_lower_hex(source_tree_digest, length=64, name="source tree digest")
    _require_lower_hex(manifest_sha256, length=64, name="manifest SHA-256")
    return _canonical_bytes(
        {
            "schema": "sfora-siglip-proxy-control-smoke-v1",
            "claim_eligible": False,
            "source_revision": source_revision,
            "source_tree_digest": source_tree_digest,
            "manifest_sha256": manifest_sha256,
            "config_sha256": _config_sha256(config),
            "steps_per_epoch": steps_per_epoch,
            "observations": [
                _smoke_observation_payload(observation) for observation in receipt.observations
            ],
            "selected_microbatch_size": receipt.selected_microbatch_size,
            "projected_seed_seconds": receipt.projected_seed_seconds,
        }
    )


def _read_smoke_receipt(
    path: Path,
    *,
    config: SiglipProxyControlConfig,
    source_revision: str,
    source_tree_digest: str,
    manifest_sha256: str,
    steps_per_epoch: int,
) -> tuple[SmokeReceipt, bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("smoke receipt is not valid JSON") from error
    if type(payload) is not dict or raw != _canonical_bytes(cast(dict[str, Any], payload)):
        raise ValueError("smoke receipt is not canonical")
    expected_keys = {
        "schema",
        "claim_eligible",
        "source_revision",
        "source_tree_digest",
        "manifest_sha256",
        "config_sha256",
        "steps_per_epoch",
        "observations",
        "selected_microbatch_size",
        "projected_seed_seconds",
    }
    if (
        set(payload) != expected_keys
        or payload["schema"] != "sfora-siglip-proxy-control-smoke-v1"
        or payload["claim_eligible"] is not False
        or payload["source_revision"] != source_revision
        or payload["source_tree_digest"] != source_tree_digest
        or payload["manifest_sha256"] != manifest_sha256
        or payload["config_sha256"] != _config_sha256(config)
        or payload["steps_per_epoch"] != steps_per_epoch
        or type(payload["observations"]) is not list
    ):
        raise ValueError("smoke receipt authority differs")
    observations: list[SmokeObservation] = []
    for row in payload["observations"]:
        if type(row) is not dict:
            raise ValueError("smoke observation schema differs")
        try:
            observations.append(SmokeObservation(**row))
        except (TypeError, ValueError) as error:
            raise ValueError("smoke observation schema differs") from error
    receipt = SmokeReceipt(
        observations=tuple(observations),
        selected_microbatch_size=payload["selected_microbatch_size"],
        projected_seed_seconds=payload["projected_seed_seconds"],
    )
    expected_bytes = _smoke_receipt_bytes(
        receipt,
        config=config,
        source_revision=source_revision,
        source_tree_digest=source_tree_digest,
        manifest_sha256=manifest_sha256,
        steps_per_epoch=steps_per_epoch,
    )
    if raw != expected_bytes or not receipt.observations:
        raise ValueError("smoke receipt evidence differs")
    selected = receipt.observations[-1]
    if (
        selected.microbatch_size != receipt.selected_microbatch_size
        or not _smoke_observation_passes(
            selected,
            config=config,
            steps_per_epoch=steps_per_epoch,
        )
    ):
        raise ValueError("smoke receipt lacks a passing selected rung")
    return receipt, raw


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


def _reap_orphan_checkpoints(directory: Path, *, seed: int) -> None:
    """Remove only never-authenticated interrupted publications for one seed."""

    _validate_checkpoint_coordinates(seed=seed, epoch=1)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint directory must be a real directory")
    changed = False
    prefix = f"seed-{seed:03d}-epoch-"
    for partial_path in directory.glob(f"{prefix}*.pt.partial"):
        partial_path.unlink()
        changed = True
    for checkpoint in directory.glob(f"{prefix}*.pt"):
        receipt = checkpoint.with_name(f"{checkpoint.name.removesuffix('.pt')}.checkpoint.json")
        if not receipt.exists() and not receipt.is_symlink():
            checkpoint.unlink()
            changed = True
    for receipt in directory.glob(f"{prefix}*.checkpoint.json"):
        checkpoint = receipt.with_name(f"{receipt.name.removesuffix('.checkpoint.json')}.pt")
        if not checkpoint.exists() and not checkpoint.is_symlink():
            receipt.unlink()
            changed = True
    if changed:
        _fsync_directory(directory)


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
    _reap_orphan_checkpoints(directory, seed=seed)
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
    if observation.failure_reason is not None:
        return False
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
        try:
            observation = run_rung(microbatch_size)
        except torch.cuda.OutOfMemoryError:
            observation = SmokeObservation(
                microbatch_size=microbatch_size,
                steps_completed=0,
                peak_process_rss_bytes=0,
                peak_cuda_allocated_bytes=0,
                peak_cuda_reserved_bytes=0,
                memory_psi_growth=0.0,
                swap_growth_bytes=0,
                examples_per_second=0.0,
                final_loss=0.0,
                complete_tower_gradient_coverage=False,
                maximum_score_disagreement=0.0,
                failure_reason="cuda-out-of-memory",
            )
            torch.cuda.empty_cache()
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


def parse_control_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the three capability-separated pooled-control phases."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--source-revision", required=True)
    smoke.add_argument("--source-tree-digest", required=True)

    rung = subparsers.add_parser("smoke-rung", help=argparse.SUPPRESS)
    rung.add_argument("--output", type=Path, required=True)
    rung.add_argument("--microbatch-size", type=int, required=True)
    rung.add_argument("--steps-per-epoch", type=int, required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--smoke", type=Path, required=True)
    train.add_argument("--seed", type=int, choices=_CONTROL_SEEDS, required=True)
    train.add_argument("--source-revision", required=True)
    train.add_argument("--source-tree-digest", required=True)
    train.add_argument("--maximum-checkpoint-bytes", type=int, default=8 * 1024**3)
    train.add_argument("--evaluation-batch-size", type=int, default=32)
    train.add_argument("--query-block", type=int, default=128)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument(
        "--seed-receipt",
        type=Path,
        action="append",
        required=True,
    )
    return parser.parse_args(arguments)


def _current_run_authority(
    *,
    source_revision: str,
    source_tree_digest: str,
    bands: ControlExampleBands,
    microbatch_size: int,
    steps_per_epoch: int,
    device: torch.device,
) -> ControlRunAuthority:
    import torchvision
    import transformers

    return ControlRunAuthority(
        source_revision=source_revision,
        source_tree_digest=source_tree_digest,
        manifest_sha256=control_manifest_sha256(bands.ordered_manifest),
        torch_version=str(torch.__version__),
        transformers_version=str(transformers.__version__),
        torchvision_version=str(torchvision.__version__),
        cuda_runtime=torch.version.cuda,
        device_name=(torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"),
        microbatch_size=microbatch_size,
        steps_per_epoch=steps_per_epoch,
    )


def _smoke_rung_result_bytes(observation: SmokeObservation) -> bytes:
    return _canonical_bytes(
        {
            "schema": "sfora-siglip-proxy-control-smoke-rung-v1",
            "claim_eligible": False,
            "observation": _smoke_observation_payload(observation),
        }
    )


def _read_smoke_rung_result(path: Path) -> SmokeObservation:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("smoke rung result is not valid JSON") from error
    if (
        type(payload) is not dict
        or set(payload) != {"schema", "claim_eligible", "observation"}
        or payload["schema"] != "sfora-siglip-proxy-control-smoke-rung-v1"
        or payload["claim_eligible"] is not False
        or type(payload["observation"]) is not dict
        or raw != _canonical_bytes(cast(dict[str, Any], payload))
    ):
        raise ValueError("smoke rung result authority differs")
    try:
        return SmokeObservation(**payload["observation"])
    except (TypeError, ValueError) as error:
        raise ValueError("smoke rung observation differs") from error


def _oom_smoke_observation(microbatch_size: int) -> SmokeObservation:
    return SmokeObservation(
        microbatch_size=microbatch_size,
        steps_completed=0,
        peak_process_rss_bytes=0,
        peak_cuda_allocated_bytes=0,
        peak_cuda_reserved_bytes=0,
        memory_psi_growth=0.0,
        swap_growth_bytes=0,
        examples_per_second=0.0,
        final_loss=0.0,
        complete_tower_gradient_coverage=False,
        maximum_score_disagreement=0.0,
        failure_reason="cuda-out-of-memory",
    )


def _run_smoke_subprocess(microbatch_size: int, *, steps_per_epoch: int) -> SmokeObservation:
    with tempfile.TemporaryDirectory(prefix="sfora-proxy-smoke-") as temporary:
        output = Path(temporary) / "rung.json"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "smoke-rung",
                "--output",
                str(output),
                "--microbatch-size",
                str(microbatch_size),
                "--steps-per-epoch",
                str(steps_per_epoch),
            ],
            check=True,
        )
        return _read_smoke_rung_result(output)


def main(arguments: list[str] | None = None) -> None:
    """Execute one capability-separated pooled-control phase."""

    args = parse_control_args(arguments)
    config = SiglipProxyControlConfig()
    if args.command == "aggregate":
        receipts = tuple(path.read_bytes() for path in args.seed_receipt)
        _write_new(args.output, control_aggregate_receipt_bytes(receipts))
        return

    device = torch.device("cuda")
    if args.command == "smoke-rung":
        bands = load_control_examples()
        try:
            observation = run_control_smoke_rung(
                config=config,
                optimization_examples=bands.optimization,
                microbatch_size=args.microbatch_size,
                steps_per_epoch=args.steps_per_epoch,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            observation = _oom_smoke_observation(args.microbatch_size)
        _write_new(args.output, _smoke_rung_result_bytes(observation))
        return

    bands = load_control_examples()
    steps_per_epoch = config.steps_per_epoch(len(bands.optimization))
    manifest_sha256 = control_manifest_sha256(bands.ordered_manifest)
    if args.command == "smoke":
        smoke_result = run_memory_smoke(
            config=config,
            steps_per_epoch=steps_per_epoch,
            run_rung=partial(_run_smoke_subprocess, steps_per_epoch=steps_per_epoch),
        )
        _write_new(
            args.output,
            _smoke_receipt_bytes(
                smoke_result,
                config=config,
                source_revision=args.source_revision,
                source_tree_digest=args.source_tree_digest,
                manifest_sha256=manifest_sha256,
                steps_per_epoch=steps_per_epoch,
            ),
        )
        return

    if args.command != "train":
        raise RuntimeError("unreachable pooled-control command")
    smoke_receipt, smoke_bytes = _read_smoke_receipt(
        args.smoke,
        config=config,
        source_revision=args.source_revision,
        source_tree_digest=args.source_tree_digest,
        manifest_sha256=manifest_sha256,
        steps_per_epoch=steps_per_epoch,
    )
    run_authority = _current_run_authority(
        source_revision=args.source_revision,
        source_tree_digest=args.source_tree_digest,
        bands=bands,
        microbatch_size=smoke_receipt.selected_microbatch_size,
        steps_per_epoch=steps_per_epoch,
        device=device,
    )
    result = run_control_seed(
        config=config,
        seed=args.seed,
        bands=bands,
        checkpoint_directory=args.output_dir / f"seed-{args.seed:03d}" / "checkpoints",
        maximum_checkpoint_bytes=args.maximum_checkpoint_bytes,
        microbatch_size=smoke_receipt.selected_microbatch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        query_block=args.query_block,
        device=device,
        smoke_receipt=smoke_receipt,
        run_authority=run_authority,
    )
    seed_receipt_bytes = control_seed_receipt_bytes(
        result=result,
        config=config,
        bands=bands,
        smoke_receipt=smoke_receipt,
        smoke_sha256=hashlib.sha256(smoke_bytes).hexdigest(),
        run_authority=run_authority,
    )
    _write_new(
        args.output_dir / f"seed-{args.seed:03d}.receipt.json",
        seed_receipt_bytes,
    )


if __name__ == "__main__":
    main()
