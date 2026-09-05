#!/usr/bin/env python3
"""Run the local-only paired Qwen geometry-control experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from sfora.data import load_image_retrieval_examples, materialize_image  # noqa: E402
from sfora.qwen_geometry_control import (  # noqa: E402
    QwenGeometryProtocol,
    build_geometry_pooler,
    derive_epoch_batches,
    initialize_geometry_pooler,
    initialize_geometry_proxies,
    learning_rate_multiplier,
    optimizer_groups,
    pool_patch_tokens,
)
from sfora.token_set_proxy_anchor import proxy_anchor_loss  # noqa: E402


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "little") + value


def _tensor_bytes(value: Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    header = f"{cpu.dtype}:{tuple(cpu.shape)}".encode("ascii")
    payload = cpu.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return _frame(header) + _frame(payload)


def state_sha256(parameters: Iterable[Tensor]) -> str:
    """Hash ordered tensor state with dtype and shape framing."""

    digest = hashlib.sha256()
    count = 0
    for ordinal, parameter in enumerate(parameters):
        digest.update(_frame(str(ordinal).encode("ascii")))
        digest.update(_tensor_bytes(parameter))
        count += 1
    if count == 0:
        raise ValueError("parameter state is empty")
    return digest.hexdigest()


def _hash_value(digest: object, value: object) -> None:
    update = digest.update  # type: ignore[attr-defined]
    if isinstance(value, Tensor):
        update(b"tensor")
        update(_tensor_bytes(value))
    elif isinstance(value, Mapping):
        update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            update(_frame(repr(key).encode("utf-8")))
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        update(b"sequence")
        for item in value:
            _hash_value(digest, item)
    else:
        update(_frame(repr(value).encode("utf-8")))


def _optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, optimizer.state_dict())
    return digest.hexdigest()


@dataclass(frozen=True)
class GeometryStepEvidence:
    """Auditable evidence from exactly one successful logical-batch update."""

    update_index: int
    loss: float
    scores: Tensor
    score_gradients: Tensor
    parameter_gradients: tuple[Tensor, ...]
    gradient_norm: float
    maximum_score_disagreement: float
    learning_rate_multiplier: float
    updated_state_sha256: str
    optimizer_state_sha256: str
    sampled_parameter_values: int
    changed_sampled_parameter_values: int
    parameter_displacement: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class GeometryCheckpointAuthority:
    """Immutable byte and experiment identity for one training checkpoint."""

    basename: str
    byte_length: int
    sha256: str
    source_commit: str
    arm: str
    seed: int
    completed_updates: int


def _validate_qwen_parameter_roles(model: QwenVisionGeometryModel) -> None:
    """Require the sole frozen visual branch and every intended trainable role."""

    visual = tuple(model.visual.named_parameters(remove_duplicate=False))
    pooler = tuple(model.pooler.named_parameters(remove_duplicate=False))
    if not visual or not pooler:
        raise ValueError("Qwen geometry parameter role authority differs")
    if any(
        parameter.requires_grad != (not name.startswith("deepstack_merger_list."))
        for name, parameter in visual
    ) or any(not parameter.requires_grad for _, parameter in pooler):
        raise ValueError("Qwen geometry parameter role authority differs")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def write_geometry_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    proxies: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    source_commit: str,
    arm: str,
    seed: int,
    completed_updates: int,
    epoch_plan_digests: tuple[str, str, str],
) -> GeometryCheckpointAuthority:
    """Atomically publish one complete checkpoint and return its byte authority."""

    protocol = QwenGeometryProtocol()
    partial = path.with_name(path.name + ".partial")
    valid = (
        isinstance(path, Path)
        and path.parent.is_dir()
        and not path.parent.is_symlink()
        and not path.exists()
        and not partial.exists()
        and _valid_lower_hex(source_commit, 40)
        and arm in protocol.arms
        and type(seed) is int
        and seed in protocol.seeds
        and type(completed_updates) is int
        and 0 <= completed_updates <= protocol.optimizer_updates
        and type(epoch_plan_digests) is tuple
        and len(epoch_plan_digests) == protocol.epochs
        and all(_valid_lower_hex(value, 64) for value in epoch_plan_digests)
    )
    if not valid:
        raise ValueError("Qwen geometry checkpoint identity differs")
    payload = {
        "arm": arm,
        "completed_updates": completed_updates,
        "epoch_plan_digests": epoch_plan_digests,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "proxies": proxies.detach(),
        "schema": "sfora-qwen-geometry-checkpoint-v1",
        "seed": seed,
        "source_commit": source_commit,
        "torch_rng_state": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state()
    with partial.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    byte_length = path.stat().st_size
    return GeometryCheckpointAuthority(
        basename=path.name,
        byte_length=byte_length,
        sha256=_sha256_path(path),
        source_commit=source_commit,
        arm=arm,
        seed=seed,
        completed_updates=completed_updates,
    )


def restore_geometry_checkpoint(
    *,
    path: Path,
    authority: GeometryCheckpointAuthority,
    model: nn.Module,
    proxies: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    source_commit: str,
    arm: str,
    seed: int,
    epoch_plan_digests: tuple[str, str, str],
) -> int:
    """Authenticate and restore one checkpoint, returning its next update index."""

    expected_authority = (
        authority.basename == path.name
        and authority.byte_length == path.stat().st_size
        and authority.sha256 == _sha256_path(path)
        and authority.source_commit == source_commit
        and authority.arm == arm
        and authority.seed == seed
        and _valid_lower_hex(authority.sha256, 64)
    )
    if not expected_authority:
        raise ValueError("Qwen geometry checkpoint byte authority differs")
    value = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = {
        "arm",
        "completed_updates",
        "epoch_plan_digests",
        "model_state",
        "optimizer_state",
        "proxies",
        "schema",
        "seed",
        "source_commit",
        "torch_rng_state",
    }
    if isinstance(value, dict) and "cuda_rng_state" in value:
        expected_keys.add("cuda_rng_state")
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value["schema"] != "sfora-qwen-geometry-checkpoint-v1"
        or value["source_commit"] != source_commit
        or value["arm"] != arm
        or value["seed"] != seed
        or value["completed_updates"] != authority.completed_updates
        or value["epoch_plan_digests"] != epoch_plan_digests
    ):
        raise ValueError("Qwen geometry checkpoint payload authority differs")
    stored_proxies = value["proxies"]
    if not isinstance(stored_proxies, Tensor) or stored_proxies.shape != proxies.shape:
        raise ValueError("Qwen geometry proxy checkpoint differs")
    model.load_state_dict(value["model_state"], strict=True)
    proxies.data.copy_(stored_proxies.to(device=proxies.device, dtype=proxies.dtype))
    optimizer.load_state_dict(value["optimizer_state"])
    torch.random.set_rng_state(value["torch_rng_state"])
    if "cuda_rng_state" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state(value["cuda_rng_state"])
    return authority.completed_updates


class QwenVisionGeometryModel(nn.Module):
    """Expose only Qwen vision features followed by one registered pooling arm."""

    def __init__(
        self,
        *,
        model: nn.Module,
        processor: object,
        token_dimensions: int,
        arm: str,
    ) -> None:
        super().__init__()
        try:
            visual = model.model.visual  # type: ignore[attr-defined]
            language = model.model.language_model  # type: ignore[attr-defined]
            language_head = model.lm_head  # type: ignore[attr-defined]
        except AttributeError as error:
            raise ValueError("Qwen vision-only model structure differs") from error
        if not isinstance(visual, nn.Module):
            raise ValueError("Qwen visual tower differs")
        if not callable(getattr(processor, "image_processor", None)):
            raise ValueError("Qwen image processor differs")
        self.visual = visual
        self.visual.float()
        self.pooler = build_geometry_pooler(arm, token_dimensions=token_dimensions)
        self.__dict__["_image_processor"] = processor.image_processor
        for parameter in (*language.parameters(), *language_head.parameters()):
            parameter.requires_grad_(False)
        for parameter in self.visual.parameters():
            parameter.requires_grad_(True)
        deepstack = getattr(self.visual, "deepstack_merger_list", None)
        if isinstance(deepstack, nn.Module):
            for parameter in deepstack.parameters():
                parameter.requires_grad_(False)
        _validate_qwen_parameter_roles(self)

    def visual_tokens(self, images: Sequence[object]) -> Tensor:
        """Return the pre-pooler patch-token plane for an image batch."""

        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)) or not images:
            raise ValueError("Qwen image batch differs")
        raw = self._image_processor(images=list(images), return_tensors="pt")
        if not isinstance(raw, Mapping) or set(raw) != {"pixel_values", "image_grid_thw"}:
            raise ValueError("Qwen image processor output differs")
        pixel_values = raw["pixel_values"]
        image_grid_thw = raw["image_grid_thw"]
        if not isinstance(pixel_values, Tensor) or not isinstance(image_grid_thw, Tensor):
            raise ValueError("Qwen image processor tensors differ")
        try:
            device = next(self.visual.parameters()).device
        except StopIteration as error:
            raise ValueError("Qwen visual tower has no parameters") from error
        try:
            visual_dtype = next(self.visual.parameters()).dtype
            merge_size = int(self.visual.spatial_merge_size)  # type: ignore[attr-defined]
        except (AttributeError, StopIteration, TypeError, ValueError) as error:
            raise ValueError("Qwen visual execution authority differs") from error
        grid = image_grid_thw.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = self.visual(
                pixel_values.to(device=device, dtype=visual_dtype),
                grid_thw=grid,
                return_dict=True,
            )
        features = getattr(output, "pooler_output", None)
        split_sizes = tuple(int(value) for value in (grid.prod(-1) // merge_size**2).tolist())
        if (
            not isinstance(features, Tensor)
            or features.ndim != 2
            or len(split_sizes) != len(images)
            or any(size < 1 for size in split_sizes)
            or sum(split_sizes) != features.shape[0]
        ):
            raise ValueError("Qwen patch feature authority differs")
        patches = torch.split(features, split_sizes)
        if any(patch.shape != patches[0].shape for patch in patches):
            raise ValueError("Qwen patch feature authority differs")
        return torch.stack(patches).float()

    def forward(self, images: Sequence[object]) -> Tensor:
        patches = self.visual_tokens(images)
        descriptors, _ = pool_patch_tokens(self.pooler, patches)
        return descriptors


def _capture_trainable_snapshot(module: nn.Module) -> dict[str, Tensor]:
    snapshot = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    if not snapshot:
        raise ValueError("trainable snapshot is empty")
    return snapshot


def _tower_block_name(parameter_name: str) -> str:
    parts = parameter_name.split(".")
    if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
        return f"blocks.{parts[1]}"
    return parts[0]


def _summarize_tower_displacement(
    tower: nn.Module, baseline: Mapping[str, Tensor]
) -> dict[str, object]:
    """Stream exact FP32 tower displacement reductions by parameter block."""

    current = {
        name: parameter
        for name, parameter in tower.named_parameters(remove_duplicate=False)
        if parameter.requires_grad
    }
    if set(current) != set(baseline):
        raise ValueError("tower displacement parameter authority differs")
    totals: dict[str, list[float | int]] = {}
    total_delta_squared = 0.0
    total_before_squared = 0.0
    total_changed = 0
    total_elements = 0
    maximum_absolute_delta = 0.0
    chunk_size = 262_144
    for name in sorted(current):
        parameter = current[name].detach().reshape(-1)
        before = baseline[name].reshape(-1)
        if before.dtype != torch.float32 or before.numel() != parameter.numel():
            raise ValueError("tower displacement baseline authority differs")
        block = _tower_block_name(name)
        block_totals = totals.setdefault(block, [0.0, 0.0, 0, 0, 0.0])
        for start in range(0, parameter.numel(), chunk_size):
            stop = min(parameter.numel(), start + chunk_size)
            actual = parameter[start:stop].to(device="cpu", dtype=torch.float64)
            expected = before[start:stop].to(dtype=torch.float64)
            delta = actual - expected
            delta_squared = float(torch.dot(delta, delta))
            before_squared = float(torch.dot(expected, expected))
            changed = int((actual != expected).sum())
            maximum = float(delta.abs().max()) if delta.numel() else 0.0
            elements = stop - start
            total_delta_squared += delta_squared
            total_before_squared += before_squared
            total_changed += changed
            total_elements += elements
            maximum_absolute_delta = max(maximum_absolute_delta, maximum)
            block_totals[0] = float(block_totals[0]) + delta_squared
            block_totals[1] = float(block_totals[1]) + before_squared
            block_totals[2] = int(block_totals[2]) + changed
            block_totals[3] = int(block_totals[3]) + elements
            block_totals[4] = max(float(block_totals[4]), maximum)
    blocks = []
    for block, values in sorted(totals.items()):
        delta_squared, before_squared, changed, elements, maximum = values
        relative = math.sqrt(float(delta_squared)) / max(
            math.sqrt(float(before_squared)), 1.0e-12
        )
        blocks.append(
            {
                "block": block,
                "changed_elements": int(changed),
                "maximum_absolute_delta": float(maximum),
                "relative_l2": relative,
                "total_elements": int(elements),
            }
        )
    protocol = QwenGeometryProtocol()
    transformer = [value for value in blocks if str(value["block"]).startswith("blocks.")]
    moving = sum(
        float(value["relative_l2"]) >= protocol.tower_displacement_floor
        for value in transformer
    )
    return {
        "blocks": blocks,
        "changed_elements": total_changed,
        "maximum_absolute_delta": maximum_absolute_delta,
        "moving_block_fraction": moving / len(transformer) if transformer else 0.0,
        "moving_transformer_blocks": moving,
        "relative_l2": math.sqrt(total_delta_squared)
        / max(math.sqrt(total_before_squared), 1.0e-12),
        "total_elements": total_elements,
        "transformer_blocks": len(transformer),
    }


def _summarize_token_displacement(
    before: Tensor, unchanged_repeat: Tensor, after: Tensor
) -> dict[str, object]:
    if before.shape != unchanged_repeat.shape or before.shape != after.shape or not before.numel():
        raise ValueError("visual token displacement shape differs")
    before64 = before.detach().to(device="cpu", dtype=torch.float64)
    repeat64 = unchanged_repeat.detach().to(device="cpu", dtype=torch.float64)
    after64 = after.detach().to(device="cpu", dtype=torch.float64)
    denominator = max(float(torch.linalg.vector_norm(before64)), 1.0e-12)
    discrepancy = float(torch.linalg.vector_norm(repeat64 - before64)) / denominator
    relative = float(torch.linalg.vector_norm(after64 - before64)) / denominator
    threshold = max(1.0e-6, 10.0 * discrepancy)
    return {
        "passed": relative > threshold,
        "relative_l2": relative,
        "threshold": threshold,
        "unchanged_repeat_discrepancy": discrepancy,
    }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )


def _write_new(path: Path, raw: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise FileExistsError("Qwen geometry output already exists")
    with partial.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _rgb_224(image: object) -> np.ndarray:
    converted = cast(Image.Image, materialize_image(image)).convert("RGB")
    resized = converted.resize((224, 224), resample=Image.Resampling.BICUBIC)
    value = np.asarray(resized, dtype=np.uint8)
    if value.shape != (224, 224, 3):
        raise ValueError("Cars image differs from RGB 224 authority")
    return value.copy(order="C")


def _configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only real-model smoke boundary."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("phase", choices=("smoke", "train"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--arm", choices=QwenGeometryProtocol().arms, required=True)
    parser.add_argument("--seed", type=int, choices=QwenGeometryProtocol().seeds, required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument("--execute-train", action="store_true")
    values = parser.parse_args(argv)
    if (values.phase == "smoke") != bool(values.execute_smoke) or (
        values.phase == "train"
    ) != bool(values.execute_train):
        parser.error("phase requires its matching explicit execution flag")
    if values.phase == "smoke" and values.checkpoint_output is not None:
        parser.error("smoke cannot publish a training checkpoint")
    if values.phase == "train" and values.checkpoint_output is None:
        parser.error("train requires --checkpoint-output")
    if len(values.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in values.source_commit
    ):
        parser.error("source commit must be 40 lowercase hex")
    if (
        values.microbatch_size < 1
        or QwenGeometryProtocol().logical_batch_size % values.microbatch_size != 0
    ):
        parser.error("microbatch size must divide the logical batch")
    invalid_output = (
        values.output.exists()
        or not values.output.parent.is_dir()
        or values.output.parent.is_symlink()
    )
    if invalid_output:
        parser.error("output must be absent beneath an existing regular directory")
    if values.checkpoint_output is not None:
        invalid_checkpoint = (
            values.checkpoint_output == values.output
            or values.checkpoint_output.exists()
            or not values.checkpoint_output.parent.is_dir()
            or values.checkpoint_output.parent.is_symlink()
        )
        if invalid_checkpoint:
            parser.error("checkpoint output must be a distinct absent regular path")
    return values


def _load_real_geometry_model(args: argparse.Namespace) -> QwenVisionGeometryModel:
    from scripts.diagnose_saga_gb10_feasibility import (
        LoadedAuthority,
        TransformersFactory,
        load_qwen_adapter,
    )
    from sfora.saga_feasibility import load_fixture_authority, load_snapshot_authority

    snapshot = load_snapshot_authority(
        root=args.model_root, manifest_path=args.snapshot_manifest
    )
    fixture = load_fixture_authority(args.fixture)
    adapter = load_qwen_adapter(
        LoadedAuthority(snapshot=snapshot, fixture=fixture), factory=TransformersFactory()
    )
    model = QwenVisionGeometryModel(
        model=adapter._model,
        processor=adapter._processor,
        token_dimensions=adapter.pooler_token_dim,
        arm=args.arm,
    )
    device = next(model.visual.parameters()).device
    model.pooler.to(device)
    initialize_geometry_pooler(model.pooler, seed=args.seed)
    return model


def _initialize_geometry_training_state(
    args: argparse.Namespace, protocol: QwenGeometryProtocol
) -> tuple[QwenVisionGeometryModel, nn.Parameter, torch.optim.AdamW]:
    model = _load_real_geometry_model(args)
    device = next(model.visual.parameters()).device
    proxies = nn.Parameter(
        torch.empty(
            len(protocol.optimization_classes),
            protocol.embedding_dimensions,
            device=device,
            dtype=torch.float32,
        )
    )
    initialize_geometry_proxies(proxies, seed=args.seed)
    groups = optimizer_groups(
        tower=model.visual,
        pooler=model.pooler,
        proxies=proxies,
        allow_frozen=True,
    )
    for group in groups:
        group["base_lr"] = group["lr"]
        group["schedule_update"] = 0
    optimizer = torch.optim.AdamW(
        groups,
        betas=protocol.adamw_betas,
        eps=protocol.adamw_epsilon,
        foreach=protocol.optimizer_foreach,
    )
    return model, proxies, optimizer


def _run_smoke_trial(
    args: argparse.Namespace,
    examples: tuple[object, ...],
    plan: object,
    epoch_plan_digests: tuple[str, str, str],
    *,
    resume_after_two: bool,
) -> dict[str, object]:
    protocol = QwenGeometryProtocol()
    torch.manual_seed(args.seed)
    model, proxies, optimizer = _initialize_geometry_training_state(args, protocol)
    device = next(model.visual.parameters()).device
    model.train()
    baseline = _capture_trainable_snapshot(model.visual)
    fixed_image = (_rgb_224(examples[plan.batches[0][0]].image),)  # type: ignore[attr-defined]
    with torch.no_grad():
        initial_tokens = model.visual_tokens(fixed_image)
        repeated_initial_tokens = model.visual_tokens(fixed_image)
    started = perf_counter_ns()
    updates: list[dict[str, object]] = []
    update_elapsed_ns: list[int] = []
    with tempfile.TemporaryDirectory(prefix="sfora-qwen-smoke-") as directory:
        for update_index, batch in enumerate(plan.batches[:3]):  # type: ignore[attr-defined]
            if resume_after_two and update_index == 2:
                checkpoint_path = Path(directory) / "update-two.pt"
                authority = write_geometry_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    proxies=proxies,
                    optimizer=optimizer,
                    source_commit=args.source_commit,
                    arm=args.arm,
                    seed=args.seed,
                    completed_updates=2,
                    epoch_plan_digests=epoch_plan_digests,
                )
                del model, proxies, optimizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model, proxies, optimizer = _initialize_geometry_training_state(args, protocol)
                model.train()
                restored_update = restore_geometry_checkpoint(
                    path=checkpoint_path,
                    authority=authority,
                    model=model,
                    proxies=proxies,
                    optimizer=optimizer,
                    source_commit=args.source_commit,
                    arm=args.arm,
                    seed=args.seed,
                    epoch_plan_digests=epoch_plan_digests,
                )
                if restored_update != update_index:
                    raise RuntimeError("Qwen geometry smoke resume position differs")
                device = next(model.visual.parameters()).device
            images = tuple(_rgb_224(examples[index].image) for index in batch)  # type: ignore[attr-defined]
            labels = torch.tensor(
                [examples[index].label for index in batch],  # type: ignore[attr-defined]
                dtype=torch.int64,
                device=device,
            )
            update_started = perf_counter_ns()
            evidence = replayed_proxy_anchor_step(
                model=model,
                proxies=proxies,
                inputs=images,
                labels=labels,
                optimizer=optimizer,
                microbatch_size=args.microbatch_size,
                update_index=update_index,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            update_elapsed_ns.append(perf_counter_ns() - update_started)
            updates.append(
                {
                    "batch_sha256": plan.batch_digests[update_index],  # type: ignore[attr-defined]
                    "changed_sampled_parameter_values": evidence.changed_sampled_parameter_values,
                    "gradient_norm": evidence.gradient_norm,
                    "loss": evidence.loss,
                    "maximum_score_disagreement": evidence.maximum_score_disagreement,
                    "optimizer_state_sha256": evidence.optimizer_state_sha256,
                    "parameter_displacement": [
                        {"changed": changed, "role": role, "sampled": sampled}
                        for role, sampled, changed in evidence.parameter_displacement
                    ],
                    "sampled_parameter_values": evidence.sampled_parameter_values,
                    "state_sha256": evidence.updated_state_sha256,
                    "update": update_index,
                }
            )
    tower_displacement = _summarize_tower_displacement(model.visual, baseline)
    with torch.no_grad():
        final_tokens = model.visual_tokens(fixed_image)
    token_displacement = _summarize_token_displacement(
        initial_tokens, repeated_initial_tokens, final_tokens
    )
    if (
        float(tower_displacement["relative_l2"])
        < protocol.tower_displacement_floor
        or float(tower_displacement["moving_block_fraction"])
        < protocol.minimum_moving_block_fraction
        or not token_displacement["passed"]
    ):
        raise RuntimeError("Qwen geometry smoke displacement gate failed")
    return {
        "elapsed_ns": perf_counter_ns() - started,
        "resume_after_two": resume_after_two,
        "token_displacement": token_displacement,
        "tower_displacement": tower_displacement,
        "update_elapsed_ns": update_elapsed_ns,
        "updates": updates,
    }


def run_smoke(args: argparse.Namespace) -> bytes:
    """Run, restore, and exactly repeat three authenticated real-data updates."""

    protocol = QwenGeometryProtocol()
    _configure_determinism()
    examples = tuple(
        row
        for row in load_image_retrieval_examples(dataset_name="cars", split="train")
        if row.label in protocol.optimization_classes
    )
    members = {
        label: tuple(index for index, row in enumerate(examples) if row.label == label)
        for label in protocol.optimization_classes
    }
    plans = tuple(
        derive_epoch_batches(members, seed=args.seed, epoch=epoch)
        for epoch in range(protocol.epochs)
    )
    plan = plans[0]
    epoch_plan_digests = cast(tuple[str, str, str], tuple(value.digest for value in plans))
    trials = (
        _run_smoke_trial(
            args, examples, plan, epoch_plan_digests, resume_after_two=False
        ),
        _run_smoke_trial(args, examples, plan, epoch_plan_digests, resume_after_two=True),
    )
    comparable = tuple(
        {
            key: value
            for key, value in trial.items()
            if key not in {"elapsed_ns", "resume_after_two", "update_elapsed_ns"}
        }
        for trial in trials
    )
    if comparable[0] != comparable[1]:
        raise RuntimeError("restored Qwen geometry smoke evidence differs")
    payload = {
        "arm": args.arm,
        "claim_eligible": False,
        "elapsed_ns": sum(int(trial["elapsed_ns"]) for trial in trials),
        "language_forward_calls": 0,
        "microbatch_size": args.microbatch_size,
        "optimizer_updates": 3,
        "peak_cuda_bytes": (
            max(1, int(torch.cuda.max_memory_reserved())) if torch.cuda.is_available() else 1
        ),
        "peak_rss_bytes": max(1, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "protocol_batch_plan_sha256": plan.digest,
        "checkpoint_resume_equal": True,
        "restored_repetition_equal": True,
        "schema": "sfora-qwen-geometry-smoke-v2",
        "seed": args.seed,
        "source_commit": args.source_commit,
        "trials": trials,
    }
    return _canonical_bytes(payload)


def run_train(args: argparse.Namespace) -> bytes:
    """Run the complete frozen optimization schedule and publish its checkpoint."""

    protocol = QwenGeometryProtocol()
    _configure_determinism()
    torch.manual_seed(args.seed)
    examples = tuple(
        row
        for row in load_image_retrieval_examples(dataset_name="cars", split="train")
        if row.label in protocol.optimization_classes
    )
    members = {
        label: tuple(index for index, row in enumerate(examples) if row.label == label)
        for label in protocol.optimization_classes
    }
    plans = tuple(
        derive_epoch_batches(members, seed=args.seed, epoch=epoch)
        for epoch in range(protocol.epochs)
    )
    model = _load_real_geometry_model(args)
    device = next(model.visual.parameters()).device
    proxies = nn.Parameter(
        torch.empty(
            len(protocol.optimization_classes),
            protocol.embedding_dimensions,
            device=device,
            dtype=torch.float32,
        )
    )
    initialize_geometry_proxies(proxies, seed=args.seed)
    groups = optimizer_groups(
        tower=model.visual,
        pooler=model.pooler,
        proxies=proxies,
        allow_frozen=True,
    )
    for group in groups:
        group["base_lr"] = group["lr"]
        group["schedule_update"] = 0
    optimizer = torch.optim.AdamW(
        groups,
        betas=protocol.adamw_betas,
        eps=protocol.adamw_epsilon,
        foreach=protocol.optimizer_foreach,
    )
    model.train()
    started = perf_counter_ns()
    epochs: list[dict[str, object]] = []
    update_index = 0
    for epoch, plan in enumerate(plans):
        epoch_started = perf_counter_ns()
        losses: list[float] = []
        maximum_gradient_norm = 0.0
        maximum_disagreement = 0.0
        terminal_state_sha256 = ""
        terminal_optimizer_sha256 = ""
        sampled_values = 0
        changed_sampled_values = 0
        role_displacement: dict[str, list[int]] = {}
        for batch in plan.batches:
            images = tuple(_rgb_224(examples[index].image) for index in batch)
            labels = torch.tensor(
                [examples[index].label for index in batch],
                dtype=torch.int64,
                device=device,
            )
            evidence = replayed_proxy_anchor_step(
                model=model,
                proxies=proxies,
                inputs=images,
                labels=labels,
                optimizer=optimizer,
                microbatch_size=args.microbatch_size,
                update_index=update_index,
            )
            losses.append(evidence.loss)
            maximum_gradient_norm = max(maximum_gradient_norm, evidence.gradient_norm)
            maximum_disagreement = max(
                maximum_disagreement, evidence.maximum_score_disagreement
            )
            sampled_values += evidence.sampled_parameter_values
            changed_sampled_values += evidence.changed_sampled_parameter_values
            for role, sampled, changed in evidence.parameter_displacement:
                totals = role_displacement.setdefault(role, [0, 0])
                totals[0] += sampled
                totals[1] += changed
            terminal_state_sha256 = evidence.updated_state_sha256
            terminal_optimizer_sha256 = evidence.optimizer_state_sha256
            update_index += 1
        epochs.append(
            {
                "elapsed_ns": perf_counter_ns() - epoch_started,
                "epoch": epoch,
                "changed_sampled_parameter_values": changed_sampled_values,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "maximum_gradient_norm": maximum_gradient_norm,
                "maximum_score_disagreement": maximum_disagreement,
                "mean_loss": sum(losses) / len(losses),
                "optimizer_state_sha256": terminal_optimizer_sha256,
                "parameter_displacement": [
                    {"changed": totals[1], "role": role, "sampled": totals[0]}
                    for role, totals in sorted(role_displacement.items())
                ],
                "plan_sha256": plan.digest,
                "sampled_parameter_values": sampled_values,
                "state_sha256": terminal_state_sha256,
                "updates": len(losses),
            }
        )
    if update_index != protocol.optimizer_updates:
        raise RuntimeError("Qwen geometry training update count differs")
    checkpoint_path = args.checkpoint_output
    if not isinstance(checkpoint_path, Path):
        raise ValueError("Qwen geometry training checkpoint path differs")
    checkpoint = write_geometry_checkpoint(
        path=checkpoint_path,
        model=model,
        proxies=proxies,
        optimizer=optimizer,
        source_commit=args.source_commit,
        arm=args.arm,
        seed=args.seed,
        completed_updates=update_index,
        epoch_plan_digests=tuple(plan.digest for plan in plans),
    )
    payload = {
        "arm": args.arm,
        "checkpoint": asdict(checkpoint),
        "claim_eligible": False,
        "elapsed_ns": perf_counter_ns() - started,
        "epochs": epochs,
        "language_forward_calls": 0,
        "microbatch_size": args.microbatch_size,
        "optimizer_updates": update_index,
        "peak_cuda_bytes": (
            max(1, int(torch.cuda.max_memory_reserved())) if torch.cuda.is_available() else 1
        ),
        "peak_rss_bytes": max(1, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "schema": "sfora-qwen-geometry-train-v1",
        "seed": args.seed,
        "source_commit": args.source_commit,
    }
    return _canonical_bytes(payload)


def _validate_replay_model(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError("logical replay refuses training batch normalization")
        if isinstance(module, nn.Dropout) and module.training and module.p > 0.0:
            raise ValueError("logical replay refuses active dropout")


def _fixed_parameter_samples(parameters: tuple[Tensor, ...]) -> Tensor:
    return torch.cat(
        tuple(
            torch.stack(
                (
                    parameter.detach().reshape(-1)[0],
                    parameter.detach().reshape(-1)[parameter.numel() // 2],
                    parameter.detach().reshape(-1)[-1],
                )
            ).to(dtype=torch.float64)
            for parameter in parameters
        )
    )


def replayed_proxy_anchor_step(
    *,
    model: nn.Module,
    proxies: nn.Parameter,
    inputs: Tensor | Sequence[object],
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
    microbatch_size: int,
    update_index: int,
) -> GeometryStepEvidence:
    """Apply Proxy Anchor once while replaying a logical batch in bounded slices."""

    protocol = QwenGeometryProtocol()
    tensor_inputs = isinstance(inputs, Tensor)
    sequence_inputs = isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes))
    batch_size = (
        int(inputs.shape[0])
        if tensor_inputs and inputs.ndim >= 1
        else len(inputs)
        if sequence_inputs
        else 0
    )
    visual = getattr(model, "visual", None)
    pooler = getattr(model, "pooler", None)
    if isinstance(visual, nn.Module) and isinstance(pooler, nn.Module):
        role_parameters = (
            (
                "tower",
                tuple(
                    parameter for parameter in visual.parameters() if parameter.requires_grad
                ),
            ),
            (
                "pooler",
                tuple(parameter for parameter in pooler.parameters() if parameter.requires_grad),
            ),
            ("proxies", (proxies,)),
        )
    else:
        role_parameters = (
            (
                "model",
                tuple(parameter for parameter in model.parameters() if parameter.requires_grad),
            ),
            ("proxies", (proxies,)),
        )
    parameters = tuple(
        parameter for _, role_values in role_parameters for parameter in role_values
    )
    if (
        not (tensor_inputs or sequence_inputs)
        or (tensor_inputs and (not inputs.is_floating_point() or inputs.ndim < 2))
        or batch_size < 1
        or labels.shape != (batch_size,)
        or labels.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("logical batch inputs and labels differ")
    if tensor_inputs and not torch.isfinite(inputs).all().item():
        raise ValueError("logical batch inputs must be finite")
    if (
        type(microbatch_size) is not int
        or microbatch_size < 1
        or microbatch_size > batch_size
        or batch_size % microbatch_size != 0
    ):
        raise ValueError("microbatch size must be a positive logical-batch divisor")
    if type(update_index) is not int or not 0 <= update_index < protocol.optimizer_updates:
        raise ValueError("update index differs from the registered schedule")
    if any(group.get("schedule_update") != update_index for group in optimizer.param_groups):
        raise ValueError("optimizer schedule position differs")
    if any("base_lr" not in group for group in optimizer.param_groups):
        raise ValueError("optimizer base learning-rate authority is absent")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("every logical replay parameter must be trainable")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("logical replay parameters are duplicated")
    if not torch.isfinite(proxies).all().item() or bool(
        (torch.linalg.vector_norm(proxies, dim=-1) <= 0).any()
    ):
        raise ValueError("class proxies must be finite and nonzero")
    if bool((labels < 0).any()) or bool((labels >= proxies.shape[0]).any()):
        raise ValueError("logical batch labels exceed the proxy authority")
    _validate_replay_model(model)

    optimizer.zero_grad(set_to_none=True)
    score_chunks: list[Tensor] = []
    with torch.no_grad():
        normalized_proxies = F.normalize(proxies, dim=-1)
        for start in range(0, batch_size, microbatch_size):
            descriptors = model(inputs[start : start + microbatch_size])
            if descriptors.ndim != 2 or descriptors.shape[1] != proxies.shape[1]:
                raise ValueError("descriptor and proxy shapes differ")
            score_chunks.append(descriptors @ normalized_proxies.T)
    scores = torch.cat(score_chunks)
    if not torch.isfinite(scores).all().item():
        raise ValueError("logical batch scores must be finite")
    score_leaf = scores.detach().requires_grad_(True)
    loss = proxy_anchor_loss(
        score_leaf,
        labels,
        alpha=protocol.proxy_anchor_alpha,
        delta=protocol.proxy_anchor_delta,
    )
    (score_gradients,) = torch.autograd.grad(loss, score_leaf)
    if not torch.isfinite(loss).item() or not torch.isfinite(score_gradients).all().item():
        raise ValueError("Proxy Anchor loss and cotangent must be finite")

    maximum_disagreement = 0.0
    for start in range(0, batch_size, microbatch_size):
        stop = start + microbatch_size
        replay_scores = model(inputs[start:stop]) @ F.normalize(proxies, dim=-1).T
        disagreement = float((replay_scores.detach() - scores[start:stop]).abs().max())
        maximum_disagreement = max(maximum_disagreement, disagreement)
        if not math.isfinite(disagreement) or disagreement > 1.0e-10:
            raise RuntimeError("logical replay score disagreement exceeds tolerance")
        torch.autograd.backward(replay_scores, score_gradients[start:stop])

    gradients: list[Tensor] = []
    for parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError("every logical replay parameter must receive a finite gradient")
        gradients.append(parameter.grad.detach().clone())
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, protocol.gradient_clip_norm, error_if_nonfinite=True
    )
    if not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0.0:
        raise RuntimeError("logical replay gradient norm must be finite and positive")

    multiplier = learning_rate_multiplier(update_index)
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier
    before_by_role = tuple(
        (role, _fixed_parameter_samples(role_values))
        for role, role_values in role_parameters
    )
    before_samples = torch.cat(tuple(values for _, values in before_by_role))
    optimizer.step()
    after_by_role = tuple(
        (role, _fixed_parameter_samples(role_values))
        for role, role_values in role_parameters
    )
    after_samples = torch.cat(tuple(values for _, values in after_by_role))
    changed_sampled_values = int((before_samples != after_samples).sum())
    displacement = tuple(
        (role, before.numel(), int((before != after).sum()))
        for (role, before), (after_role, after) in zip(
            before_by_role, after_by_role, strict=True
        )
        if role == after_role
    )
    for group in optimizer.param_groups:
        group["schedule_update"] = update_index + 1

    return GeometryStepEvidence(
        update_index=update_index,
        loss=float(loss.detach()),
        scores=scores.detach(),
        score_gradients=score_gradients.detach(),
        parameter_gradients=tuple(gradients),
        gradient_norm=float(gradient_norm),
        maximum_score_disagreement=maximum_disagreement,
        learning_rate_multiplier=multiplier,
        updated_state_sha256=state_sha256(parameters),
        optimizer_state_sha256=_optimizer_state_sha256(optimizer),
        sampled_parameter_values=before_samples.numel(),
        changed_sampled_parameter_values=changed_sampled_values,
        parameter_displacement=displacement,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw = run_smoke(args) if args.phase == "smoke" else run_train(args)
    _write_new(args.output, raw)
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
