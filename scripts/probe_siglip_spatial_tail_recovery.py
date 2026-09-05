"""Local-only SigLIP spatial-tail recovery probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

_CUDA_MEMORY_CAP_BYTES = 96 * 1024**3


def _enforce_cuda_memory_cap(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.memory_reserved(device) > _CUDA_MEMORY_CAP_BYTES:
        raise RuntimeError("spatial tail CUDA memory cap exceeded")


@dataclass(frozen=True, slots=True)
class SpatialTailFitInputs:
    """CPU-resident source/target token fields and teacher descriptors."""

    source_tokens: torch.Tensor
    target_tokens: torch.Tensor
    teacher_descriptors: torch.Tensor
    cache_bytes: int


@dataclass(frozen=True, slots=True)
class SpatialTailFitEvidence:
    """One final trained arm and its fixed-budget optimization evidence."""

    model: nn.Module
    initial_loss: float
    final_loss: float
    final_losses: tuple[float, ...]
    index_sha256: str


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the strict optimization-only spatial-tail command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-binding", type=_absolute_path, required=True)
    parser.add_argument("--control-binding-sha256", type=_lower_sha256, required=True)
    parser.add_argument("--checkpoint-seed17", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest", type=_absolute_path, required=True)
    parser.add_argument("--optimization-manifest-sha256", type=_lower_sha256, required=True)
    parser.add_argument("--optimization-image-root", type=_absolute_path, required=True)
    parser.add_argument("--artifact", type=_absolute_path, required=True)
    parser.add_argument("--result", type=_absolute_path, required=True)
    parser.add_argument("--execute-spatial-tail", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


def stream_spatial_tail_fit_inputs(
    vision_model: nn.Module,
    projection: nn.Linear,
    pixel_batches: Iterable[torch.Tensor],
    *,
    source_depth: int,
    target_depth: int,
    device: torch.device,
) -> SpatialTailFitInputs:
    """Extract exact frozen source/target token fields in one teacher pass."""

    post_layernorm = getattr(vision_model, "post_layernorm", None)
    head = getattr(vision_model, "head", None)
    if (
        not isinstance(vision_model, nn.Module)
        or vision_model.training
        or not isinstance(post_layernorm, nn.Module)
        or post_layernorm.training
        or not isinstance(head, nn.Module)
        or head.training
        or not isinstance(projection, nn.Linear)
        or projection.training
        or projection.bias is not None
        or type(source_depth) is not int
        or type(target_depth) is not int
        or not 0 < source_depth < target_depth
        or type(device) is not torch.device
        or device.type not in {"cpu", "cuda"}
    ):
        raise ValueError("spatial tail model authority differs")
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    descriptors: list[torch.Tensor] = []
    with torch.inference_mode():
        for pixels in pixel_batches:
            if (
                type(pixels) is not torch.Tensor
                or pixels.ndim != 4
                or pixels.shape[0] < 1
                or not pixels.is_floating_point()
                or not bool(torch.isfinite(pixels).all())
            ):
                raise ValueError("spatial tail pixel authority differs")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = vision_model(
                    pixel_values=pixels.to(device),
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = getattr(output, "hidden_states", None)
                pooler_output = getattr(output, "pooler_output", None)
                if (
                    type(hidden_states) not in {tuple, list}
                    or len(hidden_states) <= target_depth
                    or type(pooler_output) is not torch.Tensor
                    or pooler_output.ndim != 2
                    or pooler_output.shape[0] != pixels.shape[0]
                    or pooler_output.shape[1] != projection.in_features
                ):
                    raise ValueError("spatial tail hidden-state authority differs")
                source = hidden_states[source_depth]
                target = hidden_states[target_depth]
                if (
                    type(source) is not torch.Tensor
                    or type(target) is not torch.Tensor
                    or source.shape != target.shape
                    or source.ndim != 3
                    or source.shape[0] != pixels.shape[0]
                    or source.shape[2] != projection.in_features
                    or not bool(torch.isfinite(source).all())
                    or not bool(torch.isfinite(target).all())
                ):
                    raise ValueError("spatial tail hidden-state authority differs")
                pooled = head(post_layernorm(target))
                if target_depth == len(hidden_states) - 1 and not torch.equal(
                    pooled.float(), pooler_output.float()
                ):
                    raise ValueError("spatial tail final-depth identity differs")
                teacher = F.normalize(projection(pooler_output.float()), dim=1)
                if not bool(torch.isfinite(teacher).all()):
                    raise ValueError("spatial tail teacher descriptor differs")
            sources.append(source.half().cpu().contiguous())
            targets.append(target.half().cpu().contiguous())
            descriptors.append(teacher.float().cpu().contiguous())
            _enforce_cuda_memory_cap(device)
    if not sources:
        raise ValueError("spatial tail stream is empty")
    source_tokens = torch.cat(sources).contiguous()
    target_tokens = torch.cat(targets).contiguous()
    teacher_descriptors = torch.cat(descriptors).contiguous()
    values = (source_tokens, target_tokens, teacher_descriptors)
    return SpatialTailFitInputs(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        teacher_descriptors=teacher_descriptors,
        cache_bytes=sum(value.numel() * value.element_size() for value in values),
    )


def residual_channel_scale(
    source_tokens: torch.Tensor, target_tokens: torch.Tensor
) -> torch.Tensor:
    """Return fitting-only per-channel RMS with the registered nonzero floor."""

    if (
        type(source_tokens) is not torch.Tensor
        or type(target_tokens) is not torch.Tensor
        or source_tokens.device.type != "cpu"
        or target_tokens.device.type != "cpu"
        or source_tokens.dtype != torch.float16
        or target_tokens.dtype != torch.float16
        or source_tokens.ndim != 3
        or source_tokens.shape != target_tokens.shape
        or source_tokens.shape[0] < 2
        or source_tokens.shape[1] < 1
        or source_tokens.shape[2] < 1
        or not source_tokens.is_contiguous()
        or not target_tokens.is_contiguous()
        or not bool(torch.isfinite(source_tokens).all())
        or not bool(torch.isfinite(target_tokens).all())
    ):
        raise ValueError("spatial tail residual scale authority differs")
    difference = target_tokens.double() - source_tokens.double()
    rms = torch.sqrt(torch.mean(difference.square(), dim=(0, 1)))
    nonzero = rms[rms > 0]
    if nonzero.numel() == 0 or not bool(torch.isfinite(rms).all()):
        raise ValueError("spatial tail residual scale authority differs")
    floor = torch.median(nonzero) * 1.0e-3
    return torch.maximum(rms, floor).float().contiguous()


class FrozenTeacherReadout(nn.Module):
    """Frozen teacher LayerNorm, native pooler, and projection."""

    def __init__(self, layernorm: nn.Module, head: nn.Module, projection: nn.Linear) -> None:
        super().__init__()
        self.layernorm = copy.deepcopy(layernorm)
        self.head = copy.deepcopy(head)
        self.projection = copy.deepcopy(projection)
        self.requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.head(self.layernorm(tokens))), dim=1)


class _ResidualMlp(nn.Module):
    def __init__(self, dimensions: int, bottleneck_width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimensions)
        self.input = nn.Linear(dimensions, bottleneck_width)
        self.output = nn.Linear(bottleneck_width, dimensions)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, tokens + self.output(F.gelu(self.input(self.norm(tokens)))))


class TokenwiseTailControl(nn.Module):
    """Two residual tokenwise updates with no spatial communication."""

    def __init__(self, dimensions: int, *, bottleneck_width: int = 128) -> None:
        super().__init__()
        self.first = _ResidualMlp(dimensions, bottleneck_width)
        self.second = _ResidualMlp(dimensions, bottleneck_width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.second(self.first(tokens)))


class LatentInteractionTail(nn.Module):
    """Low-rank latent read/interact/write spatial residual update."""

    def __init__(
        self,
        dimensions: int,
        *,
        latent_width: int = 128,
        latent_count: int = 8,
        heads: int = 4,
    ) -> None:
        super().__init__()
        if latent_width % heads:
            raise ValueError("spatial tail latent width differs")
        self.input_norm = nn.LayerNorm(dimensions)
        self.input_projection = nn.Linear(dimensions, latent_width)
        self.latents = nn.Parameter(torch.empty(latent_count, latent_width))
        self.read_norm = nn.LayerNorm(latent_width)
        self.read = nn.MultiheadAttention(latent_width, heads, dropout=0.0, batch_first=True)
        self.latent_norm = nn.LayerNorm(latent_width)
        self.latent_attention = nn.MultiheadAttention(
            latent_width, heads, dropout=0.0, batch_first=True
        )
        self.latent_mlp_norm = nn.LayerNorm(latent_width)
        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_width, latent_width * 2),
            nn.GELU(),
            nn.Linear(latent_width * 2, latent_width),
        )
        self.write_norm = nn.LayerNorm(latent_width)
        self.write = nn.MultiheadAttention(latent_width, heads, dropout=0.0, batch_first=True)
        self.output = nn.Linear(latent_width, dimensions)
        nn.init.normal_(self.latents, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        spatial = self.input_projection(self.input_norm(tokens))
        latents = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        normalized_spatial = self.read_norm(spatial)
        read, _ = self.read(
            latents, normalized_spatial, normalized_spatial, need_weights=False
        )
        latents = latents + read
        normalized = self.latent_norm(latents)
        interacted, _ = self.latent_attention(
            normalized, normalized, normalized, need_weights=False
        )
        latents = latents + interacted
        latents = latents + self.latent_mlp(self.latent_mlp_norm(latents))
        written, _ = self.write(
            spatial, self.write_norm(latents), self.write_norm(latents), need_weights=False
        )
        return cast(torch.Tensor, tokens + self.output(written))


def _write_spatial_tail_artifact(
    path: Path,
    control: TokenwiseTailControl,
    treatment: LatentInteractionTail,
    readout: FrozenTeacherReadout,
) -> str:
    """Exclusively seal both arms and the frozen teacher readout."""

    from safetensors.torch import load_file, save_file

    if (
        not isinstance(path, Path)
        or not isinstance(control, TokenwiseTailControl)
        or control.training
        or not isinstance(treatment, LatentInteractionTail)
        or treatment.training
        or not isinstance(readout, FrozenTeacherReadout)
        or readout.training
        or path.exists()
        or path.is_symlink()
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        raise ValueError("spatial tail artifact authority differs")
    tensors: dict[str, torch.Tensor] = {}
    for prefix, module in (
        ("control", control),
        ("treatment", treatment),
        ("readout", readout),
    ):
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError("spatial tail artifact authority differs")
            tensors[f"{prefix}.{name}"] = tensor
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        dict(sorted(tensors.items())),
        str(partial),
        metadata={"schema": "sfora-siglip-spatial-tail-artifact-v1"},
    )
    restored = load_file(str(partial), device="cpu")
    if set(restored) != set(tensors) or any(
        not torch.equal(restored[name], value) for name, value in tensors.items()
    ):
        partial.unlink(missing_ok=True)
        raise ValueError("spatial tail artifact replay differs")
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    partial.replace(path)
    return digest


def _load_spatial_tail_artifact(
    path: Path,
    control_template: TokenwiseTailControl,
    treatment_template: LatentInteractionTail,
    readout_template: FrozenTeacherReadout,
) -> tuple[TokenwiseTailControl, LatentInteractionTail, FrozenTeacherReadout]:
    """Restore a complete sealed artifact into fresh module copies."""

    from safetensors.torch import load_file

    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("spatial tail artifact authority differs")
    values = load_file(str(path), device="cpu")
    modules = (
        ("control", copy.deepcopy(control_template).cpu()),
        ("treatment", copy.deepcopy(treatment_template).cpu()),
        ("readout", copy.deepcopy(readout_template).cpu()),
    )
    expected = {
        f"{prefix}.{name}"
        for prefix, module in modules
        for name in module.state_dict()
    }
    if set(values) != expected:
        raise ValueError("spatial tail artifact schema differs")
    restored: list[nn.Module] = []
    for prefix, module in modules:
        state = {
            name: values[f"{prefix}.{name}"]
            for name in module.state_dict()
        }
        module.load_state_dict(state, strict=True)
        module.eval()
        restored.append(module)
    return cast(
        tuple[TokenwiseTailControl, LatentInteractionTail, FrozenTeacherReadout],
        tuple(restored),
    )


def spatial_tail_loss(
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    predicted_descriptors: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    channel_scale: torch.Tensor,
) -> torch.Tensor:
    """Compute normalized token residual MSE plus descriptor cosine loss."""

    if (
        type(predicted_tokens) is not torch.Tensor
        or type(target_tokens) is not torch.Tensor
        or predicted_tokens.shape != target_tokens.shape
        or predicted_tokens.ndim != 3
        or type(predicted_descriptors) is not torch.Tensor
        or type(teacher_descriptors) is not torch.Tensor
        or predicted_descriptors.shape != teacher_descriptors.shape
        or predicted_descriptors.ndim != 2
        or predicted_descriptors.shape[0] != predicted_tokens.shape[0]
        or type(channel_scale) is not torch.Tensor
        or channel_scale.ndim != 1
        or channel_scale.shape[0] != predicted_tokens.shape[2]
        or not bool(torch.isfinite(predicted_tokens).all())
        or not bool(torch.isfinite(target_tokens).all())
        or not bool(torch.isfinite(predicted_descriptors).all())
        or not bool(torch.isfinite(teacher_descriptors).all())
        or not bool(torch.isfinite(channel_scale).all())
        or bool((channel_scale <= 0).any())
    ):
        raise ValueError("spatial tail loss authority differs")
    token_loss = torch.mean(((predicted_tokens - target_tokens) / channel_scale) ** 2)
    descriptor_loss = (
        1.0 - torch.sum(predicted_descriptors * teacher_descriptors, dim=1)
    ).mean()
    return token_loss + descriptor_loss


def _mean_spatial_tail_loss(
    model: nn.Module,
    readout: FrozenTeacherReadout,
    source: torch.Tensor,
    target: torch.Tensor,
    teacher: torch.Tensor,
    scale: torch.Tensor,
) -> float:
    weighted: list[float] = []
    with torch.no_grad():
        for start in range(0, source.shape[0], 64):
            stop = min(start + 64, source.shape[0])
            with torch.autocast(
                device_type=source.device.type,
                dtype=torch.bfloat16,
                enabled=source.device.type == "cuda",
            ):
                predicted = model(source[start:stop])
                loss = spatial_tail_loss(
                    predicted,
                    target[start:stop],
                    readout(predicted),
                    teacher[start:stop],
                    scale,
                )
            weighted.append(float(loss) * (stop - start))
    return math.fsum(weighted) / source.shape[0]


def fit_spatial_tail_arm(
    model: nn.Module,
    readout: FrozenTeacherReadout,
    source_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    teacher_descriptors: torch.Tensor,
    channel_scale: torch.Tensor,
    *,
    device: torch.device,
    updates: int = 4_000,
    batch_size: int = 64,
) -> SpatialTailFitEvidence:
    """Train one arm with the fixed matched index stream and objective."""

    if (
        not isinstance(model, (TokenwiseTailControl, LatentInteractionTail))
        or not isinstance(readout, FrozenTeacherReadout)
        or readout.training
        or type(source_tokens) is not torch.Tensor
        or type(target_tokens) is not torch.Tensor
        or source_tokens.device.type != "cpu"
        or target_tokens.device.type != "cpu"
        or source_tokens.dtype != torch.float16
        or target_tokens.dtype != torch.float16
        or source_tokens.shape != target_tokens.shape
        or source_tokens.ndim != 3
        or source_tokens.shape[0] < 2
        or not source_tokens.is_contiguous()
        or not target_tokens.is_contiguous()
        or type(teacher_descriptors) is not torch.Tensor
        or teacher_descriptors.device.type != "cpu"
        or teacher_descriptors.dtype != torch.float32
        or teacher_descriptors.ndim != 2
        or teacher_descriptors.shape[0] != source_tokens.shape[0]
        or not teacher_descriptors.is_contiguous()
        or type(channel_scale) is not torch.Tensor
        or channel_scale.device.type != "cpu"
        or channel_scale.dtype != torch.float32
        or channel_scale.shape != (source_tokens.shape[2],)
        or type(device) is not torch.device
        or device.type not in {"cpu", "cuda"}
        or type(updates) is not int
        or updates < 1
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("spatial tail fit authority differs")
    model = model.to(device).train()
    readout = readout.to(device).eval()
    source = source_tokens.to(device=device, dtype=torch.float32)
    target = target_tokens.to(device=device, dtype=torch.float32)
    teacher = teacher_descriptors.to(device)
    scale = channel_scale.to(device)
    _enforce_cuda_memory_cap(device)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=3.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.01,
        foreach=False,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260905)
    pending = torch.empty(0, dtype=torch.int64)
    index_digest = hashlib.sha256(b"sfora-spatial-tail-index-v1\0")
    initial_loss = _mean_spatial_tail_loss(model.eval(), readout, source, target, teacher, scale)
    model.train()
    losses: list[float] = []
    warmup = 200 if updates == 4_000 else max(1, updates // 20)
    for update in range(1, updates + 1):
        while pending.numel() < batch_size:
            pending = torch.cat(
                (pending, torch.randperm(source_tokens.shape[0], generator=generator))
            )
        indexes, pending = pending[:batch_size], pending[batch_size:]
        index_digest.update(indexes.numpy().tobytes(order="C"))
        if update <= warmup:
            learning_rate = 3.0e-4 * update / warmup
        else:
            progress = (update - warmup) / (updates - warmup)
            learning_rate = 3.0e-5 + 0.5 * (3.0e-4 - 3.0e-5) * (
                1.0 + math.cos(math.pi * progress)
            )
        optimizer.param_groups[0]["lr"] = learning_rate
        selected = indexes.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            predicted = model(source[selected])
            loss = spatial_tail_loss(
                predicted,
                target[selected],
                readout(predicted),
                teacher[selected],
                scale,
            )
        if not bool(torch.isfinite(loss)):
            raise ValueError("spatial tail fit loss is nonfinite")
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        _enforce_cuda_memory_cap(device)
        if update > max(0, updates - 200):
            losses.append(float(loss.detach()))
    model.eval()
    final_loss = _mean_spatial_tail_loss(model, readout, source, target, teacher, scale)
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise ValueError("spatial tail fit loss is nonfinite")
    return SpatialTailFitEvidence(
        model=model.cpu(),
        initial_loss=initial_loss,
        final_loss=final_loss,
        final_losses=tuple(losses),
        index_sha256=index_digest.hexdigest(),
    )


def _apply_spatial_tail_arm(
    model: nn.Module,
    readout: FrozenTeacherReadout,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Apply one sealed arm and frozen readout in bounded batches."""

    if (
        not isinstance(model, (TokenwiseTailControl, LatentInteractionTail))
        or model.training
        or not isinstance(readout, FrozenTeacherReadout)
        or readout.training
        or tokens.device.type != "cpu"
        or tokens.dtype != torch.float16
        or tokens.ndim != 3
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("spatial tail application authority differs")
    model = model.to(device)
    readout = readout.to(device)
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], batch_size):
            source = tokens[start : start + batch_size].to(device).float()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                descriptors = readout(model(source))
            batches.append(descriptors.float().cpu().contiguous())
            _enforce_cuda_memory_cap(device)
    return torch.cat(batches).contiguous()


def _retrieval_payload(evidence: object) -> dict[str, object]:
    correct = getattr(evidence, "correct", None)
    average_precisions = getattr(evidence, "average_precisions", None)
    if type(correct) is not tuple or type(average_precisions) is not tuple:
        raise ValueError("spatial tail retrieval evidence differs")
    return {"hits": list(correct), "average_precision": list(average_precisions)}


def main(argv: list[str] | None = None) -> int:
    """Train on fitting classes, seal both arms, then score internal development."""

    from diagnose_siglip_rsta_stage_a import (
        _load_model_state_checkpoint,
        _load_optimization_manifest,
        _parse_control_binding,
        _stage_a_transforms,
        configure_stage_a_determinism,
        load_stage_a_checkpoint_model,
        load_stage_a_siglip_runtime,
    )
    from diagnose_siglip_rsta_stage_a import _read_regular as read_rsta_regular
    from PIL import Image

    from sfora.siglip_spatial_tail_recovery import (
        build_spatial_tail_result,
        spatial_retrieval_evidence,
        spatial_tail_class_split,
    )

    arguments = parse_args(argv)
    for output in (arguments.result, arguments.artifact):
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
    binding_raw = read_rsta_regular(arguments.control_binding, role="spatial tail binding")
    if hashlib.sha256(binding_raw).hexdigest() != arguments.control_binding_sha256:
        raise ValueError("spatial tail binding digest differs")
    binding = _parse_control_binding(binding_raw)
    if (
        binding.control_complete is not True
        or binding.optimization_manifest_sha256 != arguments.optimization_manifest_sha256
        or tuple(checkpoint.seed for checkpoint in binding.checkpoints) != (17, 29, 43)
    ):
        raise ValueError("spatial tail binding authority differs")
    seed17 = binding.checkpoints[0]
    checkpoint = _load_model_state_checkpoint(arguments.checkpoint_seed17, seed17, binding)
    optimization_ids, optimization_labels, optimization_paths = _load_optimization_manifest(
        arguments.optimization_manifest,
        arguments.optimization_manifest_sha256,
        binding,
        arguments.optimization_image_root,
    )
    fit_labels, development_labels = spatial_tail_class_split(tuple(range(49)))
    fit_indexes = tuple(
        index for index, label in enumerate(optimization_labels) if label in fit_labels
    )
    development_indexes = tuple(
        index for index, label in enumerate(optimization_labels) if label in development_labels
    )
    if not fit_indexes or not development_indexes:
        raise ValueError("spatial tail class partition differs")

    configure_stage_a_determinism()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("spatial tail recovery requires CUDA bf16")
    device = torch.device("cuda")
    runtime = load_stage_a_siglip_runtime()
    model = load_stage_a_checkpoint_model(
        checkpoint,
        model_factory=runtime.model_factory,
        device=device,
    )
    runtime.disable_checkpointing(model)
    model.eval()
    vision_model = model.tower.vision_model
    projection = model.projection
    _graph_transform, evaluation_transform = _stage_a_transforms(runtime.processor)

    def pixel_batches(paths: tuple[Path, ...]) -> Iterable[torch.Tensor]:
        for start in range(0, len(paths), 32):
            tensors: list[torch.Tensor] = []
            for path in paths[start : start + 32]:
                with Image.open(path) as image:
                    tensor = evaluation_transform(image)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError("spatial tail image transform differs")
                tensors.append(tensor)
            yield torch.stack(tensors)

    fit_paths = tuple(optimization_paths[index] for index in fit_indexes)
    fitting = stream_spatial_tail_fit_inputs(
        vision_model,
        projection,
        pixel_batches(fit_paths),
        source_depth=18,
        target_depth=27,
        device=device,
    )
    scale = residual_channel_scale(fitting.source_tokens, fitting.target_tokens)
    readout = FrozenTeacherReadout(
        vision_model.post_layernorm,
        vision_model.head,
        projection,
    ).eval()
    torch.manual_seed(20260905)
    control = TokenwiseTailControl(1152).eval()
    control_fit = fit_spatial_tail_arm(
        control,
        readout,
        fitting.source_tokens,
        fitting.target_tokens,
        fitting.teacher_descriptors,
        scale,
        device=device,
    )
    torch.cuda.empty_cache()
    torch.manual_seed(20260905)
    treatment = LatentInteractionTail(1152).eval()
    treatment_fit = fit_spatial_tail_arm(
        treatment,
        readout,
        fitting.source_tokens,
        fitting.target_tokens,
        fitting.teacher_descriptors,
        scale,
        device=device,
    )
    if control_fit.index_sha256 != treatment_fit.index_sha256:
        raise ValueError("spatial tail matched index stream differs")
    artifact_sha256 = _write_spatial_tail_artifact(
        arguments.artifact,
        cast(TokenwiseTailControl, control_fit.model),
        cast(LatentInteractionTail, treatment_fit.model),
        readout,
    )
    restored_control, restored_treatment, restored_readout = _load_spatial_tail_artifact(
        arguments.artifact,
        cast(TokenwiseTailControl, control_fit.model),
        cast(LatentInteractionTail, treatment_fit.model),
        readout,
    )
    del fitting
    torch.cuda.empty_cache()

    development_paths = tuple(optimization_paths[index] for index in development_indexes)
    development = stream_spatial_tail_fit_inputs(
        vision_model,
        projection,
        pixel_batches(development_paths),
        source_depth=18,
        target_depth=27,
        device=device,
    )
    teacher = development.teacher_descriptors
    # The registered baseline is the unmodified depth-18 field through the frozen readout.
    restored_readout = restored_readout.to(device)
    baseline_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, development.source_tokens.shape[0], 32):
            source = development.source_tokens[start : start + 32].to(device).float()
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                baseline_batches.append(restored_readout(source).float().cpu())
    baseline = torch.cat(baseline_batches).contiguous()
    control_descriptors = _apply_spatial_tail_arm(
        restored_control,
        restored_readout.cpu(),
        development.source_tokens,
        device=device,
    )
    treatment_descriptors = _apply_spatial_tail_arm(
        restored_treatment,
        restored_readout.cpu(),
        development.source_tokens,
        device=device,
    )
    ids = tuple(optimization_ids[index] for index in development_indexes)
    labels = tuple(optimization_labels[index] for index in development_indexes)

    def evidence(query: torch.Tensor, gallery: torch.Tensor) -> dict[str, object]:
        return _retrieval_payload(
            spatial_retrieval_evidence(
                query,
                gallery,
                query_ids=ids,
                gallery_ids=ids,
                query_labels=labels,
                gallery_labels=labels,
            )
        )

    cells: dict[str, dict[str, object]] = {
        "baseline": {"self": evidence(baseline, baseline), "cross": evidence(baseline, teacher)},
        "tokenwise-control": {
            "self": evidence(control_descriptors, control_descriptors),
            "cross": evidence(control_descriptors, teacher),
        },
        "latent-interaction": {
            "self": evidence(treatment_descriptors, treatment_descriptors),
            "cross": evidence(treatment_descriptors, teacher),
        },
        "teacher": {"self": evidence(teacher, teacher), "cross": evidence(teacher, teacher)},
    }
    raw = build_spatial_tail_result(
        checkpoint_sha256=seed17.sha256,
        optimization_manifest_sha256=arguments.optimization_manifest_sha256,
        artifact_sha256=artifact_sha256,
        fit_labels=tuple(sorted(fit_labels)),
        development_labels=tuple(sorted(development_labels)),
        cells=cells,
        training={
            "updates": 4_000,
            "tokenwise_initial_loss": control_fit.initial_loss,
            "tokenwise_final_loss": control_fit.final_loss,
            "interaction_initial_loss": treatment_fit.initial_loss,
            "interaction_final_loss": treatment_fit.final_loss,
        },
    )
    value = json.loads(raw)
    if not isinstance(value, Mapping) or value.get("claim_eligible") is not False:
        raise ValueError("spatial tail result binding differs")
    partial = arguments.result.with_name(f"{arguments.result.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    arguments.result.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(raw)
    partial.replace(arguments.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
