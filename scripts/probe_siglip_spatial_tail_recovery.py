"""Local-only SigLIP spatial-tail recovery probe."""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
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
