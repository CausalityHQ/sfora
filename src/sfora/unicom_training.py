"""Single-device equivalents of UNICOM's class-sharded margin loss."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def padded_epoch_indices(
    *, size: int, global_batch: int, epoch: int, seed: int
) -> tuple[int, ...]:
    """Return the global union/order of UNICOM's distributed epoch sampler."""

    if any(type(value) is not int for value in (size, global_batch, epoch, seed)):
        raise TypeError("sampler values must be builtin integers")
    if size <= 0 or global_batch <= 0 or epoch < 0:
        raise ValueError("sampler size/batch must be positive and epoch nonnegative")
    total = math.ceil(size / global_batch) * global_batch
    generator = torch.Generator().manual_seed(seed + epoch)
    shuffled = torch.randperm(size, generator=generator).tolist()
    return tuple((shuffled * math.ceil(total / size))[:total])


def sample_shard_masks(
    *,
    dimension: int,
    selected: int,
    shards: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw the official argsort-of-uniform feature mask independently per shard."""

    if any(type(value) is not int for value in (dimension, selected, shards)):
        raise TypeError("mask dimensions must be builtin integers")
    if dimension <= 0 or not 0 < selected <= dimension or shards <= 0:
        raise ValueError("mask dimensions differ")
    return torch.stack(
        [
            torch.argsort(torch.rand(dimension, generator=generator, device=device))[
                :selected
            ]
            for _ in range(shards)
        ]
    )


def _class_slices(class_count: int, shard_count: int) -> tuple[slice, ...]:
    quotient, remainder = divmod(class_count, shard_count)
    start = 0
    result: list[slice] = []
    for shard in range(shard_count):
        width = quotient + int(shard < remainder)
        result.append(slice(start, start + width))
        start += width
    return tuple(result)


def sharded_mask_arcface_loss(
    embeddings: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    margin: float = 0.25,
    scale: float = 32.0,
) -> torch.Tensor:
    """Emulate UNICOM's independently masked class shards on one device.

    Each row of ``masks`` is assigned to one contiguous class shard using the
    same quotient/remainder partition as ``PartialFC_V2``. The shard logits
    are concatenated before one global cross-entropy reduction.
    """

    if embeddings.ndim != 2 or weights.ndim != 2 or embeddings.shape[1] != weights.shape[1]:
        raise ValueError("embedding and class-weight matrices must share a feature width")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must match embedding rows")
    if masks.ndim != 2 or masks.shape[0] == 0 or masks.shape[1] == 0:
        raise ValueError("masks must be a nonempty shard-by-coordinate matrix")
    if masks.dtype != torch.int64 or labels.dtype != torch.int64:
        raise TypeError("labels and masks must be int64 tensors")
    if masks.device != embeddings.device or weights.device != embeddings.device:
        raise ValueError("embeddings, weights, and masks must share a device")
    if labels.device != embeddings.device:
        raise ValueError("labels must share the embedding device")
    if torch.any(masks < 0) or torch.any(masks >= embeddings.shape[1]):
        raise ValueError("mask coordinate is outside the feature width")
    if any(torch.unique(mask).numel() != mask.numel() for mask in masks):
        raise ValueError("each shard mask must contain unique coordinates")
    if torch.any(labels < 0) or torch.any(labels >= weights.shape[0]):
        raise ValueError("label is outside the class range")
    if not math.isfinite(margin) or not 0.0 <= margin < math.pi:
        raise ValueError("margin must be finite and in [0, pi)")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    shard_logits: list[torch.Tensor] = []
    for class_slice, coordinates in zip(
        _class_slices(weights.shape[0], masks.shape[0]), masks, strict=True
    ):
        selected_embeddings = F.normalize(embeddings.index_select(1, coordinates), dim=1)
        selected_weights = F.normalize(
            weights[class_slice].index_select(1, coordinates), dim=1
        )
        shard_logits.append(F.linear(selected_embeddings, selected_weights))
    logits = torch.cat(shard_logits, dim=1).clamp(-1.0, 1.0)
    rows = torch.arange(labels.numel(), device=labels.device)
    if margin:
        with torch.no_grad():
            target = logits[rows, labels]
            logits[rows, labels] = torch.cos(torch.acos(target) + margin)
    return F.cross_entropy(logits * scale, labels)
