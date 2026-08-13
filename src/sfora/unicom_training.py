"""Single-device equivalents of UNICOM's class-sharded margin loss."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from sfora.unicom_inshop import InshopRecord


def experiment_stream_seed(seed: int, offset: int) -> int:
    """Namespace stochastic streams across experiments without adjacent-seed aliases."""

    if type(seed) is not int or type(offset) is not int:
        raise TypeError("experiment seed and stream offset must be builtin integers")
    if seed < 0 or offset < 0 or offset >= 1_000_003:
        raise ValueError("experiment seed/stream offset is outside the registered range")
    result = seed * 1_000_003 + offset
    if result > 2**63 - 1:
        raise ValueError("derived experiment stream seed exceeds int64")
    return result


def identity_holdout(
    records: tuple[InshopRecord, ...], *, fraction: float, seed: int
) -> tuple[
    tuple[InshopRecord, ...],
    tuple[InshopRecord, ...],
    tuple[InshopRecord, ...],
    dict[str, int],
]:
    """Hold out complete train identities and choose one query per held-out identity."""

    if type(records) is not tuple or not records or any(row.split != "train" for row in records):
        raise ValueError("holdout records must be a nonempty train tuple")
    if type(fraction) is not float or not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("holdout fraction must be finite and in [0, 1)")
    if type(seed) is not int:
        raise TypeError("holdout seed must be a builtin integer")
    grouped: dict[str, list[InshopRecord]] = defaultdict(list)
    for row in records:
        grouped[row.label].append(row)
    if fraction == 0.0:
        labels = sorted(grouped)
        return records, (), (), {label: index for index, label in enumerate(labels)}
    eligible = sorted(label for label, rows in grouped.items() if len(rows) >= 2)
    if len(grouped) < 2 or not eligible:
        raise ValueError("holdout needs at least one identity with two images")
    labels = np.asarray(eligible, dtype=np.str_)
    generator = np.random.Generator(np.random.PCG64(seed))
    generator.shuffle(labels)
    count = max(1, min(labels.size, round(len(eligible) * fraction)))
    heldout = set(labels[:count].tolist())
    optimization = tuple(row for row in records if row.label not in heldout)
    query: list[InshopRecord] = []
    gallery: list[InshopRecord] = []
    for label in sorted(heldout):
        rows = sorted(grouped[label], key=lambda row: str(row.image_path))
        query_index = int(generator.integers(0, len(rows)))
        query.append(rows[query_index])
        gallery.extend(row for index, row in enumerate(rows) if index != query_index)
    optimization_labels = sorted({row.label for row in optimization})
    return (
        optimization,
        tuple(query),
        tuple(gallery),
        {label: index for index, label in enumerate(optimization_labels)},
    )


def padded_epoch_indices(
    *, size: int, global_batch: int, epoch: int, seed: int, shards: int = 8
) -> tuple[int, ...]:
    """Return the global union/order of UNICOM's distributed epoch sampler."""

    if any(type(value) is not int for value in (size, global_batch, epoch, seed, shards)):
        raise TypeError("sampler values must be builtin integers")
    if size <= 0 or global_batch <= 0 or epoch < 0 or shards <= 0 or global_batch % shards != 0:
        raise ValueError("sampler size/batch must be positive and epoch nonnegative")
    rank_samples = math.ceil(size / shards)
    local_batch = global_batch // shards
    total = (rank_samples // local_batch) * global_batch
    if total == 0:
        raise ValueError("sampler has no complete distributed batch")
    distributed_total = rank_samples * shards
    generator = torch.Generator().manual_seed(experiment_stream_seed(seed, 1_000 + epoch))
    shuffled = torch.randperm(size, generator=generator).tolist()
    padded = (shuffled * math.ceil(distributed_total / size))[:distributed_total]
    return tuple(padded[:total])


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
            torch.argsort(torch.rand(dimension, generator=generator, device=device))[:selected]
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
        selected_weights = F.normalize(weights[class_slice].index_select(1, coordinates), dim=1)
        shard_logits.append(F.linear(selected_embeddings, selected_weights))
    logits = torch.cat(shard_logits, dim=1).clamp(-1.0, 1.0)
    rows = torch.arange(labels.numel(), device=labels.device)
    if margin:
        with torch.no_grad():
            target = logits[rows, labels]
            logits[rows, labels] = torch.cos(torch.acos(target) + margin)
    return F.cross_entropy(logits * scale, labels)
