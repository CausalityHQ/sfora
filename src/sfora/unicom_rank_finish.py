"""Deterministic rank-finishing primitives for the UniCOM screen."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
import torch


def identity_balanced_batches(
    labels: Sequence[str],
    *,
    batch_size: int,
    images_per_identity: int,
    seed: int,
    epoch: int,
    steps: int,
) -> tuple[tuple[int, ...], ...]:
    """Return a replayable identity-balanced index schedule for one epoch."""

    integers = (batch_size, images_per_identity, seed, epoch, steps)
    if any(type(value) is not int for value in integers):
        raise TypeError("rank-finish schedule parameters must be builtin integers")
    if (
        batch_size <= 0
        or images_per_identity < 2
        or batch_size % images_per_identity
        or seed < 0
        or epoch <= 0
        or steps <= 0
    ):
        raise ValueError("rank-finish schedule parameters differ")
    if not labels or any(type(label) is not str or not label for label in labels):
        raise ValueError("rank-finish labels differ")

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    identities_per_batch = batch_size // images_per_identity
    if (
        len(grouped) < identities_per_batch
        or any(len(indices) < images_per_identity for indices in grouped.values())
    ):
        raise ValueError("rank-finish identity inventory differs")

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence((seed, epoch))))
    identity_names = tuple(sorted(grouped))
    permutations = {
        label: list(rng.permutation(grouped[label]).tolist())
        for label in identity_names
    }
    positions = {label: 0 for label in identity_names}

    def draw(label: str) -> tuple[int, ...]:
        selected: list[int] = []
        while len(selected) < images_per_identity:
            permutation = permutations[label]
            position = positions[label]
            if position == len(permutation):
                permutation = list(rng.permutation(grouped[label]).tolist())
                permutations[label] = permutation
                position = 0
            while position < len(permutation) and len(selected) < images_per_identity:
                candidate = permutation[position]
                position += 1
                if candidate not in selected:
                    selected.append(candidate)
            positions[label] = position
        return tuple(selected)

    batches = []
    for _ in range(steps):
        selected_identities = rng.choice(
            identity_names, size=identities_per_batch, replace=False
        ).tolist()
        batch = tuple(
            index
            for label in selected_identities
            for index in draw(str(label))
        )
        batches.append(batch)
    return tuple(batches)


def smooth_ap_finish_loss(
    embeddings: torch.Tensor,
    labels: Sequence[object],
    *,
    dimensions: int = 512,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Compute the fixed Smooth-AP finish loss in deployment geometry."""

    if type(embeddings) is not torch.Tensor or not embeddings.is_floating_point():
        raise TypeError("rank-finish embeddings must be a floating tensor")
    if type(dimensions) is not int or dimensions != 512:
        raise ValueError("rank-finish dimensions differ")
    if type(temperature) is not float or temperature != 0.01:
        raise ValueError("rank-finish temperature differs")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(labels)
        or embeddings.shape[1] < dimensions
        or embeddings.shape[0] < 4
        or not torch.isfinite(embeddings).all()
        or torch.any(torch.linalg.vector_norm(embeddings.float(), dim=1) == 0.0)
    ):
        raise ValueError("rank-finish embedding inventory differs")
    if any(label is None for label in labels):
        raise ValueError("rank-finish labels differ")
    counts = Counter(labels)
    if any(count < 2 for count in counts.values()):
        raise ValueError("rank-finish positive inventory differs")

    normalized = torch.nn.functional.normalize(embeddings.float(), dim=1)[
        :, :dimensions
    ]
    distances = torch.sum(
        (normalized[:, None, :] - normalized[None, :, :]).square(), dim=2
    )
    rows = len(labels)
    encoded: dict[object, int] = {}
    label_ids = []
    for label in labels:
        label_ids.append(encoded.setdefault(label, len(encoded)))
    label_tensor = torch.tensor(label_ids, device=embeddings.device)
    identity = torch.eye(rows, device=embeddings.device, dtype=torch.bool)
    positive_mask = label_tensor[:, None].eq(label_tensor[None, :]) & ~identity
    comparisons = torch.sigmoid(
        (distances[:, :, None] - distances[:, None, :]) / temperature
    )
    candidate_mask = ~identity
    competitor_mask = candidate_mask[:, None, :] & ~identity[None, :, :]
    rank = 1.0 + (comparisons * competitor_mask).sum(dim=2)
    positive_competitor_mask = positive_mask[:, None, :] & ~identity[None, :, :]
    positive_rank = 1.0 + (comparisons * positive_competitor_mask).sum(dim=2)
    precision = positive_rank / rank
    per_anchor = (precision * positive_mask).sum(dim=1) / positive_mask.sum(dim=1)
    loss = 1.0 - per_anchor.mean()
    if not torch.isfinite(loss):
        raise ValueError("rank-finish loss is nonfinite")
    return loss


__all__ = ["identity_balanced_batches", "smooth_ap_finish_loss"]
