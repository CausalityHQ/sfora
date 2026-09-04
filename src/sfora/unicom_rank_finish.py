"""Deterministic rank-finishing primitives for the UniCOM screen."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


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


__all__ = ["identity_balanced_batches"]
