from __future__ import annotations

from collections import Counter

import pytest

from sfora.unicom_rank_finish import identity_balanced_batches


def _labels(identity_count: int = 40, images_per_identity: int = 5) -> tuple[str, ...]:
    return tuple(
        f"item-{identity:03d}"
        for identity in range(identity_count)
        for _ in range(images_per_identity)
    )


def test_identity_balanced_batches_are_exact_replay_and_epoch_separated() -> None:
    labels = _labels()
    arguments = {
        "batch_size": 128,
        "images_per_identity": 4,
        "seed": 7,
        "steps": 3,
    }

    first = identity_balanced_batches(labels, epoch=5, **arguments)
    replay = identity_balanced_batches(labels, epoch=5, **arguments)
    next_epoch = identity_balanced_batches(labels, epoch=6, **arguments)

    assert first == replay
    assert first != next_epoch
    assert len(first) == 3
    for batch in first:
        assert len(batch) == 128
        assert len(set(batch)) == 128
        counts = Counter(labels[index] for index in batch)
        assert len(counts) == 32
        assert set(counts.values()) == {4}


def test_identity_balanced_batches_cycle_only_after_identity_inventory() -> None:
    labels = _labels(images_per_identity=5)

    batches = identity_balanced_batches(
        labels,
        batch_size=128,
        images_per_identity=4,
        seed=11,
        epoch=5,
        steps=2,
    )

    by_identity: dict[str, list[int]] = {}
    for batch in batches:
        for index in batch:
            by_identity.setdefault(labels[index], []).append(index)
    assert all(
        len(indices) <= 5 or len(set(indices[:5])) == 5
        for indices in by_identity.values()
    )


@pytest.mark.parametrize(
    ("labels", "kwargs"),
    (
        (("a", "a", "b", "b"), {"batch_size": 3, "images_per_identity": 2}),
        (("a", "a", "b", "b"), {"batch_size": 4, "images_per_identity": 1}),
        (("a", "a", "a", "a"), {"batch_size": 4, "images_per_identity": 2}),
        (("a", "", "b", "b"), {"batch_size": 4, "images_per_identity": 2}),
    ),
)
def test_identity_balanced_batches_reject_invalid_or_impossible_shapes(
    labels: tuple[str, ...], kwargs: dict[str, int]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        identity_balanced_batches(labels, seed=0, epoch=5, steps=1, **kwargs)
