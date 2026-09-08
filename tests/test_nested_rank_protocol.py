from __future__ import annotations

import math

import numpy as np
import pytest

from sfora.nested_rank_protocol import (
    class_disjoint_fold,
    cluster_bootstrap_lower_bound,
    identity_balanced_schedule,
)
from sfora.representation_ceiling import deterministic_class_partition


def test_class_disjoint_fold_exactly_reuses_sealed_partition() -> None:
    labels = np.repeat(np.arange(20, dtype=np.int64), 3)
    sample_ids = tuple(f"sample-{index}" for index in range(labels.size))

    fold = class_disjoint_fold(sample_ids, labels, seed=17)
    sealed = deterministic_class_partition(
        tuple(int(label) for label in labels), fit_fraction=0.8, seed=17
    )

    assert fold.optimization == sealed.fit_row_indexes
    assert fold.validation == sealed.validation_row_indexes
    assert fold.validation_queries == fold.validation
    assert not set(labels[list(fold.optimization)]) & set(labels[list(fold.validation)])
    assert len(fold.optimization) + len(fold.validation) == labels.size


@pytest.mark.parametrize("seed", [17, 1729, 65537])
def test_class_disjoint_fold_is_deterministic_for_registered_seeds(seed: int) -> None:
    labels = np.repeat(np.arange(50, dtype=np.int64), 2)
    ids = tuple(range(labels.size))
    assert class_disjoint_fold(ids, labels, seed=seed) == class_disjoint_fold(
        ids, labels, seed=seed
    )


def test_identity_schedule_has_32_unique_labels_and_four_rows_each() -> None:
    labels = np.repeat(np.arange(34, dtype=np.int64), 2)
    rows = np.arange(labels.size, dtype=np.float32)
    teacher = np.stack((rows + 1.0, (rows % 7) + 1.0, (rows % 5) + 2.0), axis=1)

    schedule = identity_balanced_schedule(
        labels,
        teacher,
        tuple(range(labels.size)),
        seed=17,
        steps=6,
    )

    assert schedule == identity_balanced_schedule(
        labels,
        teacher,
        tuple(range(labels.size)),
        seed=17,
        steps=6,
    )
    assert len(schedule) == 6
    for batch in schedule:
        batch_labels = labels[list(batch)]
        unique, counts = np.unique(batch_labels, return_counts=True)
        assert len(batch) == 128
        assert unique.size == 32
        assert np.array_equal(counts, np.full(32, 4))


def test_identity_schedule_handles_rollover_collisions_without_duplicates() -> None:
    labels = np.repeat(np.arange(35, dtype=np.int64), 2)
    teacher = np.eye(35, dtype=np.float32)[labels]
    schedule = identity_balanced_schedule(
        labels,
        teacher,
        tuple(range(labels.size)),
        seed=1729,
        steps=10,
    )
    for batch in schedule:
        assert np.unique(labels[list(batch)]).size == 32


def test_identity_schedule_rejects_nonfinite_and_insufficient_authority() -> None:
    labels = np.repeat(np.arange(31, dtype=np.int64), 2)
    teacher = np.ones((labels.size, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="identity schedule authority"):
        identity_balanced_schedule(labels, teacher, tuple(range(labels.size)), seed=17, steps=1)
    labels = np.repeat(np.arange(32, dtype=np.int64), 2)
    teacher = np.ones((labels.size, 4), dtype=np.float32)
    teacher[0, 0] = np.nan
    with pytest.raises(ValueError, match="identity schedule authority"):
        identity_balanced_schedule(labels, teacher, tuple(range(labels.size)), seed=17, steps=1)


def test_cluster_bootstrap_is_query_weighted_and_nearest_rank() -> None:
    differences = np.asarray([1.0, 1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    classes = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)

    actual = cluster_bootstrap_lower_bound(
        differences,
        classes,
        draws=10_000,
        seed=17,
        quantile=0.05,
    )

    rng = np.random.Generator(np.random.PCG64(17))
    estimates = []
    sums = np.asarray([2.0, -4.0])
    counts = np.asarray([2, 4])
    for _ in range(10_000):
        selected = rng.integers(0, 2, size=2)
        estimates.append(float(sums[selected].sum() / counts[selected].sum()))
    expected = sorted(estimates)[math.ceil(0.05 * len(estimates)) - 1]
    assert actual == expected


def test_protocol_rejects_bool_seed_duplicate_ids_and_class_singletons() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError):
        class_disjoint_fold((0, 1, 2, 3), labels, seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        class_disjoint_fold((0, 0, 2, 3), labels, seed=17)
    with pytest.raises(ValueError):
        class_disjoint_fold((0, 1, 2), np.asarray([0, 0, 1], dtype=np.int64), seed=17)
