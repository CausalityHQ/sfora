"""Coverage and distinct-image checks for matched product-pair batches."""

from __future__ import annotations

from collections import Counter

import numpy as np

from sfora.product_pair_schedule import product_pair_batches


def test_every_eligible_product_is_seen_once_per_pass_with_distinct_images() -> None:
    labels = np.repeat(np.arange(35, dtype=np.int64), 3)
    batches = product_pair_batches(labels, products_per_batch=8, passes=3, seed=17)
    assert batches.shape == (15, 16)
    for epoch in range(3):
        observed = Counter()
        for batch in batches[epoch * 5 : (epoch + 1) * 5]:
            products = labels[batch].reshape(8, 2)
            assert np.array_equal(products[:, 0], products[:, 1])
            assert len(set(products[:, 0].tolist())) == 8
            assert np.all(batch.reshape(8, 2)[:, 0] != batch.reshape(8, 2)[:, 1])
            observed.update(products[:, 0].tolist())
        assert set(observed) == set(range(35))
        assert min(observed.values()) == 1


def test_schedule_replays_and_rejects_singletons() -> None:
    labels = np.repeat(np.arange(9, dtype=np.int64), 2)
    first = product_pair_batches(labels, products_per_batch=4, passes=2, seed=3)
    second = product_pair_batches(labels, products_per_batch=4, passes=2, seed=3)
    np.testing.assert_array_equal(first, second)
    try:
        product_pair_batches(
            np.array([0, 0, 1, 1, 2], dtype=np.int64), products_per_batch=2, passes=1, seed=3
        )
    except ValueError as error:
        assert "two images" in str(error)
    else:
        raise AssertionError("singleton product was accepted")
