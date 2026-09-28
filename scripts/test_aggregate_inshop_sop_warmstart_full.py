"""Seed/product bootstrap keeps paired deltas and query counts together."""

import numpy as np
import pytest
from aggregate_inshop_sop_warmstart_full import product_seed_bootstrap, validate_quality


def test_constant_paired_delta_has_exact_interval():
    products = np.array(["a", "a", "b", "c"])
    deltas = np.ones((3, 4), dtype=np.float64)
    assert product_seed_bootstrap(deltas, products) == (1.0, 1.0)
    assert product_seed_bootstrap(np.zeros_like(deltas), products) == (0.0, 0.0)


def test_bootstrap_shares_products_and_weights_queries():
    products = np.array(["a", "a", "a", "b", "c", "c"])
    deltas = np.array([[1, 0, 1, -1, 0, 1], [-1, 1, 0, 1, -1, 0], [0, 1, -1, 0, 1, 1]])
    rng = np.random.default_rng(179019)
    rows = [np.flatnonzero(products == name) for name in np.unique(products)]
    draws = []
    for _ in range(5_000):
        seeds = rng.integers(0, 3, 3)
        selected = np.concatenate([rows[i] for i in rng.integers(0, 3, 3)])
        draws.append(deltas[seeds][:, selected].mean())
    assert product_seed_bootstrap(deltas, products) == tuple(np.quantile(draws, [0.025, 0.975]))


def test_quality_rejects_stale_scalars_nonbinary_hits_and_nonfinite_ap():
    quality = {
        "per_query_r1": [1, 0],
        "per_query_ap": [0.5, 0.0],
        "recall_at_1": 0.5,
        "map_at_r": 0.25,
    }
    validate_quality(quality, 2)
    for field, value in (
        ("recall_at_1", 1.0),
        ("per_query_r1", [2, 0]),
        ("per_query_ap", [float("nan"), 0]),
    ):
        with pytest.raises(ValueError):
            validate_quality({**quality, field: value}, 2)
