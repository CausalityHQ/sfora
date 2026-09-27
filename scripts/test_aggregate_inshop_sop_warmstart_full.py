"""Seed/product bootstrap keeps paired deltas and query counts together."""

import numpy as np
from aggregate_inshop_sop_warmstart_full import product_seed_bootstrap


def test_constant_paired_delta_has_exact_interval():
    products = np.array(["a", "a", "b", "c"])
    deltas = np.ones((3, 4), dtype=np.float64)
    assert product_seed_bootstrap(deltas, products) == (1.0, 1.0)
    assert product_seed_bootstrap(np.zeros_like(deltas), products) == (0.0, 0.0)
