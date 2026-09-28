"""A shuffled control must preserve B class counts without changing A."""

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np


def test_sham_preserves_inventory_and_protected_domain():
    path = Path(__file__).with_name("probe_inshop_identity_pooling.py")
    assert path.exists(), "pooling falsifier is not implemented"
    spec = importlib.util.spec_from_file_location("pooling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = np.repeat(np.arange(12), np.arange(12) + 2)
    protected = labels < 3
    result = module.shuffled_b_labels(labels, protected)
    assert np.array_equal(result[protected], labels[protected])
    assert Counter(result[~protected]) == Counter(labels[~protected])
    assert np.mean(result[~protected] != labels[~protected]) > 0.5
    assert np.array_equal(result, module.shuffled_b_labels(labels, protected))
