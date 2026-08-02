from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "measure_fragmentation_confounding",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_fragmentation_confounding.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
_quintile_bins = _module._quintile_bins
measure = _module.measure


def test_quintile_bins_are_deterministic_and_collapse_duplicate_edges() -> None:
    bins, edges = _quintile_bins(np.asarray([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert edges == sorted(set(edges))
    assert bins.tolist() == [1, 1, 1, 2, 3]


def test_measure_reports_locked_adjustment_fields() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(20), 4)
    embeddings = rng.normal(size=(labels.size, 12))
    result = measure(embeddings, labels)
    assert result["eligible_classes"] == 20
    assert result["fragmented_classes"] + result["connected_classes"] == 20
    assert 0.0 <= result["retained_fraction"] <= 1.0
    assert result["correlation_order"][0] == "fragmented"
    assert np.asarray(result["pearson_correlation_matrix"]).shape == (4, 4)
