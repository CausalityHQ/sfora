"""Static PFML field-census tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_pfml_final_field",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_pfml_final_field.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_pair_field_statistics_counts_support_and_excludes_self() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0], [-0.99, 0.1]], dtype=np.float64
    )
    labels = np.asarray([0, 0, 1, 1])
    stats = _module.pair_field_statistics(
        vectors,
        labels,
        vectors,
        labels,
        delta=0.2,
        alpha=3.0,
        self_pairs=True,
        chunk_size=2,
        histogram_bins=128,
    )
    assert stats["same"]["pair_count"] == 2
    assert stats["different"]["pair_count"] == 4
    assert stats["same"]["active_pair_count"] == 0
    assert stats["different"]["active_pair_count"] == 0


def test_pair_field_statistics_detects_active_relations() -> None:
    left = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    right = np.asarray([[0.0, 1.0], [0.999, 0.045]])
    stats = _module.pair_field_statistics(
        left,
        np.asarray([0, 1]),
        right,
        np.asarray([0, 1]),
        delta=0.2,
        alpha=3.0,
        self_pairs=False,
        histogram_bins=128,
    )
    assert stats["same"]["active_pair_count"] == 2
    assert stats["different"]["active_pair_count"] == 2
    assert stats["different"]["active_radial_force_sum"] > 0


def test_proxy_occupancy_reports_effective_count() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.asarray([0, 0, 0, 0])
    proxies = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = _module.proxy_occupancy_statistics(
        embeddings, labels, proxies, np.asarray([0, 0])
    )
    row = result["per_class"][0]
    assert row["occupied_proxies"] == 2
    assert row["effective_proxy_count"] == pytest.approx(2.0)
    assert row["maximum_occupancy_share"] == 0.5


def test_pair_field_statistics_rejects_zero_norm() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        _module.pair_field_statistics(
            np.asarray([[0.0, 0.0]]),
            np.asarray([0]),
            np.asarray([[1.0, 0.0]]),
            np.asarray([0]),
            delta=0.2,
            alpha=3.0,
            self_pairs=False,
        )
