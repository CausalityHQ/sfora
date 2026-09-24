"""Cross-label neighbors must be computed on normalized, fit-only features."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_sop_label_conflicts.py"
SPEC = importlib.util.spec_from_file_location("audit_sop_label_conflicts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_cross_label_neighbor_normalizes_and_excludes_own_product() -> None:
    features = np.array([[2, 0], [1.8, 0.2], [4, 0], [0, 3]], dtype=np.float32)
    labels = np.array([1, 1, 2, 3], dtype=np.int64)
    scores, neighbors = MODULE.nearest_cross_label_cosines(features, labels, block_rows=2)
    assert neighbors.tolist() == [2, 2, 0, 1]
    assert scores[:3] == pytest.approx([1, 0.9938837, 1], abs=1e-6)
    assert scores[3] == pytest.approx(0.1104315, abs=1e-6)


def test_cross_label_neighbor_rejects_zero_vector_and_single_product() -> None:
    with pytest.raises(ValueError, match="feature inventory"):
        MODULE.nearest_cross_label_cosines(
            np.array([[0, 0], [1, 0]], dtype=np.float32),
            np.array([1, 2], dtype=np.int64),
            block_rows=2,
        )
    with pytest.raises(ValueError, match="feature inventory"):
        MODULE.nearest_cross_label_cosines(
            np.array([[1, 0], [0, 1]], dtype=np.float32),
            np.array([1, 1], dtype=np.int64),
            block_rows=2,
        )
