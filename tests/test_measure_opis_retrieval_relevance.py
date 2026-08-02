from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "measure_opis_retrieval_relevance",
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "measure_opis_retrieval_relevance.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
measure = _module.measure


def test_measure_is_deterministic_and_finite() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.20],
            [0.0, 1.0],
            [0.20, 0.98],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2])

    first = measure(embeddings, labels, negative_sample_count=1000, seed=7)
    second = measure(embeddings, labels, negative_sample_count=1000, seed=7)

    assert first == second
    assert first["class_count"] == 3
    assert first["image_count"] == 6
    assert first["grid_size"] == 101
    assert np.isfinite(first["opis"])
    assert np.isfinite(first["spearman_opis_contribution_vs_class_error"])


def test_measure_rejects_singleton_classes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        measure(
            np.eye(3, dtype=np.float32),
            np.asarray([0, 0, 1]),
            negative_sample_count=10,
        )
