from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "measure_spectral_class_connectivity",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_spectral_class_connectivity.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def test_measure_detects_disconnected_one_nn_class_graph() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.1], [-1.0, 0.0], [-0.99, 0.1]], dtype=np.float64)
    result = _module.measure(embeddings, np.zeros(4, dtype=np.int64), temperature=0.1)

    assert result["eligible_classes"] == 1
    assert result["one_nn_fragmented_fraction"] == pytest.approx(1.0)
    assert result["mean_class_size"] == pytest.approx(4.0)
    assert np.isfinite(result["median_fiedler_value"])
