from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_inshop_training_geometry",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_inshop_training_geometry.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_training_geometry_reports_margins_and_fragmentation() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0], [-0.9, 0.1], [-0.8, 0.2]],
        dtype=np.float64,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.asarray([10, 10, 10, 20, 20, 20])
    proxies = np.asarray([[1.0, 0.0], [-1.0, 0.0]])
    result = _module.analyze(embeddings, labels, proxies, np.asarray([10, 20]), chunk_size=2)

    assert result["leave_one_out_recall_at_1"] == 1.0
    assert result["negative_sample_margin_fraction"] == 0.0
    assert result["negative_proxy_margin_fraction"] == 0.0
    assert result["one_nn_fragmentation_eligible_classes"] == 2
    assert result["one_nn_fragmented_classes"] == 0
    assert result["within_class_pair_count"] == 6


def test_fragmentation_detects_two_components() -> None:
    similarity = np.asarray(
        [
            [1.0, 0.9, 0.0, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.9],
            [0.0, 0.0, 0.9, 1.0],
        ]
    )
    assert _module._is_fragmented(similarity) is True


def test_scalar_string_rejects_vector() -> None:
    with pytest.raises(ValueError, match="scalar"):
        _module._scalar_string(np.asarray(["train"]))
