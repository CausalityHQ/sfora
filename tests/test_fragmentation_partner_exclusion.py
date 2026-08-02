from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))
_spec = importlib.util.spec_from_file_location(
    "measure_fragmentation_partner_exclusion",
    _scripts / "measure_fragmentation_partner_exclusion.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
_partner_excluded_correct = _module._partner_excluded_correct
measure = _module.measure


def test_partner_exclusion_rejects_trivial_pair_support() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.0, 1.0],
            [0.001, 0.999],
        ]
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.asarray([0, 0, 1, 1])
    assert _partner_excluded_correct(embeddings, labels).tolist() == [False] * 4


def test_measure_reports_locked_partner_exclusion_fields() -> None:
    rng = np.random.default_rng(11)
    labels = np.repeat(np.arange(20), 4)
    embeddings = rng.normal(size=(labels.size, 12))
    result = measure(embeddings, labels)
    assert result["eligible_classes"] == 20
    assert result["fragmented_classes"] + result["connected_classes"] == 20
    assert 0.0 <= result["retained_fraction"] <= 1.0
    assert "adjusted_fragmented_minus_connected_partner_excluded_top1_points" in result
