from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "measure_fragmentation_identity_stability",
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "measure_fragmentation_identity_stability.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _pack(labels: np.ndarray, embeddings: np.ndarray) -> dict[str, np.ndarray]:
    return {"labels": labels, "embeddings": embeddings}


def test_kappa_is_one_for_identical_and_negative_for_opposites() -> None:
    values = np.asarray([False, False, True, True])
    assert _MODULE._kappa(values, values) == pytest.approx((1.0, 1.0))
    agreement, kappa = _MODULE._kappa(values, ~values)
    assert agreement == 0.0
    assert kappa == pytest.approx(-1.0)


def test_measure_refuses_nonidentical_class_sets() -> None:
    embeddings = np.eye(6, dtype=np.float64)
    first = _pack(np.asarray([0, 0, 0, 1, 1, 1]), embeddings)
    second = _pack(np.asarray([0, 0, 0, 2, 2, 2]), embeddings)
    with pytest.raises(ValueError, match="identity labels differ"):
        _MODULE.measure([first, second, first])
