from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "measure_fragmentation_cross_series_retrieval",
    _SCRIPTS / "measure_fragmentation_cross_series_retrieval.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_cross_series_retrieval_excludes_only_same_identity_same_series() -> None:
    # Query 0 is closest to same-identity/same-series item 1, then to the negative
    # item 4, while its cross-series positive 2 is farther away. It must be wrong.
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.6, 0.8],
            [-1.0, 0.0],
            [0.8, 0.6],
            [-0.8, 0.6],
        ],
        dtype=np.float64,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    series = np.asarray(["a", "a", "b", "a", "a", "b"])
    correct = _MODULE._cross_series_correct(embeddings, labels, series)
    assert not bool(correct[0])


def test_other_identity_with_same_series_token_remains_a_competitor() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.99, 0.01]], dtype=np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.asarray([0, 0, 1])
    series = np.asarray(["a", "b", "a"])
    correct = _MODULE._cross_series_correct(embeddings, labels, series)
    assert not bool(correct[0])
