"""Exact-content Cars196 tie-audit tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "analyze_cars_content_ties",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_cars_content_ties.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_nearest_indices_exposes_partial_topk_tie_choice() -> None:
    # Rows 1 and 2 are identical but have different labels. Query 0 is exactly
    # equidistant to them, so both rules must return a valid exact-tie member;
    # the audit records either disagreement without treating it as geometry.
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1])
    production, stable = _module._nearest_indices(embeddings, labels, chunk_size=2)
    assert production.shape == stable.shape == (4,)
    assert production[0] in {1, 2}
    assert stable[0] == 1
    assert np.sum((embeddings[0] - embeddings[1]) ** 2) == np.sum(
        (embeddings[0] - embeddings[2]) ** 2
    )
