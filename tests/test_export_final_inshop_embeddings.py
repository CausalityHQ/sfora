"""Independent In-Shop final-artifact verifier tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "export_final_inshop_embeddings",
    Path(__file__).resolve().parents[1] / "scripts" / "export_final_inshop_embeddings.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_independent_query_gallery_recall_uses_cosine_and_disjoint_rows() -> None:
    query = np.asarray([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    query_labels = np.asarray([10, 20, 30])
    gallery = np.asarray([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
    gallery_labels = np.asarray([10, 20, 99, 30])

    score = _module.independent_query_gallery_recall_at_1(
        query, query_labels, gallery, gallery_labels, chunk_size=2
    )

    assert score == pytest.approx(1.0)


def test_independent_query_gallery_recall_rejects_zero_norms() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        _module.independent_query_gallery_recall_at_1(
            np.asarray([[0.0, 0.0]]),
            np.asarray([1]),
            np.asarray([[1.0, 0.0]]),
            np.asarray([1]),
        )
