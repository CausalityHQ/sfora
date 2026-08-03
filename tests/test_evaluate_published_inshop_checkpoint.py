from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "evaluate_published_inshop_checkpoint",
    _SCRIPTS / "evaluate_published_inshop_checkpoint.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_upstream_recall_accepts_tied_negative_but_rejects_strictly_better_negative() -> None:
    queries = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    query_labels = np.asarray([0, 1])
    gallery = np.asarray(
        [
            [1.0, 0.0],  # positive for query 0 and tied negative for query 1
            [1.0, 0.0],  # positive for query 1 and tied negative for query 0
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )
    gallery_labels = np.asarray([0, 1, 2])

    assert _MODULE._upstream_recall_at_1(
        queries, query_labels, gallery, gallery_labels
    ) == 1.0

    # Move query 1's positive away while query 0's vector remains a strictly
    # better negative. Query 0 still succeeds, query 1 must fail.
    gallery[1] = np.asarray([0.0, 1.0], dtype=np.float32)
    assert _MODULE._upstream_recall_at_1(
        queries, query_labels, gallery, gallery_labels
    ) == 0.5

