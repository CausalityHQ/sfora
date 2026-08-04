from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "measure_cross_seed_query_errors",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_cross_seed_query_errors.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
measure = _MODULE.measure


def _pack(embeddings: list[list[float]], labels: list[int]) -> dict[str, object]:
    rows = len(labels)
    return {
        "embeddings": np.asarray(embeddings, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "example_ids": np.asarray([f"row-{index}" for index in range(rows)]),
        "source_paths": np.asarray([f"/row/{index}" for index in range(rows)]),
        "checkpoint_sha256": "digest",
    }


def test_measure_cross_seed_error_overlap() -> None:
    query_labels = [0, 1, 2, 3]
    gallery_labels = [0, 1, 2, 3, 4]
    gallery = _pack(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]],
        gallery_labels,
    )
    seed0_query = _pack([[1, 0], [1, 0], [-1, 0], [1, 1]], query_labels)
    seed1_query = _pack([[1, 0], [0, 1], [1, 1], [1, 1]], query_labels)

    result = measure([seed0_query, seed1_query], [gallery, gallery], chunk_size=2)

    assert result["recall_at_1"] == [0.5, 0.5]
    assert result["error_counts"] == [2, 2]
    assert result["both_wrong"] == 1
    assert result["seed0_only_correct"] == 1
    assert result["seed1_only_correct"] == 1
    assert result["error_overlap_coefficient"] == 0.5
    assert result["error_set_jaccard"] == 1 / 3
    assert result["oracle_either_seed_recall_at_1"] == 0.75
