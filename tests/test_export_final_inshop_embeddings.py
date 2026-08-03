"""Independent In-Shop final-artifact verifier tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "export_final_inshop_embeddings",
    Path(__file__).resolve().parents[1] / "scripts" / "export_final_inshop_embeddings.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _example(label: int, suffix: str) -> SimpleNamespace:
    return SimpleNamespace(label=label, example_id=suffix, image=f"/data/{suffix}.jpg")


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


def test_partition_verifier_checks_all_three_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _module,
        "EXPECTED_INSHOP_PARTITION",
        {"train": (3, 2), "query": (2, 2), "gallery": (2, 2)},
    )
    train = [_example(1, "t-1"), _example(1, "t-2"), _example(2, "t-3")]
    query = [_example(10, "q-1"), _example(11, "q-2")]
    gallery = [_example(10, "g-1"), _example(11, "g-2")]
    result = _module.verify_official_partition(
        SimpleNamespace(train=train, query=query, gallery=gallery)
    )
    assert result["train_query_identity_overlap"] == 0
    assert result["query_gallery_identity_sets_equal"] is True


def test_partition_verifier_rejects_duplicate_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _module,
        "EXPECTED_INSHOP_PARTITION",
        {"train": (3, 2), "query": (2, 2), "gallery": (2, 2)},
    )
    train = [_example(1, "same"), _example(1, "same"), _example(2, "t-3")]
    query = [_example(10, "q-1"), _example(11, "q-2")]
    gallery = [_example(10, "g-1"), _example(11, "g-2")]
    with pytest.raises(ValueError, match="duplicate In-Shop train example IDs"):
        _module.verify_official_partition(
            SimpleNamespace(train=train, query=query, gallery=gallery)
        )
