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


_NO_DUPLICATES = {
    "duplicate_groups": 0,
    "duplicate_rows": 0,
    "cross_identity_groups": 0,
    "cross_identity_rows": 0,
}


def _expect_clean_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _module,
        "EXPECTED_INSHOP_CONTENT_PROFILE",
        {"train": _NO_DUPLICATES, "query": _NO_DUPLICATES, "gallery": _NO_DUPLICATES},
    )
    monkeypatch.setattr(
        _module,
        "EXPECTED_INSHOP_CONTENT_OVERLAP",
        {
            "train_query": {"groups": 0, "rows": 0, "cross_identity_groups": 0},
            "train_gallery": {"groups": 0, "rows": 0, "cross_identity_groups": 0},
            "query_gallery": {"groups": 0, "rows": 0, "cross_identity_groups": 0},
        },
    )


def _example(
    label: int, suffix: str, root: Path, *, content: bytes | None = None
) -> SimpleNamespace:
    path = root / f"{suffix}.jpg"
    path.write_bytes(content if content is not None else suffix.encode())
    return SimpleNamespace(label=label, example_id=suffix, image=str(path))


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


def test_independent_sensitivity_reports_mixed_exact_tie() -> None:
    result = _module.independent_query_gallery_sensitivity(
        np.asarray([[1.0, 0.0]]),
        np.asarray([10]),
        np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        np.asarray([10, 20]),
    )
    assert result["canonical_float64_euclidean_recall_at_1"] == 1.0
    assert result["float64_cosine_recall_at_1"] == 1.0
    assert result["exact_tie_expected_recall_at_1"] == 0.5
    assert result["multiway_nearest_tie_queries"] == 1
    assert result["mixed_identity_nearest_tie_queries"] == 1


def test_partition_verifier_checks_all_three_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _module,
        "EXPECTED_INSHOP_PARTITION",
        {"train": (3, 2), "query": (2, 2), "gallery": (2, 2)},
    )
    _expect_clean_content(monkeypatch)
    train = [
        _example(1, "t-1", tmp_path),
        _example(1, "t-2", tmp_path),
        _example(2, "t-3", tmp_path),
    ]
    query = [_example(10, "q-1", tmp_path), _example(11, "q-2", tmp_path)]
    gallery = [_example(10, "g-1", tmp_path), _example(11, "g-2", tmp_path)]
    result = _module.verify_official_partition(
        SimpleNamespace(train=train, query=query, gallery=gallery)
    )
    assert result["train_query_identity_overlap"] == 0
    assert result["query_gallery_identity_sets_equal"] is True


def test_partition_verifier_rejects_duplicate_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _module,
        "EXPECTED_INSHOP_PARTITION",
        {"train": (3, 2), "query": (2, 2), "gallery": (2, 2)},
    )
    train = [
        _example(1, "same", tmp_path),
        _example(1, "same", tmp_path),
        _example(2, "t-3", tmp_path),
    ]
    query = [_example(10, "q-1", tmp_path), _example(11, "q-2", tmp_path)]
    gallery = [_example(10, "g-1", tmp_path), _example(11, "g-2", tmp_path)]
    with pytest.raises(ValueError, match="duplicate In-Shop train example IDs"):
        _module.verify_official_partition(
            SimpleNamespace(train=train, query=query, gallery=gallery)
        )


def test_partition_verifier_rejects_unexpected_content_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _module,
        "EXPECTED_INSHOP_PARTITION",
        {"train": (2, 2), "query": (2, 2), "gallery": (2, 2)},
    )
    _expect_clean_content(monkeypatch)
    train = [_example(1, "t-1", tmp_path), _example(2, "t-2", tmp_path)]
    query = [
        _example(10, "q-1", tmp_path, content=b"shared"),
        _example(11, "q-2", tmp_path),
    ]
    gallery = [
        _example(10, "g-1", tmp_path, content=b"shared"),
        _example(11, "g-2", tmp_path),
    ]
    with pytest.raises(ValueError, match="source-content overlap"):
        _module.verify_official_partition(
            SimpleNamespace(train=train, query=query, gallery=gallery)
        )
