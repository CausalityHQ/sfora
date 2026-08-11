from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_inshop_reciprocal_reranking.py"
SPEC = importlib.util.spec_from_file_location("evaluate_inshop_rsr", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_bundle(
    path: Path,
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    prefix: str,
) -> None:
    np.savez(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray(
            [f"{prefix}-{index}" for index in range(len(labels))], dtype=np.str_
        ),
    )


def _unit(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_loader_requires_exact_core_arrays_and_unit_finite_embeddings(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npz"
    _write_bundle(valid, _unit([[1, 0], [0, 1]]), np.asarray([1, 2]), prefix="v")
    bundle = MODULE.load_embedding_bundle(valid)
    assert bundle.embeddings.shape == (2, 2)
    assert bundle.labels.tolist() == [1, 2]
    assert bundle.example_ids.tolist() == ["v-0", "v-1"]

    missing = tmp_path / "missing.npz"
    np.savez(missing, embeddings=_unit([[1, 0]]), labels=np.asarray([1]))
    with pytest.raises(ValueError, match="example_ids"):
        MODULE.load_embedding_bundle(missing)

    nonunit = tmp_path / "nonunit.npz"
    _write_bundle(nonunit, np.asarray([[2, 0]], dtype=np.float32), np.asarray([1]), prefix="n")
    with pytest.raises(ValueError, match="unit-normalized"):
        MODULE.load_embedding_bundle(nonunit)

    nonfinite = tmp_path / "nonfinite.npz"
    _write_bundle(
        nonfinite,
        np.asarray([[np.nan, 0]], dtype=np.float32),
        np.asarray([1]),
        prefix="f",
    )
    with pytest.raises(ValueError, match="finite"):
        MODULE.load_embedding_bundle(nonfinite)


def test_train_split_is_deterministic_disjoint_and_has_one_query_per_class(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.npz"
    labels = np.repeat(np.arange(6, dtype=np.int64), 3)
    angles = np.arange(labels.size, dtype=np.float32) * 0.17
    _write_bundle(
        path,
        np.stack((np.cos(angles), np.sin(angles)), axis=1),
        labels,
        prefix="train",
    )
    bundle = MODULE.load_embedding_bundle(path)

    first = MODULE.select_train_split(bundle, max_classes=4)
    second = MODULE.select_train_split(bundle, max_classes=4)

    assert first.query.example_ids.tolist() == second.query.example_ids.tolist()
    assert first.gallery.example_ids.tolist() == second.gallery.example_ids.tolist()
    assert first.query.embeddings.shape[0] == 4
    assert set(first.query.example_ids).isdisjoint(first.gallery.example_ids)
    assert set(first.query.labels) == set(first.gallery.labels)
    assert all(np.count_nonzero(first.query.labels == label) == 1 for label in first.query.labels)


def test_recall_at_one_uses_labels_at_top_indices() -> None:
    assert MODULE.recall_at_one(
        np.asarray([10, 20, 30], dtype=np.int64),
        np.asarray([30, 10, 20], dtype=np.int64),
        np.asarray([[1], [2], [0]], dtype=np.int64),
    ) == pytest.approx(1.0)


def test_tuner_uses_only_train_bundle_and_breaks_ties_lexicographically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = MODULE.EmbeddingBundle(
        embeddings=_unit([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]]),
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        example_ids=np.asarray(["a", "b", "c", "d"], dtype=np.str_),
    )
    split = MODULE.TrainSplit(
        query=MODULE.EmbeddingBundle(
            train.embeddings[[0, 2]], train.labels[[0, 2]], train.example_ids[[0, 2]]
        ),
        gallery=MODULE.EmbeddingBundle(
            train.embeddings[[1, 3]], train.labels[[1, 3]], train.example_ids[[1, 3]]
        ),
    )
    monkeypatch.setattr(MODULE, "select_train_split", lambda bundle, max_classes: split)

    selected, rows = MODULE.tune_parameters(
        train,
        max_classes=2,
        k_values=(1,),
        candidate_depth_values=(1, 2),
        blend_values=(0.1, 0.5),
        block_size=1,
    )

    assert selected == {"k": 1, "candidate_depth": 1, "blend": 0.1}
    assert [(row["k"], row["candidate_depth"], row["blend"]) for row in rows] == [
        (1, 1, 0.1),
        (1, 1, 0.5),
        (1, 2, 0.1),
        (1, 2, 0.5),
    ]


def test_atomic_writer_is_no_clobber_and_roundtrips(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    payload = {"schema_version": "test", "value": 3}
    MODULE.write_json_atomic(destination, payload)
    assert json.loads(destination.read_text()) == payload

    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.write_json_atomic(destination, {"different": True})
    assert destination.read_bytes() == before
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_evaluation_report_binds_inputs_metrics_grid_and_decision(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    query_path = tmp_path / "query.npz"
    gallery_path = tmp_path / "gallery.npz"
    train_labels = np.repeat(np.arange(3, dtype=np.int64), 3)
    train_vectors = []
    for label in train_labels:
        angle = float(label) * 2.0
        train_vectors.append([np.cos(angle), np.sin(angle)])
    _write_bundle(train_path, _unit(train_vectors), train_labels, prefix="train")
    _write_bundle(query_path, _unit([[1, 0], [0, 1]]), np.asarray([7, 8]), prefix="query")
    _write_bundle(
        gallery_path,
        _unit([[0.99, 0.1], [0.1, 0.99], [-1, 0]]),
        np.asarray([7, 8, 9]),
        prefix="gallery",
    )

    report = MODULE.build_report(
        train_path=train_path,
        evaluations={"published": (query_path, gallery_path)},
        max_classes=3,
        k_values=(1,),
        candidate_depth_values=(2,),
        blend_values=(0.1,),
        block_size=2,
        minimum_gain=0.0015,
    )

    assert list(report) == [
        "schema_version",
        "inputs",
        "tuning",
        "evaluations",
        "minimum_gain",
        "passes_falsifier",
    ]
    assert len(report["inputs"]["train_sha256"]) == 64
    assert len(report["inputs"]["evaluations"]["published"]["query_sha256"]) == 64
    assert report["tuning"]["selected"] == {"k": 1, "candidate_depth": 2, "blend": 0.1}
    assert report["evaluations"]["published"]["raw_recall_at_one"] == pytest.approx(1.0)
    assert type(report["passes_falsifier"]) is bool
