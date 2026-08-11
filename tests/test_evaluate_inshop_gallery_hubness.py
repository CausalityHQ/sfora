from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_inshop_gallery_hubness.py"
SPEC = importlib.util.spec_from_file_location("evaluate_inshop_ghc", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _unit(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _write_bundle(
    path: Path, embeddings: np.ndarray, labels: np.ndarray, *, prefix: str
) -> None:
    np.savez(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray(
            [f"{prefix}-{index}" for index in range(len(labels))], dtype=np.str_
        ),
    )


def test_loader_requires_exact_core_arrays_and_unit_finite_embeddings(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.npz"
    _write_bundle(valid, _unit([[1, 0], [0, 1]]), np.asarray([1, 2]), prefix="v")
    bundle = MODULE.load_embedding_bundle(valid)
    assert bundle.labels.tolist() == [1, 2]
    assert bundle.example_ids.tolist() == ["v-0", "v-1"]

    missing = tmp_path / "missing.npz"
    np.savez(missing, embeddings=_unit([[1, 0]]), labels=np.asarray([1]))
    with pytest.raises(ValueError, match="example_ids"):
        MODULE.load_embedding_bundle(missing)

    nonunit = tmp_path / "nonunit.npz"
    _write_bundle(
        nonunit,
        np.asarray([[2, 0]], dtype=np.float32),
        np.asarray([1]),
        prefix="n",
    )
    with pytest.raises(ValueError, match="unit-normalized"):
        MODULE.load_embedding_bundle(nonunit)


def test_train_split_keeps_all_nonquery_rows_as_distractors() -> None:
    labels = np.repeat(np.arange(6, dtype=np.int64), 3)
    angles = np.arange(labels.size, dtype=np.float32) * 0.17
    bundle = MODULE.EmbeddingBundle(
        embeddings=np.stack((np.cos(angles), np.sin(angles)), axis=1),
        labels=labels,
        example_ids=np.asarray([f"id-{i}" for i in range(labels.size)]),
    )
    first = MODULE.select_train_split(bundle, max_classes=4)
    second = MODULE.select_train_split(bundle, max_classes=4)
    assert first.query.example_ids.tolist() == second.query.example_ids.tolist()
    assert first.gallery.example_ids.tolist() == second.gallery.example_ids.tolist()
    assert first.query.embeddings.shape[0] == 4
    assert first.gallery.embeddings.shape[0] == labels.size - 4
    assert set(first.query.example_ids).isdisjoint(first.gallery.example_ids)
    assert set(first.query.labels).issubset(first.gallery.labels)
    assert {4, 5}.issubset(first.gallery.labels)


def test_tuner_rejects_a_saturated_train_split() -> None:
    train = MODULE.EmbeddingBundle(
        embeddings=_unit(
            [[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99], [-1, 0], [-0.99, 0.1]]
        ),
        labels=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        example_ids=np.asarray(["a", "b", "c", "d", "e", "f"]),
    )
    with pytest.raises(ValueError, match="saturated"):
        MODULE.tune_k(train, max_classes=3, k_values=(1, "all"), block_size=2)


def test_pair_evaluation_reports_real_and_permuted_control() -> None:
    query = MODULE.EmbeddingBundle(
        embeddings=_unit([[1, 0], [0, 1]]),
        labels=np.asarray([7, 8], dtype=np.int64),
        example_ids=np.asarray(["q0", "q1"]),
    )
    gallery = MODULE.EmbeddingBundle(
        embeddings=_unit([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]]),
        labels=np.asarray([7, 9, 8, 10], dtype=np.int64),
        example_ids=np.asarray(["g0", "g1", "g2", "g3"]),
    )
    result = MODULE.evaluate_pair(query, gallery, k=1, block_size=2, permutation_seed=9)
    assert list(result) == [
        "raw_recall_at_one",
        "corrected_recall_at_one",
        "absolute_gain",
        "permuted_density_control_recall_at_one",
        "hubness",
    ]
    assert result["raw_recall_at_one"] == pytest.approx(1.0)
    assert result["corrected_recall_at_one"] == pytest.approx(1.0)
    assert result["absolute_gain"] == pytest.approx(0.0)
    assert type(result["permuted_density_control_recall_at_one"]) is float
    assert result["hubness"]["incoming_count_sum"] == 2
    assert result["hubness"]["maximum_count"] == 1


def test_report_refuses_to_touch_test_pairs_after_saturated_tuning(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.npz"
    query_path = tmp_path / "query.npz"
    gallery_path = tmp_path / "gallery.npz"
    _write_bundle(
        train_path,
        _unit([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]]),
        np.asarray([0, 0, 1, 1]),
        prefix="train",
    )
    _write_bundle(
        query_path, _unit([[1, 0], [0, 1]]), np.asarray([7, 8]), prefix="query"
    )
    _write_bundle(
        gallery_path,
        _unit([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]]),
        np.asarray([7, 9, 8, 10]),
        prefix="gallery",
    )
    query_path.unlink()
    gallery_path.unlink()
    with pytest.raises(ValueError, match="saturated"):
        MODULE.build_report(
            train_path=train_path,
            evaluations={"published": (query_path, gallery_path)},
            max_classes=2,
            k_values=(1, "all"),
            block_size=2,
            minimum_gain=0.001,
        )


def test_atomic_writer_roundtrips_and_never_clobbers(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    payload = {"schema_version": "test", "value": 3}
    MODULE.write_json_atomic(destination, payload)
    assert json.loads(destination.read_text()) == payload
    before = destination.read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.write_json_atomic(destination, {"different": True})
    assert destination.read_bytes() == before
    assert list(tmp_path.glob(".result.json.*.tmp")) == []
