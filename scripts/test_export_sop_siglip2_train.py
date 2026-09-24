"""TRAIN-row authority checks for the SigLIP2 substrate screen."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_sop_siglip2_train import authenticated_train_rows, export_features


class TrainOnlyArchive(dict):
    def __getitem__(self, key):
        if key.startswith("test_"):
            raise AssertionError("TEST archive key was accessed")
        return super().__getitem__(key)


def test_train_rows_match_official_train_file_without_test_access(tmp_path):
    (tmp_path / "cat").mkdir()
    for name in ("one.jpg", "two.jpg"):
        (tmp_path / "cat" / name).write_bytes(b"image")
    (tmp_path / "Ebay_train.txt").write_text(
        "image_id class_id super_class_id path\n1 7 3 cat/one.jpg\n2 7 3 cat/two.jpg\n"
    )
    archive = TrainOnlyArchive(
        train_image_ids=np.array([1, 2], dtype=np.int64),
        train_labels=np.array([7, 7], dtype=np.int64),
        train_relative_paths=np.array(["cat/one.jpg", "cat/two.jpg"]),
    )
    rows = authenticated_train_rows(archive, tmp_path)
    assert [row.image_id for row in rows] == [1, 2]
    assert [row.image_path for row in rows] == [tmp_path / "cat/one.jpg", tmp_path / "cat/two.jpg"]


def test_train_row_order_mismatch_fails_before_model_load(tmp_path):
    (tmp_path / "cat").mkdir()
    for name in ("one.jpg", "two.jpg"):
        (tmp_path / "cat" / name).write_bytes(b"image")
    (tmp_path / "Ebay_train.txt").write_text(
        "image_id class_id super_class_id path\n1 7 3 cat/one.jpg\n2 7 3 cat/two.jpg\n"
    )
    archive = TrainOnlyArchive(
        train_image_ids=np.array([2, 1], dtype=np.int64),
        train_labels=np.array([7, 7], dtype=np.int64),
        train_relative_paths=np.array(["cat/two.jpg", "cat/one.jpg"]),
    )
    with pytest.raises(ValueError, match="row authority"):
        authenticated_train_rows(archive, tmp_path)


def test_feature_export_preserves_row_order_and_rejects_nonfinite(tmp_path):
    (tmp_path / "cat").mkdir()
    for name in ("one.jpg", "two.jpg"):
        (tmp_path / "cat" / name).write_bytes(b"image")
    (tmp_path / "Ebay_train.txt").write_text(
        "image_id class_id super_class_id path\n1 7 3 cat/one.jpg\n2 7 3 cat/two.jpg\n"
    )
    rows = authenticated_train_rows(
        TrainOnlyArchive(
            train_image_ids=np.array([1, 2]),
            train_labels=np.array([7, 7]),
            train_relative_paths=np.array(["cat/one.jpg", "cat/two.jpg"]),
        ),
        tmp_path,
    )
    out = tmp_path / "features.npy"
    export_features(
        rows,
        lambda batch: np.asarray([[float(row.image_id), 2.0] for row in batch], dtype=np.float32),
        out,
        width=2,
        batch_size=1,
    )
    np.testing.assert_array_equal(np.load(out), [[1.0, 2.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="nonfinite"):
        export_features(
            rows,
            lambda batch: np.full((len(batch), 2), np.nan, dtype=np.float32),
            tmp_path / "bad.npy",
            width=2,
            batch_size=2,
        )
    assert not (tmp_path / "bad.npy").exists()
