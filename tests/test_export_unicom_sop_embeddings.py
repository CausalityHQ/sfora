from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_unicom_sop_embeddings.py"
_SPEC = importlib.util.spec_from_file_location("export_unicom_sop_embeddings", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _write_sop_fixture(root: Path) -> None:
    (root / "bicycle_final").mkdir(parents=True)
    (root / "chair_final").mkdir()
    for relative in (
        "bicycle_final/a.jpg",
        "bicycle_final/b.jpg",
        "chair_final/c.jpg",
        "chair_final/d.jpg",
    ):
        (root / relative).write_bytes(relative.encode())
    (root / "Ebay_train.txt").write_text(
        "image_id class_id super_class_id path\n"
        "1 1 1 bicycle_final/a.jpg\n"
        "2 1 1 bicycle_final/b.jpg\n",
        encoding="utf-8",
    )
    (root / "Ebay_test.txt").write_text(
        "image_id class_id super_class_id path\n3 2 2 chair_final/c.jpg\n4 2 2 chair_final/d.jpg\n",
        encoding="utf-8",
    )


def test_parse_sop_records_preserves_official_order_and_manifest(tmp_path: Path) -> None:
    _write_sop_fixture(tmp_path)

    records = _MODULE.parse_sop_records(
        tmp_path,
        expected_counts=(2, 2),
        expected_classes=(1, 1),
    )

    assert tuple(record.split for record in records) == ("train", "train", "test", "test")
    assert tuple(record.image_id for record in records) == (1, 2, 3, 4)
    assert tuple(record.label for record in records) == (1, 1, 2, 2)
    assert tuple(record.super_class_id for record in records) == (1, 1, 2, 2)
    assert tuple(record.relative_path for record in records) == (
        "bicycle_final/a.jpg",
        "bicycle_final/b.jpg",
        "chair_final/c.jpg",
        "chair_final/d.jpg",
    )
    authority = (
        b"train\0"
        b"1\0"
        b"1\0"
        b"1\0"
        b"bicycle_final/a.jpg\n"
        b"train\0"
        b"2\0"
        b"1\0"
        b"1\0"
        b"bicycle_final/b.jpg\n"
        b"test\0"
        b"3\0"
        b"2\0"
        b"2\0"
        b"chair_final/c.jpg\n"
        b"test\0"
        b"4\0"
        b"2\0"
        b"2\0"
        b"chair_final/d.jpg\n"
    )
    assert _MODULE.ordered_record_sha256(records) == hashlib.sha256(authority).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("bad-header", "header"),
        ("duplicate-path", "duplicate"),
        ("overlapping-class", "disjoint"),
        ("missing-image", "regular"),
        ("symlink-image", "regular"),
    ),
)
def test_parse_sop_records_rejects_protocol_drift(
    tmp_path: Path, mutation: str, match: str
) -> None:
    _write_sop_fixture(tmp_path)
    if mutation == "bad-header":
        (tmp_path / "Ebay_train.txt").write_text("wrong\n", encoding="utf-8")
    elif mutation == "duplicate-path":
        value = (tmp_path / "Ebay_test.txt").read_text(encoding="utf-8")
        (tmp_path / "Ebay_test.txt").write_text(
            value.replace("chair_final/c.jpg", "bicycle_final/a.jpg"), encoding="utf-8"
        )
    elif mutation == "overlapping-class":
        value = (tmp_path / "Ebay_test.txt").read_text(encoding="utf-8")
        (tmp_path / "Ebay_test.txt").write_text(
            value.replace("3 2 2", "3 1 2").replace("4 2 2", "4 1 2"), encoding="utf-8"
        )
    elif mutation == "missing-image":
        (tmp_path / "chair_final/c.jpg").unlink()
    else:
        image = tmp_path / "chair_final/c.jpg"
        image.unlink()
        image.symlink_to(tmp_path / "chair_final/d.jpg")

    with pytest.raises(ValueError, match=match):
        _MODULE.parse_sop_records(
            tmp_path,
            expected_counts=(2, 2),
            expected_classes=(1, 1),
        )


def test_export_sop_embeddings_publishes_exact_authenticated_arrays(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_sop_fixture(root)
    records = _MODULE.parse_sop_records(root, expected_counts=(2, 2), expected_classes=(1, 1))
    output = tmp_path / "bundle.npz"

    def encode(paths: tuple[Path, ...]) -> np.ndarray:
        return np.asarray(
            [[float(path.read_bytes()[0]), float(len(path.name)), 1.0] for path in paths],
            dtype=np.float32,
        )

    _MODULE.export_sop_embeddings(
        records,
        encode,
        {
            "model_identifier": "fixture-model",
            "model_revision": "ab" * 20,
            "checkpoint_sha256": "cd" * 32,
            "transform": "fixture transform",
        },
        output,
        batch_size=3,
        expected_counts=(2, 2),
        expected_classes=(1, 1),
    )

    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "metadata_json",
            "train_embeddings",
            "train_labels",
            "train_image_ids",
            "train_relative_paths",
            "test_embeddings",
            "test_labels",
            "test_image_ids",
            "test_relative_paths",
        }
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata["schema"] == "sfora-unicom-sop-embeddings-v1"
        assert metadata["ordered_record_sha256"] == _MODULE.ordered_record_sha256(records)
        assert metadata["embedding_dimension"] == 3
        assert metadata["split_counts"] == {"test": 2, "train": 2}
        assert metadata["split_classes"] == {"test": 1, "train": 1}
        assert archive["train_labels"].tolist() == [1, 1]
        assert archive["test_labels"].tolist() == [2, 2]
        assert archive["train_image_ids"].tolist() == [1, 2]
        assert archive["test_image_ids"].tolist() == [3, 4]
        assert archive["train_relative_paths"].tolist() == [
            "bicycle_final/a.jpg",
            "bicycle_final/b.jpg",
        ]
        assert archive["test_relative_paths"].tolist() == [
            "chair_final/c.jpg",
            "chair_final/d.jpg",
        ]
        for name, expected in metadata["array_sha256"].items():
            assert hashlib.sha256(archive[name].tobytes(order="C")).hexdigest() == expected
    _MODULE.load_sop_embedding_archive(
        output,
        expected_counts=(2, 2),
        expected_classes=(1, 1),
        expected_dimension=3,
    )


def test_export_sop_embeddings_rejects_existing_output_and_nonfinite_encoder(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_sop_fixture(root)
    records = _MODULE.parse_sop_records(root, expected_counts=(2, 2), expected_classes=(1, 1))
    output = tmp_path / "bundle.npz"
    metadata = {
        "model_identifier": "fixture-model",
        "model_revision": "ab" * 20,
        "checkpoint_sha256": "cd" * 32,
        "transform": "fixture transform",
    }

    def nonfinite(paths: tuple[Path, ...]) -> np.ndarray:
        return np.full((len(paths), 3), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="encoded batch"):
        _MODULE.export_sop_embeddings(
            records,
            nonfinite,
            metadata,
            output,
            expected_counts=(2, 2),
            expected_classes=(1, 1),
        )
    assert not output.exists()
    output.write_bytes(b"owned by caller")
    with pytest.raises(FileExistsError):
        _MODULE.export_sop_embeddings(
            records,
            nonfinite,
            metadata,
            output,
            expected_counts=(2, 2),
            expected_classes=(1, 1),
        )


@pytest.mark.parametrize(
    ("model", "identifier", "filename", "digest"),
    (
        (
            "b16",
            "UNICOM-ViT-B/16",
            "FP16-ViT-B-16.pt",
            "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef",
        ),
        (
            "l14-336",
            "UNICOM-ViT-L/14@336px",
            "FP16-ViT-L-14-336px.pt",
            "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
        ),
    ),
)
def test_model_authority_is_frozen(model: str, identifier: str, filename: str, digest: str) -> None:
    authority = _MODULE.model_authority(model)
    assert authority.identifier == identifier
    assert authority.checkpoint_filename == filename
    assert authority.checkpoint_sha256 == digest
    assert authority.revision == "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"


def test_parse_args_requires_explicit_local_export_surface() -> None:
    args = _MODULE.parse_args(
        [
            "--unicom-checkout",
            "/checkout",
            "--checkpoint",
            "/checkpoint.pt",
            "--dataset-root",
            "/dataset",
            "--model",
            "b16",
            "--output",
            "/output.npz",
            "--batch-size",
            "32",
            "--execute-export",
        ]
    )
    assert args.unicom_checkout == Path("/checkout")
    assert args.checkpoint == Path("/checkpoint.pt")
    assert args.dataset_root == Path("/dataset")
    assert args.output == Path("/output.npz")
    assert args.model == "b16"
    assert args.batch_size == 32
    assert args.execute_export is True

    with pytest.raises(SystemExit):
        _MODULE.parse_args(
            [
                "--unicom-checkout",
                "/checkout",
                "--checkpoint",
                "/checkpoint.pt",
                "--dataset-root",
                "/dataset",
                "--model",
                "b16",
                "--output",
                "/output.npz",
            ]
        )
    with pytest.raises(SystemExit):
        _MODULE.parse_args(["--unknown"])
