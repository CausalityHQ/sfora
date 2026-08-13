from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sfora.unicom_audit_io import load_embedding_bundle

SCRIPT = Path(__file__).parents[1] / "scripts/export_unicom_inshop_embeddings.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("export_unicom_inshop_embeddings", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _records(module, root: Path):
    records = []
    for split, labels in (
        ("train", ["a", "a", "b", "b"]),
        ("query", ["x", "y"]),
        ("gallery", ["x", "y", "x"]),
    ):
        for index, label in enumerate(labels):
            path = root / f"{split}-{index}.jpg"
            path.write_bytes(bytes([len(records) + 1]))
            records.append(module.InshopRecord(split=split, image_path=path, label=label))
    return tuple(records)


def _metadata(model_identifier: str = "UNICOM-ViT-B/16") -> dict[str, object]:
    return {
        "model_identifier": model_identifier,
        "model_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "image_list_sha256": "c" * 64,
        "transform": "official-unicom-eval-v1",
    }


def test_parse_partition_preserves_official_row_order_and_labels(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "dataset"
    (root / "Eval").mkdir(parents=True)
    (root / "Img" / "img").mkdir(parents=True)
    rows = [
        ("img/a.jpg", "item_2", "query"),
        ("img/b.jpg", "item_1", "train"),
        ("img/c.jpg", "item_2", "gallery"),
    ]
    for name, _, _ in rows:
        (root / "Img" / name).write_bytes(b"pixel")
    (root / "Eval" / "list_eval_partition.txt").write_text(
        "3\nimage_name item_id evaluation_status\n"
        + "\n".join(" ".join(row) for row in rows)
        + "\n"
    )

    records = module.parse_inshop_partition(root, expected_counts=None)

    assert [(row.split, row.image_path.name, row.label) for row in records] == [
        ("query", "a.jpg", "item_2"),
        ("train", "b.jpg", "item_1"),
        ("gallery", "c.jpg", "item_2"),
    ]


def test_export_embeddings_batches_in_official_order_and_roundtrips(tmp_path: Path) -> None:
    module = _load_script()
    records = _records(module, tmp_path)
    observed: list[str] = []

    def encode_batch(paths: tuple[Path, ...]) -> np.ndarray:
        observed.extend(path.name for path in paths)
        rows = []
        for path in paths:
            value = float(path.read_bytes()[0])
            rows.append([value, 1.0, value / 2.0])
        return np.asarray(rows, dtype=np.float32)

    output = tmp_path / "bundle.npz"
    module.export_embeddings(
        records,
        encode_batch,
        _metadata(),
        output,
        batch_size=2,
        expected_counts=(4, 2, 3),
    )

    assert observed == [record.image_path.name for record in records]
    bundle = load_embedding_bundle(
        output, expected_counts=(4, 2, 3), expected_dimension=3
    )
    assert bundle.train_labels.tolist() == ["a", "a", "b", "b"]
    assert bundle.query_labels.tolist() == ["x", "y"]
    assert bundle.gallery_labels.tolist() == ["x", "y", "x"]
    assert json.loads(str(np.load(output, allow_pickle=False)["metadata_json"]))[
        "embedding_dimension"
    ] == 3
    assert "torch" not in module.__dict__


def test_export_is_no_clobber_and_preserves_existing_bytes(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "bundle.npz"
    output.write_bytes(b"existing")
    records = _records(module, tmp_path)

    with pytest.raises(FileExistsError):
        module.export_embeddings(
            records,
            lambda paths: np.ones((len(paths), 3), dtype=np.float32),
            _metadata(),
            output,
            batch_size=2,
            expected_counts=(4, 2, 3),
        )
    assert output.read_bytes() == b"existing"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_export_rejects_nonfinite_or_wrong_batch_rows(tmp_path: Path) -> None:
    module = _load_script()
    records = _records(module, tmp_path)

    with pytest.raises(ValueError, match="batch"):
        module.export_embeddings(
            records,
            lambda paths: np.full((len(paths) - 1, 3), np.nan, dtype=np.float32),
            _metadata(),
            tmp_path / "bundle.npz",
            batch_size=2,
            expected_counts=(4, 2, 3),
        )


def test_official_encoder_loads_named_architecture_from_checkpoint_directory(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-B-16.pt"
    checkpoint.write_bytes(b"checkpoint")
    observed: list[tuple[object, ...]] = []

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: observed.append((*args, kwargs))
        or (Model(), object()),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    module._official_encoder(checkout, checkpoint)

    assert observed == [
        ("ViT-B/16", {"download_root": str(checkpoint.parent)})
    ]


def test_official_encoder_rejects_checkpoint_filename_upstream_would_ignore(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "renamed-unicom.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="FP16-ViT-B-16.pt"):
        module._official_encoder(tmp_path / "checkout", checkpoint)


def test_l14_336_model_spec_binds_official_name_file_and_digest() -> None:
    module = _load_script()

    spec = module._model_spec("ViT-L/14@336px")

    assert spec.model_identifier == "UNICOM-ViT-L/14@336px"
    assert spec.checkpoint_filename == "FP16-ViT-L-14-336px.pt"
    assert spec.checkpoint_sha256 == (
        "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
    )


def test_official_encoder_loads_l14_336_from_matching_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"checkpoint")
    observed: list[tuple[object, ...]] = []

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: observed.append((*args, kwargs))
        or (Model(), object()),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    module._official_encoder(
        checkout, checkpoint, model_name="ViT-L/14@336px"
    )

    assert observed == [
        ("ViT-L/14@336px", {"download_root": str(checkpoint.parent)})
    ]


def test_export_metadata_accepts_registered_l14_336_identifier(tmp_path: Path) -> None:
    module = _load_script()
    records = _records(module, tmp_path)

    module.export_embeddings(
        records,
        lambda paths: np.ones((len(paths), 3), dtype=np.float32),
        _metadata("UNICOM-ViT-L/14@336px"),
        tmp_path / "bundle.npz",
        batch_size=2,
        expected_counts=(4, 2, 3),
    )
