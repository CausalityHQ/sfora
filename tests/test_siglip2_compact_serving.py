"""Trained SigLIP2 serving contracts without requiring a GPU or model download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sfora.siglip2_compact_serving import (
    Siglip2CompactEncoder,
    Siglip2CompactIndex,
    _verified_official_gallery,
)


class BasisProcessor:
    def __call__(
        self, *, images: list[Image.Image], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {"pixel_values": torch.eye(1024)[: len(images)]}


class EchoVision(torch.nn.Module):
    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(pooler_output=pixel_values)


class RecordingGallery:
    def __init__(self) -> None:
        self.closed = False

    def search_packed(self, query: object) -> tuple[np.ndarray, np.ndarray]:
        assert query.codes.shape == (2, 128)
        assert int(query.codes[0, 0]) == 127
        assert int(query.codes[1, 1]) == 127
        return np.tile(np.arange(10), (2, 1)), np.ones((2, 10), dtype=np.float32)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("precision", ["fp32_autocast", "fp16_native"])
def test_serving_uses_trained_head_and_exact_packed_gallery(precision: str) -> None:
    head = torch.nn.Linear(1024, 128)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:, :128] = torch.eye(128)
        head.bias.zero_()
    vision = EchoVision().half() if precision == "fp16_native" else EchoVision()
    encoder = Siglip2CompactEncoder(BasisProcessor(), vision, head, precision, torch.device("cpu"))
    gallery = RecordingGallery()
    index = Siglip2CompactIndex(encoder, gallery)
    images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]
    ordinals, scores = index.search_images(images)
    assert ordinals.shape == (2, 10)
    assert scores.dtype == np.float32
    index.close()
    assert gallery.closed
    with pytest.raises(RuntimeError, match="closed"):
        index.search_images(images)


def test_serving_rejects_empty_query_batch() -> None:
    head = torch.nn.Linear(1024, 128)
    encoder = Siglip2CompactEncoder(
        BasisProcessor(), EchoVision(), head, "fp32_autocast", torch.device("cpu")
    )
    with pytest.raises(ValueError, match="nonempty"):
        encoder.encode_images([])


def test_serving_rejects_oversized_image_before_processing() -> None:
    head = torch.nn.Linear(1024, 128)
    encoder = Siglip2CompactEncoder(
        BasisProcessor(), EchoVision(), head, "fp32_autocast", torch.device("cpu")
    )
    with pytest.raises(ValueError, match="pixel limit"):
        encoder.encode_images([Image.new("RGB", (4097, 4097))])


def test_serving_releases_owned_encoder_and_rejects_search_after_close() -> None:
    head = torch.nn.Linear(1024, 128)
    encoder = Siglip2CompactEncoder(
        BasisProcessor(), EchoVision(), head, "fp32_autocast", torch.device("cpu")
    )
    gallery = RecordingGallery()
    index = Siglip2CompactIndex(encoder, gallery)
    index.close()
    assert gallery.closed
    assert index.encoder is None
    with pytest.raises(RuntimeError, match="closed"):
        index.search_images([Image.new("RGB", (2, 2))])


def test_artifact_loader_rejects_checkpoint_hash_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import PIL
    import torchvision
    import transformers

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    tileiras = tmp_path / "tileiras"
    tileiras.write_bytes(b"toolchain")
    monkeypatch.setenv("CUTILE_TILEIRAS_PATH", str(tileiras))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "sfora-sop-siglip2-compact-full-backbone-v1",
                "quality": {"gallery_wire_bytes_per_row": 130, "native_top10_exact": True},
                "checkpoint_sha256": "wrong",
                "train_embeddings_sha256": "wrong",
                "native_library_sha256": "wrong",
                "tileiras_sha256": hashlib.sha256(tileiras.read_bytes()).hexdigest(),
                "hardware": {
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "transformers": transformers.__version__,
                    "pillow": PIL.__version__,
                },
            }
        )
    )
    for name in ("checkpoint.pt", "embeddings.npy", "native.so"):
        (tmp_path / name).write_bytes(b"invalid")
    with pytest.raises(ValueError, match="receipt authority"):
        Siglip2CompactIndex.from_artifacts(
            model_snapshot=tmp_path,
            training_receipt=receipt,
            training_checkpoint=tmp_path / "checkpoint.pt",
            train_embeddings=tmp_path / "embeddings.npy",
            native_library=tmp_path / "native.so",
            expected_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        )


def test_artifact_loader_rejects_unpinned_receipt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    with pytest.raises(ValueError, match="receipt digest"):
        Siglip2CompactIndex.from_artifacts(
            model_snapshot=tmp_path,
            training_receipt=receipt,
            training_checkpoint=tmp_path / "checkpoint.pt",
            train_embeddings=tmp_path / "embeddings.npy",
            native_library=tmp_path / "native.so",
            expected_receipt_sha256="0" * 64,
        )


def test_official_gallery_must_match_checkpoint_precision_and_pinned_bytes(tmp_path: Path) -> None:
    embeddings = tmp_path / "test_embeddings.npy"
    np.save(embeddings, np.ones((10, 128), dtype=np.float32), allow_pickle=False)
    digest = hashlib.sha256(embeddings.read_bytes()).hexdigest()
    receipt = tmp_path / "official.json"
    payload = {
        "schema": "sfora-sop-siglip2-official-test-v1",
        "training_receipt_sha256": "a" * 64,
        "training_checkpoint_sha256": "b" * 64,
        "native_library_sha256": "c" * 64,
        "test_embeddings_sha256": digest,
        "queries": 10,
        "export_batch_size": 32,
        "inference_precision": "fp16_native",
        "public_first_batch_packed_exact": True,
        "native_top10_exact": True,
        "native_per_query_r1_equal": True,
        "gallery_wire_bytes_per_row": 130,
    }
    receipt.write_text(json.dumps(payload))
    expected = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert _verified_official_gallery(
        receipt, embeddings, expected, "a" * 64, "b" * 64, "c" * 64, "fp16_native"
    ) == (embeddings, 10)
    with pytest.raises(ValueError, match="official gallery authority"):
        _verified_official_gallery(
            receipt, embeddings, expected, "a" * 64, "b" * 64, "c" * 64, "fp32_autocast"
        )
    embeddings.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="official gallery authority"):
        _verified_official_gallery(
            receipt, embeddings, expected, "a" * 64, "b" * 64, "c" * 64, "fp16_native"
        )
