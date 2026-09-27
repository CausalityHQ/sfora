"""Trained SigLIP2 serving contracts without requiring a GPU or model download."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings
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


def test_encoder_loads_pinned_checkpoint_for_custom_gallery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WeightVision(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(pooler_output=pixel_values * self.scale)

    class SiglipImageProcessor(BasisProcessor):
        size = {"height": 256, "width": 256}
        resample = 2

    files = {}
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "processor_config.json").write_text('{"image_processor":{"image_mean":[0,0,0]}}')
    vision = WeightVision()
    with torch.no_grad():
        vision.scale.fill_(2)
    head = torch.nn.Linear(1024, 128)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, 0] = 1
        head.bias.zero_()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"vision": vision.state_dict(), "head": head.state_dict()}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fake_transformers = ModuleType("transformers")
    processor_paths: list[Path] = []

    def processor_from_pretrained(path: Path, **_kwargs: object) -> SiglipImageProcessor:
        processor_paths.append(path)
        return SiglipImageProcessor()

    fake_transformers.AutoImageProcessor = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=processor_from_pretrained
    )
    fake_transformers.AutoConfig = SimpleNamespace(  # type: ignore[attr-defined]
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(vision_config=None)
    )
    fake_transformers.SiglipVisionModel = lambda config: WeightVision()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    with pytest.raises(ValueError, match="checkpoint digest"):
        Siglip2CompactEncoder.from_checkpoint(
            model_snapshot=tmp_path,
            checkpoint=checkpoint,
            expected_checkpoint_sha256="0" * 64,
            model_file_sha256=files,
            precision="fp32_autocast",
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="model snapshot digest"):
        Siglip2CompactEncoder.from_checkpoint(
            model_snapshot=tmp_path,
            checkpoint=checkpoint,
            expected_checkpoint_sha256=digest,
            model_file_sha256={**files, "config.json": "0" * 64},
            precision="fp32_autocast",
            device=torch.device("cpu"),
        )
    encoder = Siglip2CompactEncoder.from_checkpoint(
        model_snapshot=tmp_path,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=digest,
        model_file_sha256=files,
        precision="fp32_autocast",
        device=torch.device("cpu"),
    )
    assert processor_paths == [tmp_path / "preprocessor_config.json"]
    assert torch.equal(encoder.vision.state_dict()["scale"], vision.state_dict()["scale"])
    assert torch.equal(encoder.head.weight, head.weight)
    assert encoder.encode_images([Image.new("RGB", (2, 2))]).codes.shape == (1, 128)
    with torch.no_grad():
        head.weight[0, 0] = float("nan")
    nonfinite = tmp_path / "nonfinite.pt"
    torch.save({"vision": vision.state_dict(), "head": head.state_dict()}, nonfinite)
    with pytest.raises(ValueError, match="checkpoint weights"):
        Siglip2CompactEncoder.from_checkpoint(
            model_snapshot=tmp_path,
            checkpoint=nonfinite,
            expected_checkpoint_sha256=hashlib.sha256(nonfinite.read_bytes()).hexdigest(),
            model_file_sha256=files,
            precision="fp32_autocast",
            device=torch.device("cpu"),
        )


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


def test_cuda_graph_opt_in_requires_native_fp16_cuda() -> None:
    with pytest.raises(ValueError, match="CUDA graph requires native FP16 CUDA"):
        Siglip2CompactEncoder(
            BasisProcessor(),
            EchoVision(),
            torch.nn.Linear(1024, 128),
            "fp32_autocast",
            torch.device("cpu"),
            cuda_graph_batch1=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph support")
def test_cuda_graph_capture_ignores_ambient_autocast_and_preserves_eager_output() -> None:
    class Processor:
        def __call__(
            self, *, images: list[Image.Image], return_tensors: str
        ) -> dict[str, torch.Tensor]:
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((len(images), 3, 256, 256))}

    class Vision(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, 1024, bias=False).half()

        def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(pooler_output=self.projection(pixel_values.mean(dim=(2, 3))))

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    vision = Vision().cuda().eval()
    head = torch.nn.Linear(1024, 128).cuda().eval()
    processor = Processor()
    eager = Siglip2CompactEncoder(processor, vision, head, "fp16_native", torch.device("cuda:0"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        graph = Siglip2CompactEncoder(
            processor, vision, head, "fp16_native", torch.device("cuda:0"), cuda_graph_batch1=True
        )
    assert graph._batch1_graph is not None
    assert graph._batch1_graph[2].dtype == torch.float16
    for count in (1, 2):
        images = [Image.new("RGB", (2, 2)) for _ in range(count)]
        first, second = eager.encode_images(images), graph.encode_images(images)
        assert torch.equal(first.codes, second.codes)
        assert torch.equal(first.inverse_norms, second.inverse_norms)


def test_cuda_graph_rejects_broadcastable_pixel_shape() -> None:
    class WrongShapeProcessor:
        def __call__(
            self, *, images: list[Image.Image], return_tensors: str
        ) -> dict[str, torch.Tensor]:
            return {"pixel_values": torch.ones((len(images), 3, 1, 1))}

    encoder = Siglip2CompactEncoder(
        WrongShapeProcessor(),
        EchoVision(),
        torch.nn.Linear(1024, 128),
        "fp16_native",
        torch.device("cpu"),
    )
    encoder._batch1_graph = (
        torch.zeros((1, 3, 256, 256)),
        cast(torch.cuda.CUDAGraph, None),
        torch.zeros((1, 1024)),
    )
    with pytest.raises(ValueError, match="pixel geometry"):
        encoder.encode_images([Image.new("RGB", (2, 2))])


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


def test_custom_image_gallery_uses_bounded_encoder_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    head = torch.nn.Linear(1024, 128)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[:, :128] = torch.eye(128)
        head.bias.zero_()
    encoder = Siglip2CompactEncoder(
        BasisProcessor(), EchoVision(), head, "fp32_autocast", torch.device("cpu")
    )
    paths = []
    for row in range(33):
        path = tmp_path / f"{row}.png"
        Image.new("RGB", (2, 2), (row, 0, 0)).save(path)
        paths.append(path)
    gallery = RecordingGallery()
    observed: list[PackedInt8Embeddings] = []

    def open_packed(_library: Path, packed: PackedInt8Embeddings) -> RecordingGallery:
        observed.append(packed)
        return gallery

    monkeypatch.setattr(CutilePackedInt8Gallery, "open_packed", open_packed)
    (tmp_path / "native.so").write_bytes(b"test backend")
    index = Siglip2CompactIndex.from_image_paths(
        encoder=encoder, native_library=tmp_path / "native.so", image_paths=paths
    )
    assert len(observed) == 1
    assert observed[0].codes.shape == (33, 128)
    assert observed[0].inverse_norms.shape == (33,)
    assert [int(row.argmax()) for row in observed[0].codes] == [row % 32 for row in range(33)]
    index.close()
    assert gallery.closed


def test_custom_gallery_rejects_invalid_files_before_encoding(tmp_path: Path) -> None:
    encoder = Siglip2CompactEncoder(
        BasisProcessor(),
        EchoVision(),
        torch.nn.Linear(1024, 128),
        "fp32_autocast",
        torch.device("cpu"),
    )
    missing = (tmp_path / "missing.png",) * 10
    with pytest.raises(ValueError, match="custom gallery authority"):
        Siglip2CompactIndex.from_image_paths(
            encoder=encoder, native_library=Path("relative.so"), image_paths=missing
        )
    library = tmp_path / "native.so"
    library.write_bytes(b"test backend")
    with pytest.raises(ValueError, match="custom gallery authority"):
        Siglip2CompactIndex.from_image_paths(
            encoder=encoder, native_library=library, image_paths=missing
        )


def test_custom_gallery_rejects_unpinned_native_binary(tmp_path: Path) -> None:
    encoder = Siglip2CompactEncoder(
        BasisProcessor(),
        EchoVision(),
        torch.nn.Linear(1024, 128),
        "fp32_autocast",
        torch.device("cpu"),
    )
    library = tmp_path / "native.so"
    library.write_bytes(b"candidate")
    image = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(image)
    with pytest.raises(ValueError, match="custom gallery authority"):
        Siglip2CompactIndex.from_image_paths(
            encoder=encoder,
            native_library=library,
            image_paths=[image] * 10,
            expected_native_library_sha256="0" * 64,
        )


def test_custom_gallery_rejects_mixed_official_selection_before_model_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="custom gallery selection"):
        Siglip2CompactIndex.from_artifacts(
            model_snapshot=tmp_path,
            training_receipt=tmp_path / "missing.json",
            training_checkpoint=tmp_path / "missing.pt",
            train_embeddings=tmp_path / "missing.npy",
            native_library=tmp_path / "missing.so",
            expected_receipt_sha256="0" * 64,
            custom_gallery_image_paths=(tmp_path,) * 10,
            official_gallery_receipt=tmp_path / "official.json",
        )


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
