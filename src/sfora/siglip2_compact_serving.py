"""Serve trained SigLIP2 compact image descriptors through exact packed top-10."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features

InferencePrecision = Literal["fp32_autocast", "fp16_native"]
_MAX_QUERY_IMAGES = 32
_MAX_IMAGE_PIXELS = 16_000_000


class PackedGallery(Protocol):
    def search_packed(
        self, queries: PackedInt8Embeddings
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]: ...

    def close(self) -> None: ...


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verified_official_gallery(
    receipt_path: Path,
    embeddings: Path,
    expected_receipt_sha256: str,
    training_receipt_sha256: str,
    checkpoint_sha256: str,
    native_library_sha256: str,
    precision: InferencePrecision,
) -> tuple[Path, int]:
    """Bind a pinned official gallery to the loaded checkpoint and query arithmetic."""

    if (
        not isinstance(receipt_path, Path)
        or not isinstance(embeddings, Path)
        or not isinstance(expected_receipt_sha256, str)
        or len(expected_receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_receipt_sha256)
        or not receipt_path.is_file()
        or not embeddings.is_file()
        or _sha256(receipt_path) != expected_receipt_sha256
    ):
        raise ValueError("trained SigLIP2 official gallery authority differs")
    receipt = json.loads(receipt_path.read_text())
    rows = receipt.get("queries")
    if (
        receipt.get("schema") != "sfora-sop-siglip2-official-test-v1"
        or receipt.get("training_receipt_sha256") != training_receipt_sha256
        or receipt.get("training_checkpoint_sha256") != checkpoint_sha256
        or receipt.get("native_library_sha256") != native_library_sha256
        or receipt.get("test_embeddings_sha256") != _sha256(embeddings)
        or precision != "fp16_native"
        or receipt.get("inference_precision") != precision
        or receipt.get("export_batch_size") != 32
        or receipt.get("public_first_batch_packed_exact") is not True
        or receipt.get("native_top10_exact") is not True
        or receipt.get("native_per_query_r1_equal") is not True
        or receipt.get("gallery_wire_bytes_per_row") != 130
        or type(rows) is not int
        or rows < 10
    ):
        raise ValueError("trained SigLIP2 official gallery authority differs")
    return embeddings, rows


class Siglip2CompactEncoder:
    """A trained 1024→128 SigLIP2 image encoder with an explicit precision mode."""

    def __init__(
        self,
        processor: Any,
        vision: nn.Module,
        head: nn.Linear,
        precision: InferencePrecision,
        device: torch.device,
    ) -> None:
        if (
            precision not in ("fp32_autocast", "fp16_native")
            or type(device) is not torch.device
            or type(head) is not nn.Linear
            or head.in_features != 1024
            or head.out_features != 128
            or not callable(processor)
            or not isinstance(vision, nn.Module)
        ):
            raise ValueError("trained SigLIP2 serving geometry differs")
        self.processor = processor
        self.vision = vision.eval()
        self.head = head.eval()
        self.precision = precision
        self.device = device

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Any]) -> PackedInt8Embeddings:
        """Return wire-compatible 128 signed bytes plus one f16 inverse norm per image."""

        from PIL import Image

        if self.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("trained SigLIP2 serving requires CUDA TF32 disabled")
        if not images or any(not isinstance(image, Image.Image) for image in images):
            raise ValueError("trained SigLIP2 serving needs a nonempty PIL image batch")
        if len(images) > _MAX_QUERY_IMAGES:
            raise ValueError("trained SigLIP2 serving batch limit is 32 images")
        if any(image.width * image.height > _MAX_IMAGE_PIXELS for image in images):
            raise ValueError("trained SigLIP2 serving pixel limit is 16000000")
        batch = self.processor(
            images=[image.convert("RGB") for image in images], return_tensors="pt"
        )
        if set(batch) != {"pixel_values"}:
            raise ValueError("trained SigLIP2 serving processor geometry differs")
        pixels = batch["pixel_values"].to(device=self.device)
        if self.precision == "fp16_native":
            pixels = pixels.to(dtype=torch.float16)
        with torch.amp.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.precision == "fp32_autocast" and self.device.type == "cuda",
        ):
            pooled = self.vision(pixel_values=pixels).pooler_output
        if pooled is None or pooled.shape != (len(images), 1024):
            raise ValueError("trained SigLIP2 serving pooler geometry differs")
        features = F.normalize(compact_head_features(pooled, self.head), dim=1).cpu()
        if not bool(torch.isfinite(features).all()):
            raise ValueError("trained SigLIP2 serving features are nonfinite")
        return pack_int8_unit_embeddings(features)


class Siglip2CompactIndex:
    """Own an encoder and persistent exact native gallery for external image queries."""

    def __init__(self, encoder: Siglip2CompactEncoder, gallery: PackedGallery) -> None:
        self.encoder: Siglip2CompactEncoder | None = encoder
        self.gallery: PackedGallery | None = gallery
        self._closed = False
        self._lifecycle_lock = threading.RLock()

    @classmethod
    def from_artifacts(
        cls,
        *,
        model_snapshot: Path,
        training_receipt: Path,
        training_checkpoint: Path,
        train_embeddings: Path,
        native_library: Path,
        expected_receipt_sha256: str,
        precision: InferencePrecision = "fp16_native",
        official_gallery_receipt: Path | None = None,
        official_gallery_embeddings: Path | None = None,
        expected_official_gallery_receipt_sha256: str | None = None,
    ) -> Siglip2CompactIndex:
        """Load trusted artifacts bound to an independently pinned receipt digest."""

        if (
            len(expected_receipt_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_receipt_sha256)
            or not training_receipt.is_file()
            or _sha256(training_receipt) != expected_receipt_sha256
        ):
            raise ValueError("trained SigLIP2 serving receipt digest differs")
        if precision not in ("fp32_autocast", "fp16_native") or not torch.cuda.is_available():
            raise ValueError("trained SigLIP2 serving needs CUDA and a supported precision")
        if any(
            not isinstance(path, Path) or not path.is_file()
            for path in (
                training_receipt,
                training_checkpoint,
                train_embeddings,
                native_library,
            )
        ):
            raise ValueError("trained SigLIP2 serving artifacts differ")
        receipt = json.loads(training_receipt.read_text())
        import PIL
        import torchvision
        import transformers

        stack = receipt.get("hardware", {})
        tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
        if (
            stack.get("torch") != torch.__version__
            or stack.get("torchvision") != torchvision.__version__
            or stack.get("transformers") != transformers.__version__
            or stack.get("pillow") != PIL.__version__
            or not tileiras
            or not Path(tileiras).is_file()
            or _sha256(Path(tileiras)) != receipt.get("tileiras_sha256")
            or torch.backends.cuda.matmul.allow_tf32
        ):
            raise ValueError("trained SigLIP2 serving runtime differs")
        model_hashes = receipt.get("model_file_sha256", {})
        if (
            receipt.get("schema") != "sfora-sop-siglip2-compact-full-backbone-v1"
            or receipt.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
            or receipt.get("quality", {}).get("native_top10_exact") is not True
            or receipt.get("checkpoint_sha256") != _sha256(training_checkpoint)
            or receipt.get("train_embeddings_sha256") != _sha256(train_embeddings)
            or receipt.get("native_library_sha256") != _sha256(native_library)
            or not isinstance(model_hashes, dict)
            or set(model_hashes) != {"config.json", "preprocessor_config.json", "model.safetensors"}
            or any(
                _sha256(model_snapshot / name) != digest for name, digest in model_hashes.items()
            )
        ):
            raise ValueError("trained SigLIP2 serving receipt authority differs")
        official_args = (
            official_gallery_receipt,
            official_gallery_embeddings,
            expected_official_gallery_receipt_sha256,
        )
        if any(value is not None for value in official_args):
            if (
                not isinstance(official_gallery_receipt, Path)
                or not isinstance(official_gallery_embeddings, Path)
                or not isinstance(expected_official_gallery_receipt_sha256, str)
            ):
                raise ValueError("trained SigLIP2 official gallery authority differs")
            gallery_path, gallery_rows = _verified_official_gallery(
                official_gallery_receipt,
                official_gallery_embeddings,
                expected_official_gallery_receipt_sha256,
                expected_receipt_sha256,
                receipt["checkpoint_sha256"],
                receipt["native_library_sha256"],
                precision,
            )
        else:
            gallery_path, gallery_rows = train_embeddings, receipt.get("gallery_images")
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            model_snapshot, local_files_only=True, backend="torchvision"
        )
        if (
            type(processor).__name__ != "SiglipImageProcessor"
            or processor.size.get("height") != 256
            or processor.size.get("width") != 256
            or processor.resample != 2
        ):
            raise ValueError("trained SigLIP2 serving processor differs")
        model = AutoModel.from_pretrained(
            model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        device = torch.device("cuda:0")
        vision = model.vision_model.float().to(device).eval()
        del model
        head = nn.Linear(1024, 128).to(device).eval()
        checkpoint = torch.load(training_checkpoint, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("arm") != receipt.get("arm")
            or checkpoint.get("seed") != receipt.get("seed")
            or checkpoint.get("updates") != receipt.get("updates")
        ):
            raise ValueError("trained SigLIP2 serving checkpoint identity differs")
        vision.load_state_dict(checkpoint["vision"], strict=True)
        head.load_state_dict(checkpoint["head"], strict=True)
        if precision == "fp16_native":
            vision.half()
        values = np.load(gallery_path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape != (gallery_rows, 128):
            raise ValueError("trained SigLIP2 serving gallery geometry differs")
        packed_gallery = pack_int8_unit_embeddings(torch.from_numpy(np.asarray(values).copy()))
        gallery = CutilePackedInt8Gallery.open_packed(native_library, packed_gallery)
        encoder = Siglip2CompactEncoder(processor, vision, head, precision, device)
        return cls(encoder, gallery)

    def search_images(self, images: Sequence[Any]) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Return top-10 gallery ordinals and scores; queries must be external images."""

        with self._lifecycle_lock:
            if self._closed or self.encoder is None or self.gallery is None:
                raise RuntimeError("trained SigLIP2 index is closed")
            return self.gallery.search_packed(self.encoder.encode_images(images))

    def close(self) -> None:
        with self._lifecycle_lock:
            if not self._closed:
                self._closed = True
                try:
                    if self.gallery is not None:
                        self.gallery.close()
                finally:
                    self.gallery = None
                    self.encoder = None

    def __enter__(self) -> Siglip2CompactIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["InferencePrecision", "Siglip2CompactEncoder", "Siglip2CompactIndex"]
