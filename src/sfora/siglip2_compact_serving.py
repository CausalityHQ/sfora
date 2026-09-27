"""Serve trained SigLIP2 compact image descriptors through exact packed top-10."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
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
_MAX_IMAGE_PIXELS = 16_777_216
# ponytail: cap preprocessing source pixels; tune only after measuring gallery RSS and latency.
_MAX_PREPROCESS_BATCH_PIXELS = 64_000_000
_DIRECT_PROCESSOR_SHA = "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff"


def _direct_processor_supported(digest: str) -> bool:
    import torchvision
    import transformers

    return (
        digest == _DIRECT_PROCESSOR_SHA
        and transformers.__version__ == "5.12.1"
        and torchvision.__version__ == "0.27.1+cu130"
        and torch.__version__ == "2.12.1+cu130"
    )


def _direct_preprocess(image: Any) -> torch.Tensor:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as tvf

    pixels = tvf.pil_to_tensor(image.convert("RGB")).unsqueeze(0)
    pixels = tvf.resize(pixels, [256, 256], InterpolationMode.BILINEAR, antialias=True)
    return tvf.normalize(pixels.float(), [127.5] * 3, [127.5] * 3)  # type: ignore[no-any-return]


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
        cuda_graph_batch1: bool = False,
        _use_direct_preprocess: bool = False,
    ) -> None:
        if (
            precision not in ("fp32_autocast", "fp16_native")
            or type(device) is not torch.device
            or type(head) is not nn.Linear
            or head.in_features != 1024
            or head.out_features != 128
            or not callable(processor)
            or not isinstance(vision, nn.Module)
            or type(cuda_graph_batch1) is not bool
            or type(_use_direct_preprocess) is not bool
        ):
            raise ValueError("trained SigLIP2 serving geometry differs")
        if cuda_graph_batch1 and (
            precision != "fp16_native"
            or device.type != "cuda"
            or not torch.cuda.is_available()
            or torch.backends.cuda.matmul.allow_tf32
        ):
            raise ValueError("CUDA graph requires native FP16 CUDA with TF32 disabled")
        self.processor = processor
        self.vision = vision.eval()
        self.head = head.eval()
        self.precision = precision
        self.device = device
        self._batch1_lock = threading.Lock()
        self._use_direct_preprocess = _use_direct_preprocess
        self._batch1_graph = self._capture_batch1_graph() if cuda_graph_batch1 else None

    @classmethod
    def from_checkpoint(
        cls,
        *,
        model_snapshot: Path,
        checkpoint: Path,
        expected_checkpoint_sha256: str,
        model_file_sha256: Mapping[str, str],
        precision: InferencePrecision,
        device: torch.device,
        cuda_graph_batch1: bool = False,
    ) -> Siglip2CompactEncoder:
        """Load a pinned trained encoder for any gallery built with ``from_image_paths``."""

        if (
            not isinstance(checkpoint, Path)
            or not checkpoint.is_file()
            or not isinstance(expected_checkpoint_sha256, str)
            or len(expected_checkpoint_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_checkpoint_sha256)
            or _sha256(checkpoint) != expected_checkpoint_sha256
        ):
            raise ValueError("trained SigLIP2 checkpoint digest differs")
        if (
            not isinstance(model_snapshot, Path)
            or not isinstance(model_file_sha256, Mapping)
            or set(model_file_sha256)
            != {"config.json", "preprocessor_config.json", "model.safetensors"}
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or not (model_snapshot / name).is_file()
                or _sha256(model_snapshot / name) != digest
                for name, digest in model_file_sha256.items()
            )
        ):
            raise ValueError("trained SigLIP2 model snapshot digest differs")
        if (
            precision not in ("fp32_autocast", "fp16_native")
            or type(device) is not torch.device
            or (precision == "fp16_native" and device.type != "cuda")
            or (
                device.type == "cuda"
                and (not torch.cuda.is_available() or torch.backends.cuda.matmul.allow_tf32)
            )
        ):
            raise ValueError("trained SigLIP2 checkpoint runtime differs")
        from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

        processor = AutoImageProcessor.from_pretrained(
            model_snapshot / "preprocessor_config.json",
            local_files_only=True,
            backend="torchvision",
        )
        if (
            type(processor).__name__ != "SiglipImageProcessor"
            or processor.size.get("height") != 256
            or processor.size.get("width") != 256
            or processor.resample != 2
        ):
            raise ValueError("trained SigLIP2 checkpoint processor differs")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            not isinstance(state, dict)
            or not isinstance(state.get("vision"), dict)
            or not isinstance(state.get("head"), dict)
            or any(
                not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
                for group in (state["vision"], state["head"])
                for value in group.values()
            )
        ):
            raise ValueError("trained SigLIP2 checkpoint weights differ")
        config = AutoConfig.from_pretrained(model_snapshot, local_files_only=True)
        vision = SiglipVisionModel(config.vision_config).to(device=device, dtype=torch.float32)
        head = nn.Linear(1024, 128).to(device=device, dtype=torch.float32)
        vision.load_state_dict(state["vision"], strict=True)
        head.load_state_dict(state["head"], strict=True)
        if precision == "fp16_native":
            vision.half()
        return cls(
            processor,
            vision,
            head,
            precision,
            device,
            cuda_graph_batch1,
            _use_direct_preprocess=_direct_processor_supported(
                model_file_sha256["preprocessor_config.json"]
            ),
        )

    @torch.inference_mode()
    def _capture_batch1_graph(self) -> tuple[torch.Tensor, torch.cuda.CUDAGraph, torch.Tensor]:
        with torch.cuda.device(self.device), torch.amp.autocast("cuda", enabled=False):
            pixels = torch.zeros((1, 3, 256, 256), device=self.device, dtype=torch.float16)
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(5):
                    self.vision(pixel_values=pixels)
            torch.cuda.current_stream(self.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                pooled = self.vision(pixel_values=pixels).pooler_output
        if pooled is None or pooled.shape != (1, 1024):
            raise ValueError("CUDA graph pooler geometry differs")
        return pixels, graph, pooled

    def _pack_pooled(self, pooled: torch.Tensor | None, count: int) -> PackedInt8Embeddings:
        if pooled is None or pooled.shape != (count, 1024):
            raise ValueError("trained SigLIP2 serving pooler geometry differs")
        features = F.normalize(compact_head_features(pooled, self.head), dim=1).cpu()
        if not bool(torch.isfinite(features).all()):
            raise ValueError("trained SigLIP2 serving features are nonfinite")
        return pack_int8_unit_embeddings(features)

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
            raise ValueError(f"trained SigLIP2 serving pixel limit is {_MAX_IMAGE_PIXELS}")
        if self._use_direct_preprocess and len(images) == 1:
            batch = {"pixel_values": _direct_preprocess(images[0])}
        elif sum(image.width * image.height for image in images) > _MAX_PREPROCESS_BATCH_PIXELS:
            pieces = [
                self.processor(images=[image.convert("RGB")], return_tensors="pt")
                for image in images
            ]
            if any(set(piece) != {"pixel_values"} for piece in pieces):
                raise ValueError("trained SigLIP2 serving processor geometry differs")
            batch = {"pixel_values": torch.cat([piece["pixel_values"] for piece in pieces])}
        else:
            batch = self.processor(
                images=[image.convert("RGB") for image in images], return_tensors="pt"
            )
        if set(batch) != {"pixel_values"}:
            raise ValueError("trained SigLIP2 serving processor geometry differs")
        pixels = batch["pixel_values"].to(device=self.device)
        if self.precision == "fp16_native":
            pixels = pixels.to(dtype=torch.float16)
        if self._batch1_graph is not None and len(images) == 1:
            static, graph, pooled = self._batch1_graph
            if pixels.shape != static.shape:
                raise ValueError("CUDA graph pixel geometry differs")
            with self._batch1_lock:
                static.copy_(pixels)
                graph.replay()
                return self._pack_pooled(pooled, 1)
        with torch.amp.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.precision == "fp32_autocast" and self.device.type == "cuda",
        ):
            pooled = self.vision(pixel_values=pixels).pooler_output
        return self._pack_pooled(pooled, len(images))


class Siglip2CompactIndex:
    """Own an encoder and persistent exact native gallery for external image queries."""

    def __init__(self, encoder: Siglip2CompactEncoder, gallery: PackedGallery) -> None:
        self.encoder: Siglip2CompactEncoder | None = encoder
        self.gallery: PackedGallery | None = gallery
        self._closed = False
        self._lifecycle_lock = threading.RLock()

    @classmethod
    def from_image_paths(
        cls,
        *,
        encoder: Siglip2CompactEncoder,
        native_library: Path,
        image_paths: Sequence[Path],
        expected_native_library_sha256: str | None = None,
    ) -> Siglip2CompactIndex:
        """Encode a user gallery in ordinal order and open exact native top-10."""

        from PIL import Image

        if (
            not isinstance(encoder, Siglip2CompactEncoder)
            or not isinstance(native_library, Path)
            or not native_library.is_absolute()
            or not native_library.is_file()
            or (
                expected_native_library_sha256 is not None
                and (
                    not isinstance(expected_native_library_sha256, str)
                    or len(expected_native_library_sha256) != 64
                    or any(
                        char not in "0123456789abcdef" for char in expected_native_library_sha256
                    )
                    or _sha256(native_library) != expected_native_library_sha256
                )
            )
            or not isinstance(image_paths, Sequence)
            or len(image_paths) < 10
            or any(not isinstance(path, Path) or not path.is_file() for path in image_paths)
        ):
            raise ValueError("trained SigLIP2 custom gallery authority differs")
        codes = torch.empty((len(image_paths), 128), dtype=torch.int8)
        inverse_norms = torch.empty(len(image_paths), dtype=torch.float16)
        for start in range(0, len(image_paths), _MAX_QUERY_IMAGES):
            stop = min(start + _MAX_QUERY_IMAGES, len(image_paths))
            with ExitStack() as stack:
                images = [stack.enter_context(Image.open(path)) for path in image_paths[start:stop]]
                chunk = encoder.encode_images(images)
            codes[start:stop] = chunk.codes
            inverse_norms[start:stop] = chunk.inverse_norms
        packed = PackedInt8Embeddings(codes, inverse_norms)
        return cls(encoder, CutilePackedInt8Gallery.open_packed(native_library, packed))

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
        cuda_graph_batch1: bool = False,
        official_gallery_receipt: Path | None = None,
        official_gallery_embeddings: Path | None = None,
        expected_official_gallery_receipt_sha256: str | None = None,
        custom_gallery_image_paths: Sequence[Path] | None = None,
    ) -> Siglip2CompactIndex:
        """Load trusted artifacts bound to an independently pinned receipt digest."""

        if custom_gallery_image_paths is not None and (
            not isinstance(custom_gallery_image_paths, Sequence)
            or any(
                value is not None
                for value in (
                    official_gallery_receipt,
                    official_gallery_embeddings,
                    expected_official_gallery_receipt_sha256,
                )
            )
            or len(custom_gallery_image_paths) < 10
            or not isinstance(native_library, Path)
            or not native_library.is_absolute()
            or any(
                not isinstance(path, Path) or not path.is_file()
                for path in custom_gallery_image_paths
            )
        ):
            raise ValueError("trained SigLIP2 custom gallery selection differs")
        if (
            len(expected_receipt_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_receipt_sha256)
            or not training_receipt.is_file()
            or _sha256(training_receipt) != expected_receipt_sha256
        ):
            raise ValueError("trained SigLIP2 serving receipt digest differs")
        if (
            precision not in ("fp32_autocast", "fp16_native")
            or not torch.cuda.is_available()
            or type(cuda_graph_batch1) is not bool
            or (cuda_graph_batch1 and precision != "fp16_native")
        ):
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
        if custom_gallery_image_paths is not None:
            gallery_path, gallery_rows = None, None
        elif any(value is not None for value in official_args):
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
        from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

        processor = AutoImageProcessor.from_pretrained(
            model_snapshot / "preprocessor_config.json",
            local_files_only=True,
            backend="torchvision",
        )
        if (
            type(processor).__name__ != "SiglipImageProcessor"
            or processor.size.get("height") != 256
            or processor.size.get("width") != 256
            or processor.resample != 2
        ):
            raise ValueError("trained SigLIP2 serving processor differs")
        config = AutoConfig.from_pretrained(model_snapshot, local_files_only=True)
        device = torch.device("cuda:0")
        vision = (
            SiglipVisionModel(config.vision_config).to(device=device, dtype=torch.float32).eval()
        )
        head = nn.Linear(1024, 128).to(device=device, dtype=torch.float32).eval()
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
        encoder = Siglip2CompactEncoder(
            processor,
            vision,
            head,
            precision,
            device,
            cuda_graph_batch1=cuda_graph_batch1,
            _use_direct_preprocess=_direct_processor_supported(
                model_hashes["preprocessor_config.json"]
            ),
        )
        if custom_gallery_image_paths is not None:
            return cls.from_image_paths(
                encoder=encoder,
                native_library=native_library,
                image_paths=custom_gallery_image_paths,
            )
        if gallery_path is None or gallery_rows is None:
            raise ValueError("trained SigLIP2 gallery selection differs")
        values = np.load(gallery_path, mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape != (gallery_rows, 128):
            raise ValueError("trained SigLIP2 serving gallery geometry differs")
        packed_gallery = pack_int8_unit_embeddings(torch.from_numpy(np.asarray(values).copy()))
        gallery = CutilePackedInt8Gallery.open_packed(native_library, packed_gallery)
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
