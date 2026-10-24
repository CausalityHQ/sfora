"""Serve fine-tuned SigLIP2 So400m compact descriptors through exact packed top-k search.

Encode arithmetic copies the training/official export path (train_sop_siglip2_substrate_pair.export_all
and evaluate_sop_so400m_*_dim512_official.export) op for op, so packed outputs reproduce receipts:
  batch of 64 -> processor (torchvision backend) -> vision under fp16 autocast -> pooler_output
  -> autocast disabled: head(F.normalize(pooled.float())) -> F.normalize -> pack_int8_unit_embeddings.
Search copies score_packed_full_gallery: int8 codes as float32, (q @ g.T) * q_inv * g_inv, self -inf,
stable descending argsort, queries blocked by 64.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional as F

from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings

RESOLUTIONS = (280, 384)
OUTPUT_DIMS = (128, 512)
SO400M_WIDTH = 1152
BATCH_SIZE = 64
_INDEX_MAGIC = b"SFORA-SO400M-INDEX-v1\0"


class _FP8Linear(nn.Module):
    """Dynamic FP8 (e4m3) linear: per-output-channel weight scales, per-row activation scales, fp16 out."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        fmax = torch.finfo(torch.float8_e4m3fn).max
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fmax
        self.register_buffer("w_t", (w / scale).to(torch.float8_e4m3fn).t())
        self.register_buffer("w_scale", scale.t().contiguous())
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().half(), requires_grad=False)
        self.out_features = linear.out_features
        self.fmax = fmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        x_scale = x2.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / self.fmax
        y = torch._scaled_mm((x2.float() / x_scale).to(torch.float8_e4m3fn), self.w_t, scale_a=x_scale,
                             scale_b=self.w_scale, bias=self.bias, out_dtype=torch.float16)
        return y.reshape(*x.shape[:-1], self.out_features)


def _swap_fp8(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features % 16 == 0 and child.out_features % 16 == 0:
            setattr(module, name, _FP8Linear(child))
        elif not isinstance(child, nn.MultiheadAttention):  # MHA reads out_proj.weight directly
            _swap_fp8(child)


class So400mCompactEncoder:
    """A trained So400m vision tower plus linear compact head, in the reference export arithmetic."""

    def __init__(
        self,
        processor: Any,
        vision: nn.Module,
        head: nn.Linear,
        *,
        resolution: int,
        device: torch.device,
        fp16_native: bool = False,
        fp8: bool = False,
    ) -> None:
        config = getattr(vision, "config", None)
        if (
            resolution not in RESOLUTIONS
            or type(device) is not torch.device
            or not callable(processor)
            or not isinstance(vision, nn.Module)
            or config is None
            or type(head) is not nn.Linear
            or head.in_features != config.hidden_size
            or head.out_features not in OUTPUT_DIMS
            or getattr(processor, "resample", None) != 2
            or processor.size.get("height") != resolution
            or processor.size.get("width") != resolution
        ):
            raise ValueError("So400m compact serving geometry differs")
        self.processor = processor
        self.vision = vision.eval()
        self.head = head.eval()
        self.resolution = resolution
        self.output_dim = head.out_features
        self.device = device
        # fp16 weights, no autocast: 1.32x e2e @280, held R@1 net +1/5,851 (evidence so400m-p-variants-v1).
        # fp8 (implies fp16_native): CUDA sm_89+; In-Shop official 101 img/s, R@1 net +7/14,218 (inshop-so400m-384-fp8).
        self.fp16_native = fp16_native or fp8
        if self.fp16_native:
            self.vision = self.vision.half()
        if fp8:
            _swap_fp8(self.vision)
        # Reference: 384 runs unpatched (no interpolation); 280 interpolates position embeddings.
        self.interpolate = resolution != config.image_size

    @classmethod
    def from_checkpoint(
        cls,
        model_snapshot: Path,
        checkpoint: Path,
        *,
        resolution: int,
        output_dim: int,
        device: torch.device,
        fp16_native: bool = False,
        fp8: bool = False,
    ) -> So400mCompactEncoder:
        if (
            resolution not in RESOLUTIONS
            or output_dim not in OUTPUT_DIMS
            or type(device) is not torch.device
            or not Path(model_snapshot).is_dir()
            or not Path(checkpoint).is_file()
        ):
            raise ValueError("So400m compact serving geometry differs")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        head_weight = state.get("head", {}).get("weight") if isinstance(state, dict) else None
        if head_weight is None or tuple(head_weight.shape) != (output_dim, SO400M_WIDTH):
            raise ValueError("So400m compact serving head differs from output_dim")
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            model_snapshot, local_files_only=True, backend="torchvision"
        )
        processor.size = {"height": resolution, "width": resolution}
        full = AutoModel.from_pretrained(
            model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        vision = full.vision_model.float().to(device).eval()
        del full
        head = nn.Linear(SO400M_WIDTH, output_dim).to(device).eval()
        vision.load_state_dict(state["vision"], strict=True)
        head.load_state_dict(state["head"], strict=True)
        if vision.config.hidden_size != SO400M_WIDTH or vision.config.image_size != 384:
            raise ValueError("So400m compact serving snapshot differs")
        if device.type == "cuda":
            # ponytail: process-global flag, as in the reference scripts; encode re-checks it.
            torch.backends.cuda.matmul.allow_tf32 = False
        return cls(processor, vision, head, resolution=resolution, device=device, fp16_native=fp16_native, fp8=fp8)

    @torch.inference_mode()
    def _encode_batch(self, images: Sequence[Any]) -> torch.Tensor:
        batch = self.processor(images=[image.convert("RGB") for image in images], return_tensors="pt")
        tensors = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        if tensors.get("pixel_values") is None or tensors["pixel_values"].shape[0] != len(images):
            raise ValueError("So400m compact processor batch geometry differs")
        if self.fp16_native:
            tensors["pixel_values"] = tensors["pixel_values"].half()
            pooled = self.vision(**tensors, interpolate_pos_encoding=self.interpolate).pooler_output
        else:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                pooled = self.vision(**tensors, interpolate_pos_encoding=self.interpolate).pooler_output
        if (
            pooled is None
            or pooled.shape != (len(images), self.head.in_features)
            or not bool(torch.isfinite(pooled).all())
            or bool((torch.linalg.vector_norm(pooled.float(), dim=1) == 0).any())
        ):
            raise ValueError("So400m compact pooler differs")
        with torch.autocast(device_type=self.device.type, enabled=False):
            values = self.head(F.normalize(pooled.float(), dim=1))
        return F.normalize(values, dim=1).cpu()

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        """Return unit float32 CPU rows in input order, computed in batches of 64."""

        if isinstance(images, (str, bytes)) or len(images) < 1:
            raise ValueError("So400m compact image batch differs")
        if self.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError("So400m compact serving requires TF32 matmul disabled")
        images = list(images)
        result = torch.cat(
            [self._encode_batch(images[i : i + BATCH_SIZE]) for i in range(0, len(images), BATCH_SIZE)]
        ).contiguous()
        if result.shape != (len(images), self.output_dim) or not bool(torch.isfinite(result).all()):
            raise ValueError("So400m compact descriptor geometry differs")
        return result

    def encode_packed(self, images: Sequence[Any]) -> PackedInt8Embeddings:
        return pack_int8_unit_embeddings(self.encode_images(images))


class So400mCompactIndex:
    """Exact full-gallery packed int8 search, bit-identical to score_packed_full_gallery."""

    def __init__(self, gallery: PackedInt8Embeddings, *, device: torch.device | None = None) -> None:
        if type(gallery) is not PackedInt8Embeddings:
            raise ValueError("So400m compact gallery differs")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if type(device) is not torch.device:
            raise ValueError("So400m compact gallery device differs")
        self.gallery = gallery
        self.device = device
        self._codes = gallery.codes.float().to(device)
        self._inverse = gallery.inverse_norms.float().to(device)

    def __len__(self) -> int:
        return self.gallery.codes.shape[0]

    @torch.inference_mode()
    def search(
        self,
        queries: PackedInt8Embeddings,
        k: int,
        exclude: Sequence[int] | NDArray[np.integer] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Return (ordinals, scores) of shape (queries, k); ties keep the lowest gallery index first.

        `exclude[i]` is the gallery row scored -inf for query i (self exclusion)."""

        rows = len(self)
        if (
            type(queries) is not PackedInt8Embeddings
            or queries.codes.shape[1] != self.gallery.codes.shape[1]
            or type(k) is not int
            or not 1 <= k <= rows
        ):
            raise ValueError("So400m compact search geometry differs")
        count = queries.codes.shape[0]
        excluded = None
        if exclude is not None:
            excluded = torch.as_tensor(np.asarray(exclude, dtype=np.int64))
            if excluded.shape != (count,) or bool((excluded < 0).any()) or bool((excluded >= rows).any()):
                raise ValueError("So400m compact search exclusion differs")
            excluded = excluded.to(self.device)
        code = queries.codes.float().to(self.device)
        inverse = queries.inverse_norms.float().to(self.device)
        ordinals, scores = [], []
        for start in range(0, count, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, count)
            block = (code[start:stop] @ self._codes.T) * inverse[start:stop, None] * self._inverse[None, :]
            if excluded is not None:
                block[torch.arange(stop - start, device=self.device), excluded[start:stop]] = -torch.inf
            ranked = torch.argsort(block, dim=1, descending=True, stable=True)[:, :k]
            ordinals.append(ranked.cpu())
            scores.append(block.gather(1, ranked).cpu())
        return torch.cat(ordinals).numpy().astype(np.int64), torch.cat(scores).numpy().astype(np.float32)

    def to_bytes(self) -> bytes:
        count, dimensions = self.gallery.codes.shape
        payload = _INDEX_MAGIC + struct.pack("<QQ", count, dimensions) + self.gallery.to_bytes()
        return payload + hashlib.sha256(payload).digest()

    @classmethod
    def from_bytes(cls, data: bytes, *, device: torch.device | None = None) -> So400mCompactIndex:
        header = len(_INDEX_MAGIC) + 16
        if type(data) is not bytes or len(data) < header + 32:
            raise ValueError("So400m compact index bytes differ")
        payload, digest = data[:-32], data[-32:]
        if not payload.startswith(_INDEX_MAGIC) or hashlib.sha256(payload).digest() != digest:
            raise ValueError("So400m compact index bytes differ")
        count, dimensions = struct.unpack("<QQ", payload[len(_INDEX_MAGIC) : header])
        gallery = PackedInt8Embeddings.from_bytes(payload[header:], count=count, dimensions=dimensions)
        return cls(gallery, device=device)


__all__ = ["So400mCompactEncoder", "So400mCompactIndex"]
