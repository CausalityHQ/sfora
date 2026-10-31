"""Exact shared-trunk serving for So400m seed ensembles.

Seeds trained with the lower stack frozen share embeddings + blocks [0, split) bitwise (verified 211/211 tensors
for seeds 179024/179025, split 13). The trunk runs once; each member runs only its upper blocks, post-LN, pooling
head and linear head. Output = concat(unit members)/sqrt(n): identical arithmetic to the official ensemble.
Encode cost (split + n*(27-split))/27 of one tower per member-set: 1.52x instead of 2x for n=2, split=13.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from sfora.so400m_compact_serving import BATCH_SIZE, So400mCompactEncoder


class SharedTrunkEnsemble:
    def __init__(self, members: Sequence[So400mCompactEncoder], *, split: int = 13, compile: bool = False) -> None:
        first = members[0]
        shared = [p for n, p in first.vision.named_parameters() if n.startswith("embeddings.") or
                  any(n.startswith(f"encoder.layers.{i}.") for i in range(split))]
        for m in members[1:]:
            own = dict(m.vision.named_parameters())
            names = [n for n, _ in first.vision.named_parameters() if n.startswith("embeddings.") or
                     any(n.startswith(f"encoder.layers.{i}.") for i in range(split))]
            if m.fp16_native != first.fp16_native or m.resolution != first.resolution or any(
                    not torch.equal(own[n], p) for n, p in zip(names, shared)):
                raise ValueError("ensemble members do not share the frozen trunk")
        self.members, self.split, self.first = list(members), split, first
        if compile:  # per-module; compiling the inference_mode method crashes. fp8+compile: 63.7 img/s e2e, held R@1 95.197
            for m in self.members:
                ls = m.vision.encoder.layers
                for i in range(len(ls)):
                    ls[i] = torch.compile(ls[i], mode="max-autotune-no-cudagraphs")
                m.vision.head = torch.compile(m.vision.head, mode="max-autotune-no-cudagraphs")

    @torch.inference_mode()
    def _encode_batch(self, images: Sequence[Any]) -> torch.Tensor:
        f = self.first
        px = f.processor(images=[i.convert("RGB") for i in images], return_tensors="pt")["pixel_values"].to(f.device)
        return self.encode_pixels(px).cpu()

    @torch.inference_mode()
    def encode_pixels(self, px: torch.Tensor) -> torch.Tensor:
        """Processor output (B, 3, R, R) on device -> unit ensemble rows on device."""
        f = self.first
        ctx = torch.autocast(device_type=f.device.type, dtype=torch.float16, enabled=not f.fp16_native)
        with ctx:
            h = f.vision.embeddings(px.half() if f.fp16_native else px, interpolate_pos_encoding=f.interpolate)
            for layer in f.vision.encoder.layers[: self.split]:
                h = _hidden(layer(h, None))
            outs = []
            for m in self.members:
                x = h
                for layer in m.vision.encoder.layers[self.split:]:
                    x = _hidden(layer(x, None))
                x = m.vision.post_layernorm(x)
                pooled = m.vision.head(x)
                with torch.autocast(device_type=f.device.type, enabled=False):
                    outs.append(F.normalize(m.head(F.normalize(pooled.float(), dim=1)), dim=1))
        return torch.cat(outs, 1) / math.sqrt(len(outs))

    def encode_images(self, images: Sequence[Any]) -> torch.Tensor:
        images = list(images)
        return torch.cat([self._encode_batch(images[i: i + BATCH_SIZE]) for i in range(0, len(images), BATCH_SIZE)])


def _hidden(out: Any) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out
