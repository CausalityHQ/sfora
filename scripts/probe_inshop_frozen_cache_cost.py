#!/usr/bin/env python3
"""Measure the cacheable lower-stack cost in the frozen In-Shop encoder."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoModel

    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().train()
    del full
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vision.load_state_dict(checkpoint["vision"], strict=True)
    vision.embeddings.requires_grad_(False)
    for layer in vision.encoder.layers[:12]:
        layer.requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    pixels = torch.randn(64, 3, 256, 256, device="cuda")

    def lower() -> torch.Tensor:
        hidden = vision.embeddings(pixels)
        for layer in vision.encoder.layers[:12]:
            hidden = layer(hidden, None)
        return hidden

    def upper(hidden: torch.Tensor) -> torch.Tensor:
        for layer in vision.encoder.layers[12:]:
            hidden = layer(hidden, None)
        hidden = vision.post_layernorm(hidden)
        return vision.head(hidden)

    def timed(call: object) -> float:
        torch.cuda.synchronize()
        started = time.perf_counter()
        call()  # type: ignore[operator]
        torch.cuda.synchronize()
        return time.perf_counter() - started

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            hidden = lower()
            reference = vision(pixels).pooler_output
            split = upper(hidden)
        if reference is None or not torch.equal(reference, split):
            raise ValueError("In-Shop frozen-layer split differs from full vision forward")

    def full_step() -> None:
        vision.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = vision(pixels).pooler_output
            if output is None:
                raise ValueError("In-Shop full pooler missing")
            output.square().mean().backward()

    def cached_step() -> None:
        vision.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            upper(hidden).square().mean().backward()

    def cache_step() -> None:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lower()

    for _ in range(3):
        full_step()
        cached_step()
        cache_step()
    times = {"full": [], "cached": [], "cache_generation": []}
    for _ in range(10):
        for name, call in (
            ("full", full_step),
            ("cached", cached_step),
            ("cache_generation", cache_step),
        ):
            times[name].append(timed(call))
    print(
        json.dumps(
            {
                "schema": "sfora-inshop-frozen-cache-cost-probe-v1",
                "batch_size": 64,
                "position_tokens": hidden.shape[1],
                "hidden_width": hidden.shape[2],
                "cache_dtype": str(hidden.dtype),
                "cache_bytes_per_batch": hidden.numel() * hidden.element_size(),
                "medians_seconds": {
                    name: statistics.median(values) for name, values in times.items()
                },
                "times_seconds": times,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
