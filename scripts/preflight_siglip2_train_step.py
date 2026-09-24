#!/usr/bin/env python3
"""Measure one real-image SigLIP2 vision-backbone training configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

MODEL_HASHES = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.batch_size < 2
        or args.steps < 2
        or not torch.cuda.is_available()
        or len(args.images) < 2
        or any(not path.is_file() or path.is_symlink() for path in args.images)
        or any(
            sha256(args.model_snapshot / name) != expected
            for name, expected in MODEL_HASHES.items()
        )
    ):
        raise ValueError("SigLIP2 training preflight authority differs")
    from transformers import AutoImageProcessor, AutoModel

    torch.manual_seed(179019)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    source = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = source.vision_model
    del source
    vision = vision.float().cuda().train()
    if args.gradient_checkpointing:
        vision.gradient_checkpointing_enable()
    head = nn.Linear(1024, 128).cuda().train()
    optimizer = torch.optim.AdamW(
        [
            {"params": vision.parameters(), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    selected = [images[index % len(images)] for index in range(args.batch_size)]
    input_batch = processor(images=selected, return_tensors="pt")
    tensors = {key: value.cuda() for key, value in input_batch.items() if torch.is_tensor(value)}
    labels = torch.arange(args.batch_size, device="cuda") % 2
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step_seconds = []
    losses = []
    for _ in range(args.steps):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pooled = vision(**tensors).pooler_output
            if pooled is None or pooled.shape != (args.batch_size, 1024):
                raise ValueError("SigLIP2 training preflight feature geometry differs")
            logits = head(pooled.float())
            loss = F.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError("SigLIP2 training preflight loss is nonfinite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(vision.parameters()) + list(head.parameters()), 1.0, error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_started)
        losses.append(float(loss.detach()))
        print(json.dumps({"step": len(step_seconds), "seconds": step_seconds[-1]}), flush=True)
    receipt = {
        "schema": "sfora-siglip2-train-step-preflight-v1",
        "source_sha256": sha256(Path(__file__)),
        "model_file_sha256": MODEL_HASHES,
        "image_sha256": [sha256(path) for path in args.images],
        "batch_size": args.batch_size,
        "steps": args.steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "objective": "synthetic two-label cross-entropy, representative autograd only",
        "vision_parameter_dtype": sorted({str(value.dtype) for value in vision.parameters()}),
        "step_seconds": step_seconds,
        "losses": losses,
        "queries_per_second_after_first": (
            args.batch_size * (args.steps - 1) / sum(step_seconds[1:])
        ),
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "throughput_images_per_second": receipt["queries_per_second_after_first"],
                "peak_cuda_allocated_bytes": receipt["peak_cuda_allocated_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
