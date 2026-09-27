#!/usr/bin/env python3
"""Paired TRAIN-image stage-cost diagnosis; no quality selection."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, sha256
from probe_inshop_zooclaw_source import CONFIG_SHA, MODEL_SHA, PROCESSOR_SHA, REVISION
from transformers import AutoImageProcessor, AutoModel

from sfora.unicom_inshop import parse_inshop_partition

SIGLIP_REVISION = "787800c8990e6f058423089178e718139608408c"
SIGLIP_FILES = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}
STAGES = ("decode", "processor", "host_to_device", "vision", "total")


def measure(snapshot: Path, paths: tuple[Path, ...], *, fashion: bool) -> dict:
    processor = AutoImageProcessor.from_pretrained(
        snapshot, local_files_only=True, **({"backend": "torchvision"} if fashion else {})
    )
    model = (
        AutoModel.from_pretrained(
            snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    torch.cuda.reset_peak_memory_stats()
    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    with torch.inference_mode():
        for index in range(-1, len(paths) // 32):
            batch = paths[:32] if index < 0 else paths[index * 32 : (index + 1) * 32]
            started = time.perf_counter()
            images = []
            for path in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            decoded = time.perf_counter()
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            processed = time.perf_counter()
            pixels = pixels.cuda()
            if fashion:
                pixels = pixels.half()
            torch.cuda.synchronize()
            transferred = time.perf_counter()
            output = model.get_image_features(pixel_values=pixels)
            result = output if isinstance(output, torch.Tensor) else output.pooler_output
            if result.shape != (len(batch), 768 if fashion else 1024):
                raise ValueError("In-Shop stage profile output geometry differs")
            torch.cuda.synchronize()
            finished = time.perf_counter()
            if index < 0:
                continue
            for stage, duration in zip(
                STAGES,
                (
                    decoded - started,
                    processed - decoded,
                    transferred - processed,
                    finished - transferred,
                    finished - started,
                ),
                strict=True,
            ):
                samples[stage].append(duration)
    return {
        "samples_seconds": samples,
        "median_seconds": {stage: float(np.median(samples[stage])) for stage in STAGES},
        "mean_seconds": {stage: float(np.mean(samples[stage])) for stage in STAGES},
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "siglip-snapshot", "fashion-snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--spread", action="store_true")
    args = parser.parse_args()
    expected = {
        "config.json": CONFIG_SHA,
        "preprocessor_config.json": PROCESSOR_SHA,
        "model.safetensors": MODEL_SHA,
    }
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or args.siglip_snapshot.resolve().name != SIGLIP_REVISION
        or args.fashion_snapshot.resolve().name != REVISION
        or any(
            sha256(args.siglip_snapshot / name) != digest for name, digest in SIGLIP_FILES.items()
        )
        or any(sha256(args.fashion_snapshot / name) != digest for name, digest in expected.items())
    ):
        raise ValueError("In-Shop source-stage profile authority differs")
    rows = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    if len(rows) != 25_882:
        raise ValueError("In-Shop TRAIN inventory differs")
    indexes = np.linspace(0, len(rows) - 1, 320, dtype=int) if args.spread else range(320)
    paths = tuple(rows[index].image_path for index in indexes)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "schema": "sfora-inshop-source-stage-cost-diagnostic-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "partition_sha256": PARTITION_SHA,
        "sample": (
            "320 evenly spaced official TRAIN images, 10 batches of 32; sequential arms"
            if args.spread
            else "first 320 official TRAIN images, 10 batches of 32; sequential arms"
        ),
        "siglip_large_256": measure(args.siglip_snapshot, paths, fashion=False),
    }
    torch.cuda.empty_cache()
    report["fashion_384"] = measure(args.fashion_snapshot, paths, fashion=True)
    report["gpu"] = torch.cuda.get_device_name()
    report["torch"] = torch.__version__
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {arm: report[arm]["median_seconds"] for arm in ("siglip_large_256", "fashion_384")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
