#!/usr/bin/env python3
"""TRAIN-only CUDA graph feasibility for the unchanged SigLIP2 vision stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

from export_sop_siglip2_train import MODEL_REVISION

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
RECEIPT_SHA = "07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5"
CHECKPOINT_SHA = "2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172"
MODEL_HASHES = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(samples: list[int]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(values.mean()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-archive", "dataset-root", "model-snapshot", "receipt", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.receipt) != RECEIPT_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items())
    ):
        raise ValueError("SOP CUDA graph authority differs")
    receipt = json.loads(args.receipt.read_text())
    if receipt["checkpoint_sha256"] != CHECKPOINT_SHA or receipt["seed"] != 179024:
        raise ValueError("SOP CUDA graph checkpoint differs")
    with np.load(args.source_archive, allow_pickle=False) as source:
        relative = tuple(PurePosixPath(str(path)) for path in source["train_relative_paths"][:32])
    if any(path.is_absolute() or ".." in path.parts or not path.parts for path in relative):
        raise ValueError("SOP CUDA graph image paths differ")
    paths = tuple(args.dataset_root.joinpath(*path.parts) for path in relative)
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("SOP CUDA graph image missing")
    from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(args.model_snapshot, local_files_only=True, backend="torchvision")
    config = AutoConfig.from_pretrained(args.model_snapshot, local_files_only=True)
    vision = SiglipVisionModel(config.vision_config).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vision.load_state_dict(checkpoint["vision"], strict=True)
    vision.half()
    torch.backends.cuda.matmul.allow_tf32 = False
    with_images = []
    for path in paths:
        with Image.open(path) as image:
            with_images.append(image.convert("RGB"))
    pixels32 = processor(images=with_images, return_tensors="pt")["pixel_values"].cuda().half()
    if pixels32.shape != (32, 3, 256, 256):
        raise ValueError("SOP CUDA graph input geometry differs")
    results = {}
    torch.cuda.reset_peak_memory_stats()
    for batch_size in (1, 32):
        static = torch.empty_like(pixels32[:batch_size])
        static.copy_(pixels32[:batch_size])
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(5):
                pooled = vision(pixel_values=static).pooler_output
                if pooled is None:
                    raise ValueError("SOP CUDA graph warmup pooler missing")
        torch.cuda.current_stream().wait_stream(side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = vision(pixel_values=static).pooler_output
        if captured is None or captured.shape != (batch_size, 1024):
            raise ValueError("SOP CUDA graph pooler geometry differs")
        for index in range(32 if batch_size == 1 else 1):
            pixels = pixels32[index : index + 1] if batch_size == 1 else pixels32
            eager = vision(pixel_values=pixels).pooler_output
            static.copy_(pixels)
            graph.replay()
            if eager is None or not torch.equal(eager, captured):
                raise ValueError("SOP CUDA graph pooled output differs")
        samples: dict[str, list[int]] = {"eager": [], "graph": []}
        for block in range(20):
            for arm in (("eager", "graph", "graph", "eager") if block % 2 == 0 else ("graph", "eager", "eager", "graph")):
                for call in range(10):
                    pixels = pixels32[(block * 10 + call) % 32 : (block * 10 + call) % 32 + 1] if batch_size == 1 else pixels32
                    torch.cuda.synchronize()
                    started = time.perf_counter_ns()
                    if arm == "graph":
                        static.copy_(pixels)
                        graph.replay()
                    else:
                        vision(pixel_values=pixels)
                    torch.cuda.synchronize()
                    samples[arm].append(time.perf_counter_ns() - started)
        results[str(batch_size)] = {
            "eager": summarize(samples["eager"]),
            "graph": summarize(samples["graph"]),
            "raw_ns": samples,
            "parity_inputs": 32 if batch_size == 1 else 1,
        }
        print(json.dumps({"batch": batch_size, "eager": results[str(batch_size)]["eager"], "graph": results[str(batch_size)]["graph"]}), flush=True)
    peak = torch.cuda.max_memory_allocated()
    gates = {
        "batch1_p50": results["1"]["graph"]["p50_ms"] <= 0.9 * results["1"]["eager"]["p50_ms"],
        "batch32_p95": results["32"]["graph"]["p95_ms"] <= 1.05 * results["32"]["eager"]["p95_ms"],
        "peak_cuda": peak < 3_000_000_000,
    }
    report = {
        "schema": "sfora-sop-siglip2-cuda-graph-vision-preflight-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": RECEIPT_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "source_archive_sha256": ARCHIVE_SHA,
        "image_sha256": [sha256(path) for path in paths],
        "results": results,
        "gates": gates,
        "advance": all(gates.values()),
        "peak_cuda_allocated_bytes": peak,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    payload = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
