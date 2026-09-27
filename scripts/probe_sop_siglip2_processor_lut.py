#!/usr/bin/env python3
"""TRAIN-only exact pixel and paired CPU preprocessor screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
PROCESSOR_SHA = "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lookup(processor: object) -> np.ndarray:
    ramp = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :, None], (256, 256, 3))
    pixels = processor(images=[Image.fromarray(ramp)], return_tensors="pt")["pixel_values"]
    table = pixels[0, 0, 0].numpy().copy()
    if table.shape != (256,) or not np.array_equal(
        pixels[0].numpy(), table[ramp].transpose(2, 0, 1)
    ):
        raise ValueError("processor channels or lookup arithmetic differ")
    return table


def candidate(image: Image.Image, table: np.ndarray) -> torch.Tensor:
    resized = image.convert("RGB").resize((256, 256), resample=Image.Resampling.BILINEAR)
    values = table[np.asarray(resized)]
    return torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1))).unsqueeze(0)


def summary(samples: list[int]) -> dict[str, float]:
    ms = np.asarray(samples, dtype=np.float64) / 1e6
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-archive", "dataset-root", "model-snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.model_snapshot / "preprocessor_config.json") != PROCESSOR_SHA
    ):
        raise ValueError("SOP processor source authority differs")
    torch.set_num_threads(1)
    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    if (
        processor.size.height != 256
        or processor.size.width != 256
        or processor.resample != 2
        or tuple(processor.image_mean) != (0.5,) * 3
        or tuple(processor.image_std) != (0.5,) * 3
    ):
        raise ValueError("SOP processor configuration differs")
    table = lookup(processor)
    paths = []
    seen = set()
    with np.load(args.source_archive, allow_pickle=False) as archive:
        for value in archive["train_relative_paths"]:
            relative = PurePosixPath(str(value))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("SOP image path differs")
            path = args.dataset_root.joinpath(*relative.parts)
            if not path.is_file() or path.is_symlink():
                raise ValueError("SOP image missing")
            digest = sha256(path)
            if digest not in seen:
                seen.add(digest)
                paths.append(path)
            if len(paths) == 10_000:
                break
    if len(paths) != 10_000:
        raise ValueError("SOP unique image count differs")
    timings: dict[str, list[int]] = {"control": [], "candidate": []}
    mismatch = 0
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        expected = processor(images=[rgb], return_tensors="pt")["pixel_values"]
        actual = candidate(rgb, table)
        mismatch += not torch.equal(expected, actual)
        if i < 1_000:
            arms = (
                ("control", "candidate", "candidate", "control")
                if i % 2 == 0
                else ("candidate", "control", "control", "candidate")
            )
            for arm in arms:
                start = time.perf_counter_ns()
                if arm == "control":
                    processor(images=[rgb.convert("RGB")], return_tensors="pt")
                else:
                    candidate(rgb, table)
                timings[arm].append(time.perf_counter_ns() - start)
        if (i + 1) % 1_000 == 0:
            print(json.dumps({"checked": i + 1, "mismatch": mismatch}), flush=True)
    scores = {name: summary(values) for name, values in timings.items()}
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {
        "exact": mismatch == 0,
        "p50": scores["candidate"]["p50_ms"] <= 0.85 * scores["control"]["p50_ms"],
        "p95": scores["candidate"]["p95_ms"] <= 0.90 * scores["control"]["p95_ms"],
        "rss": peak_rss_bytes < 2_000_000_000,
    }
    receipt = {
        "schema": "sfora-sop-siglip2-processor-lut-train-v1",
        "source_sha256": sha256(Path(__file__)),
        "archive_sha256": ARCHIVE_SHA,
        "processor_sha256": PROCESSOR_SHA,
        "unique_image_count": len(paths),
        "mismatch_count": mismatch,
        "timing_calls_per_arm": len(timings["control"]),
        "timing": scores,
        "peak_rss_bytes": peak_rss_bytes,
        "gates": gates,
        "advance": all(gates.values()),
        "hardware": {"cpu": os.cpu_count(), "torch": torch.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
