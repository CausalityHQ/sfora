#!/usr/bin/env python3
"""Screen a direct Torchvision v2 path against the pinned SOP TRAIN processor."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from probe_sop_siglip2_processor_lut import ARCHIVE_SHA, PROCESSOR_SHA, sha256, summary
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvf
from transformers import AutoImageProcessor


def direct(image: Image.Image) -> torch.Tensor:
    pixels = tvf.pil_to_tensor(image.convert("RGB")).unsqueeze(0)
    pixels = tvf.resize(pixels, [256, 256], InterpolationMode.BILINEAR, antialias=True)
    return tvf.normalize(pixels.float(), [127.5] * 3, [127.5] * 3)


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
        raise ValueError("SOP direct processor authority differs")
    torch.set_num_threads(20)
    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot / "preprocessor_config.json",
        local_files_only=True,
        backend="torchvision",
    )
    if (
        type(processor).__name__ != "SiglipImageProcessor"
        or processor.size.height != 256
        or processor.size.width != 256
        or processor.resample != 2
        or tuple(processor.image_mean) != (0.5,) * 3
        or tuple(processor.image_std) != (0.5,) * 3
        or processor.rescale_factor != 1 / 255
    ):
        raise ValueError("SOP direct processor configuration differs")
    paths: list[Path] = []
    seen: set[str] = set()
    with np.load(args.source_archive, allow_pickle=False) as archive:
        for value in archive["train_relative_paths"]:
            relative = PurePosixPath(str(value))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("SOP direct processor image path differs")
            path = args.dataset_root.joinpath(*relative.parts)
            if not path.is_file() or path.is_symlink():
                raise ValueError("SOP direct processor image missing")
            digest = sha256(path)
            if digest not in seen:
                seen.add(digest)
                paths.append(path)
            if len(paths) == 10_000:
                break
    if len(paths) != 10_000:
        raise ValueError("SOP direct processor image inventory differs")
    timings: dict[str, list[int]] = {"control": [], "direct": []}
    mismatch = 0
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        expected = processor(images=[rgb.convert("RGB")], return_tensors="pt")[
            "pixel_values"
        ]
        actual = direct(rgb)
        mismatch += not torch.equal(expected, actual)
        if i < 1_000:
            arms = (
                ("control", "direct", "direct", "control")
                if i % 2 == 0
                else ("direct", "control", "control", "direct")
            )
            for arm in arms:
                started = time.perf_counter_ns()
                if arm == "control":
                    processor(images=[rgb.convert("RGB")], return_tensors="pt")
                else:
                    direct(rgb)
                timings[arm].append(time.perf_counter_ns() - started)
        if (i + 1) % 1_000 == 0:
            print(json.dumps({"checked": i + 1, "mismatches": mismatch}), flush=True)
    scores = {arm: summary(values) for arm, values in timings.items()}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {
        "exact": mismatch == 0,
        "p50": scores["direct"]["p50_ms"] <= 0.85 * scores["control"]["p50_ms"],
        "p95": scores["direct"]["p95_ms"] <= 0.90 * scores["control"]["p95_ms"],
        "rss": rss < 2_000_000_000,
    }
    receipt = {
        "schema": "sfora-sop-siglip2-direct-processor-train-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "archive_sha256": ARCHIVE_SHA,
        "processor_sha256": PROCESSOR_SHA,
        "unique_image_count": len(paths),
        "mismatch_count": mismatch,
        "timing_calls_per_arm": len(timings["control"]),
        "timing": scores,
        "peak_rss_bytes": rss,
        "gates": gates,
        "advance": all(gates.values()),
        "hardware": {"cpu": os.cpu_count(), "torch": torch.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
