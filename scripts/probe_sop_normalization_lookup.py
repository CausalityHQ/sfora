#!/usr/bin/env python3
"""CPU-only finite-domain normalization spike; no product API changes."""

import argparse
import copy
import json
from pathlib import Path, PurePosixPath
import resource
import time
from types import MethodType

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tvf

import probe_sop_preprocessing_worker as base
from benchmark_sop_siglip2_cuda_graph_public import ARCHIVE_SHA, decode, sha256, stats

_table = None
_processor = None


def lookup(pixels):
    assert pixels.device.type == "cpu" and pixels.dtype == torch.uint8
    assert _table is not None
    result = torch.empty_strided(pixels.shape, pixels.stride(), dtype=torch.float32)
    np.take(_table, pixels.numpy(), out=result.numpy())
    return result


def normalize_lookup(
    self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
):
    assert do_rescale and do_normalize and rescale_factor == 1 / 255
    assert list(image_mean) == list(image_std) == [0.5] * 3
    return lookup(images)


def initialize_lookup():
    global _table, _processor
    snapshot = Path(
        "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"
    )
    base.initialize(snapshot, 20)
    assert base._direct
    values = (
        torch.arange(256, dtype=torch.uint8).reshape(1, 1, 1, 256).repeat(1, 3, 1, 1)
    )
    direct = tvf.normalize(values.float(), [127.5] * 3, [127.5] * 3)
    fused = base._processor.rescale_and_normalize(
        values, True, 1 / 255, True, (0.5,) * 3, (0.5,) * 3
    )
    base.check_pixels(direct, fused)
    _table = direct[0, 0, 0].numpy().copy()
    _processor = copy.deepcopy(base._processor)
    _processor.rescale_and_normalize = MethodType(normalize_lookup, _processor)


def process_lookup(images):
    assert 1 <= len(images) <= 32
    assert all(image.width * image.height <= 16_777_216 for image in images)
    assert sum(image.width * image.height for image in images) <= 64_000_000
    if len(images) == 1:
        pixels = tvf.pil_to_tensor(images[0].convert("RGB")).unsqueeze(0)
        return lookup(
            tvf.resize(pixels, [256, 256], InterpolationMode.BILINEAR, antialias=True)
        )
    return _processor(
        images=[image.convert("RGB") for image in images], return_tensors="pt"
    )["pixel_values"]


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    assert not args.output.exists()
    archive = Path("/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz")
    assert sha256(archive) == ARCHIVE_SHA
    with np.load(archive, allow_pickle=False) as source:
        paths = [
            PurePosixPath(str(source["train_relative_paths"][i]))
            for i in np.linspace(0, 59550, 32, dtype=np.int64)
        ]
    assert all(not p.is_absolute() and ".." not in p.parts for p in paths)
    absolute = [
        Path("/home/riomus/datasets/Stanford_Online_Products").joinpath(*p.parts)
        for p in paths
    ]
    assert all(p.is_file() and not p.is_symlink() for p in absolute)
    manifest = [
        {"relative_path": str(r), "sha256": sha256(p)} for r, p in zip(paths, absolute)
    ]
    images = decode(absolute)
    old_threads = torch.get_num_threads()
    started = time.perf_counter()
    timing = {}
    try:
        initialize_lookup()
        for size in (1, 32):
            batch = images[:size]
            reference = base.process(batch)
            for _ in range(3):
                base.check_pixels(reference, base.process(batch))
                base.check_pixels(reference, process_lookup(batch))
            raw = {"parent20": [], "lookup20": []}
            for block in range(5):
                order = (
                    ("parent20", "lookup20", "lookup20", "parent20")
                    if block % 2 == 0
                    else ("lookup20", "parent20", "parent20", "lookup20")
                )
                for arm in order:
                    for _ in range(2):
                        tick = time.perf_counter_ns()
                        actual = (
                            base.process(batch)
                            if arm == "parent20"
                            else process_lookup(batch)
                        )
                        raw[arm].append(time.perf_counter_ns() - tick)
                        base.check_pixels(reference, actual)
                        assert torch.get_num_threads() == 20
            timing[str(size)] = {
                "raw_ns": raw,
                "stats": {arm: stats(v) for arm, v in raw.items()},
                "pixels_exact": True,
            }
    finally:
        torch.set_num_threads(old_threads)
    assert all(sha256(p) == row["sha256"] for p, row in zip(absolute, manifest))
    passed = all(
        v["stats"]["lookup20"]["p95_ms"] <= 0.8 * v["stats"]["parent20"]["p95_ms"]
        and v["stats"]["parent20"]["p50_ms"] - v["stats"]["lookup20"]["p50_ms"]
        >= (0.81 if size == "1" else 14.21)
        for size, v in timing.items()
    )
    report = {
        "schema": "sfora-sop-normalization-lookup-cpu-v1",
        "decision": "CPU_PASS_DESIGN_ONLY" if passed else "KILL_CPU_LOOKUP",
        "timing": timing,
        "manifest": manifest,
        "archive_sha256": ARCHIVE_SHA,
        "serving_sha256": base.SERVING_SHA,
        "processor_sha256": base.serving._DIRECT_PROCESSOR_SHA,
        "script_sha256": sha256(Path(__file__)),
        "baseline_script_sha256": sha256(Path(base.__file__)),
        "helper_sha256": sha256(Path(decode.__code__.co_filename)),
        "table_sha256": __import__("hashlib").sha256(_table.tobytes()).hexdigest(),
        "whole_wall_seconds": time.perf_counter() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "quality_read": False,
        "claim_eligible": False,
    }
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "timing": {s: v["stats"] for s, v in timing.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
