#!/usr/bin/env python3
"""Fixed CPU/IPC preprocessing falsifier; no serving API or GPU changes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import resource
import time

import numpy as np
import torch
import torchvision
import transformers
from transformers import AutoImageProcessor

import sfora.siglip2_compact_serving as serving
from benchmark_sop_siglip2_cuda_graph_public import ARCHIVE_SHA, decode, sha256, stats

SERVING_SHA = "ef454200ca17b80917fe7aa52cf232703779d21af7e8ffe0d25f8e811724bc62"
_processor = None
_direct = False


def initialize(snapshot, threads):
    global _processor, _direct
    assert (
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
    )
    assert sha256(Path(serving.__file__)) == SERVING_SHA
    assert (
        sha256(snapshot / "preprocessor_config.json") == serving._DIRECT_PROCESSOR_SHA
    )
    torch.set_num_threads(threads)
    _processor = AutoImageProcessor.from_pretrained(
        snapshot / "preprocessor_config.json",
        local_files_only=True,
        backend="torchvision",
    )
    _direct = serving._direct_processor_supported(serving._DIRECT_PROCESSOR_SHA)


def process(images):
    assert 1 <= len(images) <= 32
    assert all(image.width * image.height <= 16_777_216 for image in images)
    assert sum(image.width * image.height for image in images) <= 64_000_000
    if _direct and len(images) == 1:
        return serving._direct_preprocess(images[0])
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
    snapshot = Path(
        "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"
    )
    with np.load(archive, allow_pickle=False) as source:
        rows = np.linspace(0, 59550, 32, dtype=np.int64)
        relative = [PurePosixPath(str(source["train_relative_paths"][i])) for i in rows]
    assert all(not r.is_absolute() and ".." not in r.parts for r in relative)
    paths = [
        Path("/home/riomus/datasets/Stanford_Online_Products").joinpath(*r.parts)
        for r in relative
    ]
    assert all(p.is_file() and not p.is_symlink() for p in paths)
    manifest = [
        {"relative_path": str(r), "sha256": sha256(p)} for r, p in zip(relative, paths)
    ]
    images = decode(paths)
    old_threads = torch.get_num_threads()
    started = time.perf_counter()
    initialize(snapshot, 20)
    reports = {}
    try:
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("spawn"),
            initializer=initialize,
            initargs=(snapshot, 1),
        ) as worker:
            tick = time.perf_counter()
            worker.submit(process, images[:1]).result(timeout=30)
            cold = time.perf_counter() - tick
            for size in (1, 32):
                batch = images[:size]
                reference = process(batch)
                assert (
                    reference.shape == (size, 3, 256, 256)
                    and reference.dtype == torch.float32
                )
                assert reference.is_contiguous() and torch.isfinite(reference).all()
                for _ in range(3):
                    assert torch.equal(reference, process(batch))
                    assert torch.equal(
                        reference, worker.submit(process, batch).result(timeout=30)
                    )
                raw = {"parent20": [], "worker1": []}
                for block in range(5):
                    order = (
                        ("parent20", "worker1", "worker1", "parent20")
                        if block % 2 == 0
                        else ("worker1", "parent20", "parent20", "worker1")
                    )
                    for arm in order:
                        for _ in range(2):
                            tick = time.perf_counter_ns()
                            actual = (
                                process(batch)
                                if arm == "parent20"
                                else worker.submit(process, batch).result(timeout=30)
                            )
                            raw[arm].append(time.perf_counter_ns() - tick)
                            assert (
                                actual.dtype == torch.float32
                                and actual.device.type == "cpu"
                                and actual.is_contiguous()
                            )
                            assert torch.equal(reference, actual)
                            assert torch.get_num_threads() == 20
                reports[str(size)] = {
                    "raw_ns": raw,
                    "stats": {arm: stats(v) for arm, v in raw.items()},
                    "pixels_exact": True,
                }
    finally:
        torch.set_num_threads(old_threads)
    assert all(sha256(p) == row["sha256"] for p, row in zip(paths, manifest))
    useful = all(
        v["stats"]["worker1"]["p95_ms"] <= 0.8 * v["stats"]["parent20"]["p95_ms"]
        and v["stats"]["parent20"]["p50_ms"] - v["stats"]["worker1"]["p50_ms"]
        >= (0.81 if size == "1" else 14.21)
        for size, v in reports.items()
    )
    report = {
        "schema": "sfora-sop-preprocessing-worker-cpu-v1",
        "decision": "CPU_PASS_DESIGN_ONLY" if useful else "KILL_CPU_STAGE_OR_IPC",
        "timing": reports,
        "cold_worker_seconds": cold,
        "archive_sha256": ARCHIVE_SHA,
        "serving_sha256": SERVING_SHA,
        "processor_sha256": serving._DIRECT_PROCESSOR_SHA,
        "direct_batch1": _direct,
        "script_sha256": sha256(Path(__file__)),
        "manifest": manifest,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "helper_sha256": sha256(Path(decode.__code__.co_filename)),
        "parent_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "child_max_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "whole_wall_seconds": time.perf_counter() - started,
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
                "cold_worker_seconds": cold,
                "timing": {s: v["stats"] for s, v in reports.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
