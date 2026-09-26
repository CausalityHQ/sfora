#!/usr/bin/env python3
"""Interleave matched public SOP image-to-native-top-10 calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from paired_latency_certification import block_bootstrap_p99_ratio
from PIL import Image

from sfora.siglip2_compact_serving import Siglip2CompactIndex

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--precision", choices=("fp32_autocast", "fp16_native"), required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--certify-batch1-p99", action="store_true")
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.native_library) != NATIVE_SHA
    ):
        raise ValueError("SOP latency source or scorer differs")
    decision = json.loads(args.decision.read_text())
    if (
        decision.get("schema") != "sfora-sop-true-freeze-paired-train-only-v1"
        or decision.get("seed") != 179024
    ):
        raise ValueError("SOP latency paired training decision differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
    rows = np.linspace(0, 59_550, 32, dtype=np.int64)
    paths = []
    image_hashes = []
    for row in rows:
        relative = PurePosixPath(str(relatives[row]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("SOP latency image path differs")
        path = args.dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP latency image missing")
        paths.append(path)
        image_hashes.append(sha256(path))
    load_seconds = {}
    sizes = (1,) if args.certify_batch1_p99 else (1, 32)
    blocks = 20 if args.certify_batch1_p99 else 200
    repeats = 250 if args.certify_batch1_p99 else 1
    raw: dict[str, dict[str, list[int]]] = {
        str(size): {"control": [], "freeze": []} for size in sizes
    }
    results: dict[str, dict[str, Any]] = {str(size): {} for size in sizes}
    with ExitStack() as stack:
        indexes = {}
        for arm in ("control", "freeze"):
            run = args.runs / f"sfora-sop-true-freeze-{arm}-179024-1000-v1"
            receipt_path = run / "receipt.json"
            receipt_sha = sha256(receipt_path)
            if decision.get("arms", {}).get(arm, {}).get("receipt_sha256") != receipt_sha:
                raise ValueError(f"SOP latency {arm} checkpoint differs")
            started = time.perf_counter()
            indexes[arm] = stack.enter_context(
                Siglip2CompactIndex.from_artifacts(
                    model_snapshot=args.model_snapshot,
                    training_receipt=receipt_path,
                    training_checkpoint=run / "checkpoint.pt",
                    train_embeddings=run / "train_embeddings.npy",
                    native_library=args.native_library,
                    expected_receipt_sha256=receipt_sha,
                    precision=args.precision,
                )
            )
            load_seconds[arm] = time.perf_counter() - started

        def call(arm: str, size: int) -> int:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            images = []
            for path in paths[:size]:
                with Image.open(BytesIO(path.read_bytes())) as image:
                    images.append(image.convert("RGB"))
            ordinals, scores = indexes[arm].search_images(images)
            torch.cuda.synchronize()
            elapsed = time.perf_counter_ns() - started
            digest = hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest()
            previous = results[str(size)].setdefault(arm, digest)
            if digest != previous:
                raise ValueError("SOP latency result changed across calls")
            return elapsed

        for size in sizes:
            for arm in ("control", "freeze"):
                for _ in range(5):
                    call(arm, size)
            torch.cuda.reset_peak_memory_stats()
            for block in range(blocks):
                order = (
                    ("control", "freeze", "freeze", "control")
                    if block % 2 == 0
                    else ("freeze", "control", "control", "freeze")
                )
                for _ in range(repeats):
                    for arm in order:
                        raw[str(size)][arm].append(call(arm, size))
                if block % max(1, blocks // 10) == max(1, blocks // 10) - 1:
                    print(json.dumps({"batch": size, "blocks_done": block + 1}), flush=True)
            if any(
                len(raw[str(size)][arm]) != blocks * repeats * 2 for arm in ("control", "freeze")
            ):
                raise ValueError("SOP latency call count differs")
            results[str(size)]["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    summary: dict[str, dict[str, Any]] = {}
    for batch_label in map(str, sizes):
        summary[batch_label] = {}
        for arm in ("control", "freeze"):
            milliseconds = np.asarray(raw[batch_label][arm], dtype=np.float64) / 1e6
            summary[batch_label][arm] = {
                "p50_ms": float(np.median(milliseconds)),
                "p95_ms": float(np.quantile(milliseconds, 0.95)),
                "mean_ms": float(milliseconds.mean()),
                "result_sha256": results[batch_label][arm],
                "raw_ns": raw[batch_label][arm],
            }
        summary[batch_label]["peak_cuda_bytes"] = results[batch_label]["peak_cuda_bytes"]
    if args.certify_batch1_p99:
        batch = raw["1"]
        p99 = block_bootstrap_p99_ratio(
            np.asarray(batch["freeze"], dtype=np.int64).reshape(blocks, 500).tolist(),
            np.asarray(batch["control"], dtype=np.int64).reshape(blocks, 500).tolist(),
        )
        passed = (
            p99["ci95_upper"] <= 1.05
            and p99["point_p50_ratio"] <= 1.05
            and p99["mean_latency_ratio"] <= 1.05
        )
        summary["1"]["p99_nonregression"] = p99
    else:
        passed = all(
            summary[size]["freeze"]["p95_ms"] <= 1.05 * summary[size]["control"]["p95_ms"]
            for size in ("1", "32")
        )
    result = {
        "schema": (
            "sfora-sop-true-freeze-public-batch1-p99-v1"
            if args.certify_batch1_p99
            else "sfora-sop-true-freeze-public-latency-v1"
        ),
        "claim_eligible": False,
        "seed": 179024,
        "precision": args.precision,
        "latency_pass": bool(passed),
        "source_sha256": sha256(Path(__file__)),
        "latency_helper_sha256": sha256(Path(block_bootstrap_p99_ratio.__code__.co_filename)),
        "source_archive_sha256": ARCHIVE_SHA,
        "decision_sha256": sha256(args.decision),
        "native_library_sha256": NATIVE_SHA,
        "query_image_ids": ids[rows].tolist(),
        "query_image_sha256": image_hashes,
        "cold_index_load_seconds": load_seconds,
        "timing": summary,
        "gpu": torch.cuda.get_device_name(),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "precision": args.precision,
                "latency_pass": passed,
                "timing": {
                    str(size): {
                        arm: summary[str(size)][arm]["p95_ms"] for arm in ("control", "freeze")
                    }
                    for size in sizes
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
