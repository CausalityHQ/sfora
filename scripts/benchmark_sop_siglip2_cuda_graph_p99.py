#!/usr/bin/env python3
"""Frozen varied-image SOP TRAIN p99 and graph-index concurrency gate."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from benchmark_sop_siglip2_cuda_graph_public import (
    ARCHIVE_SHA,
    NATIVE_SHA,
    RECEIPT_SHA,
    SERVING_SHA,
    decode,
    sha256,
)

import sfora.siglip2_compact_serving as serving
from sfora.siglip2_compact_serving import Siglip2CompactIndex

PUBLIC_SHA = "2a0f562f1a47b2888e8e925649a44641b8275ef2c19dace7ab6de581e68920d8"


def percentiles(values: list[int]) -> dict[str, float]:
    ms = np.asarray(values, dtype=np.float64) / 1e6
    return {f"p{p}_ms": float(np.percentile(ms, p)) for p in (50, 95, 99)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "source-archive",
        "dataset-root",
        "model-snapshot",
        "receipt",
        "checkpoint",
        "train-embeddings",
        "native-library",
        "public-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.receipt) != RECEIPT_SHA
        or sha256(args.native_library) != NATIVE_SHA
        or sha256(args.public_receipt) != PUBLIC_SHA
        or sha256(Path(serving.__file__)) != SERVING_SHA
    ):
        raise ValueError("SOP CUDA graph p99 authority differs")
    public = json.loads(args.public_receipt.read_text())
    if public.get("advance_p99") is not True:
        raise ValueError("SOP CUDA graph public gate did not pass")
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.source_archive, allow_pickle=False) as source:
        inventory = source["train_relative_paths"]
        paths = []
        image_hashes = []
        selected_rows = []
        seen = set()
        for row, value in enumerate(inventory):
            relative = PurePosixPath(str(value))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("SOP CUDA graph p99 image path differs")
            path = args.dataset_root.joinpath(*relative.parts)
            if not path.is_file() or path.is_symlink():
                raise ValueError("SOP CUDA graph p99 image missing")
            digest = sha256(path)
            if digest in seen:
                continue
            seen.add(digest)
            paths.append(path)
            image_hashes.append(digest)
            selected_rows.append(row)
            if len(paths) == 10_000:
                break
    if len(paths) != 10_000:
        raise ValueError("SOP CUDA graph p99 unique image count differs")
    common = dict(
        model_snapshot=args.model_snapshot,
        training_receipt=args.receipt,
        training_checkpoint=args.checkpoint,
        train_embeddings=args.train_embeddings,
        native_library=args.native_library,
        expected_receipt_sha256=RECEIPT_SHA,
        precision="fp16_native",
    )
    with ExitStack() as stack:
        indexes = {
            "eager": stack.enter_context(Siglip2CompactIndex.from_artifacts(**common)),
            "graph": stack.enter_context(
                Siglip2CompactIndex.from_artifacts(**common, cuda_graph_batch1=True)
            ),
        }
        raw = {arm: [] for arm in indexes}
        expected = []
        torch.cuda.reset_peak_memory_stats()
        for row, path in enumerate(paths):
            order = ("eager", "graph") if row % 2 == 0 else ("graph", "eager")
            results = {}
            for arm in order:
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                result = indexes[arm].search_images(decode([path]))
                torch.cuda.synchronize()
                raw[arm].append(time.perf_counter_ns() - started)
                results[arm] = result
            if not all(
                np.array_equal(a, b)
                for a, b in zip(results["eager"], results["graph"], strict=True)
            ):
                raise ValueError(f"SOP CUDA graph p99 top-10 differs at row {row}")
            if row < 100:
                expected.append(results["eager"])
            if (row + 1) % 1000 == 0:
                print(json.dumps({"images_done": row + 1}), flush=True)

        def concurrent_search(row: int) -> bool:
            got = indexes["graph"].search_images(decode([paths[row]]))
            return all(np.array_equal(a, b) for a, b in zip(got, expected[row], strict=True))

        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrency_exact = all(pool.map(concurrent_search, range(100)))
        peak = torch.cuda.max_memory_allocated()
    summary = {arm: percentiles(values) for arm, values in raw.items()}
    gates = {
        "p50": summary["graph"]["p50_ms"] <= 0.95 * summary["eager"]["p50_ms"],
        "p99": summary["graph"]["p99_ms"] <= summary["eager"]["p99_ms"],
        "concurrency_exact": concurrency_exact,
        "peak_cuda": peak < 3_000_000_000,
    }
    report = {
        "schema": "sfora-sop-siglip2-cuda-graph-p99-v2",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "serving_sha256": SERVING_SHA,
        "public_receipt_sha256": PUBLIC_SHA,
        "source_archive_sha256": ARCHIVE_SHA,
        "image_sha256": image_hashes,
        "source_archive_rows": selected_rows,
        "raw_ns": raw,
        "summary": summary,
        "gates": gates,
        "promote": all(gates.values()),
        "peak_cuda_allocated_bytes": peak,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"summary": summary, "gates": gates}), flush=True)


if __name__ == "__main__":
    main()
