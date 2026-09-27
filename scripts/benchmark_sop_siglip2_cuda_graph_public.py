#!/usr/bin/env python3
"""Frozen SOP TRAIN public-path screen for the optional batch-1 CUDA graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

import sfora.siglip2_compact_serving as serving
from sfora.siglip2_compact_serving import Siglip2CompactIndex

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
RECEIPT_SHA = "07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
SERVING_SHA = "0616ef30da0a33741a38fe390f834a0431b1885cd18bc40ea6ed593e2749e9ad"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(BytesIO(path.read_bytes())) as image:
            images.append(image.convert("RGB"))
    return images


def stats(raw: list[int]) -> dict[str, float]:
    ms = np.asarray(raw, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.median(ms)),
        "p95_ms": float(np.quantile(ms, 0.95)),
        "mean_ms": float(ms.mean()),
    }


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
        or sha256(Path(serving.__file__)) != SERVING_SHA
    ):
        raise ValueError("SOP public CUDA graph authority differs")
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.source_archive, allow_pickle=False) as source:
        relatives = tuple(PurePosixPath(str(path)) for path in source["train_relative_paths"][:32])
    if any(path.is_absolute() or ".." in path.parts or not path.parts for path in relatives):
        raise ValueError("SOP public CUDA graph paths differ")
    paths = [args.dataset_root.joinpath(*path.parts) for path in relatives]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("SOP public CUDA graph image missing")
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
        expected: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        for batch in (1, 32):
            for row in range(32 if batch == 1 else 1):
                images = decode(paths[row : row + 1] if batch == 1 else paths)
                outputs = []
                for index in indexes.values():
                    assert index.encoder is not None and index.gallery is not None
                    packed = index.encoder.encode_images(images)
                    result = index.gallery.search_packed(packed)
                    outputs.append((packed, result))
                first, second = outputs
                if (
                    not torch.equal(first[0].codes, second[0].codes)
                    or not torch.equal(first[0].inverse_norms, second[0].inverse_norms)
                    or not np.array_equal(first[1][0], second[1][0])
                    or not np.array_equal(first[1][1], second[1][1])
                ):
                    raise ValueError("SOP public CUDA graph packed top-10 differs")
                expected[(batch, row)] = first[1]
        torch.cuda.reset_peak_memory_stats()
        raw = {str(batch): {"eager": [], "graph": []} for batch in (1, 32)}
        for batch, blocks in ((1, 20), (32, 5)):
            for block in range(blocks):
                order = (
                    ("eager", "graph", "graph", "eager")
                    if block % 2 == 0
                    else ("graph", "eager", "eager", "graph")
                )
                for arm in order:
                    for call in range(10):
                        row = (block * 10 + call) % 32 if batch == 1 else 0
                        selected = paths[row : row + 1] if batch == 1 else paths
                        torch.cuda.synchronize()
                        started = time.perf_counter_ns()
                        result = indexes[arm].search_images(decode(selected))
                        torch.cuda.synchronize()
                        raw[str(batch)][arm].append(time.perf_counter_ns() - started)
                        prior = expected[(batch, row)]
                        if not np.array_equal(result[0], prior[0]) or not np.array_equal(
                            result[1], prior[1]
                        ):
                            raise ValueError("SOP public CUDA graph repeated top-10 differs")
                print(json.dumps({"batch": batch, "blocks_done": block + 1}), flush=True)
        results = {
            batch: {arm: stats(values) for arm, values in arms.items()}
            for batch, arms in raw.items()
        }
        peak = torch.cuda.max_memory_allocated()
    gates = {
        "batch1_p50": results["1"]["graph"]["p50_ms"] <= 0.95 * results["1"]["eager"]["p50_ms"],
        "batch1_p95": results["1"]["graph"]["p95_ms"] <= results["1"]["eager"]["p95_ms"],
        "batch32_p95": results["32"]["graph"]["p95_ms"] <= 1.05 * results["32"]["eager"]["p95_ms"],
        "peak_cuda": peak < 3_000_000_000,
    }
    report = {
        "schema": "sfora-sop-siglip2-cuda-graph-public-screen-v2",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "serving_sha256": SERVING_SHA,
        "receipt_sha256": RECEIPT_SHA,
        "image_sha256": [sha256(path) for path in paths],
        "raw_ns": raw,
        "results": results,
        "gates": gates,
        "advance_p99": all(gates.values()),
        "peak_cuda_allocated_bytes": peak,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    payload = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"results": results, "advance_p99": report["advance_p99"]}), flush=True)


if __name__ == "__main__":
    main()
