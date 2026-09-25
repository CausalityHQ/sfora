#!/usr/bin/env python3
"""Pair live PIL-image-to-native-top-10 timing for trained SOP control and bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

from sfora.representation_ceiling import deterministic_class_partition
from sfora.siglip2_compact_serving import Siglip2CompactIndex

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
CONTROL_RECEIPT_SHA256 = "d88167bfcbf8152ee912c8382061afaf248e45fe52ae24cf5f1a5da739477fc3"
BANK_RECEIPT_SHA256 = "2e73ee0e6252c91d54c815d52581abd0d303de09a56776a2fcd5b5e7908c981c"
SEED = 179019


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_order(block: int) -> tuple[str, str]:
    """AB/BA/BA/AB balances position every four blocks."""
    return ("control", "bank") if block % 4 in (0, 3) else ("bank", "control")


def summarize_ns(values: list[int]) -> dict[str, float | int]:
    if not values or any(value <= 0 for value in values):
        raise ValueError("live latency samples differ")
    milliseconds = np.asarray(values, dtype=np.float64) / 1e6
    return {
        "calls": len(values),
        "p50_ms": float(np.quantile(milliseconds, 0.5)),
        "p95_ms": float(np.quantile(milliseconds, 0.95)),
        "p99_ms": float(np.quantile(milliseconds, 0.99)),
        "mean_ms": float(milliseconds.mean()),
        "calls_per_second": float(len(values) / (sum(values) / 1e9)),
    }


def image_panel(
    archive: Path, dataset_root: Path, query_sha256: str
) -> tuple[list[Image.Image], list[int], list[str], list[str]]:
    with np.load(archive, allow_pickle=False) as source:
        labels = np.asarray(source["train_labels"], dtype=np.int64)
        ids = np.asarray(source["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(source["train_relative_paths"]).astype(str)
    if labels.shape != (59_551,) or ids.shape != labels.shape or relatives.shape != labels.shape:
        raise ValueError("SOP live TRAIN inventory differs")
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=SEED
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if len(held) != 5_851 or hashlib.sha256(ids[held].tobytes()).hexdigest() != query_sha256:
        raise ValueError("SOP live holdout inventory differs")
    rows = held[np.linspace(0, len(held) - 1, 32, dtype=int)]
    images: list[Image.Image] = []
    digests: list[str] = []
    for row in rows:
        relative = PurePosixPath(str(relatives[row]))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("SOP live image path differs")
        path = dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP live image missing or symlinked")
        digests.append(sha256(path))
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images, ids[rows].astype(int).tolist(), digests, labels[rows].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=100)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.blocks < 20
        or args.blocks % 4 != 0
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP live timing authority differs")
    directories = {"control": args.control_dir, "bank": args.bank_dir}
    expected = {"control": CONTROL_RECEIPT_SHA256, "bank": BANK_RECEIPT_SHA256}
    receipts = {}
    for arm, directory in directories.items():
        receipt = directory / "receipt.json"
        if sha256(receipt) != expected[arm]:
            raise ValueError("SOP live training receipt differs")
        receipts[arm] = json.loads(receipt.read_text())
    if (
        receipts["control"]["query_image_ids_sha256"] != receipts["bank"]["query_image_ids_sha256"]
        or receipts["control"]["model_file_sha256"] != receipts["bank"]["model_file_sha256"]
        or receipts["control"]["source_sha256"] != receipts["bank"]["source_sha256"]
        or receipts["control"]["native_library_sha256"] != receipts["bank"]["native_library_sha256"]
    ):
        raise ValueError("SOP live paired model authority differs")
    images, image_ids, image_hashes, labels = image_panel(
        args.source_archive, args.dataset_root, receipts["control"]["query_image_ids_sha256"]
    )
    indexes = {}
    try:
        for arm, directory in directories.items():
            indexes[arm] = Siglip2CompactIndex.from_artifacts(
                model_snapshot=args.model_snapshot,
                training_receipt=directory / "receipt.json",
                training_checkpoint=directory / "checkpoint.pt",
                train_embeddings=directory / "train_embeddings.npy",
                native_library=args.native_library,
                expected_receipt_sha256=expected[arm],
                precision="fp16_native",
            )
        torch.cuda.reset_peak_memory_stats()
        result = {}
        for batch_size in (1, 32):
            panel = images[:batch_size]
            for arm in ("control", "bank"):
                for _ in range(5):
                    indexes[arm].search_images(panel)
            samples = {arm: [] for arm in ("control", "bank")}
            output_hashes = {arm: set() for arm in ("control", "bank")}
            for block in range(args.blocks):
                for arm in paired_order(block):
                    started = time.perf_counter_ns()
                    ordinals, scores = indexes[arm].search_images(panel)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter_ns() - started
                    if (
                        ordinals.shape != (batch_size, 10)
                        or scores.shape != ordinals.shape
                        or not np.isfinite(scores).all()
                    ):
                        raise ValueError("SOP live native top-10 geometry differs")
                    samples[arm].append(elapsed)
                    output_hashes[arm].add(
                        hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest()
                    )
            if any(len(output_hashes[arm]) != 1 for arm in output_hashes):
                raise ValueError("SOP live output changed between paired calls")
            result[str(batch_size)] = {
                arm: {
                    **summarize_ns(samples[arm]),
                    "queries_per_second": batch_size
                    * len(samples[arm])
                    / (sum(samples[arm]) / 1e9),
                    "top10_result_sha256": next(iter(output_hashes[arm])),
                }
                for arm in ("control", "bank")
            }
        peak = torch.cuda.max_memory_allocated()
    finally:
        for index in indexes.values():
            index.close()
    output = {
        "schema": "sfora-sop-siglip2-bank-live-paired-diagnostic-v1",
        "claim_eligible": False,
        "timing_scope": (
            "preloaded PIL images to exact native top-10; "
            "no disk read; descriptive p99"
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "blocks": args.blocks,
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "control_receipt_sha256": CONTROL_RECEIPT_SHA256,
        "bank_receipt_sha256": BANK_RECEIPT_SHA256,
        "query_image_ids": image_ids,
        "query_image_sha256": image_hashes,
        "query_labels": labels,
        "timing": result,
        "peak_cuda_allocated_bytes_after_loading": peak,
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(output, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
