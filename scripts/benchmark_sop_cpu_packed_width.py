"""Measure exact CPU packed-search width cost on authenticated SOP train rows.

The 128-D arm uses the normalized first 128 pretrained coordinates solely as
a width-matched timing control. This receipt makes no retrieval-quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.packed_int8_search import CpuPackedInt8Gallery

ARCHIVE_SHA = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
WARMUP = 25
CALLS = 200


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_model() -> str:
    try:
        report = subprocess.check_output(["lscpu"], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return platform.processor()
    for line in report.splitlines():
        if line.startswith("Model name:"):
            return line.partition(":")[2].strip()
    return platform.processor()


def take_rows(packed: PackedInt8Embeddings, count: int) -> PackedInt8Embeddings:
    return PackedInt8Embeddings(
        codes=packed.codes[:count].contiguous(),
        inverse_norms=packed.inverse_norms[:count].contiguous(),
    )


def verify_exact(gallery: PackedInt8Embeddings, query: PackedInt8Embeddings) -> None:
    scorer = CpuPackedInt8Gallery.open_packed(gallery)
    actual_ordinals, actual_scores = scorer.search_packed(query, k=10)
    codes = gallery.codes.float().T
    gallery_norms = gallery.inverse_norms.float()
    for row in range(query.codes.shape[0]):
        scores = (
            (query.codes[row].float() @ codes) * query.inverse_norms[row].float() * gallery_norms
        ).numpy()
        expected = np.lexsort((np.arange(len(scores)), -scores))[:10]
        if not np.array_equal(actual_ordinals[row], expected):
            raise ValueError("CPU packed top-k ordinals differ from stable full sort")
        if not np.array_equal(actual_scores[row], scores[expected]):
            raise ValueError("CPU packed top-k scores differ from reference")


def timed_cell(gallery: CpuPackedInt8Gallery, query: PackedInt8Embeddings) -> dict[str, object]:
    for _ in range(WARMUP):
        gallery.search_packed(query, k=10)
    elapsed = np.empty(CALLS, dtype=np.float64)
    last: tuple[np.ndarray, np.ndarray] | None = None
    for index in range(CALLS):
        started = time.perf_counter_ns()
        last = gallery.search_packed(query, k=10)
        elapsed[index] = (time.perf_counter_ns() - started) / 1_000_000
    assert last is not None
    return {
        "calls": CALLS,
        "warmup_calls": WARMUP,
        "p50_ms": float(np.percentile(elapsed, 50)),
        "p95_ms": float(np.percentile(elapsed, 95)),
        "p99_ms_diagnostic_only": float(np.percentile(elapsed, 99)),
        "mean_ms": float(elapsed.mean()),
        "completed_top1_ordinal": int(last[0][0, 0]),
        "timed_ms": elapsed.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.source_archive) != ARCHIVE_SHA:
        raise ValueError("SOP packed CPU benchmark input or output differs")
    torch.set_num_threads(4)
    with np.load(args.source_archive, allow_pickle=False) as archive:
        source = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
    if source.shape != (59_551, 768):
        raise ValueError("SOP packed CPU benchmark row inventory differs")
    values = torch.from_numpy(source)
    result: dict[str, object] = {
        "schema": "sfora-sop-cpu-packed-width-cost-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": "official train 59551-image gallery; first 1 or 32 gallery rows reused as queries",
        "source_archive_sha256": ARCHIVE_SHA,
        "script_sha256": sha256(Path(__file__)),
        "gallery_rows": 59_551,
        "k": 10,
        "gallery_block_rows": 65_536,
        "threads": 4,
        "hardware": {
            "cpu_model": cpu_model(),
            "logical_cpus": os.cpu_count(),
            "architecture": platform.machine(),
            "torch": torch.__version__,
        },
        "warning": (
            "128D prefix is a timing control, not a learned compact descriptor; "
            "200 calls do not certify p99"
        ),
        "cells": {},
    }
    cells: dict[str, object] = {}
    for width in (128, 768):
        normalized = F.normalize(values[:, :width].contiguous(), dim=1)
        started = time.perf_counter()
        packed = pack_int8_unit_embeddings(normalized)
        pack_seconds = time.perf_counter() - started
        started = time.perf_counter()
        gallery = CpuPackedInt8Gallery.open_packed(packed)
        prepare_seconds = time.perf_counter() - started
        verify_exact(packed, take_rows(packed, 4))
        batch1 = timed_cell(gallery, take_rows(packed, 1))
        batch32 = timed_cell(gallery, take_rows(packed, 32))
        cells[str(width)] = {
            "bytes_per_item": width + 2,
            "gallery_wire_bytes": 59_551 * (width + 2),
            "pack_seconds": pack_seconds,
            "gallery_prepare_seconds": prepare_seconds,
            "batch1": batch1,
            "batch32": batch32,
        }
        print(
            json.dumps(
                {
                    "width": width,
                    "batch1_p50_ms": batch1["p50_ms"],
                    "batch32_p50_ms": batch32["p50_ms"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    result["cells"] = cells
    result["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
