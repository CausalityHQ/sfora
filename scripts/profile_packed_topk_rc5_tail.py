"""Locate per-call Python/native stalls in the frozen packed top-k replay."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

_MANIFEST_SHA256 = "2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799"
_API_SHA256 = "7e585fa716ad79b0ad9f998ac6eb63a4f9c7a89409eefee2d9367885686c3818"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[int], percentile: int) -> int:
    return sorted(values)[math.ceil(len(values) * percentile / 100) - 1]


def _verify_fixture(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json"
    if _sha256(manifest_path) != _MANIFEST_SHA256:
        raise ValueError("fixture manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        f"gallery_{rows}_{kind}.bin"
        for rows in (1_000_000, 1_000_003)
        for kind in ("codes", "norms")
    } | {f"query_{batch}_{kind}.bin" for batch in (1, 32) for kind in ("codes", "norms")}
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != expected:
        raise ValueError("fixture file set differs")
    for name in expected:
        authority = files[name]
        input_path = path / name
        if (
            input_path.stat().st_size != authority["bytes"]
            or _sha256(input_path) != authority["sha256"]
        ):
            raise ValueError(f"fixture bytes differ: {name}")
    return manifest


def _arrays(path: Path, batch: int) -> tuple[np.ndarray, np.ndarray]:
    codes = np.fromfile(path / f"query_{batch}_codes.bin", dtype=np.int8).reshape(batch, 128)
    norms = np.fromfile(path / f"query_{batch}_norms.bin", dtype="<f2")
    return codes, norms


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--api-root", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-allocations", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _verify_fixture(args.fixture)
    api_path = args.api_root / "sfora/cutile_int8.py"
    if _sha256(api_path) != _API_SHA256:
        raise ValueError("Python API hash differs")
    sys.path.insert(0, str(args.api_root.resolve()))
    from sfora.cutile_int8 import CutilePackedInt8Gallery

    rows = 1_000_000
    gallery_codes = np.fromfile(args.fixture / "gallery_1000000_codes.bin", dtype=np.int8).reshape(
        rows, 128
    )
    gallery_norms = np.fromfile(args.fixture / "gallery_1000000_norms.bin", dtype="<f2")
    events: list[dict[str, object]] = []
    batches: dict[str, object] = {}
    call_label = "outside"

    def gc_callback(phase: str, info: dict[str, int]) -> None:
        events.append(
            {
                "time_ns": time.perf_counter_ns(),
                "call": call_label,
                "phase": phase,
                "generation": info["generation"],
                "collected": info.get("collected"),
                "uncollectable": info.get("uncollectable"),
            }
        )

    gc.callbacks.append(gc_callback)
    try:
        with CutilePackedInt8Gallery.open(
            args.library.resolve(), gallery_codes, gallery_norms
        ) as gallery:
            native_search = gallery._library.sfora_cutile_int8_search
            native_calls: list[dict[str, int]] = []

            def timed_native(*native_args: object) -> int:
                started = time.perf_counter_ns()
                status = native_search(*native_args)
                native_calls.append({"start_ns": started, "end_ns": time.perf_counter_ns()})
                return status

            gallery._library.sfora_cutile_int8_search = timed_native
            if args.trace_allocations:
                tracemalloc.start(5)
            for batch in (1, 32):
                codes, norms = _arrays(args.fixture, batch)
                reference_path = args.expected / f"exact_1000000_{batch}.json"
                reference = json.loads(reference_path.read_text())
                ordinals, scores = gallery.search(codes, norms, k=10)
                if ordinals.reshape(-1).tolist() != reference["fused_ordinals"]:
                    raise AssertionError(f"batch {batch} ordinals differ")
                if scores.reshape(-1).view("<u4").tolist() != reference["fused_score_bits"]:
                    raise AssertionError(f"batch {batch} score bits differ")
                for _ in range(5):
                    gallery.search(codes, norms, k=10)
                samples: list[dict[str, object]] = []
                for index in range(50):
                    call_label = f"{batch}:{index}"
                    before = time.perf_counter_ns()
                    native_before = len(native_calls)
                    blocks_before = sys.getallocatedblocks()
                    gc_before = list(gc.get_count())
                    faults_before = resource.getrusage(resource.RUSAGE_SELF)
                    allocation_before = (
                        tracemalloc.get_traced_memory() if args.trace_allocations else None
                    )
                    gallery.search(codes, norms, k=10)
                    end = time.perf_counter_ns()
                    allocation_after = (
                        tracemalloc.get_traced_memory() if args.trace_allocations else None
                    )
                    faults_after = resource.getrusage(resource.RUSAGE_SELF)
                    call_label = "outside"
                    calls = native_calls[native_before:]
                    samples.append(
                        {
                            "index": index,
                            "start_ns": before,
                            "end_ns": end,
                            "total_ns": end - before,
                            "native_calls": calls,
                            "native_ns": sum(call["end_ns"] - call["start_ns"] for call in calls),
                            "allocated_blocks_before": blocks_before,
                            "allocated_blocks_after": sys.getallocatedblocks(),
                            "gc_count_before": gc_before,
                            "gc_count_after": list(gc.get_count()),
                            "minor_faults_delta": faults_after.ru_minflt - faults_before.ru_minflt,
                            "major_faults_delta": faults_after.ru_majflt - faults_before.ru_majflt,
                            "tracemalloc_before": allocation_before,
                            "tracemalloc_after": allocation_after,
                        }
                    )
                latencies = [int(sample["total_ns"]) for sample in samples]
                batches[str(batch)] = {
                    "reference_sha256": _sha256(reference_path),
                    "warmups": 5,
                    "samples": samples,
                    "p50_ns": _percentile(latencies, 50),
                    "p95_ns": _percentile(latencies, 95),
                    "p99_ns": _percentile(latencies, 99),
                }
    finally:
        gc.callbacks.remove(gc_callback)
        if args.trace_allocations:
            tracemalloc.stop()
    receipt = {
        "schema": "sfora-packed-topk-rc5-tail-trace-v1",
        "library_sha256": _sha256(args.library),
        "api_sha256": _sha256(api_path),
        "fixture_manifest_sha256": _MANIFEST_SHA256,
        "trace_allocations": args.trace_allocations,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "batches": batches,
        "gc_events": events,
    }
    args.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "library_sha256": receipt["library_sha256"],
                "batch_32_p99_ns": batches["32"]["p99_ns"],
                "gc_events": len(events),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
