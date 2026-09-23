"""Replay frozen packed top-k calls with minimal Python GC event logging."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
from array import array
from pathlib import Path

import numpy as np

_FIXTURE_MANIFEST_SHA256 = "2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799"
_RC4_API_SHA256 = "7e585fa716ad79b0ad9f998ac6eb63a4f9c7a89409eefee2d9367885686c3818"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _percentile(samples: list[int], percent: int) -> int:
    return sorted(samples)[math.ceil(len(samples) * percent / 100) - 1]


def verify_fixture(fixture: Path, expected_manifest_sha256: str) -> set[str]:
    manifest_path = fixture / "manifest.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("RC4 fixture manifest differs")
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("RC4 fixture file list differs")
    for name, authority in files.items():
        path = fixture / name
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(authority, dict)
            or not path.is_file()
            or path.stat().st_size != authority.get("bytes")
            or _sha256(path) != authority.get("sha256")
        ):
            raise ValueError(f"RC4 fixture bytes differ: {name}")
    return set(files)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-gc", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("RC4 library replay output already exists")
    expected_files = {
        f"gallery_{rows}_{kind}.bin"
        for rows in (1_000_000, 1_000_003)
        for kind in ("codes", "norms")
    } | {f"query_{batch}_{kind}.bin" for batch in (1, 32) for kind in ("codes", "norms")}
    if verify_fixture(args.fixture, _FIXTURE_MANIFEST_SHA256) != expected_files:
        raise ValueError("RC4 fixture inputs differ")
    if _sha256(args.api_root / "sfora/cutile_int8.py") != _RC4_API_SHA256:
        raise ValueError("RC4 Python API differs")
    sys.path.insert(0, str(args.api_root.resolve()))
    from sfora.cutile_int8 import CutilePackedInt8Gallery

    clock_offset_ns = time.time_ns() - time.perf_counter_ns()
    gc_events: list[dict[str, object]] = []
    current_batch = 0
    current_index = -1

    def gc_callback(phase: str, info: dict[str, int]) -> None:
        gc_events.append(
            {
                "time_ns": time.perf_counter_ns(),
                "batch": current_batch,
                "index": current_index,
                "phase": phase,
                "generation": info["generation"],
                "collected": info.get("collected"),
            }
        )

    if args.disable_gc:
        gc.disable()
    gc_enabled = gc.isenabled()
    gc_threshold = gc.get_threshold()
    gc_count_initial = gc.get_count()
    gc.callbacks.append(gc_callback)
    batches: dict[str, dict[str, object]] = {}
    for rows in (1_000_000, 1_000_003):
        gallery_codes = np.fromfile(
            args.fixture / f"gallery_{rows}_codes.bin", dtype=np.int8
        ).reshape(rows, 128)
        gallery_norms = np.fromfile(args.fixture / f"gallery_{rows}_norms.bin", dtype="<f2")
        with CutilePackedInt8Gallery.open(
            args.library.resolve(), gallery_codes, gallery_norms
        ) as gallery:
            for batch in (1, 32):
                current_batch = batch
                current_index = -1
                query_codes = np.fromfile(
                    args.fixture / f"query_{batch}_codes.bin", dtype=np.int8
                ).reshape(batch, 128)
                query_norms = np.fromfile(args.fixture / f"query_{batch}_norms.bin", dtype="<f2")
                started = time.perf_counter_ns()
                ordinals, scores = gallery.search(query_codes, query_norms, k=10)
                first_call_ns = time.perf_counter_ns() - started
                reference = json.loads((args.expected / f"exact_{rows}_{batch}.json").read_text())
                exact_ordinals = ordinals.reshape(-1).tolist() == reference["fused_ordinals"]
                exact_scores = (
                    scores.reshape(-1).view("<u4").tolist() == reference["fused_score_bits"]
                )
                if not exact_ordinals or not exact_scores:
                    raise AssertionError(
                        f"RC4 library exactness failed: {rows} rows, batch {batch}"
                    )
                result: dict[str, object] = {
                    "exact_ordinals": exact_ordinals,
                    "exact_score_bits": exact_scores,
                    "first_call_ns": first_call_ns,
                }
                if rows == 1_000_000:
                    for _ in range(5):
                        gallery.search(query_codes, query_norms, k=10)
                    samples = []
                    sample_starts = array("Q", [0]) * 50
                    sample_ends = array("Q", [0]) * 50
                    for index in range(50):
                        current_index = index
                        started = time.perf_counter_ns()
                        gallery.search(query_codes, query_norms, k=10)
                        ended = time.perf_counter_ns()
                        samples.append(ended - started)
                        sample_starts[index] = started
                        sample_ends[index] = ended
                    current_index = -1
                    result.update(
                        warmups=5,
                        end_to_end_ns=samples,
                        sample_start_ns=sample_starts.tolist(),
                        sample_end_ns=sample_ends.tolist(),
                        p50_ns=_percentile(samples, 50),
                        p95_ns=_percentile(samples, 95),
                        p99_ns=_percentile(samples, 99),
                        queries_per_second=batch * 1e9 / (sum(samples) / len(samples)),
                    )
                batches[f"{rows}_{batch}"] = result
    gc.callbacks.remove(gc_callback)
    receipt = {
        "schema": "sfora-rc5-packed-library-timeline-v1",
        "library_sha256": _sha256(args.library),
        "api_sha256": _sha256(args.api_root / "sfora/cutile_int8.py"),
        "fixture_manifest_sha256": _sha256(args.fixture / "manifest.json"),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gc_enabled": gc_enabled,
        "gc_threshold": gc_threshold,
        "gc_count_initial": gc_count_initial,
        "clock_offset_ns": clock_offset_ns,
        "gc_events": gc_events,
        "batches": batches,
    }
    args.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    print(
        json.dumps(
            {
                "library_sha256": receipt["library_sha256"],
                "batches": {
                    key: {name: value for name, value in result.items() if name != "end_to_end_ns"}
                    for key, result in batches.items()
                },
                "process_peak_rss_bytes": receipt["process_peak_rss_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
