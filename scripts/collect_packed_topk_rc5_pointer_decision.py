"""Verify the frozen RC5 pointer-dispatch gate from raw replay receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import tarfile
import tempfile
from pathlib import Path

_FIXTURE_SHA256 = "2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799"
_BASELINE_API_SHA256 = "7e585fa716ad79b0ad9f998ac6eb63a4f9c7a89409eefee2d9367885686c3818"
_CANDIDATE_API_SHA256 = "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409"
_BASELINE_LIBRARY_SHA256 = "a9e583881201760088f3d99c180367e587ca68219c79dfe1760a3467f06b8dea"
_CANDIDATE_LIBRARY_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
_CANDIDATE_RUST_SHA256 = "5c5628b5bac9ac4109e3c8133db5928b2c1755ecce89f0c50f217c3a5d653a42"
_NSIGHT_TRACE_SHA256 = "b2e46beee43c255756cea7a69987b15f2e0ab3e5f7294c08410c88826d5b817e"
_NSIGHT_SQLITE_SHA256 = "a38a72df3040b4e93ff69c62c9dc7b545286c0974fc638c6c2c42e93077162bf"
_REFERENCE_SHA256 = {
    "1000000_1": "ca1e391d3c942d127d3f4cc3ba42b888b1e37710e42bc0ac3f1a62e57cb004a4",
    "1000000_32": "afbb2fe2421ca9b147cfc135dafce6fd016ec356bdc58e8c6c24e47ee4200e62",
    "1000003_1": "e915b606fec42a77f7641fd5c482857033980ef0974627d51f0be10e80cbf9fc",
    "1000003_32": "6d2379b579b503732acf002875675b23ba5d707409aa4e2dd6c15649284a741c",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tarfile.open(args.archive, "r:gz") as archive:

        def read(name: str) -> bytes:
            stream = archive.extractfile(name)
            if stream is None:
                raise ValueError(f"missing {name}")
            return stream.read()

        def load(name: str) -> dict:
            return json.loads(read(name))

        original_gc = load("raw/original_gc_objects.json")
        candidate_gc = load("raw/candidate_gc_objects.json")
        pairs = [
            (load(f"raw/pair{i}_rc3.json"), load(f"raw/pair{i}_candidate.json")) for i in (1, 2)
        ]
        memory = load("raw/combined_memory_profile_fixed.json")
        pool = load("raw/combined_pool_peak.json")
        order = load("raw/run_order.json")
        if sha256(read("authority/manifest.json")) != _FIXTURE_SHA256:
            raise AssertionError("fixture manifest authority differs")
        for key, digest in _REFERENCE_SHA256.items():
            if sha256(read(f"authority/exact_{key}.json")) != digest:
                raise AssertionError(f"exact reference authority differs: {key}")
        if sha256(read("authority/baseline_cutile_int8.py")) != _BASELINE_API_SHA256:
            raise AssertionError("baseline API authority differs")
        if sha256(read("source/cutile_int8.py")) != _CANDIDATE_API_SHA256:
            raise AssertionError("candidate API authority differs")
        if sha256(read("source/topk.rs")) != _CANDIDATE_RUST_SHA256:
            raise AssertionError("candidate Rust source authority differs")
        if (
            sha256(read("raw/combined_memory_profile_fixed.nsys-rep")) != _NSIGHT_TRACE_SHA256
            or pool["trace_sha256"] != _NSIGHT_TRACE_SHA256
        ):
            raise AssertionError("Nsight trace hash differs")
        sqlite_bytes = read("raw/combined_memory_profile_fixed.sqlite")
        if sha256(sqlite_bytes) != _NSIGHT_SQLITE_SHA256:
            raise AssertionError("Nsight SQLite export hash differs")
        for item, name in zip(
            order["files"],
            ("pair1_rc3.json", "pair1_candidate.json", "pair2_candidate.json", "pair2_rc3.json"),
            strict=True,
        ):
            if item["name"] != name or item["sha256"] != sha256(read(f"raw/{name}")):
                raise AssertionError("paired run order receipt differs")
        if [item["mtime_ns"] for item in order["files"]] != sorted(
            item["mtime_ns"] for item in order["files"]
        ):
            raise AssertionError("paired run order differs")

    with tempfile.TemporaryDirectory() as temp:
        sqlite_path = Path(temp) / "nsight.sqlite"
        sqlite_path.write_bytes(sqlite_bytes)
        with sqlite3.connect(sqlite_path) as connection:
            event_count, pool_peak, reserved_peak = connection.execute(
                "SELECT COUNT(*), MAX(localMemoryPoolUtilizedSize), "
                "MAX(localMemoryPoolSize) FROM CUDA_GPU_MEMORY_USAGE_EVENTS"
            ).fetchone()
    if event_count != pool["cuda_memory_events"] or pool_peak != pool["tracked_pool_peak_bytes"]:
        raise AssertionError("Nsight pool summary differs from events")
    if reserved_peak != pool["tracked_pool_reserved_peak_bytes"]:
        raise AssertionError("Nsight pool reserve differs from events")

    def tracked(receipt: dict) -> int:
        return sum(
            count
            for kind, count in receipt["new_tracked_object_types"]
            if kind in {"builtins.dict", "ctypes.c_void_p"}
        )

    baseline_objects = tracked(original_gc)
    candidate_objects = tracked(candidate_gc)
    if baseline_objects != 400 or candidate_objects > baseline_objects * 0.2:
        raise AssertionError("causal object gate failed")
    for gc_receipt, api in (
        (original_gc, _BASELINE_API_SHA256),
        (candidate_gc, _CANDIDATE_API_SHA256),
    ):
        if (
            gc_receipt["api_sha256"] != api
            or gc_receipt["library_sha256"] != _BASELINE_LIBRARY_SHA256
            or gc_receipt["fixture_manifest_sha256"] != _FIXTURE_SHA256
            or gc_receipt["reference_sha256"] != _REFERENCE_SHA256["1000000_32"]
        ):
            raise AssertionError("GC probe identity differs")

    pair_rows = []
    passed = True
    for index, (baseline, candidate) in enumerate(pairs, 1):
        for receipt, library, api in (
            (baseline, _BASELINE_LIBRARY_SHA256, _BASELINE_API_SHA256),
            (candidate, _CANDIDATE_LIBRARY_SHA256, _CANDIDATE_API_SHA256),
        ):
            if (
                receipt["fixture_manifest_sha256"] != _FIXTURE_SHA256
                or receipt["library_sha256"] != library
                or receipt["api_sha256"] != api
            ):
                raise AssertionError("paired replay identity differs")
        for batch in (1, 32):
            key = f"1000000_{batch}"
            old = baseline["batches"][key]
            new = candidate["batches"][key]
            if (
                old["reference_sha256"] != _REFERENCE_SHA256[key]
                or new["reference_sha256"] != _REFERENCE_SHA256[key]
            ):
                raise AssertionError("paired exact references differ")
            extra_key = f"1000003_{batch}"
            for extra in (baseline["batches"][extra_key], candidate["batches"][extra_key]):
                if not extra["exact_ordinals"] or not extra["exact_score_bits"]:
                    raise AssertionError("nonmultiple gallery exactness failed")
            for extra in (baseline["batches"][extra_key], candidate["batches"][extra_key]):
                if extra["reference_sha256"] != _REFERENCE_SHA256[extra_key]:
                    raise AssertionError("nonmultiple gallery reference differs")
            for result in (old, new):
                samples = result["end_to_end_ns"]
                if len(samples) != 50 or result["warmups"] != 5:
                    raise AssertionError("replay sample count changed")
                ordered = sorted(samples)
                for percentile in (50, 95, 99):
                    if result[f"p{percentile}_ns"] != ordered[math.ceil(50 * percentile / 100) - 1]:
                        raise AssertionError("percentile does not match raw samples")
                throughput = batch * 1e9 / (sum(samples) / len(samples))
                if not math.isclose(result["queries_per_second"], throughput, rel_tol=1e-12):
                    raise AssertionError("throughput does not match raw samples")
                if not result["exact_ordinals"] or not result["exact_score_bits"]:
                    raise AssertionError("exactness failed")
            improvement = 1 - new["p99_ns"] / old["p99_ns"]
            batch_pass = improvement >= 0.2 if batch == 32 else improvement >= -0.05
            passed &= batch_pass
            pair_rows.append(
                {
                    "pair": index,
                    "batch": batch,
                    "baseline_p50_ms": old["p50_ns"] / 1e6,
                    "baseline_p95_ms": old["p95_ns"] / 1e6,
                    "baseline_p99_ms": old["p99_ns"] / 1e6,
                    "candidate_p50_ms": new["p50_ns"] / 1e6,
                    "candidate_p95_ms": new["p95_ns"] / 1e6,
                    "candidate_p99_ms": new["p99_ns"] / 1e6,
                    "baseline_queries_per_second": old["queries_per_second"],
                    "candidate_queries_per_second": new["queries_per_second"],
                    "p99_improvement_fraction": improvement,
                    "passed": batch_pass,
                }
            )
        passed &= candidate["process_peak_rss_bytes"] < 2 * 1024**3

    passed &= pool_peak < 200_000_000
    if memory["library_sha256"] != _CANDIDATE_LIBRARY_SHA256:
        raise AssertionError("memory profile native library differs")
    if memory["api_sha256"] != _CANDIDATE_API_SHA256:
        raise AssertionError("memory profile Python API differs")
    receipt = {
        "schema": "sfora-packed-topk-rc5-pointer-decision-v1",
        "raw_archive_sha256": sha256(args.archive.read_bytes()),
        "baseline_library_sha256": pairs[0][0]["library_sha256"],
        "candidate_library_sha256": pairs[0][1]["library_sha256"],
        "baseline_api_sha256": original_gc["api_sha256"],
        "candidate_api_sha256": candidate_gc["api_sha256"],
        "retained_gc_objects_baseline": baseline_objects,
        "retained_gc_objects_candidate": candidate_objects,
        "tracked_cuda_pool_peak_bytes": pool_peak,
        "pair_rows": pair_rows,
        "release_gate_passed": passed,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"release_gate_passed": passed, "archive_sha256": receipt["raw_archive_sha256"]})
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
