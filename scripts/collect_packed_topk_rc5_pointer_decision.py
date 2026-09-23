"""Verify the frozen RC5 pointer-dispatch gate from raw replay receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path


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

        def load(name: str) -> dict:
            stream = archive.extractfile(name)
            if stream is None:
                raise ValueError(f"missing {name}")
            return json.load(stream)

        original_gc = load("raw/original_gc_objects.json")
        candidate_gc = load("raw/candidate_gc_objects.json")
        pairs = [
            (load(f"raw/pair{i}_rc3.json"), load(f"raw/pair{i}_candidate.json")) for i in (1, 2)
        ]
        memory = load("raw/combined_memory_profile_fixed.json")
        pool = load("raw/combined_pool_peak.json")
        trace_stream = archive.extractfile("raw/combined_memory_profile_fixed.nsys-rep")
        if trace_stream is None or sha256(trace_stream.read()) != pool["trace_sha256"]:
            raise AssertionError("Nsight trace hash differs")

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
    if original_gc["library_sha256"] != candidate_gc["library_sha256"]:
        raise AssertionError("GC probe libraries differ")

    pair_rows = []
    passed = True
    for index, (baseline, candidate) in enumerate(pairs, 1):
        if baseline["fixture_manifest_sha256"] != candidate["fixture_manifest_sha256"]:
            raise AssertionError("paired fixture manifests differ")
        if baseline["library_sha256"] != pairs[0][0]["library_sha256"]:
            raise AssertionError("baseline library changed")
        if candidate["library_sha256"] != pairs[0][1]["library_sha256"]:
            raise AssertionError("candidate library changed")
        if baseline["api_sha256"] != original_gc["api_sha256"]:
            raise AssertionError("baseline API changed")
        if candidate["api_sha256"] != candidate_gc["api_sha256"]:
            raise AssertionError("candidate API changed")
        for batch in (1, 32):
            key = f"1000000_{batch}"
            old = baseline["batches"][key]
            new = candidate["batches"][key]
            if old["reference_sha256"] != new["reference_sha256"]:
                raise AssertionError("paired exact references differ")
            extra_key = f"1000003_{batch}"
            for extra in (baseline["batches"][extra_key], candidate["batches"][extra_key]):
                if not extra["exact_ordinals"] or not extra["exact_score_bits"]:
                    raise AssertionError("nonmultiple gallery exactness failed")
            if (
                baseline["batches"][extra_key]["reference_sha256"]
                != candidate["batches"][extra_key]["reference_sha256"]
            ):
                raise AssertionError("nonmultiple gallery references differ")
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

    pool_peak = pool["tracked_pool_peak_bytes"]
    passed &= pool_peak < 200_000_000
    passed &= memory["library_sha256"] == pairs[0][1]["library_sha256"]
    passed &= memory["api_sha256"] == pairs[0][1]["api_sha256"]
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


if __name__ == "__main__":
    main()
