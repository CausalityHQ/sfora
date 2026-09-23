"""Replay an explicit packed top-k library on the frozen RC4 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _percentile(samples: list[int], percent: int) -> int:
    return sorted(samples)[math.ceil(len(samples) * percent / 100) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--api-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("RC4 library replay output already exists")
    sys.path.insert(0, str(args.api_root.resolve()))
    from sfora.cutile_int8 import CutilePackedInt8Gallery

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
                    for _ in range(50):
                        started = time.perf_counter_ns()
                        gallery.search(query_codes, query_norms, k=10)
                        samples.append(time.perf_counter_ns() - started)
                    result.update(
                        warmups=5,
                        end_to_end_ns=samples,
                        p50_ns=_percentile(samples, 50),
                        p95_ns=_percentile(samples, 95),
                        p99_ns=_percentile(samples, 99),
                        queries_per_second=batch * 1e9 / (sum(samples) / len(samples)),
                    )
                batches[f"{rows}_{batch}"] = result
    receipt = {
        "schema": "sfora-rc4-packed-library-replay-v1",
        "library_sha256": _sha256(args.library),
        "api_sha256": _sha256(args.api_root / "sfora/cutile_int8.py"),
        "fixture_manifest_sha256": _sha256(args.fixture / "manifest.json"),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
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
