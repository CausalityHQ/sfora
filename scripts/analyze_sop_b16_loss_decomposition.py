#!/usr/bin/env python3
"""Class-clustered paired intervals for the pretrained B/16 SOP compression screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019
REPLICATES = 5_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--screen-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.archive) != ARCHIVE_SHA256
        or sha256(args.screen) != args.screen_sha256
    ):
        raise ValueError("SOP B/16 interval input differs")
    with np.load(args.archive, allow_pickle=False) as archive:
        labels = np.ascontiguousarray(archive["test_labels"], dtype=np.int64)
    screen = json.loads(args.screen.read_text())
    if (
        screen.get("schema") != "sfora-unicom-b16-sop-pretrained-screen-v2"
        or screen.get("source_archive_sha256") != ARCHIVE_SHA256
        or labels.shape != (60_502,)
        or len(np.unique(labels)) != 11_316
    ):
        raise ValueError("SOP B/16 interval protocol differs")
    _classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    arm_names = ("float", "pca_float", "pca_packed")
    pairs = (("pca_float", "float"), ("pca_packed", "pca_float"), ("pca_packed", "float"))
    rows = []
    rng = np.random.Generator(np.random.PCG64(SEED))
    for metric in ("per_query_r1", "per_query_ap"):
        values = {
            name: np.asarray(screen[name]["score"][metric], dtype=np.float64)
            for name in arm_names
        }
        if any(
            array.shape != labels.shape or not np.isfinite(array).all()
            for array in values.values()
        ):
            raise ValueError("SOP B/16 interval per-query evidence differs")
        for candidate, baseline in pairs:
            delta = values[candidate] - values[baseline]
            group_sums = np.bincount(inverse, weights=delta, minlength=len(counts))
            samples = np.empty(REPLICATES, dtype=np.float64)
            for start in range(0, REPLICATES, 64):
                stop = min(start + 64, REPLICATES)
                draws = rng.integers(0, len(counts), size=(stop - start, len(counts)))
                samples[start:stop] = group_sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            lo, hi = np.quantile(samples, (0.025, 0.975), method="linear")
            rows.append(
                {
                    "metric": metric,
                    "candidate": candidate,
                    "baseline": baseline,
                    "delta": float(delta.mean()),
                    "class_cluster_bootstrap_ci95": [float(lo), float(hi)],
                    "candidate_only_positive_queries": int(np.count_nonzero(delta > 0)),
                    "baseline_only_positive_queries": int(np.count_nonzero(delta < 0)),
                }
            )
    result = {
        "schema": "sfora-unicom-b16-sop-pretrained-loss-decomposition-v1",
        "claim_eligible": False,
        "screen_sha256": args.screen_sha256,
        "source_archive_sha256": ARCHIVE_SHA256,
        "test_queries": len(labels),
        "test_classes": len(counts),
        "bootstrap_seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "resampling_unit": "official SOP test product identity",
        "script_sha256": sha256(Path(__file__)),
        "rows": rows,
    }
    payload = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
