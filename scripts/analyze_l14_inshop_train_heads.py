#!/usr/bin/env python3
"""Apply the frozen In-Shop train-head advancement rule to one pinned receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RECEIPT_SHA256 = "7bb00c1aea0cbcd2c8bdc7ab044ddd81e65f8024ad6eeb4d3eb9ba4adb688826"
SCORERS = ("pca128_packed_cosine", "upstream_prefix512_euclidean")
SEED = 179019
RESAMPLES = 10_000
LOWER_BOUND = -0.002


def clustered_recall_interval(
    labels: tuple[int, ...],
    earlier: tuple[float, ...],
    later: tuple[float, ...],
    *,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    """Resample identities and retain all their query outcomes together."""

    if (
        len(labels) < 2
        or len(earlier) != len(labels)
        or len(later) != len(labels)
        or resamples < 2
        or any(value not in (0.0, 1.0) for value in (*earlier, *later))
    ):
        raise ValueError("In-Shop paired recall inventory differs")
    classes, inverse = np.unique(np.asarray(labels), return_inverse=True)
    difference = np.asarray(later, dtype=np.float64) - np.asarray(earlier, dtype=np.float64)
    sums = np.bincount(inverse, weights=difference, minlength=len(classes))
    counts = np.bincount(inverse, minlength=len(classes))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(classes), size=(resamples, len(classes)))
    bootstrap = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {
        "difference": float(difference.mean()),
        "interval_95": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
        "identity_clusters": len(classes),
        "wins": int((difference > 0).sum()),
        "losses": int((difference < 0).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-l14-inshop-train-heads", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("In-Shop head analysis output exists")
    receipt_bytes = args.receipt.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != RECEIPT_SHA256:
        raise ValueError("In-Shop head receipt differs")
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema") != "sfora-l14-heads-inshop-train-query-gallery-v1":
        raise ValueError("In-Shop head receipt schema differs")
    query_labels = tuple(
        receipt["validation_labels"][index] for index in receipt["query_indexes"]
    )
    original = receipt["quality"]["original"]
    rank512 = receipt["quality"]["rank512"]
    exact_fused = receipt["quality"]["exact_fused"]
    rank_comparisons = {
        scorer: clustered_recall_interval(
            query_labels,
            tuple(original[scorer]["per_query_r1"]),
            tuple(rank512[scorer]["per_query_r1"]),
            seed=SEED,
            resamples=RESAMPLES,
        )
        for scorer in SCORERS
    }
    exact_changes = {
        scorer: sum(
            a != b
            for a, b in zip(
                original[scorer]["per_query_r1"],
                exact_fused[scorer]["per_query_r1"],
                strict=True,
            )
        )
        for scorer in original
    }
    result = {
        "schema": "sfora-l14-inshop-train-heads-decision-v1",
        "claim_eligible": False,
        "receipt_sha256": RECEIPT_SHA256,
        "bootstrap_seed": SEED,
        "bootstrap_resamples": RESAMPLES,
        "rank512_minimum_lower_bound": LOWER_BOUND,
        "rank512": rank_comparisons,
        "rank512_advances": all(
            value["interval_95"][0] >= LOWER_BOUND for value in rank_comparisons.values()
        ),
        "exact_fused_changed_r1_queries": exact_changes,
        "exact_fused_max_abs_difference": receipt["exact_fused_max_abs_difference"],
        "exact_fused_advances": (
            receipt["exact_fused_max_abs_difference"] <= 5e-6
            and all(changes == 0 for changes in exact_changes.values())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
