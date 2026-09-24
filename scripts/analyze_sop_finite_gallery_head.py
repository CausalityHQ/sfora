#!/usr/bin/env python3
"""Apply the frozen SOP train-only head screen advancement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
RECEIPT_SHA256 = "5d30439310f5b53e985a88c2f6db078476376c980b767c7c2f9d0f3feb6b6629"
BOOTSTRAPS = 10_000
SEED = 179019


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clustered_interval(
    candidate: np.ndarray, control: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Pair outcomes within query and resample entire heldout products."""

    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    products, group = np.unique(labels, return_inverse=True)
    if len(products) != 1_132 or difference.shape != labels.shape:
        raise ValueError("SOP paired query/product inventory differs")
    totals = np.bincount(group, weights=difference, minlength=len(products))
    sizes = np.bincount(group, minlength=len(products))
    rng = np.random.Generator(np.random.PCG64(SEED))
    resamples = rng.integers(0, len(products), size=(BOOTSTRAPS, len(products)))
    replicated = totals[resamples].sum(axis=1) / sizes[resamples].sum(axis=1)
    lower, upper = np.quantile(replicated, [0.025, 0.975])
    return {
        "delta_percentage_points": float(difference.mean() * 100),
        "lower_95_percentage_points": float(lower * 100),
        "upper_95_percentage_points": float(upper * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.receipt) != RECEIPT_SHA256
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP head analysis authority differs")
    with args.receipt.open() as stream:
        receipt = json.load(stream)
    if (
        receipt["schema"] != "sfora-sop-finite-gallery-head-screen-v1"
        or receipt["seed"] != SEED
        or receipt["fit_rows"] != 53_700
        or receipt["holdout_rows"] != 5_851
    ):
        raise ValueError("SOP head receipt inventory differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        source_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    partition = deterministic_class_partition(
        tuple(map(int, source_labels)), fit_fraction=0.9, seed=SEED
    )
    labels = source_labels[list(partition.validation_row_indexes)]
    arms = receipt["results"]
    comparisons: dict[str, dict[str, float]] = {}
    for score_name in ("all_train_gallery", "holdout_only"):
        for control in ("inbatch_probability", "arcface", "bank_log", "pca"):
            key = f"bank_probability_minus_{control}_{score_name}"
            comparisons[key] = clustered_interval(
                np.asarray(arms["bank_probability"][score_name]["per_query_r1"]),
                np.asarray(arms[control][score_name]["per_query_r1"]),
                labels,
            )
    full_vs_batch = comparisons["bank_probability_minus_inbatch_probability_all_train_gallery"]
    holdout_vs_batch = comparisons["bank_probability_minus_inbatch_probability_holdout_only"]
    full_vs_arcface = comparisons["bank_probability_minus_arcface_all_train_gallery"]
    gate = {
        "full_gallery_gain_at_least_one_point": full_vs_batch["delta_percentage_points"] >= 1.0,
        "full_gallery_interval_excludes_zero": full_vs_batch["lower_95_percentage_points"] > 0,
        "holdout_only_no_regression": holdout_vs_batch["delta_percentage_points"] >= 0,
        "full_gallery_matches_arcface": full_vs_arcface["delta_percentage_points"] >= 0,
    }
    output = {
        "schema": "sfora-sop-finite-gallery-head-decision-v1",
        "claim_eligible": False,
        "receipt_sha256": RECEIPT_SHA256,
        "source_archive_sha256": ARCHIVE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "bootstrap": "10000 paired product-cluster draws with NumPy PCG64 seed 179019",
        "heldout_query_products": len(np.unique(labels)),
        "comparisons": comparisons,
        "gate": gate,
        "advance": all(gate.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(output, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"gate": gate, "advance": output["advance"], "comparisons": comparisons}))


if __name__ == "__main__":
    main()
