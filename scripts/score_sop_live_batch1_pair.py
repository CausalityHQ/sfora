#!/usr/bin/env python3
"""Pair full live batch-one SOP TRAIN Recall@1 outcomes by heldout product."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
DRAW_COUNT = 5_000
SEED = 179019


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def product_bootstrap(delta: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=delta, minlength=len(classes))
    random = np.random.default_rng(SEED)
    draws: np.ndarray = np.empty(DRAW_COUNT, dtype=np.float64)
    for start in range(0, DRAW_COUNT, 512):
        stop = min(start + 512, DRAW_COUNT)
        selected = random.integers(0, len(classes), size=(stop - start, len(classes)))
        draws[start:stop] = totals[selected].sum(axis=1) / counts[selected].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "point": float(delta.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP live batch-one paired source differs")
    candidate = json.loads(args.candidate_receipt.read_text())
    baseline = json.loads(args.baseline_receipt.read_text())
    if (
        candidate.get("schema") != "sfora-sop-siglip2-live-batch1-quality-v1"
        or baseline.get("schema") != "sfora-sop-unicom-live-batch1-quality-v1"
        or candidate.get("query_image_ids_sha256") != baseline.get("query_image_ids_sha256")
        or candidate.get("native_library_sha256") != baseline.get("native_library_sha256")
    ):
        raise ValueError("SOP live batch-one paired receipt authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != (59_551,) or ids.shape != labels.shape:
        raise ValueError("SOP live batch-one paired TRAIN inventory differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    held = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if (
        len(held) != 5_851
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != candidate["query_image_ids_sha256"]
    ):
        raise ValueError("SOP live batch-one paired holdout inventory differs")
    candidate_r1 = np.asarray(candidate["per_query_live_r1"], dtype=np.float64)
    baseline_r1 = np.asarray(baseline["per_query_live_r1"], dtype=np.float64)
    if (
        candidate_r1.shape != (5_851,)
        or baseline_r1.shape != candidate_r1.shape
        or not np.isin(candidate_r1, [0, 1]).all()
        or not np.isin(baseline_r1, [0, 1]).all()
        or abs(float(candidate_r1.mean()) - candidate["live_batch1_r1"]) > 1e-12
        or abs(float(baseline_r1.mean()) - baseline["live_batch1_r1"]) > 1e-12
    ):
        raise ValueError("SOP live batch-one paired outcome inventory differs")
    interval = product_bootstrap(candidate_r1 - baseline_r1, labels[held])
    receipt = {
        "schema": "sfora-sop-live-batch1-paired-quality-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout, full TRAIN gallery",
        "fit_images": len(partition.fit_row_indexes),
        "holdout_queries": len(held),
        "heldout_products": len(np.unique(labels[held])),
        "gallery_images": len(labels),
        "seed": SEED,
        "bootstrap_draws": DRAW_COUNT,
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "candidate_receipt_sha256": sha256(args.candidate_receipt),
        "baseline_receipt_sha256": sha256(args.baseline_receipt),
        "query_image_ids_sha256": candidate["query_image_ids_sha256"],
        "candidate_live_batch1_r1": float(candidate_r1.mean()),
        "baseline_live_batch1_r1": float(baseline_r1.mean()),
        "candidate_minus_baseline_r1": interval,
        "discordant_queries": int(np.count_nonzero(candidate_r1 != baseline_r1)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt["candidate_minus_baseline_r1"]), flush=True)


if __name__ == "__main__":
    main()
