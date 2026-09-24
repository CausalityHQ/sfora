#!/usr/bin/env python3
"""Apply the frozen no-regression gate for exploratory SOP-to-CUB transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analyze_sop_finite_gallery_head import sha256

RECEIPT_SHA256 = "bcfeb5f043b663b8417a8490f2d95b7692334b3c56cf02ab2b8f35b2d8b0e116"
SAMPLES = 10_000
SEED = 179019


def class_interval(candidate: np.ndarray, control: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    classes, index = np.unique(labels, return_inverse=True)
    if difference.shape != labels.shape or len(classes) != 100 or len(labels) != 5_924:
        raise ValueError("SOP tail CUB paired population differs")
    totals = np.bincount(index, weights=difference, minlength=len(classes))
    sizes = np.bincount(index, minlength=len(classes))
    samples = np.random.Generator(np.random.PCG64(SEED)).integers(
        0, len(classes), size=(SAMPLES, len(classes))
    )
    draws = totals[samples].sum(axis=1) / sizes[samples].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_percentage_points": float(difference.mean() * 100),
        "lower_95_percentage_points": float(lower * 100),
        "upper_95_percentage_points": float(upper * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or sha256(args.receipt) != RECEIPT_SHA256:
        raise ValueError("SOP tail CUB decision authority differs")
    receipt = json.loads(args.receipt.read_text())
    if (
        receipt["schema"] != "sfora-sop-tail-cub-transfer-v1"
        or receipt["rows"] != 5_924
        or receipt["gallery_bytes_per_item"] != 130
        or set(receipt["results"]) != {"head_arcface", "tail_arcface_eval"}
    ):
        raise ValueError("SOP tail CUB receipt geometry differs")
    labels = np.asarray(receipt["labels"], dtype=np.int64)
    results = receipt["results"]
    comparisons: dict[str, dict[str, float]] = {}
    for wire in ("packed", "float"):
        control = results["head_arcface"][wire]
        candidate = results["tail_arcface_eval"][wire]
        for metric, vector in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
            for score in (control, candidate):
                if not np.isclose(np.mean(score[vector]), score[metric], rtol=0, atol=1e-10):
                    raise ValueError("SOP tail CUB per-query arithmetic differs")
            comparisons[f"{wire}_{metric}"] = class_interval(
                np.asarray(candidate[vector]), np.asarray(control[vector]), labels
            )
    gate = {
        "packed_r1_no_point_regression": comparisons["packed_recall_at_1"]["delta_percentage_points"] >= 0,
        "packed_map_at_r_no_point_regression": comparisons["packed_map_at_r"]["delta_percentage_points"] >= 0,
    }
    output = {
        "schema": "sfora-sop-tail-cub-transfer-decision-v1",
        "claim_eligible": False,
        "receipt_sha256": RECEIPT_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "bootstrap": "10000 paired class-cluster draws with NumPy PCG64 seed 179019; exploratory uncertainty only",
        "comparisons": comparisons,
        "gate": gate,
        "advance_to_cars_transfer": all(gate.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(output, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"gate": gate, "comparisons": comparisons}, sort_keys=True))


if __name__ == "__main__":
    main()
