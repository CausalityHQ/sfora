#!/usr/bin/env python3
"""Apply the frozen SOP cached teacher-tail product-cluster advancement gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analyze_sop_finite_gallery_head import clustered_interval, sha256
from sfora.representation_ceiling import deterministic_class_partition

RECEIPT_SHA256 = "331b8055b970704b1269b3229e0b2114f594033bfe1690c9af736b4631b46aa1"
SOURCE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019


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
        or sha256(args.source_archive) != SOURCE_SHA256
    ):
        raise ValueError("SOP cached teacher-tail decision authority differs")
    with args.receipt.open() as stream:
        receipt = json.load(stream)
    if (
        receipt["schema"] != "sfora-sop-cached-teacher-tail-screen-v1"
        or receipt["script_sha256"]
        != "df9b0eb5f22fb82aeaf92ee5dd8df4355f971937ecf5ff5b3ee0ae49c35effd4"
        or receipt["seed"] != SEED
        or receipt["fit_rows"] != 53_700
        or receipt["holdout_rows"] != 5_851
        or receipt["updates_per_arm"] != 1_595
        or set(receipt["results"])
        != {"head_arcface", "head_teacher", "tail_arcface", "tail_teacher"}
    ):
        raise ValueError("SOP cached teacher-tail run differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        source_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    partition = deterministic_class_partition(
        tuple(map(int, source_labels)), fit_fraction=0.9, seed=SEED
    )
    labels = source_labels[list(partition.validation_row_indexes)]
    if len(labels) != 5_851 or len(np.unique(labels)) != 1_132:
        raise ValueError("SOP heldout product inventory differs")
    arms = receipt["results"]
    for arm in arms.values():
        for gallery in ("holdout_only", "all_train_gallery"):
            scores = arm[gallery]
            for metric, vector in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
                if not np.isclose(np.mean(scores[vector]), scores[metric], rtol=0, atol=1e-10):
                    raise ValueError("SOP metric vector/mean differs")
    comparisons: dict[str, dict[str, float]] = {}
    for control in ("head_teacher", "tail_arcface", "head_arcface"):
        for gallery in ("holdout_only", "all_train_gallery"):
            for metric in ("per_query_r1", "per_query_ap"):
                key = f"tail_teacher_minus_{control}_{gallery}_{metric}"
                comparisons[key] = clustered_interval(
                    np.asarray(arms["tail_teacher"][gallery][metric]),
                    np.asarray(arms[control][gallery][metric]),
                    labels,
                )
    head = comparisons["tail_teacher_minus_head_teacher_holdout_only_per_query_r1"]
    tail = comparisons["tail_teacher_minus_tail_arcface_holdout_only_per_query_r1"]
    full = comparisons["tail_teacher_minus_tail_arcface_all_train_gallery_per_query_r1"]
    map_r = comparisons["tail_teacher_minus_tail_arcface_holdout_only_per_query_ap"]
    gate = {
        "holdout_gain_at_least_one_point_over_head_teacher": head["delta_percentage_points"] >= 1.0,
        "holdout_gain_at_least_one_point_over_tail_arcface": tail["delta_percentage_points"] >= 1.0,
        "head_teacher_paired_product_lower_above_zero": head["lower_95_percentage_points"] > 0,
        "tail_arcface_paired_product_lower_above_zero": tail["lower_95_percentage_points"] > 0,
        "full_gallery_r1_no_regression_vs_tail_arcface": full["delta_percentage_points"] >= 0,
        "holdout_map_at_r_no_regression_vs_tail_arcface": map_r["delta_percentage_points"] >= 0,
    }
    output = {
        "schema": "sfora-sop-cached-teacher-tail-decision-v1",
        "claim_eligible": False,
        "receipt_sha256": RECEIPT_SHA256,
        "source_archive_sha256": SOURCE_SHA256,
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
