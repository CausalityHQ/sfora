#!/usr/bin/env python3
"""Apply the frozen matched-mode final-block SOP TRAIN advancement gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analyze_sop_finite_gallery_head import clustered_interval, sha256
from sfora.representation_ceiling import deterministic_class_partition

ORIGINAL_SHA256 = "331b8055b970704b1269b3229e0b2114f594033bfe1690c9af736b4631b46aa1"
CANDIDATE_SHA256 = "cb3b03d814f08e0acf9e992f1569bc099566d77409b6895cdb2d3ba763e354cb"
PREFLIGHT_SHA256 = "1a915131f70ba6645a66aef8b3cf52a32c735e10f89de117b94137859978d95b"
SOURCE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.original) != ORIGINAL_SHA256
        or sha256(args.candidate) != CANDIDATE_SHA256
        or sha256(args.preflight) != PREFLIGHT_SHA256
        or sha256(args.source_archive) != SOURCE_SHA256
    ):
        raise ValueError("SOP tail mode-control authority differs")
    original = json.loads(args.original.read_text())
    candidate = json.loads(args.candidate.read_text())
    preflight = json.loads(args.preflight.read_text())
    if (
        original["schema"] != "sfora-sop-cached-teacher-tail-screen-v1"
        or candidate["schema"] != "sfora-sop-tail-eval-control-screen-v1"
        or preflight["schema"] != "sfora-sop-tail-eval-control-preflight-v1"
        or candidate["script_sha256"] != preflight["script_sha256"]
        or candidate["script_sha256"]
        != "9fd785a2fb5ac1c077bad8452f49e08d8a6a4de76f22f9790101cc779237ea65"
        or any(receipt["seed"] != SEED for receipt in (original, candidate, preflight))
        or any(receipt["fit_rows"] != 53_700 or receipt["holdout_rows"] != 5_851
               for receipt in (original, candidate, preflight))
        or candidate["updates_per_arm"] != 1_595
        or original["updates_per_arm"] != 1_595
        or candidate["schedule_sha256"] != original["schedule_sha256"]
        or candidate["learning_rates"] != original["learning_rates"]
        or set(candidate["results"]) != {"tail_arcface_eval"}
        or abs(preflight["results"]["tail_arcface_eval"]["last_arcface_loss"]
               - 6.143701553344727) > 1e-6
        or not preflight["results"]["tail_arcface_eval"]["tail_gradient_present"]
    ):
        raise ValueError("SOP tail mode-control execution differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        source_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    partition = deterministic_class_partition(
        tuple(map(int, source_labels)), fit_fraction=0.9, seed=SEED
    )
    labels = source_labels[list(partition.validation_row_indexes)]
    if len(labels) != 5_851 or len(np.unique(labels)) != 1_132:
        raise ValueError("SOP tail mode-control heldout products differ")
    old = original["results"]["head_arcface"]
    new = candidate["results"]["tail_arcface_eval"]
    comparisons: dict[str, dict[str, float]] = {}
    for gallery in ("holdout_only", "all_train_gallery"):
        for metric, vector in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
            for arm in (old, new):
                score = arm[gallery]
                if not np.isclose(np.mean(score[vector]), score[metric], rtol=0, atol=1e-10):
                    raise ValueError("SOP tail mode-control metric vector differs")
            comparisons[f"{gallery}_{metric}"] = clustered_interval(
                np.asarray(new[gallery][vector]), np.asarray(old[gallery][vector]), labels
            )
    hold = comparisons["holdout_only_recall_at_1"]
    gate = {
        "holdout_r1_gain_at_least_one_point": hold["delta_percentage_points"] >= 1,
        "holdout_r1_product_lower_above_zero": hold["lower_95_percentage_points"] > 0,
        "holdout_map_at_r_no_regression": comparisons["holdout_only_map_at_r"]["delta_percentage_points"] >= 0,
        "full_gallery_r1_no_regression": comparisons["all_train_gallery_recall_at_1"]["delta_percentage_points"] >= 0,
    }
    output = {
        "schema": "sfora-sop-tail-eval-control-decision-v1",
        "claim_eligible": False,
        "original_receipt_sha256": ORIGINAL_SHA256,
        "candidate_receipt_sha256": CANDIDATE_SHA256,
        "preflight_receipt_sha256": PREFLIGHT_SHA256,
        "source_archive_sha256": SOURCE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "bootstrap": "10000 paired product-cluster draws with NumPy PCG64 seed 179019",
        "heldout_query_products": 1_132,
        "comparisons": comparisons,
        "gate": gate,
        "advance_to_exploratory_transfer": all(gate.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(output, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"gate": gate, "comparisons": comparisons}, sort_keys=True))


if __name__ == "__main__":
    main()
