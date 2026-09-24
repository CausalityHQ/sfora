#!/usr/bin/env python3
"""Apply the frozen SOP token MaxSim signal gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from analyze_sop_finite_gallery_head import clustered_interval, sha256
from sfora.representation_ceiling import deterministic_class_partition

RECEIPT_SHA256 = "5394028503619469a504ae86bc2dd3aa3a3d900ac60ad5033ac988def89539fe"
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
        raise ValueError("SOP token MaxSim decision authority differs")
    receipt = json.loads(args.receipt.read_text())
    if (
        receipt["schema"] != "sfora-sop-token-maxsim-top32-v1"
        or receipt["script_sha256"]
        != "771537b6f6b29da1d59c49b5209a95c7b62f4baeaedc27aafe5baec1207d640e"
        or receipt["seed"] != SEED
        or receipt["heldout_rows"] != 5_851
        or receipt["heldout_products"] != 1_132
        or receipt["shortlist"] != 32
    ):
        raise ValueError("SOP token MaxSim execution differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        source_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    partition = deterministic_class_partition(
        tuple(map(int, source_labels)), fit_fraction=0.9, seed=SEED
    )
    labels = source_labels[list(partition.validation_row_indexes)]
    comparisons: dict[str, dict[str, float]] = {}
    for metric, vector in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        for arm in ("baseline", "maxsim"):
            score = receipt[arm]
            if not np.isclose(np.mean(score[vector]), score[metric], rtol=0, atol=1e-7):
                raise ValueError("SOP token MaxSim per-query arithmetic differs")
        comparisons[metric] = clustered_interval(
            np.asarray(receipt["maxsim"][vector]),
            np.asarray(receipt["baseline"][vector]),
            labels,
        )
    r1 = comparisons["recall_at_1"]
    gate = {
        "r1_gain_at_least_one_point": r1["delta_percentage_points"] >= 1.0,
        "r1_product_lower_above_zero": r1["lower_95_percentage_points"] > 0,
        "map_at_r_no_regression": comparisons["map_at_r"]["delta_percentage_points"] >= 0,
    }
    base = np.asarray(receipt["baseline"]["per_query_r1"], dtype=np.int8)
    candidate = np.asarray(receipt["maxsim"]["per_query_r1"], dtype=np.int8)
    output = {
        "schema": "sfora-sop-token-maxsim-top32-decision-v1",
        "claim_eligible": False,
        "receipt_sha256": RECEIPT_SHA256,
        "source_archive_sha256": SOURCE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "bootstrap": "10000 paired product-cluster draws with NumPy PCG64 seed 179019",
        "comparisons": comparisons,
        "discordant_queries": {
            "maxsim_wins": int(((candidate == 1) & (base == 0)).sum()),
            "baseline_wins": int(((candidate == 0) & (base == 1)).sum()),
        },
        "positive_in_top32": receipt["positive_in_shortlist"],
        "gate": gate,
        "advance_to_side_branch": all(gate.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(output, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"gate": gate, "comparisons": comparisons, "discordant_queries": output["discordant_queries"]}))


if __name__ == "__main__":
    main()
