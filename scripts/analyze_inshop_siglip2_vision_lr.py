#!/usr/bin/env python3
"""Apply the frozen TRAIN-only In-Shop vision-rate gate to one paired seed."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from analyze_inshop_siglip2_unseen_gallery import paired_gate
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

TRAINER_SHA = "d616513d3850b0268fa80ee28f2c16d4d852c0d25b889376650df72c5cb01290"
PREFLIGHT_SHAS = {
    179024: "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034",
    179025: "9dc46a73d7000996028551600ae84a6e6fe1d8d3e113d6503547a2fc8ca927f7",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    seed = preflight.get("seed")
    if (
        args.output.exists()
        or seed not in PREFLIGHT_SHAS
        or sha256(args.preflight) != PREFLIGHT_SHAS[seed]
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop vision-rate readout authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[index].label for index in held])
    receipts = []
    for run, rate in ((args.control, 1e-5), (args.treatment, 3e-5)):
        receipt_path = run / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        quality = receipt.get("quality")
        schedule = preflight["schedules"]["1000"]
        if (
            receipt.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
            or receipt.get("arm") != "control"
            or receipt.get("seed") != seed
            or receipt.get("updates") != 1000
            or receipt.get("vision_lr") != rate
            or receipt.get("source_sha256") != TRAINER_SHA
            or receipt.get("preflight_sha256") != PREFLIGHT_SHAS[seed]
            or receipt.get("partition_sha256") != PARTITION_SHA
            or receipt.get("schedule_sha256") != schedule["sha256"]
            or receipt.get("held_rows_sha256") != preflight["held_sha256"]
            or receipt.get("fit_rows_sha256") != preflight["fit_sha256"]
            or receipt.get("rank_active_updates") != schedule["rank_active_updates"]
            or receipt.get("rank_inactive_steps") != schedule["rank_inactive_steps"]
            or receipt.get("batch_size") != 64
            or receipt.get("workers") != 4
            or receipt.get("rank_coefficient") != 8.0
            or receipt.get("frozen_encoder_blocks") != []
            or receipt.get("held_rows") != len(held)
            or sha256(run / "checkpoint.pt") != receipt.get("checkpoint_sha256")
            or not isinstance(quality, dict)
            or len(receipt.get("preclip_grad_norms", ())) != 1000
            or not all(math.isfinite(value) for value in receipt["preclip_grad_norms"])
        ):
            raise ValueError("In-Shop vision-rate training receipt differs")
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(quality[key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))
                or abs(float(values.mean()) - quality[mean]) > 1e-8
            ):
                raise ValueError("In-Shop vision-rate quality vector differs")
        receipts.append((receipt, sha256(receipt_path)))
    control, treatment = (row[0] for row in receipts)
    common = (
        "first_input_batch_sha256",
        "feature_receipt_sha256",
        "features_sha256",
        "model_file_sha256",
        "pca_sha256",
        "fit_rows_sha256",
        "held_rows_sha256",
        "schedule_sha256",
        "source_files_sha256",
        "hardware",
    )
    if any(control[key] != treatment[key] for key in common):
        raise ValueError("In-Shop vision-rate paired inputs differ")
    gate = paired_gate(
        np.asarray(control["quality"]["per_query_r1"], dtype=np.float64),
        np.asarray(treatment["quality"]["per_query_r1"], dtype=np.float64),
        np.asarray(control["quality"]["per_query_ap"], dtype=np.float64),
        np.asarray(treatment["quality"]["per_query_ap"], dtype=np.float64),
        labels,
    )
    wall = [row["training_wall_including_member_bank_init_seconds"] for row in (control, treatment)]
    peak = [row["training_peak_cuda_allocated_bytes"] for row in (control, treatment)]
    if any(not math.isfinite(value) or value <= 0 for value in wall + peak):
        raise ValueError("In-Shop vision-rate resource receipt differs")
    cost_pass = wall[1] <= 1.1 * wall[0] and peak[1] <= 1.1 * peak[0]
    report = {
        "schema": "sfora-inshop-vision-lr-paired-train-only-v1",
        "claim_eligible": False,
        "seed": seed,
        "held_queries_and_gallery": len(held),
        "paired_gate": gate,
        "cost_pass": cost_pass,
        "advance_next_seed": gate["gate_pass"] and cost_pass,
        "control": {
            "receipt_sha256": receipts[0][1],
            "packed_r1": control["quality"]["recall_at_1"],
            "packed_map_at_r": control["quality"]["map_at_r"],
            "training_wall_seconds": wall[0],
            "peak_cuda_allocated_bytes": peak[0],
        },
        "treatment": {
            "receipt_sha256": receipts[1][1],
            "packed_r1": treatment["quality"]["recall_at_1"],
            "packed_map_at_r": treatment["quality"]["map_at_r"],
            "training_wall_seconds": wall[1],
            "peak_cuda_allocated_bytes": peak[1],
        },
        "preflight_sha256": PREFLIGHT_SHAS[seed],
        "trainer_sha256": TRAINER_SHA,
        "source_sha256": sha256(Path(__file__)),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {"seed": seed, "advance_next_seed": report["advance_next_seed"], "paired_gate": gate}
        )
    )


if __name__ == "__main__":
    main()
