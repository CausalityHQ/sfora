#!/usr/bin/env python3
"""Apply the frozen full-held In-Shop coordinate-subspace gate."""

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

PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"
TRAINER_SHAS = (
    "d616513d3850b0268fa80ee28f2c16d4d852c0d25b889376650df72c5cb01290",
    "daae20d2a29fab8cd60d1974ed8f324e6e507b10f974b651468dbe1a58978651",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop subspace readout authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[index].label for index in held])
    receipts = []
    for run, arm, source, width in (
        (args.control, "control", TRAINER_SHAS[0], None),
        (args.treatment, "subspace", TRAINER_SHAS[1], 64),
    ):
        path = run / "receipt.json"
        receipt = json.loads(path.read_text())
        quality = receipt.get("quality")
        expected = preflight["schedules"]["1000"]
        if (
            receipt.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
            or receipt.get("arm") != arm
            or receipt.get("seed") != 179024
            or receipt.get("updates") != 1000
            or receipt.get("vision_lr") != 1e-5
            or receipt.get("source_sha256") != source
            or receipt.get("training_coordinates") != width
            or receipt.get("preflight_sha256") != PREFLIGHT_SHA
            or receipt.get("partition_sha256") != PARTITION_SHA
            or receipt.get("schedule_sha256") != expected["sha256"]
            or receipt.get("fit_rows_sha256") != preflight["fit_sha256"]
            or receipt.get("held_rows_sha256") != preflight["held_sha256"]
            or receipt.get("rank_active_updates") != expected["rank_active_updates"]
            or receipt.get("rank_inactive_steps") != expected["rank_inactive_steps"]
            or receipt.get("batch_size") != 64
            or receipt.get("rank_coefficient") != 8.0
            or receipt.get("frozen_encoder_blocks") != []
            or sha256(run / "checkpoint.pt") != receipt.get("checkpoint_sha256")
            or not isinstance(quality, dict)
            or len(receipt.get("preclip_grad_norms", ())) != 1000
            or not all(math.isfinite(value) for value in receipt["preclip_grad_norms"])
        ):
            raise ValueError(f"In-Shop subspace {arm} receipt differs")
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(quality[key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))
                or abs(float(values.mean()) - quality[mean]) > 1e-8
            ):
                raise ValueError(f"In-Shop subspace {arm} quality vector differs")
        receipts.append((receipt, sha256(path)))
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
        "hardware",
    )
    if any(control[key] != treatment[key] for key in common):
        raise ValueError("In-Shop subspace paired inputs differ")
    sources = []
    for receipt in (control, treatment):
        shared = {
            Path(path).name: digest
            for path, digest in receipt["source_files_sha256"].items()
            if not Path(path).name.startswith("train_inshop_siglip2_unseen_gallery")
        }
        sources.append(shared)
    if sources[0] != sources[1] or len(sources[0]) != 9:
        raise ValueError("In-Shop subspace shared source differs")
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
        raise ValueError("In-Shop subspace resource receipt differs")
    cost_pass = wall[1] <= 1.1 * wall[0] and peak[1] <= 1.1 * peak[0]
    report = {
        "schema": "sfora-inshop-subspace-paired-train-only-v1",
        "claim_eligible": False,
        "seed": 179024,
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
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"advance_next_seed": report["advance_next_seed"], "gate": gate}))


if __name__ == "__main__":
    main()
