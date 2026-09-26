#!/usr/bin/env python3
"""Apply the frozen TRAIN-only true lower-stack freeze gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import cast

import numpy as np
from analyze_inshop_siglip2_unseen_gallery import paired_gate
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"
ARMS: tuple[tuple[str, str, list[int]], ...] = (
    ("control", "d616513d3850b0268fa80ee28f2c16d4d852c0d25b889376650df72c5cb01290", []),
    ("freeze", "1edb0bf0fe51003719f893a50975d17be80283747f93ef980ad45550bee621dd", list(range(12))),
    (
        "freeze_emb",
        "a01f3cbc6754393bbdbc2aa3b89cdd3c852d703adffea4c22933fbad4759cb0e",
        list(range(12)),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop true-freeze readout authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[index].label for index in held])
    receipts = []
    for run, (arm, source, frozen) in zip(
        (args.control, args.freeze, args.treatment), ARMS, strict=True
    ):
        path = run / "receipt.json"
        r = json.loads(path.read_text())
        quality = r.get("quality")
        expected = preflight["schedules"]["1000"]
        if (
            r.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
            or r.get("arm") != arm
            or r.get("seed") != 179024
            or r.get("updates") != 1000
            or r.get("vision_lr", 1e-5) != 1e-5
            or r.get("source_sha256") != source
            or r.get("frozen_encoder_blocks") != frozen
            or r.get("frozen_embeddings", False) != (arm == "freeze_emb")
            or r.get("preflight_sha256") != PREFLIGHT_SHA
            or r.get("partition_sha256") != PARTITION_SHA
            or r.get("schedule_sha256") != expected["sha256"]
            or r.get("fit_rows_sha256") != preflight["fit_sha256"]
            or r.get("held_rows_sha256") != preflight["held_sha256"]
            or r.get("rank_active_updates") != expected["rank_active_updates"]
            or r.get("rank_inactive_steps") != expected["rank_inactive_steps"]
            or r.get("batch_size") != 64
            or r.get("rank_coefficient") != 8.0
            or sha256(run / "checkpoint.pt") != r.get("checkpoint_sha256")
            or not isinstance(quality, dict)
            or len(r.get("preclip_grad_norms", ())) != 1000
            or not all(math.isfinite(value) for value in r["preclip_grad_norms"])
        ):
            raise ValueError(f"In-Shop true-freeze {arm} receipt differs")
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(quality[key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))
                or abs(float(values.mean()) - quality[mean]) > 1e-8
            ):
                raise ValueError(f"In-Shop true-freeze {arm} quality vector differs")
        receipts.append((r, sha256(path)))
    control, freeze, treatment = (row[0] for row in receipts)
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
    if any(r[key] != control[key] for r in (freeze, treatment) for key in common):
        raise ValueError("In-Shop true-freeze paired inputs differ")
    shared_sources = [
        {
            Path(path).name: digest
            for path, digest in r["source_files_sha256"].items()
            if not Path(path).name.startswith("train_inshop_siglip2_unseen_gallery")
        }
        for r in (control, freeze, treatment)
    ]
    if len(shared_sources[0]) != 9 or any(
        sources != shared_sources[0] for sources in shared_sources[1:]
    ):
        raise ValueError("In-Shop true-freeze shared sources differ")
    gate = paired_gate(
        np.asarray(control["quality"]["per_query_r1"], dtype=np.float64),
        np.asarray(treatment["quality"]["per_query_r1"], dtype=np.float64),
        np.asarray(control["quality"]["per_query_ap"], dtype=np.float64),
        np.asarray(treatment["quality"]["per_query_ap"], dtype=np.float64),
        labels,
    )
    r1 = cast(dict[str, float], gate["r1_product_bootstrap"])
    wall = [
        r["training_wall_including_member_bank_init_seconds"] for r in (control, freeze, treatment)
    ]
    peak = [r["training_peak_cuda_allocated_bytes"] for r in (control, freeze, treatment)]
    if any(not math.isfinite(value) or value <= 0 for value in wall + peak):
        raise ValueError("In-Shop true-freeze resource receipt differs")
    quality_pass = (
        treatment["quality"]["recall_at_1"] >= control["quality"]["recall_at_1"]
        and r1["lower_95"] > -0.003
        and treatment["quality"]["map_at_r"] >= control["quality"]["map_at_r"]
    )
    cost_pass = wall[2] <= 0.9 * wall[1] and peak[2] <= 0.9 * peak[1]
    report = {
        "schema": "sfora-inshop-true-freeze-paired-train-only-v1",
        "claim_eligible": False,
        "seed": 179024,
        "held_queries_and_gallery": len(held),
        "paired_control_gate": gate,
        "quality_pass": quality_pass,
        "cost_pass": cost_pass,
        "advance_fresh_seeds": quality_pass and cost_pass,
        "arms": {
            name: {
                "receipt_sha256": receipts[index][1],
                "packed_r1": r["quality"]["recall_at_1"],
                "packed_map_at_r": r["quality"]["map_at_r"],
                "training_wall_seconds": wall[index],
                "peak_cuda_allocated_bytes": peak[index],
            }
            for index, (name, r) in enumerate(
                zip(("control", "freeze", "freeze_emb"), (control, freeze, treatment), strict=True)
            )
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "advance_fresh_seeds": report["advance_fresh_seeds"],
                "quality_pass": quality_pass,
                "cost_pass": cost_pass,
                "paired_control_gate": gate,
            }
        )
    )


if __name__ == "__main__":
    main()
