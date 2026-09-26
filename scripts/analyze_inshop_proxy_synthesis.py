#!/usr/bin/env python3
"""Apply the frozen seed-179024 Proxy Synthesis TRAIN-only gate."""

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
SOURCES = (
    "1edb0bf0fe51003719f893a50975d17be80283747f93ef980ad45550bee621dd",
    "a9172a0b439238cd943d9d9e0aaed614c5c508bb94db4b1190b6aa87ad8cf2a6",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop readout authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[index].label for index in held])
    receipts = []
    for path, arm, source in (
        (args.control, "control", SOURCES[0]),
        (args.treatment, "proxy_synthesis", SOURCES[1]),
    ):
        receipt = json.loads(path.read_text())
        quality = receipt.get("quality")
        if (
            receipt.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
            or receipt.get("arm") != arm
            or receipt.get("source_sha256") != source
            or receipt.get("seed") != 179024
            or receipt.get("updates") != 1000
            or receipt.get("batch_size") != 64
            or (arm == "proxy_synthesis" and receipt.get("vision_lr") != 1e-5)
            or receipt.get("preflight_sha256") != PREFLIGHT_SHA
            or receipt.get("partition_sha256") != PARTITION_SHA
            or receipt.get("schedule_sha256") != preflight["schedules"]["1000"]["sha256"]
            or receipt.get("fit_rows_sha256") != preflight["fit_sha256"]
            or receipt.get("held_rows_sha256") != preflight["held_sha256"]
            or receipt.get("held_rows") != len(held)
            or receipt.get("frozen_encoder_blocks") != []
            or len(receipt.get("preclip_grad_norms", ())) != 1000
            or not all(math.isfinite(x) for x in receipt["preclip_grad_norms"])
            or len(receipt.get("step_seconds", ())) != 1000
            or not all(math.isfinite(x) and x > 0 for x in receipt["step_seconds"])
            or not isinstance(quality, dict)
        ):
            raise ValueError(f"In-Shop {arm} receipt differs")
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(quality[key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))
                or abs(float(values.mean()) - quality[mean]) > 1e-8
            ):
                raise ValueError(f"In-Shop {arm} quality vector differs")
        receipts.append((receipt, sha256(path)))
    control, treatment = (item[0] for item in receipts)
    for receipt, checkpoint in zip(
        (control, treatment),
        (args.control_checkpoint, args.treatment_checkpoint),
        strict=True,
    ):
        if sha256(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("In-Shop checkpoint digest differs")
    shared = (
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
    if (
        any(control[key] != treatment[key] for key in shared)
        or len(control["first_input_batch_sha256"]) != 10
    ):
        raise ValueError("In-Shop paired inputs differ")
    sources = []
    for receipt in (control, treatment):
        shared_sources = {
            Path(path).name: digest
            for path, digest in receipt["source_files_sha256"].items()
            if not Path(path).name.startswith("train_inshop_siglip2_unseen_gallery")
        }
        sources.append(shared_sources)
    if sources[0] != sources[1] or len(sources[0]) != 9:
        raise ValueError("In-Shop shared source differs")
    for receipt in (control, treatment):
        wall = receipt["training_wall_including_member_bank_init_seconds"]
        peak = receipt["training_peak_cuda_allocated_bytes"]
        if not math.isfinite(wall) or wall <= 0 or not isinstance(peak, int) or peak <= 0:
            raise ValueError("In-Shop training cost differs")
    if treatment.get("synthetic_classes_per_step") != 64 or not treatment.get(
        "proxy_synthesis_stream_sha256"
    ):
        raise ValueError("Proxy Synthesis stream differs")
    gate = paired_gate(
        *(
            np.asarray(receipt["quality"][key], dtype=np.float64)
            for key in ("per_query_r1", "per_query_ap")
            for receipt in (control, treatment)
        ),
        labels,
    )
    wall_ratio = (
        treatment["training_wall_including_member_bank_init_seconds"]
        / control["training_wall_including_member_bank_init_seconds"]
    )
    peak_ratio = (
        treatment["training_peak_cuda_allocated_bytes"]
        / control["training_peak_cuda_allocated_bytes"]
    )
    report = {
        "schema": "sfora-inshop-proxy-synthesis-paired-train-only-v1",
        "claim_eligible": False,
        "seed": 179024,
        "held_queries_and_gallery": len(held),
        "paired_gate": gate,
        "training_wall_ratio": wall_ratio,
        "peak_cuda_ratio": peak_ratio,
        "advance_next_seed": gate["gate_pass"] and wall_ratio <= 1.1 and peak_ratio <= 1.1,
        **{
            arm: {
                "receipt_sha256": digest,
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "packed_r1": receipt["quality"]["recall_at_1"],
                "packed_map_at_r": receipt["quality"]["map_at_r"],
                "training_wall_seconds": receipt[
                    "training_wall_including_member_bank_init_seconds"
                ],
                "training_images_per_second": 64000
                / receipt["training_wall_including_member_bank_init_seconds"],
                "peak_cuda_allocated_bytes": receipt["training_peak_cuda_allocated_bytes"],
            }
            for arm, (receipt, digest) in zip(("control", "treatment"), receipts, strict=True)
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"advance_next_seed": report["advance_next_seed"], "gate": gate}))


if __name__ == "__main__":
    main()
