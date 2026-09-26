#!/usr/bin/env python3
"""Apply the frozen seed-179024 valid-anchor TRAIN-only rank gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"
BASE_RECEIPT_SHA = "fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89"
TREATMENT_SOURCE_SHA = "57981249fda8ee2d7950d631db444c0d28e18fc3f23a7d38ff24ccc2a7b54388"
FULL_CONTROL_R1 = 0.9855544090800857


def positive_training_costs(receipt: dict[str, object]) -> tuple[float, int]:
    wall = receipt.get("training_wall_including_member_bank_init_seconds")
    peak = receipt.get("training_peak_cuda_allocated_bytes")
    if (
        type(wall) not in (float, int)
        or not math.isfinite(wall)
        or wall <= 0
        or type(peak) is not int
        or peak <= 0
    ):
        raise ValueError("In-Shop training cost differs")
    return float(wall), peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.baseline) != BASE_RECEIPT_SHA
    ):
        raise ValueError("In-Shop valid-anchor authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[index].label for index in held])
    receipts = []
    for path, checkpoint, arm, source, recovered in (
        (
            args.baseline,
            args.baseline_checkpoint,
            "freeze_emb",
            "a01f3cbc6754393bbdbc2aa3b89cdd3c852d703adffea4c22933fbad4759cb0e",
            0,
        ),
        (args.treatment, args.treatment_checkpoint, "freeze_emb_rank", TREATMENT_SOURCE_SHA, 77),
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
            or receipt.get("vision_lr") != 1e-5
            or receipt.get("preflight_sha256") != PREFLIGHT_SHA
            or receipt.get("partition_sha256") != PARTITION_SHA
            or receipt.get("schedule_sha256") != preflight["schedules"]["1000"]["sha256"]
            or receipt.get("fit_rows_sha256") != preflight["fit_sha256"]
            or receipt.get("held_rows_sha256") != preflight["held_sha256"]
            or receipt.get("held_rows") != len(held)
            or receipt.get("frozen_encoder_blocks") != list(range(12))
            or receipt.get("frozen_embeddings") is not True
            or receipt.get("recovered_rank_updates", 0) != recovered
            or receipt.get("rank_inactive_steps")
            != preflight["schedules"]["1000"]["rank_inactive_steps"]
            or len(receipt.get("preclip_grad_norms", ())) != 1000
            or not all(math.isfinite(x) for x in receipt["preclip_grad_norms"])
            or len(receipt.get("step_seconds", ())) != 1000
            or not all(math.isfinite(x) and x > 0 for x in receipt["step_seconds"])
            or sha256(checkpoint) != receipt.get("checkpoint_sha256")
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
                raise ValueError(f"In-Shop {arm} per-query quality differs")
        receipts.append((receipt, sha256(path)))
    baseline, treatment = (item[0] for item in receipts)
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
    if any(baseline[key] != treatment[key] for key in shared):
        raise ValueError("In-Shop paired inputs differ")
    sources = []
    for receipt in (baseline, treatment):
        sources.append(
            {
                Path(path).name: digest
                for path, digest in receipt["source_files_sha256"].items()
                if not Path(path).name.startswith("train_inshop_siglip2_unseen_gallery")
            }
        )
    if sources[0] != sources[1] or len(sources[0]) != 9:
        raise ValueError("In-Shop shared source differs")
    interval = product_bootstrap(
        np.asarray(treatment["quality"]["per_query_r1"], dtype=np.float64)
        - np.asarray(baseline["quality"]["per_query_r1"], dtype=np.float64),
        labels,
    )
    baseline_wall, baseline_peak = positive_training_costs(baseline)
    treatment_wall, treatment_peak = positive_training_costs(treatment)
    wall_ratio = treatment_wall / baseline_wall
    peak_ratio = treatment_peak / baseline_peak
    passed = (
        treatment["quality"]["recall_at_1"] >= FULL_CONTROL_R1
        and treatment["quality"]["map_at_r"] >= baseline["quality"]["map_at_r"]
        and interval["lower_95"] > -0.003
        and wall_ratio <= 1.1
        and peak_ratio <= 1.1
    )
    report = {
        "schema": "sfora-inshop-valid-anchor-rank-train-only-v1",
        "claim_eligible": False,
        "seed": 179024,
        "held_queries_and_gallery": len(held),
        "r1_product_bootstrap": interval,
        "training_wall_ratio": wall_ratio,
        "peak_cuda_ratio": peak_ratio,
        "advance_fresh_seeds": passed,
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
            for arm, (receipt, digest) in zip(("baseline", "treatment"), receipts, strict=True)
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"advance_fresh_seeds": passed, "interval": interval}))


if __name__ == "__main__":
    main()
