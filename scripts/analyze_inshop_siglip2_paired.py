#!/usr/bin/env python3
"""Select one In-Shop arm from the frozen three-seed TRAIN holdout only."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_coverage import PARTITION_SHA256, fit_and_holdout, sha256
from train_inshop_siglip2_compact import FIT_SHA256, HELD_SHA256, SEEDS, UPDATES

from sfora.unicom_inshop import parse_inshop_partition

ARMS = ("bank", "float")
TRAINER_SHA256 = "f6af548085c2c7578d1dfe059bd3348675e6a68bc668e595346a6748182ddb83"
PREFLIGHT_SHA256 = "3102f89577583bfda526865d5a34b1712df0787c376e6e699cc8ff83333d814b"
COST_SHA256 = "6fb8c5acb616e97cf7d739641a01ed87daa7a67fbe6374ac43c66cb1ed92d12c"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or sha256(args.preflight) != PREFLIGHT_SHA256
    ):
        raise ValueError("In-Shop paired readout authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in train)
    fit, held = fit_and_holdout(labels)
    if len(fit) != 23_342 or len(held) != 2_540:
        raise ValueError("In-Shop paired readout split differs")
    held_labels = np.asarray([labels[index] for index in held])
    rows: dict[str, dict[str, dict[str, object]]] = {}
    per_query: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        rows[str(seed)] = {}
        per_query[seed] = {}
        first_input = None
        paired_reference = None
        for arm in ARMS:
            run = args.run_base / f"sfora-inshop-siglip2-{arm}-{seed}-v1"
            path = run / "receipt.json"
            receipt = json.loads(path.read_text())
            quality = receipt.get("quality")
            sources = receipt.get("source_files_sha256")
            if (
                receipt.get("schema") != "sfora-inshop-siglip2-compact-paired-train-v1"
                or receipt.get("arm") != arm
                or receipt.get("seed") != seed
                or receipt.get("updates") != UPDATES
                or receipt.get("batch_size") != 64
                or receipt.get("source_sha256") != TRAINER_SHA256
                or receipt.get("preflight_sha256") != PREFLIGHT_SHA256
                or receipt.get("bank_cost_sha256") != COST_SHA256
                or receipt.get("partition_sha256") != PARTITION_SHA256
                or receipt.get("fit_rows_sha256") != FIT_SHA256
                or receipt.get("held_rows_sha256") != HELD_SHA256
                or receipt.get("schedule_sha256") != preflight["schedules"][str(seed)]["sha256"]
                or receipt.get("rank_coefficient") != {"bank": 8.0, "float": 21.93}[arm]
                or receipt.get("rank_active_updates")
                != preflight["schedules"][str(seed)]["rank_active_updates"]
                or receipt.get("rank_inactive_steps")
                != preflight["schedules"][str(seed)]["singleton_steps"]
                or not isinstance(sources, dict)
                or any(sha256(Path(name)) != digest for name, digest in sources.items())
                or sha256(run / "checkpoint.pt") != receipt.get("checkpoint_sha256")
                or not isinstance(quality, dict)
                or len(receipt.get("step_seconds", ())) != UPDATES
                or not all(math.isfinite(value) and value > 0 for value in receipt["step_seconds"])
                or len(receipt.get("first_input_batch_sha256", ())) != 10
            ):
                raise ValueError(f"In-Shop seed {seed} {arm} receipt differs")
            if first_input is None:
                first_input = receipt["first_input_batch_sha256"]
            elif receipt["first_input_batch_sha256"] != first_input:
                raise ValueError(f"In-Shop seed {seed} paired input differs")
            common = (
                "feature_receipt_sha256",
                "features_sha256",
                "model_file_sha256",
                "pca_sha256",
                "schedule_sha256",
                "source_files_sha256",
                "hardware",
            )
            if paired_reference is None:
                paired_reference = receipt
            elif any(receipt[key] != paired_reference[key] for key in common):
                raise ValueError(f"In-Shop seed {seed} paired geometry differs")
            for metric, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
                values = np.asarray(quality[key], dtype=np.float64)
                if (
                    values.shape != (len(held),)
                    or not np.isfinite(values).all()
                    or abs(float(values.mean()) - quality[metric]) > 1e-8
                ):
                    raise ValueError(f"In-Shop seed {seed} {arm} quality differs")
            per_query[seed][arm] = np.asarray(quality["per_query_r1"], dtype=np.float64)
            wall = receipt["training_wall_including_member_bank_init_seconds"]
            rows[str(seed)][arm] = {
                "receipt_sha256": sha256(path),
                "packed_r1": quality["recall_at_1"],
                "packed_map_at_r": quality["map_at_r"],
                "training_wall_seconds": wall,
                "training_images_per_second": 64_000 / wall,
                "training_peak_cuda_allocated_bytes": receipt["training_peak_cuda_allocated_bytes"],
                "export_seconds": receipt["export_seconds"],
                "score_seconds": receipt["score_seconds"],
            }
    means = {
        arm: {
            name: float(np.mean([rows[str(seed)][arm][name] for seed in SEEDS]))
            for name in (
                "packed_r1",
                "packed_map_at_r",
                "training_wall_seconds",
                "training_images_per_second",
            )
        }
        for arm in ARMS
    }
    delta = np.mean([per_query[seed]["bank"] - per_query[seed]["float"] for seed in SEEDS], axis=0)
    selected = max(ARMS, key=lambda arm: (means[arm]["packed_r1"], means[arm]["packed_map_at_r"]))
    result = {
        "schema": "sfora-inshop-siglip2-paired-holdout-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN product-disjoint fit/holdout; no official query/gallery",
        "seeds": SEEDS,
        "arms": rows,
        "means": means,
        "bank_minus_float_seedwise_r1": [
            float((per_query[seed]["bank"] - per_query[seed]["float"]).mean()) for seed in SEEDS
        ],
        "bank_minus_float_product_bootstrap": product_bootstrap(delta, held_labels),
        "interval_scope": "product bootstrap conditional on three fixed trained seeds",
        "selected_arm": selected,
        "selection_rule": "highest three-seed mean packed R1; mAP@R tie breaker; bank on exact tie",
        "source_sha256": sha256(Path(__file__)),
        "preflight_sha256": PREFLIGHT_SHA256,
        "partition_sha256": PARTITION_SHA256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"selected_arm": selected, "means": means}), flush=True)


if __name__ == "__main__":
    main()
