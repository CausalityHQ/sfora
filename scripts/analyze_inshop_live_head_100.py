#!/usr/bin/env python3
"""Apply the frozen 100-update In-Shop live-head bank screen."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

TRAINER_SHA = "2bbcf55ff4e07ad63f5052ddcbb1fa706b8e239b71ca96c627cb2690393d6799"
PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "preflight", "control", "treatment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
    ):
        raise ValueError("In-Shop live-head authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[row].label for row in held])
    preflight = json.loads(args.preflight.read_text())
    if (
        len(held) != 12_599
        or preflight["fit_sha256"] != digest_rows(fit)
        or preflight["held_sha256"] != digest_rows(held)
    ):
        raise ValueError("In-Shop live-head split differs")
    runs = {
        name: json.loads((path / "receipt.json").read_text())
        for name, path in (("detached", args.control), ("live", args.treatment))
    }
    control, live = runs["detached"], runs["live"]
    common = (
        "schema",
        "seed",
        "updates",
        "batch_size",
        "rank_coefficient",
        "vision_lr",
        "preflight_sha256",
        "feature_receipt_sha256",
        "features_sha256",
        "partition_sha256",
        "model_file_sha256",
        "fit_rows_sha256",
        "held_rows_sha256",
        "schedule_sha256",
        "first_input_batch_sha256",
        "pca_sha256",
        "hardware",
        "source_sha256",
        "source_files_sha256",
        "rank_active_updates",
        "frozen_encoder_blocks",
        "frozen_embeddings",
        "freeze_first_blocks",
        "half_fit_products",
        "tail_blocks_dropped",
    )
    if any(control[key] != live[key] for key in common):
        raise ValueError("In-Shop live-head paired inputs differ")
    vectors = {}
    for name, run in runs.items():
        path = args.control if name == "detached" else args.treatment
        if (
            run["source_sha256"] != TRAINER_SHA
            or run["arm"] != ("freeze_emb" if name == "detached" else "freeze_emb_live")
            or run["live_head_bank"] != (name == "live")
            or run["bank_width"] != (128 if name == "detached" else 1024)
            or run["seed"] != 179024
            or run["updates"] != 100
            or run["frozen_encoder_blocks"] != list(range(16))
            or not run["frozen_embeddings"]
            or run["half_fit_products"]
            or run["tail_blocks_dropped"]
            or run["fit_rows_sha256"] != digest_rows(fit)
            or run["held_rows_sha256"] != digest_rows(held)
            or sha256(path / "checkpoint.pt") != run["checkpoint_sha256"]
            or len(run["preclip_grad_norms"]) != 100
            or not all(math.isfinite(x) for x in run["preclip_grad_norms"])
        ):
            raise ValueError(f"In-Shop {name} bank receipt differs")
        vectors[name] = {}
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(run["quality"][key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or abs(float(values.mean()) - run["quality"][mean]) > 1e-8
            ):
                raise ValueError(f"In-Shop {name} bank quality differs")
            vectors[name][key] = values
    ap_delta = product_bootstrap(
        vectors["live"]["per_query_ap"] - vectors["detached"]["per_query_ap"], labels
    )
    r1_delta = float(np.mean(vectors["live"]["per_query_r1"] - vectors["detached"]["per_query_r1"]))
    wall_ratio = (
        live["training_wall_including_member_bank_init_seconds"]
        / control["training_wall_including_member_bank_init_seconds"]
    )
    peak_ratio = (
        live["training_peak_cuda_allocated_bytes"] / control["training_peak_cuda_allocated_bytes"]
    )
    advance = (
        ap_delta["point"] >= 0.004
        and ap_delta["lower_95"] > 0
        and r1_delta >= 0
        and wall_ratio <= 1.05
        and peak_ratio <= 1.15
    )
    report = {
        "schema": "sfora-inshop-live-head-bank-100-train-gate-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": {
            name: sha256(path / "receipt.json")
            for name, path in (("detached", args.control), ("live", args.treatment))
        },
        "held_rows": len(held),
        "live_minus_detached_map_at_r_product_bootstrap": ap_delta,
        "live_minus_detached_r1": r1_delta,
        "training_wall_ratio": wall_ratio,
        "peak_cuda_ratio": peak_ratio,
        "advance_full_seed": advance,
        "arms": {
            name: {
                "packed_r1": run["quality"]["recall_at_1"],
                "packed_map_at_r": run["quality"]["map_at_r"],
                "training_wall_seconds": run["training_wall_seconds"],
                "training_peak_cuda_allocated_bytes": run["training_peak_cuda_allocated_bytes"],
            }
            for name, run in runs.items()
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
