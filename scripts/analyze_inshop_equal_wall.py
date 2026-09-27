#!/usr/bin/env python3
"""Decide the frozen same-source In-Shop equal-wall TRAIN pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_unseen_gallery import digest_rows, schedule, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

TRAINER_SHA = "de7a3fde54decd78c0c42aa29973c8c714d01c4a85a6a6ce1d19358642470801"
PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"


def gate_pass(r1_lower: float, map_delta: float, wall_ratio: float, peak_ratio: float) -> bool:
    return r1_lower > 0 and map_delta >= -0.002 and wall_ratio <= 1.02 and peak_ratio <= 1


def schedule_digest(batches: tuple[tuple[int, ...], ...]) -> str:
    return hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.preflight) != PREFLIGHT_SHA:
        raise ValueError("equal-wall decision authority differs")
    preflight = json.loads(args.preflight.read_text())
    rows = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in rows)
    fit, held = split(labels)
    if (
        len(rows) != 25_882
        or len(fit) != 13_283
        or len(held) != 12_599
        or preflight.get("seed") != 179024
        or preflight.get("fit_sha256") != digest_rows(fit)
        or preflight.get("held_sha256") != digest_rows(held)
    ):
        raise ValueError("equal-wall split differs")
    fit_labels = tuple(labels[i] for i in fit)
    held_labels = np.asarray([labels[i] for i in held])
    runs = {}
    common = (
        "schema",
        "seed",
        "batch_size",
        "workers",
        "rank_coefficient",
        "vision_lr",
        "feature_receipt_sha256",
        "features_sha256",
        "model_file_sha256",
        "fit_rows_sha256",
        "held_rows_sha256",
        "pca_sha256",
        "source_sha256",
        "source_files_sha256",
        "first_input_batch_sha256",
        "hardware",
    )
    for name, arm, updates, horizon in (
        ("control-1000", "control", 1_000, 1_000),
        ("freeze-1533", "freeze_emb", 1_533, 3_000),
    ):
        directory = args.run_root / name
        path = directory / "receipt.json"
        receipt = json.loads(path.read_text())
        quality = receipt.get("quality")
        batches = schedule(fit_labels, horizon, seed=179024)
        inactive = preflight["schedules"][str(horizon)]["rank_inactive_steps"]
        expected_inactive = [step for step in inactive if step <= updates]
        if (
            receipt.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
            or receipt.get("arm") != arm
            or receipt.get("seed") != 179024
            or receipt.get("updates") != updates
            or receipt.get("source_sha256") != TRAINER_SHA
            or receipt.get("preflight_sha256") != PREFLIGHT_SHA
            or receipt.get("fit_rows_sha256") != digest_rows(fit)
            or receipt.get("held_rows_sha256") != digest_rows(held)
            or receipt.get("schedule_sha256") != schedule_digest(batches)
            or receipt.get("executed_schedule_sha256") != schedule_digest(batches[:updates])
            or receipt.get("rank_inactive_steps") != expected_inactive
            or receipt.get("rank_active_updates") != updates - len(expected_inactive)
            or receipt.get("batch_size") != 64
            or receipt.get("frozen_encoder_blocks")
            != (list(range(12)) if arm == "freeze_emb" else [])
            or receipt.get("frozen_embeddings") != (arm == "freeze_emb")
            or len(receipt.get("first_input_batch_sha256", ())) != 10
            or len(receipt.get("step_seconds", ())) != updates
            or len(receipt.get("preclip_grad_norms", ())) != updates
            or not isinstance(quality, dict)
            or sha256(directory / "checkpoint.pt") != receipt.get("checkpoint_sha256")
            or any(
                sha256(Path(p)) != digest for p, digest in receipt["source_files_sha256"].items()
            )
        ):
            raise ValueError(f"equal-wall {name} authority differs")
        r1 = np.asarray(quality["per_query_r1"], dtype=np.float64)
        ap = np.asarray(quality["per_query_ap"], dtype=np.float64)
        wall = receipt["training_wall_including_member_bank_init_seconds"]
        peak = receipt["training_peak_cuda_allocated_bytes"]
        if (
            r1.shape != (len(held),)
            or ap.shape != (len(held),)
            or not np.isin(r1, (0, 1)).all()
            or not np.isfinite(ap).all()
            or np.any((ap < 0) | (ap > 1))
            or abs(float(r1.mean()) - quality["recall_at_1"]) > 1e-12
            or abs(float(ap.mean()) - quality["map_at_r"]) > 1e-7
            or not math.isfinite(wall)
            or wall <= 0
            or peak <= 0
        ):
            raise ValueError(f"equal-wall {name} metrics differ")
        runs[name] = {"receipt": receipt, "sha256": sha256(path), "r1": r1, "ap": ap}
    control = runs["control-1000"]
    treatment = runs["freeze-1533"]
    for key in common:
        if control["receipt"][key] != treatment["receipt"][key]:
            raise ValueError(f"equal-wall pair differs: {key}")
    r1_delta = product_bootstrap(treatment["r1"] - control["r1"], held_labels)
    ap_delta = product_bootstrap(treatment["ap"] - control["ap"], held_labels)
    control_wall = control["receipt"]["training_wall_including_member_bank_init_seconds"]
    treatment_wall = treatment["receipt"]["training_wall_including_member_bank_init_seconds"]
    control_peak = control["receipt"]["training_peak_cuda_allocated_bytes"]
    treatment_peak = treatment["receipt"]["training_peak_cuda_allocated_bytes"]
    wall_ratio = treatment_wall / control_wall
    peak_ratio = treatment_peak / control_peak
    report = {
        "schema": "sfora-inshop-equal-wall-train-decision-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN; 12,599 product-disjoint held-only symmetric gallery",
        "seed": 179024,
        "arms": {
            name: {
                "receipt_sha256": run["sha256"],
                "packed_r1": float(run["r1"].mean()),
                "packed_map_at_r": float(run["ap"].mean()),
                "training_wall_including_bank_seconds": run["receipt"][
                    "training_wall_including_member_bank_init_seconds"
                ],
                "training_images_per_second": 64
                * run["receipt"]["updates"]
                / run["receipt"]["training_wall_including_member_bank_init_seconds"],
                "peak_cuda_allocated_bytes": run["receipt"]["training_peak_cuda_allocated_bytes"],
            }
            for name, run in runs.items()
        },
        "r1_delta_product_bootstrap": r1_delta,
        "map_at_r_delta_product_bootstrap": ap_delta,
        "wall_ratio": wall_ratio,
        "peak_cuda_ratio": peak_ratio,
        "advance_seed179026": gate_pass(
            r1_delta["lower_95"], ap_delta["point"], wall_ratio, peak_ratio
        ),
        "source_sha256": sha256(Path(__file__)),
        "bootstrap_source_sha256": sha256(Path(product_bootstrap.__code__.co_filename)),
        "preflight_sha256": PREFLIGHT_SHA,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "advance_seed179026": report["advance_seed179026"],
                "r1_delta": r1_delta,
                "map_delta": ap_delta["point"],
                "wall_ratio": wall_ratio,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
