#!/usr/bin/env python3
"""Read the frozen, paired In-Shop unseen-gallery seed replication."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from analyze_inshop_siglip2_unseen_gallery import paired_gate
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

SEEDS = (179023, 179024, 179025)
NEW_SEEDS = SEEDS[1:]
ARMS = ("control", "freeze")


def clipped_steps(receipt: dict[str, Any]) -> int | None:
    norms = receipt.get("preclip_grad_norms")
    return None if norms is None else sum(value > 1 for value in norms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if (
        args.output.exists()
        or protocol.get("schema") != "sfora-inshop-unseen-gallery-seed-replication-protocol-v1"
        or protocol.get("seeds") != list(SEEDS)
        or protocol.get("arms") != list(ARMS)
        or protocol.get("analyzer_sha256") != sha256(Path(__file__))
        or protocol.get("paired_gate_sha256") != sha256(Path(paired_gate.__code__.co_filename))
        or protocol.get("bootstrap_sha256") != sha256(Path(product_bootstrap.__code__.co_filename))
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop seed replication authority differs")
    old_decision_path = args.run_base / "sfora-inshop-unseen-gallery-decision-v1.json"
    old_decision = json.loads(old_decision_path.read_text())
    if (
        sha256(old_decision_path) != protocol["original_decision_sha256"]
        or old_decision.get("selected_arm") != "freeze"
        or not old_decision["deltas_vs_control"]["freeze"]["gate_pass"]
    ):
        raise ValueError("In-Shop original selection differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[index].label for index in held])
    per_seed = {}
    all_delta = []
    reference_held_sha = None
    reference_inputs = None
    for seed in SEEDS:
        preflight = None
        preflight_sha = None
        if seed in NEW_SEEDS:
            preflight_path = args.run_base / f"inshop-unseen-gallery-preflight-{seed}-v2.json"
            preflight_sha = sha256(preflight_path)
            if preflight_sha != protocol["preflight_sha256"][str(seed)]:
                raise ValueError(f"In-Shop {seed} preflight hash differs")
            preflight = json.loads(preflight_path.read_text())
            if preflight.get("seed") != seed or preflight.get("held_rows") != len(held):
                raise ValueError(f"In-Shop {seed} preflight differs")
        receipts = {}
        for arm in ARMS:
            version = "v1" if seed == SEEDS[0] else "v2"
            run = args.run_base / f"sfora-inshop-unseen-{arm}-{seed}-{version}"
            path = run / "receipt.json"
            receipt = json.loads(path.read_text())
            expected_sha = old_decision["arms"][arm]["receipt_sha256"] if seed == SEEDS[0] else None
            if (
                (expected_sha is not None and sha256(path) != expected_sha)
                or receipt.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
                or receipt.get("arm") != arm
                or receipt.get("seed") != seed
                or receipt.get("updates") != 1_000
                or receipt.get("batch_size") != 64
                or receipt.get("rank_coefficient") != 8.0
                or receipt.get("held_rows") != len(held)
                or receipt.get("frozen_encoder_blocks")
                != (list(range(12)) if arm == "freeze" else [])
                or sha256(run / "checkpoint.pt") != receipt.get("checkpoint_sha256")
            ):
                raise ValueError(f"In-Shop {seed} {arm} receipt differs")
            if seed in NEW_SEEDS:
                assert preflight is not None
                schedule = preflight["schedules"]["1000"]
                durations = np.asarray(receipt.get("step_seconds", ()), dtype=np.float64)
                gradients = np.asarray(receipt.get("preclip_grad_norms", ()), dtype=np.float64)
                if (
                    receipt.get("source_sha256") != protocol["trainer_sha256"]
                    or receipt.get("source_files_sha256") != protocol["source_files_sha256"]
                    or receipt.get("preflight_sha256") != preflight_sha
                    or receipt.get("fit_rows_sha256") != preflight["fit_sha256"]
                    or receipt.get("held_rows_sha256") != preflight["held_sha256"]
                    or receipt.get("schedule_sha256") != schedule["sha256"]
                    or receipt.get("rank_inactive_steps") != schedule["rank_inactive_steps"]
                    or receipt.get("rank_active_updates") != schedule["rank_active_updates"]
                    or receipt.get("workers") != 4
                    or len(receipt.get("first_input_batch_sha256", ())) != 10
                    or durations.shape != (1000,)
                    or gradients.shape != (1000,)
                    or not np.isfinite(durations).all()
                    or not np.isfinite(gradients).all()
                    or np.any(durations <= 0)
                    or np.any(gradients < 0)
                    or not math.isfinite(receipt.get("first_loss", float("nan")))
                    or not math.isfinite(receipt.get("last_loss", float("nan")))
                    or not math.isfinite(
                        receipt.get(
                            "training_wall_including_member_bank_init_seconds", float("nan")
                        )
                    )
                    or receipt["training_wall_including_member_bank_init_seconds"] <= 0
                    or receipt.get("training_peak_cuda_allocated_bytes", 0) <= 0
                    or any(
                        sha256(Path(name)) != digest
                        for name, digest in receipt["source_files_sha256"].items()
                    )
                ):
                    raise ValueError(f"In-Shop {seed} {arm} training authority differs")
            held_sha = receipt.get("held_rows_sha256")
            if reference_held_sha is None:
                reference_held_sha = held_sha
            elif held_sha != reference_held_sha:
                raise ValueError("In-Shop held rows differ across seeds")
            inputs = tuple(
                receipt.get(key)
                for key in (
                    "feature_receipt_sha256",
                    "features_sha256",
                    "model_file_sha256",
                    "pca_sha256",
                    "fit_rows_sha256",
                    "held_rows_sha256",
                    "hardware",
                )
            )
            if reference_inputs is None:
                reference_inputs = inputs
            elif inputs != reference_inputs:
                raise ValueError("In-Shop source inputs differ across seeds")
            quality = receipt.get("quality")
            if not isinstance(quality, dict):
                raise ValueError(f"In-Shop {seed} {arm} quality missing")
            for mean_name, vector_name in (
                ("recall_at_1", "per_query_r1"),
                ("map_at_r", "per_query_ap"),
            ):
                values = np.asarray(quality[vector_name], dtype=np.float64)
                if (
                    values.shape != (len(held),)
                    or abs(float(values.mean()) - quality[mean_name]) > 1e-8
                ):
                    raise ValueError(f"In-Shop {seed} {arm} quality mean differs")
            receipts[arm] = receipt
        control, freeze = (receipts[arm] for arm in ARMS)
        common = (
            "feature_receipt_sha256",
            "features_sha256",
            "model_file_sha256",
            "pca_sha256",
            "fit_rows_sha256",
            "held_rows_sha256",
            "schedule_sha256",
            "source_files_sha256",
            "hardware",
            "first_input_batch_sha256",
        )
        if any(control[key] != freeze[key] for key in common):
            raise ValueError(f"In-Shop {seed} paired inputs differ")
        result = paired_gate(
            np.asarray(control["quality"]["per_query_r1"], dtype=np.float64),
            np.asarray(freeze["quality"]["per_query_r1"], dtype=np.float64),
            np.asarray(control["quality"]["per_query_ap"], dtype=np.float64),
            np.asarray(freeze["quality"]["per_query_ap"], dtype=np.float64),
            labels,
        )
        all_delta.append(
            np.asarray(freeze["quality"]["per_query_r1"], dtype=np.float64)
            - np.asarray(control["quality"]["per_query_r1"], dtype=np.float64)
        )
        version = "v1" if seed == SEEDS[0] else "v2"
        result["arms"] = {
            arm: {
                "receipt_sha256": sha256(
                    args.run_base / f"sfora-inshop-unseen-{arm}-{seed}-{version}" / "receipt.json"
                ),
                "packed_r1": receipts[arm]["quality"]["recall_at_1"],
                "packed_map_at_r": receipts[arm]["quality"]["map_at_r"],
                "training_wall_seconds": receipts[arm][
                    "training_wall_including_member_bank_init_seconds"
                ],
                "peak_cuda_allocated_bytes": receipts[arm]["training_peak_cuda_allocated_bytes"],
                "clipped_steps": clipped_steps(receipts[arm]),
            }
            for arm in ARMS
        }
        per_seed[str(seed)] = result
    pooled = product_bootstrap(np.mean(all_delta, axis=0), labels)
    passed = all(per_seed[str(seed)]["gate_pass"] for seed in SEEDS) and pooled["lower_95"] > 0
    report = {
        "schema": "sfora-inshop-unseen-gallery-seed-replication-decision-v1",
        "claim_eligible": False,
        "seed_replication_pass": passed,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "pooled_r1_product_bootstrap": pooled,
        "interval_scope": (
            "product bootstrap conditional on three fixed trained seeds; no seed-uncertainty claim"
        ),
        "split": "official In-Shop TRAIN product-disjoint held-only symmetric gallery",
        "held_queries_and_gallery": len(held),
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": sha256(Path(__file__)),
        "paired_gate_sha256": sha256(Path(paired_gate.__code__.co_filename)),
        "bootstrap_sha256": sha256(Path(product_bootstrap.__code__.co_filename)),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"seed_replication_pass": passed, "pooled": pooled}), flush=True)


if __name__ == "__main__":
    main()
