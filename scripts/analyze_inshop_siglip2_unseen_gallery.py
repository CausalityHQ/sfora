#!/usr/bin/env python3
"""Choose a new In-Shop training arm using the unseen TRAIN gallery only."""

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

PREFLIGHT_SHA = "d5e22c6a331acbdf3b143b9836597593c2317f9488b99b72f61a4c54baf18eb8"
TRAINER_SHA = "7acea65835a5c079c4b8ad625e0144f41943cf24506ac3265444d77c540f25e3"
ARMS = ("control", "freeze", "budget")


def paired_gate(
    control_r1: np.ndarray,
    freeze_r1: np.ndarray,
    control_ap: np.ndarray,
    freeze_ap: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    arrays = (control_r1, freeze_r1, control_ap, freeze_ap)
    if (
        any(values.shape != labels.shape or not np.isfinite(values).all() for values in arrays)
        or any(np.any((values != 0) & (values != 1)) for values in arrays[:2])
        or any(np.any((values < 0) | (values > 1)) for values in arrays[2:])
    ):
        raise ValueError("In-Shop paired metric vectors differ")
    r1 = product_bootstrap(freeze_r1 - control_r1, labels)
    ap = product_bootstrap(freeze_ap - control_ap, labels)
    return {
        "r1_product_bootstrap": r1,
        "map_at_r_product_bootstrap": ap,
        "gate_pass": r1["lower_95"] > 0 and ap["point"] >= -0.005,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop unseen-gallery readout authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    held_labels = np.asarray([train[index].label for index in held])
    rows = {}
    per_query = {}
    reference = None
    for arm in ARMS:
        run = args.run_base / f"sfora-inshop-unseen-{arm}-179023-v1"
        path = run / "receipt.json"
        r = json.loads(path.read_text())
        updates = 3_000 if arm == "budget" else 1_000
        expected = preflight["schedules"][str(updates)]
        quality = r.get("quality")
        if (
            r.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
            or r.get("arm") != arm
            or r.get("seed") != 179023
            or r.get("updates") != updates
            or r.get("batch_size") != 64
            or r.get("source_sha256") != TRAINER_SHA
            or r.get("preflight_sha256") != PREFLIGHT_SHA
            or r.get("partition_sha256") != PARTITION_SHA
            or r.get("fit_rows") != 13_283
            or r.get("held_rows") != len(held)
            or r.get("fit_rows_sha256") != preflight["fit_sha256"]
            or r.get("held_rows_sha256") != preflight["held_sha256"]
            or r.get("schedule_sha256") != expected["sha256"]
            or r.get("rank_inactive_steps") != expected["rank_inactive_steps"]
            or r.get("rank_active_updates") != expected["rank_active_updates"]
            or r.get("rank_coefficient") != 8.0
            or r.get("frozen_encoder_blocks") != (list(range(12)) if arm == "freeze" else [])
            or sha256(run / "checkpoint.pt") != r.get("checkpoint_sha256")
            or not isinstance(quality, dict)
            or len(r.get("step_seconds", ())) != updates
            or not all(math.isfinite(v) and v > 0 for v in r["step_seconds"])
            or len(r.get("first_input_batch_sha256", ())) != 10
        ):
            raise ValueError(f"In-Shop unseen-gallery {arm} receipt differs")
        for name, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
            values = np.asarray(quality[key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))
                or abs(float(values.mean()) - quality[name]) > 1e-8
            ):
                raise ValueError(f"In-Shop unseen-gallery {arm} quality differs")
        common = (
            "first_input_batch_sha256",
            "feature_receipt_sha256",
            "features_sha256",
            "model_file_sha256",
            "pca_sha256",
            "source_files_sha256",
            "hardware",
        )
        if reference is None:
            reference = r
        elif any(r[key] != reference[key] for key in common):
            raise ValueError(f"In-Shop unseen-gallery {arm} paired inputs differ")
        sources = r["source_files_sha256"]
        if any(sha256(Path(name)) != digest for name, digest in sources.items()):
            raise ValueError(f"In-Shop unseen-gallery {arm} source differs")
        per_query[arm] = np.asarray(quality["per_query_r1"], dtype=np.float64)
        wall = r["training_wall_including_member_bank_init_seconds"]
        rows[arm] = {
            "receipt_sha256": sha256(path),
            "packed_r1": quality["recall_at_1"],
            "packed_map_at_r": quality["map_at_r"],
            "training_wall_seconds": wall,
            "training_images_per_second": updates * 64 / wall,
            "peak_cuda_allocated_bytes": r["training_peak_cuda_allocated_bytes"],
        }
    deltas = {}
    passing = []
    for arm in ("freeze", "budget"):
        interval = product_bootstrap(per_query[arm] - per_query["control"], held_labels)
        map_delta = rows[arm]["packed_map_at_r"] - rows["control"]["packed_map_at_r"]
        passed = interval["lower_95"] > 0 and map_delta >= -0.005
        deltas[arm] = {
            "product_bootstrap": interval,
            "map_at_r_delta": map_delta,
            "gate_pass": passed,
        }
        if passed:
            passing.append(arm)
    selected = (
        max(passing, key=lambda arm: (rows[arm]["packed_r1"], rows[arm]["packed_map_at_r"]))
        if passing
        else None
    )
    report = {
        "schema": "sfora-inshop-siglip2-unseen-gallery-decision-v1",
        "claim_eligible": False,
        "split": "official TRAIN product-disjoint held-only symmetric gallery",
        "held_queries_and_gallery": len(held),
        "held_products": len(set(held_labels)),
        "seed": 179023,
        "arms": rows,
        "deltas_vs_control": deltas,
        "selected_arm": selected,
        "interval_scope": "product bootstrap conditional on one fixed trained seed",
        "selection_rule": "bootstrap R1 lower 95% > 0 and mAP@R delta >= -0.005; best R1",
        "source_sha256": sha256(Path(__file__)),
        "preflight_sha256": PREFLIGHT_SHA,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"selected_arm": selected, "deltas": deltas}), flush=True)


if __name__ == "__main__":
    main()
