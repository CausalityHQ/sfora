#!/usr/bin/env python3
"""Apply the frozen 100-update In-Shop encoder-depth feasibility gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHT_SHA = "4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254"
TRAINER_SHA = "92e60e603a88951a071b778ba1f6b2d1c362af9a81fffd78bac117e6c7a823a9"
EXPORTER_SHA = "30f46c7536f546a264295da9c22de9cde0738bd02f5ca1bd51dcd1a9b8a659b6"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "preflight",
        "full-cache",
        "depth22-cache",
        "control",
        "treatment",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
    ):
        raise ValueError("In-Shop depth screen authority differs")
    preflight = json.loads(args.preflight.read_text())
    records = tuple(
        row for row in parse_inshop_partition(args.dataset_root) if row.split == "train"
    )
    fit, held = split(tuple(row.label for row in records))
    if preflight["fit_sha256"] != digest_rows(fit) or preflight["held_sha256"] != digest_rows(held):
        raise ValueError("In-Shop depth screen split differs")
    caches = [
        json.loads((path / "receipt.json").read_text())
        for path in (args.full_cache, args.depth22_cache)
    ]
    runs = [
        json.loads((path / "receipt.json").read_text()) for path in (args.control, args.treatment)
    ]
    for depth, cache, run, cache_dir, run_dir in zip(
        (0, 2),
        caches,
        runs,
        (args.full_cache, args.depth22_cache),
        (args.control, args.treatment),
        strict=True,
    ):
        if (
            cache["schema"] != "sfora-inshop-siglip2-train-feature-export-v1"
            or cache.get("tail_blocks_dropped", 0) != depth
            or (depth == 2 and cache["source_sha256"] != EXPORTER_SHA)
            or sha256(cache_dir / "train_features.npy") != cache["features_sha256"]
            or run["schema"] != "sfora-inshop-siglip2-unseen-gallery-train-v1"
            or run["source_sha256"] != TRAINER_SHA
            or run["feature_receipt_sha256"] != sha256(cache_dir / "receipt.json")
            or run["features_sha256"] != cache["features_sha256"]
            or run["tail_blocks_dropped"] != depth
            or run["seed"] != 179026
            or run["updates"] != 100
            or run["arm"] != "freeze_emb"
            or run["fit_rows_sha256"] != digest_rows(fit)
            or run["held_rows_sha256"] != digest_rows(held)
            or run["preflight_sha256"] != PREFLIGHT_SHA
            or sha256(run_dir / "checkpoint.pt") != run["checkpoint_sha256"]
            or len(run["preclip_grad_norms"]) != 100
            or not all(math.isfinite(value) for value in run["preclip_grad_norms"])
        ):
            raise ValueError(f"In-Shop depth {depth} receipt differs")
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(run["quality"][key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or abs(float(values.mean()) - run["quality"][mean]) > 1e-8
            ):
                raise ValueError(f"In-Shop depth {depth} quality differs")
    shared = (
        "source_files_sha256",
        "model_file_sha256",
        "partition_sha256",
        "fit_rows_sha256",
        "held_rows_sha256",
        "schedule_sha256",
        "first_input_batch_sha256",
        "rank_coefficient",
        "batch_size",
        "vision_lr",
        "hardware",
    )
    if any(runs[0][key] != runs[1][key] for key in shared):
        raise ValueError("In-Shop depth paired inputs differ")
    c, t = runs
    wall_ratio = t["training_wall_seconds"] / c["training_wall_seconds"]
    r1_delta = t["quality"]["recall_at_1"] - c["quality"]["recall_at_1"]
    map_delta = t["quality"]["map_at_r"] - c["quality"]["map_at_r"]
    advance = wall_ratio <= 0.95 and r1_delta >= -0.005 and map_delta >= -0.005
    report = {
        "schema": "sfora-inshop-depth22-100-update-train-only-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "cache_receipt_sha256": [
            sha256(path / "receipt.json") for path in (args.full_cache, args.depth22_cache)
        ],
        "run_receipt_sha256": [
            sha256(path / "receipt.json") for path in (args.control, args.treatment)
        ],
        "held_rows": len(held),
        "wall_ratio": wall_ratio,
        "packed_r1_delta": r1_delta,
        "packed_map_at_r_delta": map_delta,
        "advance_batch1_latency_screen": advance,
        "arms": {
            name: {
                "packed_r1": run["quality"]["recall_at_1"],
                "packed_map_at_r": run["quality"]["map_at_r"],
                "training_wall_seconds": run["training_wall_seconds"],
                "training_peak_cuda_allocated_bytes": run["training_peak_cuda_allocated_bytes"],
                "cache_export_seconds": cache["encode_wall_seconds"],
            }
            for name, cache, run in zip(("full24", "depth22"), caches, runs, strict=True)
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
