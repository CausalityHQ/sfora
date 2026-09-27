#!/usr/bin/env python3
"""Score one frozen 1,000-update SOP warm-start pair on TRAIN held roles."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from compare_inshop_sop_warmstart_100 import packed_quality
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from score_inshop_crop_view_pair import (
    GALLERY_SHA,
    PARTITION_SHA,
    QUERY_SHA,
    bootstrap_lower,
    roles,
    sha256,
)

from sfora.unicom_inshop import parse_inshop_partition

BASE = Path("/home/riomus/runs")
DATA = Path("/home/riomus/datasets/In-shop Clothes Retrieval Benchmark")
SOP_SHA = "2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172"
HELD_SHA = "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--seed", type=int, choices=(179024, 179025, 179026), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = BASE / f"sfora-inshop-sop-warmstart-{args.seed}-1000-pair-v1.json"
    if output.exists() or sha256(DATA / "Eval/list_eval_partition.txt") != PARTITION_SHA:
        raise ValueError("In-Shop full-budget paired authority differs")
    paths = {
        name: BASE / f"sfora-inshop-sop-warmstart-{name}-{args.seed}-1000-v1"
        for name in ("control", "treatment")
    }
    receipts = {
        name: json.loads((path / "receipt.json").read_text()) for name, path in paths.items()
    }
    control, treatment = receipts.values()
    paired = (
        "source_sha256",
        "source_files_sha256",
        "fit_rows_sha256",
        "held_rows_sha256",
        "schedule_sha256",
        "executed_schedule_sha256",
        "first_input_batch_sha256",
        "preflight_sha256",
        "model_file_sha256",
    )
    if (
        any(control[key] != treatment[key] for key in paired)
        or any(
            r["arm"] != "freeze_emb" or r["seed"] != args.seed or r["updates"] != 1_000
            for r in receipts.values()
        )
        or control["vision_init_sha256"] is not None
        or treatment["vision_init_sha256"] != SOP_SHA
    ):
        raise ValueError("In-Shop full-budget paired training differs")
    train = tuple(row for row in parse_inshop_partition(DATA) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, tuple(train[row].image_path for row in held), DATA)
    if (
        len(train) != 25_882
        or len(held) != 12_599
        or digest_rows(held) != HELD_SHA
        or len(query) != 6_354
        or len(gallery) != 6_245
        or hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != QUERY_SHA
        or hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest() != GALLERY_SHA
    ):
        raise ValueError("In-Shop full-budget held roles differ")
    quality = {}
    for name, path in paths.items():
        receipt = receipts[name]
        values_path = path / "held_values.npy"
        values = np.load(values_path, allow_pickle=False)
        if (
            sha256(values_path) != receipt["held_values_sha256"]
            or values.shape != (12_599, 128)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or sha256(path / "checkpoint.pt") != receipt["checkpoint_sha256"]
            or len(receipt["preclip_grad_norms"]) != 1_000
            or not np.isfinite(receipt["preclip_grad_norms"]).all()
            or not np.isfinite([receipt["first_loss"], receipt["last_loss"]]).all()
        ):
            raise ValueError(f"{name} full-budget checkpoint or gradients differ")
        quality[name] = packed_quality(values, labels, query, gallery)
    delta = np.asarray(quality["treatment"]["per_query_r1"]) - np.asarray(
        quality["control"]["per_query_r1"]
    )
    products = np.asarray([labels[row] for row in query])
    gain = float(delta.mean())
    lower = bootstrap_lower(delta, products)
    upper = -bootstrap_lower(-delta, products)
    wall_ratio = treatment["training_wall_seconds"] / control["training_wall_seconds"]
    cuda_ratio = (
        treatment["training_peak_cuda_allocated_bytes"]
        / control["training_peak_cuda_allocated_bytes"]
    )
    report = {
        "schema": "sfora-inshop-sop-warmstart-full-pair-v1",
        "claim_eligible": False,
        "seed": args.seed,
        "split": "official In-Shop TRAIN held products; packed 6354 query / 6245 gallery",
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": {name: sha256(path / "receipt.json") for name, path in paths.items()},
        "asymmetric_packed": quality,
        "symmetric_packed": {name: receipt["quality"] for name, receipt in receipts.items()},
        "r1_gain_percentage_points": 100 * gain,
        "r1_paired_product_bootstrap_95_pp": [100 * lower, 100 * upper],
        "training_wall_seconds": {
            name: receipt["training_wall_seconds"] for name, receipt in receipts.items()
        },
        "training_peak_cuda_allocated_bytes": {
            name: receipt["training_peak_cuda_allocated_bytes"]
            for name, receipt in receipts.items()
        },
        "training_wall_ratio": wall_ratio,
        "training_cuda_ratio": cuda_ratio,
        "first_seed_gross_kill": gain <= -0.005
        or quality["treatment"]["map_at_r"] <= quality["control"]["map_at_r"] - 0.010,
        "score_wall_seconds": time.perf_counter() - started,
    }
    with output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "seed",
                    "r1_gain_percentage_points",
                    "r1_paired_product_bootstrap_95_pp",
                    "first_seed_gross_kill",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
