#!/usr/bin/env python3
"""Apply the frozen In-Shop half-fit product scaling screen."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from train_inshop_siglip2_unseen_gallery import half_fit_products

from sfora.unicom_inshop import parse_inshop_partition

BASELINE_SHA = "fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89"
PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"
TRAINER_SHA = "aea6718f4b03e0bbc326f4f9d6547ea20f9da778f336fdb3cae69fbf1f756e28"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "preflight", "baseline", "half", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.baseline / "receipt.json") != BASELINE_SHA
    ):
        raise ValueError("In-Shop fit-product scaling authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in train)
    full_fit, held = split(labels)
    half_fit = half_fit_products(labels, full_fit)
    held_labels = np.asarray([labels[row] for row in held])
    baseline, half = (
        json.loads((path / "receipt.json").read_text()) for path in (args.baseline, args.half)
    )
    baseline_helpers = {
        Path(path).name: digest
        for path, digest in baseline["source_files_sha256"].items()
        if Path(path).name != "train_inshop_siglip2_unseen_gallery.py"
    }
    half_helpers = {
        Path(path).name: digest
        for path, digest in half["source_files_sha256"].items()
        if Path(path).name != "train_inshop_siglip2_unseen_gallery.py"
    }
    shared = (
        "preflight_sha256",
        "held_rows_sha256",
        "partition_sha256",
        "model_file_sha256",
        "feature_receipt_sha256",
        "features_sha256",
        "seed",
        "updates",
        "batch_size",
        "rank_coefficient",
        "vision_lr",
    )
    if (
        preflight["fit_sha256"] != digest_rows(full_fit)
        or preflight["held_sha256"] != digest_rows(held)
        or any(baseline[key] != half[key] for key in shared)
        or baseline_helpers != half_helpers
        or baseline["fit_rows_sha256"] != digest_rows(full_fit)
        or half["full_fit_rows_sha256"] != digest_rows(full_fit)
        or half["fit_rows_sha256"] != digest_rows(half_fit)
        or half["source_sha256"] != TRAINER_SHA
        or not half["half_fit_products"]
        or half["fit_rows"] != len(half_fit)
        or half["held_rows"] != len(held)
        or half["arm"] != baseline["arm"] != "freeze_emb"
        or half["seed"] != 179024
        or half["updates"] != 1_000
        or sha256(args.half / "checkpoint.pt") != half["checkpoint_sha256"]
        or len(half["preclip_grad_norms"]) != 1_000
        or not all(math.isfinite(value) for value in half["preclip_grad_norms"])
    ):
        raise ValueError("In-Shop fit-product scaling run differs")
    quality = []
    for receipt in (baseline, half):
        q = receipt["quality"]
        vectors = []
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(q[key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or abs(float(values.mean()) - q[mean]) > 1e-8
            ):
                raise ValueError("In-Shop fit-product scaling quality vector differs")
            vectors.append(values)
        quality.append(vectors)
    delta = quality[0][0] - quality[1][0]
    bootstrap = product_bootstrap(delta, held_labels)
    advance = bootstrap["point"] >= 0.004 and bootstrap["lower_95"] > 0
    report = {
        "schema": "sfora-inshop-fit-product-scaling-train-only-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "baseline_receipt_sha256": BASELINE_SHA,
        "half_receipt_sha256": sha256(args.half / "receipt.json"),
        "half_checkpoint_sha256": half["checkpoint_sha256"],
        "full_fit_products": len({labels[row] for row in full_fit}),
        "half_fit_products": len({labels[row] for row in half_fit}),
        "held_products": len(set(held_labels)),
        "r1_full_minus_half_product_bootstrap": bootstrap,
        "advance_full_train": advance,
        "arms": {
            name: {
                "fit_rows": receipt["fit_rows"],
                "packed_r1": receipt["quality"]["recall_at_1"],
                "packed_map_at_r": receipt["quality"]["map_at_r"],
                "training_wall_seconds": receipt["training_wall_seconds"],
                "training_peak_cuda_allocated_bytes": receipt["training_peak_cuda_allocated_bytes"],
            }
            for name, receipt in (("full_fit", baseline), ("half_fit", half))
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
