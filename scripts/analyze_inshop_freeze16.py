#!/usr/bin/env python3
"""Apply the frozen first-16-block In-Shop TRAIN gate."""

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

BASELINE_SHA = "fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89"
PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"
TRAINER_SHA = "0da378d00167e7e6d3a76bdf8ee4dd90fe6f82d05c142c82145767de0da78a6b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "preflight", "baseline", "treatment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.baseline / "receipt.json") != BASELINE_SHA
    ):
        raise ValueError("In-Shop freeze16 authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    held_labels = np.asarray([train[row].label for row in held])
    if preflight["fit_sha256"] != digest_rows(fit) or preflight["held_sha256"] != digest_rows(held):
        raise ValueError("In-Shop freeze16 split differs")
    control = json.loads((args.baseline / "receipt.json").read_text())
    treatment = json.loads((args.treatment / "receipt.json").read_text())
    common = (
        "schema",
        "arm",
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
    )

    def helpers(run: dict) -> dict[str, str]:
        return {
            Path(path).name: digest
            for path, digest in run["source_files_sha256"].items()
            if not Path(path).name.startswith("train_inshop_siglip2_unseen_gallery")
        }

    if (
        any(control[key] != treatment[key] for key in common)
        or helpers(control) != helpers(treatment)
        or treatment["source_sha256"] != TRAINER_SHA
        or treatment["frozen_encoder_blocks"] != list(range(16))
        or control["frozen_encoder_blocks"] != list(range(12))
        or not treatment["frozen_embeddings"]
        or treatment["freeze_first_blocks"] != 16
        or treatment["tail_blocks_dropped"] != 0
        or treatment["half_fit_products"]
        or treatment["seed"] != 179024
        or treatment["updates"] != 1_000
        or treatment["fit_rows_sha256"] != digest_rows(fit)
        or treatment["held_rows_sha256"] != digest_rows(held)
        or sha256(args.treatment / "checkpoint.pt") != treatment["checkpoint_sha256"]
        or len(treatment["preclip_grad_norms"]) != 1_000
        or not all(math.isfinite(x) for x in treatment["preclip_grad_norms"])
    ):
        raise ValueError("In-Shop freeze16 paired run differs")
    vectors = []
    for run in (control, treatment):
        metrics = []
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(run["quality"][key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or abs(float(values.mean()) - run["quality"][mean]) > 1e-8
            ):
                raise ValueError("In-Shop freeze16 quality vector differs")
            metrics.append(values)
        vectors.append(metrics)
    r1_delta = float(np.mean(vectors[1][0] - vectors[0][0]))
    map_bootstrap = product_bootstrap(vectors[1][1] - vectors[0][1], held_labels)
    wall_ratio = treatment["training_wall_seconds"] / control["training_wall_seconds"]
    advance = (
        map_bootstrap["point"] >= 0
        and map_bootstrap["lower_95"] >= -0.003
        and r1_delta >= -0.002
        and wall_ratio <= 0.95
    )
    report = {
        "schema": "sfora-inshop-freeze16-train-only-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "baseline_receipt_sha256": BASELINE_SHA,
        "treatment_receipt_sha256": sha256(args.treatment / "receipt.json"),
        "treatment_checkpoint_sha256": treatment["checkpoint_sha256"],
        "held_rows": len(held),
        "packed_r1_delta": r1_delta,
        "packed_map_at_r_product_bootstrap": map_bootstrap,
        "training_wall_ratio": wall_ratio,
        "advance_independent_seeds": advance,
        "arms": {
            name: {
                "packed_r1": run["quality"]["recall_at_1"],
                "packed_map_at_r": run["quality"]["map_at_r"],
                "training_wall_seconds": run["training_wall_seconds"],
                "training_peak_cuda_allocated_bytes": run["training_peak_cuda_allocated_bytes"],
            }
            for name, run in (("freeze12", control), ("freeze16", treatment))
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
