#!/usr/bin/env python3
"""Apply the frozen mAP@R-aligned first-16-block In-Shop TRAIN gate."""

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

FREEZE12_SHA = "fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89"
FREEZE16_SHA = "63e97cebb8cc0e354e87ed837f2739123e5cdb2078a38827848f5abef0c768cd"
PREFLIGHT_SHA = "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034"
CHANGED_SOURCES = {
    "train_inshop_siglip2_unseen_gallery.py",
    "train_sop_siglip2_compact.py",
    "deployed_code_rank.py",
}
TREATMENT_SOURCES = {
    "train_inshop_siglip2_unseen_gallery.py": (
        "f92f792c52165350c16e59943fe8f06749f7dd0e8b5e7b3006ded9f683dd772e"
    ),
    "train_sop_siglip2_compact.py": (
        "ebc0986f112eb8ba72943595ac6ef93f5328c0bed105c2990cb8c70d7b3b0495"
    ),
    "deployed_code_rank.py": "435bd73d9a3dde04c9daa819ec3694e228b4cadcc2523c35e4c877ae5d034870",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "preflight", "freeze12", "freeze16", "treatment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.freeze12 / "receipt.json") != FREEZE12_SHA
        or sha256(args.freeze16 / "receipt.json") != FREEZE16_SHA
    ):
        raise ValueError("In-Shop mAP@R gate authority differs")
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[row].label for row in held])
    if preflight["fit_sha256"] != digest_rows(fit) or preflight["held_sha256"] != digest_rows(held):
        raise ValueError("In-Shop mAP@R split differs")
    runs = {
        name: json.loads((path / "receipt.json").read_text())
        for name, path in (
            ("freeze12", args.freeze12),
            ("freeze16", args.freeze16),
            ("mapr", args.treatment),
        )
    }
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
    )
    control = runs["freeze12"]
    if any(any(run[key] != control[key] for key in common) for run in runs.values()):
        raise ValueError("In-Shop mAP@R paired inputs differ")

    def unchanged_sources(run: dict) -> dict[str, str]:
        return {
            Path(path).name: digest
            for path, digest in run["source_files_sha256"].items()
            if Path(path).name not in CHANGED_SOURCES
            and not Path(path).name.startswith("train_inshop_siglip2_unseen_gallery")
        }

    if any(unchanged_sources(run) != unchanged_sources(control) for run in runs.values()):
        raise ValueError("In-Shop mAP@R unmodified sources differ")
    treatment_sources = {
        Path(path).name: digest
        for path, digest in runs["mapr"]["source_files_sha256"].items()
        if Path(path).name in CHANGED_SOURCES
    }
    if (
        treatment_sources != TREATMENT_SOURCES
        or runs["mapr"]["source_sha256"]
        != TREATMENT_SOURCES["train_inshop_siglip2_unseen_gallery.py"]
    ):
        raise ValueError("In-Shop mAP@R treatment source differs")
    for name, run in runs.items():
        path = {"freeze12": args.freeze12, "freeze16": args.freeze16, "mapr": args.treatment}[name]
        expected_blocks = 12 if name == "freeze12" else 16
        if (
            run["arm"] != ("freeze_emb_mapr" if name == "mapr" else "freeze_emb")
            or run["frozen_encoder_blocks"] != list(range(expected_blocks))
            or not run["frozen_embeddings"]
            or run.get("freeze_first_blocks", 12 if name == "freeze12" else None) != expected_blocks
            or run["seed"] != 179024
            or run["updates"] != 1_000
            or run.get("half_fit_products", False if name == "freeze12" else None)
            or run.get("tail_blocks_dropped", 0 if name == "freeze12" else None) != 0
            or run["fit_rows_sha256"] != digest_rows(fit)
            or run["held_rows_sha256"] != digest_rows(held)
            or sha256(path / "checkpoint.pt") != run["checkpoint_sha256"]
            or len(run["preclip_grad_norms"]) != 1_000
            or not all(math.isfinite(x) for x in run["preclip_grad_norms"])
        ):
            raise ValueError(f"In-Shop mAP@R {name} run differs")
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for name, run in runs.items():
        vectors[name] = {}
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(run["quality"][key], dtype=np.float64)
            if (
                values.shape != (len(held),)
                or not np.isfinite(values).all()
                or abs(float(values.mean()) - run["quality"][mean]) > 1e-8
            ):
                raise ValueError(f"In-Shop mAP@R {name} quality differs")
            vectors[name][key] = values
    map_delta = product_bootstrap(
        vectors["mapr"]["per_query_ap"] - vectors["freeze12"]["per_query_ap"], labels
    )
    r1_delta = float(np.mean(vectors["mapr"]["per_query_r1"] - vectors["freeze12"]["per_query_r1"]))
    wall_ratio = (
        runs["mapr"]["training_wall_including_member_bank_init_seconds"]
        / control["training_wall_including_member_bank_init_seconds"]
    )
    advance = (
        map_delta["point"] >= 0
        and map_delta["lower_95"] >= -0.003
        and r1_delta >= -0.002
        and wall_ratio <= 0.95
    )
    report = {
        "schema": "sfora-inshop-mapr-freeze16-train-gate-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": {
            "freeze12": FREEZE12_SHA,
            "freeze16": FREEZE16_SHA,
            "mapr": sha256(args.treatment / "receipt.json"),
        },
        "held_rows": len(held),
        "mapr_minus_freeze12_map_at_r_product_bootstrap": map_delta,
        "mapr_minus_freeze12_r1": r1_delta,
        "mapr_to_freeze12_training_wall_ratio": wall_ratio,
        "advance_independent_seeds": advance,
        "arms": {
            name: {
                "packed_r1": run["quality"]["recall_at_1"],
                "packed_map_at_r": run["quality"]["map_at_r"],
                "training_wall_including_bank_init_seconds": run[
                    "training_wall_including_member_bank_init_seconds"
                ],
                "peak_cuda_allocated_bytes": run["training_peak_cuda_allocated_bytes"],
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
