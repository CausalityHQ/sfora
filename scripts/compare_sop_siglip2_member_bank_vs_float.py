#!/usr/bin/env python3
"""Descriptive same-source bank-versus-in-batch SmoothAP SOP TRAIN comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import (
    ARCHIVE_SHA256,
    COST_SHA256,
    PREFLIGHT_SHA256,
    SEED,
    TRAINER_SHA256,
    product_bootstrap,
)

from sfora.representation_ceiling import deterministic_class_partition


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_pair(float_arm: dict, bank_arm: dict) -> None:
    common = (
        "schema",
        "split",
        "seed",
        "updates",
        "batch_size",
        "workers",
        "images_per_identity",
        "fit_images",
        "fit_products",
        "holdout_queries",
        "gallery_images",
        "source_sha256",
        "source_files_sha256",
        "model_file_sha256",
        "source_archive_sha256",
        "source_features_sha256",
        "source_export_receipt_sha256",
        "ordered_rows_sha256",
        "native_library_sha256",
        "tileiras_sha256",
        "source_pca_sha256",
        "initial_head_sha256",
        "initial_classifier_sha256",
        "schedule_sha256",
        "first_input_batch_sha256",
        "query_image_ids_sha256",
        "rank_coefficient",
        "grad_scaler_initial_scale",
        "optimizer",
        "precision",
        "augmentation",
        "hardware",
    )
    if (
        float_arm.get("arm") != "float_rank"
        or bank_arm.get("arm") != "float_rank_member_bank"
        or float_arm.get("source_sha256") != TRAINER_SHA256
        or bank_arm.get("source_sha256") != TRAINER_SHA256
        or float_arm.get("updates") != 1_000
        or float_arm.get("seed") != SEED
        or float_arm.get("holdout_queries") != 5_851
        or float_arm.get("member_bank_preflight_sha256") is not None
        or float_arm.get("member_bank_cost_sha256") is not None
        or bank_arm.get("member_bank_preflight_sha256") != PREFLIGHT_SHA256
        or bank_arm.get("member_bank_cost_sha256") != COST_SHA256
        or len(float_arm.get("first_input_batch_sha256", [])) != 10
        or any(float_arm.get(key) != bank_arm.get(key) for key in common)
        or any(
            row.get("quality", {}).get("native_top10_exact") is not True
            or row.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
            for row in (float_arm, bank_arm)
        )
    ):
        raise ValueError("SOP bank-versus-in-batch authority differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--float-receipt", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP bank-versus-in-batch source differs")
    float_arm = json.loads(args.float_receipt.read_text())
    bank_arm = json.loads(args.bank_receipt.read_text())
    validate_pair(float_arm, bank_arm)
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != (59_551,) or ids.shape != labels.shape:
        raise ValueError("SOP bank-versus-in-batch TRAIN inventory differs")
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=SEED
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if (
        len(held) != 5_851
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != float_arm["query_image_ids_sha256"]
    ):
        raise ValueError("SOP bank-versus-in-batch holdout differs")
    intervals = {}
    for metric, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        difference = np.asarray(bank_arm["quality"][key], dtype=np.float64) - np.asarray(
            float_arm["quality"][key], dtype=np.float64
        )
        if difference.shape != (len(held),):
            raise ValueError("SOP bank-versus-in-batch query outcomes differ")
        intervals[metric] = product_bootstrap(difference, labels[held])
        if (
            abs(
                intervals[metric]["point"]
                - (bank_arm["quality"][metric] - float_arm["quality"][metric])
            )
            > 1e-6
        ):
            raise ValueError("SOP bank-versus-in-batch metric summary differs")
    result = {
        "schema": "sfora-sop-siglip2-bank-versus-in-batch-v1",
        "claim_eligible": False,
        "confirmatory_gate": "none preregistered for this secondary attribution comparison",
        "split": "SOP official TRAIN product-disjoint holdout, full TRAIN gallery",
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "float_receipt_sha256": sha256(args.float_receipt),
        "bank_receipt_sha256": sha256(args.bank_receipt),
        "seed": SEED,
        "holdout_products": 1_132,
        "float": {
            "recall_at_1": float_arm["quality"]["recall_at_1"],
            "map_at_r": float_arm["quality"]["map_at_r"],
            "training_wall_seconds": float_arm["training_wall_seconds"],
        },
        "bank": {
            "recall_at_1": bank_arm["quality"]["recall_at_1"],
            "map_at_r": bank_arm["quality"]["map_at_r"],
            "training_wall_seconds_including_init": bank_arm[
                "training_wall_including_member_bank_init_seconds"
            ],
        },
        "paired_product_bootstrap": intervals,
        "training_wall_ratio": (
            bank_arm["training_wall_including_member_bank_init_seconds"]
            / float_arm["training_wall_seconds"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(intervals, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
