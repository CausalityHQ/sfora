#!/usr/bin/env python3
"""Paired SOP TRAIN holdout decision for same-source member-bank training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
TRAINER_SHA256 = "32cce40aa7133bfc602c41b8d5fdb6e76609c71336b907a87c964cbb58348ade"
PREFLIGHT_SHA256 = "54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c"
COST_SHA256 = "8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9"
SEED = 179019
DRAWS = 5_000


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def product_bootstrap(delta: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    classes, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=delta, minlength=len(classes))
    random = np.random.default_rng(SEED)
    draws = np.empty(DRAWS, dtype=np.float64)
    for start in range(0, DRAWS, 512):
        stop = min(start + 512, DRAWS)
        picked = random.integers(0, len(classes), size=(stop - start, len(classes)))
        draws[start:stop] = totals[picked].sum(axis=1) / counts[picked].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"point": float(delta.mean()), "lower_95": float(lower), "upper_95": float(upper)}


def validate_pair(control: dict, treatment: dict) -> None:
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
        control.get("arm") != "arcface"
        or treatment.get("arm") != "float_rank_member_bank"
        or control.get("source_sha256") != TRAINER_SHA256
        or treatment.get("source_sha256") != TRAINER_SHA256
        or control.get("updates") != 1_000
        or control.get("seed") != SEED
        or control.get("fit_images") != 53_700
        or control.get("holdout_queries") != 5_851
        or control.get("gallery_images") != 59_551
        or treatment.get("member_bank_preflight_sha256") != PREFLIGHT_SHA256
        or treatment.get("member_bank_cost_sha256") != COST_SHA256
        or control.get("member_bank_preflight_sha256") is not None
        or control.get("member_bank_cost_sha256") is not None
        or len(control.get("first_input_batch_sha256", [])) != 10
        or any(control.get(key) != treatment.get(key) for key in common)
        or any(
            receipt.get("quality", {}).get("native_top10_exact") is not True
            or receipt.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
            or receipt.get("rank_to_arcface_head_gradient_ratio") != []
            for receipt in (control, treatment)
        )
    ):
        raise ValueError("SOP member-bank matched arm authority differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--control-receipt", type=Path, required=True)
    parser.add_argument("--treatment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP member-bank comparison authority differs")
    control = json.loads(args.control_receipt.read_text())
    treatment = json.loads(args.treatment_receipt.read_text())
    validate_pair(control, treatment)
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != (59_551,) or ids.shape != labels.shape:
        raise ValueError("SOP member-bank TRAIN inventory differs")
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=SEED
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if (
        len(held) != 5_851
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != control["query_image_ids_sha256"]
    ):
        raise ValueError("SOP member-bank holdout inventory differs")
    intervals = {}
    for metric, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        difference = np.asarray(treatment["quality"][key], dtype=np.float64) - np.asarray(
            control["quality"][key], dtype=np.float64
        )
        if difference.shape != (len(held),):
            raise ValueError("SOP member-bank query outcomes differ")
        intervals[metric] = product_bootstrap(difference, labels[held])
        if (
            abs(
                intervals[metric]["point"]
                - (treatment["quality"][metric] - control["quality"][metric])
            )
            > 1e-6
        ):
            raise ValueError("SOP member-bank metric summary differs")
    wall_ratio = (
        treatment["training_wall_including_member_bank_init_seconds"]
        / control["training_wall_seconds"]
    )
    gate = bool(
        intervals["recall_at_1"]["point"] >= 0.01
        and intervals["recall_at_1"]["lower_95"] > 0
        and intervals["map_at_r"]["point"] >= -0.005
        and wall_ratio <= 1.15
    )
    result = {
        "schema": "sfora-sop-siglip2-member-bank-matched-decision-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout, full TRAIN gallery",
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "control_receipt_sha256": sha256(args.control_receipt),
        "treatment_receipt_sha256": sha256(args.treatment_receipt),
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "holdout_products": 1_132,
        "control": {
            "recall_at_1": control["quality"]["recall_at_1"],
            "map_at_r": control["quality"]["map_at_r"],
            "training_wall_seconds": control["training_wall_seconds"],
            "training_peak_cuda_allocated_bytes": control["training_peak_cuda_allocated_bytes"],
        },
        "treatment": {
            "recall_at_1": treatment["quality"]["recall_at_1"],
            "map_at_r": treatment["quality"]["map_at_r"],
            "training_wall_including_member_bank_init_seconds": treatment[
                "training_wall_including_member_bank_init_seconds"
            ],
            "training_peak_cuda_allocated_bytes": treatment["training_peak_cuda_allocated_bytes"],
        },
        "paired_product_bootstrap": intervals,
        "training_wall_ratio": wall_ratio,
        "screen_gate_pass": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"screen_gate_pass": gate, "wall_ratio": wall_ratio}), flush=True)


if __name__ == "__main__":
    main()
