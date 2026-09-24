#!/usr/bin/env python3
"""Compare matched SOP TRAIN compact arms using heldout-product resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
TRAINER_SHA256 = "95de5b80fda488db18715308588a3d56e53ecf6e4df99313877426274346b1c8"
ARMS = ("arcface", "float_rank", "packed_rank")
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
    draws: np.ndarray = np.empty(DRAWS, dtype=np.float64)
    for start in range(0, DRAWS, 512):
        stop = min(start + 512, DRAWS)
        selected = random.integers(0, len(classes), size=(stop - start, len(classes)))
        draws[start:stop] = totals[selected].sum(axis=1) / counts[selected].sum(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "point": float(delta.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def compare(
    candidate: dict[str, Any], reference: dict[str, Any], labels: np.ndarray
) -> dict[str, Any]:
    answer: dict[str, Any] = {}
    for metric, per_query in (
        ("recall_at_1", "per_query_r1"),
        ("map_at_r", "per_query_ap"),
    ):
        values = np.asarray(candidate["quality"][per_query], dtype=np.float64)
        baseline = np.asarray(reference["quality"][per_query], dtype=np.float64)
        if values.shape != labels.shape or baseline.shape != labels.shape:
            raise ValueError("SOP matched arm paired query geometry differs")
        interval = product_bootstrap(values - baseline, labels)
        if (
            abs(interval["point"] - (candidate["quality"][metric] - reference["quality"][metric]))
            > 1e-6
        ):
            raise ValueError("SOP matched arm paired summary differs")
        answer[metric] = interval
    answer["training_wall_ratio"] = (
        candidate["training_wall_seconds"] / reference["training_wall_seconds"]
    )
    answer["screen_gate_pass"] = bool(
        answer["recall_at_1"]["point"] >= 0.01
        and answer["recall_at_1"]["lower_95"] > 0
        and answer["map_at_r"]["point"] >= 0
        and answer["training_wall_ratio"] <= 1.15
    )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(f"--{arm}-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP matched arm source authority differs")
    paths = {arm: getattr(args, f"{arm}_receipt") for arm in ARMS}
    receipts = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    control = receipts["arcface"]
    common_fields = (
        "seed",
        "updates",
        "batch_size",
        "workers",
        "fit_images",
        "fit_products",
        "holdout_queries",
        "gallery_images",
        "model_file_sha256",
        "hardware",
        "source_files_sha256",
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
        "rank_coefficient",
        "grad_scaler_initial_scale",
        "optimizer",
        "precision",
        "augmentation",
        "query_image_ids_sha256",
    )
    for arm in ARMS:
        receipt = receipts[arm]
        if (
            receipt.get("schema") != "sfora-sop-siglip2-compact-full-backbone-v1"
            or receipt.get("arm") != arm
            or receipt.get("claim_eligible") is not False
            or receipt.get("updates") != 1_000
            or receipt.get("batch_size") != 64
            or receipt.get("workers") != 2
            or receipt.get("seed") != SEED
            or receipt.get("grad_scaler_initial_scale") != 128.0
            or receipt.get("source_files_sha256", {}).get("scripts/train_sop_siglip2_compact.py")
            != TRAINER_SHA256
            or len(receipt.get("first_input_batch_sha256", [])) != 10
            or len(receipt.get("rank_to_arcface_head_gradient_ratio", [])) != 0
            or any(receipt.get(field) != control.get(field) for field in common_fields)
            or receipt.get("quality") is None
            or receipt["quality"].get("native_top10_exact") is not True
            or receipt["quality"].get("gallery_wire_bytes_per_row") != 130
        ):
            raise ValueError(f"SOP matched {arm} arm authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != (59_551,) or ids.shape != labels.shape:
        raise ValueError("SOP matched TRAIN row inventory differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    held = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if (
        len(held) != 5_851
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != control["query_image_ids_sha256"]
    ):
        raise ValueError("SOP matched heldout inventory differs")
    comparisons = {
        "packed_rank_minus_arcface": compare(receipts["packed_rank"], control, labels[held]),
        "float_rank_minus_arcface": compare(receipts["float_rank"], control, labels[held]),
        "packed_rank_minus_float_rank": compare(
            receipts["packed_rank"], receipts["float_rank"], labels[held]
        ),
    }
    result = {
        "schema": "sfora-sop-siglip2-matched-1000-update-decision-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout, full TRAIN gallery",
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "seed": SEED,
        "bootstrap_draws": DRAWS,
        "heldout_products": 1_132,
        "arms": {
            arm: {
                "receipt_sha256": sha256(path),
                "recall_at_1": receipts[arm]["quality"]["recall_at_1"],
                "map_at_r": receipts[arm]["quality"]["map_at_r"],
                "training_wall_seconds": receipts[arm]["training_wall_seconds"],
                "export_seconds": receipts[arm]["export_seconds"],
                "score_seconds": receipts[arm]["score_seconds"],
                "training_peak_cuda_allocated_bytes": receipts[arm][
                    "training_peak_cuda_allocated_bytes"
                ],
            }
            for arm, path in paths.items()
        },
        "paired_product_bootstrap": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps({key: value["screen_gate_pass"] for key, value in comparisons.items()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
