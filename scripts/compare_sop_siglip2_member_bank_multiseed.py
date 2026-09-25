#!/usr/bin/env python3
"""Source-bound three-seed SOP TRAIN member-bank continuation decision."""

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
    DRAWS,
    PREFLIGHT_SHA256,
    TRAINER_SHA256,
    product_bootstrap,
)

from sfora.representation_ceiling import deterministic_class_partition

SEEDS = (179019, 179020, 179021, 179022)
REPLICATION_SEEDS = SEEDS[1:]
ARMS = ("arcface", "float_rank", "bank")
EXPLORATORY_RECEIPT_SHA256 = {
    "arcface": "d88167bfcbf8152ee912c8382061afaf248e45fe52ae24cf5f1a5da739477fc3",
    "float_rank": "99d6492b687fe35be850ccf16b22fe778b0fc82c620f5c3ea0277d8b42139e97",
    "bank": "2e73ee0e6252c91d54c815d52581abd0d303de09a56776a2fcd5b5e7908c981c",
}
COMMON_WITHIN_SEED = (
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
COMMON_ACROSS_SEEDS = tuple(
    key
    for key in COMMON_WITHIN_SEED
    if key not in ("seed", "schedule_sha256", "first_input_batch_sha256")
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_three_arms(seed: int, arcface: dict, floating: dict, bank: dict) -> None:
    arms = (arcface, floating, bank)
    if (
        seed not in SEEDS
        or tuple(row.get("arm") for row in arms)
        != ("arcface", "float_rank", "float_rank_member_bank")
        or any(row.get("seed") != seed or row.get("updates") != 1_000 for row in arms)
        or any(row.get("source_sha256") != TRAINER_SHA256 for row in arms)
        or any(
            row.get("fit_images") != 53_700
            or row.get("holdout_queries") != 5_851
            or row.get("gallery_images") != 59_551
            for row in arms
        )
        or any(row.get("source_archive_sha256") != ARCHIVE_SHA256 for row in arms)
        or any(
            row.get("quality", {}).get("native_top10_exact") is not True
            or row.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
            for row in arms
        )
        or len(arcface.get("first_input_batch_sha256", [])) != 10
        or any(row.get(key) != arcface.get(key) for row in arms[1:] for key in COMMON_WITHIN_SEED)
        or any(
            row.get("member_bank_preflight_sha256") is not None
            or row.get("member_bank_cost_sha256") is not None
            for row in arms[:2]
        )
        or bank.get("member_bank_preflight_sha256") != PREFLIGHT_SHA256
        or bank.get("member_bank_cost_sha256") != COST_SHA256
    ):
        raise ValueError("SOP three-arm matched authority differs")


def continuation_gate(comparisons: dict) -> bool:
    for name in ("float_rank", "arcface"):
        row = comparisons[name]
        if (
            len(row["replication_seedwise_r1"]) != 3
            or len(row["replication_seedwise_wall_ratio"]) != 3
            or not all(value > 0 for value in row["replication_seedwise_r1"])
            or row["replication_pooled_r1"]["lower_95"] <= 0
            or row["replication_pooled_map_at_r"]["point"] < 0
            or not all(value <= 1.15 for value in row["replication_seedwise_wall_ratio"])
        ):
            return False
    return comparisons["float_rank"]["replication_pooled_r1"]["point"] >= 0.005


def receipt_paths(run_base: Path) -> dict[int, dict[str, Path]]:
    first = {
        "arcface": "control",
        "float_rank": "float",
        "bank": "treatment",
    }
    return {
        seed: {
            arm: run_base
            / (
                f"sfora-siglip2-member-bank-{first[arm]}-1000-v1/receipt.json"
                if seed == SEEDS[0]
                else f"sfora-siglip2-member-bank-multiseed-{seed}-{arm}-v1/receipt.json"
            )
            for arm in ARMS
        }
        for seed in SEEDS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP multi-seed source or output differs")
    paths = receipt_paths(args.run_base)
    if any(sha256(paths[SEEDS[0]][arm]) != EXPLORATORY_RECEIPT_SHA256[arm] for arm in ARMS):
        raise ValueError("SOP exploratory receipt pin differs")
    receipts = {
        seed: {arm: json.loads(path.read_text()) for arm, path in paths[seed].items()}
        for seed in SEEDS
    }
    for seed in SEEDS:
        validate_three_arms(seed, *(receipts[seed][arm] for arm in ARMS))
    reference = receipts[SEEDS[0]]["arcface"]
    if any(
        row.get(key) != reference.get(key)
        for seed in SEEDS[1:]
        for row in receipts[seed].values()
        for key in COMMON_ACROSS_SEEDS
    ):
        raise ValueError("SOP cross-seed fixed authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != (59_551,) or ids.shape != labels.shape:
        raise ValueError("SOP TRAIN inventory differs")
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=179019
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if (
        len(held) != 5_851
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != reference["query_image_ids_sha256"]
    ):
        raise ValueError("SOP holdout inventory differs")
    result_rows = {}
    for seed in SEEDS:
        result_rows[str(seed)] = {}
        for arm in ARMS:
            row = receipts[seed][arm]
            quality = row["quality"]
            for metric, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
                values = np.asarray(quality[key], dtype=np.float64)
                if values.shape != (len(held),) or abs(values.mean() - quality[metric]) > 1e-6:
                    raise ValueError("SOP per-query metric authority differs")
            wall = (
                row["training_wall_including_member_bank_init_seconds"]
                if arm == "bank"
                else row["training_wall_seconds"]
            )
            result_rows[str(seed)][arm] = {
                "receipt_sha256": sha256(paths[seed][arm]),
                "recall_at_1": quality["recall_at_1"],
                "map_at_r": quality["map_at_r"],
                "training_wall_accounted_seconds": wall,
                "member_bank_init_seconds": row["member_bank_init_seconds"],
                "training_images_per_second": 64_000 / wall,
                "training_peak_cuda_allocated_bytes": row["training_peak_cuda_allocated_bytes"],
                "export_seconds": row["export_seconds"],
                "score_seconds": row["score_seconds"],
                "gallery_wire_bytes_per_row": quality["gallery_wire_bytes_per_row"],
            }
    comparisons = {}
    for control in ("float_rank", "arcface"):
        by_metric = {}
        for metric, key in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
            deltas = [
                np.asarray(receipts[seed]["bank"]["quality"][key], dtype=np.float64)
                - np.asarray(receipts[seed][control]["quality"][key], dtype=np.float64)
                for seed in SEEDS
            ]
            by_metric[metric] = {
                "seedwise": [product_bootstrap(delta, labels[held]) for delta in deltas],
                "replication_pooled": product_bootstrap(np.mean(deltas[1:], axis=0), labels[held]),
            }
        wall_ratios = [
            result_rows[str(seed)]["bank"]["training_wall_accounted_seconds"]
            / result_rows[str(seed)][control]["training_wall_accounted_seconds"]
            for seed in SEEDS
        ]
        comparisons[control] = {
            "seedwise_r1": [row["point"] for row in by_metric["r1"]["seedwise"]],
            "seedwise_wall_ratio": wall_ratios,
            "replication_seedwise_r1": [row["point"] for row in by_metric["r1"]["seedwise"][1:]],
            "replication_seedwise_wall_ratio": wall_ratios[1:],
            "seedwise_product_bootstrap": {
                key: by_metric[key]["seedwise"] for key in ("r1", "map_at_r")
            },
            "replication_pooled_r1": by_metric["r1"]["replication_pooled"],
            "replication_pooled_map_at_r": by_metric["map_at_r"]["replication_pooled"],
        }
    result = {
        "schema": "sfora-sop-siglip2-member-bank-multiseed-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout; full TRAIN gallery",
        "seeds": SEEDS,
        "replication_seeds": REPLICATION_SEEDS,
        "exploratory_seed": SEEDS[0],
        "bootstrap_draws": DRAWS,
        "holdout_queries": len(held),
        "holdout_products": 1_132,
        "source_sha256": sha256(Path(__file__)),
        "bootstrap_source_sha256": sha256(
            Path(__file__).with_name("compare_sop_siglip2_member_bank_arms.py")
        ),
        "source_archive_sha256": ARCHIVE_SHA256,
        "arms": result_rows,
        "comparisons": comparisons,
        "continuation_gate_pass": continuation_gate(comparisons),
        "interval_scope": (
            "product bootstrap conditional on three post-selection trained seeds; "
            "not seed-population confidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "continuation_gate_pass": result["continuation_gate_pass"],
                "comparisons": comparisons,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
