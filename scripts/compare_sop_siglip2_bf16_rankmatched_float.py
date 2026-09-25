#!/usr/bin/env python3
"""Source-bound SOP TRAIN comparison of BF16 bank and rank-matched float arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_bf16_member_bank_multiseed import (
    COMMON_ACROSS_SEEDS,
    SEEDS,
    validate_metric_values,
)
from compare_sop_siglip2_member_bank_arms import ARCHIVE_SHA256, DRAWS, product_bootstrap
from compare_sop_siglip2_member_bank_multiseed import COMMON_WITHIN_SEED

from sfora.representation_ceiling import deterministic_class_partition

GATE_SHA256 = "70a682d158c914d00505df8d9fd2e53dab5abe3f067a43d472a5fd5d411ad6ce"
DIAGNOSTIC_SHA256 = "54685249fdc2dd199a28d59469f9b107f9e68ac4a01f34ac5abd7d5a68e7a390"
OLD_TRAINER_SHA256 = "ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23"
NEW_TRAINER_SHA256 = "328cdfbd4d35ae8037d1130c5fc889d60fe0ec25cf185c0c9ecc714a475120e4"
TRAINER_RELATIVE = "scripts/train_sop_siglip2_compact.py"
PAIRED_KEYS = tuple(
    key
    for key in COMMON_WITHIN_SEED
    if key not in ("source_sha256", "source_files_sha256", "rank_coefficient")
)
ACROSS_SEED_KEYS = tuple(
    key
    for key in COMMON_ACROSS_SEEDS
    if key not in ("source_sha256", "source_files_sha256", "rank_coefficient")
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_authority(gate: dict, diagnostic: dict) -> None:
    if (
        gate.get("schema") != "sfora-sop-siglip2-bf16-member-bank-multiseed-v1"
        or gate.get("continuation_gate_pass") is not True
        or gate.get("seeds") != list(SEEDS)
        or gate.get("source_archive_sha256") != ARCHIVE_SHA256
        or diagnostic.get("schema") != "sfora-sop-siglip2-bf16-rank-contribution-v1"
        or diagnostic.get("gate_sha256") != GATE_SHA256
        or diagnostic.get("rank_coefficient_float_control") != 21.93
    ):
        raise ValueError("SOP rank-matched source authority differs")


def validate_pair(seed: int, bank: dict, floating: dict, gate: dict) -> None:
    old_sources = bank.get("source_files_sha256", {})
    new_sources = floating.get("source_files_sha256", {})
    if (
        seed not in SEEDS
        or bank.get("arm") != "float_rank_member_bank"
        or floating.get("arm") != "float_rank"
        or bank.get("source_sha256") != OLD_TRAINER_SHA256
        or floating.get("source_sha256") != NEW_TRAINER_SHA256
        or bank.get("rank_coefficient") != 8.0
        or floating.get("rank_coefficient") != 21.93
        or old_sources.get(TRAINER_RELATIVE) != OLD_TRAINER_SHA256
        or new_sources.get(TRAINER_RELATIVE) != NEW_TRAINER_SHA256
        or set(old_sources) != set(new_sources)
        or any(
            old_sources[key] != new_sources[key] for key in old_sources if key != TRAINER_RELATIVE
        )
        or bank.get("seed") != seed
        or floating.get("seed") != seed
        or any(bank.get(key) != floating.get(key) for key in PAIRED_KEYS)
        or any(row.get("train_vision_dtype") != "bf16" for row in (bank, floating))
        or any(
            row.get("updates") != 1_000 or len(row.get("step_seconds", [])) != 1_000
            for row in (bank, floating)
        )
        or any(
            row.get("quality", {}).get("native_top10_exact") is not True
            or row.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
            for row in (bank, floating)
        )
        or len(bank.get("first_input_batch_sha256", [])) != 10
        or gate.get("arms", {}).get(str(seed), {}).get("bank", {}).get("receipt_sha256") is None
    ):
        raise ValueError("SOP rank-matched paired authority differs")


def validate_canary(canary: dict, original: dict) -> None:
    if (
        canary.get("source_sha256") != NEW_TRAINER_SHA256
        or canary.get("arm") != "float_rank"
        or canary.get("seed") != SEEDS[0]
        or canary.get("rank_coefficient") != 21.93
        or canary.get("train_vision_dtype") != "bf16"
        or canary.get("updates") != 1
        or canary.get("quality") is not None
        or len(canary.get("step_seconds", [])) != 1
        or not math.isfinite(canary.get("first_loss", math.nan))
        or len(canary.get("rank_to_arcface_head_gradient_ratio", [])) != 1
        or not math.isfinite(canary["rank_to_arcface_head_gradient_ratio"][0])
        or any(
            canary.get(key) != original.get(key)
            for key in ("initial_head_sha256", "initial_classifier_sha256", "model_file_sha256")
        )
        or canary.get("first_input_batch_sha256", [None])[0]
        != original.get("first_input_batch_sha256", [None])[0]
    ):
        raise ValueError("SOP rank-matched canary authority differs")


def rankmatched_gate(
    seedwise_r1: list[float], pooled_r1: dict, pooled_map: dict, wall: list[float]
) -> bool:
    values = (*seedwise_r1, pooled_r1["point"], pooled_r1["lower_95"], pooled_map["point"], *wall)
    return (
        len(seedwise_r1) == len(wall) == 3
        and all(math.isfinite(value) for value in values)
        and all(value > 0 for value in seedwise_r1)
        and pooled_r1["point"] >= 0.005
        and pooled_r1["lower_95"] > 0
        and pooled_map["point"] >= 0
        and all(value <= 1.15 for value in wall)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.gate) != GATE_SHA256
        or sha256(args.diagnostic) != DIAGNOSTIC_SHA256
    ):
        raise ValueError("SOP rank-matched input authority or output differs")
    gate = json.loads(args.gate.read_text())
    diagnostic = json.loads(args.diagnostic.read_text())
    validate_authority(gate, diagnostic)
    canary_path = args.run_base / "sfora-siglip2-bf16-rankmatched-179023-canary-v1/receipt.json"
    original_path = (
        args.run_base / "sfora-siglip2-bf16-member-bank-179023-float_rank-v1/receipt.json"
    )
    validate_canary(json.loads(canary_path.read_text()), json.loads(original_path.read_text()))
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != ids.shape or labels.shape != (59_551,):
        raise ValueError("SOP rank-matched TRAIN inventory differs")
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=179019
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if len(held) != 5_851 or len(np.unique(labels[held])) != 1_132:
        raise ValueError("SOP rank-matched holdout inventory differs")
    rows: dict[str, dict] = {}
    deltas: dict[str, list[np.ndarray]] = {"r1": [], "map_at_r": []}
    walls: list[float] = []
    bank_reference: dict | None = None
    float_reference: dict | None = None
    for seed in SEEDS:
        bank_path = args.run_base / f"sfora-siglip2-bf16-member-bank-{seed}-bank-v1/receipt.json"
        float_path = args.run_base / f"sfora-siglip2-bf16-rankmatched-{seed}-float-v1/receipt.json"
        if sha256(bank_path) != gate["arms"][str(seed)]["bank"]["receipt_sha256"]:
            raise ValueError("SOP rank-matched bank receipt differs from passed gate")
        bank = json.loads(bank_path.read_text())
        floating = json.loads(float_path.read_text())
        validate_pair(seed, bank, floating, gate)
        if (
            bank.get("query_image_ids_sha256") != hashlib.sha256(ids[held].tobytes()).hexdigest()
            or (
                bank_reference is not None
                and any(bank.get(key) != bank_reference.get(key) for key in ACROSS_SEED_KEYS)
            )
            or (
                float_reference is not None
                and any(floating.get(key) != float_reference.get(key) for key in ACROSS_SEED_KEYS)
            )
        ):
            raise ValueError("SOP rank-matched cross-seed or holdout authority differs")
        bank_reference = bank
        float_reference = floating
        bank_wall = bank["training_wall_including_member_bank_init_seconds"]
        float_wall = floating["training_wall_seconds"]
        if not all(math.isfinite(value) and value > 0 for value in (bank_wall, float_wall)):
            raise ValueError("SOP rank-matched training wall differs")
        walls.append(bank_wall / float_wall)
        for metric, key in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
            stated = "recall_at_1" if metric == "r1" else metric
            bank_values = np.asarray(bank["quality"][key], dtype=np.float64)
            float_values = np.asarray(floating["quality"][key], dtype=np.float64)
            validate_metric_values(bank_values, bank["quality"][stated], len(held))
            validate_metric_values(float_values, floating["quality"][stated], len(held))
            deltas[metric].append(bank_values - float_values)
        rows[str(seed)] = {
            "bank_receipt_sha256": sha256(bank_path),
            "matched_float_receipt_sha256": sha256(float_path),
            "bank_r1": bank["quality"]["recall_at_1"],
            "matched_float_r1": floating["quality"]["recall_at_1"],
            "bank_map_at_r": bank["quality"]["map_at_r"],
            "matched_float_map_at_r": floating["quality"]["map_at_r"],
            "bank_accounted_training_wall_seconds": bank_wall,
            "matched_float_training_wall_seconds": float_wall,
            "bank_images_per_second": 64_000 / bank_wall,
            "matched_float_images_per_second": 64_000 / float_wall,
            "bank_peak_cuda_allocated_bytes": bank["training_peak_cuda_allocated_bytes"],
            "matched_float_peak_cuda_allocated_bytes": floating[
                "training_peak_cuda_allocated_bytes"
            ],
            "bank_export_seconds": bank["export_seconds"],
            "matched_float_export_seconds": floating["export_seconds"],
            "bank_score_seconds": bank["score_seconds"],
            "matched_float_score_seconds": floating["score_seconds"],
            "gallery_wire_bytes_per_row": 130,
            "bank_over_matched_float_wall_ratio": walls[-1],
        }
    metrics = {
        name: {
            "seedwise": [product_bootstrap(delta, labels[held]) for delta in values],
            "mean": product_bootstrap(np.mean(values, axis=0), labels[held]),
        }
        for name, values in deltas.items()
    }
    seedwise_r1 = [row["point"] for row in metrics["r1"]["seedwise"]]
    result = {
        "schema": "sfora-sop-siglip2-bf16-rankmatched-float-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout; full TRAIN gallery",
        "seeds": SEEDS,
        "holdout_queries": len(held),
        "holdout_products": 1_132,
        "bootstrap_draws": DRAWS,
        "source_sha256": sha256(Path(__file__)),
        "bootstrap_source_sha256": sha256(
            Path(__file__).with_name("compare_sop_siglip2_member_bank_arms.py")
        ),
        "metric_source_sha256": sha256(
            Path(__file__).with_name("compare_sop_siglip2_bf16_member_bank_multiseed.py")
        ),
        "source_archive_sha256": ARCHIVE_SHA256,
        "gate_sha256": GATE_SHA256,
        "diagnostic_sha256": DIAGNOSTIC_SHA256,
        "old_trainer_sha256": OLD_TRAINER_SHA256,
        "new_trainer_sha256": NEW_TRAINER_SHA256,
        "rank_coefficient_bank": 8.0,
        "rank_coefficient_matched_float": 21.93,
        "canary_receipt_sha256": sha256(canary_path),
        "arms": rows,
        "bank_minus_matched_float": metrics,
        "bank_over_matched_float_wall_ratio": walls,
        "bank_specific_screen_pass": rankmatched_gate(
            seedwise_r1, metrics["r1"]["mean"], metrics["map_at_r"]["mean"], walls
        ),
        "interval_scope": (
            "product bootstrap conditional on three trained seeds and selected TRAIN holdout"
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
                "bank_specific_screen_pass": result["bank_specific_screen_pass"],
                "bank_minus_matched_float": metrics,
                "bank_over_matched_float_wall_ratio": walls,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
