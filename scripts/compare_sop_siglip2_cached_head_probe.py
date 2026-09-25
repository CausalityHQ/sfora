#!/usr/bin/env python3
"""Paired, source-bound decision for the SOP cached-head feasibility probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_bf16_member_bank_multiseed import validate_metric_values
from compare_sop_siglip2_member_bank_arms import ARCHIVE_SHA256, DRAWS, product_bootstrap
from train_sop_siglip2_cached_head_probe import (
    ARMS,
    EXPECTED_CLASSIFIER_SHA256,
    EXPECTED_HEAD_SHA256,
    EXPECTED_PCA_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    FEATURE_SHA256,
    SEED,
    STEPS,
    sha256,
    source_manifest,
)

from sfora.representation_ceiling import deterministic_class_partition

BANK_BASELINE_SHA256 = "20dcf88633c302502d5e7f9cd0a432c9b4186fc1939da420203e52464648ad3d"
BANK_BASELINE_SECONDS = 1146.9084727950394
EXPORT_RECEIPT_SHA256 = "3d49e039c72c0677591835752133caf1cbfd0ea483c23a901773748b44d843fb"
CACHE_ENCODE_SECONDS = 508.51690101856366
COMMON = (
    "schema",
    "claim_eligible",
    "split",
    "seed",
    "steps",
    "source_archive_sha256",
    "source_cache_sha256",
    "export_receipt_sha256",
    "source_files_sha256",
    "native_library_sha256",
    "tileiras_sha256",
    "initial_head_sha256",
    "initial_classifier_sha256",
    "source_pca_sha256",
    "schedule_sha256",
    "first_input_batch_sha256",
    "query_image_ids_sha256",
    "cache_encode_seconds",
)


def validate_pair(arcface: dict, bank: dict, query_sha256: str) -> None:
    if (
        (arcface.get("arm"), bank.get("arm")) != ARMS
        or any(
            row.get("schema") != "sfora-sop-siglip2-cached-head-probe-v1" for row in (arcface, bank)
        )
        or any(row.get("seed") != SEED or row.get("steps") != STEPS for row in (arcface, bank))
        or any(row.get("claim_eligible") is not False for row in (arcface, bank))
        or any(row.get("source_archive_sha256") != ARCHIVE_SHA256 for row in (arcface, bank))
        or any(row.get("source_cache_sha256") != FEATURE_SHA256 for row in (arcface, bank))
        or any(row.get("export_receipt_sha256") != EXPORT_RECEIPT_SHA256 for row in (arcface, bank))
        or any(row.get("initial_head_sha256") != EXPECTED_HEAD_SHA256 for row in (arcface, bank))
        or any(
            row.get("initial_classifier_sha256") != EXPECTED_CLASSIFIER_SHA256
            for row in (arcface, bank)
        )
        or any(row.get("source_pca_sha256") != EXPECTED_PCA_SHA256 for row in (arcface, bank))
        or any(row.get("schedule_sha256") != EXPECTED_SCHEDULE_SHA256 for row in (arcface, bank))
        or any(row.get("query_image_ids_sha256") != query_sha256 for row in (arcface, bank))
        or any(row.get("cache_encode_seconds") != CACHE_ENCODE_SECONDS for row in (arcface, bank))
        or any(len(row.get("step_seconds", [])) != STEPS for row in (arcface, bank))
        or any(len(row.get("first_input_batch_sha256", [])) != 10 for row in (arcface, bank))
        or any(not math.isfinite(row.get("first_loss", math.nan)) for row in (arcface, bank))
        or any(not math.isfinite(row.get("last_loss", math.nan)) for row in (arcface, bank))
        or any(
            row.get("quality", {}).get("native_top10_exact") is not True
            or row.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
            for row in (arcface, bank)
        )
        or any(arcface.get(key) != bank.get(key) for key in COMMON)
    ):
        raise ValueError("SOP cached-head paired authority differs")


def survival_gate(delta_r1: dict, delta_map: dict, bank_r1: float, wall: float) -> bool:
    values = (delta_r1["point"], delta_r1["lower_95"], delta_map["point"], bank_r1, wall)
    return (
        all(math.isfinite(value) for value in values)
        and delta_r1["point"] >= 0.005
        and delta_r1["lower_95"] > 0
        and delta_map["point"] >= 0
        and bank_r1 >= 0.85
        and wall <= 0.5 * BANK_BASELINE_SECONDS
    )


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
        raise ValueError("SOP cached-head comparison source or output differs")
    baseline_path = args.run_base / "sfora-siglip2-bf16-member-bank-179023-bank-v1/receipt.json"
    if sha256(baseline_path) != BANK_BASELINE_SHA256:
        raise ValueError("SOP cached-head bank baseline receipt differs")
    baseline = json.loads(baseline_path.read_text())
    if baseline.get("training_wall_including_member_bank_init_seconds") != BANK_BASELINE_SECONDS:
        raise ValueError("SOP cached-head bank baseline wall differs")
    paths = {
        arm: args.run_base / f"sfora-siglip2-cached-head-{SEED}-{arm}-v1/receipt.json"
        for arm in ARMS
    }
    receipts = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    if receipts["arcface"].get("source_files_sha256") != source_manifest():
        raise ValueError("SOP cached-head source package differs")
    script_digest = sha256(Path(__file__).with_name("train_sop_siglip2_cached_head_probe.py"))
    if any(
        row.get("source_files_sha256", {}).get("scripts/train_sop_siglip2_cached_head_probe.py")
        != script_digest
        for row in receipts.values()
    ):
        raise ValueError("SOP cached-head executed probe source differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != ids.shape or labels.shape != (59_551,):
        raise ValueError("SOP cached-head TRAIN inventory differs")
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=179019
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if len(held) != 5_851 or len(np.unique(labels[held])) != 1_132:
        raise ValueError("SOP cached-head holdout differs")
    validate_pair(
        receipts["arcface"],
        receipts["live_head_bank"],
        hashlib.sha256(ids[held].tobytes()).hexdigest(),
    )
    metrics = {}
    for name, key, stated in (
        ("r1", "per_query_r1", "recall_at_1"),
        ("map_at_r", "per_query_ap", "map_at_r"),
    ):
        values = {}
        for arm in ARMS:
            quality = receipts[arm]["quality"]
            values[arm] = np.asarray(quality[key], dtype=np.float64)
            validate_metric_values(values[arm], quality[stated], len(held))
        metrics[name] = product_bootstrap(
            values["live_head_bank"] - values["arcface"], labels[held]
        )
    bank = receipts["live_head_bank"]
    wall = bank["accounted_cache_plus_head_train_seconds"]
    if (
        not math.isfinite(wall)
        or wall <= 0
        or abs(
            wall
            - (
                bank["cache_encode_seconds"]
                + bank["head_initialization_seconds"]
                + bank["training_wall_seconds"]
            )
        )
        > 1e-6
    ):
        raise ValueError("SOP cached-head accounted training wall differs")
    result = {
        "schema": "sfora-sop-siglip2-cached-head-comparison-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout; full TRAIN gallery",
        "source_sha256": sha256(Path(__file__)),
        "bootstrap_source_sha256": sha256(
            Path(__file__).with_name("compare_sop_siglip2_member_bank_arms.py")
        ),
        "source_archive_sha256": ARCHIVE_SHA256,
        "bootstrap_draws": DRAWS,
        "baseline_receipt_sha256": BANK_BASELINE_SHA256,
        "baseline_accounted_training_wall_seconds": BANK_BASELINE_SECONDS,
        "probe_receipt_sha256": {arm: sha256(path) for arm, path in paths.items()},
        "arms": {
            arm: {
                "r1": receipts[arm]["quality"]["recall_at_1"],
                "map_at_r": receipts[arm]["quality"]["map_at_r"],
                "cache_plus_head_training_seconds": receipts[arm][
                    "accounted_cache_plus_head_train_seconds"
                ],
                "head_training_peak_torch_allocated_bytes": receipts[arm][
                    "training_peak_cuda_allocated_bytes"
                ],
            }
            for arm in ARMS
        },
        "live_head_minus_arcface": metrics,
        "live_head_over_full_bank_training_wall_ratio": wall / BANK_BASELINE_SECONDS,
        "survival_gate_pass": survival_gate(
            metrics["r1"], metrics["map_at_r"], bank["quality"]["recall_at_1"], wall
        ),
        "interval_scope": (
            "single-seed product bootstrap on an already-used TRAIN holdout; no seed uncertainty"
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
                "survival_gate_pass": result["survival_gate_pass"],
                "live_head_minus_arcface": metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
