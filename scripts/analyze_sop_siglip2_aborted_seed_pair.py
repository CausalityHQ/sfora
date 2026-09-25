#!/usr/bin/env python3
"""Preserve a matched SOP pair when its preregistered third arm fails."""

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
    TRAINER_SHA256,
    product_bootstrap,
)
from compare_sop_siglip2_member_bank_multiseed import COMMON_WITHIN_SEED

from sfora.representation_ceiling import deterministic_class_partition

FIT_SEED = 179019
FAILED_SEED = 179020
FAILURE_TEXT = "The total norm of order 2.0 for gradients from `parameters` is non-finite"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--float-receipt", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--arcface-output-dir", type=Path, required=True)
    parser.add_argument("--failure-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or not args.arcface_output_dir.is_dir()
        or (args.arcface_output_dir / "receipt.json").exists()
        or FAILURE_TEXT not in args.failure_journal.read_text()
        or "RUN seed=179020 arm=arcface" not in args.failure_journal.read_text()
        or "Failed with result 'exit-code'" not in args.failure_journal.read_text()
    ):
        raise ValueError("aborted seed authority differs")
    floating = json.loads(args.float_receipt.read_text())
    bank = json.loads(args.bank_receipt.read_text())
    if (
        (floating.get("arm"), bank.get("arm")) != ("float_rank", "float_rank_member_bank")
        or any(row.get("seed") != FAILED_SEED for row in (floating, bank))
        or any(row.get("source_sha256") != TRAINER_SHA256 for row in (floating, bank))
        or any(row.get("source_archive_sha256") != ARCHIVE_SHA256 for row in (floating, bank))
        or any(row.get("updates") != 1000 for row in (floating, bank))
        or any(row.get("holdout_queries") != 5851 for row in (floating, bank))
        or any(bank.get(key) != floating.get(key) for key in COMMON_WITHIN_SEED)
        or floating.get("member_bank_preflight_sha256") is not None
        or floating.get("member_bank_cost_sha256") is not None
        or bank.get("member_bank_preflight_sha256") != PREFLIGHT_SHA256
        or bank.get("member_bank_cost_sha256") != COST_SHA256
        or any(
            row.get("quality", {}).get("native_top10_exact") is not True
            or row.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
            for row in (floating, bank)
        )
    ):
        raise ValueError("aborted seed pair differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != (59_551,) or ids.shape != labels.shape:
        raise ValueError("aborted seed SOP inventory differs")
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=FIT_SEED
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if (
        len(held) != 5_851
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != floating["query_image_ids_sha256"]
    ):
        raise ValueError("aborted seed holdout differs")
    intervals = {}
    for metric, per_query in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        difference = np.asarray(bank["quality"][per_query], dtype=np.float64) - np.asarray(
            floating["quality"][per_query], dtype=np.float64
        )
        if difference.shape != (len(held),):
            raise ValueError("aborted seed query outcomes differ")
        intervals[metric] = product_bootstrap(difference, labels[held])
        if (
            abs(
                intervals[metric]["point"] - (bank["quality"][metric] - floating["quality"][metric])
            )
            > 1e-6
        ):
            raise ValueError("aborted seed metric summary differs")
    result = {
        "schema": "sfora-sop-siglip2-aborted-seed-pair-v1",
        "claim_eligible": False,
        "replication_gate_pass": False,
        "failure": "preregistered ArcFace control produced a non-finite gradient before step 550",
        "split": "SOP official TRAIN product-disjoint holdout, full TRAIN gallery",
        "seed": FAILED_SEED,
        "holdout_queries": len(held),
        "holdout_products": 1_132,
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "float_receipt_sha256": sha256(args.float_receipt),
        "bank_receipt_sha256": sha256(args.bank_receipt),
        "failure_journal_sha256": sha256(args.failure_journal),
        "float": {
            "recall_at_1": floating["quality"]["recall_at_1"],
            "map_at_r": floating["quality"]["map_at_r"],
            "training_wall_seconds": floating["training_wall_seconds"],
        },
        "bank": {
            "recall_at_1": bank["quality"]["recall_at_1"],
            "map_at_r": bank["quality"]["map_at_r"],
            "training_wall_seconds_including_init": bank[
                "training_wall_including_member_bank_init_seconds"
            ],
        },
        "paired_product_bootstrap": intervals,
        "training_wall_ratio": bank["training_wall_including_member_bank_init_seconds"]
        / floating["training_wall_seconds"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
