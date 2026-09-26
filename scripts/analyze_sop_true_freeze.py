#!/usr/bin/env python3
"""Apply the frozen SOP TRAIN-only true lower-stack freeze gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap

from sfora.representation_ceiling import deterministic_class_partition

TRAINER_SHA = "c5b8786c352ce6c8bedce9a5963ef43e2c18db61974e3c141c698227a23f1b3c"
ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
PREFLIGHT_SHA = "54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c"
COST_SHA = "8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=179024, choices=(179024, 179026, 179027))
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.source_root / "scripts/train_sop_siglip2_compact.py") != TRAINER_SHA
        or sha256(args.source_archive) != ARCHIVE_SHA
    ):
        raise ValueError("SOP true-freeze source authority differs")
    receipts = []
    for run, frozen in ((args.control, False), (args.treatment, True)):
        path = run / "receipt.json"
        r = json.loads(path.read_text())
        quality = r.get("quality")
        source_files = r.get("source_files_sha256")
        if (
            r.get("schema") != "sfora-sop-siglip2-compact-full-backbone-v1"
            or r.get("claim_eligible") is not False
            or r.get("split") != "SOP official TRAIN product-disjoint fit/holdout; no TEST rows"
            or r.get("arm") != "float_rank_member_bank"
            or r.get("member_bank_preflight_sha256") != PREFLIGHT_SHA
            or r.get("member_bank_cost_sha256") != COST_SHA
            or r.get("seed") != args.seed
            or r.get("updates") != 1000
            or r.get("batch_size") != 64
            or r.get("train_vision_dtype") != "bf16"
            or r.get("coverage_first_schedule") is not True
            or r.get("rank_coefficient") != 8.0
            or r.get("freeze_lower_stack") is not frozen
            or r.get("frozen_encoder_blocks") != (list(range(12)) if frozen else [])
            or r.get("fit_images") != 53_700
            or r.get("fit_products") != 10_186
            or r.get("holdout_queries") != 5_851
            or r.get("gallery_images") != 59_551
            or r.get("source_sha256") != TRAINER_SHA
            or r.get("source_archive_sha256") != ARCHIVE_SHA
            or r.get("native_library_sha256") != NATIVE_SHA
            or not isinstance(source_files, dict)
            or any(
                sha256(args.source_root / relative) != digest
                for relative, digest in source_files.items()
            )
            or sha256(run / "checkpoint.pt") != r.get("checkpoint_sha256")
            or sha256(run / "train_embeddings.npy") != r.get("train_embeddings_sha256")
            or not isinstance(quality, dict)
            or quality.get("gallery_wire_bytes_per_row") != 130
            or quality.get("native_top10_exact") is not True
            or quality.get("native_per_query_r1_equal") is not True
            or len(r.get("step_seconds", ())) != 1000
            or not all(math.isfinite(value) and value > 0 for value in r["step_seconds"])
        ):
            raise ValueError(f"SOP true-freeze frozen={frozen} receipt differs")
        for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
            values = np.asarray(quality[key], dtype=np.float64)
            if (
                values.shape != (5_851,)
                or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))
                or abs(float(values.mean()) - quality[mean]) > 1e-6
            ):
                raise ValueError(f"SOP true-freeze frozen={frozen} quality vector differs")
        receipts.append((r, sha256(path)))
    control, treatment = (row[0] for row in receipts)
    common = (
        "source_files_sha256",
        "model_file_sha256",
        "source_features_sha256",
        "source_export_receipt_sha256",
        "ordered_rows_sha256",
        "tileiras_sha256",
        "source_pca_sha256",
        "initial_head_sha256",
        "initial_classifier_sha256",
        "schedule_sha256",
        "first_input_batch_sha256",
        "query_image_ids_sha256",
        "hardware",
    )
    if any(control[key] != treatment[key] for key in common):
        raise ValueError("SOP true-freeze paired authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=179019
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if (
        labels.shape != (59_551,)
        or held.shape != (5_851,)
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != control["query_image_ids_sha256"]
    ):
        raise ValueError("SOP true-freeze held inventory differs")
    intervals = {
        metric: product_bootstrap(
            np.asarray(treatment["quality"][key], dtype=np.float64)
            - np.asarray(control["quality"][key], dtype=np.float64),
            labels[held],
        )
        for metric, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap"))
    }
    wall = [r["training_wall_including_member_bank_init_seconds"] for r in (control, treatment)]
    peak = [r["training_peak_cuda_allocated_bytes"] for r in (control, treatment)]
    if any(not math.isfinite(value) or value <= 0 for value in wall + peak):
        raise ValueError("SOP true-freeze resource receipt differs")
    quality_pass = (
        treatment["quality"]["recall_at_1"] >= control["quality"]["recall_at_1"]
        and intervals["recall_at_1"]["lower_95"] > -0.005
        and treatment["quality"]["map_at_r"] >= control["quality"]["map_at_r"]
    )
    cost_pass = wall[1] <= 0.85 * wall[0] and peak[1] <= 0.80 * peak[0]
    report = {
        "schema": "sfora-sop-true-freeze-paired-train-only-v1",
        "claim_eligible": False,
        "seed": args.seed,
        "held_queries": len(held),
        "paired_product_bootstrap": intervals,
        "quality_pass": quality_pass,
        "cost_pass": cost_pass,
        "advance_fresh_seeds": quality_pass and cost_pass,
        "arms": {
            name: {
                "receipt_sha256": receipts[index][1],
                "packed_r1": r["quality"]["recall_at_1"],
                "packed_map_at_r": r["quality"]["map_at_r"],
                "training_wall_seconds": wall[index],
                "peak_cuda_allocated_bytes": peak[index],
                "export_seconds": r["export_seconds"],
                "score_seconds": r["score_seconds"],
            }
            for index, (name, r) in enumerate(
                zip(("control", "freeze"), (control, treatment), strict=True)
            )
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "advance_fresh_seeds": report["advance_fresh_seeds"],
                "quality_pass": quality_pass,
                "cost_pass": cost_pass,
                "r1_delta": intervals["recall_at_1"],
            }
        )
    )


if __name__ == "__main__":
    main()
