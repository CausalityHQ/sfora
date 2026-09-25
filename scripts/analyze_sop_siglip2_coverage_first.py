#!/usr/bin/env python3
"""Source-bound, TRAIN-only readout for the frozen coverage-first pair."""

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

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
OLD_BANK_SHA256 = "20dcf88633c302502d5e7f9cd0a432c9b4186fc1939da420203e52464648ad3d"
OLD_FLOAT_SHA256 = "4086dee4178e1d6a8ca66599ab8fcdd99121ffd914dc5816a9f83e13763a1ada"
OLD_GATE_SHA256 = "70a682d158c914d00505df8d9fd2e53dab5abe3f067a43d472a5fd5d411ad6ce"
CANARY_FLOAT_SHA256 = "0e1ae4aaf4460c7ce879ca39a8dda182e21e73a4fb8ab5f9ea9726b4ce97c06d"
CANARY_BANK_SHA256 = "67e29c0f21779a5afd6f47af35f4f10b887f786bd827363791899631f447b95b"
TRAINER_SHA256 = "400f6ef2d992e449ff7eabf53c2982b88db0889df0586bbf94875a1b04a6e118"
SAMPLER_SHA256 = "bb97a0e0e97c0452ba48e10caf003b95c19d42ec9ce84204190be4b727d966a3"
SCHEDULE_SHA256 = "fe5453718569a6ee7c5dd5a308bd3da5d22eca647f9c513a76b41614a62b9ed4"
FIRST_INPUT_SHA256 = "3ccc597c3d37d9907c5674ad1dc8321d98f60fd3e4f6e5b8ee800d5975cabc08"
NATIVE_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
COMMON = (
    "schema",
    "split",
    "seed",
    "updates",
    "batch_size",
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
    "train_vision_dtype",
    "coverage_first_schedule",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--old-bank", required=True, type=Path)
    parser.add_argument("--old-float", required=True, type=Path)
    parser.add_argument("--old-gate", required=True, type=Path)
    parser.add_argument("--canary-float", required=True, type=Path)
    parser.add_argument("--canary-bank", required=True, type=Path)
    parser.add_argument("--new-fixed-float", required=True, type=Path)
    parser.add_argument("--new-matched-float", required=True, type=Path)
    parser.add_argument("--new-bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.old_bank) != OLD_BANK_SHA256
        or sha256(args.old_float) != OLD_FLOAT_SHA256
        or sha256(args.old_gate) != OLD_GATE_SHA256
        or sha256(args.canary_float) != CANARY_FLOAT_SHA256
        or sha256(args.canary_bank) != CANARY_BANK_SHA256
        or sha256(args.source_root / "scripts/train_sop_siglip2_compact.py") != TRAINER_SHA256
        or sha256(args.source_root / "src/sfora/unicom_rank_finish.py") != SAMPLER_SHA256
    ):
        raise ValueError("coverage-first source authority differs")
    old = json.loads(args.old_bank.read_text())
    old_float = json.loads(args.old_float.read_text())
    old_gate = json.loads(args.old_gate.read_text())
    canaries = {
        "float": json.loads(args.canary_float.read_text()),
        "bank": json.loads(args.canary_bank.read_text()),
    }
    ratios = {}
    for name, row in canaries.items():
        observed = row.get("rank_to_arcface_head_gradient_ratio")
        if (
            row.get("source_sha256") != TRAINER_SHA256
            or row.get("updates") != 1
            or row.get("quality") is not None
            or row.get("rank_coefficient") != 8.0
            or row.get("first_input_batch_sha256", [None])[0] != FIRST_INPUT_SHA256
            or not isinstance(observed, list)
            or len(observed) != 1
            or not math.isfinite(observed[0])
            or observed[0] <= 0
        ):
            raise ValueError("coverage-first gradient canary differs")
        ratios[name] = observed[0]
    if abs(8 * ratios["bank"] / ratios["float"] - 58.64) > 0.01:
        raise ValueError("coverage-first float calibration differs")
    if (
        old_gate.get("schema") != "sfora-sop-siglip2-bf16-member-bank-multiseed-v1"
        or old_gate.get("seeds") != [179023, 179024, 179025]
        or old_gate.get("arms", {}).get("179023", {}).get("bank", {}).get("receipt_sha256")
        != OLD_BANK_SHA256
    ):
        raise ValueError("coverage-first old three-seed gate differs")
    paths = {
        "fixed_float": args.new_fixed_float,
        "matched_float": args.new_matched_float,
        "bank": args.new_bank,
    }
    arms = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for name, row in arms.items():
        receipt_path = paths[name]
        checkpoint_path = receipt_path.with_name("checkpoint.pt")
        sources = row.get("source_files_sha256")
        quality = row.get("quality")
        if (
            row.get("schema") != "sfora-sop-siglip2-compact-full-backbone-v1"
            or row.get("claim_eligible") is not False
            or row.get("seed") != 179023
            or row.get("updates") != 1_000
            or row.get("batch_size") != 64
            or row.get("train_vision_dtype") != "bf16"
            or row.get("coverage_first_schedule") is not True
            or row.get("schedule_sha256") != SCHEDULE_SHA256
            or row.get("source_sha256") != TRAINER_SHA256
            or row.get("source_archive_sha256") != ARCHIVE_SHA256
            or row.get("native_library_sha256") != NATIVE_SHA256
            or row.get("arm") != ("float_rank_member_bank" if name == "bank" else "float_rank")
            or row.get("rank_coefficient")
            != {"fixed_float": 21.93, "matched_float": 58.64, "bank": 8.0}[name]
            or not isinstance(sources, dict)
            or sources.get("src/sfora/unicom_rank_finish.py") != SAMPLER_SHA256
            or any(
                sha256(args.source_root / relative) != digest
                for relative, digest in sources.items()
            )
            or sha256(checkpoint_path) != row.get("checkpoint_sha256")
            or not isinstance(quality, dict)
            or quality.get("native_top10_exact") is not True
            or quality.get("gallery_wire_bytes_per_row") != 130
            or len(row.get("step_seconds", ())) != 1_000
            or not all(math.isfinite(value) and value > 0 for value in row["step_seconds"])
            or len(row.get("first_input_batch_sha256", ())) != 10
            or row["first_input_batch_sha256"][0] != FIRST_INPUT_SHA256
        ):
            raise ValueError(f"coverage-first {name} arm authority differs")
    bank = arms["bank"]
    if (
        any(bank.get(key) != row.get(key) for key in COMMON for row in arms.values())
        or bank.get("member_bank_preflight_sha256")
        != "54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c"
        or bank.get("member_bank_cost_sha256")
        != "8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9"
        or any(
            arms[name].get(key) is not None
            for name in ("fixed_float", "matched_float")
            for key in ("member_bank_preflight_sha256", "member_bank_cost_sha256")
        )
        or bank["query_image_ids_sha256"] != old["query_image_ids_sha256"]
        or bank["query_image_ids_sha256"] != old_float["query_image_ids_sha256"]
    ):
        raise ValueError("coverage-first paired authority differs")
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
        or ids.shape != labels.shape
        or held.shape != (5_851,)
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != bank["query_image_ids_sha256"]
    ):
        raise ValueError("coverage-first TRAIN holdout differs")
    intervals = {}
    for metric, per_query in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        values = {}
        for name, row in (
            ("old_bank", old),
            ("old_float", old_float),
            ("bank", bank),
            ("fixed_float", arms["fixed_float"]),
            ("matched_float", arms["matched_float"]),
        ):
            values[name] = np.asarray(row["quality"][per_query], dtype=np.float64)
            if (
                values[name].shape != held.shape
                or not np.isfinite(values[name]).all()
                or abs(values[name].mean() - row["quality"][metric]) > 1e-6
            ):
                raise ValueError("coverage-first per-query quality differs")
        intervals[metric] = {
            "bank_minus_old_bank": product_bootstrap(
                values["bank"] - values["old_bank"], labels[held]
            ),
            "fixed_float_minus_old_float": product_bootstrap(
                values["fixed_float"] - values["old_float"], labels[held]
            ),
            **{
                f"bank_minus_{name}": product_bootstrap(values["bank"] - values[name], labels[held])
                for name in ("fixed_float", "matched_float")
            },
        }
    old_banks = [old_gate["arms"][str(seed)]["bank"] for seed in old_gate["seeds"]]
    old_mean = {
        "recall_at_1": float(np.mean([row["recall_at_1"] for row in old_banks])),
        "map_at_r": float(np.mean([row["map_at_r"] for row in old_banks])),
        "training_wall_seconds": float(
            np.mean([row["training_wall_accounted_seconds"] for row in old_banks])
        ),
        "training_peak_cuda_allocated_bytes": max(
            row["training_peak_cuda_allocated_bytes"] for row in old_banks
        ),
    }
    bank_wall = bank["training_wall_including_member_bank_init_seconds"]
    gate = (
        bank["quality"]["recall_at_1"] - old_mean["recall_at_1"] >= 0.003
        and bank["quality"]["map_at_r"] >= old_mean["map_at_r"]
        and bank_wall <= 1.03 * old_mean["training_wall_seconds"]
        and bank["training_peak_cuda_allocated_bytes"]
        <= 1.05 * old_mean["training_peak_cuda_allocated_bytes"]
        and all(
            intervals["recall_at_1"][f"bank_minus_{name}"]["point"] >= 0.006
            and intervals["recall_at_1"][f"bank_minus_{name}"]["lower_95"] > 0
            for name in ("fixed_float", "matched_float")
        )
    )
    result = {
        "schema": "sfora-sop-siglip2-bf16-coverage-first-screen-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout; full TRAIN gallery",
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "old_bank_receipt_sha256": OLD_BANK_SHA256,
        "old_float_receipt_sha256": OLD_FLOAT_SHA256,
        "old_gate_sha256": OLD_GATE_SHA256,
        "gradient_canary_sha256": {
            "float": CANARY_FLOAT_SHA256,
            "bank": CANARY_BANK_SHA256,
        },
        "first_step_gradient_ratios_at_coefficient_8": ratios,
        "old_bank_three_seed_mean": old_mean,
        "new_receipt_sha256": {name: sha256(path) for name, path in paths.items()},
        "schedule_sha256": SCHEDULE_SHA256,
        "arms": {
            name: {
                "packed_r1": row["quality"]["recall_at_1"],
                "packed_map_at_r": row["quality"]["map_at_r"],
                "training_wall_seconds": row["training_wall_including_member_bank_init_seconds"]
                if name == "bank"
                else row["training_wall_seconds"],
                "training_peak_cuda_allocated_bytes": row["training_peak_cuda_allocated_bytes"],
            }
            for name, row in arms.items()
        },
        "paired_product_bootstrap": intervals,
        "replication_gate_pass": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {"replication_gate_pass": gate, "paired_product_bootstrap": intervals}, sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
