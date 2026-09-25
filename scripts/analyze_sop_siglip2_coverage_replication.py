#!/usr/bin/env python3
"""Source-bound three-seed SOP TRAIN coverage-first replication readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from analyze_sop_siglip2_coverage_first import (
    ARCHIVE_SHA256,
    COMMON,
    NATIVE_SHA256,
    OLD_GATE_SHA256,
    SAMPLER_SHA256,
    TRAINER_SHA256,
    sha256,
)
from compare_sop_siglip2_member_bank_arms import product_bootstrap

from sfora.representation_ceiling import deterministic_class_partition

SEEDS = (179023, 179024, 179025)
ARMS = ("fixed_float", "matched_float", "bank")
SCHEDULES = (
    "fe5453718569a6ee7c5dd5a308bd3da5d22eca647f9c513a76b41614a62b9ed4",
    "4bdc9606b58cac9c34ec16f9578f8111bdc06026926d9c9eb51f50e774779799",
    "1eef43537600973d97dcfbc37df1c147d527eb387d9e4509921b994a836b246c",
)
FIRST_SEED_HASHES = {
    "bank": "778634601b58584be21e21f56a95ed163f85b19accf7fb8c36d56b655df78f62",
    "fixed_float": "1e91a491aa0f1ab5e170e55430112410fa49cf3ec0f526e2334f496318461476",
    "matched_float": "9c7d4fd1b5b920cfc3d6ef3f8e13070a95077da18736cbfab103a70a9099609e",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--old-gate", required=True, type=Path)
    parser.add_argument("--run-base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.old_gate) != OLD_GATE_SHA256
        or sha256(args.source_root / "scripts/train_sop_siglip2_compact.py") != TRAINER_SHA256
        or sha256(args.source_root / "src/sfora/unicom_rank_finish.py") != SAMPLER_SHA256
    ):
        raise ValueError("replication source authority differs")
    old = json.loads(args.old_gate.read_text())
    if old.get("schema") != "sfora-sop-siglip2-bf16-member-bank-multiseed-v1" or old.get(
        "seeds"
    ) != list(SEEDS):
        raise ValueError("old bank gate differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    held = np.asarray(
        deterministic_class_partition(
            tuple(map(int, labels)), fit_fraction=0.9, seed=179019
        ).validation_row_indexes,
        dtype=np.int64,
    )
    if labels.shape != (59_551,) or ids.shape != labels.shape or held.shape != (5_851,):
        raise ValueError("TRAIN holdout inventory differs")
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    receipts: dict[int, dict[str, dict[str, Any]]] = {}
    for seed, schedule in zip(SEEDS, SCHEDULES, strict=True):
        receipts[seed] = {}
        rows[str(seed)] = {}
        for arm in ARMS:
            path = args.run_base / f"sfora-siglip2-bf16-coverage-{seed}-{arm}-v1/receipt.json"
            digest = sha256(path)
            if seed == SEEDS[0] and digest != FIRST_SEED_HASHES[arm]:
                raise ValueError(f"first-seed {arm} receipt differs")
            row = json.loads(path.read_text())
            quality = row.get("quality")
            sources = row.get("source_files_sha256")
            if (
                row.get("schema") != "sfora-sop-siglip2-compact-full-backbone-v1"
                or row.get("claim_eligible") is not False
                or row.get("seed") != seed
                or row.get("updates") != 1_000
                or row.get("batch_size") != 64
                or row.get("train_vision_dtype") != "bf16"
                or row.get("coverage_first_schedule") is not True
                or row.get("schedule_sha256") != schedule
                or row.get("source_sha256") != TRAINER_SHA256
                or row.get("source_archive_sha256") != ARCHIVE_SHA256
                or row.get("native_library_sha256") != NATIVE_SHA256
                or row.get("arm") != ("float_rank_member_bank" if arm == "bank" else "float_rank")
                or row.get("rank_coefficient")
                != {"bank": 8.0, "fixed_float": 21.93, "matched_float": 58.64}[arm]
                or not isinstance(sources, dict)
                or sources.get("src/sfora/unicom_rank_finish.py") != SAMPLER_SHA256
                or any(sha256(args.source_root / name) != value for name, value in sources.items())
                or sha256(path.with_name("checkpoint.pt")) != row.get("checkpoint_sha256")
                or not isinstance(quality, dict)
                or quality.get("native_top10_exact") is not True
                or quality.get("gallery_wire_bytes_per_row") != 130
                or len(row.get("step_seconds", ())) != 1_000
                or not all(math.isfinite(v) and v > 0 for v in row["step_seconds"])
                or len(row.get("first_input_batch_sha256", ())) != 10
            ):
                raise ValueError(f"seed {seed} arm {arm} authority differs")
            for metric, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
                values = np.asarray(quality[key], dtype=np.float64)
                if (
                    values.shape != held.shape
                    or not np.isfinite(values).all()
                    or abs(values.mean() - quality[metric]) > 1e-6
                ):
                    raise ValueError(f"seed {seed} arm {arm} {metric} differs")
            wall = (
                row["training_wall_including_member_bank_init_seconds"]
                if arm == "bank"
                else row["training_wall_seconds"]
            )
            rows[str(seed)][arm] = {
                "receipt_sha256": digest,
                "packed_r1": quality["recall_at_1"],
                "packed_map_at_r": quality["map_at_r"],
                "training_wall_seconds": wall,
                "training_images_per_second": 64_000 / wall,
                "peak_cuda_allocated_bytes": row["training_peak_cuda_allocated_bytes"],
                "export_seconds": row["export_seconds"],
                "score_seconds": row["score_seconds"],
            }
            receipts[seed][arm] = row
        bank = receipts[seed]["bank"]
        if (
            any(bank.get(key) != receipts[seed][arm].get(key) for key in COMMON for arm in ARMS)
            or bank.get("member_bank_preflight_sha256")
            != "54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c"
            or bank.get("member_bank_cost_sha256")
            != "8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9"
            or any(
                receipts[seed][arm].get(key) is not None
                for arm in ("fixed_float", "matched_float")
                for key in ("member_bank_preflight_sha256", "member_bank_cost_sha256")
            )
            or bank["query_image_ids_sha256"] != hashlib.sha256(ids[held].tobytes()).hexdigest()
        ):
            raise ValueError(f"seed {seed} paired authority differs")
    reference = receipts[SEEDS[0]]["bank"]
    fixed_keys = tuple(
        k
        for k in COMMON
        if k
        not in {
            "seed",
            "schedule_sha256",
            "first_input_batch_sha256",
            "initial_head_sha256",
            "initial_classifier_sha256",
        }
    )
    if any(
        row.get(key) != reference.get(key)
        for seed in SEEDS[1:]
        for row in receipts[seed].values()
        for key in (*fixed_keys, "hardware")
    ):
        raise ValueError("cross-seed authority differs")
    old_rows = [old["arms"][str(seed)]["bank"] for seed in SEEDS]
    old_mean = {
        "packed_r1": float(np.mean([r["recall_at_1"] for r in old_rows])),
        "packed_map_at_r": float(np.mean([r["map_at_r"] for r in old_rows])),
        "training_wall_seconds": float(
            np.mean([r["training_wall_accounted_seconds"] for r in old_rows])
        ),
        "peak_cuda_allocated_bytes": max(r["training_peak_cuda_allocated_bytes"] for r in old_rows),
    }
    means = {
        arm: {
            metric: float(np.mean([rows[str(seed)][arm][metric] for seed in SEEDS]))
            for metric in ("packed_r1", "packed_map_at_r", "training_wall_seconds")
        }
        for arm in ARMS
    }
    intervals: dict[str, dict[str, Any]] = {}
    for arm in ("fixed_float", "matched_float"):
        deltas = [
            np.asarray(receipts[seed]["bank"]["quality"]["per_query_r1"], dtype=np.float64)
            - np.asarray(receipts[seed][arm]["quality"]["per_query_r1"], dtype=np.float64)
            for seed in SEEDS
        ]
        intervals[arm] = {
            "seedwise_r1_delta": [float(x.mean()) for x in deltas],
            "mean_product_bootstrap": product_bootstrap(np.mean(deltas, axis=0), labels[held]),
        }
    gate = (
        means["bank"]["packed_r1"] - old_mean["packed_r1"] >= 0.003
        and means["bank"]["packed_map_at_r"] >= old_mean["packed_map_at_r"]
        and means["bank"]["training_wall_seconds"] <= 1.03 * old_mean["training_wall_seconds"]
        and max(rows[str(seed)]["bank"]["peak_cuda_allocated_bytes"] for seed in SEEDS)
        <= 1.05 * old_mean["peak_cuda_allocated_bytes"]
        and all(
            means["bank"]["packed_r1"] - means[arm]["packed_r1"] >= 0.006
            and means["bank"]["packed_map_at_r"] >= means[arm]["packed_map_at_r"]
            and all(delta > 0 for delta in intervals[arm]["seedwise_r1_delta"][1:])
            and intervals[arm]["mean_product_bootstrap"]["lower_95"] > 0
            for arm in ("fixed_float", "matched_float")
        )
    )
    result = {
        "schema": "sfora-sop-siglip2-bf16-coverage-replication-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout; full TRAIN gallery",
        "seeds": SEEDS,
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA256,
        "old_gate_sha256": OLD_GATE_SHA256,
        "schedule_sha256": dict(zip(SEEDS, SCHEDULES, strict=True)),
        "arms": rows,
        "means": means,
        "old_bank_three_seed_mean": old_mean,
        "comparisons": intervals,
        "replication_gate_pass": gate,
        "interval_scope": (
            "product bootstrap conditional on fixed trained seeds; not training-seed confidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {"replication_gate_pass": gate, "means": means, "comparisons": intervals},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
