#!/usr/bin/env python3
"""Apply the frozen two-seed In-Shop valid-anchor TRAIN gate."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHTS = {
    179026: "4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254",
    179027: "45d9fb0cd46844dccc9b25673634f047475f62991d42b3e61069d3823574d80b",
}
SOURCE = "76e20c328df6e632b387486e4761fb26994b74eff4c0a944c23400a40c4985cc"


def read_run(run: Path, *, seed: int, arm: str, preflight: dict, held: int) -> tuple[dict, str]:
    path = run / "receipt.json"
    receipt = json.loads(path.read_text())
    quality = receipt.get("quality")
    former = preflight["schedules"]["1000"]["rank_inactive_steps"]
    treatment = arm == "freeze_emb_rank"
    if (
        receipt.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
        or receipt.get("arm") != arm
        or receipt.get("seed") != seed
        or receipt.get("source_sha256") != SOURCE
        or receipt.get("updates") != 1000
        or receipt.get("batch_size") != 64
        or receipt.get("vision_lr") != 1e-5
        or receipt.get("preflight_sha256") != PREFLIGHTS[seed]
        or receipt.get("partition_sha256") != PARTITION_SHA
        or receipt.get("fit_rows_sha256") != preflight["fit_sha256"]
        or receipt.get("held_rows_sha256") != preflight["held_sha256"]
        or receipt.get("held_rows") != held
        or receipt.get("schedule_sha256") != preflight["schedules"]["1000"]["sha256"]
        or receipt.get("frozen_encoder_blocks") != list(range(12))
        or receipt.get("frozen_embeddings") is not True
        or receipt.get("rank_inactive_steps") != ([] if treatment else former)
        or receipt.get("formerly_rank_inactive_steps") != (former if treatment else [])
        or receipt.get("rank_active_updates") != (1000 if treatment else 1000 - len(former))
        or receipt.get("recovered_rank_updates") != (len(former) if treatment else 0)
        or len(receipt.get("preclip_grad_norms", ())) != 1000
        or not all(math.isfinite(x) for x in receipt["preclip_grad_norms"])
        or len(receipt.get("step_seconds", ())) != 1000
        or not all(math.isfinite(x) and x > 0 for x in receipt["step_seconds"])
        or sha256(run / "checkpoint.pt") != receipt.get("checkpoint_sha256")
        or not isinstance(quality, dict)
    ):
        raise ValueError(f"In-Shop seed {seed} {arm} receipt differs")
    for key, mean in (("per_query_r1", "recall_at_1"), ("per_query_ap", "map_at_r")):
        values = np.asarray(quality[key], dtype=np.float64)
        if (
            values.shape != (held,)
            or not np.isfinite(values).all()
            or np.any((values < 0) | (values > 1))
            or (key == "per_query_r1" and np.any((values != 0) & (values != 1)))
            or abs(float(values.mean()) - quality[mean]) > 1e-8
        ):
            raise ValueError(f"In-Shop seed {seed} {arm} per-query scores differ")
    wall = receipt.get("training_wall_including_member_bank_init_seconds")
    peak = receipt.get("training_peak_cuda_allocated_bytes")
    if (
        type(wall) not in (int, float)
        or not math.isfinite(wall)
        or wall <= 0
        or type(peak) is not int
        or peak <= 0
    ):
        raise ValueError(f"In-Shop seed {seed} {arm} cost differs")
    return receipt, sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop confirmation authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[index].label for index in held])
    rows = {}
    r1_deltas = []
    ap_deltas = []
    for seed in PREFLIGHTS:
        preflight_path = args.preflight_dir / f"preflight-{seed}.json"
        if sha256(preflight_path) != PREFLIGHTS[seed]:
            raise ValueError(f"In-Shop seed {seed} preflight differs")
        preflight = json.loads(preflight_path.read_text())
        if (
            preflight.get("seed") != seed
            or preflight.get("source_sha256")
            != "b6556df2d2011cb23c6b5ce25458c54679646b1ea7989fc47cb67e6a5142906e"
            or preflight.get("fit_sha256") != digest_rows(fit)
            or preflight.get("held_sha256") != digest_rows(held)
            or preflight.get("held_rows") != len(held)
        ):
            raise ValueError(f"In-Shop seed {seed} split differs")
        runs = {}
        for arm in ("freeze_emb", "freeze_emb_rank"):
            run = args.runs_dir / f"sfora-inshop-valid-anchor-confirm-{arm}-{seed}-v1"
            runs[arm] = read_run(run, seed=seed, arm=arm, preflight=preflight, held=len(held))
        baseline, treatment = (runs[arm][0] for arm in ("freeze_emb", "freeze_emb_rank"))
        common = (
            "first_input_batch_sha256",
            "feature_receipt_sha256",
            "features_sha256",
            "model_file_sha256",
            "pca_sha256",
            "fit_rows_sha256",
            "held_rows_sha256",
            "schedule_sha256",
            "hardware",
            "source_files_sha256",
        )
        if (
            any(baseline[key] != treatment[key] for key in common)
            or len(baseline["first_input_batch_sha256"]) != 10
        ):
            raise ValueError(f"In-Shop seed {seed} paired inputs differ")
        r1_delta = np.asarray(treatment["quality"]["per_query_r1"], dtype=np.float64) - np.asarray(
            baseline["quality"]["per_query_r1"], dtype=np.float64
        )
        ap_delta = np.asarray(treatment["quality"]["per_query_ap"], dtype=np.float64) - np.asarray(
            baseline["quality"]["per_query_ap"], dtype=np.float64
        )
        r1_deltas.append(r1_delta)
        ap_deltas.append(ap_delta)
        wall_ratio = (
            treatment["training_wall_including_member_bank_init_seconds"]
            / baseline["training_wall_including_member_bank_init_seconds"]
        )
        peak_ratio = (
            treatment["training_peak_cuda_allocated_bytes"]
            / baseline["training_peak_cuda_allocated_bytes"]
        )
        rows[str(seed)] = {
            "r1_product_bootstrap": product_bootstrap(r1_delta, labels),
            "map_at_r_delta": float(ap_delta.mean()),
            "training_wall_ratio": wall_ratio,
            "peak_cuda_ratio": peak_ratio,
            "per_seed_pass": float(r1_delta.mean()) > 0
            and float(ap_delta.mean()) >= -0.005
            and wall_ratio <= 1.1
            and peak_ratio <= 1.1,
            **{
                name: {
                    "receipt_sha256": digest,
                    "checkpoint_sha256": receipt["checkpoint_sha256"],
                    "packed_r1": receipt["quality"]["recall_at_1"],
                    "packed_map_at_r": receipt["quality"]["map_at_r"],
                    "training_wall_seconds": receipt[
                        "training_wall_including_member_bank_init_seconds"
                    ],
                    "peak_cuda_allocated_bytes": receipt["training_peak_cuda_allocated_bytes"],
                }
                for name, (receipt, digest) in (
                    ("baseline", runs["freeze_emb"]),
                    ("treatment", runs["freeze_emb_rank"]),
                )
            },
        }
    pooled_r1 = product_bootstrap(np.mean(r1_deltas, axis=0), labels)
    pooled_map = float(np.mean(ap_deltas))
    advance = (
        all(row["per_seed_pass"] for row in rows.values())
        and pooled_r1["lower_95"] > 0
        and pooled_map >= 0
    )
    report = {
        "schema": "sfora-inshop-valid-anchor-confirmation-train-only-v1",
        "claim_eligible": False,
        "held_queries_and_gallery": len(held),
        "seeds": rows,
        "pooled_r1_product_bootstrap": pooled_r1,
        "pooled_map_at_r_delta": pooled_map,
        "advance_to_official_gate": advance,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"advance_to_official_gate": advance, "pooled_r1": pooled_r1}))


if __name__ == "__main__":
    main()
