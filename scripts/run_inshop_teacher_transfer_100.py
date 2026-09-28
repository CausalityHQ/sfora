#!/usr/bin/env python3
"""One frozen serial cache/control/teacher/(conditional sham) TRAIN retrieval gate."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from compare_inshop_sop_warmstart_100 import packed_quality
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from score_inshop_crop_view_pair import GALLERY_SHA, QUERY_SHA, bootstrap_lower, roles, sha256

from sfora.unicom_inshop import parse_inshop_partition


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "base-snapshot",
        "large-snapshot",
        "teacher-checkpoint",
        "smoke-receipt",
        "preflight",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("teacher gate output already exists")
    args.output_dir.mkdir(parents=True)
    scripts = Path(__file__).parent

    def child(name, command):
        torch.cuda.empty_cache()
        with (args.output_dir / f"{name}.log").open("wb") as log:
            subprocess.run(
                [sys.executable, str(scripts / command[0]), *map(str, command[1:])],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=min(240, max(1, 900 - (time.perf_counter() - started))),
            )
        print(
            json.dumps({"terminal": name, "elapsed_seconds": time.perf_counter() - started}),
            flush=True,
        )

    cache = args.output_dir / "cache"
    child(
        "fit_cache",
        [
            "export_inshop_siglip2_train_features.py",
            "--dataset-root",
            args.dataset_root,
            "--model-snapshot",
            args.base_snapshot,
            "--output-dir",
            cache,
            "--teacher-transfer-smoke-receipt",
            args.smoke_receipt,
        ],
    )
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, tuple(train[row].image_path for row in held), args.dataset_root)
    if (
        len(query) != 6354
        or len(gallery) != 6245
        or digest_rows(held) != "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"
    ):
        raise ValueError("teacher gate held authority differs")
    import hashlib

    if any(
        hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != expected
        for rows, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
    ):
        raise ValueError("teacher gate role authority differs")
    qlabels = np.asarray([labels[row] for row in query])
    arms, quality, receipt_hashes = {}, {}, {}

    def arm(name):
        destination = args.output_dir / name
        child(
            name,
            [
                "train_inshop_siglip2_unseen_gallery.py",
                "--dataset-root",
                args.dataset_root,
                "--model-snapshot",
                args.base_snapshot,
                "--features-dir",
                cache,
                "--preflight",
                args.preflight,
                "--preflight-sha256",
                sha256(args.preflight),
                "--output-dir",
                destination,
                "--arm",
                "control",
                "--updates",
                "100",
                "--seed",
                "179024",
                "--teacher-transfer",
                name,
                "--teacher-transfer-smoke-receipt",
                args.smoke_receipt,
                "--teacher-model-snapshot",
                args.large_snapshot,
                "--teacher-checkpoint",
                args.teacher_checkpoint,
            ],
        )
        receipt_path = destination / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        values_path = destination / "held_values.npy"
        if (
            sha256(values_path) != receipt["held_values_sha256"]
            or sha256(destination / "checkpoint.pt") != receipt["checkpoint_sha256"]
        ):
            raise ValueError("teacher gate held/checkpoint hash differs")
        values = np.load(values_path, allow_pickle=False)
        if (
            values.shape != (12599, 128)
            or not np.isfinite(values).all()
            or len(receipt["all_step_losses"]) != 100
            or len(receipt["first_input_batch_sha256"]) != 100
            or not np.isfinite(receipt["all_step_losses"]).all()
        ):
            raise ValueError("teacher gate stable update inventory differs")
        if arms:
            control = arms["control"]
            fields = (
                "source_sha256",
                "source_files_sha256",
                "fit_rows_sha256",
                "held_rows_sha256",
                "executed_schedule_sha256",
                "first_input_batch_sha256",
                "pca_sha256",
                "features_sha256",
                "preflight_sha256",
                "model_file_sha256",
            )
            if any(receipt[field] != control[field] for field in fields):
                raise ValueError("teacher gate paired inputs differ")
        arms[name] = receipt
        receipt_hashes[name] = sha256(receipt_path)
        quality[name] = packed_quality(values, labels, query, gallery)
        (args.output_dir / f"{name}_packed.json").write_text(
            json.dumps(quality[name], sort_keys=True, allow_nan=False) + "\n"
        )

    arm("control")
    arm("true")
    delta = np.asarray(quality["true"]["per_query_r1"]) - np.asarray(
        quality["control"]["per_query_r1"]
    )
    ap_delta = np.asarray(quality["true"]["per_query_ap"]) - np.asarray(
        quality["control"]["per_query_ap"]
    )
    lower, ap_lower = bootstrap_lower(delta, qlabels), bootstrap_lower(ap_delta, qlabels)
    criteria = {
        "teacher_gain": float(delta.mean()) >= 0.003,
        "teacher_r1_lower": lower > 0,
        "map_guard": float(ap_delta.mean()) >= 0 and ap_lower >= -0.002,
        "large_gross_quality_floor": quality["true"]["recall_at_1"] >= 0.921999,
        "train_cost": arms["true"]["training_wall_seconds"] <= 113.864,
        "memory": arms["true"]["training_peak_cuda_allocated_bytes"] < 32 * 1024**3,
    }
    if all(criteria.values()):
        arm("sham")
        criteria["teacher_specific"] = (
            quality["true"]["recall_at_1"] > quality["sham"]["recall_at_1"]
            and quality["true"]["map_at_r"] >= quality["sham"]["map_at_r"]
        )
    result = {
        "schema": "sfora-inshop-teacher-transfer-100-v1",
        "claim_eligible": False,
        "evaluation_exposure": "reused official TRAIN holdout; exploratory",
        "decision": "GO_PAIRED_SEED_GATE_DESIGN" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "r1_delta": float(delta.mean()),
        "r1_delta_95_lower": lower,
        "map_delta": float(ap_delta.mean()),
        "map_delta_95_lower": ap_lower,
        "quality": quality,
        "training_receipt_sha256": receipt_hashes,
        "training_cost": {
            name: {
                key: receipt[key]
                for key in (
                    "training_wall_seconds",
                    "whole_arm_wall_seconds",
                    "training_peak_cuda_allocated_bytes",
                    "export_seconds",
                    "score_seconds",
                    "teacher_forward_seconds",
                )
            }
            for name, receipt in arms.items()
        },
        "cache_receipt_sha256": sha256(cache / "receipt.json"),
        "cache_receipt": json.loads((cache / "receipt.json").read_text()),
        "teacher_historical_training_wall_seconds": 745.063,
        "source_sha256": sha256(Path(__file__)),
        "main_wall_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("quality", "cache_receipt", "training_cost")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
