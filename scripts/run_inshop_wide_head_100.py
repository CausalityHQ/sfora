#!/usr/bin/env python3
"""One frozen serial native128/folded128 paired TRAIN-only100-update gate."""

import argparse
import hashlib
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


def record_child_failure(output_dir, name, error, quality, hashes, elapsed, *, source_main=False):
    result = {
        "schema": "sfora-inshop-source-main-100-v1"
        if source_main
        else "sfora-inshop-wide-main-head-100-v1",
        "claim_eligible": False,
        "decision": "KILL_CHILD_BUDGET"
        if isinstance(error, subprocess.TimeoutExpired)
        else "KILL_CHILD_FAILURE",
        "failed_arm": name,
        "exception": type(error).__name__,
        "child_timeout_seconds": getattr(error, "timeout", None),
        "last_completed_quality": {
            arm: {key: value[key] for key in ("recall_at_1", "map_at_r")}
            for arm, value in quality.items()
        },
        "training_receipt_sha256": hashes,
        "main_wall_seconds": elapsed,
        "failed_arm_quality": None,
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "features-dir",
        "preflight",
        "qualification",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-main", action="store_true")
    args = parser.parse_args()
    proposal = "source" if args.source_main else "wide"
    if args.output_dir.exists() or sha256(args.qualification) != (
        "552859f6fad15f093ec1d540391b594fd9ff7ea0458fe942c5523d3ebc3a4c9b"
        if args.source_main
        else "eb4c8c1bf2a059277b92493fc71612d1f4ce9b53504c3ae1d6346155bf01591e"
    ):
        raise ValueError("wide100 authority/output differs")
    args.output_dir.mkdir(parents=True)
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, tuple(train[row].image_path for row in held), args.dataset_root)
    if (
        len(query) != 6354
        or len(gallery) != 6245
        or digest_rows(held) != "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"
    ):
        raise ValueError("wide100 held metadata authority differs")
    if any(
        hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != expected
        for rows, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
    ):
        raise ValueError("wide100 held roles differ")
    qlabels = np.asarray([labels[row] for row in query])
    arms, quality, hashes = {}, {}, {}
    for name, width in (("control", 128), (proposal, 128 if args.source_main else 256)):
        destination = args.output_dir / name
        command = [
            sys.executable,
            str(Path(__file__).with_name("train_inshop_siglip2_unseen_gallery.py")),
            "--dataset-root",
            str(args.dataset_root),
            "--model-snapshot",
            str(args.model_snapshot),
            "--features-dir",
            str(args.features_dir),
            "--preflight",
            str(args.preflight),
            "--preflight-sha256",
            sha256(args.preflight),
            "--output-dir",
            str(destination),
            "--arm",
            "freeze_emb",
            "--updates",
            "100",
            "--seed",
            "179024",
            "--training-width",
            str(width),
            "--wide-head-qualification",
            str(args.qualification),
        ]
        if args.source_main:
            command[-2:] = [
                "--source-main-smoke",
                name,
                "--source-main-qualification",
                str(args.qualification),
            ]
        torch.cuda.empty_cache()
        try:
            with (args.output_dir / f"{name}.log").open("wb") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=min(280, max(1, 600 - (time.perf_counter() - started))),
                )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            record_child_failure(
                args.output_dir,
                name,
                error,
                quality,
                hashes,
                time.perf_counter() - started,
                source_main=args.source_main,
            )
            raise
        receipt_path = destination / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        values_path = destination / "held_values.npy"
        if (
            sha256(values_path) != receipt["held_values_sha256"]
            or sha256(destination / "checkpoint.pt") != receipt["checkpoint_sha256"]
        ):
            raise ValueError("wide100 output/checkpoint authority differs")
        values = np.load(values_path, allow_pickle=False)
        if (
            values.shape != (12599, 128)
            or not np.isfinite(values).all()
            or len(receipt["all_step_losses"]) != 100
            or len(receipt["first_input_batch_sha256"]) != 100
            or not all(receipt["matched_public32_parity"].values())
        ):
            raise ValueError("wide100 update/export/parity inventory differs")
        if args.source_main:
            smoke = json.loads(args.qualification.read_text())["arms"][name]
            if receipt["first_input_batch_sha256"][:17] != smoke[
                "first_input_batch_sha256"
            ] or not np.allclose(
                receipt["all_step_losses"][:17], smoke["all_step_losses"], rtol=0, atol=1e-5
            ):
                raise ValueError("source MAIN100 prefix differs from qualified17")
        if arms:
            paired = (
                "source_sha256",
                "source_files_sha256",
                "fit_rows_sha256",
                "held_rows_sha256",
                "executed_schedule_sha256",
                "first_input_batch_sha256",
                "features_sha256",
                "preflight_sha256",
                "model_file_sha256",
                "export_batch_size",
            )
            if any(receipt[field] != arms["control"][field] for field in paired):
                raise ValueError("wide100 paired inputs differ")
        arms[name] = receipt
        hashes[name] = sha256(receipt_path)
        quality[name] = packed_quality(values, labels, query, gallery)
        (args.output_dir / f"{name}_packed.json").write_text(
            json.dumps(quality[name], sort_keys=True, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "terminal": name,
                    "r1": quality[name]["recall_at_1"],
                    "map_at_r": quality[name]["map_at_r"],
                    "elapsed": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if name == "control" and quality[name]["recall_at_1"] < 0.922:
            (args.output_dir / "receipt.json").write_text(
                json.dumps(
                    {
                        "decision": "KILL_INVALID_NATIVE_CONTROL",
                        "quality": quality,
                        "training_receipt_sha256": hashes,
                    }
                )
                + "\n"
            )
            return
    delta = np.asarray(quality[proposal]["per_query_r1"]) - np.asarray(
        quality["control"]["per_query_r1"]
    )
    ap_delta = np.asarray(quality[proposal]["per_query_ap"]) - np.asarray(
        quality["control"]["per_query_ap"]
    )
    lower = bootstrap_lower(ap_delta, qlabels)
    control, wide = arms["control"], arms[proposal]
    criteria = {
        "map_gain": float(ap_delta.mean()) >= 0.01,
        "map_lower": lower > 0,
        "r1_guard": float(delta.mean()) >= 0,
        "training_cost": wide["training_wall_seconds"] <= 1.10 * control["training_wall_seconds"],
        "median_step": float(np.median(wide["step_seconds"]))
        <= 1.10 * float(np.median(control["step_seconds"])),
        "memory": wide["training_peak_cuda_allocated_bytes"]
        <= control["training_peak_cuda_allocated_bytes"] + 2**30,
        "whole_arm_calibration_cost": wide["whole_arm_wall_seconds"]
        <= (1.15 if args.source_main else 2) * control["whole_arm_wall_seconds"],
        "no_collapse": all(
            wide["width_terminal_geometry"][k] >= 0.5 * wide["width_initial_geometry"][k]
            for k in ("variance", "effective_rank")
        ),
        "whole_budget": time.perf_counter() - started <= 600,
    }
    result = {
        "schema": "sfora-inshop-source-main-100-v1"
        if args.source_main
        else "sfora-inshop-wide-main-head-100-v1",
        "source_main": args.source_main,
        "claim_eligible": False,
        "evaluation_exposure": "observed officialTRAIN held identities; exploratory",
        "decision": "GO_PAIRED_SEED_GATE_DESIGN" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "r1_delta": float(delta.mean()),
        "r1_delta_95": [bootstrap_lower(delta, qlabels), -bootstrap_lower(-delta, qlabels)],
        "map_delta": float(ap_delta.mean()),
        "map_delta_95": [lower, -bootstrap_lower(-ap_delta, qlabels)],
        "rescues": int((delta > 0).sum()),
        "regressions": int((delta < 0).sum()),
        "quality": quality,
        "training_receipt_sha256": hashes,
        "training_cost": {
            name: {
                k: r[k]
                for k in (
                    "training_wall_seconds",
                    "whole_arm_wall_seconds",
                    "training_peak_cuda_allocated_bytes",
                    "export_seconds",
                    "score_seconds",
                    "head_classifier_init_seconds",
                    "width_fold",
                )
            }
            for name, r in arms.items()
        },
        "qualification_sha256": sha256(args.qualification),
        "source_sha256": sha256(Path(__file__)),
        "query_rows_sha256": QUERY_SHA,
        "gallery_rows_sha256": GALLERY_SHA,
        "main_wall_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "decision",
                    "criteria",
                    "r1_delta",
                    "r1_delta_95",
                    "map_delta",
                    "map_delta_95",
                    "main_wall_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
