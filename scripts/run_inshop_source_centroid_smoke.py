#!/usr/bin/env python3
"""Run one serial, wall-bounded paired TRAIN-fit smoke; stop negative arms."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def probe_failures(probes):
    failures = []
    ratios = [probe["weighted_auxiliary_to_main_gradient_ratio"] for probe in probes]
    if all(ratio < 0.01 for ratio in ratios):
        failures.append("weak_auxiliary_encoder_gradient")
    if all(probe["saturated_target_fraction"] > 0.5 for probe in probes):
        failures.append("saturated_auxiliary_targets")
    if any(ratio > 0.25 for ratio in ratios):
        failures.append("dominant_auxiliary_encoder_gradient")
    return failures


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "features-dir",
        "preflight",
        "source-centroid-receipt",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(exist_ok=False)
    base = [sys.executable, str(Path(__file__).with_name("train_inshop_siglip2_unseen_gallery.py"))]
    for name in (
        "dataset-root",
        "model-snapshot",
        "features-dir",
        "preflight",
        "source-centroid-receipt",
    ):
        base.extend((f"--{name}", str(getattr(args, name.replace("-", "_")))))
    base.extend(
        (
            "--preflight-sha256",
            args.preflight_sha256,
            "--arm",
            "freeze_emb",
            "--seed",
            "179024",
            "--updates",
            "17",
        )
    )
    arms, failures, process_results = {}, [], []
    for arm in ("control", "auxiliary"):
        output = args.output_dir / arm
        command = base + ["--output-dir", str(output), "--source-centroid-smoke", arm]
        with (args.output_dir / f"{arm}.log").open("x") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                exit_code = child.wait(timeout=max(0.01, 120 - (time.perf_counter() - started)))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                exit_code = 124
        process_results.append({"arm": arm, "pid": child.pid, "exit_code": exit_code})
        if exit_code:
            failures.append(f"{arm}_process_exit_{exit_code}")
            break
        receipt_path = output / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        arms[arm] = {
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "receipt": receipt,
        }
        failures.extend(probe_failures(receipt["source_centroid_probes"]))
        if failures:
            break
    if len(arms) == 2 and not failures:
        control, auxiliary = (arms[name]["receipt"] for name in ("control", "auxiliary"))
        for key in (
            "fit_rows_sha256",
            "pca_sha256",
            "executed_schedule_sha256",
            "first_input_batch_sha256",
            "source_files_sha256",
            "model_file_sha256",
        ):
            if control[key] != auxiliary[key]:
                failures.append(f"pairing_{key}")
        if any(len(arm["receipt"]["first_input_batch_sha256"]) != 17 for arm in arms.values()):
            failures.append("incomplete_17_input_hashes")
        c_clip = np.minimum(1, 1 / (np.asarray(control["preclip_grad_norms"]) + 1e-6))
        a_clip = np.minimum(1, 1 / (np.asarray(auxiliary["preclip_grad_norms"]) + 1e-6))
        clipping_ratio = float(np.median(a_clip / c_clip))
        if abs(clipping_ratio - 1) > 0.1:
            failures.append("shared_clipping_confounded")
        for key in ("compact_effective_rank", "compact_centered_variance"):
            if (
                auxiliary["source_centroid_probes"][-1][key]
                < 0.8 * control["source_centroid_probes"][-1][key]
            ):
                failures.append(f"collapse_{key}")
    else:
        clipping_ratio = None
    wall = time.perf_counter() - started
    if wall > 120:
        failures.append("paired_whole_wall_budget")
    report = {
        "schema": "sfora-inshop-source-centroid-paired-smoke-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN fit only; no held export",
        "arms": arms,
        "process_results": process_results,
        "whole_paired_wall_seconds": wall,
        "median_paired_clipping_multiplier_ratio": clipping_ratio,
        "failures": failures,
        "advance": len(arms) == 2 and not failures,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "arms"}), flush=True)


if __name__ == "__main__":
    main()
