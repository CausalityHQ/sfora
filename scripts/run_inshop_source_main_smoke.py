#!/usr/bin/env python3
"""One serial native/source MAIN17 pair, TRAIN fit only,120s total watchdog."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from train_sop_siglip2_compact import sha256


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "features-dir",
        "preflight",
        "cached-gate",
        "native-control",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or sha256(args.native_control)
        != "83c977aa6d57fc5c0aff10824b8ffa0ac38c5d47cfb43c4e88679848e301063b"
    ):
        raise ValueError("source MAIN native/output authority differs")
    if (
        sha256(args.cached_gate)
        != "f5bec2865b8aeb86f66003ada0dc5902616d98e043a3d3b8e1716ead95c41203"
    ):
        raise ValueError("source MAIN cache authority differs")
    previous = json.loads(args.native_control.read_text())
    args.output_dir.mkdir(parents=True)
    arms = {}
    for name in ("control", "source"):
        command = [
            sys.executable,
            str(Path(__file__).with_name("train_inshop_siglip2_unseen_gallery.py")),
        ]
        for key in ("dataset-root", "model-snapshot", "features-dir", "preflight"):
            command += [f"--{key}", str(getattr(args, key.replace("-", "_")))]
        command += [
            "--preflight-sha256",
            sha256(args.preflight),
            "--arm",
            "freeze_emb",
            "--updates",
            "17",
            "--seed",
            "179024",
            "--workers",
            "4",
            "--source-main-smoke",
            name,
            "--source-main-receipt",
            str(args.cached_gate),
            "--output-dir",
            str(args.output_dir / name),
        ]
        try:
            with (args.output_dir / f"{name}.log").open("wb") as log:
                subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=max(0.1, 120 - (time.perf_counter() - started)),
                )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
            receipt = {
                "schema": "sfora-inshop-source-main-smoke-v1",
                "claim_eligible": False,
                "decision": "KILL_CHILD_BUDGET"
                if isinstance(error, subprocess.TimeoutExpired)
                else "KILL_CHILD_FAILURE",
                "failed_arm": name,
                "exception": type(error).__name__,
                "arms": arms,
                "quality": None,
                "whole_wall_seconds": time.perf_counter() - started,
            }
            (args.output_dir / "receipt.json").write_text(
                json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n"
            )
            raise
        arms[name] = json.loads((args.output_dir / name / "receipt.json").read_text())
        print(
            json.dumps({"terminal_arm": name, "elapsed": time.perf_counter() - started}), flush=True
        )
        if name == "control" and (
            arms[name]["first_input_batch_sha256"] != previous["first_input_batch_sha256"]
            or not np.allclose(
                arms[name]["all_step_losses"], previous["all_step_losses"], rtol=0, atol=1e-5
            )
        ):
            (args.output_dir / "receipt.json").write_text(
                json.dumps(
                    {
                        "decision": "KILL_NATIVE_PAIRING",
                        "claim_eligible": False,
                        "arms": arms,
                        "quality": None,
                    }
                )
                + "\n"
            )
            return
    control, source = arms["control"], arms["source"]
    initial, terminal = source["source_main_probes"]
    criteria = {
        "all17_pixels": control["first_input_batch_sha256"] == source["first_input_batch_sha256"],
        "initial_head_bank_vision": all(
            control["source_main_initializer"][key] == source["source_main_initializer"][key]
            for key in ("head_sha256", "bank_sha256", "vision_sha256")
        ),
        "17_stable_updates": all(
            a["updates"] == 17 and a["rank_active_updates"] == 15 for a in arms.values()
        ),
        "singleton_head_unchanged": source["singleton_head_checks"]
        == [{"step": 2, "unchanged": True}, {"step": 7, "unchanged": True}]
        and source["optimizer_head_steps"] == [15, 15],
        "initial_encoder_pressure": initial["main_rank_norm_ratio"] >= 0.1,
        "terminal_encoder_pressure": terminal["main_encoder_gradient_norm"]
        >= 0.25 * initial["main_encoder_gradient_norm"],
        "compact_geometry": all(
            terminal["geometry"][key] >= 0.9 * initial["geometry"][key]
            for key in ("variance", "effective_rank")
        ),
        "training_cost": source["training_wall_seconds"] <= 1.10 * control["training_wall_seconds"],
        "median_step_cost": np.median(source["step_seconds"][1:])
        <= 1.10 * np.median(control["step_seconds"][1:]),
        "peak_memory": source["training_peak_cuda_allocated_bytes"]
        <= control["training_peak_cuda_allocated_bytes"] + 2**30,
        "public32_exact": all(all(a["matched_public32_parity"].values()) for a in arms.values()),
        "no_held_read": all(
            a["quality"] is None and a["export_seconds"] is None for a in arms.values()
        ),
        "whole_budget": time.perf_counter() - started <= 120,
    }
    criteria = {key: bool(value) for key, value in criteria.items()}
    receipt = {
        "schema": "sfora-inshop-source-main-smoke-v1",
        "claim_eligible": False,
        "decision": "GO_FROZEN_100_GATE" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "arms": arms,
        "quality": None,
        "source_sha256": sha256(Path(__file__)),
        "whole_wall_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"decision": receipt["decision"], "criteria": criteria}), flush=True)


if __name__ == "__main__":
    main()
