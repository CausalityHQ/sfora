#!/usr/bin/env python3
"""One bounded serial native/product-PCA encoder17 pair; no held export."""

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
        != "b6c684bcdfb90b3644c5ef66a3de1c1a5d736f46003474775e57c7af2a1c51bf"
        or sha256(args.cached_gate)
        != "0d6a2418429bf90cf1e0f4fce5aa7d3a11e74fa77e44888014998f8e656d6df6"
    ):
        raise ValueError("centroid encoder smoke authority differs")
    previous = json.loads(args.native_control.read_text())
    args.output_dir.mkdir(parents=True)
    arms = {}
    criteria = {}
    for name in ("control", "products"):
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
            "--centroid-pca-smoke",
            name,
            "--centroid-pca-receipt",
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
            result = {
                "decision": "KILL_CHILD_BUDGET"
                if isinstance(error, subprocess.TimeoutExpired)
                else "KILL_CHILD_FAILURE",
                "failed_arm": name,
                "arms": arms,
                "claim_eligible": False,
                "quality": None,
                "whole_wall_seconds": time.perf_counter() - started,
            }
            (args.output_dir / "receipt.json").write_text(json.dumps(result, sort_keys=True) + "\n")
            raise
        arms[name] = json.loads((args.output_dir / name / "receipt.json").read_text())
        current = arms[name]
        criteria[f"{name}_finite17"] = (
            len(current["all_step_losses"]) == 17
            and np.isfinite(current["all_step_losses"]).all()
            and len(current["width_history"]) == 17
        )
        criteria[f"{name}_public_live_parity"] = all(current["matched_public32_parity"].values())
        criteria[f"{name}_geometry"] = all(
            current["width_terminal_geometry"][key]
            >= fraction * current["width_initial_geometry"][key]
            for key, fraction in (("variance", 0.5), ("effective_rank", 0.8))
        )
        if name == "control":
            criteria["native_pixels"] = (
                current["first_input_batch_sha256"] == previous["first_input_batch_sha256"]
            )
            criteria["native_losses"] = np.allclose(
                current["all_step_losses"], previous["all_step_losses"], rtol=0, atol=1e-5
            )
            criteria["native_initializer"] = (
                current["pca_sha256"] == previous["pca_sha256"]
                and current["centroid_pca_initializer"]["vision_sha256"]
                == previous["source_main_initializer"]["vision_sha256"]
                and current["centroid_pca_initializer"]["head_sha256"]
                == previous["source_main_initializer"]["head_sha256"]
                and current["centroid_pca_initializer"]["bank_sha256"]
                == previous["source_main_initializer"]["bank_sha256"]
            )
        print(json.dumps({"terminal_arm": name, "criteria": criteria}), flush=True)
        if not all(criteria.values()):
            break
    if "products" in arms:
        control, product = arms["control"], arms["products"]
        criteria["paired_pixels17"] = (
            len(control["first_input_batch_sha256"]) == 17
            and control["first_input_batch_sha256"] == product["first_input_batch_sha256"]
        )
        criteria["paired_authority"] = all(
            control[key] == product[key]
            for key in (
                "features_sha256",
                "fit_rows_sha256",
                "schedule_sha256",
                "model_file_sha256",
                "source_files_sha256",
                "executed_schedule_sha256",
            )
        )
        ci, pi = control["centroid_pca_initializer"], product["centroid_pca_initializer"]
        criteria["paired_initializer"] = ci["vision_sha256"] == pi["vision_sha256"] and all(
            ci[key] != pi[key] for key in ("head_sha256", "classifier_sha256", "bank_sha256")
        )
        criteria["step_cost"] = np.median(product["step_seconds"][1:]) <= 1.05 * np.median(
            control["step_seconds"][1:]
        )
        criteria["cuda_cost"] = (
            product["training_peak_cuda_allocated_bytes"]
            <= 1.005 * control["training_peak_cuda_allocated_bytes"]
        )
    criteria["whole_pair_budget"] = time.perf_counter() - started <= 120
    result = {
        "schema": "sfora-inshop-centroid-pca-smoke-v1",
        "claim_eligible": False,
        "decision": "GO_FROZEN_100_GATE" if len(arms) == 2 and all(criteria.values()) else "KILL",
        "criteria": {key: bool(value) for key, value in criteria.items()},
        "arms": arms,
        "quality": None,
        "whole_wall_seconds": time.perf_counter() - started,
        "native_control_sha256": sha256(args.native_control),
        "cached_gate_sha256": sha256(args.cached_gate),
        "source_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}), flush=True)


if __name__ == "__main__":
    main()
