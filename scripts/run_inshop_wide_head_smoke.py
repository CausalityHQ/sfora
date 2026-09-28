#!/usr/bin/env python3
"""One paired17-update TRAIN-fit-only width smoke,120s whole-job deadline."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import sha256

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.siglip2_compact_serving import Siglip2CompactEncoder
from sfora.unicom_inshop import parse_inshop_partition


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
    if args.output_dir.exists():
        raise ValueError("wide smoke output already exists")
    previous = json.loads(args.native_control.read_text())
    if (
        sha256(args.native_control)
        != "83c977aa6d57fc5c0aff10824b8ffa0ac38c5d47cfb43c4e88679848e301063b"
    ):
        raise ValueError("wide smoke native control authority differs")
    args.output_dir.mkdir(parents=True)
    arms = {}
    for width in (128, 256):
        destination = args.output_dir / str(width)
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
            "--arm",
            "freeze_emb",
            "--updates",
            "17",
            "--seed",
            "179024",
            "--workers",
            "4",
            "--training-width",
            str(width),
            "--wide-head-smoke-receipt",
            str(args.cached_gate),
            "--output-dir",
            str(destination),
        ]
        with (args.output_dir / f"{width}.log").open("wb") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=max(1, 120 - (time.perf_counter() - started)),
            )
        arms[width] = json.loads((destination / "receipt.json").read_text())
        if width == 128 and (
            arms[width]["first_input_batch_sha256"] != previous["first_input_batch_sha256"]
            or not np.allclose(
                arms[width]["all_step_losses"], previous["all_step_losses"], rtol=0, atol=1e-5
            )
        ):
            raise ValueError("fresh native smoke differs from original17-step control")
        print(
            json.dumps({"terminal_width": width, "elapsed": time.perf_counter() - started}),
            flush=True,
        )
    control, wide = arms[128], arms[256]
    histories = {
        width: np.asarray(
            [
                [step["group_gradient_norms"][name] for name in ("vision", "head", "classifier")]
                for step in arms[width]["width_history"]
            ]
        )
        for width in arms
    }
    ratio = histories[256][:, 0] / histories[128][:, 0]
    share_ratio = (histories[256][:, 0] / np.linalg.norm(histories[256], axis=1)) / (
        histories[128][:, 0] / np.linalg.norm(histories[128], axis=1)
    )
    destination = args.output_dir / "256"
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = Siglip2CompactEncoder.from_checkpoint(
        model_snapshot=args.model_snapshot,
        checkpoint=destination / "folded_checkpoint.pt",
        expected_checkpoint_sha256=wide["width_fold"]["folded_checkpoint_sha256"],
        model_file_sha256=MODEL_HASHES,
        precision="fp32_autocast",
        device=torch.device("cuda"),
    )
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, _ = split(tuple(row.label for row in train))
    from PIL import Image

    images = []
    for row in fit[:64]:
        with Image.open(train[row].image_path) as image:
            images.append(image.convert("RGB"))
    probe = torch.load(destination / "fold_probe.pt", map_location="cpu", weights_only=True)
    actual = encoder.encode_images(images)
    expected = pack_int8_unit_embeddings(probe["folded"])
    exact = torch.equal(actual.codes, expected.codes) and torch.equal(
        actual.inverse_norms, expected.inverse_norms
    )
    composed = pack_int8_unit_embeddings(probe["composed"])
    flips = expected.codes != composed.codes
    max_flip = int((expected.codes.int() - composed.codes.int()).abs().max())
    try:
        Siglip2CompactEncoder.from_checkpoint(
            model_snapshot=args.model_snapshot,
            checkpoint=destination / "checkpoint.pt",
            expected_checkpoint_sha256=wide["checkpoint_sha256"],
            model_file_sha256=MODEL_HASHES,
            precision="fp32_autocast",
            device=torch.device("cuda"),
        )
    except RuntimeError as error:
        if "size mismatch" not in str(error):
            raise
        rejected = True
    else:
        rejected = False
    criteria = {
        "matched_all17_pixels": control["first_input_batch_sha256"]
        == wide["first_input_batch_sha256"],
        "encoder_pressure": bool(np.all((ratio[[0, -1]] >= 0.25) & (ratio[[0, -1]] <= 4))),
        "encoder_share": 0.5 <= float(np.median(share_ratio)) <= 2,
        "wide_step_cost": float(np.median(wide["step_seconds"]))
        <= 1.05 * float(np.median(control["step_seconds"])),
        "wide_vram": wide["training_peak_cuda_allocated_bytes"]
        <= control["training_peak_cuda_allocated_bytes"] + 2**30,
        "no_collapse": all(
            wide["width_terminal_geometry"][name] >= 0.5 * wide["width_initial_geometry"][name]
            for name in ("variance", "effective_rank")
        ),
        "exact_native_reload": exact,
        "raw_wide_rejected": rejected,
        "association_flips_at_most_one_lsb": max_flip <= 1,
        "whole_budget": time.perf_counter() - started <= 120,
    }
    receipt = {
        "schema": "sfora-inshop-wide-main-head-smoke-v1",
        "claim_eligible": False,
        "decision": "GO_FROZEN_100_GATE" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "arms": {str(width): arms[width] for width in arms},
        "initial_terminal_encoder_norm_ratios": ratio[[0, -1]].tolist(),
        "median_encoder_share_ratio": float(np.median(share_ratio)),
        "association_code_flip_fraction": float(flips.float().mean()),
        "association_max_lsb": max_flip,
        "whole_wall_seconds": time.perf_counter() - started,
        "source_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "criteria": criteria,
                "whole_wall_seconds": receipt["whole_wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
