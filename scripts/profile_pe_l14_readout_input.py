#!/usr/bin/env python3
"""Actual frozen seventeen B64 inputs: CPU cost attribution, no model/update."""

import statistics
import time
from pathlib import Path

import torch

import pe_l14_training as l14
import train_inshop_pe_pair as pair
from core.vision_encoder.transforms import get_image_transform


def main():
    root = Path(__file__).resolve().parent
    output = root / "readout-input-profile-v1.json"
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    args, frozen, prior = l14.control(root)
    native = get_image_transform(224)
    samples, hashes = [], []
    for step, batch in enumerate(frozen["batches"][:17], 1):
        tick = time.perf_counter()
        images, rgb = pair.augmented_images(
            args.dataset_root, frozen["fit_manifest"], batch, step
        )
        middle = time.perf_counter()
        pixels = pair.pixels(native, images, "pe")
        end = time.perf_counter()
        assert rgb == prior["rgb_sha256"][step - 1]
        assert list(pixels.shape) == [64, 3, 224, 224]
        assert pixels.is_contiguous() and torch.isfinite(pixels).all()
        hashes.append(pair.smoke.digest({"pixels": pixels}))
        samples.append(
            {
                "decode_aug_hash": middle - tick,
                "native_pixels": end - middle,
                "whole_input": end - tick,
            }
        )
    pair.smoke.save(
        output,
        {
            "script_sha256": pair.sha(Path(__file__)),
            "samples": samples,
            "median_step_3_17_seconds": {
                name: statistics.median(row[name] for row in samples[2:])
                for name in samples[0]
            },
            "pixels_sha256": hashes,
            "rgb_sha256": prior["rgb_sha256"][:17],
            "torch_threads": 8,
            "cuda": False,
            "optimizer_updates": 0,
            "source_encoder_included": False,
            "quality_read": False,
        },
    )
    print("COMPLETE CPU actual seventeen B64 inputs; no source/training/quality")


if __name__ == "__main__":
    main()
