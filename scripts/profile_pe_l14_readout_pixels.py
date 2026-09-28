#!/usr/bin/env python3
"""CPU native B64 pixel-stage cost, exact frozen actual augmented input."""

import statistics
import time
from pathlib import Path

import torch
from torchvision import transforms

import pe_l14_training as l14
import train_inshop_pe_pair as pair
from core.vision_encoder.transforms import get_image_transform


def main():
    root = Path(__file__).resolve().parent
    output = root / "readout-pixel-profile-v1.json"
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    args, frozen, prior = l14.control(root)
    images, rgb = pair.augmented_images(
        args.dataset_root, frozen["fit_manifest"], frozen["batches"][0], 1
    )
    assert rgb == prior["rgb_sha256"][0]
    native = get_image_transform(224)
    assert isinstance(native.transforms[-2], transforms.ToTensor) and isinstance(
        native.transforms[-1], transforms.Normalize
    )
    prepare = transforms.Compose(native.transforms[:-2])
    normalize = transforms.Compose(native.transforms[-2:])
    prepared = [prepare(image) for image in images]
    reference = pair.pixels(native, images, "pe")
    assert torch.equal(reference, torch.stack([normalize(i) for i in prepared]))
    times = {
        n: [] for n in ("prepare", "tensor_normalize_stack", "whole_native_pixels")
    }
    for repeat in range(12):
        tick = time.perf_counter()
        ready = [prepare(i) for i in images]
        middle = time.perf_counter()
        pixels = torch.stack([normalize(i) for i in ready])
        end = time.perf_counter()
        assert torch.equal(pixels, reference) and pixels.stride() == reference.stride()
        if repeat >= 2:
            times["prepare"].append(middle - tick)
            times["tensor_normalize_stack"].append(end - middle)
            times["whole_native_pixels"].append(end - tick)
    pair.smoke.save(
        output,
        {
            "script_sha256": pair.sha(Path(__file__)),
            "rgb_sha256": rgb,
            "pixels_sha256": pair.smoke.digest({"pixels": reference}),
            "shape": list(reference.shape),
            "strides": list(reference.stride()),
            "torch_threads": 8,
            "samples": times,
            "median_seconds": {n: statistics.median(t) for n, t in times.items()},
            "decode_augmentation_included": False,
            "source_encoder_included": False,
            "training_included": False,
            "quality_read": False,
            "optimizer_updates": 0,
            "cuda": False,
        },
    )
    print(
        "COMPLETE actual B64 native CPU pixel stage profile; exact pixel/stride authority, no model/quality/update"
    )


if __name__ == "__main__":
    main()
