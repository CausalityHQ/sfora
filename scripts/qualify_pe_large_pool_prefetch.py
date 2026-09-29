#!/usr/bin/env python3
"""Actual seventeen native CPU inputs: serial/worker bitwise parity."""

import json
from pathlib import Path

import torch

import pe_large_pool as l14
import train_inshop_pe_pair as pair
from transformers import AutoImageProcessor
from pe_l14_prefetch import prefetch


def main():
    root = Path(__file__).resolve().parent
    output = Path("/home/riomus/runs/sfora-large-native-pool-prefetch-cpu-v1")
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    cpu = Path("/home/riomus/runs/sfora-large-native-pool-cpu-v3/preflight.json")
    assert (
        pair.sha(cpu)
        == "8def5e7b45595d6d6831f691fc34d46afb255fa834efc38bff7b5d478c7d0754"
    )
    prior_cpu = json.loads(cpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in prior_cpu["code"].items())
    args, frozen, prior = l14.control(root)
    native = AutoImageProcessor.from_pretrained(
        args.large_snapshot, local_files_only=True, backend="torchvision"
    )
    assert json.loads(json.dumps(native.to_dict())) == prior_cpu["environment"]["processor"]

    def prepare(index):
        images, rgb = pair.augmented_images(
            args.dataset_root,
            frozen["fit_manifest"],
            frozen["batches"][index],
            index + 1,
        )
        pixels = pair.pixels(native, images, "large")
        assert rgb == prior["rgb_sha256"][index]
        assert pixels.device.type == "cpu" and not pixels.requires_grad
        return pixels, rgb

    rng = torch.random.get_rng_state().clone()
    serial = [prepare(index) for index in range(17)]
    assert torch.equal(rng, torch.random.get_rng_state())
    hashes = []
    for index, (pixels, rgb) in enumerate(prefetch(prepare, 17)):
        expected, expected_rgb = serial[index]
        assert rgb == expected_rgb and torch.equal(pixels, expected)
        assert pixels.stride() == expected.stride() and pixels.is_contiguous()
        hashes.append(pair.smoke.digest({"pixels": pixels}))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert hashes[0] == frozen["initializers"]["large"]["first_pixels_sha256"]
    output.mkdir(exist_ok=False)
    pair.smoke.save(
        output / "preflight.json",
        {
            "original_native_cpu_sha256": pair.sha(cpu),
            "code": {
                **prior_cpu["code"],
                **{
                    name: pair.sha(root / name)
                    for name in (
                        "pe_l14_prefetch.py",
                        "qualify_pe_large_pool_prefetch.py",
                    )
                },
            },
            "pixels_sha256": hashes,
            "rgb_sha256": prior["rgb_sha256"][:17],
            "serial_worker_pixels_exact": True,
            "strides_exact": True,
            "caller_cpu_rng_unchanged": True,
            "worker_threads": 1,
            "pending_batches_maximum": 1,
            "cuda": False,
            "optimizer_updates": 0,
            "quality_read": False,
        },
    )
    print(
        "PASS actual seventeen B64 serial/worker pixels, RGB, strides, RNG exact; no training/quality"
    )


if __name__ == "__main__":
    main()
