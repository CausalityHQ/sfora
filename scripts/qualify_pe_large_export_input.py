#!/usr/bin/env python3
"""Actual native B32 held input parity and CPU preparation attribution, no scores."""

import json
import time
from pathlib import Path

import torch
from transformers import AutoImageProcessor

import pe_large_pool as native
from pe_l14_prefetch import prefetch
from pe_prefetched_export import export_prefetched


def main():
    pair = native.pair
    root = Path(__file__).resolve().parent
    output = Path("/home/riomus/runs/sfora-large-pool-export-input-cpu-v2")
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    control, frozen, _ = native.control(root)
    cpu = Path("/home/riomus/runs/sfora-large-native-pool-cpu-v3/preflight.json")
    assert pair.sha(cpu) == "8def5e7b45595d6d6831f691fc34d46afb255fa834efc38bff7b5d478c7d0754"
    initial = json.loads(cpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in initial["code"].items())
    processor = AutoImageProcessor.from_pretrained(control.large_snapshot, local_files_only=True, backend="torchvision")
    assert json.loads(json.dumps(processor.to_dict())) == initial["environment"]["processor"]
    rows = frozen["held_manifest"]
    batches = [rows[i * 32 : (i + 1) * 32] for i in range(17)] + [rows[-23:]]
    assert len(rows) == 12599 and len(rows) % 32 == 23

    def prepare(index):
        batch = batches[index]
        tick = time.perf_counter()
        images, rgb = pair.augmented_images(control.dataset_root, batch, tuple(range(len(batch))), None)
        image_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        pixels = pair.pixels(processor, images, "large")
        assert pixels.device.type == "cpu" and not pixels.requires_grad and pixels.is_contiguous()
        return pixels, rgb, image_seconds, time.perf_counter() - tick

    rng = torch.random.get_rng_state().clone()
    serial = [prepare(i) for i in range(len(batches))]
    assert torch.equal(rng, torch.random.get_rng_state())
    pixel_hashes = []
    for index, (pixels, rgb, _, _) in enumerate(prefetch(prepare, len(batches))):
        assert torch.equal(pixels, serial[index][0]) and rgb == serial[index][1]
        assert pixels.stride() == serial[index][0].stride()
        pixel_hashes.append(pair.smoke.digest({"pixels": pixels}))
    assert torch.equal(rng, torch.random.get_rng_state())
    # Exercise the production callback/writer seam using independently made real pixels.
    def prepare_rows(batch):
        images, _ = pair.augmented_images(control.dataset_root, batch, tuple(range(len(batch))), None)
        return pair.pixels(processor, images, "large")

    fixture_rows = rows[:67]
    expected_batches = [prepare_rows(fixture_rows[i:i + 32]) for i in range(0, 67, 32)]
    position = 0

    def encode_pixels(batch, pixels):
        nonlocal position
        expected = expected_batches[position]
        position += 1
        assert torch.equal(pixels, expected)
        return pixels.flatten(1)[:, :2].numpy().copy()

    output.mkdir(exist_ok=False)
    path = output / "input-writer-fixture.npy"
    export_prefetched(rows[:67], prepare_rows, encode_pixels, path, width=2)
    path.unlink()
    assert torch.equal(rng, torch.random.get_rng_state())
    pair.smoke.save(output / "preflight.json", {
        "original_native_cpu_sha256": pair.sha(cpu),
        "code": {**initial["code"], **{n: pair.sha(root / n) for n in ("pe_l14_prefetch.py", "pe_prefetched_export.py", "check_pe_prefetched_export.py", "qualify_pe_large_export_input.py")}},
        "serial_worker_pixels_rgb_stride_exact": True,
        "caller_cpu_rng_unchanged": True,
        "original_writer_real_pixel_callback_exact": True,
        "batch_sizes": [len(b) for b in batches],
        "pixels_sha256": pixel_hashes,
        "rgb_sha256": [s[1] for s in serial],
        "serial_image_authority_decode_rgb_seconds": [s[2] for s in serial],
        "serial_native_processor_seconds": [s[3] for s in serial],
        "pending_batches_maximum": 1,
        "cuda": False, "optimizer_updates": 0, "quality_read": False,
    })
    print("PASS actual B32 held serial/worker pixels/RGB/stride/RNG, final23 and original writer; no model-quality read")


if __name__ == "__main__":
    main()
