#!/usr/bin/env python3
"""Fit-only serial/overlap native export attribution; no learning or quality."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pe_large_pool as native
from pe_prefetched_export import export_prefetched


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pair = native.pair
    root = Path(__file__).resolve().parent
    assert not args.output.exists()
    manifest = root / "export-overlap-measurement-execution.json"
    assert pair.sha(manifest) == args.execution_sha256
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items())
    cpu = Path("/home/riomus/runs/sfora-large-pool-export-input-cpu-v2/preflight.json")
    assert pair.sha(cpu) == "3b62ba2005a6598ae3da5ff83ca67a06ce12de98650ef33165ad59d3572935c0"
    qualified = json.loads(cpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in qualified["code"].items())
    gpu = Path("/home/riomus/runs/sfora-large-native-pool-gpu-v1/preflight.json")
    assert pair.sha(gpu) == "9f49b23c1128eca8b62af6330abaf34f189882d3051bc148beb7223c65f2aa85"
    gpu_qualified = json.loads(gpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in gpu_qualified["code"].items())
    checkpoint = Path("/home/riomus/runs/sfora-large-native-pool-pilot-v1/pe.pt")
    assert pair.sha(checkpoint) == "a7e3d8d413aba67dd0d1472d5b70537a72de902ecbdac505833af9cbcaa1a796"
    training_path = checkpoint.parent / "training.json"
    assert pair.sha(training_path) == "ebf8521b70e46280824ada916fa83bb7488eb856afb865edc6630b636d24d409"
    training = json.loads(training_path.read_text())
    control, frozen, _ = native.control(root)
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, processor = pair.smoke.load_arm(control, "large")
    native.freeze(model)
    assert json.loads(json.dumps(native.environment(model, processor))) == gpu_qualified["environment"]
    assert pair.smoke.digest(native.frozen_state(model)) == training["frozen_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(saved["vision"], strict=True)
    assert len(model.state_dict()) == 400
    assert pair.smoke.digest(native.frozen_state(model)) == training["frozen_sha256"]
    assert pair.smoke.digest(dict(model.head.named_parameters())) == training["final_group_sha256"]["native_pool"]
    head = nn.Linear(1024, 128)
    head.load_state_dict(saved["head"], strict=True)
    assert pair.smoke.digest(head.state_dict()) == training["final_group_sha256"]["compact_head"]
    model.cuda().eval()
    head.cuda().eval()
    other = type(model)(copy.deepcopy(model.config)).float().eval()
    other.load_state_dict(saved["vision"], strict=True)
    other.cuda()
    other_head = nn.Linear(1024, 128)
    other_head.load_state_dict(saved["head"], strict=True)
    other_head.cuda().eval()
    del saved
    original_state = pair.smoke.digest(native.whole_state(model))
    assert original_state == pair.smoke.digest(native.whole_state(other))
    runtime = native.runtime_identity(model)
    assert runtime == native.runtime_identity(other)
    torch.cuda.reset_peak_memory_stats()
    rows = frozen["fit_manifest"][:17 * 32]
    args.output.mkdir(exist_ok=False)
    measures, pixels_hashes = {}, {}

    @torch.inference_mode()
    def encode(pixels, record):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        x = pixels.cuda()
        a, b = native.verified_features(other, model, x)
        va = F.normalize(pair.smoke.compact_head_features(a, other_head).float(), dim=1)
        vb = F.normalize(pair.smoke.compact_head_features(b, head).float(), dim=1)
        assert torch.equal(va, vb)
        values = va.cpu().numpy()
        torch.cuda.synchronize()
        record.append(time.perf_counter() - tick)
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        return values

    # Warm the actual same B32 independent-forward/head path before either timing arm.
    images, _ = pair.augmented_images(control.dataset_root, rows[:32], tuple(range(32)), None)
    encode(pair.pixels(processor, images, "large"), [])
    del images
    for mode in ("serial", "overlap"):
        prep_seconds, forward_seconds, pixel_hashes = [], [], []

        def prepare(batch):
            tick = time.perf_counter()
            images, _ = pair.augmented_images(control.dataset_root, batch, tuple(range(len(batch))), None)
            pixels = pair.pixels(processor, images, "large")
            prep_seconds.append(time.perf_counter() - tick)
            assert pixels.device.type == "cpu" and not pixels.requires_grad
            pixel_hashes.append(pair.smoke.digest({"pixels": pixels}))
            return pixels

        rng = torch.random.get_rng_state().clone()
        torch.cuda.synchronize()
        tick = time.perf_counter()
        path = args.output / (mode + "-fit.npy")
        if mode == "serial":
            pair.export_features(rows, lambda batch: encode(prepare(batch), forward_seconds), path, width=128, batch_size=32)
        else:
            export_prefetched(rows, prepare, lambda batch, pixels: encode(pixels, forward_seconds), path, width=128)
        wall = time.perf_counter() - tick
        assert torch.equal(rng, torch.random.get_rng_state())
        assert len(forward_seconds) == len(prep_seconds) == 17
        pixels_hashes[mode] = pixel_hashes
        measures[mode] = {"whole_export_seconds": wall, "cpu_prepare_seconds": prep_seconds, "paired_h2d_encoder_heads_d2h_seconds": forward_seconds, "vectors_sha256": pair.sha(path)}
        print(json.dumps({"mode": mode, "whole_export_seconds": wall}), flush=True)
    assert pixels_hashes["serial"] == pixels_hashes["overlap"]
    assert np.array_equal(np.load(args.output / "serial-fit.npy"), np.load(args.output / "overlap-fit.npy"))
    for mode in measures:
        (args.output / (mode + "-fit.npy")).unlink()
    assert original_state == pair.smoke.digest(native.whole_state(model)) == pair.smoke.digest(native.whole_state(other))
    assert runtime == native.runtime_identity(model) == native.runtime_identity(other)
    assert all(pair.sha(root / n) == h for n, h in code.items())
    pair.smoke.save(args.output / "receipt.json", {"pass": True, "measurements": measures, "checkpoint_sha256": pair.sha(checkpoint), "execution_sha256": args.execution_sha256, "cpu_input_sha256": pair.sha(cpu), "native_gpu_qualification_sha256": pair.sha(gpu), "fit_images": 544, "held_images": 0, "optimizer_updates": 0, "quality_read": False, "official_read": False, "two_distinct_strict_loaded_checkpoint_copies": True, "same_b32_native_pixels_exact": True, "serial_overlap_vectors_exact": True, "whole_state_runtime_unchanged": True, "caller_cpu_rng_unchanged": True, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "diagnostic_vectors_discarded": True})
    print("PASS fit-only unchanged saved native checkpoint; serial/overlap exact pixels/vectors and attributed measured execution, no training/quality")


if __name__ == "__main__":
    main()
