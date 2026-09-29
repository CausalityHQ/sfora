#!/usr/bin/env python3
"""Actual native AMP weight-cache exact parity and bounded fit-only speed gate."""

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
from pe_immutable_amp import build_cache, cached_call


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pair = native.pair
    root = Path(__file__).resolve().parent
    assert not args.output.exists()
    cpu_path = Path("/home/riomus/runs/sfora-large-immutable-amp-cpu-v1/preflight.json")
    assert pair.sha(cpu_path) == "f48b025eb5921d51cfb7ca98f8500c2c32e23840bf77cf6fc37621b966173591"
    info = json.loads(cpu_path.read_text())
    execution = root / "immutable-amp-gpu-execution.json"
    assert pair.sha(execution) == args.execution_sha256
    code = json.loads(execution.read_text())
    assert all(pair.sha(root / n) == h for n, h in {**info["code"], **code}.items())
    for path, digest in info["functional_source_sha256"].items():
        assert pair.sha(Path(path)) == digest
    checkpoint = Path("/home/riomus/runs/sfora-large-native-pool-pilot-v1/pe.pt")
    assert pair.sha(checkpoint) == info["checkpoint_sha256"]
    assert pair.sha(checkpoint.parent / "training.json") == info["training_sha256"]
    control, frozen, _ = native.control(root)
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, processor = pair.smoke.load_arm(control, "large")
    native.freeze(model)
    assert json.loads(json.dumps(native.environment(model, processor))) == info["native_environment"]
    assert pair.smoke.digest(native.frozen_state(model)) == info["frozen_complement_sha256"]
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(saved["vision"], strict=True)
    model.eval()
    assert len(model.state_dict()) == 400
    assert pair.smoke.digest(native.whole_state(model)) == info["whole_checkpoint_state_sha256"]
    other = type(model)(copy.deepcopy(model.config)).float().eval()
    other.load_state_dict(saved["vision"], strict=True)
    heads = [nn.Linear(1024, 128).eval() for _ in range(2)]
    for h in heads:
        h.load_state_dict(saved["head"], strict=True)
        h.cuda()
    del saved
    model.cuda()
    other.cuda()
    models = (model, other)
    caches = tuple(build_cache(m) for m in models)
    assert all(sorted(c[0]) == info["cached_parameter_names"] for c in caches)
    runtime = native.runtime_identity(model)
    assert runtime == native.runtime_identity(other)
    assert all(pair.smoke.digest(native.whole_state(m)) == info["whole_checkpoint_state_sha256"] for m in models)
    torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(exist_ok=False)
    times = {"eager": [], "cached": []}
    equality = []
    token_shapes = []

    @torch.inference_mode()
    def encode(mode, x):
        vectors = []
        for m, h, cache in zip(models, heads, caches, strict=True):
            if mode == "eager":
                with torch.autocast("cuda", dtype=torch.float16):
                    source = m(pixel_values=x).pooler_output.float()
            else:
                source = cached_call(m, cache, x).pooler_output.float()
            vectors.append((source, F.normalize(pair.smoke.compact_head_features(source, h).float(), dim=1)))
        assert torch.equal(vectors[0][0], vectors[1][0])
        assert torch.equal(vectors[0][1], vectors[1][1])
        return vectors[0]

    def prepare(rows):
        images, _ = pair.augmented_images(control.dataset_root, rows, tuple(range(len(rows))), None)
        return pair.pixels(processor, images, "large").cuda()

    warm = prepare(frozen["fit_manifest"][:32])
    for mode in times:
        encode(mode, warm)
    del warm
    rng = torch.random.get_rng_state().clone()
    for index in range(18):
        rows = frozen["fit_manifest"][index * 32:(index + 1) * 32] if index < 17 else frozen["fit_manifest"][-23:]
        x = prepare(rows)
        results = {}
        for mode in (("eager", "cached") if index % 2 == 0 else ("cached", "eager")):
            hooks = []
            if index in (0, 17):
                for m in models:
                    def tokens(module, inputs):
                        native.assert_frozen_tokens(inputs)
                        token_shapes.append(list(inputs[0].shape))
                    hooks.append(m.head.register_forward_pre_hook(tokens))
            try:
                torch.cuda.synchronize()
                tick = time.perf_counter()
                result = encode(mode, x)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
            finally:
                for hook in hooks:
                    hook.remove()
            if index < 17:
                times[mode].append(elapsed)
            results[mode] = result
        assert torch.equal(results["eager"][0], results["cached"][0]), "native cached pooled output differs"
        assert torch.equal(results["eager"][1], results["cached"][1]), "native cached compact vector differs"
        # Wire parity is checked separately from floating-point vector parity.
        from sfora.joint_relational_compaction import pack_int8_unit_embeddings
        a = pack_int8_unit_embeddings(results["eager"][1].cpu())
        b = pack_int8_unit_embeddings(results["cached"][1].cpu())
        assert np.array_equal(a.codes, b.codes) and np.array_equal(a.inverse_norms, b.inverse_norms)
        equality.append(len(rows))
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
    assert torch.equal(rng, torch.random.get_rng_state())
    assert runtime == native.runtime_identity(model) == native.runtime_identity(other)
    assert all(pair.smoke.digest(native.whole_state(m)) == info["whole_checkpoint_state_sha256"] for m in models)
    assert all(p.dtype == torch.float32 for m in models for p in m.parameters())
    for path, digest in info["functional_source_sha256"].items():
        assert pair.sha(Path(path)) == digest
    assert all(pair.sha(root / n) == h for n, h in code.items())
    means = {k: float(np.mean(v)) for k, v in times.items()}
    ratio = means["cached"] / means["eager"]
    pair.smoke.save(args.output / "receipt.json", {
        "parity_pass": True, "advance": ratio <= 0.90,
        "execution_sha256": args.execution_sha256, "cpu_preflight_sha256": pair.sha(cpu_path),
        "checkpoint_sha256": pair.sha(checkpoint), "timing_seconds": times,
        "mean_paired_encoder_head_seconds": means, "mean_ratio": ratio,
        "cached_parameter_tensors_per_model": info["cached_parameter_tensors"],
        "fit_batch_sizes": equality, "native_pool_token_shapes": token_shapes,
        "pooled_vectors_packed_codes_inverse_norms_exact": True,
        "two_strict_loaded_copies_independent_whole_forwards": True,
        "whole_native_f32_state_runtime_unchanged": True,
        "caller_cpu_rng_unchanged": True,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_updates": 0, "held_images": 0, "quality_read": False, "official_read": False,
    })
    print("GO immutable AMP execution qualification" if ratio <= 0.90 else "KILL fixed cache speed gate", flush=True)


if __name__ == "__main__":
    main()
