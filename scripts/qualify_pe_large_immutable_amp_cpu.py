#!/usr/bin/env python3
"""Actual saved native checkpoint CPU FP16 cache parity and role/state guards."""

import inspect
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import pe_large_pool as native
from pe_immutable_amp import build_cache, cached_call


def main():
    pair = native.pair
    root = Path(__file__).resolve().parent
    output = Path("/home/riomus/runs/sfora-large-immutable-amp-cpu-v1")
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    gpu_path = Path("/home/riomus/runs/sfora-large-native-pool-gpu-v1/preflight.json")
    assert pair.sha(gpu_path) == "9f49b23c1128eca8b62af6330abaf34f189882d3051bc148beb7223c65f2aa85"
    gpu = json.loads(gpu_path.read_text())
    assert all(pair.sha(root / n) == h for n, h in gpu["code"].items())
    control, frozen, _ = native.control(root)
    checkpoint = Path("/home/riomus/runs/sfora-large-native-pool-pilot-v1/pe.pt")
    assert pair.sha(checkpoint) == "a7e3d8d413aba67dd0d1472d5b70537a72de902ecbdac505833af9cbcaa1a796"
    training_path = checkpoint.parent / "training.json"
    assert pair.sha(training_path) == "ebf8521b70e46280824ada916fa83bb7488eb856afb865edc6630b636d24d409"
    training = json.loads(training_path.read_text())
    model, processor = pair.smoke.load_arm(control, "large")
    native.freeze(model)
    assert pair.smoke.digest(native.frozen_state(model)) == training["frozen_sha256"]
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(saved["vision"], strict=True)
    model.eval()
    assert len(model.state_dict()) == 400
    assert pair.smoke.digest(native.frozen_state(model)) == training["frozen_sha256"]
    assert pair.smoke.digest(dict(model.head.named_parameters())) == training["final_group_sha256"]["native_pool"]
    head = nn.Linear(1024, 128).eval()
    head.load_state_dict(saved["head"], strict=True)
    assert pair.smoke.digest(head.state_dict()) == training["final_group_sha256"]["compact_head"]
    del saved
    original = pair.smoke.digest(native.whole_state(model))
    runtime = native.runtime_identity(model)
    cache = build_cache(model)
    assert all(not n.startswith(("embeddings.position_embedding", "post_layernorm", "head.probe", "head.layernorm")) for n in cache[0])
    assert all("layer_norm" not in n for n in cache[0])
    assert all(v.dtype == torch.float16 and not v.requires_grad for v in cache[0].values())
    images, _ = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], (0, 1), None)
    pixels = pair.pixels(processor, images, "large")
    token_inputs = []

    def tokens(module, inputs):
        assert module is model.head
        native.assert_frozen_tokens(inputs)
        token_inputs.append(list(inputs[0].shape))

    hook = model.head.register_forward_pre_hook(tokens)
    try:
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.float16):
            eager = model(pixel_values=pixels).pooler_output.float()
        cached = cached_call(model, cache, pixels).pooler_output.float()
    finally:
        hook.remove()
    assert token_inputs == [[2, 256, 1024], [2, 256, 1024]]
    assert torch.equal(eager, cached), "actual native CPU AMP cached output differs"
    with torch.no_grad():
        a = F.normalize(pair.smoke.compact_head_features(eager, head).float(), dim=1)
        b = F.normalize(pair.smoke.compact_head_features(cached, head).float(), dim=1)
    assert torch.equal(a, b)
    assert native.runtime_identity(model) == runtime
    assert pair.smoke.digest(native.whole_state(model)) == original
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with torch.no_grad():
        original_probe = model.head.probe.clone()
        try:
            model.head.probe.add_(0.01)
            try:
                cached_call(model, cache, pixels)
            except AssertionError as error:
                assert str(error) == "source changed"
            else:
                raise AssertionError("updated native probe accepted stale cache")
        finally:
            model.head.probe.copy_(original_probe)
    assert pair.smoke.digest(native.whole_state(model)) == original
    assert native.runtime_identity(model) == runtime
    output.mkdir(exist_ok=False)
    pair.smoke.save(output / "preflight.json", {
        "code": {**gpu["code"], **{n: pair.sha(root / n) for n in ("pe_immutable_amp.py", "check_pe_immutable_amp.py", "qualify_pe_large_immutable_amp_cpu.py")}},
        "native_gpu_qualification_sha256": pair.sha(gpu_path),
        "checkpoint_sha256": pair.sha(checkpoint),
        "training_sha256": pair.sha(training_path),
        "native_environment": native.environment(model, processor),
        "functional_source_sha256": {inspect.getfile(f): pair.sha(Path(inspect.getfile(f))) for f in (torch.func.functional_call, torch.nn.utils.stateless._functional_call)},
        "cached_parameter_names": sorted(cache[0]),
        "cached_parameter_tensors": len(cache[0]),
        "cached_elements": sum(v.numel() for v in cache[0].values()),
        "whole_checkpoint_state_sha256": original,
        "frozen_complement_sha256": training["frozen_sha256"],
        "runtime_identity": runtime,
        "native_cpu_fp16_pool_and_head_exact": True,
        "native_pool_tokens_f32_no_graph": token_inputs,
        "native_f32_state_preserved": True,
        "native_mutation_rejected": True,
        "fit_images": 2, "held_images": 0, "optimizer_updates": 0, "quality_read": False, "cuda": False,
    })
    print("PASS actual saved native400-key F32 state, CPU AMP cached pooled/head bits, F32 graph-free pool tokens, mutation rejection; no training/quality")


if __name__ == "__main__":
    main()
