#!/usr/bin/env python3
"""One actual native FP16 final-block qualification; no optimizer update or held read."""

import argparse
import copy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pe_large_final_block as native
from qualify_pe_large_pool_gpu import attention_nodes
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

CPU = Path("/home/riomus/runs/sfora-large-final-block-cpu-v3/preflight.json")
CPU_SHA = "98b6b7a1f5136e99c0d5ed09d2c2c963b38651b3b4a21660b92eefceb3559032"


def startup(root, execution_sha):
    pair = native.base.pair
    manifest = root / "large-final-block-gpu-execution.json"
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), "GPU execution code differs"
    assert pair.sha(CPU) == CPU_SHA
    info = json.loads(CPU.read_text())
    assert all(pair.sha(root / n) == h for n, h in info["code"].items()), "CPU-qualified code differs"
    assert info["optimizer_updates"] == 0 and info["cuda"] is False and info["quality_read"] is False
    assert info["native_trainable_tensors"] == 16 and info["optimizer_tensors"] == 19
    for path, digest in info["environment"]["native_files"].items():
        assert pair.sha(Path(path)) == digest
    control, frozen, prior = native.base.control(root)
    return control, frozen, prior, info, code


def numerical_flags():
    return {"deterministic": torch.are_deterministic_algorithms_enabled(), "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(), "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction, "cudnn_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic, "flash_sdpa": torch.backends.cuda.flash_sdp_enabled(), "efficient_sdpa": torch.backends.cuda.mem_efficient_sdp_enabled(), "cudnn_sdpa": torch.backends.cuda.cudnn_sdp_enabled(), "math_sdpa": torch.backends.cuda.math_sdp_enabled(), "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def fp16(model, pixels):
    assert not torch.is_autocast_enabled("cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        return model(pixel_values=pixels).pooler_output.float()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--check-startup-only", action="store_true")
    args = parser.parse_args()
    base, pair, root = native.base, native.base.pair, Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, prior, info, code = startup(root, args.execution_sha256)
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        print("PASS native GPU startup authority; no CUDA/images/quality")
        return
    assert not args.output.exists() and torch.cuda.is_available()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flags = numerical_flags()
    assert flags["deterministic"] and not flags["warn_only"]
    args.output.mkdir(exist_ok=False)
    model, processor = pair.smoke.load_arm(control, "large")
    inventory = native.freeze(model)
    assert inventory == {k: tuple(v) for k, v in info["inventory"].items()}
    assert pair.smoke.digest(base.whole_state(model)) == info["whole_original_source_sha256"]
    assert json.loads(json.dumps(native.environment(model, processor))) == info["environment"]
    images, rgb = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], frozen["batches"][0], 1)
    first = pair.pixels(processor, images, "large")
    assert rgb == prior["rgb_sha256"][0] and pair.smoke.digest({"pixels": first}) == frozen["initializers"]["large"]["first_pixels_sha256"]
    del first, images
    torch.cuda.reset_peak_memory_stats()
    model.cuda().train()
    head = nn.Linear(1024, 128)
    with np.load(base.INIT / "initializers.npz", allow_pickle=False) as init:
        head.load_state_dict({k: torch.from_numpy(init["large.head." + k]) for k in ("weight", "bias")})
        classifier = nn.Parameter(torch.from_numpy(init["large.classifier"].copy()).cuda())
        bank = torch.from_numpy(init["large.bank"].copy()).cuda()
    head.cuda()
    active = [p for p in model.parameters() if p.requires_grad]
    members = active + list(head.parameters()) + [classifier]
    optimizer = torch.optim.AdamW([{"params": active, "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-4}, {"params": [classifier], "lr": 1e-4}], weight_decay=0.05)
    assert not optimizer.state and len(members) == len({id(p) for p in members}) == 19
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in members}
    _, scaler = pair.smoke.training_precision("fp16", device="cuda")
    assert scaler.get_scale() == 128
    images, _ = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], (0, 1), None)
    pixels = pair.pixels(processor, images, "large").cuda()
    with torch.no_grad():
        full_precision = model(pixel_values=pixels).pooler_output.float()
        baseline = fp16(model, pixels)
        cached = torch.from_numpy(np.load(control.cache / "large.fit.npy", mmap_mode="r")[:2].copy()).cuda()
        calibration = {"fp16_fp32": F.cosine_similarity(baseline, full_precision).tolist(), "cached_fresh": F.cosine_similarity(cached, baseline).tolist()}
        assert all(min(v) >= 0.999 for v in calibration.values())
    seen = {}

    def block_input(module, inputs):
        assert module is model.encoder.layers[-1] and len(inputs) == 2 and inputs[1] is None
        base.assert_frozen_tokens((inputs[0],))
        seen["block_input"] = list(inputs[0].shape)

    def pool_input(module, inputs):
        assert module is model.head and len(inputs) == 1
        assert inputs[0].requires_grad and inputs[0].grad_fn is not None and inputs[0].dtype == torch.float32
        seen["pool_input"] = list(inputs[0].shape)

    hooks = [model.encoder.layers[-1].register_forward_pre_hook(block_input), model.head.register_forward_pre_hook(pool_input)]
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
            source = fp16(model, pixels)
            backends = attention_nodes(source.grad_fn)
            raw = pair.smoke.compact_head_features(source, head)
            index = torch.tensor((0, 1), device="cuda")
            positives = pair.smoke.member_bank_positive_ordinals(np.asarray(frozen["target"], dtype=np.int64), allow_singletons=True).cuda()
            loss = pair.smoke.sharded_mask_arcface_loss(raw, classifier, torch.tensor(frozen["target"], device="cuda")[index], torch.arange(128, device="cuda").unsqueeze(0), margin=0.3, scale=64)
            loss += 8 * pair.smoke.member_bank_rank_loss(raw, bank, head, positives[index], index, live_head=False)
            assert torch.isfinite(source).all() and torch.isfinite(raw).all() and torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
    operators = sorted({e.key for e in profile.key_averages() if any(s in e.key.lower() for s in ("scaled_dot_product", "bmm", "softmax"))})
    kernels = sorted({e.name for e in profile.events() if "cuda" in str(e.device_type).lower()})
    gradients = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in model.named_parameters() if p.requires_grad}
    extra_gradients = [float(p.grad.norm()) if p.grad is not None else None for p in [*head.parameters(), classifier]]
    pair.smoke.save(args.output / "gradient-observation.json", {"advance": False, "partial_evidence_only": True, "data_gradient_norms": gradients, "head_proxy_gradient_norms": extra_gradients, "backward_nodes": backends, "operator_names": operators, "cuda_kernel_names": kernels, "numerical_flags": flags, "graph": seen, "calibration": calibration, "scale": scaler.get_scale(), "optimizer_updates": 0, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
    assert seen == {"block_input": [2, 256, 1024], "pool_input": [2, 256, 1024]} and torch.equal(source, baseline)
    assert len(gradients) == 16 and all(v is not None and np.isfinite(v) and v > 0 for v in (*gradients.values(), *extra_gradients)), "native FP16 data gradient invalid/zero"
    assert all((p.grad is None) == (not p.requires_grad) and (p.grad is None or torch.isfinite(p.grad).all()) for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in [*head.parameters(), classifier])
    assert backends.get("BmmBackward0", 0) >= 2 and backends.get("SoftmaxBackward0", 0) >= 1 and sum(v for k, v in backends.items() if "ScaledDotProduct" in k) == 1
    assert kernels and any("backward" in n.lower() for n in operators)
    assert pair.smoke.digest(base.whole_state(model)) == info["whole_original_source_sha256"]
    assert pair.smoke.digest(native.frozen_state(model)) == info["frozen_complement_sha256"]
    model.zero_grad(set_to_none=True)
    model.eval()
    parameters = (model.encoder.layers[-1].self_attn.q_proj.bias, model.encoder.layers[-1].mlp.fc2.bias)
    before = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            parameters[0][0].add_(0.01)
            changed_attention = fp16(model, pixels)
            assert not torch.equal(changed_attention, baseline)
            parameters[1][0].add_(0.01)
            changed = fp16(model, pixels)
            assert not torch.equal(changed, changed_attention)
        with TemporaryDirectory(dir=root, prefix="gpu-native-roundtrip-") as tmp:
            path = Path(tmp) / "native.pt"
            torch.save({"vision": model.state_dict(), "head": head.state_dict()}, path)
            saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            loaded = type(model)(copy.deepcopy(model.config)).float().eval()
            loaded.load_state_dict(saved["vision"], strict=True)
            native.freeze(loaded)
            loaded.cuda()
            loaded_head = nn.Linear(1024, 128).eval()
            loaded_head.load_state_dict(saved["head"], strict=True)
            loaded_head.cuda()
            with torch.no_grad():
                assert torch.equal(fp16(loaded, pixels), changed)
            cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
            with native.verify_export(loaded, model) as (encode, scopes):
                a, b = encode(pixels)
                va = F.normalize(pair.smoke.compact_head_features(a, loaded_head).float(), dim=1)
                vb = F.normalize(pair.smoke.compact_head_features(b, head).float(), dim=1)
                assert torch.isfinite(va).all() and torch.equal(va, vb) and torch.equal(a, changed)
            assert len(scopes) == 1 and all(scopes[0][k] for k in ("loaded_amp_inside", "captured_block_amp_inside", "live_amp_inside")) and not scopes[0]["between_encoder_amp_enabled"]
            pa, pb = pack_int8_unit_embeddings(va.cpu()), pack_int8_unit_embeddings(vb.cpu())
            assert np.array_equal(pa.codes, pb.codes) and np.array_equal(pa.inverse_norms, pb.inverse_norms)
            assert torch.equal(cpu_rng, torch.random.get_rng_state()) and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
            assert pair.smoke.digest(native.frozen_state(loaded)) == info["frozen_complement_sha256"]
    finally:
        with torch.no_grad():
            for p, value in zip(parameters, before, strict=True):
                p.copy_(value)
    assert pair.smoke.digest(base.whole_state(model)) == info["whole_original_source_sha256"]
    with torch.no_grad():
        assert torch.equal(fp16(model, pixels), baseline)
    assert json.loads(json.dumps(native.environment(model, processor))) == info["environment"] and numerical_flags() == flags
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert torch.cuda.max_memory_allocated() < 10_000_000_000
    pair.smoke.save(args.output / "preflight.json", {"code": code, "native_cpu_authority_sha256": CPU_SHA, "environment": info["environment"], "inventory": inventory, "whole_original_source_sha256": info["whole_original_source_sha256"], "frozen_complement_sha256": info["frozen_complement_sha256"], "data_gradient_norms": gradients, "head_proxy_gradient_norms": extra_gradients, "backward_nodes": backends, "operator_names": operators, "cuda_kernel_names": kernels, "numerical_flags": flags, "scope_observations": scopes, "graph": seen, "calibration": calibration, "updated_strict400_reload_whole_calibration_suffix_head_packed_exact": True, "original_source_restored": True, "cpu_cuda_rng_preserved_in_verification": True, "optimizer_updates": 0, "held_images": 0, "quality_read": False, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
    print("PASS actual native FP16 final-block graph/backward, strict whole calibration/suffix/head/packed parity and restored source; no updates/held/quality")


if __name__ == "__main__":
    main()
