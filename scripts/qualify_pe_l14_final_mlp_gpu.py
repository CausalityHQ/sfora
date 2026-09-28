#!/usr/bin/env python3
"""Actual original L14 final-MLP/pool gradients and strict native boundary FP16 GPU parity."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pe_l14_final_mlp as pool
import pe_l14_readout as readout
import pe_l14_training as l14
import train_inshop_pe_pair as pair
from pe_core_training import named_training_parameters
from core.vision_encoder.pe import VisionTransformer
from train_pe_l14_readout import cuda_budget


def sdpa_nodes(root):
    seen, pending, nodes = set(), [root], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if "ScaledDotProduct" in type(node).__name__:
            nodes.add(node)
        pending.extend(child for child, _ in node.next_functions)
    return nodes


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cpu-preflight-sha256", required=True)
    parser.add_argument("--execution-sha256", required=True)
    options = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = Path("/home/riomus/runs/sfora-pe-l14-final-mlp-gpu-v1")
    assert not output.exists()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    cpu = Path("/home/riomus/runs/sfora-pe-l14-final-mlp-cpu-v1/preflight.json")
    assert pair.sha(cpu) == options.cpu_preflight_sha256
    authority = json.loads(cpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in authority["code"].items())
    execution = root / "final-mlp-gpu-execution.json"
    assert pair.sha(execution) == options.execution_sha256
    code = json.loads(execution.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items())
    args, frozen, _ = l14.control(root)
    assert pair.sha(readout.INIT / "preflight.json") == readout.INIT_SHA
    initial = json.loads((readout.INIT / "preflight.json").read_text())
    assert pair.sha(readout.INIT / "initializers.npz") == initial["initializers_sha256"]
    model, processor = l14.load()
    assert model.pool_type == "attn" and model.attn_pool.probe.shape == (1, 1, 1024)
    images, _ = pair.augmented_images(
        args.dataset_root, frozen["fit_manifest"], (0, 1), None
    )
    pixels = pair.pixels(processor, images, "pe")
    inventory = pool.freeze(model)
    assert set(inventory["trainable"]) == (
        {"attn_pool." + n for n, _ in model.attn_pool.named_parameters()}
        | {
            "transformer.resblocks.23.mlp." + n
            for n, _ in model.transformer.resblocks[-1].mlp.named_parameters()
        }
    )
    original = pair.smoke.digest(pool.frozen_state(model))
    whole = pair.smoke.digest(readout.frozen_state(model))
    assert original == authority["frozen_complement_sha256"]
    assert whole == authority["whole_original_source_sha256"]
    assert torch.cuda.is_available()
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    model.cuda().train()
    pixels = pixels.cuda()
    with torch.no_grad():
        fp32 = model(pixels).float()
        with torch.autocast("cuda", dtype=torch.float16):
            baseline = model(pixels).float()
    cached = torch.from_numpy(
        np.load(l14.CACHE / "l14.fit.npy", mmap_mode="r")[:2].copy()
    ).cuda()
    calibration = {
        "fp16_fp32": torch.nn.functional.cosine_similarity(baseline, fp32).tolist(),
        "cached_fresh": torch.nn.functional.cosine_similarity(
            cached, baseline
        ).tolist(),
    }
    print(json.dumps({"calibration": calibration}), flush=True)
    assert all(min(values) >= 0.999 for values in calibration.values())
    cuda_budget("calibrated-source")
    head = nn.Linear(1024, 128)
    with np.load(readout.INIT / "initializers.npz", allow_pickle=False) as a:
        head.load_state_dict(
            {k: torch.from_numpy(a["pe.head." + k]) for k in ("weight", "bias")}
        )
        classifier = nn.Parameter(torch.from_numpy(a["pe.classifier"].copy()))
        bank = torch.from_numpy(a["pe.bank"].copy())
    head.cuda()
    classifier = nn.Parameter(classifier.detach().cuda())
    bank = bank.cuda()
    runtime = pool.runtime_identity(model)
    token_inputs = []

    def check_tokens(module, inputs):
        pool.assert_frozen_tokens(module, inputs)
        assert inputs[0].shape == (2, 257, 1024)
        token_inputs.append(
            {"shape": list(inputs[0].shape), "dtype": str(inputs[0].dtype)}
        )

    pool_graph = []
    attention_nodes = {}

    def check_pool_graph(module, inputs):
        assert inputs[0].requires_grad and inputs[0].grad_fn is not None
        pool_graph.append(True)

    def record_attention(name):
        def record(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if name == "final_self_attention":
                assert not value.requires_grad and value.grad_fn is None
            attention_nodes[name] = sdpa_nodes(value.grad_fn)

        return record

    observers = [
        model.transformer.resblocks[-1].mlp.register_forward_pre_hook(check_tokens),
        model.attn_pool.register_forward_pre_hook(check_pool_graph),
        model.transformer.resblocks[-1].attn.register_forward_hook(
            record_attention("final_self_attention")
        ),
        model.attn_pool.attn.register_forward_hook(record_attention("pool_attention")),
    ]
    guard = model.transformer.resblocks[-1].register_forward_pre_hook(check_tokens)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            source = model(pixels).float()
    finally:
        guard.remove()
        for observer in observers:
            observer.remove()
    assert pool_graph == [True]
    assert len(token_inputs) == 2 and pool.runtime_identity(model) == runtime
    assert torch.equal(source, baseline), "initial native output changed with pool role"
    raw = pair.smoke.compact_head_features(source, head)
    index = torch.tensor((0, 1), device="cuda")
    positives = pair.smoke.member_bank_positive_ordinals(
        np.asarray(frozen["target"], dtype=np.int64), allow_singletons=True
    )
    positives = positives.cuda()
    loss = pair.smoke.sharded_mask_arcface_loss(
        raw,
        classifier,
        torch.tensor(frozen["target"], device="cuda")[index],
        torch.arange(128, device="cuda").unsqueeze(0),
        margin=0.3,
        scale=64,
    ) + 8 * pair.smoke.member_bank_rank_loss(
        raw, bank, head, positives[index], index, live_head=False
    )
    backend_nodes = {
        "final_self_attention": sorted(
            type(n).__name__ for n in attention_nodes["final_self_attention"]
        ),
        "pool_attention": sorted(
            type(n).__name__
            for n in attention_nodes["pool_attention"]
            - attention_nodes["final_self_attention"]
        ),
    }
    assert (
        backend_nodes["final_self_attention"] == []
        and len(backend_nodes["pool_attention"]) == 1
    )
    print(json.dumps({"attributed_sdpa_autograd_nodes": backend_nodes}), flush=True)
    _, scaler = pair.smoke.training_precision("fp16", device="cuda")
    assert scaler.get_scale() == 128
    scaler.scale(loss).backward()
    cuda_budget("actual-pool-backward")
    gradients = {}
    for name, parameter in named_training_parameters(model, "pe"):
        assert (parameter.grad is None) == (not parameter.requires_grad), name
        if parameter.requires_grad:
            assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0, (
                name
            )
            gradients[name] = float(parameter.grad.norm())
    assert all(
        torch.isfinite(p.grad).all() and p.grad.norm() > 0
        for p in [*head.parameters(), classifier]
    )
    assert pair.smoke.digest(pool.frozen_state(model)) == original
    assert pair.smoke.digest(readout.frozen_state(model)) == whole
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for _, p in named_training_parameters(model, "pe")
                    if p.requires_grad
                ],
                "lr": 1e-5,
            },
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    members = [p for group in optimizer.param_groups for p in group["params"]]
    assert not optimizer.state and len({id(p) for p in members}) == len(members)
    assert {id(p) for p in members} == {
        id(p) for _, p in named_training_parameters(model, "pe") if p.requires_grad
    } | {id(p) for p in head.parameters()} | {id(classifier)}
    model.eval()
    runtime = pool.runtime_identity(model)
    saved_probe = model.attn_pool.probe.detach().clone()
    saved_bias = model.transformer.resblocks[-1].mlp.c_proj.bias[0].detach().clone()
    with torch.no_grad():
        model.transformer.resblocks[-1].mlp.c_proj.bias[0].add_(0.01)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        block_changed = model(pixels).float()
    assert not torch.equal(block_changed, baseline)
    with torch.no_grad():
        model.attn_pool.probe.add_(0.01)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        changed = model(pixels).float()
    assert not torch.equal(changed, block_changed)
    assert not torch.equal(changed, baseline)
    with tempfile.TemporaryDirectory(
        dir=root, prefix="final-mlp-roundtrip-"
    ) as directory:
        checkpoint = Path(directory) / "native.pt"
        torch.save(
            {
                "vision": model.state_dict(),
                "rope": model.rope.rope.state_dict(),
                "rope_freq": model.rope.freq,
                "head": head.state_dict(),
            },
            checkpoint,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        loaded = (
            VisionTransformer.from_config("PE-Core-L14-336", pretrained=False)
            .float()
            .eval()
        )
        loaded.load_state_dict(state["vision"], strict=True)
        loaded.rope.rope.load_state_dict(state["rope"], strict=True)
        loaded.rope.update_grid(torch.device("cpu"), 16, 16)
        loaded_head = nn.Linear(1024, 128).eval()
        loaded_head.load_state_dict(state["head"], strict=True)
        assert torch.equal(loaded.rope.freq, state["rope_freq"])
        loaded.cuda()
        loaded_head.cuda()
        cuda_budget("strict-native-reload")
        assert pair.smoke.digest(readout.frozen_state(loaded)) == pair.smoke.digest(
            readout.frozen_state(model)
        )
        assert pair.smoke.digest(pool.frozen_state(loaded)) == original
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            assert torch.equal(loaded(pixels).float(), changed)
        assert pool.runtime_identity(loaded) == runtime
        with torch.autocast("cuda", dtype=torch.float16):
            loaded_source, live_source = pool.verified_features(loaded, model, pixels)
        assert torch.equal(loaded_source, changed) and torch.equal(
            loaded_source, live_source
        )
        assert torch.equal(
            torch.nn.functional.normalize(
                pair.smoke.compact_head_features(loaded_source, loaded_head).float(),
                dim=1,
            ),
            torch.nn.functional.normalize(
                pair.smoke.compact_head_features(live_source, head).float(), dim=1
            ),
        )
        for parameter in (
            model.transformer.resblocks[-1].mlp.c_proj.bias[0],
            model.attn_pool.probe,
        ):
            negative = parameter.detach().clone()
            with torch.no_grad():
                parameter.add_(0.01)
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    pool.verified_features(loaded, model, pixels)
            except AssertionError:
                pass
            else:
                raise AssertionError("changed actual FP16 native suffix accepted")
            finally:
                with torch.no_grad():
                    parameter.copy_(negative)
            assert not loaded.transformer.resblocks[-1]._forward_hooks
        assert not loaded.transformer.resblocks[-1]._forward_hooks
        assert pool.runtime_identity(loaded) == pool.runtime_identity(model) == runtime
    with torch.no_grad():
        model.attn_pool.probe.copy_(saved_probe)
        model.transformer.resblocks[-1].mlp.c_proj.bias[0].copy_(saved_bias)
    assert pair.smoke.digest(readout.frozen_state(model)) == whole
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        restored = model(pixels).float()
    assert torch.equal(restored, baseline) and not optimizer.state
    cuda_budget("restored-native-state")
    output.mkdir(exist_ok=False)
    pair.smoke.save(
        output / "preflight.json",
        {
            "code": {
                **authority["code"],
                **{n: pair.sha(root / n) for n in ("qualify_pe_l14_final_mlp_gpu.py",)},
            },
            "native_cpu_authority_sha256": pair.sha(cpu),
            "qualification_execution_sha256": options.execution_sha256,
            "calibration": calibration,
            "attributed_sdpa_autograd_nodes": backend_nodes,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "initializers_sha256": initial["initializers_sha256"],
            "inventory": inventory,
            "frozen_complement_sha256": original,
            "whole_original_source_sha256": whole,
            "actual_native_gradient_norms": gradients,
            "runtime_identity": runtime,
            "actual_frozen_token_input": token_inputs[0],
            "final_mlp_trainable_parameters": sum(
                p.numel() for p in model.transformer.resblocks[-1].mlp.parameters()
            ),
            "native_trainable_tensors": len(gradients),
            "pool_trainable_parameters": sum(
                p.numel() for p in model.attn_pool.parameters()
            ),
            "initial_native_output_exact": True,
            "updated_native_roundtrip_exact": True,
            "shared_actual_final_block_suffix_exact": True,
            "hook_removed": True,
            "temporary_mlp_and_pool_restored": True,
            "mlp_and_pool_mismatch_rejected": True,
            "mlp_perturbation_observable": True,
            "pool_perturbation_observable": True,
            "pool_input_has_graph": True,
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "cuda": True,
        },
    )
    print(
        "PASS actual FP16 GPU native final-MLP/pool gradients, frozen complement, output identity, perturbed strict roundtrip/boundary/head parity; no updates/quality"
    )


if __name__ == "__main__":
    main()
