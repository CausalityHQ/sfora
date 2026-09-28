#!/usr/bin/env python3
"""Actual original L14 pool gradients and strict native boundary CPU parity."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pe_l14_final_block as pool
import pe_l14_readout as readout
import pe_l14_training as l14
import train_inshop_pe_pair as pair
from pe_core_training import named_training_parameters
from core.vision_encoder.pe import VisionTransformer


def main():
    root = Path(__file__).resolve().parent
    output = Path("/home/riomus/runs/sfora-pe-l14-final-block-cpu-v2")
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    cpu = Path("/home/riomus/runs/sfora-pe-l14-native-pool-cpu-v2/preflight.json")
    assert (
        pair.sha(cpu)
        == "1f11069e3675890b75b474095c01a135078c346cc02e69979823ccc60899d154"
    )
    authority = json.loads(cpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in authority["code"].items())
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
    with torch.no_grad():
        baseline = model(pixels)
    inventory = pool.freeze(model)
    assert set(inventory["trainable"]) == (
        {"attn_pool." + n for n, _ in model.attn_pool.named_parameters()}
        | {
            "transformer.resblocks.23." + n
            for n, _ in model.transformer.resblocks[-1].named_parameters()
        }
    )
    original = pair.smoke.digest(pool.frozen_state(model))
    whole = pair.smoke.digest(readout.frozen_state(model))
    head = nn.Linear(1024, 128)
    with np.load(readout.INIT / "initializers.npz", allow_pickle=False) as a:
        head.load_state_dict(
            {k: torch.from_numpy(a["pe.head." + k]) for k in ("weight", "bias")}
        )
        classifier = nn.Parameter(torch.from_numpy(a["pe.classifier"].copy()))
        bank = torch.from_numpy(a["pe.bank"].copy())
    runtime = pool.runtime_identity(model)
    token_inputs = []

    def check_tokens(module, inputs):
        pool.assert_frozen_tokens(module, inputs)
        assert inputs[0].shape == (2, 257, 1024)
        token_inputs.append(
            {"shape": list(inputs[0].shape), "dtype": str(inputs[0].dtype)}
        )

    pool_graph = []

    def check_pool_graph(module, inputs):
        assert inputs[0].requires_grad and inputs[0].grad_fn is not None
        pool_graph.append(True)

    pool_guard = model.attn_pool.register_forward_pre_hook(check_pool_graph)
    guard = model.transformer.resblocks[-1].register_forward_pre_hook(check_tokens)
    try:
        source = model(pixels)
    finally:
        guard.remove()
        pool_guard.remove()
    assert pool_graph == [True]
    assert len(token_inputs) == 1 and pool.runtime_identity(model) == runtime
    assert torch.equal(source, baseline), "initial native output changed with pool role"
    raw = pair.smoke.compact_head_features(source, head)
    index = torch.tensor((0, 1))
    positives = pair.smoke.member_bank_positive_ordinals(
        np.asarray(frozen["target"], dtype=np.int64), allow_singletons=True
    )
    loss = pair.smoke.sharded_mask_arcface_loss(
        raw,
        classifier,
        torch.tensor(frozen["target"])[index],
        torch.arange(128).unsqueeze(0),
        margin=0.3,
        scale=64,
    ) + 8 * pair.smoke.member_bank_rank_loss(
        raw, bank, head, positives[index], index, live_head=False
    )
    loss.backward()
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
    saved_probe = model.attn_pool.probe.detach().clone()
    saved_bias = model.transformer.resblocks[-1].ln_2.bias.detach().clone()
    with torch.no_grad():
        model.transformer.resblocks[-1].ln_2.bias.add_(0.01)
        block_changed = model(pixels)
        assert not torch.equal(block_changed, baseline)
        model.attn_pool.probe.add_(0.01)
        changed = model(pixels)
        assert not torch.equal(changed, block_changed)
    assert not torch.equal(changed, baseline)
    with tempfile.TemporaryDirectory(
        dir=root, prefix="final-block-roundtrip-"
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
        assert pair.smoke.digest(readout.frozen_state(loaded)) == pair.smoke.digest(
            readout.frozen_state(model)
        )
        assert pair.smoke.digest(pool.frozen_state(loaded)) == original
        loaded_source, live_source = pool.verified_features(loaded, model, pixels)
        assert torch.equal(loaded_source, changed) and torch.equal(
            loaded_source, live_source
        )
        assert torch.equal(
            pair.smoke.compact_head_features(loaded_source, loaded_head),
            pair.smoke.compact_head_features(live_source, head),
        )
        for parameter in (
            model.transformer.resblocks[-1].ln_2.bias,
            model.attn_pool.probe,
        ):
            negative = parameter.detach().clone()
            with torch.no_grad():
                parameter.add_(0.01)
            try:
                pool.verified_features(loaded, model, pixels)
            except AssertionError:
                pass
            else:
                raise AssertionError("changed actual native suffix accepted")
            finally:
                with torch.no_grad():
                    parameter.copy_(negative)
            assert not loaded.transformer.resblocks[-1]._forward_hooks
        assert not loaded.transformer.resblocks[-1]._forward_hooks
        assert pool.runtime_identity(loaded) == pool.runtime_identity(model) == runtime
    with torch.no_grad():
        model.attn_pool.probe.copy_(saved_probe)
        model.transformer.resblocks[-1].ln_2.bias.copy_(saved_bias)
    assert pair.smoke.digest(readout.frozen_state(model)) == whole
    with torch.no_grad():
        assert torch.equal(model(pixels), baseline)
    output.mkdir(exist_ok=False)
    pair.smoke.save(
        output / "preflight.json",
        {
            "code": {
                **authority["code"],
                **{
                    n: pair.sha(root / n)
                    for n in (
                        "pe_l14_native_pool.py",
                        "pe_l14_final_block.py",
                        "qualify_pe_l14_final_block.py",
                        "check_pe_l14_final_block.py",
                    )
                },
            },
            "pool_cpu_authority_sha256": pair.sha(cpu),
            "initializers_sha256": initial["initializers_sha256"],
            "inventory": inventory,
            "frozen_complement_sha256": original,
            "whole_original_source_sha256": whole,
            "actual_native_gradient_norms": gradients,
            "runtime_identity": runtime,
            "actual_frozen_token_input": token_inputs[0],
            "final_block_trainable_parameters": sum(
                p.numel() for p in model.transformer.resblocks[-1].parameters()
            ),
            "native_trainable_tensors": len(gradients),
            "pool_trainable_parameters": sum(
                p.numel() for p in model.attn_pool.parameters()
            ),
            "initial_native_output_exact": True,
            "updated_native_roundtrip_exact": True,
            "shared_actual_final_block_suffix_exact": True,
            "hook_removed": True,
            "block_and_pool_mismatch_rejected": True,
            "pool_input_has_graph": True,
            "temporary_block_and_pool_restored": True,
            "block_perturbation_observable": True,
            "pool_perturbation_observable": True,
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "cuda": False,
        },
    )
    print(
        "PASS actual native final-block/pool gradients, frozen complement, output identity, perturbed strict roundtrip/boundary/head parity; no updates/quality"
    )


if __name__ == "__main__":
    main()
