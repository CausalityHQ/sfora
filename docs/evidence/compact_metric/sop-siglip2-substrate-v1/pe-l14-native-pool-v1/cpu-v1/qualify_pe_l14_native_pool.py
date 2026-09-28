#!/usr/bin/env python3
"""Actual original L14 pool gradients and strict native boundary CPU parity."""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pe_l14_native_pool as pool
import pe_l14_readout as readout
import pe_l14_training as l14
import train_inshop_pe_pair as pair
from pe_core_training import named_training_parameters
from core.vision_encoder.pe import VisionTransformer


def main():
    root = Path(__file__).resolve().parent
    output = Path("/home/riomus/runs/sfora-pe-l14-native-pool-cpu-v1")
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    cpu = Path("/home/riomus/runs/sfora-pe-l14-prefetch-cpu-v1/preflight.json")
    assert (
        pair.sha(cpu)
        == "234d7ece495c74a294ff18e3cc7f12cf301244d6cc322783d8e26a9db0e2c273"
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
    assert set(inventory["trainable"]) == {
        "attn_pool." + n for n, _ in model.attn_pool.named_parameters()
    }
    original = pair.smoke.digest(pool.frozen_state(model))
    whole = pair.smoke.digest(readout.frozen_state(model))
    head = nn.Linear(1024, 128)
    with np.load(readout.INIT / "initializers.npz", allow_pickle=False) as a:
        head.load_state_dict(
            {k: torch.from_numpy(a["pe.head." + k]) for k in ("weight", "bias")}
        )
        classifier = nn.Parameter(torch.from_numpy(a["pe.classifier"].copy()))
        bank = torch.from_numpy(a["pe.bank"].copy())
    source = model(pixels)
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
            {"params": model.attn_pool.parameters(), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    members = [p for group in optimizer.param_groups for p in group["params"]]
    assert not optimizer.state and len({id(p) for p in members}) == len(members)
    model.eval()
    saved_probe = model.attn_pool.probe.detach().clone()
    with torch.no_grad():
        model.attn_pool.probe.add_(0.01)
        changed = model(pixels)
    assert not torch.equal(changed, baseline)
    with tempfile.TemporaryDirectory(dir=root, prefix="pool-roundtrip-") as directory:
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
        assert not loaded.attn_pool._forward_hooks
    with torch.no_grad():
        model.attn_pool.probe.copy_(saved_probe)
    assert pair.smoke.digest(readout.frozen_state(model)) == whole
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
                        "qualify_pe_l14_native_pool.py",
                        "check_pe_l14_native_pool.py",
                    )
                },
            },
            "input_cpu_authority_sha256": pair.sha(cpu),
            "initializers_sha256": initial["initializers_sha256"],
            "inventory": inventory,
            "frozen_complement_sha256": original,
            "whole_original_source_sha256": whole,
            "actual_pool_gradient_norms": gradients,
            "pool_trainable_parameters": sum(
                p.numel() for p in model.attn_pool.parameters()
            ),
            "initial_native_output_exact": True,
            "updated_native_roundtrip_exact": True,
            "shared_actual_pool_boundary_exact": True,
            "hook_removed": True,
            "temporary_pool_restored": True,
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "cuda": False,
        },
    )
    print(
        "PASS actual native pool gradients, frozen complement, output identity, perturbed strict roundtrip/boundary/head parity; no updates/quality"
    )


if __name__ == "__main__":
    main()
