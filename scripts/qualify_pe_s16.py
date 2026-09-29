#!/usr/bin/env python3
"""Actual S16 full-encoder CPU gradients and independent native reload check."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pe_s16_training as native
from core.vision_encoder.pe import VisionTransformer
from pe_core_training import named_training_parameters


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--initializer-sha256", required=True)
    args = parser.parse_args()
    pair = native.pair
    root = Path(__file__).resolve().parent
    init = Path("/home/riomus/runs/sfora-pe-s16-init-v1")
    output = Path("/home/riomus/runs/sfora-pe-s16-cpu-v1")
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    assert pair.sha(init / "preflight.json") == args.initializer_sha256
    authority = json.loads((init / "preflight.json").read_text())
    assert all(pair.sha(root / n) == h for n, h in authority["code"].items())
    assert pair.sha(init / "initializers.npz") == authority["initializers_sha256"]
    control, frozen, _ = native.control(root)
    model, processor = native.load()
    inventory = native.freeze(model)
    assert inventory == {
        k: tuple(v) for k, v in authority["initializers"]["inventory"].items()
    }
    runtime = native.runtime_identity(model)
    original = pair.smoke.digest(native.whole_state(model))
    foreign = pair.smoke.digest(native.frozen_state(model))
    assert foreign == authority["initializers"]["frozen_sha256"]
    head = nn.Linear(512, 128)
    with np.load(init / "initializers.npz", allow_pickle=False) as a:
        head.load_state_dict(
            {k: torch.from_numpy(a["pe.head." + k]) for k in ("weight", "bias")}
        )
        classifier = nn.Parameter(torch.from_numpy(a["pe.classifier"].copy()))
        bank = torch.from_numpy(a["pe.bank"].copy())
    images, _ = pair.augmented_images(
        control.dataset_root, frozen["fit_manifest"], (0, 1), None
    )
    pixels = pair.pixels(processor, images, "pe")
    with torch.no_grad():
        baseline = model(pixels)
    source = model(pixels)
    assert torch.equal(source, baseline)
    raw = native.compact_features(source, head)
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
    )
    loss = loss + 8 * pair.smoke.member_bank_rank_loss(
        raw, bank, head, positives[index], index, live_head=False
    )
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, parameter in named_training_parameters(model, "pe"):
        assert (parameter.grad is None) == (not parameter.requires_grad), name
        if parameter.requires_grad:
            assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0, (
                name
            )
            gradients[name] = float(parameter.grad.norm())
    assert len(gradients) == 163
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        for p in [*head.parameters(), classifier]
    )
    assert pair.smoke.digest(native.whole_state(model)) == original
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    members = [p for g in optimizer.param_groups for p in g["params"]]
    assert not optimizer.state and len(members) == len({id(p) for p in members}) == 166
    assert {id(p) for p in members} == {
        id(p) for _, p in named_training_parameters(model, "pe") if p.requires_grad
    } | {id(p) for p in head.parameters()} | {id(classifier)}
    model.zero_grad(set_to_none=True)
    parameters = (model.conv1.weight, model.attn_pool.probe)
    saved = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            parameters[0].view(-1)[0].add_(0.01)
            stem_changed = model(pixels)
            assert not torch.equal(stem_changed, baseline)
            parameters[1].add_(0.01)
            changed = model(pixels)
            assert not torch.equal(changed, stem_changed)
        with tempfile.TemporaryDirectory(
            dir=root, prefix="s16-roundtrip-"
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
            state = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            loaded = (
                VisionTransformer.from_config("PE-Core-S16-384", pretrained=False)
                .float()
                .eval()
            )
            assert (
                len(state["vision"]) == 163
                and state["vision"].keys() == loaded.state_dict().keys()
            )
            loaded.load_state_dict(state["vision"], strict=True)
            loaded.rope.rope.load_state_dict(state["rope"], strict=True)
            loaded.rope.update_grid(torch.device("cpu"), 14, 14)
            loaded_head = nn.Linear(512, 128).eval()
            loaded_head.load_state_dict(state["head"], strict=True)
            assert torch.equal(loaded.rope.freq, state["rope_freq"])
            assert pair.smoke.digest(native.whole_state(loaded)) == pair.smoke.digest(
                native.whole_state(model)
            )
            assert pair.smoke.digest(native.frozen_state(loaded)) == foreign
            a, b = native.verified_features(loaded, model, pixels)
            assert torch.equal(a, changed) and torch.equal(a, b)
            assert torch.equal(
                native.compact_features(a, loaded_head),
                native.compact_features(b, head),
            )
            for parameter in parameters:
                negative = parameter.detach().clone()
                try:
                    with torch.no_grad():
                        parameter.add_(0.01)
                    try:
                        native.verified_features(loaded, model, pixels)
                    except AssertionError:
                        pass
                    else:
                        raise AssertionError("changed whole native encoder accepted")
                finally:
                    with torch.no_grad():
                        parameter.copy_(negative)
            assert (
                native.runtime_identity(loaded)
                == native.runtime_identity(model)
                == runtime
            )
    finally:
        with torch.no_grad():
            for parameter, value in zip(parameters, saved, strict=True):
                parameter.copy_(value)
    assert pair.smoke.digest(native.whole_state(model)) == original
    assert pair.smoke.digest(native.frozen_state(model)) == foreign
    with torch.no_grad():
        assert torch.equal(model(pixels), baseline)
    output.mkdir(exist_ok=False)
    pair.smoke.save(
        output / "preflight.json",
        {
            "code": {**authority["code"], **pair.smoke.authority()},
            "initializer_preflight_sha256": args.initializer_sha256,
            "initializers_sha256": authority["initializers_sha256"],
            "inventory": inventory,
            "whole_original_source_sha256": original,
            "frozen_complement_sha256": foreign,
            "runtime_identity": runtime,
            "actual_native_gradient_norms": gradients,
            "actual_native_head_proxy_gradient_norms": [
                float(p.grad.norm()) for p in [*head.parameters(), classifier]
            ],
            "actual_native_loss": float(loss.detach()),
            "native_trainable_tensors": len(gradients),
            "native_trainable_parameters": sum(p.numel() for p in model.parameters()),
            "optimizer_tensors": len(members),
            "initial_native_output_exact": True,
            "updated_native_roundtrip_exact": True,
            "whole_live_and_loaded_forward_exact": True,
            "stem_and_pool_mismatch_rejected": True,
            "temporary_stem_and_pool_restored": True,
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "cuda": False,
        },
    )
    print(
        "PASS actual full-native S16 163 gradients/166 optimizer tensors, independent whole-encoder strict reload/head bits, negative controls/restoration; no updates/quality"
    )


if __name__ == "__main__":
    main()
