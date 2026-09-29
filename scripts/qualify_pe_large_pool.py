#!/usr/bin/env python3
"""Actual Large pooling-only CPU gradients and independent native reload check."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pe_large_pool as native
import copy
from pe_core_training import named_training_parameters


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pair = native.pair
    root = Path(__file__).resolve().parent
    init = native.INIT
    output = args.output
    assert not output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, prior = native.control(root)
    authority = frozen
    assert all(pair.sha(root / n) == h for n, h in authority["code"].items())
    assert pair.sha(init / "initializers.npz") == authority["initializers_sha256"]
    model, processor = pair.smoke.load_arm(control, "large")
    assert sum(p.numel() for p in model.parameters()) == 315956224
    assert len(model.state_dict()) == 400
    inventory = native.freeze(model)
    runtime = native.runtime_identity(model)
    original = pair.smoke.digest(native.whole_state(model))
    foreign = pair.smoke.digest(native.frozen_state(model))
    original_environment = native.environment(model, processor)
    head = nn.Linear(1024, 128)
    with np.load(init / "initializers.npz", allow_pickle=False) as a:
        head.load_state_dict(
            {k: torch.from_numpy(a["large.head." + k]) for k in ("weight", "bias")}
        )
        classifier = nn.Parameter(torch.from_numpy(a["large.classifier"].copy()))
        bank = torch.from_numpy(a["large.bank"].copy())
    fit_features = torch.from_numpy(
        np.load(control.cache / "large.fit.npy", allow_pickle=False)
    )
    own_head, own_classifier, pca_sha = pair.smoke.initialize_head_and_classifier(
        fit_features, tuple(frozen["target"]), allow_singletons=True
    )
    assert pca_sha == frozen["initializers"]["large"]["pca_sha256"]
    assert pair.smoke.digest(own_head.state_dict()) == pair.smoke.digest(
        head.state_dict()
    )
    assert torch.equal(own_classifier, classifier)
    own_bank = torch.nn.functional.normalize(
        pair.smoke.compact_head_features(fit_features, own_head).detach(), dim=1
    )
    assert torch.equal(own_bank, bank)
    del fit_features, own_head, own_classifier, own_bank
    images, rgb = pair.augmented_images(
        control.dataset_root, frozen["fit_manifest"], frozen["batches"][0], 1
    )
    first_pixels = pair.pixels(processor, images, "large")
    assert rgb == prior["rgb_sha256"][0]
    assert (
        pair.smoke.digest({"pixels": first_pixels})
        == frozen["initializers"]["large"]["first_pixels_sha256"]
    )
    del first_pixels, images
    images, _ = pair.augmented_images(
        control.dataset_root, frozen["fit_manifest"], (0, 1), None
    )
    pixels = pair.pixels(processor, images, "large")
    with torch.no_grad():
        baseline = model(pixel_values=pixels).pooler_output
    token_inputs = []

    def check_input(module, inputs):
        assert module is model.head
        native.assert_frozen_tokens(inputs)
        assert inputs[0].shape == (2, 256, 1024)
        assert (
            inputs[0].dtype == torch.float32
            and not inputs[0].requires_grad
            and inputs[0].grad_fn is None
        )
        token_inputs.append(list(inputs[0].shape))

    hook = model.head.register_forward_pre_hook(check_input)
    try:
        source = model(pixel_values=pixels).pooler_output
    finally:
        hook.remove()
    assert (
        token_inputs == [[2, 256, 1024]] and native.runtime_identity(model) == runtime
    )
    assert torch.equal(source, baseline)
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
    )
    loss = loss + 8 * pair.smoke.member_bank_rank_loss(
        raw, bank, head, positives[index], index, live_head=False
    )
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, parameter in named_training_parameters(model, "large"):
        assert (parameter.grad is None) == (not parameter.requires_grad), name
        if parameter.requires_grad:
            assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0, (
                name
            )
            gradients[name] = float(parameter.grad.norm())
    assert set(gradients) == set(inventory["trainable"]) and gradients
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
        for p in [*head.parameters(), classifier]
    )
    assert pair.smoke.digest(native.whole_state(model)) == original
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    members = [p for g in optimizer.param_groups for p in g["params"]]
    assert (
        not optimizer.state
        and len(members) == len({id(p) for p in members}) == len(gradients) + 3
    )
    assert {id(p) for p in members} == {
        id(p) for _, p in named_training_parameters(model, "large") if p.requires_grad
    } | {id(p) for p in head.parameters()} | {id(classifier)}
    model.zero_grad(set_to_none=True)
    parameters = (model.head.probe, model.head.mlp.fc2.bias)
    saved = [p.detach().clone() for p in parameters]
    try:
        with torch.no_grad():
            parameters[0].add_(0.01)
            stem_changed = model(pixel_values=pixels).pooler_output
            assert not torch.equal(stem_changed, baseline)
            parameters[1].add_(0.01)
            changed = model(pixel_values=pixels).pooler_output
            assert not torch.equal(changed, stem_changed)
        with tempfile.TemporaryDirectory(
            dir=root, prefix="s16-roundtrip-"
        ) as directory:
            checkpoint = Path(directory) / "native.pt"
            torch.save(
                {
                    "vision": model.state_dict(),
                    "head": head.state_dict(),
                },
                checkpoint,
            )
            state = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            loaded = type(model)(copy.deepcopy(model.config)).float().eval()
            assert state["vision"].keys() == loaded.state_dict().keys()
            loaded.load_state_dict(state["vision"], strict=True)
            loaded_head = nn.Linear(1024, 128).eval()
            loaded_head.load_state_dict(state["head"], strict=True)
            assert pair.smoke.digest(native.whole_state(loaded)) == pair.smoke.digest(
                native.whole_state(model)
            )
            assert pair.smoke.digest(native.frozen_state(loaded)) == foreign
            a, b = native.verified_features(loaded, model, pixels)
            assert torch.equal(a, changed) and torch.equal(a, b)
            assert torch.equal(
                pair.smoke.compact_head_features(a, loaded_head),
                pair.smoke.compact_head_features(b, head),
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
                        raise AssertionError("changed native Large pool accepted")
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
        assert torch.equal(model(pixel_values=pixels).pooler_output, baseline)
    assert native.environment(model, processor) == original_environment
    output.mkdir(exist_ok=False)
    pair.smoke.save(
        output / "preflight.json",
        {
            "code": {
                **authority["code"],
                **pair.smoke.authority(),
                "check_pe_large_pool.py": pair.sha(root / "check_pe_large_pool.py"),
            },
            "environment": original_environment,
            "actual_frozen_pool_input": token_inputs[0],
            "source_state_keys": len(model.state_dict()),
            "initializer_preflight_sha256": pair.sha(init / "preflight.json"),
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
            "native_trainable_parameters": sum(
                p.numel() for p in model.head.parameters()
            ),
            "optimizer_tensors": len(members),
            "initial_native_output_exact": True,
            "updated_native_roundtrip_exact": True,
            "whole_live_and_loaded_forward_exact": True,
            "pool_probe_and_mlp_mismatch_rejected": True,
            "temporary_pool_restored": True,
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "cuda": False,
        },
    )
    print(
        "PASS actual native Large pool gradients/optimizer tensors, independent whole-encoder strict reload/head bits, negative controls/restoration; no updates/quality"
    )


if __name__ == "__main__":
    main()
