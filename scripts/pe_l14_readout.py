#!/usr/bin/env python3
"""Frozen native L14, own initialized supervised compact readout only."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pe_l14_training as l14
import train_inshop_pe_pair as pair
from pe_core_training import named_training_parameters

INIT = Path("/home/riomus/runs/sfora-pe-l14-init-v1")
INIT_SHA = "be5a8bdcfd1749340f220d5669cebfb1c0d8b30d3c08667a0b72a4fb38701277"


def freeze(model):
    model.requires_grad_(False)
    model.rope.rope.requires_grad_(False)
    return {
        "frozen": tuple(
            n for n, p in named_training_parameters(model, "pe") if not p.requires_grad
        ),
        "trainable": tuple(
            n for n, p in named_training_parameters(model, "pe") if p.requires_grad
        ),
    }


def frozen_state(model):
    values = dict(model.state_dict())
    values.update(
        ("rope.rope." + n, v) for n, v in model.rope.rope.state_dict().items()
    )
    values["rope.freq"] = model.rope.freq
    return values


def preflight(root, output):
    assert not torch.cuda.is_available()
    args, frozen, prior = l14.control(root)
    assert pair.sha(INIT / "preflight.json") == INIT_SHA
    initial = json.loads((INIT / "preflight.json").read_text())
    assert all(pair.sha(root / n) == h for n, h in initial["code"].items())
    assert pair.sha(INIT / "initializers.npz") == initial["initializers_sha256"]
    model, processor = l14.load()
    inventory = freeze(model)
    assert not inventory["trainable"] and len(inventory["frozen"]) == 308
    original = pair.smoke.digest(frozen_state(model))
    head = nn.Linear(1024, 128)
    with np.load(INIT / "initializers.npz", allow_pickle=False) as a:
        head.load_state_dict(
            {k: torch.from_numpy(a["pe.head." + k]) for k in ("weight", "bias")}
        )
        classifier = nn.Parameter(torch.from_numpy(a["pe.classifier"].copy()))
        bank = torch.from_numpy(a["pe.bank"].copy())
    images, _ = pair.augmented_images(
        args.dataset_root, frozen["fit_manifest"], (0, 1), None
    )
    with torch.no_grad():
        source = model(pair.pixels(processor, images, "pe"))
    raw = pair.smoke.compact_head_features(source, head)
    idx = torch.tensor((0, 1))
    positives = pair.smoke.member_bank_positive_ordinals(
        np.asarray(frozen["target"], dtype=np.int64), allow_singletons=True
    )
    loss = pair.smoke.sharded_mask_arcface_loss(
        raw,
        classifier,
        torch.tensor(frozen["target"])[idx],
        torch.arange(128).unsqueeze(0),
        margin=0.3,
        scale=64,
    ) + 8 * pair.smoke.member_bank_rank_loss(
        raw, bank, head, positives[idx], idx, live_head=False
    )
    params = list(head.parameters()) + [classifier]
    loss.backward()
    assert torch.isfinite(loss) and all(
        torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in params
    )
    assert all(
        p.grad is None and not p.requires_grad
        for _, p in named_training_parameters(model, "pe")
    )
    assert pair.smoke.digest(frozen_state(model)) == original
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    assert (
        len(params) == 3 and len({id(p) for p in params}) == 3 and not optimizer.state
    )
    images, rgb = pair.augmented_images(
        args.dataset_root, frozen["fit_manifest"], frozen["batches"][0], 1
    )
    pixels = pair.pixels(processor, images, "pe")
    assert (
        rgb == prior["rgb_sha256"][0]
        and pair.smoke.digest({"pixels": pixels})
        == initial["initializers"]["first_pixels_sha256"]
    )
    output.mkdir(exist_ok=False)
    pair.smoke.save(
        output / "preflight.json",
        {
            "code": {
                **pair.smoke.authority(),
                **{
                    n: pair.sha(root / n)
                    for n in ("check_pe_l14_readout.py", "run_pe_l14_readout.sh")
                },
            },
            "original_initializer_preflight_sha256": INIT_SHA,
            "initializers_sha256": initial["initializers_sha256"],
            "inventory": inventory,
            "whole_source_frozen_sha256": original,
            "first_rgb_sha256": rgb,
            "first_pixels_sha256": pair.smoke.digest({"pixels": pixels}),
            "native_frozen_parameters": sum(
                p.numel() for _, p in named_training_parameters(model, "pe")
            ),
            "trainable_readout_parameters": sum(p.numel() for p in params),
            "trainable_tensors": 3,
            "actual_native_loss": float(loss.detach()),
            "actual_native_head_proxy_gradient_norms": [
                float(p.grad.norm()) for p in params
            ],
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "cuda": False,
        },
    )
    print(
        "PASS actual native frozen source/foreign state/no encoder gradient, own head/proxy data gradients/disjoint coverage and matched augmentedRGB224"
    )
