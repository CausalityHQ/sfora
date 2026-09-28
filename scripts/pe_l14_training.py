#!/usr/bin/env python3
"""Fixed L14 source, native half12 prefix and fit-only initialization."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import export_pe_l14_fit as fit
import train_inshop_pe_pair as pair
from core.vision_encoder.pe import VisionTransformer
from core.vision_encoder.transforms import get_image_transform
from pe_core_authority import visual_state
from pe_core_training import named_training_parameters

CACHE = Path("/home/riomus/runs/sfora-pe-l14-fit-v1")
CACHE_SHA = "9f9a1c6e91740cbd8c80fd5e90c7f0e631e000e6a031eb1276e94ab11f8a870c"
AUDIT_SHA = "c19ca161820bdda4c316b839a37b4d6daa9b229b2585c7382c2038e420309819"


def control(root):
    args = SimpleNamespace(
        root=root,
        output=Path("/home/riomus/runs/sfora-pe-augmented-100-v2"),
        preflight_sha256="41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293",
        cache=Path("/home/riomus/runs/sfora-pe-fullfit-cache-v3"),
        mechanics_dir=Path("/home/riomus/runs/sfora-pe-fp16-smoke-v1"),
        dataset_root=fit.DATA,
        large_snapshot=Path(
            "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"
        ),
    )
    frozen = pair.check_startup(args)
    assert (
        pair.sha(args.output / "receipt.json")
        == "d02a1c176f023e81bd9a0bb526ff3294e677e88dcc0724b9306e1ca884177dbb"
    )
    prior = json.loads((args.output / "pe.json").read_text())
    assert prior == json.loads((args.output / "receipt.json").read_text())["arms"]["pe"]
    assert (
        pair.sha(CACHE / "receipt.json") == CACHE_SHA
        and pair.sha(CACHE / "cpu-audit.json") == AUDIT_SHA
    )
    cache = json.loads((CACHE / "receipt.json").read_text())
    assert cache["pass"] and not cache["quality_read"]
    assert pair.sha(CACHE / "preflight.json") == cache["preflight_sha256"]
    fp = json.loads((CACHE / "preflight.json").read_text())
    assert fp["manifest"] == frozen["fit_manifest"]
    assert all(pair.sha(root / n) == h for n, h in fp["code"].items())
    assert pair.sha(CACHE / "l14.fit.npy") == cache["features_sha256"]
    fit.source_authority()
    return args, frozen, prior


def load():
    source = fit.source_authority()
    state = visual_state(
        torch.load(
            source["checkpoint"], weights_only=True, map_location="cpu", mmap=True
        )
    )
    model = (
        VisionTransformer.from_config("PE-Core-L14-336", pretrained=False)
        .float()
        .eval()
    )
    assert state.keys() == model.state_dict().keys() and len(state) == 307
    model.load_state_dict(state, strict=True)
    assert pair.smoke.digest(model.state_dict()) == source["source_state_sha256"]
    assert sum(p.numel() for p in model.parameters()) == 317151232
    model.rope.update_grid(torch.device("cpu"), 16, 16)
    assert (
        pair.smoke.digest(model.rope.rope.state_dict()) == source["foreign_rope_sha256"]
    )
    assert pair.smoke.digest({"freq": model.rope.freq}) == source["foreign_grid_sha256"]
    return model, get_image_transform(224)


def freeze(model):
    blocks = model.transformer.resblocks
    if len(blocks) != 24 or model.layers != 24 or model.width != 1024:
        raise ValueError("L14 native depth/width differs")
    if any(b.attn.rope is not model.rope for b in blocks):
        raise ValueError("L14 native rotary aliases differ")
    model.requires_grad_(True)
    for module in (model.conv1, model.ln_pre, model.rope.rope, *blocks[:12]):
        module.requires_grad_(False)
    model.class_embedding.requires_grad_(False)
    model.positional_embedding.requires_grad_(False)
    return {
        k: tuple(
            n
            for n, p in named_training_parameters(model, "pe")
            if p.requires_grad == active
        )
        for k, active in (("frozen", False), ("trainable", True))
    }


def frozen_state(model, inventory):
    roots = ("conv1.", "ln_pre.") + tuple(
        f"transformer.resblocks.{i}." for i in range(12)
    )
    values = {
        n: v
        for n, v in model.state_dict().items()
        if n in inventory["frozen"] or n.startswith(roots)
    }
    values.update(
        ("rope.rope." + n, v) for n, v in model.rope.rope.state_dict().items()
    )
    values["rope.freq"] = model.rope.freq
    return values


def preflight(root, output):
    assert not torch.cuda.is_available()
    args, frozen, prior = control(root)
    source = torch.from_numpy(np.load(CACHE / "l14.fit.npy", allow_pickle=False))
    head, classifier, pca_sha = pair.smoke.initialize_head_and_classifier(
        source, tuple(frozen["target"]), allow_singletons=True
    )
    bank = F.normalize(pair.smoke.compact_head_features(source, head).detach(), dim=1)
    assert torch.isfinite(bank).all() and (bank.norm(dim=1) > 0).all()
    index = torch.tensor(frozen["batches"][frozen["rank_active"].index(True)])
    query = source[index].clone().requires_grad_()
    raw = pair.smoke.compact_head_features(query, head)
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
    gradients = torch.autograd.grad(loss, (query, head.weight, head.bias, classifier))
    assert torch.isfinite(loss) and all(
        torch.isfinite(g).all() and g.norm() > 0 for g in gradients
    )
    info = {
        "pca_sha256": pca_sha,
        "cached_objective": float(loss.detach()),
        "cached_gradient_norms": [float(g.norm()) for g in gradients],
        "head_sha256": pair.smoke.digest(head.state_dict()),
        "classifier_sha256": pair.smoke.digest({"classifier": classifier}),
        "bank_sha256": pair.smoke.digest({"bank": bank}),
    }
    output.mkdir(exist_ok=False)
    with (output / "initializers.npz").open("xb") as stream:
        np.savez(
            stream,
            **{
                "pe.head.weight": head.weight.detach().numpy(),
                "pe.head.bias": head.bias.detach().numpy(),
                "pe.classifier": classifier.detach().numpy(),
                "pe.bank": bank.numpy(),
            },
        )
    del source, bank, gradients, loss, raw, query, positives
    model, processor = load()
    inventory = freeze(model)
    images, rgb_sha = pair.augmented_images(
        args.dataset_root, frozen["fit_manifest"], frozen["batches"][0], 1
    )
    assert rgb_sha == prior["rgb_sha256"][0]
    x = pair.pixels(processor, images, "pe")
    assert x.shape == (64, 3, 224, 224)
    info.update(
        {
            "inventory": inventory,
            "frozen_sha256": pair.smoke.digest(frozen_state(model, inventory)),
            "first_rgb_sha256": rgb_sha,
            "first_pixels_sha256": pair.smoke.digest({"pixels": x}),
            "trainable_parameters": sum(
                p.numel()
                for _, p in named_training_parameters(model, "pe")
                if p.requires_grad
            ),
            "frozen_parameters": sum(
                p.numel()
                for _, p in named_training_parameters(model, "pe")
                if not p.requires_grad
            ),
        }
    )
    params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(head.parameters())
        + [classifier]
    )
    assert len({id(p) for p in params}) == len(params)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    assert not optimizer.state
    extra = {
        n: pair.sha(root / n)
        for n in ("check_pe_l14_training.py", "run_pe_l14_training.sh")
    }
    pair.smoke.save(
        output / "preflight.json",
        {
            "code": {**pair.smoke.authority(), **extra},
            "initializers_sha256": pair.sha(output / "initializers.npz"),
            "initializers": info,
            "cache_receipt_sha256": CACHE_SHA,
            "cache_audit_sha256": AUDIT_SHA,
            "optimizer_updates": 0,
            "held_images": 0,
            "cuda": False,
            "quality_read": False,
        },
    )
    print(
        "PASS actual native L14 half12 inventory/source/foreign rotary, own full-fit initialization/objective, disjoint AdamW coverage and matched RGB224; no optimizer update"
    )
