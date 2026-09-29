#!/usr/bin/env python3
"""Fixed S16 source, full native encoder and fit-only initialization."""

import json
import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import export_pe_s16_fit as fit
import train_inshop_pe_pair as pair
from core.vision_encoder.pe import VisionTransformer
from core.vision_encoder.transforms import get_image_transform
from pe_core_authority import visual_state
from pe_core_training import named_training_parameters
from sfora.representation_ceiling import fit_centered_pca

CACHE = Path("/home/riomus/runs/sfora-pe-s16-fit-v1")
CACHE_SHA = "214b951de7656cbf83485b16adecd61509ad5c19da4af7e60ffddc0755d35665"
AUDIT_SHA = "f74de562c3538821389a5e76aa3fddb27cf8c731be288839b160d1f26b79c6ff"


def initialize(features, labels):
    if (
        features.ndim != 2
        or features.shape != (len(labels), 512)
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or not torch.isfinite(features).all()
    ):
        raise ValueError("S16 initialization inventory differs")
    normalized = F.normalize(features, dim=1)
    names = tuple(sorted(set(labels)))
    indexes = {label: index for index, label in enumerate(names)}
    pca = fit_centered_pca(normalized, dimensions=128)
    head = nn.Linear(512, 128)
    with torch.no_grad():
        head.weight.copy_(pca.components)
        head.bias.copy_(-(pca.components @ pca.mean))
    projected = pca.apply(normalized)
    sums = torch.zeros(len(names), 128)
    counts = torch.zeros(len(names), dtype=torch.int64)
    for row, label in enumerate(labels):
        sums[indexes[label]] += projected[row]
        counts[indexes[label]] += 1
    assert counts.min() >= 1
    classifier = nn.Parameter(F.normalize(sums, dim=1))
    pca_sha = hashlib.sha256(
        pca.mean.numpy().tobytes() + pca.components.numpy().tobytes()
    ).hexdigest()
    return head, classifier, pca_sha


def compact_features(source, head):
    if (
        type(source) is not torch.Tensor
        or source.ndim != 2
        or source.shape[1] != 512
        or not source.is_floating_point()
        or not torch.isfinite(source).all()
        or (torch.linalg.vector_norm(source.float(), dim=1) == 0).any()
        or type(head) is not nn.Linear
        or head.in_features != 512
        or head.out_features != 128
        or source.device != head.weight.device
    ):
        raise ValueError("S16 compact source geometry differs")
    with torch.autocast(device_type=source.device.type, enabled=False):
        return head(F.normalize(source.float(), dim=1))


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
    assert pair.sha(CACHE / "s16.fit.npy") == cache["features_sha256"]
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
        VisionTransformer.from_config("PE-Core-S16-384", pretrained=False)
        .float()
        .eval()
    )
    assert state.keys() == model.state_dict().keys() and len(state) == 163
    model.load_state_dict(state, strict=True)
    assert pair.smoke.digest(model.state_dict()) == source["source_state_sha256"]
    assert sum(p.numel() for p in model.parameters()) == 23782656
    model.rope.update_grid(torch.device("cpu"), 14, 14)
    assert (
        pair.smoke.digest(model.rope.rope.state_dict()) == source["foreign_rope_sha256"]
    )
    assert pair.smoke.digest({"freq": model.rope.freq}) == source["foreign_grid_sha256"]
    return model, get_image_transform(224)


def freeze(model):
    assert model.layers == 12 and model.width == 384
    assert all(b.attn.rope is model.rope for b in model.transformer.resblocks)
    model.requires_grad_(True)
    model.rope.rope.requires_grad_(False)
    inventory = {
        k: tuple(
            n
            for n, p in named_training_parameters(model, "pe")
            if p.requires_grad == active
        )
        for k, active in (("frozen", False), ("trainable", True))
    }
    assert set(inventory["trainable"]) == set(dict(model.named_parameters()))
    assert len(inventory["trainable"]) == 163
    assert all(n.startswith("rope.rope.") for n in inventory["frozen"])
    return inventory


def frozen_state(model, inventory=None):
    values = {"rope.rope." + n: v for n, v in model.rope.rope.state_dict().items()}
    values.update(("rope.runtime." + n, v) for n, v in model.rope.rope.named_buffers())
    values["rope.freq"] = model.rope.freq
    return values


def whole_state(model):
    return {**model.state_dict(), **frozen_state(model)}


def runtime_identity(model):
    assert model.layers == 12 and model.width == 384 and model.pool_type == "attn"
    assert model.rope.grid_size == (14, 14)
    assert all(b.attn.rope is model.rope for b in model.transformer.resblocks)
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    return {
        "native_config": [
            model.image_size,
            model.layers,
            model.width,
            model.patch_size,
            model.posemb_grid_size,
            model.use_cls_token,
            model.use_abs_posemb,
            model.use_rope2d,
            model.pool_type,
        ],
        "pool_heads": model.attn_pool.num_heads,
        "grid": model.rope.grid_size,
        "frequency_device": str(model.rope.freq.device),
        "foreign_buffer_devices": {
            n: str(v.device) for n, v in model.rope.rope.named_buffers()
        },
    }


@torch.no_grad()
def verified_features(loaded, live, images):
    assert loaded is not live and not loaded.training and not live.training
    assert runtime_identity(loaded) == runtime_identity(live)
    assert not torch.is_autocast_enabled("cuda")
    with (
        torch.autocast("cuda", dtype=torch.float16) if images.is_cuda else nullcontext()
    ):
        a = loaded(images).float()
    with (
        torch.autocast("cuda", dtype=torch.float16) if images.is_cuda else nullcontext()
    ):
        b = live(images).float()
    assert torch.equal(a, b), "whole live/strict native encoder differs"
    return a, b


def preflight(root, output):
    assert not torch.cuda.is_available()
    args, frozen, prior = control(root)
    source = torch.from_numpy(np.load(CACHE / "s16.fit.npy", allow_pickle=False))
    head, classifier, pca_sha = initialize(source, tuple(frozen["target"]))
    bank = F.normalize(compact_features(source, head).detach(), dim=1)
    assert torch.isfinite(bank).all() and (bank.norm(dim=1) > 0).all()
    index = torch.tensor(frozen["batches"][frozen["rank_active"].index(True)])
    query = source[index].clone().requires_grad_()
    raw = compact_features(query, head)
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
    pair.smoke.save(
        output / "preflight.json",
        {
            "code": pair.smoke.authority(),
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
        "PASS actual native S16 full encoder inventory/source/foreign rotary, own full-fit initialization/objective, disjoint AdamW coverage and matched RGB224; no optimizer update"
    )
