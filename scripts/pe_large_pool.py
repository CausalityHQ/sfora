#!/usr/bin/env python3
"""Fixed native SigLIP2 Large pooling-only allocation and original authority."""

import inspect
import importlib.metadata
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

import train_inshop_pe_pair as pair

INIT = Path("/home/riomus/runs/sfora-pe-augmented-100-v2")


def control(root):
    args = SimpleNamespace(
        root=root,
        output=INIT,
        preflight_sha256="41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293",
        cache=Path("/home/riomus/runs/sfora-pe-fullfit-cache-v3"),
        mechanics_dir=Path("/home/riomus/runs/sfora-pe-fp16-smoke-v1"),
        dataset_root=Path("/home/riomus/datasets/inshop_official_standard"),
        large_snapshot=Path(
            "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"
        ),
    )
    frozen = pair.check_startup(args)
    assert (
        pair.sha(INIT / "receipt.json")
        == "d02a1c176f023e81bd9a0bb526ff3294e677e88dcc0724b9306e1ca884177dbb"
    )
    prior = json.loads((INIT / "large.json").read_text())
    assert prior == json.loads((INIT / "receipt.json").read_text())["arms"]["large"]
    return args, frozen, prior


def freeze(model):
    assert model.config.hidden_size == 1024 and model.config.num_hidden_layers == 24
    assert model.config.image_size == 256 and len(model.encoder.layers) == 24
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    inventory = {
        key: tuple(n for n, p in model.named_parameters() if p.requires_grad == active)
        for key, active in (("frozen", False), ("trainable", True))
    }
    assert set(inventory["trainable"]) == {
        "head." + n for n, _ in model.head.named_parameters()
    }
    return inventory


def whole_state(model):
    return {
        **model.state_dict(),
        **{"runtime." + n: v for n, v in model.named_buffers()},
    }


def frozen_state(model):
    return {
        n: v
        for n, v in whole_state(model).items()
        if not n.startswith(("head.", "runtime.head."))
    }


def assert_frozen_tokens(inputs):
    assert len(inputs) == 1
    tokens = inputs[0]
    assert tokens.ndim == 3 and tokens.shape[1:] == (256, 1024)
    assert (
        tokens.dtype == torch.float32
        and not tokens.requires_grad
        and tokens.grad_fn is None
    )


def runtime_identity(model):
    assert model.config.hidden_size == 1024 and model.config.num_hidden_layers == 24
    assert (
        model.config.image_size == 256
        and model.use_head
        and len(model.encoder.layers) == 24
    )
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
    return {
        "config": model.config.to_dict(),
        "attention_implementation": model.config._attn_implementation,
        "checkpointing": model.is_gradient_checkpointing,
        "pool_heads": model.head.attention.num_heads,
        "normalization_eps": {
            n: m.eps
            for n, m in model.named_modules()
            if isinstance(m, torch.nn.LayerNorm)
        },
        "dropout": {
            n: m.p for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)
        },
        "buffer_devices": {n: str(v.device) for n, v in model.named_buffers()},
    }


def environment(model, processor):
    return {
        "classes": {
            "vision": type(model).__module__ + "." + type(model).__name__,
            "pool": type(model.head).__module__ + "." + type(model.head).__name__,
            "processor": type(processor).__module__ + "." + type(processor).__name__,
        },
        "versions": {
            n: importlib.metadata.version(n)
            for n in (
                "torch",
                "torchvision",
                "transformers",
                "Pillow",
                "numpy",
                "safetensors",
            )
        },
        "native_files": {
            inspect.getfile(cls): pair.sha(Path(inspect.getfile(cls)))
            for cls in (
                type(model),
                type(model.head),
                type(processor),
                torch.nn.MultiheadAttention,
            )
        },
        "processor": processor.to_dict(),
    }


@torch.no_grad()
def verified_features(loaded, live, pixels):
    assert loaded is not live and not loaded.training and not live.training
    assert runtime_identity(loaded) == runtime_identity(live)
    assert not torch.is_autocast_enabled("cuda")
    with (
        torch.autocast("cuda", dtype=torch.float16) if pixels.is_cuda else nullcontext()
    ):
        a = loaded(pixel_values=pixels).pooler_output.float()
    with (
        torch.autocast("cuda", dtype=torch.float16) if pixels.is_cuda else nullcontext()
    ):
        b = live(pixel_values=pixels).pooler_output.float()
    assert torch.equal(a, b), "whole live/strict native Large encoder differs"
    return a, b
