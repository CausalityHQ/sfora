#!/usr/bin/env python3
"""Exact native Large pooling-only roles and malformed-geometry rejection."""

from types import SimpleNamespace

import torch
from torch import nn

from pe_large_pool import freeze, frozen_state, assert_frozen_tokens


def main():
    model = nn.Module()
    model.config = SimpleNamespace(
        hidden_size=1024, num_hidden_layers=24, image_size=256
    )
    model.embeddings = nn.Linear(2, 2)
    model.encoder = nn.Module()
    model.encoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(24)])
    model.post_layernorm = nn.LayerNorm(2)
    model.head = nn.Linear(2, 2)
    inventory = freeze(model)
    assert inventory["trainable"] == ("head.weight", "head.bias")
    assert all(
        p.requires_grad == n.startswith("head.") for n, p in model.named_parameters()
    )
    assert set(inventory["frozen"]) <= set(frozen_state(model))
    assert not any(n.startswith("head.") for n in frozen_state(model))
    original = model.embeddings.weight.detach().clone()
    model.head.weight.sum().backward()
    assert model.head.weight.grad is not None and model.embeddings.weight.grad is None
    assert torch.equal(model.embeddings.weight, original)
    tokens = torch.randn(2, 256, 1024)
    assert_frozen_tokens((tokens,))
    for bad in ((tokens, None), (tokens.half(),), (tokens.clone().requires_grad_(),)):
        try:
            assert_frozen_tokens(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong native input boundary accepted")
    for field, value in (
        ("hidden_size", 1),
        ("image_size", 224),
        ("num_hidden_layers", 12),
    ):
        saved = getattr(model.config, field)
        setattr(model.config, field, value)
        try:
            freeze(model)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong native geometry accepted")
        setattr(model.config, field, saved)
    print(
        "PASS exact head-only gradient/frozen complement and malformed native Large geometry"
    )


if __name__ == "__main__":
    main()
