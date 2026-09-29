#!/usr/bin/env python3
"""Two-block-only roles preserve the original pooling operator and reject bad geometry."""

from types import SimpleNamespace

import torch
from torch import nn

import pe_large_final_two as native


def main():
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=1024, num_hidden_layers=24, image_size=256)
    model.embeddings = nn.Linear(2, 2)
    model.encoder = nn.Module()
    model.encoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(24)])
    model.post_layernorm = nn.LayerNorm(2)
    model.head = nn.Linear(2, 2)
    inventory = native.freeze(model)
    assert inventory["trainable"] == ("encoder.layers.22.weight", "encoder.layers.22.bias", "encoder.layers.23.weight", "encoder.layers.23.bias")
    assert all(p.requires_grad == n.startswith(("encoder.layers.22.", "encoder.layers.23.")) for n, p in model.named_parameters())
    state = native.frozen_state(model)
    assert "head.weight" in state and "head.bias" in state
    assert not any(n.startswith(("encoder.layers.22.", "encoder.layers.23.")) for n in state)
    assert set(inventory["frozen"]) <= set(state)
    model.encoder.layers[-1](torch.ones(1, 2)).sum().backward()
    assert model.encoder.layers[-1].weight.grad is not None
    assert model.head.weight.grad is None and model.encoder.layers[0].weight.grad is None
    for name, value in (("hidden_size", 1), ("num_hidden_layers", 12), ("image_size", 224)):
        original = getattr(model.config, name)
        setattr(model.config, name, value)
        try:
            native.freeze(model)
        except AssertionError:
            pass
        else:
            raise AssertionError("bad native geometry accepted")
        setattr(model.config, name, original)
    print("PASS exact final-two-block roles, pretrained pool in frozen complement and bad geometry rejection; CPU fixture only")


if __name__ == "__main__":
    main()
