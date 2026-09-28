#!/usr/bin/env python3
"""Pool-only gradients, exact boundary replay, and hook cleanup on failure."""

import copy
from types import SimpleNamespace

import torch
from torch import nn

from pe_l14_native_pool import freeze, frozen_state, verified_features


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = 4
        self.attn_pool = nn.Linear(4, 4)
        self.proj = nn.Parameter(torch.eye(4))
        self.rope = SimpleNamespace(rope=nn.Linear(2, 2), freq=torch.arange(4))

    def forward(self, tokens):
        return self.attn_pool(tokens).squeeze(1) @ self.proj


def main():
    model = Toy()
    inventory = freeze(model)
    assert set(inventory["trainable"]) == {"attn_pool.weight", "attn_pool.bias"}
    original = {n: t.clone() for n, t in frozen_state(model).items()}
    images = torch.randn(2, 1, 4)
    model(images).square().sum().backward()
    assert model.proj.grad is None
    assert all(
        p.grad is not None and p.grad.norm() > 0 for p in model.attn_pool.parameters()
    )
    assert all(torch.equal(t, original[n]) for n, t in frozen_state(model).items())
    loaded = copy.deepcopy(model).eval()
    model.eval()
    expected = loaded(images)
    source, live = verified_features(loaded, model, images)
    assert torch.equal(source, expected) and torch.equal(source, live)
    assert not loaded.attn_pool._forward_hooks
    with torch.no_grad():
        model.attn_pool.bias.add_(1)
    try:
        verified_features(loaded, model, images)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed native pool accepted")
    assert not loaded.attn_pool._forward_hooks
    print(
        "PASS pool-only gradients, frozen complement, exact replay, changed pool rejection/hook cleanup"
    )


if __name__ == "__main__":
    main()
