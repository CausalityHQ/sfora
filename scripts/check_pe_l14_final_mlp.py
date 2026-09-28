#!/usr/bin/env python3
"""Learned native suffix, immutable prefix, and mismatch/hook cleanup checks."""

import copy
from types import SimpleNamespace

import torch
from torch import nn

from pe_l14_final_mlp import freeze, frozen_state, verified_features


class Block(nn.Module):
    def __init__(self, width, output):
        super().__init__()
        assert width == output
        self.attn = nn.Linear(width, width)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 8), nn.GELU(), nn.Linear(8, width))

    def forward(self, x, attn_mask=None):
        assert attn_mask is None
        x = x + self.attn(x)
        return x + self.mlp(self.ln_2(x))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = 4
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([Block(4, 4) for _ in range(24)])
        self.attn_pool = nn.Linear(4, 4)
        self.ln_post = nn.LayerNorm(4)
        self.proj = nn.Parameter(torch.eye(4))
        self.rope = SimpleNamespace(rope=nn.Linear(2, 2), freq=torch.arange(4))

    def _pool(self, x):
        return self.attn_pool(x).squeeze(1)

    def forward(self, x):
        for block in self.transformer.resblocks:
            x = block(x, attn_mask=None)
        return self._pool(self.ln_post(x)) @ self.proj


def main():
    model = Toy()
    roles = freeze(model)
    expected = {
        "transformer.resblocks.23.mlp." + n
        for n, _ in model.transformer.resblocks[-1].mlp.named_parameters()
    } | {"attn_pool." + n for n, _ in model.attn_pool.named_parameters()}
    assert set(roles["trainable"]) == expected
    original = {n: t.clone() for n, t in frozen_state(model).items()}
    images = torch.randn(2, 1, 4)
    model(images).square().sum().backward()
    assert all((p.grad is None) == (not p.requires_grad) for p in model.parameters())
    assert all(p.grad.norm() > 0 for p in model.parameters() if p.requires_grad)
    assert all(torch.equal(t, original[n]) for n, t in frozen_state(model).items())
    loaded = copy.deepcopy(model).eval()
    model.eval()
    source, live = verified_features(loaded, model, images)
    assert torch.equal(source, loaded(images)) and torch.equal(source, live)
    for parameter in (
        model.transformer.resblocks[-1].mlp[-1].bias[:1],
        model.attn_pool.bias,
    ):
        saved = parameter.detach().clone()
        with torch.no_grad():
            parameter.add_(1)
        try:
            verified_features(loaded, model, images)
        except AssertionError:
            pass
        else:
            raise AssertionError("changed learned native suffix accepted")
        assert not loaded.transformer.resblocks[-1]._forward_hooks
        with torch.no_grad():
            parameter.copy_(saved)
    print(
        "PASS final-MLP/pool gradients, frozen prefix, exact suffix, mismatch rejection/hook cleanup"
    )


if __name__ == "__main__":
    main()
