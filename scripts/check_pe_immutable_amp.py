#!/usr/bin/env python3
"""CPU native-operator parity, original-state restoration and mutation rejection."""

import torch
from torch import nn
from pe_immutable_amp import build_cache, cached_call


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.position = nn.Parameter(torch.randn(1, 1, 4))
        self.norm = nn.LayerNorm(4)
        self.linear = nn.Linear(4, 4)
        self.attention = nn.MultiheadAttention(4, 2, batch_first=True)
        self.probe = nn.Parameter(torch.randn(1, 1, 4))
        self.fail = False

    def forward(self, pixel_values):
        tokens = self.conv(pixel_values).flatten(2).transpose(1, 2) + self.position
        tokens = self.norm(tokens)
        assert tokens.dtype == torch.float32
        tokens = self.linear(tokens).float()
        pooled = self.attention(self.probe.expand(tokens.shape[0], -1, -1), tokens, tokens)[0]
        if self.fail:
            raise RuntimeError("forward failure")
        return self.probe + pooled


def main():
    assert not torch.cuda.is_available()
    torch.manual_seed(179032)
    model = Fixture().eval()
    state = {n: p.clone() for n, p in model.state_dict().items()}
    cache = build_cache(model)
    assert set(cache[0]) == {"conv.weight", "conv.bias", "linear.weight", "linear.bias", "attention.in_proj_weight", "attention.in_proj_bias", "attention.out_proj.weight", "attention.out_proj.bias"}
    with torch.no_grad():
        for count in (1, 3):
            x = torch.randn(count, 3, 2, 2)
            with torch.autocast("cpu", dtype=torch.float16):
                eager = model(x)
            assert torch.equal(eager, cached_call(model, cache, x))
        model.fail = True
        try:
            cached_call(model, cache, x)
        except RuntimeError as error:
            assert str(error) == "forward failure"
        else:
            raise AssertionError("forward failure swallowed")
        model.fail = False
        assert all(p.dtype == torch.float32 and torch.equal(p, state[n]) for n, p in model.state_dict().items())
        for expected in ("source changed", "cache changed"):
            target = model.probe if expected == "source changed" else cache[0]["conv.weight"]
            original = target.clone()
            target.add_(0.01)
            try:
                cached_call(model, cache, x)
            except AssertionError as error:
                assert str(error) == expected
            else:
                raise AssertionError("mutated source/cache accepted")
            target.copy_(original)
            cache = build_cache(model)
        model.train()
        try:
            cached_call(model, cache, x)
        except AssertionError:
            pass
        else:
            raise AssertionError("training accepted")
        model.eval()
    print("PASS CPU FP16 autocast Conv/Linear/MHA parity, F32 norm/probe, exception restoration and source/cache mutation rejection")


if __name__ == "__main__":
    main()
