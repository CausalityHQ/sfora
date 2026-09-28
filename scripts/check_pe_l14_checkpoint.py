#!/usr/bin/env python3
"""Checkpoint frozen-input parameter gradients and preserve native state/eval."""

import torch
from torch import nn
from pe_l14_checkpoint import wrap_block


def main():
    torch.manual_seed(179032)
    block = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    before = {n: p.detach().clone() for n, p in block.state_dict().items()}
    baseline = block(x)
    grads = torch.autograd.grad(baseline.square().sum(), tuple(block.parameters()))
    wrap_block(block)
    candidate = block(x)
    loaded = torch.autograd.grad(candidate.square().sum(), tuple(block.parameters()))
    assert not x.requires_grad and torch.equal(baseline, candidate)
    assert all(torch.equal(a, b) for a, b in zip(grads, loaded, strict=True))
    assert all(torch.equal(p, before[n]) for n, p in block.state_dict().items())
    with torch.no_grad():
        assert torch.equal(block(x), baseline)
    block.eval()
    assert torch.equal(block(x), baseline)
    try:
        wrap_block(block)
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate wrapping accepted")
    print(
        "PASS frozen-input exact output/parameter gradients, unchanged state/eval and duplicate rejection"
    )


if __name__ == "__main__":
    main()
