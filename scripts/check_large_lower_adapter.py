#!/usr/bin/env python3
"""Small CPU regression for the proposed frozen-prefix adapter."""

import torch
from torch import nn

from large_lower_adapter import install, merge


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(16, 16)
        self.self_attn.v_proj = nn.Linear(16, 16)


class Vision(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(Block() for _ in range(24))


def main():
    torch.manual_seed(7)
    vision = Vision()
    for layer in vision.encoder.layers[:12]:
        layer.requires_grad_(False)
    baseline = {k: v.clone() for k, v in vision.state_dict().items()}
    x = torch.randn(4, 16)
    original_output = vision.encoder.layers[0].self_attn.q_proj(x).detach()
    sites = install(vision, rank=8, seed=179032, width=16)
    assert len(sites) == 24
    assert all(torch.equal(vision.state_dict()[f"encoder.layers.{i}.self_attn.q_proj.parametrizations.weight.original"], baseline[f"encoder.layers.{i}.self_attn.q_proj.weight"]) for i in range(12))
    assert torch.equal(vision.encoder.layers[0].self_attn.q_proj(x), original_output)
    optimizer = torch.optim.AdamW([p for p in vision.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    vision.encoder.layers[0].self_attn.q_proj(x).square().sum().backward()
    first = sites[0][0].parametrizations.weight[0]
    assert first.B.grad is not None and first.B.grad.norm() > 0
    assert first.A.grad is not None and not first.A.grad.any()
    optimizer.step()
    changed_output = vision.encoder.layers[0].self_attn.q_proj(x).detach()
    assert not torch.equal(changed_output, original_output)
    assert all(torch.equal(getattr(vision.encoder.layers[i].self_attn.q_proj.parametrizations.weight, "original"), baseline[f"encoder.layers.{i}.self_attn.q_proj.weight"]) for i in range(12))
    merge(sites)
    assert set(vision.state_dict()) == set(baseline)
    assert torch.equal(vision.encoder.layers[0].self_attn.q_proj(x), changed_output)
    print("PASS lower q/v identity, gradients, frozen originals and strict merged keys")


if __name__ == "__main__":
    main()
