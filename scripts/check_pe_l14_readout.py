#!/usr/bin/env python3
"""Readout-only role: native source/foreign frozen, no encoder gradients."""

from types import SimpleNamespace
import torch
from torch import nn
from pe_l14_readout import freeze, frozen_state
from pe_core_training import named_training_parameters


def main():
    source = nn.Linear(4, 4)
    source.rope = SimpleNamespace(rope=nn.Linear(2, 2), freq=torch.arange(4))
    inventory = freeze(source)
    assert not inventory["trainable"] and len(inventory["frozen"]) == 4
    original = {n: t.clone() for n, t in frozen_state(source).items()}
    head = nn.Linear(4, 2)
    with torch.no_grad():
        values = source(torch.randn(2, 4))
    head(values).square().sum().backward()
    assert all(
        p.grad is None and not p.requires_grad
        for _, p in named_training_parameters(source, "pe")
    )
    assert all(p.grad is not None and p.grad.norm() > 0 for p in head.parameters())
    assert all(torch.equal(t, original[n]) for n, t in frozen_state(source).items())
    print(
        "PASS frozen native/foreign role, live readout gradient and unchanged source state"
    )


if __name__ == "__main__":
    main()
