#!/usr/bin/env python3
"""Fixed native L14 half-prefix inventory and malformed-alias rejection."""

from types import SimpleNamespace
import torch
from torch import nn
from pe_l14_training import freeze, frozen_state
from pe_core_training import named_training_parameters


def main():
    model = nn.Module()
    model.width, model.layers = 1024, 24
    model.conv1, model.ln_pre, model.ln_post = (
        nn.Linear(2, 2),
        nn.Linear(2, 2),
        nn.Linear(2, 2),
    )
    model.class_embedding = nn.Parameter(torch.zeros(2))
    model.positional_embedding = nn.Parameter(torch.zeros(2, 2))
    model.transformer = nn.Module()
    model.transformer.resblocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(24)])
    model.rope = SimpleNamespace(rope=nn.Linear(2, 2), freq=torch.arange(4))
    for b in model.transformer.resblocks:
        b.attn = SimpleNamespace(rope=model.rope)
    inventory = freeze(model)
    expected = tuple(
        n
        for n, p in named_training_parameters(model, "pe")
        if n.startswith(
            ("conv1.", "ln_pre.", "rope.", "class_embedding", "positional_embedding")
        )
        or any(n.startswith(f"transformer.resblocks.{i}.") for i in range(12))
    )
    assert inventory["frozen"] == expected
    assert set(inventory["frozen"]).isdisjoint(inventory["trainable"])
    assert set(inventory["frozen"]) | set(inventory["trainable"]) == set(
        dict(named_training_parameters(model, "pe"))
    )
    state = frozen_state(model, inventory)
    assert set(expected) <= set(state) and torch.equal(
        state["rope.freq"], model.rope.freq
    )
    for bad in ("width", "layers", "alias"):
        if bad == "alias":
            model.transformer.resblocks[0].attn.rope = None
        else:
            setattr(model, bad, 1)
        try:
            freeze(model)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad {bad} accepted")
        model.width, model.layers = 1024, 24
        model.transformer.resblocks[0].attn.rope = model.rope
    print(
        "PASS exact half12 L14 frozen/trainable inventory and wrong width/depth/rotary rejection"
    )


if __name__ == "__main__":
    main()
