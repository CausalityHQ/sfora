#!/usr/bin/env python3
"""Native PE/Large prefix authority, shared by matched training probes."""

from types import SimpleNamespace

import torch
from torch import nn


def named_training_parameters(vision, arm):
    yield from vision.named_parameters()
    if arm == "pe":
        yield from (
            ("rope.rope." + name, value) for name, value in vision.rope.rope.named_parameters()
        )


def freeze_prefix(vision, arm):
    if arm == "pe":
        blocks = vision.transformer.resblocks
        if len(blocks) != 12 or vision.width != 768 or vision.layers != 12:
            raise ValueError("PE native depth/width differs")
        if any(block.attn.rope is not vision.rope for block in blocks):
            raise ValueError("PE rotary aliases differ")
        vision.requires_grad_(True)
        for module in (vision.conv1, vision.ln_pre, vision.rope.rope, *blocks[:6]):
            module.requires_grad_(False)
        vision.class_embedding.requires_grad_(False)
        vision.positional_embedding.requires_grad_(False)
    elif arm == "large":
        blocks = vision.encoder.layers
        if len(blocks) != 24:
            raise ValueError("Large native depth differs")
        vision.requires_grad_(True)
        for module in (vision.embeddings, *blocks[:12]):
            module.requires_grad_(False)
    else:
        raise ValueError("unsupported training source")
    return {
        "frozen": tuple(
            name
            for name, value in named_training_parameters(vision, arm)
            if not value.requires_grad
        ),
        "trainable": tuple(
            name for name, value in named_training_parameters(vision, arm) if value.requires_grad
        ),
    }


def frozen_state(vision, arm, inventory):
    if arm == "pe":
        roots = ("conv1.", "ln_pre.") + tuple(f"transformer.resblocks.{i}." for i in range(6))
    elif arm == "large":
        roots = ("embeddings.",) + tuple(f"encoder.layers.{i}." for i in range(12))
    else:
        raise ValueError("unsupported training source")
    values = {
        name: value
        for name, value in vision.state_dict().items()
        if name in inventory["frozen"] or name.startswith(roots)
    }
    if arm == "pe":
        values.update(
            ("rope.rope." + name, value) for name, value in vision.rope.rope.state_dict().items()
        )
        if getattr(vision.rope, "freq", None) is not None:
            values["rope.freq"] = vision.rope.freq
    return values


def self_test():
    for arm, depth, frozen_blocks in [("pe", 12, 6), ("large", 24, 12)]:
        model = nn.Module()
        blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(depth)])
        if arm == "pe":
            model.width, model.layers = 768, depth
            model.conv1, model.ln_pre = (
                nn.Linear(2, 2),
                nn.Linear(2, 2),
            )
            model.rope = SimpleNamespace(rope=nn.Linear(2, 2), freq=torch.arange(4))
            for block in blocks:
                block.attn = SimpleNamespace(rope=model.rope)
            model.class_embedding = nn.Parameter(torch.zeros(2))
            model.positional_embedding = nn.Parameter(torch.zeros(2, 2))
            model.transformer = nn.Module()
            model.transformer.resblocks = blocks
            model.ln_post, model.attn_pool = nn.Linear(2, 2), nn.Linear(2, 2)
            model.proj = nn.Parameter(torch.ones(2, 2))
            prefixes = ("conv1.", "ln_pre.", "rope.", "class_embedding", "positional_embedding")
            block_prefix = "transformer.resblocks."
        else:
            model.embeddings = nn.Linear(2, 2)
            model.encoder = nn.Module()
            model.encoder.layers = blocks
            model.head = nn.Linear(2, 2)
            prefixes, block_prefix = ("embeddings.",), "encoder.layers."
        inventory = freeze_prefix(model, arm)
        expected = tuple(
            name
            for name, _ in named_training_parameters(model, arm)
            if name.startswith(prefixes)
            or any(name.startswith(block_prefix + str(i) + ".") for i in range(frozen_blocks))
        )
        assert tuple(inventory["frozen"]) == expected
        assert set(inventory["trainable"]) | set(inventory["frozen"]) == set(
            dict(named_training_parameters(model, arm))
        )
        assert not set(inventory["trainable"]) & set(inventory["frozen"])
        for name, value in named_training_parameters(model, arm):
            assert value.requires_grad == (name not in expected)
        assert set(expected).issubset(frozen_state(model, arm, inventory))
        if arm == "pe":
            assert torch.equal(frozen_state(model, arm, inventory)["rope.freq"], model.rope.freq)
            blocks[0].attn.rope = None
            try:
                freeze_prefix(model, arm)
            except ValueError:
                pass
            else:
                raise AssertionError("wrong rotary alias accepted")
            blocks[0].attn.rope = model.rope
        blocks.append(nn.Linear(2, 2))
        try:
            freeze_prefix(model, arm)
        except ValueError:
            pass
        else:
            raise AssertionError("wrong depth accepted")
    print(
        "PASS exact native stem/pre-LN/embedding/block freeze inventories and wrong-depth rejection"
    )


if __name__ == "__main__":
    self_test()
