#!/usr/bin/env python3
"""Native upper-block activation replay; state keys and parameters stay native."""

from functools import wraps

import torch
from torch.utils.checkpoint import checkpoint


def wrap_block(block):
    assert not getattr(block, "_sfora_checkpointed", False)
    native = block.forward

    @wraps(native)
    def forward(*args, **kwargs):
        if block.training and torch.is_grad_enabled():
            return checkpoint(
                native, *args, use_reentrant=False, preserve_rng_state=True, **kwargs
            )
        return native(*args, **kwargs)

    block.forward = forward
    block._sfora_checkpointed = True


def enable(model):
    blocks = model.transformer.resblocks
    assert len(blocks) == 24 and model.width == 1024 and model.layers == 24
    assert not model.transformer.grad_checkpointing
    assert all(not p.requires_grad for block in blocks[:12] for p in block.parameters())
    assert all(p.requires_grad for block in blocks[12:] for p in block.parameters())
    assert all(not getattr(b, "_sfora_checkpointed", False) for b in blocks)
    assert all(b.attn.rope is model.rope for b in blocks)
    for block in blocks[12:]:
        wrap_block(block)
