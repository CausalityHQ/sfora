"""Existing native attention pool learns; shared immutable trunk stays native."""

import torch

import pe_l14_readout as readout
from pe_core_training import named_training_parameters


def freeze(model):
    readout.freeze(model)
    model.attn_pool.requires_grad_(True)
    return {
        key: tuple(
            n
            for n, p in named_training_parameters(model, "pe")
            if p.requires_grad == active
        )
        for key, active in (("frozen", False), ("trainable", True))
    }


def frozen_state(model):
    return {
        n: value
        for n, value in readout.frozen_state(model).items()
        if not n.startswith("attn_pool.")
    }


@torch.no_grad()
def verified_features(loaded, live, images):
    assert loaded is not live and not loaded.training and not live.training
    captured = []

    def verify_pool(module, inputs, output):
        assert module is loaded.attn_pool and len(inputs) == 1 and not captured
        tokens = inputs[0]
        assert (
            tokens.ndim == 3
            and tokens.shape[0] == images.shape[0]
            and tokens.shape[-1] == loaded.width
        )
        assert output.shape == (images.shape[0], 1, loaded.width)
        other = live.attn_pool(tokens)
        assert torch.equal(output, other), "live/strict-loaded native pool differs"
        captured.append(other.squeeze(1) @ live.proj)

    hook = loaded.attn_pool.register_forward_hook(verify_pool)
    try:
        source = loaded(images)
        assert len(captured) == 1, "native pool boundary missing"
        assert torch.equal(source, captured[0]), (
            "live/strict-loaded native projection differs"
        )
        return source.float(), captured[0].float()
    finally:
        hook.remove()
