"""Existing final native block and pool learn; first23 blocks remain frozen."""

import torch

import pe_l14_native_pool as pool
from pe_core_training import named_training_parameters


def freeze(model):
    pool.freeze(model)
    assert len(model.transformer.resblocks) == 24
    model.transformer.resblocks[-1].requires_grad_(True)
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
        for n, value in pool.frozen_state(model).items()
        if not n.startswith("transformer.resblocks.23.")
    }


def runtime_identity(model):
    values = pool.runtime_identity(model)
    assert not model.transformer.grad_checkpointing
    return values


def assert_frozen_tokens(module, inputs):
    pool.assert_frozen_tokens(module, inputs)
    tokens = inputs[0]
    assert tokens.ndim == 3
    if tokens.shape[-1] == 1024:
        assert tokens.shape[1:] == (257, 1024)


@torch.no_grad()
def verified_features(loaded, live, images):
    assert loaded is not live and not loaded.training and not live.training
    captured = []

    def verify_suffix(module, inputs, kwargs, output):
        assert module is loaded.transformer.resblocks[-1] and not captured
        assert kwargs == {"attn_mask": None}
        assert_frozen_tokens(module, inputs)
        other = live.transformer.resblocks[-1](inputs[0], **kwargs)
        assert torch.equal(output, other), (
            "live/strict-loaded final native block differs"
        )
        other = live._pool(live.ln_post(other))
        assert other.shape == (images.shape[0], live.width)
        captured.append(other @ live.proj)

    hook = loaded.transformer.resblocks[-1].register_forward_hook(
        verify_suffix, with_kwargs=True
    )
    try:
        source = loaded(images)
        assert len(captured) == 1, "native final block boundary missing"
        assert torch.equal(source, captured[0]), (
            "live/strict-loaded native suffix differs"
        )
        return source.float(), captured[0].float()
    finally:
        hook.remove()
        captured.clear()
