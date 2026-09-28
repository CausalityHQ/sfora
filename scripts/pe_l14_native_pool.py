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
    values = {
        n: value
        for n, value in readout.frozen_state(model).items()
        if not n.startswith("attn_pool.")
    }
    values.update(
        ("rope.runtime." + n, value) for n, value in model.rope.rope.named_buffers()
    )
    return values


def assert_frozen_tokens(module, inputs):
    assert len(inputs) == 1
    tokens = inputs[0]
    assert (
        tokens.dtype == torch.float32
        and not tokens.requires_grad
        and tokens.grad_fn is None
    )


def runtime_identity(model):
    assert model.layers == 24 and model.width == 1024 and model.pool_type == "attn"
    assert model.rope.grid_size == (16, 16)
    assert all(block.attn.rope is model.rope for block in model.transformer.resblocks)
    assert not any(
        module._forward_hooks or module._forward_pre_hooks for module in model.modules()
    )
    return {
        "native_config": [
            model.layers,
            model.width,
            model.patch_size,
            model.posemb_grid_size,
            model.use_cls_token,
            model.use_abs_posemb,
            model.use_rope2d,
            model.pool_type,
        ],
        "pool_heads": model.attn_pool.num_heads,
        "grid": model.rope.grid_size,
        "frequency_device": str(model.rope.freq.device),
        "foreign_buffer_devices": {
            n: str(value.device) for n, value in model.rope.rope.named_buffers()
        },
    }


@torch.no_grad()
def verified_features(loaded, live, images):
    assert loaded is not live and not loaded.training and not live.training
    captured = []

    def verify_pool(module, inputs, output):
        assert module is loaded.attn_pool and len(inputs) == 1 and not captured
        tokens = inputs[0]
        assert_frozen_tokens(module, inputs)
        assert (
            tokens.ndim == 3
            and tokens.shape[0] == images.shape[0]
            and tokens.shape[-1] == loaded.width
        )
        assert output.shape == (images.shape[0], 1, loaded.width)
        if loaded.width == 1024:
            assert tokens.shape == (images.shape[0], 257, 1024)
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
