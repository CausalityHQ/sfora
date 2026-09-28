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
    assert model.transformer.layers == len(model.transformer.resblocks) == 24
    assert model.transformer.width == 1024 and not model.transformer.grad_checkpointing
    assert all(module.training == model.training for module in model.modules())
    attention = []
    for block in model.transformer.resblocks:
        a = block.attn
        assert (a.embed_dim, a.num_heads, a.head_dim, a.scale) == (1024, 16, 64, 0.125)
        assert all(
            isinstance(x, torch.nn.Identity)
            for x in (block.ls_1, block.ls_2, block.drop_path1, block.drop_path2)
        )
        assert block.mlp.c_fc.out_features == 4096
        attention.append([a.embed_dim, a.num_heads, a.head_dim, a.scale])
    norms = {
        n: module.eps
        for n, module in model.named_modules()
        if isinstance(module, torch.nn.LayerNorm)
    }
    assert len(norms) == 51 and all(eps == 1e-5 for eps in norms.values())
    assert model.attn_pool.attn.dropout == 0 and model.attn_pool.attn.batch_first
    assert model.rope.dim == 64 and model.rope.use_cls_token
    values.update(
        {
            "transformer_layers": 24,
            "checkpointing": False,
            "block_attention": attention,
            "normalization_eps": norms,
            "training": model.training,
            "foreign_training": model.rope.rope.training,
        }
    )
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
    assert not any(
        module._forward_hooks or module._forward_pre_hooks
        for model in (loaded, live)
        for module in model.modules()
    )
    if loaded.width == 1024:
        assert runtime_identity(loaded) == runtime_identity(live)
        assert loaded.rope is not live.rope
        assert torch.equal(loaded.rope.freq, live.rope.freq)
        assert loaded.rope.freq.device == live.rope.freq.device
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
