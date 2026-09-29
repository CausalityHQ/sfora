"""Read-only native FP16 weight casts; single caller, original F32 state restored."""

import torch
from torch import nn


def stamp(values):
    return tuple((n, id(v), v._version, v.shape, v.dtype, v.device, v.requires_grad) for n, v in values.items())


def source_stamp(model):
    assert not any(m.training for m in model.modules()), "training unsupported"
    return stamp({**dict(model.named_parameters()), **dict(model.named_buffers())})


def build_cache(model):
    assert not torch.is_inference_mode_enabled()
    original = source_stamp(model)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert not any(hasattr(m, "parametrizations") for m in model.modules())
    overrides = {}
    with torch.no_grad():
        for prefix, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                names = ("weight", "bias")
            elif isinstance(module, nn.MultiheadAttention):
                assert module._qkv_same_embed_dim
                names = ("in_proj_weight", "in_proj_bias")
            else:
                continue
            for name in names:
                parameter = getattr(module, name)
                if parameter is not None:
                    full_name = prefix + "." + name if prefix else name
                    assert full_name not in overrides
                    overrides[full_name] = parameter.detach().to(dtype=torch.float16)
    assert overrides and source_stamp(model) == original
    return overrides, original, stamp(overrides)


@torch.inference_mode()
def cached_call(model, cache, pixels):
    # ponytail: temporary functional weights require a single caller; add the serving lock before exposing concurrent calls.
    overrides, original, converted = cache
    assert source_stamp(model) == original, "source changed"
    assert stamp(overrides) == converted, "cache changed"
    assert not torch.is_autocast_enabled(pixels.device.type)
    try:
        with torch.autocast(pixels.device.type, dtype=torch.float16):
            return torch.func.functional_call(model, overrides, (), {"pixel_values": pixels})
    finally:
        assert source_stamp(model) == original, "source changed"
        assert stamp(overrides) == converted, "cache changed"
