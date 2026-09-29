"""Final native token block learns; pretrained pool and prefix stay frozen."""

from contextlib import contextmanager, nullcontext

import torch

import pe_large_pool as base
from pe_immutable_amp import source_stamp


def freeze(model):
    base.freeze(model)
    model.requires_grad_(False)
    model.encoder.layers[-1].requires_grad_(True)
    return {
        role: tuple(n for n, p in model.named_parameters() if p.requires_grad == active)
        for role, active in (("frozen", False), ("trainable", True))
    }


def frozen_state(model):
    return {n: v for n, v in base.whole_state(model).items() if not n.startswith(("encoder.layers.23.", "runtime.encoder.layers.23."))}


@contextmanager
def verify_export(loaded, live):
    # ponytail: private single-caller qualification; serving concurrency needs its own lock.
    assert loaded is not live
    runtime = base.runtime_identity(loaded)
    assert runtime == base.runtime_identity(live)
    assert all(p.dtype == torch.float32 for m in (loaded, live) for p in m.parameters())
    assert all(p.requires_grad == n.startswith("encoder.layers.23.") for m in (loaded, live) for n, p in m.named_parameters())
    original = base.pair.smoke.digest(base.whole_state(loaded))
    assert original == base.pair.smoke.digest(base.whole_state(live)), "native pair state differs"
    stamps = (source_stamp(loaded), source_stamp(live))
    captured = []
    block = loaded.encoder.layers[-1]

    def capture(module, inputs, kwargs, output):
        assert module is block and not captured
        assert len(inputs) == 2 and inputs[1] is None and not kwargs
        base.assert_frozen_tokens((inputs[0],))
        captured.append((inputs, output))

    hook = block.register_forward_hook(capture, with_kwargs=True)

    @torch.no_grad()
    def encode(pixels):
        assert not torch.is_autocast_enabled(pixels.device.type)
        assert (source_stamp(loaded), source_stamp(live)) == stamps, "native pair changed"
        assert loaded.config.to_dict() == live.config.to_dict() == runtime["config"]
        assert all(not m._forward_pre_hooks and (m._forward_hooks == {hook.id: capture} if m is block else not m._forward_hooks) for model in (loaded, live) for m in model.modules()), "unexpected observer"
        def scope():
            return torch.autocast("cuda", dtype=torch.float16) if pixels.is_cuda else nullcontext()

        try:
            with scope():
                a = loaded(pixel_values=pixels).pooler_output.float()
            assert len(captured) == 1
            inputs, expected = captured[0]
            with scope():
                output = live.encoder.layers[-1](*inputs)
                assert torch.equal(output, expected), "native final block differs"
                b = live.head(live.post_layernorm(output)).float()
            assert torch.equal(a, b), "native suffix differs"
            assert (source_stamp(loaded), source_stamp(live)) == stamps, "native pair changed"
            return a, b
        finally:
            captured.clear()

    try:
        yield encode
        assert (source_stamp(loaded), source_stamp(live)) == stamps, "native pair changed"
    finally:
        hook.remove()
        captured.clear()
    assert base.runtime_identity(loaded) == base.runtime_identity(live) == runtime
    assert all(base.pair.smoke.digest(base.whole_state(m)) == original for m in (loaded, live))
