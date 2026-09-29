"""Last two native token blocks learn; pretrained pool and prefix stay frozen."""

from contextlib import contextmanager, nullcontext
import inspect
from pathlib import Path

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import output_capturing

import pe_large_pool as base
from pe_immutable_amp import source_stamp


def freeze(model):
    base.freeze(model)
    model.requires_grad_(False)
    for block in model.encoder.layers[-2:]:
        block.requires_grad_(True)
    return {
        role: tuple(n for n, p in model.named_parameters() if p.requires_grad == active)
        for role, active in (("frozen", False), ("trainable", True))
    }


def frozen_state(model):
    return {n: v for n, v in base.whole_state(model).items() if not n.startswith(("encoder.layers.22.", "encoder.layers.23.", "runtime.encoder.layers.22.", "runtime.encoder.layers.23."))}


def environment(model, processor):
    values = base.environment(model, processor)
    assert model.config._attn_implementation == "sdpa"
    for obj in (ALL_ATTENTION_FUNCTIONS["sdpa"], torch.nn.functional.multi_head_attention_forward, torch.amp.autocast_mode, output_capturing):
        path = Path(inspect.getfile(obj))
        values["native_files"][str(path)] = base.pair.sha(path)
    return values


@contextmanager
def verify_export(loaded, live):
    # ponytail: private single-caller qualification; serving concurrency needs its own lock.
    assert loaded is not live
    runtime = base.runtime_identity(loaded)
    assert runtime == base.runtime_identity(live)
    assert all(p.dtype == torch.float32 for m in (loaded, live) for p in m.parameters())
    assert all(p.requires_grad == n.startswith(("encoder.layers.22.", "encoder.layers.23.")) for m in (loaded, live) for n, p in m.named_parameters())
    original = base.pair.smoke.digest(base.whole_state(loaded))
    assert original == base.pair.smoke.digest(base.whole_state(live)), "native pair state differs"
    stamps = (source_stamp(loaded), source_stamp(live))
    captured = []
    observations = []
    block, final = loaded.encoder.layers[-2:]

    def capture(module, inputs, kwargs, output):
        assert module is block and not captured
        assert len(inputs) == 2 and inputs[1] is None and not kwargs
        base.assert_frozen_tokens((inputs[0],))
        assert output.dtype == torch.float32 and torch.isfinite(output).all(), "native block type/nonfinite"
        captured.append((inputs, output, torch.is_autocast_enabled(output.device.type)))

    def capture_final(module, inputs, kwargs, output):
        assert module is final and len(captured) == 1
        assert len(inputs) == 2 and inputs[1] is None and not kwargs
        base.assert_frozen_tokens((inputs[0],))
        assert torch.equal(inputs[0], captured[0][1]), "loaded adapted chain differs"
        assert output.dtype == torch.float32 and torch.isfinite(output).all()
        captured.append((inputs, output, torch.is_autocast_enabled(output.device.type)))

    hook = block.register_forward_hook(capture, with_kwargs=True)
    final_hook = final.register_forward_hook(capture_final, with_kwargs=True)

    @torch.no_grad()
    def encode(pixels):
        assert not torch.is_autocast_enabled(pixels.device.type)
        assert pixels.dtype == torch.float32 and all(v.device == pixels.device for m in (loaded, live) for v in (*m.parameters(), *m.buffers())), "native pair device/type differs"
        assert (source_stamp(loaded), source_stamp(live)) == stamps, "native pair changed"
        assert loaded.config.to_dict() == live.config.to_dict() == runtime["config"]
        assert all(not m._forward_pre_hooks and (m._forward_hooks == {hook.id: capture} if m is block else m._forward_hooks == {final_hook.id: capture_final} if m is final else not m._forward_hooks) for model in (loaded, live) for m in model.modules()), "unexpected observer"
        def scope():
            return torch.autocast("cuda", dtype=torch.float16) if pixels.is_cuda else nullcontext()

        try:
            with scope():
                raw_a = loaded(pixel_values=pixels).pooler_output
                loaded_amp = torch.is_autocast_enabled(pixels.device.type)
                a = raw_a.float()
            between_amp = torch.is_autocast_enabled(pixels.device.type)
            assert not between_amp
            assert len(captured) == 2
            inputs, expected, block_amp = captured[0]
            _, final_expected, final_amp = captured[1]
            with scope():
                live_amp = torch.is_autocast_enabled(pixels.device.type)
                output = live.encoder.layers[-2](*inputs)
                assert output.dtype == expected.dtype and torch.isfinite(output).all(), "native block type/nonfinite"
                assert torch.equal(output, expected), "native first adapted block differs"
                output = live.encoder.layers[-1](output, None)
                assert output.dtype == final_expected.dtype and torch.isfinite(output).all()
                assert torch.equal(output, final_expected), "native final block differs"
                raw_b = live.head(live.post_layernorm(output))
                b = raw_b.float()
            assert raw_a.dtype == raw_b.dtype == (torch.float16 if pixels.is_cuda else torch.float32)
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), "native pool nonfinite"
            assert torch.equal(a, b), "native suffix differs"
            assert (source_stamp(loaded), source_stamp(live)) == stamps, "native pair changed"
            observations.append({"loaded_amp_inside": loaded_amp, "captured_block_amp_inside": block_amp, "captured_final_block_amp_inside": final_amp, "between_encoder_amp_enabled": between_amp, "live_amp_inside": live_amp, "autocast_cache_configuration_enabled": torch.is_autocast_cache_enabled(), "loaded_raw_pool_dtype": str(raw_a.dtype), "live_raw_pool_dtype": str(raw_b.dtype), "block_dtype": str(output.dtype), "pooled_outputs_finite": True})
            return a, b
        finally:
            captured.clear()

    try:
        yield encode, observations
        assert (source_stamp(loaded), source_stamp(live)) == stamps, "native pair changed"
    finally:
        hook.remove()
        final_hook.remove()
        captured.clear()
    assert base.runtime_identity(loaded) == base.runtime_identity(live) == runtime
    assert all(base.pair.smoke.digest(base.whole_state(m)) == original for m in (loaded, live)), "native pair state changed"
