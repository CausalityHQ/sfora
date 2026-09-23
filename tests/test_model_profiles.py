"""The bundled SOP projection must load from the package and serve packed codes."""

from __future__ import annotations

import torch

from sfora.model_profiles import load_oml_sop_compact_encoder


def test_oml_sop_profile_loads_and_produces_130_byte_rows() -> None:
    encoder = load_oml_sop_compact_encoder()
    assert encoder.weight.shape == (128, 384)
    assert encoder.sha256 == "07e6e0f38dae4d509fe1e27b10aa650acda1f08d799faa025793cf7686a78cd4"
    values = torch.randn(10, 384, generator=torch.Generator().manual_seed(23))
    packed = encoder.encode_packed(values.contiguous())
    assert packed.codes.shape == (10, 128)
    assert packed.inverse_norms.shape == (10,)
    assert packed.codes.numel() + packed.inverse_norms.numel() * 2 == 10 * 130
