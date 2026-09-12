from __future__ import annotations

import pytest
import torch

from sfora.residual_quantization import (
    ResidualQuantizationSpec,
    ResidualQuantizer,
    fit_residual_quantizer,
)


def _quantizer() -> ResidualQuantizer:
    return ResidualQuantizer.from_codebooks(
        ResidualQuantizationSpec(dimensions=2, stages=2, codebook_size=3),
        torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
            ],
            dtype=torch.float32,
        ),
    )


def test_residual_quantizer_uses_one_byte_per_full_dimensional_stage() -> None:
    spec = ResidualQuantizationSpec(dimensions=128, stages=24)
    assert spec.bytes_per_vector == 24
    assert spec.codebook_shape == (24, 256, 128)


def test_greedy_encoding_updates_the_full_residual_with_stable_ties() -> None:
    quantizer = _quantizer()
    values = torch.tensor([[1.0, 0.0], [0.1, 2.4]], dtype=torch.float32)

    codes = quantizer.hard_encode(values)

    assert codes.dtype == torch.uint8
    assert codes.tolist() == [[0, 1], [2, 2]]
    torch.testing.assert_close(
        quantizer.hard_decode(codes),
        torch.tensor([[0.5, 0.0], [0.0, 2.5]]),
        rtol=0.0,
        atol=0.0,
    )


def test_additive_dot_tables_equal_explicit_decoded_dot_products() -> None:
    quantizer = _quantizer()
    gallery = torch.tensor([[1.9, 0.2], [0.1, 2.4], [-0.1, 0.1]], dtype=torch.float32)
    queries = torch.tensor([[0.3, -0.7], [1.0, 0.5]], dtype=torch.float32)
    codes = quantizer.hard_encode(gallery)

    observed = quantizer.asymmetric_dot_scores(queries, codes)
    expected = queries @ quantizer.hard_decode(codes).T

    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)


def test_encoding_does_not_lose_nearest_codeword_to_large_common_translation() -> None:
    quantizer = ResidualQuantizer.from_codebooks(
        ResidualQuantizationSpec(dimensions=2, stages=1, codebook_size=2),
        torch.tensor([[[10_001.0, 10_000.0], [10_000.0, 10_000.0]]]),
    )
    assert quantizer.hard_encode(torch.tensor([[10_000.0, 10_000.0]])).tolist() == [[1]]


def test_residual_quantizer_fit_is_seeded_and_reduces_residual_energy() -> None:
    values = torch.tensor(
        [
            [-1.1, -0.9],
            [-1.0, -1.0],
            [-0.9, -1.1],
            [0.9, 1.1],
            [1.0, 1.0],
            [1.1, 0.9],
            [-1.0, 1.0],
            [-0.9, 1.1],
            [1.0, -1.0],
            [1.1, -0.9],
        ],
        dtype=torch.float32,
    )
    spec = ResidualQuantizationSpec(dimensions=2, stages=2, codebook_size=3)

    first = fit_residual_quantizer(values, spec, seed=7, maximum_iterations=20)
    second = fit_residual_quantizer(values, spec, seed=7, maximum_iterations=20)

    torch.testing.assert_close(first.codebooks, second.codebooks, rtol=0.0, atol=0.0)
    restored = first.hard_decode(first.hard_encode(values))
    baseline = values.square().mean()
    observed = float((values - restored.detach()).square().mean())
    assert observed < float(baseline) * 0.1


@pytest.mark.parametrize(
    "spec",
    (
        lambda: ResidualQuantizationSpec(dimensions=True, stages=2),
        lambda: ResidualQuantizationSpec(dimensions=2, stages=0),
        lambda: ResidualQuantizationSpec(dimensions=2, stages=2, codebook_size=257),
    ),
)
def test_residual_quantization_spec_rejects_invalid_geometry(spec: object) -> None:
    with pytest.raises(ValueError, match="residual quantization spec"):
        spec()  # type: ignore[operator]


def test_residual_quantizer_rejects_nonfinite_inputs_and_out_of_range_codes() -> None:
    quantizer = _quantizer()
    invalid = torch.ones((2, 2), dtype=torch.float32)
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="residual quantization input"):
        quantizer.hard_encode(invalid)
    with pytest.raises(ValueError, match="residual quantization code"):
        quantizer.hard_decode(torch.tensor([[0, 7]], dtype=torch.uint8))
