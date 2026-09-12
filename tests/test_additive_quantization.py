from __future__ import annotations

import math

import pytest
import torch

from sfora.additive_quantization import (
    AdditiveFitSpec,
    AdditiveQuantizationSpec,
    AdditiveQuantizer,
    additive_reconstruction_loss,
    fit_additive_quantizer,
    padded_product_codebooks,
)


def _quantizer() -> AdditiveQuantizer:
    return AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=2, stages=2, codebook_size=2),
        torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
    )


def test_additive_codec_uses_exactly_one_byte_per_full_dimensional_stage() -> None:
    spec = AdditiveQuantizationSpec(dimensions=128, stages=24)

    assert spec.bytes_per_vector == 24
    assert spec.codebook_shape == (24, 256, 128)


def test_padded_product_codebooks_preserve_the_exact_product_reconstruction() -> None:
    product = (
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32).reshape(2, 2),
        torch.tensor([[[5.0], [6.0]]], dtype=torch.float32).reshape(2, 1),
    )
    codes = torch.tensor([[0, 1], [1, 0]], dtype=torch.uint8)

    additive = padded_product_codebooks((2, 1), product)
    decoded = AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=3, stages=2, codebook_size=2), additive
    ).hard_decode(codes)

    torch.testing.assert_close(
        additive,
        torch.tensor(
            [
                [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
                [[0.0, 0.0, 5.0], [0.0, 0.0, 6.0]],
            ],
            dtype=torch.float32,
        ),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        decoded,
        torch.tensor([[1.0, 2.0, 6.0], [3.0, 4.0, 5.0]], dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_padded_product_codebooks_preserve_ragged_128d_product_reconstruction() -> None:
    block_dimensions = (5,) * 16 + (6,) * 8
    generator = torch.Generator().manual_seed(907)
    product = tuple(
        torch.randn((4, width), generator=generator, dtype=torch.float32)
        for width in block_dimensions
    )
    codes = torch.randint(
        0,
        4,
        (7, len(block_dimensions)),
        generator=generator,
        dtype=torch.uint8,
    )

    additive = padded_product_codebooks(block_dimensions, product)
    observed = AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=128, stages=24, codebook_size=4), additive
    ).hard_decode(codes)
    expected = torch.cat(
        [codebook[codes[:, stage].long()] for stage, codebook in enumerate(product)], dim=1
    )

    assert additive.shape == (24, 4, 128)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_additive_lookup_scores_equal_direct_full_vector_dot_products() -> None:
    quantizer = _quantizer()
    codes = torch.tensor([[1, 1], [0, 1], [1, 0]], dtype=torch.uint8)
    queries = torch.tensor([[0.25, -0.75], [1.0, 0.5]], dtype=torch.float32)

    observed = quantizer.asymmetric_dot_scores(queries, codes)
    expected = queries @ quantizer.hard_decode(codes).T

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_coordinate_descent_changes_all_stages_using_the_complete_residual() -> None:
    quantizer = _quantizer()
    values = torch.tensor([[0.9, 0.8], [0.1, 0.2]], dtype=torch.float32)
    initial = torch.zeros((2, 2), dtype=torch.uint8)

    observed = quantizer.coordinate_descent_encode(
        values, initial_codes=initial, sweeps=1, parallel_weight=0.0
    )

    assert observed.tolist() == [[1, 1], [0, 0]]


def test_coordinate_descent_retains_the_incumbent_on_an_exact_tie() -> None:
    quantizer = AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=2, stages=1, codebook_size=2),
        torch.zeros((1, 2, 2), dtype=torch.float32),
    )

    observed = quantizer.coordinate_descent_encode(
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        initial_codes=torch.tensor([[1]], dtype=torch.uint8),
        sweeps=2,
        parallel_weight=3.0,
    )

    assert observed.tolist() == [[1]]


def test_coordinate_descent_honors_the_registered_stage_order() -> None:
    quantizer = AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=1, stages=2, codebook_size=2),
        torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]], dtype=torch.float32),
    )
    values = torch.tensor([[1.0]], dtype=torch.float32)
    initial = torch.tensor([[0, 0]], dtype=torch.uint8)

    first_then_second = quantizer.coordinate_descent_encode(
        values,
        initial_codes=initial,
        sweeps=1,
        parallel_weight=0.0,
        stage_order=(0, 1),
    )
    second_then_first = quantizer.coordinate_descent_encode(
        values,
        initial_codes=initial,
        sweeps=1,
        parallel_weight=0.0,
        stage_order=(1, 0),
    )

    assert first_then_second.tolist() == [[1, 0]]
    assert second_then_first.tolist() == [[0, 1]]


def test_anisotropic_objective_penalizes_parallel_error_by_registered_weight() -> None:
    values = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    reconstructed = torch.tensor([[0.5, 0.5], [0.0, 1.0]], dtype=torch.float32)

    observed = additive_reconstruction_loss(values, reconstructed, parallel_weight=3.0)

    # Squared errors are 0.5 and 1.0; dot residuals are 0.5 and 2.0.
    assert float(observed) == pytest.approx((0.5 + 1.0 + 3.0 * (0.25 + 4.0)) / 2.0)


@pytest.mark.parametrize(
    "builder",
    (
        lambda: AdditiveQuantizationSpec(dimensions=True, stages=2),
        lambda: AdditiveQuantizationSpec(dimensions=2, stages=0),
        lambda: AdditiveQuantizationSpec(dimensions=2, stages=2, codebook_size=257),
    ),
)
def test_additive_spec_rejects_invalid_wire_geometry(builder: object) -> None:
    with pytest.raises(ValueError, match="additive quantization spec"):
        builder()  # type: ignore[operator]


def test_additive_codec_rejects_nonfinite_values_and_malformed_codes() -> None:
    quantizer = _quantizer()
    invalid = torch.tensor([[float("nan"), 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="additive quantization input"):
        quantizer.coordinate_descent_encode(
            invalid,
            initial_codes=torch.zeros((1, 2), dtype=torch.uint8),
            sweeps=1,
            parallel_weight=0.0,
        )
    with pytest.raises(ValueError, match="additive quantization code"):
        quantizer.hard_decode(torch.tensor([[0, 2]], dtype=torch.uint8))


def test_alternating_fit_is_deterministic_and_never_loses_its_best_objective() -> None:
    values = torch.tensor([[0.9, 0.8], [1.1, 0.9], [0.1, 0.2], [0.0, 0.1]], dtype=torch.float32)
    initial = _quantizer()
    initial_codes = torch.tensor([[1, 1], [1, 1], [0, 0], [0, 0]], dtype=torch.uint8)
    fit_spec = AdditiveFitSpec(
        rounds=2,
        assignment_sweeps=1,
        codebook_epochs=2,
        batch_size=2,
        learning_rates=(0.05, 0.01),
        parallel_weight=0.0,
        seed=11,
    )

    first = fit_additive_quantizer(
        values,
        initial_codebooks=initial.detached_codebooks(),
        initial_codes=initial_codes,
        fit_spec=fit_spec,
    )
    second = fit_additive_quantizer(
        values,
        initial_codebooks=initial.detached_codebooks(),
        initial_codes=initial_codes,
        fit_spec=fit_spec,
    )

    assert len(first.objectives) == 3
    assert all(
        right <= left
        for left, right in zip(first.objectives[:-1], first.objectives[1:], strict=True)
    )
    assert first.objectives[-1] < first.objectives[0]
    assert first.objectives[-1] == pytest.approx(
        float(
            additive_reconstruction_loss(
                values,
                first.quantizer.hard_decode(first.codes),
                parallel_weight=fit_spec.parallel_weight,
            ).detach()
        ),
        abs=0.0,
    )
    assert first.objectives == second.objectives
    assert first.assignment_churn_ppm == second.assignment_churn_ppm
    assert len(first.fit_stage_utilization_ppm) == fit_spec.rounds
    assert all(len(values) == 2 for values in first.fit_stage_utilization_ppm)
    assert first.fit_stage_utilization_ppm == second.fit_stage_utilization_ppm
    torch.testing.assert_close(first.codes, second.codes, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        first.quantizer.detached_codebooks(),
        second.quantizer.detached_codebooks(),
        rtol=0.0,
        atol=0.0,
    )


def test_alternating_fit_accepts_inference_mode_source_embeddings() -> None:
    with torch.inference_mode():
        values = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        initial_codebooks = torch.tensor([[[0.8, 0.2], [0.2, 0.8]]], dtype=torch.float32)
        initial_codes = torch.tensor([[0], [1]], dtype=torch.uint8)

    result = fit_additive_quantizer(
        values,
        initial_codebooks=initial_codebooks,
        initial_codes=initial_codes,
        fit_spec=AdditiveFitSpec(
            rounds=1,
            assignment_sweeps=1,
            codebook_epochs=1,
            batch_size=2,
            learning_rates=(0.01,),
            parallel_weight=3.0,
            seed=0,
        ),
    )

    assert math.isfinite(result.objectives[-1])
