from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from sfora.product_quantization import (
    ProductQuantizationSpec,
    ProductQuantizer,
    fit_product_quantizer,
    neighborhood_adc_distillation_loss,
)


def _quantizer() -> ProductQuantizer:
    spec = ProductQuantizationSpec(block_dimensions=(2, 1), codebook_size=3)
    return ProductQuantizer.from_codebooks(
        spec,
        (
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
            torch.tensor([[-1.0], [0.0], [2.0]], dtype=torch.float32),
        ),
    )


def test_product_quantization_spec_supports_exact_128_dimension_24_byte_layout() -> None:
    spec = ProductQuantizationSpec(block_dimensions=(5,) * 16 + (6,) * 8)

    assert spec.dimensions == 128
    assert spec.bytes_per_vector == 24
    assert spec.codebook_size == 256


@pytest.mark.parametrize(
    ("dimensions", "size"),
    [
        ((2, 0), 256),
        ((2, True), 256),
        ((2, 1), 1),
        ((2, 1), 257),
        ([2, 1], 256),
    ],
)
def test_product_quantization_spec_rejects_invalid_wire_geometry(
    dimensions: object, size: object
) -> None:
    with pytest.raises(ValueError, match="product quantization spec"):
        ProductQuantizationSpec(block_dimensions=dimensions, codebook_size=size)  # type: ignore[arg-type]


def test_hard_codes_use_nearest_codeword_and_lowest_index_ties() -> None:
    quantizer = _quantizer()
    values = torch.tensor([[0.5, 0.0, -0.5], [0.0, 1.9, 1.6]], dtype=torch.float32)

    codes = quantizer.hard_encode(values)

    assert codes.dtype == torch.uint8
    assert codes.shape == (2, 2)
    assert codes.tolist() == [[0, 0], [2, 2]]
    torch.testing.assert_close(
        quantizer.hard_decode(codes),
        torch.tensor([[0.0, 0.0, -1.0], [0.0, 2.0, 2.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_asymmetric_distance_matches_explicit_hard_decode() -> None:
    quantizer = _quantizer()
    queries = torch.tensor([[0.2, 0.4, -0.5], [1.0, -0.5, 1.0]], dtype=torch.float32)
    gallery = torch.tensor(
        [[0.1, 0.1, -0.9], [0.9, 0.1, 0.2], [0.0, 1.8, 1.8]], dtype=torch.float32
    )
    codes = quantizer.hard_encode(gallery)

    observed = quantizer.asymmetric_squared_distances(queries, codes)
    restored = quantizer.hard_decode(codes)
    expected = (queries[:, None, :] - restored[None, :, :]).square().sum(dim=-1)

    torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
    restored_quantizer = ProductQuantizer.from_codebooks(
        quantizer.spec, quantizer.detached_codebooks()
    )
    assert restored_quantizer.hard_encode(gallery).tolist() == codes.tolist()


def test_product_quantizer_fit_is_seeded_and_uses_every_dimension() -> None:
    generator = torch.Generator().manual_seed(11)
    values = torch.randn((512, 6), generator=generator)
    spec = ProductQuantizationSpec(block_dimensions=(2, 1, 3), codebook_size=8)

    first = fit_product_quantizer(values, spec, seed=5, maximum_iterations=10)
    second = fit_product_quantizer(values, spec, seed=5, maximum_iterations=10)

    for left, right in zip(first.detached_codebooks(), second.detached_codebooks(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    restored = first.hard_decode(first.hard_encode(values)).detach()
    assert restored.shape == values.shape
    assert float((values - restored).square().mean()) < float(values.square().mean())


def test_straight_through_forward_is_hard_and_gradients_reach_rows_and_codebooks() -> None:
    quantizer = _quantizer()
    values = torch.tensor([[0.2, 0.1, -0.8], [0.9, 0.2, 1.8]], requires_grad=True)

    reconstructed, codes = quantizer.straight_through(values)
    expected = quantizer.hard_decode(codes).detach()
    torch.testing.assert_close(reconstructed.detach(), expected, rtol=0.0, atol=0.0)

    weights = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]])
    (reconstructed * weights).sum().backward()

    torch.testing.assert_close(values.grad, weights, rtol=0.0, atol=0.0)
    first_gradient = quantizer.codebooks[0].grad
    second_gradient = quantizer.codebooks[1].grad
    assert first_gradient is not None and second_gradient is not None
    torch.testing.assert_close(first_gradient[0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(first_gradient[1], torch.tensor([-2.0, 1.0]))
    torch.testing.assert_close(second_gradient[0], torch.tensor([3.0]))
    torch.testing.assert_close(second_gradient[2], torch.tensor([0.5]))


def test_neighborhood_adc_distillation_matches_independent_terms_and_backpropagates() -> None:
    quantizer = _quantizer()
    student_queries = F.normalize(
        torch.tensor([[0.8, 0.2, -0.3], [0.1, 0.9, 0.4]], requires_grad=True), dim=-1
    )
    student_gallery = F.normalize(
        torch.tensor(
            [
                [[0.7, 0.1, -0.2], [0.0, 0.9, 0.3], [-0.8, 0.1, 0.2]],
                [[0.2, 0.8, 0.5], [0.9, 0.0, -0.1], [-0.2, -0.7, 0.4]],
            ],
            requires_grad=True,
        ),
        dim=-1,
    )
    teacher_queries = F.normalize(student_queries.detach() + 0.05, dim=-1)
    teacher_gallery = F.normalize(student_gallery.detach() - 0.03, dim=-1)

    observed = neighborhood_adc_distillation_loss(
        student_queries,
        student_gallery,
        teacher_queries,
        teacher_gallery,
        quantizer,
        temperature=0.2,
        float_weight=0.25,
        reconstruction_weight=0.1,
    )

    hard = quantizer.hard_decode(
        quantizer.hard_encode(student_gallery.detach().reshape(-1, 3))
    ).reshape_as(student_gallery)
    teacher_logits = torch.einsum("bd,bcd->bc", teacher_queries, teacher_gallery) / 0.2
    adc_logits = -0.5 * (student_queries[:, None, :] - hard).square().sum(dim=-1) / 0.2
    float_logits = torch.einsum("bd,bcd->bc", student_queries, student_gallery) / 0.2
    probabilities = F.softmax(teacher_logits, dim=-1)
    expected_adc = F.kl_div(F.log_softmax(adc_logits, dim=-1), probabilities, reduction="batchmean")
    expected_float = F.kl_div(
        F.log_softmax(float_logits, dim=-1), probabilities, reduction="batchmean"
    )
    expected_reconstruction = F.mse_loss(hard, student_gallery.detach())

    torch.testing.assert_close(observed.adc_kl.detach(), expected_adc)
    torch.testing.assert_close(observed.float_kl.detach(), expected_float)
    torch.testing.assert_close(observed.reconstruction.detach(), expected_reconstruction)
    expected_total = expected_adc + 0.25 * expected_float + 0.1 * expected_reconstruction
    torch.testing.assert_close(observed.total.detach(), expected_total)
    observed.total.backward()
    assert student_queries.grad_fn is not None
    assert student_gallery.grad_fn is not None
    assert all(codebook.grad is not None for codebook in quantizer.codebooks)


def test_reconstruction_gradient_attracts_rows_and_selected_codewords() -> None:
    quantizer = _quantizer()
    student_queries = F.normalize(torch.tensor([[1.0, 0.2], [0.1, 1.0]]), dim=-1)
    student_queries = torch.cat((student_queries, torch.zeros((2, 1))), dim=1)
    student_gallery = F.normalize(
        torch.tensor(
            [
                [[0.6, 0.8, 0.1], [0.8, 0.1, -0.5]],
                [[0.1, 0.9, 0.3], [-0.7, 0.2, 0.4]],
            ]
        ),
        dim=-1,
    ).requires_grad_()
    teacher_queries = F.normalize(student_queries + 0.01, dim=-1)
    teacher_gallery = F.normalize(student_gallery.detach() - 0.01, dim=-1)
    loss = neighborhood_adc_distillation_loss(
        student_queries,
        student_gallery,
        teacher_queries,
        teacher_gallery,
        quantizer,
        temperature=0.2,
        float_weight=0.0,
        reconstruction_weight=1.0,
    )
    codes = quantizer.hard_encode(student_gallery.detach().reshape(-1, 3))
    hard = quantizer.hard_decode(codes).detach().reshape_as(student_gallery)

    loss.reconstruction.backward()

    assert student_gallery.grad is not None
    displacement = student_gallery.detach() - hard
    assert float((student_gallery.grad * displacement).sum()) > 0.0


def _nonfinite(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    result[0, 0] = float("nan")
    return result


@pytest.mark.parametrize("mutation", (_nonfinite, lambda value: value[:, :2]))
def test_product_quantizer_rejects_nonfinite_or_wrong_width(mutation: object) -> None:
    quantizer = _quantizer()
    values = mutation(torch.ones((2, 3), dtype=torch.float32))  # type: ignore[operator]
    with pytest.raises(ValueError, match="product quantization input"):
        quantizer.hard_encode(values)


def test_distillation_rejects_invalid_temperature() -> None:
    quantizer = _quantizer()
    query = F.normalize(torch.ones((2, 3)), dim=-1)
    gallery = F.normalize(torch.ones((2, 3, 3)), dim=-1)
    with pytest.raises(ValueError, match="distillation authority"):
        neighborhood_adc_distillation_loss(
            query,
            gallery,
            query,
            gallery,
            quantizer,
            temperature=0.0,
            float_weight=0.25,
            reconstruction_weight=0.1,
        )
