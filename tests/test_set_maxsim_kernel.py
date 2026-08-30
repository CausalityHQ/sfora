from __future__ import annotations

import pytest
import torch

from sfora.kernels.set_maxsim import fused_set_maxsim, symmetric_set_maxsim_reference


def test_triton_dot_refuses_tf32_input_precision() -> None:
    import inspect

    import sfora.kernels.set_maxsim as module

    assert 'input_precision="ieee"' in inspect.getsource(module)


def test_symmetric_set_maxsim_matches_hand_computation() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gallery = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    query_weights = torch.tensor([[0.75, 0.25]])
    gallery_weights = torch.tensor([[0.5, 0.5], [0.25, 0.75]])

    scores = symmetric_set_maxsim_reference(
        query,
        gallery,
        query_weights=query_weights,
        gallery_weights=gallery_weights,
    )

    # Exact match: both directed scores are one.  In the second gallery item,
    # q->g is .75 and g->q is one, so the symmetric score is .875.
    torch.testing.assert_close(scores, torch.tensor([[1.0, 0.875]]), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("query_weights", "gallery_weights", "message"),
    [
        (torch.tensor([[0.4, 0.4]]), torch.tensor([[0.5, 0.5]]), "query weights"),
        (torch.tensor([[0.5, 0.5]]), torch.tensor([[0.2, 0.2]]), "gallery weights"),
    ],
)
def test_symmetric_set_maxsim_rejects_non_probability_weights(
    query_weights: torch.Tensor,
    gallery_weights: torch.Tensor,
    message: str,
) -> None:
    tokens = torch.eye(2).unsqueeze(0)

    with pytest.raises(ValueError, match=message):
        symmetric_set_maxsim_reference(
            tokens,
            tokens,
            query_weights=query_weights,
            gallery_weights=gallery_weights,
        )


def test_fused_set_maxsim_cpu_dispatches_to_exact_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.nn.functional.normalize(torch.randn(3, 4, 8, generator=generator), dim=-1)
    gallery = torch.nn.functional.normalize(torch.randn(5, 6, 8, generator=generator), dim=-1)
    query_weights = torch.softmax(torch.randn(3, 4, generator=generator), dim=-1)
    gallery_weights = torch.softmax(torch.randn(5, 6, generator=generator), dim=-1)

    expected = symmetric_set_maxsim_reference(
        query,
        gallery,
        query_weights=query_weights,
        gallery_weights=gallery_weights,
    )
    actual = fused_set_maxsim(
        query,
        gallery,
        query_weights=query_weights,
        gallery_weights=gallery_weights,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fused_set_maxsim_uses_reference_when_only_weights_require_grad() -> None:
    tokens = torch.eye(2).unsqueeze(0)
    weights = torch.tensor([[0.5, 0.5]], requires_grad=True)

    score = fused_set_maxsim(
        tokens,
        tokens,
        query_weights=weights,
        gallery_weights=weights,
    )
    score.sum().backward()

    assert score.dtype == torch.float32
    assert weights.grad is not None


def test_reference_promotes_half_inputs_and_weights_to_float32() -> None:
    tokens = torch.eye(2, dtype=torch.float16).unsqueeze(0)
    weights = torch.tensor([[0.5, 0.5]], dtype=torch.float16)

    score = symmetric_set_maxsim_reference(
        tokens,
        tokens,
        query_weights=weights,
        gallery_weights=weights,
    )

    assert score.dtype == torch.float32
    torch.testing.assert_close(score, torch.ones_like(score), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton parity")
def test_fused_set_maxsim_triton_matches_float32_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(23)
    query = torch.nn.functional.normalize(
        torch.randn(7, 16, 128, generator=generator, device="cuda", dtype=torch.float16),
        dim=-1,
    )
    gallery = torch.nn.functional.normalize(
        torch.randn(11, 12, 128, generator=generator, device="cuda", dtype=torch.float16),
        dim=-1,
    )
    query_weights = torch.softmax(torch.randn(7, 16, generator=generator, device="cuda"), dim=-1)
    gallery_weights = torch.softmax(torch.randn(11, 12, generator=generator, device="cuda"), dim=-1)

    expected = symmetric_set_maxsim_reference(
        query.float(),
        gallery.float(),
        query_weights=query_weights,
        gallery_weights=gallery_weights,
    )
    actual = fused_set_maxsim(
        query,
        gallery,
        query_weights=query_weights,
        gallery_weights=gallery_weights,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
