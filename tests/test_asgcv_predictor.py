from __future__ import annotations

import pytest
import torch

from sfora.asgcv_predictor import (
    AsgcvPatchGradientPredictor,
    predictor_state_sha256,
)


def _predictor() -> AsgcvPatchGradientPredictor:
    torch.manual_seed(7)
    return AsgcvPatchGradientPredictor(channel_dimensions=32, predictor_rank=16)


def test_predictor_is_pair_exchange_equivariant_and_rank_bounded() -> None:
    predictor = _predictor()
    tokens = torch.linspace(-1.0, 1.0, 2 * 2 * 20 * 32).reshape(2, 2, 20, 32)
    relation_signs = torch.tensor([1, -1], dtype=torch.int8)

    observed = predictor(tokens, relation_signs)
    swapped = predictor(tokens[:, [1, 0]], relation_signs)

    assert observed.shape == tokens.shape
    assert bool(torch.isfinite(observed).all())
    torch.testing.assert_close(swapped, observed[:, [1, 0]], rtol=0.0, atol=2e-7)
    for image_ordinal in range(2):
        singular_values = torch.linalg.svdvals(observed[0, image_ordinal].double())
        assert singular_values[16] <= singular_values[0] * 1e-6


def test_predictor_trains_itself_but_never_backpropagates_into_stopped_tokens() -> None:
    predictor = _predictor()
    tokens = torch.randn(2, 2, 5, 32)
    relation_signs = torch.tensor([-1, 1], dtype=torch.int8)

    prediction = predictor(tokens, relation_signs)
    prediction.square().mean().backward()
    assert tokens.grad is None
    assert all(parameter.grad is not None for parameter in predictor.parameters())

    detached = predictor.predict_detached(tokens, relation_signs)
    assert detached.requires_grad is False
    assert detached.grad_fn is None

    with pytest.raises(ValueError):
        predictor(tokens.requires_grad_(True), relation_signs)


def test_predictor_rejects_shape_relation_rank_and_finiteness_drift() -> None:
    predictor = _predictor()
    tokens = torch.randn(2, 2, 5, 32)
    signs = torch.tensor([-1, 1], dtype=torch.int8)

    for invalid_tokens, invalid_signs in (
        (tokens[:, :1], signs),
        (tokens[:, :, :, :31], signs),
        (tokens, signs[:1]),
        (tokens, signs.to(torch.int64)),
        (tokens, torch.tensor([0, 1], dtype=torch.int8)),
    ):
        with pytest.raises(ValueError):
            predictor(invalid_tokens, invalid_signs)

    nonfinite = tokens.clone()
    nonfinite[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        predictor(nonfinite, signs)

    for dimensions, rank in ((0, 16), (32, True), (32, 15), (8, 16)):
        with pytest.raises(ValueError):
            AsgcvPatchGradientPredictor(
                channel_dimensions=dimensions,
                predictor_rank=rank,
            )


def test_predictor_state_digest_is_deterministic_and_mutation_sensitive() -> None:
    predictor = _predictor()
    first = predictor_state_sha256(predictor)
    second = predictor_state_sha256(predictor)
    assert first == second
    assert len(first) == 64

    with torch.no_grad():
        next(predictor.parameters()).view(-1)[0].add_(1.0)
    assert predictor_state_sha256(predictor) != first

    predictor.train()
    assert predictor_state_sha256(predictor) == predictor_state_sha256(predictor.eval())
