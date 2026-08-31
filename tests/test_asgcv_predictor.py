from __future__ import annotations

import pytest
import torch

from sfora.asgcv import AsgcvSrhtAuthority, srht_gradient_sketch
from sfora.asgcv_predictor import (
    AsgcvPatchGradientPredictor,
    predictor_state_sha256,
    predictor_training_loss,
    prepare_asgcv_stratum,
    torch_asgcv_stratum_gradient,
    torch_srht_gradient_sketch,
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


def test_torch_srht_matches_scalar_authority_and_backpropagates() -> None:
    authority = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="00" * 32,
    ).validated()
    field = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64, requires_grad=True)

    observed = torch_srht_gradient_sketch(field, authority)
    expected = srht_gradient_sketch(field.detach().numpy(), authority)
    torch.testing.assert_close(observed, torch.from_numpy(expected), rtol=0.0, atol=1e-15)
    observed.square().sum().backward()
    assert field.grad is not None and bool(torch.isfinite(field.grad).all())


def test_predictor_training_loss_has_exact_dense_and_srht_controls() -> None:
    authority = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="00" * 32,
    ).validated()
    exact = torch.arange(1, 1 + 2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    perfect = exact.clone().requires_grad_(True)
    assert predictor_training_loss(perfect, exact, authority).item() == pytest.approx(0.0)

    zero = torch.zeros_like(exact, requires_grad=True)
    loss = predictor_training_loss(zero, exact, authority)
    assert loss.item() == pytest.approx(2.0, abs=1e-6)
    loss.backward()
    assert zero.grad is not None and bool(torch.isfinite(zero.grad).all())

    scaled_loss = predictor_training_loss(
        torch.zeros_like(exact),
        exact * 7.0,
        authority,
    )
    assert scaled_loss.item() == pytest.approx(loss.item(), abs=1e-6)


def test_predictor_training_loss_rejects_teacher_and_tensor_authority_drift() -> None:
    authority = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="12" * 32,
    ).validated()
    exact = torch.randn(1, 2, 3, 4)
    predicted = torch.randn_like(exact, requires_grad=True)

    with pytest.raises(ValueError):
        predictor_training_loss(predicted, exact.requires_grad_(True), authority)
    exact = exact.detach()
    with pytest.raises(ValueError):
        predictor_training_loss(predicted[:, :, :, :3], exact, authority)
    with pytest.raises(ValueError):
        predictor_training_loss(predicted.double(), exact.double(), authority)
    nonfinite = predicted.detach().clone()
    nonfinite[0, 0, 0, 0] = torch.inf
    with pytest.raises(ValueError):
        predictor_training_loss(nonfinite, exact, authority)


def test_prepared_stratum_seals_predictor_before_selection_and_estimator() -> None:
    predictor = _predictor()
    tokens = torch.randn(8, 2, 5, 32)
    signs = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.int8)

    prepared = prepare_asgcv_stratum(
        predictor,
        tokens,
        signs,
        selection_seed="00" * 32,
        optimizer_step=1,
        stratum_ordinal=0,
    )
    assert prepared.selected_index == 1
    assert prepared.predictor_state_sha256 == predictor_state_sha256(predictor)
    assert prepared.predicted.requires_grad is False
    assert prepared.predicted.shape == tokens.shape

    exact_selected = prepared.predicted[prepared.selected_index] + 0.25
    observed = torch_asgcv_stratum_gradient(prepared, exact_selected, predictor=predictor)
    expected = (
        prepared.predicted.mean(dim=0)
        + exact_selected
        - prepared.predicted[prepared.selected_index]
    )
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_prepared_stratum_rejects_same_step_predictor_update_and_input_drift() -> None:
    predictor = _predictor()
    tokens = torch.randn(8, 2, 5, 32)
    signs = torch.ones(8, dtype=torch.int8)
    prepared = prepare_asgcv_stratum(
        predictor,
        tokens,
        signs,
        selection_seed="12" * 32,
        optimizer_step=4,
        stratum_ordinal=3,
    )
    exact_selected = prepared.predicted[prepared.selected_index].clone()

    with torch.no_grad():
        next(predictor.parameters()).view(-1)[0].add_(1.0)
    with pytest.raises(ValueError):
        torch_asgcv_stratum_gradient(prepared, exact_selected, predictor=predictor)

    clean_predictor = _predictor()
    with pytest.raises(ValueError):
        prepare_asgcv_stratum(
            clean_predictor,
            tokens[:7],
            signs[:7],
            selection_seed="12" * 32,
            optimizer_step=4,
            stratum_ordinal=3,
        )

    clean_prepared = prepare_asgcv_stratum(
        clean_predictor,
        tokens,
        signs,
        selection_seed="12" * 32,
        optimizer_step=4,
        stratum_ordinal=3,
    )
    with pytest.raises(ValueError):
        torch_asgcv_stratum_gradient(
            clean_prepared,
            exact_selected.requires_grad_(True),
            predictor=clean_predictor,
        )

    with torch.no_grad():
        clean_prepared.predicted[0, 0, 0, 0].add_(1.0)
    with pytest.raises(ValueError):
        torch_asgcv_stratum_gradient(
            clean_prepared,
            exact_selected.detach(),
            predictor=clean_predictor,
        )
