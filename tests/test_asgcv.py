from __future__ import annotations

import numpy as np
import pytest

from sfora.asgcv import (
    AsgcvAuthority,
    asgcv_stratum_gradient,
    evaluate_e0,
    exhaustive_selection_mean,
    low_rank_gradient_field,
    normalized_residual_energy,
    selection_variance_ratio,
)


def _fields() -> tuple[np.ndarray, np.ndarray]:
    exact = np.arange(8 * 3 * 4, dtype=np.float64).reshape(8, 3, 4) / 10.0
    offsets = np.linspace(-0.35, 0.35, num=8, dtype=np.float64)[:, None, None]
    predicted = exact + offsets
    return exact, predicted


def test_authority_is_exact_and_rejects_concrete_type_drift() -> None:
    authority = AsgcvAuthority(stratum_size=8, predictor_rank=16).validated()
    assert authority.to_mapping() == {
        "schema": "sfora-asgcv-authority-v1",
        "stratum_size": 8,
        "predictor_rank": 16,
        "accumulator_dtype": "float64",
        "selection_policy": "one-uniform-index-per-eight-pair-stratum-v1",
    }
    assert AsgcvAuthority.from_mapping(authority.to_mapping()) == authority

    for mutation in (
        {**authority.to_mapping(), "stratum_size": True},
        {**authority.to_mapping(), "stratum_size": 7},
        {**authority.to_mapping(), "predictor_rank": 15},
        {**authority.to_mapping(), "accumulator_dtype": "float32"},
        {**authority.to_mapping(), "extra": 1},
    ):
        with pytest.raises(ValueError):
            AsgcvAuthority.from_mapping(mutation)


def test_stratum_estimator_matches_registered_formula_and_is_unbiased() -> None:
    exact, predicted = _fields()
    selected_index = 3

    observed = asgcv_stratum_gradient(
        predicted,
        exact[selected_index],
        selected_index=selected_index,
    )
    expected = predicted.mean(axis=0) + exact[selected_index] - predicted[selected_index]
    np.testing.assert_array_equal(observed, expected)

    selection_mean = exhaustive_selection_mean(exact, predicted)
    np.testing.assert_allclose(selection_mean, exact.mean(axis=0), rtol=0.0, atol=4e-15)


def test_estimator_rejects_shape_dtype_index_and_finiteness_drift() -> None:
    exact, predicted = _fields()
    with pytest.raises(ValueError):
        asgcv_stratum_gradient(predicted.astype(np.float32), exact[0], selected_index=0)
    with pytest.raises(ValueError):
        asgcv_stratum_gradient(predicted[:7], exact[0], selected_index=0)
    with pytest.raises(ValueError):
        asgcv_stratum_gradient(predicted, exact[0, :, :3], selected_index=0)
    with pytest.raises(ValueError):
        asgcv_stratum_gradient(predicted, exact[0], selected_index=True)
    with pytest.raises(ValueError):
        asgcv_stratum_gradient(predicted, exact[0], selected_index=8)
    nonfinite = predicted.copy()
    nonfinite[2, 1, 1] = np.nan
    with pytest.raises(ValueError):
        asgcv_stratum_gradient(nonfinite, exact[0], selected_index=0)


def test_residual_energy_and_selection_variance_have_exact_controls() -> None:
    exact, predicted = _fields()
    expected_energy = float(np.square(exact - predicted).sum() / np.square(exact).sum())
    assert normalized_residual_energy(exact, predicted) == pytest.approx(expected_energy)

    assert selection_variance_ratio(exact, exact.copy()) == pytest.approx(0.0)
    assert selection_variance_ratio(exact, np.zeros_like(exact)) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        normalized_residual_energy(np.zeros_like(exact), np.zeros_like(exact))
    with pytest.raises(ValueError):
        selection_variance_ratio(np.ones_like(exact), np.ones_like(exact))


def test_low_rank_field_uses_registered_orientation_and_float64_accumulation() -> None:
    patch_factors = np.arange(3 * 2, dtype=np.float64).reshape(3, 2) / 7.0
    channel_factors = np.arange(4 * 2, dtype=np.float64).reshape(4, 2) / 11.0

    observed = low_rank_gradient_field(patch_factors, channel_factors, predictor_rank=2)
    np.testing.assert_array_equal(observed, patch_factors @ channel_factors.T)

    with pytest.raises(ValueError):
        low_rank_gradient_field(patch_factors.astype(np.float32), channel_factors, predictor_rank=2)
    with pytest.raises(ValueError):
        low_rank_gradient_field(patch_factors, channel_factors[:, :1], predictor_rank=2)
    with pytest.raises(ValueError):
        low_rank_gradient_field(patch_factors, channel_factors, predictor_rank=True)


def test_e0_metrics_pass_only_when_every_registered_gate_passes() -> None:
    exact, _ = _fields()
    exact_batch = np.stack((exact, exact * 1.25))
    projection = np.eye(4, dtype=np.float64)

    perfect = evaluate_e0(exact_batch, exact_batch.copy(), projection)
    assert perfect.passed is True
    assert perfect.to_mapping() == {
        "schema": "sfora-asgcv-e0-metrics-v1",
        "pair_count": 16,
        "dense_gradient_cosine_ppm": 1_000_000,
        "projected_gradient_cosine_ppm": 1_000_000,
        "patch_salience_spearman_ppm": 1_000_000,
        "normalized_residual_energy_ppm": 0,
        "selection_variance_ratio_ppm": 0,
        "passed": True,
    }
    assert type(perfect).from_mapping(perfect.to_mapping()) == perfect

    for mutation in (
        {**perfect.to_mapping(), "pair_count": True},
        {**perfect.to_mapping(), "pair_count": 15},
        {**perfect.to_mapping(), "dense_gradient_cosine_ppm": 1_000_001},
        {**perfect.to_mapping(), "normalized_residual_energy_ppm": -1},
        {**perfect.to_mapping(), "selection_variance_ratio_ppm": -1},
        {**perfect.to_mapping(), "passed": False},
        {**perfect.to_mapping(), "extra": 1},
    ):
        with pytest.raises(ValueError):
            type(perfect).from_mapping(mutation)

    reversed_prediction = evaluate_e0(exact_batch, -exact_batch, projection)
    assert reversed_prediction.passed is False
    assert reversed_prediction.dense_gradient_cosine_ppm == -1_000_000
    assert reversed_prediction.normalized_residual_energy_ppm == 4_000_000


def test_e0_rejects_projection_batch_and_degenerate_salience_drift() -> None:
    exact, _ = _fields()
    exact_batch = np.stack((exact, exact * 1.25))
    projection = np.eye(4, dtype=np.float64)

    with pytest.raises(ValueError):
        evaluate_e0(exact_batch.astype(np.float32), exact_batch, projection)
    with pytest.raises(ValueError):
        evaluate_e0(exact_batch, exact_batch[:, :, :, :3], projection)
    with pytest.raises(ValueError):
        evaluate_e0(exact_batch, exact_batch, projection.astype(np.float32))
    with pytest.raises(ValueError):
        evaluate_e0(exact_batch, exact_batch, projection[:, :3])
    nonfinite = projection.copy()
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError):
        evaluate_e0(exact_batch, exact_batch, nonfinite)

    constant_patch_norms = np.ones((2, 8, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        evaluate_e0(constant_patch_norms, constant_patch_norms.copy(), projection)
