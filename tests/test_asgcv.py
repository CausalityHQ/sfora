from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from sfora.asgcv import (
    AsgcvAuthority,
    AsgcvE0CapacityFloor,
    AsgcvSrhtAuthority,
    asgcv_stratum_gradient,
    canonical_e0_result_bytes,
    canonical_gradient_sample_bytes,
    evaluate_e0,
    evaluate_e0_capacity_floor,
    exhaustive_selection_mean,
    low_rank_gradient_field,
    normalized_residual_energy,
    select_stratum_index,
    selection_schedule_sha256,
    selection_variance_ratio,
    srht_gradient_sketch,
    srht_signs_and_rows,
    validate_e0_result_bytes,
    validate_e0_result_context,
    validate_e0_result_inputs,
    validate_gradient_sample_bundle,
    validate_gradient_sample_bytes,
    validate_gradient_sample_context,
    validate_gradient_sample_inputs,
)
from sfora.asgcv_protocol import (
    AsgcvCompletionProtocol,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    assemble_asgcv_eligible_schedule,
    build_asgcv_pair_schedule,
    classify_asgcv_completion_group,
)


def _fields() -> tuple[np.ndarray, np.ndarray]:
    exact = np.arange(8 * 2 * 3 * 4, dtype=np.float64).reshape(8, 2, 3, 4) / 10.0
    offsets = np.linspace(-0.35, 0.35, num=8, dtype=np.float64)[:, None, None, None]
    predicted = exact + offsets
    return exact, predicted


def _e0_srht() -> AsgcvSrhtAuthority:
    return AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="56" * 32,
    ).validated()


def _independent_e0_block(first: np.ndarray) -> np.ndarray:
    second32 = first.astype(np.float32)
    multiplier = np.where(
        np.arange(second32.size) % 2 == 0,
        np.float32(1.0001),
        np.float32(0.9999),
    ).reshape(second32.shape)
    second32 = (second32 * multiplier).astype(np.float32)
    return second32.astype(np.float64)


def _e0_batch() -> np.ndarray:
    exact, _ = _fields()
    two_strata = np.stack((exact, exact * 1.25))
    return np.tile(two_strata, (32, 1, 1, 1, 1)).astype(np.float32).astype(np.float64)


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
        asgcv_stratum_gradient(predicted, exact[0, :, :, :3], selected_index=0)
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


def test_e0_second_seed_cannot_launder_first_seed_prediction_error() -> None:
    first = _e0_batch()
    first_mean = np.mean(first, axis=1, keepdims=True, dtype=np.float64)
    predicted = first + 0.85 * first_mean
    honest_second = _independent_e0_block(first)
    forged_second = predicted + 1e-6 * (first - predicted)

    honest = evaluate_e0(
        first,
        predicted,
        _e0_srht(),
        second_exact=honest_second,
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    forged = evaluate_e0(
        first,
        predicted,
        _e0_srht(),
        second_exact=forged_second,
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )

    first_block_residual_ppm = int(
        np.ceil(
            np.square(first - predicted).sum(dtype=np.float64)
            / np.square(first).sum(dtype=np.float64)
            * 1_000_000
        )
    )
    assert honest.normalized_residual_energy_ppm >= first_block_residual_ppm > 350_000
    assert forged.normalized_residual_energy_ppm >= first_block_residual_ppm
    assert forged.passed is False


def test_e0_registered_metrics_are_invariant_to_exact_block_order() -> None:
    first = _e0_batch()
    second = _independent_e0_block(first)
    predicted = first.copy()

    forward = evaluate_e0(
        first,
        predicted,
        _e0_srht(),
        second_exact=second,
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    reverse = evaluate_e0(
        second,
        predicted,
        _e0_srht(),
        second_exact=first,
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )

    assert forward == reverse


def test_e0_accepts_a_low_noise_ideal_predictor_when_seed_error_crosses_negative() -> None:
    signal = _e0_batch()
    noise = 0.01 * np.sign(signal)
    first = signal + noise
    second = signal - noise

    metrics = evaluate_e0(
        first,
        signal,
        _e0_srht(),
        second_exact=second,
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )

    assert metrics.normalized_residual_energy_ppm < 350_000


def test_e0_capacity_floor_closes_impossible_predictor_families() -> None:
    low_rank = np.zeros((64, 2, 3, 17), dtype=np.float64)
    low_rank[:, :, :, 0] = np.arange(64 * 2 * 3, dtype=np.float64).reshape(64, 2, 3) + 1.0
    viable = evaluate_e0_capacity_floor(low_rank, low_rank.copy())
    assert (
        viable
        == AsgcvE0CapacityFloor(
            pair_count=64,
            conditional_variance_floor_ppm=0,
            fixed_channel_residual_floor_ppm=0,
            per_sample_rank_residual_floor_ppm=0,
            passed=True,
        ).validated()
    )
    assert type(viable).from_mapping(viable.to_mapping()) == viable

    noisy = evaluate_e0_capacity_floor(low_rank, -low_rank)
    assert noisy.conditional_variance_floor_ppm == 2_000_000
    assert noisy.passed is False

    rotating_channels = np.zeros((64, 2, 1, 32), dtype=np.float64)
    for pair_index in range(64):
        rotating_channels[pair_index, :, 0, pair_index % 32] = 1.0
    fixed_only = evaluate_e0_capacity_floor(
        rotating_channels,
        rotating_channels.copy(),
    )
    assert 500_000 <= fixed_only.fixed_channel_residual_floor_ppm <= 500_001
    assert fixed_only.per_sample_rank_residual_floor_ppm == 0
    assert fixed_only.passed is False

    identity = np.broadcast_to(
        np.eye(32, dtype=np.float64),
        (64, 2, 32, 32),
    ).copy()
    rank_limited = evaluate_e0_capacity_floor(identity, identity.copy())
    assert 500_000 <= rank_limited.fixed_channel_residual_floor_ppm <= 500_001
    assert 500_000 <= rank_limited.per_sample_rank_residual_floor_ppm <= 500_001
    assert rank_limited.passed is False

    for mutation in (
        {**viable.to_mapping(), "pair_count": 63},
        {**viable.to_mapping(), "pair_count": 65},
        {**viable.to_mapping(), "conditional_variance_floor_ppm": -1},
        {**viable.to_mapping(), "fixed_channel_residual_floor_ppm": True},
        {**viable.to_mapping(), "passed": False},
        {**viable.to_mapping(), "extra": 1},
    ):
        with pytest.raises(ValueError):
            type(viable).from_mapping(mutation)


def test_e0_capacity_floor_rejects_dtype_shape_finiteness_and_zero_energy() -> None:
    gradients = np.ones((64, 2, 2, 17), dtype=np.float64)
    for first, second in (
        (gradients.astype(np.float32), gradients),
        (gradients, gradients[:, :, :, :-1]),
        (gradients[:, :1], gradients[:, :1]),
        (np.zeros_like(gradients), np.zeros_like(gradients)),
    ):
        with pytest.raises(ValueError):
            evaluate_e0_capacity_floor(first, second)
    nonfinite = gradients.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_e0_capacity_floor(nonfinite, gradients)


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
    exact_batch = _e0_batch()
    srht = _e0_srht()

    perfect = evaluate_e0(
        exact_batch,
        exact_batch.copy(),
        srht,
        second_exact=_independent_e0_block(exact_batch),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    assert perfect.passed is True
    assert perfect.to_mapping() == {
        "schema": "sfora-asgcv-e0-metrics-v5",
        "pair_count": 512,
        "dense_gradient_cosine_ppm": 1_000_000,
        "projected_gradient_cosine_ppm": 1_000_000,
        "patch_salience_spearman_ppm": 1_000_000,
        "normalized_residual_energy_ppm": 1,
        "selection_variance_ratio_ppm": 1,
        "mean_agreement_upper_ppm": 20,
        "preclip_p99_ratio_ppm": 1_000_000,
        "exact_clip_rate_ppm": 0,
        "asgcv_clip_rate_ppm": 0,
        "clip_rate_delta_ppm": 0,
        "semantic_wall_ratio_ppm": 350_000,
        "peak_cuda_reserved_bytes": 1_000_000,
        "passed": True,
    }
    assert type(perfect).from_mapping(perfect.to_mapping()) == perfect

    for mutation in (
        {**perfect.to_mapping(), "pair_count": True},
        {**perfect.to_mapping(), "pair_count": 15},
        {**perfect.to_mapping(), "pair_count": 504},
        {**perfect.to_mapping(), "dense_gradient_cosine_ppm": 1_000_001},
        {**perfect.to_mapping(), "normalized_residual_energy_ppm": -1},
        {**perfect.to_mapping(), "selection_variance_ratio_ppm": -1},
        {**perfect.to_mapping(), "mean_agreement_upper_ppm": -1},
        {**perfect.to_mapping(), "preclip_p99_ratio_ppm": -1},
        {**perfect.to_mapping(), "exact_clip_rate_ppm": 1_000_001},
        {**perfect.to_mapping(), "clip_rate_delta_ppm": -1},
        {**perfect.to_mapping(), "semantic_wall_ratio_ppm": 350_001},
        {**perfect.to_mapping(), "peak_cuda_reserved_bytes": True},
        {**perfect.to_mapping(), "peak_cuda_reserved_bytes": 96 * 1024**3 + 1},
        {**perfect.to_mapping(), "passed": False},
        {**perfect.to_mapping(), "extra": 1},
    ):
        with pytest.raises(ValueError):
            type(perfect).from_mapping(mutation)

    reversed_prediction = evaluate_e0(
        exact_batch,
        -exact_batch,
        srht,
        second_exact=_independent_e0_block(exact_batch),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    assert reversed_prediction.passed is False
    assert reversed_prediction.dense_gradient_cosine_ppm == -1_000_000
    assert reversed_prediction.normalized_residual_energy_ppm >= 4_000_000


def test_e0_clip_proxy_is_invariant_to_uniform_field_scale() -> None:
    exact = _e0_batch()

    def evaluate(scale: float) -> tuple[int, int, int]:
        metrics = evaluate_e0(
            exact * scale,
            exact.copy() * scale,
            _e0_srht(),
            second_exact=_independent_e0_block(exact * scale),
            selection_seed_sha256="ab" * 32,
            peak_cuda_reserved_bytes=1_000_000,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )
        return (
            metrics.exact_clip_rate_ppm,
            metrics.asgcv_clip_rate_ppm,
            metrics.clip_rate_delta_ppm,
        )

    assert evaluate(1e-6) == evaluate(1e6)


def test_e0_clip_proxy_uses_the_registered_higher_method_p90() -> None:
    scales = np.arange(1, 65, dtype=np.float64).reshape(64, 1, 1, 1, 1)
    exact = _e0_batch() * scales
    metrics = evaluate_e0(
        exact,
        exact.copy(),
        _e0_srht(),
        second_exact=_independent_e0_block(exact),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )

    assert metrics.exact_clip_rate_ppm == 93_750
    assert metrics.asgcv_clip_rate_ppm == 93_750
    assert metrics.clip_rate_delta_ppm == 0


def test_e0_clip_proxy_detects_inflation_above_the_exact_p90_reference() -> None:
    exact = _e0_batch()
    predicted = exact.copy()
    exact_estimates = np.mean(exact, axis=1, dtype=np.float64)
    exact_norms = np.linalg.norm(exact_estimates.reshape(64, -1), axis=1)
    upper_half = set(np.argsort(exact_norms)[32:].tolist())
    for stratum in range(64):
        if stratum not in upper_half:
            continue
        selected = select_stratum_index(
            "ab" * 32,
            optimizer_step=0,
            stratum_ordinal=stratum,
        )
        target_delta = 0.9 * exact_estimates[stratum]
        for rollout in range(8):
            if rollout != selected:
                predicted[stratum, rollout] += (8.0 / 7.0) * target_delta

    metrics = evaluate_e0(
        exact,
        predicted,
        _e0_srht(),
        second_exact=_independent_e0_block(exact),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )

    assert metrics.preclip_p99_ratio_ppm <= 2_000_000
    assert metrics.clip_rate_delta_ppm > 50_000
    assert metrics.passed is False


def test_e0_clip_proxy_outlier_cannot_mask_other_tail_crossings() -> None:
    exact = np.tile(_e0_batch(), (4, 1, 1, 1, 1))
    exact *= np.arange(1, 257, dtype=np.float64).reshape(256, 1, 1, 1, 1)
    predicted = exact.copy()
    exact_estimates = np.mean(exact, axis=1, dtype=np.float64)
    for stratum in range(128, 256):
        selected = select_stratum_index(
            "ab" * 32,
            optimizer_step=0,
            stratum_ordinal=stratum,
        )
        multiplier = 1e30 if stratum == 255 else 1.5
        target_delta = (multiplier - 1.0) * exact_estimates[stratum]
        for rollout in range(8):
            if rollout != selected:
                predicted[stratum, rollout] += (8.0 / 7.0) * target_delta

    metrics = evaluate_e0(
        exact,
        predicted,
        _e0_srht(),
        second_exact=_independent_e0_block(exact),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )

    assert metrics.preclip_p99_ratio_ppm <= 2_000_000
    assert metrics.clip_rate_delta_ppm > 50_000


def test_e0_mean_agreement_gate_detects_selection_aligned_error() -> None:
    exact = _e0_batch()
    predicted = exact.copy()
    for stratum_ordinal in range(exact.shape[0]):
        selected = select_stratum_index(
            "ab" * 32,
            optimizer_step=0,
            stratum_ordinal=stratum_ordinal,
        )
        predicted[stratum_ordinal, selected] += 10.0
    metrics = evaluate_e0(
        exact,
        predicted,
        _e0_srht(),
        second_exact=_independent_e0_block(exact),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    assert metrics.mean_agreement_upper_ppm > 150_000
    assert metrics.passed is False


def test_e0_rejects_srht_batch_and_degenerate_salience_drift() -> None:
    exact_batch = _e0_batch()
    srht = _e0_srht()
    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch.astype(np.float32),
            exact_batch,
            srht,
            second_exact=_independent_e0_block(exact_batch),
            selection_seed_sha256="ab" * 32,
            peak_cuda_reserved_bytes=1_000_000,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )
    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch,
            exact_batch[..., :3],
            srht,
            second_exact=_independent_e0_block(exact_batch),
            selection_seed_sha256="ab" * 32,
            peak_cuda_reserved_bytes=1_000_000,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )
    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch,
            exact_batch,
            np.eye(4, dtype=np.float64),
            second_exact=_independent_e0_block(exact_batch),
            selection_seed_sha256="ab" * 32,
            peak_cuda_reserved_bytes=1_000_000,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )
    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch,
            exact_batch,
            AsgcvSrhtAuthority(
                input_dimensions=3,
                padded_dimensions=4,
                output_dimensions=2,
                seed_sha256="56" * 32,
            ).validated(),
            second_exact=_independent_e0_block(exact_batch),
            selection_seed_sha256="ab" * 32,
            peak_cuda_reserved_bytes=1_000_000,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )
    constant_patch_norms = np.ones((2, 8, 2, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        evaluate_e0(
            constant_patch_norms,
            constant_patch_norms.copy(),
            srht,
            second_exact=_independent_e0_block(constant_patch_norms),
            selection_seed_sha256="ab" * 32,
            peak_cuda_reserved_bytes=1_000_000,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )

    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch,
            exact_batch,
            srht,
            second_exact=_independent_e0_block(exact_batch),
            selection_seed_sha256=True,
            peak_cuda_reserved_bytes=1_000_000,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )


def test_e0_fails_closed_when_control_variate_increases_clipping() -> None:
    predicted_batch = _e0_batch()
    exact_batch = predicted_batch.copy()
    srht = _e0_srht()
    exact_means = np.mean(exact_batch, axis=1, dtype=np.float64)
    scale = 0.1 / float(
        np.max(np.linalg.norm(exact_means.reshape(exact_means.shape[0], -1), axis=1))
    )
    exact_batch *= scale
    predicted_batch *= scale
    for stratum_ordinal in range(exact_batch.shape[0]):
        selected = select_stratum_index(
            "ab" * 32,
            optimizer_step=0,
            stratum_ordinal=stratum_ordinal,
        )
        exact_batch[stratum_ordinal, selected] += 0.5

    metrics = evaluate_e0(
        exact_batch,
        predicted_batch,
        srht,
        second_exact=_independent_e0_block(exact_batch),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    assert metrics.preclip_p99_ratio_ppm > 2_000_000
    assert metrics.exact_clip_rate_ppm <= 100_000
    assert metrics.asgcv_clip_rate_ppm == 1_000_000
    assert metrics.clip_rate_delta_ppm > 50_000
    assert metrics.passed is False


def test_e0_fails_closed_when_semantic_wall_ratio_exceeds_gate() -> None:
    exact_batch = _e0_batch()
    srht = _e0_srht()
    metrics = evaluate_e0(
        exact_batch,
        exact_batch.copy(),
        srht,
        second_exact=_independent_e0_block(exact_batch),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=351,
    )
    assert metrics.semantic_wall_ratio_ppm == 351_000
    assert metrics.passed is False

    memory_metrics = evaluate_e0(
        exact_batch,
        exact_batch.copy(),
        srht,
        second_exact=_independent_e0_block(exact_batch),
        selection_seed_sha256="ab" * 32,
        peak_cuda_reserved_bytes=96 * 1024**3 + 1,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    assert memory_metrics.passed is False

    for exact_wall, asgcv_wall in ((True, 350), (1_000, True), (0, 1), (1_000, 0)):
        with pytest.raises(ValueError):
            evaluate_e0(
                exact_batch,
                exact_batch.copy(),
                srht,
                second_exact=_independent_e0_block(exact_batch),
                selection_seed_sha256="ab" * 32,
                peak_cuda_reserved_bytes=1_000_000,
                exact_semantic_wall_ns=exact_wall,
                asgcv_semantic_wall_ns=asgcv_wall,
            )
    for peak_memory in (True, 0):
        with pytest.raises(ValueError):
            evaluate_e0(
                exact_batch,
                exact_batch.copy(),
                srht,
                second_exact=_independent_e0_block(exact_batch),
                selection_seed_sha256="ab" * 32,
                peak_cuda_reserved_bytes=peak_memory,
                exact_semantic_wall_ns=1_000,
                asgcv_semantic_wall_ns=350,
            )


def test_selection_stream_is_source_bound_exact_and_independent_of_gradients() -> None:
    assert select_stratum_index("00" * 32, optimizer_step=0, stratum_ordinal=0) == 3
    assert select_stratum_index("00" * 32, optimizer_step=1, stratum_ordinal=0) == 1
    assert select_stratum_index("00" * 32, optimizer_step=1, stratum_ordinal=1) == 2
    assert select_stratum_index("ff" * 32, optimizer_step=2**32, stratum_ordinal=17) == 1
    assert select_stratum_index("12" * 32, optimizer_step=99, stratum_ordinal=1234) == 4

    assert (
        selection_schedule_sha256("00" * 32, optimizer_steps=3, strata_per_step=4)
        == "677eb50e03a6e697f3c4d28782c800582eaf1842244dcbf323c950e33901fe5d"
    )

    for seed, step, ordinal in (
        ("0" * 63, 0, 0),
        ("gg" * 32, 0, 0),
        ("00" * 32, True, 0),
        ("00" * 32, -1, 0),
        ("00" * 32, 0, True),
        ("00" * 32, 0, -1),
        ("00" * 32, 2**64, 0),
    ):
        with pytest.raises(ValueError):
            select_stratum_index(seed, optimizer_step=step, stratum_ordinal=ordinal)

    with pytest.raises(ValueError):
        selection_schedule_sha256("00" * 32, optimizer_steps=0, strata_per_step=4)


def test_srht_authority_locks_padding_signs_rows_and_scaling() -> None:
    authority = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="00" * 32,
    ).validated()
    assert authority.to_mapping() == {
        "schema": "sfora-asgcv-srht-authority-v1",
        "input_dimensions": 4,
        "padded_dimensions": 4,
        "output_dimensions": 2,
        "seed_sha256": "00" * 32,
        "accumulator_dtype": "float64",
        "normalization": "orthonormal-hadamard-times-sqrt-padded-over-output-v1",
    }
    assert AsgcvSrhtAuthority.from_mapping(authority.to_mapping()) == authority
    signs, rows = srht_signs_and_rows(authority)
    np.testing.assert_array_equal(signs, np.asarray([1.0, -1.0, 1.0, -1.0]))
    np.testing.assert_array_equal(rows, np.asarray([3, 0], dtype=np.int64))

    field = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    observed = srht_gradient_sketch(field, authority)
    np.testing.assert_allclose(
        observed,
        np.asarray([[-2.0 * np.sqrt(2.0), -np.sqrt(2.0)]], dtype=np.float64),
        rtol=0.0,
        atol=1e-15,
    )


def test_srht_rejects_authority_shape_dtype_and_nonfinite_drift() -> None:
    authority = AsgcvSrhtAuthority(
        input_dimensions=3,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="12" * 32,
    ).validated()
    field = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert srht_gradient_sketch(field, authority).shape == (2, 2)

    for mutation in (
        {**authority.to_mapping(), "input_dimensions": True},
        {**authority.to_mapping(), "padded_dimensions": 8},
        {**authority.to_mapping(), "output_dimensions": 5},
        {**authority.to_mapping(), "seed_sha256": "0" * 63},
        {**authority.to_mapping(), "normalization": "none"},
    ):
        with pytest.raises(ValueError):
            AsgcvSrhtAuthority.from_mapping(mutation)

    with pytest.raises(ValueError):
        srht_gradient_sketch(field.astype(np.float32), authority)
    with pytest.raises(ValueError):
        srht_gradient_sketch(field[:, :2], authority)
    nonfinite = field.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError):
        srht_gradient_sketch(nonfinite, authority)


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _e0_partition() -> AsgcvPartitionAuthority:
    return AsgcvPartitionAuthority(
        source_manifest_sha256="9a" * 32,
        partition_seed_sha256="bc" * 32,
        predictor_train_class_ids=(0, 1),
        e0_validation_class_ids=(2, 3),
        e1_optimization_class_ids=(4, 5),
    ).validated()


def _e0_result_bytes(*, reverse_exact_blocks: bool = False) -> bytes:
    exact_batch = _e0_batch()
    second_batch = _independent_e0_block(exact_batch)
    if reverse_exact_blocks:
        exact_batch, second_batch = second_batch, exact_batch
    return canonical_e0_result_bytes(
        source_commit="1" * 40,
        dataset_manifest_sha256="2" * 64,
        partition_manifest_sha256=_e0_partition().sha256(),
        predictor_state_sha256="3" * 64,
        selection_seed_sha256="ab" * 32,
        first_exact=exact_batch,
        second_exact=second_batch,
        predicted=_e0_batch(),
        srht_authority=_e0_srht(),
        peak_cuda_reserved_bytes=1_000_000,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )


def test_e0_result_is_canonical_claim_ineligible_and_binds_every_array() -> None:
    raw = _e0_result_bytes()
    assert _e0_result_bytes(reverse_exact_blocks=True) == raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    result = validate_e0_result_bytes(raw)

    assert result["schema"] == "sfora-asgcv-e0-result-v5"
    assert result["claim_eligible"] is False
    assert result["source_commit"] == "1" * 40
    assert result["partition_manifest_sha256"] == _e0_partition().sha256()
    assert result["semantic_wall_ns"] == {"asgcv": 350, "exact": 1_000}
    assert result["metrics"]["semantic_wall_ratio_ppm"] == 350_000
    assert result["metrics"]["passed"] is True
    assert set(result["arrays"]) == {
        "first_exact_gradients",
        "second_exact_gradients",
        "predicted_gradients",
    }
    assert result["selection_seed_sha256"] == "ab" * 32
    assert result["selection_schedule_sha256"] == selection_schedule_sha256(
        "ab" * 32,
        optimizer_steps=1,
        strata_per_step=64,
    )
    assert result["srht_authority"] == _e0_srht().to_mapping()
    assert result["arrays"]["first_exact_gradients"]["shape"] == [64, 8, 2, 3, 4]
    assert result["arrays"]["first_exact_gradients"]["dtype"] == "float64-le"
    assert len(result["arrays"]["first_exact_gradients"]["sha256"]) == 64


def test_e0_result_context_binds_partition_and_predictor_state() -> None:
    raw = _e0_result_bytes()
    assert (
        validate_e0_result_context(
            raw,
            partition_authority=_e0_partition(),
            predictor_state_sha256="3" * 64,
        )["result_sha256"]
        == validate_e0_result_bytes(raw)["result_sha256"]
    )

    with pytest.raises(ValueError):
        validate_e0_result_context(
            raw,
            partition_authority=replace(
                _e0_partition(),
                partition_seed_sha256="cd" * 32,
            ).validated(),
            predictor_state_sha256="3" * 64,
        )
    with pytest.raises(ValueError):
        validate_e0_result_context(
            raw,
            partition_authority=_e0_partition(),
            predictor_state_sha256="4" * 64,
        )


def test_e0_result_rejects_semantic_rehash_and_byte_authority_drift() -> None:
    raw = _e0_result_bytes()
    baseline = json.loads(raw)

    mutations: list[dict[str, object]] = []
    claim = json.loads(raw)
    claim["claim_eligible"] = True
    mutations.append(claim)
    time_drift = json.loads(raw)
    time_drift["semantic_wall_ns"]["asgcv"] = 351
    mutations.append(time_drift)
    metric_drift = json.loads(raw)
    metric_drift["metrics"]["semantic_wall_ratio_ppm"] = 349_999
    mutations.append(metric_drift)
    shape_drift = json.loads(raw)
    shape_drift["arrays"]["first_exact_gradients"]["shape"][0] = True
    mutations.append(shape_drift)
    srht_drift = json.loads(raw)
    srht_drift["srht_authority"]["seed_sha256"] = "0" * 63
    mutations.append(srht_drift)
    partition_drift = json.loads(raw)
    partition_drift["partition_manifest_sha256"] = True
    mutations.append(partition_drift)
    selection_drift = json.loads(raw)
    selection_drift["selection_schedule_sha256"] = "4" * 64
    mutations.append(selection_drift)

    for mutation in mutations:
        unsigned = dict(mutation)
        unsigned.pop("result_sha256", None)
        mutation["result_sha256"] = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        with pytest.raises(ValueError):
            validate_e0_result_bytes(_canonical_json_bytes(mutation))

    noncanonical = json.dumps(baseline, sort_keys=False).encode() + b"\n"
    with pytest.raises(ValueError):
        validate_e0_result_bytes(noncanonical)

    baseline["result_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_e0_result_bytes(_canonical_json_bytes(baseline))


def test_e0_result_reopens_every_bound_input_before_acceptance() -> None:
    raw = _e0_result_bytes()
    exact_batch = _e0_batch()

    assert (
        validate_e0_result_inputs(
            raw,
            first_exact=exact_batch,
            second_exact=_independent_e0_block(exact_batch),
            predicted=exact_batch.copy(),
        )["result_sha256"]
        == validate_e0_result_bytes(raw)["result_sha256"]
    )

    mutated = exact_batch.copy()
    mutated[0, 0, 0, 0] = np.nextafter(mutated[0, 0, 0, 0], np.inf)
    with pytest.raises(ValueError):
        validate_e0_result_inputs(
            raw,
            first_exact=mutated,
            second_exact=_independent_e0_block(exact_batch),
            predicted=exact_batch.copy(),
        )

    with pytest.raises(ValueError):
        validate_e0_result_inputs(
            raw,
            first_exact=exact_batch,
            second_exact=mutated,
            predicted=exact_batch.copy(),
        )


def _gradient_sample_arrays() -> tuple[np.ndarray, np.ndarray]:
    tokens = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) / 7.0
    gradient = np.flip(tokens, axis=-1).copy() / 11.0
    return tokens, gradient


def _gradient_sample_bytes() -> bytes:
    tokens, gradient = _gradient_sample_arrays()
    return canonical_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        completion_group_sha256="4" * 64,
        completion_protocol_sha256="6" * 64,
        eligible_schedule_sha256="7" * 64,
        pooler_state_sha256="8" * 64,
        eligible_pair_ordinal=5,
        candidate_pair_ordinal=11,
        pair_ordinals=(17, 29),
        relation_sign=-1,
        grpo_loss=0.125,
        attention_kl=0.375,
        generated_tokens=64,
        patch_tokens=tokens,
        exact_gradient=gradient,
    )


def test_gradient_sample_is_canonical_and_reopens_exact_fp32_arrays() -> None:
    raw = _gradient_sample_bytes()
    value = validate_gradient_sample_bytes(raw)
    tokens, gradient = _gradient_sample_arrays()

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert value["schema"] == "sfora-asgcv-gradient-sample-v4"
    assert value["claim_eligible"] is False
    assert value["pair_ordinals"] == [17, 29]
    assert value["eligible_pair_ordinal"] == 5
    assert value["candidate_pair_ordinal"] == 11
    assert value["relation_sign"] == -1
    assert value["pooler_state_sha256"] == "8" * 64
    assert "predictor_state_sha256" not in value
    assert value["replay_branch_count"] == 8
    assert value["losses"] == {"attention_kl": 0.375, "grpo": 0.125, "semantic": 0.5}
    assert value["arrays"]["patch_tokens"]["shape"] == [2, 3, 4]
    assert value["arrays"]["patch_tokens"]["dtype"] == "float32-le"
    assert (
        validate_gradient_sample_inputs(
            raw,
            patch_tokens=tokens,
            exact_gradient=gradient,
        )["sample_sha256"]
        == value["sample_sha256"]
    )


def test_gradient_sample_rejects_identity_type_shape_and_array_drift() -> None:
    raw = _gradient_sample_bytes()
    tokens, gradient = _gradient_sample_arrays()
    mutations = []
    for path, replacement in (
        (("claim_eligible",), True),
        (("pooler_state_sha256",), True),
        (("pair_ordinals",), [17, 17]),
        (("eligible_pair_ordinal",), True),
        (("relation_sign",), 0),
        (("replay_branch_count",), 7),
        (("generated_tokens",), True),
        (("losses", "semantic"), 0.49),
        (("arrays", "exact_gradient", "shape"), [2, 4, 3]),
    ):
        value = json.loads(raw)
        target = value
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        unsigned = dict(value)
        unsigned.pop("sample_sha256")
        value["sample_sha256"] = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        mutations.append(value)
    for value in mutations:
        with pytest.raises(ValueError):
            validate_gradient_sample_bytes(_canonical_json_bytes(value))

    changed = gradient.copy()
    changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.float32(np.inf))
    with pytest.raises(ValueError):
        validate_gradient_sample_inputs(
            raw,
            patch_tokens=tokens,
            exact_gradient=changed,
        )


def test_gradient_sample_context_cross_binds_refill_pair_group_and_protocol() -> None:
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11,),
        different_prefix_ids=(21,),
        terminal_token_ids=(99,),
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="12" * 32,
        model_revision="3" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=1024,
    ).validated()
    example_ids = tuple(f"cars-{index:02d}" for index in range(32))
    labels = tuple(index // 4 for index in range(32))
    candidates = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="ab" * 32,
        pair_count=16,
    )
    groups = []
    for pair in candidates.pairs:
        correct = (11,) if pair.relation_sign == 1 else (21,)
        wrong = (21,) if pair.relation_sign == 1 else (11,)
        completions = tuple(
            (*correct, 30 + index, 99) if index < 4 else (*wrong, 50 + index, 99)
            for index in range(8)
        )
        groups.append(
            classify_asgcv_completion_group(
                completions,
                pair.relation_sign,
                protocol,
                rollout_authority=rollout,
                candidate_pair_ordinal=pair.ordinal,
            )
        )
    eligible = assemble_asgcv_eligible_schedule(
        candidates,
        tuple(groups),
        target_pair_count=8,
    )
    eligible_index = 0
    candidate_index = eligible.candidate_ordinals[eligible_index]
    pair = candidates.pairs[candidate_index]
    group = groups[candidate_index]
    tokens, gradient = _gradient_sample_arrays()
    raw = canonical_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        completion_group_sha256=group.sha256(),
        completion_protocol_sha256=protocol.sha256(),
        eligible_schedule_sha256=eligible.sha256(),
        pooler_state_sha256="8" * 64,
        eligible_pair_ordinal=eligible_index,
        candidate_pair_ordinal=candidate_index,
        pair_ordinals=(pair.left_index, pair.right_index),
        relation_sign=pair.relation_sign,
        grpo_loss=0.125,
        attention_kl=0.375,
        generated_tokens=64,
        patch_tokens=tokens,
        exact_gradient=gradient,
    )

    assert (
        validate_gradient_sample_context(
            raw,
            eligible_schedule=eligible,
            candidate_schedule=candidates,
            completion_groups=tuple(groups),
        )["sample_sha256"]
        == validate_gradient_sample_bytes(raw)["sample_sha256"]
    )
    assert (
        validate_gradient_sample_bundle(
            raw,
            patch_tokens=tokens,
            exact_gradient=gradient,
            protocol=protocol,
            rollout_authority=rollout,
            eligible_schedule=eligible,
            candidate_schedule=candidates,
            completion_groups=tuple(groups),
            example_ids=example_ids,
            labels=labels,
        )["sample_sha256"]
        == validate_gradient_sample_bytes(raw)["sample_sha256"]
    )

    with pytest.raises(ValueError, match="context"):
        validate_gradient_sample_context(
            raw,
            eligible_schedule=eligible,
            candidate_schedule=candidates,
            completion_groups=tuple(reversed(groups)),
        )
