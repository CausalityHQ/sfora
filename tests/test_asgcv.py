from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sfora.asgcv import (
    AsgcvAuthority,
    AsgcvSrhtAuthority,
    asgcv_stratum_gradient,
    canonical_e0_result_bytes,
    canonical_gradient_sample_bytes,
    evaluate_e0,
    exhaustive_selection_mean,
    low_rank_gradient_field,
    normalized_residual_energy,
    select_stratum_index,
    selection_schedule_sha256,
    selection_variance_ratio,
    srht_gradient_sketch,
    srht_signs_and_rows,
    validate_e0_result_bytes,
    validate_e0_result_inputs,
    validate_gradient_sample_bytes,
    validate_gradient_sample_inputs,
)


def _fields() -> tuple[np.ndarray, np.ndarray]:
    exact = np.arange(8 * 3 * 4, dtype=np.float64).reshape(8, 3, 4) / 10.0
    offsets = np.linspace(-0.35, 0.35, num=8, dtype=np.float64)[:, None, None]
    predicted = exact + offsets
    return exact, predicted


def _e0_srht() -> AsgcvSrhtAuthority:
    return AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="56" * 32,
    ).validated()


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
    srht = _e0_srht()
    exact_preclip_norms = np.asarray([0.5, 0.75, 1.25, 1.5], dtype=np.float64)

    perfect = evaluate_e0(
        exact_batch,
        exact_batch.copy(),
        srht,
        exact_preclip_norms=exact_preclip_norms,
        asgcv_preclip_norms=exact_preclip_norms.copy(),
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    assert perfect.passed is True
    assert perfect.to_mapping() == {
        "schema": "sfora-asgcv-e0-metrics-v1",
        "pair_count": 16,
        "dense_gradient_cosine_ppm": 1_000_000,
        "projected_gradient_cosine_ppm": 1_000_000,
        "patch_salience_spearman_ppm": 1_000_000,
        "normalized_residual_energy_ppm": 0,
        "selection_variance_ratio_ppm": 0,
        "preclip_p99_ratio_ppm": 1_000_000,
        "exact_clip_rate_ppm": 500_000,
        "asgcv_clip_rate_ppm": 500_000,
        "clip_rate_delta_ppm": 0,
        "semantic_wall_ratio_ppm": 350_000,
        "passed": True,
    }
    assert type(perfect).from_mapping(perfect.to_mapping()) == perfect

    for mutation in (
        {**perfect.to_mapping(), "pair_count": True},
        {**perfect.to_mapping(), "pair_count": 15},
        {**perfect.to_mapping(), "dense_gradient_cosine_ppm": 1_000_001},
        {**perfect.to_mapping(), "normalized_residual_energy_ppm": -1},
        {**perfect.to_mapping(), "selection_variance_ratio_ppm": -1},
        {**perfect.to_mapping(), "preclip_p99_ratio_ppm": -1},
        {**perfect.to_mapping(), "exact_clip_rate_ppm": 1_000_001},
        {**perfect.to_mapping(), "clip_rate_delta_ppm": -1},
        {**perfect.to_mapping(), "semantic_wall_ratio_ppm": 350_001},
        {**perfect.to_mapping(), "passed": False},
        {**perfect.to_mapping(), "extra": 1},
    ):
        with pytest.raises(ValueError):
            type(perfect).from_mapping(mutation)

    reversed_prediction = evaluate_e0(
        exact_batch,
        -exact_batch,
        srht,
        exact_preclip_norms=exact_preclip_norms,
        asgcv_preclip_norms=exact_preclip_norms.copy(),
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    assert reversed_prediction.passed is False
    assert reversed_prediction.dense_gradient_cosine_ppm == -1_000_000
    assert reversed_prediction.normalized_residual_energy_ppm == 4_000_000


def test_e0_rejects_srht_batch_and_degenerate_salience_drift() -> None:
    exact, _ = _fields()
    exact_batch = np.stack((exact, exact * 1.25))
    srht = _e0_srht()
    norms = np.asarray([0.5, 0.75, 1.25, 1.5], dtype=np.float64)

    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch.astype(np.float32),
            exact_batch,
            srht,
            exact_preclip_norms=norms,
            asgcv_preclip_norms=norms,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )
    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch,
            exact_batch[:, :, :, :3],
            srht,
            exact_preclip_norms=norms,
            asgcv_preclip_norms=norms,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )
    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch,
            exact_batch,
            np.eye(4, dtype=np.float64),
            exact_preclip_norms=norms,
            asgcv_preclip_norms=norms,
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
            exact_preclip_norms=norms,
            asgcv_preclip_norms=norms,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )
    constant_patch_norms = np.ones((2, 8, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        evaluate_e0(
            constant_patch_norms,
            constant_patch_norms.copy(),
            srht,
            exact_preclip_norms=norms,
            asgcv_preclip_norms=norms,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )

    with pytest.raises(ValueError):
        evaluate_e0(
            exact_batch,
            exact_batch,
            srht,
            exact_preclip_norms=norms.astype(np.float32),
            asgcv_preclip_norms=norms,
            exact_semantic_wall_ns=1_000,
            asgcv_semantic_wall_ns=350,
        )


def test_e0_fails_closed_when_control_variate_increases_clipping() -> None:
    exact, _ = _fields()
    exact_batch = np.stack((exact, exact * 1.25))
    srht = _e0_srht()
    exact_norms = np.full(20, 0.5, dtype=np.float64)
    asgcv_norms = np.full(20, 3.0, dtype=np.float64)

    metrics = evaluate_e0(
        exact_batch,
        exact_batch.copy(),
        srht,
        exact_preclip_norms=exact_norms,
        asgcv_preclip_norms=asgcv_norms,
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )
    assert metrics.preclip_p99_ratio_ppm == 6_000_000
    assert metrics.exact_clip_rate_ppm == 0
    assert metrics.asgcv_clip_rate_ppm == 1_000_000
    assert metrics.clip_rate_delta_ppm == 1_000_000
    assert metrics.passed is False


def test_e0_fails_closed_when_semantic_wall_ratio_exceeds_gate() -> None:
    exact, _ = _fields()
    exact_batch = np.stack((exact, exact * 1.25))
    srht = _e0_srht()
    norms = np.asarray([0.5, 0.75, 1.25, 1.5], dtype=np.float64)

    metrics = evaluate_e0(
        exact_batch,
        exact_batch.copy(),
        srht,
        exact_preclip_norms=norms,
        asgcv_preclip_norms=norms.copy(),
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=351,
    )
    assert metrics.semantic_wall_ratio_ppm == 351_000
    assert metrics.passed is False

    for exact_wall, asgcv_wall in ((True, 350), (1_000, True), (0, 1), (1_000, 0)):
        with pytest.raises(ValueError):
            evaluate_e0(
                exact_batch,
                exact_batch.copy(),
                srht,
                exact_preclip_norms=norms,
                asgcv_preclip_norms=norms.copy(),
                exact_semantic_wall_ns=exact_wall,
                asgcv_semantic_wall_ns=asgcv_wall,
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


def _e0_result_bytes() -> bytes:
    exact, _ = _fields()
    exact_batch = np.stack((exact, exact * 1.25))
    norms = np.asarray([0.5, 0.75, 1.25, 1.5], dtype=np.float64)
    return canonical_e0_result_bytes(
        source_commit="1" * 40,
        dataset_manifest_sha256="2" * 64,
        partition_manifest_sha256="5" * 64,
        predictor_state_sha256="3" * 64,
        selection_schedule_sha256="4" * 64,
        exact=exact_batch,
        predicted=exact_batch.copy(),
        srht_authority=_e0_srht(),
        exact_preclip_norms=norms,
        asgcv_preclip_norms=norms.copy(),
        exact_semantic_wall_ns=1_000,
        asgcv_semantic_wall_ns=350,
    )


def test_e0_result_is_canonical_claim_ineligible_and_binds_every_array() -> None:
    raw = _e0_result_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    result = validate_e0_result_bytes(raw)

    assert result["schema"] == "sfora-asgcv-e0-result-v1"
    assert result["claim_eligible"] is False
    assert result["source_commit"] == "1" * 40
    assert result["partition_manifest_sha256"] == "5" * 64
    assert result["semantic_wall_ns"] == {"asgcv": 350, "exact": 1_000}
    assert result["metrics"]["semantic_wall_ratio_ppm"] == 350_000
    assert result["metrics"]["passed"] is True
    assert set(result["arrays"]) == {
        "asgcv_preclip_norms",
        "exact_gradients",
        "exact_preclip_norms",
        "predicted_gradients",
    }
    assert result["srht_authority"] == _e0_srht().to_mapping()
    assert result["arrays"]["exact_gradients"]["shape"] == [2, 8, 3, 4]
    assert result["arrays"]["exact_gradients"]["dtype"] == "float64-le"
    assert len(result["arrays"]["exact_gradients"]["sha256"]) == 64


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
    shape_drift["arrays"]["exact_gradients"]["shape"][0] = True
    mutations.append(shape_drift)
    srht_drift = json.loads(raw)
    srht_drift["srht_authority"]["seed_sha256"] = "0" * 63
    mutations.append(srht_drift)
    partition_drift = json.loads(raw)
    partition_drift["partition_manifest_sha256"] = True
    mutations.append(partition_drift)

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
    exact, _ = _fields()
    exact_batch = np.stack((exact, exact * 1.25))
    norms = np.asarray([0.5, 0.75, 1.25, 1.5], dtype=np.float64)

    assert (
        validate_e0_result_inputs(
            raw,
            exact=exact_batch,
            predicted=exact_batch.copy(),
            exact_preclip_norms=norms,
            asgcv_preclip_norms=norms.copy(),
        )["result_sha256"]
        == validate_e0_result_bytes(raw)["result_sha256"]
    )

    mutated = exact_batch.copy()
    mutated[0, 0, 0, 0] = np.nextafter(mutated[0, 0, 0, 0], np.inf)
    with pytest.raises(ValueError):
        validate_e0_result_inputs(
            raw,
            exact=mutated,
            predicted=exact_batch.copy(),
            exact_preclip_norms=norms,
            asgcv_preclip_norms=norms.copy(),
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
    assert value["schema"] == "sfora-asgcv-gradient-sample-v1"
    assert value["claim_eligible"] is False
    assert value["pair_ordinals"] == [17, 29]
    assert value["relation_sign"] == -1
    assert value["replay_branch_count"] == 8
    assert value["losses"] == {"attention_kl": 0.375, "grpo": 0.125, "semantic": 0.5}
    assert value["arrays"]["patch_tokens"]["shape"] == [2, 3, 4]
    assert value["arrays"]["patch_tokens"]["dtype"] == "float32-le"
    assert validate_gradient_sample_inputs(
        raw,
        patch_tokens=tokens,
        exact_gradient=gradient,
    )["sample_sha256"] == value["sample_sha256"]


def test_gradient_sample_rejects_identity_type_shape_and_array_drift() -> None:
    raw = _gradient_sample_bytes()
    tokens, gradient = _gradient_sample_arrays()
    mutations = []
    for path, replacement in (
        (("claim_eligible",), True),
        (("pair_ordinals",), [17, 17]),
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
