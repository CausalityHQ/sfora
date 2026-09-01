from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sfora.asgcv import (
    AsgcvSrhtAuthority,
    canonical_e0_result_bytes,
    select_stratum_index,
)
from sfora.asgcv_bias import (
    canonical_e0_selection_audit_bytes,
    projected_mean_error_potentials,
    randomization_mean_p_value_ppm,
    randomization_selection_indices,
    validate_e0_selection_audit_bytes,
)


def _manual_draw(seed: str, draw: int, strata: int) -> tuple[int, ...]:
    values: list[int] = []
    block = 0
    while len(values) < strata:
        digest = hashlib.sha256(
            b"sfora-asgcv-e0-mean-null-v1\0"
            + bytes.fromhex(seed)
            + draw.to_bytes(8, "big")
            + block.to_bytes(8, "big")
        ).digest()
        values.extend(byte & 7 for byte in digest)
        block += 1
    return tuple(values[:strata])


def test_randomization_selection_stream_is_source_bound_and_exact() -> None:
    seed = "12" * 32
    observed = randomization_selection_indices(seed, draw_count=3, stratum_count=64)
    assert observed.dtype == np.uint8
    assert observed.shape == (3, 64)
    assert tuple(int(value) for value in observed[0]) == _manual_draw(seed, 0, 64)
    assert tuple(int(value) for value in observed[2]) == _manual_draw(seed, 2, 64)
    np.testing.assert_array_equal(
        observed,
        randomization_selection_indices(seed, draw_count=3, stratum_count=64),
    )
    assert not np.array_equal(
        observed,
        randomization_selection_indices("34" * 32, draw_count=3, stratum_count=64),
    )


def test_randomization_selection_stream_rejects_unbounded_or_concrete_type_inputs() -> None:
    for seed, draws, strata in (
        (True, 3, 64),
        ("12" * 32, True, 64),
        ("12" * 32, 0, 64),
        ("12" * 32, 10_001, 64),
        ("12" * 32, 3, True),
        ("12" * 32, 3, 0),
        ("12" * 32, 3, 513),
    ):
        with pytest.raises(ValueError):
            randomization_selection_indices(seed, draw_count=draws, stratum_count=strata)


def test_randomization_mean_gate_is_exact_and_detects_extreme_realized_bias() -> None:
    strata = 64
    scalar = np.arange(8, dtype=np.float64) - 3.5
    potential = np.broadcast_to(scalar[None, :, None], (strata, 8, 1)).copy()
    null = randomization_selection_indices("56" * 32, draw_count=4096, stratum_count=strata)

    extreme = np.full(strata, 7, dtype=np.uint8)
    assert randomization_mean_p_value_ppm(potential, extreme, null) < 1_000

    ordinary = null[0].copy()
    observed = randomization_mean_p_value_ppm(potential, ordinary, null[1:])
    assert 0 <= observed <= 1_000_000
    assert (
        randomization_mean_p_value_ppm(
            np.zeros_like(potential),
            ordinary,
            null[1:],
        )
        == 1_000_000
    )


def test_randomization_mean_gate_rejects_shape_type_and_nonfinite_drift() -> None:
    potential = np.zeros((2, 8, 3), dtype=np.float64)
    observed = np.zeros(2, dtype=np.uint8)
    null = np.zeros((3, 2), dtype=np.uint8)
    for bad_potential, bad_observed, bad_null in (
        (potential.astype(np.float32), observed, null),
        (potential[:, :7], observed, null),
        (potential, observed.astype(np.int64), null),
        (potential, observed[:1], null),
        (potential, observed, null.astype(np.int64)),
        (potential, observed, null[:, :1]),
    ):
        with pytest.raises(ValueError):
            randomization_mean_p_value_ppm(bad_potential, bad_observed, bad_null)
    potential[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        randomization_mean_p_value_ppm(potential, observed, null)


def test_projected_error_potentials_preserve_patch_cancellation_structure() -> None:
    exact = np.arange(2 * 8 * 2 * 3 * 4, dtype=np.float64).reshape(2, 8, 2, 3, 4) / 17.0
    predicted = exact + np.linspace(-0.4, 0.3, 8, dtype=np.float64)[None, :, None, None, None]
    srht = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="78" * 32,
    ).validated()

    observed = projected_mean_error_potentials(exact, predicted, srht)
    np.testing.assert_allclose(observed.mean(axis=1), 0.0, atol=1e-15, rtol=0.0)
    cancelling = np.zeros_like(exact)
    cancelling[:, :, 1, 0, 0] = 1.0
    cancelling[:, :, 1, 1, 0] = 2.0
    cancelling[:, :, 1, 2, 0] = -3.0
    cancelling[:, 0] *= 2.0
    structured = projected_mean_error_potentials(cancelling, np.zeros_like(cancelling), srht)
    assert bool(np.count_nonzero(structured))

    with pytest.raises(ValueError):
        projected_mean_error_potentials(exact.astype(np.float32), predicted, srht)
    with pytest.raises(ValueError):
        projected_mean_error_potentials(exact, predicted[..., :3], srht)


def _selection_audit_result(
    exact: np.ndarray,
    predicted: np.ndarray,
    srht: AsgcvSrhtAuthority,
) -> bytes:
    return canonical_e0_result_bytes(
        source_commit="1" * 40,
        dataset_manifest_sha256="2" * 64,
        partition_manifest_sha256="3" * 64,
        predictor_state_sha256="4" * 64,
        selection_seed_sha256="ab" * 32,
        exact=exact,
        predicted=predicted,
        srht_authority=srht,
        peak_cuda_reserved_bytes=1,
        exact_semantic_wall_ns=10,
        asgcv_semantic_wall_ns=1,
    )


def test_selection_audit_derives_seeds_indices_and_potentials_from_e0() -> None:
    exact = np.arange(64 * 8 * 2 * 3 * 4, dtype=np.float64).reshape(64, 8, 2, 3, 4) / 17.0
    predicted = exact + np.linspace(-0.4, 0.3, 8, dtype=np.float64)[None, :, None, None, None]
    srht = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="78" * 32,
    ).validated()
    perfect_result = _selection_audit_result(exact, exact.copy(), srht)
    perfect = validate_e0_selection_audit_bytes(
        canonical_e0_selection_audit_bytes(perfect_result, exact=exact, predicted=exact.copy())
    )
    assert perfect["selection_independence_p_value_ppm"] == 1_000_000
    assert perfect["selection_independence_z_ppm"] == 0
    assert perfect["randomization_draws"] == 10_000

    for ordinal in range(64):
        selected = select_stratum_index(
            "ab" * 32,
            optimizer_step=0,
            stratum_ordinal=ordinal,
        )
        predicted[ordinal, selected] += 10.0
    result = _selection_audit_result(exact, predicted, srht)
    raw = canonical_e0_selection_audit_bytes(result, exact=exact, predicted=predicted)
    observed = validate_e0_selection_audit_bytes(raw)
    assert observed["selection_independence_p_value_ppm"] < 20_000
    expected_null_seed = hashlib.sha256(
        b"sfora-asgcv-e0-mean-null-seed-v1\0" + bytes.fromhex(json.loads(result)["result_sha256"])
    ).hexdigest()
    assert observed["null_seed_sha256"] == expected_null_seed

    with pytest.raises(ValueError):
        canonical_e0_selection_audit_bytes(
            result,
            exact=exact[:63],
            predicted=predicted[:63],
        )


def test_projected_selection_bias_z_matches_diagonal_free_closed_form() -> None:
    exact = np.arange(64 * 8 * 2 * 3 * 4, dtype=np.float64).reshape(64, 8, 2, 3, 4) / 17.0
    predicted = exact + np.linspace(-0.4, 0.3, 8, dtype=np.float64)[None, :, None, None, None]
    srht = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="78" * 32,
    ).validated()
    seed = "ab" * 32
    potentials = projected_mean_error_potentials(exact, predicted, srht)
    indices = np.asarray(
        [
            select_stratum_index(seed, optimizer_step=0, stratum_ordinal=ordinal)
            for ordinal in range(64)
        ],
        dtype=np.uint8,
    )
    selected = potentials[np.arange(64), indices]
    observed_u = 0.0
    for left in range(64):
        for right in range(64):
            if left != right:
                observed_u += float(np.dot(selected[left], selected[right]))
    covariance = np.einsum("sip,siq->spq", potentials, potentials) / 8.0
    variance = 0.0
    for left in range(64):
        for right in range(left + 1, 64):
            variance += 4.0 * float(np.sum(covariance[left] * covariance[right]))
    expected = int(round(abs(observed_u) / np.sqrt(variance) * 1_000_000))

    result = _selection_audit_result(exact, predicted, srht)
    audit = validate_e0_selection_audit_bytes(
        canonical_e0_selection_audit_bytes(result, exact=exact, predicted=predicted)
    )
    assert audit["selection_independence_z_ppm"] == expected
