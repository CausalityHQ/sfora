from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sfora.asgcv import AsgcvSrhtAuthority, srht_gradient_sketch
from sfora.asgcv_bias import (
    projected_mean_agreement_p_value_ppm,
    projected_mean_error_potentials,
    randomization_mean_p_value_ppm,
    randomization_selection_indices,
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
    assert randomization_mean_p_value_ppm(
        np.zeros_like(potential),
        ordinary,
        null[1:],
    ) == 1_000_000


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


def test_projected_mean_error_potentials_center_every_stratum_exactly() -> None:
    exact = np.arange(2 * 8 * 2 * 3 * 4, dtype=np.float64).reshape(2, 8, 2, 3, 4) / 17.0
    predicted = exact + np.linspace(-0.4, 0.3, 8, dtype=np.float64)[None, :, None, None, None]
    srht = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="78" * 32,
    ).validated()

    observed = projected_mean_error_potentials(exact, predicted, srht)
    projected = srht_gradient_sketch((exact - predicted).reshape(-1, 4), srht).reshape(
        2, 8, 2, 3, 2
    )
    reduced = projected.sum(axis=(2, 3), dtype=np.float64)
    expected = reduced - reduced.mean(axis=1, keepdims=True, dtype=np.float64)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_allclose(observed.mean(axis=1), 0.0, atol=1e-15, rtol=0.0)

    with pytest.raises(ValueError):
        projected_mean_error_potentials(exact.astype(np.float32), predicted, srht)
    with pytest.raises(ValueError):
        projected_mean_error_potentials(exact, predicted[..., :3], srht)


def test_projected_mean_agreement_derives_indices_and_potentials_from_authority() -> None:
    exact = np.arange(64 * 8 * 2 * 3 * 4, dtype=np.float64).reshape(64, 8, 2, 3, 4) / 17.0
    predicted = exact + np.linspace(-0.4, 0.3, 8, dtype=np.float64)[None, :, None, None, None]
    srht = AsgcvSrhtAuthority(
        input_dimensions=4,
        padded_dimensions=4,
        output_dimensions=2,
        seed_sha256="78" * 32,
    ).validated()
    assert projected_mean_agreement_p_value_ppm(
        exact,
        exact.copy(),
        srht,
        selection_seed_sha256="ab" * 32,
        null_seed_sha256="cd" * 32,
    ) == 1_000_000
    observed = projected_mean_agreement_p_value_ppm(
        exact,
        predicted,
        srht,
        selection_seed_sha256="ab" * 32,
        null_seed_sha256="cd" * 32,
    )
    assert 0 <= observed <= 1_000_000

    with pytest.raises(ValueError):
        projected_mean_agreement_p_value_ppm(
            exact,
            predicted,
            srht,
            selection_seed_sha256=True,
            null_seed_sha256="cd" * 32,
        )
