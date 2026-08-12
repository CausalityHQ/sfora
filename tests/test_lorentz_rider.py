from __future__ import annotations

import numpy as np
import pytest

from sfora.lorentz_rider import (
    column_permutation_null,
    delta_bruteforce,
    gromov_delta_rel,
    l0_subsample_indices,
    medoid_index,
    pairwise_chord_distances,
    spectrum_gaussian_null,
)


def test_four_point_oracles_and_scale_invariance() -> None:
    line = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]).astype(np.float64)
    cycle4 = np.asarray(
        [[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]],
        dtype=np.float64,
    )
    square = pairwise_chord_distances(
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    )

    assert delta_bruteforce(line) == 0.0
    assert 2.0 * delta_bruteforce(cycle4) / cycle4.max() == 1.0
    assert 2.0 * delta_bruteforce(square) / square.max() == pytest.approx(
        2.0 - np.sqrt(2.0)
    )
    base = medoid_index(square)
    assert gromov_delta_rel(17.0 * square, base).relative == pytest.approx(
        gromov_delta_rel(square, base).relative
    )


def test_l0_subsamples_are_exact_sorted_and_reproducible() -> None:
    first = l0_subsample_indices(25_882, 0, 0)

    assert first.shape == (2_000,)
    assert first.dtype == np.int64
    assert np.array_equal(first, np.sort(first))
    assert np.array_equal(first, l0_subsample_indices(25_882, 0, 0))
    assert not np.array_equal(first, l0_subsample_indices(25_882, 0, 1))


def test_fixed_base_estimate_is_bounded_by_bruteforce() -> None:
    generator = np.random.Generator(np.random.PCG64(17))
    values = generator.normal(size=(9, 3)).astype(np.float32)
    distances = pairwise_chord_distances(values)
    exact = delta_bruteforce(distances)

    for base in range(distances.shape[0]):
        estimate = gromov_delta_rel(distances, base)
        assert estimate.delta <= exact + 1e-12
        assert exact <= 2.0 * estimate.delta + 1e-12


def test_nulls_preserve_registered_structure() -> None:
    values = np.arange(60, dtype=np.float32).reshape(12, 5)
    permuted = column_permutation_null(values, 7500)

    for column in range(values.shape[1]):
        assert np.array_equal(np.sort(permuted[:, column]), np.sort(values[:, column]))
    gaussian = spectrum_gaussian_null(values, 7500)
    assert gaussian.shape == values.shape
    assert gaussian.dtype == np.float32
    assert np.isfinite(gaussian).all()
