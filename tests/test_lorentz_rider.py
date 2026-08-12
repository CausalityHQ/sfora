from __future__ import annotations

import numpy as np
import pytest

from sfora.lorentz_rider import (
    FrozenPCA,
    apply_frozen_pca,
    column_permutation_null,
    delta_bruteforce,
    fit_frozen_pca,
    gromov_delta_rel,
    l0_subsample_indices,
    lorentz_distance_block,
    lorentz_lift,
    lorentz_mips_scores,
    medoid_index,
    pairwise_chord_distances,
    scale_for_target_median,
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
    assert gromov_delta_rel(cycle4, 0).relative == 1.0
    assert gromov_delta_rel(square, base).relative == pytest.approx(2.0 - np.sqrt(2.0))
    assert gromov_delta_rel(17.0 * square, base).relative == pytest.approx(
        gromov_delta_rel(square, base).relative
    )


def test_l0_subsamples_are_exact_sorted_and_reproducible() -> None:
    first = l0_subsample_indices(25_882, 0, 0)

    assert first.shape == (2_000,)
    assert first.dtype == np.int64
    assert np.array_equal(first, np.sort(first))
    assert np.unique(first).size == 2_000
    assert np.array_equal(first, l0_subsample_indices(25_882, 0, 0))
    assert not np.array_equal(first, l0_subsample_indices(25_882, 0, 1))
    assert not np.array_equal(first, l0_subsample_indices(25_882, 1, 0))


def test_medoid_uses_minimum_row_sum_and_lowest_tie() -> None:
    distances = np.asarray(
        [[0, 1, 4, 4], [1, 0, 2, 2], [4, 2, 0, 3], [4, 2, 3, 0]], dtype=np.float64
    )
    assert medoid_index(distances) == 1
    tied = np.asarray([[0, 1, 2], [1, 0, 2], [2, 2, 0]], dtype=np.float64)
    assert medoid_index(tied) == 0


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
    generator = np.random.Generator(np.random.PCG64(9))
    values = generator.normal(size=(40, 5)).astype(np.float32)
    values[:, 1] = values[:, 0]
    permuted = column_permutation_null(values, 7500)

    for column in range(values.shape[1]):
        assert np.array_equal(np.sort(permuted[:, column]), np.sort(values[:, column]))
    assert not np.array_equal(permuted[:, 0], permuted[:, 1])
    gaussian = spectrum_gaussian_null(values, 7500)
    assert gaussian.shape == values.shape
    assert gaussian.dtype == np.float32
    assert np.isfinite(gaussian).all()
    assert np.allclose(np.mean(gaussian, axis=0), np.mean(values, axis=0), atol=2e-6)
    observed_spectrum = np.linalg.eigvalsh(
        np.cov(values.astype(np.float64), rowvar=False, bias=True)
    )
    null_spectrum = np.linalg.eigvalsh(
        np.cov(gaussian.astype(np.float64), rowvar=False, bias=True)
    )
    assert np.allclose(null_spectrum, observed_spectrum, rtol=2e-5, atol=2e-7)
    assert column_permutation_null(values.astype(np.float64), 7500).dtype == np.float32


def test_pca_uses_descending_variance_components_in_coordinate_order() -> None:
    train = np.asarray(
        [
            [10, 0, 0],
            [-10, 0, 0],
            [0, 3, 0],
            [0, -3, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.float32,
    )

    fit = fit_frozen_pca(train, 2)

    assert type(fit) is FrozenPCA
    assert fit.mean.tolist() == [0.0, 0.0, 0.0]
    assert np.array_equal(
        fit.components,
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )


def test_pca_component_sign_uses_lowest_loading_index_on_tie() -> None:
    train = np.asarray([[1, -1], [-1, 1], [2, -2], [-2, 2]], dtype=np.float32)

    fit = fit_frozen_pca(train, 1)

    expected = np.float32(1.0 / np.sqrt(2.0))
    assert fit.components[0, 0] == pytest.approx(expected)
    assert fit.components[0, 1] == pytest.approx(-expected)


def test_apply_frozen_pca_uses_frozen_train_mean_not_eval_mean() -> None:
    fit = FrozenPCA(
        mean=np.asarray([1, 2], dtype=np.float32),
        components=np.asarray([[1, 0], [0, 1]], dtype=np.float32),
    )
    evaluation = np.asarray([[10, 7], [14, -1], [-2, 11]], dtype=np.float32)

    projected = apply_frozen_pca(evaluation, fit)

    assert np.array_equal(
        projected,
        np.asarray([[9, 5], [13, -3], [-3, 9]], dtype=np.float32),
    )


def test_pca_canonicalization_rejects_eval_leakage_and_mutants() -> None:
    generator = np.random.Generator(np.random.PCG64(41))
    train = generator.normal(size=(12, 4)).astype(np.float32)
    first = fit_frozen_pca(train, 3)
    mutated_eval = np.full((2, 4), 10_000.0, dtype=np.float32)

    assert fit_frozen_pca(train, 3) == first
    projected = apply_frozen_pca(mutated_eval, first)
    assert np.isfinite(projected).all()
    assert np.array_equal(first.mean, np.mean(train.astype(np.float64), axis=0).astype(np.float32))


def test_scale_uses_exact_target_median_and_rejects_degenerate_data() -> None:
    train = np.asarray([[3, 4], [0, 2], [0, 0]], dtype=np.float32)
    scale = scale_for_target_median(train, 1.0)
    radii = np.linalg.norm(train.astype(np.float64), axis=1)
    assert scale == 1.0 / float(np.median(radii))
    with pytest.raises(ValueError, match="median"):
        scale_for_target_median(np.zeros((3, 2), dtype=np.float32), 1.0)


def test_lift_satisfies_hyperboloid_and_mips_selects_nearest() -> None:
    values = np.asarray([[1, 0], [-1, 0], [0, 0]], dtype=np.float32)

    lifted = lorentz_lift(values, 1.0)

    constraint = -(lifted[:, 0] ** 2) + np.sum(lifted[:, 1:] ** 2, axis=1)
    assert np.allclose(constraint, -1.0, atol=1e-6)
    assert lifted[2].tolist() == [1.0, 0.0, 0.0]
    scores = lorentz_mips_scores(lifted[:1], lifted)
    assert int(np.argmax(scores[0])) == 0


def test_lift_clips_radius_at_registered_maximum_without_changing_direction() -> None:
    values = np.asarray([[30, 40], [-6, 8]], dtype=np.float32)

    lifted = lorentz_lift(values, 1.0)

    radius = np.arccosh(lifted[:, 0].astype(np.float64))
    assert np.allclose(radius, 2.5, atol=2e-6, rtol=0.0)
    expected_directions = np.asarray([[0.6, 0.8], [-0.6, 0.8]], dtype=np.float32)
    observed_directions = lifted[:, 1:] / np.linalg.norm(
        lifted[:, 1:], axis=1, keepdims=True
    )
    assert np.allclose(observed_directions, expected_directions, atol=1e-7, rtol=0.0)


def test_wrong_sign_selects_farthest() -> None:
    lifted = lorentz_lift(np.asarray([[1, 0], [-1, 0]], dtype=np.float32), 1.0)
    wrong_query = lifted[:1].copy()
    wrong_query[:, 1:] *= -1

    assert int(np.argmax(wrong_query @ lifted.T)) == 1
    assert int(np.argmax(lorentz_mips_scores(lifted[:1], lifted)[0])) == 0
    expected = -lifted[0, 0] * lifted[:, 0] + lifted[0, 1:] @ lifted[:, 1:].T
    assert np.array_equal(lorentz_mips_scores(lifted[:1], lifted)[0], expected)


def test_ambient_euclidean_dot_is_not_a_lorentz_score() -> None:
    query = np.asarray([[0.18905339, -0.52274847]], dtype=np.float32)
    gallery = np.asarray(
        [[-0.41306356, -2.4414673], [1.7997074, 1.1441659], [-0.32542282, 0.7738066]],
        dtype=np.float32,
    )
    lifted_query = lorentz_lift(query, 1.0)
    lifted_gallery = lorentz_lift(gallery, 1.0)

    assert int(np.argmax(lifted_query @ lifted_gallery.T)) == 0
    assert int(np.argmax(lorentz_mips_scores(lifted_query, lifted_gallery)[0])) == 2


def test_fp32_distance_matches_nonnegative_fp64_oracle() -> None:
    values = np.asarray(
        [[1, 0], [np.cos(1e-4), np.sin(1e-4)], [-1, 0]], dtype=np.float32
    )
    lifted = lorentz_lift(values, 2.5)

    actual = lorentz_distance_block(lifted, lifted)

    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.all(actual >= 0.0)
    assert np.allclose(np.diag(actual), 0.0, atol=2e-4)
    assert np.array_equal(np.argsort(actual[0], kind="stable"), np.asarray([0, 1, 2]))
    a = np.arccosh(lifted[:, 0].astype(np.float64))
    sinh_a = np.sinh(a)
    direction = np.zeros_like(lifted[:, 1:], dtype=np.float64)
    nonzero = sinh_a > 0.0
    direction[nonzero] = lifted[nonzero, 1:].astype(np.float64) / sinh_a[nonzero, None]
    radius_difference = a[:, None] - a[None, :]
    direction_difference = direction[:, None, :] - direction[None, :, :]
    oracle_u = 2.0 * np.sinh(radius_difference / 2.0) ** 2 + (
        sinh_a[:, None]
        * sinh_a[None, :]
        * 0.5
        * np.sum(direction_difference**2, axis=2, dtype=np.float64)
    )
    oracle = np.log1p(oracle_u + np.sqrt(oracle_u * (oracle_u + 2.0)))
    assert np.allclose(actual.astype(np.float64), oracle, atol=3e-4, rtol=3e-4)


def test_distance_uses_ordered_fp32_intermediates() -> None:
    values = np.asarray(
        [
            [-0.9891214, -0.36778665, 1.2879252, 0.19397442, 0.9202309],
            [0.09716732, -1.5259304, 1.1921661, -0.67108965, 1.0002694],
        ],
        dtype=np.float32,
    )
    lifted = lorentz_lift(values, 0.30731173920785104)

    actual = lorentz_distance_block(lifted[:1], lifted[1:])
    spatial = lifted[:1, None, 1:] - lifted[None, 1:, 1:]
    temporal = lifted[:1, None, 0] - lifted[None, 1:, 0]
    expected_u = (
        np.sum(spatial * spatial, axis=2, dtype=np.float32) - temporal * temporal
    ) / np.float32(2.0)
    np.maximum(expected_u, np.float32(0.0), out=expected_u)
    expected = np.log1p(
        expected_u + np.sqrt(expected_u * (expected_u + np.float32(2.0)))
    ).astype(np.float32)

    assert np.array_equal(actual, expected)
    spatial64 = spatial.astype(np.float64)
    temporal64 = temporal.astype(np.float64)
    u64 = (np.sum(spatial64 * spatial64, axis=2) - temporal64 * temporal64) / 2.0
    widened = np.log1p(u64 + np.sqrt(u64 * (u64 + 2.0))).astype(np.float32)
    assert not np.array_equal(actual, widened)


def test_distance_clips_accepted_negative_roundoff_to_zero() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    gallery = np.asarray([[np.nextafter(np.float32(1.0), np.float32(2.0)), 0.0]], dtype=np.float32)

    distance = lorentz_distance_block(query, gallery)

    assert np.array_equal(distance, np.zeros((1, 1), dtype=np.float32))


@pytest.mark.parametrize(
    "bad",
    [
        np.asfortranarray(np.ones((3, 2), dtype=np.float32)),
        np.ones((3, 2), dtype=np.float16),
        np.asarray([[1, 0], [np.nan, 1]], dtype=np.float32),
    ],
)
def test_public_arithmetic_rejects_noncanonical_matrices(bad: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_frozen_pca(bad, 1)


def test_public_arithmetic_rejects_ndarray_subclasses() -> None:
    class ArraySubclass(np.ndarray):
        pass

    bad = np.ones((3, 2), dtype=np.float32).view(ArraySubclass)

    with pytest.raises(TypeError):
        fit_frozen_pca(bad, 1)


def test_lorentz_scoring_rejects_off_manifold_inputs() -> None:
    valid = lorentz_lift(np.asarray([[1, 0], [0, 1]], dtype=np.float32), 1.0)
    invalid = valid.copy()
    invalid[0, 0] += np.float32(1e-3)

    with pytest.raises(ValueError, match="constraint"):
        lorentz_mips_scores(invalid, valid)


def test_small_scale_distance_converges_to_euclidean() -> None:
    values = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    scale = 1e-3
    lifted = lorentz_lift(values, scale)
    distance = float(lorentz_distance_block(lifted[:1], lifted[1:])[0, 0])
    expected = float(np.linalg.norm(values[0].astype(np.float64) - values[1]))
    assert distance / scale == pytest.approx(expected, rel=2e-4)
