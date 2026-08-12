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
    l1_decision,
    lorentz_distance_block,
    lorentz_lift,
    lorentz_mips_scores,
    medoid_index,
    paired_identity_max_interval,
    pairwise_chord_distances,
    scale_for_target_median,
    score_control_blocks,
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
    assert 2.0 * delta_bruteforce(square) / square.max() == pytest.approx(2.0 - np.sqrt(2.0))
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
    assert first[:5].tolist() == [5, 20, 26, 32, 51]
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


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64),
        np.asarray([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64),
        np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64),
    ],
)
def test_distance_consumers_reject_invalid_metric_matrices(bad: np.ndarray) -> None:
    for consumer in (medoid_index, delta_bruteforce):
        with pytest.raises(ValueError):
            consumer(bad)


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
    null_spectrum = np.linalg.eigvalsh(np.cov(gaussian.astype(np.float64), rowvar=False, bias=True))
    assert np.allclose(null_spectrum, observed_spectrum, rtol=2e-5, atol=2e-7)
    assert column_permutation_null(values.astype(np.float64), 7500).dtype == np.float32
    assert np.allclose(
        gaussian[0],
        np.asarray(
            [0.7399556, 0.7399556, 0.9934853, -0.47605044, 0.6247305],
            dtype=np.float32,
        ),
        atol=5e-7,
        rtol=0.0,
    )


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


def test_pca_components_diagonalize_covariance_in_descending_order() -> None:
    train = np.random.Generator(np.random.PCG64(41)).normal(size=(12, 4)).astype(np.float32)

    fit = fit_frozen_pca(train, 3)

    centered = train.astype(np.float64) - fit.mean.astype(np.float64)
    covariance = centered.T @ centered / train.shape[0]
    rotated = fit.components.astype(np.float64) @ covariance @ fit.components.T
    assert np.allclose(rotated - np.diag(np.diag(rotated)), np.zeros((3, 3)), atol=2e-7, rtol=0.0)
    assert np.all(np.diff(np.diag(rotated)) < 0.0)
    assert np.allclose(
        fit.components @ fit.components.T,
        np.eye(3, dtype=np.float32),
        atol=2e-7,
        rtol=0.0,
    )


def test_pca_component_sign_uses_largest_loading_not_first_coordinate() -> None:
    train = np.random.Generator(np.random.PCG64(1)).normal(size=(20, 4)).astype(np.float32)

    fit = fit_frozen_pca(train, 3)

    for component in fit.components:
        pivot = int(np.argmax(np.abs(component)))
        assert component[pivot] > 0.0
    assert int(np.argmax(np.abs(fit.components[1]))) == 2


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
    observed_directions = lifted[:, 1:] / np.linalg.norm(lifted[:, 1:], axis=1, keepdims=True)
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
    values = np.asarray([[1, 0], [np.cos(1e-4), np.sin(1e-4)], [-1, 0]], dtype=np.float32)
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
    expected = np.log1p(expected_u + np.sqrt(expected_u * (expected_u + np.float32(2.0)))).astype(
        np.float32
    )

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
    fit = FrozenPCA(
        mean=np.zeros(2, dtype=np.float32),
        components=np.eye(2, dtype=np.float32),
    )
    with pytest.raises((TypeError, ValueError)):
        apply_frozen_pca(bad, fit)
    with pytest.raises((TypeError, ValueError)):
        lorentz_lift(bad, 1.0)


def test_public_arithmetic_rejects_ndarray_subclasses() -> None:
    class ArraySubclass(np.ndarray):
        pass

    bad = np.ones((3, 2), dtype=np.float32).view(ArraySubclass)

    with pytest.raises(TypeError):
        fit_frozen_pca(bad, 1)
    fit = FrozenPCA(
        mean=np.zeros(2, dtype=np.float32),
        components=np.eye(2, dtype=np.float32),
    )
    with pytest.raises(TypeError):
        apply_frozen_pca(bad, fit)
    with pytest.raises(TypeError):
        lorentz_lift(bad, 1.0)


def test_lorentz_scoring_rejects_off_manifold_inputs() -> None:
    valid = lorentz_lift(np.asarray([[1, 0], [0, 1]], dtype=np.float32), 1.0)
    invalid = valid.copy()
    invalid[0, 0] += np.float32(1e-3)

    with pytest.raises(ValueError, match="constraint"):
        lorentz_mips_scores(invalid, valid)


def test_lorentz_scoring_rejects_lower_hyperboloid_sheet() -> None:
    valid = lorentz_lift(np.asarray([[1, 0], [0, 1]], dtype=np.float32), 1.0)
    lower_sheet = valid.copy()
    lower_sheet[:, 0] *= np.float32(-1.0)

    with pytest.raises(ValueError, match="time coordinates"):
        lorentz_mips_scores(lower_sheet, valid)


def test_small_scale_distance_converges_to_euclidean() -> None:
    values = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    scale = 1e-3
    lifted = lorentz_lift(values, scale)
    distance = float(lorentz_distance_block(lifted[:1], lifted[1:])[0, 0])
    expected = float(np.linalg.norm(values[0].astype(np.float64) - values[1]))
    assert distance / scale == pytest.approx(expected, rel=2e-4)


def test_l1_cannot_pass_from_endpoint_difference_alone() -> None:
    labels = np.asarray(["a", "a", "b", "b"])
    euclidean = np.asarray([True, True, False, False])
    cosine = np.asarray([False, False, True, True])
    interiors = np.stack([euclidean, cosine])

    result = paired_identity_max_interval(
        euclidean,
        interiors,
        labels,
        samples=200,
        seed=205,
    )

    assert result.point <= 0.0
    assert result.lower <= 0.0
    assert len(result.replicates_sha256) == 64


def test_identity_bootstrap_resamples_whole_clusters_and_recomputes_maximum() -> None:
    labels = np.asarray(["a", "a", "b", "c", "c", "c"])
    baseline = np.asarray([True, False, True, False, False, False])
    interiors = np.asarray(
        [
            [True, True, False, False, False, True],
            [False, False, True, True, True, False],
        ],
        dtype=np.bool_,
    )

    actual = paired_identity_max_interval(baseline, interiors, labels, samples=32, seed=205)

    groups = (np.asarray([0, 1]), np.asarray([2]), np.asarray([3, 4, 5]))
    generator = np.random.Generator(np.random.PCG64(205))
    replicates = []
    for sampled in generator.integers(0, 3, size=(32, 3)):
        indices = np.concatenate([groups[index] for index in sampled])
        gains = interiors[:, indices].mean(axis=1) - baseline[indices].mean()
        replicates.append(float(np.max(gains)))
    bounds = np.percentile(np.asarray(replicates), [2.5, 97.5])
    point = float(np.max(interiors.mean(axis=1) - baseline.mean()))

    assert actual.point == point
    assert actual.lower == pytest.approx(float(bounds[0]))
    assert actual.upper == pytest.approx(float(bounds[1]))
    assert actual.standard_errors == pytest.approx(point / np.std(replicates, ddof=1))


def test_score_controls_follow_registered_radial_formulas_and_order() -> None:
    query = np.asarray([[3, 4], [0, 2]], dtype=np.float32)
    gallery = np.asarray([[4, 3], [-3, 4], [0, 1]], dtype=np.float32)
    scale = 0.2
    clip = 2.5

    score_chunks = score_control_blocks(query, gallery, scale, clip)

    assert list(score_chunks) == [
        "lorentz",
        "pca_euclidean",
        "pca_cosine",
        "spatial_only",
        "power_1",
        "power_3",
    ]
    scores = {}
    for name, chunks in score_chunks.items():
        materialized = tuple(chunks)
        assert materialized
        assert all(chunk.ndim == 2 for chunk in materialized)
        scores[name] = np.concatenate(materialized, axis=0)
    query64 = query.astype(np.float64)
    gallery64 = gallery.astype(np.float64)
    query_radius = np.minimum(scale * np.linalg.norm(query64, axis=1), clip)
    gallery_radius = np.minimum(scale * np.linalg.norm(gallery64, axis=1), clip)
    cosine = (query64 @ gallery64.T) / (
        np.linalg.norm(query64, axis=1)[:, None] * np.linalg.norm(gallery64, axis=1)[None, :]
    )
    expected_lorentz = (
        np.tanh(query_radius)[:, None] * np.sinh(gallery_radius)[None, :] * cosine
        - np.cosh(gallery_radius)[None, :]
    )
    expected_spatial = (
        np.sinh(query_radius)[:, None] * np.sinh(gallery_radius)[None, :] * cosine
        - 0.5 * np.sinh(gallery_radius)[None, :] ** 2
    )
    assert np.allclose(scores["lorentz"], expected_lorentz, atol=2e-7)
    assert np.allclose(scores["spatial_only"], expected_spatial, atol=2e-7)
    for power in (1, 3):
        query_h = query_radius**power
        gallery_h = gallery_radius**power
        expected_power = query_h[:, None] / np.sqrt(1.0 + query_h[:, None] ** 2) * gallery_h[
            None, :
        ] * cosine - np.sqrt(1.0 + gallery_h[None, :] ** 2)
        assert np.allclose(scores[f"power_{power}"], expected_power, atol=2e-7)


def test_score_control_endpoint_rankings_match_euclidean_and_cosine() -> None:
    query = np.asarray([[1.0, 0.2]], dtype=np.float32)
    gallery = np.asarray([[0.9, 0.4], [2.0, 0.1], [-0.2, 1.0]], dtype=np.float32)
    gallery = gallery[[2, 0, 1]]

    small_chunks = score_control_blocks(query, gallery, 0.125, 2.5)
    saturated_chunks = score_control_blocks(query, gallery, 1e6, 2.5)
    small = {name: np.concatenate(tuple(chunks), axis=0) for name, chunks in small_chunks.items()}
    saturated = {
        name: np.concatenate(tuple(chunks), axis=0) for name, chunks in saturated_chunks.items()
    }

    assert np.array_equal(
        np.argsort(-small["lorentz"][0], kind="stable"),
        np.argsort(-small["pca_euclidean"][0], kind="stable"),
    )
    assert np.array_equal(
        np.argsort(-saturated["lorentz"][0], kind="stable"),
        np.argsort(-saturated["pca_cosine"][0], kind="stable"),
    )


def test_score_controls_stream_bounded_query_chunks() -> None:
    query = np.ones((513, 2), dtype=np.float32)
    gallery = np.asarray([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    controls = score_control_blocks(query, gallery, 0.5, 2.5)

    for chunks in controls.values():
        materialized = tuple(chunks)
        assert [chunk.shape for chunk in materialized] == [
            (256, 3),
            (256, 3),
            (1, 3),
        ]


def test_distance_gallery_chunking_preserves_stable_formula() -> None:
    values = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)
    lifted = lorentz_lift(values, 1.0)

    chunked = lorentz_distance_block(lifted[:2], lifted, gallery_chunk_size=1)
    ordinary = lorentz_distance_block(lifted[:2], lifted, gallery_chunk_size=4)

    assert np.array_equal(chunked, ordinary)


def test_function_family_tie_closes_geometry_claim() -> None:
    tied = l1_decision(
        endpoint_gain=0.02,
        endpoint_lower=0.01,
        standard_errors=4.0,
        spatial_gain=0.01,
        power_gain=0.0,
    )
    surviving = l1_decision(
        endpoint_gain=0.02,
        endpoint_lower=0.01,
        standard_errors=4.0,
        spatial_gain=0.01,
        power_gain=0.01,
    )

    assert tied.status == "CLOSE_GEOMETRY"
    assert surviving.status == "GEOMETRY_SURVIVES"
