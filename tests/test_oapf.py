from __future__ import annotations

import numpy as np
import pytest

from sfora.oapf import (
    class_balanced_pair_weights,
    deterministic_view_seed,
    fit_standardizer,
    fit_weighted_logistic,
    fixed_class_folds,
    heldout_pair_compatibility,
    logistic_probabilities,
    orbit_radius_and_rms,
    same_class_pairs,
    weighted_auc,
    within_class_derangement,
)


def test_view_seed_is_stable_and_pack_specific() -> None:
    first = deterministic_view_seed("A", "inshop-train-img.jpg", 0)
    assert first == deterministic_view_seed("A", "inshop-train-img.jpg", 0)
    assert first != deterministic_view_seed("B", "inshop-train-img.jpg", 0)


def test_radius_uses_euclidean_q90_and_rms_is_distinct() -> None:
    canonical = np.asarray([[1.0, 0.0]], dtype=np.float64)
    views = np.asarray(
        [
            [
                [1.0, 0.0],
                [0.8, 0.6],
                [0.6, 0.8],
                [0.0, 1.0],
                [-0.6, 0.8],
                [-0.8, 0.6],
            ]
        ],
        dtype=np.float64,
    )
    radius, rms = orbit_radius_and_rms(canonical, views)
    expected = np.quantile(np.linalg.norm(views[0] - canonical[0], axis=1), 0.9)
    assert radius[0] == expected
    assert radius[0] != rms[0]


def test_radius_rejects_wrong_view_count_or_unnormalized_embeddings() -> None:
    canonical = np.asarray([[1.0, 0.0]], dtype=np.float64)
    six_views = np.repeat(canonical[:, None, :], 6, axis=1)
    with pytest.raises(ValueError, match="six"):
        orbit_radius_and_rms(canonical, six_views[:, :5])
    with pytest.raises(ValueError, match="L2-normalized"):
        orbit_radius_and_rms(canonical * 2.0, six_views)


def test_pair_outcome_counts_both_directions_across_six_views() -> None:
    canonical = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float64)
    views = np.repeat(canonical[:, None, :], 6, axis=1)
    nearest_negative = np.full((2, 6), 1.0, dtype=np.float64)
    continuous, binary = heldout_pair_compatibility(
        canonical,
        views,
        nearest_negative,
        np.asarray([0]),
        np.asarray([1]),
    )
    assert continuous.tolist() == [1.0]
    assert binary.tolist() == [1]


def test_pair_outcome_uses_ten_of_twelve_and_strict_positive_margin() -> None:
    canonical = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    views = np.repeat(canonical[:, None, :], 6, axis=1)
    nearest_negative = np.asarray(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    left = np.asarray([0], dtype=np.int64)
    right = np.asarray([1], dtype=np.int64)

    continuous, binary = heldout_pair_compatibility(canonical, views, nearest_negative, left, right)
    assert continuous.tolist() == [0.75]
    assert binary.tolist() == [0]

    nearest_negative[1, 4] = 1.0
    continuous, binary = heldout_pair_compatibility(canonical, views, nearest_negative, left, right)
    assert continuous.tolist() == [10.0 / 12.0]
    assert binary.tolist() == [1]


def test_pair_weights_give_classes_equal_mass() -> None:
    labels = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    left, right, pair_labels = same_class_pairs(labels)
    assert list(zip(left.tolist(), right.tolist(), strict=True)) == [
        (0, 1),
        (0, 2),
        (1, 2),
        (3, 4),
    ]
    weights = class_balanced_pair_weights(pair_labels)
    assert np.isclose(weights[pair_labels == 0].sum(), 0.5)
    assert np.isclose(weights[pair_labels == 1].sum(), 0.5)


def test_derangement_never_crosses_classes_or_leaves_multisample_values_fixed() -> None:
    values = np.arange(6, dtype=np.float64)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    permuted = within_class_derangement(values, labels, seed=174000)
    for label in (0, 1):
        mask = labels == label
        assert set(permuted[mask]) == set(values[mask])
        assert np.all(permuted[mask] != values[mask])


def test_derangement_swaps_size_two_and_preserves_singleton() -> None:
    values = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    permuted = within_class_derangement(values, labels, seed=174000)
    assert permuted.tolist() == [20.0, 10.0, 30.0]


def test_fixed_class_folds_are_deterministic_and_class_disjoint() -> None:
    labels = np.arange(20, dtype=np.int64)
    first = fixed_class_folds(labels)
    second = fixed_class_folds(labels)
    assert first == second
    assert set(first) == set(labels.tolist())
    assert set(first.values()) == set(range(5))
    assert all(list(first.values()).count(fold) == 4 for fold in range(5))


def test_weighted_auc_respects_sample_weights() -> None:
    targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.9, 0.1, 0.8, 0.2], dtype=np.float64)
    uniform = weighted_auc(targets, scores, np.ones(4, dtype=np.float64))
    emphasized = weighted_auc(targets, scores, np.asarray([10.0, 1.0, 1.0, 10.0], dtype=np.float64))
    assert emphasized < uniform


def test_constrained_radius_direction_cannot_flip_its_sign() -> None:
    radius = np.linspace(-2.0, 2.0, 40)[:, None]
    targets = (radius[:, 0] > 0.0).astype(np.int64)
    weights = np.ones(40, dtype=np.float64)
    standardizer = fit_standardizer(radius, weights)
    features = standardizer.apply(radius)
    real = fit_weighted_logistic(features, targets, weights, nonnegative_from=0)
    inverse = fit_weighted_logistic(-features, targets, weights, nonnegative_from=0)
    real_probability = logistic_probabilities(features, real)
    inverse_probability = logistic_probabilities(-features, inverse)
    assert real[-1] > 0.0
    assert inverse[-1] == 0.0
    assert real_probability[-1] > inverse_probability[-1]
