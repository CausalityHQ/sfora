import numpy as np
import pytest

from sfora.losses import group_triplet_margin_loss, triplet_margin_loss


def test_triplet_margin_loss_is_zero_when_negative_is_far_enough() -> None:
    anchors = np.array([[0.0, 0.0]])
    positives = np.array([[0.1, 0.0]])
    negatives = np.array([[2.0, 0.0]])

    loss = triplet_margin_loss(anchors, positives, negatives, margin=0.5)

    assert loss == pytest.approx(0.0)


def test_triplet_margin_loss_penalizes_margin_violations() -> None:
    anchors = np.array([[0.0, 0.0]])
    positives = np.array([[1.0, 0.0]])
    negatives = np.array([[1.2, 0.0]])

    loss = triplet_margin_loss(anchors, positives, negatives, margin=0.5)

    assert loss == pytest.approx(0.3)


def test_group_triplet_matches_standard_triplet_for_singleton_groups_without_group_terms() -> None:
    anchors = np.array([[[0.0, 0.0]]])
    positives = np.array([[[1.0, 0.0]]])
    negatives = np.array([[[1.2, 0.0]]])

    grouped = group_triplet_margin_loss(
        anchors,
        positives,
        negatives,
        margin=0.5,
        hard_weight=0.0,
        spread_weight=0.0,
    )
    standard = triplet_margin_loss(
        anchors[:, 0, :],
        positives[:, 0, :],
        negatives[:, 0, :],
        margin=0.5,
    )

    assert grouped == pytest.approx(standard)


def test_group_triplet_penalizes_dispersed_positive_groups_with_good_centroids() -> None:
    anchors = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    positives = np.array([[[-2.0, 0.0], [2.0, 0.0]]])
    negatives = np.array([[[3.0, 0.0], [3.0, 0.0]]])

    centroid_only = group_triplet_margin_loss(
        anchors,
        positives,
        negatives,
        margin=0.1,
        hard_weight=0.0,
        spread_weight=0.0,
    )
    group_aware = group_triplet_margin_loss(
        anchors,
        positives,
        negatives,
        margin=0.1,
        hard_weight=0.0,
        spread_weight=0.25,
    )

    assert centroid_only == pytest.approx(0.0)
    assert group_aware > 0.0
