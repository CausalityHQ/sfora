import numpy as np
import pytest

from sfora.losses import barrier_energy_loss, redundant_connectivity_loss


def test_barrier_energy_is_finite_and_uses_foreign_proxy_saddle() -> None:
    anchors = np.asarray([[1.0, 0.0]], dtype=np.float64)
    positives = np.asarray([[0.0, 1.0]], dtype=np.float64)
    proxies = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64
    )
    labels = np.asarray([0], dtype=np.int64)
    value = barrier_energy_loss(
        anchors, positives, proxies, labels, temperature=0.1, path_points=9
    )
    assert np.isfinite(value)
    assert value > 0.0


def test_barrier_energy_drops_when_foreign_proxy_moves_away() -> None:
    anchors = np.asarray([[1.0, 0.0]], dtype=np.float64)
    positives = np.asarray([[0.0, 1.0]], dtype=np.float64)
    near_foreign = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    far_foreign = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    labels = np.asarray([0], dtype=np.int64)
    near = barrier_energy_loss(anchors, positives, near_foreign, labels, temperature=0.1, path_points=9)
    far = barrier_energy_loss(anchors, positives, far_foreign, labels, temperature=0.1, path_points=9)
    assert np.isfinite(near)
    assert np.isfinite(far)
    assert near > far


def test_redundant_connectivity_matches_weighted_triangle_tree_mass() -> None:
    # Three equally similar points have three spanning trees, each with weight
    # w^2.  The negative log tree mass is therefore -log(3*w^2).
    embeddings = np.asarray(
        [[1.0, 0.0], [-0.5, np.sqrt(3.0) / 2.0], [-0.5, -np.sqrt(3.0) / 2.0]],
        dtype=np.float64,
    )
    got = redundant_connectivity_loss(embeddings, tau=0.0, scale=1.0, eps=1e-12)
    w = 1.0 / (1.0 + np.exp(0.5))
    assert got == pytest.approx(-np.log(3.0 * w * w), rel=1e-8)


def test_redundant_connectivity_prefers_two_redundant_paths() -> None:
    redundant = np.asarray([[1.0, 0.0], [0.99, 0.141067], [0.98, 0.198997]], dtype=np.float64)
    chain = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    assert redundant_connectivity_loss(redundant, tau=0.5, scale=0.1) < redundant_connectivity_loss(
        chain, tau=0.5, scale=0.1
    )
