import numpy as np
import pytest

from sfora.losses import redundant_connectivity_loss


def test_redundant_connectivity_matches_weighted_triangle_tree_mass() -> None:
    # Three equally similar points have three spanning trees, each with weight
    # w^2.  The negative log tree mass is therefore -log(3*w^2).
    embeddings = np.asarray(
        [[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0], [-0.5, np.sqrt(3.0) / 2.0]],
        dtype=np.float64,
    )
    got = redundant_connectivity_loss(embeddings, tau=0.0, scale=1.0, eps=1e-12)
    w = 1.0 / (1.0 + np.exp(-1.0))
    assert got == pytest.approx(-np.log(3.0 * w * w), rel=1e-8)


def test_redundant_connectivity_prefers_two_redundant_paths() -> None:
    redundant = np.asarray([[1.0, 0.0], [0.99, 0.141067], [0.98, 0.198997]], dtype=np.float64)
    chain = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    assert redundant_connectivity_loss(redundant, tau=0.5, scale=0.1) < redundant_connectivity_loss(
        chain, tau=0.5, scale=0.1
    )
