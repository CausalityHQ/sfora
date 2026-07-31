import numpy as np

from sfora.ipsr import build_ipsr_preferences


def test_ipsr_selects_closest_compatible_peer_that_is_still_contradicted() -> None:
    anchors = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],  # compatible, farther
            [0.9, 0.4359],  # compatible, closer: must be selected
            [0.99, 0.1411],  # incompatible and closest
        ]
    )
    signatures = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]]
    )
    result = build_ipsr_preferences(
        anchors,
        signatures,
        np.zeros(4, dtype=np.int64),
        np.ones(4, dtype=bool),
    )
    assert result.preferred_indices[0] == 2
    assert result.unknown_indices[0] == 3
    assert result.mean_initial_loss > np.log(2.0)


def test_ipsr_does_not_invent_preference_without_geometric_contradiction() -> None:
    anchors = np.array([[1.0, 0.0], [0.99, 0.1], [0.8, 0.6]])
    signatures = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    result = build_ipsr_preferences(
        anchors,
        signatures,
        np.zeros(3, dtype=np.int64),
        np.ones(3, dtype=bool),
    )
    assert result.preferred_indices[0] == -1
    assert result.unknown_indices[0] == -1
