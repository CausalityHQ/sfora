import numpy as np

from sfora.arcg import diagnose_arcg_graph, normalized_response_signatures


def test_graph_rejects_close_disagreement_and_accepts_distant_agreement() -> None:
    # Pair 0-1 is closest in anchor space but has opposing signatures; pair 0-3
    # is farthest and has an agreeing signature. This is ARCG's defining asymmetry.
    anchors = np.array([[1.0, 0.0], [0.99, 0.01], [0.7, 0.7], [-0.8, 0.2]])
    signatures = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    result = diagnose_arcg_graph(
        anchors, signatures, np.zeros(4, dtype=np.int64), np.ones(4, dtype=bool)
    )
    assert result.closest_quartile_rejected_fraction == 1.0
    assert result.farthest_quartile_accepted_fraction == 1.0


def test_response_signatures_are_normalized_and_flat_profile_is_invalid() -> None:
    anchors = np.repeat([[1.0, 0.0]], 4, axis=0)
    angles = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.1, 0.2],
            [0.2, 0.3, 0.1],
            [0.2, 0.2, 0.2],
        ]
    )
    transformed = np.stack(
        [np.stack([np.cos(row), np.sin(row)], axis=1) for row in angles], axis=0
    )
    signatures, valid = normalized_response_signatures(anchors, transformed)
    assert valid.tolist() == [True, True, True, False]
    np.testing.assert_allclose(np.linalg.norm(signatures[:3], axis=1), 1.0)
    np.testing.assert_array_equal(signatures[3], 0.0)
