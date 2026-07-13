import numpy as np
import pytest

from sfora.evaluation import (
    embedding_space_diagnostics_on_split,
    linear_probe_score,
    linear_probe_score_on_split,
    retrieval_score_on_split,
)


def test_linear_probe_scores_linearly_separable_embeddings() -> None:
    embeddings = np.array(
        [
            [-2.0, -1.8],
            [-1.8, -2.2],
            [-2.2, -1.9],
            [2.0, 1.8],
            [1.8, 2.2],
            [2.2, 1.9],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    score = linear_probe_score(embeddings, labels, test_size=0.5, random_state=7)

    assert score.accuracy == pytest.approx(1.0)
    assert score.macro_f1 == pytest.approx(1.0)
    assert score.train_accuracy == pytest.approx(1.0)
    assert score.train_macro_f1 == pytest.approx(1.0)
    assert score.confusion_matrix.shape == (2, 2)
    assert np.trace(score.confusion_matrix) == score.confusion_matrix.sum()


def test_linear_probe_scores_explicit_train_test_split() -> None:
    train_embeddings = np.array(
        [
            [-2.0, -1.8],
            [-1.8, -2.2],
            [2.0, 1.8],
            [1.8, 2.2],
        ]
    )
    train_labels = np.array([0, 0, 1, 1])
    test_embeddings = np.array([[-2.2, -1.9], [2.2, 1.9]])
    test_labels = np.array([0, 1])

    score = linear_probe_score_on_split(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        random_state=7,
    )

    assert score.accuracy == pytest.approx(1.0)
    assert score.macro_f1 == pytest.approx(1.0)
    assert score.train_accuracy == pytest.approx(1.0)
    assert score.train_macro_f1 == pytest.approx(1.0)
    assert score.confusion_matrix.tolist() == [[1, 0], [0, 1]]


def test_retrieval_scores_precision_at_1_and_map_at_r_on_explicit_split() -> None:
    gallery_embeddings = np.array(
        [
            [-2.0, -2.0],
            [-1.8, -2.2],
            [2.0, 2.0],
            [2.2, 1.8],
        ]
    )
    gallery_labels = np.array([0, 0, 1, 1])
    query_embeddings = np.array(
        [
            [-2.1, -1.9],
            [1.9, 2.1],
        ]
    )
    query_labels = np.array([0, 1])

    score = retrieval_score_on_split(
        gallery_embeddings,
        gallery_labels,
        query_embeddings,
        query_labels,
    )

    assert score.precision_at_1 == pytest.approx(1.0)
    assert score.map_at_r == pytest.approx(1.0)
    assert score.mean_relevant_items == pytest.approx(2.0)


def test_retrieval_map_at_r_penalizes_late_relevant_neighbors() -> None:
    gallery_embeddings = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.4, 0.0],
            [0.6, 0.0],
        ]
    )
    gallery_labels = np.array([1, 0, 1, 0])
    query_embeddings = np.array([[0.25, 0.0]])
    query_labels = np.array([1])

    score = retrieval_score_on_split(
        gallery_embeddings,
        gallery_labels,
        query_embeddings,
        query_labels,
    )

    assert score.precision_at_1 == pytest.approx(0.0)
    assert score.map_at_r == pytest.approx(0.25)
    assert score.mean_relevant_items == pytest.approx(2.0)


def test_retrieval_query_limit_uses_deterministic_stratified_subset() -> None:
    gallery_embeddings = np.array(
        [
            [-2.0, -2.0],
            [-1.8, -2.2],
            [2.0, 2.0],
            [2.2, 1.8],
        ]
    )
    gallery_labels = np.array([0, 0, 1, 1])
    query_embeddings = np.array(
        [
            [-2.1, -1.9],
            [-2.0, -2.1],
            [-1.9, -2.0],
            [1.9, 2.1],
            [2.1, 1.9],
            [2.0, 2.2],
        ]
    )
    query_labels = np.array([0, 0, 0, 1, 1, 1])

    score = retrieval_score_on_split(
        gallery_embeddings,
        gallery_labels,
        query_embeddings,
        query_labels,
        query_limit=4,
        random_state=7,
    )

    assert score.evaluated_queries == 4
    assert score.total_queries == 6
    assert score.precision_at_1 == pytest.approx(1.0)
    assert score.map_at_r == pytest.approx(1.0)


def test_embedding_space_diagnostics_quantify_linear_class_geometry() -> None:
    train_embeddings = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    train_labels = np.array([0, 0, 1, 1])
    test_embeddings = np.array([[-1.5, 0.0], [1.5, 0.0]])
    test_labels = np.array([0, 1])

    diagnostics = embedding_space_diagnostics_on_split(
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
    )

    assert diagnostics.train_within_class_radius == pytest.approx(0.5)
    assert diagnostics.test_within_class_radius == pytest.approx(0.0)
    assert diagnostics.train_centroid_gap == pytest.approx(3.0)
    assert diagnostics.train_test_centroid_drift == pytest.approx(0.0)
    assert diagnostics.signal_to_noise_ratio == pytest.approx(6.0)
    assert diagnostics.drift_to_gap_ratio == pytest.approx(0.0)


def test_linear_probe_requires_at_least_two_classes() -> None:
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.array([1, 1])

    with pytest.raises(ValueError, match="at least two classes"):
        linear_probe_score(embeddings, labels)
