import numpy as np
import pytest

from sfora.training import (
    Objective,
    ProjectionHeadTrainingConfig,
    ProjectionTrainingConfig,
    _group_supervised_contrastive_gradient,
    _make_group_triplet_indices,
    _memory_hard_triplet_gradient,
    _normalization_gradient,
    _normalize,
    _objective_gradient,
    _radius_variance_gradient,
    _supervised_contrastive_gradient,
    train_embedding_table,
    train_projection_head,
)


def _overlapping_embeddings() -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, -0.2],
            [0.3, -0.1],
            [0.4, 0.2],
            [0.5, -0.1],
            [0.6, 0.0],
            [0.7, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    return embeddings, labels


def _close_class_centroids(label_count: int) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    for label in range(label_count):
        center = np.array([0.001 * label, 0.0], dtype=np.float64)
        embeddings.append(center + np.array([0.0, 0.001], dtype=np.float64))
        embeddings.append(center + np.array([0.0, -0.001], dtype=np.float64))
        labels.extend([label, label])
    return np.stack(embeddings), np.asarray(labels, dtype=np.int64)


def test_triplet_training_reduces_triplet_loss() -> None:
    embeddings, labels = _overlapping_embeddings()

    result = train_embedding_table(
        embeddings,
        labels,
        ProjectionTrainingConfig(objective="triplet", steps=40, learning_rate=0.05, margin=0.4),
    )

    assert result.objective == "triplet"
    assert result.final_triplet_loss < result.initial_triplet_loss
    assert result.transformed_embeddings.shape == embeddings.shape
    assert len(result.history) == 41


def test_group_training_reduces_group_loss() -> None:
    embeddings, labels = _overlapping_embeddings()

    result = train_embedding_table(
        embeddings,
        labels,
        ProjectionTrainingConfig(
            objective="group",
            group_size=2,
            steps=40,
            learning_rate=0.05,
            margin=0.4,
            spread_weight=0.2,
        ),
    )

    assert result.objective == "group"
    assert result.final_group_loss < result.initial_group_loss
    assert result.transformed_embeddings.shape == embeddings.shape
    assert len(result.history) == 41


def test_group_triplet_mining_requires_distinct_anchor_and_positive_groups() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="at least two groups per label"):
        _make_group_triplet_indices(labels, group_size=4)


def test_group_triplet_mining_can_shuffle_groups_reproducibly() -> None:
    labels = np.array([0] * 8 + [1] * 8, dtype=np.int64)

    unshuffled = _make_group_triplet_indices(labels, group_size=2)
    first = _make_group_triplet_indices(labels, group_size=2, random_state=11)
    second = _make_group_triplet_indices(labels, group_size=2, random_state=11)

    assert first == second
    assert first != unshuffled
    assert sorted(index for triplet in first for group in triplet for index in group) == sorted(
        index for triplet in unshuffled for group in triplet for index in group
    )


def test_group_supcon_uses_normalized_group_centroid_representatives() -> None:
    embeddings = np.array(
        [
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [-1.0, 1.0],
            [-2.0, 0.0],
            [-1.0, -1.0],
            [0.0, -2.0],
            [1.0, -1.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    config = ProjectionTrainingConfig(objective="group_supcon", group_size=2, temperature=0.2)
    group_triplets = _make_group_triplet_indices(labels, group_size=2)

    actual = _group_supervised_contrastive_gradient(embeddings, labels, group_triplets, config)

    groups = [anchor for anchor, _, _ in group_triplets]
    group_labels = np.asarray([labels[group[0]] for group in groups], dtype=np.int64)
    raw_centroids = np.stack([embeddings[list(group)].mean(axis=0) for group in groups])
    centroid_gradient = _supervised_contrastive_gradient(
        _normalize(raw_centroids),
        group_labels,
        config,
    )
    raw_centroid_gradient = _normalization_gradient(raw_centroids, centroid_gradient)
    expected = np.zeros_like(embeddings)
    for group, group_gradient in zip(groups, raw_centroid_gradient, strict=True):
        expected[list(group)] += group_gradient / len(group)

    assert np.allclose(actual, expected)


def test_triplet_projection_head_training_reduces_triplet_loss() -> None:
    embeddings, labels = _overlapping_embeddings()

    result = train_projection_head(
        embeddings,
        labels,
        ProjectionHeadTrainingConfig(
            objective="triplet",
            steps=40,
            learning_rate=0.05,
            margin=0.4,
        ),
    )

    assert result.objective == "triplet"
    assert result.final_triplet_loss < result.initial_triplet_loss
    assert result.projection_matrix.shape == (2, 2)
    assert result.transformed_embeddings.shape == embeddings.shape
    assert len(result.history) == 41


def test_group_projection_head_training_reduces_group_loss() -> None:
    embeddings, labels = _overlapping_embeddings()

    result = train_projection_head(
        embeddings,
        labels,
        ProjectionHeadTrainingConfig(
            objective="group",
            group_size=2,
            steps=40,
            learning_rate=0.05,
            margin=0.4,
            spread_weight=0.2,
        ),
    )

    assert result.objective == "group"
    assert result.final_group_loss < result.initial_group_loss
    assert result.projection_matrix.shape == (2, 2)
    assert result.transformed_embeddings.shape == embeddings.shape
    assert len(result.history) == 41


def test_projection_head_supports_hybrid_xbm_radius_objective() -> None:
    embeddings, labels = _overlapping_embeddings()

    result = train_projection_head(
        embeddings,
        labels,
        ProjectionHeadTrainingConfig(
            objective="hybrid_xbm_radius",
            group_size=2,
            steps=5,
            learning_rate=0.01,
            margin=0.4,
            xbm_weight=0.2,
            radius_weight=0.05,
            variance_weight=0.05,
        ),
    )

    assert result.objective == "hybrid_xbm_radius"
    assert result.projection_matrix.shape == (2, 2)
    assert result.transformed_embeddings.shape == embeddings.shape
    assert len(result.history) == 6


def test_radius_variance_gradient_stays_stable_with_many_labels() -> None:
    small_embeddings, small_labels = _close_class_centroids(label_count=5)
    large_embeddings, large_labels = _close_class_centroids(label_count=100)
    config = ProjectionTrainingConfig(
        objective="hybrid_xbm_radius",
        margin=1.0,
        radius_weight=0.05,
        variance_weight=0.05,
    )

    small_gradient = _radius_variance_gradient(small_embeddings, small_labels, config)
    large_gradient = _radius_variance_gradient(large_embeddings, large_labels, config)
    small_max_norm = np.linalg.norm(small_gradient, axis=1).max()
    large_max_norm = np.linalg.norm(large_gradient, axis=1).max()

    assert large_max_norm <= small_max_norm * 3.0


def test_radius_gradient_expands_classes_below_target_radius() -> None:
    embeddings = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    config = ProjectionTrainingConfig(
        objective="hybrid_xbm_radius",
        margin=0.5,
        radius_weight=1.0,
        radius_target=2.0,
        variance_weight=0.0,
    )

    gradient = _radius_variance_gradient(embeddings, labels, config)
    updated = embeddings - (0.1 * gradient)

    before_radius = np.linalg.norm(embeddings[:2] - embeddings[:2].mean(axis=0), axis=1).mean()
    after_radius = np.linalg.norm(updated[:2] - updated[:2].mean(axis=0), axis=1).mean()
    assert after_radius > before_radius


def test_memory_hard_triplet_gradient_uses_detached_memory_negatives() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [4.0, 0.0],
            [4.2, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    memory_embeddings = np.array(
        [
            [0.1, 0.0],
            [0.35, 0.0],
        ],
        dtype=np.float64,
    )
    memory_labels = np.array([0, 1], dtype=np.int64)
    config = ProjectionTrainingConfig(objective="hybrid_xbm", margin=0.5)

    current_gradient = _memory_hard_triplet_gradient(
        embeddings,
        labels,
        memory_embeddings=np.empty((0, 2), dtype=np.float64),
        memory_labels=np.empty((0,), dtype=np.int64),
        config=config,
    )
    memory_gradient = _memory_hard_triplet_gradient(
        embeddings,
        labels,
        memory_embeddings=memory_embeddings,
        memory_labels=memory_labels,
        config=config,
    )

    assert np.allclose(current_gradient, np.zeros_like(embeddings))
    assert np.linalg.norm(memory_gradient[0]) > 0.0


@pytest.mark.parametrize(
    "objective",
    [
        "batch_hard_triplet",
        "hard_group",
        "supcon",
        "proxy_nca",
        "proxy_anchor",
        "cosface",
        "arcface",
        "group_supcon",
        "group_supcon_xbm_radius",
    ],
)
def test_projection_head_supports_stronger_mining_objectives(objective: Objective) -> None:
    embeddings, labels = _overlapping_embeddings()

    result = train_projection_head(
        embeddings,
        labels,
        ProjectionHeadTrainingConfig(
            objective=objective,
            group_size=2,
            steps=5,
            learning_rate=0.01,
            margin=0.4,
            normalize_projected_embeddings=True,
        ),
    )

    assert result.objective == objective
    assert result.projection_matrix.shape == (2, 2)
    assert result.transformed_embeddings.shape == embeddings.shape
    assert len(result.history) == 6


def test_group_supcon_keeps_point_supcon_gradient_component() -> None:
    embeddings, labels = _overlapping_embeddings()
    config = ProjectionTrainingConfig(
        objective="group_supcon",
        group_size=2,
        temperature=0.2,
        triplet_weight=0.7,
        group_weight=1.3,
    )
    group_triplets = _make_group_triplet_indices(labels, group_size=2)

    gradient = _objective_gradient(
        embeddings,
        labels,
        [],
        group_triplets,
        config,
    )

    point_gradient = _supervised_contrastive_gradient(embeddings, labels, config)
    group_gradient = _group_supervised_contrastive_gradient(
        embeddings,
        labels,
        group_triplets,
        config,
    )
    assert np.allclose(gradient, (0.7 * point_gradient) + (1.3 * group_gradient))


def test_projection_head_can_train_and_return_normalized_projected_embeddings() -> None:
    embeddings, labels = _overlapping_embeddings()
    scaled_embeddings = embeddings * np.array(
        [[1.0], [2.0], [3.0], [4.0], [1.5], [2.5], [3.5], [4.5]]
    )

    result = train_projection_head(
        scaled_embeddings,
        labels,
        ProjectionHeadTrainingConfig(
            objective="hybrid_xbm_radius",
            group_size=2,
            steps=3,
            learning_rate=0.01,
            margin=0.4,
            normalize_projected_embeddings=True,
        ),
    )

    norms = np.linalg.norm(result.transformed_embeddings, axis=1)
    assert np.allclose(norms[norms > 0.0], 1.0)


def test_projection_head_selects_best_checkpoint_with_selection_callback() -> None:
    embeddings, labels = _overlapping_embeddings()
    scored_steps: list[int] = []

    def validation_score(projection: np.ndarray, step: int) -> float:
        scored_steps.append(step)
        return 1.0 if step == 0 else -float(step)

    result = train_projection_head(
        embeddings,
        labels,
        ProjectionHeadTrainingConfig(
            objective="triplet",
            steps=5,
            learning_rate=0.05,
            margin=0.4,
            selection_score_callback=validation_score,
        ),
    )

    assert scored_steps == [0, 1, 2, 3, 4, 5]
    assert result.selected_step == 0
    assert result.selection_score == 1.0
    assert np.allclose(result.projection_matrix, np.eye(2))
