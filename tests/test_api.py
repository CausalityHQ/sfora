import numpy as np
import pytest

from sfora import GroupLearningProjector, fit_sfora_projection


def _toy_embeddings() -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.array(
        [
            [1.00, 0.00],
            [0.95, 0.05],
            [0.90, -0.05],
            [0.85, 0.10],
            [-1.00, 0.00],
            [-0.95, -0.05],
            [-0.90, 0.05],
            [-0.85, -0.10],
        ],
        dtype=np.float64,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    return embeddings, labels


def test_sfora_projector_fits_and_transforms_embeddings() -> None:
    embeddings, labels = _toy_embeddings()

    projector = GroupLearningProjector(
        objective="group_supcon_xbm_radius",
        group_size=2,
        steps=4,
        learning_rate=0.01,
        normalize_embeddings=True,
    ).fit(embeddings, labels)

    transformed = projector.transform(embeddings)

    assert transformed.shape == embeddings.shape
    assert projector.training_result is not None
    assert projector.selected_step >= 0
    assert np.allclose(np.linalg.norm(transformed, axis=1), 1.0)


def test_sfora_projector_can_select_checkpoint_with_validation_split() -> None:
    embeddings, labels = _toy_embeddings()
    validation_embeddings = embeddings + np.array(
        [[0.01, 0.00], [0.00, 0.01], [-0.01, 0.00], [0.00, -0.01]] * 2,
        dtype=np.float64,
    )

    projector = GroupLearningProjector(
        objective="triplet",
        group_size=2,
        steps=3,
        learning_rate=0.01,
        normalize_embeddings=True,
    ).fit(
        embeddings,
        labels,
        validation_embeddings=validation_embeddings,
        validation_labels=labels,
        validation_query_limit=4,
    )

    assert projector.training_result is not None
    assert projector.selection_score is not None
    assert 0 <= projector.selected_step <= 3


def test_sfora_projector_exposes_memory_and_group_shuffle_knobs() -> None:
    projector = GroupLearningProjector(
        xbm_memory_size=64,
        shuffle_groups_each_step=True,
    )

    assert projector.config.xbm_memory_size == 64
    assert projector.config.shuffle_groups_each_step is True


def test_sfora_projector_requires_complete_validation_split() -> None:
    embeddings, labels = _toy_embeddings()
    projector = GroupLearningProjector()

    with pytest.raises(ValueError, match="validation_embeddings and validation_labels"):
        projector.fit(embeddings, labels, validation_embeddings=embeddings)


def test_fit_sfora_projection_returns_fitted_projector() -> None:
    embeddings, labels = _toy_embeddings()

    projector = fit_sfora_projection(
        embeddings,
        labels,
        objective="hybrid_xbm_radius",
        group_size=2,
        steps=3,
        learning_rate=0.01,
    )

    assert isinstance(projector, GroupLearningProjector)
    assert projector.training_result is not None


def test_sfora_projector_requires_fit_before_transform() -> None:
    projector = GroupLearningProjector()

    with pytest.raises(ValueError, match="fit"):
        projector.transform(np.zeros((2, 2), dtype=np.float64))
