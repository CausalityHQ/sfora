from __future__ import annotations

import numpy as np
import pytest

from sfora.unicom_shard_audit import (
    arcface_joint_objective,
    audit_shard_sensitivity,
    class_to_shard_permutations,
    contiguous_shard_sizes,
    select_shard_panel,
    shard_decision,
    trial_masks,
)


def _training_fixture(*, classes: int = 12, rows_per_class: int = 5, dimension: int = 8):
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    for class_index in range(classes):
        for row_index in range(rows_per_class):
            vector = np.full(dimension, 0.05, dtype=np.float32)
            vector[class_index % dimension] += 1.0 + row_index / 10.0
            embeddings.append(vector)
            labels.append(f"class-{class_index:02d}")
    return np.ascontiguousarray(np.stack(embeddings)), np.asarray(labels)


@pytest.mark.parametrize(
    ("class_count", "expected"),
    [(8, (2, 2, 2, 2)), (10, (3, 3, 2, 2)), (3, (1, 1, 1, 0))],
)
def test_contiguous_shard_sizes_match_partialfc_rule(
    class_count: int, expected: tuple[int, int, int, int]
) -> None:
    assert contiguous_shard_sizes(class_count) == expected


def test_panel_selects_seeded_classes_then_restores_sorted_order() -> None:
    embeddings, labels = _training_fixture()
    panel = select_shard_panel(
        embeddings,
        labels,
        class_count=8,
        examples_per_class=4,
        seed=205,
    )
    all_classes = np.unique(labels)
    chosen = np.random.Generator(np.random.PCG64(205)).choice(
        all_classes.size, 8, replace=False
    )
    expected_classes = np.sort(all_classes[chosen])

    assert panel.embeddings.shape == (32, 8)
    assert panel.prototypes.shape == (8, 8)
    assert panel.shard_sizes == (2, 2, 2, 2)
    assert panel.class_labels.tolist() == expected_classes.tolist()
    assert panel.labels.tolist() == np.repeat(expected_classes, 4).tolist()


def test_panel_uses_first_rows_and_all_rows_for_prototype_mean() -> None:
    embeddings, labels = _training_fixture(classes=4)
    panel = select_shard_panel(
        embeddings,
        labels,
        class_count=4,
        examples_per_class=4,
        seed=205,
    )

    for class_index, label in enumerate(panel.class_labels):
        source = embeddings[labels == label]
        expected_rows = source[:4]
        assert np.array_equal(
            panel.embeddings[class_index * 4 : (class_index + 1) * 4], expected_rows
        )
        mean = source.astype(np.float64).mean(axis=0)
        expected_prototype = (mean / np.linalg.norm(mean)).astype(np.float32)
        assert panel.prototypes[class_index] == pytest.approx(expected_prototype)


def test_panel_rejects_classes_with_too_few_rows() -> None:
    embeddings, labels = _training_fixture(classes=4, rows_per_class=3)

    with pytest.raises(ValueError, match="eligible identities"):
        select_shard_panel(
            embeddings,
            labels,
            class_count=4,
            examples_per_class=4,
            seed=205,
        )


def _objective_fixture():
    embeddings = np.array(
        [[1.0, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 1.0]], dtype=np.float32
    )
    prototypes = np.array(
        [
            [1.0, 0.1, 0.2, 0.3],
            [0.2, 1.0, 0.3, 0.1],
            [0.3, 0.2, 1.0, 0.1],
            [0.1, 0.3, 0.2, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 3], dtype=np.int64)
    assignments = np.array([0, 1, 2, 3], dtype=np.int64)
    masks = (
        np.array([0, 1], dtype=np.int64),
        np.array([1, 2], dtype=np.int64),
        np.array([2, 3], dtype=np.int64),
        np.array([0, 3], dtype=np.int64),
    )
    return embeddings, labels, prototypes, assignments, masks


def _straight_through_oracle(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    assignments: np.ndarray,
    masks: tuple[np.ndarray, ...],
    *,
    margin: float = 0.25,
    scale: float = 32.0,
    mathematical_target_derivative: bool = False,
):
    rows, dimension = embeddings.shape
    classes = prototypes.shape[0]
    logits = np.empty((rows, classes), dtype=np.float64)
    normalized_embeddings: dict[int, np.ndarray] = {}
    normalized_prototypes: dict[int, np.ndarray] = {}
    for shard, mask in enumerate(masks):
        selected_embeddings = embeddings[:, mask].astype(np.float64)
        selected_prototypes = prototypes[:, mask].astype(np.float64)
        normalized_embeddings[shard] = selected_embeddings / np.linalg.norm(
            selected_embeddings, axis=1, keepdims=True
        )
        normalized_prototypes[shard] = selected_prototypes / np.linalg.norm(
            selected_prototypes, axis=1, keepdims=True
        )
        columns = np.flatnonzero(assignments == shard)
        logits[:, columns] = (
            normalized_embeddings[shard] @ normalized_prototypes[shard][columns].T
        )
    rows_index = np.arange(rows)
    targets = np.clip(logits[rows_index, labels], -1.0, 1.0)
    logits[rows_index, labels] = np.cos(np.arccos(targets) + margin)
    logits *= scale
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    losses = -np.log(probabilities[rows_index, labels])
    score_gradient = probabilities
    score_gradient[rows_index, labels] -= 1.0
    if mathematical_target_derivative:
        target_factors = np.sin(np.arccos(targets) + margin) / np.sqrt(1.0 - targets**2)
        score_gradient[rows_index, labels] *= target_factors
    score_gradient *= scale / rows
    gradient = np.zeros((rows, dimension), dtype=np.float64)
    for shard, mask in enumerate(masks):
        columns = np.flatnonzero(assignments == shard)
        direction_gradient = (
            score_gradient[:, columns] @ normalized_prototypes[shard][columns]
        )
        selected = embeddings[:, mask].astype(np.float64)
        norms = np.linalg.norm(selected, axis=1, keepdims=True)
        unit = normalized_embeddings[shard]
        tangent = (
            direction_gradient
            - unit * np.sum(unit * direction_gradient, axis=1, keepdims=True)
        ) / norms
        gradient[:, mask] += tangent
    return float(losses.mean()), losses, logits.argmax(axis=1), gradient


def test_arcface_joint_objective_matches_independent_straight_through_oracle() -> None:
    fixture = _objective_fixture()
    expected = _straight_through_oracle(*fixture)

    result = arcface_joint_objective(*fixture)

    assert result.loss == pytest.approx(expected[0], abs=1e-10)
    assert result.per_example_loss == pytest.approx(expected[1], abs=1e-10)
    assert np.array_equal(result.predictions, expected[2])
    assert result.embedding_gradient == pytest.approx(expected[3], abs=1e-10)


def test_arcface_target_gradient_is_straight_through_not_mathematical_derivative() -> None:
    embeddings, labels, prototypes, assignments, masks = _objective_fixture()
    result = arcface_joint_objective(
        embeddings, labels, prototypes, assignments, masks, margin=0.25, scale=32.0
    )
    expected = _straight_through_oracle(
        embeddings, labels, prototypes, assignments, masks, margin=0.25, scale=32.0
    )[3]
    mathematical = _straight_through_oracle(
        embeddings,
        labels,
        prototypes,
        assignments,
        masks,
        margin=0.25,
        scale=32.0,
        mathematical_target_derivative=True,
    )[3]

    assert result.embedding_gradient == pytest.approx(expected, abs=1e-10)
    assert not np.allclose(result.embedding_gradient, mathematical, atol=1e-8, rtol=1e-8)


def test_trial_masks_use_exact_registered_streams() -> None:
    masks = trial_masks(dimension=8, selected=4, trial=3)
    for rank, actual in enumerate(masks):
        expected = np.sort(
            np.random.Generator(np.random.PCG64(1000 + 3 * 4 + rank)).choice(
                8, 4, replace=False
            )
        )
        assert np.array_equal(actual, expected)


def test_class_permutations_use_exact_registered_stream() -> None:
    actual = class_to_shard_permutations(class_count=8, trial=2, count=3)
    baseline = np.repeat(np.arange(4, dtype=np.int64), 2)
    generator = np.random.Generator(np.random.PCG64(3002))
    expected = tuple(generator.permutation(baseline) for _ in range(3))

    for produced, oracle in zip(actual, expected, strict=True):
        assert np.array_equal(produced, oracle)


@pytest.mark.parametrize(
    (
        "loss_range",
        "independent_mse",
        "coherent_mse",
        "invariance",
        "prediction_change",
        "finite",
        "expected",
    ),
    [
        (1e-3, 1.25, 1.0, 1e-6, 0.10, True, "SHARD_SENSITIVE"),
        (1e-3 - 1e-12, 2.0, 1.0, 0.0, 1.0, True, "SHARD_NULL"),
        (1e-3, 1.249999, 1.0, 0.0, 1.0, True, "SHARD_NULL"),
        (1e-3, 2.0, 1.0, 1e-6 + 1e-12, 1.0, True, "SHARD_NULL"),
        (1e-3, 2.0, 1.0, 0.0, 0.10 - 1e-12, True, "SHARD_NULL"),
        (1e-3, 2.0, 1.0, 0.0, 1.0, False, "SHARD_NULL"),
    ],
)
def test_shard_decision_boundaries(
    loss_range: float,
    independent_mse: float,
    coherent_mse: float,
    invariance: float,
    prediction_change: float,
    finite: bool,
    expected: str,
) -> None:
    decision = shard_decision(
        independent_loss_range=loss_range,
        independent_gradient_mse=independent_mse,
        coherent_gradient_mse=coherent_mse,
        coherent_invariance_error=invariance,
        prediction_change_rate=prediction_change,
        all_finite=finite,
    )

    assert decision == expected


def test_shard_audit_is_deterministic_and_finite_on_real_embedding_panel() -> None:
    embeddings, labels = _training_fixture(classes=12, rows_per_class=5, dimension=8)
    kwargs = {
        "class_count": 8,
        "examples_per_class": 2,
        "selected": 4,
        "trials": 2,
        "permutations": 3,
    }

    first = audit_shard_sensitivity(embeddings, labels, **kwargs)
    second = audit_shard_sensitivity(embeddings, labels, **kwargs)

    assert first == second
    assert first.trials == 2
    assert first.permutations_per_trial == 3
    assert first.all_finite is True
    assert 0.0 <= first.mask_union_coverage <= 1.0
