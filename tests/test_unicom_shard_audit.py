from __future__ import annotations

import numpy as np
import pytest

import sfora.unicom_shard_audit as shard_module
from sfora.unicom_shard_audit import (
    ObjectiveResult,
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
    chosen = np.random.Generator(np.random.PCG64(205)).choice(all_classes.size, 8, replace=False)
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
    embeddings = np.array([[1.0, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 1.0]], dtype=np.float32)
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
        logits[:, columns] = normalized_embeddings[shard] @ normalized_prototypes[shard][columns].T
    rows_index = np.arange(rows)
    targets = np.clip(logits[rows_index, labels], -1.0, 1.0)
    predictions = logits.argmax(axis=1)
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
        direction_gradient = score_gradient[:, columns] @ normalized_prototypes[shard][columns]
        selected = embeddings[:, mask].astype(np.float64)
        norms = np.linalg.norm(selected, axis=1, keepdims=True)
        unit = normalized_embeddings[shard]
        tangent = (
            direction_gradient - unit * np.sum(unit * direction_gradient, axis=1, keepdims=True)
        ) / norms
        gradient[:, mask] += tangent
    return float(losses.mean()), losses, predictions, gradient


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


def test_audit_measures_coherent_invariance_across_class_placements(monkeypatch) -> None:
    embeddings, labels = _training_fixture(classes=8, rows_per_class=4, dimension=8)

    def placement_sensitive_objective(
        values: np.ndarray,
        label_indices: np.ndarray,
        prototypes: np.ndarray,
        assignments: np.ndarray,
        masks: tuple[np.ndarray, ...],
        *,
        margin: float,
        scale: float,
    ) -> ObjectiveResult:
        assert margin == 0.25
        assert scale == 32.0
        coherent = all(np.array_equal(mask, masks[0]) for mask in masks[1:])
        placement_score = float(
            np.dot(np.arange(1, assignments.size + 1, dtype=np.float64), assignments)
        )
        if not coherent:
            multiplier = 10.0
        elif masks[0].size == values.shape[1]:
            multiplier = 100.0
        else:
            multiplier = 1.0
        loss = multiplier * placement_score
        return ObjectiveResult(
            loss=loss,
            per_example_loss=np.full(values.shape[0], loss, dtype=np.float64),
            predictions=np.zeros(values.shape[0], dtype=np.int64),
            embedding_gradient=np.zeros_like(values, dtype=np.float64),
        )

    monkeypatch.setattr(shard_module, "arcface_joint_objective", placement_sensitive_objective)

    result = audit_shard_sensitivity(
        embeddings,
        labels,
        class_count=8,
        examples_per_class=2,
        selected=4,
        trials=1,
        permutations=3,
    )

    assignments = class_to_shard_permutations(class_count=8, trial=0, count=3)
    coherent_losses = [
        float(np.dot(np.arange(1, 9, dtype=np.float64), assignment)) for assignment in assignments
    ]
    expected_coherent_range = max(coherent_losses) - min(coherent_losses)
    assert result.independent_loss_range == 10.0 * expected_coherent_range
    assert result.coherent_placement_control_error == expected_coherent_range


def test_coherent_masks_remove_class_placement_dependence_on_real_objective() -> None:
    embeddings, labels, prototypes, assignments, masks = _objective_fixture()
    moved_assignments = np.array([1, 0, 3, 2], dtype=np.int64)
    coherent_masks = (masks[0], masks[0], masks[0], masks[0])

    independent_before = arcface_joint_objective(embeddings, labels, prototypes, assignments, masks)
    independent_after = arcface_joint_objective(
        embeddings, labels, prototypes, moved_assignments, masks
    )
    coherent_before = arcface_joint_objective(
        embeddings, labels, prototypes, assignments, coherent_masks
    )
    coherent_after = arcface_joint_objective(
        embeddings, labels, prototypes, moved_assignments, coherent_masks
    )

    assert abs(independent_before.loss - independent_after.loss) > 1e-3
    assert coherent_before.loss == coherent_after.loss


def test_trial_masks_use_exact_registered_streams() -> None:
    masks = trial_masks(dimension=8, selected=4, trial=3)
    for rank, actual in enumerate(masks):
        expected = np.sort(
            np.random.Generator(np.random.PCG64(1000 + 3 * 4 + rank)).choice(8, 4, replace=False)
        )
        assert np.array_equal(actual, expected)


def test_audit_routes_registered_rank_count_into_panel_masks_and_assignments(
    monkeypatch,
) -> None:
    embeddings, labels = _training_fixture(classes=8, rows_per_class=4, dimension=8)
    real_panel = shard_module.select_shard_panel
    real_masks = shard_module.trial_masks
    real_assignments = shard_module.class_to_shard_permutations
    observed: list[tuple[str, int | None]] = []

    def panel(*args, **kwargs):
        observed.append(("panel", kwargs.get("world_size")))
        return real_panel(*args, **kwargs)

    def masks(*args, **kwargs):
        observed.append(("masks", kwargs.get("rank_count")))
        return real_masks(*args, **kwargs)

    def assignments(*args, **kwargs):
        observed.append(("assignments", kwargs.get("world_size")))
        return real_assignments(*args, **kwargs)

    monkeypatch.setattr(shard_module, "select_shard_panel", panel)
    monkeypatch.setattr(shard_module, "trial_masks", masks)
    monkeypatch.setattr(shard_module, "class_to_shard_permutations", assignments)

    result = audit_shard_sensitivity(
        embeddings,
        labels,
        class_count=8,
        examples_per_class=2,
        selected=4,
        trials=1,
        permutations=2,
    )

    assert result.config.shard_rank_count == 4
    assert observed == [("panel", 4), ("masks", 4), ("assignments", 4)]


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
        coherent_placement_control_error=invariance,
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
