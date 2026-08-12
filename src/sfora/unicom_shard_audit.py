"""Single-process emulation of UNICOM's four-shard sampled-subspace objective."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfora.unicom_retrieval_audit import l2_normalize


@dataclass(frozen=True)
class ShardPanel:
    embeddings: np.ndarray
    labels: np.ndarray
    prototypes: np.ndarray
    class_labels: np.ndarray
    shard_sizes: tuple[int, int, int, int]


@dataclass(frozen=True)
class ObjectiveResult:
    loss: float
    per_example_loss: np.ndarray
    predictions: np.ndarray
    embedding_gradient: np.ndarray


@dataclass(frozen=True)
class ShardAudit:
    trials: int
    permutations_per_trial: int
    independent_loss_range: float
    independent_loss_std: float
    independent_gradient_mse: float
    coherent_gradient_mse: float
    independent_gradient_cosine_distance: float
    coherent_gradient_cosine_distance: float
    coherent_invariance_error: float
    prediction_change_rate: float
    mask_union_coverage: float
    all_finite: bool
    decision: str


def contiguous_shard_sizes(
    class_count: int, world_size: int = 4
) -> tuple[int, int, int, int]:
    if type(class_count) is not int or type(world_size) is not int:
        raise TypeError("class_count and world_size must be builtin integers")
    if class_count <= 0 or world_size != 4:
        raise ValueError("class_count must be positive and world_size must be four")
    sizes = tuple(
        class_count // world_size + int(rank < class_count % world_size)
        for rank in range(world_size)
    )
    return sizes  # type: ignore[return-value]


def _validate_training_inputs(embeddings: np.ndarray, labels: np.ndarray) -> None:
    if type(embeddings) is not np.ndarray or embeddings.dtype != np.float32:
        raise TypeError("train embeddings must be a float32 NumPy array")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("train embeddings must be a nonempty matrix")
    if not embeddings.flags.c_contiguous or not np.isfinite(embeddings).all():
        raise ValueError("train embeddings must be finite and C-contiguous")
    if type(labels) is not np.ndarray or labels.ndim != 1 or labels.shape[0] != len(embeddings):
        raise ValueError("train labels must be a vector matching embedding rows")
    if labels.dtype.kind not in {"U", "S", "O"}:
        raise TypeError("train labels must contain strings")
    if any(type(label) is not str or not label for label in labels.tolist()):
        raise TypeError("train labels must contain nonempty builtin strings")


def select_shard_panel(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    *,
    class_count: int = 64,
    examples_per_class: int = 4,
    seed: int = 205,
) -> ShardPanel:
    _validate_training_inputs(train_embeddings, train_labels)
    if (
        type(class_count) is not int
        or type(examples_per_class) is not int
        or type(seed) is not int
    ):
        raise TypeError("panel parameters must be builtin integers")
    if class_count <= 0 or examples_per_class <= 0:
        raise ValueError("panel sizes must be positive")

    unique_labels, counts = np.unique(train_labels, return_counts=True)
    eligible = unique_labels[counts >= examples_per_class]
    if eligible.size < class_count:
        raise ValueError("not enough eligible identities for the requested panel")
    generator = np.random.Generator(np.random.PCG64(seed))
    selected_indices = generator.choice(eligible.size, class_count, replace=False)
    class_labels = np.sort(eligible[selected_indices])

    rows: list[np.ndarray] = []
    row_labels: list[str] = []
    prototypes: list[np.ndarray] = []
    for label in class_labels.tolist():
        source = np.ascontiguousarray(train_embeddings[train_labels == label])
        rows.append(source[:examples_per_class])
        row_labels.extend([label] * examples_per_class)
        mean = np.mean(source.astype(np.float64), axis=0, dtype=np.float64).astype(np.float32)
        prototypes.append(mean)

    prototype_matrix = l2_normalize(np.ascontiguousarray(np.stack(prototypes)))
    return ShardPanel(
        embeddings=np.ascontiguousarray(np.concatenate(rows, axis=0)),
        labels=np.asarray(row_labels),
        prototypes=prototype_matrix,
        class_labels=np.asarray(class_labels),
        shard_sizes=contiguous_shard_sizes(class_count),
    )


def _validate_objective_inputs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    assignments: np.ndarray,
    masks: tuple[np.ndarray, ...],
) -> None:
    _validate_training_inputs(embeddings, np.asarray(["x"] * len(embeddings)))
    if type(prototypes) is not np.ndarray or prototypes.dtype != np.float32:
        raise TypeError("prototypes must be a float32 NumPy array")
    if (
        prototypes.ndim != 2
        or prototypes.shape[0] == 0
        or prototypes.shape[1] != embeddings.shape[1]
        or not prototypes.flags.c_contiguous
        or not np.isfinite(prototypes).all()
    ):
        raise ValueError("prototypes must be a finite C-contiguous matrix")
    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.ndim != 1
        or labels.shape[0] != embeddings.shape[0]
        or np.any(labels < 0)
        or np.any(labels >= prototypes.shape[0])
    ):
        raise ValueError("labels must be valid int64 class indices")
    if (
        type(assignments) is not np.ndarray
        or assignments.dtype != np.int64
        or assignments.shape != (prototypes.shape[0],)
        or np.any(assignments < 0)
        or np.any(assignments >= 4)
        or set(assignments.tolist()) != {0, 1, 2, 3}
    ):
        raise ValueError("assignments must map every class to one of four nonempty shards")
    if type(masks) is not tuple or len(masks) != 4:
        raise TypeError("masks must be an exact four-element tuple")
    for mask in masks:
        if (
            type(mask) is not np.ndarray
            or mask.dtype != np.int64
            or mask.ndim != 1
            or mask.size == 0
            or np.any(mask < 0)
            or np.any(mask >= embeddings.shape[1])
            or not np.array_equal(mask, np.unique(mask))
        ):
            raise ValueError("every mask must contain sorted unique in-range coordinates")


def arcface_joint_objective(
    embeddings: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    assignments: np.ndarray,
    masks: tuple[np.ndarray, ...],
    *,
    margin: float = 0.25,
    scale: float = 32.0,
) -> ObjectiveResult:
    """Evaluate the joint sharded softmax and official straight-through gradient."""

    _validate_objective_inputs(embeddings, labels, prototypes, assignments, masks)
    if (
        type(margin) is not float
        or type(scale) is not float
        or not np.isfinite(margin)
        or not np.isfinite(scale)
        or margin < 0.0
        or scale <= 0.0
    ):
        raise ValueError("margin and scale must be finite builtin floats in range")

    rows, dimension = embeddings.shape
    class_count = prototypes.shape[0]
    logits = np.empty((rows, class_count), dtype=np.float64)
    normalized_embeddings: list[np.ndarray] = []
    normalized_prototypes: list[np.ndarray] = []
    selected_embeddings: list[np.ndarray] = []

    for shard, mask in enumerate(masks):
        embedding_slice = embeddings[:, mask].astype(np.float64)
        prototype_slice = prototypes[:, mask].astype(np.float64)
        embedding_norms = np.linalg.norm(embedding_slice, axis=1, keepdims=True)
        prototype_norms = np.linalg.norm(prototype_slice, axis=1, keepdims=True)
        if np.any(embedding_norms == 0.0) or np.any(prototype_norms == 0.0):
            raise ValueError("selected subspace contains a zero-norm row")
        embedding_unit = embedding_slice / embedding_norms
        prototype_unit = prototype_slice / prototype_norms
        columns = np.flatnonzero(assignments == shard)
        logits[:, columns] = embedding_unit @ prototype_unit[columns].T
        selected_embeddings.append(embedding_slice)
        normalized_embeddings.append(embedding_unit)
        normalized_prototypes.append(prototype_unit)

    row_indices = np.arange(rows)
    target_cosines = np.clip(logits[row_indices, labels], -1.0, 1.0)
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    logits[row_indices, labels] = np.cos(np.arccos(target_cosines) + margin)
    logits *= scale
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True, dtype=np.float64)
    per_example_loss = -np.log(probabilities[row_indices, labels])

    score_gradient = probabilities.copy()
    score_gradient[row_indices, labels] -= 1.0
    score_gradient *= scale / rows
    embedding_gradient = np.zeros((rows, dimension), dtype=np.float64)
    for shard, mask in enumerate(masks):
        columns = np.flatnonzero(assignments == shard)
        direction_gradient = score_gradient[:, columns] @ normalized_prototypes[shard][columns]
        unit = normalized_embeddings[shard]
        norms = np.linalg.norm(selected_embeddings[shard], axis=1, keepdims=True)
        tangent_gradient = (
            direction_gradient
            - unit * np.sum(unit * direction_gradient, axis=1, keepdims=True)
        ) / norms
        embedding_gradient[:, mask] += tangent_gradient

    if not (
        np.isfinite(per_example_loss).all()
        and np.isfinite(embedding_gradient).all()
        and np.isfinite(probabilities).all()
    ):
        raise ValueError("objective produced a nonfinite value")
    return ObjectiveResult(
        loss=float(np.mean(per_example_loss, dtype=np.float64)),
        per_example_loss=per_example_loss,
        predictions=predictions,
        embedding_gradient=embedding_gradient,
    )


def trial_masks(
    *, dimension: int, selected: int = 512, trial: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        type(dimension) is not int
        or type(selected) is not int
        or type(trial) is not int
        or dimension <= 0
        or selected <= 0
        or selected > dimension
        or trial < 0
    ):
        raise ValueError("invalid mask parameters")
    masks = tuple(
        np.sort(
            np.random.Generator(np.random.PCG64(1000 + trial * 4 + rank)).choice(
                dimension, selected, replace=False
            )
        ).astype(np.int64)
        for rank in range(4)
    )
    return masks  # type: ignore[return-value]


def class_to_shard_permutations(
    *, class_count: int, trial: int, count: int = 16
) -> tuple[np.ndarray, ...]:
    if (
        type(class_count) is not int
        or type(trial) is not int
        or type(count) is not int
        or class_count < 4
        or trial < 0
        or count <= 0
    ):
        raise ValueError("invalid permutation parameters")
    baseline = np.concatenate(
        [np.full(size, rank, dtype=np.int64) for rank, size in enumerate(
            contiguous_shard_sizes(class_count)
        )]
    )
    generator = np.random.Generator(np.random.PCG64(3000 + trial))
    return tuple(generator.permutation(baseline) for _ in range(count))


def shard_decision(
    *,
    independent_loss_range: float,
    independent_gradient_mse: float,
    coherent_gradient_mse: float,
    coherent_invariance_error: float,
    prediction_change_rate: float,
    all_finite: bool,
) -> str:
    scalar_values = (
        independent_loss_range,
        independent_gradient_mse,
        coherent_gradient_mse,
        coherent_invariance_error,
        prediction_change_rate,
    )
    if any(type(value) is not float or not np.isfinite(value) for value in scalar_values):
        raise TypeError("shard decision metrics must be finite builtin floats")
    if type(all_finite) is not bool:
        raise TypeError("all_finite must be a builtin bool")
    sensitive = (
        all_finite
        and independent_loss_range >= 1e-3
        and independent_gradient_mse >= 1.25 * coherent_gradient_mse
        and coherent_invariance_error <= 1e-6
        and prediction_change_rate >= 0.10
    )
    return "SHARD_SENSITIVE" if sensitive else "SHARD_NULL"


def _gradient_cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0 if left_norm == right_norm else 1.0
    cosine = float(np.dot(left_flat, right_flat) / (left_norm * right_norm))
    return float(1.0 - np.clip(cosine, -1.0, 1.0))


def audit_shard_sensitivity(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    *,
    class_count: int = 64,
    examples_per_class: int = 4,
    selected: int = 512,
    trials: int = 32,
    permutations: int = 16,
) -> ShardAudit:
    """Audit class-shard placement sensitivity on one frozen real-embedding panel."""

    if type(trials) is not int or trials <= 0 or type(permutations) is not int:
        raise ValueError("trials and permutations must be positive builtin integers")
    if permutations <= 0:
        raise ValueError("trials and permutations must be positive builtin integers")
    panel = select_shard_panel(
        train_embeddings,
        train_labels,
        class_count=class_count,
        examples_per_class=examples_per_class,
        seed=205,
    )
    if type(selected) is not int or selected <= 0 or selected > panel.embeddings.shape[1]:
        raise ValueError("selected must be within the embedding dimension")
    label_indices = np.searchsorted(panel.class_labels, panel.labels).astype(np.int64)
    full_mask = np.arange(panel.embeddings.shape[1], dtype=np.int64)
    full_masks = (full_mask, full_mask, full_mask, full_mask)

    independent_losses: list[float] = []
    independent_ranges: list[float] = []
    independent_mse: list[float] = []
    coherent_mse: list[float] = []
    independent_cosine: list[float] = []
    coherent_cosine: list[float] = []
    coherent_invariance: list[float] = []
    prediction_changes: list[float] = []
    coverages: list[float] = []

    for trial in range(trials):
        independent_masks = trial_masks(
            dimension=panel.embeddings.shape[1], selected=selected, trial=trial
        )
        coherent_masks = (
            independent_masks[0],
            independent_masks[0],
            independent_masks[0],
            independent_masks[0],
        )
        assignments = class_to_shard_permutations(
            class_count=class_count, trial=trial, count=permutations
        )
        trial_independent_losses: list[float] = []
        reference_predictions: np.ndarray | None = None
        class_order = np.random.Generator(np.random.PCG64(4000 + trial)).permutation(
            class_count
        )
        inverse_order = np.empty(class_count, dtype=np.int64)
        inverse_order[class_order] = np.arange(class_count, dtype=np.int64)
        permuted_prototypes = np.ascontiguousarray(panel.prototypes[class_order])
        permuted_labels = inverse_order[label_indices]
        for assignment in assignments:
            independent = arcface_joint_objective(
                panel.embeddings,
                label_indices,
                panel.prototypes,
                assignment,
                independent_masks,
            )
            coherent = arcface_joint_objective(
                panel.embeddings,
                label_indices,
                panel.prototypes,
                assignment,
                coherent_masks,
            )
            full = arcface_joint_objective(
                panel.embeddings,
                label_indices,
                panel.prototypes,
                assignment,
                full_masks,
            )
            consistent_permutation = arcface_joint_objective(
                panel.embeddings,
                permuted_labels,
                permuted_prototypes,
                np.ascontiguousarray(assignment[class_order]),
                independent_masks,
            )
            trial_independent_losses.append(independent.loss)
            coherent_invariance.append(
                float(abs(consistent_permutation.loss - independent.loss))
            )
            independent_mse.append(
                float(np.mean((independent.embedding_gradient - full.embedding_gradient) ** 2))
            )
            coherent_mse.append(
                float(np.mean((coherent.embedding_gradient - full.embedding_gradient) ** 2))
            )
            independent_cosine.append(
                _gradient_cosine_distance(independent.embedding_gradient, full.embedding_gradient)
            )
            coherent_cosine.append(
                _gradient_cosine_distance(coherent.embedding_gradient, full.embedding_gradient)
            )
            if reference_predictions is None:
                reference_predictions = independent.predictions
            else:
                prediction_changes.append(
                    float(np.mean(independent.predictions != reference_predictions))
                )
        independent_losses.extend(trial_independent_losses)
        independent_ranges.append(
            float(max(trial_independent_losses) - min(trial_independent_losses))
        )
        union = np.unique(np.concatenate(independent_masks))
        coverages.append(float(union.size / panel.embeddings.shape[1]))

    scalar_arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            independent_losses,
            independent_ranges,
            independent_mse,
            coherent_mse,
            independent_cosine,
            coherent_cosine,
            coherent_invariance,
            prediction_changes or [0.0],
            coverages,
        )
    )
    all_finite = bool(all(np.isfinite(values).all() for values in scalar_arrays))
    loss_range = float(np.max(scalar_arrays[1]))
    independent_gradient_mse = float(np.mean(scalar_arrays[2], dtype=np.float64))
    coherent_gradient_mse = float(np.mean(scalar_arrays[3], dtype=np.float64))
    invariance_error = float(np.max(scalar_arrays[6]))
    prediction_change_rate = float(np.mean(scalar_arrays[7], dtype=np.float64))
    decision = shard_decision(
        independent_loss_range=loss_range,
        independent_gradient_mse=independent_gradient_mse,
        coherent_gradient_mse=coherent_gradient_mse,
        coherent_invariance_error=invariance_error,
        prediction_change_rate=prediction_change_rate,
        all_finite=all_finite,
    )
    return ShardAudit(
        trials=trials,
        permutations_per_trial=permutations,
        independent_loss_range=loss_range,
        independent_loss_std=float(np.std(scalar_arrays[0])),
        independent_gradient_mse=independent_gradient_mse,
        coherent_gradient_mse=coherent_gradient_mse,
        independent_gradient_cosine_distance=float(np.mean(scalar_arrays[4])),
        coherent_gradient_cosine_distance=float(np.mean(scalar_arrays[5])),
        coherent_invariance_error=invariance_error,
        prediction_change_rate=prediction_change_rate,
        mask_union_coverage=float(np.mean(scalar_arrays[8])),
        all_finite=all_finite,
        decision=decision,
    )
