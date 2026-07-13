from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from sfora.losses import group_triplet_margin_loss, triplet_margin_loss

Objective = Literal[
    "triplet",
    "batch_hard_triplet",
    "group",
    "hard_group",
    "supcon",
    "proxy_nca",
    "proxy_anchor",
    "cosface",
    "arcface",
    "group_supcon",
    "hybrid",
    "hybrid_xbm",
    "hybrid_radius",
    "hybrid_xbm_radius",
    "group_supcon_xbm_radius",
]


class ProjectionTrainingConfig(BaseModel):
    """Configuration for lightweight embedding-table objective optimization."""

    objective: Objective = "triplet"
    group_size: int = Field(default=4, ge=1)
    steps: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=0.03, gt=0.0)
    margin: float = Field(default=0.5, ge=0.0)
    hard_weight: float = Field(default=0.5, ge=0.0)
    spread_weight: float = Field(default=0.1, ge=0.0)
    triplet_weight: float = Field(default=1.0, ge=0.0)
    group_weight: float = Field(default=1.0, ge=0.0)
    xbm_weight: float = Field(default=0.25, ge=0.0)
    xbm_memory_size: int = Field(default=1024, ge=0)
    radius_weight: float = Field(default=0.05, ge=0.0)
    radius_target: float = Field(default=0.0, ge=0.0)
    variance_weight: float = Field(default=0.05, ge=0.0)
    temperature: float = Field(default=0.07, gt=0.0)
    shuffle_groups_each_step: bool = False


class ProjectionHeadTrainingConfig(ProjectionTrainingConfig):
    """Configuration for training a reusable linear projection head."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dimensions: int | None = None
    normalize_projected_embeddings: bool = False
    selection_score_callback: Callable[[NDArray[np.float64], int], float] | None = Field(
        default=None,
        exclude=True,
    )
    seed: int = 0


@dataclass(frozen=True)
class ProjectionTrainingResult:
    """Result from optimizing a trainable embedding table."""

    objective: Objective
    transformed_embeddings: NDArray[np.float64]
    initial_triplet_loss: float
    final_triplet_loss: float
    initial_group_loss: float
    final_group_loss: float
    history: list[float]


@dataclass(frozen=True)
class ProjectionHeadTrainingResult(ProjectionTrainingResult):
    """Result from optimizing a reusable projection matrix."""

    projection_matrix: NDArray[np.float64]
    selected_step: int
    selection_score: float | None


def train_embedding_table(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    config: ProjectionTrainingConfig | None = None,
) -> ProjectionTrainingResult:
    """Optimize embeddings directly with triplet or group-triplet SGD updates."""
    resolved_config = config or ProjectionTrainingConfig()
    trained = np.asarray(embeddings, dtype=np.float64).copy()
    label_array = np.asarray(labels, dtype=np.int64)
    _validate_inputs(trained, label_array)

    triplets = _make_triplet_indices(label_array)
    group_triplets = _make_required_group_triplet_indices(label_array, resolved_config)
    initial_triplet_loss, initial_group_loss = _losses(
        trained, triplets, group_triplets, resolved_config
    )
    history = [_objective_loss(initial_triplet_loss, initial_group_loss, resolved_config.objective)]

    memory_embeddings = np.empty((0, trained.shape[1]), dtype=np.float64)
    memory_labels = np.empty((0,), dtype=np.int64)
    for step in range(1, resolved_config.steps + 1):
        step_group_triplets = _step_group_triplets(
            label_array,
            resolved_config,
            fallback=group_triplets,
            step=step,
        )
        gradient = _objective_gradient(
            trained,
            label_array,
            triplets,
            step_group_triplets,
            resolved_config,
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
        )
        trained -= resolved_config.learning_rate * gradient
        memory_embeddings, memory_labels = _updated_memory(
            memory_embeddings,
            memory_labels,
            trained,
            label_array,
            memory_size=resolved_config.xbm_memory_size,
        )
        triplet_loss, group_loss = _losses(trained, triplets, group_triplets, resolved_config)
        history.append(_objective_loss(triplet_loss, group_loss, resolved_config.objective))

    final_triplet_loss, final_group_loss = _losses(
        trained, triplets, group_triplets, resolved_config
    )
    return ProjectionTrainingResult(
        objective=resolved_config.objective,
        transformed_embeddings=trained,
        initial_triplet_loss=initial_triplet_loss,
        final_triplet_loss=final_triplet_loss,
        initial_group_loss=initial_group_loss,
        final_group_loss=final_group_loss,
        history=history,
    )


def train_projection_head(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    config: ProjectionHeadTrainingConfig | None = None,
) -> ProjectionHeadTrainingResult:
    """Optimize a linear projection head with triplet or group-triplet objectives."""
    resolved_config = config or ProjectionHeadTrainingConfig()
    input_embeddings = np.asarray(embeddings, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    _validate_inputs(input_embeddings, label_array)

    projection = _initial_projection(input_embeddings.shape[1], resolved_config)
    triplets = _make_triplet_indices(label_array)
    group_triplets = _make_required_group_triplet_indices(label_array, resolved_config)
    transformed = _project_embeddings(input_embeddings, projection, resolved_config)
    initial_triplet_loss, initial_group_loss = _losses(
        transformed, triplets, group_triplets, resolved_config
    )
    history = [_objective_loss(initial_triplet_loss, initial_group_loss, resolved_config.objective)]
    best_projection = projection.copy()
    selected_step = 0
    selection_score = _selection_score(resolved_config, projection, selected_step)
    best_selection_score = selection_score
    memory_embeddings = np.empty(
        (0, resolved_config.output_dimensions or input_embeddings.shape[1]),
        dtype=np.float64,
    )
    memory_labels = np.empty((0,), dtype=np.int64)

    for step in range(1, resolved_config.steps + 1):
        step_group_triplets = _step_group_triplets(
            label_array,
            resolved_config,
            fallback=group_triplets,
            step=step,
        )
        raw_transformed = input_embeddings @ projection
        transformed = (
            _normalize(raw_transformed)
            if resolved_config.normalize_projected_embeddings
            else raw_transformed
        )
        transformed_gradient = _objective_gradient(
            transformed,
            label_array,
            triplets,
            step_group_triplets,
            resolved_config,
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
        )
        raw_gradient = (
            _normalization_gradient(raw_transformed, transformed_gradient)
            if resolved_config.normalize_projected_embeddings
            else transformed_gradient
        )
        projection -= resolved_config.learning_rate * input_embeddings.T @ raw_gradient
        projected_for_metrics = _project_embeddings(input_embeddings, projection, resolved_config)
        memory_embeddings, memory_labels = _updated_memory(
            memory_embeddings,
            memory_labels,
            projected_for_metrics,
            label_array,
            memory_size=resolved_config.xbm_memory_size,
        )
        triplet_loss, group_loss = _losses(
            projected_for_metrics, triplets, group_triplets, resolved_config
        )
        history.append(_objective_loss(triplet_loss, group_loss, resolved_config.objective))
        current_score = _selection_score(resolved_config, projection, step)
        if current_score is not None and (
            best_selection_score is None or current_score > best_selection_score
        ):
            best_projection = projection.copy()
            selected_step = step
            best_selection_score = current_score

    if best_selection_score is not None:
        projection = best_projection
    else:
        selected_step = resolved_config.steps
    transformed = _project_embeddings(input_embeddings, projection, resolved_config)
    final_triplet_loss, final_group_loss = _losses(
        transformed, triplets, group_triplets, resolved_config
    )
    return ProjectionHeadTrainingResult(
        objective=resolved_config.objective,
        transformed_embeddings=transformed,
        initial_triplet_loss=initial_triplet_loss,
        final_triplet_loss=final_triplet_loss,
        initial_group_loss=initial_group_loss,
        final_group_loss=final_group_loss,
        history=history,
        projection_matrix=projection,
        selected_step=selected_step,
        selection_score=best_selection_score,
    )


def _selection_score(
    config: ProjectionHeadTrainingConfig,
    projection: NDArray[np.float64],
    step: int,
) -> float | None:
    if config.selection_score_callback is None:
        return None
    return float(config.selection_score_callback(projection.copy(), step))


def _initial_projection(
    input_dimensions: int,
    config: ProjectionHeadTrainingConfig,
) -> NDArray[np.float64]:
    output_dimensions = config.output_dimensions or input_dimensions
    if output_dimensions < 1:
        raise ValueError("output_dimensions must be at least 1")
    if output_dimensions == input_dimensions:
        return np.eye(input_dimensions, dtype=np.float64)

    rng = np.random.default_rng(config.seed)
    return rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(input_dimensions),
        size=(input_dimensions, output_dimensions),
    )


def _project_embeddings(
    embeddings: NDArray[np.float64],
    projection: NDArray[np.float64],
    config: ProjectionHeadTrainingConfig,
) -> NDArray[np.float64]:
    projected = embeddings @ projection
    if config.normalize_projected_embeddings:
        return _normalize(projected)
    return projected


def _normalize(embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _normalization_gradient(
    raw_embeddings: NDArray[np.float64],
    normalized_gradient: NDArray[np.float64],
) -> NDArray[np.float64]:
    norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
    safe_norms = np.maximum(norms, 1e-12)
    normalized = raw_embeddings / safe_norms
    radial_component = np.sum(normalized_gradient * normalized, axis=1, keepdims=True)
    return (normalized_gradient - normalized * radial_component) / safe_norms


def _validate_inputs(embeddings: NDArray[np.float64], labels: NDArray[np.int64]) -> None:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must contain the same number of examples")
    if np.unique(labels).shape[0] < 2:
        raise ValueError("training requires at least two labels")


def _make_triplet_indices(labels: NDArray[np.int64]) -> list[tuple[int, int, int]]:
    triplets: list[tuple[int, int, int]] = []
    for label in np.unique(labels):
        same_indices = np.flatnonzero(labels == label)
        other_indices = np.flatnonzero(labels != label)
        if same_indices.shape[0] < 2:
            raise ValueError("triplet training requires at least two examples per label")
        for position, anchor_index in enumerate(same_indices):
            positive_index = same_indices[(position + 1) % same_indices.shape[0]]
            negative_index = other_indices[position % other_indices.shape[0]]
            triplets.append((int(anchor_index), int(positive_index), int(negative_index)))
    return triplets


def _make_group_triplet_indices(
    labels: NDArray[np.int64],
    *,
    group_size: int,
    require_distinct_positive: bool = True,
    random_state: int | None = None,
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    rng = None if random_state is None else np.random.default_rng(random_state)
    grouped_by_label: dict[int, list[tuple[int, ...]]] = {}
    for label in np.unique(labels):
        label_indices = np.flatnonzero(labels == label)
        if rng is not None:
            label_indices = label_indices.copy()
            rng.shuffle(label_indices)
        usable_count = label_indices.shape[0] - (label_indices.shape[0] % group_size)
        if usable_count < group_size:
            raise ValueError("group training requires at least group_size examples per label")
        if require_distinct_positive and usable_count < group_size * 2:
            raise ValueError(
                "group training requires at least two groups per label; "
                "increase min_per_class or reduce group_size"
            )
        grouped_by_label[int(label)] = [
            tuple(int(index) for index in label_indices[start : start + group_size])
            for start in range(0, usable_count, group_size)
        ]
        if rng is not None:
            rng.shuffle(grouped_by_label[int(label)])

    group_triplets: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for label in sorted(grouped_by_label):
        same_groups = grouped_by_label[label]
        other_groups = [
            group
            for other_label in sorted(grouped_by_label)
            if other_label != label
            for group in grouped_by_label[other_label]
        ]
        for position, anchor_group in enumerate(same_groups):
            group_triplets.append(
                (
                    anchor_group,
                    same_groups[(position + 1) % len(same_groups)],
                    other_groups[position % len(other_groups)],
                )
            )
    return group_triplets


def _make_required_group_triplet_indices(
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
    *,
    random_state: int | None = None,
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    if not _uses_group_objective(config.objective):
        return []
    return _make_group_triplet_indices(
        labels,
        group_size=config.group_size,
        require_distinct_positive=True,
        random_state=random_state,
    )


def _step_group_triplets(
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
    *,
    fallback: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    step: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    if not config.shuffle_groups_each_step or not _uses_group_objective(config.objective):
        return fallback
    seed = getattr(config, "seed", 0)
    return _make_required_group_triplet_indices(labels, config, random_state=seed + step)


def _losses(
    embeddings: NDArray[np.float64],
    triplets: list[tuple[int, int, int]],
    group_triplets: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    config: ProjectionTrainingConfig,
) -> tuple[float, float]:
    anchors = np.stack([embeddings[anchor] for anchor, _, _ in triplets])
    positives = np.stack([embeddings[positive] for _, positive, _ in triplets])
    negatives = np.stack([embeddings[negative] for _, _, negative in triplets])
    triplet_loss = triplet_margin_loss(anchors, positives, negatives, margin=config.margin)
    if not group_triplets:
        return triplet_loss, 0.0

    anchor_groups = np.stack([embeddings[list(anchor)] for anchor, _, _ in group_triplets])
    positive_groups = np.stack([embeddings[list(positive)] for _, positive, _ in group_triplets])
    negative_groups = np.stack([embeddings[list(negative)] for _, _, negative in group_triplets])
    group_loss = group_triplet_margin_loss(
        anchor_groups,
        positive_groups,
        negative_groups,
        margin=config.margin,
        hard_weight=config.hard_weight,
        spread_weight=config.spread_weight,
    )
    return triplet_loss, group_loss


def _objective_loss(triplet_loss: float, group_loss: float, objective: Objective) -> float:
    if objective in {
        "triplet",
        "batch_hard_triplet",
        "supcon",
        "proxy_nca",
        "proxy_anchor",
        "cosface",
        "arcface",
    }:
        return triplet_loss
    if objective in {"group", "hard_group", "group_supcon"}:
        return group_loss
    return triplet_loss + group_loss


def _objective_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    triplets: list[tuple[int, int, int]],
    group_triplets: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    config: ProjectionTrainingConfig,
    *,
    memory_embeddings: NDArray[np.float64] | None = None,
    memory_labels: NDArray[np.int64] | None = None,
) -> NDArray[np.float64]:
    if config.objective == "triplet":
        return _triplet_gradient(embeddings, triplets, config)
    if config.objective == "batch_hard_triplet":
        return _hard_triplet_gradient(embeddings, labels, config)
    if config.objective == "group":
        return _group_gradient(embeddings, group_triplets, config)
    if config.objective == "hard_group":
        return _hard_group_gradient(embeddings, labels, group_triplets, config)
    if config.objective == "supcon":
        return _supervised_contrastive_gradient(embeddings, labels, config)
    if config.objective == "proxy_nca":
        return _proxy_nca_gradient(embeddings, labels, config)
    if config.objective == "proxy_anchor":
        return _proxy_anchor_gradient(embeddings, labels, config)
    if config.objective == "cosface":
        return _cosface_gradient(embeddings, labels, config)
    if config.objective == "arcface":
        return _arcface_gradient(embeddings, labels, config)
    if config.objective == "group_supcon":
        return config.triplet_weight * _supervised_contrastive_gradient(
            embeddings, labels, config
        ) + config.group_weight * _group_supervised_contrastive_gradient(
            embeddings, labels, group_triplets, config
        )
    if config.objective == "group_supcon_xbm_radius":
        xbm_gradient = _xbm_gradient(embeddings, labels, config, memory_embeddings, memory_labels)
        return _group_supcon_xbm_radius_gradient(
            embeddings,
            labels,
            group_triplets,
            config,
            xbm_gradient=xbm_gradient,
        )

    gradient = (config.triplet_weight * _triplet_gradient(embeddings, triplets, config)) + (
        config.group_weight * _group_gradient(embeddings, group_triplets, config)
    )
    if config.objective in {"hybrid_xbm", "hybrid_xbm_radius"}:
        xbm_gradient = _xbm_gradient(embeddings, labels, config, memory_embeddings, memory_labels)
        gradient += config.xbm_weight * xbm_gradient
    if config.objective in {"hybrid_radius", "hybrid_xbm_radius"}:
        gradient += _radius_variance_gradient(embeddings, labels, config)
    return gradient


def _xbm_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
    memory_embeddings: NDArray[np.float64] | None,
    memory_labels: NDArray[np.int64] | None,
) -> NDArray[np.float64]:
    if (
        config.xbm_memory_size > 0
        and memory_embeddings is not None
        and memory_labels is not None
        and memory_embeddings.shape[0] > 0
    ):
        return _memory_hard_triplet_gradient(
            embeddings,
            labels,
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
            config=config,
        )
    return _hard_triplet_gradient(embeddings, labels, config)


def _uses_group_objective(objective: Objective) -> bool:
    return objective in {
        "group",
        "hard_group",
        "hybrid",
        "hybrid_xbm",
        "hybrid_radius",
        "hybrid_xbm_radius",
        "group_supcon",
        "group_supcon_xbm_radius",
    }


def _group_supcon_xbm_radius_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    group_triplets: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    config: ProjectionTrainingConfig,
    *,
    xbm_gradient: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (
        config.triplet_weight * _supervised_contrastive_gradient(embeddings, labels, config)
        + config.group_weight
        * _group_supervised_contrastive_gradient(embeddings, labels, group_triplets, config)
        + config.xbm_weight * xbm_gradient
        + config.hard_weight * _hard_group_gradient(embeddings, labels, group_triplets, config)
        + _radius_variance_gradient(embeddings, labels, config)
    )


def _hard_triplet_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    triplets: list[tuple[int, int, int]] = []
    embedding_norms = np.sum(embeddings * embeddings, axis=1)
    chunk_size = 2048
    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        chunk = embeddings[start:stop]
        chunk_indices = np.arange(start, stop)
        distances = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + embedding_norms[np.newaxis, :]
            - (2.0 * chunk @ embeddings.T)
        )
        distances = np.maximum(distances, 0.0)
        same_label = labels[chunk_indices, np.newaxis] == labels[np.newaxis, :]
        same_label[np.arange(chunk_indices.shape[0]), chunk_indices] = False
        other_label = ~same_label
        other_label[np.arange(chunk_indices.shape[0]), chunk_indices] = False
        positive_distances = np.where(same_label, distances, -np.inf)
        negative_distances = np.where(other_label, distances, np.inf)
        positive_indices = np.argmax(positive_distances, axis=1)
        negative_indices = np.argmin(negative_distances, axis=1)
        has_positive = np.isfinite(
            positive_distances[np.arange(chunk_indices.shape[0]), positive_indices]
        )
        has_negative = np.isfinite(
            negative_distances[np.arange(chunk_indices.shape[0]), negative_indices]
        )
        for row_position, anchor_index in enumerate(chunk_indices):
            if not (has_positive[row_position] and has_negative[row_position]):
                continue
            triplets.append(
                (
                    int(anchor_index),
                    int(positive_indices[row_position]),
                    int(negative_indices[row_position]),
                )
            )
    return (
        _triplet_gradient(embeddings, triplets, config) if triplets else np.zeros_like(embeddings)
    )


def _memory_hard_triplet_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    memory_embeddings: NDArray[np.float64],
    memory_labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    if memory_embeddings.shape[0] == 0:
        candidate_embeddings = embeddings
        candidate_labels = labels
    else:
        candidate_embeddings = np.concatenate([embeddings, memory_embeddings], axis=0)
        candidate_labels = np.concatenate([labels, memory_labels], axis=0)

    gradient = np.zeros_like(embeddings)
    triplet_count = 0
    current_count = embeddings.shape[0]
    candidate_norms = np.sum(candidate_embeddings * candidate_embeddings, axis=1)
    chunk_size = 2048
    for start in range(0, current_count, chunk_size):
        stop = min(start + chunk_size, current_count)
        chunk = embeddings[start:stop]
        chunk_indices = np.arange(start, stop)
        distances = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + candidate_norms[np.newaxis, :]
            - (2.0 * chunk @ candidate_embeddings.T)
        )
        distances = np.maximum(distances, 0.0)
        same_label = labels[chunk_indices, np.newaxis] == candidate_labels[np.newaxis, :]
        same_label[np.arange(chunk_indices.shape[0]), chunk_indices] = False
        other_label = ~same_label
        other_label[np.arange(chunk_indices.shape[0]), chunk_indices] = False
        positive_distances = np.where(same_label, distances, -np.inf)
        negative_distances = np.where(other_label, distances, np.inf)
        positive_indices = np.argmax(positive_distances, axis=1)
        negative_indices = np.argmin(negative_distances, axis=1)
        has_positive = np.isfinite(
            positive_distances[np.arange(chunk_indices.shape[0]), positive_indices]
        )
        has_negative = np.isfinite(
            negative_distances[np.arange(chunk_indices.shape[0]), negative_indices]
        )
        for row_position, anchor_index in enumerate(chunk_indices):
            if not (has_positive[row_position] and has_negative[row_position]):
                continue
            positive_index = int(positive_indices[row_position])
            negative_index = int(negative_indices[row_position])
            anchor = embeddings[anchor_index]
            positive = candidate_embeddings[positive_index]
            negative = candidate_embeddings[negative_index]
            positive_distance = _distance(anchor, positive)
            negative_distance = _distance(anchor, negative)
            if positive_distance - negative_distance + config.margin <= 0.0:
                continue

            positive_unit = _unit(anchor - positive)
            negative_unit = _unit(anchor - negative)
            gradient[anchor_index] += positive_unit - negative_unit
            if positive_index < current_count:
                gradient[positive_index] += -positive_unit
            if negative_index < current_count:
                gradient[negative_index] += negative_unit
            triplet_count += 1

    return gradient / max(triplet_count, 1)


def _updated_memory(
    memory_embeddings: NDArray[np.float64],
    memory_labels: NDArray[np.int64],
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    memory_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    if memory_size == 0:
        return memory_embeddings[:0], memory_labels[:0]
    combined_embeddings = np.concatenate([memory_embeddings, embeddings.copy()], axis=0)
    combined_labels = np.concatenate([memory_labels, labels.copy()], axis=0)
    if combined_embeddings.shape[0] <= memory_size:
        return combined_embeddings, combined_labels
    return combined_embeddings[-memory_size:], combined_labels[-memory_size:]


def _hard_group_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    group_triplets: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    groups = [anchor for anchor, _, _ in group_triplets]
    if not groups:
        return np.zeros_like(embeddings)

    group_labels = np.asarray([labels[group[0]] for group in groups], dtype=np.int64)
    centroids = np.stack([embeddings[list(group)].mean(axis=0) for group in groups])
    gradient = np.zeros_like(embeddings)

    for anchor_position, anchor_indices in enumerate(groups):
        same_label = group_labels == group_labels[anchor_position]
        same_label[anchor_position] = False
        other_label = group_labels != group_labels[anchor_position]
        if not np.any(same_label) or not np.any(other_label):
            continue

        distances = np.linalg.norm(centroids - centroids[anchor_position], axis=1)
        positive_position = int(np.argmax(np.where(same_label, distances, -np.inf)))
        negative_position = int(np.argmin(np.where(other_label, distances, np.inf)))
        positive_indices = groups[positive_position]
        negative_indices = groups[negative_position]
        positive_group = embeddings[list(positive_indices)]
        negative_group = embeddings[list(negative_indices)]
        anchor_group = embeddings[list(anchor_indices)]

        _add_centroid_margin_gradient(
            gradient,
            anchor_indices,
            positive_indices,
            negative_indices,
            centroids[anchor_position],
            centroids[positive_position],
            centroids[negative_position],
            config.margin,
        )
        _add_hard_member_gradient(
            gradient,
            positive_indices,
            negative_indices,
            anchor_indices,
            positive_group,
            negative_group,
            centroids[anchor_position],
            config,
        )
        _add_spread_gradient(
            gradient,
            anchor_indices,
            anchor_group,
            centroids[anchor_position],
            config,
        )
        _add_spread_gradient(
            gradient,
            positive_indices,
            positive_group,
            centroids[positive_position],
            config,
        )
        _add_spread_gradient(
            gradient,
            negative_indices,
            negative_group,
            centroids[negative_position],
            config,
        )

    return gradient / max(len(groups), 1)


def _supervised_contrastive_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    gradient = np.zeros_like(embeddings)
    anchor_count = embeddings.shape[0]
    if anchor_count < 2:
        return gradient

    chunk_size = 512
    scale = 1.0 / config.temperature
    for start in range(0, anchor_count, chunk_size):
        stop = min(start + chunk_size, anchor_count)
        chunk = embeddings[start:stop]
        chunk_indices = np.arange(start, stop)
        similarities = scale * (chunk @ embeddings.T)
        similarities[np.arange(chunk_indices.shape[0]), chunk_indices] = -np.inf

        positive_mask = labels[chunk_indices, np.newaxis] == labels[np.newaxis, :]
        positive_mask[np.arange(chunk_indices.shape[0]), chunk_indices] = False
        positive_counts = positive_mask.sum(axis=1, keepdims=True)
        valid_rows = positive_counts[:, 0] > 0
        if not np.any(valid_rows):
            continue

        valid_similarities = similarities[valid_rows]
        row_max = np.max(valid_similarities, axis=1, keepdims=True)
        exp_similarities = np.exp(valid_similarities - row_max)
        exp_similarities[~np.isfinite(valid_similarities)] = 0.0
        probabilities = exp_similarities / np.maximum(
            exp_similarities.sum(axis=1, keepdims=True),
            1e-12,
        )
        target = positive_mask[valid_rows].astype(np.float64) / positive_counts[valid_rows]
        coefficients = scale * (probabilities - target) / anchor_count
        valid_chunk = chunk[valid_rows]
        valid_indices = chunk_indices[valid_rows]

        gradient[valid_indices] += coefficients @ embeddings
        gradient += coefficients.T @ valid_chunk

    return gradient


def _group_supervised_contrastive_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    group_triplets: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    groups = [anchor for anchor, _, _ in group_triplets]
    if not groups:
        return np.zeros_like(embeddings)

    group_labels = np.asarray([labels[group[0]] for group in groups], dtype=np.int64)
    raw_centroids = np.stack([embeddings[list(group)].mean(axis=0) for group in groups])
    normalized_centroids = _normalize(raw_centroids)
    centroid_gradient = _supervised_contrastive_gradient(
        normalized_centroids,
        group_labels,
        config,
    )
    centroid_gradient = _normalization_gradient(raw_centroids, centroid_gradient)
    gradient = np.zeros_like(embeddings)
    for group, group_gradient in zip(groups, centroid_gradient, strict=True):
        _add_group_centroid_gradient(gradient, group, group_gradient)
    return gradient


def _proxy_nca_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    unique_labels = np.asarray(sorted(np.unique(labels).tolist()), dtype=np.int64)
    if unique_labels.shape[0] < 2:
        return np.zeros_like(embeddings)

    centroids = np.stack([embeddings[labels == label].mean(axis=0) for label in unique_labels])
    label_positions = {int(label): position for position, label in enumerate(unique_labels)}
    targets = np.asarray([label_positions[int(label)] for label in labels], dtype=np.int64)
    gradient = np.zeros_like(embeddings)
    scale = 1.0 / config.temperature

    chunk_size = 1024
    centroid_norms = np.sum(centroids * centroids, axis=1)
    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        chunk = embeddings[start:stop]
        target_positions = targets[start:stop]
        squared_distances = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + centroid_norms[np.newaxis, :]
            - (2.0 * chunk @ centroids.T)
        )
        logits = -scale * np.maximum(squared_distances, 0.0)
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        probabilities[np.arange(stop - start), target_positions] -= 1.0
        gradient[start:stop] = 2.0 * scale * probabilities @ centroids / embeddings.shape[0]

    return gradient


def _cosface_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    unique_labels = np.asarray(sorted(np.unique(labels).tolist()), dtype=np.int64)
    if unique_labels.shape[0] < 2:
        return np.zeros_like(embeddings)

    normalized_embeddings = _normalize(embeddings)
    centroids = _normalize(
        np.stack([normalized_embeddings[labels == label].mean(axis=0) for label in unique_labels])
    )
    label_positions = {int(label): position for position, label in enumerate(unique_labels)}
    targets = np.asarray([label_positions[int(label)] for label in labels], dtype=np.int64)
    normalized_gradient = np.zeros_like(embeddings)
    scale = 1.0 / config.temperature

    chunk_size = 1024
    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        chunk = normalized_embeddings[start:stop]
        target_positions = targets[start:stop]
        logits = scale * (chunk @ centroids.T)
        logits[np.arange(stop - start), target_positions] -= scale * config.margin
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        probabilities[np.arange(stop - start), target_positions] -= 1.0
        normalized_gradient[start:stop] = scale * probabilities @ centroids / embeddings.shape[0]

    return _normalization_gradient(embeddings, normalized_gradient)


def _proxy_anchor_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    unique_labels = np.asarray(sorted(np.unique(labels).tolist()), dtype=np.int64)
    if unique_labels.shape[0] < 2:
        return np.zeros_like(embeddings)

    normalized_embeddings = _normalize(embeddings)
    centroids = _normalize(
        np.stack([normalized_embeddings[labels == label].mean(axis=0) for label in unique_labels])
    )
    label_positions = {int(label): position for position, label in enumerate(unique_labels)}
    targets = np.asarray([label_positions[int(label)] for label in labels], dtype=np.int64)
    scale = 1.0 / config.temperature
    class_count = unique_labels.shape[0]

    similarities = normalized_embeddings @ centroids.T
    positive_exp = np.exp(
        -scale * (similarities[np.arange(embeddings.shape[0]), targets] - config.margin)
    )
    positive_denominators = np.ones(class_count, dtype=np.float64)
    np.add.at(positive_denominators, targets, positive_exp)
    negative_exp = np.exp(scale * (similarities + config.margin))
    negative_exp[np.arange(embeddings.shape[0]), targets] = 0.0
    negative_denominators = 1.0 + negative_exp.sum(axis=0)

    coefficients = scale * negative_exp / (negative_denominators[np.newaxis, :] * class_count)
    coefficients[np.arange(embeddings.shape[0]), targets] = (
        -scale * positive_exp / (positive_denominators[targets] * class_count)
    )
    normalized_gradient = coefficients @ centroids / embeddings.shape[0]
    return _normalization_gradient(embeddings, normalized_gradient)


def _arcface_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    unique_labels = np.asarray(sorted(np.unique(labels).tolist()), dtype=np.int64)
    if unique_labels.shape[0] < 2:
        return np.zeros_like(embeddings)

    normalized_embeddings = _normalize(embeddings)
    centroids = _normalize(
        np.stack([normalized_embeddings[labels == label].mean(axis=0) for label in unique_labels])
    )
    label_positions = {int(label): position for position, label in enumerate(unique_labels)}
    targets = np.asarray([label_positions[int(label)] for label in labels], dtype=np.int64)
    normalized_gradient = np.zeros_like(embeddings)
    scale = 1.0 / config.temperature
    margin_cos = float(np.cos(config.margin))
    margin_sin = float(np.sin(config.margin))

    chunk_size = 1024
    for start in range(0, embeddings.shape[0], chunk_size):
        stop = min(start + chunk_size, embeddings.shape[0])
        chunk = normalized_embeddings[start:stop]
        target_positions = targets[start:stop]
        cosine_logits = chunk @ centroids.T
        target_cosines = np.clip(
            cosine_logits[np.arange(stop - start), target_positions],
            -1.0 + 1e-6,
            1.0 - 1e-6,
        )
        target_sines = np.sqrt(np.maximum(1.0 - (target_cosines * target_cosines), 1e-12))
        logits = scale * cosine_logits
        logits[np.arange(stop - start), target_positions] = scale * (
            (target_cosines * margin_cos) - (target_sines * margin_sin)
        )
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        coefficients = probabilities * scale
        target_derivatives = margin_cos + ((target_cosines / target_sines) * margin_sin)
        coefficients[np.arange(stop - start), target_positions] = (
            (probabilities[np.arange(stop - start), target_positions] - 1.0)
            * scale
            * target_derivatives
        )
        normalized_gradient[start:stop] = coefficients @ centroids / embeddings.shape[0]

    return _normalization_gradient(embeddings, normalized_gradient)


def _radius_variance_gradient(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    gradient = np.zeros_like(embeddings)
    centroids: list[tuple[int, NDArray[np.float64], NDArray[np.int64]]] = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        if indices.shape[0] < 2:
            continue
        members = embeddings[indices]
        centroid = members.mean(axis=0)
        centered = members - centroid
        if config.radius_weight > 0.0:
            distances = np.linalg.norm(centered, axis=1)
            mean_radius = float(np.mean(distances))
            radius_error = mean_radius - config.radius_target
            if radius_error != 0.0:
                units = np.divide(
                    centered,
                    np.maximum(distances[:, np.newaxis], 1e-12),
                    out=np.zeros_like(centered),
                    where=distances[:, np.newaxis] > 0.0,
                )
                gradient[indices] += config.radius_weight * radius_error * units / indices.shape[0]
        gradient[indices] += config.variance_weight * centered / indices.shape[0]
        centroids.append((int(label), centroid, indices))

    centroid_forces = [np.zeros_like(centroid) for _, centroid, _ in centroids]
    active_pair_counts = np.zeros(len(centroids), dtype=np.int64)
    for left_position, (_, left_centroid, _) in enumerate(centroids):
        for right_position in range(left_position + 1, len(centroids)):
            _, right_centroid, _ = centroids[right_position]
            distance = _distance(left_centroid, right_centroid)
            if distance >= config.margin:
                continue
            unit = _unit(left_centroid - right_centroid)
            centroid_forces[left_position] += -unit
            centroid_forces[right_position] += unit
            active_pair_counts[left_position] += 1
            active_pair_counts[right_position] += 1

    for position, (_, _, indices) in enumerate(centroids):
        active_pair_count = active_pair_counts[position]
        if active_pair_count == 0:
            continue
        average_force = centroid_forces[position] / active_pair_count
        gradient[indices] += average_force / indices.shape[0]
    return gradient


def _triplet_gradient(
    embeddings: NDArray[np.float64],
    triplets: list[tuple[int, int, int]],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    gradient = np.zeros_like(embeddings)
    for anchor_index, positive_index, negative_index in triplets:
        anchor = embeddings[anchor_index]
        positive = embeddings[positive_index]
        negative = embeddings[negative_index]
        positive_distance = _distance(anchor, positive)
        negative_distance = _distance(anchor, negative)
        if positive_distance - negative_distance + config.margin <= 0.0:
            continue

        positive_unit = _unit(anchor - positive)
        negative_unit = _unit(anchor - negative)
        gradient[anchor_index] += positive_unit - negative_unit
        gradient[positive_index] += -positive_unit
        gradient[negative_index] += negative_unit

    return gradient / max(len(triplets), 1)


def _group_gradient(
    embeddings: NDArray[np.float64],
    group_triplets: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    config: ProjectionTrainingConfig,
) -> NDArray[np.float64]:
    gradient = np.zeros_like(embeddings)
    for anchor_indices, positive_indices, negative_indices in group_triplets:
        anchor_group = embeddings[list(anchor_indices)]
        positive_group = embeddings[list(positive_indices)]
        negative_group = embeddings[list(negative_indices)]
        anchor_centroid = anchor_group.mean(axis=0)
        positive_centroid = positive_group.mean(axis=0)
        negative_centroid = negative_group.mean(axis=0)

        _add_centroid_margin_gradient(
            gradient,
            anchor_indices,
            positive_indices,
            negative_indices,
            anchor_centroid,
            positive_centroid,
            negative_centroid,
            config.margin,
        )
        _add_hard_member_gradient(
            gradient,
            positive_indices,
            negative_indices,
            anchor_indices,
            positive_group,
            negative_group,
            anchor_centroid,
            config,
        )
        _add_spread_gradient(gradient, anchor_indices, anchor_group, anchor_centroid, config)
        _add_spread_gradient(gradient, positive_indices, positive_group, positive_centroid, config)
        _add_spread_gradient(gradient, negative_indices, negative_group, negative_centroid, config)

    return gradient / max(len(group_triplets), 1)


def _add_centroid_margin_gradient(
    gradient: NDArray[np.float64],
    anchor_indices: tuple[int, ...],
    positive_indices: tuple[int, ...],
    negative_indices: tuple[int, ...],
    anchor_centroid: NDArray[np.float64],
    positive_centroid: NDArray[np.float64],
    negative_centroid: NDArray[np.float64],
    margin: float,
) -> None:
    positive_distance = _distance(anchor_centroid, positive_centroid)
    negative_distance = _distance(anchor_centroid, negative_centroid)
    if positive_distance - negative_distance + margin <= 0.0:
        return

    positive_unit = _unit(anchor_centroid - positive_centroid)
    negative_unit = _unit(anchor_centroid - negative_centroid)
    anchor_grad = positive_unit - negative_unit
    positive_grad = -positive_unit
    negative_grad = negative_unit
    _add_group_centroid_gradient(gradient, anchor_indices, anchor_grad)
    _add_group_centroid_gradient(gradient, positive_indices, positive_grad)
    _add_group_centroid_gradient(gradient, negative_indices, negative_grad)


def _add_hard_member_gradient(
    gradient: NDArray[np.float64],
    positive_indices: tuple[int, ...],
    negative_indices: tuple[int, ...],
    anchor_indices: tuple[int, ...],
    positive_group: NDArray[np.float64],
    negative_group: NDArray[np.float64],
    anchor_centroid: NDArray[np.float64],
    config: ProjectionTrainingConfig,
) -> None:
    if config.hard_weight == 0.0:
        return

    positive_distances = np.linalg.norm(positive_group - anchor_centroid, axis=1)
    negative_distances = np.linalg.norm(negative_group - anchor_centroid, axis=1)
    positive_position = int(np.argmax(positive_distances))
    negative_position = int(np.argmin(negative_distances))
    positive_distance = float(positive_distances[positive_position])
    negative_distance = float(negative_distances[negative_position])
    if positive_distance - negative_distance + config.margin <= 0.0:
        return

    positive_unit = _unit(positive_group[positive_position] - anchor_centroid)
    negative_unit = _unit(negative_group[negative_position] - anchor_centroid)
    gradient[positive_indices[positive_position]] += config.hard_weight * positive_unit
    gradient[negative_indices[negative_position]] += -config.hard_weight * negative_unit
    anchor_grad = config.hard_weight * (-positive_unit + negative_unit)
    _add_group_centroid_gradient(gradient, anchor_indices, anchor_grad)


def _add_spread_gradient(
    gradient: NDArray[np.float64],
    indices: tuple[int, ...],
    group: NDArray[np.float64],
    centroid: NDArray[np.float64],
    config: ProjectionTrainingConfig,
) -> None:
    if config.spread_weight == 0.0:
        return

    for position, index in enumerate(indices):
        gradient[index] += config.spread_weight * _unit(group[position] - centroid) / len(indices)


def _add_group_centroid_gradient(
    gradient: NDArray[np.float64],
    indices: tuple[int, ...],
    centroid_gradient: NDArray[np.float64],
) -> None:
    for index in indices:
        gradient[index] += centroid_gradient / len(indices)


def _distance(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(left - right))


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector)
    return vector / norm
