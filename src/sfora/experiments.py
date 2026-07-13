from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from sfora.evaluation import ProbeScore, linear_probe_score
from sfora.losses import group_triplet_margin_loss, triplet_margin_loss
from sfora.training import ProjectionTrainingConfig, train_embedding_table


class SyntheticExperimentConfig(BaseModel):
    """Configuration for the deterministic local smoke experiment."""

    samples_per_class: int = Field(default=24, ge=4)
    dimensions: int = Field(default=8, ge=2)
    group_size: int = Field(default=4, ge=1)
    seed: int = 0
    class_gap: float = Field(default=1.2, gt=0.0)
    noise: float = Field(default=1.1, gt=0.0)
    margin: float = Field(default=0.5, ge=0.0)
    test_size: float = Field(default=0.3, gt=0.0, lt=1.0)


class TrainableSyntheticExperimentConfig(SyntheticExperimentConfig):
    """Configuration for trainable synthetic triplet/group comparison."""

    train_steps: int = Field(default=80, ge=1)
    learning_rate: float = Field(default=0.03, gt=0.0)
    hard_weight: float = Field(default=0.5, ge=0.0)
    spread_weight: float = Field(default=0.1, ge=0.0)


@dataclass(frozen=True)
class MethodMetrics:
    """Comparable metrics for one representation space."""

    triplet_loss: float
    group_loss: float
    probe: ProbeScore


@dataclass(frozen=True)
class ExperimentResult:
    """Serializable output from a representation comparison experiment."""

    name: str
    config: SyntheticExperimentConfig
    methods: dict[str, MethodMetrics]


def run_synthetic_experiment(config: SyntheticExperimentConfig | None = None) -> ExperimentResult:
    """Run a deterministic smoke comparison over synthetic vector spaces."""
    resolved_config = config or SyntheticExperimentConfig()
    embeddings, labels = _make_synthetic_embeddings(resolved_config)
    _validate_groupable(labels, resolved_config.group_size)

    representations = {
        "raw": embeddings,
        "triplet_baseline": _class_compacted_embeddings(embeddings, labels, strength=0.45),
        "sfora": _class_compacted_embeddings(embeddings, labels, strength=0.75),
    }

    return ExperimentResult(
        name="synthetic-smoke",
        config=resolved_config,
        methods={
            name: _score_representation(representation, labels, resolved_config)
            for name, representation in representations.items()
        },
    )


def run_trainable_synthetic_experiment(
    config: TrainableSyntheticExperimentConfig | None = None,
) -> ExperimentResult:
    """Run a trainable synthetic comparison for triplet and group objectives."""
    resolved_config = config or TrainableSyntheticExperimentConfig()
    embeddings, labels = _make_synthetic_embeddings(resolved_config)
    _validate_groupable(labels, resolved_config.group_size)

    triplet_training = train_embedding_table(
        embeddings,
        labels,
        ProjectionTrainingConfig(
            objective="triplet",
            group_size=resolved_config.group_size,
            steps=resolved_config.train_steps,
            learning_rate=resolved_config.learning_rate,
            margin=resolved_config.margin,
            hard_weight=resolved_config.hard_weight,
            spread_weight=resolved_config.spread_weight,
        ),
    )
    group_training = train_embedding_table(
        embeddings,
        labels,
        ProjectionTrainingConfig(
            objective="group",
            group_size=resolved_config.group_size,
            steps=resolved_config.train_steps,
            learning_rate=resolved_config.learning_rate,
            margin=resolved_config.margin,
            hard_weight=resolved_config.hard_weight,
            spread_weight=resolved_config.spread_weight,
        ),
    )

    return ExperimentResult(
        name="synthetic-trainable",
        config=resolved_config,
        methods={
            "raw": _score_representation(embeddings, labels, resolved_config),
            "triplet_trained": _score_representation(
                triplet_training.transformed_embeddings,
                labels,
                resolved_config,
            ),
            "group_trained": _score_representation(
                group_training.transformed_embeddings,
                labels,
                resolved_config,
            ),
        },
    )


def write_experiment_report(result: ExperimentResult, output_path: Path) -> Path:
    """Persist an experiment result as stable, human-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_to_json(result), encoding="utf-8")
    return output_path


def _make_synthetic_embeddings(
    config: SyntheticExperimentConfig,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(config.seed)
    centers = np.zeros((2, config.dimensions), dtype=np.float64)
    centers[0, 0] = -config.class_gap / 2.0
    centers[1, 0] = config.class_gap / 2.0

    labels = np.repeat(np.array([0, 1], dtype=np.int64), config.samples_per_class)
    embeddings = centers[labels] + rng.normal(
        loc=0.0,
        scale=config.noise,
        size=(labels.shape[0], config.dimensions),
    )
    return embeddings, labels


def _validate_groupable(labels: NDArray[np.int64], group_size: int) -> None:
    for label in np.unique(labels):
        if np.count_nonzero(labels == label) % group_size != 0:
            raise ValueError("samples_per_class must divide evenly by group_size")


def _class_compacted_embeddings(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    strength: float,
) -> NDArray[np.float64]:
    compacted = embeddings.copy()
    global_center = embeddings.mean(axis=0)
    for label in np.unique(labels):
        label_mask = labels == label
        class_center = embeddings[label_mask].mean(axis=0)
        centered = embeddings[label_mask] - class_center
        separation = class_center + strength * (class_center - global_center)
        compacted[label_mask] = separation + (1.0 - strength) * centered
    return compacted


def _score_representation(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    config: SyntheticExperimentConfig,
) -> MethodMetrics:
    anchors, positives, negatives = _make_triplets(embeddings, labels)
    anchor_groups, positive_groups, negative_groups = _make_group_triplets(
        embeddings,
        labels,
        group_size=config.group_size,
    )

    return MethodMetrics(
        triplet_loss=triplet_margin_loss(
            anchors,
            positives,
            negatives,
            margin=config.margin,
        ),
        group_loss=group_triplet_margin_loss(
            anchor_groups,
            positive_groups,
            negative_groups,
            margin=config.margin,
        ),
        probe=linear_probe_score(
            embeddings,
            labels,
            test_size=config.test_size,
            random_state=config.seed,
        ),
    )


def _make_triplets(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    anchors: list[NDArray[np.float64]] = []
    positives: list[NDArray[np.float64]] = []
    negatives: list[NDArray[np.float64]] = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        same_indices = np.flatnonzero(labels == label)
        other_indices = np.flatnonzero(labels != label)
        for position, anchor_index in enumerate(same_indices):
            positive_index = same_indices[(position + 1) % same_indices.shape[0]]
            negative_index = other_indices[position % other_indices.shape[0]]
            anchors.append(embeddings[anchor_index])
            positives.append(embeddings[positive_index])
            negatives.append(embeddings[negative_index])

    return np.vstack(anchors), np.vstack(positives), np.vstack(negatives)


def _make_group_triplets(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    *,
    group_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    anchor_groups: list[NDArray[np.float64]] = []
    positive_groups: list[NDArray[np.float64]] = []
    negative_groups: list[NDArray[np.float64]] = []
    unique_labels = np.unique(labels)

    grouped_by_label = {
        int(label): embeddings[labels == label].reshape(-1, group_size, embeddings.shape[1])
        for label in unique_labels
    }

    for label in unique_labels:
        same_groups = grouped_by_label[int(label)]
        negative_groups_for_label = grouped_by_label[int(unique_labels[unique_labels != label][0])]
        for position, anchor_group in enumerate(same_groups):
            anchor_groups.append(anchor_group)
            positive_groups.append(same_groups[(position + 1) % same_groups.shape[0]])
            negative_groups.append(
                negative_groups_for_label[position % negative_groups_for_label.shape[0]]
            )

    return (
        np.stack(anchor_groups),
        np.stack(positive_groups),
        np.stack(negative_groups),
    )


def _to_json(result: ExperimentResult) -> str:
    import json

    return json.dumps(_to_payload(result), indent=2, sort_keys=True) + "\n"


def _to_payload(result: ExperimentResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "config": result.config.model_dump(),
        "methods": {
            name: {
                "triplet_loss": metrics.triplet_loss,
                "group_loss": metrics.group_loss,
                "probe": {
                    **asdict(metrics.probe),
                    "confusion_matrix": metrics.probe.confusion_matrix.tolist(),
                },
            }
            for name, metrics in result.methods.items()
        },
    }
