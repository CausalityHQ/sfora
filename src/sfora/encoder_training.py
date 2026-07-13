from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from sfora.data import (
    TextExample,
    TextGroupTriplet,
    TextTriplet,
    mine_group_triplets,
    mine_triplets,
)
from sfora.evaluation import (
    EmbeddingSpaceDiagnostics,
    ProbeScore,
    RetrievalScore,
    embedding_space_diagnostics_on_split,
    linear_probe_score_on_split,
    retrieval_score_on_split,
)
from sfora.losses import group_triplet_margin_loss, triplet_margin_loss

EncoderObjective = Literal[
    "triplet",
    "group",
    "hybrid",
    "hybrid_xbm",
    "hybrid_radius",
    "hybrid_xbm_radius",
    "all",
]


class EncoderTrainingConfig(BaseModel):
    """Configuration for trainable SentenceTransformers-style encoder objectives."""

    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    group_size: int = Field(default=4, ge=1)
    batch_size: int = Field(default=32, ge=1)
    train_steps: int = Field(default=80, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    margin: float = Field(default=0.5, ge=0.0)
    hard_weight: float = Field(default=0.5, ge=0.0)
    spread_weight: float = Field(default=0.1, ge=0.0)
    triplet_weight: float = Field(default=1.0, ge=0.0)
    group_weight: float = Field(default=1.0, ge=0.0)
    xbm_weight: float = Field(default=0.25, ge=0.0)
    xbm_size: int = Field(default=256, ge=1)
    radius_weight: float = Field(default=0.05, ge=0.0)
    variance_weight: float = Field(default=0.05, ge=0.0)
    normalize_embeddings: bool = True
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    retrieval_query_limit: int | None = Field(default=None, ge=1)
    seed: int = 0
    objectives: tuple[EncoderObjective, ...] = (
        "triplet",
        "group",
        "hybrid",
        "hybrid_xbm",
        "hybrid_radius",
        "hybrid_xbm_radius",
    )


class TrainableTextEncoder(Protocol):
    """Minimal protocol for trainable text encoders used by the runner."""

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> NDArray[np.float64]:
        """Return one embedding per input text."""

    def fit(
        self,
        examples: list[TextExample],
        *,
        objective: EncoderObjective,
        config: EncoderTrainingConfig,
    ) -> list[float]:
        """Train the encoder for one objective and return objective loss history."""


EncoderFactory = Callable[[str], TrainableTextEncoder]


@dataclass(frozen=True)
class EncoderTrainingMethodMetrics:
    """Metrics for one fine-tuned encoder representation."""

    dimensions: int
    initial_triplet_loss: float
    triplet_loss: float
    initial_group_loss: float
    group_loss: float
    objective_history: list[float]
    initial_probe: ProbeScore
    probe: ProbeScore
    initial_retrieval: RetrievalScore
    retrieval: RetrievalScore
    initial_space: EmbeddingSpaceDiagnostics
    space: EmbeddingSpaceDiagnostics


@dataclass(frozen=True)
class EncoderTrainingResult:
    """Serializable output for trainable encoder comparisons."""

    name: str
    config: EncoderTrainingConfig
    examples: int
    train_examples: int
    test_examples: int
    triplets: int
    group_triplets: int
    methods: dict[str, EncoderTrainingMethodMetrics]


def run_encoder_training(
    examples: list[TextExample],
    config: EncoderTrainingConfig | None = None,
    *,
    encoder_factory: EncoderFactory | None = None,
) -> EncoderTrainingResult:
    """Fine-tune text encoders with standard triplet and sfora objectives."""
    resolved_config = config or EncoderTrainingConfig()
    train_examples, test_examples = _split_examples(examples, resolved_config)
    return run_encoder_training_on_split(
        train_examples,
        test_examples,
        resolved_config,
        encoder_factory=encoder_factory,
    )


def run_encoder_training_on_split(
    train_examples: list[TextExample],
    test_examples: list[TextExample],
    config: EncoderTrainingConfig | None = None,
    *,
    encoder_factory: EncoderFactory | None = None,
) -> EncoderTrainingResult:
    """Fine-tune text encoders on explicit train examples and evaluate on test examples."""
    resolved_config = config or EncoderTrainingConfig()
    triplets = mine_triplets(train_examples)
    group_triplets = mine_group_triplets(train_examples, group_size=resolved_config.group_size)
    train_labels = np.array([example.label for example in train_examples], dtype=np.int64)
    test_labels = np.array([example.label for example in test_examples], dtype=np.int64)
    factory = encoder_factory or _load_trainable_sentence_transformer
    methods: dict[str, EncoderTrainingMethodMetrics] = {}

    for objective in resolved_config.objectives:
        # Re-seed before each encoder so init/dropout are reproducible and objective
        # comparisons do not depend on run order.
        _seed_everything(resolved_config.seed)
        encoder = factory(resolved_config.model_name)
        initial_embeddings = encoder.encode(
            [example.text for example in train_examples],
            batch_size=resolved_config.batch_size,
            normalize_embeddings=resolved_config.normalize_embeddings,
        )
        initial_test_embeddings = encoder.encode(
            [example.text for example in test_examples],
            batch_size=resolved_config.batch_size,
            normalize_embeddings=resolved_config.normalize_embeddings,
        )
        initial_triplet_loss, initial_group_loss = _losses_for_embeddings(
            initial_embeddings,
            train_examples,
            triplets,
            group_triplets,
            resolved_config,
        )
        initial_probe = linear_probe_score_on_split(
            initial_embeddings,
            train_labels,
            initial_test_embeddings,
            test_labels,
            random_state=resolved_config.seed,
        )
        initial_retrieval = retrieval_score_on_split(
            initial_embeddings,
            train_labels,
            initial_test_embeddings,
            test_labels,
            query_limit=resolved_config.retrieval_query_limit,
            random_state=resolved_config.seed,
        )
        initial_space = embedding_space_diagnostics_on_split(
            initial_embeddings,
            train_labels,
            initial_test_embeddings,
            test_labels,
        )
        history = encoder.fit(train_examples, objective=objective, config=resolved_config)
        final_embeddings = encoder.encode(
            [example.text for example in train_examples],
            batch_size=resolved_config.batch_size,
            normalize_embeddings=resolved_config.normalize_embeddings,
        )
        final_test_embeddings = encoder.encode(
            [example.text for example in test_examples],
            batch_size=resolved_config.batch_size,
            normalize_embeddings=resolved_config.normalize_embeddings,
        )
        triplet_loss, group_loss = _losses_for_embeddings(
            final_embeddings,
            train_examples,
            triplets,
            group_triplets,
            resolved_config,
        )
        methods[f"{objective}_finetuned:{resolved_config.model_name}"] = (
            EncoderTrainingMethodMetrics(
                dimensions=final_embeddings.shape[1],
                initial_triplet_loss=initial_triplet_loss,
                triplet_loss=triplet_loss,
                initial_group_loss=initial_group_loss,
                group_loss=group_loss,
                objective_history=history,
                initial_probe=initial_probe,
                probe=linear_probe_score_on_split(
                    final_embeddings,
                    train_labels,
                    final_test_embeddings,
                    test_labels,
                    random_state=resolved_config.seed,
                ),
                initial_retrieval=initial_retrieval,
                retrieval=retrieval_score_on_split(
                    final_embeddings,
                    train_labels,
                    final_test_embeddings,
                    test_labels,
                    query_limit=resolved_config.retrieval_query_limit,
                    random_state=resolved_config.seed,
                ),
                initial_space=initial_space,
                space=embedding_space_diagnostics_on_split(
                    final_embeddings,
                    train_labels,
                    final_test_embeddings,
                    test_labels,
                ),
            )
        )

    return EncoderTrainingResult(
        name="sentence-transformer-training",
        config=resolved_config,
        examples=len(train_examples) + len(test_examples),
        train_examples=len(train_examples),
        test_examples=len(test_examples),
        triplets=len(triplets),
        group_triplets=len(group_triplets),
        methods=methods,
    )


def write_encoder_training_report(result: EncoderTrainingResult, output_path: Path) -> Path:
    """Persist an encoder training result as stable, human-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_to_json(result), encoding="utf-8")
    return output_path


def _losses_for_embeddings(
    embeddings: NDArray[np.float64],
    examples: list[TextExample],
    triplets: list[TextTriplet],
    group_triplets: list[TextGroupTriplet],
    config: EncoderTrainingConfig,
) -> tuple[float, float]:
    row_by_id = {example.example_id: index for index, example in enumerate(examples)}
    anchors, positives, negatives = _triplet_arrays(embeddings, triplets, row_by_id)
    anchor_groups, positive_groups, negative_groups = _group_triplet_arrays(
        embeddings,
        group_triplets,
        row_by_id,
    )
    return (
        triplet_margin_loss(anchors, positives, negatives, margin=config.margin),
        group_triplet_margin_loss(
            anchor_groups,
            positive_groups,
            negative_groups,
            margin=config.margin,
            hard_weight=config.hard_weight,
            spread_weight=config.spread_weight,
        ),
    )


def _split_examples(
    examples: list[TextExample],
    config: EncoderTrainingConfig,
) -> tuple[list[TextExample], list[TextExample]]:
    labels = np.array([example.label for example in examples], dtype=np.int64)
    indices = np.arange(len(examples))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    train_examples = [examples[int(index)] for index in sorted(train_indices)]
    test_examples = [examples[int(index)] for index in sorted(test_indices)]
    return train_examples, test_examples


def _triplet_arrays(
    embeddings: NDArray[np.float64],
    triplets: list[TextTriplet],
    row_by_id: dict[str, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.stack([embeddings[row_by_id[triplet.anchor.example_id]] for triplet in triplets]),
        np.stack([embeddings[row_by_id[triplet.positive.example_id]] for triplet in triplets]),
        np.stack([embeddings[row_by_id[triplet.negative.example_id]] for triplet in triplets]),
    )


def _group_triplet_arrays(
    embeddings: NDArray[np.float64],
    group_triplets: list[TextGroupTriplet],
    row_by_id: dict[str, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.stack(
            [_group_array(embeddings, triplet.anchor, row_by_id) for triplet in group_triplets]
        ),
        np.stack(
            [_group_array(embeddings, triplet.positive, row_by_id) for triplet in group_triplets]
        ),
        np.stack(
            [_group_array(embeddings, triplet.negative, row_by_id) for triplet in group_triplets]
        ),
    )


def _group_array(
    embeddings: NDArray[np.float64],
    group: tuple[TextExample, ...],
    row_by_id: dict[str, int],
) -> NDArray[np.float64]:
    return np.stack([embeddings[row_by_id[example.example_id]] for example in group])


def _seed_everything(seed: int) -> None:
    """Seed Python/NumPy/Torch/Transformers so encoder init + dropout are reproducible."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def _load_trainable_sentence_transformer(model_name: str) -> TrainableTextEncoder:
    return _SentenceTransformerFineTuner(model_name)


class _SentenceTransformerFineTuner:
    """Runtime adapter for fine-tuning real SentenceTransformers models."""

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install the research extra to fine-tune encoders: "
                "uv sync --group dev --extra research"
            ) from error

        self._torch: Any = torch
        self._model: Any = SentenceTransformer(model_name)

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> NDArray[np.float64]:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float64)

    def fit(
        self,
        examples: list[TextExample],
        *,
        objective: EncoderObjective,
        config: EncoderTrainingConfig,
    ) -> list[float]:
        self._model.train()
        optimizer = self._torch.optim.AdamW(self._model.parameters(), lr=config.learning_rate)
        history: list[float] = []
        triplets = mine_triplets(examples)
        group_triplets = mine_group_triplets(examples, group_size=config.group_size)
        memory_embeddings: Any | None = None
        memory_labels: Any | None = None

        for step in range(config.train_steps):
            optimizer.zero_grad()
            if objective == "triplet":
                loss = self._triplet_batch_loss(triplets, step, config)
            elif objective == "group":
                loss = self._group_batch_loss(group_triplets, step, config)
            else:
                loss = self._hybrid_batch_loss(triplets, group_triplets, step, config)
                if objective in {"hybrid_xbm", "hybrid_xbm_radius", "all"}:
                    xbm_loss, memory_embeddings, memory_labels = self._xbm_batch_loss(
                        examples,
                        step,
                        config,
                        memory_embeddings,
                        memory_labels,
                    )
                    loss = loss + (config.xbm_weight * xbm_loss)
                if objective in {"hybrid_radius", "hybrid_xbm_radius", "all"}:
                    loss = loss + self._radius_variance_batch_loss(examples, step, config)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu().item()))

        return history

    def _triplet_batch_loss(
        self,
        triplets: list[TextTriplet],
        step: int,
        config: EncoderTrainingConfig,
    ) -> Any:
        batch_count = max(1, config.batch_size // 3)
        batch = _cyclic_slice(triplets, start=step * batch_count, count=batch_count)
        texts = [
            example.text
            for triplet in batch
            for example in (triplet.anchor, triplet.positive, triplet.negative)
        ]
        embeddings = self._forward_texts(texts, normalize_embeddings=config.normalize_embeddings)
        embeddings = embeddings.reshape(len(batch), 3, embeddings.shape[-1])
        anchors = embeddings[:, 0, :]
        positives = embeddings[:, 1, :]
        negatives = embeddings[:, 2, :]
        positive_distances = self._torch.linalg.vector_norm(anchors - positives, dim=-1)
        negative_distances = self._torch.linalg.vector_norm(anchors - negatives, dim=-1)
        return self._torch.relu(positive_distances - negative_distances + config.margin).mean()

    def _hybrid_batch_loss(
        self,
        triplets: list[TextTriplet],
        group_triplets: list[TextGroupTriplet],
        step: int,
        config: EncoderTrainingConfig,
    ) -> Any:
        return (config.triplet_weight * self._triplet_batch_loss(triplets, step, config)) + (
            config.group_weight * self._group_batch_loss(group_triplets, step, config)
        )

    def _group_batch_loss(
        self,
        group_triplets: list[TextGroupTriplet],
        step: int,
        config: EncoderTrainingConfig,
    ) -> Any:
        examples_per_group_triplet = 3 * config.group_size
        batch_count = max(1, config.batch_size // examples_per_group_triplet)
        batch = _cyclic_slice(group_triplets, start=step * batch_count, count=batch_count)
        texts = [
            example.text
            for triplet in batch
            for group in (triplet.anchor, triplet.positive, triplet.negative)
            for example in group
        ]
        embeddings = self._forward_texts(texts, normalize_embeddings=config.normalize_embeddings)
        embeddings = embeddings.reshape(len(batch), 3, config.group_size, embeddings.shape[-1])
        anchor_groups = embeddings[:, 0, :, :]
        positive_groups = embeddings[:, 1, :, :]
        negative_groups = embeddings[:, 2, :, :]
        anchor_centroids = anchor_groups.mean(dim=1)
        positive_centroids = positive_groups.mean(dim=1)
        negative_centroids = negative_groups.mean(dim=1)

        centroid_losses = self._margin_violation(
            self._dist(anchor_centroids, positive_centroids),
            self._dist(anchor_centroids, negative_centroids),
            config.margin,
        )
        hard_losses = self._margin_violation(
            self._dist(positive_groups, anchor_centroids[:, None, :]).max(dim=1).values,
            self._dist(negative_groups, anchor_centroids[:, None, :]).min(dim=1).values,
            config.margin,
        )
        spread_penalties = (
            self._dist(anchor_groups, anchor_centroids[:, None, :]).mean(dim=1)
            + self._dist(positive_groups, positive_centroids[:, None, :]).mean(dim=1)
            + self._dist(negative_groups, negative_centroids[:, None, :]).mean(dim=1)
        )
        return (
            centroid_losses
            + (config.hard_weight * hard_losses)
            + (config.spread_weight * spread_penalties)
        ).mean()

    def _xbm_batch_loss(
        self,
        examples: list[TextExample],
        step: int,
        config: EncoderTrainingConfig,
        memory_embeddings: Any | None,
        memory_labels: Any | None,
    ) -> tuple[Any, Any, Any]:
        batch = _cyclic_slice(examples, start=step * config.batch_size, count=config.batch_size)
        labels = self._torch.tensor(
            [example.label for example in batch],
            device=self._model.device,
            dtype=self._torch.long,
        )
        embeddings = self._forward_texts(
            [example.text for example in batch],
            normalize_embeddings=config.normalize_embeddings,
        )
        if memory_embeddings is None or memory_labels is None:
            loss = embeddings.new_zeros(())
        else:
            distances = self._torch.cdist(embeddings, memory_embeddings)
            same_label = labels[:, None] == memory_labels[None, :]
            different_label = ~same_label
            has_positive = same_label.any(dim=1)
            has_negative = different_label.any(dim=1)
            valid = has_positive & has_negative
            if bool(valid.any()):
                # Hardest-positive / hardest-negative mining (XBM's intent). Using the
                # MAX same-label distance also sidesteps the cyclic-memory self-match
                # collapse: an example's own repeated copy is the *smallest* same-label
                # distance, so taking the max ignores it (the previous min picked it,
                # driving the positive term to ~0 and killing the signal).
                positive_distances = (
                    distances.masked_fill(~same_label, float("-inf")).max(dim=1).values
                )
                negative_distances = (
                    distances.masked_fill(~different_label, float("inf")).min(dim=1).values
                )
                loss = self._torch.relu(
                    positive_distances[valid] - negative_distances[valid] + config.margin
                ).mean()
            else:
                loss = embeddings.new_zeros(())

        next_embeddings = embeddings.detach()
        next_labels = labels.detach()
        if memory_embeddings is not None and memory_labels is not None:
            next_embeddings = self._torch.cat([memory_embeddings, next_embeddings], dim=0)
            next_labels = self._torch.cat([memory_labels, next_labels], dim=0)
        if next_embeddings.shape[0] > config.xbm_size:
            next_embeddings = next_embeddings[-config.xbm_size :]
            next_labels = next_labels[-config.xbm_size :]
        return loss, next_embeddings, next_labels

    def _radius_variance_batch_loss(
        self,
        examples: list[TextExample],
        step: int,
        config: EncoderTrainingConfig,
    ) -> Any:
        batch = _cyclic_slice(examples, start=step * config.batch_size, count=config.batch_size)
        labels = self._torch.tensor(
            [example.label for example in batch],
            device=self._model.device,
            dtype=self._torch.long,
        )
        embeddings = self._forward_texts(
            [example.text for example in batch],
            normalize_embeddings=config.normalize_embeddings,
        )
        penalties: list[Any] = []
        centroids: list[Any] = []
        for label in self._torch.unique(labels):
            members = embeddings[labels == label]
            if members.shape[0] < 2:
                continue
            centroid = members.mean(dim=0)
            centroids.append(centroid)
            distances = self._dist(members, centroid[None, :])
            penalties.append(config.radius_weight * distances.mean())
            penalties.append(config.variance_weight * distances.var(unbiased=False))
        if len(centroids) >= 2:
            stacked_centroids = self._torch.stack(centroids)
            centroid_distances = self._torch.pdist(stacked_centroids)
            penalties.append(self._torch.relu(config.margin - centroid_distances).mean())
        if not penalties:
            return embeddings.new_zeros(())
        return self._torch.stack(penalties).sum()

    def _forward_texts(self, texts: list[str], *, normalize_embeddings: bool) -> Any:
        features = self._model.tokenize(texts)
        features = {
            name: value.to(self._model.device) if hasattr(value, "to") else value
            for name, value in features.items()
        }
        embeddings = self._model(features)["sentence_embedding"]
        if normalize_embeddings:
            embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def _dist(self, left: Any, right: Any) -> Any:
        return self._torch.linalg.vector_norm(left - right, dim=-1)

    def _margin_violation(
        self, positive_distances: Any, negative_distances: Any, margin: float
    ) -> Any:
        return self._torch.relu(positive_distances - negative_distances + margin)


def _cyclic_slice[T](items: list[T], *, start: int, count: int) -> list[T]:
    return [items[(start + offset) % len(items)] for offset in range(count)]


def _to_json(result: EncoderTrainingResult) -> str:
    import json

    return json.dumps(_to_payload(result), indent=2, sort_keys=True) + "\n"


def _to_payload(result: EncoderTrainingResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "config": result.config.model_dump(),
        "examples": result.examples,
        "train_examples": result.train_examples,
        "test_examples": result.test_examples,
        "triplets": result.triplets,
        "group_triplets": result.group_triplets,
        "methods": {
            name: {
                "dimensions": metrics.dimensions,
                "initial_triplet_loss": metrics.initial_triplet_loss,
                "triplet_loss": metrics.triplet_loss,
                "initial_group_loss": metrics.initial_group_loss,
                "group_loss": metrics.group_loss,
                "objective_history": metrics.objective_history,
                "initial_probe": {
                    **asdict(metrics.initial_probe),
                    "confusion_matrix": metrics.initial_probe.confusion_matrix.tolist(),
                },
                "probe": {
                    **asdict(metrics.probe),
                    "confusion_matrix": metrics.probe.confusion_matrix.tolist(),
                },
                "initial_retrieval": asdict(metrics.initial_retrieval),
                "retrieval": asdict(metrics.retrieval),
                "initial_space": asdict(metrics.initial_space),
                "space": asdict(metrics.space),
            }
            for name, metrics in result.methods.items()
        },
    }
