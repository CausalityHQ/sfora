from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sfora.training import ProjectionHeadTrainingConfig, train_projection_head


class TextBaselineConfig(BaseModel):
    """Configuration for frozen text-vector baseline evaluation."""

    group_size: int = Field(default=4, ge=1)
    max_features: int = Field(default=4096, ge=16)
    margin: float = Field(default=0.5, ge=0.0)
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    seed: int = 0
    train_projection_heads: bool = False
    projection_steps: int = Field(default=80, ge=1)
    projection_learning_rate: float = Field(default=0.03, gt=0.0)
    projection_output_dimensions: int | None = Field(default=None, ge=1)


class SentenceTransformerBaselineConfig(BaseModel):
    """Configuration for frozen SentenceTransformers encoder evaluation."""

    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    group_size: int = Field(default=4, ge=1)
    batch_size: int = Field(default=32, ge=1)
    normalize_embeddings: bool = True
    margin: float = Field(default=0.5, ge=0.0)
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    seed: int = 0


class SentenceTransformerModelSuiteConfig(BaseModel):
    """Configuration for comparing multiple frozen SentenceTransformers encoders."""

    model_names: tuple[str, ...] = (
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    group_size: int = Field(default=4, ge=1)
    batch_size: int = Field(default=32, ge=1)
    normalize_embeddings: bool = True
    margin: float = Field(default=0.5, ge=0.0)
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    seed: int = 0


BaselineConfig = (
    TextBaselineConfig | SentenceTransformerBaselineConfig | SentenceTransformerModelSuiteConfig
)


class SentenceEncoder(Protocol):
    """Minimal encoder protocol shared by real and test sentence encoders."""

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Return one embedding per input text."""


EncoderFactory = Callable[[str], SentenceEncoder]


@dataclass(frozen=True)
class TextMethodMetrics:
    """Metrics for one frozen text representation."""

    dimensions: int
    triplet_loss: float
    group_loss: float
    probe: ProbeScore
    retrieval: RetrievalScore
    space: EmbeddingSpaceDiagnostics


@dataclass(frozen=True)
class TextBaselineResult:
    """Serializable output for frozen text representation baselines."""

    name: str
    config: BaselineConfig
    examples: int
    triplets: int
    group_triplets: int
    methods: dict[str, TextMethodMetrics]


def run_text_baseline(
    examples: list[TextExample],
    config: TextBaselineConfig | None = None,
) -> TextBaselineResult:
    """Evaluate frozen text vectors with probe and triplet/group objectives."""
    resolved_config = config or TextBaselineConfig()
    triplets = mine_triplets(examples)
    group_triplets = mine_group_triplets(examples, group_size=resolved_config.group_size)
    labels = np.array([example.label for example in examples], dtype=np.int64)

    tfidf_embeddings = _tfidf_embeddings(examples, max_features=resolved_config.max_features)
    methods = {
        "tfidf_word": _score_text_representation(
            tfidf_embeddings,
            labels,
            examples,
            triplets,
            group_triplets,
            resolved_config,
        )
    }

    if resolved_config.train_projection_heads:
        methods.update(
            _train_tfidf_projection_heads(
                tfidf_embeddings,
                labels,
                examples,
                triplets,
                group_triplets,
                resolved_config,
            )
        )

    return TextBaselineResult(
        name="text-baseline",
        config=resolved_config,
        examples=len(examples),
        triplets=len(triplets),
        group_triplets=len(group_triplets),
        methods=methods,
    )


def run_sentence_transformer_baseline(
    examples: list[TextExample],
    config: SentenceTransformerBaselineConfig | None = None,
    *,
    encoder_factory: EncoderFactory | None = None,
) -> TextBaselineResult:
    """Evaluate a frozen SentenceTransformers encoder on text examples."""
    resolved_config = config or SentenceTransformerBaselineConfig()
    triplets = mine_triplets(examples)
    group_triplets = mine_group_triplets(examples, group_size=resolved_config.group_size)
    labels = np.array([example.label for example in examples], dtype=np.int64)
    embeddings = _sentence_transformer_embeddings(
        examples,
        config=resolved_config,
        encoder_factory=encoder_factory,
    )
    method_name = f"sentence_transformer:{resolved_config.model_name}"

    return TextBaselineResult(
        name="sentence-transformer-baseline",
        config=resolved_config,
        examples=len(examples),
        triplets=len(triplets),
        group_triplets=len(group_triplets),
        methods={
            method_name: _score_text_representation(
                embeddings,
                labels,
                examples,
                triplets,
                group_triplets,
                resolved_config,
            )
        },
    )


def run_sentence_transformer_model_suite(
    examples: list[TextExample],
    config: SentenceTransformerModelSuiteConfig | None = None,
    *,
    encoder_factory: EncoderFactory | None = None,
) -> TextBaselineResult:
    """Evaluate multiple frozen SentenceTransformers encoders on the same sample."""
    resolved_config = config or SentenceTransformerModelSuiteConfig()
    triplets = mine_triplets(examples)
    group_triplets = mine_group_triplets(examples, group_size=resolved_config.group_size)
    labels = np.array([example.label for example in examples], dtype=np.int64)
    methods: dict[str, TextMethodMetrics] = {}

    for model_name in resolved_config.model_names:
        embeddings = _sentence_transformer_embeddings(
            examples,
            config=SentenceTransformerBaselineConfig(
                model_name=model_name,
                group_size=resolved_config.group_size,
                batch_size=resolved_config.batch_size,
                normalize_embeddings=resolved_config.normalize_embeddings,
                margin=resolved_config.margin,
                test_size=resolved_config.test_size,
                seed=resolved_config.seed,
            ),
            encoder_factory=encoder_factory,
        )
        methods[f"sentence_transformer:{model_name}"] = _score_text_representation(
            embeddings,
            labels,
            examples,
            triplets,
            group_triplets,
            resolved_config,
        )

    return TextBaselineResult(
        name="sentence-transformer-model-suite",
        config=resolved_config,
        examples=len(examples),
        triplets=len(triplets),
        group_triplets=len(group_triplets),
        methods=methods,
    )


def write_text_baseline_report(result: TextBaselineResult, output_path: Path) -> Path:
    """Persist a text baseline result as stable, human-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_to_json(result), encoding="utf-8")
    return output_path


def _tfidf_embeddings(
    examples: list[TextExample],
    *,
    max_features: int,
) -> NDArray[np.float64]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=max_features,
        ngram_range=(1, 2),
        norm="l2",
    )
    matrix = vectorizer.fit_transform([example.text for example in examples])
    return np.asarray(matrix.toarray(), dtype=np.float64)


def _sentence_transformer_embeddings(
    examples: list[TextExample],
    *,
    config: SentenceTransformerBaselineConfig,
    encoder_factory: EncoderFactory | None,
) -> NDArray[np.float64]:
    factory = encoder_factory or _load_sentence_transformer
    encoder = factory(config.model_name)
    embeddings = encoder.encode(
        [example.text for example in examples],
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float64)


def _train_tfidf_projection_heads(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    examples: list[TextExample],
    triplets: list[TextTriplet],
    group_triplets: list[TextGroupTriplet],
    config: TextBaselineConfig,
) -> dict[str, TextMethodMetrics]:
    projection_config = ProjectionHeadTrainingConfig(
        group_size=config.group_size,
        steps=config.projection_steps,
        learning_rate=config.projection_learning_rate,
        margin=config.margin,
        output_dimensions=config.projection_output_dimensions,
        seed=config.seed,
    )
    triplet_projection = train_projection_head(
        embeddings,
        labels,
        projection_config.model_copy(update={"objective": "triplet"}),
    )
    group_projection = train_projection_head(
        embeddings,
        labels,
        projection_config.model_copy(update={"objective": "group"}),
    )

    return {
        "tfidf_triplet_projection": _score_text_representation(
            triplet_projection.transformed_embeddings,
            labels,
            examples,
            triplets,
            group_triplets,
            config,
        ),
        "tfidf_group_projection": _score_text_representation(
            group_projection.transformed_embeddings,
            labels,
            examples,
            triplets,
            group_triplets,
            config,
        ),
    }


def _score_text_representation(
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    examples: list[TextExample],
    triplets: list[TextTriplet],
    group_triplets: list[TextGroupTriplet],
    config: BaselineConfig,
) -> TextMethodMetrics:
    row_by_id = {example.example_id: index for index, example in enumerate(examples)}
    anchors, positives, negatives = _triplet_arrays(embeddings, triplets, row_by_id)
    anchor_groups, positive_groups, negative_groups = _group_triplet_arrays(
        embeddings,
        group_triplets,
        row_by_id,
    )
    train_indices, test_indices = _stratified_indices(
        labels,
        test_size=config.test_size,
        seed=config.seed,
    )
    train_embeddings = embeddings[train_indices]
    test_embeddings = embeddings[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    return TextMethodMetrics(
        dimensions=embeddings.shape[1],
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
        probe=linear_probe_score_on_split(
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels,
            random_state=config.seed,
        ),
        retrieval=retrieval_score_on_split(
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels,
        ),
        space=embedding_space_diagnostics_on_split(
            train_embeddings,
            train_labels,
            test_embeddings,
            test_labels,
        ),
    )


def _stratified_indices(
    labels: NDArray[np.int64],
    *,
    test_size: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    indices = np.arange(labels.shape[0], dtype=np.int64)
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def _load_sentence_transformer(model_name: str) -> SentenceEncoder:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install the research extra to load SentenceTransformers: "
            "uv sync --group dev --extra research"
        ) from error

    return SentenceTransformer(model_name)  # type: ignore[no-any-return]


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


def _to_json(result: TextBaselineResult) -> str:
    import json

    return json.dumps(_to_payload(result), indent=2, sort_keys=True) + "\n"


def _to_payload(result: TextBaselineResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "config": result.config.model_dump(),
        "examples": result.examples,
        "triplets": result.triplets,
        "group_triplets": result.group_triplets,
        "methods": {
            name: {
                "dimensions": metrics.dimensions,
                "triplet_loss": metrics.triplet_loss,
                "group_loss": metrics.group_loss,
                "probe": {
                    **asdict(metrics.probe),
                    "confusion_matrix": metrics.probe.confusion_matrix.tolist(),
                },
                "retrieval": asdict(metrics.retrieval),
                "space": asdict(metrics.space),
            }
            for name, metrics in result.methods.items()
        },
    }
