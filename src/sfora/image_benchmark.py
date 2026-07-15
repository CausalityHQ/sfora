from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from sfora.data import ImageDatasetName, ImageExample
from sfora.evaluation import (
    EmbeddingSpaceDiagnostics,
    embedding_space_diagnostics_on_split,
)
from sfora.training import Objective, ProjectionHeadTrainingConfig, train_projection_head

ImageObjective = Literal[
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


class ImageBenchmarkConfig(BaseModel):
    """Configuration for frozen-image-backbone retrieval benchmarks."""

    dataset_name: ImageDatasetName = "cub"
    model_names: tuple[str, ...] = (
        "facebook/dinov2-small",
        "openai/clip-vit-base-patch32",
        "google/siglip-base-patch16-224",
    )
    objectives: tuple[ImageObjective, ...] = (
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
    )
    group_size: int = Field(default=4, ge=1)
    batch_size: int = Field(default=64, ge=1)
    train_steps: int = Field(default=80, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
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
    output_dimensions: int | None = Field(default=None, ge=1)
    normalize_embeddings: bool = True
    shuffle_groups_each_step: bool = False
    embedding_cache_dir: Path | None = None
    limit_per_class: int | None = Field(default=None, ge=1)
    min_per_class: int | None = Field(default=None, ge=1)
    max_classes: int | None = Field(default=None, ge=2)
    projection_train_limit: int | None = Field(default=None, ge=2)
    projection_validation_fraction: float = Field(default=0.2, ge=0.0, lt=1.0)
    projection_validation_limit: int | None = Field(default=4096, ge=2)
    validation_query_limit: int | None = Field(default=1024, ge=1)
    retrieval_query_limit: int | None = Field(default=None, ge=1)
    seed: int = 0


class FrozenImageEncoder(Protocol):
    """Minimal image encoder interface used by the image retrieval runner."""

    def encode(
        self,
        images: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> NDArray[np.float64]:
        """Return one embedding per image."""


ImageEncoderFactory = Callable[[str], FrozenImageEncoder]


@dataclass(frozen=True)
class ImageRetrievalMetrics:
    """Self-retrieval metrics on a held-out image split."""

    precision_at_1: float
    recall_at_1: float
    recall_at_2: float
    recall_at_4: float
    recall_at_8: float
    map_at_r: float
    mean_relevant_items: float
    evaluated_queries: int
    total_queries: int


@dataclass(frozen=True)
class ImageBenchmarkMethodMetrics:
    """Metrics for one model/objective row in an image retrieval benchmark."""

    model_name: str
    objective: str
    display_name: str
    dimensions: int
    retrieval: ImageRetrievalMetrics
    recall_at_1: float
    recall_at_2: float
    recall_at_4: float
    recall_at_8: float
    map_at_r: float
    recall_at_1_delta: float
    map_at_r_delta: float
    initial_triplet_loss: float | None
    triplet_loss: float | None
    initial_group_loss: float | None
    group_loss: float | None
    objective_history: list[float]
    selected_step: int | None
    selection_score: float | None
    space: EmbeddingSpaceDiagnostics | None


@dataclass(frozen=True)
class ImageBenchmarkResult:
    """Serializable output for image metric-learning benchmark comparisons."""

    name: str
    dataset_name: ImageDatasetName
    config: ImageBenchmarkConfig
    examples: int
    train_examples: int
    projection_train_examples: int
    projection_validation_examples: int
    test_examples: int
    best_method: str | None
    methods: dict[str, ImageBenchmarkMethodMetrics]


def run_image_benchmark(
    *,
    train_examples: list[ImageExample],
    test_examples: list[ImageExample],
    config: ImageBenchmarkConfig | None = None,
    encoder_factory: ImageEncoderFactory | None = None,
) -> ImageBenchmarkResult:
    """Compare frozen image backbones and projection-head objectives on retrieval."""
    resolved_config = config or ImageBenchmarkConfig()
    factory = encoder_factory or _load_transformers_image_encoder
    train_labels = np.asarray([example.label for example in train_examples], dtype=np.int64)
    test_labels = np.asarray([example.label for example in test_examples], dtype=np.int64)
    projection_min_per_class = _projection_min_per_class(resolved_config)
    projection_train_indices, projection_validation_indices = _projection_train_validation_indices(
        train_labels,
        train_limit=resolved_config.projection_train_limit,
        validation_fraction=resolved_config.projection_validation_fraction,
        validation_limit=resolved_config.projection_validation_limit,
        train_min_per_class=projection_min_per_class,
        random_state=resolved_config.seed,
    )
    methods: dict[str, ImageBenchmarkMethodMetrics] = {}

    for model_name in resolved_config.model_names:
        encoder: FrozenImageEncoder | None = None
        train_embeddings, encoder = _encode_examples_with_cache(
            model_name=model_name,
            split_name="train",
            examples=train_examples,
            config=resolved_config,
            encoder=encoder,
            encoder_factory=factory,
        )
        test_embeddings, encoder = _encode_examples_with_cache(
            model_name=model_name,
            split_name="test",
            examples=test_examples,
            config=resolved_config,
            encoder=encoder,
            encoder_factory=factory,
        )
        del encoder
        projection_train_embeddings = train_embeddings[projection_train_indices]
        projection_train_labels = train_labels[projection_train_indices]
        projection_validation_embeddings = train_embeddings[projection_validation_indices]
        projection_validation_labels = train_labels[projection_validation_indices]
        frozen_retrieval = image_self_retrieval_score(
            test_embeddings,
            test_labels,
            query_limit=resolved_config.retrieval_query_limit,
            random_state=resolved_config.seed,
        )
        frozen_key = f"frozen:{model_name}"
        methods[frozen_key] = _method_metrics(
            model_name=model_name,
            objective="frozen",
            display_name="Frozen",
            embeddings=test_embeddings,
            labels=test_labels,
            retrieval=frozen_retrieval,
            baseline=frozen_retrieval,
            objective_history=[],
            initial_triplet_loss=None,
            triplet_loss=None,
            initial_group_loss=None,
            group_loss=None,
        )

        for objective in resolved_config.objectives:
            training = train_projection_head(
                projection_train_embeddings,
                projection_train_labels,
                ProjectionHeadTrainingConfig(
                    objective=_projection_objective(objective),
                    group_size=resolved_config.group_size,
                    steps=resolved_config.train_steps,
                    learning_rate=resolved_config.learning_rate,
                    margin=resolved_config.margin,
                    hard_weight=resolved_config.hard_weight,
                    spread_weight=resolved_config.spread_weight,
                    triplet_weight=resolved_config.triplet_weight,
                    group_weight=resolved_config.group_weight,
                    xbm_weight=resolved_config.xbm_weight,
                    xbm_memory_size=resolved_config.xbm_memory_size,
                    radius_weight=resolved_config.radius_weight,
                    radius_target=resolved_config.radius_target,
                    variance_weight=resolved_config.variance_weight,
                    output_dimensions=resolved_config.output_dimensions,
                    normalize_projected_embeddings=resolved_config.normalize_embeddings,
                    shuffle_groups_each_step=resolved_config.shuffle_groups_each_step,
                    selection_score_callback=_projection_selection_callback(
                        validation_embeddings=projection_validation_embeddings,
                        validation_labels=projection_validation_labels,
                        normalize_embeddings=resolved_config.normalize_embeddings,
                        query_limit=(
                            resolved_config.validation_query_limit
                            if resolved_config.validation_query_limit is not None
                            else resolved_config.retrieval_query_limit
                        ),
                        random_state=resolved_config.seed,
                    ),
                    seed=resolved_config.seed,
                ),
            )
            projected_test = test_embeddings @ training.projection_matrix
            if resolved_config.normalize_embeddings:
                projected_test = _normalize(projected_test)
            retrieval = image_self_retrieval_score(
                projected_test,
                test_labels,
                query_limit=resolved_config.retrieval_query_limit,
                random_state=resolved_config.seed,
            )
            method_key = f"{objective}_projection:{model_name}"
            methods[method_key] = _method_metrics(
                model_name=model_name,
                objective=objective,
                display_name=objective_display_name(objective),
                embeddings=projected_test,
                labels=test_labels,
                retrieval=retrieval,
                baseline=frozen_retrieval,
                objective_history=training.history,
                initial_triplet_loss=training.initial_triplet_loss,
                triplet_loss=training.final_triplet_loss,
                initial_group_loss=training.initial_group_loss,
                group_loss=training.final_group_loss,
                selected_step=training.selected_step,
                selection_score=training.selection_score,
            )

    best_method = _best_method_name(methods)
    return ImageBenchmarkResult(
        name="image-retrieval-benchmark",
        dataset_name=resolved_config.dataset_name,
        config=resolved_config,
        examples=len(train_examples) + len(test_examples),
        train_examples=len(train_examples),
        projection_train_examples=int(projection_train_indices.shape[0]),
        projection_validation_examples=int(projection_validation_indices.shape[0]),
        test_examples=len(test_examples),
        best_method=best_method,
        methods=methods,
    )


def image_self_retrieval_score(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    query_limit: int | None = None,
    random_state: int = 0,
) -> ImageRetrievalMetrics:
    """Evaluate retrieval within a held-out image split, excluding the query itself."""
    embedding_array = np.asarray(embeddings, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    if embedding_array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if label_array.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if embedding_array.shape[0] != label_array.shape[0]:
        raise ValueError("embeddings and labels must contain the same number of examples")
    if embedding_array.shape[0] < 2:
        raise ValueError("self retrieval requires at least two examples")
    if query_limit is not None and query_limit < 1:
        raise ValueError("query_limit must be at least 1")

    query_indices = np.arange(embedding_array.shape[0], dtype=np.int64)
    if query_limit is not None and query_limit < query_indices.shape[0]:
        query_indices = _stratified_query_indices(
            label_array,
            query_limit=query_limit,
            random_state=random_state,
        )

    precision_at_1_values: list[float] = []
    recall_at_k_values: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}
    average_precisions: list[float] = []
    relevant_counts: list[int] = []
    label_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(label_array, return_counts=True), strict=True)
    }
    query_relevant_counts = np.asarray(
        [label_counts[int(label_array[index])] - 1 for index in query_indices],
        dtype=np.int64,
    )
    max_relevant_count = int(query_relevant_counts.max(initial=0))
    top_k = min(embedding_array.shape[0] - 1, max(8, max_relevant_count))
    embedding_norms = np.sum(embedding_array * embedding_array, axis=1)
    chunk_size = 1024
    for start in range(0, query_indices.shape[0], chunk_size):
        chunk_indices = query_indices[start : start + chunk_size]
        query_embeddings = embedding_array[chunk_indices]
        distances = (
            np.sum(query_embeddings * query_embeddings, axis=1, keepdims=True)
            + embedding_norms[np.newaxis, :]
            - (2.0 * query_embeddings @ embedding_array.T)
        )
        distances = np.maximum(distances, 0.0)
        distances[np.arange(chunk_indices.shape[0]), chunk_indices] = np.inf
        if top_k < embedding_array.shape[0]:
            top_indices = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
            top_distances = np.take_along_axis(distances, top_indices, axis=1)
            top_order = np.argsort(top_distances, axis=1, kind="stable")
            orders = np.take_along_axis(top_indices, top_order, axis=1)
        else:
            orders = np.argsort(distances, axis=1, kind="stable")
        for row_position, query_index in enumerate(chunk_indices):
            query_label = label_array[query_index]
            matches = label_array == query_label
            matches[query_index] = False
            ordered_matches = matches[orders[row_position]]
            relevant_count = int(matches.sum())
            if relevant_count == 0:
                continue

            precision_at_1_values.append(float(ordered_matches[0]))
            for k in recall_at_k_values:
                recall_at_k_values[k].append(float(bool(ordered_matches[:k].any())))

            top_r_matches = ordered_matches[:relevant_count]
            relevant_ranks = np.flatnonzero(top_r_matches) + 1
            precisions = [float(top_r_matches[:rank].sum() / rank) for rank in relevant_ranks]
            average_precisions.append(float(sum(precisions) / relevant_count))
            relevant_counts.append(relevant_count)

    if not average_precisions:
        raise ValueError("self retrieval requires at least two examples for one label")

    return ImageRetrievalMetrics(
        precision_at_1=float(np.mean(precision_at_1_values)),
        recall_at_1=float(np.mean(recall_at_k_values[1])),
        recall_at_2=float(np.mean(recall_at_k_values[2])),
        recall_at_4=float(np.mean(recall_at_k_values[4])),
        recall_at_8=float(np.mean(recall_at_k_values[8])),
        map_at_r=float(np.mean(average_precisions)),
        mean_relevant_items=float(np.mean(relevant_counts)),
        evaluated_queries=len(average_precisions),
        total_queries=int(embedding_array.shape[0]),
    )


def image_query_gallery_retrieval_score(
    query_embeddings: NDArray[np.floating],
    query_labels: NDArray[np.integer],
    gallery_embeddings: NDArray[np.floating],
    gallery_labels: NDArray[np.integer],
    *,
    query_limit: int | None = None,
    random_state: int = 0,
) -> ImageRetrievalMetrics:
    """Retrieve each query against a disjoint gallery (the In-Shop / consumer-to-shop
    protocol). Unlike self-retrieval there is no leave-one-out: the query set and the
    gallery set are separate, so nothing is excluded from the ranking. A query counts
    as relevant to a gallery item when they share the identity label.

    This is a standalone scoring **primitive** (no dataset loader wires it yet). It
    returns the project's uniform ``ImageRetrievalMetrics`` — recall at **1/2/4/8** and
    MAP@R — so ``recall_at_1`` is directly comparable; note the In-Shop *paper* headline
    is R@1/10/20/30, which this fixed-cutoff metric does not report."""
    q_emb = np.asarray(query_embeddings, dtype=np.float64)
    g_emb = np.asarray(gallery_embeddings, dtype=np.float64)
    q_lab = np.asarray(query_labels, dtype=np.int64)
    g_lab = np.asarray(gallery_labels, dtype=np.int64)
    if q_emb.ndim != 2 or g_emb.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")
    if q_emb.shape[1] != g_emb.shape[1]:
        raise ValueError("query and gallery embeddings must share the feature dimension")
    if q_emb.shape[0] != q_lab.shape[0] or g_emb.shape[0] != g_lab.shape[0]:
        raise ValueError("embeddings and labels must contain the same number of examples")
    if q_emb.shape[0] < 1 or g_emb.shape[0] < 1:
        raise ValueError("query and gallery must each contain at least one example")
    if query_limit is not None and query_limit < 1:
        raise ValueError("query_limit must be at least 1")

    query_indices = np.arange(q_emb.shape[0], dtype=np.int64)
    if query_limit is not None and query_limit < query_indices.shape[0]:
        query_indices = _stratified_query_indices(
            q_lab, query_limit=query_limit, random_state=random_state
        )

    gallery_label_counts = {
        int(label): int(count)
        for label, count in zip(*np.unique(g_lab, return_counts=True), strict=True)
    }
    query_relevant_counts = np.asarray(
        [gallery_label_counts.get(int(q_lab[index]), 0) for index in query_indices],
        dtype=np.int64,
    )
    max_relevant_count = int(query_relevant_counts.max(initial=0))
    top_k = min(g_emb.shape[0], max(8, max_relevant_count))

    precision_at_1_values: list[float] = []
    recall_at_k_values: dict[int, list[float]] = {1: [], 2: [], 4: [], 8: []}
    average_precisions: list[float] = []
    relevant_counts: list[int] = []
    gallery_norms = np.sum(g_emb * g_emb, axis=1)
    chunk_size = 1024
    for start in range(0, query_indices.shape[0], chunk_size):
        chunk_indices = query_indices[start : start + chunk_size]
        chunk = q_emb[chunk_indices]
        distances = (
            np.sum(chunk * chunk, axis=1, keepdims=True)
            + gallery_norms[np.newaxis, :]
            - (2.0 * chunk @ g_emb.T)
        )
        distances = np.maximum(distances, 0.0)
        if top_k < g_emb.shape[0]:
            top_indices = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
            top_distances = np.take_along_axis(distances, top_indices, axis=1)
            top_order = np.argsort(top_distances, axis=1, kind="stable")
            orders = np.take_along_axis(top_indices, top_order, axis=1)
        else:
            orders = np.argsort(distances, axis=1, kind="stable")
        for row_position, query_index in enumerate(chunk_indices):
            matches = g_lab == q_lab[query_index]
            relevant_count = int(matches.sum())
            if relevant_count == 0:
                continue
            ordered_matches = matches[orders[row_position]]
            precision_at_1_values.append(float(ordered_matches[0]))
            for k in recall_at_k_values:
                recall_at_k_values[k].append(float(bool(ordered_matches[:k].any())))
            top_r_matches = ordered_matches[:relevant_count]
            relevant_ranks = np.flatnonzero(top_r_matches) + 1
            precisions = [float(top_r_matches[:rank].sum() / rank) for rank in relevant_ranks]
            average_precisions.append(float(sum(precisions) / relevant_count))
            relevant_counts.append(relevant_count)

    if not average_precisions:
        raise ValueError("no query shares an identity label with any gallery item")

    return ImageRetrievalMetrics(
        precision_at_1=float(np.mean(precision_at_1_values)),
        recall_at_1=float(np.mean(recall_at_k_values[1])),
        recall_at_2=float(np.mean(recall_at_k_values[2])),
        recall_at_4=float(np.mean(recall_at_k_values[4])),
        recall_at_8=float(np.mean(recall_at_k_values[8])),
        map_at_r=float(np.mean(average_precisions)),
        mean_relevant_items=float(np.mean(relevant_counts)),
        evaluated_queries=len(average_precisions),
        total_queries=int(q_emb.shape[0]),
    )


def objective_display_name(objective: str) -> str:
    """Return the human-readable objective name used in reports."""
    names = {
        "frozen": "Frozen",
        "triplet": "Triplet",
        "batch_hard_triplet": "Batch-Hard Triplet",
        "group": "Group",
        "hard_group": "Hard Group",
        "supcon": "Supervised Contrastive",
        "proxy_nca": "Proxy-NCA",
        "proxy_anchor": "Proxy Anchor",
        "cosface": "CosFace",
        "arcface": "ArcFace",
        "group_supcon": "Group SupCon",
        "hybrid": "Hybrid",
        "hybrid_xbm": "Hybrid + XBM",
        "hybrid_radius": "Hybrid + Radius",
        "hybrid_xbm_radius": "Hybrid + XBM + Radius",
        "group_supcon_xbm_radius": "Group SupCon + XBM + Radius",
        "all": "Hybrid + XBM + Radius",
    }
    return names.get(objective, objective.replace("_", " ").title())


def _encode_examples_with_cache(
    *,
    model_name: str,
    split_name: str,
    examples: list[ImageExample],
    config: ImageBenchmarkConfig,
    encoder: FrozenImageEncoder | None,
    encoder_factory: ImageEncoderFactory,
) -> tuple[NDArray[np.float64], FrozenImageEncoder | None]:
    cache_path = _embedding_cache_path(
        cache_dir=config.embedding_cache_dir,
        dataset_name=config.dataset_name,
        split_name=split_name,
        model_name=model_name,
        normalize_embeddings=config.normalize_embeddings,
        examples=examples,
    )
    if cache_path is not None:
        cached = _load_cached_embeddings(cache_path, examples)
        if cached is not None:
            return cached, encoder

    resolved_encoder = encoder or encoder_factory(model_name)
    embeddings = resolved_encoder.encode(
        [example.image for example in examples],
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
    )
    if cache_path is not None:
        _write_cached_embeddings(cache_path, embeddings, examples)
    return embeddings, resolved_encoder


def _embedding_cache_path(
    *,
    cache_dir: Path | None,
    dataset_name: str,
    split_name: str,
    model_name: str,
    normalize_embeddings: bool,
    examples: list[ImageExample],
) -> Path | None:
    if cache_dir is None:
        return None
    payload = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "model_name": model_name,
        "normalize_embeddings": normalize_embeddings,
        "examples": [(example.example_id, example.label) for example in examples],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    safe_model_name = "".join(character if character.isalnum() else "-" for character in model_name)
    safe_model_name = safe_model_name.strip("-") or "model"
    return cache_dir / f"{dataset_name}_{split_name}_{safe_model_name}_{digest}.npz"


def _load_cached_embeddings(
    cache_path: Path,
    examples: list[ImageExample],
) -> NDArray[np.float64] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
            example_ids = payload["example_ids"].astype(str).tolist()
            labels = payload["labels"].astype(np.int64).tolist()
    except (OSError, KeyError, ValueError):
        return None

    expected_ids = [example.example_id for example in examples]
    expected_labels = [example.label for example in examples]
    if example_ids != expected_ids or labels != expected_labels:
        return None
    if embeddings.ndim != 2 or embeddings.shape[0] != len(examples):
        return None
    return embeddings


def _write_cached_embeddings(
    cache_path: Path,
    embeddings: NDArray[np.float64],
    examples: list[ImageExample],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=np.asarray(embeddings, dtype=np.float64),
        example_ids=np.asarray([example.example_id for example in examples], dtype=str),
        labels=np.asarray([example.label for example in examples], dtype=np.int64),
    )


def write_image_benchmark_report(result: ImageBenchmarkResult, output_path: Path) -> Path:
    """Persist an image benchmark result as stable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_to_json(result), encoding="utf-8")
    return output_path


def _method_metrics(
    *,
    model_name: str,
    objective: str,
    display_name: str,
    embeddings: NDArray[np.float64],
    labels: NDArray[np.int64],
    retrieval: ImageRetrievalMetrics,
    baseline: ImageRetrievalMetrics,
    objective_history: list[float],
    initial_triplet_loss: float | None,
    triplet_loss: float | None,
    initial_group_loss: float | None,
    group_loss: float | None,
    selected_step: int | None = None,
    selection_score: float | None = None,
) -> ImageBenchmarkMethodMetrics:
    space = None
    if np.unique(labels).shape[0] >= 2:
        space = embedding_space_diagnostics_on_split(embeddings, labels, embeddings, labels)
    return ImageBenchmarkMethodMetrics(
        model_name=model_name,
        objective=objective,
        display_name=display_name,
        dimensions=int(embeddings.shape[1]),
        retrieval=retrieval,
        recall_at_1=retrieval.recall_at_1,
        recall_at_2=retrieval.recall_at_2,
        recall_at_4=retrieval.recall_at_4,
        recall_at_8=retrieval.recall_at_8,
        map_at_r=retrieval.map_at_r,
        recall_at_1_delta=retrieval.recall_at_1 - baseline.recall_at_1,
        map_at_r_delta=retrieval.map_at_r - baseline.map_at_r,
        initial_triplet_loss=initial_triplet_loss,
        triplet_loss=triplet_loss,
        initial_group_loss=initial_group_loss,
        group_loss=group_loss,
        objective_history=objective_history,
        selected_step=selected_step,
        selection_score=selection_score,
        space=space,
    )


def _best_method_name(methods: dict[str, ImageBenchmarkMethodMetrics]) -> str | None:
    if not methods:
        return None
    return max(methods, key=lambda name: _best_method_score(methods[name]))


def _best_method_score(method: ImageBenchmarkMethodMetrics) -> float:
    return method.map_at_r


def _projection_objective(objective: ImageObjective) -> Objective:
    return objective


def _projection_min_per_class(config: ImageBenchmarkConfig) -> int:
    group_objectives = {
        "group",
        "hard_group",
        "hybrid",
        "hybrid_xbm",
        "hybrid_radius",
        "hybrid_xbm_radius",
        "group_supcon",
        "group_supcon_xbm_radius",
    }
    if any(objective in group_objectives for objective in config.objectives):
        return config.group_size * 2
    return 2


def _projection_selection_callback(
    *,
    validation_embeddings: NDArray[np.float64],
    validation_labels: NDArray[np.int64],
    normalize_embeddings: bool,
    query_limit: int | None,
    random_state: int,
) -> Callable[[NDArray[np.float64], int], float] | None:
    if validation_embeddings.shape[0] < 2 or np.unique(validation_labels).shape[0] < 2:
        return None

    def score(projection: NDArray[np.float64], step: int) -> float:
        del step
        projected = validation_embeddings @ projection
        if normalize_embeddings:
            projected = _normalize(projected)
        return image_self_retrieval_score(
            projected,
            validation_labels,
            query_limit=query_limit,
            random_state=random_state,
        ).map_at_r

    return score


def _normalize(embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _stratified_query_indices(
    labels: NDArray[np.int64],
    *,
    query_limit: int,
    random_state: int,
) -> NDArray[np.int64]:
    rng = np.random.default_rng(random_state)
    grouped = {
        label: np.flatnonzero(labels == label) for label in sorted(np.unique(labels).tolist())
    }
    ordered_labels = np.asarray(list(grouped), dtype=np.int64)
    rng.shuffle(ordered_labels)
    class_count = min(query_limit, len(ordered_labels))
    sampled_labels = sorted(int(label) for label in ordered_labels[:class_count])
    base_quota = query_limit // class_count
    remainder = query_limit % class_count
    selected: list[int] = []
    for position, label in enumerate(sampled_labels):
        indices = grouped[label].copy()
        rng.shuffle(indices)
        quota = min(len(indices), base_quota + (1 if position < remainder else 0))
        selected.extend(int(index) for index in indices[:quota])
    return np.asarray(sorted(selected), dtype=np.int64)


def _stratified_subset_indices(
    labels: NDArray[np.int64],
    *,
    limit: int | None,
    min_per_class: int,
    random_state: int,
) -> NDArray[np.int64]:
    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    if limit is None or limit >= all_indices.shape[0]:
        return all_indices
    if limit < min_per_class * 2:
        raise ValueError("projection_train_limit must fit at least two classes")

    rng = np.random.default_rng(random_state)
    grouped = {
        int(label): np.flatnonzero(labels == label) for label in sorted(np.unique(labels).tolist())
    }
    eligible_labels = [
        label for label, indices in grouped.items() if indices.shape[0] >= min_per_class
    ]
    max_label_count = min(len(eligible_labels), limit // min_per_class)
    if max_label_count < 2:
        raise ValueError("projection training subset requires at least two eligible classes")

    shuffled_labels = np.asarray(eligible_labels, dtype=np.int64)
    rng.shuffle(shuffled_labels)
    selected_labels = sorted(int(label) for label in shuffled_labels[:max_label_count])
    remaining = limit - (max_label_count * min_per_class)
    selected: list[int] = []
    for label in selected_labels:
        indices = grouped[label].copy()
        rng.shuffle(indices)
        take = min_per_class
        if remaining > 0:
            extra = min(remaining, indices.shape[0] - min_per_class)
            take += extra
            remaining -= extra
        selected.extend(int(index) for index in indices[:take])
    return np.asarray(sorted(selected), dtype=np.int64)


def _projection_train_validation_indices(
    labels: NDArray[np.int64],
    *,
    train_limit: int | None,
    validation_fraction: float,
    validation_limit: int | None,
    train_min_per_class: int,
    random_state: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if validation_fraction == 0.0:
        return (
            _stratified_subset_indices(
                labels,
                limit=train_limit,
                min_per_class=train_min_per_class,
                random_state=random_state,
            ),
            np.asarray([], dtype=np.int64),
        )

    grouped = {
        int(label): np.flatnonzero(labels == label) for label in sorted(np.unique(labels).tolist())
    }
    train_eligible_labels = [
        label for label, indices in grouped.items() if indices.shape[0] >= train_min_per_class
    ]
    validation_eligible_labels = [
        label for label, indices in grouped.items() if indices.shape[0] >= 2
    ]
    shared_labels = [
        label for label in train_eligible_labels if label in validation_eligible_labels
    ]
    validation_label_count = int(round(len(shared_labels) * validation_fraction))
    if validation_label_count < 2 or len(shared_labels) - validation_label_count < 2:
        return (
            _stratified_subset_indices(
                labels,
                limit=train_limit,
                min_per_class=train_min_per_class,
                random_state=random_state,
            ),
            np.asarray([], dtype=np.int64),
        )

    rng = np.random.default_rng(random_state)
    shuffled_labels = np.asarray(shared_labels, dtype=np.int64)
    rng.shuffle(shuffled_labels)
    validation_labels = set(int(label) for label in shuffled_labels[:validation_label_count])
    train_labels = [
        int(label) for label in train_eligible_labels if int(label) not in validation_labels
    ]
    train_indices = _subset_indices_from_labels(
        grouped,
        labels=train_labels,
        limit=train_limit,
        min_per_class=train_min_per_class,
        random_state=random_state,
    )
    validation_indices = _subset_indices_from_labels(
        grouped,
        labels=sorted(validation_labels),
        limit=validation_limit,
        min_per_class=2,
        random_state=random_state + 1,
    )
    return train_indices, validation_indices


def _subset_indices_from_labels(
    grouped: dict[int, NDArray[np.int64]],
    *,
    labels: list[int],
    limit: int | None,
    min_per_class: int,
    random_state: int,
) -> NDArray[np.int64]:
    if len(labels) < 2:
        raise ValueError("projection subset requires at least two eligible classes")
    if limit is not None and limit < min_per_class * 2:
        raise ValueError("projection subset limit must fit at least two classes")

    rng = np.random.default_rng(random_state)
    max_label_count = len(labels) if limit is None else min(len(labels), limit // min_per_class)
    if max_label_count < 2:
        raise ValueError("projection subset requires at least two eligible classes")
    shuffled_labels = np.asarray(labels, dtype=np.int64)
    rng.shuffle(shuffled_labels)
    selected_labels = sorted(int(label) for label in shuffled_labels[:max_label_count])
    remaining = 0 if limit is None else limit - (max_label_count * min_per_class)
    selected: list[int] = []
    for label in selected_labels:
        indices = grouped[label].copy()
        rng.shuffle(indices)
        take = indices.shape[0] if limit is None else min_per_class
        if limit is not None and remaining > 0:
            extra = min(remaining, indices.shape[0] - min_per_class)
            take += extra
            remaining -= extra
        selected.extend(int(index) for index in indices[:take])
    return np.asarray(sorted(selected), dtype=np.int64)


def _load_transformers_image_encoder(model_name: str) -> FrozenImageEncoder:
    return _TransformersImageEncoder(model_name)


class _TransformersImageEncoder:
    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError(
                "Install the research extra to run image benchmarks: "
                "uv sync --group dev --extra research"
            ) from error

        self._torch: Any = torch
        processor_cls: Any = AutoImageProcessor
        model_cls: Any = AutoModel
        self._processor: Any = processor_cls.from_pretrained(model_name)
        self._model: Any = model_cls.from_pretrained(model_name)
        self._device: Any = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()

    def encode(
        self,
        images: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> NDArray[np.float64]:
        batches: list[NDArray[np.float64]] = []
        with self._torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size]
                features = self._processor(images=batch, return_tensors="pt")
                features = {
                    name: value.to(self._device) if hasattr(value, "to") else value
                    for name, value in features.items()
                }
                if hasattr(self._model, "get_image_features"):
                    output = self._model.get_image_features(**features)
                else:
                    model_output = self._model(**features)
                    output = self._embedding_tensor(model_output)
                output = self._embedding_tensor(output)
                if normalize_embeddings:
                    output = self._torch.nn.functional.normalize(output, p=2, dim=-1)
                batches.append(output.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(batches, axis=0)

    def _embedding_tensor(self, output: Any) -> Any:
        if self._torch.is_tensor(output):
            return output
        image_embeds = getattr(output, "image_embeds", None)
        if self._torch.is_tensor(image_embeds):
            return image_embeds
        pooler_output = getattr(output, "pooler_output", None)
        if self._torch.is_tensor(pooler_output):
            return pooler_output
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if last_hidden_state is not None and self._torch.is_tensor(last_hidden_state):
            return last_hidden_state[:, 0, :]
        if isinstance(output, (tuple, list)) and output:
            return self._embedding_tensor(output[0])
        raise RuntimeError("image encoder did not return a tensor-like embedding output")


def _to_json(result: ImageBenchmarkResult) -> str:
    return json.dumps(_to_payload(result), indent=2, sort_keys=True) + "\n"


def _to_payload(result: ImageBenchmarkResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "dataset_name": result.dataset_name,
        "config": result.config.model_dump(mode="json"),
        "examples": result.examples,
        "train_examples": result.train_examples,
        "projection_train_examples": result.projection_train_examples,
        "projection_validation_examples": result.projection_validation_examples,
        "test_examples": result.test_examples,
        "best_method": result.best_method,
        "methods": {
            name: {
                "model_name": metrics.model_name,
                "objective": metrics.objective,
                "display_name": metrics.display_name,
                "dimensions": metrics.dimensions,
                "retrieval": asdict(metrics.retrieval),
                "precision_at_1": metrics.retrieval.precision_at_1,
                "recall_at_1": metrics.recall_at_1,
                "recall_at_2": metrics.recall_at_2,
                "recall_at_4": metrics.recall_at_4,
                "recall_at_8": metrics.recall_at_8,
                "map_at_r": metrics.map_at_r,
                "recall_at_1_delta": metrics.recall_at_1_delta,
                "map_at_r_delta": metrics.map_at_r_delta,
                "initial_triplet_loss": metrics.initial_triplet_loss,
                "triplet_loss": metrics.triplet_loss,
                "initial_group_loss": metrics.initial_group_loss,
                "group_loss": metrics.group_loss,
                "objective_history": metrics.objective_history,
                "selected_step": metrics.selected_step,
                "selection_score": metrics.selection_score,
                "space": None if metrics.space is None else asdict(metrics.space),
            }
            for name, metrics in result.methods.items()
        },
    }
