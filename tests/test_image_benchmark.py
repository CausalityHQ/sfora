import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from sfora.data import ImageExample
from sfora.image_benchmark import (
    ImageBenchmarkConfig,
    ImageBenchmarkMethodMetrics,
    _best_method_name,
    _projection_min_per_class,
    _stratified_query_indices,
    image_query_gallery_retrieval_score,
    image_self_retrieval_score,
    run_image_benchmark,
    write_image_benchmark_report,
)


def test_query_gallery_retrieval_scores_cross_set() -> None:
    # Gallery has two items per identity; queries are near their own identity's
    # gallery vectors, so top-1 is always the same identity -> R@1 == 1.
    rng = np.random.default_rng(0)
    centers = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0]), 2: np.array([-1.0, 0.0])}
    gallery_emb, gallery_lab, query_emb, query_lab = [], [], [], []
    for label, center in centers.items():
        for _ in range(2):
            gallery_emb.append(center + 0.01 * rng.standard_normal(2))
            gallery_lab.append(label)
        query_emb.append(center + 0.01 * rng.standard_normal(2))
        query_lab.append(label)
    metrics = image_query_gallery_retrieval_score(
        np.array(query_emb), np.array(query_lab), np.array(gallery_emb), np.array(gallery_lab)
    )
    assert metrics.recall_at_1 == 1.0
    assert metrics.evaluated_queries == 3
    assert 0.0 < metrics.map_at_r <= 1.0


def test_query_gallery_retrieval_skips_queries_absent_from_gallery() -> None:
    # Query label 9 has no gallery match -> skipped; the one matchable query scores.
    gallery_emb = np.array([[1.0, 0.0], [1.0, 0.0]])
    gallery_lab = np.array([0, 0])
    query_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    query_lab = np.array([0, 9])
    metrics = image_query_gallery_retrieval_score(query_emb, query_lab, gallery_emb, gallery_lab)
    assert metrics.evaluated_queries == 1
    assert metrics.total_queries == 2
    assert metrics.recall_at_1 == 1.0


def _image_examples(prefix: str, labels: tuple[int, ...]) -> list[ImageExample]:
    return [
        ImageExample(
            example_id=f"{prefix}-{label}-{index}", image=f"{prefix}-{label}-{index}", label=label
        )
        for label in labels
        for index in range(4)
    ]


def _image_examples_with_count(
    prefix: str,
    labels: tuple[int, ...],
    *,
    examples_per_label: int,
) -> list[ImageExample]:
    return [
        ImageExample(
            example_id=f"{prefix}-{label}-{index}", image=f"{prefix}-{label}-{index}", label=label
        )
        for label in labels
        for index in range(examples_per_label)
    ]


@dataclass
class FakeImageEncoder:
    model_name: str
    encode_calls: int = 0

    def encode(
        self,
        images: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> NDArray[np.float64]:
        self.encode_calls += 1
        assert batch_size == 8
        rows = []
        model_shift = 0.1 if self.model_name == "fake-dino" else -0.1
        for image in images:
            parts = str(image).split("-")
            label = int(parts[-2])
            index = int(parts[-1])
            rows.append(
                [
                    float(label) + model_shift,
                    float(index) / 10.0,
                    1.0 if label % 2 == 0 else -1.0,
                ]
            )
        embeddings = np.asarray(rows, dtype=np.float64)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings


def _factory(model_name: str) -> FakeImageEncoder:
    return FakeImageEncoder(model_name)


def test_image_self_retrieval_scores_perfect_neighbors_with_query_excluded() -> None:
    embeddings = np.asarray(
        [
            [-2.0, -2.0],
            [-1.9, -2.1],
            [2.0, 2.0],
            [2.1, 1.9],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    score = image_self_retrieval_score(embeddings, labels)

    assert score.precision_at_1 == pytest.approx(1.0)
    assert score.recall_at_1 == pytest.approx(1.0)
    assert score.map_at_r == pytest.approx(1.0)
    assert score.mean_relevant_items == pytest.approx(1.0)
    assert score.evaluated_queries == 4
    assert score.total_queries == 4


def test_image_self_retrieval_map_at_r_penalizes_late_relevant_neighbors() -> None:
    embeddings = np.asarray([[0.0], [1.0], [0.2], [0.3]], dtype=np.float64)
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)

    score = image_self_retrieval_score(embeddings, labels)

    assert score.precision_at_1 == pytest.approx(0.5)
    assert score.recall_at_1 == pytest.approx(0.5)
    assert score.map_at_r == pytest.approx(0.5)
    assert score.mean_relevant_items == pytest.approx(1.0)


def test_image_self_retrieval_query_limit_uses_deterministic_subset() -> None:
    embeddings = np.asarray([[float(index), 0.0] for index in range(12)], dtype=np.float64)
    labels = np.asarray([0] * 4 + [1] * 4 + [2] * 4, dtype=np.int64)

    first = image_self_retrieval_score(embeddings, labels, query_limit=6, random_state=7)
    second = image_self_retrieval_score(embeddings, labels, query_limit=6, random_state=7)

    assert first.evaluated_queries == 6
    assert first.total_queries == 12
    assert first == second


def test_stratified_query_limit_samples_classes_when_limit_is_smaller_than_class_count() -> None:
    labels = np.asarray([label for label in range(20) for _ in range(2)], dtype=np.int64)

    selected = _stratified_query_indices(labels, query_limit=5, random_state=11)

    assert selected.shape == (5,)
    selected_labels = labels[selected].tolist()
    assert len(set(selected_labels)) == 5
    assert selected_labels != [0, 1, 2, 3, 4]


def test_run_image_benchmark_compares_models_and_human_named_objectives() -> None:
    result = run_image_benchmark(
        train_examples=_image_examples("train", (0, 1, 2)),
        test_examples=_image_examples("test", (0, 1, 2)),
        config=ImageBenchmarkConfig(
            dataset_name="cub",
            model_names=("fake-dino", "fake-clip"),
            objectives=(
                "batch_hard_triplet",
                "hard_group",
                "proxy_nca",
                "proxy_anchor",
                "cosface",
                "arcface",
                "supcon",
                "group_supcon",
                "hybrid_xbm_radius",
                "group_supcon_xbm_radius",
            ),
            group_size=2,
            batch_size=8,
            train_steps=5,
            learning_rate=0.01,
            retrieval_query_limit=6,
            seed=5,
        ),
        encoder_factory=_factory,
    )

    assert result.name == "image-retrieval-benchmark"
    assert result.dataset_name == "cub"
    assert result.train_examples == 12
    assert result.test_examples == 12
    assert result.best_method is not None
    assert set(result.methods) == {
        "frozen:fake-dino",
        "batch_hard_triplet_projection:fake-dino",
        "hard_group_projection:fake-dino",
        "proxy_nca_projection:fake-dino",
        "proxy_anchor_projection:fake-dino",
        "cosface_projection:fake-dino",
        "arcface_projection:fake-dino",
        "supcon_projection:fake-dino",
        "group_supcon_projection:fake-dino",
        "hybrid_xbm_radius_projection:fake-dino",
        "group_supcon_xbm_radius_projection:fake-dino",
        "frozen:fake-clip",
        "batch_hard_triplet_projection:fake-clip",
        "hard_group_projection:fake-clip",
        "proxy_nca_projection:fake-clip",
        "proxy_anchor_projection:fake-clip",
        "cosface_projection:fake-clip",
        "arcface_projection:fake-clip",
        "supcon_projection:fake-clip",
        "group_supcon_projection:fake-clip",
        "hybrid_xbm_radius_projection:fake-clip",
        "group_supcon_xbm_radius_projection:fake-clip",
    }
    assert result.methods["batch_hard_triplet_projection:fake-dino"].display_name == (
        "Batch-Hard Triplet"
    )
    assert result.methods["hard_group_projection:fake-dino"].display_name == "Hard Group"
    assert result.methods["proxy_nca_projection:fake-dino"].display_name == "Proxy-NCA"
    assert result.methods["proxy_anchor_projection:fake-dino"].display_name == "Proxy Anchor"
    assert result.methods["cosface_projection:fake-dino"].display_name == "CosFace"
    assert result.methods["arcface_projection:fake-dino"].display_name == "ArcFace"
    assert result.methods["supcon_projection:fake-dino"].display_name == "Supervised Contrastive"
    assert result.methods["group_supcon_projection:fake-dino"].display_name == "Group SupCon"
    assert result.methods["group_supcon_xbm_radius_projection:fake-dino"].display_name == (
        "Group SupCon + XBM + Radius"
    )
    hybrid = result.methods["hybrid_xbm_radius_projection:fake-dino"]
    assert hybrid.objective == "hybrid_xbm_radius"
    assert hybrid.display_name == "Hybrid + XBM + Radius"
    assert hybrid.retrieval.evaluated_queries == 6
    assert hybrid.recall_at_1 <= 1.0
    assert hybrid.map_at_r <= 1.0


def test_run_image_benchmark_can_train_projection_on_stratified_subset() -> None:
    result = run_image_benchmark(
        train_examples=_image_examples("train", (0, 1, 2)),
        test_examples=_image_examples("test", (0, 1, 2)),
        config=ImageBenchmarkConfig(
            dataset_name="cub",
            model_names=("fake-dino",),
            objectives=("hybrid_xbm_radius",),
            group_size=2,
            batch_size=8,
            train_steps=3,
            projection_train_limit=8,
            seed=5,
        ),
        encoder_factory=_factory,
    )

    assert result.train_examples == 12
    assert result.projection_train_examples == 8
    assert result.test_examples == 12


def test_run_image_benchmark_records_memory_and_group_shuffle_config() -> None:
    result = run_image_benchmark(
        train_examples=_image_examples("train", (0, 1, 2)),
        test_examples=_image_examples("test", (0, 1, 2)),
        config=ImageBenchmarkConfig(
            dataset_name="cub",
            model_names=("fake-dino",),
            objectives=("group_supcon_xbm_radius",),
            group_size=2,
            batch_size=8,
            train_steps=2,
            xbm_memory_size=32,
            shuffle_groups_each_step=True,
            seed=5,
        ),
        encoder_factory=_factory,
    )

    assert result.config.xbm_memory_size == 32
    assert result.config.shuffle_groups_each_step is True


def test_run_image_benchmark_reuses_cached_frozen_embeddings(tmp_path: Path) -> None:
    encoder = FakeImageEncoder("fake-dino")

    def factory(model_name: str) -> FakeImageEncoder:
        assert model_name == "fake-dino"
        return encoder

    config = ImageBenchmarkConfig(
        dataset_name="cub",
        model_names=("fake-dino",),
        objectives=("triplet",),
        group_size=2,
        batch_size=8,
        train_steps=1,
        embedding_cache_dir=tmp_path,
        seed=5,
    )

    run_image_benchmark(
        train_examples=_image_examples("train", (0, 1)),
        test_examples=_image_examples("test", (0, 1)),
        config=config,
        encoder_factory=factory,
    )
    run_image_benchmark(
        train_examples=_image_examples("train", (0, 1)),
        test_examples=_image_examples("test", (0, 1)),
        config=config,
        encoder_factory=factory,
    )

    assert encoder.encode_calls == 2
    result_cache_files = list(tmp_path.glob("*.npz"))
    assert result_cache_files
    assert len(result_cache_files) == 2


def test_projection_training_subset_keeps_two_group_sized_class_blocks() -> None:
    result = run_image_benchmark(
        train_examples=_image_examples_with_count(
            "train",
            tuple(range(6)),
            examples_per_label=8,
        ),
        test_examples=_image_examples_with_count("test", tuple(range(6)), examples_per_label=8),
        config=ImageBenchmarkConfig(
            dataset_name="sop",
            model_names=("fake-dino",),
            objectives=("hybrid_xbm_radius",),
            group_size=4,
            batch_size=8,
            train_steps=2,
            projection_train_limit=16,
            seed=5,
        ),
        encoder_factory=_factory,
    )

    assert result.train_examples == 48
    assert result.projection_train_examples == 16


def test_image_benchmark_uses_disjoint_projection_validation_split() -> None:
    result = run_image_benchmark(
        train_examples=_image_examples_with_count(
            "train",
            tuple(range(6)),
            examples_per_label=8,
        ),
        test_examples=_image_examples_with_count("test", tuple(range(6)), examples_per_label=8),
        config=ImageBenchmarkConfig(
            dataset_name="sop",
            model_names=("fake-dino",),
            objectives=("hybrid_xbm",),
            group_size=4,
            batch_size=8,
            train_steps=2,
            projection_train_limit=16,
            projection_validation_fraction=0.5,
            projection_validation_limit=16,
            seed=5,
        ),
        encoder_factory=_factory,
    )

    method = result.methods["hybrid_xbm_projection:fake-dino"]
    assert result.projection_train_examples == 16
    assert result.projection_validation_examples == 16
    assert method.selected_step is not None
    assert method.selection_score is not None


def test_projection_min_per_class_requires_two_groups_for_hard_group() -> None:
    config = ImageBenchmarkConfig(objectives=("hard_group",), group_size=5)

    assert _projection_min_per_class(config) == 10


def test_projection_min_per_class_requires_two_groups_for_group_supcon() -> None:
    config = ImageBenchmarkConfig(objectives=("group_supcon",), group_size=5)

    assert _projection_min_per_class(config) == 10


def test_projection_min_per_class_requires_two_groups_for_group_supcon_xbm_radius() -> None:
    config = ImageBenchmarkConfig(objectives=("group_supcon_xbm_radius",), group_size=5)

    assert _projection_min_per_class(config) == 10


def test_best_method_can_remain_frozen_when_projection_is_worse() -> None:
    methods = cast(
        dict[str, ImageBenchmarkMethodMetrics],
        {
            "frozen:fake-dino": SimpleNamespace(map_at_r=0.8),
            "hybrid_xbm_projection:fake-dino": SimpleNamespace(map_at_r=0.7),
        },
    )

    assert _best_method_name(methods) == "frozen:fake-dino"


def test_write_image_benchmark_report_persists_retrieval_metrics(tmp_path: Path) -> None:
    result = run_image_benchmark(
        train_examples=_image_examples("train", (0, 1)),
        test_examples=_image_examples("test", (0, 1)),
        config=ImageBenchmarkConfig(
            dataset_name="cars",
            model_names=("fake-dino",),
            objectives=("hybrid_xbm_radius",),
            group_size=2,
            batch_size=8,
            train_steps=3,
            retrieval_query_limit=4,
        ),
        encoder_factory=_factory,
    )

    output_path = write_image_benchmark_report(result, tmp_path / "image.json")

    payload = json.loads(output_path.read_text())
    assert payload["name"] == "image-retrieval-benchmark"
    assert payload["dataset_name"] == "cars"
    assert payload["config"]["objectives"] == ["hybrid_xbm_radius"]
    assert "frozen:fake-dino" in payload["methods"]
    assert payload["methods"]["hybrid_xbm_radius_projection:fake-dino"]["display_name"] == (
        "Hybrid + XBM + Radius"
    )
    assert "recall_at_1" in payload["methods"]["frozen:fake-dino"]
    assert "map_at_r_delta" in payload["methods"]["hybrid_xbm_radius_projection:fake-dino"]
