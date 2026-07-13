import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sfora.data import TextExample
from sfora.encoder_training import (
    EncoderObjective,
    EncoderTrainingConfig,
    run_encoder_training,
    run_encoder_training_on_split,
    write_encoder_training_report,
)
from sfora.training import ProjectionHeadTrainingConfig, train_projection_head


def _examples() -> list[TextExample]:
    negatives = [
        "bad dull awful film",
        "boring weak slow movie",
        "poor flat tedious story",
        "awful dull weak acting",
        "slow boring bad scenes",
        "flat poor tedious film",
    ]
    positives = [
        "great vivid excellent film",
        "moving sharp joyful movie",
        "strong bright wonderful story",
        "excellent vivid sharp acting",
        "joyful moving great scenes",
        "bright strong wonderful film",
    ]
    return [
        TextExample(example_id=f"neg-{index}", text=text, label=0)
        for index, text in enumerate(negatives)
    ] + [
        TextExample(example_id=f"pos-{index}", text=text, label=1)
        for index, text in enumerate(positives)
    ]


@dataclass
class FakeTrainableEncoder:
    projection: NDArray[np.float64]
    fit_example_counts: list[int]

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> NDArray[np.float64]:
        assert batch_size == 4
        base = np.array([_features(text) for text in texts], dtype=np.float64)
        embeddings = base @ self.projection
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings

    def fit(
        self,
        examples: list[TextExample],
        *,
        objective: EncoderObjective,
        config: EncoderTrainingConfig,
    ) -> list[float]:
        labels = np.array([example.label for example in examples], dtype=np.int64)
        base = np.array([_features(example.text) for example in examples], dtype=np.float64)
        projection_objective: Literal["triplet", "group"] = (
            "triplet" if objective == "triplet" else "group"
        )
        result = train_projection_head(
            base,
            labels,
            ProjectionHeadTrainingConfig(
                objective=projection_objective,
                group_size=config.group_size,
                steps=config.train_steps,
                learning_rate=config.learning_rate,
                margin=config.margin,
                hard_weight=config.hard_weight,
                spread_weight=config.spread_weight,
            ),
        )
        self.projection = result.projection_matrix
        return result.history


def _features(text: str) -> list[float]:
    negative_words = {"awful", "bad", "boring", "dull", "flat", "poor", "slow", "tedious", "weak"}
    positive_words = {
        "bright",
        "excellent",
        "great",
        "joyful",
        "moving",
        "sharp",
        "strong",
        "vivid",
        "wonderful",
    }
    tokens = text.split()
    return [
        float(sum(token in negative_words for token in tokens)),
        float(sum(token in positive_words for token in tokens)),
        len(tokens) / 10.0,
    ]


def _fake_encoder_factory(_model_name: str) -> FakeTrainableEncoder:
    collapsed_sentiment_projection = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return FakeTrainableEncoder(projection=collapsed_sentiment_projection, fit_example_counts=[])


def test_run_encoder_training_compares_triplet_and_group_objectives() -> None:
    result = run_encoder_training(
        _examples(),
        EncoderTrainingConfig(
            model_name="fake-mini-encoder",
            group_size=2,
            batch_size=4,
            train_steps=30,
            learning_rate=0.05,
            test_size=0.33,
            seed=3,
        ),
        encoder_factory=_fake_encoder_factory,
    )

    assert result.name == "sentence-transformer-training"
    assert result.examples == 12
    assert result.train_examples == 8
    assert result.test_examples == 4
    assert set(result.methods) == {
        "group_finetuned:fake-mini-encoder",
        "hybrid_finetuned:fake-mini-encoder",
        "hybrid_radius_finetuned:fake-mini-encoder",
        "hybrid_xbm_finetuned:fake-mini-encoder",
        "hybrid_xbm_radius_finetuned:fake-mini-encoder",
        "triplet_finetuned:fake-mini-encoder",
    }
    triplet_metrics = result.methods["triplet_finetuned:fake-mini-encoder"]
    group_metrics = result.methods["group_finetuned:fake-mini-encoder"]
    assert triplet_metrics.triplet_loss < triplet_metrics.initial_triplet_loss
    assert group_metrics.group_loss < group_metrics.initial_group_loss
    assert triplet_metrics.initial_probe.macro_f1 <= 1.0
    assert group_metrics.initial_probe.macro_f1 <= 1.0
    assert triplet_metrics.initial_probe.train_macro_f1 <= 1.0
    assert group_metrics.probe.train_macro_f1 <= 1.0
    assert triplet_metrics.initial_retrieval.precision_at_1 <= 1.0
    assert group_metrics.retrieval.map_at_r <= 1.0
    assert triplet_metrics.initial_space.signal_to_noise_ratio >= 0.0
    assert group_metrics.space.drift_to_gap_ratio >= 0.0
    assert triplet_metrics.dimensions == 3
    assert group_metrics.dimensions == 3
    assert triplet_metrics.probe.accuracy >= 0.75
    assert group_metrics.probe.accuracy >= 0.75


def test_run_encoder_training_on_split_uses_explicit_train_and_test_examples() -> None:
    examples = _examples()
    train_examples = examples[:4] + examples[6:10]
    test_examples = examples[4:6] + examples[10:12]

    result = run_encoder_training_on_split(
        train_examples,
        test_examples,
        EncoderTrainingConfig(
            model_name="fake-mini-encoder",
            group_size=2,
            batch_size=4,
            train_steps=10,
            learning_rate=0.05,
            seed=3,
            retrieval_query_limit=2,
            objectives=("triplet",),
        ),
        encoder_factory=_fake_encoder_factory,
    )

    assert result.examples == 12
    assert result.train_examples == 8
    assert result.test_examples == 4
    assert result.triplets == 8
    assert result.group_triplets == 4
    assert set(result.methods) == {"triplet_finetuned:fake-mini-encoder"}
    metrics = result.methods["triplet_finetuned:fake-mini-encoder"]
    assert metrics.initial_probe.confusion_matrix.sum() == 4
    assert metrics.probe.confusion_matrix.sum() == 4
    assert metrics.initial_retrieval.evaluated_queries == 2
    assert metrics.retrieval.evaluated_queries == 2


def test_write_encoder_training_report_persists_json(tmp_path: Path) -> None:
    result = run_encoder_training(
        _examples(),
        EncoderTrainingConfig(
            model_name="fake-mini-encoder",
            group_size=2,
            batch_size=4,
            train_steps=10,
            learning_rate=0.05,
            test_size=0.33,
            seed=3,
        ),
        encoder_factory=_fake_encoder_factory,
    )
    output_path = tmp_path / "encoder_training.json"

    written_path = write_encoder_training_report(result, output_path)

    payload = json.loads(written_path.read_text())
    assert payload["name"] == "sentence-transformer-training"
    assert payload["examples"] == 12
    assert payload["train_examples"] == 8
    assert payload["test_examples"] == 4
    assert "group_finetuned:fake-mini-encoder" in payload["methods"]
    assert "initial_group_loss" in payload["methods"]["group_finetuned:fake-mini-encoder"]
    assert "initial_probe" in payload["methods"]["group_finetuned:fake-mini-encoder"]
    assert "macro_f1" in payload["methods"]["group_finetuned:fake-mini-encoder"]["initial_probe"]
    assert "train_macro_f1" in payload["methods"]["group_finetuned:fake-mini-encoder"]["probe"]
    assert "initial_retrieval" in payload["methods"]["group_finetuned:fake-mini-encoder"]
    assert "map_at_r" in payload["methods"]["group_finetuned:fake-mini-encoder"]["retrieval"]
    assert "initial_space" in payload["methods"]["group_finetuned:fake-mini-encoder"]
    assert (
        "signal_to_noise_ratio" in payload["methods"]["group_finetuned:fake-mini-encoder"]["space"]
    )
