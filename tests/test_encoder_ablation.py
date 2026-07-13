import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sfora.data import TextExample
from sfora.encoder_ablation import (
    EncoderAblationConfig,
    run_encoder_ablation,
    write_encoder_ablation_report,
)
from sfora.encoder_training import EncoderObjective, EncoderTrainingConfig
from sfora.training import ProjectionHeadTrainingConfig, train_projection_head


def _examples() -> list[TextExample]:
    negatives = [f"negative dull review {index}" for index in range(6)]
    positives = [f"positive bright review {index}" for index in range(6)]
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
    tokens = text.split()
    return [
        float("negative" in tokens),
        float("positive" in tokens),
        len(tokens) / 10.0,
    ]


def _fake_encoder_factory(_model_name: str) -> FakeTrainableEncoder:
    return FakeTrainableEncoder(
        projection=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )


def test_run_encoder_ablation_ranks_training_strength_trials() -> None:
    result = run_encoder_ablation(
        _examples(),
        EncoderAblationConfig(
            model_name="fake-mini-encoder",
            objectives=("triplet", "group"),
            train_steps=(5, 10),
            learning_rates=(0.05,),
            group_sizes=(1, 2),
            batch_size=4,
            test_size=0.33,
            seed=3,
        ),
        encoder_factory=_fake_encoder_factory,
    )

    assert result.name == "sentence-transformer-ablation"
    assert len(result.trials) == 8
    assert [trial.rank for trial in result.trials] == list(range(1, 9))
    assert {trial.objective for trial in result.trials} == {"triplet", "group"}
    assert {trial.group_size for trial in result.trials} == {1, 2}
    assert result.best_trial.macro_f1 == max(trial.macro_f1 for trial in result.trials)
    assert result.best_trial.f1_generalization_gap >= 0.0
    assert result.best_trial.map_at_r_delta is not None


def test_write_encoder_ablation_report_persists_json(tmp_path: Path) -> None:
    result = run_encoder_ablation(
        _examples(),
        EncoderAblationConfig(
            model_name="fake-mini-encoder",
            objectives=("group",),
            train_steps=(5,),
            learning_rates=(0.05,),
            group_sizes=(1, 2),
            batch_size=4,
            test_size=0.33,
            seed=3,
        ),
        encoder_factory=_fake_encoder_factory,
    )
    output_path = tmp_path / "encoder_ablation.json"

    written_path = write_encoder_ablation_report(result, output_path)

    payload = json.loads(written_path.read_text())
    assert payload["name"] == "sentence-transformer-ablation"
    assert payload["best_trial"]["objective"] == "group"
    assert payload["best_trial"]["group_size"] in {1, 2}
    assert payload["trials"][0]["train_steps"] == 5
    assert "group_size" in payload["trials"][0]
    assert "train_macro_f1_delta" in payload["trials"][0]
