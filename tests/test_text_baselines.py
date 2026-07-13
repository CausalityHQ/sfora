import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sfora.data import TextExample
from sfora.text_baselines import (
    SentenceTransformerBaselineConfig,
    SentenceTransformerModelSuiteConfig,
    TextBaselineConfig,
    run_sentence_transformer_baseline,
    run_sentence_transformer_model_suite,
    run_text_baseline,
    write_text_baseline_report,
)


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


def test_run_text_baseline_scores_tfidf_representation() -> None:
    result = run_text_baseline(
        _examples(),
        TextBaselineConfig(group_size=3, test_size=0.33, seed=3, max_features=128),
    )

    assert result.name == "text-baseline"
    assert set(result.methods) == {"tfidf_word"}
    assert result.config.group_size == 3
    assert result.examples == 12
    assert result.triplets == 12
    assert result.group_triplets == 4
    assert result.methods["tfidf_word"].dimensions > 0
    assert result.methods["tfidf_word"].probe.accuracy >= 0.75
    assert result.methods["tfidf_word"].triplet_loss >= 0.0
    assert result.methods["tfidf_word"].group_loss >= 0.0


def _examples_for_heads() -> list[TextExample]:
    # Nine rows per class so that after the honest train/test split the train side
    # still holds at least two groups per label for group-head training.
    negatives = [
        "bad dull awful film",
        "boring weak slow movie",
        "poor flat tedious story",
        "awful dull weak acting",
        "slow boring bad scenes",
        "flat poor tedious film",
        "grim bleak drab plot",
        "weak grim slow cast",
        "dull bleak poor script",
    ]
    positives = [
        "great vivid excellent film",
        "moving sharp joyful movie",
        "strong bright wonderful story",
        "excellent vivid sharp acting",
        "joyful moving great scenes",
        "bright strong wonderful film",
        "superb lively radiant plot",
        "sharp superb joyful cast",
        "vivid lively strong script",
    ]
    return [
        TextExample(example_id=f"neg-{index}", text=text, label=0)
        for index, text in enumerate(negatives)
    ] + [
        TextExample(example_id=f"pos-{index}", text=text, label=1)
        for index, text in enumerate(positives)
    ]


def test_run_text_baseline_can_train_projection_heads() -> None:
    result = run_text_baseline(
        _examples_for_heads(),
        TextBaselineConfig(
            group_size=3,
            test_size=0.33,
            seed=3,
            max_features=128,
            train_projection_heads=True,
            projection_steps=30,
            projection_learning_rate=0.05,
        ),
    )

    assert set(result.methods) == {
        "tfidf_group_projection",
        "tfidf_triplet_projection",
        "tfidf_word",
    }
    assert (
        result.methods["tfidf_triplet_projection"].triplet_loss
        < result.methods["tfidf_word"].triplet_loss
    )
    assert (
        result.methods["tfidf_group_projection"].group_loss
        < result.methods["tfidf_word"].group_loss
    )


def test_write_text_baseline_report_persists_json(tmp_path: Path) -> None:
    result = run_text_baseline(_examples(), TextBaselineConfig(group_size=3, seed=5))
    output_path = tmp_path / "baseline.json"

    written_path = write_text_baseline_report(result, output_path)

    payload = json.loads(written_path.read_text())
    assert payload["name"] == "text-baseline"
    assert payload["examples"] == 12
    assert payload["methods"]["tfidf_word"]["dimensions"] > 0
    assert "confusion_matrix" in payload["methods"]["tfidf_word"]["probe"]
    assert "retrieval" in payload["methods"]["tfidf_word"]
    assert "map_at_r" in payload["methods"]["tfidf_word"]["retrieval"]
    assert "space" in payload["methods"]["tfidf_word"]
    assert "signal_to_noise_ratio" in payload["methods"]["tfidf_word"]["space"]


class FakeSentenceEncoder:
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> NDArray[np.float64]:
        assert batch_size == 4
        assert normalize_embeddings is True
        assert show_progress_bar is False
        embeddings = []
        for text in texts:
            polarity = (
                1.0 if any(word in text for word in ["great", "excellent", "joyful"]) else -1.0
            )
            length_feature = len(text.split()) / 10.0
            embeddings.append([polarity, length_feature])
        return np.array(embeddings, dtype=np.float64)


def test_run_sentence_transformer_baseline_uses_injected_encoder() -> None:
    result = run_sentence_transformer_baseline(
        _examples(),
        SentenceTransformerBaselineConfig(
            model_name="fake-mini-encoder",
            group_size=3,
            batch_size=4,
            test_size=0.33,
            seed=3,
        ),
        encoder_factory=lambda _model_name: FakeSentenceEncoder(),
    )

    assert result.name == "sentence-transformer-baseline"
    assert set(result.methods) == {"sentence_transformer:fake-mini-encoder"}
    assert result.config.model_dump()["model_name"] == "fake-mini-encoder"
    assert result.methods["sentence_transformer:fake-mini-encoder"].dimensions == 2
    assert result.methods["sentence_transformer:fake-mini-encoder"].probe.accuracy >= 0.75
    assert result.methods["sentence_transformer:fake-mini-encoder"].retrieval.precision_at_1 >= 0.0
    assert result.methods["sentence_transformer:fake-mini-encoder"].retrieval.map_at_r >= 0.0
    assert (
        result.methods["sentence_transformer:fake-mini-encoder"].space.signal_to_noise_ratio >= 0.0
    )


def test_run_sentence_transformer_model_suite_scores_multiple_models() -> None:
    loaded_models: list[str] = []

    def fake_factory(model_name: str) -> FakeSentenceEncoder:
        loaded_models.append(model_name)
        return FakeSentenceEncoder()

    result = run_sentence_transformer_model_suite(
        _examples(),
        SentenceTransformerModelSuiteConfig(
            model_names=("fake-mini-a", "fake-mini-b"),
            group_size=3,
            batch_size=4,
            test_size=0.33,
            seed=3,
        ),
        encoder_factory=fake_factory,
    )

    assert result.name == "sentence-transformer-model-suite"
    assert loaded_models == ["fake-mini-a", "fake-mini-b"]
    assert set(result.methods) == {
        "sentence_transformer:fake-mini-a",
        "sentence_transformer:fake-mini-b",
    }
    assert result.config.model_dump()["model_names"] == ("fake-mini-a", "fake-mini-b")
    assert result.methods["sentence_transformer:fake-mini-a"].retrieval.map_at_r >= 0.0
