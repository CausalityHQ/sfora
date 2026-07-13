import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from sfora.cli import _default_report_artifacts, app
from sfora.data import ImageExample, TextExample
from sfora.encoder_training import EncoderObjective, EncoderTrainingConfig
from sfora.training import ProjectionHeadTrainingConfig, train_projection_head


def test_smoke_command_runs_as_subcommand() -> None:
    result = CliRunner().invoke(app, ["smoke"])

    assert result.exit_code == 0
    assert "triplet_loss" in result.output
    assert "linear_probe_accuracy" in result.output


def test_synthetic_command_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "synthetic.json"

    result = CliRunner().invoke(
        app,
        [
            "synthetic",
            "--output",
            str(output_path),
            "--samples-per-class",
            "8",
            "--dimensions",
            "4",
            "--group-size",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "synthetic-smoke" in result.output


def test_synthetic_train_command_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "synthetic_trainable.json"

    result = CliRunner().invoke(
        app,
        [
            "synthetic-train",
            "--output",
            str(output_path),
            "--samples-per-class",
            "8",
            "--dimensions",
            "4",
            "--group-size",
            "2",
            "--train-steps",
            "20",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "synthetic-trainable" in result.output
    assert "group_trained" in output_path.read_text()


def test_synthetic_ablation_command_writes_report(tmp_path: Path) -> None:
    output_path = tmp_path / "synthetic_ablation.json"

    result = CliRunner().invoke(
        app,
        [
            "synthetic-ablation",
            "--output",
            str(output_path),
            "--samples-per-class",
            "8",
            "--dimensions",
            "4",
            "--group-sizes",
            "2,4",
            "--hard-weights",
            "0.0",
            "--spread-weights",
            "0.0,0.2",
            "--train-steps",
            "10",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    text = output_path.read_text()
    assert "synthetic-ablation" in result.output
    assert "best_trial" in text


def test_report_data_command_writes_astro_site_data(tmp_path: Path) -> None:
    artifact_path = tmp_path / "image.json"
    artifact_path.write_text(
        json.dumps(
            {
                "name": "image-retrieval-benchmark",
                "dataset_name": "sop",
                "examples": 48,
                "train_examples": 24,
                "test_examples": 24,
                "config": {
                    "model_names": ["fake"],
                    "objectives": ["supcon", "group_supcon", "group_supcon_xbm_radius"],
                    "projection_train_limit": 12,
                    "train_steps": 80,
                    "group_size": 4,
                    "validation_query_limit": 8,
                },
                "methods": {
                    "frozen:fake": {
                        "model_name": "fake",
                        "objective": "frozen",
                        "display_name": "Frozen",
                        "recall_at_1": 0.4,
                        "map_at_r": 0.2,
                        "recall_at_1_delta": 0.0,
                        "map_at_r_delta": 0.0,
                        "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                    },
                    "supcon_projection:fake": {
                        "model_name": "fake",
                        "objective": "supcon",
                        "display_name": "Supervised Contrastive",
                        "recall_at_1": 0.5,
                        "map_at_r": 0.3,
                        "recall_at_1_delta": 0.1,
                        "map_at_r_delta": 0.1,
                        "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                    },
                    "group_supcon_projection:fake": {
                        "model_name": "fake",
                        "objective": "group_supcon",
                        "display_name": "Group SupCon",
                        "recall_at_1": 0.55,
                        "map_at_r": 0.34,
                        "recall_at_1_delta": 0.15,
                        "map_at_r_delta": 0.14,
                        "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                    },
                    "group_supcon_xbm_radius_projection:fake": {
                        "model_name": "fake",
                        "objective": "group_supcon_xbm_radius",
                        "display_name": "Group SupCon + XBM + Radius",
                        "recall_at_1": 0.6,
                        "map_at_r": 0.36,
                        "recall_at_1_delta": 0.2,
                        "map_at_r_delta": 0.16,
                        "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "report-data.json"

    result = CliRunner().invoke(
        app,
        [
            "report-data",
            "--artifact",
            str(artifact_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["claim"]["headline"].startswith(
        "Power study: Group SupCon + XBM + Radius is best"
    )
    assert "frozen-backbone image datasets" in payload["claim"]["headline"]
    assert (
        "same-architecture ResNet-50/512 paper-protocol claim remains pending"
        in payload["claim"]["detail"]
    )
    assert payload["formula"]["text"] == "(ours MAP@R - previous MAP@R) / previous MAP@R"
    assert payload["mainResults"][0]["dataset"] == "Stanford Online Products"
    assert payload["mainResults"][0]["prior"]["methodName"] == "Supervised Contrastive (SupCon)"
    assert payload["mainResults"][0]["priorResultGain"] == pytest.approx(0.2)
    assert payload["supconComparisons"][0]["dataset"] == "Stanford Online Products"
    assert payload["supconComparisons"][0]["groupAdvantage"] == pytest.approx(0.04)
    assert payload["supconComparisons"][0]["fullAdvantage"] == pytest.approx(0.06)
    assert payload["protocol"]["datasets"][0]["dataset"] == "Stanford Online Products"
    assert payload["protocol"]["datasets"][0]["trainExamples"] == 24
    assert payload["protocol"]["datasets"][0]["testExamples"] == 24
    assert payload["protocol"]["datasets"][0]["evaluatedQueries"] == 24
    assert payload["protocol"]["backbones"] == ["fake"]
    assert payload["protocol"]["objectiveCount"] == 3
    assert payload["protocol"]["trainSteps"] == 80
    assert payload["protocol"]["projectionTrainLimit"] == 12
    assert payload["findings"]["datasetCount"] == 1
    assert payload["findings"]["priorWins"] == 1
    assert payload["findings"]["supconWins"] == 1
    assert payload["findings"]["rawGroupSupconWins"] == 1
    assert payload["findings"]["rawGroupSupconRegressions"] == 0
    assert payload["findings"]["bestResultGainDataset"] == "Stanford Online Products"
    assert payload["findings"]["bestResultGain"] == pytest.approx(0.2)
    assert any(
        method["name"] == "Group SupCon + XBM + Radius" for method in payload["methodCatalog"]
    )
    assert payload["publishedReferences"]["controlledProtocol"].startswith(
        "Our power experiment measures"
    )
    assert payload["publishedReferences"]["bestRows"]
    assert len(payload["publishedReferences"]["latestRows"]) == 12
    assert payload["publishedReferences"]["historicalRecallNote"].startswith(
        "The CUB R@1 55.1 value"
    )
    assert payload["publishedReferences"]["noisyLabelBaselineNote"].startswith(
        "PFML CVPR 2025 also reports"
    )
    noisy_triplet_rows = [
        row
        for row in payload["publishedReferences"]["noisyLabelBaselineRows"]
        if row["method"] == "Triplet" and row["dataset"] == "CUB"
    ]
    assert noisy_triplet_rows
    assert noisy_triplet_rows[0]["recallAt1Percent"] == pytest.approx(55.1)
    assert noisy_triplet_rows[0]["backbone"] == "ResNet-50 / 512-dim"
    assert noisy_triplet_rows[0]["comparisonScope"] == "noisy_label_resnet_context"
    assert "20% label-noise" in noisy_triplet_rows[0]["note"]
    damlrrm_rows = [
        row
        for row in payload["publishedReferences"]["historicalRecallRows"]
        if row["method"] == "DAMLRRM"
    ]
    assert damlrrm_rows
    assert damlrrm_rows[0]["dataset"] == "CUB"
    assert damlrrm_rows[0]["recallAt1Percent"] == pytest.approx(55.1)
    assert damlrrm_rows[0]["comparisonScope"] == "architecture_context"
    assert "not a plain Triplet baseline" in damlrrm_rows[0]["note"]
    assert not any(
        row["method"] == "DAMLRRM" for row in payload["publishedReferences"]["primaryMapRows"]
    )
    assert any(
        row["method"] == "CouCE" and row["dataset"] == "CUB"
        for row in payload["publishedReferences"]["latestRows"]
    )
    couce_rows = [
        row for row in payload["publishedReferences"]["latestRows"] if row["method"] == "CouCE"
    ]
    assert couce_rows
    assert all(
        row["comparisonScope"] == "same_backbone_training_module_context" for row in couce_rows
    )
    assert all("CouCE training modules" in row["backbone"] for row in couce_rows)
    cub_map_rows = [
        row
        for row in payload["publishedReferences"]["primaryMapRows"]
        if row["method"] == "Cont. + XBM" and row["dataset"] == "CUB"
    ]
    assert cub_map_rows
    assert cub_map_rows[0]["comparisonScope"] == "non_resnet_map_context"
    assert "BN-Inception" in cub_map_rows[0]["backbone"]
    assert not any(
        row["method"] == "PFML" for row in payload["publishedReferences"]["primaryMapRows"]
    )
    assert not any(
        row["method"] == "CouCE" for row in payload["publishedReferences"]["primaryMapRows"]
    )
    assert any(
        row["method"] == "SGSL" and row["dataset"] == "CUB" and row["mapAtRPercent"] is None
        for row in payload["publishedReferences"]["latestRows"]
    )
    assert any(
        row["method"] == "PFML"
        and row["dataset"] == "CUB"
        and row["source"] == "PFML CVPR 2025"
        and row["mapAtRPercent"] is None
        and row["recallAt1Percent"] == pytest.approx(73.4)
        for row in payload["publishedReferences"]["latestRows"]
    )
    assert any(
        row["method"] == "PFML"
        and row["dataset"] == "CUB"
        and row["comparisonScope"] == "same_architecture_recall_target"
        and row["recallAt1Percent"] == pytest.approx(73.4)
        for row in payload["publishedReferences"]["primaryResnetRecallRows"]
    )
    assert any(
        row["method"] == "PFML"
        and row["dataset"] == "Cars196"
        and row["comparisonScope"] == "same_architecture_recall_target"
        and row["recallAt1Percent"] == pytest.approx(92.7)
        for row in payload["publishedReferences"]["primaryResnetRecallRows"]
    )
    assert not any(
        row["method"] == "CouCE"
        for row in payload["publishedReferences"]["primaryResnetRecallRows"]
    )
    assert any(
        row["source"] == "HPL WACV 2022" and row["dataset"] == "SOP"
        for row in payload["publishedReferences"]["rows"]
    )
    assert payload["endToEndRows"] == []
    assert payload["imageRows"]


def test_imdb_mine_command_writes_triplet_summary(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "imdb_mining.json"

    def fake_loader(*, split: str, limit_per_class: int, seed: int) -> list[TextExample]:
        assert split == "train"
        assert limit_per_class == 4
        assert seed == 5
        return [
            TextExample(example_id=f"neg-{index}", text=f"negative {index}", label=0)
            for index in range(4)
        ] + [
            TextExample(example_id=f"pos-{index}", text=f"positive {index}", label=1)
            for index in range(4)
        ]

    monkeypatch.setattr("sfora.cli.load_imdb_examples", fake_loader)

    result = CliRunner().invoke(
        app,
        [
            "imdb-mine",
            "--output",
            str(output_path),
            "--limit-per-class",
            "4",
            "--group-size",
            "2",
            "--seed",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "group_triplets" in output_path.read_text()


def test_imdb_mine_command_reports_missing_research_extra(monkeypatch: Any) -> None:
    def missing_loader(*, split: str, limit_per_class: int, seed: int) -> list[TextExample]:
        raise RuntimeError("Install the research extra")

    monkeypatch.setattr("sfora.cli.load_imdb_examples", missing_loader)

    result = CliRunner().invoke(app, ["imdb-mine"])

    assert result.exit_code == 1
    assert "Install the research extra" in result.output
    assert "Traceback" not in result.output


def test_imdb_baseline_command_writes_text_baseline_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "imdb_baseline.json"

    def fake_loader(*, split: str, limit_per_class: int, seed: int) -> list[TextExample]:
        assert split == "train"
        assert limit_per_class == 6
        assert seed == 9
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

    monkeypatch.setattr("sfora.cli.load_imdb_examples", fake_loader)

    result = CliRunner().invoke(
        app,
        [
            "imdb-baseline",
            "--output",
            str(output_path),
            "--limit-per-class",
            "6",
            "--group-size",
            "3",
            "--seed",
            "9",
            "--train-projection-heads",
            "--projection-steps",
            "30",
            "--projection-learning-rate",
            "0.05",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    text = output_path.read_text()
    assert "tfidf_group_projection" in text
    assert "tfidf_triplet_projection" in text
    assert "tfidf_word" in text
    assert "text-baseline" in result.output


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
        rows = []
        for text in texts:
            polarity = 1.0 if "positive" in text else -1.0
            rows.append([polarity, len(text) / 100.0])
        return np.array(rows, dtype=np.float64)


def test_imdb_encoder_baseline_command_writes_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "imdb_encoder_baseline.json"

    def fake_loader(*, split: str, limit_per_class: int, seed: int) -> list[TextExample]:
        assert split == "train"
        assert limit_per_class == 6
        assert seed == 4
        return [
            TextExample(example_id=f"neg-{index}", text=f"negative review {index}", label=0)
            for index in range(6)
        ] + [
            TextExample(example_id=f"pos-{index}", text=f"positive review {index}", label=1)
            for index in range(6)
        ]

    monkeypatch.setattr("sfora.cli.load_imdb_examples", fake_loader)
    monkeypatch.setattr(
        "sfora.text_baselines._load_sentence_transformer",
        lambda _model_name: FakeSentenceEncoder(),
    )

    result = CliRunner().invoke(
        app,
        [
            "imdb-encoder-baseline",
            "--output",
            str(output_path),
            "--model-name",
            "fake-mini-encoder",
            "--limit-per-class",
            "6",
            "--group-size",
            "3",
            "--batch-size",
            "4",
            "--seed",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "sentence_transformer:fake-mini-encoder" in output_path.read_text()
    assert "sentence-transformer-baseline" in result.output


def test_image_benchmark_command_writes_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_benchmark.json"

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        assert dataset_name == "cub"
        assert limit_per_class == 4
        assert min_per_class == 2
        assert max_classes == 3
        assert seed == 11
        labels = (0, 1, 2) if split == "train" else (100, 101, 102)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(4)
        ]

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr(
        "sfora.image_benchmark._load_transformers_image_encoder",
        lambda model_name: FakeCliImageEncoder(model_name),
    )

    result = CliRunner().invoke(
        app,
        [
            "image-benchmark",
            "--output",
            str(output_path),
            "--dataset-name",
            "cub",
            "--model-names",
            "fake-dino,fake-clip",
            "--objectives",
            "triplet,hybrid_xbm_radius",
            "--limit-per-class",
            "4",
            "--max-classes",
            "3",
            "--min-per-class",
            "2",
            "--group-size",
            "2",
            "--batch-size",
            "8",
            "--train-steps",
            "3",
            "--triplet-weight",
            "0.8",
            "--group-weight",
            "0.35",
            "--hard-weight",
            "0.45",
            "--spread-weight",
            "0.02",
            "--output-dimensions",
            "2",
            "--projection-train-limit",
            "8",
            "--retrieval-query-limit",
            "6",
            "--xbm-memory-size",
            "32",
            "--xbm-weight",
            "0.15",
            "--radius-weight",
            "0.07",
            "--radius-target",
            "0.2",
            "--variance-weight",
            "0.03",
            "--embedding-cache-dir",
            str(tmp_path / "cache"),
            "--shuffle-groups-each-step",
            "--seed",
            "11",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["name"] == "image-retrieval-benchmark"
    assert payload["dataset_name"] == "cub"
    assert payload["config"]["limit_per_class"] == 4
    assert payload["config"]["min_per_class"] == 2
    assert payload["config"]["max_classes"] == 3
    assert payload["config"]["output_dimensions"] == 2
    assert payload["config"]["projection_train_limit"] == 8
    assert payload["config"]["triplet_weight"] == 0.8
    assert payload["config"]["group_weight"] == 0.35
    assert payload["config"]["hard_weight"] == 0.45
    assert payload["config"]["spread_weight"] == 0.02
    assert payload["config"]["xbm_memory_size"] == 32
    assert payload["config"]["xbm_weight"] == 0.15
    assert payload["config"]["radius_weight"] == 0.07
    assert payload["config"]["radius_target"] == 0.2
    assert payload["config"]["variance_weight"] == 0.03
    assert payload["config"]["embedding_cache_dir"] == str(tmp_path / "cache")
    assert payload["config"]["shuffle_groups_each_step"] is True
    assert payload["projection_train_examples"] == 8
    assert "frozen:fake-dino" in payload["methods"]
    assert "hybrid_xbm_radius_projection:fake-clip" in payload["methods"]
    assert "Hybrid + XBM + Radius" in output_path.read_text()


def test_image_benchmark_defaults_to_two_groups_per_class_for_group_objectives(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_benchmark.json"
    seen_min_per_class: list[int | None] = []

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        assert dataset_name == "sop"
        assert limit_per_class is None
        assert max_classes is None
        assert seed == 7
        seen_min_per_class.append(min_per_class)
        labels = (0, 1, 2) if split == "train" else (100, 101, 102)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(8)
        ]

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr(
        "sfora.image_benchmark._load_transformers_image_encoder",
        lambda model_name: FakeCliImageEncoder(model_name),
    )

    result = CliRunner().invoke(
        app,
        [
            "image-benchmark",
            "--output",
            str(output_path),
            "--dataset-name",
            "sop",
            "--model-names",
            "fake-dino",
            "--objectives",
            "hybrid_xbm",
            "--group-size",
            "4",
            "--batch-size",
            "8",
            "--train-steps",
            "2",
            "--projection-train-limit",
            "16",
            "--retrieval-query-limit",
            "6",
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert seen_min_per_class == [8, 8]
    payload = json.loads(output_path.read_text())
    assert payload["config"]["min_per_class"] == 8
    assert payload["config"]["limit_per_class"] is None
    assert payload["config"]["max_classes"] is None


def test_image_end_to_end_command_passes_triplet_and_backbone_lr_knobs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_end_to_end.json"
    captured: dict[str, Any] = {}

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        labels = (0, 1) if split == "train" else (100, 101)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(4)
        ]

    def fake_run(
        *,
        train_examples: list[ImageExample],
        test_examples: list[ImageExample],
        config: Any,
        progress_callback: Any,
    ) -> Any:
        captured["config"] = config
        return SimpleNamespace(
            name="image-end-to-end-benchmark",
            dataset_name=config.dataset_name,
            protocol=config.protocol,
            train_examples=len(train_examples),
            test_examples=len(test_examples),
            methods={},
        )

    def fake_write(result: Any, output: Path) -> Path:
        config = captured["config"]
        output.write_text(
            json.dumps(
                {
                    "objectives": list(config.objectives),
                    "backbone_learning_rate": config.backbone_learning_rate,
                    "triplet_margin": config.triplet_margin,
                    "train_augmentation": config.train_augmentation,
                    "freeze_batch_norm": config.freeze_batch_norm,
                    "checkpoint_selection_interval": config.checkpoint_selection_interval,
                    "checkpoint_selection_metric": config.checkpoint_selection_metric,
                    "checkpoint_selection_query_limit": config.checkpoint_selection_query_limit,
                    "checkpoint_selection_validation_fraction": (
                        config.checkpoint_selection_validation_fraction
                    ),
                    "teacher_similarity_weight": config.teacher_similarity_weight,
                    "label_noise_fraction": config.label_noise_fraction,
                    "point_weight": config.point_weight,
                    "group_weight": config.group_weight,
                    "proxy_weight": config.proxy_weight,
                    "proxy_count_per_class": config.proxy_count_per_class,
                    "proxy_learning_rate_multiplier": config.proxy_learning_rate_multiplier,
                    "potential_weight": config.potential_weight,
                    "potential_delta": config.potential_delta,
                    "potential_alpha": config.potential_alpha,
                }
            ),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr("sfora.cli.run_image_end_to_end_benchmark", fake_run)
    monkeypatch.setattr("sfora.cli.write_image_end_to_end_report", fake_write)

    result = CliRunner().invoke(
        app,
        [
            "image-end-to-end",
            "--output",
            str(output_path),
            "--dataset-name",
            "cub",
            "--objectives",
            "triplet,batch_hard_triplet,group_potential,group_potential_xbm",
            "--train-steps",
            "1",
            "--group-size",
            "2",
            "--backbone-learning-rate",
            "0.00003",
            "--triplet-margin",
            "0.35",
            "--train-augmentation",
            "center_crop",
            "--update-batch-norm",
            "--checkpoint-selection-interval",
            "50",
            "--checkpoint-selection-metric",
            "recall_at_1",
            "--checkpoint-selection-query-limit",
            "128",
            "--checkpoint-selection-validation-fraction",
            "0.2",
            "--teacher-similarity-weight",
            "1.5",
            "--label-noise-fraction",
            "0.2",
            "--point-weight",
            "1.0",
            "--group-weight",
            "0.25",
            "--proxy-weight",
            "0.5",
            "--proxy-count-per-class",
            "3",
            "--proxy-learning-rate-multiplier",
            "50",
            "--potential-weight",
            "0.75",
            "--potential-delta",
            "0.2",
            "--potential-alpha",
            "4",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["objectives"] == [
        "triplet",
        "batch_hard_triplet",
        "group_potential",
        "group_potential_xbm",
    ]
    assert payload["backbone_learning_rate"] == pytest.approx(3e-5)
    assert payload["triplet_margin"] == pytest.approx(0.35)
    assert payload["train_augmentation"] == "center_crop"
    assert payload["freeze_batch_norm"] is False
    assert payload["checkpoint_selection_interval"] == 50
    assert payload["checkpoint_selection_metric"] == "recall_at_1"
    assert payload["checkpoint_selection_query_limit"] == 128
    assert payload["checkpoint_selection_validation_fraction"] == pytest.approx(0.2)
    assert payload["teacher_similarity_weight"] == pytest.approx(1.5)
    assert payload["label_noise_fraction"] == pytest.approx(0.2)
    assert payload["point_weight"] == pytest.approx(1.0)
    assert payload["group_weight"] == pytest.approx(0.25)
    assert payload["proxy_weight"] == pytest.approx(0.5)
    assert payload["proxy_count_per_class"] == 3
    assert payload["proxy_learning_rate_multiplier"] == pytest.approx(50.0)
    assert payload["potential_weight"] == pytest.approx(0.75)
    assert payload["potential_delta"] == pytest.approx(0.2)
    assert payload["potential_alpha"] == pytest.approx(4.0)


def test_image_end_to_end_command_accepts_protocol_repair_knobs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_end_to_end_protocol_repair.json"
    captured: dict[str, Any] = {}

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        labels = (0, 1) if split == "train" else (100, 101)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(4)
        ]

    def fake_run(
        *,
        train_examples: list[ImageExample],
        test_examples: list[ImageExample],
        config: Any,
        progress_callback: Any,
    ) -> Any:
        captured["config"] = config
        return SimpleNamespace(
            name="image-end-to-end-benchmark",
            dataset_name=config.dataset_name,
            protocol=config.protocol,
            train_examples=len(train_examples),
            test_examples=len(test_examples),
            methods={},
        )

    def fake_write(result: Any, output: Path) -> Path:
        config = captured["config"]
        output.write_text(
            json.dumps(
                {
                    "protocol": config.protocol,
                    "objectives": list(config.objectives),
                    "optimizer": config.optimizer,
                    "weight_decay": config.weight_decay,
                    "warmup_epochs": config.warmup_epochs,
                    "lr_schedule": config.lr_schedule,
                    "lr_step_epochs": config.lr_step_epochs,
                    "lr_gamma": config.lr_gamma,
                    "samples_per_class": config.samples_per_class,
                    "pretrained_weights": config.pretrained_weights,
                    "head_pooling": config.head_pooling,
                    "embedding_head_init": config.embedding_head_init,
                    "xbm_start_step": config.xbm_start_step,
                    "proxy_anchor_alpha": config.proxy_anchor_alpha,
                    "proxy_anchor_delta": config.proxy_anchor_delta,
                    "hist_tau": config.hist_tau,
                    "hist_alpha": config.hist_alpha,
                }
            ),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr("sfora.cli.run_image_end_to_end_benchmark", fake_run)
    monkeypatch.setattr("sfora.cli.write_image_end_to_end_report", fake_write)

    result = CliRunner().invoke(
        app,
        [
            "image-end-to-end",
            "--output",
            str(output_path),
            "--dataset-name",
            "cub",
            "--protocol",
            "pfml-resnet50-512",
            "--objectives",
            "proxy_anchor,pfml",
            "--train-steps",
            "1",
            "--optimizer",
            "adamw",
            "--weight-decay",
            "0.00005",
            "--warmup-epochs",
            "3",
            "--lr-schedule",
            "step",
            "--lr-step-epochs",
            "7",
            "--lr-gamma",
            "0.25",
            "--samples-per-class",
            "5",
            "--pretrained-weights",
            "v1",
            "--head-pooling",
            "avg_max",
            "--embedding-head-init",
            "kaiming_normal",
            "--xbm-start-step",
            "9",
            "--proxy-anchor-alpha",
            "16",
            "--proxy-anchor-delta",
            "0.2",
            "--hist-tau",
            "24",
            "--hist-alpha",
            "1.15",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["protocol"] == "pfml-resnet50-512"
    assert payload["objectives"] == ["proxy_anchor", "pfml"]
    assert payload["optimizer"] == "adamw"
    assert payload["weight_decay"] == pytest.approx(5e-5)
    assert payload["warmup_epochs"] == 3
    assert payload["lr_schedule"] == "step"
    assert payload["lr_step_epochs"] == 7
    assert payload["lr_gamma"] == pytest.approx(0.25)
    assert payload["samples_per_class"] == 5
    assert payload["pretrained_weights"] == "v1"
    assert payload["head_pooling"] == "avg_max"
    assert payload["embedding_head_init"] == "kaiming_normal"
    assert payload["xbm_start_step"] == 9
    assert payload["proxy_anchor_alpha"] == pytest.approx(16.0)
    assert payload["proxy_anchor_delta"] == pytest.approx(0.2)
    assert payload["hist_tau"] == pytest.approx(24.0)
    assert payload["hist_alpha"] == pytest.approx(1.15)


def test_image_end_to_end_command_accepts_gsi_knobs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_end_to_end_gsi.json"
    captured: dict[str, Any] = {}

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        labels = (0, 1) if split == "train" else (100, 101)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(4)
        ]

    def fake_run(
        *,
        train_examples: list[ImageExample],
        test_examples: list[ImageExample],
        config: Any,
        progress_callback: Any,
    ) -> Any:
        captured["config"] = config
        return SimpleNamespace(
            name="image-end-to-end-benchmark",
            dataset_name=config.dataset_name,
            protocol=config.protocol,
            train_examples=len(train_examples),
            test_examples=len(test_examples),
            methods={},
        )

    def fake_write(result: Any, output: Path) -> Path:
        config = captured["config"]
        output.write_text(
            json.dumps(
                {
                    "objectives": list(config.objectives),
                    "gsi_weight": config.gsi_weight,
                    "gsi_floor": config.gsi_floor,
                    "gsi_top_k": config.gsi_top_k,
                    "gsi_min_group_size": config.gsi_min_group_size,
                    "gsi_variance_floor": config.gsi_variance_floor,
                    "gsi_start_epoch": config.gsi_start_epoch,
                    "gsi_axis_mode": config.gsi_axis_mode,
                }
            ),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr("sfora.cli.run_image_end_to_end_benchmark", fake_run)
    monkeypatch.setattr("sfora.cli.write_image_end_to_end_report", fake_write)

    result = CliRunner().invoke(
        app,
        [
            "image-end-to-end",
            "--output",
            str(output_path),
            "--dataset-name",
            "cub",
            "--protocol",
            "proxy-anchor-resnet50-512",
            "--objectives",
            "proxy_anchor_gsi,pfml_gsi",
            "--train-steps",
            "1",
            "--gsi-weight",
            "0.7",
            "--gsi-floor",
            "0.05",
            "--gsi-top-k",
            "5",
            "--gsi-min-group-size",
            "3",
            "--gsi-variance-floor",
            "0.001",
            "--gsi-start-epoch",
            "2",
            "--gsi-axis-mode",
            "random",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["objectives"] == ["proxy_anchor_gsi", "pfml_gsi"]
    assert payload["gsi_weight"] == pytest.approx(0.7)
    assert payload["gsi_floor"] == pytest.approx(0.05)
    assert payload["gsi_top_k"] == 5
    assert payload["gsi_min_group_size"] == 3
    assert payload["gsi_variance_floor"] == pytest.approx(0.001)
    assert payload["gsi_start_epoch"] == 2
    assert payload["gsi_axis_mode"] == "random"


def test_image_end_to_end_command_accepts_bgsi_knobs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_end_to_end_bgsi.json"
    captured: dict[str, Any] = {}

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        labels = (0, 1) if split == "train" else (100, 101)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(4)
        ]

    def fake_run(
        *,
        train_examples: list[ImageExample],
        test_examples: list[ImageExample],
        config: Any,
        progress_callback: Any,
    ) -> Any:
        captured["config"] = config
        return SimpleNamespace(
            name="image-end-to-end-benchmark",
            dataset_name=config.dataset_name,
            protocol=config.protocol,
            train_examples=len(train_examples),
            test_examples=len(test_examples),
            methods={},
        )

    def fake_write(result: Any, output: Path) -> Path:
        config = captured["config"]
        output.write_text(
            json.dumps(
                {
                    "objectives": list(config.objectives),
                    "bgsi_weight": config.bgsi_weight,
                    "bgsi_floor": config.bgsi_floor,
                    "bgsi_top_k": config.bgsi_top_k,
                    "bgsi_temperature": config.bgsi_temperature,
                    "bgsi_start_epoch": config.bgsi_start_epoch,
                    "bgsi_min_group_size": config.bgsi_min_group_size,
                    "bgsi_variance_floor": config.bgsi_variance_floor,
                    "bgsi_axis_mode": config.bgsi_axis_mode,
                    "bgsi_ema_momentum": config.bgsi_ema_momentum,
                    "bgsi_min_axis_observations": config.bgsi_min_axis_observations,
                    "bgsi_use_axis_agreement_gate": config.bgsi_use_axis_agreement_gate,
                    "bgsi_axis_agreement": config.bgsi_axis_agreement,
                }
            ),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr("sfora.cli.run_image_end_to_end_benchmark", fake_run)
    monkeypatch.setattr("sfora.cli.write_image_end_to_end_report", fake_write)

    result = CliRunner().invoke(
        app,
        [
            "image-end-to-end",
            "--output",
            str(output_path),
            "--dataset-name",
            "cub",
            "--protocol",
            "proxy-anchor-resnet50-512",
            "--objectives",
            "proxy_anchor_bgsi",
            "--train-steps",
            "1",
            "--bgsi-weight",
            "1.0",
            "--bgsi-floor",
            "0.005",
            "--bgsi-top-k",
            "2",
            "--bgsi-temperature",
            "0.2",
            "--bgsi-start-epoch",
            "0",
            "--bgsi-min-group-size",
            "3",
            "--bgsi-variance-floor",
            "0.0002",
            "--bgsi-axis-mode",
            "ema_boundary",
            "--bgsi-ema-momentum",
            "0.8",
            "--bgsi-min-axis-observations",
            "3",
            "--no-bgsi-use-axis-agreement-gate",
            "--bgsi-axis-agreement",
            "0.25",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text())
    assert payload["objectives"] == ["proxy_anchor_bgsi"]
    assert payload["bgsi_weight"] == pytest.approx(1.0)
    assert payload["bgsi_floor"] == pytest.approx(0.005)
    assert payload["bgsi_top_k"] == 2
    assert payload["bgsi_temperature"] == pytest.approx(0.2)
    assert payload["bgsi_start_epoch"] == 0
    assert payload["bgsi_min_group_size"] == 3
    assert payload["bgsi_variance_floor"] == pytest.approx(0.0002)
    assert payload["bgsi_axis_mode"] == "ema_boundary"
    assert payload["bgsi_ema_momentum"] == pytest.approx(0.8)
    assert payload["bgsi_min_axis_observations"] == 3
    assert payload["bgsi_use_axis_agreement_gate"] is False
    assert payload["bgsi_axis_agreement"] == pytest.approx(0.25)


def test_image_end_to_end_command_preserves_protocol_objectives_when_omitted(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_end_to_end_pfml.json"
    captured: dict[str, Any] = {}

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        labels = (0, 1) if split == "train" else (100, 101)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(4)
        ]

    def fake_run(
        *,
        train_examples: list[ImageExample],
        test_examples: list[ImageExample],
        config: Any,
        progress_callback: Any,
    ) -> Any:
        captured["config"] = config
        return SimpleNamespace(
            name="image-end-to-end-benchmark",
            dataset_name=config.dataset_name,
            protocol=config.protocol,
            train_examples=len(train_examples),
            test_examples=len(test_examples),
            methods={},
        )

    def fake_write(result: Any, output: Path) -> Path:
        config = captured["config"]
        output.write_text(
            json.dumps(
                {
                    "objectives": list(config.objectives),
                    "proxy_count_per_class": config.proxy_count_per_class,
                    "optimizer": config.optimizer,
                    "lr_schedule": config.lr_schedule,
                }
            ),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr("sfora.cli.run_image_end_to_end_benchmark", fake_run)
    monkeypatch.setattr("sfora.cli.write_image_end_to_end_report", fake_write)

    result = CliRunner().invoke(
        app,
        [
            "image-end-to-end",
            "--output",
            str(output_path),
            "--dataset-name",
            "cub",
            "--protocol",
            "pfml-resnet50-512",
            "--train-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(output_path.read_text())
    assert payload["objectives"] == ["frozen_pretrained", "pfml"]
    assert payload["proxy_count_per_class"] == 15
    assert payload["optimizer"] == "adam"
    assert payload["lr_schedule"] == "cosine"


def test_image_end_to_end_command_defaults_to_legacy_objectives_when_omitted(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_end_to_end_legacy.json"
    captured: dict[str, Any] = {}

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        labels = (0, 1) if split == "train" else (100, 101)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(4)
        ]

    def fake_run(
        *,
        train_examples: list[ImageExample],
        test_examples: list[ImageExample],
        config: Any,
        progress_callback: Any,
    ) -> Any:
        captured["config"] = config
        return SimpleNamespace(
            name="image-end-to-end-benchmark",
            dataset_name=config.dataset_name,
            protocol=config.protocol,
            train_examples=len(train_examples),
            test_examples=len(test_examples),
            methods={},
        )

    def fake_write(result: Any, output: Path) -> Path:
        config = captured["config"]
        output.write_text(
            json.dumps({"objectives": list(config.objectives)}),
            encoding="utf-8",
        )
        return output

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr("sfora.cli.run_image_end_to_end_benchmark", fake_run)
    monkeypatch.setattr("sfora.cli.write_image_end_to_end_report", fake_write)

    for protocol in ("sota-resnet50-512", "hpl-resnet50-512"):
        result = CliRunner().invoke(
            app,
            [
                "image-end-to-end",
                "--output",
                str(output_path),
                "--dataset-name",
                "cub",
                "--protocol",
                protocol,
                "--train-steps",
                "1",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(output_path.read_text())
        assert payload["objectives"] == ["frozen_pretrained", "group_supcon_xbm_radius"]


def test_image_end_to_end_command_keeps_preset_train_augmentation_when_omitted(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "image_end_to_end_aug.json"
    captured: dict[str, Any] = {}

    def fake_loader(
        *,
        dataset_name: str,
        split: str,
        limit_per_class: int | None,
        min_per_class: int | None,
        max_classes: int | None,
        seed: int,
    ) -> list[ImageExample]:
        labels = (0, 1) if split == "train" else (100, 101)
        return [
            ImageExample(
                example_id=f"{split}-{label}-{index}",
                image=f"{split}-{label}-{index}",
                label=label,
            )
            for label in labels
            for index in range(4)
        ]

    def fake_run(
        *,
        train_examples: list[ImageExample],
        test_examples: list[ImageExample],
        config: Any,
        progress_callback: Any,
    ) -> Any:
        captured["config"] = config
        return SimpleNamespace(
            name="image-end-to-end-benchmark",
            dataset_name=config.dataset_name,
            protocol=config.protocol,
            train_examples=len(train_examples),
            test_examples=len(test_examples),
            methods={},
        )

    def fake_write(result: Any, output: Path) -> Path:
        output.write_text("{}", encoding="utf-8")
        return output

    monkeypatch.setattr("sfora.cli.load_image_retrieval_examples", fake_loader)
    monkeypatch.setattr("sfora.cli.run_image_end_to_end_benchmark", fake_run)
    monkeypatch.setattr("sfora.cli.write_image_end_to_end_report", fake_write)

    result = CliRunner().invoke(
        app,
        [
            "image-end-to-end",
            "--output",
            str(output_path),
            "--dataset-name",
            "cub",
            "--protocol",
            "proxy-anchor-resnet50-512",
            "--train-steps",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert captured["config"].train_augmentation == "full_res_crop"

    result = CliRunner().invoke(
        app,
        [
            "image-end-to-end",
            "--output",
            str(output_path),
            "--dataset-name",
            "cub",
            "--protocol",
            "proxy-anchor-resnet50-512",
            "--train-steps",
            "1",
            "--train-augmentation",
            "standard",
        ],
    )
    assert result.exit_code == 0
    assert captured["config"].train_augmentation == "standard"


class FakeCliImageEncoder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(
        self,
        images: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> NDArray[np.float64]:
        assert batch_size == 8
        rows = []
        for image in images:
            label = int(str(image).split("-")[-2])
            index = int(str(image).split("-")[-1])
            rows.append([float(label), float(index), 1.0])
        embeddings = np.asarray(rows, dtype=np.float64)
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings


def test_imdb_encoder_models_command_writes_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "imdb_encoder_models.json"

    def fake_loader(*, split: str, limit_per_class: int, seed: int) -> list[TextExample]:
        assert split == "train"
        assert limit_per_class == 6
        assert seed == 4
        return [
            TextExample(example_id=f"neg-{index}", text=f"negative review {index}", label=0)
            for index in range(6)
        ] + [
            TextExample(example_id=f"pos-{index}", text=f"positive review {index}", label=1)
            for index in range(6)
        ]

    monkeypatch.setattr("sfora.cli.load_imdb_examples", fake_loader)
    monkeypatch.setattr(
        "sfora.text_baselines._load_sentence_transformer",
        lambda _model_name: FakeSentenceEncoder(),
    )

    result = CliRunner().invoke(
        app,
        [
            "imdb-encoder-models",
            "--output",
            str(output_path),
            "--model-names",
            "fake-mini-a,fake-mini-b",
            "--limit-per-class",
            "6",
            "--group-size",
            "3",
            "--batch-size",
            "4",
            "--seed",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    text = output_path.read_text()
    assert '"name": "sentence-transformer-model-suite"' in text
    assert "sentence_transformer:fake-mini-a" in text
    assert "sentence_transformer:fake-mini-b" in text
    assert "sentence-transformer-model-suite" in result.output


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
        base = np.array([_trainable_features(text) for text in texts], dtype=np.float64)
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
        base = np.array([_trainable_features(example.text) for example in examples])
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
            ),
        )
        self.projection = result.projection_matrix
        return result.history


def _trainable_features(text: str) -> list[float]:
    return [
        float("negative" in text),
        float("positive" in text),
        len(text.split()) / 10.0,
    ]


def _fake_trainable_encoder_factory(_model_name: str) -> FakeTrainableEncoder:
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


def test_imdb_encoder_train_command_writes_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "imdb_encoder_training.json"

    def fake_loader(*, split: str, limit_per_class: int, seed: int) -> list[TextExample]:
        assert split == "train"
        assert limit_per_class == 12
        assert seed == 4
        return [
            TextExample(example_id=f"neg-{index}", text=f"negative review {index}", label=0)
            for index in range(12)
        ] + [
            TextExample(example_id=f"pos-{index}", text=f"positive review {index}", label=1)
            for index in range(12)
        ]

    monkeypatch.setattr("sfora.cli.load_imdb_examples", fake_loader)
    monkeypatch.setattr(
        "sfora.encoder_training._load_trainable_sentence_transformer",
        _fake_trainable_encoder_factory,
    )

    result = CliRunner().invoke(
        app,
        [
            "imdb-encoder-train",
            "--output",
            str(output_path),
            "--model-name",
            "fake-mini-encoder",
            "--limit-per-class",
            "12",
            "--group-size",
            "3",
            "--batch-size",
            "4",
            "--train-steps",
            "8",
            "--learning-rate",
            "0.05",
            "--test-size",
            "0.5",
            "--seed",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    text = output_path.read_text()
    assert "triplet_finetuned:fake-mini-encoder" in text
    assert "group_finetuned:fake-mini-encoder" in text
    assert '"train_examples": 12' in text
    assert '"test_examples": 12' in text
    assert "sentence-transformer-training" in result.output


def test_imdb_encoder_train_command_can_use_official_test_split(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "imdb_encoder_training_official.json"
    calls: list[tuple[str, int, int]] = []

    def fake_loader(*, split: str, limit_per_class: int, seed: int) -> list[TextExample]:
        calls.append((split, limit_per_class, seed))
        count = 6 if split == "train" else 4
        return [
            TextExample(example_id=f"{split}-neg-{index}", text=f"negative review {index}", label=0)
            for index in range(count)
        ] + [
            TextExample(example_id=f"{split}-pos-{index}", text=f"positive review {index}", label=1)
            for index in range(count)
        ]

    monkeypatch.setattr("sfora.cli.load_imdb_examples", fake_loader)
    monkeypatch.setattr(
        "sfora.encoder_training._load_trainable_sentence_transformer",
        _fake_trainable_encoder_factory,
    )

    result = CliRunner().invoke(
        app,
        [
            "imdb-encoder-train",
            "--output",
            str(output_path),
            "--model-name",
            "fake-mini-encoder",
            "--limit-per-class",
            "6",
            "--test-limit-per-class",
            "4",
            "--official-test-split",
            "--group-size",
            "3",
            "--batch-size",
            "4",
            "--train-steps",
            "8",
            "--learning-rate",
            "0.05",
            "--retrieval-query-limit",
            "4",
            "--seed",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert calls == [("train", 6, 4), ("test", 4, 4)]
    payload = json.loads(output_path.read_text())
    assert payload["examples"] == 20
    assert payload["train_examples"] == 12
    assert payload["test_examples"] == 8
    assert payload["config"]["retrieval_query_limit"] == 4
    triplet = payload["methods"]["triplet_finetuned:fake-mini-encoder"]
    assert triplet["initial_retrieval"]["evaluated_queries"] == 4
    assert triplet["initial_retrieval"]["total_queries"] == 8


def test_imdb_encoder_ablation_command_writes_report(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_path = tmp_path / "imdb_encoder_ablation.json"

    def fake_loader(*, split: str, limit_per_class: int, seed: int) -> list[TextExample]:
        assert split == "train"
        assert limit_per_class == 12
        assert seed == 4
        return [
            TextExample(example_id=f"neg-{index}", text=f"negative review {index}", label=0)
            for index in range(12)
        ] + [
            TextExample(example_id=f"pos-{index}", text=f"positive review {index}", label=1)
            for index in range(12)
        ]

    monkeypatch.setattr("sfora.cli.load_imdb_examples", fake_loader)
    monkeypatch.setattr(
        "sfora.encoder_training._load_trainable_sentence_transformer",
        _fake_trainable_encoder_factory,
    )

    result = CliRunner().invoke(
        app,
        [
            "imdb-encoder-ablation",
            "--output",
            str(output_path),
            "--model-name",
            "fake-mini-encoder",
            "--limit-per-class",
            "12",
            "--objectives",
            "triplet,group",
            "--train-steps-grid",
            "4,8",
            "--learning-rates",
            "0.05",
            "--group-sizes",
            "1,3",
            "--batch-size",
            "4",
            "--test-size",
            "0.5",
            "--seed",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    text = output_path.read_text()
    assert '"name": "sentence-transformer-ablation"' in text
    assert '"objective": "group"' in text
    assert '"group_size": 1' in text
    assert '"group_size": 3' in text
    assert '"train_macro_f1_delta"' in text
    assert "sentence-transformer-ablation" in result.output


def test_remote_plan_command_writes_shell_script(tmp_path: Path) -> None:
    output_path = tmp_path / "run_remote.sh"

    result = CliRunner().invoke(
        app,
        [
            "remote-plan",
            "--output",
            str(output_path),
            "--command",
            "uv run --group dev sfora synthetic-train "
            "--output reports/generated/synthetic_trainable.json",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    text = output_path.read_text()
    assert "ssh riomus@192.168.1.35" in text
    assert "synthetic-train" in text
    assert "remote-run-plan" in result.output


def test_remote_plan_command_accepts_explicit_local_dir(tmp_path: Path) -> None:
    output_path = tmp_path / "run_remote.sh"

    result = CliRunner().invoke(
        app,
        [
            "remote-plan",
            "--output",
            str(output_path),
            "--local-dir",
            "/tmp/sfora",
        ],
    )

    assert result.exit_code == 0
    text = output_path.read_text()
    assert 'LOCAL_DIR="' not in text
    assert "/tmp/sfora/" in text


def test_report_build_command_writes_report_and_hf_card(tmp_path: Path) -> None:
    artifact_path = tmp_path / "experiment.json"
    artifact_path.write_text(
        json.dumps(
            {
                "name": "synthetic-trainable",
                "methods": {
                    "group_trained": {
                        "triplet_loss": 0.3,
                        "group_loss": 0.5,
                        "probe": {"accuracy": 0.9, "macro_f1": 0.88},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "REPORT.md"
    card_path = tmp_path / "README.md"

    result = CliRunner().invoke(
        app,
        [
            "report-build",
            "--artifact",
            str(artifact_path),
            "--output",
            str(report_path),
            "--hf-card-output",
            str(card_path),
        ],
    )

    assert result.exit_code == 0
    assert "# Group Learning Report" in report_path.read_text()
    assert "library_name: sentence-transformers" in card_path.read_text()
    assert "report-build" in result.output


def test_report_site_command_writes_html_page(tmp_path: Path) -> None:
    artifact_path = tmp_path / "encoder_training.json"
    artifact_path.write_text(
        json.dumps(
            {
                "name": "sentence-transformer-training",
                "examples": 256,
                "methods": {
                    "group_finetuned:mini": {
                        "triplet_loss": 0.45,
                        "group_loss": 0.79,
                        "probe": {"accuracy": 0.72, "macro_f1": 0.71},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "site" / "index.html"

    result = CliRunner().invoke(
        app,
        [
            "report-site",
            "--artifact",
            str(artifact_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Group trained (mini)" in output_path.read_text(encoding="utf-8")
    assert "report-site" in result.output


def test_bgsi_gate_command_summarizes_hard_seed_discriminator(tmp_path: Path) -> None:
    generated = tmp_path / "reports" / "generated"
    generated.mkdir(parents=True)

    def write_end_to_end_artifact(
        path: Path, objective: str, retrieval: dict[str, float], diagnostics: dict[str, float]
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "name": "image-end-to-end-benchmark",
                    "methods": {
                        f"{objective}_end_to_end:resnet50": {
                            "objective": objective,
                            "retrieval": retrieval,
                            "gsi_diagnostics": diagnostics,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    write_end_to_end_artifact(
        generated / "image_end_to_end_cub.pa_bgsi_pair_w03_60e_seed1.json",
        "proxy_anchor",
        {"recall_at_1": 0.6923, "map_at_r": 0.2796},
        {"boundary_axis_rho_mean": 0.0259, "boundary_axis_rho_p90": 0.0443},
    )
    write_end_to_end_artifact(
        generated / "image_end_to_end_cub.pa_bgsi_ema_w03_60e_seed1.json",
        "proxy_anchor_bgsi",
        {"recall_at_1": 0.6910, "map_at_r": 0.2820},
        {
            "active_fraction_mean": 0.7,
            "boundary_axis_rho_mean": 0.0261,
            "boundary_axis_rho_p90": 0.0430,
            "bgsi_axis_coverage_mean": 0.8,
            "bgsi_ema_ready_fraction_mean": 0.9,
        },
    )
    write_end_to_end_artifact(
        generated / "image_end_to_end_cub.pa_bgsi_permuted_w03_60e_seed1.json",
        "proxy_anchor_bgsi",
        {"recall_at_1": 0.6905, "map_at_r": 0.2810},
        {
            "active_fraction_mean": 0.7,
            "boundary_axis_rho_mean": 0.0263,
            "boundary_axis_rho_p90": 0.0440,
            "bgsi_axis_coverage_mean": 0.8,
            "bgsi_ema_ready_fraction_mean": 0.9,
        },
    )
    write_end_to_end_artifact(
        generated / "image_end_to_end_cub.pa_bgsi_random_w03_60e_seed1.json",
        "proxy_anchor_bgsi",
        {"recall_at_1": 0.6902, "map_at_r": 0.2805},
        {
            "active_fraction_mean": 0.7,
            "boundary_axis_rho_mean": 0.0264,
            "boundary_axis_rho_p90": 0.0441,
            "bgsi_axis_coverage_mean": 1.0,
            "bgsi_ema_ready_fraction_mean": 0.0,
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "bgsi-gate",
            "--generated-dir",
            str(generated),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "baseline R@1=0.6923 MAP@R=0.2796 boundary=0.0259/0.0443" in result.output
    assert "ema_boundary R@1=0.6910 dR@1=-0.0013 MAP@R=0.2820 dMAP@R=+0.0024" in result.output
    assert (
        "gate baseline_ok=True controls_ok=True coverage_ok=True diagnostic_ok=True PASS=True"
        in result.output
    )


def test_default_report_artifacts_prefer_curated_full_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    generated = tmp_path / "reports" / "generated"
    archive = tmp_path / "reports" / "archive"
    generated.mkdir(parents=True)
    archive.mkdir(parents=True)
    (generated / "imdb_encoder_training.json").write_text("{}", encoding="utf-8")
    (generated / "image_retrieval_cub.json").write_text("{}", encoding="utf-8")
    (generated / "image_end_to_end_cub.json").write_text("{}", encoding="utf-8")
    (generated / "image_end_to_end_cub.bn_freeze_full.json").write_text("{}", encoding="utf-8")
    (generated / "image_end_to_end_cars.bn_freeze_full.json").write_text("{}", encoding="utf-8")
    (generated / "image_end_to_end_sop.pfml200_full.json").write_text("{}", encoding="utf-8")
    (generated / "image_end_to_end_sop.pfml200_proxy_gw025_valsel_full.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cars.pfml200_proxy_gw025_full.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cars.pfml200_proxy_gw025_valsel_full.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.pfml200_gw025_full.json").write_text("{}", encoding="utf-8")
    (generated / "image_end_to_end_cub.pfml200_proxy_gw025_full.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.pfml200_proxy_gw025_valsel_full.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.pfml200_potential_gw025_valsel_full.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.stability_teacher20_splitlr.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.full_tune_g8_xbm010_r0.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.full_tune_g8_xbm025_r0.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.full_tune_g16_xbm010_r0.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.proxy_potential_200e.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.group_potential_200e.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.group_potential_40e_g4.json").write_text(
        "{}", encoding="utf-8"
    )
    (generated / "image_end_to_end_cub.triplet_noisy20_pfml_table2.json").write_text(
        "{}", encoding="utf-8"
    )
    (archive / "imdb_encoder_training.full.remote.json").write_text("{}", encoding="utf-8")
    (archive / "image_retrieval_cub.remote.json").write_text("{}", encoding="utf-8")
    (archive / "image_end_to_end_cub.remote.json").write_text("{}", encoding="utf-8")

    artifacts = _default_report_artifacts()

    assert Path("reports/archive/imdb_encoder_training.full.remote.json") in artifacts
    assert Path("reports/generated/imdb_encoder_training.json") not in artifacts
    assert Path("reports/generated/image_retrieval_cub.json") in artifacts
    assert Path("reports/archive/image_retrieval_cub.remote.json") not in artifacts
    assert Path("reports/generated/image_end_to_end_cub.json") not in artifacts
    assert Path("reports/archive/image_end_to_end_cub.remote.json") not in artifacts
    assert Path("reports/generated/image_end_to_end_cub.bn_freeze_full.json") not in artifacts
    assert Path("reports/generated/image_end_to_end_cars.bn_freeze_full.json") not in artifacts
    assert Path("reports/generated/image_end_to_end_sop.pfml200_full.json") not in artifacts
    assert (
        Path("reports/generated/image_end_to_end_sop.pfml200_proxy_gw025_valsel_full.json")
        in artifacts
    )
    assert Path("reports/generated/image_end_to_end_cub.pfml200_gw025_full.json") in artifacts
    assert (
        Path("reports/generated/image_end_to_end_cub.pfml200_proxy_gw025_full.json")
        not in artifacts
    )
    assert (
        Path("reports/generated/image_end_to_end_cub.pfml200_proxy_gw025_valsel_full.json")
        in artifacts
    )
    assert (
        Path("reports/generated/image_end_to_end_cub.pfml200_potential_gw025_valsel_full.json")
        in artifacts
    )
    assert (
        Path("reports/generated/image_end_to_end_cub.stability_teacher20_splitlr.json") in artifacts
    )
    assert Path("reports/generated/image_end_to_end_cub.full_tune_g8_xbm010_r0.json") in artifacts
    assert Path("reports/generated/image_end_to_end_cub.full_tune_g8_xbm025_r0.json") in artifacts
    assert Path("reports/generated/image_end_to_end_cub.full_tune_g16_xbm010_r0.json") in artifacts
    assert Path("reports/generated/image_end_to_end_cub.proxy_potential_200e.json") in artifacts
    assert Path("reports/generated/image_end_to_end_cub.group_potential_200e.json") in artifacts
    assert Path("reports/generated/image_end_to_end_cub.group_potential_40e_g4.json") in artifacts
    assert (
        Path("reports/generated/image_end_to_end_cub.triplet_noisy20_pfml_table2.json") in artifacts
    )
    assert (
        Path("reports/generated/image_end_to_end_cars.pfml200_proxy_gw025_full.json")
        not in artifacts
    )
    assert (
        Path("reports/generated/image_end_to_end_cars.pfml200_proxy_gw025_valsel_full.json")
        in artifacts
    )


def test_hf_publish_command_builds_dry_run_bundle(tmp_path: Path) -> None:
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "README.md").write_text("# sfora\n", encoding="utf-8")
    (tmp_path / "reports" / "archive").mkdir(parents=True)
    (tmp_path / "reports" / "archive" / "result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "reports" / "REPORT.md").write_text("# Report\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "results.md").write_text("# Plan\n", encoding="utf-8")
    (tmp_path / "src" / "sfora").mkdir(parents=True)
    (tmp_path / "src" / "sfora" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sfora'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    output_dir = tmp_path / "dist" / "hf_publish"

    result = CliRunner().invoke(
        app,
        [
            "hf-publish",
            "--repo-id",
            "romanbartusiak/sfora",
            "--project-root",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert (output_dir / "README.md").exists()
    assert (output_dir / "MANIFEST.json").exists()
    assert "uploaded" in result.output
    assert "False" in result.output
