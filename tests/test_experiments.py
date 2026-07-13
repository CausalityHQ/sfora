import json
from pathlib import Path

import pytest

from sfora.experiments import (
    SyntheticExperimentConfig,
    TrainableSyntheticExperimentConfig,
    run_synthetic_experiment,
    run_trainable_synthetic_experiment,
    write_experiment_report,
)


def test_synthetic_experiment_returns_comparable_method_metrics() -> None:
    result = run_synthetic_experiment(
        SyntheticExperimentConfig(samples_per_class=12, dimensions=6, group_size=3, seed=11)
    )

    assert result.name == "synthetic-smoke"
    assert set(result.methods) == {"raw", "triplet_baseline", "sfora"}
    assert result.methods["sfora"].group_loss < result.methods["raw"].group_loss
    assert result.methods["sfora"].probe.accuracy >= result.methods["raw"].probe.accuracy


def test_synthetic_experiment_validates_group_size() -> None:
    with pytest.raises(ValueError, match="divide"):
        run_synthetic_experiment(
            SyntheticExperimentConfig(samples_per_class=10, dimensions=4, group_size=3)
        )


def test_write_experiment_report_persists_json(tmp_path: Path) -> None:
    result = run_synthetic_experiment(
        SyntheticExperimentConfig(samples_per_class=8, dimensions=4, group_size=2, seed=7)
    )
    output_path = tmp_path / "synthetic.json"

    written_path = write_experiment_report(result, output_path)

    payload = json.loads(written_path.read_text())
    assert payload["name"] == "synthetic-smoke"
    assert payload["config"]["group_size"] == 2
    assert sorted(payload["methods"]) == ["raw", "sfora", "triplet_baseline"]
    assert payload["methods"]["sfora"]["group_loss"] >= 0.0


def test_trainable_synthetic_experiment_compares_triplet_and_group_training() -> None:
    result = run_trainable_synthetic_experiment(
        TrainableSyntheticExperimentConfig(
            samples_per_class=8,
            dimensions=4,
            group_size=2,
            train_steps=30,
            learning_rate=0.05,
            seed=11,
        )
    )

    assert result.name == "synthetic-trainable"
    assert set(result.methods) == {"raw", "triplet_trained", "group_trained"}
    assert result.methods["triplet_trained"].triplet_loss < result.methods["raw"].triplet_loss
    assert result.methods["group_trained"].group_loss < result.methods["raw"].group_loss
