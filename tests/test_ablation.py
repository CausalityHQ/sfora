import json
from pathlib import Path

from sfora.ablation import (
    SyntheticAblationConfig,
    run_synthetic_ablation,
    write_ablation_report,
)


def test_run_synthetic_ablation_returns_sorted_trials() -> None:
    result = run_synthetic_ablation(
        SyntheticAblationConfig(
            samples_per_class=8,
            dimensions=4,
            group_sizes=(2, 4),
            hard_weights=(0.0, 0.5),
            spread_weights=(0.0,),
            train_steps=15,
            seed=3,
        )
    )

    assert result.name == "synthetic-ablation"
    assert len(result.trials) == 4
    assert result.best_trial.group_loss == min(trial.group_loss for trial in result.trials)
    assert [trial.rank for trial in result.trials] == [1, 2, 3, 4]
    assert {trial.group_size for trial in result.trials} == {2, 4}


def test_write_ablation_report_persists_json(tmp_path: Path) -> None:
    result = run_synthetic_ablation(
        SyntheticAblationConfig(
            samples_per_class=8,
            dimensions=4,
            group_sizes=(2,),
            hard_weights=(0.0,),
            spread_weights=(0.0, 0.2),
            train_steps=10,
            seed=5,
        )
    )
    output_path = tmp_path / "ablation.json"

    written_path = write_ablation_report(result, output_path)

    payload = json.loads(written_path.read_text())
    assert payload["name"] == "synthetic-ablation"
    assert len(payload["trials"]) == 2
    assert payload["best_trial"]["rank"] == 1
