from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sfora.experiments import (
    TrainableSyntheticExperimentConfig,
    run_trainable_synthetic_experiment,
)


class SyntheticAblationConfig(BaseModel):
    """Configuration for a trainable synthetic sfora ablation grid."""

    samples_per_class: int = Field(default=24, ge=4)
    dimensions: int = Field(default=8, ge=2)
    group_sizes: tuple[int, ...] = (2, 4)
    hard_weights: tuple[float, ...] = (0.0, 0.5)
    spread_weights: tuple[float, ...] = (0.0, 0.1)
    train_steps: int = Field(default=80, ge=1)
    learning_rate: float = Field(default=0.03, gt=0.0)
    seed: int = 0


@dataclass(frozen=True)
class SyntheticAblationTrial:
    """One ranked ablation trial."""

    rank: int
    group_size: int
    hard_weight: float
    spread_weight: float
    triplet_loss: float
    group_loss: float
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class SyntheticAblationResult:
    """Serializable result for a synthetic ablation grid."""

    name: str
    config: SyntheticAblationConfig
    best_trial: SyntheticAblationTrial
    trials: list[SyntheticAblationTrial]


def run_synthetic_ablation(
    config: SyntheticAblationConfig | None = None,
) -> SyntheticAblationResult:
    """Run and rank trainable synthetic sfora ablation trials."""
    resolved_config = config or SyntheticAblationConfig()
    unranked_trials: list[SyntheticAblationTrial] = []

    for group_size in resolved_config.group_sizes:
        for hard_weight in resolved_config.hard_weights:
            for spread_weight in resolved_config.spread_weights:
                experiment = run_trainable_synthetic_experiment(
                    TrainableSyntheticExperimentConfig(
                        samples_per_class=resolved_config.samples_per_class,
                        dimensions=resolved_config.dimensions,
                        group_size=group_size,
                        hard_weight=hard_weight,
                        spread_weight=spread_weight,
                        train_steps=resolved_config.train_steps,
                        learning_rate=resolved_config.learning_rate,
                        seed=resolved_config.seed,
                    )
                )
                metrics = experiment.methods["group_trained"]
                unranked_trials.append(
                    SyntheticAblationTrial(
                        rank=0,
                        group_size=group_size,
                        hard_weight=hard_weight,
                        spread_weight=spread_weight,
                        triplet_loss=metrics.triplet_loss,
                        group_loss=metrics.group_loss,
                        accuracy=metrics.probe.accuracy,
                        macro_f1=metrics.probe.macro_f1,
                    )
                )

    ranked_trials = [
        SyntheticAblationTrial(rank=rank, **_trial_without_rank(trial))
        for rank, trial in enumerate(
            sorted(
                unranked_trials,
                key=lambda trial: (
                    trial.group_loss,
                    -trial.accuracy,
                    trial.triplet_loss,
                    trial.group_size,
                    trial.hard_weight,
                    trial.spread_weight,
                ),
            ),
            start=1,
        )
    ]
    return SyntheticAblationResult(
        name="synthetic-ablation",
        config=resolved_config,
        best_trial=ranked_trials[0],
        trials=ranked_trials,
    )


def write_ablation_report(result: SyntheticAblationResult, output_path: Path) -> Path:
    """Persist an ablation result as stable, human-readable JSON."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_payload(result), indent=2, sort_keys=True) + "\n")
    return output_path


def _trial_without_rank(trial: SyntheticAblationTrial) -> dict[str, Any]:
    payload = asdict(trial)
    payload.pop("rank")
    return payload


def _to_payload(result: SyntheticAblationResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "config": result.config.model_dump(),
        "best_trial": asdict(result.best_trial),
        "trials": [asdict(trial) for trial in result.trials],
    }
