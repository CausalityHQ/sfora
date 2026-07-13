from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from sfora.data import TextExample
from sfora.encoder_training import (
    EncoderFactory,
    EncoderObjective,
    EncoderTrainingConfig,
    run_encoder_training,
)


class EncoderAblationConfig(BaseModel):
    """Configuration for neural encoder fine-tuning ablations."""

    model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    objectives: tuple[EncoderObjective, ...] = (
        "triplet",
        "group",
        "hybrid",
        "hybrid_xbm",
        "hybrid_radius",
        "hybrid_xbm_radius",
    )
    train_steps: tuple[int, ...] = (20, 80)
    learning_rates: tuple[float, ...] = (2e-5,)
    group_sizes: tuple[int, ...] = Field(default=(4, 8, 16), min_length=1)
    batch_size: int = Field(default=32, ge=1)
    test_size: float = Field(default=0.25, gt=0.0, lt=1.0)
    seed: int = 0

    @field_validator("group_sizes")
    @classmethod
    def _group_sizes_are_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(group_size < 1 for group_size in value):
            raise ValueError("group_sizes must contain only positive integers")
        return value


@dataclass(frozen=True)
class EncoderAblationTrial:
    """One ranked trainable encoder ablation trial."""

    rank: int
    objective: EncoderObjective
    group_size: int
    train_steps: int
    learning_rate: float
    macro_f1: float
    f1_delta: float
    train_macro_f1: float
    train_macro_f1_delta: float
    f1_generalization_gap: float
    precision_at_1: float
    precision_at_1_delta: float
    map_at_r: float
    map_at_r_delta: float
    signal_to_noise_ratio_delta: float
    drift_to_gap_ratio_delta: float


@dataclass(frozen=True)
class EncoderAblationResult:
    """Serializable result for neural encoder ablation trials."""

    name: str
    config: EncoderAblationConfig
    examples: int
    train_examples: int
    test_examples: int
    best_trial: EncoderAblationTrial
    trials: list[EncoderAblationTrial]


def run_encoder_ablation(
    examples: list[TextExample],
    config: EncoderAblationConfig | None = None,
    *,
    encoder_factory: EncoderFactory | None = None,
) -> EncoderAblationResult:
    """Run and rank trainable encoder ablations by held-out F1 first."""
    resolved_config = config or EncoderAblationConfig()
    unranked_trials: list[EncoderAblationTrial] = []
    examples_count = 0
    train_examples = 0
    test_examples = 0

    for objective in resolved_config.objectives:
        for group_size in resolved_config.group_sizes:
            for train_steps in resolved_config.train_steps:
                for learning_rate in resolved_config.learning_rates:
                    result = run_encoder_training(
                        examples,
                        EncoderTrainingConfig(
                            model_name=resolved_config.model_name,
                            objectives=(objective,),
                            train_steps=train_steps,
                            learning_rate=learning_rate,
                            group_size=group_size,
                            batch_size=resolved_config.batch_size,
                            test_size=resolved_config.test_size,
                            seed=resolved_config.seed,
                        ),
                        encoder_factory=encoder_factory,
                    )
                    examples_count = result.examples
                    train_examples = result.train_examples
                    test_examples = result.test_examples
                    metrics = next(iter(result.methods.values()))
                    unranked_trials.append(
                        EncoderAblationTrial(
                            rank=0,
                            objective=objective,
                            group_size=group_size,
                            train_steps=train_steps,
                            learning_rate=learning_rate,
                            macro_f1=metrics.probe.macro_f1,
                            f1_delta=metrics.probe.macro_f1 - metrics.initial_probe.macro_f1,
                            train_macro_f1=metrics.probe.train_macro_f1,
                            train_macro_f1_delta=(
                                metrics.probe.train_macro_f1 - metrics.initial_probe.train_macro_f1
                            ),
                            f1_generalization_gap=(
                                metrics.probe.train_macro_f1 - metrics.probe.macro_f1
                            ),
                            precision_at_1=metrics.retrieval.precision_at_1,
                            precision_at_1_delta=(
                                metrics.retrieval.precision_at_1
                                - metrics.initial_retrieval.precision_at_1
                            ),
                            map_at_r=metrics.retrieval.map_at_r,
                            map_at_r_delta=metrics.retrieval.map_at_r
                            - metrics.initial_retrieval.map_at_r,
                            signal_to_noise_ratio_delta=(
                                metrics.space.signal_to_noise_ratio
                                - metrics.initial_space.signal_to_noise_ratio
                            ),
                            drift_to_gap_ratio_delta=(
                                metrics.space.drift_to_gap_ratio
                                - metrics.initial_space.drift_to_gap_ratio
                            ),
                        )
                    )

    ranked_trials = [
        EncoderAblationTrial(rank=rank, **_trial_without_rank(trial))
        for rank, trial in enumerate(
            sorted(
                unranked_trials,
                key=lambda trial: (
                    -trial.macro_f1,
                    trial.f1_generalization_gap,
                    -trial.map_at_r_delta,
                    trial.train_steps,
                    trial.learning_rate,
                    trial.group_size,
                    trial.objective,
                ),
            ),
            start=1,
        )
    ]
    return EncoderAblationResult(
        name="sentence-transformer-ablation",
        config=resolved_config,
        examples=examples_count,
        train_examples=train_examples,
        test_examples=test_examples,
        best_trial=ranked_trials[0],
        trials=ranked_trials,
    )


def write_encoder_ablation_report(result: EncoderAblationResult, output_path: Path) -> Path:
    """Persist an encoder ablation result as stable, human-readable JSON."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_payload(result), indent=2, sort_keys=True) + "\n")
    return output_path


def _trial_without_rank(trial: EncoderAblationTrial) -> dict[str, Any]:
    payload = asdict(trial)
    payload.pop("rank")
    return payload


def _to_payload(result: EncoderAblationResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "config": result.config.model_dump(),
        "examples": result.examples,
        "train_examples": result.train_examples,
        "test_examples": result.test_examples,
        "best_trial": asdict(result.best_trial),
        "trials": [asdict(trial) for trial in result.trials],
    }
