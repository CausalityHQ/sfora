"""Multi-seed benchmark runner for method bricks.

Compose a method from :mod:`sfora.method` bricks and benchmark it on a dataset over
several seeds, getting a typed :class:`BenchmarkResult` with per-metric mean and
standard deviation:

    from sfora.method import herd, pa_distill, ProxyAnchor
    from sfora.benchmark import benchmark, grid

    benchmark(herd(), dataset="cub", seeds=[0, 1, 2])
    grid({"HERD": herd(), "PA+distill": pa_distill(), "PA": ProxyAnchor()},
         datasets=["cub", "cars"], seeds=[0, 1, 2])

The actual training is delegated to an injectable ``runner`` (default: the verified
``run_image_end_to_end_benchmark`` trainer), so the aggregation logic is unit-tested
without a GPU.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sfora.catalog import Dataset, Protocol
from sfora.data import ImageDatasetName
from sfora.image_end_to_end import EndToEndProtocol, ImageEndToEndConfig, config_for_protocol
from sfora.method import LossFn, Objective, build_config, custom_losses_of

__all__ = ["BenchmarkResult", "Dataset", "Protocol", "SeedRun", "TrainRunner", "benchmark", "grid"]

_METRICS = ("recall_at_1", "recall_at_2", "recall_at_4", "recall_at_8", "map_at_r")


# A custom eval metric: (test_embeddings, test_labels) -> scalar, computed at each
# eval interval and tracked as a training curve alongside loss and recall_at_1.
MetricFn = Callable[[NDArray[np.floating], NDArray[np.integer]], float]


@dataclass(frozen=True)
class SeedRun:
    """One seed's result: final/best scalar metrics + named training curves.

    ``metrics`` must contain ``recall_at_1/2/4/8`` and ``map_at_r`` (plus any custom
    metric's final value). ``curves`` maps a name -> per-eval-interval values; the
    default runner provides ``"loss"``, ``"recall_at_1"``, and one entry per custom
    metric. Empty if the runner does not track curves.
    """

    metrics: Mapping[str, float]
    curves: Mapping[str, tuple[float, ...]] = field(default_factory=dict)


# A runner trains one config and returns a SeedRun. Injectable so aggregation is
# testable without torch.
TrainRunner = Callable[[ImageEndToEndConfig], SeedRun]


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated metrics for one method on one dataset over seeds.

    The default runner reports **best-over-training** metrics (the full retrieval set
    at the peak test-R@1 epoch — the standard DML protocol the papers report), by
    enabling per-epoch test evaluation. A custom ``runner`` may report whatever it
    likes; the fields are R@1/2/4/8 and MAP@R, mean ± std over seeds.
    """

    method: str
    dataset: ImageDatasetName
    seeds: tuple[int, ...]
    recall_at_1: float
    recall_at_1_std: float
    recall_at_1_per_seed: tuple[float, ...]
    recall_at_2: float
    recall_at_4: float
    recall_at_8: float
    map_at_r: float
    # Per-seed named training curves (e.g. "loss", "recall_at_1", custom metrics).
    curves_per_seed: tuple[Mapping[str, tuple[float, ...]], ...] = ()

    def summary(self) -> str:
        return (
            f"{self.method} · {self.dataset}: R@1 {self.recall_at_1:.4f} "
            f"± {self.recall_at_1_std:.4f} (seeds {list(self.seeds)})"
        )

    def mean_curve(self, name: str) -> tuple[float, ...]:
        """Mean of the named curve across seeds, truncated to the shortest seed."""
        series = [c[name] for c in self.curves_per_seed if name in c and c[name]]
        if not series:
            return ()
        length = min(len(s) for s in series)
        return tuple(statistics.mean(s[i] for s in series) for i in range(length))


def benchmark(
    method: Objective,
    *,
    dataset: ImageDatasetName,
    seeds: Sequence[int] = (0,),
    protocol: EndToEndProtocol = Protocol.PROXY_ANCHOR_R50_512,
    overrides: Mapping[str, object] | None = None,
    metrics: Mapping[str, MetricFn] | None = None,
    runner: TrainRunner | None = None,
    label: str | None = None,
) -> BenchmarkResult:
    """Benchmark a method brick on a dataset over seeds; returns aggregated metrics.

    ``overrides`` are dataset/training config fields that **take precedence over the
    brick's fields** (applied after the method compiles). Unknown or out-of-range
    override values raise, rather than being silently dropped. ``metrics`` are custom
    ``(embeddings, labels) -> float`` eval metrics computed each eval interval and
    exposed as curves (default runner only). ``label`` sets the result's method label.
    """
    if not seeds:
        raise ValueError("benchmark requires at least one seed")
    if overrides:
        unknown = sorted(set(overrides) - set(ImageEndToEndConfig.model_fields))
        if unknown:
            raise ValueError(f"unknown override field(s): {unknown}")
    # The default runner computes custom metrics + dispatches any CustomObjective loss;
    # a custom runner owns its own metrics/losses.
    losses = custom_losses_of(method)
    run: TrainRunner = runner or (
        lambda cfg: _default_runner(cfg, extra_metrics=metrics or {}, custom_losses=losses)
    )
    base = config_for_protocol(protocol, dataset_name=dataset)

    runs: list[SeedRun] = []
    for seed in seeds:
        config = build_config(method, base)
        if overrides:
            # overrides win over brick fields, and are re-validated (not silently kept).
            config = ImageEndToEndConfig.model_validate({**config.model_dump(), **dict(overrides)})
        config = config.model_copy(update={"dataset_name": dataset, "seed": int(seed)})
        seed_run = run(config)
        missing = sorted(set(_METRICS) - set(seed_run.metrics))
        if missing:
            raise ValueError(f"runner did not return required metric(s): {missing}")
        runs.append(seed_run)

    def agg(metric: str) -> float:
        return statistics.mean(float(r.metrics[metric]) for r in runs)

    r1 = [float(r.metrics["recall_at_1"]) for r in runs]
    return BenchmarkResult(
        method=label or method.name,
        dataset=dataset,
        seeds=tuple(int(s) for s in seeds),
        recall_at_1=statistics.mean(r1),
        recall_at_1_std=statistics.pstdev(r1) if len(r1) > 1 else 0.0,
        recall_at_1_per_seed=tuple(r1),
        recall_at_2=agg("recall_at_2"),
        recall_at_4=agg("recall_at_4"),
        recall_at_8=agg("recall_at_8"),
        map_at_r=agg("map_at_r"),
        curves_per_seed=tuple(dict(r.curves) for r in runs),
    )


def grid(
    methods: Mapping[str, Objective] | Sequence[Objective],
    *,
    datasets: Sequence[ImageDatasetName],
    seeds: Sequence[int] = (0,),
    protocol: EndToEndProtocol = Protocol.PROXY_ANCHOR_R50_512,
    overrides: Mapping[str, object] | None = None,
    metrics: Mapping[str, MetricFn] | None = None,
    runner: TrainRunner | None = None,
) -> list[BenchmarkResult]:
    """Benchmark every method on every dataset; returns a flat list of results.

    ``methods`` may be a plain sequence of bricks (labelled by each brick's
    ``.name``) or a mapping of custom label -> brick.
    """
    # Preserve custom mapping labels (a sequence is labelled by each brick's .name).
    labelled: list[tuple[str | None, Objective]] = (
        [(k, v) for k, v in methods.items()]
        if isinstance(methods, Mapping)
        else [(None, m) for m in methods]
    )
    results: list[BenchmarkResult] = []
    for dataset in datasets:
        for label, method in labelled:
            results.append(
                benchmark(
                    method,
                    dataset=dataset,
                    seeds=seeds,
                    protocol=protocol,
                    overrides=overrides,
                    metrics=metrics,
                    runner=runner,
                    label=label,
                )
            )
    return results


def _default_runner(
    config: ImageEndToEndConfig,
    *,
    extra_metrics: Mapping[str, MetricFn] | None = None,
    custom_losses: Mapping[str, LossFn] | None = None,
) -> SeedRun:
    """Load the dataset, train one config, and extract scalar metrics + training curves."""
    from sfora.data import load_image_retrieval_examples
    from sfora.image_end_to_end import run_image_end_to_end_benchmark

    train_examples = load_image_retrieval_examples(
        dataset_name=config.dataset_name, split="train", seed=config.seed
    )
    test_examples = load_image_retrieval_examples(
        dataset_name=config.dataset_name, split="test", seed=config.seed
    )
    # A method brick compiles to exactly one trained objective; require that so the
    # extracted metrics are unambiguous (not "whichever objective happened to run last").
    if len(config.objectives) != 1:
        raise ValueError(
            f"the benchmark runner expects a single-objective config, got {config.objectives}"
        )
    # Track best-over-training (peak test R@1) — the project's headline protocol — unless
    # the caller already set an eval cadence.
    if config.eval_test_interval_epochs <= 0:
        config = config.model_copy(update={"eval_test_interval_epochs": 5})
    result = run_image_end_to_end_benchmark(
        train_examples=train_examples,
        test_examples=test_examples,
        config=config,
        extra_eval_metrics=dict(extra_metrics) if extra_metrics else None,
        custom_losses=dict(custom_losses) if custom_losses else None,
    )
    trained = [m for m in result.methods.values() if m.objective == config.objectives[0]]
    if not trained:
        raise RuntimeError(f"trainer returned no metrics for objective {config.objectives[0]}")
    metrics = trained[-1]
    # Prefer the full best-over-training metric set; fall back to final-epoch metrics.
    source: object = metrics.best_test_retrieval if metrics.best_test_retrieval else metrics
    scalars = {name: float(getattr(source, name)) for name in _METRICS}

    curves: dict[str, tuple[float, ...]] = {}
    if metrics.loss_history:
        curves["loss"] = tuple(float(x) for x in metrics.loss_history)
    if metrics.test_recall_history:
        curves["recall_at_1"] = tuple(float(x) for x in metrics.test_recall_history)
    for name, series in (metrics.extra_metric_curves or {}).items():
        values = tuple(float(x) for x in series)
        curves[name] = values
        scalars[name] = values[-1] if values else float("nan")  # final custom-metric value
    return SeedRun(metrics=scalars, curves=curves)
