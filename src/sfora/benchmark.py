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
from dataclasses import dataclass

from sfora.catalog import Dataset, Protocol
from sfora.data import ImageDatasetName
from sfora.image_end_to_end import EndToEndProtocol, ImageEndToEndConfig, config_for_protocol
from sfora.method import Objective, build_config

__all__ = ["BenchmarkResult", "Dataset", "Protocol", "TrainRunner", "benchmark", "grid"]

# A runner trains one config and returns its metrics as a name -> value mapping
# (at least "recall_at_1"). Injectable so the aggregation is testable without torch.
TrainRunner = Callable[[ImageEndToEndConfig], Mapping[str, float]]

_METRICS = ("recall_at_1", "recall_at_2", "recall_at_4", "recall_at_8", "map_at_r")


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated metrics for one method on one dataset over seeds.

    Metrics are the trainer's reported retrieval on the test split (its primary
    ``recall_at_1`` is the **final-epoch** model). The project's headline numbers use
    the *best-over-training* protocol (peak test R@1), which the trainer tracks only
    as a diagnostic — reproduce those via the remote scripts, not this runner.
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

    def summary(self) -> str:
        return (
            f"{self.method} · {self.dataset}: R@1 {self.recall_at_1:.4f} "
            f"± {self.recall_at_1_std:.4f} (seeds {list(self.seeds)})"
        )


def benchmark(
    method: Objective,
    *,
    dataset: ImageDatasetName,
    seeds: Sequence[int] = (0,),
    protocol: EndToEndProtocol = Protocol.PROXY_ANCHOR_R50_512,
    overrides: Mapping[str, object] | None = None,
    runner: TrainRunner | None = None,
    label: str | None = None,
) -> BenchmarkResult:
    """Benchmark a method brick on a dataset over seeds; returns aggregated metrics.

    ``overrides`` are dataset/training config fields that **take precedence over the
    brick's fields** (applied after the method compiles). Unknown or out-of-range
    override values raise, rather than being silently dropped. ``label`` sets the
    result's method label (defaults to ``method.name``).
    """
    if not seeds:
        raise ValueError("benchmark requires at least one seed")
    if overrides:
        unknown = sorted(set(overrides) - set(ImageEndToEndConfig.model_fields))
        if unknown:
            raise ValueError(f"unknown override field(s): {unknown}")
    run = runner or _default_runner
    base = config_for_protocol(protocol, dataset_name=dataset)

    per_seed_metrics: list[Mapping[str, float]] = []
    for seed in seeds:
        config = build_config(method, base)
        if overrides:
            # overrides win over brick fields, and are re-validated (not silently kept).
            config = ImageEndToEndConfig.model_validate({**config.model_dump(), **dict(overrides)})
        config = config.model_copy(update={"dataset_name": dataset, "seed": int(seed)})
        metrics = run(config)
        missing = sorted(set(_METRICS) - set(metrics))
        if missing:
            raise ValueError(f"runner did not return required metric(s): {missing}")
        per_seed_metrics.append(metrics)

    def agg(metric: str) -> float:
        return statistics.mean(float(m[metric]) for m in per_seed_metrics)

    r1 = [float(m["recall_at_1"]) for m in per_seed_metrics]
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
    )


def grid(
    methods: Mapping[str, Objective] | Sequence[Objective],
    *,
    datasets: Sequence[ImageDatasetName],
    seeds: Sequence[int] = (0,),
    protocol: EndToEndProtocol = Protocol.PROXY_ANCHOR_R50_512,
    overrides: Mapping[str, object] | None = None,
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
                    runner=runner,
                    label=label,
                )
            )
    return results


def _default_runner(config: ImageEndToEndConfig) -> Mapping[str, float]:
    """Load the dataset, train one config with the verified trainer, extract metrics."""
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
    result = run_image_end_to_end_benchmark(
        train_examples=train_examples, test_examples=test_examples, config=config
    )
    trained = [m for m in result.methods.values() if m.objective == config.objectives[0]]
    if not trained:
        raise RuntimeError(f"trainer returned no metrics for objective {config.objectives[0]}")
    metrics = trained[-1]
    return {name: float(getattr(metrics, name)) for name in _METRICS}
