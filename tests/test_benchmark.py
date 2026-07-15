from collections.abc import Callable, Mapping

from sfora.benchmark import benchmark, grid
from sfora.image_end_to_end import ImageEndToEndConfig
from sfora.method import HIST, ProxyAnchor, herd, pa_distill

_Runner = Callable[[ImageEndToEndConfig], Mapping[str, float]]


def _fake_runner(
    recall_by_seed: dict[int, float],
) -> tuple[_Runner, list[ImageEndToEndConfig]]:
    seen: list[ImageEndToEndConfig] = []

    def run(config: ImageEndToEndConfig) -> Mapping[str, float]:
        seen.append(config)
        r = recall_by_seed[config.seed]
        return {
            "recall_at_1": r,
            "recall_at_2": r + 0.05,
            "recall_at_4": r + 0.10,
            "recall_at_8": r + 0.15,
            "map_at_r": r - 0.30,
        }

    return run, seen


def test_benchmark_aggregates_mean_and_std_over_seeds() -> None:
    run, seen = _fake_runner({0: 0.70, 1: 0.72, 2: 0.74})
    result = benchmark(herd(), dataset="cub", seeds=[0, 1, 2], runner=run)

    assert result.method == "IsNorm(Distill(HIST))"
    assert result.dataset == "cub"
    assert result.recall_at_1_per_seed == (0.70, 0.72, 0.74)
    assert abs(result.recall_at_1 - 0.72) < 1e-9
    assert result.recall_at_1_std > 0.0
    assert abs(result.recall_at_2 - 0.77) < 1e-9
    # each seed ran exactly once, with the brick's objective + the right seed
    assert [c.seed for c in seen] == [0, 1, 2]
    assert all(c.objectives == ("hist",) for c in seen)
    assert all(c.embedding_layer_norm is True for c in seen)
    assert all(c.dataset_name == "cub" for c in seen)


def test_benchmark_single_seed_zero_std() -> None:
    run, _ = _fake_runner({0: 0.88})
    result = benchmark(pa_distill(), dataset="cars", seeds=[0], runner=run)
    assert result.recall_at_1 == 0.88
    assert result.recall_at_1_std == 0.0


def test_benchmark_requires_a_seed() -> None:
    run, _ = _fake_runner({})
    try:
        benchmark(HIST(), dataset="cub", seeds=[], runner=run)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for empty seeds")


def test_grid_mapping_preserves_custom_labels() -> None:
    run, seen = _fake_runner({0: 0.5})
    methods = {"HERD": herd(), "PA": ProxyAnchor()}
    results = grid(methods, datasets=["cub", "cars"], seeds=[0], runner=run)
    assert len(results) == 4  # 2 methods x 2 datasets
    # mapping keys become the result labels (not the brick .name)
    assert {(r.method, r.dataset) for r in results} == {
        ("HERD", "cub"),
        ("HERD", "cars"),
        ("PA", "cub"),
        ("PA", "cars"),
    }
    # ProxyAnchor bricks carry a proxy per class; HERD bricks do not.
    assert any(c.proxy_count_per_class == 1 and c.objectives == ("proxy_anchor",) for c in seen)


def test_benchmark_rejects_unknown_override_key() -> None:
    import pytest

    run, _ = _fake_runner({0: 0.7})
    with pytest.raises(ValueError, match="unknown override"):
        benchmark(herd(), dataset="cub", seeds=[0], overrides={"hist_taau": 7.0}, runner=run)


def test_overrides_take_precedence_over_brick_fields() -> None:
    run, seen = _fake_runner({0: 0.7})
    benchmark(HIST(tau=99.0), dataset="cub", seeds=[0], overrides={"hist_tau": 16.0}, runner=run)
    assert seen[0].hist_tau == 16.0  # explicit override wins over the brick's tau


def test_benchmark_requires_all_metrics_from_runner() -> None:
    import pytest

    def bad_runner(config: ImageEndToEndConfig) -> Mapping[str, float]:
        return {"recall_at_1": 0.7}  # missing recall_at_2/4/8, map_at_r

    with pytest.raises(ValueError, match="required metric"):
        benchmark(herd(), dataset="cub", seeds=[0], runner=bad_runner)


def test_grid_sequence_labels_by_brick_name() -> None:
    run, _ = _fake_runner({0: 0.5})
    results = grid([herd(), ProxyAnchor()], datasets=["cub"], seeds=[0], runner=run)
    assert {r.method for r in results} == {"IsNorm(Distill(HIST))", "ProxyAnchor"}


def test_type_safe_constants_are_the_underlying_literals() -> None:
    from sfora.catalog import Dataset, Protocol

    assert Dataset.CUB == "cub"
    assert Dataset.ALL == ("cub", "cars", "sop")
    assert Protocol.PROXY_ANCHOR_R50_512 == "proxy-anchor-resnet50-512"

    run, seen = _fake_runner({0: 0.7})
    result = benchmark(herd(), dataset=Dataset.CUB, seeds=[0], runner=run)
    assert result.dataset == "cub"
    assert seen[0].protocol == Protocol.PROXY_ANCHOR_R50_512


def test_grid_accepts_a_plain_sequence_of_methods() -> None:
    run, _ = _fake_runner({0: 0.5})
    from sfora.catalog import Dataset

    results = grid([herd(), ProxyAnchor()], datasets=Dataset.ALL, seeds=[0], runner=run)
    assert len(results) == 6  # 2 methods x 3 datasets
    # labelled by each brick's .name, no manual string keys needed
    assert {r.method for r in results} == {"IsNorm(Distill(HIST))", "ProxyAnchor"}
