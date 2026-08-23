from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_probe import (
    PROBE_LEARNING_RATE,
    ProbeFit,
    ProbeGradientMetrics,
    ProbeMetrics,
    ProbeSeedStatistics,
    ProbeSplit,
    class_mean_head,
    compare_probe_gradients,
    evaluate_probe_heads,
    fit_spherical_probe,
    probe_decision,
    split_probe_records,
)
from sfora.unicom_training import sample_shard_masks, sharded_mask_arcface_logits


def _record(label: str, name: str) -> InshopRecord:
    return InshopRecord(split="train", image_path=Path("/dataset") / name, label=label)


def test_split_probe_records_uses_seeded_validation_and_keeps_singletons() -> None:
    records = (
        _record("c", "03_1_front.jpg"),
        _record("a", "02_3_side.jpg"),
        _record("a", "01_1_front.jpg"),
        _record("b", "07_1_front.jpg"),
        _record("c", "03_2_side.jpg"),
        _record("a", "01_2_back.jpg"),
    )

    actual = split_probe_records(records, {"a": 0, "b": 1, "c": 2})

    assert type(actual) is ProbeSplit
    assert actual.fitting == (
        _record("a", "02_3_side.jpg"),
        _record("b", "07_1_front.jpg"),
        _record("c", "03_1_front.jpg"),
    )
    assert actual.validation == (
        _record("a", "01_1_front.jpg"),
        _record("c", "03_2_side.jpg"),
    )
    assert actual.validation_group_represented == (False, True)
    assert actual.validation_class_count == 2
    assert actual.singleton_class_count == 1
    assert actual.excluded_same_series_count == 1


def test_split_probe_records_reports_unrepresented_acquisition_series() -> None:
    actual = split_probe_records(
        (
            _record("a", "01_1_front.jpg"),
            _record("a", "02_1_front.jpg"),
            _record("a", "03_1_front.jpg"),
        ),
        {"a": 0},
    )

    assert actual.validation == (_record("a", "01_1_front.jpg"),)
    assert actual.validation_group_represented == (False,)
    assert actual.excluded_same_series_count == 0


@pytest.mark.parametrize(
    ("records", "labels"),
    (
        ((_record("a", "a-1.jpg"), _record("a", "a-2.jpg")), {"a": 1}),
        ((_record("a", "a-1.jpg"), _record("a", "a-2.jpg")), {"a": 0, "b": 1}),
        ((_record("a", "a-1.jpg"), InshopRecord("query", Path("/dataset/a-2.jpg"), "a")), {"a": 0}),
    ),
)
def test_split_probe_records_rejects_invalid_class_inventory(
    records: tuple[InshopRecord, ...], labels: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        split_probe_records(records, labels)


def test_class_mean_head_matches_fp64_normalized_class_means() -> None:
    features = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1], dtype=torch.int64)

    actual = class_mean_head(features, labels, class_count=2)

    normalized = torch.nn.functional.normalize(features, dim=1).double()
    expected = torch.stack((normalized[:2].mean(dim=0), normalized[2:].mean(dim=0)))
    expected = torch.nn.functional.normalize(expected, dim=1).float()
    expected *= 0.01 * (4.0**0.5)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"


@pytest.mark.parametrize(
    ("features", "labels", "class_count"),
    (
        (torch.ones(2, 4, dtype=torch.float64), torch.tensor([0, 1]), 2),
        (torch.ones(2, 4), torch.tensor([0, 1], dtype=torch.int32), 2),
        (torch.ones(2, 4), torch.tensor([0, 2]), 2),
        (torch.ones(2, 4), torch.tensor([0, 0]), 2),
        (torch.tensor([[0.0, 0.0], [1.0, 0.0]]), torch.tensor([0, 1]), 2),
    ),
)
def test_class_mean_head_rejects_invalid_features_or_labels(
    features: torch.Tensor, labels: torch.Tensor, class_count: int
) -> None:
    with pytest.raises((TypeError, ValueError)):
        class_mean_head(features, labels, class_count)


def _separable_probe_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_count = 8
    dimension = 768
    labels = torch.arange(class_count, dtype=torch.int64).repeat_interleave(4)
    features = torch.zeros(labels.numel(), dimension, dtype=torch.float32)
    features[torch.arange(labels.numel()), labels] = 1.0
    features[:, class_count] = torch.linspace(0.01, 0.04, labels.numel())
    features = features.contiguous()
    initial = torch.roll(torch.eye(class_count, dimension, dtype=torch.float32), shifts=1, dims=0)
    initial = torch.nn.functional.normalize(initial, dim=1) * (0.01 * dimension**0.5)
    return features, labels, initial


def test_fit_spherical_probe_decreases_loss_and_preserves_exact_row_norm() -> None:
    features, labels, initial = _separable_probe_fixture()
    original_features = features.clone()
    original_initial = initial.clone()

    result = fit_spherical_probe(
        features,
        labels,
        initial,
        steps=64,
        batch_size=8,
    )

    assert result.steps == 64
    assert PROBE_LEARNING_RATE == 1e-4
    assert result.final_loss < result.initial_loss
    assert not torch.equal(result.head, initial)
    expected_norm = 0.01 * features.shape[1] ** 0.5
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.head, dim=1),
        torch.full((8,), expected_norm),
        rtol=2e-6,
        atol=2e-7,
    )
    assert torch.equal(features, original_features)
    assert torch.equal(initial, original_initial)
    cosine = torch.nn.functional.cosine_similarity(result.head, initial, dim=1)
    assert result.row_cosine_min == float(cosine.min())
    assert result.row_cosine_p05 == float(torch.quantile(cosine.double(), 0.05))
    assert result.row_cosine_median == float(torch.quantile(cosine.double(), 0.5))
    assert result.row_cosine_mean == float(cosine.double().mean())


def test_fit_spherical_probe_uses_independent_diagnostic_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.unicom_probe as probe_module

    features, labels, initial = _separable_probe_fixture()
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = probe_module.sharded_mask_arcface_loss

    def capture(
        batch_features: torch.Tensor,
        head: torch.Tensor,
        batch_labels: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((batch_features.detach().clone(), masks.detach().clone()))
        return original(batch_features, head, batch_labels, masks)

    monkeypatch.setattr(probe_module, "sharded_mask_arcface_loss", capture)
    fit_spherical_probe(features, labels, initial, steps=1, batch_size=8)

    assert len(calls) == 3
    assert not torch.equal(calls[0][0], calls[1][0])
    assert not torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(calls[0][0], calls[2][0])
    assert torch.equal(calls[0][1], calls[2][1])


def test_fit_spherical_probe_is_byte_deterministic_for_registered_streams() -> None:
    features, labels, initial = _separable_probe_fixture()

    first = fit_spherical_probe(features, labels, initial, steps=8, batch_size=8)
    second = fit_spherical_probe(features, labels, initial, steps=8, batch_size=8)

    assert torch.equal(first.head, second.head)
    assert first.initial_loss == second.initial_loss
    assert first.final_loss == second.final_loss
    assert first.steps == second.steps


@pytest.mark.parametrize(
    "mutation", ("feature_dtype", "label_dtype", "zero_row", "nan", "empty_class")
)
def test_fit_spherical_probe_rejects_invalid_tensor_contract(mutation: str) -> None:
    features, labels, initial = _separable_probe_fixture()
    if mutation == "feature_dtype":
        features = features.double()
    elif mutation == "label_dtype":
        labels = labels.int()
    elif mutation == "zero_row":
        initial[0].zero_()
    elif mutation == "nan":
        features[0, 0] = float("nan")
    elif mutation == "empty_class":
        labels[labels == 7] = 6

    with pytest.raises((TypeError, ValueError)):
        fit_spherical_probe(features, labels, initial, steps=2, batch_size=8)


def test_evaluate_probe_heads_uses_margin_free_accuracy_and_retains_paired_losses() -> None:
    features, labels, initial = _separable_probe_fixture()
    probe = torch.roll(initial, shifts=-1, dims=0).contiguous()
    represented = tuple(index % 2 == 0 for index in range(labels.numel()))

    actual = evaluate_probe_heads(
        features,
        labels,
        {"class_mean": initial, "spherical_probe": probe},
        validation_group_represented=represented,
        mask_sets=2,
    )

    assert tuple(actual) == ("class_mean", "spherical_probe")
    assert all(type(value) is ProbeMetrics for value in actual.values())
    expected: dict[str, tuple[tuple[float, ...], int, float, float]] = {}
    generator = torch.Generator().manual_seed(23_003)
    per_mask = {"class_mean": [], "spherical_probe": []}
    correct = {"class_mean": 0, "spherical_probe": 0}
    represented_loss = {"class_mean": 0.0, "spherical_probe": 0.0}
    unrepresented_loss = {"class_mean": 0.0, "spherical_probe": 0.0}
    represented_tensor = torch.tensor(represented, dtype=torch.bool)
    for _index in range(2):
        masks = sample_shard_masks(
            dimension=768,
            selected=512,
            shards=8,
            generator=generator,
            device=torch.device("cpu"),
        )
        for name, head in {"class_mean": initial, "spherical_probe": probe}.items():
            logits = sharded_mask_arcface_logits(features, head, labels, masks)
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            per_mask[name].append(float(losses.double().mean()))
            represented_loss[name] += float(losses[represented_tensor].double().sum())
            unrepresented_loss[name] += float(losses[~represented_tensor].double().sum())
            margin_free = sharded_mask_arcface_logits(
                features, head, labels, masks, margin=0.0
            )
            correct[name] += int(torch.count_nonzero(margin_free.argmax(dim=1) == labels))
    observations = labels.numel() * 2
    represented_observations = int(represented_tensor.sum()) * 2
    unrepresented_observations = int((~represented_tensor).sum()) * 2
    for name in per_mask:
        expected[name] = (
            tuple(per_mask[name]),
            correct[name],
            represented_loss[name] / represented_observations,
            unrepresented_loss[name] / unrepresented_observations,
        )

    for name in actual:
        assert actual[name].per_mask_mean_losses == expected[name][0]
        assert actual[name].mean_loss == sum(expected[name][0]) / 2
        assert actual[name].correct_count == expected[name][1]
        assert actual[name].observation_count == observations
        assert actual[name].accuracy == expected[name][1] / observations
        assert actual[name].represented_mean_loss == expected[name][2]
        assert actual[name].unrepresented_mean_loss == expected[name][3]


def test_evaluate_probe_heads_allows_singleton_classes_absent_from_validation() -> None:
    features, labels, initial = _separable_probe_fixture()
    keep = labels != 7
    validation_features = features[keep].contiguous()
    validation_labels = labels[keep].contiguous()
    represented = tuple(index % 2 == 0 for index in range(validation_labels.numel()))

    actual = evaluate_probe_heads(
        validation_features,
        validation_labels,
        {"class_mean": initial, "spherical_probe": initial.clone()},
        validation_group_represented=represented,
        mask_sets=2,
    )

    assert actual["class_mean"].observation_count == validation_labels.numel() * 2


def test_compare_probe_gradients_matches_independent_autograd_oracle() -> None:
    from sfora.unicom_training import experiment_stream_seed, padded_epoch_indices

    features, labels, initial = _separable_probe_fixture()
    probe = torch.roll(initial, shifts=-1, dims=0).contiguous()

    actual = compare_probe_gradients(
        features,
        labels,
        {"class_mean": initial, "spherical_probe": probe},
        fit_seed=2,
        batch_size=8,
    )

    indices = padded_epoch_indices(
        size=features.shape[0],
        global_batch=8,
        epoch=0,
        seed=experiment_stream_seed(2, 23_005),
        shards=8,
    )[:8]
    batch_labels = labels[list(indices)]
    generator = torch.Generator().manual_seed(experiment_stream_seed(2, 23_005))
    masks = sample_shard_masks(
        dimension=768,
        selected=512,
        shards=8,
        generator=generator,
        device=torch.device("cpu"),
    )
    gradients = []
    for head in (initial, probe):
        batch = features[list(indices)].detach().clone().requires_grad_(True)
        logits = sharded_mask_arcface_logits(batch, head, batch_labels, masks)
        loss = torch.nn.functional.cross_entropy(logits, batch_labels)
        gradients.append(torch.autograd.grad(loss, batch)[0])
    mean_gradient, probe_gradient = gradients
    per_row = torch.nn.functional.cosine_similarity(mean_gradient, probe_gradient, dim=1)

    assert type(actual) is ProbeGradientMetrics
    assert actual.class_mean_l2 == float(torch.linalg.vector_norm(mean_gradient).double())
    assert actual.spherical_probe_l2 == float(torch.linalg.vector_norm(probe_gradient).double())
    assert actual.relative_l2_difference == float(
        torch.linalg.vector_norm(probe_gradient - mean_gradient).double()
        / torch.linalg.vector_norm(mean_gradient).double()
    )
    assert actual.cosine_min == float(per_row.double().min())
    assert actual.cosine_p05 == float(torch.quantile(per_row.double(), 0.05))
    assert actual.cosine_median == float(torch.quantile(per_row.double(), 0.5))
    assert actual.cosine_mean == float(per_row.double().mean())


def _metrics(loss: float, accuracy: float, *, count: int = 100) -> ProbeMetrics:
    return ProbeMetrics(
        mean_loss=loss,
        accuracy=accuracy,
        correct_count=round(accuracy * count),
        observation_count=count,
        per_mask_mean_losses=tuple(loss for _ in range(64)),
        represented_mean_loss=loss,
        unrepresented_mean_loss=loss,
    )


def _metrics_from_deltas(deltas: tuple[float, ...]) -> ProbeMetrics:
    losses = tuple(1.0 - delta for delta in deltas)
    return ProbeMetrics(
        mean_loss=math.fsum(losses) / len(losses),
        accuracy=0.8,
        correct_count=80,
        observation_count=100,
        per_mask_mean_losses=losses,
        represented_mean_loss=0.9,
        unrepresented_mean_loss=0.9,
    )


def _fit() -> ProbeFit:
    return ProbeFit(
        head=torch.ones(8, 768, dtype=torch.float32),
        initial_loss=2.0,
        final_loss=1.9,
        steps=512,
        row_cosine_min=0.8,
        row_cosine_p05=0.9,
        row_cosine_median=0.97,
        row_cosine_mean=0.95,
    )


def _gradient() -> ProbeGradientMetrics:
    return ProbeGradientMetrics(
        class_mean_l2=1.0,
        spherical_probe_l2=1.0,
        relative_l2_difference=0.1,
        cosine_min=0.98,
        cosine_p05=0.985,
        cosine_median=0.99,
        cosine_mean=0.99,
    )


def test_probe_decision_promotes_at_registered_inclusive_boundaries() -> None:
    decision = probe_decision(
        class_mean=_metrics(1.0, 0.8),
        probe_fits={seed: _fit() for seed in range(3)},
        probe_metrics={seed: _metrics(0.9, 0.8) for seed in range(3)},
        probe_gradients={seed: _gradient() for seed in range(3)},
    )

    assert decision.status == "PROMOTE"
    assert tuple(decision.per_seed_predicates) == (0, 1, 2)
    assert all(all(values.values()) for values in decision.per_seed_predicates.values())
    assert all(
        type(value) is ProbeSeedStatistics
        for value in decision.per_seed_statistics.values()
    )
    assert decision.per_seed_statistics[0] == ProbeSeedStatistics(
        mean_paired_loss_delta=pytest.approx(0.1),
        paired_95_lower_bound=pytest.approx(0.1),
        positive_mask_count=64,
        accuracy_delta=0.0,
        represented_loss_delta=pytest.approx(0.1),
        unrepresented_loss_delta=pytest.approx(0.1),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "fit",
        "loss",
        "paired_ci",
        "majority",
        "accuracy",
        "represented",
        "unrepresented",
        "head_min",
        "head_mean",
        "gradient",
        "cosine",
    ),
)
def test_probe_decision_closes_each_failed_scientific_predicate(mutation: str) -> None:
    fits = {seed: _fit() for seed in range(3)}
    metrics = {seed: _metrics(0.9, 0.8) for seed in range(3)}
    gradients = {seed: _gradient() for seed in range(3)}
    arguments = {
        "class_mean": _metrics(1.0, 0.8),
        "probe_fits": fits,
        "probe_metrics": metrics,
        "probe_gradients": gradients,
    }
    if mutation == "fit":
        fits[1] = replace(fits[1], final_loss=fits[1].initial_loss)
    elif mutation == "loss":
        metrics[1] = _metrics(1.0, 0.8)
    elif mutation == "paired_ci":
        metrics[1] = _metrics_from_deltas((0.2,) * 48 + (-0.5,) * 16)
    elif mutation == "majority":
        metrics[1] = _metrics_from_deltas((0.2,) * 47 + (-0.01,) * 17)
    elif mutation == "accuracy":
        metrics[1] = _metrics(0.9, 0.79)
    elif mutation == "represented":
        value = _metrics(0.9, 0.8)
        metrics[1] = replace(value, represented_mean_loss=1.0)
    elif mutation == "unrepresented":
        value = _metrics(0.9, 0.8)
        metrics[1] = replace(value, unrepresented_mean_loss=1.0)
    elif mutation == "head_min":
        fits[1] = replace(_fit(), row_cosine_min=0.799)
    elif mutation == "head_mean":
        fits[1] = replace(_fit(), row_cosine_mean=0.949)
    elif mutation == "gradient":
        gradients[1] = replace(_gradient(), relative_l2_difference=0.049)
    elif mutation == "cosine":
        gradients[1] = replace(_gradient(), cosine_median=0.996)

    assert probe_decision(**arguments).status == "CLOSE_DIRECTION"
