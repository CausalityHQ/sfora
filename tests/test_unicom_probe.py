from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_probe import class_mean_head, fit_spherical_probe, split_probe_records


def _record(label: str, name: str) -> InshopRecord:
    return InshopRecord(split="train", image_path=Path("/dataset") / name, label=label)


def test_split_probe_records_uses_last_sorted_path_per_class_for_validation() -> None:
    records = (
        _record("b", "b-2.jpg"),
        _record("a", "a-3.jpg"),
        _record("a", "a-1.jpg"),
        _record("b", "b-1.jpg"),
        _record("a", "a-2.jpg"),
    )

    fitting, validation = split_probe_records(records, {"a": 0, "b": 1})

    assert fitting == (
        _record("a", "a-1.jpg"),
        _record("a", "a-2.jpg"),
        _record("b", "b-1.jpg"),
    )
    assert validation == (_record("a", "a-3.jpg"), _record("b", "b-2.jpg"))


@pytest.mark.parametrize(
    ("records", "labels"),
    (
        ((_record("a", "a-1.jpg"),), {"a": 0}),
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
