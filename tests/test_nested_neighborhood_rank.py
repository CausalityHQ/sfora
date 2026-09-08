from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from sfora.nested_neighborhood_rank import (
    NestedRankConfig,
    NestedRankHead,
    asymmetric_neighborhood_loss,
    nested_proxy_anchor_loss,
    smooth_ap_loss,
)


def _unit(rows: int, dimensions: int) -> torch.Tensor:
    values = torch.arange(1, rows * dimensions + 1, dtype=torch.float32)
    return F.normalize(values.reshape(rows, dimensions), dim=1)


def test_config_and_head_emit_independently_normalized_prefixes() -> None:
    config = NestedRankConfig(
        input_dim=8,
        hidden_dim=16,
        output_dim=8,
        widths=(4, 8),
        class_count=4,
    )
    head = NestedRankHead(config)

    result = head(torch.randn(6, 8, dtype=torch.float32))

    assert tuple(result) == (4, 8)
    assert head.residual_scale.item() == 0.0
    for width, values in result.items():
        assert values.shape == (6, width)
        torch.testing.assert_close(values.norm(dim=1), torch.ones(6))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_dim", True),
        ("hidden_dim", 0),
        ("output_dim", 4),
        ("widths", (8, 4)),
        ("class_count", False),
    ],
)
def test_config_rejects_invalid_concrete_authority(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "input_dim": 8,
        "hidden_dim": 16,
        "output_dim": 8,
        "widths": (4, 8),
        "class_count": 4,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match="nested rank configuration"):
        NestedRankConfig(**kwargs)  # type: ignore[arg-type]


def test_head_rejects_shape_dtype_nonfinite_and_zero_norm_inputs() -> None:
    head = NestedRankHead(
        NestedRankConfig(input_dim=4, hidden_dim=8, output_dim=4, widths=(2, 4), class_count=2)
    )
    bad = [
        torch.ones(2, 3),
        torch.ones(2, 4, dtype=torch.float64),
        torch.tensor([[1.0, 2.0, 3.0, math.nan], [1.0, 2.0, 3.0, 4.0]]),
        torch.zeros(2, 4),
    ]
    for values in bad:
        with pytest.raises(ValueError, match="nested rank features"):
            head(values)


def test_proxy_anchor_matches_scalar_formula_and_shared_prefixes() -> None:
    raw = torch.tensor(
        [[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        requires_grad=True,
    )
    rows = F.normalize(torch.tensor([[2.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 1.0]]), dim=1)
    embeddings = {2: F.normalize(rows[:, :2], dim=1), 4: rows}
    labels = torch.tensor([0, 1], dtype=torch.int64)

    actual = nested_proxy_anchor_loss(
        embeddings,
        labels,
        raw,
        width_weights={2: 0.25, 4: 1.0},
        scale=2.0,
        margin=0.1,
    )

    expected = torch.zeros((), dtype=torch.float32)
    for width, weight in ((2, 0.25), (4, 1.0)):
        proxies = F.normalize(raw[:, :width], dim=1)
        scores = embeddings[width] @ proxies.T
        positive = torch.stack(
            [
                torch.log1p(torch.exp(-2.0 * (scores[index, label] - 0.1)))
                for index, label in enumerate(labels)
            ]
        ).mean()
        negative = torch.stack(
            [
                torch.log1p(torch.exp(2.0 * (scores[:, proxy] + 0.1))[labels != proxy].sum())
                for proxy in range(3)
            ]
        ).mean()
        expected = expected + weight * (positive + negative)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_asymmetric_neighborhood_masks_classes_and_detaches_targets_and_keys() -> None:
    queries = _unit(4, 4).detach().requires_grad_(True)
    keys = _unit(4, 4).detach().requires_grad_(True)
    teacher = torch.flip(_unit(4, 5), dims=(1,)).detach().requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    sample_ids = torch.tensor([10, 11, 20, 21], dtype=torch.int64)

    loss = asymmetric_neighborhood_loss(
        queries,
        keys,
        teacher,
        labels,
        sample_ids,
        temperature=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert queries.grad is not None and queries.grad.abs().sum() > 0
    assert keys.grad is None
    assert teacher.grad is None


def test_asymmetric_neighborhood_rejects_rows_without_off_class_keys() -> None:
    rows = _unit(3, 4)
    with pytest.raises(ValueError, match="neighborhood key inventory"):
        asymmetric_neighborhood_loss(
            rows,
            rows,
            _unit(3, 5),
            torch.zeros(3, dtype=torch.int64),
            torch.arange(3, dtype=torch.int64),
            temperature=0.1,
        )


def test_smooth_ap_matches_scalar_reference_and_has_finite_gradients() -> None:
    embeddings = _unit(4, 3).detach().requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    sample_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    temperature = 0.2

    actual = smooth_ap_loss(embeddings, labels, sample_ids, temperature=temperature)

    distances = 1.0 - embeddings @ embeddings.T
    aps: list[torch.Tensor] = []
    for query in range(4):
        valid = sample_ids != sample_ids[query]
        positives = valid & (labels == labels[query])
        terms: list[torch.Tensor] = []
        for positive in torch.where(positives)[0]:
            comparisons = torch.sigmoid(
                (distances[query, positive] - distances[query, valid]) / temperature
            )
            full_rank = 0.5 + comparisons.sum()
            positive_rank = 0.5 + comparisons[positives[valid]].sum()
            terms.append(positive_rank / full_rank)
        aps.append(torch.stack(terms).mean())
    expected = 1.0 - torch.stack(aps).mean()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


def test_losses_reject_nonfinite_temperature_shape_and_positive_inventory() -> None:
    rows = _unit(3, 4)
    labels = torch.tensor([0, 1, 2], dtype=torch.int64)
    ids = torch.arange(3, dtype=torch.int64)
    with pytest.raises(ValueError):
        nested_proxy_anchor_loss({4: rows}, labels[:2], rows)
    with pytest.raises(ValueError):
        asymmetric_neighborhood_loss(rows, rows, rows, labels, ids, temperature=math.nan)
    with pytest.raises(ValueError, match="Smooth-AP positive inventory"):
        smooth_ap_loss(rows, labels, ids, temperature=0.1)
