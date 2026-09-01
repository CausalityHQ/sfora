from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from sfora.asgcv_verdict_marginal import (
    collapsed_grpo_verdict_field,
    collapsed_verdict_coefficient,
    collapsed_verdict_probability,
    torch_collapsed_grpo_verdict_loss,
)


def _exhaustive_binary_grpo_field(probability: float, score_gap_gradient: np.ndarray) -> np.ndarray:
    total = np.zeros_like(score_gap_gradient, dtype=np.float64)
    group_size = 8
    for rewards in itertools.product((0, 1), repeat=group_size):
        correct = sum(rewards)
        event_probability = probability**correct * (1.0 - probability) ** (
            group_size - correct
        )
        if correct in {0, group_size}:
            continue
        mean = correct / group_size
        scale = math.sqrt(mean * (1.0 - mean))
        gradient = np.zeros_like(total)
        for reward in rewards:
            advantage = (reward - mean) / scale
            score_gradient = (
                (1.0 - probability) * score_gap_gradient
                if reward == 1
                else -probability * score_gap_gradient
            )
            gradient -= advantage * score_gradient / group_size
        total += event_probability * gradient
    return total


@pytest.mark.parametrize("probability", [0.01, 0.2, 0.5, 0.83, 0.99])
def test_collapsed_verdict_field_matches_exhaustive_eight_rollout_expectation(
    probability: float,
) -> None:
    score_gap_gradient = np.array([[1.25, -0.5], [0.0, 2.0]], dtype=np.float64)
    expected = _exhaustive_binary_grpo_field(probability, score_gap_gradient)
    actual = collapsed_grpo_verdict_field(probability, score_gap_gradient)
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-14)
    assert np.allclose(
        actual,
        -collapsed_verdict_coefficient(probability) * score_gap_gradient,
        rtol=0.0,
        atol=0.0,
    )


def test_collapsed_verdict_field_is_zero_at_deterministic_outcomes_and_rejects_drift() -> None:
    gradient = np.ones((2, 3), dtype=np.float64)
    assert np.array_equal(collapsed_grpo_verdict_field(0.0, gradient), np.zeros_like(gradient))
    assert np.array_equal(collapsed_grpo_verdict_field(1.0, gradient), np.zeros_like(gradient))
    for probability in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="probability"):
            collapsed_verdict_coefficient(probability)
    with pytest.raises(ValueError, match="gradient"):
        collapsed_grpo_verdict_field(0.5, np.array([float("inf")], dtype=np.float64))
    with pytest.raises(ValueError, match="float64"):
        collapsed_grpo_verdict_field(0.5, gradient.astype(np.float32))


def test_collapsed_verdict_probability_is_stable_and_order_sensitive() -> None:
    assert collapsed_verdict_probability(0.0, 0.0) == 0.5
    high = collapsed_verdict_probability(1_000.0, -1_000.0)
    low = collapsed_verdict_probability(-1_000.0, 1_000.0)
    assert high == 1.0
    assert low == 0.0
    assert collapsed_verdict_probability(2.0, -0.5) == pytest.approx(
        1.0 - collapsed_verdict_probability(-0.5, 2.0), abs=3e-16
    )
    for scores in ((float("nan"), 0.0), (0.0, float("inf")), (0, 0.0)):
        with pytest.raises(ValueError, match="score"):
            collapsed_verdict_probability(*scores)


def test_torch_collapsed_loss_backpropagates_the_authoritative_detached_field() -> None:
    features = torch.tensor([0.3, -0.7, 1.2], dtype=torch.float32, requires_grad=True)
    correct_weights = torch.tensor([0.5, 0.2, -0.4], dtype=torch.float32)
    incorrect_weights = torch.tensor([-0.1, 0.6, 0.3], dtype=torch.float32)
    correct_score = torch.dot(features, correct_weights)
    incorrect_score = torch.dot(features, incorrect_weights)
    loss = torch_collapsed_grpo_verdict_loss(correct_score, incorrect_score)
    loss.backward()
    assert features.grad is not None

    probability = collapsed_verdict_probability(
        float(correct_score.detach()), float(incorrect_score.detach())
    )
    expected = collapsed_grpo_verdict_field(
        probability,
        (correct_weights - incorrect_weights).double().numpy(),
    )
    np.testing.assert_allclose(features.grad.double().numpy(), expected, rtol=2e-7, atol=1e-8)
    assert loss.dtype == torch.float32

    with pytest.raises(ValueError, match="scalar"):
        torch_collapsed_grpo_verdict_loss(features, incorrect_score)
    with pytest.raises(ValueError, match="finite"):
        torch_collapsed_grpo_verdict_loss(
            torch.tensor(float("nan")), torch.tensor(0.0)
        )
