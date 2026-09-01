"""Analytic collapsed-verdict control fields for eight-rollout ASG-CV."""

from __future__ import annotations

import math

import numpy as np
import torch

ASGCV_VERDICT_GROUP_SIZE = 8


def collapsed_verdict_probability(correct_score: float, incorrect_score: float) -> float:
    """Normalize two finite teacher-forced branch scores without overflow."""

    if (
        type(correct_score) is not float
        or not math.isfinite(correct_score)
        or type(incorrect_score) is not float
        or not math.isfinite(incorrect_score)
    ):
        raise ValueError("ASG-CV collapsed verdict score differs")
    gap = correct_score - incorrect_score
    if gap >= 0.0:
        probability = 1.0 / (1.0 + math.exp(-gap))
    else:
        exponent = math.exp(gap)
        probability = exponent / (1.0 + exponent)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("ASG-CV collapsed verdict probability differs")
    return probability


def collapsed_verdict_coefficient(probability: float) -> float:
    """Return E[sqrt(M(K-M))/K] for M ~ Binomial(K, probability)."""

    if (
        type(probability) is not float
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("ASG-CV collapsed verdict probability differs")
    group_size = ASGCV_VERDICT_GROUP_SIZE
    coefficient = math.fsum(
        math.comb(group_size, correct)
        * probability**correct
        * (1.0 - probability) ** (group_size - correct)
        * math.sqrt(correct * (group_size - correct))
        / group_size
        for correct in range(1, group_size)
    )
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("ASG-CV collapsed verdict coefficient differs")
    return coefficient


def collapsed_grpo_verdict_field(
    probability: float,
    score_gap_gradient: np.ndarray,
) -> np.ndarray:
    """Return the exact binary-verdict marginal of population-normalized GRPO."""

    if (
        type(score_gap_gradient) is not np.ndarray
        or score_gap_gradient.dtype != np.dtype(np.float64)
        or score_gap_gradient.ndim == 0
        or not bool(np.isfinite(score_gap_gradient).all())
    ):
        raise ValueError("ASG-CV collapsed verdict float64 gradient differs")
    result = -collapsed_verdict_coefficient(probability) * score_gap_gradient
    if not bool(np.isfinite(result).all()):
        raise ValueError("ASG-CV collapsed verdict gradient differs")
    return np.ascontiguousarray(result, dtype=np.float64)


def torch_collapsed_grpo_verdict_loss(
    correct_score: torch.Tensor,
    incorrect_score: torch.Tensor,
) -> torch.Tensor:
    """Build a scalar whose gradient is the detached collapsed-verdict field."""

    if (
        type(correct_score) is not torch.Tensor
        or type(incorrect_score) is not torch.Tensor
        or correct_score.ndim != 0
        or incorrect_score.ndim != 0
        or correct_score.dtype != torch.float32
        or incorrect_score.dtype != torch.float32
        or correct_score.device != incorrect_score.device
    ):
        raise ValueError("ASG-CV collapsed verdict scalar score differs")
    if not bool(torch.isfinite(correct_score)) or not bool(torch.isfinite(incorrect_score)):
        raise ValueError("ASG-CV collapsed verdict score is not finite")
    probability = collapsed_verdict_probability(
        float(correct_score.detach()),
        float(incorrect_score.detach()),
    )
    coefficient = collapsed_verdict_coefficient(probability)
    loss = -coefficient * (correct_score - incorrect_score)
    if not bool(torch.isfinite(loss)):
        raise ValueError("ASG-CV collapsed verdict loss is not finite")
    return loss
