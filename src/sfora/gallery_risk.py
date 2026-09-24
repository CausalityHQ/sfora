"""Finite-gallery success surrogate for training compact retrieval embeddings."""

from __future__ import annotations

import torch


def finite_gallery_success(
    soft_outranking_count: torch.Tensor, *, pool_size: int, sample_size: int
) -> torch.Tensor:
    """Estimate P(recall@1) for a uniform gallery sampled without replacement.

    ``soft_outranking_count`` may be a differentiable approximation to the
    number of wrong items beating a query's best positive. Integer inputs
    return the exact hypergeometric probability. Values for which fewer than
    ``sample_size`` non-outranking items remain have probability zero. A
    float64 log-gamma calculation preserves small probability differences
    when the negative pool contains tens of thousands of items.
    """

    return finite_gallery_log_success(
        soft_outranking_count, pool_size=pool_size, sample_size=sample_size
    ).exp()


def finite_gallery_log_success(
    soft_outranking_count: torch.Tensor, *, pool_size: int, sample_size: int
) -> torch.Tensor:
    """Log of finite-gallery success, preserving gradients at tiny probabilities."""

    if (
        type(pool_size) is not int
        or type(sample_size) is not int
        or pool_size < 1
        or not 1 <= sample_size <= pool_size
        or type(soft_outranking_count) is not torch.Tensor
        or soft_outranking_count.ndim != 1
        or soft_outranking_count.numel() < 1
        or soft_outranking_count.dtype not in (torch.float32, torch.float64)
        or not bool(torch.isfinite(soft_outranking_count).all())
        or bool((soft_outranking_count < 0).any())
        or bool((soft_outranking_count > pool_size).any())
    ):
        raise ValueError("gallery sampling inventory differs")
    counts = soft_outranking_count.double()
    feasible = counts <= pool_size - sample_size
    good = pool_size - counts.clamp(max=pool_size - sample_size)
    log_probability = (
        torch.lgamma(good + 1)
        - torch.lgamma(good - sample_size + 1)
        - torch.lgamma(torch.tensor(pool_size + 1.0, dtype=counts.dtype, device=counts.device))
        + torch.lgamma(
            torch.tensor(pool_size - sample_size + 1.0, dtype=counts.dtype, device=counts.device)
        )
    )
    log_probability = torch.where(feasible, log_probability, -torch.inf)
    return log_probability.to(dtype=soft_outranking_count.dtype)


__all__ = ["finite_gallery_log_success", "finite_gallery_success"]
