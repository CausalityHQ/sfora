"""Checks for the differentiable finite-gallery sampling objective."""

from __future__ import annotations

import math

import pytest
import torch

from sfora.gallery_risk import finite_gallery_log_success, finite_gallery_success


def test_finite_gallery_success_matches_hypergeometric_integer_counts() -> None:
    counts = torch.tensor([0.0, 1.0, 3.0, 9.0], dtype=torch.float64)
    actual = finite_gallery_success(counts, pool_size=10, sample_size=4)
    expected = [
        math.comb(10 - int(count), 4) / math.comb(10, 4) if count <= 6 else 0.0 for count in counts
    ]
    assert actual.tolist() == pytest.approx(expected, abs=1e-13)


def test_finite_gallery_success_has_saturating_gradient() -> None:
    counts = torch.tensor([0.25, 5.0, 40.0], dtype=torch.float32, requires_grad=True)
    success = finite_gallery_success(counts, pool_size=100, sample_size=20)
    success.sum().backward()
    assert bool(torch.isfinite(success).all())
    assert counts.grad is not None
    assert bool(torch.isfinite(counts.grad).all())
    assert bool((counts.grad < 0).all())
    assert abs(float(counts.grad[0])) > abs(float(counts.grad[-1]))


@pytest.mark.parametrize("pool,sample", [(0, 1), (3, 0), (3, 4)])
def test_finite_gallery_success_rejects_invalid_sizes(pool: int, sample: int) -> None:
    with pytest.raises(ValueError, match="gallery sampling"):
        finite_gallery_success(torch.tensor([0.0]), pool_size=pool, sample_size=sample)


def test_finite_gallery_success_clamps_impossible_sample_without_nan() -> None:
    counts = torch.tensor([7.0, 8.0], requires_grad=True)
    success = finite_gallery_success(counts, pool_size=10, sample_size=4)
    assert success.tolist() == [0.0, 0.0]
    success.sum().backward()
    assert counts.grad is not None
    assert counts.grad.tolist() == [0.0, 0.0]


def test_log_success_retains_gradient_after_probability_underflows() -> None:
    counts = torch.tensor([1000.0], requires_grad=True)
    log_success = finite_gallery_log_success(counts, pool_size=53_700, sample_size=5_000)
    assert bool(torch.isfinite(log_success).all())
    assert float(log_success.detach()[0]) < -90
    (-log_success).sum().backward()
    assert counts.grad is not None
    assert bool(torch.isfinite(counts.grad).all())
    assert float(counts.grad[0]) > 0
