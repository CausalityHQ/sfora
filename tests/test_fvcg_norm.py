from __future__ import annotations

import math

import pytest
import torch

from sfora.fvcg_norm import combine_norm_stabilized_gradients


def _flat(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat(tuple(value.flatten() for value in values))


def test_norm_stabilized_field_is_invariant_to_positive_semantic_scale() -> None:
    dml = (torch.tensor([3.0, 4.0]),)
    semantic = (torch.tensor([0.0, 2.0]),)

    small = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)
    large = combine_norm_stabilized_gradients(
        dml, tuple(value * 1_000_000.0 for value in semantic), rho=0.25
    )

    assert torch.equal(_flat(small.gradients), _flat(large.gradients))
    assert small.dml_norm == pytest.approx(5.0)
    assert small.applied_semantic_norm == pytest.approx(1.25)
    assert small.applied_to_dml_ratio_ppm == 250_000


def test_norm_stabilized_field_removes_only_conflicting_component() -> None:
    dml = (torch.tensor([2.0, 0.0]),)
    semantic = (torch.tensor([-3.0, 4.0]),)

    result = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)

    assert result.raw_dot == pytest.approx(-6.0)
    assert result.projected_dot == pytest.approx(0.0, abs=1.0e-7)
    assert result.safe_semantic_norm == pytest.approx(4.0)
    assert torch.allclose(result.gradients[0], torch.tensor([2.0, 0.5]))
    assert 5_000 <= result.combined_cosine_distance_ppm <= 50_000


def test_norm_stabilized_field_retains_nonconflicting_parallel_component() -> None:
    result = combine_norm_stabilized_gradients(
        (torch.tensor([2.0, 0.0]),),
        (torch.tensor([3.0, 4.0]),),
        rho=0.25,
    )

    assert result.raw_dot == pytest.approx(6.0)
    assert result.projected_dot == pytest.approx(6.0)
    assert torch.allclose(result.gradients[0], torch.tensor([2.3, 0.4]))


@pytest.mark.parametrize(
    ("dml", "semantic", "rho"),
    [
        ((torch.zeros(2),), (torch.ones(2),), 0.25),
        ((torch.ones(2),), (torch.zeros(2),), 0.25),
        ((torch.ones(2),), (torch.tensor([math.nan, 1.0]),), 0.25),
        ((torch.ones(2),), (torch.ones(3),), 0.25),
        ((torch.ones(2),), (torch.ones(2),), 0.0),
    ],
)
def test_norm_stabilized_field_fails_closed(
    dml: tuple[torch.Tensor, ...],
    semantic: tuple[torch.Tensor, ...],
    rho: float,
) -> None:
    with pytest.raises(ValueError, match="FVCG-Norm"):
        combine_norm_stabilized_gradients(dml, semantic, rho=rho)


def test_norm_stabilized_field_is_deterministic_and_fp32() -> None:
    dml = (torch.tensor([1.0, 2.0], dtype=torch.bfloat16),)
    semantic = (torch.tensor([2.0, -0.5], dtype=torch.bfloat16),)

    first = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)
    second = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)

    assert first == second
    assert first.gradients[0].dtype == torch.float32
