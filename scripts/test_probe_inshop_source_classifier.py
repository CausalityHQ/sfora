"""The discarded gradient component is orthogonal to the compact-head span."""

import torch
from probe_inshop_source_classifier import prototype_scores, unused_gradient


def test_unused_gradient_removes_head_and_source_directions():
    weight = torch.tensor([[1.0, 0.0, 0.0]])
    unit = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    gradient = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    result = unused_gradient(gradient, weight, unit)
    assert torch.equal(result[0], torch.tensor([0.0, 0.0, 4.0]))
    assert torch.equal(result[1], torch.tensor([0.0, 3.0, 4.0]))
    assert torch.equal(result @ weight.T, torch.zeros(2, 1))
    assert torch.equal((result * unit).sum(1), torch.zeros(2))


def test_prototype_scores_remove_query_from_positive():
    fit = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    scores = prototype_scores(
        fit[:1], fit, torch.tensor([0, 0, 1]), torch.tensor([0]), torch.tensor([0])
    )
    assert torch.equal(scores, torch.tensor([[0.0, 1.0]]))


def test_qr_projection_for_trained_nonorthonormal_head():
    weight = torch.tensor([[2.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    basis = torch.linalg.qr(weight.T, mode="reduced").Q.T
    unit = torch.tensor([[1.0, 0.0, 0.0]])
    result = unused_gradient(torch.tensor([[2.0, 3.0, 4.0]]), basis, unit)
    assert torch.allclose(result, torch.tensor([[0.0, 0.0, 4.0]]), atol=1e-6)
    assert torch.allclose(result @ weight.T, torch.zeros(1, 2), atol=1e-6)
