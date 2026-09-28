"""The discarded gradient component is orthogonal to the compact-head span."""

import torch
from probe_inshop_source_classifier import unused_gradient


def test_unused_gradient_removes_head_and_source_directions():
    weight = torch.tensor([[1.0, 0.0, 0.0]])
    unit = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    gradient = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]])
    result = unused_gradient(gradient, weight, unit)
    assert torch.equal(result[0], torch.tensor([0.0, 0.0, 4.0]))
    assert torch.equal(result[1], torch.tensor([0.0, 3.0, 4.0]))
    assert torch.equal(result @ weight.T, torch.zeros(2, 1))
    assert torch.equal((result * unit).sum(1), torch.zeros(2))
