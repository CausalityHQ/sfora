import torch
from probe_inshop_arcface_derivative import analytic_interior_arcface_logits

from sfora.unicom_training import sharded_mask_arcface_logits


def test_analytic_interior_derivative_matches_finite_difference_and_native_forward():
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(3, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    weights = torch.randn(5, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    labels = torch.tensor([0, 2, 4])
    masks = torch.arange(4).reshape(1, -1)

    def objective(x, w):
        return analytic_interior_arcface_logits(x, w, labels, masks, margin=0.3, scale=64)

    assert torch.equal(
        objective(features, weights),
        sharded_mask_arcface_logits(features, weights, labels, masks, margin=0.3, scale=64),
    )
    assert torch.autograd.gradcheck(objective, (features, weights), eps=1e-6, atol=1e-5)


def test_pole_policy_remains_finite_with_identical_forward():
    features = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    labels, masks = torch.tensor([0, 0]), torch.tensor([[0, 1]])
    logits = analytic_interior_arcface_logits(features, weights, labels, masks)
    assert torch.equal(
        logits, sharded_mask_arcface_logits(features, weights, labels, masks, margin=0.3, scale=64)
    )
    torch.nn.functional.cross_entropy(logits, labels).backward()
    assert torch.isfinite(features.grad).all() and torch.isfinite(weights.grad).all()
