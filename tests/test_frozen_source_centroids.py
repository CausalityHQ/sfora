import torch
from torch.nn import functional as F

from sfora.sop_compact_training import frozen_source_centroid_loss
from sfora.unicom_training import sharded_mask_arcface_loss


def test_frozen_source_loss_native_parity_and_gradient_routing():
    torch.manual_seed(7)
    source = torch.randn(4, 6, requires_grad=True)
    mean = torch.randn(6, requires_grad=True)
    prototypes = torch.randn(3, 6, requires_grad=True)
    head = torch.nn.Linear(6, 2)
    target = torch.tensor([0, 1, 2, 0])
    head(source)  # Main-path parameters must receive no auxiliary gradient.
    loss = frozen_source_centroid_loss(source, mean, prototypes, target)
    expected = sharded_mask_arcface_loss(
        F.normalize(source, dim=1) - mean.detach(),
        prototypes.detach(),
        target,
        torch.arange(6).unsqueeze(0),
        margin=0,
        scale=64,
    )
    assert torch.allclose(loss, expected)
    actual_grad = torch.autograd.grad(loss, source, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, source)[0]
    assert torch.allclose(actual_grad, expected_grad)
    loss.backward()
    assert source.grad is not None and source.grad.norm() > 0
    assert mean.grad is None and prototypes.grad is None and head.weight.grad is None
