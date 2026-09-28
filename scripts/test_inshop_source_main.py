import torch
import train_inshop_siglip2_unseen_gallery as trainer
from torch import nn
from torch.nn import functional as F

from sfora.unicom_training import sharded_mask_arcface_loss


def test_source_main_preserves_native_gradients_and_inactive_head():
    torch.manual_seed(13)
    source = torch.randn(4, 1024, requires_grad=True)
    mean = torch.randn(1024, requires_grad=True) * 0.01
    proxy = nn.Parameter(torch.randn(3, 1024))
    head = nn.Linear(1024, 128)
    labels = torch.tensor([0, 1, 2, 0])
    actual = trainer.source_main_arcface_loss(source, mean, proxy, labels)
    expected = sharded_mask_arcface_loss(
        F.normalize(source, dim=1) - mean.detach(),
        proxy,
        labels,
        torch.arange(1024).unsqueeze(0),
        margin=0.3,
        scale=64,
    )
    actual_grad = torch.autograd.grad(
        actual,
        (source, proxy, mean, head.weight, head.bias),
        allow_unused=True,
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(expected, (source, proxy))
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grad[:2], expected_grad, strict=True))
    assert actual_grad[2:] == (None, None, None)
    assert all(g.isfinite().all() and g.norm() > 0 for g in actual_grad[:2])
    optimizer = torch.optim.AdamW(list(head.parameters()) + [proxy], lr=1e-4)
    head(source.detach()).square().mean().backward()
    optimizer.step()
    before = {key: value.clone() for key, value in head.state_dict().items()}
    states = [
        {key: value.clone() for key, value in optimizer.state[p].items()} for p in head.parameters()
    ]
    optimizer.zero_grad(set_to_none=True)
    trainer.source_main_arcface_loss(source, mean, proxy, labels).backward()
    assert trainer.gradient_norm(head.parameters()) == 0
    optimizer.step()
    assert all(torch.equal(value, before[key]) for key, value in head.state_dict().items())
    assert all(
        torch.equal(value, states[i][key])
        for i, p in enumerate(head.parameters())
        for key, value in optimizer.state[p].items()
    )
