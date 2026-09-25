"""Live-head ranking must train the head through cached candidate rows."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sfora.deployed_code_rank import smooth_ap_bank_loss
from sfora.live_head_bank import live_head_bank_loss


def test_live_head_matches_detached_bank_forward_but_adds_candidate_gradient() -> None:
    generator = torch.Generator().manual_seed(179023)
    source = F.normalize(torch.randn(6, 1024, generator=generator), dim=1)
    head = nn.Linear(1024, 128)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * 0.03)
        head.bias.copy_(torch.randn(head.bias.shape, generator=generator) * 0.01)
    anchors = F.normalize(torch.randn(2, 128, generator=generator), dim=1)
    positives = torch.tensor([[1], [0]], dtype=torch.long)
    self_rows = torch.tensor([0, 1], dtype=torch.long)

    actual = live_head_bank_loss(anchors, source, head, positives, self_rows)
    projected = F.normalize(head(source), dim=1)
    expected = smooth_ap_bank_loss(anchors, projected, positives, self_rows)
    assert actual.item() == pytest.approx(expected.item(), abs=1e-6)

    candidate_gradient = torch.autograd.grad(actual, head.weight)[0]
    assert not expected.requires_grad
    assert torch.isfinite(candidate_gradient).all()
    assert candidate_gradient.abs().sum() > 0


def test_live_head_keeps_cached_source_detached() -> None:
    generator = torch.Generator().manual_seed(179024)
    source = F.normalize(torch.randn(6, 1024, generator=generator), dim=1)
    source.requires_grad_()
    head = nn.Linear(1024, 128)
    anchors = F.normalize(torch.randn(2, 128, generator=generator), dim=1)
    loss = live_head_bank_loss(
        anchors,
        source,
        head,
        torch.tensor([[1], [0]], dtype=torch.long),
        torch.tensor([0, 1], dtype=torch.long),
    )
    loss.backward()
    assert source.grad is None
    assert head.weight.grad is not None
