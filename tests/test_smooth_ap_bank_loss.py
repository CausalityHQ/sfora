"""Exact small-gallery control for the full-bank SmoothAP surrogate."""

import pytest
import torch
from torch.nn import functional as F

from sfora.deployed_code_rank import smooth_ap_bank_loss, smooth_ap_float_loss


def test_bank_loss_matches_full_gallery_smooth_ap_forward() -> None:
    torch.manual_seed(179019)
    values = F.normalize(torch.randn(7, 128, dtype=torch.float32), dim=1)
    labels = [0, 0, 0, 1, 1, 2, 2]
    positives = torch.tensor([[1, 2], [0, 2], [0, 1], [4, -1], [3, -1], [6, -1], [5, -1]])
    actual = smooth_ap_bank_loss(values, values, positives, torch.arange(7))
    expected = smooth_ap_float_loss(values, labels)
    assert actual.item() == pytest.approx(expected.item(), abs=1e-6)


def test_bank_loss_backpropagates_through_anchors() -> None:
    torch.manual_seed(179020)
    bank = F.normalize(torch.randn(6, 128, dtype=torch.float32), dim=1)
    anchors = bank[:2].clone().requires_grad_()
    positives = torch.tensor([[1], [0]])
    loss = smooth_ap_bank_loss(anchors, bank, positives, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(anchors.grad).all()
    assert anchors.grad.abs().sum() > 0
