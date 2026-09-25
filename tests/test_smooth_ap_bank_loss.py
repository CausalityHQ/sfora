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


def test_bank_loss_gradient_matches_independent_detached_gallery_reference() -> None:
    torch.manual_seed(179021)
    bank = F.normalize(torch.randn(8, 128, dtype=torch.float32), dim=1).requires_grad_()
    with torch.no_grad():
        bank[3].copy_(bank[2])  # tied scores must not change the formula
    anchors = F.normalize(torch.randn(2, 128, dtype=torch.float32), dim=1).requires_grad_()
    positives = torch.tensor([[1, 2], [4, -1]])
    self_rows = torch.tensor([0, 3])
    actual = smooth_ap_bank_loss(anchors, bank, positives, self_rows)

    reference_precisions = []
    for i in range(2):
        scores = anchors[i] @ bank.detach().T
        member_rows = [int(row) for row in positives[i] if row >= 0]
        terms = []
        for positive in member_rows:
            denominator = 1.0 + sum(
                torch.sigmoid((scores[other] - scores[positive]) * 200.0)
                for other in range(8)
                if other != int(self_rows[i]) and other != positive
            )
            numerator = 1.0 + sum(
                torch.sigmoid((scores[other] - scores[positive]) * 200.0)
                for other in member_rows
                if other != positive
            )
            terms.append(numerator / denominator)
        reference_precisions.append(sum(terms) / len(terms))
    expected = 1.0 - sum(reference_precisions) / len(reference_precisions)
    actual_grad = torch.autograd.grad(actual, anchors, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, anchors)[0]
    assert actual.item() == pytest.approx(expected.item(), abs=1e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-5, rtol=1e-5)
    actual.backward()
    assert bank.grad is None
