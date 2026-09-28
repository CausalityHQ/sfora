"""Soft semantic targets have an attainable optimum and ignore missing captions."""

import pytest
import torch

from sfora.frozen_text_supervision import frozen_text_loss


def test_text_target_is_stationary_at_its_own_prototype():
    text = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]]), dim=1)
    source = text[[0]].clone().requires_grad_()
    loss = frozen_text_loss(source, text, torch.tensor([0]))
    loss.backward()
    assert torch.isfinite(loss)
    assert source.grad.norm() < 1e-5


def test_missing_caption_masks_only_auxiliary_rows():
    text = torch.eye(2)
    source = torch.tensor([[0.8, 0.2], [0.2, 0.8]], requires_grad=True)
    loss = frozen_text_loss(source, text, torch.tensor([0, -1]))
    assert loss == frozen_text_loss(source[:1], text, torch.tensor([0]))
    loss.backward()
    assert torch.equal(source.grad[1], torch.zeros(2))
    with pytest.raises(ValueError):
        frozen_text_loss(source, text, torch.tensor([-1, -1]))


def test_soft_teacher_remains_fp32_inside_autocast():
    text = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]]), dim=1)
    source = text[[0]].clone().requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = frozen_text_loss(source, text, torch.tensor([0]))
    loss.backward()
    assert source.grad.norm() < 1e-5
