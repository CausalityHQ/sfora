"""One frozen angular geometry term; CE and member-bank rank remain unchanged."""
import torch
from torch.nn import functional as F


def anchor(raw, teacher):
    assert raw.shape == teacher.shape and raw.ndim == 2 and raw.shape[1] == 128
    assert teacher.dtype == torch.float32 and not teacher.requires_grad and teacher.grad_fn is None
    return 32 * (F.normalize(raw.float(), dim=1) - teacher).square().sum(dim=1).mean()


def check():
    teacher = torch.zeros(2, 128)
    teacher[:, 0] = 1
    assert anchor(teacher, teacher) == 0 and anchor(-teacher, teacher) == 128
    orthogonal = torch.zeros_like(teacher)
    orthogonal[:, 1] = 2
    orthogonal.requires_grad_()
    loss = anchor(orthogonal, teacher)
    assert loss == 64
    loss.backward()
    assert torch.isfinite(orthogonal.grad).all() and orthogonal.grad.norm() > 0
    assert torch.equal((orthogonal.grad * orthogonal).sum(1), torch.zeros(2))
    try:
        anchor(orthogonal, teacher.requires_grad_())
    except AssertionError:
        pass
    else:
        raise AssertionError('teacher gradients accepted')
