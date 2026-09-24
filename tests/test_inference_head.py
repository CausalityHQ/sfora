import pytest
import torch
from torch import nn

from sfora.inference_head import fold_eval_affine_head


def test_fold_eval_affine_head_matches_two_linear_bn_layers() -> None:
    torch.manual_seed(17)
    head = nn.Sequential(
        nn.Linear(7, 5, bias=False, dtype=torch.float64),
        nn.BatchNorm1d(5, eps=2e-5, dtype=torch.float64),
        nn.Linear(5, 3, bias=False, dtype=torch.float64),
        nn.BatchNorm1d(3, eps=2e-5, dtype=torch.float64),
    ).eval()
    with torch.no_grad():
        for layer in (head[1], head[3]):
            layer.running_mean.copy_(torch.randn_like(layer.running_mean))
            layer.running_var.copy_(torch.rand_like(layer.running_var) + 0.1)
            layer.weight.copy_(torch.randn_like(layer.weight))
            layer.bias.copy_(torch.randn_like(layer.bias))
    values = torch.randn(9, 7, dtype=torch.float64)
    fused = fold_eval_affine_head(head)
    torch.testing.assert_close(fused(values), head(values), atol=1e-12, rtol=1e-12)
    assert fused.training is False


def test_fold_eval_affine_head_rejects_training_or_non_affine_head() -> None:
    head = nn.Sequential(
        nn.Linear(7, 5, bias=False),
        nn.BatchNorm1d(5),
        nn.Linear(5, 3, bias=False),
        nn.BatchNorm1d(3),
    )
    with pytest.raises(ValueError, match="evaluation"):
        fold_eval_affine_head(head)
    head.eval()
    head[2] = nn.Sequential(nn.Linear(5, 3), nn.ReLU())
    with pytest.raises(ValueError, match="geometry"):
        fold_eval_affine_head(head)
