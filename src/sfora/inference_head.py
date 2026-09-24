"""Execution-preserving inference transformations for affine metric heads."""

from __future__ import annotations

import torch
from torch import nn


def fold_eval_affine_head(head: nn.Sequential) -> nn.Linear:
    """Fold Linear→BatchNorm→Linear→BatchNorm into one affine layer.

    The transformation is algebraically exact in real arithmetic for an
    evaluation-mode head. Floating-point outputs may differ by rounding.
    The caller can keep its original head for a checked serving fallback.
    """

    if type(head) is not nn.Sequential or len(head) != 4 or head.training:
        raise ValueError("affine head must be in evaluation mode")
    first, bn1, second, bn2 = head
    if (
        type(first) is not nn.Linear
        or type(second) is not nn.Linear
        or type(bn1) is not nn.BatchNorm1d
        or type(bn2) is not nn.BatchNorm1d
        or any(layer.training for layer in head)
        or not bn1.affine
        or not bn2.affine
        or not bn1.track_running_stats
        or not bn2.track_running_stats
        or first.out_features != bn1.num_features
        or second.in_features != bn1.num_features
        or second.out_features != bn2.num_features
    ):
        raise ValueError("affine head geometry differs")
    parameters = tuple(head.parameters())
    buffers = (bn1.running_mean, bn1.running_var, bn2.running_mean, bn2.running_var)
    if (
        any(buffer is None for buffer in buffers)
        or len({value.device for value in (*parameters, *buffers)}) != 1
        or len({value.dtype for value in (*parameters, *buffers)}) != 1
        or not parameters[0].is_floating_point()
        or any(not bool(torch.isfinite(value).all()) for value in (*parameters, *buffers))
        or any(bool((value < 0).any()) for value in (bn1.running_var, bn2.running_var))
    ):
        raise ValueError("affine head parameter authority differs")
    with torch.no_grad(), torch.autocast(device_type=first.weight.device.type, enabled=False):
        scale1 = bn1.weight / torch.sqrt(bn1.running_var + bn1.eps)
        scale2 = bn2.weight / torch.sqrt(bn2.running_var + bn2.eps)
        first_bias = first.bias if first.bias is not None else torch.zeros_like(scale1)
        second_bias = second.bias if second.bias is not None else torch.zeros_like(scale2)
        shift1 = scale1 * (first_bias - bn1.running_mean) + bn1.bias
        left = (scale2[:, None] * second.weight) * scale1[None, :]
        fused = nn.Linear(
            first.in_features,
            second.out_features,
            bias=True,
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        fused.weight.copy_(left @ first.weight)
        fused.bias.copy_(
            scale2 * (second.weight @ shift1 + second_bias - bn2.running_mean) + bn2.bias
        )
    return fused.eval()
