"""Fixed depth surgery and exact logical-batch recovery for SigLIP compression."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.siglip_proxy_control import PooledProxyAnchorModel, _validate_replay_module
from sfora.token_set_proxy_anchor import proxy_anchor_loss

RETAINED_BLOCKS = (1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27)


def prune_siglip_student(model: PooledProxyAnchorModel) -> PooledProxyAnchorModel:
    """Copy a 27-block tower, retaining the fixed distributed 18-block subset."""
    vision = getattr(model.tower, "vision_model", None)
    encoder = getattr(vision, "encoder", None)
    layers = getattr(encoder, "layers", None)
    config = getattr(vision, "config", None)
    if (
        not isinstance(layers, nn.ModuleList)
        or len(layers) != 27
        or type(getattr(config, "num_hidden_layers", None)) is not int
        or getattr(config, "num_hidden_layers", None) != 27
    ):
        raise ValueError("depth recovery requires the complete 27-block SigLIP topology")
    student = copy.deepcopy(model)
    student_vision = cast(Any, student.tower.vision_model)
    student_vision.encoder.layers = nn.ModuleList(
        student_vision.encoder.layers[index - 1] for index in RETAINED_BLOCKS
    )
    student_vision.config.num_hidden_layers = len(RETAINED_BLOCKS)
    return student


def recovery_multiplier(update: int) -> float:
    """Return the fixed positive-learning-rate 198-update recovery schedule."""
    if type(update) is not int or not 1 <= update <= 198:
        raise ValueError("recovery update must be a concrete integer in 1..198")
    if update <= 10:
        return update / 10.0
    return 0.1 + 0.45 * (1.0 + math.cos(math.pi * (update - 10) / 188))


def _unit_descriptors(value: torch.Tensor) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] < 2
        or value.shape[1] < 1
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("descriptors require a finite floating matrix with at least two rows")
    norms = torch.linalg.vector_norm(value.float(), dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0.0):
        raise ValueError("descriptors must be unit normalized")


def relational_cross_entropy(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Transfer full-batch off-diagonal neighbours with a frozen teacher target."""
    _unit_descriptors(student)
    _unit_descriptors(teacher)
    if student.shape != teacher.shape or student.device != teacher.device:
        raise ValueError("teacher/student descriptor shape or device differs")
    n = student.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=student.device)
    z, t = student.float(), teacher.detach().float()
    # Selecting off-diagonal entries avoids undefined zero times negative infinity.
    s_logits = ((z @ z.T)[mask] / 0.1).reshape(n, n - 1)
    t_logits = ((t @ t.T)[mask] / 0.1).reshape(n, n - 1)
    return -(t_logits.softmax(dim=1) * s_logits.log_softmax(dim=1)).sum(dim=1).mean()


@dataclass(frozen=True)
class RecoveryBackwardEvidence:
    """Detached logical-batch objectives and replay consistency, not quality."""

    loss: torch.Tensor
    proxy_loss: torch.Tensor
    relational_loss: torch.Tensor
    maximum_descriptor_disagreement: float


def recomputed_recovery_backward(
    model: PooledProxyAnchorModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    teacher_descriptors: torch.Tensor | None = None,
    microbatch_size: int,
    descriptor_tolerance: float = 2e-5,
) -> RecoveryBackwardEvidence:
    """Replay exact PA(+relational) descriptor cotangents, with proxy grads once.

    Inputs must already contain the materialized logical-batch augmentation.
    No optimizer update or teacher forward occurs inside this operator.
    """
    if (
        not isinstance(inputs, torch.Tensor)
        or inputs.ndim < 2
        or inputs.shape[0] < 2
        or not inputs.is_floating_point()
        or not bool(torch.isfinite(inputs).all())
    ):
        raise ValueError("recovery inputs must be finite floating logical-batch tensors")
    n = int(inputs.shape[0])
    if (
        type(microbatch_size) is not int
        or microbatch_size < 1
        or microbatch_size > n
        or n % microbatch_size
    ):
        raise ValueError("microbatch size must be a positive divisor of the logical batch")
    if (
        labels.shape != (n,)
        or labels.dtype not in (torch.int32, torch.int64)
        or labels.device != inputs.device
        or bool((labels < 0).any())
        or bool((labels >= model.class_count).any())
    ):
        raise ValueError("recovery labels differ from the logical-batch proxy authority")
    if (
        type(descriptor_tolerance) is not float
        or not math.isfinite(descriptor_tolerance)
        or not 0 <= descriptor_tolerance <= 2e-5
    ):
        raise ValueError("descriptor tolerance exceeds the finite registered bound")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("recovery replay requires cleared gradients")
    if not model.proxies.requires_grad or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("recovery requires FP32 model state and trainable proxies")
    _validate_replay_module(model)
    with torch.no_grad():
        descriptors = torch.cat(
            [
                model.encode(inputs[start : start + microbatch_size])
                for start in range(0, n, microbatch_size)
            ]
        )
    _unit_descriptors(descriptors)
    z = descriptors.detach().requires_grad_(True)
    raw_proxies = model.proxies.detach().requires_grad_(True)
    if not bool(torch.isfinite(raw_proxies).all()) or bool(
        (torch.linalg.vector_norm(raw_proxies, dim=1) <= 0).any()
    ):
        raise ValueError("recovery proxies must be finite and nonzero")
    scores = z @ F.normalize(raw_proxies, dim=1).T
    pa = proxy_anchor_loss(scores, labels, alpha=32.0, delta=0.1)
    rel = (
        z.new_zeros(())
        if teacher_descriptors is None
        else relational_cross_entropy(z, teacher_descriptors)
    )
    loss = pa + rel
    dz, dp = torch.autograd.grad(loss, (z, raw_proxies))
    if not all(bool(torch.isfinite(x).all()) for x in (loss, dz, dp)):
        raise ValueError("recovery objective or cotangents are nonfinite")
    maximum = 0.0
    for start in range(0, n, microbatch_size):
        stop = start + microbatch_size
        replay = model.encode(inputs[start:stop])
        disagreement = float((replay.detach() - descriptors[start:stop]).abs().max())
        if not math.isfinite(disagreement) or disagreement > descriptor_tolerance:
            raise RuntimeError("recovery descriptor replay disagreement exceeds tolerance")
        maximum = max(maximum, disagreement)
        torch.autograd.backward(replay, dz[start:stop])
    model.proxies.grad = dp.detach().clone()
    for name, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all())):
            raise RuntimeError(f"recovery parameter gradient is absent or nonfinite: {name}")
    return RecoveryBackwardEvidence(loss.detach(), pa.detach(), rel.detach(), maximum)


def speed_gate(windows: list[dict[str, Any]]) -> bool:
    """Require every matched p95 ratio <=0.75, with exact integer arithmetic."""
    if type(windows) is not list or len(windows) != 3:
        raise ValueError("speed evidence requires exactly three paired windows")
    passed = True
    for window in windows:
        if type(window) is not dict or set(window) != {"pipeline", "encoder"}:
            raise ValueError("speed evidence scopes differ")
        for measurements in window.values():
            if type(measurements) is not dict or set(measurements) != {"full", "student"}:
                raise ValueError("speed evidence model pair differs")
            for samples in measurements.values():
                if (
                    type(samples) is not list
                    or len(samples) != 100
                    or any(type(sample) is not int or sample <= 0 for sample in samples)
                ):
                    raise ValueError(
                        "speed samples must be exactly 100 positive integer nanoseconds"
                    )
            passed &= (
                sorted(measurements["student"])[94] * 4 <= sorted(measurements["full"])[94] * 3
            )
    return passed
