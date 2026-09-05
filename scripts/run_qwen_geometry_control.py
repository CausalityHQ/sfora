#!/usr/bin/env python3
"""Run the local-only paired Qwen geometry-control experiment."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sfora.qwen_geometry_control import QwenGeometryProtocol, learning_rate_multiplier
from sfora.token_set_proxy_anchor import proxy_anchor_loss


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "little") + value


def _tensor_bytes(value: Tensor) -> bytes:
    cpu = value.detach().to(device="cpu").contiguous()
    header = f"{cpu.dtype}:{tuple(cpu.shape)}".encode("ascii")
    payload = cpu.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
    return _frame(header) + _frame(payload)


def state_sha256(parameters: Iterable[Tensor]) -> str:
    """Hash ordered tensor state with dtype and shape framing."""

    digest = hashlib.sha256()
    count = 0
    for ordinal, parameter in enumerate(parameters):
        digest.update(_frame(str(ordinal).encode("ascii")))
        digest.update(_tensor_bytes(parameter))
        count += 1
    if count == 0:
        raise ValueError("parameter state is empty")
    return digest.hexdigest()


def _hash_value(digest: object, value: object) -> None:
    update = digest.update  # type: ignore[attr-defined]
    if isinstance(value, Tensor):
        update(b"tensor")
        update(_tensor_bytes(value))
    elif isinstance(value, Mapping):
        update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            update(_frame(repr(key).encode("utf-8")))
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        update(b"sequence")
        for item in value:
            _hash_value(digest, item)
    else:
        update(_frame(repr(value).encode("utf-8")))


def _optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, optimizer.state_dict())
    return digest.hexdigest()


@dataclass(frozen=True)
class GeometryStepEvidence:
    """Auditable evidence from exactly one successful logical-batch update."""

    update_index: int
    loss: float
    scores: Tensor
    score_gradients: Tensor
    parameter_gradients: tuple[Tensor, ...]
    gradient_norm: float
    maximum_score_disagreement: float
    learning_rate_multiplier: float
    updated_state_sha256: str
    optimizer_state_sha256: str


def _validate_replay_model(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError("logical replay refuses training batch normalization")
        if isinstance(module, nn.Dropout) and module.training and module.p > 0.0:
            raise ValueError("logical replay refuses active dropout")


def replayed_proxy_anchor_step(
    *,
    model: nn.Module,
    proxies: nn.Parameter,
    inputs: Tensor,
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
    microbatch_size: int,
    update_index: int,
) -> GeometryStepEvidence:
    """Apply Proxy Anchor once while replaying a logical batch in bounded slices."""

    protocol = QwenGeometryProtocol()
    batch_size = int(inputs.shape[0]) if isinstance(inputs, Tensor) and inputs.ndim >= 1 else 0
    parameters = (*model.parameters(), proxies)
    if (
        not isinstance(inputs, Tensor)
        or not inputs.is_floating_point()
        or inputs.ndim < 2
        or batch_size < 1
        or labels.shape != (batch_size,)
        or labels.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("logical batch inputs and labels differ")
    if not torch.isfinite(inputs).all().item():
        raise ValueError("logical batch inputs must be finite")
    if (
        type(microbatch_size) is not int
        or microbatch_size < 1
        or microbatch_size > batch_size
        or batch_size % microbatch_size != 0
    ):
        raise ValueError("microbatch size must be a positive logical-batch divisor")
    if type(update_index) is not int or not 0 <= update_index < protocol.optimizer_updates:
        raise ValueError("update index differs from the registered schedule")
    if any(group.get("schedule_update") != update_index for group in optimizer.param_groups):
        raise ValueError("optimizer schedule position differs")
    if any("base_lr" not in group for group in optimizer.param_groups):
        raise ValueError("optimizer base learning-rate authority is absent")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("every logical replay parameter must be trainable")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("logical replay parameters are duplicated")
    if not torch.isfinite(proxies).all().item() or bool(
        (torch.linalg.vector_norm(proxies, dim=-1) <= 0).any()
    ):
        raise ValueError("class proxies must be finite and nonzero")
    if bool((labels < 0).any()) or bool((labels >= proxies.shape[0]).any()):
        raise ValueError("logical batch labels exceed the proxy authority")
    _validate_replay_model(model)

    optimizer.zero_grad(set_to_none=True)
    score_chunks: list[Tensor] = []
    with torch.no_grad():
        normalized_proxies = F.normalize(proxies, dim=-1)
        for start in range(0, batch_size, microbatch_size):
            descriptors = model(inputs[start : start + microbatch_size])
            if descriptors.ndim != 2 or descriptors.shape[1] != proxies.shape[1]:
                raise ValueError("descriptor and proxy shapes differ")
            score_chunks.append(descriptors @ normalized_proxies.T)
    scores = torch.cat(score_chunks)
    if not torch.isfinite(scores).all().item():
        raise ValueError("logical batch scores must be finite")
    score_leaf = scores.detach().requires_grad_(True)
    loss = proxy_anchor_loss(
        score_leaf,
        labels,
        alpha=protocol.proxy_anchor_alpha,
        delta=protocol.proxy_anchor_delta,
    )
    (score_gradients,) = torch.autograd.grad(loss, score_leaf)
    if not torch.isfinite(loss).item() or not torch.isfinite(score_gradients).all().item():
        raise ValueError("Proxy Anchor loss and cotangent must be finite")

    maximum_disagreement = 0.0
    for start in range(0, batch_size, microbatch_size):
        stop = start + microbatch_size
        replay_scores = model(inputs[start:stop]) @ F.normalize(proxies, dim=-1).T
        disagreement = float((replay_scores.detach() - scores[start:stop]).abs().max())
        maximum_disagreement = max(maximum_disagreement, disagreement)
        if not math.isfinite(disagreement) or disagreement > 1.0e-10:
            raise RuntimeError("logical replay score disagreement exceeds tolerance")
        torch.autograd.backward(replay_scores, score_gradients[start:stop])

    gradients: list[Tensor] = []
    for parameter in parameters:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError("every logical replay parameter must receive a finite gradient")
        gradients.append(parameter.grad.detach().clone())
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        parameters, protocol.gradient_clip_norm, error_if_nonfinite=True
    )
    if not math.isfinite(float(gradient_norm)) or float(gradient_norm) <= 0.0:
        raise RuntimeError("logical replay gradient norm must be finite and positive")

    multiplier = learning_rate_multiplier(update_index)
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * multiplier
    optimizer.step()
    for group in optimizer.param_groups:
        group["schedule_update"] = update_index + 1

    return GeometryStepEvidence(
        update_index=update_index,
        loss=float(loss.detach()),
        scores=scores.detach(),
        score_gradients=score_gradients.detach(),
        parameter_gradients=tuple(gradients),
        gradient_norm=float(gradient_norm),
        maximum_score_disagreement=maximum_disagreement,
        learning_rate_multiplier=multiplier,
        updated_state_sha256=state_sha256(parameters),
        optimizer_state_sha256=_optimizer_state_sha256(optimizer),
    )
