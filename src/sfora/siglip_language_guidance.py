"""Training-only class-centroid language guidance; no inference or dataset API."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch

from sfora.siglip_depth_recovery import (
    RecoveryBackwardEvidence,
    _unit_descriptors,
    recomputed_recovery_backward,
    relational_cross_entropy,
)
from sfora.siglip_proxy_control import PooledProxyAnchorModel, _validate_replay_module
from sfora.token_set_proxy_anchor import proxy_anchor_loss


@dataclass(frozen=True)
class LanguageBackwardEvidence(RecoveryBackwardEvidence):
    """Detached loss terms and replay equality; not retrieval-quality evidence."""

    language_loss: torch.Tensor


def recomputed_language_backward(
    model: PooledProxyAnchorModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    text_gram: torch.Tensor | None,
    teacher_descriptors: torch.Tensor | None = None,
    microbatch_size: int,
) -> LanguageBackwardEvidence:
    """Replay a single combined PA(+relational+language) full-batch cotangent.

    No optimizer step, teacher forward, text encoding or augmentation occurs
    here. The base control delegates to the unchanged verified recovery path.
    """
    if text_gram is None:
        base = recomputed_recovery_backward(
            model,
            inputs,
            labels,
            teacher_descriptors=teacher_descriptors,
            microbatch_size=microbatch_size,
        )
        return LanguageBackwardEvidence(
            base.loss,
            base.proxy_loss,
            base.relational_loss,
            base.maximum_descriptor_disagreement,
            base.loss.new_zeros(()),
        )
    if (
        inputs.ndim < 2
        or len(inputs) < 2
        or not inputs.is_floating_point()
        or not bool(torch.isfinite(inputs).all())
        or type(microbatch_size) is not int
        or microbatch_size < 1
        or microbatch_size > len(inputs)
        or len(inputs) % microbatch_size
        or labels.shape != (len(inputs),)
        or labels.device != inputs.device
        or labels.dtype not in (torch.int32, torch.int64)
        or bool((labels < 0).any())
        or bool((labels >= model.class_count).any())
        or text_gram.shape != (model.class_count, model.class_count)
    ):
        raise ValueError("language replay batch/proxy/target authority differs")
    if any(
        p.grad is not None or not p.requires_grad or p.dtype != torch.float32
        for p in model.parameters()
    ):
        raise ValueError("language replay requires cleared trainable FP32 parameters")
    _validate_replay_module(model)
    n = len(inputs)
    with torch.no_grad():
        descriptors = torch.cat(
            [
                model.encode(inputs[start : start + microbatch_size])
                for start in range(0, n, microbatch_size)
            ]
        )
    _unit_descriptors(descriptors)
    z = descriptors.detach().requires_grad_(True)
    proxies = model.proxies.detach().requires_grad_(True)
    if not bool(torch.isfinite(proxies).all()) or bool((proxies.norm(dim=1) <= 0).any()):
        raise ValueError("language replay proxies must be finite and nonzero")
    scores = z @ torch.nn.functional.normalize(proxies, dim=1).T
    pa = proxy_anchor_loss(scores, labels, alpha=32.0, delta=0.1)
    rel = (
        z.new_zeros(())
        if teacher_descriptors is None
        else relational_cross_entropy(z, teacher_descriptors)
    )
    language = language_centroid_cross_entropy(z, labels, text_gram)
    loss = pa + rel + language
    dz, dp = torch.autograd.grad(loss, (z, proxies))
    if not all(bool(torch.isfinite(v).all()) for v in (loss, dz, dp)):
        raise ValueError("language objective or descriptor/proxy gradients are nonfinite")
    maximum = 0.0
    for start in range(0, n, microbatch_size):
        stop = start + microbatch_size
        replay = model.encode(inputs[start:stop])
        disagreement = float((replay.detach() - descriptors[start:stop]).abs().max())
        if not math.isfinite(disagreement) or disagreement > 2e-5:
            raise RuntimeError("language descriptor replay disagreement exceeds tolerance")
        maximum = max(maximum, disagreement)
        torch.autograd.backward(replay, dz[start:stop])
    model.proxies.grad = dp.detach().clone()
    for name, p in model.named_parameters():
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise RuntimeError(f"language parameter gradient absent/nonfinite: {name}")
    return LanguageBackwardEvidence(
        loss.detach(), pa.detach(), rel.detach(), maximum, language.detach()
    )


def standardized_text_gram(text: torch.Tensor) -> torch.Tensor:
    """Freeze text cosine relations using global off-diagonal population moments."""
    _unit_descriptors(text)
    with torch.no_grad():
        values = text.detach().float()
        gram = values @ values.T
        mask = ~torch.eye(len(values), dtype=torch.bool, device=values.device)
        off = gram[mask]
        mean = off.mean()
        deviation = off.std(correction=0)
        if not bool(torch.isfinite(deviation)) or float(deviation) <= 0:
            raise ValueError("text relations need finite nonzero off-diagonal variation")
        result = torch.zeros_like(gram)
        result[mask] = (off - mean) / deviation
        if not bool(torch.isfinite(result).all()):
            raise ValueError("standardized text relations are nonfinite")
        return result


def language_centroid_cross_entropy(
    descriptors: torch.Tensor, labels: torch.Tensor, text_gram: torch.Tensor
) -> torch.Tensor:
    """Match balanced image-class centroids to frozen subset text relations.

    Scientific callers enforce30 classes/four images; reduced balanced CPU
    fixtures exercise the same operation. Both image-similarity endpoints
    receive gradients; the text target never does. Temperature is fixed0.1.
    """
    _unit_descriptors(descriptors)
    if (
        labels.shape != (len(descriptors),)
        or labels.dtype not in (torch.int32, torch.int64)
        or labels.device != descriptors.device
        or text_gram.ndim != 2
        or text_gram.shape[0] != text_gram.shape[1]
        or text_gram.device != descriptors.device
        or not text_gram.is_floating_point()
        or not bool(torch.isfinite(text_gram).all())
        or bool((labels < 0).any())
        or bool((labels >= text_gram.shape[0]).any())
    ):
        raise ValueError("language labels or target matrix differ from descriptor authority")
    classes, counts = cast(
        tuple[torch.Tensor, torch.Tensor], torch.unique(labels, sorted=True, return_counts=True)
    )
    if len(classes) < 3 or int(counts[0]) < 2 or not bool((counts == counts[0]).all()):
        raise ValueError("language loss requires at least three balanced multi-image class groups")
    z = descriptors.float()
    means = torch.stack([z[labels == c].mean(dim=0) for c in classes])
    norms = torch.linalg.vector_norm(means, dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError("image class centroid is zero or nonfinite")
    centers = means / norms
    k = len(classes)
    mask = ~torch.eye(k, dtype=torch.bool, device=z.device)
    target = text_gram.detach().float()[classes][:, classes]
    q = target[mask].reshape(k, k - 1).softmax(dim=1)
    logits = ((centers @ centers.T)[mask] / 0.1).reshape(k, k - 1)
    return cast(torch.Tensor, -(q * logits.log_softmax(dim=1)).sum(dim=1).mean())
