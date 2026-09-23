"""Matched ArcFace and rank objectives for compact full-backbone SOP training."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

import torch
from torch import nn
from torch.nn import functional as F

from sfora.deployed_code_rank import smooth_ap_float_loss, smooth_ap_packed_loss
from sfora.unicom_training import sharded_mask_arcface_loss

RANK_COEFFICIENT = 2.0


class CompactTrainingArm(StrEnum):
    ARCFACE = "arcface"
    FLOAT_RANK = "float_rank"
    PACKED_RANK = "packed_rank"


def compact_head_features(source: torch.Tensor, head: nn.Linear) -> torch.Tensor:
    """Apply a compact head in the unit-source geometry used by PCA fitting."""

    if (
        type(source) is not torch.Tensor
        or source.ndim != 2
        or source.shape[1] != 768
        or not source.is_floating_point()
        or not bool(torch.isfinite(source).all())
        or bool((torch.linalg.vector_norm(source.float(), dim=1) == 0).any())
        or type(head) is not nn.Linear
        or head.in_features != 768
        or head.out_features != 128
        or source.device != head.weight.device
    ):
        raise ValueError("SOP compact source geometry differs")
    with torch.autocast(device_type=source.device.type, enabled=False):
        return head(F.normalize(source.float(), dim=1))


def compact_training_terms(
    features: torch.Tensor,
    classifier: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    arm: CompactTrainingArm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ArcFace and rank terms while preserving the same classifier.

    The rank arms add a fixed-weight SmoothAP term to the control. Every
    arm receives the same 128-dimensional head output, labels, and masks.
    Float32 objective arithmetic is explicit even under backbone AMP.
    """

    if type(arm) is not CompactTrainingArm:
        raise ValueError("SOP compact training arm differs")
    if (
        type(features) is not torch.Tensor
        or features.ndim != 2
        or features.shape[1] != 128
        or not features.is_floating_point()
        or not isinstance(classifier, torch.Tensor)
        or classifier.dtype != torch.float32
        or classifier.ndim != 2
        or classifier.shape[1] != 128
        or type(labels) is not torch.Tensor
        or labels.dtype != torch.int64
        or labels.shape != (features.shape[0],)
        or type(masks) is not torch.Tensor
        or masks.dtype != torch.int64
        or features.device != classifier.device
        or features.device != labels.device
        or features.device != masks.device
    ):
        raise ValueError("SOP compact training tensors differ")
    if min(Counter(labels.tolist()).values(), default=0) < 2:
        raise ValueError("SOP compact positive inventory differs")
    with torch.autocast(device_type=features.device.type, enabled=False):
        float_features = features.float()
        control = sharded_mask_arcface_loss(
            float_features, classifier, labels, masks, margin=0.3, scale=64.0
        )
        if arm is CompactTrainingArm.ARCFACE:
            return control, control.new_zeros(())
        label_list = labels.tolist()
        rank = (
            smooth_ap_float_loss(float_features, label_list)
            if arm is CompactTrainingArm.FLOAT_RANK
            else smooth_ap_packed_loss(float_features, label_list)
        )
        return control, rank


def compact_training_loss(
    features: torch.Tensor,
    classifier: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    arm: CompactTrainingArm,
) -> torch.Tensor:
    """Return the matched ArcFace control plus the selected rank term."""

    control, rank = compact_training_terms(features, classifier, labels, masks, arm=arm)
    return control + RANK_COEFFICIENT * rank


__all__ = [
    "CompactTrainingArm",
    "RANK_COEFFICIENT",
    "compact_head_features",
    "compact_training_loss",
    "compact_training_terms",
]
