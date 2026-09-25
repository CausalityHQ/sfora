"""SmoothAP candidate scores with a current head over cached source features."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def live_head_bank_loss(
    anchors: torch.Tensor,
    source_bank: torch.Tensor,
    head: nn.Linear,
    positive_ordinals: torch.Tensor,
    self_ordinals: torch.Tensor,
    *,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Rank current anchors against all fit sources projected by the live head.

    The source bank contains detached, unit-normalized encoder outputs. Its
    rows may be stale, but the candidate projection uses the current head and
    contributes gradients to that head. Every candidate is projected before
    final output normalization, matching the deployed representation.
    """

    if (
        anchors.ndim != 2
        or source_bank.ndim != 2
        or type(head) is not nn.Linear
        or head.in_features != source_bank.shape[1]
        or head.out_features != anchors.shape[1]
        or anchors.shape[0] != positive_ordinals.shape[0]
        or anchors.shape[0] != len(self_ordinals)
        or anchors.dtype != torch.float32
        or source_bank.dtype != torch.float32
        or positive_ordinals.dtype != torch.long
        or self_ordinals.dtype != torch.long
        or anchors.device != source_bank.device
        or anchors.device != head.weight.device
        or anchors.device != positive_ordinals.device
        or anchors.device != self_ordinals.device
        or type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or bool((self_ordinals < 0).any())
        or bool((self_ordinals >= len(source_bank)).any())
        or bool((positive_ordinals < -1).any())
        or bool((positive_ordinals >= len(source_bank)).any())
    ):
        raise ValueError("live-head bank geometry differs")
    valid = positive_ordinals >= 0
    if not bool(valid.any(dim=1).all()) or bool(
        ((positive_ordinals == self_ordinals[:, None]) & valid).any()
    ):
        raise ValueError("live-head bank positive inventory differs")
    with torch.autocast(device_type=anchors.device.type, enabled=False):
        projected = F.normalize(head(source_bank.detach()), dim=1)
        scores = anchors @ projected.T
        safe = positive_ordinals.clamp_min(0)
        positive_scores = scores.gather(1, safe)
        sigmoid_all = torch.sigmoid(
            (scores[:, None, :] - positive_scores[:, :, None]) * (2.0 / temperature)
        )
        candidate_valid = (
            torch.arange(len(source_bank), device=source_bank.device)[None, :]
            != self_ordinals[:, None]
        )
        candidate_rank = 0.5 + (sigmoid_all * candidate_valid[:, None, :]).sum(dim=2)
        positive_rank = 0.5 + (
            torch.sigmoid(
                (positive_scores[:, None, :] - positive_scores[:, :, None]) * (2.0 / temperature)
            )
            * valid[:, None, :]
        ).sum(dim=2)
        precision = positive_rank / candidate_rank
        return 1.0 - ((precision * valid).sum(dim=1) / valid.sum(dim=1)).mean()


__all__ = ["live_head_bank_loss"]
