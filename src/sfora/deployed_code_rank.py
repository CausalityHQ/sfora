"""Differentiable ranking through the deployed int8-128 plus f16 norm scorer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
from torch.nn import functional as F


def _unit_features(features: torch.Tensor) -> torch.Tensor:
    if (
        type(features) is not torch.Tensor
        or features.dtype != torch.float32
        or features.ndim != 2
        or features.shape[0] < 2
        or features.shape[1] != 128
        or not bool(torch.isfinite(features).all())
        or bool((torch.linalg.vector_norm(features, dim=1) == 0).any())
    ):
        raise ValueError("packed rank feature inventory differs")
    return F.normalize(features, dim=1)


def packed_cosine_ste(features: torch.Tensor) -> torch.Tensor:
    """Return exact packed-score forward values with straight-through gradients.

    The deployed wire stores round(127 * unit feature) as int8 and the
    reciprocal code norm as f16. Its scorer multiplies the integer dot by
    those two f16 values after widening to f32. Rounding's backward path is
    the identity; f16 conversion also uses a straight-through gradient.
    """

    with torch.autocast(device_type=features.device.type, enabled=False):
        unit = _unit_features(features)
        scaled = unit * 127.0
        rounded = torch.round(scaled).clamp(-127.0, 127.0)
        codes = scaled + (rounded - scaled).detach()
        inverse = torch.linalg.vector_norm(codes, dim=1).reciprocal()
        rounded_inverse = inverse.to(torch.float16).to(torch.float32)
        inverse_ste = inverse + (rounded_inverse - inverse).detach()
        dots = codes @ codes.T
        return dots * inverse_ste[:, None] * inverse_ste[None, :]


def _smooth_ap_scores(
    scores: torch.Tensor,
    labels: Sequence[object],
    *,
    temperature: float,
) -> torch.Tensor:
    if type(temperature) is not float or temperature <= 0.0:
        raise ValueError("packed rank temperature differs")
    rows = scores.shape[0]
    if len(labels) != rows or rows < 4 or any(label is None for label in labels):
        raise ValueError("packed rank labels differ")
    counts = Counter(labels)
    if any(count < 2 for count in counts.values()):
        raise ValueError("packed rank positive inventory differs")

    encoded: dict[object, int] = {}
    label_ids = [encoded.setdefault(label, len(encoded)) for label in labels]
    labels_tensor = torch.tensor(label_ids, device=scores.device)
    identity = torch.eye(rows, device=scores.device, dtype=torch.bool)
    positives = labels_tensor[:, None].eq(labels_tensor[None, :]) & ~identity
    distances = 2.0 - 2.0 * scores
    comparisons = torch.sigmoid((distances[:, :, None] - distances[:, None, :]) / temperature)
    competitors = (~identity)[:, None, :] & ~identity[None, :, :]
    rank = 1.0 + (comparisons * competitors).sum(dim=2)
    positive_competitors = positives[:, None, :] & ~identity[None, :, :]
    positive_rank = 1.0 + (comparisons * positive_competitors).sum(dim=2)
    precision = positive_rank / rank
    per_anchor = (precision * positives).sum(dim=1) / positives.sum(dim=1)
    loss = 1.0 - per_anchor.mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("packed rank loss is nonfinite")
    return loss


def smooth_ap_float_loss(
    features: torch.Tensor,
    labels: Sequence[object],
    *,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Matched 128-dimensional float control for the packed rank loss."""

    with torch.autocast(device_type=features.device.type, enabled=False):
        unit = _unit_features(features)
        return _smooth_ap_scores(unit @ unit.T, labels, temperature=temperature)


def smooth_ap_bank_loss(
    anchors: torch.Tensor,
    bank: torch.Tensor,
    positive_ordinals: torch.Tensor,
    self_ordinals: torch.Tensor,
    *,
    temperature: float = 0.01,
) -> torch.Tensor:
    """SmoothAP against detached full-bank candidates in O(B x P x N).

    `positive_ordinals` is a padded B x P matrix of gallery row indexes; -1
    denotes padding. Each positive set excludes that anchor's own row. The
    caller owns bank refresh and must supply unit fp32 vectors.
    """

    if (
        anchors.ndim != 2
        or bank.ndim != 2
        or anchors.shape[1] != bank.shape[1]
        or anchors.shape[0] != positive_ordinals.shape[0]
        or anchors.shape[0] != len(self_ordinals)
        or anchors.dtype != torch.float32
        or bank.dtype != torch.float32
        or positive_ordinals.dtype != torch.long
        or self_ordinals.dtype != torch.long
        or anchors.device != bank.device
        or anchors.device != positive_ordinals.device
        or anchors.device != self_ordinals.device
        or type(temperature) is not float
        or temperature <= 0.0
        or bool((self_ordinals < 0).any())
        or bool((self_ordinals >= len(bank)).any())
        or bool((positive_ordinals < -1).any())
        or bool((positive_ordinals >= len(bank)).any())
    ):
        raise ValueError("bank rank geometry differs")
    valid = positive_ordinals >= 0
    if not bool(valid.any(dim=1).all()) or bool(
        ((positive_ordinals == self_ordinals[:, None]) & valid).any()
    ):
        raise ValueError("bank rank positive inventory differs")
    with torch.autocast(device_type=anchors.device.type, enabled=False):
        scores = anchors @ bank.detach().T
        safe = positive_ordinals.clamp_min(0)
        positive_scores = scores.gather(1, safe)
        sigmoid_all = torch.sigmoid(
            (scores[:, None, :] - positive_scores[:, :, None]) * (2.0 / temperature)
        )
        candidate_valid = (
            torch.arange(len(bank), device=bank.device)[None, :] != (self_ordinals[:, None])
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


def smooth_ap_packed_loss(
    features: torch.Tensor,
    labels: Sequence[object],
    *,
    temperature: float = 0.01,
) -> torch.Tensor:
    """Smooth AP on the exact forward scores of the 130-byte packed wire."""

    with torch.autocast(device_type=features.device.type, enabled=False):
        return _smooth_ap_scores(packed_cosine_ste(features), labels, temperature=temperature)


__all__ = [
    "packed_cosine_ste",
    "smooth_ap_bank_loss",
    "smooth_ap_float_loss",
    "smooth_ap_packed_loss",
]
