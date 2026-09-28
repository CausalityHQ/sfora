"""Training-only supervision from fixed semantic prototype similarities."""

from typing import Literal

import torch
from torch.nn import functional as F


def frozen_text_loss(
    source: torch.Tensor,
    prototypes: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: Literal["none", "mean"] = "mean",
) -> torch.Tensor:
    """Soft CE at fixed scale 32; target -1 means no caption for that row.

    Targets preserve the text teacher's relative similarities, including
    identical descriptions. This is a surrogate, not native SigLIP2 loss.
    The teacher is detached; no serving component or text gradient is added.
    """
    if (
        source.ndim != 2
        or prototypes.ndim != 2
        or source.shape[1] != prototypes.shape[1]
        or target.shape != (source.shape[0],)
        or target.dtype != torch.long
        or not torch.isfinite(source).all()
        or not torch.isfinite(prototypes).all()
        or not bool((source.norm(dim=1) > 0).all())
        or not bool((prototypes.norm(dim=1) > 0).all())
        or bool(((target < -1) | (target >= len(prototypes))).any())
    ):
        raise ValueError("frozen text supervision geometry differs")
    valid = target >= 0
    if not bool(valid.any()):
        raise ValueError("batch has no caption-supervised row")
    with torch.autocast(device_type=source.device.type, enabled=False):
        text = F.normalize(prototypes.detach().float(), dim=1)
        images = F.normalize(source[valid].float(), dim=1)
        distribution = (32 * (text[target[valid]] @ text.T)).softmax(dim=1)
        return F.cross_entropy(32 * (images @ text.T), distribution, reduction=reduction)
