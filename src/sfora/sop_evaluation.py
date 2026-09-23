"""Deterministic leave-one-out retrieval scoring for the official SOP split."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


def score_symmetric(
    values: torch.Tensor,
    labels: torch.Tensor,
    *,
    block_rows: int = 64,
    inverse_norms: torch.Tensor | None = None,
) -> Mapping[str, object]:
    """Score all SOP queries against other test rows with ordinal tie-breaking.

    With ``inverse_norms``, ``values`` are widened signed-byte codes and
    scores use the deployed integer-dot-times-two-f16-inverse-norm arithmetic.
    Without it, rows are normalized before cosine scoring. Stable descending
    sorting gives the lower gallery ordinal priority on exact score ties.
    """

    if (
        type(values) is not torch.Tensor
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < 4
        or values.shape[1] < 2
        or not bool(torch.isfinite(values).all())
        or type(labels) is not torch.Tensor
        or labels.dtype != torch.int64
        or labels.shape != (values.shape[0],)
        or labels.device != values.device
        or type(block_rows) is not int
        or block_rows < 1
    ):
        raise ValueError("SOP symmetric scoring inventory differs")
    if inverse_norms is not None and (
        type(inverse_norms) is not torch.Tensor
        or inverse_norms.dtype != torch.float16
        or inverse_norms.shape != (values.shape[0],)
        or inverse_norms.device != values.device
        or not bool(torch.isfinite(inverse_norms).all())
        or bool((inverse_norms <= 0).any())
        or bool((values.abs() > 127).any())
        or not bool(torch.equal(values, values.round()))
    ):
        raise ValueError("SOP packed scoring inventory differs")
    if inverse_norms is None:
        if bool((torch.linalg.vector_norm(values, dim=1) == 0).any()):
            raise ValueError("SOP symmetric scoring inventory differs")
        matrix = F.normalize(values, dim=1)
        inverse = None
    else:
        matrix = values
        inverse = inverse_norms.float()
    _classes, inverse_labels, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    relevant = counts[inverse_labels] - 1
    if int(relevant.min()) < 1:
        raise ValueError("SOP symmetric positive inventory differs")
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device=values.device)
    gallery_rows = torch.arange(len(values), device=values.device)
    per_query_ap: list[float] = []
    per_query_r1: list[float] = []
    for start in range(0, len(values), block_rows):
        stop = min(start + block_rows, len(values))
        scores = matrix[start:stop] @ matrix.T
        if inverse is not None:
            scores = scores * inverse[start:stop, None] * inverse[None, :]
        local = torch.arange(stop - start, device=values.device)
        scores[local, gallery_rows[start:stop]] = -torch.inf
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = labels[ranked].eq(labels[start:stop, None])
        precision = torch.cumsum(matches, dim=1) / ranks[None, :]
        valid = ranks[None, :] <= relevant[start:stop, None]
        ap = (precision * matches * valid).sum(dim=1) / relevant[start:stop]
        per_query_ap.extend(float(x) for x in ap.cpu().tolist())
        per_query_r1.extend(float(x) for x in matches[:, 0].cpu().tolist())
    return {
        "queries": len(values),
        "max_relevant": width,
        "recall_at_1": sum(per_query_r1) / len(per_query_r1),
        "map_at_r": sum(per_query_ap) / len(per_query_ap),
        "per_query_r1": per_query_r1,
        "per_query_ap": per_query_ap,
    }


__all__ = ["score_symmetric"]
