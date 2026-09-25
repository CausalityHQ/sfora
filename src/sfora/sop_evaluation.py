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
    prefix_euclidean_dimensions: int | None = None,
) -> Mapping[str, object]:
    """Score all SOP queries against other test rows with ordinal tie-breaking.

    With ``inverse_norms``, ``values`` are widened signed-byte codes and
    scores use the deployed integer-dot-times-two-f16-inverse-norm arithmetic.
    Without it, rows are normalized before cosine scoring. With
    ``prefix_euclidean_dimensions``, normalize the full rows first, truncate,
    then rank by Euclidean distance as UNICOM's SOP reference evaluator does.
    Stable descending sorting gives the lower gallery ordinal priority on
    exact score ties.
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
    if prefix_euclidean_dimensions is not None and (
        type(prefix_euclidean_dimensions) is not int
        or not 2 <= prefix_euclidean_dimensions < values.shape[1]
        or inverse_norms is not None
    ):
        raise ValueError("SOP reference prefix scoring inventory differs")
    if inverse_norms is None:
        if bool((torch.linalg.vector_norm(values, dim=1) == 0).any()):
            raise ValueError("SOP symmetric scoring inventory differs")
        matrix = F.normalize(values, dim=1)
        if prefix_euclidean_dimensions is not None:
            matrix = matrix[:, :prefix_euclidean_dimensions].contiguous()
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
    rank_width = max(width, min(1_000, len(values) - 1))
    ranks = torch.arange(1, rank_width + 1, device=values.device)
    gallery_rows = torch.arange(len(values), device=values.device)
    gallery_squared_norms = (
        (matrix * matrix).sum(dim=1) if prefix_euclidean_dimensions is not None else None
    )
    per_query_ap: list[float] = []
    per_query_r1: list[float] = []
    per_query_recall: dict[int, list[float]] = {10: [], 100: [], 1_000: []}
    for start in range(0, len(values), block_rows):
        stop = min(start + block_rows, len(values))
        scores = matrix[start:stop] @ matrix.T
        if inverse is not None:
            scores = scores * inverse[start:stop, None] * inverse[None, :]
        if gallery_squared_norms is not None:
            scores = 2 * scores - gallery_squared_norms[None, :]
        local = torch.arange(stop - start, device=values.device)
        scores[local, gallery_rows[start:stop]] = -torch.inf
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :rank_width]
        matches = labels[ranked].eq(labels[start:stop, None])
        precision = torch.cumsum(matches, dim=1) / ranks[None, :]
        valid = ranks[None, :] <= relevant[start:stop, None]
        ap = (precision * matches * valid).sum(dim=1) / relevant[start:stop]
        per_query_ap.extend(float(x) for x in ap.cpu().tolist())
        per_query_r1.extend(float(x) for x in matches[:, 0].cpu().tolist())
        for k, values_at_k in per_query_recall.items():
            values_at_k.extend(float(x) for x in matches[:, :k].any(dim=1).cpu().tolist())
    return {
        "queries": len(values),
        "max_relevant": width,
        "recall_at_1": sum(per_query_r1) / len(per_query_r1),
        **{f"recall_at_{k}": sum(rows) / len(rows) for k, rows in per_query_recall.items()},
        "map_at_r": sum(per_query_ap) / len(per_query_ap),
        "per_query_r1": per_query_r1,
        **{f"per_query_r{k}": rows for k, rows in per_query_recall.items()},
        "per_query_ap": per_query_ap,
    }


@torch.inference_mode()
def score_gallery_r1(
    values: torch.Tensor,
    labels: torch.Tensor,
    query_rows: torch.Tensor,
    *,
    block_rows: int = 64,
    inverse_norms: torch.Tensor | None = None,
) -> Mapping[str, object]:
    """Score selected self-excluded queries against the full gallery.

    Float rows use unit cosine. With f16 inverse norms, rows are widened
    signed-byte codes and the deployed packed cosine arithmetic is used.
    ``argmax`` selects the first gallery ordinal on an exact score tie.
    """

    if (
        type(values) is not torch.Tensor
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 2
        or not bool(torch.isfinite(values).all())
        or type(labels) is not torch.Tensor
        or labels.dtype != torch.int64
        or labels.shape != (len(values),)
        or labels.device != values.device
        or type(query_rows) is not torch.Tensor
        or query_rows.dtype != torch.int64
        or query_rows.ndim != 1
        or query_rows.device != values.device
        or len(query_rows) == 0
        or bool(((query_rows < 0) | (query_rows >= len(values))).any())
        or len(torch.unique(query_rows)) != len(query_rows)
        or type(block_rows) is not int
        or block_rows < 1
    ):
        raise ValueError("SOP full-gallery scoring inventory differs")
    if inverse_norms is not None and (
        type(inverse_norms) is not torch.Tensor
        or inverse_norms.dtype != torch.float16
        or inverse_norms.shape != (len(values),)
        or inverse_norms.device != values.device
        or not bool(torch.isfinite(inverse_norms).all())
        or bool((inverse_norms <= 0).any())
        or bool((values.abs() > 127).any())
        or not bool(torch.equal(values, values.round()))
    ):
        raise ValueError("SOP full-gallery packed inventory differs")
    matrix = F.normalize(values, dim=1) if inverse_norms is None else values
    inverse = None if inverse_norms is None else inverse_norms.float()
    transpose = matrix.T.contiguous()
    nearest: list[int] = []
    correct: list[int] = []
    for start in range(0, len(query_rows), block_rows):
        rows = query_rows[start : start + block_rows]
        scores = matrix[rows] @ transpose
        if inverse is not None:
            scores = scores * inverse[rows, None] * inverse[None, :]
        scores[torch.arange(len(rows), device=values.device), rows] = -torch.inf
        winners = torch.argmax(scores, dim=1)
        nearest.extend(int(row) for row in winners.cpu().tolist())
        correct.extend(int(value) for value in (labels[winners] == labels[rows]).cpu().tolist())
    return {
        "queries": len(query_rows),
        "recall_at_1": sum(correct) / len(correct),
        "per_query_r1": correct,
        "nearest_ordinals": nearest,
    }


__all__ = ["score_symmetric", "score_gallery_r1"]
