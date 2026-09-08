#!/usr/bin/env python3
"""Train and evaluate relational linear compaction on official SOP."""

from __future__ import annotations

import math
from collections import Counter
from typing import TypedDict

import torch

from sfora.joint_relational_compaction import PackedInt8Embeddings


class SymmetricScore(TypedDict):
    """Per-row and aggregate official symmetric retrieval evidence."""

    map_at_r: float
    per_query_ap: tuple[float, ...]
    per_query_r1: tuple[float, ...]
    r1: float


def _lexicographic_candidates(scores: torch.Tensor, width: int) -> torch.Tensor:
    """Select score-descending, ordinal-ascending candidates exactly."""

    retained = min(width + 1, scores.shape[1])
    values, indexes = torch.topk(scores, k=retained, dim=1, largest=True, sorted=False)
    ordinal_order = torch.argsort(indexes, dim=1, stable=True)
    indexes = indexes.gather(1, ordinal_order)
    values = values.gather(1, ordinal_order)
    score_order = torch.argsort(values, dim=1, descending=True, stable=True)
    indexes = indexes.gather(1, score_order)
    values = values.gather(1, score_order)
    if retained > width:
        ambiguous = values[:, width - 1] == values[:, width]
        for row in torch.nonzero(ambiguous, as_tuple=False).flatten().tolist():
            boundary = values[row, width - 1]
            candidates = torch.nonzero(scores[row] >= boundary, as_tuple=False).flatten()
            candidates = torch.sort(candidates).values
            candidate_values = scores[row, candidates]
            order = torch.argsort(candidate_values, descending=True, stable=True)
            indexes[row, :width] = candidates[order[:width]]
    return indexes[:, :width]


def score_symmetric(
    embeddings: torch.Tensor | PackedInt8Embeddings,
    labels: tuple[int, ...],
    *,
    candidate_width: int,
    device: torch.device,
) -> SymmetricScore:
    """Evaluate leave-self-out MAP@R and Recall@1 on one symmetric split."""

    if isinstance(embeddings, PackedInt8Embeddings):
        row_count = embeddings.codes.shape[0]
        packed = True
    elif type(embeddings) is torch.Tensor:
        row_count = len(embeddings)
        packed = False
        norms = torch.linalg.vector_norm(embeddings.detach().double(), dim=1)
        if (
            embeddings.device.type != "cpu"
            or embeddings.dtype != torch.float32
            or embeddings.ndim != 2
            or embeddings.shape[1] < 2
            or not bool(torch.isfinite(embeddings).all())
            or not bool((torch.abs(norms - 1.0) <= 2e-5).all())
        ):
            raise ValueError("SOP floating embedding authority differs")
    else:
        raise ValueError("SOP scoring representation differs")
    if (
        type(labels) is not tuple
        or len(labels) != row_count
        or any(type(label) is not int or label < 1 for label in labels)
        or type(candidate_width) is not int
        or candidate_width < 1
        or candidate_width >= row_count
        or type(device) is not torch.device
    ):
        raise ValueError("SOP candidate authority differs")
    counts = Counter(labels)
    if any(count < 2 for count in counts.values()):
        raise ValueError("SOP singleton class differs")
    if max(counts.values()) - 1 > candidate_width:
        raise ValueError("SOP candidate width is smaller than a positive set")
    rankings = []
    with torch.inference_mode():
        for start in range(0, row_count, 256):
            stop = min(start + 256, row_count)
            if packed:
                assert isinstance(embeddings, PackedInt8Embeddings)
                query = PackedInt8Embeddings(
                    codes=embeddings.codes[start:stop].contiguous(),
                    inverse_norms=embeddings.inverse_norms[start:stop].contiguous(),
                )
                scores = query.cosine_similarity(embeddings, device=device)
            else:
                assert isinstance(embeddings, torch.Tensor)
                scores = embeddings[start:stop].to(device) @ embeddings.to(device).T
            rows = torch.arange(stop - start, device=device)
            columns = torch.arange(start, stop, device=device)
            scores[rows, columns] = -torch.inf
            rankings.append(_lexicographic_candidates(scores, candidate_width).cpu())
    ranked = torch.cat(rankings)
    aps = []
    hits = []
    for ranking, label in zip(ranked.tolist(), labels, strict=True):
        positives = counts[label] - 1
        found = 0
        terms = []
        for rank, index in enumerate(ranking[:positives], start=1):
            if labels[index] == label:
                found += 1
                terms.append(found / rank)
        aps.append(math.fsum(terms) / positives)
        hits.append(float(labels[ranking[0]] == label))
    return SymmetricScore(
        map_at_r=math.fsum(aps) / row_count,
        per_query_ap=tuple(aps),
        per_query_r1=tuple(hits),
        r1=math.fsum(hits) / row_count,
    )
