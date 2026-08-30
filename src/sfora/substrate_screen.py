"""Leakage-safe frozen-substrate screening for image retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

SUBSTRATE_F0_CLASSES = frozenset(range(82, 98))


@dataclass(frozen=True)
class SubstrateScreenMetrics:
    """Exact leave-one-out retrieval evidence for one frozen descriptor substrate."""

    correct: int
    queries: int
    recall_at_1: float


@dataclass(frozen=True)
class SubstrateRetrievalError:
    """One exact leave-one-out nearest-neighbour classification error."""

    query_position: int
    nearest_position: int
    query_label: int
    nearest_label: int


@dataclass(frozen=True)
class SubstrateScreenEvidence:
    """Metrics plus ordered query-level errors from the same score pass."""

    metrics: SubstrateScreenMetrics
    errors: tuple[SubstrateRetrievalError, ...]


def validate_substrate_holdout(*, split: str, labels: torch.Tensor) -> None:
    """Require the exact burned Cars train-class band used for substrate selection."""

    if split != "train":
        raise ValueError("the substrate screen is restricted to the train split")
    if labels.ndim != 1 or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("holdout labels must be an integer vector")
    if frozenset(int(label) for label in labels.tolist()) != SUBSTRATE_F0_CLASSES:
        raise ValueError("the holdout must contain exactly Cars train classes 82 through 97")
    counts = torch.bincount(labels.cpu().to(torch.int64), minlength=98)[82:98]
    if bool((counts < 2).any()):
        raise ValueError("every holdout class needs at least two images")


def score_frozen_substrate(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    query_block: int,
) -> SubstrateScreenMetrics:
    """Compute exact fp32 leave-one-out cosine Recall@1 with lowest-index ties."""

    return score_frozen_substrate_evidence(
        embeddings,
        labels,
        query_block=query_block,
    ).metrics


def score_frozen_substrate_evidence(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    query_block: int,
) -> SubstrateScreenEvidence:
    """Compute metrics and preserve every incorrect nearest-neighbour identity."""

    if query_block < 1:
        raise ValueError("query_block must be positive")
    if embeddings.ndim != 2 or labels.shape != (embeddings.shape[0],):
        raise ValueError("embedding and label shapes differ")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must use an integer dtype")
    if embeddings.shape[0] < 2 or not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite and contain at least two rows")
    embeddings = embeddings.float()
    norms = torch.linalg.vector_norm(embeddings, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1.0e-6, rtol=0.0):
        raise ValueError("incoming descriptors must have unit norm")
    normalized = F.normalize(embeddings, dim=-1)
    count = int(labels.numel())
    labels_cpu = labels.detach().cpu()
    errors: list[SubstrateRetrievalError] = []
    for start in range(0, count, query_block):
        stop = min(start + query_block, count)
        scores = normalized[start:stop] @ normalized.T
        rows = torch.arange(stop - start, device=scores.device)
        scores[rows, torch.arange(start, stop, device=scores.device)] = -torch.inf
        nearest_cpu = scores.argmax(dim=1).detach().cpu().tolist()
        for offset, nearest_position in enumerate(nearest_cpu):
            query_position = start + offset
            if nearest_position == query_position:
                raise RuntimeError("leave-one-out scoring selected the query itself")
            query_label = int(labels_cpu[query_position])
            nearest_label = int(labels_cpu[nearest_position])
            if query_label != nearest_label:
                errors.append(
                    SubstrateRetrievalError(
                        query_position=query_position,
                        nearest_position=int(nearest_position),
                        query_label=query_label,
                        nearest_label=nearest_label,
                    )
                )
    correct = count - len(errors)
    return SubstrateScreenEvidence(
        metrics=SubstrateScreenMetrics(
            correct=correct,
            queries=count,
            recall_at_1=correct / count,
        ),
        errors=tuple(errors),
    )
