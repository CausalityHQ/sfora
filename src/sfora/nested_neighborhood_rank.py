"""Dataset-agnostic nested embeddings and retrieval objectives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _finite_positive_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value) and value > 0.0


@dataclass(frozen=True, slots=True)
class NestedRankConfig:
    """Shape authority for an ordered residual embedding head."""

    input_dim: int
    hidden_dim: int = 1024
    output_dim: int = 128
    widths: tuple[int, ...] = (32, 128)
    class_count: int = 1

    def __post_init__(self) -> None:
        valid_widths = (
            type(self.widths) is tuple
            and bool(self.widths)
            and all(_positive_int(width) for width in self.widths)
            and tuple(sorted(set(self.widths))) == self.widths
            and self.widths[-1] == self.output_dim
        )
        if not (
            _positive_int(self.input_dim)
            and _positive_int(self.hidden_dim)
            and _positive_int(self.output_dim)
            and _positive_int(self.class_count)
            and valid_widths
        ):
            raise ValueError("nested rank configuration differs")


class NestedRankHead(nn.Module):
    """Residual projection whose ordered prefixes are independently normalized."""

    def __init__(self, config: NestedRankConfig) -> None:
        super().__init__()
        if not isinstance(config, NestedRankConfig):
            raise ValueError("nested rank configuration differs")
        self.config = config
        self.skip = nn.Linear(config.input_dim, config.output_dim, bias=False)
        self.hidden = nn.Linear(config.input_dim, config.hidden_dim, bias=False)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.output_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        if (
            type(features) is not torch.Tensor
            or features.ndim != 2
            or features.shape[1] != self.config.input_dim
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
            or torch.any(torch.linalg.vector_norm(features, dim=1) == 0)
        ):
            raise ValueError("nested rank features differ")
        dense = self.skip(features) + self.residual_scale * self.output(
            F.gelu(self.norm(self.hidden(features)))
        )
        if not torch.isfinite(dense).all() or torch.any(
            torch.linalg.vector_norm(dense, dim=1) == 0
        ):
            raise ValueError("nested rank features differ")
        return {width: F.normalize(dense[:, :width], dim=1) for width in self.config.widths}


def _labels(labels: torch.Tensor, *, rows: int, class_count: int | None = None) -> None:
    if (
        type(labels) is not torch.Tensor
        or labels.dtype != torch.int64
        or labels.ndim != 1
        or labels.shape[0] != rows
        or (rows and int(labels.min()) < 0)
        or (class_count is not None and rows and int(labels.max()) >= class_count)
    ):
        raise ValueError("nested rank labels differ")


def nested_proxy_anchor_loss(
    embeddings: Mapping[int, torch.Tensor],
    labels: torch.Tensor,
    raw_proxies: torch.Tensor,
    *,
    width_weights: Mapping[int, float] | None = None,
    scale: float = 32.0,
    margin: float = 0.1,
) -> torch.Tensor:
    """Compute the registered Proxy-Anchor objective over nested widths."""

    if (
        not embeddings
        or type(raw_proxies) is not torch.Tensor
        or raw_proxies.dtype != torch.float32
        or raw_proxies.ndim != 2
        or not torch.isfinite(raw_proxies).all()
        or not _finite_positive_float(scale)
        or type(margin) is not float
        or not math.isfinite(margin)
        or margin < 0.0
    ):
        raise ValueError("Proxy-Anchor authority differs")
    ordered_widths = tuple(embeddings)
    if ordered_widths != tuple(sorted(set(ordered_widths))):
        raise ValueError("Proxy-Anchor authority differs")
    rows = next(iter(embeddings.values())).shape[0]
    _labels(labels, rows=rows, class_count=raw_proxies.shape[0])
    resolved_weights: Mapping[int, float] = (
        {width: 1.0 for width in ordered_widths} if width_weights is None else width_weights
    )
    if tuple(resolved_weights) != ordered_widths or any(
        not _finite_positive_float(weight) for weight in resolved_weights.values()
    ):
        raise ValueError("Proxy-Anchor authority differs")

    total = raw_proxies.new_zeros(())
    present = torch.unique(labels, sorted=True)
    for width, values in embeddings.items():
        if (
            type(width) is not int
            or width <= 0
            or width > raw_proxies.shape[1]
            or type(values) is not torch.Tensor
            or values.dtype != torch.float32
            or values.ndim != 2
            or values.shape != (rows, width)
            or values.device != raw_proxies.device
            or labels.device != values.device
            or not torch.isfinite(values).all()
        ):
            raise ValueError("Proxy-Anchor authority differs")
        proxies = F.normalize(raw_proxies[:, :width], dim=1)
        scores = values @ proxies.T
        positive_terms = []
        for proxy in present:
            selected = scores[labels == proxy, proxy]
            positive_terms.append(torch.log1p(torch.exp(-scale * (selected - margin)).sum()))
        negative_terms = []
        for proxy in range(raw_proxies.shape[0]):
            selected = scores[labels != proxy, proxy]
            negative_terms.append(torch.log1p(torch.exp(scale * (selected + margin)).sum()))
        width_loss = torch.stack(positive_terms).mean() + torch.stack(negative_terms).mean()
        total = total + resolved_weights[width] * width_loss
    if not torch.isfinite(total):
        raise ValueError("Proxy-Anchor result differs")
    return total


def asymmetric_neighborhood_loss(
    student_queries: torch.Tensor,
    student_keys: torch.Tensor,
    teacher_rows: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Preserve an off-class teacher topology through query-only gradients."""

    rows = student_queries.shape[0] if type(student_queries) is torch.Tensor else -1
    _labels(labels, rows=rows)
    if (
        type(student_queries) is not torch.Tensor
        or type(student_keys) is not torch.Tensor
        or type(teacher_rows) is not torch.Tensor
        or type(sample_ids) is not torch.Tensor
        or student_queries.dtype != torch.float32
        or student_keys.dtype != torch.float32
        or teacher_rows.dtype != torch.float32
        or sample_ids.dtype != torch.int64
        or student_queries.ndim != 2
        or student_keys.shape != student_queries.shape
        or teacher_rows.ndim != 2
        or teacher_rows.shape[0] != rows
        or sample_ids.shape != (rows,)
        or len(
            {
                student_queries.device,
                student_keys.device,
                teacher_rows.device,
                labels.device,
                sample_ids.device,
            }
        )
        != 1
        or not _finite_positive_float(temperature)
        or not torch.isfinite(student_queries).all()
        or not torch.isfinite(student_keys).all()
        or not torch.isfinite(teacher_rows).all()
    ):
        raise ValueError("neighborhood authority differs")
    off_class = labels[:, None] != labels[None, :]
    distinct_sample = sample_ids[:, None] != sample_ids[None, :]
    valid = off_class & distinct_sample
    if not torch.all(valid.any(dim=1)):
        raise ValueError("neighborhood key inventory differs")
    teacher_similarity = (
        F.normalize(teacher_rows.detach(), dim=1) @ F.normalize(teacher_rows.detach(), dim=1).T
    )
    student_similarity = (
        F.normalize(student_queries, dim=1) @ F.normalize(student_keys.detach(), dim=1).T
    )
    teacher_probability = F.softmax(
        teacher_similarity.masked_fill(~valid, -torch.inf) / temperature, dim=1
    )
    student_log_probability = F.log_softmax(
        student_similarity.masked_fill(~valid, -torch.inf) / temperature, dim=1
    )
    safe_log_probability = torch.where(valid, student_log_probability, 0.0)
    loss = -(teacher_probability * safe_log_probability).sum(dim=1).mean()
    if not torch.isfinite(loss):
        raise ValueError("neighborhood result differs")
    return loss


def smooth_ap_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Compute a cosine-distance Smooth-AP surrogate with exact self exclusion."""

    rows = embeddings.shape[0] if type(embeddings) is torch.Tensor else -1
    _labels(labels, rows=rows)
    if (
        type(embeddings) is not torch.Tensor
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or type(sample_ids) is not torch.Tensor
        or sample_ids.dtype != torch.int64
        or sample_ids.shape != (rows,)
        or sample_ids.device != embeddings.device
        or labels.device != embeddings.device
        or not _finite_positive_float(temperature)
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("Smooth-AP authority differs")
    distances = 1.0 - F.normalize(embeddings, dim=1) @ F.normalize(embeddings, dim=1).T
    query_aps = []
    for query in range(rows):
        valid = sample_ids != sample_ids[query]
        positives = valid & (labels == labels[query])
        if not torch.any(positives):
            raise ValueError("Smooth-AP positive inventory differs")
        terms = []
        for positive in torch.where(positives)[0]:
            comparisons = torch.sigmoid(
                (distances[query, positive] - distances[query, valid]) / temperature
            )
            full_rank = 0.5 + comparisons.sum()
            positive_rank = 0.5 + comparisons[positives[valid]].sum()
            terms.append(positive_rank / full_rank)
        query_aps.append(torch.stack(terms).mean())
    loss = 1.0 - torch.stack(query_aps).mean()
    if not torch.isfinite(loss):
        raise ValueError("Smooth-AP result differs")
    return loss
