"""Small cached-feature adapters for foundation retrieval experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration shared by the nested adapter and its margin-softmax loss."""

    input_dim: int
    hidden_dim: int = 1_024
    output_dim: int = 512
    prefixes: tuple[int, ...] = (64, 128, 256, 512)
    class_count: int = 1
    margin: float = 0.2
    scale: float = 32.0

    def __post_init__(self) -> None:
        for name in ("input_dim", "hidden_dim", "output_dim", "class_count"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive builtin integer")
        if (
            type(self.prefixes) is not tuple
            or not self.prefixes
            or any(type(width) is not int or width <= 0 for width in self.prefixes)
            or tuple(sorted(set(self.prefixes))) != self.prefixes
            or self.prefixes[-1] != self.output_dim
        ):
            raise ValueError("prefixes must be increasing unique widths ending at output_dim")
        if (
            type(self.margin) is not float
            or not isfinite(self.margin)
            or not 0.0 <= self.margin < 1.0
        ):
            raise ValueError("margin must be a finite builtin float in [0, 1)")
        if type(self.scale) is not float or not isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be a positive finite builtin float")


@dataclass(frozen=True)
class HardenedRetrievalFold:
    """One unseen-identity evaluation fold with a permanent distractor pool."""

    optimization: np.ndarray
    query: np.ndarray
    gallery: np.ndarray
    distractor: np.ndarray


class NestedResidualAdapter(nn.Module):
    """A compact residual MLP whose ordered coordinates form nested descriptors."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.hidden = nn.Linear(config.input_dim, config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.output_dim)
        self.skip = nn.Linear(config.input_dim, config.output_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.class_proxies = nn.Parameter(torch.empty(config.class_count, config.output_dim))
        nn.init.normal_(self.class_proxies, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_adapter_inputs(inputs, input_dim=self.config.input_dim)
        normalized = self.input_norm(inputs)
        residual = self.output(F.gelu(self.hidden(normalized)))
        return cast(torch.Tensor, self.skip(inputs) + self.residual_scale * residual)


class NestedLinearAdapter(nn.Module):
    """Bias-free linear control trained with the same proxies and nested loss."""

    def __init__(self, config: AdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.input_dim, config.output_dim, bias=False)
        self.class_proxies = nn.Parameter(torch.empty(config.class_count, config.output_dim))
        nn.init.normal_(self.class_proxies, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _validate_adapter_inputs(inputs, input_dim=self.config.input_dim)
        return cast(torch.Tensor, self.projection(inputs))


def initialize_residual_from_linear(
    residual: NestedResidualAdapter,
    linear: NestedLinearAdapter,
) -> None:
    """Warm-start a nonlinear adapter as the exact fitted linear mapping."""

    if not isinstance(residual, NestedResidualAdapter) or not isinstance(
        linear, NestedLinearAdapter
    ):
        raise ValueError("warm start requires residual and linear adapter instances")
    if residual.config != linear.config:
        raise ValueError("warm-start adapter configurations differ")
    with torch.no_grad():
        residual.skip.weight.copy_(linear.projection.weight)
        residual.class_proxies.copy_(linear.class_proxies)
        residual.residual_scale.zero_()


def _validate_adapter_inputs(inputs: torch.Tensor, *, input_dim: int) -> None:
    if not torch.is_tensor(inputs) or inputs.dtype != torch.float32:
        raise ValueError("adapter inputs must be FP32 tensors")
    if inputs.ndim != 2 or inputs.shape[1] != input_dim:
        raise ValueError("adapter inputs have the wrong rank or width")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("adapter inputs must be finite")


def nested_embeddings(
    embeddings: torch.Tensor,
    prefixes: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """Return independently normalized ordered prefixes."""

    if not torch.is_tensor(embeddings) or embeddings.dtype != torch.float32:
        raise ValueError("embeddings must be an FP32 tensor")
    if embeddings.ndim != 2 or not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite rank-2 values")
    if (
        type(prefixes) is not tuple
        or not prefixes
        or tuple(sorted(set(prefixes))) != prefixes
        or prefixes[-1] != embeddings.shape[1]
    ):
        raise ValueError("prefixes differ from the embedding width")
    return {width: F.normalize(embeddings[:, :width], p=2, dim=1) for width in prefixes}


def identity_balanced_batches(
    labels: np.ndarray,
    *,
    identities_per_batch: int,
    images_per_identity: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Build one deterministic epoch of P-by-K cached-feature batches."""

    if type(labels) is not np.ndarray or labels.dtype != np.dtype("int64") or labels.ndim != 1:
        raise ValueError("labels must be a rank-1 int64 NumPy array")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative builtin integer")
    for name, value in (
        ("identities_per_batch", identities_per_batch),
        ("images_per_identity", images_per_identity),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive builtin integer")
    groups = {
        int(label): np.flatnonzero(labels == label)
        for label in np.unique(labels)
        if int(np.count_nonzero(labels == label)) >= images_per_identity
    }
    eligible = np.asarray(sorted(groups), dtype=np.int64)
    if eligible.size < identities_per_batch:
        raise ValueError("too few eligible identities for one balanced batch")
    rng = np.random.default_rng(seed)
    identity_order = rng.permutation(eligible)
    usable = (identity_order.size // identities_per_batch) * identities_per_batch
    batches: list[np.ndarray] = []
    for start in range(0, usable, identities_per_batch):
        indexes = [
            rng.choice(groups[int(label)], size=images_per_identity, replace=False)
            for label in identity_order[start : start + identities_per_batch]
        ]
        batches.append(np.concatenate(indexes).astype(np.int64, copy=False))
    return tuple(batches)


def hardened_retrieval_folds(
    labels: np.ndarray,
    *,
    seed: int,
    heldout_fraction: float = 0.2,
) -> tuple[HardenedRetrievalFold, ...]:
    """Split the original unseen identities into four eval folds and distractors."""

    if type(labels) is not np.ndarray or labels.dtype != np.dtype("int64") or labels.ndim != 1:
        raise ValueError("labels must be a rank-1 int64 NumPy array")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative builtin integer")
    if (
        type(heldout_fraction) is not float
        or not isfinite(heldout_fraction)
        or not 0.0 < heldout_fraction < 1.0
    ):
        raise ValueError("heldout_fraction must be a finite builtin float in (0, 1)")
    groups = {
        int(label): np.flatnonzero(labels == label)
        for label in np.unique(labels)
        if int(np.count_nonzero(labels == label)) >= 2
    }
    if len(groups) < 25:
        raise ValueError("hardened retrieval requires at least 25 eligible identities")
    rng = np.random.default_rng(seed)
    eligible = np.asarray(sorted(groups), dtype=np.int64)
    heldout_count = min(
        max(5, int(round(len(eligible) * heldout_fraction))),
        len(eligible) - 5,
    )
    heldout = eligible[rng.permutation(len(eligible))[:heldout_count]]
    heldout_set = {int(label) for label in heldout}
    optimization = np.concatenate(
        [groups[label] for label in sorted(set(groups) - heldout_set)]
    ).astype(np.int64, copy=False)
    identity_folds = tuple(np.array_split(rng.permutation(heldout), 5))
    distractor = np.concatenate([groups[int(label)] for label in identity_folds[4]])
    result = []
    for identity_fold in identity_folds[:4]:
        query_rows = []
        gallery_rows = []
        for label in sorted(int(value) for value in identity_fold):
            order = rng.permutation(groups[label])
            query_count = max(1, len(order) // 2)
            query_rows.append(order[:query_count])
            gallery_rows.append(order[query_count:])
        result.append(
            HardenedRetrievalFold(
                optimization=optimization,
                query=np.concatenate(query_rows).astype(np.int64, copy=False),
                gallery=np.concatenate(gallery_rows).astype(np.int64, copy=False),
                distractor=distractor.astype(np.int64, copy=False),
            )
        )
    return tuple(result)


def retrieval_recall_at_1(
    query: torch.Tensor,
    query_labels: np.ndarray,
    gallery: torch.Tensor,
    gallery_labels: np.ndarray,
    *,
    chunk_size: int = 2_048,
) -> float:
    """Compute exact cosine Recall@1 without materializing the full score matrix."""

    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive builtin integer")
    for name, value in (("query_labels", query_labels), ("gallery_labels", gallery_labels)):
        if type(value) is not np.ndarray or value.dtype != np.dtype("int64") or value.ndim != 1:
            raise ValueError(f"{name} must be a rank-1 int64 NumPy array")
    if (
        not torch.is_tensor(query)
        or not torch.is_tensor(gallery)
        or query.dtype != torch.float32
        or gallery.dtype != torch.float32
        or query.ndim != 2
        or gallery.ndim != 2
        or query.shape[1] != gallery.shape[1]
        or query.shape[0] != query_labels.shape[0]
        or gallery.shape[0] != gallery_labels.shape[0]
    ):
        raise ValueError("query/gallery tensors and labels differ")
    query = F.normalize(query, p=2, dim=1)
    gallery = F.normalize(gallery, p=2, dim=1)
    correct = 0
    for start in range(0, query.shape[0], chunk_size):
        neighbors = (query[start : start + chunk_size] @ gallery.T).argmax(dim=1)
        predicted = gallery_labels[neighbors.detach().cpu().numpy()]
        correct += int(np.count_nonzero(predicted == query_labels[start : start + chunk_size]))
    return float(correct / query.shape[0])


def retrieval_map_at_r(
    query: torch.Tensor,
    query_labels: np.ndarray,
    gallery: torch.Tensor,
    gallery_labels: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    """Compute exact mAP@R, where R is each query's gallery relevant count."""

    _validate_retrieval_inputs(
        query, query_labels, gallery, gallery_labels, chunk_size=chunk_size
    )
    query = F.normalize(query, p=2, dim=1)
    gallery = F.normalize(gallery, p=2, dim=1)
    relevant_counts = np.asarray(
        [np.count_nonzero(gallery_labels == label) for label in query_labels], dtype=np.int64
    )
    if bool((relevant_counts == 0).any()):
        raise ValueError("every query must have at least one relevant gallery row")
    average_precisions: list[np.ndarray] = []
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        maximum_r = int(relevant_counts[start:stop].max())
        top = torch.topk(
            query[start:stop] @ gallery.T,
            k=maximum_r,
            dim=1,
            largest=True,
            sorted=True,
        ).indices.detach().cpu().numpy()
        matches = gallery_labels[top] == query_labels[start:stop, None]
        ranks = np.arange(1, maximum_r + 1, dtype=np.float64)
        cumulative = np.cumsum(matches, axis=1)
        precision = cumulative / ranks[None, :]
        rows = []
        for row, relevant in enumerate(relevant_counts[start:stop]):
            numerator = (precision[row, :relevant] * matches[row, :relevant]).sum()
            rows.append(float(numerator / relevant))
        average_precisions.append(np.asarray(rows, dtype=np.float64))
    return float(np.concatenate(average_precisions).mean())


def _validate_retrieval_inputs(
    query: torch.Tensor,
    query_labels: np.ndarray,
    gallery: torch.Tensor,
    gallery_labels: np.ndarray,
    *,
    chunk_size: int,
) -> None:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive builtin integer")
    for name, value in (("query_labels", query_labels), ("gallery_labels", gallery_labels)):
        if type(value) is not np.ndarray or value.dtype != np.dtype("int64") or value.ndim != 1:
            raise ValueError(f"{name} must be a rank-1 int64 NumPy array")
    if (
        not torch.is_tensor(query)
        or not torch.is_tensor(gallery)
        or query.dtype != torch.float32
        or gallery.dtype != torch.float32
        or query.ndim != 2
        or gallery.ndim != 2
        or query.shape[1] != gallery.shape[1]
        or query.shape[0] != query_labels.shape[0]
        or gallery.shape[0] != gallery_labels.shape[0]
    ):
        raise ValueError("query/gallery tensors and labels differ")


def ridge_stitch(
    source: torch.Tensor,
    target: torch.Tensor,
    optimization_indexes: torch.Tensor,
    *,
    regularization: float,
) -> torch.Tensor:
    """Fit a centered ridge map on optimization rows and transform every source row."""

    if (
        not torch.is_tensor(source)
        or not torch.is_tensor(target)
        or source.dtype != torch.float32
        or target.dtype != torch.float32
        or source.ndim != 2
        or target.ndim != 2
        or source.shape[0] != target.shape[0]
        or not bool(torch.isfinite(source).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise ValueError("ridge source and target must be aligned finite FP32 matrices")
    if (
        not torch.is_tensor(optimization_indexes)
        or optimization_indexes.dtype != torch.int64
        or optimization_indexes.ndim != 1
        or optimization_indexes.numel() < source.shape[1]
        or bool((optimization_indexes < 0).any())
        or bool((optimization_indexes >= source.shape[0]).any())
        or torch.unique(optimization_indexes).numel() != optimization_indexes.numel()
    ):
        raise ValueError("ridge optimization indexes differ from unique valid int64 rows")
    if (
        type(regularization) is not float
        or not isfinite(regularization)
        or regularization <= 0.0
    ):
        raise ValueError("ridge regularization must be a positive finite builtin float")
    fit_source = source[optimization_indexes]
    fit_target = target[optimization_indexes]
    source_mean = fit_source.mean(dim=0, keepdim=True)
    target_mean = fit_target.mean(dim=0, keepdim=True)
    x = fit_source - source_mean
    y = fit_target - target_mean
    gram = x.T @ x
    scale = torch.trace(gram) / gram.shape[0]
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    weight = torch.linalg.solve(gram + regularization * scale * identity, x.T @ y)
    return cast(torch.Tensor, (source - source_mean) @ weight + target_mean)


def cosine_margin_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_proxies: torch.Tensor,
    config: AdapterConfig,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Average CosFace classification over all registered nested prefixes."""

    if (
        not torch.is_tensor(labels)
        or labels.dtype != torch.int64
        or labels.ndim != 1
        or labels.shape[0] != embeddings.shape[0]
        or bool((labels < 0).any())
        or bool((labels >= config.class_count).any())
    ):
        raise ValueError("labels must be valid contiguous int64 class indexes")
    if (
        not torch.is_tensor(class_proxies)
        or class_proxies.dtype != torch.float32
        or class_proxies.shape != (config.class_count, config.output_dim)
        or not bool(torch.isfinite(class_proxies).all())
    ):
        raise ValueError("class proxies differ from the configured FP32 matrix")
    outputs = nested_embeddings(embeddings, config.prefixes)
    losses: dict[int, torch.Tensor] = {}
    for width, output in outputs.items():
        proxies = F.normalize(class_proxies[:, :width], p=2, dim=1)
        logits = output @ proxies.T
        target = F.one_hot(labels, num_classes=config.class_count).to(dtype=logits.dtype)
        logits = config.scale * (logits - config.margin * target)
        losses[width] = F.cross_entropy(logits, labels)
    return torch.stack(tuple(losses.values())).mean(), losses


def trainable_parameter_count(module: Any) -> int:
    """Count trainable parameters without accepting arbitrary iterables."""

    if not isinstance(module, nn.Module):
        raise ValueError("module must be a torch.nn.Module")
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
