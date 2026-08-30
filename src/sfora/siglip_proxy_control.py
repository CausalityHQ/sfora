"""Frozen authority for the SigLIP-so400m pooled Proxy Anchor control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.substrate_screen import (
    SUBSTRATE_F0_CLASSES,
    SubstrateScreenMetrics,
    score_frozen_substrate,
)
from sfora.token_set_proxy_anchor import proxy_anchor_loss
from sfora.token_set_screen import F1_TRAIN_CLASSES, F1_VALIDATION_CLASSES

_SMOKE_MICROBATCH_LADDER = (120, 60, 40, 30, 24, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1)


def _concrete_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, tuple):
        actual_tuple = cast(tuple[object, ...], actual)
        return len(actual_tuple) == len(expected) and all(
            _concrete_equal(left, right)
            for left, right in zip(actual_tuple, expected, strict=True)
        )
    return bool(actual == expected)


@dataclass(frozen=True)
class SiglipProxyControlConfig:
    """The prospectively frozen pooled-control configuration."""

    model_name: str = "google/siglip-so400m-patch14-384"
    model_revision: str = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
    dataset_name: str = "tanganke/stanford_cars"
    dataset_revision: str = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
    seeds: tuple[int, ...] = (17, 29, 43)
    input_dimensions: int = 1152
    embedding_dimensions: int = 512
    logical_batch_size: int = 120
    images_per_class: int = 4
    classes_per_batch: int = 30
    proxy_anchor_alpha: float = 32.0
    proxy_anchor_delta: float = 0.1
    tower_learning_rate: float = 1.0e-5
    projection_learning_rate: float = 1.0e-4
    proxy_learning_rate: float = 1.0e-2
    weight_decay: float = 1.0e-4
    train_epochs: int = 60
    warmup_epochs: int = 5
    decay_epochs: tuple[int, ...] = (10, 20, 30, 40, 50)
    decay_gamma: float = 0.5
    gradient_clip_norm: float = 10.0
    replay_score_tolerance: float = 2.0e-5
    smoke_microbatch_ladder: tuple[int, ...] = _SMOKE_MICROBATCH_LADDER
    combined_memory_limit_bytes: int = 96 * 1024**3
    maximum_projected_seed_hours: float = 24.0

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "model_name": "google/siglip-so400m-patch14-384",
            "model_revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
            "dataset_name": "tanganke/stanford_cars",
            "dataset_revision": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
            "seeds": (17, 29, 43),
            "input_dimensions": 1152,
            "embedding_dimensions": 512,
            "logical_batch_size": 120,
            "images_per_class": 4,
            "classes_per_batch": 30,
            "proxy_anchor_alpha": 32.0,
            "proxy_anchor_delta": 0.1,
            "tower_learning_rate": 1.0e-5,
            "projection_learning_rate": 1.0e-4,
            "proxy_learning_rate": 1.0e-2,
            "weight_decay": 1.0e-4,
            "train_epochs": 60,
            "warmup_epochs": 5,
            "decay_epochs": (10, 20, 30, 40, 50),
            "decay_gamma": 0.5,
            "gradient_clip_norm": 10.0,
            "replay_score_tolerance": 2.0e-5,
            "smoke_microbatch_ladder": _SMOKE_MICROBATCH_LADDER,
            "combined_memory_limit_bytes": 96 * 1024**3,
            "maximum_projected_seed_hours": 24.0,
        }
        for name, value in expected.items():
            if not _concrete_equal(getattr(self, name), value):
                raise ValueError(f"{name} differs from the frozen SigLIP pooled-control contract")

    def steps_per_epoch(self, optimization_examples: int) -> int:
        """Resolve the registered floor-defined epoch compute budget."""

        if type(optimization_examples) is not int or optimization_examples <= 0:
            raise ValueError("optimization example count must be a positive concrete integer")
        return optimization_examples // self.logical_batch_size


@dataclass(frozen=True)
class ControlSplit:
    """Authenticated evidence roles and their observed example counts."""

    optimization_classes: frozenset[int]
    clean_validation_classes: frozenset[int]
    burned_diagnostic_classes: frozenset[int]
    optimization_examples: int
    clean_validation_examples: int
    burned_diagnostic_examples: int


class PooledProxyAnchorModel(nn.Module):
    """A vision tower, bias-free pooled projection, and normalized class proxies."""

    def __init__(
        self,
        *,
        tower: nn.Module,
        input_dimensions: int,
        embedding_dimensions: int,
        class_count: int,
    ) -> None:
        super().__init__()
        if type(input_dimensions) is not int or input_dimensions < 1:
            raise ValueError("input_dimensions must be a positive concrete integer")
        if type(embedding_dimensions) is not int or embedding_dimensions < 1:
            raise ValueError("embedding_dimensions must be a positive concrete integer")
        if type(class_count) is not int or class_count < 2:
            raise ValueError("class_count must be a concrete integer of at least two")
        self.tower = tower
        self.projection = nn.Linear(input_dimensions, embedding_dimensions, bias=False)
        self.proxies = nn.Parameter(torch.empty(class_count, embedding_dimensions))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.normal_(self.proxies, std=0.01)

    @property
    def class_count(self) -> int:
        """Return the number of trainable class proxies."""

        return int(self.proxies.shape[0])

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Produce the registered unit-normalized fp32 pooled descriptor."""

        pooled = self.tower(inputs)
        if pooled.ndim != 2 or pooled.shape[1] != self.projection.in_features:
            raise ValueError("tower output differs from the registered pooled shape")
        projected = self.projection(pooled).float()
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("projected descriptors must be finite")
        if bool((torch.linalg.vector_norm(projected, dim=1) <= 0).any()):
            raise ValueError("projected descriptors must have nonzero norm")
        return F.normalize(projected, dim=1)

    def class_scores(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return fp32 cosine scores against every normalized proxy."""

        embeddings = self.encode(inputs)
        proxies = self.proxies.float()
        if not bool(torch.isfinite(proxies).all()):
            raise ValueError("class proxies must be finite")
        if bool((torch.linalg.vector_norm(proxies, dim=1) <= 0).any()):
            raise ValueError("class proxies must have nonzero norm")
        return embeddings @ F.normalize(proxies, dim=1).T


@dataclass(frozen=True)
class ReplayBackwardEvidence:
    """Exact full-batch score cotangents and replay-consistency evidence."""

    loss: torch.Tensor
    scores: torch.Tensor
    score_gradients: torch.Tensor
    maximum_score_disagreement: float


@dataclass(frozen=True)
class NearestClassMargins:
    """Per-query nearest class similarities and their mean margin evidence."""

    nearest_positive_cosine: torch.Tensor
    nearest_negative_cosine: torch.Tensor
    margin: torch.Tensor
    mean_nearest_positive_cosine: float
    mean_nearest_negative_cosine: float
    mean_margin: float


@dataclass(frozen=True)
class ControlBandEvidence:
    """One band's authoritative Recall@1 and separate margin evidence."""

    retrieval: SubstrateScreenMetrics
    margins: NearestClassMargins


@dataclass(frozen=True)
class SeedControlEvidence:
    """Initial/final scalar evidence required to summarize one scientific seed."""

    seed: int
    train_initial_margin: float
    train_final_margin: float
    clean_initial_recall_at_1: float
    clean_final_recall_at_1: float
    clean_initial_margin: float
    clean_final_margin: float
    burned_initial_margin: float
    burned_final_margin: float


@dataclass(frozen=True)
class SeedControlSummary:
    """Derived per-seed transfer changes without checkpoint selection."""

    seed: int
    train_margin_change: float
    clean_recall_change: float
    clean_margin_change: float
    burned_margin_change: float
    memorization_to_transfer_ratio: float | None


def _validate_replay_module(model: PooledProxyAnchorModel) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            raise ValueError("recomputed replay refuses training batch normalization")
        if isinstance(module, nn.Dropout) and module.training and module.p > 0.0:
            raise ValueError("recomputed replay refuses active dropout")
        for attribute in ("dropout", "attention_dropout", "drop_path"):
            value = getattr(module, attribute, 0.0)
            if isinstance(value, float) and value != 0.0:
                raise ValueError(f"recomputed replay refuses nonzero {attribute}")


def recomputed_proxy_anchor_backward(
    model: PooledProxyAnchorModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    microbatch_size: int,
    alpha: float,
    delta: float,
    score_tolerance: float,
) -> ReplayBackwardEvidence:
    """Backpropagate one exact logical-batch Proxy Anchor score cotangent by replay."""

    if not isinstance(inputs, torch.Tensor) or not inputs.is_floating_point() or inputs.ndim < 2:
        raise ValueError("logical-batch inputs must be a floating tensor")
    batch_size = int(inputs.shape[0])
    if batch_size < 1 or labels.shape != (batch_size,):
        raise ValueError("logical-batch inputs and labels are misaligned")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("logical-batch labels must use an integer dtype")
    if bool((labels < 0).any()) or bool((labels >= model.class_count).any()):
        raise ValueError("logical-batch labels exceed the proxy classes")
    if (
        type(microbatch_size) is not int
        or microbatch_size < 1
        or microbatch_size > batch_size
        or batch_size % microbatch_size != 0
    ):
        raise ValueError("microbatch size must be a positive divisor of the logical batch")
    if type(score_tolerance) is not float or score_tolerance < 0.0:
        raise ValueError("score tolerance must be a nonnegative concrete float")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError("recomputed replay requires cleared parameter gradients")
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("logical-batch inputs must be finite")
    _validate_replay_module(model)

    score_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, batch_size, microbatch_size):
            score_chunks.append(model.class_scores(inputs[start : start + microbatch_size]))
    scores = torch.cat(score_chunks, dim=0)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("logical-batch scores must be finite")
    score_leaf = scores.detach().requires_grad_(True)
    loss = proxy_anchor_loss(score_leaf, labels, alpha=alpha, delta=delta)
    (score_gradients,) = torch.autograd.grad(loss, score_leaf)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(score_gradients).all()):
        raise ValueError("Proxy Anchor loss and score gradients must be finite")

    maximum_disagreement = 0.0
    for start in range(0, batch_size, microbatch_size):
        stop = start + microbatch_size
        replay_scores = model.class_scores(inputs[start:stop])
        disagreement = float((replay_scores.detach() - scores[start:stop]).abs().max())
        maximum_disagreement = max(maximum_disagreement, disagreement)
        if disagreement > score_tolerance:
            raise RuntimeError("recomputed score disagreement exceeds the registered tolerance")
        torch.autograd.backward(replay_scores, score_gradients[start:stop])

    tower_gradient_norm = 0.0
    for parameter in model.tower.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("every trainable tower parameter must receive a finite gradient")
        tower_gradient_norm += float(torch.sum(parameter.grad.detach().float().square()))
    if tower_gradient_norm <= 0.0:
        raise RuntimeError("aggregate tower gradient norm must be positive")

    return ReplayBackwardEvidence(
        loss=loss.detach(),
        scores=scores.detach(),
        score_gradients=score_gradients.detach(),
        maximum_score_disagreement=maximum_disagreement,
    )


def nearest_class_margins(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    query_block: int,
) -> NearestClassMargins:
    """Compute exact fp32 nearest-positive and nearest-negative cosine margins."""

    if type(query_block) is not int or query_block < 1:
        raise ValueError("query_block must be a positive concrete integer")
    if embeddings.ndim != 2 or labels.shape != (embeddings.shape[0],):
        raise ValueError("embedding and label shapes differ")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must use an integer dtype")
    if embeddings.shape[0] < 2 or not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite and contain at least two rows")
    embeddings_fp32 = embeddings.float()
    norms = torch.linalg.vector_norm(embeddings_fp32, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1.0e-6, rtol=0.0):
        raise ValueError("incoming descriptors must have unit norm")
    labels_device = labels.to(device=embeddings.device, dtype=torch.int64)
    unique, counts = torch.unique(labels_device, return_counts=True)
    if unique.numel() < 2 or bool((counts < 2).any()):
        raise ValueError("margin evidence requires two classes and two examples per class")

    positives: list[torch.Tensor] = []
    negatives: list[torch.Tensor] = []
    count = int(labels.numel())
    for start in range(0, count, query_block):
        stop = min(start + query_block, count)
        scores = embeddings_fp32[start:stop] @ embeddings_fp32.T
        same_class = labels_device[start:stop, None] == labels_device[None, :]
        rows = torch.arange(stop - start, device=embeddings.device)
        same_class[rows, torch.arange(start, stop, device=embeddings.device)] = False
        positive_scores = scores.masked_fill(~same_class, -torch.inf)
        negative_scores = scores.masked_fill(same_class, -torch.inf)
        negative_scores[rows, torch.arange(start, stop, device=embeddings.device)] = -torch.inf
        positives.append(positive_scores.max(dim=1).values)
        negatives.append(negative_scores.max(dim=1).values)
    nearest_positive = torch.cat(positives)
    nearest_negative = torch.cat(negatives)
    margin = nearest_positive - nearest_negative
    if not bool(torch.isfinite(nearest_positive).all()):
        raise RuntimeError("nearest-positive evidence is incomplete")
    if not bool(torch.isfinite(nearest_negative).all()):
        raise RuntimeError("nearest-negative evidence is incomplete")
    return NearestClassMargins(
        nearest_positive_cosine=nearest_positive,
        nearest_negative_cosine=nearest_negative,
        margin=margin,
        mean_nearest_positive_cosine=float(nearest_positive.mean()),
        mean_nearest_negative_cosine=float(nearest_negative.mean()),
        mean_margin=float(margin.mean()),
    )


def evaluate_control_band(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    query_block: int,
) -> ControlBandEvidence:
    """Evaluate one isolated gallery through the existing Recall@1 authority."""

    retrieval = score_frozen_substrate(embeddings, labels, query_block=query_block)
    margins = nearest_class_margins(embeddings, labels, query_block=query_block)
    return ControlBandEvidence(retrieval=retrieval, margins=margins)


def summarize_control_seeds(
    evidence: tuple[SeedControlEvidence, ...],
) -> tuple[SeedControlSummary, ...]:
    """Derive the frozen three-seed clean and memorization-to-transfer changes."""

    if type(evidence) is not tuple or tuple(item.seed for item in evidence) != (17, 29, 43):
        raise ValueError("control evidence must contain the exact seeds 17, 29, and 43")
    summaries: list[SeedControlSummary] = []
    for item in evidence:
        values = (
            item.train_initial_margin,
            item.train_final_margin,
            item.clean_initial_recall_at_1,
            item.clean_final_recall_at_1,
            item.clean_initial_margin,
            item.clean_final_margin,
            item.burned_initial_margin,
            item.burned_final_margin,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise ValueError("control seed evidence must contain concrete finite floats")
        if (
            not 0.0 <= item.clean_initial_recall_at_1 <= 1.0
            or not 0.0 <= item.clean_final_recall_at_1 <= 1.0
        ):
            raise ValueError("control seed recalls must lie in [0, 1]")
        train_change = item.train_final_margin - item.train_initial_margin
        burned_change = item.burned_final_margin - item.burned_initial_margin
        summaries.append(
            SeedControlSummary(
                seed=item.seed,
                train_margin_change=train_change,
                clean_recall_change=(
                    item.clean_final_recall_at_1 - item.clean_initial_recall_at_1
                ),
                clean_margin_change=item.clean_final_margin - item.clean_initial_margin,
                burned_margin_change=burned_change,
                memorization_to_transfer_ratio=(
                    burned_change / train_change if train_change > 0.0 else None
                ),
            )
        )
    return tuple(summaries)


def _validate_labels(labels: Any, *, role: str) -> torch.Tensor:
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.numel() == 0
        or labels.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(f"{role} labels must be a nonempty integer vector")
    return labels.detach().cpu().to(torch.int64)


def _require_band(
    labels: torch.Tensor,
    *,
    expected: frozenset[int],
    role: str,
    minimum: int,
) -> None:
    observed = frozenset(int(value) for value in labels.tolist())
    if observed != expected:
        raise ValueError(f"{role} labels must contain the exact {role} classes")
    counts = {label: int((labels == label).sum()) for label in expected}
    if any(count < minimum for count in counts.values()):
        words = "four" if minimum == 4 else "two"
        raise ValueError(f"every {role} class must contain at least {words} examples")


def validate_control_partition(
    *,
    optimization_labels: torch.Tensor,
    clean_validation_labels: torch.Tensor,
    burned_diagnostic_labels: torch.Tensor,
) -> ControlSplit:
    """Require the three exact, disjoint Cars train-class evidence bands."""

    optimization = _validate_labels(optimization_labels, role="optimization")
    clean = _validate_labels(clean_validation_labels, role="clean-validation")
    burned = _validate_labels(burned_diagnostic_labels, role="burned-diagnostic")
    _require_band(
        optimization,
        expected=F1_TRAIN_CLASSES,
        role="optimization",
        minimum=4,
    )
    _require_band(
        clean,
        expected=F1_VALIDATION_CLASSES,
        role="clean-validation",
        minimum=2,
    )
    _require_band(
        burned,
        expected=SUBSTRATE_F0_CLASSES,
        role="burned-diagnostic",
        minimum=2,
    )
    return ControlSplit(
        optimization_classes=F1_TRAIN_CLASSES,
        clean_validation_classes=F1_VALIDATION_CLASSES,
        burned_diagnostic_classes=SUBSTRATE_F0_CLASSES,
        optimization_examples=int(optimization.numel()),
        clean_validation_examples=int(clean.numel()),
        burned_diagnostic_examples=int(burned.numel()),
    )
