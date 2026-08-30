"""Paired frozen-feature training for the preregistered TSPA F1 screen."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from sfora.token_set_proxy_anchor import (
    TokenSetProxyAnchorHead,
    proxy_anchor_loss,
    token_proxy_diversity,
    token_set_proxy_anchor_objective,
)
from sfora.token_set_screen import (
    cross_label_token_permutation,
    leave_one_out_recall_at_one,
    validate_f1_class_partition,
)

F1Arm = Literal["pooled", "tspa", "token-shuffled-tspa"]


@dataclass(frozen=True)
class FrozenTokenSetSplit:
    """Frozen SigLIP global/token features and their Cars labels."""

    global_features: torch.Tensor
    token_features: torch.Tensor
    pretrained_attention: torch.Tensor
    labels: torch.Tensor

    def validate(self) -> None:
        count = self.global_features.shape[0]
        if (
            self.global_features.ndim != 2
            or self.token_features.ndim != 3
            or self.token_features.shape[0] != count
            or self.token_features.shape[2] != self.global_features.shape[1]
            or self.pretrained_attention.shape != self.token_features.shape[:2]
            or self.labels.shape != (count,)
        ):
            raise ValueError("frozen token-set split shapes differ")
        if count == 0 or self.labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("frozen token-set split labels are invalid")
        floating = (self.global_features, self.token_features, self.pretrained_attention)
        if any(not tensor.is_floating_point() for tensor in floating):
            raise ValueError("frozen token-set features must be floating tensors")
        if any(not torch.isfinite(tensor).all() for tensor in floating):
            raise ValueError("frozen token-set features must be finite")
        if bool((self.pretrained_attention < 0).any()) or bool(
            (self.pretrained_attention.sum(dim=1) <= 0).any()
        ):
            raise ValueError("frozen attention must be nonnegative with positive mass")


@dataclass(frozen=True)
class F1TrainingConfig:
    """Frozen optimization and representation settings for one paired F1 arm."""

    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    global_dimensions: int = 512
    token_dimensions: int = 128
    token_proxies_per_class: int = 16
    set_weight: float = 0.25
    query_block: int = 32
    proxy_anchor_alpha: float = 32.0
    proxy_anchor_delta: float = 0.1
    diversity_weight: float = 0.1
    diversity_margin: float = 0.5
    collapse_threshold: float = 0.95

    def validate(self) -> None:
        integers = (
            self.epochs,
            self.batch_size,
            self.global_dimensions,
            self.token_dimensions,
            self.token_proxies_per_class,
            self.query_block,
        )
        if min(integers) < 1:
            raise ValueError("F1 integer settings must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("F1 optimizer settings are invalid")
        if not 0.0 <= self.set_weight <= 1.0:
            raise ValueError("F1 set weight must lie in [0, 1]")
        if self.proxy_anchor_alpha <= 0.0 or not 0.0 <= self.proxy_anchor_delta < 1.0:
            raise ValueError("F1 Proxy Anchor settings are invalid")
        if self.diversity_weight < 0.0 or not -1.0 <= self.diversity_margin <= 1.0:
            raise ValueError("F1 diversity settings are invalid")
        if not -1.0 <= self.collapse_threshold <= 1.0:
            raise ValueError("F1 collapse threshold is invalid")


@dataclass(frozen=True)
class F1ArmResult:
    """Final-checkpoint-only evidence for one paired mechanism-screen arm."""

    arm: F1Arm
    seed: int
    final_training_objective: float
    objective_kind: str
    validation_recall_at_1: float
    mean_token_proxy_cosine: float | None
    collapse_exceeded: bool | None


class PooledProxyAnchorHead(nn.Module):
    """Same-substrate pooled projection and Proxy Anchor control."""

    def __init__(self, *, input_dimensions: int, dimensions: int, classes: int) -> None:
        super().__init__()
        if min(input_dimensions, dimensions, classes) < 1:
            raise ValueError("pooled head dimensions must be positive")
        self.projection = nn.Linear(input_dimensions, dimensions, bias=False)
        self.proxies = nn.Parameter(torch.empty(classes, dimensions))
        nn.init.normal_(self.proxies, std=0.02)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.projection.in_features:
            raise ValueError("pooled features have the wrong shape")
        embeddings = F.normalize(self.projection(features), dim=-1)
        scores = embeddings.float() @ F.normalize(self.proxies, dim=-1).float().T
        return scores, embeddings


def initialize_paired_f1_heads(
    *,
    input_dimensions: int,
    classes: int,
    global_dimensions: int,
    token_dimensions: int,
    token_proxies_per_class: int,
    set_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[PooledProxyAnchorHead, TokenSetProxyAnchorHead, TokenSetProxyAnchorHead]:
    """Create paired arms with identical global projection/proxy initialization."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        pooled = PooledProxyAnchorHead(
            input_dimensions=input_dimensions,
            dimensions=global_dimensions,
            classes=classes,
        )
        torch.manual_seed(seed)
        tspa = TokenSetProxyAnchorHead(
            input_dimensions=input_dimensions,
            global_dimensions=global_dimensions,
            token_dimensions=token_dimensions,
            classes=classes,
            token_proxies_per_class=token_proxies_per_class,
            set_weight=set_weight,
        )
        with torch.no_grad():
            tspa.global_proxies.copy_(pooled.proxies)
        shuffled = copy.deepcopy(tspa)
    return pooled.to(device), tspa.to(device), shuffled.to(device)


def _pooled_leave_one_out(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    query_block: int,
) -> float:
    embeddings = F.normalize(embeddings.detach(), dim=-1)
    correct = 0
    for start in range(0, embeddings.shape[0], query_block):
        stop = min(start + query_block, embeddings.shape[0])
        scores = embeddings[start:stop].float() @ embeddings.float().T
        rows = torch.arange(stop - start, device=labels.device)
        columns = torch.arange(start, stop, device=labels.device)
        scores[rows, columns] = -torch.inf
        correct += int((labels[scores.argmax(dim=1)] == labels[start:stop]).sum())
    return correct / embeddings.shape[0]


def train_f1_arm(
    *,
    arm: F1Arm,
    head: PooledProxyAnchorHead | TokenSetProxyAnchorHead,
    train: FrozenTokenSetSplit,
    validation: FrozenTokenSetSplit,
    config: F1TrainingConfig,
    seed: int,
    device: torch.device,
) -> F1ArmResult:
    """Optimize one arm and evaluate only its immutable final checkpoint."""

    config.validate()
    train.validate()
    validation.validate()
    validate_f1_class_partition(
        train_labels=train.labels,
        validation_labels=validation.labels,
    )
    if train.global_features.shape[1] != validation.global_features.shape[1]:
        raise ValueError("F1 train and validation feature dimensions differ")
    if arm == "pooled" and not isinstance(head, PooledProxyAnchorHead):
        raise ValueError("pooled arm requires a pooled head")
    if arm != "pooled" and not isinstance(head, TokenSetProxyAnchorHead):
        raise ValueError("token-set arms require a TSPA head")
    if isinstance(head, PooledProxyAnchorHead):
        if (
            head.projection.out_features != config.global_dimensions
            or head.proxies.shape != (49, config.global_dimensions)
        ):
            raise ValueError("pooled head differs from the F1 configuration")
    else:
        if (
            head.global_projection.out_features != config.global_dimensions
            or head.token_projection.out_features != config.token_dimensions
            or head.token_proxies.shape
            != (49, config.token_proxies_per_class, config.token_dimensions)
            or not torch.equal(
                head.set_weight_tensor.detach().cpu(),
                torch.tensor(config.set_weight, dtype=torch.float32),
            )
        ):
            raise ValueError("TSPA head differs from the F1 configuration")

    shuffle = (
        cross_label_token_permutation(train.labels, seed=seed)
        if arm == "token-shuffled-tspa"
        else None
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    order_generator = torch.Generator(device="cpu").manual_seed(seed)
    final_training_loss: torch.Tensor | None = None
    head.train()
    for _epoch in range(config.epochs):
        order = torch.randperm(train.labels.numel(), generator=order_generator)
        epoch_loss_sum = torch.zeros((), device=device)
        epoch_batches = 0
        for start in range(0, order.numel(), config.batch_size):
            indexes = order[start : start + config.batch_size]
            labels = train.labels[indexes].to(device=device, dtype=torch.int64)
            globals_ = train.global_features[indexes].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            if arm == "pooled":
                pooled_head = cast(PooledProxyAnchorHead, head)
                scores, _ = pooled_head(globals_)
                loss = proxy_anchor_loss(
                    scores,
                    labels,
                    alpha=config.proxy_anchor_alpha,
                    delta=config.proxy_anchor_delta,
                )
            else:
                tspa_head = cast(TokenSetProxyAnchorHead, head)
                token_indexes = shuffle[indexes] if shuffle is not None else indexes
                tokens = train.token_features[token_indexes].to(device=device, dtype=torch.float32)
                attention = train.pretrained_attention[token_indexes].to(
                    device=device,
                    dtype=torch.float32,
                )
                output = tspa_head(globals_, tokens, attention)
                objective = token_set_proxy_anchor_objective(
                    output.class_scores,
                    labels,
                    tspa_head.token_proxies,
                    diversity_weight=config.diversity_weight,
                    diversity_margin=config.diversity_margin,
                    collapse_threshold=config.collapse_threshold,
                )
                loss = objective.total
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss_sum += loss.detach()
            epoch_batches += 1
        if epoch_batches == 0:
            raise RuntimeError("F1 optimization produced no batches")
        final_training_loss = epoch_loss_sum / epoch_batches

    head.eval()
    validation_labels = validation.labels.to(device=device, dtype=torch.int64)
    with torch.inference_mode():
        if arm == "pooled":
            pooled_head = cast(PooledProxyAnchorHead, head)
            _, embeddings = pooled_head(
                validation.global_features.to(device=device, dtype=torch.float32)
            )
            recall = _pooled_leave_one_out(
                embeddings,
                validation_labels,
                query_block=config.query_block,
            )
            mean_cosine = None
            collapse = None
        else:
            tspa_head = cast(TokenSetProxyAnchorHead, head)
            global_embeddings, token_embeddings, token_weights = tspa_head.encode(
                validation.global_features.to(device=device, dtype=torch.float32),
                validation.token_features.to(device=device, dtype=torch.float32),
                validation.pretrained_attention.to(device=device, dtype=torch.float32),
            )
            recall = leave_one_out_recall_at_one(
                global_embeddings,
                token_embeddings,
                token_weights,
                validation_labels,
                set_weight=config.set_weight,
                query_block=config.query_block,
            )
            _, cosine = token_proxy_diversity(tspa_head.token_proxies)
            mean_cosine = float(cosine)
            collapse = mean_cosine > config.collapse_threshold
    if final_training_loss is None:
        raise RuntimeError("F1 optimization produced no batches")
    return F1ArmResult(
        arm=arm,
        seed=seed,
        final_training_objective=float(final_training_loss),
        objective_kind=(
            "proxy-anchor" if arm == "pooled" else "proxy-anchor-plus-proxy-diversity"
        ),
        validation_recall_at_1=recall,
        mean_token_proxy_cosine=mean_cosine,
        collapse_exceeded=collapse,
    )
