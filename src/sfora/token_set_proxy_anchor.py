"""Trainable Token-Set Proxy Anchor scoring and objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TokenSetProxyAnchorOutput:
    """Normalized embeddings, saliency weights, and image-to-class scores."""

    class_scores: torch.Tensor
    global_embeddings: torch.Tensor
    token_embeddings: torch.Tensor
    token_weights: torch.Tensor


@dataclass(frozen=True)
class TokenSetProxyAnchorObjective:
    """Bound Proxy Anchor, diversity, and collapse evidence for one batch."""

    total: torch.Tensor
    proxy_anchor: torch.Tensor
    diversity: torch.Tensor
    mean_token_proxy_cosine: torch.Tensor
    collapse_exceeded: torch.Tensor


def select_attention_tokens(
    tokens: torch.Tensor,
    attention: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select highest-attention tokens with a stable lowest-index tie break."""

    if tokens.ndim != 3 or attention.shape != tokens.shape[:2]:
        raise ValueError("attention must provide one score per token")
    if not 1 <= top_k <= tokens.shape[1]:
        raise ValueError("top_k must fit the token count")
    if not torch.isfinite(tokens).all() or not torch.isfinite(attention).all():
        raise ValueError("tokens and attention must be finite")
    if bool((attention < 0).any()) or bool((attention.sum(dim=1) <= 0).any()):
        raise ValueError("attention must be nonnegative with positive mass")
    indices = torch.argsort(attention, dim=1, descending=True, stable=True)[:, :top_k]
    selected = torch.gather(tokens, 1, indices[:, :, None].expand(-1, -1, tokens.shape[2]))
    selected_attention = torch.gather(attention, 1, indices)
    weights = selected_attention / selected_attention.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    return selected, weights, indices


def _normalized(
    matrix: torch.Tensor,
    *,
    name: str,
    validate_values: bool = True,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    if not matrix.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if validate_values and not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite floating tensor")
    if validate_values and bool((matrix.norm(dim=-1) < epsilon).any()):
        raise ValueError(f"{name} must have nonzero rows")
    return F.normalize(matrix, dim=-1, eps=epsilon)


def token_set_class_scores(
    global_embeddings: torch.Tensor,
    token_embeddings: torch.Tensor,
    token_weights: torch.Tensor,
    global_proxies: torch.Tensor,
    token_proxies: torch.Tensor,
    *,
    set_weight: float | torch.Tensor,
    validate_values: bool = True,
) -> torch.Tensor:
    """Evaluate the preregistered global/set image-to-class score."""

    if global_embeddings.ndim != 2 or token_embeddings.ndim != 3:
        raise ValueError("global and token embeddings must be rank two and three")
    if global_proxies.ndim != 2 or token_proxies.ndim != 3:
        raise ValueError("global and token proxies must be rank two and three")
    if isinstance(set_weight, torch.Tensor):
        if set_weight.ndim != 0 or set_weight.device != global_embeddings.device:
            raise ValueError("tensor set_weight must be a scalar on the embedding device")
        if validate_values and bool((set_weight < 0.0) | (set_weight > 1.0)):
            raise ValueError("set_weight must lie in [0, 1]")
    elif not 0.0 <= set_weight <= 1.0:
        raise ValueError("set_weight must lie in [0, 1]")
    batch, tokens, _ = token_embeddings.shape
    if global_embeddings.shape[0] != batch or token_weights.shape != (batch, tokens):
        raise ValueError("embedding and token-weight batches differ")
    if global_proxies.shape[0] != token_proxies.shape[0]:
        raise ValueError("global and token proxy class counts differ")
    if global_embeddings.shape[1] != global_proxies.shape[1]:
        raise ValueError("global embedding and proxy dimensions differ")
    if token_embeddings.shape[2] != token_proxies.shape[2]:
        raise ValueError("token embedding and proxy dimensions differ")
    devices = {
        global_embeddings.device,
        token_embeddings.device,
        token_weights.device,
        global_proxies.device,
        token_proxies.device,
    }
    if len(devices) != 1:
        raise ValueError("embeddings, weights, and proxies must share a device")
    if not token_weights.is_floating_point():
        raise ValueError("token weights must be floating values")
    if validate_values and not torch.isfinite(token_weights).all():
        raise ValueError("token weights must be finite floating values")
    if validate_values and (
        bool((token_weights < 0).any())
        or not torch.allclose(
            token_weights.sum(dim=1),
            torch.ones(batch, dtype=token_weights.dtype, device=token_weights.device),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    ):
        raise ValueError("token weights must be probability vectors")

    global_embeddings = _normalized(
        global_embeddings,
        name="global embeddings",
        validate_values=validate_values,
    )
    token_embeddings = _normalized(
        token_embeddings,
        name="token embeddings",
        validate_values=validate_values,
    )
    global_proxies = _normalized(
        global_proxies,
        name="global proxies",
        validate_values=validate_values,
    )
    token_proxies = _normalized(
        token_proxies,
        name="token proxies",
        validate_values=validate_values,
    )
    global_scores = global_embeddings.float() @ global_proxies.float().T
    interactions = torch.einsum(
        "bkd,cmd->bckm",
        token_embeddings.float(),
        token_proxies.float(),
    )
    set_scores = (interactions.max(dim=3).values * token_weights.float()[:, None, :]).sum(dim=2)
    return (1.0 - set_weight) * global_scores + set_weight * set_scores


def proxy_anchor_loss(
    class_scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 32.0,
    delta: float = 0.1,
) -> torch.Tensor:
    """Compute Proxy Anchor with stable per-class log-sum-exp reductions."""

    if class_scores.ndim != 2 or labels.shape != (class_scores.shape[0],):
        raise ValueError("class scores and labels have incompatible shapes")
    if class_scores.shape[0] == 0:
        raise ValueError("Proxy Anchor refuses an empty batch")
    if not class_scores.is_floating_point() or not torch.isfinite(class_scores).all():
        raise ValueError("class scores must be finite floating values")
    if alpha <= 0.0 or not 0.0 <= delta < 1.0:
        raise ValueError("Proxy Anchor alpha and delta are invalid")
    labels = labels.to(dtype=torch.int64, device=class_scores.device)
    classes = class_scores.shape[1]
    if bool((labels < 0).any()) or bool((labels >= classes).any()):
        raise ValueError("labels lie outside the proxy class range")
    positives = F.one_hot(labels, num_classes=classes).to(torch.bool)
    zero = torch.zeros((1, classes), dtype=class_scores.dtype, device=class_scores.device)
    positive_logits = (-alpha * (class_scores - delta)).masked_fill(~positives, -torch.inf)
    negative_logits = (alpha * (class_scores + delta)).masked_fill(positives, -torch.inf)
    positive_terms = torch.logsumexp(torch.cat((zero, positive_logits), dim=0), dim=0)
    negative_terms = torch.logsumexp(torch.cat((zero, negative_logits), dim=0), dim=0)
    valid_positive = positives.any(dim=0)
    return positive_terms[valid_positive].mean() + negative_terms.mean()


def token_proxy_diversity(
    token_proxies: torch.Tensor,
    *,
    margin: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the diversity hinge and mean within-class off-diagonal cosine."""

    if token_proxies.ndim != 3 or not -1.0 <= margin <= 1.0:
        raise ValueError("token proxies or diversity margin are invalid")
    proxies = _normalized(token_proxies, name="token proxies")
    per_class = proxies.shape[1]
    if per_class < 2:
        zero = proxies.sum() * 0.0
        return zero, zero
    similarities = proxies @ proxies.transpose(1, 2)
    off_diagonal = ~torch.eye(per_class, dtype=torch.bool, device=proxies.device)
    values = similarities[:, off_diagonal]
    return F.relu(values - margin).mean(), values.mean()


def token_set_proxy_anchor_objective(
    class_scores: torch.Tensor,
    labels: torch.Tensor,
    token_proxies: torch.Tensor,
    *,
    diversity_weight: float = 0.1,
    diversity_margin: float = 0.5,
    collapse_threshold: float = 0.95,
) -> TokenSetProxyAnchorObjective:
    """Compose the preregistered Proxy Anchor and proxy-diversity objective."""

    if diversity_weight < 0.0 or not -1.0 <= collapse_threshold <= 1.0:
        raise ValueError("diversity weight or collapse threshold is invalid")
    proxy_anchor = proxy_anchor_loss(class_scores, labels, alpha=32.0, delta=0.1)
    diversity, mean_cosine = token_proxy_diversity(token_proxies, margin=diversity_margin)
    return TokenSetProxyAnchorObjective(
        total=proxy_anchor + diversity_weight * diversity,
        proxy_anchor=proxy_anchor,
        diversity=diversity,
        mean_token_proxy_cosine=mean_cosine,
        collapse_exceeded=mean_cosine.detach() > collapse_threshold,
    )


class TokenSetProxyAnchorHead(nn.Module):
    """Projection, residual saliency, and set-valued class proxies for TSPA."""

    def __init__(
        self,
        *,
        input_dimensions: int,
        global_dimensions: int,
        token_dimensions: int,
        classes: int,
        token_proxies_per_class: int,
        set_weight: float,
    ) -> None:
        super().__init__()
        if (
            min(
                input_dimensions,
                global_dimensions,
                token_dimensions,
                classes,
                token_proxies_per_class,
            )
            < 1
        ):
            raise ValueError("TSPA dimensions, classes, and proxy count must be positive")
        if not 0.0 <= set_weight <= 1.0:
            raise ValueError("set_weight must lie in [0, 1]")
        self.set_weight_tensor: torch.Tensor
        self.register_buffer("set_weight_tensor", torch.tensor(set_weight, dtype=torch.float32))
        self.global_projection = nn.Linear(input_dimensions, global_dimensions, bias=False)
        self.token_projection = nn.Linear(input_dimensions, token_dimensions, bias=False)
        self.saliency_residual = nn.Linear(input_dimensions, 1, bias=False)
        nn.init.zeros_(self.saliency_residual.weight)
        self.global_proxies = nn.Parameter(torch.empty(classes, global_dimensions))
        self.token_proxies = nn.Parameter(
            torch.empty(classes, token_proxies_per_class, token_dimensions)
        )
        nn.init.normal_(self.global_proxies, std=0.02)
        nn.init.normal_(self.token_proxies, std=0.02)

    def encode(
        self,
        global_features: torch.Tensor,
        token_features: torch.Tensor,
        pretrained_attention: torch.Tensor,
        *,
        validate_values: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if global_features.ndim != 2 or token_features.ndim != 3:
            raise ValueError("global and token features must be rank two and three")
        if global_features.shape[0] != token_features.shape[0]:
            raise ValueError("global and token feature batches differ")
        if global_features.shape[1] != self.global_projection.in_features:
            raise ValueError("global feature dimension differs")
        if token_features.shape[2] != self.token_projection.in_features:
            raise ValueError("token feature dimension differs")
        if pretrained_attention.shape != token_features.shape[:2]:
            raise ValueError("pretrained attention does not match token features")
        if validate_values and (
            not torch.isfinite(pretrained_attention).all() or bool((pretrained_attention < 0).any())
        ):
            raise ValueError("pretrained attention must be finite and nonnegative")
        if validate_values and bool((pretrained_attention.sum(dim=1) <= 0).any()):
            raise ValueError("pretrained attention must have positive mass")

        global_embeddings = _normalized(
            self.global_projection(global_features),
            name="projected global embeddings",
            validate_values=validate_values,
        )
        token_embeddings = _normalized(
            self.token_projection(token_features),
            name="projected token embeddings",
            validate_values=validate_values,
        )
        residual_logits = self.saliency_residual(token_features).squeeze(-1)
        token_weights = torch.softmax(
            pretrained_attention.clamp_min(1.0e-12).log() + residual_logits,
            dim=1,
        )
        return global_embeddings, token_embeddings, token_weights

    def forward(
        self,
        global_features: torch.Tensor,
        token_features: torch.Tensor,
        pretrained_attention: torch.Tensor,
        *,
        validate_values: bool = False,
    ) -> TokenSetProxyAnchorOutput:
        global_embeddings, token_embeddings, token_weights = self.encode(
            global_features,
            token_features,
            pretrained_attention,
            validate_values=validate_values,
        )
        class_scores = token_set_class_scores(
            global_embeddings,
            token_embeddings,
            token_weights,
            self.global_proxies,
            self.token_proxies,
            set_weight=self.set_weight_tensor,
            validate_values=False,
        )
        return TokenSetProxyAnchorOutput(
            class_scores=class_scores,
            global_embeddings=global_embeddings,
            token_embeddings=token_embeddings,
            token_weights=token_weights,
        )
