"""Type-safe, composable *method* bricks for SFORA.

A method is a **base loss + composable modifiers**, which is exactly the research
finding: our EMA-teacher relational distillation universally improves any base, so
the strongest method per dataset is that base with the distillation stacked on it.

    from sfora.method import HIST, ProxyAnchor, Distill, IsNorm, herd, pa_distill

    HERD       = IsNorm(Distill(HIST()))     # == herd();      best on CUB
    PADistill  = Distill(ProxyAnchor())      # == pa_distill(); best on Cars

Every brick is an immutable ``Objective``: it takes an ``ImageEndToEndConfig`` and
returns a new one with its fields set (``configure``). Modifiers (``Distill``,
``IsNorm``) wrap another ``Objective`` and layer their own fields on top, so the
composition is type-checked and order-independent where it should be. Bricks train
through the existing, verified trainer via :func:`build_config`, so composing them
cannot regress the benchmarked numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sfora.image_end_to_end import ImageEndToEndConfig


@runtime_checkable
class Objective(Protocol):
    """A composable training objective: a base loss or a modifier wrapping one."""

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``IsNorm(Distill(HIST))``."""

    def configure(self, config: ImageEndToEndConfig) -> ImageEndToEndConfig:
        """Return ``config`` updated with this brick's fields (pure, no mutation)."""


# ─────────────────────────────  base-loss bricks  ─────────────────────────────
@dataclass(frozen=True)
class HIST:
    """HIST hypergraph semantic-tuplet loss (per-class Gaussian prototypes + HGNN)."""

    tau: float = 32.0
    alpha: float = 0.9
    lambda_s: float = 1.0

    @property
    def name(self) -> str:
        return "HIST"

    def configure(self, config: ImageEndToEndConfig) -> ImageEndToEndConfig:
        return config.model_copy(
            update={
                "objectives": ("hist",),
                "proxy_count_per_class": 0,
                "hist_tau": self.tau,
                "hist_alpha": self.alpha,
                "hist_lambda_s": self.lambda_s,
            }
        )


@dataclass(frozen=True)
class ProxyAnchor:
    """Proxy Anchor loss (one learnable proxy per class, soft margins)."""

    alpha: float = 32.0
    delta: float = 0.1
    proxies_per_class: int = 1

    @property
    def name(self) -> str:
        return "ProxyAnchor"

    def configure(self, config: ImageEndToEndConfig) -> ImageEndToEndConfig:
        return config.model_copy(
            update={
                "objectives": ("proxy_anchor",),
                "proxy_count_per_class": self.proxies_per_class,
                "proxy_anchor_alpha": self.alpha,
                "proxy_anchor_delta": self.delta,
            }
        )


@dataclass(frozen=True)
class FusedHistProxyAnchor:
    """Single model trained with HIST + weight*ProxyAnchor.

    Note: benchmarked as a *negative* result — a compromise worse than the better
    base on each dataset. Kept as a brick so the finding stays reproducible.
    """

    fusion_weight: float = 1.0

    @property
    def name(self) -> str:
        return "FusedHistProxyAnchor"

    def configure(self, config: ImageEndToEndConfig) -> ImageEndToEndConfig:
        return config.model_copy(
            update={
                "objectives": ("hist_proxy_anchor",),
                "proxy_count_per_class": 1,
                "proxy_fusion_weight": self.fusion_weight,
            }
        )


# ──────────────────────────────  modifier bricks  ─────────────────────────────
@dataclass(frozen=True)
class Distill:
    """Add the EMA-teacher relational self-distillation on top of any base.

    This is SFORA's core contribution: a slow momentum copy of the model produces a
    soft neighbourhood distribution over the batch and the student is trained to
    match it. It improves every base loss on every dataset benchmarked.
    """

    base: Objective
    weight: float = 1.0
    momentum: float = 0.999
    tau: float = 0.1

    @property
    def name(self) -> str:
        return f"Distill({self.base.name})"

    def configure(self, config: ImageEndToEndConfig) -> ImageEndToEndConfig:
        return self.base.configure(config).model_copy(
            update={
                "ema_distill_weight": self.weight,
                "ema_momentum": self.momentum,
                "ema_distill_tau": self.tau,
            }
        )


@dataclass(frozen=True)
class IsNorm:
    """Add the reference ``LayerNorm`` (no-affine) ``is_norm`` head to any base."""

    base: Objective

    @property
    def name(self) -> str:
        return f"IsNorm({self.base.name})"

    def configure(self, config: ImageEndToEndConfig) -> ImageEndToEndConfig:
        return self.base.configure(config).model_copy(update={"embedding_layer_norm": True})


# ─────────────────────────────  headline methods  ────────────────────────────
def herd(*, distill_weight: float = 1.0, momentum: float = 0.999, tau: float = 0.1) -> Objective:
    """HERD = HIST + is_norm head + EMA-teacher relational distillation (best on CUB)."""
    return IsNorm(Distill(HIST(), weight=distill_weight, momentum=momentum, tau=tau))


def pa_distill(
    *, distill_weight: float = 1.0, momentum: float = 0.999, tau: float = 0.1
) -> Objective:
    """Proxy Anchor + EMA-teacher relational distillation (best on Cars)."""
    return Distill(ProxyAnchor(), weight=distill_weight, momentum=momentum, tau=tau)


def build_config(method: Objective, base: ImageEndToEndConfig) -> ImageEndToEndConfig:
    """Compile a method brick into a runnable ``ImageEndToEndConfig`` from a base."""
    return method.configure(base)
