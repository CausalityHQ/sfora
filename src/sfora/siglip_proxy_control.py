"""Frozen authority for the SigLIP-so400m pooled Proxy Anchor control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch

from sfora.substrate_screen import SUBSTRATE_F0_CLASSES
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
