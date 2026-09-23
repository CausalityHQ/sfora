"""UNICOM-like SOP update schedule for matched retrieval controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReferenceRecipe:
    steps_per_epoch: int
    total_updates: int
    peak_learning_rates: tuple[float, float, float]
    weight_decay: float = 0.0
    margin: float = 0.25
    scale: float = 32.0
    grad_scaler_growth_interval: int = 1_000_000_000


def reference_recipe(*, fit_images: int, batch_size: int, epochs: int = 64) -> ReferenceRecipe:
    """Return SOP B/16 launch-script hyperparameters and update budget."""

    if fit_images < 2 or batch_size < 2 or epochs < 1:
        raise ValueError("reference SOP fit inventory or training budget differs")
    steps_per_epoch = math.ceil(fit_images / batch_size)
    return ReferenceRecipe(
        steps_per_epoch=steps_per_epoch,
        total_updates=epochs * steps_per_epoch,
        peak_learning_rates=(1e-5, 1e-4, 1e-4),
    )


def reference_scheduler(
    optimizer: torch.optim.Optimizer, recipe: ReferenceRecipe
) -> torch.optim.lr_scheduler.OneCycleLR:
    if len(optimizer.param_groups) != len(recipe.peak_learning_rates):
        raise ValueError("reference SOP optimizer parameter groups differ")
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=list(recipe.peak_learning_rates),
        total_steps=recipe.total_updates,
        pct_start=0.1,
    )


def reference_train_transform(image_size: int = 224):
    """Use the authenticated UNICOM retrieval script's training augmentation."""

    if image_size not in (224, 336):
        raise ValueError("reference SOP image size differs")
    from timm.data import create_transform

    return create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
