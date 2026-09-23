"""Reference-like SOP training controls keep the upstream budget explicit."""

import math

import pytest
import torch
from PIL import Image

from sfora.sop_reference_recipe import (
    reference_recipe,
    reference_scheduler,
    reference_train_transform,
)


def test_reference_budget_tracks_sop_b16_launch_script():
    recipe = reference_recipe(fit_images=53_700, batch_size=64)
    assert recipe.steps_per_epoch == math.ceil(53_700 / 64)
    assert recipe.total_updates == 53_760
    assert recipe.peak_learning_rates == (1e-5, 1e-4, 1e-4)
    assert recipe.weight_decay == 0.0
    assert recipe.margin == 0.25
    assert recipe.scale == 32.0
    assert recipe.grad_scaler_growth_interval == 1_000_000_000


def test_reference_scheduler_warms_then_decays_all_parameter_groups():
    recipe = reference_recipe(fit_images=256, batch_size=64, epochs=2)
    params = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(3)]
    optimizer = torch.optim.AdamW(
        [{"params": [param]} for param in params], weight_decay=recipe.weight_decay
    )
    scheduler = reference_scheduler(optimizer, recipe)
    assert len(scheduler.get_last_lr()) == 3
    assert scheduler.total_steps == 8
    assert [group["max_lr"] for group in optimizer.param_groups] == list(recipe.peak_learning_rates)
    initial = scheduler.get_last_lr()
    for _ in range(recipe.total_updates):
        optimizer.step()
        scheduler.step()
    assert all(final < peak for final, peak in zip(scheduler.get_last_lr(), initial, strict=True))


def test_reference_scheduler_rejects_missing_classifier_group():
    recipe = reference_recipe(fit_images=256, batch_size=64, epochs=2)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))])
    with pytest.raises(ValueError, match="parameter groups"):
        reference_scheduler(optimizer, recipe)


def test_reference_augmentation_produces_normalized_224_pixel_images():
    transform = reference_train_transform(224)
    description = repr(transform)
    assert "RandAugment" in description
    assert "RandomErasing" in description
    assert "bicubic" in description.lower()
    image = Image.new("RGB", (300, 280), color=(127, 90, 40))
    output = transform(image)
    assert output.shape == (3, 224, 224)
    assert bool(torch.isfinite(output).all())
