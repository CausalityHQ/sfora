"""Check the causal pairing used by the resolution continuation screen."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_sop_b16_resolution_continuation import (  # noqa: E402
    augmented_image,
    optimizer_step_range,
)


def test_224_and_upsampled_336_share_exact_cropped_pixels():
    rng = np.random.default_rng(29)
    image = Image.fromarray(rng.integers(0, 256, (395, 281, 3), dtype=np.uint8))
    for occurrence in (0, 17, 255):
        base = augmented_image(image, occurrence, "b16_224")
        control = augmented_image(image, occurrence, "b16_336_upsampled")
        expected = F.interpolate(
            base.unsqueeze(0), size=(336, 336), mode="bicubic", align_corners=False
        ).squeeze(0)
        assert torch.equal(control, expected)
        assert torch.equal(base, augmented_image(image, occurrence, "b16_224"))


def test_native_detail_has_same_sampling_seed_but_more_source_pixels():
    pixels = np.arange(384 * 384 * 3, dtype=np.uint32).reshape(384, 384, 3)
    image = Image.fromarray(pixels.astype(np.uint8))
    native = augmented_image(image, 3, "b16_336_detail")
    enlarged = augmented_image(image, 3, "b16_336_upsampled")
    assert native.shape == enlarged.shape == (3, 336, 336)
    assert not torch.equal(native, enlarged)


def test_deep_copied_adamw_state_keeps_each_arm_clock_independent():
    weight = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([weight])
    weight.square().sum().backward()
    optimizer.step()
    initial = optimizer.state_dict()
    for _ in range(2):
        arm_weight = torch.nn.Parameter(torch.tensor([1.0]))
        arm_optimizer = torch.optim.AdamW([arm_weight])
        arm_optimizer.load_state_dict(copy.deepcopy(initial))
        assert optimizer_step_range(arm_optimizer) == (1, 1)
        arm_weight.square().sum().backward()
        arm_optimizer.step()
        assert optimizer_step_range(arm_optimizer) == (2, 2)
        assert initial["state"][0]["step"].item() == 1
