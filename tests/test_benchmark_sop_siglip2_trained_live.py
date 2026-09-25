"""Contracts for trained live-query self exclusion."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_sop_siglip2_trained_live.py"
SPEC = importlib.util.spec_from_file_location("benchmark_sop_siglip2_trained_live", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_first_nonself_skips_query_image_even_when_it_is_first() -> None:
    assert MODULE.first_nonself(np.asarray([[7, 19, 2]], dtype=np.int64), 7) == 19


def test_first_nonself_rejects_missing_nonself_result() -> None:
    with pytest.raises(ValueError, match="nonself top-10"):
        MODULE.first_nonself(np.asarray([[7, 7]], dtype=np.int64), 7)


class EchoVision(torch.nn.Module):
    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(pooler_output=pixel_values)


def test_native_fp16_dispatch_casts_pixels_before_the_vision_tower() -> None:
    pixels = torch.ones((1, 3, 2, 2), dtype=torch.float32)
    result = MODULE.pooled_output(EchoVision(), pixels, "fp16_native")
    assert result.dtype == torch.float16
    assert pixels.dtype == torch.float32


def test_precision_dispatch_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="inference precision"):
        MODULE.pooled_output(EchoVision(), torch.ones((1, 3, 2, 2)), "unknown")


def test_paired_order_balances_first_and_second_position() -> None:
    assert [MODULE.paired_order(i) for i in range(4)] == [
        ("fp32_autocast", "fp16_native"),
        ("fp16_native", "fp32_autocast"),
        ("fp16_native", "fp32_autocast"),
        ("fp32_autocast", "fp16_native"),
    ]
