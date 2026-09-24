"""Paired live-timing gates for the pretrained substrate screen."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_sop_siglip2_vs_unicom import alternating_pair_order, verify_live_features


def test_alternating_pair_order_balances_first_runner():
    assert [alternating_pair_order(index) for index in range(4)] == [
        ("unicom_l14_336", "siglip2_l16_256"),
        ("siglip2_l16_256", "unicom_l14_336"),
        ("unicom_l14_336", "siglip2_l16_256"),
        ("siglip2_l16_256", "unicom_l14_336"),
    ]


def test_live_feature_parity_rejects_wrong_encoder_or_preprocessing():
    cached = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    assert verify_live_features(cached.clone(), cached) == 1.0
    with pytest.raises(ValueError, match="live feature parity"):
        verify_live_features(cached[[1, 0]], cached)
