"""Singleton anchors must not discard valid rank supervision from a batch."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import train_inshop_siglip2_unseen_gallery as trainer  # noqa: E402
from analyze_inshop_valid_anchor_rank import positive_training_costs  # noqa: E402
from preflight_inshop_siglip2_unseen_gallery import schedule  # noqa: E402
from train_sop_siglip2_compact import member_bank_rank_loss  # noqa: E402


def test_valid_anchor_rank_keeps_original_batch_denominator() -> None:
    torch.manual_seed(179024)
    features = torch.randn(2, 128, requires_grad=True)
    bank = torch.randn(4, 128)
    bank = torch.nn.functional.normalize(bank, dim=1)
    positives = torch.tensor([[1], [-1]], dtype=torch.long)
    ordinals = torch.tensor([0, 2], dtype=torch.long)
    head = torch.nn.Linear(128, 128)
    actual = trainer.valid_anchor_rank_loss(features, bank, head, positives, ordinals)
    expected = 0.5 * member_bank_rank_loss(
        features[:1], bank, head, positives[:1], ordinals[:1], live_head=False
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert features.grad is not None
    assert float(features.grad[0].abs().sum()) > 0
    assert torch.equal(features.grad[1], torch.zeros_like(features.grad[1]))
    with pytest.raises(ValueError, match="valid anchor"):
        trainer.valid_anchor_rank_loss(features, bank, head, positives.fill_(-1), ordinals)


def test_valid_anchor_gate_rejects_invalid_costs() -> None:
    good = {
        "training_wall_including_member_bank_init_seconds": 748.64,
        "training_peak_cuda_allocated_bytes": 12_939_458_560,
    }
    assert positive_training_costs(good) == (748.64, 12_939_458_560)
    for key, bad in (
        ("training_wall_including_member_bank_init_seconds", -1.0),
        ("training_wall_including_member_bank_init_seconds", float("nan")),
        ("training_peak_cuda_allocated_bytes", 0),
    ):
        with pytest.raises(ValueError, match="training cost"):
            positive_training_costs({**good, key: bad})


def test_equal_wall_extension_uses_frozen_three_thousand_step_prefix() -> None:
    labels = tuple(str(i) for i in range(16) for _ in range(4))
    assert trainer.schedule_horizon(1_000) == 1_000
    assert trainer.schedule_horizon(1_533) == 3_000
    assert schedule(labels, 3_000, seed=179024)[:1_533] == schedule(
        labels, 1_533, seed=179024
    )
