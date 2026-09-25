"""Contracts for the pinned SigLIP2 processor and compact trainer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

SCRIPT = Path(__file__).parents[1] / "scripts" / "train_sop_siglip2_compact.py"
SPEC = importlib.util.spec_from_file_location("train_sop_siglip2_compact", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PixelOnlyProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((len(images), 3, 256, 256))}


def test_collate_accepts_pinned_pixel_only_processor() -> None:
    rows = [(Image.new("RGB", (32, 32)), 1), (Image.new("RGB", (32, 32)), 2)]
    batch, labels = MODULE.make_collate(PixelOnlyProcessor())(rows)
    assert set(batch) == {"pixel_values"}
    assert batch["pixel_values"].shape == (2, 3, 256, 256)
    assert labels.tolist() == [1, 2]


def test_collate_rejects_missing_pixel_values() -> None:
    rows = [(Image.new("RGB", (32, 32)), 1)]
    with pytest.raises(ValueError, match="processor batch geometry"):
        MODULE.make_collate(lambda **_kwargs: {})(rows)


def test_training_receipt_binds_executed_package_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    MODULE.assert_source_imports()
    loaded = sys.modules["sfora.sop_compact_training"]
    monkeypatch.setattr(loaded, "__file__", str(tmp_path / "stale.py"))
    with pytest.raises(ValueError, match="source differs from receipt"):
        MODULE.assert_source_imports()


def test_training_precision_selects_bf16_without_loss_scaling() -> None:
    dtype, scaler = MODULE.training_precision("bf16", device="cpu")
    assert dtype is torch.bfloat16
    assert not scaler.is_enabled()
    assert scaler.get_scale() == 1.0


def test_training_precision_keeps_fp16_scaling() -> None:
    dtype, scaler = MODULE.training_precision("fp16", device="cpu")
    assert dtype is torch.float16
    assert scaler.is_enabled()
    assert scaler.get_scale() == MODULE.GRAD_SCALER_INITIAL_SCALE


def test_rank_coefficient_accepts_frozen_float_control_and_rejects_nonfinite() -> None:
    assert MODULE.validated_rank_coefficient(21.93) == 21.93
    assert MODULE.validated_rank_coefficient(8.0) == 8.0
    for value in (float("nan"), float("inf"), -1.0, 0.0, 257.0):
        with pytest.raises(ValueError, match="rank coefficient"):
            MODULE.validated_rank_coefficient(value)


def test_ordered_source_digest_detects_reassigned_feature_rows() -> None:
    ids = np.arange(59_551, dtype=np.int64)
    labels = ids // 5
    paths = np.full(59_551, "same.jpg")
    baseline = MODULE.ordered_rows_sha256(ids, labels, paths)
    swapped_ids = ids.copy()
    swapped_ids[0], swapped_ids[1] = swapped_ids[1], swapped_ids[0]
    assert MODULE.ordered_rows_sha256(swapped_ids, labels, paths) != baseline


def test_packed_full_gallery_scoring_excludes_self_and_keeps_class_mates() -> None:
    codes = torch.zeros((6, 128), dtype=torch.float32)
    for pair in range(3):
        codes[2 * pair : 2 * pair + 2, pair] = 127
    inverse = torch.full((6,), 1 / 127, dtype=torch.float16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    held = torch.tensor([0, 2, 4], dtype=torch.int64)
    scored = MODULE.score_packed_full_gallery(
        codes, inverse, labels, held, device=torch.device("cpu")
    )
    assert scored["recall_at_1"] == 1.0
    assert scored["map_at_r"] == 1.0
    assert scored["per_query_r1"] == [1.0, 1.0, 1.0]


def test_packed_full_gallery_uses_lower_ordinal_on_equal_scores() -> None:
    codes = torch.zeros((4, 128), dtype=torch.float32)
    codes[:, 0] = 127
    inverse = torch.full((4,), 1 / 127, dtype=torch.float16)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    held = torch.tensor([0, 2], dtype=torch.int64)
    scored = MODULE.score_packed_full_gallery(
        codes, inverse, labels, held, device=torch.device("cpu")
    )
    assert scored["per_query_r1"] == [1.0, 0.0]


def test_member_bank_positive_rows_exclude_self_and_pad() -> None:
    classes = np.asarray([2, 1, 2, 1, 2], dtype=np.int64)
    assert MODULE.member_bank_positive_ordinals(classes).tolist() == [
        [2, 4],
        [3, -1],
        [0, 4],
        [1, -1],
        [0, 2],
    ]


def test_member_bank_singleton_row_can_be_left_empty_for_skipped_rank_batches() -> None:
    classes = np.asarray([2, 1, 2, 3, 2], dtype=np.int64)
    with pytest.raises(ValueError, match="no positive"):
        MODULE.member_bank_positive_ordinals(classes)
    assert MODULE.member_bank_positive_ordinals(classes, allow_singletons=True).tolist() == [
        [2, 4],
        [-1, -1],
        [0, 4],
        [-1, -1],
        [0, 2],
    ]


def test_member_bank_refresh_uses_last_augmented_view_for_duplicate_row() -> None:
    rows, positions = MODULE.member_bank_refresh_rows((5, 3, 5, 4, 3))
    assert rows == (3, 4, 5)
    assert positions == (4, 3, 2)


def test_live_head_bank_keeps_unit_sources_and_refreshes_from_encoder() -> None:
    generator = torch.Generator().manual_seed(179025)
    source = torch.randn(4, 1024, generator=generator)
    head = torch.nn.Linear(1024, 128)
    initial = MODULE.member_bank_initial_values(source, head, live_head=True)
    torch.testing.assert_close(initial, F.normalize(source, dim=1))
    assert initial.shape == (4, 1024)
    old_initial = MODULE.member_bank_initial_values(source, head, live_head=False)
    torch.testing.assert_close(old_initial, F.normalize(head(F.normalize(source, dim=1)), dim=1))
    assert old_initial.shape == (4, 128)

    current_source = torch.randn(4, 1024, generator=generator, requires_grad=True)
    current_head = head(F.normalize(current_source, dim=1))
    positions = torch.tensor([3, 1])
    refreshed = MODULE.member_bank_refresh_values(
        current_source, current_head, positions, live_head=True
    )
    torch.testing.assert_close(refreshed, F.normalize(current_source.detach()[positions], dim=1))
    assert not refreshed.requires_grad
    old_refreshed = MODULE.member_bank_refresh_values(
        current_source, current_head, positions, live_head=False
    )
    torch.testing.assert_close(old_refreshed, F.normalize(current_head.detach()[positions], dim=1))


def test_live_head_trainer_loss_reaches_head_through_cached_candidates() -> None:
    generator = torch.Generator().manual_seed(179026)
    source = F.normalize(torch.randn(6, 1024, generator=generator), dim=1)
    head = torch.nn.Linear(1024, 128)
    anchors = F.normalize(torch.randn(2, 128, generator=generator), dim=1)
    positives = torch.tensor([[1], [0]], dtype=torch.long)
    self_rows = torch.tensor([0, 1], dtype=torch.long)
    live = MODULE.member_bank_rank_loss(anchors, source, head, positives, self_rows, live_head=True)
    gradient = torch.autograd.grad(live, head.weight)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_live_head_cost_receipt_rejects_slow_candidate() -> None:
    receipt = {
        "schema": "sfora-sop-siglip2-live-head-isolated-cost-v1",
        "claim_eligible": False,
        "device": "cuda",
        "hardware": "NVIDIA GB10",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "matmul_allow_tf32": False,
        "rows": 53_700,
        "anchors": 64,
        "positives_per_anchor": 11,
        "timed_blocks": 4,
        "calls_per_block_per_arm": 10,
        "forward_loss_difference": 0.0,
        "persistent_bank_bytes": {
            "detached_bank": 53_700 * 128 * 4,
            "live_head": 53_700 * 1024 * 4,
        },
        "arms": {
            "detached_bank": {"calls": 40, "p95_ms": 14.0},
            "live_head": {
                "calls": 40,
                "p95_ms": 151.0,
                "max_incremental_allocated_bytes": 500_000_000,
            },
        },
        "source_sha256": {
            "script": "bfcf51e3e6e9fee5ba4a0c45cf144177e8571aa00681236ebaeff9350ec09fb3",
            "live_loss": MODULE.sha256(Path(MODULE.live_head_bank_loss.__code__.co_filename)),
            "detached_loss": "a57a1b8cb4722a12dbc8fd255255632b05aff918c8c66e46e5708c2da9e4854a",
        },
    }
    with pytest.raises(ValueError, match="live-head cost gate"):
        MODULE.validate_live_head_cost_receipt(receipt)
    receipt["arms"]["live_head"]["p95_ms"] = 100.0
    MODULE.validate_live_head_cost_receipt(receipt)
    receipt["matmul_allow_tf32"] = True
    with pytest.raises(ValueError, match="live-head cost gate"):
        MODULE.validate_live_head_cost_receipt(receipt)
    receipt["matmul_allow_tf32"] = False
    receipt["source_sha256"]["live_loss"] = "0" * 64
    with pytest.raises(ValueError, match="live-head cost gate"):
        MODULE.validate_live_head_cost_receipt(receipt)
