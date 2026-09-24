"""Contracts for the pinned SigLIP2 processor and compact trainer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

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
