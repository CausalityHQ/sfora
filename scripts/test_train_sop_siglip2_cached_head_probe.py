"""The cached-head probe counts a pinned source-feature acquisition cost."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import train_sop_siglip2_cached_head_probe as probe


def test_export_receipt_binds_cache_and_nonfree_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/compact_metric/sop-siglip2-substrate-v1/export-receipt.json"
    )
    receipt = json.loads(evidence.read_text())
    archive = Path("archive.npz")
    features = Path("train_features.npy")
    native = Path("score.so")
    digests = {
        archive: probe.ARCHIVE_SHA256,
        features: probe.FEATURE_SHA256,
        native: probe.NATIVE_SHA256,
    }
    monkeypatch.setattr(probe, "sha256", lambda path: digests[path])
    assert probe.validate_export(receipt, archive, features, native) == pytest.approx(
        508.51690101856366
    )
    for field, value in (("encode_wall_seconds", math.nan), ("features_sha256", "0" * 64)):
        changed = copy.deepcopy(receipt)
        changed[field] = value
        with pytest.raises(ValueError, match="source authority"):
            probe.validate_export(changed, archive, features, native)
