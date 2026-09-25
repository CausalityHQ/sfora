"""The cached-head gate rejects a changed matched recipe or nonfinite result."""

from __future__ import annotations

import copy
import math

import pytest
from compare_sop_siglip2_cached_head_probe import survival_gate, validate_pair
from train_sop_siglip2_cached_head_probe import (
    EXPECTED_CLASSIFIER_SHA256,
    EXPECTED_HEAD_SHA256,
    EXPECTED_PCA_SHA256,
    EXPECTED_SCHEDULE_SHA256,
)


def paired_receipts() -> tuple[dict, dict]:
    common = {
        "schema": "sfora-sop-siglip2-cached-head-probe-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout; full TRAIN gallery",
        "seed": 179023,
        "steps": 1000,
        "source_archive_sha256": (
            "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
        ),
        "source_cache_sha256": ("d15f76e459b90836df2807260a34d06692e2f2e5b8ba17b3449ed56500b8357a"),
        "export_receipt_sha256": (
            "3d49e039c72c0677591835752133caf1cbfd0ea483c23a901773748b44d843fb"
        ),
        "source_files_sha256": {"probe": "digest"},
        "native_library_sha256": "native",
        "tileiras_sha256": "toolchain",
        "initial_head_sha256": EXPECTED_HEAD_SHA256,
        "initial_classifier_sha256": EXPECTED_CLASSIFIER_SHA256,
        "source_pca_sha256": EXPECTED_PCA_SHA256,
        "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        "first_input_batch_sha256": ["batch"] * 10,
        "query_image_ids_sha256": "queries",
        "cache_encode_seconds": 508.51690101856366,
        "step_seconds": [0.01] * 1000,
        "first_loss": 3.0,
        "last_loss": 2.0,
        "quality": {"native_top10_exact": True, "gallery_wire_bytes_per_row": 130},
    }
    return {**copy.deepcopy(common), "arm": "arcface"}, {
        **copy.deepcopy(common),
        "arm": "live_head_bank",
    }


def test_pair_requires_identical_frozen_sources_and_schedule() -> None:
    arcface, bank = paired_receipts()
    validate_pair(arcface, bank, "queries")
    bank["source_files_sha256"]["probe"] = "changed"
    with pytest.raises(ValueError, match="paired authority"):
        validate_pair(arcface, bank, "queries")
    bank = paired_receipts()[1]
    bank["schedule_sha256"] = "changed"
    with pytest.raises(ValueError, match="paired authority"):
        validate_pair(arcface, bank, "queries")


def test_survival_gate_requires_quality_and_half_time() -> None:
    r1 = {"point": 0.01, "lower_95": 0.002}
    mapped = {"point": 0.001}
    assert survival_gate(r1, mapped, 0.86, 560.0)
    assert not survival_gate(r1, mapped, 0.86, 580.0)
    assert not survival_gate({**r1, "lower_95": math.nan}, mapped, 0.86, 560.0)
    assert not survival_gate(r1, mapped, 0.84, 560.0)
