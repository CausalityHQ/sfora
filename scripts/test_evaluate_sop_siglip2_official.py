"""Official evaluation stays gated and verifies each image against pinned bytes."""

import hashlib
from pathlib import Path

import pytest
from evaluate_sop_siglip2_official import VerifiedRows, validate_decision_arm
from PIL import Image


def test_decision_requires_post_selection_seed_and_pinned_receipt() -> None:
    decision = {
        "schema": "sfora-sop-siglip2-member-bank-multiseed-v1",
        "continuation_gate_pass": True,
        "replication_seeds": [179020, 179021, 179022],
        "arms": {"179020": {"bank": {"receipt_sha256": "a" * 64}}},
    }
    receipt = {
        "seed": 179020,
        "arm": "float_rank_member_bank",
        "quality": {"native_top10_exact": True},
    }
    validate_decision_arm(decision, 179020, "bank", "a" * 64, receipt)
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(
            {**decision, "continuation_gate_pass": False}, 179020, "bank", "a" * 64, receipt
        )
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(decision, 179019, "bank", "a" * 64, {**receipt, "seed": 179019})
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(decision, 179020, "bank", "b" * 64, receipt)


def test_verified_rows_reject_changed_image_bytes(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(path)
    digest = hashlib.sha256(path.read_bytes()).digest()
    rows = VerifiedRows((path,), (7,), digest)
    image, label = rows[0]
    assert image.size == (3, 2) and label == 7
    Image.new("RGB", (3, 2), color=(11, 20, 30)).save(path)
    with pytest.raises(ValueError, match="bytes"):
        rows[0]
