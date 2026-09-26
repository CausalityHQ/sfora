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


def test_bf16_decision_binds_three_seed_float_controls() -> None:
    decision = {
        "schema": "sfora-sop-siglip2-bf16-rankmatched-float-v1",
        "claim_eligible": False,
        "bank_specific_screen_pass": True,
        "seeds": [179023, 179024, 179025],
        "arms": {
            "179023": {
                "bank_receipt_sha256": "a" * 64,
                "matched_float_receipt_sha256": "a" * 64,
                "original_float_receipt_sha256": "a" * 64,
            }
        },
    }
    receipt = {
        "seed": 179023,
        "arm": "float_rank",
        "rank_coefficient": 21.93,
        "train_vision_dtype": "bf16",
        "updates": 1_000,
        "source_sha256": "328cdfbd4d35ae8037d1130c5fc889d60fe0ec25cf185c0c9ecc714a475120e4",
        "quality": {"native_top10_exact": True},
    }
    assert (
        validate_decision_arm(decision, 179023, "matched_float", "a" * 64, receipt) == "float_rank"
    )
    old_source = "ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23"
    assert (
        validate_decision_arm(
            decision,
            179023,
            "bank",
            "a" * 64,
            {
                **receipt,
                "arm": "float_rank_member_bank",
                "rank_coefficient": 8.0,
                "source_sha256": old_source,
            },
        )
        == "float_rank_member_bank"
    )
    assert (
        validate_decision_arm(
            decision,
            179023,
            "original_float",
            "a" * 64,
            {**receipt, "rank_coefficient": 8.0, "source_sha256": old_source},
        )
        == "float_rank"
    )
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(decision, 179023, "matched_float", "b" * 64, receipt)
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(
            decision, 179023, "matched_float", "a" * 64, {**receipt, "rank_coefficient": 8.0}
        )
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(decision, 179020, "matched_float", "a" * 64, receipt)


def test_coverage_decision_accepts_only_pinned_bank() -> None:
    decision = {
        "schema": "sfora-sop-siglip2-bf16-coverage-replication-v1",
        "claim_eligible": False,
        "replication_gate_pass": True,
        "seeds": [179023, 179024, 179025],
        "arms": {"179025": {"bank": {"receipt_sha256": "a" * 64}}},
    }
    receipt = {
        "seed": 179025,
        "arm": "float_rank_member_bank",
        "rank_coefficient": 8.0,
        "train_vision_dtype": "bf16",
        "updates": 1_000,
        "source_sha256": "400f6ef2d992e449ff7eabf53c2982b88db0889df0586bbf94875a1b04a6e118",
        "quality": {"native_top10_exact": True},
    }
    assert validate_decision_arm(decision, 179025, "bank", "a" * 64, receipt) == "float_rank_member_bank"
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(decision, 179025, "bank", "b" * 64, receipt)
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(decision, 179025, "matched_float", "a" * 64, receipt)
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm({**decision, "replication_gate_pass": False}, 179025, "bank", "a" * 64, receipt)


def test_true_freeze_decision_binds_both_paired_arms() -> None:
    decision = {
        "schema": "sfora-sop-true-freeze-paired-train-only-v1",
        "claim_eligible": False,
        "seed": 179026,
        "advance_fresh_seeds": True,
        "quality_pass": True,
        "cost_pass": True,
        "arms": {
            "control": {"receipt_sha256": "a" * 64},
            "freeze": {"receipt_sha256": "b" * 64},
        },
    }
    receipt = {
        "seed": 179026,
        "arm": "float_rank_member_bank",
        "source_sha256": "c5b8786c352ce6c8bedce9a5963ef43e2c18db61974e3c141c698227a23f1b3c",
        "rank_coefficient": 8.0,
        "train_vision_dtype": "bf16",
        "updates": 1_000,
        "freeze_lower_stack": True,
        "frozen_encoder_blocks": list(range(12)),
        "quality": {"native_top10_exact": True},
    }
    assert (
        validate_decision_arm(decision, 179026, "freeze", "b" * 64, receipt)
        == "float_rank_member_bank"
    )
    control = {**receipt, "freeze_lower_stack": False, "frozen_encoder_blocks": []}
    assert (
        validate_decision_arm(decision, 179026, "control", "a" * 64, control)
        == "float_rank_member_bank"
    )
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(decision, 179026, "freeze", "b" * 64, control)
    with pytest.raises(ValueError, match="gate"):
        validate_decision_arm(decision, 179026, "freeze", "a" * 64, receipt)


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
