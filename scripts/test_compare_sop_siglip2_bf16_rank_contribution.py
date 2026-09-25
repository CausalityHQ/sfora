"""Fit-only rank diagnostic must preserve paired first-batch authority."""

import pytest
from compare_sop_siglip2_bf16_rank_contribution import (
    TRAINER_SHA256,
    coefficient_from_ratios,
    validate_diagnostic_pair,
    validate_gate_receipt_hashes,
)


def test_coefficient_uses_median_paired_gradient_ratio() -> None:
    assert coefficient_from_ratios([(1.0, 2.0), (2.0, 6.0), (4.0, 8.0)]) == 16.0
    with pytest.raises(ValueError, match="coefficient"):
        coefficient_from_ratios([(0.0, 1.0)] * 3)


def test_diagnostic_rejects_changed_first_batch() -> None:
    common = {
        "seed": 179023,
        "source_sha256": TRAINER_SHA256,
        "source_files_sha256": {"scripts/train_sop_siglip2_compact.py": TRAINER_SHA256},
        "train_vision_dtype": "bf16",
        "first_input_batch_sha256": ["same"],
        "initial_head_sha256": "head",
        "initial_classifier_sha256": "classifier",
        "model_file_sha256": {"model.safetensors": "model"},
        "source_archive_sha256": "archive",
        "query_image_ids_sha256": "queries",
        "quality": None,
    }
    float_diag = {
        **common,
        "arm": "float_rank",
        "updates": 1,
        "rank_to_arcface_head_gradient_ratio": [0.5],
    }
    bank_diag = {
        **common,
        "arm": "float_rank_member_bank",
        "updates": 1,
        "rank_to_arcface_head_gradient_ratio": [1.0],
    }
    float_full = {**common, "arm": "float_rank", "updates": 1000}
    bank_full = {**common, "arm": "float_rank_member_bank", "updates": 1000}
    assert validate_diagnostic_pair(179023, float_diag, bank_diag, float_full, bank_full) == (
        0.5,
        1.0,
    )
    with pytest.raises(ValueError, match="diagnostic"):
        validate_diagnostic_pair(
            179023,
            float_diag,
            {**bank_diag, "first_input_batch_sha256": ["changed"]},
            float_full,
            bank_full,
        )


def test_gate_hashes_bind_full_training_receipts(tmp_path) -> None:
    floating = tmp_path / "float.json"
    bank = tmp_path / "bank.json"
    floating.write_text("float")
    bank.write_text("bank")
    from hashlib import sha256

    gate = {
        "arms": {
            "179023": {
                "float_rank": {"receipt_sha256": sha256(b"float").hexdigest()},
                "bank": {"receipt_sha256": sha256(b"bank").hexdigest()},
            }
        }
    }
    paths = {"float_rank": floating, "bank": bank}
    validate_gate_receipt_hashes(gate, 179023, paths)
    bank.write_text("changed")
    with pytest.raises(ValueError, match="gate receipt"):
        validate_gate_receipt_hashes(gate, 179023, paths)
