"""The paired live probe balances arm order and reports latency in milliseconds."""

import pytest
from benchmark_sop_siglip2_bank_live import (
    BF16_TRAINER_SHA256,
    paired_order,
    summarize_ns,
    validate_bf16_pair,
)


def test_paired_order_balances_position_every_four_blocks() -> None:
    assert [paired_order(i) for i in range(4)] == [
        ("control", "bank"),
        ("bank", "control"),
        ("bank", "control"),
        ("control", "bank"),
    ]


def test_summary_keeps_ns_to_ms_units_and_count() -> None:
    summary = summarize_ns([1_000_000, 2_000_000, 3_000_000, 4_000_000])
    assert summary["calls"] == 4
    assert summary["p50_ms"] == 2.5
    assert summary["mean_ms"] == 2.5


def test_bf16_pair_rejects_unmatched_authority() -> None:
    common = {
        "seed": 179024,
        "source_sha256": BF16_TRAINER_SHA256,
        "source_files_sha256": {"scripts/train_sop_siglip2_compact.py": BF16_TRAINER_SHA256},
        "train_vision_dtype": "bf16",
        "updates": 1000,
        "schedule_sha256": "schedule",
        "first_input_batch_sha256": ["batch"] * 10,
        "initial_head_sha256": "head",
        "initial_classifier_sha256": "classifier",
        "model_file_sha256": {"model.safetensors": "model"},
        "source_archive_sha256": "archive",
        "query_image_ids_sha256": "queries",
        "native_library_sha256": "native",
        "tileiras_sha256": "toolchain",
        "quality": {"native_top10_exact": True, "gallery_wire_bytes_per_row": 130},
    }
    control = {**common, "arm": "arcface"}
    bank = {**common, "arm": "float_rank_member_bank"}
    validate_bf16_pair(control, bank)
    with pytest.raises(ValueError, match="BF16 pair"):
        validate_bf16_pair(control, {**bank, "schedule_sha256": "changed"})
    with pytest.raises(ValueError, match="BF16 pair"):
        validate_bf16_pair(control, {**bank, "arm": "float_rank"})
