"""The new replication analyzer rejects mixed precision and unpaired arms."""

import json
from pathlib import Path

import pytest
from compare_sop_siglip2_bf16_member_bank_multiseed import validate_three_arms

EVIDENCE = (
    Path(__file__).resolve().parents[1] / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
)


def _synthetic_bf16_seed() -> tuple[dict, dict, dict]:
    rows = tuple(
        json.loads((EVIDENCE / name).read_text())
        for name in (
            "member-bank-control-1000-v1.json",
            "member-bank-float-1000-v1.json",
            "member-bank-treatment-1000-v1.json",
        )
    )
    for row in rows:
        row["seed"] = 179023
        row["source_sha256"] = (
            "ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23"
        )
        row["source_files_sha256"]["scripts/train_sop_siglip2_compact.py"] = row["source_sha256"]
        row["train_vision_dtype"] = "bf16"
        row["precision"] = "fp32 parameters, bf16 vision autocast, fp32 objective, no loss scaling"
        row["grad_scaler_initial_scale"] = 1.0
    return rows


def test_bf16_analyzer_requires_matched_precision() -> None:
    arcface, floating, bank = _synthetic_bf16_seed()
    validate_three_arms(179023, arcface, floating, bank)
    bank["train_vision_dtype"] = "fp16"
    with pytest.raises(ValueError, match="matched authority"):
        validate_three_arms(179023, arcface, floating, bank)


def test_bf16_analyzer_rejects_unpaired_schedule() -> None:
    arcface, floating, bank = _synthetic_bf16_seed()
    floating["schedule_sha256"] = "wrong"
    with pytest.raises(ValueError, match="matched authority"):
        validate_three_arms(179023, arcface, floating, bank)
