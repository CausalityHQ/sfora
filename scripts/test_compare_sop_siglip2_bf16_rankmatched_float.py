"""The rank-matched control accepts only the frozen training change."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
from compare_sop_siglip2_bf16_rankmatched_float import (
    NEW_TRAINER_SHA256,
    TRAINER_RELATIVE,
    rankmatched_gate,
    validate_authority,
    validate_canary,
    validate_original_float,
    validate_pair,
)

EVIDENCE = (
    Path(__file__).resolve().parents[1] / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
)


def test_frozen_gate_and_diagnostic_authority() -> None:
    gate = json.loads((EVIDENCE / "bf16-member-bank-multiseed-v1.json").read_text())
    diagnostic = json.loads((EVIDENCE / "bf16-rank-contribution-analysis-v1.json").read_text())
    validate_authority(gate, diagnostic)
    diagnostic["rank_coefficient_float_control"] = 8.0
    with pytest.raises(ValueError, match="source authority"):
        validate_authority(gate, diagnostic)


def test_only_trainer_source_and_coefficient_may_change() -> None:
    bank = json.loads((EVIDENCE / "bf16-seed179023-bank-v1.json").read_text())
    gate = json.loads((EVIDENCE / "bf16-member-bank-multiseed-v1.json").read_text())
    floating = copy.deepcopy(bank)
    floating["arm"] = "float_rank"
    floating["source_sha256"] = NEW_TRAINER_SHA256
    floating["source_files_sha256"][TRAINER_RELATIVE] = NEW_TRAINER_SHA256
    floating["rank_coefficient"] = 21.93
    validate_pair(179023, bank, floating, gate)
    floating["source_files_sha256"]["src/sfora/deployed_code_rank.py"] = "0" * 64
    with pytest.raises(ValueError, match="paired authority"):
        validate_pair(179023, bank, floating, gate)
    floating["source_files_sha256"] = copy.deepcopy(bank["source_files_sha256"])
    floating["source_files_sha256"][TRAINER_RELATIVE] = NEW_TRAINER_SHA256
    floating["rank_to_arcface_head_gradient_ratio"] = [0.1] * 1000
    with pytest.raises(ValueError, match="paired authority"):
        validate_pair(179023, bank, floating, gate)


def test_rankmatched_gate_rejects_nan_and_threshold_misses() -> None:
    r1 = [0.006, 0.007, 0.008]
    pooled = {"point": 0.007, "lower_95": 0.003}
    mapped = {"point": 0.01}
    wall = [1.01, 1.02, 1.03]
    assert rankmatched_gate(r1, pooled, mapped, wall)
    assert not rankmatched_gate(r1, {**pooled, "lower_95": math.nan}, mapped, wall)
    assert not rankmatched_gate(r1, pooled, mapped, [1.01, 1.16, 1.03])
    assert not rankmatched_gate([0.006, -0.001, 0.008], pooled, mapped, wall)


def test_canary_requires_finite_matched_first_step() -> None:
    original = json.loads((EVIDENCE / "bf16-seed179023-float-rank-v1.json").read_text())
    diagnostic = json.loads((EVIDENCE / "bf16-rank-contribution-analysis-v1.json").read_text())
    canary = copy.deepcopy(original)
    canary["source_sha256"] = NEW_TRAINER_SHA256
    canary["rank_coefficient"] = 21.93
    canary["updates"] = 1
    canary["quality"] = None
    canary["step_seconds"] = canary["step_seconds"][:1]
    canary["first_input_batch_sha256"] = canary["first_input_batch_sha256"][:1]
    canary["source_files_sha256"][TRAINER_RELATIVE] = NEW_TRAINER_SHA256
    canary["rank_to_arcface_head_gradient_ratio"] = [
        diagnostic["first_step_ratios"]["179023"]["float_ratio"] * 21.93 / 8.0
    ]
    validate_canary(canary, original, diagnostic)
    canary["rank_to_arcface_head_gradient_ratio"] = [math.nan]
    with pytest.raises(ValueError, match="canary authority"):
        validate_canary(canary, original, diagnostic)
    canary["rank_to_arcface_head_gradient_ratio"] = [0.1]
    with pytest.raises(ValueError, match="canary authority"):
        validate_canary(canary, original, diagnostic)
    canary["rank_to_arcface_head_gradient_ratio"] = [
        diagnostic["first_step_ratios"]["179023"]["float_ratio"] * 21.93 / 8.0
    ]
    canary["source_files_sha256"]["src/sfora/deployed_code_rank.py"] = "0" * 64
    with pytest.raises(ValueError, match="canary authority"):
        validate_canary(canary, original, diagnostic)


def test_original_float_is_bound_to_matched_recipe() -> None:
    bank = json.loads((EVIDENCE / "bf16-seed179023-bank-v1.json").read_text())
    original = json.loads((EVIDENCE / "bf16-seed179023-float-rank-v1.json").read_text())
    matched = copy.deepcopy(original)
    matched["source_sha256"] = NEW_TRAINER_SHA256
    matched["source_files_sha256"][TRAINER_RELATIVE] = NEW_TRAINER_SHA256
    matched["rank_coefficient"] = 21.93
    validate_original_float(179023, original, matched, bank)
    original["rank_to_arcface_head_gradient_ratio"] = [0.1] * 1_000
    with pytest.raises(ValueError, match="original float authority"):
        validate_original_float(179023, original, matched, bank)
