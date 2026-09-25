"""Multi-seed decisions must retain paired authority and reject one failed seed."""

import json
from pathlib import Path

import pytest
from compare_sop_siglip2_member_bank_multiseed import continuation_gate, validate_three_arms

EVIDENCE = (
    Path(__file__).resolve().parents[1] / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
)


def _first_seed() -> tuple[dict, dict, dict]:
    return tuple(
        json.loads((EVIDENCE / name).read_text())
        for name in (
            "member-bank-control-1000-v1.json",
            "member-bank-float-1000-v1.json",
            "member-bank-treatment-1000-v1.json",
        )
    )


def test_validates_real_first_seed_and_rejects_unpaired_schedule() -> None:
    control, floating, bank = _first_seed()
    validate_three_arms(179019, control, floating, bank)
    floating["schedule_sha256"] = "wrong"
    with pytest.raises(ValueError, match="authority"):
        validate_three_arms(179019, control, floating, bank)


def test_one_negative_seed_blocks_continuation_even_if_mean_passes() -> None:
    comparisons = {
        "float_rank": {
            "replication_seedwise_r1": [0.015, 0.015, -0.001],
            "replication_pooled_r1": {"point": 0.00967, "lower_95": 0.004},
            "replication_pooled_map_at_r": {"point": 0.01},
            "replication_seedwise_wall_ratio": [1.01, 1.02, 1.01],
        },
        "arcface": {
            "replication_seedwise_r1": [0.01, 0.01, 0.01],
            "replication_pooled_r1": {"point": 0.01, "lower_95": 0.005},
            "replication_pooled_map_at_r": {"point": 0.01},
            "replication_seedwise_wall_ratio": [1.01, 1.02, 1.01],
        },
    }
    assert continuation_gate(comparisons) is False
    comparisons["float_rank"]["replication_seedwise_r1"][2] = 0.001
    assert continuation_gate(comparisons) is True


def test_exploratory_seed_cannot_pay_for_weak_replication() -> None:
    comparisons = {
        control: {
            "seedwise_r1": [0.8, 0.002, 0.002, 0.002],
            "replication_seedwise_r1": [0.002, 0.002, 0.002],
            "replication_pooled_r1": {"point": 0.002, "lower_95": 0.001},
            "replication_pooled_map_at_r": {"point": 0.01},
            "replication_seedwise_wall_ratio": [1.01, 1.01, 1.01],
        }
        for control in ("float_rank", "arcface")
    }
    assert continuation_gate(comparisons) is False
