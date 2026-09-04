from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluate_unicom_rank_finish_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location("rank_finish_confirmation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _result(seed: int, delta: float, r1: float = 0.0, r10: float = 0.0):
    metrics = {
        "map_at_r": MODULE.BASELINE["map_at_r"] + delta,
        "recall_at_1": MODULE.BASELINE["recall_at_1"] + r1,
        "recall_at_10": MODULE.BASELINE["recall_at_10"] + r10,
        "recall_at_20": 1.0,
        "recall_at_30": 1.0,
    }
    value = {
        "schema": "unicom-rank-finish-screen-v1",
        "claim_eligible": False,
        "source_commit": str(seed) * 40,
        "baseline": dict(MODULE.BASELINE),
        "history": [{"epoch": 8, "metrics": metrics, "steps": 161}],
        "decision": {
            "status": "PROMOTE",
            "epoch6_delta_map": 0.01,
            "epoch8_deltas": {
                "map_at_r": delta,
                "recall_at_1": r1,
                "recall_at_10": r10,
            },
        },
        "status": "PROMOTE",
    }
    if seed:
        value["finish_seed"] = seed
        value["model_artifact"] = {
            "path": f"/tmp/seed-{seed}.pt",
            "sha256": str(seed + 3) * 64,
            "bytes": 100 + seed,
        }
    return value


def test_confirmation_requires_all_seeds_and_mean_promotion() -> None:
    decision = MODULE.evaluate_confirmation(
        [_result(0, 0.013), _result(1, 0.011), _result(2, 0.009)]
    )

    assert decision["status"] == "CONFIRM"
    assert decision["confirmation_mean_delta_map_at_r"] == pytest.approx(0.010)
    assert [row["finish_seed"] for row in decision["seeds"]] == [0, 1, 2]


def test_confirmation_mean_excludes_discovery_seed() -> None:
    decision = MODULE.evaluate_confirmation(
        [_result(0, 0.100), _result(1, 0.011), _result(2, 0.009)]
    )

    assert decision["confirmation_mean_delta_map_at_r"] == pytest.approx(0.010)


@pytest.mark.parametrize(
    "results",
    (
        [_result(0, 0.013), _result(1, 0.011), _result(2, 0.0029)],
        [_result(0, 0.013), _result(1, 0.011), _result(2, 0.009, r1=-0.0011)],
        [_result(0, 0.010), _result(1, 0.010), _result(2, 0.009)],
    ),
)
def test_confirmation_rejects_seed_or_mean_failure(results) -> None:
    assert MODULE.evaluate_confirmation(results)["status"] == "REJECT"


def test_confirmation_rejects_seed_order_and_stale_decision() -> None:
    values = [_result(0, 0.013), _result(1, 0.011), _result(2, 0.009)]
    with pytest.raises(ValueError):
        MODULE.evaluate_confirmation([values[1], values[0], values[2]])
    values[2]["decision"]["epoch8_deltas"]["map_at_r"] = 0.5
    with pytest.raises(ValueError):
        MODULE.evaluate_confirmation(values)


def test_canonical_confirmation_is_claim_ineligible() -> None:
    result = {
        "schema": "unicom-rank-finish-confirmation-v1",
        "claim_eligible": False,
        "status": "REJECT",
    }
    payload = MODULE.canonical_result_bytes(result)

    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    assert json.loads(payload) == result
