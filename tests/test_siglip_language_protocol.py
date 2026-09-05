"""Literal gates reject false language gains and unaccounted pilot costs."""

from __future__ import annotations

from typing import Any

import pytest

from sfora.siglip_language_protocol import (
    fixed_language_permutation,
    language_pilot_decision,
    pilot_training_projection,
)


def _cells() -> dict[str, dict[str, Any]]:
    return {
        "base": {"queries": 2746, "correct": 2596, "map_at_r": 0.792},
        "permuted": {"queries": 2746, "correct": 2597, "map_at_r": 0.793},
        "correct": {"queries": 2746, "correct": 2611, "map_at_r": 0.793},
    }


def test_derangement_reproduces_frozen_correspondence() -> None:
    permutation = fixed_language_permutation()
    assert permutation == (
        15,
        6,
        24,
        23,
        43,
        13,
        40,
        39,
        21,
        42,
        33,
        14,
        7,
        11,
        16,
        1,
        19,
        20,
        29,
        8,
        32,
        17,
        27,
        45,
        46,
        12,
        48,
        41,
        0,
        30,
        38,
        4,
        25,
        31,
        2,
        3,
        28,
        18,
        36,
        34,
        26,
        47,
        10,
        5,
        35,
        9,
        22,
        37,
        44,
    )
    assert sorted(permutation) == list(range(49))
    assert all(i != p for i, p in enumerate(permutation))


@pytest.mark.parametrize(
    ("hits", "score", "passed"),
    [(2611, 0.793, True), (2610, 0.793, False), (2611, 0.7929999, False)],
)
def test_correct_language_must_beat_both_controls(
    hits: int,
    score: float,
    passed: bool,
) -> None:
    cells = _cells()
    cells["correct"].update(correct=hits, map_at_r=score)
    result = language_pilot_decision(cells)
    assert result["passed"] is passed
    assert result["claim_eligible"] is False
    assert result["required_hits"] == 2611
    assert result["required_map_at_r"] == 0.793


def test_teacher_floor_applies_when_both_controls_deteriorate() -> None:
    cells = _cells()
    for name in ("base", "permuted"):
        cells[name].update(correct=2500, map_at_r=0.7)
    cells["correct"].update(correct=2610, map_at_r=0.7913744556922272)
    assert language_pilot_decision(cells)["passed"] is True
    cells["correct"]["correct"] = 2609
    assert language_pilot_decision(cells)["passed"] is False
    cells["correct"].update(correct=2610, map_at_r=0.7913744556922271)
    assert language_pilot_decision(cells)["passed"] is False


@pytest.mark.parametrize("arm", ["base", "correct", "permuted"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("queries", True),
        ("queries", 2745),
        ("correct", True),
        ("correct", -1),
        ("correct", 2747),
        ("map_at_r", True),
        ("map_at_r", float("nan")),
        ("map_at_r", float("inf")),
        ("map_at_r", -0.01),
        ("map_at_r", 1.01),
    ],
)
def test_malformed_measurements_are_invalid_not_quality_failures(
    arm: str,
    key: str,
    value: object,
) -> None:
    cells = _cells()
    cells[arm][key] = value
    with pytest.raises(ValueError):
        language_pilot_decision(cells)


@pytest.mark.parametrize("mutation", ["missing-arm", "extra-arm", "missing-field"])
def test_missing_or_extra_evidence_cannot_select_a_winner(mutation: str) -> None:
    cells = _cells()
    if mutation == "missing-arm":
        del cells["permuted"]
    elif mutation == "extra-arm":
        cells["extra"] = dict(cells["base"])
    else:
        del cells["correct"]["map_at_r"]
    with pytest.raises(ValueError):
        language_pilot_decision(cells)


def test_projection_includes_slowest_step_all_arms_and_reserves() -> None:
    assert pilot_training_projection(50.0, [30.0] * 6) == 4400.0
    assert pilot_training_projection(50.0, [10.0] * 5 + [40.0]) == 5150.0
    assert pilot_training_projection(0.0, [68.0] * 6) == 7200.0


@pytest.mark.parametrize(
    ("spent", "samples"),
    [
        (-1.0, [30.0] * 6),
        (True, [30.0] * 6),
        (float("nan"), [30.0] * 6),
        (float("inf"), [30.0] * 6),
        (0.0, [30.0] * 5),
        (0.0, [30.0] * 7),
        (0.0, [30.0] * 5 + [0.0]),
        (0.0, [30.0] * 5 + [-1.0]),
        (0.0, [30.0] * 5 + [float("nan")]),
        (0.0, [30.0] * 5 + [float("inf")]),
        (0.0, [30.0] * 5 + [True]),
        (0.0, [1e308] * 6),
    ],
)
def test_invalid_timing_cannot_authorize_a_campaign(
    spent: float,
    samples: list[float],
) -> None:
    with pytest.raises(ValueError):
        pilot_training_projection(spent, samples)
