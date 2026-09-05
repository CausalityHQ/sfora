"""Fixed language-pilot controls and gates, independent of models and data IO."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal


def fixed_language_permutation() -> tuple[int, ...]:
    """Return the preregistered single cycle, never an outcome-selected shuffle."""
    order = sorted(
        range(49),
        key=lambda i: (
            hashlib.sha256(f"sfora-language-permutation-v1:17:{i}".encode()).digest(),
            i,
        ),
    )
    mapping = [0] * 49
    for source, destination in zip(order, order[1:] + order[:1], strict=True):
        mapping[source] = destination
    return tuple(mapping)


def language_pilot_decision(cells: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Require the literal hit margin and MAP floor over both measured controls."""
    if set(cells) != {"base", "correct", "permuted"}:
        raise ValueError("language pilot requires exactly three measured arms")
    hits: dict[str, int] = {}
    maps: dict[str, Decimal] = {}
    for arm, cell in cells.items():
        queries, correct, score = cell.get("queries"), cell.get("correct"), cell.get("map_at_r")
        if type(queries) is not int or queries != 2746:
            raise ValueError("language pilot query cardinality differs")
        if type(correct) is not int or not 0 <= correct <= 2746:
            raise ValueError("language pilot hit count invalid")
        if type(score) is not float or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("language pilot MAP invalid")
        hits[arm] = correct
        maps[arm] = Decimal(str(score))
    required_hits = max(hits["base"], hits["permuted"], 2596) + 14
    required_map = max(maps["base"], maps["permuted"], Decimal("0.7913744556922272"))
    gates = {"recall": hits["correct"] >= required_hits, "map": maps["correct"] >= required_map}
    passed = all(gates.values())
    return {
        "claim_eligible": False,
        "surface": "exploratory-reuse-49..81",
        "passed": passed,
        "disposition": "exploratory-advancement" if passed else "fixed-quality-failure",
        "required_hits": required_hits,
        "required_map_at_r": float(required_map),
        "gates": gates,
    }


def pilot_training_projection(spent_seconds: float, update_seconds: Sequence[float]) -> float:
    """Account for all60 updates,25% margin and2100s save/evaluation reserves."""
    if (
        type(spent_seconds) not in (int, float)
        or not math.isfinite(spent_seconds)
        or spent_seconds < 0
        or len(update_seconds) != 6
        or any(
            type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in update_seconds
        )
    ):
        raise ValueError("language pilot requires finite spent time and six positive timings")
    projected = spent_seconds + 60 * max(update_seconds) * 1.25 + 300 + 1800
    if not math.isfinite(projected):
        raise ValueError("language pilot projected duration overflow")
    return projected
