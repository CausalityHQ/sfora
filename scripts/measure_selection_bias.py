#!/usr/bin/env python
"""Quantify the upward bias in best-over-training, per arm and per paired comparison.

Best-over-training reports `max` over ~50-60 test evaluations of one run. A maximum over
noisy observations of a curve overshoots the curve's true peak, and the overshoot grows
with the observation noise. So two arms with identical true peaks report different
numbers if one trains less stably -- the noisier arm wins on the protocol alone.

That is a claim about every DML paper using this protocol, including ours, so it is
worth measuring rather than asserting.

Estimator. At the selected epoch `i` the reported value `h[i]` contains that epoch's own
noise, which is exactly the term selection exploits. Estimate the underlying curve at `i`
from its NEIGHBOURS ONLY -- a leave-one-out local mean over +/-k epochs, excluding `i`.
Near a peak the curve is locally flat, so the neighbour mean is a reasonable estimate of
the trend, and it is uncontaminated by the noise that won the argmax.

    overshoot = h[argmax] - loo(h, argmax)

`overshoot` is then the per-run selection bonus. The number that matters is not its
size, which partly reflects local curvature, but whether it DIFFERS between two arms
being compared -- because that difference transfers directly into the reported delta.

Usage:
    uv run python scripts/measure_selection_bias.py 'reports/generated/*cub*.json'
"""

from __future__ import annotations

import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, cast

from sfora.data import ImageDatasetName
from sfora.image_recipes import BaseMethod, recipe_digest, resolve_recipe

_ARM = re.compile(
    r"image_end_to_end_(?P<dataset>[a-z0-9]+)\.(?P<arm>[^.]+)\..*?"
    r"(?P<digest>[0-9a-f]{12})_seed(?P<seed>\d+)\.json$"
)


def leave_one_out_local_mean(history: list[float], index: int, half_width: int = 2) -> float:
    """Estimate the underlying curve at `index` from neighbours only.

    Excluding `index` is the whole point: its own noise is what selection exploited.
    """
    low = max(0, index - half_width)
    high = min(len(history), index + half_width + 1)
    neighbours = [history[j] for j in range(low, high) if j != index]
    if not neighbours:
        raise ValueError("cannot estimate a trend from a single-point history")
    return sum(neighbours) / len(neighbours)


def selection_overshoot(history: list[float], half_width: int = 2) -> tuple[float, float, int]:
    """Return (reported max, leave-one-out estimate at the argmax, argmax index)."""
    if not history:
        raise ValueError("empty history")
    index = max(range(len(history)), key=lambda i: history[i])
    return history[index], leave_one_out_local_mean(history, index, half_width), index


def _recall_history(payload: dict[str, Any]) -> list[float] | None:
    for method in payload.get("methods", {}).values():
        history = method.get("test_recall_history")
        if history:
            return [float(x) for x in history]
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sd(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    average = _mean(values)
    return math.sqrt(sum((v - average) ** 2 for v in values) / (len(values) - 1))


def _current_digest(dataset: str, arm: str) -> str | None:
    """Resolve the current recipe digest instead of guessing from stale artifacts."""
    hist_arms = {
        "hist",
        "herd",
        "herd_bnfix",
    }
    base_method = "hist" if arm in hist_arms else "proxy_anchor"
    selector = "auto" if arm in {"proxy_anchor", "hist"} else arm
    try:
        recipe = resolve_recipe(
            selector,
            base_method=cast(BaseMethod, base_method),
            dataset=cast(ImageDatasetName, dataset),
        )
    except (KeyError, ValueError):
        return None
    return recipe_digest(recipe)[:12]


def main(pattern: str) -> int:
    # Keyed by digest as well as arm. A recipe ID is stable across recipe FIXES, so
    # superseded runs share an ID at a different digest -- In-Shop Proxy Anchor has
    # pre- and post-`22f7dd6` artifacts at 0.865 vs 0.902, and mixing them fabricates a
    # +3.4 pt effect. The first draft of this script did exactly that.
    runs: dict[tuple[str, str, str], dict[int, tuple[float, float]]] = {}
    for path in sorted(glob.glob(pattern)):
        match = _ARM.search(Path(path).name)
        if match is None:
            continue
        payload = json.loads(Path(path).read_text())
        history = _recall_history(payload)
        if history is None or len(history) < 5:
            continue
        reported_value, estimated, _ = selection_overshoot(history)
        key = (match["dataset"], match["arm"], match["digest"])
        runs.setdefault(key, {})[int(match["seed"])] = (reported_value, estimated)

    if not runs:
        print(f"no artifacts with a usable recall history matched {pattern!r}")
        return 1

    seen_arms: dict[tuple[str, str], list[str]] = {}
    for dataset, arm, digest in runs:
        seen_arms.setdefault((dataset, arm), []).append(digest)
    ambiguous = {k for k, v in seen_arms.items() if len(v) > 1}

    header = (
        f"{'dataset/arm':32} {'digest':>13} {'n':>2} "
        f"{'reported':>9} {'corrected':>10} {'overshoot':>10}"
    )
    print(header)
    for (dataset, arm, digest), seeds in sorted(runs.items()):
        reported = [v[0] for v in seeds.values()]
        corrected = [v[1] for v in seeds.values()]
        overshoot = [(a - b) * 100 for a, b in zip(reported, corrected, strict=True)]
        flag = "  <- SUPERSEDED?" if (dataset, arm) in ambiguous else ""
        print(
            f"{dataset + '/' + arm:32} {digest:>13} {len(seeds):>2} {_mean(reported):>9.4f} "
            f"{_mean(corrected):>10.4f} {_mean(overshoot):>9.3f} pt{flag}"
        )

    pairs = [
        ("pa_distill", "proxy_anchor"),
        ("pa_ema_avg", "proxy_anchor"),
        ("pa_ema_avg_fast", "proxy_anchor"),
        ("pa_ema_avg_m95", "proxy_anchor"),
        ("pa_ema_avg_m90", "proxy_anchor"),
        ("pa_ema_avg_bnfix", "proxy_anchor"),
        ("pa_distill_fast", "proxy_anchor"),
        ("pa_distill_avg", "proxy_anchor"),
        ("herd", "hist"),
        ("narrow128_distill", "narrow128"),
        ("narrow64_distill", "narrow64"),
    ]
    print("\nPaired deltas as reported vs after removing each arm's selection bonus:")
    print(f"{'comparison':34} {'n':>2} {'reported':>10} {'corrected':>10} {'inflation':>10}")
    for derived, base in pairs:
        for dataset in sorted({d for d, _, _ in runs}):
            derived_digest = _current_digest(dataset, derived)
            base_digest = _current_digest(dataset, base)
            left_candidates = [
                v
                for (d, a, digest), v in runs.items()
                if d == dataset and a == derived and digest == derived_digest
            ]
            right_candidates = [
                v
                for (d, a, digest), v in runs.items()
                if d == dataset and a == base and digest == base_digest
            ]
            if derived_digest is None or base_digest is None:
                print(
                    f"{dataset + '/' + derived + '-' + base:34} "
                    "REFUSED: current recipe digest could not be resolved"
                )
                continue
            left = left_candidates[0] if len(left_candidates) == 1 else None
            right = right_candidates[0] if len(right_candidates) == 1 else None
            if not left or not right:
                continue
            shared = sorted(set(left) & set(right))
            if not shared:
                continue
            reported_delta = [(left[s][0] - right[s][0]) * 100 for s in shared]
            corrected_delta = [(left[s][1] - right[s][1]) * 100 for s in shared]
            inflation = _mean(reported_delta) - _mean(corrected_delta)
            label = f"{dataset}/{derived}-{base}"
            print(
                f"{label:34} {len(shared):>2} {_mean(reported_delta):>+9.3f} "
                f"{_mean(corrected_delta):>+10.3f} {inflation:>+9.3f} pt"
            )
            print(f"{'':34}    sd {_sd(reported_delta):>6.3f}     sd {_sd(corrected_delta):>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "reports/generated/*.json"))
