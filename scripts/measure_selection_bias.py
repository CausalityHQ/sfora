#!/usr/bin/env python
"""Diagnose the local peak gap in best-over-training curves.

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

`peak_gap` is a descriptive diagnostic, not an identified selection-bias correction.
It mixes selection noise with local curvature, endpoint slope, autocorrelation, and
nonstationarity.  In particular, it is nonzero for noiseless curved or increasing
histories.  It must therefore never be presented as a corrected benchmark score.

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
    """Return (reported max, local-neighbour estimate, argmax index).

    Their difference is a local peak gap, not an unbiased selection-bias estimate.
    """
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


def _artifact_identity(path: Path, payload: dict[str, Any]) -> tuple[str, str, str, int] | None:
    match = _ARM.search(path.name)
    if match is not None:
        return match["dataset"], match["arm"], match["digest"], int(match["seed"])
    config = payload.get("config")
    methods = payload.get("methods")
    if not isinstance(config, dict) or not isinstance(methods, dict) or len(methods) != 1:
        return None
    dataset = config.get("dataset_name")
    digest = config.get("recipe_digest")
    seed = config.get("seed")
    method_key = next(iter(methods))
    recipe_base_method = config.get("recipe_base_method")
    recipe_delta = config.get("recipe_delta")
    arm = (
        recipe_base_method
        if isinstance(recipe_base_method, str) and recipe_delta in ({}, None)
        else method_key
    )
    if (
        not isinstance(dataset, str)
        or not isinstance(arm, str)
        or not isinstance(digest, str)
        or len(digest) < 12
        or not isinstance(seed, int)
    ):
        return None
    return dataset, arm, digest[:12], seed


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
        payload = json.loads(Path(path).read_text())
        identity = _artifact_identity(Path(path), payload)
        if identity is None:
            continue
        history = _recall_history(payload)
        if history is None or len(history) < 5:
            continue
        reported_value, estimated, _ = selection_overshoot(history)
        dataset, arm, digest, seed = identity
        key = (dataset, arm, digest)
        runs.setdefault(key, {})[seed] = (reported_value, estimated)

    if not runs:
        print(f"no artifacts with a usable recall history matched {pattern!r}")
        return 1

    seen_arms: dict[tuple[str, str], list[str]] = {}
    for dataset, arm, digest in runs:
        seen_arms.setdefault((dataset, arm), []).append(digest)
    ambiguous = {k for k, v in seen_arms.items() if len(v) > 1}

    header = (
        f"{'dataset/arm':32} {'digest':>13} {'n':>2} "
        f"{'reported':>9} {'local trend':>11} {'peak gap':>10}"
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
        ("pa_dual_ema", "proxy_anchor"),
        ("pa_dual_ema_bnfix", "proxy_anchor"),
        ("pa_distill_fast", "proxy_anchor"),
        ("pa_distill_avg", "proxy_anchor"),
        ("ipsr", "proxy_anchor"),
        ("tird", "proxy_anchor"),
        ("pa_fiedler", "pa_ipc4"),
        ("herd", "hist"),
        ("narrow128_distill", "narrow128"),
        ("narrow64_distill", "narrow64"),
    ]
    print("\nPaired deltas: raw versus local-neighbour diagnostic (not bias-corrected):")
    print(f"{'comparison':34} {'n':>2} {'reported':>10} {'local':>10} {'gap diff':>10}")
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
