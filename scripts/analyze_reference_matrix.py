#!/usr/bin/env python3
"""Paired analysis of the corrected reference-recipe matrix.

Reads `image_end_to_end_*.json` artifacts produced by `run_reference_matrix.sh`
and reports, per dataset, the **paired per-seed** effect of EMA relational
distillation on each base loss.

Statistical stance (important, and deliberately conservative):

With n=3 paired seeds there is no honest confidence interval. A 3-sample
bootstrap resamples from 3 points and its CI is an artifact of that fact. What
*is* defensible from 3 paired observations:

  * every paired delta and its mean (the effect size);
  * the seed spread of the base arm, so the reader can judge effect vs noise;
  * an exact paired sign/permutation test, whose two-sided floor is
    p = 2^-(n-1). For n=3 the smallest attainable p is 0.25 - i.e. **3 seeds can
    never reach p<0.05**, no matter how large the effect.

So 3 seeds are enough to *direct* the next decision and nowhere near enough to
*publish* a small effect. This script prints the required seed count to reach
p<0.05 under perfect sign consistency (n>=6, two-sided) so the gap is explicit.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal, cast

BASE_OF: dict[str, str] = {"pa_distill": "proxy_anchor", "herd": "hist"}
ARM_ORDER = ["proxy_anchor", "pa_distill", "hist", "herd"]

BaseMethodName = Literal["proxy_anchor", "hist"]


@dataclass(frozen=True)
class Run:
    """One current-digest reference-recipe run."""

    best: float
    recipe_id: str
    layer_norm: object
    path: str


def expected_digests() -> dict[tuple[str, str], str]:
    """Current registry digest for every (dataset, arm) we analyse.

    Artifacts are matched on this digest, never on `recipe_id` alone. A recipe ID
    is stable across recipe *fixes*, so several superseded runs can share one ID
    with different digests -- e.g. In-Shop Proxy Anchor has both pre- and
    post-`22f7dd6` ("use full partitions for reference recipes") artifacts, at
    R@1 0.844 vs 0.902. Keying on the ID silently mixes them.
    """
    from sfora.image_recipes import RecipeUnavailableError, recipe_digest, resolve_recipe

    wanted: dict[str, tuple[BaseMethodName, str]] = {
        "proxy_anchor": ("proxy_anchor", "auto"),
        "pa_distill": ("proxy_anchor", "pa_distill"),
        "hist": ("hist", "auto"),
        "herd": ("hist", "herd"),
    }
    digests: dict[tuple[str, str], str] = {}
    for dataset in ("cub", "cars", "sop", "inshop", "inat2018"):
        for arm, (base, selector) in wanted.items():
            try:
                recipe = resolve_recipe(selector, base_method=base, dataset=dataset)
            except (RecipeUnavailableError, ValueError, KeyError):
                continue
            digests[(dataset, arm)] = recipe_digest(recipe)
    return digests


def collect(report_dir: Path) -> dict[tuple[str, str, int], Run]:
    """Map (dataset, method, seed) -> record for current-digest reference runs."""
    digests = expected_digests()
    found: dict[tuple[str, str, int], Run] = {}
    superseded = 0
    for path in sorted(report_dir.glob("image_end_to_end_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        config = payload.get("config") or {}
        recipe_id = config.get("recipe_id")
        if not recipe_id or config.get("recipe_track") != "reference":
            continue
        dataset, seed = config.get("dataset_name"), config.get("seed")
        distill = float(config.get("ema_distill_weight") or 0.0)
        base = config.get("recipe_base_method") or (
            "hist" if "hist" in str(recipe_id) else "proxy_anchor"
        )
        if base == "hist":
            method = "herd" if distill > 0 else "hist"
        else:
            method = "pa_distill" if distill > 0 else "proxy_anchor"
        expected = digests.get((str(dataset), method))
        if expected is None or config.get("recipe_digest") != expected:
            superseded += 1
            continue
        scores = [
            float(cast(float, m.get("best_test_recall_at_1")))
            for m in (payload.get("methods") or {}).values()
            if isinstance(m.get("best_test_recall_at_1"), (int, float))
        ]
        if not scores or dataset is None or seed is None:
            continue
        key = (str(dataset), method, int(seed))
        if key in found:
            raise SystemExit(
                f"Duplicate current-digest artifact for {key}: "
                f"{found[key].path} and {path.name}. Refusing to guess."
            )
        found[key] = Run(
            best=max(scores),
            recipe_id=str(recipe_id),
            layer_norm=config.get("embedding_layer_norm"),
            path=path.name,
        )
    if superseded:
        print(f"(ignored {superseded} superseded / non-current-digest artifact(s))\n")
    return found


def exact_sign_test_p(deltas: list[float]) -> float:
    """Two-sided exact paired permutation p-value over sign flips."""
    n = len(deltas)
    if n == 0:
        return float("nan")
    observed = abs(sum(deltas) / n)
    extreme = sum(
        1
        for signs in product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, deltas, strict=True)) / n) >= observed - 1e-12
    )
    return float(extreme) / float(2**n)


def paired_t(deltas: list[float]) -> tuple[float, float]:
    """Paired t statistic and two-tailed p for the mean of `deltas` vs 0 (df = n-1).

    Reported alongside -- not instead of -- the exact sign test. The two answer
    different questions and can disagree sharply at n=3: the t-test can reach
    p<0.05 but only by assuming the paired differences are normal, which three
    points cannot evidence; the sign test assumes nothing but cannot go below
    2^-(n-1). Quote both, and treat n=3 as screening either way.
    """
    n = len(deltas)
    if n < 2:
        return float("nan"), float("nan")
    mean = sum(deltas) / n
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    sem = math.sqrt(var / n)
    if sem == 0.0:
        return float("inf"), 0.0
    t = mean / sem
    # Two-tailed p from the Student-t CDF via the regularised incomplete beta.
    df = n - 1
    x = df / (df + t * t)
    p = _betainc_half(df / 2.0, x)
    return t, min(1.0, max(0.0, p))


def _betainc_half(a: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, 1/2), used for the Student-t tail."""
    b = 0.5
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    # Continued fraction (Lentz) for the standard incomplete-beta expansion.
    tiny = 1e-30
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (tiny if abs(d) < tiny else d)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        num = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + num * d
        c = 1.0 + num / (tiny if abs(c) < tiny else c)
        d = 1.0 / (tiny if abs(d) < tiny else d)
        h *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + num * d
        c = 1.0 + num / (tiny if abs(c) < tiny else c)
        d = 1.0 / (tiny if abs(d) < tiny else d)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return math.exp(log_front) * h / a


def summarize(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports/generated"))
    parser.add_argument(
        "--gate",
        type=float,
        default=0.5,
        help="preregistered gate in R@1 percentage points (default 0.5)",
    )
    args = parser.parse_args()

    found = collect(args.reports)
    if not found:
        print(f"No reference-track artifacts under {args.reports}.")
        return 1

    by_dataset: dict[str, set[int]] = defaultdict(set)
    for dataset, _method, seed in found:
        by_dataset[dataset].add(seed)

    print("=" * 78)
    print("CORRECTED REFERENCE-RECIPE MATRIX - paired analysis")
    print("=" * 78)

    for dataset in sorted(by_dataset):
        seeds = sorted(by_dataset[dataset])
        print(f"\n## {dataset}   seeds present: {seeds}")

        header = "".join(f"{'seed ' + str(s):>10}" for s in seeds)
        print(f"\n  {'arm':<14}{header}{'mean':>10}{'sd':>9}")
        for arm in ARM_ORDER:
            row = [found.get((dataset, arm, s)) for s in seeds]
            cells = "".join(f"{rec.best:>10.4f}" if rec else f"{'-':>10}" for rec in row)
            present = [rec.best for rec in row if rec]
            if present:
                mean, sd = summarize(present)
                print(f"  {arm:<14}{cells}{mean:>10.4f}{sd:>9.4f}")
            else:
                print(f"  {arm:<14}{cells}{'-':>10}{'-':>9}")

        # Guard: the legacy confound was an unequal LayerNorm setting across arms.
        for derived, base in BASE_OF.items():
            lns = {
                found[(dataset, arm, s)].layer_norm
                for arm in (base, derived)
                for s in seeds
                if (dataset, arm, s) in found
            }
            if len(lns) > 1:
                print(
                    f"\n  !! CONFOUND: {base} vs {derived} differ in embedding_layer_norm {lns}."
                    "  Deltas below are NOT a clean distillation estimate."
                )

        print("\n  paired effect of distillation (derived - base), R@1 points:")
        for derived, base in BASE_OF.items():
            deltas = [
                100.0 * (found[(dataset, derived, s)].best - found[(dataset, base, s)].best)
                for s in seeds
                if (dataset, derived, s) in found and (dataset, base, s) in found
            ]
            if not deltas:
                print(f"    {derived:<12} vs {base:<13} - incomplete")
                continue
            mean, sd = summarize(deltas)
            per_seed = ", ".join(f"{d:+.2f}" for d in deltas)
            p_sign = exact_sign_test_p(deltas)
            t, p_t = paired_t(deltas)
            verdict = "PASS" if mean >= args.gate and all(d > 0 for d in deltas) else "FAIL"
            print(
                f"    {derived:<12} vs {base:<13} n={len(deltas)}  "
                f"per-seed [{per_seed}]  mean {mean:+.3f}  sd {sd:.3f}"
            )
            print(
                f"      paired t={t:+.2f} (df={len(deltas) - 1}) p_t={p_t:.4f}   "
                f"exact sign p={p_sign:.3f}   "
                f"gate(>={args.gate:+.2f} & all-positive): {verdict}"
            )

    print("\n" + "-" * 78)
    print("Reading the p-values: an exact paired sign test over n seeds has floor")
    print("p = 2^-(n-1) two-sided. n=3 -> 0.25, n=4 -> 0.125, n=5 -> 0.0625,")
    print("n=6 -> 0.031. Publishing a small effect therefore needs >= 6 seeds;")
    print("3 seeds can only direct the next decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
