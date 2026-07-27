#!/usr/bin/env python3
"""Diagnose the "spiky" similarity geometry of saved embeddings.

Quantifies three things that aggregate R@1 hides:

  skew(N_k)   skewness of the k-occurrence distribution -- how often each point is
              SOMEONE ELSE's neighbour. High skew means a few "hubs" dominate
              retrieval while "antihubs" are never returned. The standard hubness
              statistic (Radovanovic et al., JMLR 2010).
  cross-cls   fraction of k-NN edges crossing a class boundary. This is the "spiky"
              rate: how often a sample's nearest neighbours are the wrong class --
              the situation proxy losses assume away by collapsing each class onto a
              prototype.
  antihub%    fraction of points appearing in NOBODY's top-k. Effectively
              unretrievable, and invisible to mean R@1.

Measured on this repo's CUB embeddings: cross-class rate 35-42%, antihubs 5-8%, and
skew correlates -0.82 with R@1 *within* dataset (n=17 CUB models, n=5 Cars).

IMPORTANT: that correlation is DIAGNOSTIC, NOT CAUSAL. `--correct` applies the two
standard label-free hubness corrections at retrieval time; both cut skew sharply
(1.7 -> 0.65 for CSLS) while *reducing* R@1 (-0.65 pt CSLS, -3.16 pt Sinkhorn, mean
over 17 CUB models). Do not build a hubness-minimising training loss on the strength
of the correlation alone -- intervening on hubness makes retrieval worse on
well-trained embeddings. It helped only the weakest model measured
(`baseline_frozen`, skew 2.95), consistent with the repeated pattern that generic
corrections help broken systems and hurt well-tuned ones.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
from typing import Any

import numpy as np


def _unit(embeddings: Any) -> Any:
    return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)


def recall_at_1(similarity: Any, labels: Any) -> float:
    scores = similarity.copy()
    np.fill_diagonal(scores, -np.inf)
    return float((labels[np.argmax(scores, axis=1)] == labels).mean())


def hubness_skew(similarity: Any, k: int = 10) -> float:
    """Skewness of the k-occurrence distribution."""
    scores = similarity.copy()
    np.fill_diagonal(scores, -np.inf)
    neighbours = np.argpartition(-scores, k, axis=1)[:, :k]
    occurrence = np.bincount(neighbours.ravel(), minlength=scores.shape[0]).astype(float)
    spread = float(occurrence.std())
    return float(((occurrence - occurrence.mean()) ** 3).mean() / (spread**3 + 1e-12))


def geometry(embeddings: Any, labels: Any, k: int = 10) -> tuple[float, float, float, float]:
    unit = _unit(embeddings)
    similarity = unit @ unit.T
    scores = similarity.copy()
    np.fill_diagonal(scores, -np.inf)
    neighbours = np.argpartition(-scores, k, axis=1)[:, :k]
    occurrence = np.bincount(neighbours.ravel(), minlength=len(unit)).astype(float)
    cross_class = float((labels[neighbours] != labels[:, None]).mean())
    antihub = float((occurrence == 0).mean())
    return hubness_skew(similarity, k), cross_class, recall_at_1(similarity, labels), antihub


def csls(similarity: Any, k: int = 10) -> Any:
    """Cross-domain similarity local scaling (Conneau et al., ICLR 2018)."""
    scores = similarity.copy()
    np.fill_diagonal(scores, -np.inf)
    local = np.sort(scores, axis=1)[:, -k:].mean(axis=1)
    return 2 * similarity - local[:, None] - local[None, :]


def sinkhorn(similarity: Any, epsilon: float = 0.05, iterations: int = 10) -> Any:
    """Doubly-stochastic normalisation: exactly uniform k-occurrence, i.e. no hubs."""
    log_kernel = similarity / epsilon
    rows = np.zeros(similarity.shape[0])
    columns = np.zeros(similarity.shape[1])
    log_uniform = -float(np.log(similarity.shape[0]))
    for _ in range(iterations):
        rows = log_uniform - np.log(np.exp(log_kernel + columns[None, :]).sum(axis=1) + 1e-30)
        columns = log_uniform - np.log(np.exp(log_kernel + rows[:, None]).sum(axis=0) + 1e-30)
    return log_kernel + rows[:, None] + columns[None, :]


def load(pattern: str, min_size: int, max_size: int) -> list[tuple[str, Any, Any]]:
    found: list[tuple[str, Any, Any]] = []
    for path in sorted(glob.glob(pattern)):
        try:
            payload = np.load(path)
        except (OSError, ValueError):
            continue
        if "embeddings" not in payload.files:
            continue
        embeddings, labels = payload["embeddings"], payload["labels"]
        if not min_size <= embeddings.shape[0] <= max_size:
            continue
        found.append((os.path.basename(path), embeddings, labels))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb", default="reports/emb/*.npz")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--min-size", type=int, default=1000)
    parser.add_argument("--max-size", type=int, default=100000)
    parser.add_argument(
        "--correct",
        action="store_true",
        help="also apply CSLS / Sinkhorn hubness corrections and report the R@1 delta",
    )
    args = parser.parse_args()

    models = load(args.emb, args.min_size, args.max_size)
    if not models:
        print(f"No embeddings matched {args.emb}")
        return 1

    print(f"{'model':32}{'skew':>8}{'cross-cls':>11}{'R@1':>9}{'antihub%':>10}")
    by_size: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
    for name, embeddings, labels in models:
        skew, cross, r1, antihub = geometry(embeddings, labels, args.k)
        by_size[int(embeddings.shape[0])].append((skew, r1))
        print(f"{name[:32]:32}{skew:8.2f}{cross:11.3f}{r1:9.4f}{100 * antihub:10.1f}")

    print("\n=== within-dataset correlation (guards against a difficulty confound) ===")
    for size, rows in sorted(by_size.items()):
        if len(rows) < 4:
            continue
        skews = np.array([row[0] for row in rows])
        recalls = np.array([row[1] for row in rows])
        label = {5924: "CUB", 8131: "Cars"}.get(size, f"n_test={size}")
        print(
            f"{label:8} models={len(rows):3}  corr(skew, R@1) = "
            f"{np.corrcoef(skews, recalls)[0, 1]:+.3f}"
        )

    if args.correct:
        print("\n=== does REDUCING hubness help? (label-free, transductive) ===")
        print(f"{'model':32}{'R@1':>9}{'+CSLS':>9}{'+Sinkhorn':>11}")
        csls_deltas: list[float] = []
        sinkhorn_deltas: list[float] = []
        for name, embeddings, labels in models:
            unit = _unit(embeddings)
            similarity = unit @ unit.T
            base = recall_at_1(similarity, labels)
            corrected = recall_at_1(csls(similarity, args.k), labels)
            balanced = recall_at_1(sinkhorn(similarity), labels)
            csls_deltas.append(100 * (corrected - base))
            sinkhorn_deltas.append(100 * (balanced - base))
            print(f"{name[:32]:32}{base:9.4f}{corrected:9.4f}{balanced:11.4f}")
        print(
            f"\nmean delta: CSLS {np.mean(csls_deltas):+.2f} pt, "
            f"Sinkhorn {np.mean(sinkhorn_deltas):+.2f} pt  (n={len(models)})"
        )
        print(
            "Both cut hubness and LOSE recall on trained embeddings: the correlation is\n"
            "diagnostic, not causal. Do not build a hubness-minimising loss on it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
