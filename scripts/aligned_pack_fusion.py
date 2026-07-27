#!/usr/bin/env python3
"""Fuse a model pack into ONE embedding by orthogonal alignment, not concatenation.

THE PROBLEM. Ensembling is the only effect in this project comfortably above the
noise floor: a 5-model concatenated pack scores ~+4.7 R@1 points over a single model.
But concatenation multiplies embedding dimension - and therefore retrieval index size
and ANN query cost - by the pack size. A 5-model pack is a 2560-dim index.

WHY THE OBVIOUS FIX FAILS. In classification, "model soups" (Wortsman et al., ICML
2022) average weights and get ensemble-like gains at 1x inference cost. That does not
transfer to metric learning: cosine retrieval is invariant to rotation, so nothing
during training pins down the basis, and two independently-trained embedding spaces
are related by an arbitrary orthogonal transform. Averaging them directly destroys
both. The `naive mean` row below measures exactly that damage.

THE FIX. That rotation is a gauge freedom, and it is removable. Orthogonal Procrustes
gives the R minimising ||A - B R||, and this repo has already established that an
UNCENTERED orthonormal map is cosine-preserving (the 2560->2048 lossless compression
result). So: align every pack member into a common basis, then average. One model's
worth of dimensions, most of the pack's accuracy.

PROTOCOL. Rotations are fitted on a CLASS-DISJOINT half of the data and retrieval is
measured only on the other half, whose classes the rotation never saw - the same
discipline as the repo's train-clean compression result. Procrustes needs only image
CORRESPONDENCE (two models' embeddings of the same image), never labels, so the fit is
label-free as well as class-disjoint.

MEASURED on CUB (mean over 5 random class splits, gain vs the BEST single member,
which is a deliberately tough baseline since it is a max over N):

    HIST x5    +2.17 pt   (range +1.52 .. +2.62)   ~77% of the concat gain at d=512
    HERD x9    +1.42 pt   (range -0.03 .. +2.30)

Concatenation still wins outright; alignment buys roughly three quarters of it at 1/N
the index size. Anchor choice barely matters (<0.3 pt).
"""

from __future__ import annotations

import argparse
import glob
from typing import Any

import numpy as np


def unit(embeddings: Any) -> Any:
    return embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)


def recall_at_1(embeddings: Any, labels: Any) -> float:
    normalised = unit(embeddings)
    similarity = normalised @ normalised.T
    np.fill_diagonal(similarity, -np.inf)
    return float((labels[np.argmax(similarity, axis=1)] == labels).mean())


def procrustes(source: Any, target: Any) -> Any:
    """Orthogonal R minimising ||target - source @ R||."""
    left, _, right = np.linalg.svd(source.T @ target)
    return left @ right


def evaluate(mats: list[Any], labels: Any, seed: int, anchor: int) -> dict[str, float]:
    classes = np.unique(labels)
    order = np.random.default_rng(seed).permutation(len(classes))
    fit_classes = set(classes[order[: len(classes) // 2]].tolist())
    fit = np.array([i for i, y in enumerate(labels) if int(y) in fit_classes])
    evaluation = np.array([i for i, y in enumerate(labels) if int(y) not in fit_classes])
    eval_labels = labels[evaluation]

    aligned = [
        mats[j][evaluation]
        if j == anchor
        else mats[j][evaluation] @ procrustes(mats[j][fit], mats[anchor][fit])
        for j in range(len(mats))
    ]
    singles = [recall_at_1(m[evaluation], eval_labels) for m in mats]
    return {
        "best_single": max(singles),
        "concat": recall_at_1(np.concatenate([m[evaluation] for m in mats], axis=1), eval_labels),
        "naive_mean": recall_at_1(np.mean([m[evaluation] for m in mats], axis=0), eval_labels),
        "aligned_mean": recall_at_1(np.mean(aligned, axis=0), eval_labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="reports/emb/hist_only_seed*.npz")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--anchor", type=int, default=0)
    args = parser.parse_args()

    files = sorted(glob.glob(args.pack))
    if len(files) < 3:
        print(f"Need at least 3 pack members, matched {len(files)} for {args.pack}")
        return 1

    mats: list[Any] = []
    labels: Any = None
    for path in files:
        payload = np.load(path)
        mats.append(unit(payload["embeddings"]).astype(np.float64))
        labels = payload["labels"]

    dimension = mats[0].shape[1]
    print(f"{len(mats)} members, d={dimension}, concat d={dimension * len(mats)}\n")
    print(f"{'split':>6}{'best single':>13}{'concat':>10}{'naive':>10}{'ALIGNED':>10}{'gain':>9}")
    gains = []
    for seed in range(args.splits):
        row = evaluate(mats, labels, seed, args.anchor)
        gain = 100 * (row["aligned_mean"] - row["best_single"])
        gains.append(gain)
        print(
            f"{seed:>6}{row['best_single']:>13.4f}{row['concat']:>10.4f}"
            f"{row['naive_mean']:>10.4f}{row['aligned_mean']:>10.4f}{gain:>+9.2f}"
        )
    print(
        f"\nmean gain over best single: {np.mean(gains):+.2f} pt "
        f"(sd {np.std(gains):.2f}), at 1/{len(mats)} the dimension"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
