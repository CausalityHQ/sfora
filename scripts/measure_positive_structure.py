#!/usr/bin/env python3
"""Measure whether within-class positive relations survive independent runs.

The input packs must contain `embeddings`, `labels`, and `example_ids` for the
same training examples.  The script aligns by ID, normalizes embeddings, then
reports cross-run rank correlation and top-k neighbour overlap within each
class.  It never reads test examples or test scores.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr  # type: ignore[import-untyped]


def load_aligned(paths: list[Path]) -> tuple[list[np.ndarray], np.ndarray]:
    """Load packs and align every run to the first pack's sorted example IDs."""
    if len(paths) < 2:
        raise ValueError("at least two independent training-embedding packs are required")
    embeddings: list[np.ndarray] = []
    reference_ids: np.ndarray | None = None
    reference_labels: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as pack:
            missing = {"embeddings", "labels", "example_ids"} - set(pack.files)
            if missing:
                raise ValueError(f"{path}: missing {sorted(missing)}")
            order = np.argsort(pack["example_ids"])
            ids = np.asarray(pack["example_ids"])[order]
            labels = np.asarray(pack["labels"])[order]
            matrix = np.asarray(pack["embeddings"], dtype=np.float64)[order]
        if reference_ids is None:
            reference_ids, reference_labels = ids, labels
        else:
            assert reference_labels is not None
            if np.array_equal(ids, reference_ids) and np.array_equal(labels, reference_labels):
                pass
            else:
                raise ValueError(f"{path}: example IDs or labels do not match the first pack")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        embeddings.append(matrix / np.clip(norms, 1e-12, None))
    assert reference_labels is not None
    return embeddings, reference_labels


def measure(
    embeddings: list[np.ndarray], labels: np.ndarray, ks: tuple[int, ...]
) -> dict[str, object]:
    """Return class-local cross-run rank and neighbourhood-stability metrics."""
    rank_correlations: list[float] = []
    jaccards: dict[int, list[float]] = {k: [] for k in ks}
    chances: dict[int, list[float]] = {k: [] for k in ks}
    all_run_nearest_agreement: list[bool] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        count = len(indices)
        if count <= max(ks):
            continue
        similarities = [matrix[indices] @ matrix[indices].T for matrix in embeddings]
        triangle = np.triu_indices(count, 1)
        for left, right in itertools.combinations(range(len(embeddings)), 2):
            statistic = spearmanr(
                similarities[left][triangle], similarities[right][triangle]
            ).statistic
            rank_correlations.append(float(statistic))
        for query in range(count):
            rankings: list[np.ndarray] = []
            for similarity in similarities:
                ranking = np.argsort(-similarity[query])
                rankings.append(ranking[ranking != query])
            all_run_nearest_agreement.append(len({int(ranking[0]) for ranking in rankings}) == 1)
            for k in ks:
                for left, right in itertools.combinations(range(len(embeddings)), 2):
                    left_set = set(map(int, rankings[left][:k]))
                    right_set = set(map(int, rankings[right][:k]))
                    jaccards[k].append(len(left_set & right_set) / len(left_set | right_set))
                    chances[k].append(k / (2 * (count - 1) - k))
    return {
        "rank_mean": float(np.mean(rank_correlations)),
        "rank_median": float(np.median(rank_correlations)),
        "rank_p10": float(np.quantile(rank_correlations, 0.1)),
        "rank_p90": float(np.quantile(rank_correlations, 0.9)),
        "nearest_all_runs": float(np.mean(all_run_nearest_agreement)),
        "jaccard": {k: float(np.mean(values)) for k, values in jaccards.items()},
        "chance": {k: float(np.mean(chances[k])) for k in ks},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", nargs="+", type=Path)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    args = parser.parse_args()
    ks = tuple(sorted(set(args.k)))
    embeddings, labels = load_aligned(args.packs)
    result = measure(embeddings, labels, ks)
    print(f"runs={len(embeddings)} images={len(labels)} classes={len(np.unique(labels))}")
    print(
        "within-class pair-rank Spearman: "
        f"mean={result['rank_mean']:.4f} median={result['rank_median']:.4f} "
        f"p10={result['rank_p10']:.4f} p90={result['rank_p90']:.4f}"
    )
    jaccard = result["jaccard"]
    chance = result["chance"]
    assert isinstance(jaccard, dict) and isinstance(chance, dict)
    for k in ks:
        print(
            f"top-{k} neighbour Jaccard={jaccard[k]:.4f} "
            f"chance={chance[k]:.4f} ratio={jaccard[k] / chance[k]:.2f}x"
        )
    print(f"same nearest positive in every run={result['nearest_all_runs']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
