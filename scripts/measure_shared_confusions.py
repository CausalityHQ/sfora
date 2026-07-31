#!/usr/bin/env python3
"""Measure whether local same-class neighbours share negative-class confusions.

Each input is a training-embedding pack.  For every run, the script constructs
class centroids, ranks each image against all *other* class centroids, and asks
whether its closest same-class images have more similar negative-class profiles
than its remaining same-class images.  Test examples and retrieval scores are
never read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from measure_positive_structure import load_aligned
from scipy.stats import rankdata  # type: ignore[import-untyped]


def row_correlation(matrix: np.ndarray) -> np.ndarray:
    """Return row-wise Pearson correlations for an already finite matrix."""
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    normalized = centered / np.clip(
        np.linalg.norm(centered, axis=1, keepdims=True), 1e-12, None
    )
    return normalized @ normalized.T


def measure_run(embeddings: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float]:
    """Return top-k versus other-positive profile correlation and pair AUC."""
    classes = np.unique(labels)
    centroids = np.stack([embeddings[labels == label].mean(axis=0) for label in classes])
    centroids /= np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    class_to_column = {label: column for column, label in enumerate(classes)}
    profile_ranks = np.empty((len(labels), len(classes) - 1), dtype=np.float64)
    for row, (embedding, label) in enumerate(zip(embeddings, labels, strict=True)):
        scores = embedding @ centroids.T
        scores = np.delete(scores, class_to_column[label])
        profile_ranks[row] = rankdata(scores)

    top_values: list[float] = []
    other_values: list[float] = []
    pair_embedding_similarity: list[float] = []
    pair_profile_similarity: list[float] = []
    for label in classes:
        indices = np.flatnonzero(labels == label)
        similarities = embeddings[indices] @ embeddings[indices].T
        profile_correlations = row_correlation(profile_ranks[indices])
        for query in range(len(indices)):
            order = np.argsort(-similarities[query])
            order = order[order != query]
            top = order[:k]
            other = order[k:]
            top_values.extend(profile_correlations[query, top])
            other_values.extend(profile_correlations[query, other])
        triangle = np.triu_indices(len(indices), 1)
        pair_embedding_similarity.extend(similarities[triangle])
        pair_profile_similarity.extend(profile_correlations[triangle])

    embedding_values = np.asarray(pair_embedding_similarity)
    profile_values = np.asarray(pair_profile_similarity)
    pair_corr = float(np.corrcoef(embedding_values, profile_values)[0, 1])
    return float(np.mean(top_values) - np.mean(other_values)), pair_corr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packs", nargs="+", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    embeddings, labels = load_aligned(args.packs)
    effects: list[float] = []
    correlations: list[float] = []
    for path, matrix in zip(args.packs, embeddings, strict=True):
        effect, correlation = measure_run(matrix, labels, args.k)
        effects.append(effect)
        correlations.append(correlation)
        print(
            f"{path.name}: top-{args.k} profile-correlation advantage={effect:.4f} "
            f"embedding/profile pair-correlation={correlation:.4f}"
        )
    print(
        f"mean: top-{args.k} advantage={np.mean(effects):.4f} "
        f"pair-correlation={np.mean(correlations):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
