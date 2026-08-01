#!/usr/bin/env python3
"""Measure the Gate-1 diagnostics for spectral class connectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _one_nn_is_disconnected(similarities: np.ndarray) -> bool:
    count = similarities.shape[0]
    masked = similarities.copy()
    np.fill_diagonal(masked, -np.inf)
    neighbours = np.argmax(masked, axis=1)
    adjacency = np.zeros((count, count), dtype=bool)
    adjacency[np.arange(count), neighbours] = True
    adjacency |= adjacency.T
    seen = {0}
    pending = [0]
    while pending:
        source = pending.pop()
        for target in np.flatnonzero(adjacency[source]):
            index = int(target)
            if index not in seen:
                seen.add(index)
                pending.append(index)
    return len(seen) != count


def measure(embeddings: np.ndarray, labels: np.ndarray, *, temperature: float) -> dict[str, Any]:
    normalized = embeddings.astype(np.float64, copy=True)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    fragmented: list[bool] = []
    same_as_farthest: list[bool] = []
    singleton_cut: list[bool] = []
    sizes: list[int] = []
    fiedler_values: list[float] = []
    for label in np.unique(labels):
        members = normalized[labels == label]
        count = members.shape[0]
        if count < 3:
            continue
        similarities = members @ members.T
        fragmented.append(_one_nn_is_disconnected(similarities))

        affinities = np.exp((similarities - 1.0) / temperature)
        np.fill_diagonal(affinities, 0.0)
        laplacian = np.diag(affinities.sum(axis=1)) - affinities
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        fiedler = eigenvectors[:, 1]
        upper = np.triu_indices(count, 1)
        derivatives = (fiedler[upper[0]] - fiedler[upper[1]]) ** 2
        same_as_farthest.append(int(np.argmax(derivatives)) == int(np.argmin(similarities[upper])))
        signs = fiedler >= 0.0
        singleton_cut.append(min(int(signs.sum()), int(count - signs.sum())) == 1)
        sizes.append(count)
        fiedler_values.append(float(eigenvalues[1]))

    return {
        "eligible_classes": len(sizes),
        "mean_class_size": float(np.mean(sizes)),
        "min_class_size": min(sizes),
        "max_class_size": max(sizes),
        "one_nn_fragmented_count": int(np.sum(fragmented)),
        "one_nn_fragmented_fraction": float(np.mean(fragmented)),
        "fiedler_max_derivative_equals_farthest_count": int(np.sum(same_as_farthest)),
        "fiedler_max_derivative_equals_farthest_fraction": float(np.mean(same_as_farthest)),
        "singleton_fiedler_cut_fraction": float(np.mean(singleton_cut)),
        "median_fiedler_value": float(np.median(fiedler_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    payload = np.load(args.pack)
    result = measure(payload["embeddings"], payload["labels"], temperature=args.temperature)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
