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
    class_rows: list[tuple[int, bool, float]] = []
    predicted = np.empty(labels.shape[0], dtype=labels.dtype)
    for low in range(0, labels.shape[0], 512):
        high = min(low + 512, labels.shape[0])
        similarities = normalized[low:high] @ normalized.T
        similarities[np.arange(high - low), np.arange(low, high)] = -np.inf
        predicted[low:high] = labels[np.argmax(similarities, axis=1)]
    correct = predicted == labels
    for label in np.unique(labels):
        members = normalized[labels == label]
        count = members.shape[0]
        if count < 3:
            continue
        similarities = members @ members.T
        is_fragmented = _one_nn_is_disconnected(similarities)
        fragmented.append(is_fragmented)
        class_rows.append((count, is_fragmented, float(np.mean(correct[labels == label]))))

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

    fragmented_accuracy = [accuracy for _, flag, accuracy in class_rows if flag]
    connected_accuracy = [accuracy for _, flag, accuracy in class_rows if not flag]
    matched_deltas: list[float] = []
    matched_weights: list[int] = []
    for size in sorted({size for size, _, _ in class_rows}):
        left = [accuracy for n, flag, accuracy in class_rows if n == size and flag]
        right = [accuracy for n, flag, accuracy in class_rows if n == size and not flag]
        if left and right:
            matched_deltas.append(float(np.mean(left) - np.mean(right)))
            matched_weights.append(min(len(left), len(right)))

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
        "fragmented_class_balanced_top1": float(np.mean(fragmented_accuracy)),
        "connected_class_balanced_top1": float(np.mean(connected_accuracy)),
        "fragmented_minus_connected_top1_points": float(
            100.0 * (np.mean(fragmented_accuracy) - np.mean(connected_accuracy))
        ),
        "size_matched_fragmented_minus_connected_top1_points": float(
            100.0 * np.average(matched_deltas, weights=matched_weights)
        ),
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
