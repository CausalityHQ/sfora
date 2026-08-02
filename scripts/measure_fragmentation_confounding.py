#!/usr/bin/env python3
"""Measure the preregistered adjusted In-Shop fragmentation association."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def _quintile_bins(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    edges = np.unique(np.quantile(values, [0.2, 0.4, 0.6, 0.8]))
    return np.searchsorted(edges, values, side="right"), [float(x) for x in edges]


def measure(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    normalized = embeddings.astype(np.float64, copy=True)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)

    predicted = np.empty(labels.shape[0], dtype=labels.dtype)
    for low in range(0, labels.shape[0], 512):
        high = min(low + 512, labels.shape[0])
        similarities = normalized[low:high] @ normalized.T
        similarities[np.arange(high - low), np.arange(low, high)] = -np.inf
        predicted[low:high] = labels[np.argmax(similarities, axis=1)]
    correct = predicted == labels

    unique_labels = np.unique(labels)
    centroids = np.stack([normalized[labels == label].mean(axis=0) for label in unique_labels])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    centroid_similarities = centroids @ centroids.T
    np.fill_diagonal(centroid_similarities, -np.inf)
    nearest_foreign = centroid_similarities.max(axis=1)
    foreign_by_label = {label: nearest_foreign[i] for i, label in enumerate(unique_labels)}

    rows: list[tuple[int, bool, float, float, float]] = []
    for label in unique_labels:
        mask = labels == label
        members = normalized[mask]
        count = members.shape[0]
        if count < 3:
            continue
        similarities = members @ members.T
        upper = np.triu_indices(count, 1)
        rows.append(
            (
                count,
                _one_nn_is_disconnected(similarities),
                float(np.mean(correct[mask])),
                float(np.mean(similarities[upper])),
                float(foreign_by_label[label]),
            )
        )

    within_values = np.asarray([row[3] for row in rows])
    foreign_values = np.asarray([row[4] for row in rows])
    within_bins, within_edges = _quintile_bins(within_values)
    foreign_bins, foreign_edges = _quintile_bins(foreign_values)

    cells: dict[tuple[int, int, int], dict[bool, list[float]]] = defaultdict(
        lambda: {False: [], True: []}
    )
    for row, within_bin, foreign_bin in zip(rows, within_bins, foreign_bins, strict=True):
        size, fragmented, outcome, _, _ = row
        cells[(size, int(within_bin), int(foreign_bin))][fragmented].append(outcome)

    differences: list[float] = []
    weights: list[int] = []
    retained_fragmented = 0
    retained_connected = 0
    for groups in cells.values():
        fragmented = groups[True]
        connected = groups[False]
        if not fragmented or not connected:
            continue
        differences.append(float(np.mean(fragmented) - np.mean(connected)))
        weights.append(min(len(fragmented), len(connected)))
        retained_fragmented += len(fragmented)
        retained_connected += len(connected)

    exposure = np.asarray([float(row[1]) for row in rows])
    outcome = np.asarray([row[2] for row in rows])
    correlation_matrix = np.corrcoef(
        np.stack([exposure, within_values, foreign_values, outcome], axis=0)
    )
    retained = retained_fragmented + retained_connected
    adjusted = float(100.0 * np.average(differences, weights=weights)) if weights else None
    return {
        "eligible_classes": len(rows),
        "fragmented_classes": int(exposure.sum()),
        "connected_classes": int(len(rows) - exposure.sum()),
        "within_similarity_quintile_edges": within_edges,
        "foreign_centroid_quintile_edges": foreign_edges,
        "matched_cells": len(weights),
        "effective_matched_weight": int(sum(weights)),
        "retained_classes": retained,
        "retained_fraction": float(retained / len(rows)),
        "retained_fragmented_classes": retained_fragmented,
        "retained_connected_classes": retained_connected,
        "adjusted_fragmented_minus_connected_top1_points": adjusted,
        "correlation_order": [
            "fragmented",
            "within_class_mean_cosine",
            "nearest_foreign_centroid_cosine",
            "class_leave_one_out_top1",
        ],
        "pearson_correlation_matrix": correlation_matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = np.load(args.pack)
    result = measure(payload["embeddings"], payload["labels"])
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
