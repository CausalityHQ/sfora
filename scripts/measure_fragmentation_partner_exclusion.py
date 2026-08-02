#!/usr/bin/env python3
"""Audit fragmentation after deleting each query's nearest same-class partner."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from measure_fragmentation_confounding import _one_nn_is_disconnected, _quintile_bins


def _partner_excluded_correct(normalized: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return top-1 correctness after excluding self and closest same-class image."""
    partner = np.full(labels.shape[0], -1, dtype=np.int64)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if indices.size < 2:
            continue
        similarities = normalized[indices] @ normalized[indices].T
        np.fill_diagonal(similarities, -np.inf)
        partner[indices] = indices[np.argmax(similarities, axis=1)]

    predicted = np.empty(labels.shape[0], dtype=labels.dtype)
    for low in range(0, labels.shape[0], 512):
        high = min(low + 512, labels.shape[0])
        similarities = normalized[low:high] @ normalized.T
        rows = np.arange(high - low)
        similarities[rows, np.arange(low, high)] = -np.inf
        valid = partner[low:high] >= 0
        similarities[rows[valid], partner[low:high][valid]] = -np.inf
        predicted[low:high] = labels[np.argmax(similarities, axis=1)]
    return predicted == labels


def measure(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    normalized = embeddings.astype(np.float64, copy=True)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    correct = _partner_excluded_correct(normalized, labels)

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

    exposure = np.asarray([bool(row[1]) for row in rows])
    outcomes = np.asarray([row[2] for row in rows])
    retained = retained_fragmented + retained_connected
    return {
        "eligible_classes": len(rows),
        "fragmented_classes": int(exposure.sum()),
        "connected_classes": int((~exposure).sum()),
        "fragmented_class_balanced_partner_excluded_top1": float(outcomes[exposure].mean()),
        "connected_class_balanced_partner_excluded_top1": float(outcomes[~exposure].mean()),
        "unadjusted_fragmented_minus_connected_partner_excluded_top1_points": float(
            100.0 * (outcomes[exposure].mean() - outcomes[~exposure].mean())
        ),
        "within_similarity_quintile_edges": within_edges,
        "foreign_centroid_quintile_edges": foreign_edges,
        "matched_cells": len(weights),
        "effective_matched_weight": int(sum(weights)),
        "retained_classes": retained,
        "retained_fraction": float(retained / len(rows)),
        "retained_fragmented_classes": retained_fragmented,
        "retained_connected_classes": retained_connected,
        "adjusted_fragmented_minus_connected_partner_excluded_top1_points": (
            float(100.0 * np.average(differences, weights=weights)) if weights else None
        ),
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
