#!/usr/bin/env python3
"""Measure fragmentation association for cross-series-only training retrieval."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from measure_fragmentation_acquisition_alignment import _parse
from measure_fragmentation_confounding import (
    _one_nn_is_disconnected,
    _quintile_bins,
)
from measure_fragmentation_series_confounding import _adjust


def _cross_series_correct(
    embeddings: np.ndarray, labels: np.ndarray, series: np.ndarray
) -> np.ndarray:
    predicted = np.empty(len(labels), dtype=labels.dtype)
    for low in range(0, len(labels), 256):
        high = min(low + 256, len(labels))
        similarities = embeddings[low:high] @ embeddings.T
        for row, query in enumerate(range(low, high)):
            forbidden = (labels == labels[query]) & (series == series[query])
            similarities[row, forbidden] = -np.inf
        predicted[low:high] = labels[np.argmax(similarities, axis=1)]
    return predicted == labels


def measure(pack: dict[str, np.ndarray]) -> dict[str, Any]:
    embeddings = np.asarray(pack["embeddings"], dtype=np.float64)
    labels = np.asarray(pack["labels"])
    example_ids = np.asarray(pack["example_ids"])
    if len(np.unique(example_ids)) != len(example_ids):
        raise ValueError("duplicate example_ids")
    series = np.asarray([_parse(value)[0] for value in example_ids])
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    correct = _cross_series_correct(embeddings, labels, series)

    unique_labels = np.unique(labels)
    centroids = np.stack([embeddings[labels == label].mean(axis=0) for label in unique_labels])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    centroid_similarities = centroids @ centroids.T
    np.fill_diagonal(centroid_similarities, -np.inf)
    nearest_foreign = centroid_similarities.max(axis=1)

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(unique_labels):
        mask = labels == label
        members = embeddings[mask]
        if len(members) < 3:
            continue
        series_sizes = tuple(sorted(Counter(series[mask]).values()))
        if len(series_sizes) < 2:
            continue
        similarities = members @ members.T
        upper = np.triu_indices(len(members), 1)
        rows.append(
            {
                "series_signature": series_sizes,
                "fragmented": _one_nn_is_disconnected(similarities),
                "outcome": float(np.mean(correct[mask])),
                "within": float(np.mean(similarities[upper])),
                "foreign": float(nearest_foreign[index]),
            }
        )

    within_bins, within_edges = _quintile_bins(np.asarray([row["within"] for row in rows]))
    foreign_bins, foreign_edges = _quintile_bins(np.asarray([row["foreign"] for row in rows]))
    fragmented = [bool(row["fragmented"]) for row in rows]
    outcomes = [float(row["outcome"]) for row in rows]
    keys = [
        (row["series_signature"], int(within_bin), int(foreign_bin))
        for row, within_bin, foreign_bin in zip(rows, within_bins, foreign_bins, strict=True)
    ]
    adjusted = _adjust(keys, fragmented, outcomes)
    adjusted["retained_fraction"] = float(adjusted["retained_classes"] / len(rows))
    fragmented_outcomes = [
        value for value, exposure in zip(outcomes, fragmented, strict=True) if exposure
    ]
    connected_outcomes = [
        value for value, exposure in zip(outcomes, fragmented, strict=True) if not exposure
    ]
    return {
        "eligible_multiseries_classes": len(rows),
        "fragmented_classes": int(sum(fragmented)),
        "connected_classes": int(len(rows) - sum(fragmented)),
        "fragmented_class_balanced_cross_series_top1": float(np.mean(fragmented_outcomes)),
        "connected_class_balanced_cross_series_top1": float(np.mean(connected_outcomes)),
        "unadjusted_fragmented_minus_connected_cross_series_top1_points": float(
            100.0 * (np.mean(fragmented_outcomes) - np.mean(connected_outcomes))
        ),
        "within_similarity_quintile_edges": within_edges,
        "foreign_centroid_quintile_edges": foreign_edges,
        "exact_series_size_signature": adjusted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(dict(np.load(args.pack)))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
