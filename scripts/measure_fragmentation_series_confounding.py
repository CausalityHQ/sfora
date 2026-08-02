#!/usr/bin/env python3
"""Adjust In-Shop fragmentation association for filename-series composition."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import numpy as np
from measure_fragmentation_acquisition_alignment import _parse
from measure_fragmentation_confounding import (
    _one_nn_is_disconnected,
    _quintile_bins,
)


def _adjust(
    keys: list[Hashable], fragmented: list[bool], outcomes: list[float]
) -> dict[str, float | int | None]:
    cells: dict[Hashable, dict[bool, list[float]]] = defaultdict(lambda: {False: [], True: []})
    for key, exposure, outcome in zip(keys, fragmented, outcomes, strict=True):
        cells[key][exposure].append(outcome)
    differences: list[float] = []
    weights: list[int] = []
    retained_fragmented = 0
    retained_connected = 0
    for groups in cells.values():
        if not groups[True] or not groups[False]:
            continue
        differences.append(float(np.mean(groups[True]) - np.mean(groups[False])))
        weights.append(min(len(groups[True]), len(groups[False])))
        retained_fragmented += len(groups[True])
        retained_connected += len(groups[False])
    retained = retained_fragmented + retained_connected
    return {
        "matched_cells": len(weights),
        "effective_matched_weight": int(sum(weights)),
        "retained_classes": retained,
        "retained_fragmented_classes": retained_fragmented,
        "retained_connected_classes": retained_connected,
        "adjusted_fragmented_minus_connected_top1_points": (
            float(100.0 * np.average(differences, weights=weights)) if weights else None
        ),
    }


def measure(pack: dict[str, np.ndarray]) -> dict[str, Any]:
    embeddings = np.asarray(pack["embeddings"], dtype=np.float64)
    labels = np.asarray(pack["labels"])
    example_ids = np.asarray(pack["example_ids"])
    if len(np.unique(example_ids)) != len(example_ids):
        raise ValueError("duplicate example_ids")
    series = np.asarray([_parse(value)[0] for value in example_ids])
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    predicted = np.empty(labels.shape[0], dtype=labels.dtype)
    for low in range(0, len(labels), 512):
        high = min(low + 512, len(labels))
        similarities = embeddings[low:high] @ embeddings.T
        similarities[np.arange(high - low), np.arange(low, high)] = -np.inf
        predicted[low:high] = labels[np.argmax(similarities, axis=1)]
    correct = predicted == labels

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
        count = len(members)
        if count < 3:
            continue
        similarities = members @ members.T
        upper = np.triu_indices(count, 1)
        series_sizes = tuple(sorted(Counter(series[mask]).values()))
        rows.append(
            {
                "size": count,
                "series_count": len(series_sizes),
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
    primary_keys = [
        (row["series_signature"], int(within_bin), int(foreign_bin))
        for row, within_bin, foreign_bin in zip(rows, within_bins, foreign_bins, strict=True)
    ]
    secondary_keys = [
        (row["size"], row["series_count"], int(within_bin), int(foreign_bin))
        for row, within_bin, foreign_bin in zip(rows, within_bins, foreign_bins, strict=True)
    ]
    primary = _adjust(primary_keys, fragmented, outcomes)
    secondary = _adjust(secondary_keys, fragmented, outcomes)
    for result in (primary, secondary):
        result["retained_fraction"] = float(result["retained_classes"] / len(rows))
    return {
        "eligible_classes": len(rows),
        "fragmented_classes": int(sum(fragmented)),
        "within_similarity_quintile_edges": within_edges,
        "foreign_centroid_quintile_edges": foreign_edges,
        "exact_series_size_signature": primary,
        "exact_size_and_series_count": secondary,
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
