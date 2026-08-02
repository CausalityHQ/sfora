#!/usr/bin/env python3
"""Measure preregistered class-level fragmentation stability across seed packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _disconnected(similarities: np.ndarray) -> bool:
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


def _per_class(pack: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.asarray(pack["embeddings"], dtype=np.float64)
    labels = np.asarray(pack["labels"])
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    predicted = np.empty(labels.shape[0], dtype=labels.dtype)
    for low in range(0, len(labels), 512):
        high = min(low + 512, len(labels))
        similarities = embeddings[low:high] @ embeddings.T
        similarities[np.arange(high - low), np.arange(low, high)] = -np.inf
        predicted[low:high] = labels[np.argmax(similarities, axis=1)]
    correct = predicted == labels

    class_labels: list[Any] = []
    fragmented: list[bool] = []
    outcome: list[float] = []
    for label in np.unique(labels):
        mask = labels == label
        if int(mask.sum()) < 3:
            continue
        class_labels.append(label)
        fragmented.append(_disconnected(embeddings[mask] @ embeddings[mask].T))
        outcome.append(float(np.mean(correct[mask])))
    return np.asarray(class_labels), np.asarray(fragmented), np.asarray(outcome)


def _kappa(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    agreement = float(np.mean(left == right))
    p_left = float(np.mean(left))
    p_right = float(np.mean(right))
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    value = (agreement - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    return agreement, float(value)


def measure(packs: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    if len(packs) != 3:
        raise ValueError("exactly three seed packs are required")
    rows = [_per_class(pack) for pack in packs]
    reference_labels = rows[0][0]
    for seed, (labels, _, _) in enumerate(rows[1:], start=1):
        if not np.array_equal(labels, reference_labels):
            raise ValueError(f"eligible identity labels differ for seed {seed}")

    indicators = np.stack([row[1] for row in rows], axis=0)
    outcomes = np.stack([row[2] for row in rows], axis=0)
    pairwise: dict[str, dict[str, float]] = {}
    kappas: list[float] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        agreement, kappa = _kappa(indicators[left], indicators[right])
        pairwise[f"seed{left}_seed{right}"] = {"agreement": agreement, "kappa": kappa}
        kappas.append(kappa)

    counts = indicators.sum(axis=0)
    count_fractions = {str(i): float(np.mean(counts == i)) for i in range(4)}
    stable_fragmented = counts == 3
    stable_connected = counts == 0
    mean_outcome = outcomes.mean(axis=0)
    stable_gap = None
    if stable_fragmented.any() and stable_connected.any():
        stable_gap = float(
            100.0 * (mean_outcome[stable_fragmented].mean() - mean_outcome[stable_connected].mean())
        )
    return {
        "eligible_classes": int(len(reference_labels)),
        "fragmented_fraction_by_seed": [float(x) for x in indicators.mean(axis=1)],
        "pairwise": pairwise,
        "mean_pairwise_kappa": float(np.mean(kappas)),
        "minimum_pairwise_kappa": float(np.min(kappas)),
        "fragmented_seed_count_fractions": count_fractions,
        "stable_fragmented_classes": int(stable_fragmented.sum()),
        "stable_connected_classes": int(stable_connected.sum()),
        "stable_fragmented_minus_connected_top1_points": stable_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    loaded = [dict(np.load(path)) for path in args.packs]
    result = measure(loaded)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
