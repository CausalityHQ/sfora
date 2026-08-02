#!/usr/bin/env python3
"""Measure preregistered component-partition stability across three seed packs."""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np


def _components(similarities: np.ndarray, k: int) -> np.ndarray:
    count = len(similarities)
    if not 0 < k < count:
        raise ValueError("k must be positive and smaller than the class")
    masked = similarities.copy()
    np.fill_diagonal(masked, -np.inf)
    neighbours = np.argpartition(masked, -k, axis=1)[:, -k:]
    adjacency = np.zeros((count, count), dtype=bool)
    adjacency[np.arange(count)[:, None], neighbours] = True
    adjacency |= adjacency.T
    labels = np.full(count, -1, dtype=np.int64)
    component = 0
    for start in range(count):
        if labels[start] >= 0:
            continue
        labels[start] = component
        pending = [start]
        while pending:
            source = pending.pop()
            for target in np.flatnonzero(adjacency[source]):
                if labels[target] < 0:
                    labels[target] = component
                    pending.append(int(target))
        component += 1
    return labels


def _adjusted_rand(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
        raise ValueError("partitions must have equal length")
    _, left_inverse = np.unique(left, return_inverse=True)
    _, right_inverse = np.unique(right, return_inverse=True)
    table = np.zeros((left_inverse.max() + 1, right_inverse.max() + 1), dtype=np.int64)
    np.add.at(table, (left_inverse, right_inverse), 1)
    pairs = comb(len(left), 2)
    if pairs == 0:
        return 1.0
    joint = sum(comb(int(value), 2) for value in table.ravel())
    left_pairs = sum(comb(int(value), 2) for value in table.sum(axis=1))
    right_pairs = sum(comb(int(value), 2) for value in table.sum(axis=0))
    expected = left_pairs * right_pairs / pairs
    maximum = 0.5 * (left_pairs + right_pairs)
    denominator = maximum - expected
    return float((joint - expected) / denominator) if denominator else 1.0


def _pack_partitions(pack: dict[str, np.ndarray]) -> dict[Any, dict[int, np.ndarray]]:
    embeddings = np.asarray(pack["embeddings"], dtype=np.float64)
    labels = np.asarray(pack["labels"])
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    result: dict[Any, dict[int, np.ndarray]] = {}
    for label in np.unique(labels):
        vectors = embeddings[labels == label]
        if len(vectors) < 3:
            continue
        similarities = vectors @ vectors.T
        row = {1: _components(similarities, 1)}
        if len(vectors) >= 6:
            row[2] = _components(similarities, 2)
        result[label.item() if hasattr(label, "item") else label] = row
    return result


def _validate_packs(packs: list[dict[str, np.ndarray]]) -> None:
    if len(packs) != 3:
        raise ValueError("exactly three packs are required")
    reference_ids = np.asarray(packs[0]["example_ids"])
    reference_labels = np.asarray(packs[0]["labels"])
    for seed, pack in enumerate(packs):
        ids = np.asarray(pack["example_ids"])
        labels = np.asarray(pack["labels"])
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"seed {seed} contains duplicate example_ids")
        if not np.array_equal(ids, reference_ids):
            raise ValueError(f"example_ids differ for seed {seed}")
        if not np.array_equal(labels, reference_labels):
            raise ValueError(f"sample labels differ for seed {seed}")


def measure(packs: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    _validate_packs(packs)
    partitions = [_pack_partitions(pack) for pack in packs]
    labels = sorted(partitions[0])
    stable = [
        label
        for label in labels
        if all(
            len(partitions[seed][label][1]) > 0 and len(np.unique(partitions[seed][label][1])) > 1
            for seed in range(3)
        )
    ]
    pairwise: dict[str, dict[str, float | int]] = {}
    macro_values: list[float] = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        pair_labels = [
            label
            for label in labels
            if len(np.unique(partitions[left][label][1])) > 1
            and len(np.unique(partitions[right][label][1])) > 1
        ]
        stable_aris = np.asarray(
            [
                _adjusted_rand(partitions[left][label][1], partitions[right][label][1])
                for label in stable
            ]
        )
        sizes = np.asarray([len(partitions[left][label][1]) for label in stable])
        broad_aris = np.asarray(
            [
                _adjusted_rand(partitions[left][label][1], partitions[right][label][1])
                for label in pair_labels
            ]
        )
        macro = float(stable_aris.mean())
        macro_values.append(macro)
        pairwise[f"seed{left}_seed{right}"] = {
            "stable_cohort_macro_ari": macro,
            "stable_cohort_size_weighted_ari": float(np.average(stable_aris, weights=sizes)),
            "pairwise_disconnected_cohort_classes": len(pair_labels),
            "pairwise_disconnected_cohort_macro_ari": float(broad_aris.mean()),
        }
    component_counts = np.asarray(
        [[len(np.unique(partitions[seed][label][1])) for seed in range(3)] for label in stable]
    )
    eligible_k2 = [label for label in stable if len(partitions[0][label][1]) >= 6]
    k2_disconnected = np.asarray(
        [
            [len(np.unique(partitions[seed][label][2])) > 1 for label in eligible_k2]
            for seed in range(3)
        ],
        dtype=bool,
    )
    return {
        "eligible_classes": len(labels),
        "stable_k1_fragmented_classes": len(stable),
        "pairwise": pairwise,
        "mean_stable_cohort_macro_ari": float(np.mean(macro_values)),
        "minimum_stable_cohort_macro_ari": float(np.min(macro_values)),
        "exact_three_seed_component_count_agreement": float(
            np.mean(np.all(component_counts == component_counts[:, :1], axis=1))
        ),
        "k2_eligible_stable_k1_classes": len(eligible_k2),
        "k2_disconnected_fraction_by_seed": [float(x) for x in k2_disconnected.mean(axis=1)],
        "k2_disconnected_all_three_fraction": float(np.mean(np.all(k2_disconnected, axis=0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure([dict(np.load(path)) for path in args.packs])
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
