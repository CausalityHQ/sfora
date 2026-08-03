#!/usr/bin/env python3
"""Independently verify exposure support for a spectral-fragmentation result."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _fragmented_by_union_find(similarities: np.ndarray) -> bool:
    """Return whether the symmetrized directed 1-NN graph is disconnected."""
    count = similarities.shape[0]
    scores = np.asarray(similarities, dtype=np.float64).copy()
    np.fill_diagonal(scores, -np.inf)
    nearest = np.argmax(scores, axis=1)
    parent = np.arange(count)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for source, target in enumerate(nearest.tolist()):
        left = root(source)
        right = root(int(target))
        if left != right:
            parent[right] = left
    return len({root(index) for index in range(count)}) != 1


def verify(
    embeddings: np.ndarray,
    labels: np.ndarray,
    result: dict[str, Any],
    *,
    minimum_per_arm: int,
    minimum_effect_points: float,
) -> dict[str, Any]:
    vectors = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    if vectors.ndim != 2 or labels.ndim != 1 or len(vectors) != len(labels):
        raise ValueError("embeddings and labels must have matching rows")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("embeddings contain a zero-norm row")
    vectors /= norms

    arms_by_size: dict[int, set[bool]] = defaultdict(set)
    fragmented_count = 0
    eligible_count = 0
    for label in np.unique(labels):
        members = vectors[labels == label]
        if len(members) < 3:
            continue
        fragmented = _fragmented_by_union_find(members @ members.T)
        eligible_count += 1
        fragmented_count += int(fragmented)
        arms_by_size[len(members)].add(fragmented)

    connected_count = eligible_count - fragmented_count
    common_sizes = sorted(size for size, arms in arms_by_size.items() if arms == {False, True})
    if eligible_count == 0:
        raise ValueError("pack contains no classes with at least three examples")
    expected_eligible = int(result["eligible_classes"])
    expected_fragmented = int(result["one_nn_fragmented_count"])
    expected_fraction = float(result["one_nn_fragmented_fraction"])
    if eligible_count != expected_eligible:
        raise ValueError(
            f"eligible count differs from diagnostic: {eligible_count} != {expected_eligible}"
        )
    if fragmented_count != expected_fragmented:
        raise ValueError(
            "fragmented count differs from diagnostic: "
            f"{fragmented_count} != {expected_fragmented}"
        )
    observed_fraction = fragmented_count / eligible_count
    if not np.isclose(observed_fraction, expected_fraction, atol=1e-12, rtol=1e-12):
        raise ValueError("fragmented fraction differs from diagnostic")

    effect = float(result["size_matched_fragmented_minus_connected_top1_points"])
    support_pass = (
        fragmented_count >= minimum_per_arm
        and connected_count >= minimum_per_arm
        and bool(common_sizes)
    )
    return {
        "status": "verified",
        "eligible_classes": eligible_count,
        "fragmented_classes": fragmented_count,
        "connected_classes": connected_count,
        "common_exact_size_strata": common_sizes,
        "common_exact_size_strata_count": len(common_sizes),
        "minimum_classes_per_arm": minimum_per_arm,
        "support_gate_pass": support_pass,
        "size_matched_effect_points": effect,
        "minimum_effect_points": minimum_effect_points,
        "effect_gate_pass": effect >= minimum_effect_points,
        "registered_gate_pass": support_pass and effect >= minimum_effect_points,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-per-arm", type=int, default=100)
    parser.add_argument("--minimum-effect-points", type=float, default=1.0)
    args = parser.parse_args()
    if args.minimum_per_arm < 1:
        parser.error("--minimum-per-arm must be positive")
    with np.load(args.pack, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"])
        labels = np.asarray(archive["labels"])
    result = json.loads(args.result.read_text(encoding="utf-8"))
    payload = verify(
        embeddings,
        labels,
        result,
        minimum_per_arm=args.minimum_per_arm,
        minimum_effect_points=args.minimum_effect_points,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
