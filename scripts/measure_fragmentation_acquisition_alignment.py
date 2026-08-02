#!/usr/bin/env python3
"""Relate stable In-Shop graph components to filename acquisition/view tags."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from measure_fragmentation_partition_stability import (
    _adjusted_rand,
    _components,
    _validate_packs,
)

_FILENAME = re.compile(r"^(?P<series>[^_]+)_(?P<pose>[0-9]+)_(?P<view>[^.]+)\.jpg$")


def _parse(example_id: Any) -> tuple[str, str, str]:
    match = _FILENAME.fullmatch(Path(str(example_id)).name)
    if match is None:
        raise ValueError(f"unparseable In-Shop example_id: {example_id!r}")
    return match.group("series"), match.group("pose"), match.group("view")


def _summary(values: list[float], sizes: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(sizes, dtype=np.float64)
    return {
        "eligible_classes": len(values),
        "macro_mean_ari": float(array.mean()),
        "class_size_weighted_mean_ari": float(np.average(array, weights=weights)),
        "class_macro_standard_error": float(array.std(ddof=1) / np.sqrt(len(array))),
    }


def measure(packs: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    _validate_packs(packs)
    labels = np.asarray(packs[0]["labels"])
    parsed = [_parse(value) for value in packs[0]["example_ids"]]
    series = np.asarray([row[0] for row in parsed])
    poses = np.asarray([row[1] for row in parsed])
    views = np.asarray([row[2] for row in parsed])
    if any(len(set(views[poses == pose])) != 1 for pose in np.unique(poses)):
        raise ValueError("numeric pose token is not redundant with view descriptor")

    embeddings = [np.asarray(pack["embeddings"], dtype=np.float64) for pack in packs]
    for matrix in embeddings:
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    stable_rows: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for label in np.unique(labels):
        mask = labels == label
        if int(mask.sum()) < 3:
            continue
        components = [_components(matrix[mask] @ matrix[mask].T, 1) for matrix in embeddings]
        if all(len(np.unique(component)) > 1 for component in components):
            stable_rows.append((mask, components))

    by_seed: list[dict[str, dict[str, float | int]]] = []
    paired_differences: list[list[float]] = [[], [], []]
    for seed in range(3):
        seed_result: dict[str, dict[str, float | int]] = {}
        for name, tags in (("series", series), ("view", views)):
            values: list[float] = []
            sizes: list[int] = []
            for mask, components in stable_rows:
                class_tags = tags[mask]
                if len(np.unique(class_tags)) < 2:
                    continue
                values.append(_adjusted_rand(components[seed], class_tags))
                sizes.append(int(mask.sum()))
            seed_result[name] = _summary(values, sizes)
        for mask, components in stable_rows:
            class_series = series[mask]
            class_views = views[mask]
            if len(np.unique(class_series)) < 2 or len(np.unique(class_views)) < 2:
                continue
            paired_differences[seed].append(
                _adjusted_rand(components[seed], class_series)
                - _adjusted_rand(components[seed], class_views)
            )
        by_seed.append(seed_result)

    paired_means = [float(np.mean(values)) for values in paired_differences]
    return {
        "parsed_examples": len(parsed),
        "pose_to_view_is_one_to_one": True,
        "stable_k1_fragmented_classes": len(stable_rows),
        "by_seed": by_seed,
        "paired_series_minus_view_macro_ari_by_seed": paired_means,
        "mean_paired_series_minus_view_macro_ari": float(np.mean(paired_means)),
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
