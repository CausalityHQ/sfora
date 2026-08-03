#!/usr/bin/env python3
"""Measure class-exposure coupling to raw Proxy Anchor radii on official SOP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON only after a complete same-filesystem write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def unit_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero-norm row")
    return array / norms


def correlation(function: object, left: np.ndarray, right: np.ndarray) -> float:
    result = function(left, right)  # type: ignore[operator]
    return float(result.statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint must be explicitly final_training_state")
    state = checkpoint["state_dict"]
    proxies = np.asarray(state["metric_proxies"].numpy(), dtype=np.float64)
    proxy_labels = np.asarray(state["metric_proxy_labels"].numpy(), dtype=np.int64)
    if len(np.unique(proxy_labels)) != 11_318 or proxies.shape != (11_318, 512):
        raise ValueError(f"unexpected official SOP proxy shape: {proxies.shape}")

    archive = np.load(args.embeddings, allow_pickle=False)
    if str(archive["artifact_selection"]) != "final_training_state":
        raise ValueError("embedding artifact must be final_training_state")
    embeddings = unit_rows(np.asarray(archive["embeddings"], dtype=np.float64))
    labels = np.asarray(archive["labels"], dtype=np.int64)
    if embeddings.shape != (59_551, 512) or len(np.unique(labels)) != 11_318:
        raise ValueError(f"unexpected official SOP training embeddings: {embeddings.shape}")
    if set(labels.tolist()) != set(proxy_labels.tolist()):
        raise ValueError("proxy and training-embedding product sets differ")

    proxy_by_label = {int(label): index for index, label in enumerate(proxy_labels)}
    counts: list[float] = []
    norms: list[float] = []
    proxy_centroid_cosines: list[float] = []
    sample_centroid_cosines: list[float] = []
    normalized_proxies = unit_rows(proxies)
    for label in sorted(proxy_by_label):
        members = embeddings[labels == label]
        centroid = unit_rows(members.mean(axis=0, keepdims=True))[0]
        proxy_index = proxy_by_label[label]
        counts.append(float(len(members)))
        norms.append(float(np.linalg.norm(proxies[proxy_index])))
        proxy_centroid_cosines.append(float(normalized_proxies[proxy_index] @ centroid))
        sample_centroid_cosines.append(float(np.mean(members @ centroid)))

    count_array = np.asarray(counts)
    norm_array = np.asarray(norms)
    proxy_alignment = np.asarray(proxy_centroid_cosines)
    sample_cohesion = np.asarray(sample_centroid_cosines)
    quartile = len(norm_array) // 4
    order = np.argsort(norm_array)
    low = order[:quartile]
    high = order[-quartile:]
    result = {
        "classes": len(norm_array),
        "proxy_norm_mean": float(norm_array.mean()),
        "proxy_norm_sd": float(norm_array.std(ddof=1)),
        "proxy_norm_min": float(norm_array.min()),
        "proxy_norm_max": float(norm_array.max()),
        "pearson_class_count_proxy_norm": correlation(pearsonr, count_array, norm_array),
        "spearman_class_count_proxy_norm": correlation(spearmanr, count_array, norm_array),
        "spearman_class_count_inverse_proxy_norm": correlation(
            spearmanr, count_array, 1.0 / norm_array
        ),
        "spearman_proxy_norm_proxy_centroid_cosine": correlation(
            spearmanr, norm_array, proxy_alignment
        ),
        "spearman_proxy_norm_sample_centroid_cosine": correlation(
            spearmanr, norm_array, sample_cohesion
        ),
        "high_minus_low_norm_quartile_proxy_centroid_cosine": float(
            proxy_alignment[high].mean() - proxy_alignment[low].mean()
        ),
        "high_minus_low_norm_quartile_sample_centroid_cosine": float(
            sample_cohesion[high].mean() - sample_cohesion[low].mean()
        ),
        "class_count_min": int(count_array.min()),
        "class_count_max": int(count_array.max()),
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
