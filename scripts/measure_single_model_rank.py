#!/usr/bin/env python3
"""Measure train-fit PCA retrieval rank for one embedding model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RANKS = (8, 16, 32, 64, 128, 256, 384, 512)


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(values.dtype).eps)


def recall_at_one(embeddings: np.ndarray, labels: np.ndarray, chunk: int = 512) -> float:
    normalized = _normalize(np.asarray(embeddings, dtype=np.float32))
    labels = np.asarray(labels)
    correct = 0
    for start in range(0, normalized.shape[0], chunk):
        stop = min(start + chunk, normalized.shape[0])
        similarities = normalized[start:stop] @ normalized.T
        rows = np.arange(stop - start)
        similarities[rows, np.arange(start, stop)] = -np.inf
        neighbours = np.argmax(similarities, axis=1)
        correct += int(np.count_nonzero(labels[neighbours] == labels[start:stop]))
    return correct / normalized.shape[0]


def measure(train_path: Path, test_path: Path) -> dict[str, object]:
    with np.load(train_path, allow_pickle=False) as train_pack:
        train = np.asarray(train_pack["embeddings"], dtype=np.float64)
    with np.load(test_path, allow_pickle=False) as test_pack:
        test = np.asarray(test_pack["embeddings"], dtype=np.float64)
        labels = np.asarray(test_pack["labels"])

    mean = train.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(train - mean, full_matrices=False)
    variance = singular_values**2
    cumulative = np.cumsum(variance) / variance.sum()

    results: dict[str, object] = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "unprojected_recall_at_one": recall_at_one(test, labels),
        "ranks": {},
    }
    centered_test = test - mean
    rank_results = results["ranks"]
    assert isinstance(rank_results, dict)
    for rank in RANKS:
        if rank > right.shape[0]:
            continue
        projected = centered_test @ right[:rank].T
        rank_results[str(rank)] = {
            "recall_at_one": recall_at_one(projected, labels),
            "cumulative_train_variance": float(cumulative[rank - 1]),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_pack", type=Path)
    parser.add_argument("test_pack", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.train_pack, args.test_pack)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
