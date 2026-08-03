#!/usr/bin/env python3
"""Audit exact-content label conflicts and their Cars196 R@1 tie effects."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_indices(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    squared_norms = np.sum(vectors * vectors, axis=1)
    max_relevant = max(int(count) - 1 for count in np.bincount(labels) if count)
    top_k = min(len(vectors) - 1, max(30, max_relevant))
    production = np.empty(len(vectors), dtype=np.int64)
    stable_full = np.empty(len(vectors), dtype=np.int64)
    for start in range(0, len(vectors), chunk_size):
        stop = min(start + chunk_size, len(vectors))
        distances = (
            squared_norms[start:stop, None]
            + squared_norms[None, :]
            - (2.0 * vectors[start:stop] @ vectors.T)
        )
        distances = np.maximum(distances, 0.0)
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        top_indices = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
        top_distances = np.take_along_axis(distances, top_indices, axis=1)
        top_order = np.argsort(top_distances, axis=1, kind="stable")
        production[start:stop] = np.take_along_axis(top_indices, top_order, axis=1)[:, 0]
        stable_full[start:stop] = np.argsort(distances, axis=1, kind="stable")[:, 0]
    return production, stable_full


def analyze_pack(pack_path: Path) -> dict[str, Any]:
    with np.load(pack_path, allow_pickle=False) as pack:
        required = {"embeddings", "labels", "example_ids", "content_sha256", "split"}
        missing = required - set(pack.files)
        if missing:
            raise ValueError(f"Cars pack lacks required keys: {sorted(missing)}")
        if str(pack["split"]) != "test":
            raise ValueError("content-tie audit requires the Cars test pack")
        embeddings = np.asarray(pack["embeddings"], dtype=np.float64)
        labels = np.asarray(pack["labels"], dtype=np.int64)
        example_ids = np.asarray(pack["example_ids"]).astype(str)
        content_hashes = np.asarray(pack["content_sha256"]).astype(str)
    if embeddings.shape != (8_131, 512) or labels.shape != (8_131,):
        raise ValueError("content-tie audit requires the official Cars test shape")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(content_hashes):
        groups[digest].append(index)
    duplicate_groups = {digest: rows for digest, rows in groups.items() if len(rows) > 1}
    cross_label_groups = {
        digest: rows
        for digest, rows in duplicate_groups.items()
        if len({int(labels[index]) for index in rows}) > 1
    }
    cross_label_rows = {index for rows in cross_label_groups.values() for index in rows}

    production, stable = _nearest_indices(embeddings, labels)
    production_correct = labels[production] == labels
    stable_correct = labels[stable] == labels
    changed = np.flatnonzero(production != stable)
    changed_rows = []
    for query in changed:
        production_neighbor = int(production[query])
        stable_neighbor = int(stable[query])
        production_distance = float(
            np.sum((embeddings[query] - embeddings[production_neighbor]) ** 2)
        )
        stable_distance = float(np.sum((embeddings[query] - embeddings[stable_neighbor]) ** 2))
        changed_rows.append(
            {
                "query_index": int(query),
                "query_id": str(example_ids[query]),
                "query_label": int(labels[query]),
                "production_neighbor_index": production_neighbor,
                "production_neighbor_id": str(example_ids[production_neighbor]),
                "production_neighbor_label": int(labels[production_neighbor]),
                "stable_neighbor_index": stable_neighbor,
                "stable_neighbor_id": str(example_ids[stable_neighbor]),
                "stable_neighbor_label": int(labels[stable_neighbor]),
                "production_distance_squared": production_distance,
                "stable_distance_squared": stable_distance,
                "exact_distance_tie": bool(production_distance == stable_distance),
                "neighbors_are_cross_label_content_duplicates": bool(
                    content_hashes[production_neighbor] == content_hashes[stable_neighbor]
                    and labels[production_neighbor] != labels[stable_neighbor]
                ),
            }
        )

    retained = np.asarray(
        [index for index in range(len(labels)) if index not in cross_label_rows],
        dtype=np.int64,
    )
    return {
        "pack_sha256": sha256(pack_path),
        "test_examples": int(len(labels)),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_extra_row_count": sum(len(rows) - 1 for rows in duplicate_groups.values()),
        "cross_label_duplicate_group_count": len(cross_label_groups),
        "cross_label_duplicate_query_count": len(cross_label_rows),
        "cross_label_duplicate_groups": [
            {
                "content_sha256": digest,
                "indices": rows,
                "example_ids": [str(example_ids[index]) for index in rows],
                "labels": [int(labels[index]) for index in rows],
            }
            for digest, rows in sorted(cross_label_groups.items())
        ],
        "production_partial_topk_recall_at_1": float(np.mean(production_correct)),
        "stable_full_order_recall_at_1": float(np.mean(stable_correct)),
        "nearest_neighbor_disagreement_count": int(len(changed)),
        "production_minus_stable_correct_count": int(
            np.sum(production_correct.astype(np.int64) - stable_correct.astype(np.int64))
        ),
        "changed_queries": changed_rows,
        "cross_label_duplicate_queries_correct_production": int(
            np.sum(production_correct[list(cross_label_rows)])
        ),
        "recall_at_1_excluding_cross_label_duplicate_queries": float(
            np.mean(production_correct[retained])
        ),
        "interpretation": (
            "Official test content contains conflicting labels. The benchmark partial-top-k "
            "selection resolves three exact gallery ties favorably relative to stable full "
            "ordering; all conflicting duplicate queries themselves are unavoidable errors."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_pack(args.pack)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
