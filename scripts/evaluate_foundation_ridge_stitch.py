#!/usr/bin/env python3
"""Measure whether source embeddings contain the strong comparator geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from sfora.foundation_adapter import (
    hardened_retrieval_folds,
    retrieval_map_at_r,
    retrieval_recall_at_1,
    ridge_stitch,
)


def load_cache(path: Path) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))
    ids = tuple(str(value) for value in metadata["ids"])
    labels = np.asarray([int(value) for value in metadata["labels"]], dtype=np.int64)
    return embeddings, ids, labels, str(metadata["key"]["arm"])


def metrics(
    embeddings: torch.Tensor,
    labels: np.ndarray,
    query: np.ndarray,
    gallery: np.ndarray,
) -> dict[str, float]:
    return {
        "recall_at_1": retrieval_recall_at_1(
            embeddings[query], labels[query], embeddings[gallery], labels[gallery]
        ),
        "map_at_r": retrieval_map_at_r(
            embeddings[query], labels[query], embeddings[gallery], labels[gallery]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--regularization", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    source, source_ids, source_labels, source_arm = load_cache(args.source_cache)
    target, target_ids, target_labels, target_arm = load_cache(args.target_cache)
    if source_ids != target_ids or not np.array_equal(source_labels, target_labels):
        raise ValueError("source and target cache rows differ")
    device = torch.device(args.device)
    source_tensor = torch.from_numpy(source).to(device)
    target_tensor = torch.from_numpy(target).to(device)
    rows: list[dict[str, Any]] = []
    for index, fold in enumerate(hardened_retrieval_folds(source_labels, seed=args.seed)):
        stitched = ridge_stitch(
            source_tensor,
            target_tensor,
            torch.from_numpy(fold.optimization).to(device),
            regularization=args.regularization,
        )
        gallery = np.concatenate((fold.gallery, fold.distractor))
        rows.append(
            {
                "fold": index,
                "source": metrics(source_tensor, source_labels, fold.query, gallery),
                "stitched": metrics(stitched, source_labels, fold.query, gallery),
                "target": metrics(target_tensor, target_labels, fold.query, gallery),
            }
        )
    output = {
        "source_arm": source_arm,
        "target_arm": target_arm,
        "regularization": args.regularization,
        "folds": rows,
        "means": {
            arm: {
                metric: float(
                    np.mean([cast(dict[str, float], row[arm])[metric] for row in rows])
                )
                for metric in ("recall_at_1", "map_at_r")
            }
            for arm in ("source", "stitched", "target")
        },
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
