#!/usr/bin/env python3
"""Evaluate raw cached embeddings on distractor-hardened identity folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from sfora.foundation_adapter import (
    hardened_retrieval_folds,
    retrieval_map_at_r,
    retrieval_recall_at_1,
)


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))
    labels = np.asarray([int(value) for value in metadata["labels"]], dtype=np.int64)
    if embeddings.shape[0] != labels.shape[0] or not np.isfinite(embeddings).all():
        raise ValueError("cache embeddings and labels differ or are nonfinite")
    return embeddings, labels, str(metadata["key"]["arm"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--projection", choices=("raw", "truncate512", "pca512"), default="raw")
    args = parser.parse_args()
    embeddings, labels, arm = load_cache(args.cache)
    rows = []
    for index, fold in enumerate(hardened_retrieval_folds(labels, seed=args.seed)):
        if args.projection == "truncate512":
            projected = embeddings[:, :512]
        elif args.projection == "pca512":
            pca = PCA(
                n_components=512,
                whiten=False,
                svd_solver="randomized",
                random_state=args.seed,
            )
            pca.fit(embeddings[fold.optimization])
            projected = pca.transform(embeddings).astype(np.float32, copy=False)
        else:
            projected = embeddings
        values = torch.from_numpy(projected)
        gallery_index = np.concatenate((fold.gallery, fold.distractor))
        query = values[fold.query]
        gallery = values[gallery_index]
        query_labels = labels[fold.query]
        gallery_labels = labels[gallery_index]
        rows.append(
            {
                "fold": index,
                "query_rows": len(fold.query),
                "gallery_rows": len(gallery_index),
                "distractor_rows": len(fold.distractor),
                "recall_at_1": retrieval_recall_at_1(
                    query, query_labels, gallery, gallery_labels
                ),
                "map_at_r": retrieval_map_at_r(
                    query, query_labels, gallery, gallery_labels
                ),
            }
        )
    output = {
        "arm": arm,
        "seed": args.seed,
        "projection": args.projection,
        "folds": rows,
        "mean_recall_at_1": float(np.mean([row["recall_at_1"] for row in rows])),
        "mean_map_at_r": float(np.mean([row["map_at_r"] for row in rows])),
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
