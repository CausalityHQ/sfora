#!/usr/bin/env python3
"""Fit one train-only stitch and evaluate it on cached official source features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sfora.foundation_adapter import (
    fit_ridge_stitch,
    hardened_retrieval_folds,
    retrieval_map_at_r,
    retrieval_recall_at_1,
)


def load_cache(path: Path) -> tuple[torch.Tensor, tuple[str, ...], np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))
    return (
        torch.from_numpy(embeddings),
        tuple(str(value) for value in metadata["ids"]),
        np.asarray([int(value) for value in metadata["labels"]], dtype=np.int64),
        str(metadata["key"]["arm"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--target-train", type=Path, required=True)
    parser.add_argument("--source-query", type=Path, required=True)
    parser.add_argument("--source-gallery", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    source, source_ids, train_labels, source_arm = load_cache(args.source_train)
    target, target_ids, target_labels, target_arm = load_cache(args.target_train)
    query, _, query_labels, query_arm = load_cache(args.source_query)
    gallery, _, gallery_labels, gallery_arm = load_cache(args.source_gallery)
    if source_ids != target_ids or not np.array_equal(train_labels, target_labels):
        raise ValueError("source and target training rows differ")
    if source_arm != query_arm or source_arm != gallery_arm:
        raise ValueError("source train/query/gallery arms differ")
    device = torch.device(args.device)
    source = source.to(device)
    target = target.to(device)
    optimization = hardened_retrieval_folds(train_labels, seed=0)[0].optimization
    model = fit_ridge_stitch(
        source,
        target,
        torch.from_numpy(optimization).to(device),
        regularization=args.regularization,
    )
    with torch.no_grad():
        mapped_query = model.transform(query.to(device))
        mapped_gallery = model.transform(gallery.to(device))
    output = {
        "source_arm": source_arm,
        "target_arm": target_arm,
        "regularization": args.regularization,
        "optimization_rows": len(optimization),
        "recall_at_1": retrieval_recall_at_1(
            mapped_query, query_labels, mapped_gallery, gallery_labels
        ),
        "map_at_r": retrieval_map_at_r(
            mapped_query, query_labels, mapped_gallery, gallery_labels
        ),
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
