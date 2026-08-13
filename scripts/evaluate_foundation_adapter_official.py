#!/usr/bin/env python3
"""Evaluate one fixed cached-feature adapter checkpoint on official cached rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sfora.foundation_adapter import (
    AdapterConfig,
    NestedLinearAdapter,
    nested_embeddings,
    retrieval_map_at_r,
    retrieval_recall_at_1,
)


def load_cache(path: Path) -> tuple[torch.Tensor, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))
    return (
        torch.from_numpy(embeddings),
        np.asarray([int(value) for value in metadata["labels"]], dtype=np.int64),
        str(metadata["key"]["arm"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--gallery-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved["model"] != "linear" or saved["fixed_epoch"] != 20:
        raise ValueError("official adapter must be the fixed-20-epoch linear candidate")
    config = AdapterConfig(**saved["config"])
    model = NestedLinearAdapter(config)
    model.load_state_dict(saved["state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    query, query_labels, query_arm = load_cache(args.query_cache)
    gallery, gallery_labels, gallery_arm = load_cache(args.gallery_cache)
    if query_arm != gallery_arm:
        raise ValueError("official query and gallery arms differ")
    with torch.no_grad():
        query_outputs = nested_embeddings(model(query.to(device)), config.prefixes)
        gallery_outputs = nested_embeddings(model(gallery.to(device)), config.prefixes)
    metrics = {
        str(width): {
            "recall_at_1": retrieval_recall_at_1(
                query_outputs[width], query_labels, gallery_outputs[width], gallery_labels
            ),
            "map_at_r": retrieval_map_at_r(
                query_outputs[width], query_labels, gallery_outputs[width], gallery_labels
            ),
        }
        for width in config.prefixes
    }
    print(json.dumps({"arm": query_arm, "metrics": metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
