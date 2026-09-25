#!/usr/bin/env python3
"""TRAIN-only pretrained source and PCA-128 retrieval ceiling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_coverage import PARTITION_SHA256, fit_and_holdout
from train_sop_siglip2_compact import initialize_head_and_classifier, sha256

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_inshop import parse_inshop_partition

FEATURE_SHA = "f232584491bf4ed1cf75daa7fcc03f218e7db5df797f4d57dd2bf034ac110885"


@torch.inference_mode()
def recall_at_1(
    values: torch.Tensor,
    labels: torch.Tensor,
    queries: tuple[int, ...],
    gallery: tuple[int, ...],
    inverse: torch.Tensor | None = None,
    *,
    device: torch.device,
) -> float:
    x = values.to(device)
    ids = labels.to(device)
    query_rows = torch.tensor(queries, device=device)
    gallery_rows = torch.tensor(gallery, device=device)
    code = x[gallery_rows]
    norm = inverse.to(device) if inverse is not None else None
    correct = 0
    for block in query_rows.split(64):  # type: ignore[no-untyped-call]
        scores = x[block] @ code.T
        if norm is not None:
            scores = scores * norm[block, None] * norm[gallery_rows][None, :]
        scores[
            torch.arange(len(block), device=device), torch.searchsorted(gallery_rows, block)
        ] = -torch.inf
        nearest = gallery_rows[torch.argmax(scores, dim=1)]
        correct += int((ids[nearest] == ids[block]).sum())
    return correct / len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or sha256(args.features) != FEATURE_SHA
    ):
        raise ValueError("In-Shop source ceiling authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    names = tuple(row.label for row in train)
    fit, held = fit_and_holdout(names)
    source = np.load(args.features, mmap_mode="r")
    if source.shape != (25_882, 1024) or source.dtype != np.float32 or len(held) != 2_540:
        raise ValueError("In-Shop source ceiling geometry differs")
    encoded = {name: index for index, name in enumerate(sorted(set(names)))}
    labels = torch.tensor([encoded[name] for name in names], dtype=torch.int64)
    all_rows = tuple(range(len(train)))
    source_tensor = torch.from_numpy(np.asarray(source).copy())
    normalized = torch.nn.functional.normalize(source_tensor, dim=1)
    fit_names = tuple(names[i] for i in fit)
    fit_ids = {name: index for index, name in enumerate(sorted(set(fit_names)))}
    head, _, pca_sha = initialize_head_and_classifier(
        source_tensor[list(fit)], tuple(fit_ids[name] for name in fit_names), allow_singletons=True
    )
    projected = torch.nn.functional.normalize(compact_head_features(source_tensor, head), dim=1)
    packed = pack_int8_unit_embeddings(projected)
    rows = {
        "source_1024_float": (normalized, None),
        "pca_128_float": (projected, None),
        "pca_128_int8_norm": (packed.codes.float(), packed.inverse_norms),
    }
    report = {
        "schema": "sfora-inshop-siglip2-source-ceiling-train-only-v1",
        "claim_eligible": False,
        "feature_sha256": FEATURE_SHA,
        "pca_sha256": pca_sha,
        "held_queries": len(held),
        "full_gallery": len(train),
        "held_gallery": len(held),
        "source_sha256": sha256(Path(__file__)),
        "quality": {
            name: {
                "full_gallery_r1": recall_at_1(
                    value, labels, held, all_rows, inverse, device=torch.device("cuda")
                ),
                "held_gallery_r1": recall_at_1(
                    value, labels, held, held, inverse, device=torch.device("cuda")
                ),
            }
            for name, (value, inverse) in rows.items()
        },
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report["quality"]), flush=True)


if __name__ == "__main__":
    main()
