#!/usr/bin/env python3
"""Diagnose whether the frozen DINOv2 In-Shop failure starts before PCA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch.nn import functional as F

from sfora.unicom_inshop import parse_inshop_partition

FEATURE_SHA = "63c327a182293d28cdbee1375bf803dbef2e5c6f350299aec4f4536d5be8cf47"
RECEIPT_SHA = "64343ad8744ec2784c37c5700fc7daeed37107e37af2e4ba96eff4b8d61ca3ac"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.run_dir / "receipt.json"
    feature_path = args.run_dir / "train_features.npy"
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(receipt_path) != RECEIPT_SHA
        or sha256(feature_path) != FEATURE_SHA
    ):
        raise ValueError("DINOv2 source diagnostic authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    receipt = json.loads(receipt_path.read_text())
    if digest_rows(held) != receipt["held_rows_sha256"]:
        raise ValueError("DINOv2 source diagnostic held split differs")
    features = np.load(feature_path, mmap_mode="r")
    if features.shape != (25_882, 1024) or features.dtype != np.float32:
        raise ValueError("DINOv2 source diagnostic feature geometry differs")
    codes = F.normalize(torch.from_numpy(np.asarray(features[list(held)]).copy()), dim=1).cuda()
    labels = tuple(train[index].label for index in held)
    hits = []
    for start in range(0, len(held), 64):
        block = codes[start : start + 64]
        scores = block @ codes.T
        scores[
            torch.arange(len(block), device="cuda"), start + torch.arange(len(block), device="cuda")
        ] = -torch.inf
        top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, 0].cpu().tolist()
        hits.extend(labels[start + offset] == labels[index] for offset, index in enumerate(top))
    report = {
        "schema": "sfora-inshop-dinov2-source-raw-diagnostic-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "feature_sha256": FEATURE_SHA,
        "source_receipt_sha256": RECEIPT_SHA,
        "raw_1024_float_recall_at_1": sum(hits) / len(hits),
        "raw_1024_float_hits": sum(hits),
        "packed_128_recall_at_1": receipt["quality"]["recall_at_1"],
        "queries_and_gallery": len(held),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
