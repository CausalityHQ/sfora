#!/usr/bin/env python3
"""Compare cached pretrained and faithfully reloaded trained source on In-Shop TRAIN."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from diagnose_inshop_siglip2_trained_width import float_top1_hits
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch.nn import functional as F

from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHT_SHA = "d5e22c6a331acbdf3b143b9836597593c2317f9488b99b72f61a4c54baf18eb8"
FEATURE_RECEIPT_SHA = "da3a2cdfb3d4fc90bfb562c15c9eb0e3b1747b6547883501b596f00ef8ec6e1b"
TRAINED_WIDTH_SHA = "9f98ddf5bcced51b20e6f31aa752f61642d4bc7c4352872b15af8566502e65b1"
TRAINED_HELPER_SHA = "e0a30d81a975c2fd363a2ee03df86a0c16d467fd1f7d188f7099b274047ec354"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--trained-width", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helper_file = sys.modules[float_top1_hits.__module__].__file__
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.features_dir / "receipt.json") != FEATURE_RECEIPT_SHA
        or sha256(args.trained_width) != TRAINED_WIDTH_SHA
        or helper_file is None
        or sha256(Path(helper_file)) != TRAINED_HELPER_SHA
    ):
        raise ValueError("In-Shop pretrained-source authority differs")
    feature_receipt = json.loads((args.features_dir / "receipt.json").read_text())
    trained = json.loads(args.trained_width.read_text())
    preflight = json.loads(args.preflight.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    held_labels = np.asarray([train[index].label for index in held])
    trained_hits = np.asarray(trained["quality"]["source_per_query_r1"], dtype=np.float64)
    features = np.load(args.features_dir / "train_features.npy", mmap_mode="r")
    if (
        feature_receipt.get("schema") != "sfora-inshop-siglip2-train-feature-export-v1"
        or feature_receipt.get("partition_sha256") != PARTITION_SHA
        or feature_receipt.get("model_file_sha256") != trained.get("model_file_sha256")
        or sha256(args.features_dir / "train_features.npy")
        != feature_receipt.get("features_sha256")
        or features.shape != (25_882, 1024)
        or features.dtype != np.float32
        or preflight.get("held_sha256") != digest_rows(held)
        or trained.get("preflight_sha256") != PREFLIGHT_SHA
        or trained.get("packed_receipt_parity", {}).get("exact") is not True
        or trained_hits.shape != (len(held),)
    ):
        raise ValueError("In-Shop pretrained-source feature geometry differs")
    labels = {name: index for index, name in enumerate(sorted(set(held_labels)))}
    label_ids = torch.tensor([labels[name] for name in held_labels], device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = F.normalize(torch.from_numpy(np.asarray(features[list(held)]).copy()).cuda(), dim=1)
    pretrained_hits = float_top1_hits(values, label_ids)
    delta = product_bootstrap(pretrained_hits - trained_hits, held_labels)
    report = {
        "schema": "sfora-inshop-siglip2-pretrained-source-train-only-v1",
        "claim_eligible": False,
        "split": "official TRAIN product-disjoint held-only symmetric gallery",
        "held_queries_and_gallery": len(held),
        "pretrained_source_float_r1": float(pretrained_hits.mean()),
        "trained_source_float_r1": float(trained_hits.mean()),
        "pretrained_minus_trained_product_bootstrap": delta,
        "pretrained_per_query_r1": pretrained_hits.tolist(),
        "feature_receipt_sha256": FEATURE_RECEIPT_SHA,
        "features_sha256": feature_receipt["features_sha256"],
        "trained_width_receipt_sha256": TRAINED_WIDTH_SHA,
        "trained_helper_sha256": TRAINED_HELPER_SHA,
        "preflight_sha256": PREFLIGHT_SHA,
        "source_sha256": sha256(Path(__file__)),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "pretrained_source_float_r1",
                    "trained_source_float_r1",
                    "pretrained_minus_trained_product_bootstrap",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
