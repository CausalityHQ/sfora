#!/usr/bin/env python3
"""Check whether pretrained-feature hard products yield a useful TRAIN readout."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from torch.nn import functional as F

from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHT_SHA = "d5e22c6a331acbdf3b143b9836597593c2317f9488b99b72f61a4c54baf18eb8"
FEATURE_RECEIPT_SHA = "da3a2cdfb3d4fc90bfb562c15c9eb0e3b1747b6547883501b596f00ef8ec6e1b"
RECEIPT_SHAS = {
    "control": "c240755f35566038104a8f671c3fab86d1b12658bec55fe666d0df84e707160b",
    "freeze": "14049b181066b35a0bec216e577932f8af9b792cf187a7a2fdddf5e4c5eee1c9",
    "lr3x": "16a46ba17cf1778fdc3cdf628ca3854d8c15c7aaa09521b30c8473de0a0f8029",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    for arm in RECEIPT_SHAS:
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = {}
    for arm, digest in RECEIPT_SHAS.items():
        path = getattr(args, arm)
        if sha256(path) != digest:
            raise ValueError(f"In-Shop {arm} hard-readout receipt differs")
        receipts[arm] = json.loads(path.read_text())
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.features_dir / "receipt.json") != FEATURE_RECEIPT_SHA
    ):
        raise ValueError("In-Shop hard-readout authority differs")
    feature_receipt = json.loads((args.features_dir / "receipt.json").read_text())
    features = np.load(args.features_dir / "train_features.npy", mmap_mode="r")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = np.asarray([train[index].label for index in held])
    if (
        features.shape != (25_882, 1024)
        or features.dtype != np.float32
        or sha256(args.features_dir / "train_features.npy")
        != feature_receipt.get("features_sha256")
        or json.loads(args.preflight.read_text()).get("held_sha256") != digest_rows(held)
        or any(
            receipt.get("held_rows_sha256") != digest_rows(held)
            or len(receipt.get("quality", {}).get("per_query_r1", ())) != len(held)
            for receipt in receipts.values()
        )
    ):
        raise ValueError("In-Shop hard-readout split or feature geometry differs")
    names, inverse = np.unique(labels, return_inverse=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = F.normalize(torch.from_numpy(np.asarray(features[list(held)]).copy()).cuda(), dim=1)
    ids = torch.from_numpy(inverse).cuda()
    per_image_max = []
    with torch.inference_mode():
        for start in range(0, len(held), 256):
            stop = min(start + 256, len(held))
            scores = values[start:stop] @ values.T
            scores[ids[start:stop, None] == ids[None, :]] = -torch.inf
            per_image_max.extend(scores.max(dim=1).values.cpu().tolist())
    hardness = np.full(len(names), -np.inf, dtype=np.float64)
    np.maximum.at(hardness, inverse, per_image_max)
    if not np.isfinite(hardness).all():
        raise ValueError("In-Shop hard-product score differs")
    selected = sorted(range(len(names)), key=lambda index: (-hardness[index], names[index]))[
        : math.ceil(len(names) / 4)
    ]
    hard = np.isin(inverse, selected)
    if hard.sum() < 1:
        raise ValueError("In-Shop hard-product inventory differs")
    hits = {
        arm: np.asarray(receipt["quality"]["per_query_r1"], dtype=np.float64)
        for arm, receipt in receipts.items()
    }
    control = hits["control"]
    full_miss = float(1 - control.mean())
    hard_miss = float(1 - control[hard].mean())
    hard_misses = int((1 - control[hard]).sum())
    rows = {}
    for arm in RECEIPT_SHAS:
        arm_hits = hits[arm]
        receipt = receipts[arm]
        rows[arm] = {
            "full_packed_r1": float(arm_hits.mean()),
            "hard_packed_r1": float(arm_hits[hard].mean()),
            "full_map_at_r": receipt["quality"]["map_at_r"],
            "hard_map_at_r": float(
                np.asarray(receipt["quality"]["per_query_ap"], dtype=np.float64)[hard].mean()
            ),
            "hard_minus_control_product_bootstrap": (
                None
                if arm == "control"
                else product_bootstrap((arm_hits - control)[hard], labels[hard])
            ),
            "receipt_sha256": RECEIPT_SHAS[arm],
        }
    report = {
        "schema": "sfora-inshop-hard-product-readout-train-only-v1",
        "claim_eligible": False,
        "held_products": len(names),
        "hard_products": len(selected),
        "hard_queries": int(hard.sum()),
        "hard_product_names": names[selected].tolist(),
        "control_full_miss_rate": full_miss,
        "control_hard_miss_rate": hard_miss,
        "control_hard_misses": hard_misses,
        "miss_rate_ratio": hard_miss / full_miss,
        "useful_selection_readout": hard_misses >= 100 and hard_miss >= 3 * full_miss,
        "arms": rows,
        "feature_receipt_sha256": FEATURE_RECEIPT_SHA,
        "features_sha256": feature_receipt["features_sha256"],
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
                    "hard_products",
                    "hard_queries",
                    "control_hard_misses",
                    "miss_rate_ratio",
                    "useful_selection_readout",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
