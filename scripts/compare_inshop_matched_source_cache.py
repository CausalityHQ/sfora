#!/usr/bin/env python3
"""Compare pinned pretrained and SOP source caches on TRAIN held roles."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from probe_inshop_sop_product_prior import HELD_SHA, score
from score_inshop_crop_view_pair import (
    GALLERY_SHA,
    PARTITION_SHA,
    QUERY_SHA,
    bootstrap_lower,
    roles,
    sha256,
)

from sfora.unicom_inshop import parse_inshop_partition

BASE = Path("/home/riomus/runs")
DATA = Path("/home/riomus/datasets/In-shop Clothes Retrieval Benchmark")
CACHES = {
    "pretrained": BASE / "sfora-inshop-siglip2-train-features-v1",
    "sop": BASE / "sfora-inshop-sop-warmstart-cache-v1",
}
SOP_SHA = "2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172"
OUTPUT = BASE / "sfora-inshop-sop-warmstart-smoke-v1/matched_source_quality.json"


def main() -> None:
    started = time.perf_counter()
    if OUTPUT.exists() or sha256(DATA / "Eval/list_eval_partition.txt") != PARTITION_SHA:
        raise ValueError("In-Shop TRAIN authority differs")
    train = tuple(row for row in parse_inshop_partition(DATA) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if len(train) != 25_882 or len(held) != 12_599 or digest_rows(held) != HELD_SHA:
        raise ValueError("In-Shop product split differs")
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, tuple(train[row].image_path for row in held), DATA)
    if (
        sha256_rows(query) != QUERY_SHA
        or sha256_rows(gallery) != GALLERY_SHA
        or len(query) != 6_354
        or len(gallery) != 6_245
    ):
        raise ValueError("In-Shop held roles differ")
    torch.backends.cuda.matmul.allow_tf32 = False
    quality = {}
    receipts = {}
    for arm, path in CACHES.items():
        receipt = json.loads((path / "receipt.json").read_text())
        if (
            receipt["schema"] != "sfora-inshop-siglip2-train-feature-export-v1"
            or receipt["model_file_sha256"] != MODEL_HASHES
            or receipt["partition_sha256"] != PARTITION_SHA
            or receipt["rows"] != len(train)
            or receipt["batch_size"] != 32
            or receipt.get("vision_init_sha256") != (SOP_SHA if arm == "sop" else None)
            or receipt["features_sha256"] != sha256(path / "train_features.npy")
        ):
            raise ValueError(f"{arm} source cache authority differs")
        features = np.load(path / "train_features.npy", mmap_mode="r", allow_pickle=False)
        if features.shape != (len(train), 1024) or features.dtype != np.float32:
            raise ValueError(f"{arm} source cache geometry differs")
        quality[arm] = score(
            torch.from_numpy(np.asarray(features[list(held)]).copy()), labels, query, gallery
        )
        receipts[arm] = sha256(path / "receipt.json")
    delta = np.asarray(quality["sop"]["per_query_r1"]) - np.asarray(
        quality["pretrained"]["per_query_r1"]
    )
    products = np.asarray([labels[row] for row in query])
    lower = bootstrap_lower(delta, products)
    upper = -bootstrap_lower(-delta, products)
    gain = float(delta.mean())
    report = {
        "schema": "sfora-inshop-matched-source-cache-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN held products; 6354 query / 6245 gallery",
        "source_sha256": sha256(Path(__file__)),
        "cache_receipt_sha256": receipts,
        "quality": quality,
        "r1_gain_percentage_points": 100 * gain,
        "r1_paired_product_bootstrap_95_pp": [100 * lower, 100 * upper],
        "gate_passed": gain >= 0.01
        and lower > 0
        and quality["sop"]["map_at_r"] >= quality["pretrained"]["map_at_r"],
        "score_wall_seconds": time.perf_counter() - started,
    }
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "r1_gain_percentage_points",
                    "r1_paired_product_bootstrap_95_pp",
                    "gate_passed",
                    "score_wall_seconds",
                )
            }
        ),
        flush=True,
    )


def sha256_rows(rows: list[int]) -> str:
    return hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest()


if __name__ == "__main__":
    main()
