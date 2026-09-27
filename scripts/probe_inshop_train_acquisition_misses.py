#!/usr/bin/env python3
"""Frozen metadata-only acquisition screen on the In-Shop TRAIN held roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from measure_fragmentation_acquisition_alignment import _parse
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import GALLERY_SHA, QUERY_SHA, roles, sha256

from sfora.unicom_inshop import parse_inshop_partition

PARTITION_SHA = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
RECEIPT_SHA = "d48e2c382fcfb7aa4807b00cc8b2a76fe098aebfa77334dbc2fd8402d36b46a4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.receipt) != RECEIPT_SHA
    ):
        raise ValueError("In-Shop acquisition screen authority differs")
    receipt = json.loads(args.receipt.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    paths = tuple(train[row].image_path for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (
        len(query) != 6354
        or len(gallery) != 6245
        or receipt.get("receipt_sha256")
        != "76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc"
        or receipt.get("checkpoint_sha256")
        != "ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089"
        or len(receipt.get("actual_proxy_per_query_r1", [])) != len(query)
    ):
        raise ValueError("In-Shop acquisition screen role or receipt differs")
    if (
        hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != QUERY_SHA
        or hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest() != GALLERY_SHA
    ):
        raise ValueError("In-Shop acquisition screen role hashes differ")
    series = tuple(_parse(path)[0] for path in paths)
    gallery_series: dict[str, set[str]] = {}
    for row in gallery:
        gallery_series.setdefault(labels[row], set()).add(series[row])
    hits = receipt["actual_proxy_per_query_r1"]
    if any(hit not in (0, 1, 0.0, 1.0) for hit in hits):
        raise ValueError("In-Shop acquisition screen hits differ")
    no_same = [series[row] not in gallery_series[labels[row]] for row in query]
    misses = [hit == 0 for hit in hits]
    no_same_queries = sum(no_same)
    no_same_misses = sum(a and b for a, b in zip(no_same, misses, strict=True))
    total_misses = sum(misses)
    if total_misses != 151 or not 0 < no_same_queries < len(query):
        raise ValueError("In-Shop acquisition screen inventory differs")
    same_queries = len(query) - no_same_queries
    same_misses = total_misses - no_same_misses
    enriched = no_same_misses * same_queries >= 2 * same_misses * no_same_queries
    report = {
        "schema": "sfora-inshop-acquisition-miss-f0a-train-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "partition_sha256": PARTITION_SHA,
        "receipt_sha256": RECEIPT_SHA,
        "query_sha256": QUERY_SHA,
        "gallery_sha256": GALLERY_SHA,
        "queries": len(query),
        "gallery_rows": len(gallery),
        "misses": total_misses,
        "no_same_group_queries": no_same_queries,
        "no_same_group_misses": no_same_misses,
        "no_same_group_miss_fraction": no_same_misses / total_misses,
        "no_same_group_r1": 1 - no_same_misses / no_same_queries,
        "same_group_r1": 1 - same_misses / same_queries,
        "miss_rate_at_least_2x": enriched,
        "advance_f0b": no_same_misses / total_misses >= 0.35 and enriched,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
