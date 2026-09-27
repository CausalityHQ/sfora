#!/usr/bin/env python3
"""Post hoc AP deltas by held-product size for two failed speed arms."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, sha256, split

from sfora.unicom_inshop import parse_inshop_partition

DEPTH_CONTROL_SHA = "3da086a15c4ec25b6255d300da13d0220d5311fd6d5babff62d8aaa872ca0e03"
DEPTH_TREATMENT_SHA = "6ed19555bd00a226af98aa1caeb261a8c1d0ef057a365ebc413cc97aef5cb160"
FREEZE_CONTROL_SHA = "fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89"
FREEZE_TREATMENT_SHA = "63e97cebb8cc0e354e87ed837f2739123e5cdb2078a38827848f5abef0c768cd"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    for name in (
        "depth-control",
        "depth-treatment",
        "freeze-control",
        "freeze-treatment",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("In-Shop positive-count diagnostic authority differs")
    paths = (
        (args.depth_control, args.depth_treatment),
        (args.freeze_control, args.freeze_treatment),
    )
    hashes = ((DEPTH_CONTROL_SHA, DEPTH_TREATMENT_SHA), (FREEZE_CONTROL_SHA, FREEZE_TREATMENT_SHA))
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = [train[row].label for row in held]
    counts = Counter(labels)
    sizes = np.asarray([counts[label] for label in labels])
    panels = {}
    for name, pair, expected in zip(("depth22_100", "freeze16_1000"), paths, hashes, strict=True):
        receipts = []
        for path, digest in zip(pair, expected, strict=True):
            if sha256(path) != digest:
                raise ValueError("In-Shop positive-count receipt differs")
            receipts.append(json.loads(path.read_text()))
        values = [np.asarray(r["quality"]["per_query_ap"], dtype=np.float64) for r in receipts]
        if any(v.shape != (len(held),) or not np.isfinite(v).all() for v in values):
            raise ValueError("In-Shop positive-count AP vector differs")
        delta = values[1] - values[0]
        groups = {}
        for low, high in ((2, 3), (4, 5), (6, 8), (9, 100)):
            mask = (sizes >= low) & (sizes <= high)
            groups[f"{low}-{high}"] = {
                "queries": int(mask.sum()),
                "mean_ap_delta": float(delta[mask].mean()),
            }
        panels[name] = {"mean_ap_delta": float(delta.mean()), "groups": groups}
    report = {
        "schema": "sfora-inshop-positive-count-posthoc-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "held_rows": len(held),
        "panels": panels,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
