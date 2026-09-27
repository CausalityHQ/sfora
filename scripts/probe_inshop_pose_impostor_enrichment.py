#!/usr/bin/env python3
"""Metadata-only falsifier for same-pose top impostors on fixed In-Shop TRAIN roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from sfora.unicom_inshop import parse_inshop_partition

MISS_RECEIPT_SHA = "8dfe23577547b188ae37aa0b83f564bd1dd8c411edb066f474724ccd2e89667c"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"


def ratio(matches: int, expected: float) -> float:
    if expected <= 0:
        raise ValueError("same-pose gallery expectation must be positive")
    return matches / expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "preflight", "miss-receipt", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA:
        raise ValueError("In-Shop partition authority differs")
    receipt_sha = sha256(args.miss_receipt)
    receipt = json.loads(args.miss_receipt.read_text())
    if receipt_sha != MISS_RECEIPT_SHA or receipt["miss_count"] != 151:
        raise ValueError("In-Shop miss receipt authority differs")

    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if digest_rows(held) != json.loads(args.preflight.read_text())["held_sha256"]:
        raise ValueError("In-Shop held split differs")
    paths = tuple(train[i].image_path for i in held)
    labels = tuple(train[i].label for i in held)
    pose = tuple(path.stem.rsplit("_", 1)[-1] for path in paths)
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        grouped[label].append(i)
    query: list[int] = []
    gallery: list[int] = []
    for label in sorted(grouped):
        rows = sorted(grouped[label], key=lambda i: hashlib.sha256(str(paths[i].relative_to(args.dataset_root)).encode()).digest())
        n = max(1, min(len(rows) - 1, round(len(rows) / 2)))
        gallery.extend(rows[:n])
        query.extend(rows[n:])
    query.sort()
    gallery.sort()
    hashes = tuple(hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() for rows in (query, gallery))
    if hashes != (QUERY_SHA, GALLERY_SHA):
        raise ValueError("In-Shop asymmetric roles differ")

    gallery_pose = Counter(pose[i] for i in gallery)
    gallery_set = set(gallery)
    seen: set[int] = set()
    observed = 0
    expected = 0.0
    by_pose: dict[str, list[float]] = defaultdict(lambda: [0, 0, 0.0])
    for miss in receipt["misses"]:
        q, impostor = (int(miss[key]) for key in ("query_held_row", "best_impostor_held_row"))
        if q not in query or q in seen or impostor not in gallery_set or labels[q] == labels[impostor]:
            raise ValueError("miss row identity or role differs")
        seen.add(q)
        same = int(pose[q] == pose[impostor])
        baseline = gallery_pose[pose[q]] / len(gallery)
        observed += same
        expected += baseline
        row = by_pose[pose[q]]
        row[0] += 1
        row[1] += same
        row[2] += baseline
    if len(seen) != 151 or len(query) != 6354 or len(gallery) != 6245:
        raise ValueError("In-Shop census cardinality differs")
    enrichment = ratio(observed, expected)
    report = {
        "schema": "sfora-inshop-pose-impostor-f1-train-only-v1",
        "claim_eligible": False,
        "miss_receipt_sha256": receipt_sha,
        "partition_sha256": PARTITION_SHA,
        "preflight_sha256": sha256(args.preflight),
        "query_sha256": QUERY_SHA,
        "gallery_sha256": GALLERY_SHA,
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "miss_count": len(seen),
        "same_pose_top_impostors": observed,
        "expected_same_pose_under_gallery_frequency": expected,
        "enrichment": enrichment,
        "advance_to_training": enrichment >= 2.0,
        "by_pose": {key: {"misses": int(v[0]), "same_pose": int(v[1]), "expected": v[2]} for key, v in sorted(by_pose.items())},
    }
    args.output.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    assert ratio(2, 0.5) == 4.0
    main()
