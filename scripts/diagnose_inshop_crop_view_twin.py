#!/usr/bin/env python3
"""Metadata-only same-framing-positive test on frozen In-Shop TRAIN misses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from preflight_inshop_siglip2_unseen_gallery import split

from sfora.unicom_inshop import parse_inshop_partition

PARTITION_SHA = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
MISSES_SHA = "8dfe23577547b188ae37aa0b83f564bd1dd8c411edb066f474724ccd2e89667c"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"
FRAMINGS = {"front", "side", "back", "full", "additional", "flat"}
TARGET = {"additional", "full", "flat"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def roles(
    labels: tuple[str, ...], paths: tuple[Path, ...], root: Path
) -> tuple[list[int], list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row, label in enumerate(labels):
        grouped[label].append(row)
    query, gallery = [], []
    for label in sorted(grouped):
        rows = sorted(
            grouped[label],
            key=lambda row: hashlib.sha256(str(paths[row].relative_to(root)).encode()).digest(),
        )
        count = max(1, min(len(rows) - 1, round(len(rows) / 2)))
        gallery.extend(rows[:count])
        query.extend(rows[count:])
    return sorted(query), sorted(gallery)


def framing(path: Path) -> str:
    value = path.stem.rsplit("_", 1)[-1]
    if value not in FRAMINGS:
        raise ValueError("In-Shop framing differs")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--misses-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.misses_receipt) != MISSES_SHA
    ):
        raise ValueError("In-Shop crop-view authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    paths = tuple(train[row].image_path for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (len(query), len(gallery)) != (6354, 6245) or any(
        hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != expected
        for rows, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
    ):
        raise ValueError("In-Shop crop-view roles differ")
    prior = json.loads(args.misses_receipt.read_text())
    misses = [item["query_held_row"] for item in prior["misses"]]
    if (
        prior.get("miss_count") != 151
        or len(misses) != 151
        or len(set(misses)) != 151
        or not set(misses).issubset(query)
    ):
        raise ValueError("In-Shop crop-view misses differ")
    missed = set(misses)
    gallery_framings = {(labels[row], framing(paths[row])) for row in gallery}
    counts = {"with_twin": {"queries": 0, "misses": 0}, "without_twin": {"queries": 0, "misses": 0}}
    by_framing = {name: {"queries": 0, "misses": 0} for name in sorted(FRAMINGS)}
    for row in query:
        pose = framing(paths[row])
        by_framing[pose]["queries"] += 1
        by_framing[pose]["misses"] += row in missed
        if pose in TARGET:
            group = "with_twin" if (labels[row], pose) in gallery_framings else "without_twin"
            counts[group]["queries"] += 1
            counts[group]["misses"] += row in missed
    if by_framing != {
        "additional": {"queries": 1324, "misses": 46},
        "back": {"queries": 1294, "misses": 30},
        "flat": {"queries": 82, "misses": 6},
        "front": {"queries": 1505, "misses": 29},
        "full": {"queries": 817, "misses": 26},
        "side": {"queries": 1332, "misses": 14},
    }:
        raise ValueError("In-Shop crop-view framing census differs")
    with_twin, without_twin = counts["with_twin"], counts["without_twin"]
    rates = {
        name: value["misses"] / value["queries"] if value["queries"] else None
        for name, value in counts.items()
    }
    ratio = (
        rates["without_twin"] / rates["with_twin"]
        if rates["without_twin"] is not None and rates["with_twin"]
        else None
    )
    advance = (
        min(with_twin["queries"], without_twin["queries"]) >= 100
        and min(with_twin["misses"], without_twin["misses"]) >= 1
        and ratio is not None
        and ratio >= 1.5
    )
    report = {
        "schema": "sfora-inshop-crop-view-twin-train-preflight-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "partition_sha256": PARTITION_SHA,
        "misses_receipt_sha256": MISSES_SHA,
        "query_sha256": QUERY_SHA,
        "gallery_sha256": GALLERY_SHA,
        "by_framing": by_framing,
        "groups": counts,
        "rates": rates,
        "no_twin_to_twin_miss_rate_ratio": ratio,
        "advance_to_training": advance,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps({"groups": counts, "ratio": ratio, "advance_to_training": advance}), flush=True
    )


if __name__ == "__main__":
    main()
