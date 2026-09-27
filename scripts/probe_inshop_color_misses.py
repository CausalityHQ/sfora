#!/usr/bin/env python3
"""TRAIN-only color-distance falsifier for archived In-Shop packed misses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import GALLERY_SHA, QUERY_SHA, roles, sha256

from sfora.unicom_inshop import parse_inshop_partition

PARTITION_SHA = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
EXPECTED_SHA = "d48e2c382fcfb7aa4807b00cc8b2a76fe098aebfa77334dbc2fd8402d36b46a4"
MISSES_SHA = "8dfe23577547b188ae37aa0b83f564bd1dd8c411edb066f474724ccd2e89667c"


def color(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        image = image.crop((width // 5, height // 5, 4 * width // 5, 4 * height // 5))
        pixels = np.asarray(image.resize((32, 32)), dtype=np.uint8)
    return (
        np.concatenate(
            [np.histogram(pixels[:, :, channel], bins=8, range=(0, 256))[0] for channel in range(3)]
        ).astype(np.float64)
        / 1024
    )


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.sum((left - right) ** 2 / (left + right + 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "expected", "misses", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.expected) != EXPECTED_SHA
        or sha256(args.misses) != MISSES_SHA
    ):
        raise ValueError("In-Shop color screen authority differs")
    expected = json.loads(args.expected.read_text())
    misses = json.loads(args.misses.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    paths = tuple(train[row].image_path for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (
        len(query) != 6354
        or len(gallery) != 6245
        or len(misses.get("misses", [])) != 151
        or len(expected.get("held_image_sha256", [])) != len(held)
        or hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != QUERY_SHA
        or hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest() != GALLERY_SHA
    ):
        raise ValueError("In-Shop color screen role inventory differs")
    needed = {
        row[key]
        for row in misses["misses"]
        for key in ("query_held_row", "best_positive_held_row", "best_impostor_held_row")
    }
    if any(type(row) is not int or not 0 <= row < len(held) for row in needed):
        raise ValueError("In-Shop color screen row inventory differs")
    features = {}
    for row in sorted(needed):
        if sha256(paths[row]) != expected["held_image_sha256"][row]:
            raise ValueError("In-Shop color screen image digest differs")
        features[row] = color(paths[row])
    comparisons = []
    for miss in misses["misses"]:
        q, p, n = (
            miss[key]
            for key in ("query_held_row", "best_positive_held_row", "best_impostor_held_row")
        )
        if (
            q not in query
            or p not in gallery
            or n not in gallery
            or not (labels[q] == labels[p] != labels[n])
        ):
            raise ValueError("In-Shop color screen miss triple differs")
        positive = distance(features[q], features[p])
        negative = distance(features[q], features[n])
        comparisons.append(
            {"query_held_row": q, "positive_distance": positive, "negative_distance": negative}
        )
    advantages = np.asarray(
        [row["negative_distance"] - row["positive_distance"] for row in comparisons]
    )
    report = {
        "schema": "sfora-inshop-color-miss-f0-train-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "partition_sha256": PARTITION_SHA,
        "expected_receipt_sha256": EXPECTED_SHA,
        "miss_receipt_sha256": MISSES_SHA,
        "query_rows_sha256": QUERY_SHA,
        "gallery_rows_sha256": GALLERY_SHA,
        "images_hashed": len(features),
        "misses": len(comparisons),
        "closer_positive": int(np.count_nonzero(advantages > 0)),
        "closer_positive_fraction": float(np.mean(advantages > 0)),
        "median_distance_advantage": float(np.median(advantages)),
        "advance": bool(np.mean(advantages > 0) >= 0.60 and np.median(advantages) > 0.02),
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(report, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "misses",
                    "closer_positive",
                    "closer_positive_fraction",
                    "median_distance_advantage",
                    "advance",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
