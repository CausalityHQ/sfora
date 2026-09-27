#!/usr/bin/env python3
"""Archive ordered class labels used by the completed CUB/Cars transfer intervals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from export_unicom_cub_embeddings import ordered_record_sha256, parse_cub_records
from verify_sop_siglip2_cars_transfer import CACHE_SHA, DATASET, FINGERPRINTS, REVISION

ROOT = Path("/home/riomus/runs")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    cub = tuple(
        row
        for row in parse_cub_records(
            Path("/home/riomus/datasets/CUB_200_2011_official/extracted/CUB_200_2011")
        )
        if row.split == "test"
    )
    cars = load_dataset(DATASET, revision=REVISION)
    if {split: cars[split]._fingerprint for split in ("train", "test")} != FINGERPRINTS:
        raise ValueError("Cars label source differs")
    cache = {
        Path(entry["filename"]).name: Path(entry["filename"])
        for split in ("train", "test")
        for entry in cars[split].cache_files
    }
    if set(cache) != set(CACHE_SHA) or any(
        REVISION not in path.parts or sha(path) != CACHE_SHA[name] for name, path in cache.items()
    ):
        raise ValueError("Cars label cache differs")
    labels = {
        "cub": {
            "schema": "sfora-cub-transfer-ordered-labels-v1",
            "source_sha256": sha(Path(__file__)),
            "cub_test_records_sha256": ordered_record_sha256(cub),
            "labels": [row.label for row in cub],
        },
        "cars": {
            "schema": "sfora-cars-transfer-ordered-labels-v1",
            "source_sha256": sha(Path(__file__)),
            "dataset_revision": REVISION,
            "source_fingerprints": FINGERPRINTS,
            "cache_file_sha256": CACHE_SHA,
            "labels": [
                int(label)
                for split in ("train", "test")
                for label in cars[split]["label"]
                if int(label) >= 98
            ],
        },
    }
    for name, value in labels.items():
        target = ROOT / f"sfora-sop-{name}-transfer-public-v1/labels.json"
        if target.exists():
            raise ValueError(f"{name} labels already archived")
        target.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        print(name, len(value["labels"]), sha(target))


if __name__ == "__main__":
    main()
