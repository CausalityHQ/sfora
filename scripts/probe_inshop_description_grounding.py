#!/usr/bin/env python3
"""TRAIN-fit-only eligibility screen for semantic product descriptions."""

import argparse
import json
import re
from pathlib import Path

from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from score_inshop_crop_view_pair import PARTITION_SHA, sha256

METADATA_SHA = "78f320b9fe59ed0d08d803a599f512c06c05327065052b33495cbd4dab6c4b87"
FIT_SHA = "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be"


def fit_captions(rows: list[dict], labels: set[str]) -> dict[str, str]:
    captions = {}
    for row in rows:
        name = row["item"]
        if name not in labels:
            continue
        description = row.get("description")
        color = row.get("color")
        if (
            not isinstance(description, list)
            or not description
            or not isinstance(description[0], str)
            or not isinstance(color, str)
        ):
            continue
        text = " ".join(f"{color.strip()}. {description[0].strip()}".split())
        if re.search(r"\bid_[a-z0-9]+\b", text, flags=re.IGNORECASE):
            raise ValueError("product identity leaked into description")
        if len(text) >= 20:
            if name in captions and captions[name] != text:
                raise ValueError("conflicting fit product description")
            captions[name] = text
    return captions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("partition", "metadata", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(args.metadata) != METADATA_SHA
    ):
        raise ValueError("description screen authority differs")
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != FIT_SHA:
        raise ValueError("description fit split differs")
    names = {labels[row] for row in fit}
    captions = fit_captions(json.loads(args.metadata.read_text()), names)
    coverage = len(captions) / len(names)
    unique_fraction = len(set(captions.values())) / len(captions) if captions else 0
    report = {
        "schema": "sfora-inshop-description-eligibility-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN fit products only; no held or official outcome",
        "source_sha256": sha256(Path(__file__)),
        "metadata_sha256": METADATA_SHA,
        "fit_rows_sha256": FIT_SHA,
        "fit_products": len(names),
        "usable_captions": len(captions),
        "coverage": coverage,
        "unique_caption_fraction": unique_fraction,
        "advance": coverage >= 0.95 and unique_fraction >= 0.90,
        "encoder_export_or_training": False,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
