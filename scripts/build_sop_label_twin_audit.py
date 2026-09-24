#!/usr/bin/env python3
"""Make a seeded contact sheet of SOP TRAIN gallery-induced errors for visual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate_sop_reference_checkpoint import EXPECTED_SOP_RECORD_SHA256
from export_unicom_sop_embeddings import ordered_record_sha256, parse_sop_records
from PIL import Image, ImageDraw, ImageOps
from train_sop_compact_backbone import sha256

CENSUS_SHA256 = "71909711f5bab9f8d79350fae91441318020f01cfc8a079ea344dac4b20997a1"
SEED = 179019
AUDIT_PAIRS = 50
PAIRS_PER_PAGE = 5
IMAGE_BOX = (235, 210)


def paste_image(canvas: Image.Image, path: Path, left: int, top: int) -> None:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail(IMAGE_BOX, Image.Resampling.LANCZOS)
        canvas.paste(image, (left + (IMAGE_BOX[0] - image.width) // 2, top))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--sop-root", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-sop-label-twin-audit", action="store_true", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or sha256(args.census) != CENSUS_SHA256:
        raise ValueError("SOP label-twin audit input differs")
    records = parse_sop_records(args.sop_root)
    if ordered_record_sha256(records) != EXPECTED_SOP_RECORD_SHA256:
        raise ValueError("SOP ordered record authority differs")
    train = tuple(row for row in records if row.split == "train")
    labels = np.asarray([row.label for row in train], dtype=np.int64)
    census = json.loads(args.census.read_text())
    query = np.asarray(census["query_row_indexes"], dtype=np.int64)
    full = np.asarray(census["full_top1_row_indexes"], dtype=np.int64)
    hold = np.asarray(census["holdout_top1_row_indexes"], dtype=np.int64)
    cosine = np.asarray(census["full_top1_cosines"], dtype=np.float64)
    if (
        len(query) != 5_851
        or len(full) != len(query)
        or len(hold) != len(query)
        or np.any(query < 0)
        or np.any(full < 0)
        or np.any(hold < 0)
        or np.any(query >= len(train))
        or np.any(full >= len(train))
        or np.any(hold >= len(train))
    ):
        raise ValueError("SOP label-twin audit census geometry differs")
    new_errors = np.flatnonzero(
        (labels[hold] == labels[query]) & (labels[full] != labels[query])
    )
    if len(new_errors) != 436:
        raise ValueError("SOP gallery-induced error population differs")
    selected = np.sort(np.random.default_rng(SEED).choice(new_errors, AUDIT_PAIRS, replace=False))
    manifest = []
    args.output_dir.mkdir(parents=True)
    for page in range(AUDIT_PAIRS // PAIRS_PER_PAGE):
        canvas = Image.new("RGB", (510, PAIRS_PER_PAGE * 250), "white")
        draw = ImageDraw.Draw(canvas)
        for slot in range(PAIRS_PER_PAGE):
            position = page * PAIRS_PER_PAGE + slot
            query_slot = int(selected[position])
            query_row = int(query[query_slot])
            wrong_row = int(full[query_slot])
            query_record = train[query_row]
            wrong_record = train[wrong_row]
            top = slot * 250
            paste_image(canvas, query_record.image_path, 10, top + 8)
            paste_image(canvas, wrong_record.image_path, 265, top + 8)
            draw.text(
                (10, top + 220),
                f"{position + 1:02d} Q {query_record.image_id} / {query_record.label}  "
                f"W {wrong_record.image_id} / {wrong_record.label}  cos {cosine[query_slot]:.3f}",
                fill="black",
            )
            manifest.append(
                {
                    "ordinal": position + 1,
                    "query_slot": query_slot,
                    "query_image_id": query_record.image_id,
                    "wrong_image_id": wrong_record.image_id,
                    "query_product": query_record.label,
                    "wrong_product": wrong_record.label,
                    "query_relative_path": query_record.relative_path,
                    "wrong_relative_path": wrong_record.relative_path,
                    "query_image_sha256": hashlib.sha256(
                        query_record.image_path.read_bytes()
                    ).hexdigest(),
                    "wrong_image_sha256": hashlib.sha256(
                        wrong_record.image_path.read_bytes()
                    ).hexdigest(),
                    "cosine": float(cosine[query_slot]),
                    "page": page + 1,
                    "slot": slot + 1,
                }
            )
        canvas.save(args.output_dir / f"page-{page + 1:02d}.png")
    result = {
        "schema": "sfora-sop-label-twin-visual-audit-v1",
        "claim_eligible": False,
        "population": "436 SOP TRAIN queries newly wrong after adding fit-identity rows",
        "seed": SEED,
        "selected_count": len(manifest),
        "census_sha256": CENSUS_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "pairs": manifest,
        "human_labels": "pending independent visual review; contact sheet alone is not a label",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "pairs": len(manifest)}))


if __name__ == "__main__":
    main()
