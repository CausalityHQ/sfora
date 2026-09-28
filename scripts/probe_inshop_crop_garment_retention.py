#!/usr/bin/env python3
"""Read-only native augmentation support audit; TRAIN fit only, CPU only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from probe_inshop_bbox_context import BBOX_SHA, boxes_from_file
from torchvision.transforms import RandomResizedCrop
from train_sop_siglip2_compact import sha256

from sfora.unicom_inshop import parse_inshop_partition


def retained_area(box, crop):
    """Inclusive1-based garment box versus torchvision top,left,height,width."""
    x1, y1, x2, y2 = box
    top, left, height, width = crop
    if not (1 <= x1 < x2 and 1 <= y1 < y2 and height > 0 and width > 0):
        raise ValueError("crop/garment rectangle differs")
    intersection = max(0, min(x2, left + width) - max(x1 - 1, left)) * max(
        0, min(y2, top + height) - max(y1 - 1, top)
    )
    return intersection / ((x2 - x1 + 1) * (y2 - y1 + 1))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    partition = args.dataset_root / "Eval/list_eval_partition.txt"
    bbox = args.dataset_root / "Anno/list_bbox_inshop.txt"
    if args.output.exists() or sha256(partition) != PARTITION_SHA or sha256(bbox) != BBOX_SHA:
        raise ValueError("garment retention source authority differs")
    train = tuple(r for r in parse_inshop_partition(args.dataset_root) if r.split == "train")
    fit, _ = split(tuple(r.label for r in train))
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("garment retention fit differs")
    boxes = boxes_from_file(bbox)
    names = {
        i: train[i].image_path.relative_to(args.dataset_root / "Img").as_posix() for i in fit
    }
    selected = sorted(fit, key=lambda i: hashlib.sha256(names[i].encode()).digest())[:512]
    torch.manual_seed(179024)
    records = []
    for row in selected:
        item = train[row]
        box = boxes[names[row]]
        with Image.open(item.image_path) as image:
            if not (1 <= box[0] < box[2] <= image.width and 1 <= box[1] < box[3] <= image.height):
                raise ValueError("garment retention box outside image")
            crops = [
                RandomResizedCrop.get_params(image, scale=(0.8, 1), ratio=(0.75, 4 / 3))
                for _ in range(8)
            ]
            size = image.size
        records.append(
            {
                "train_row": row,
                "image_name": names[row],
                "image_sha256": sha256(item.image_path),
                "size": size,
                "box": box,
                "crops": crops,
                "retained_area": [retained_area(box, c) for c in crops],
            }
        )
        if time.perf_counter() - started > 120:
            raise ValueError("garment retention CPU budget exceeded")
    values = [v for r in records for v in r["retained_area"]]
    criteria = {
        "inventory": len(values) == 4096,
        "support_loss": sum(v < 0.9 for v in values) / len(values) >= 0.1,
        "major_support_loss": sum(v < 0.75 for v in values) / len(values) >= 0.05,
        "cpu_budget": time.perf_counter() - started <= 120,
    }
    result = {
        "schema": "sfora-inshop-crop-garment-retention-v1",
        "claim_eligible": False,
        "decision": "GO_PAIRED_REPRESENTATION_DIAGNOSTIC" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "rows": records,
        "split": "officialTRAIN fit only",
        "below_90_count": sum(v < 0.9 for v in values),
        "below_75_count": sum(v < 0.75 for v in values),
        "partition_sha256": PARTITION_SHA,
        "bbox_sha256": BBOX_SHA,
        "fit_sha256": digest_rows(fit),
        "source_sha256": sha256(Path(__file__)),
        "cpu_wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    assert retained_area((1, 1, 10, 10), (0, 0, 10, 10)) == 1
    assert retained_area((1, 1, 10, 10), (0, 5, 10, 10)) == 0.5
    assert retained_area((1, 1, 10, 10), (10, 10, 10, 10)) == 0
    main()
