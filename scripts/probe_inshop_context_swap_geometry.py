#!/usr/bin/env python3
"""CPU-only, TRAIN-fit context-swap geometry and pixel authority gate."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from probe_inshop_bbox_context import BBOX_SHA, boxes_from_file
from train_sop_siglip2_compact import sha256

from sfora.unicom_inshop import parse_inshop_partition


def context_mask(boxes, size):
    width, height = size
    mask = np.ones((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        if not (1 <= x1 < x2 <= width and 1 <= y1 < y2 <= height):
            raise ValueError("context garment box outside image")
        mask[y1 - 1 : y2, x1 - 1 : x2] = False
    return mask


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root
    bbox = root / "Anno/list_bbox_inshop.txt"
    if (
        args.output.exists()
        or sha256(root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(bbox) != BBOX_SHA
    ):
        raise ValueError("context-swap source authority differs")
    train = tuple(r for r in parse_inshop_partition(root) if r.split == "train")
    fit, _ = split(tuple(r.label for r in train))
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("context-swap fit differs")
    names = {i: train[i].image_path.relative_to(root / "Img").as_posix() for i in fit}
    ordered = sorted(fit, key=lambda i: hashlib.sha256(names[i].encode()).digest())
    products, poses = defaultdict(list), defaultdict(list)

    def pose(i):
        return Path(names[i]).stem.rsplit("_", 1)[-1]

    for i in ordered:
        products[train[i].label].append(i)
        poses[pose(i)].append(i)
    labels = sorted(
        (p for p, rows in products.items() if len(rows) >= 3),
        key=lambda p: hashlib.sha256(str(p).encode()).digest(),
    )[:512]
    if len(labels) != 512:
        raise ValueError("context-swap product inventory differs")
    boxes = boxes_from_file(bbox)
    records = []
    for label in labels:
        q = products[label][0]
        same = [i for i in products[label][1:] if pose(i) == pose(q)]
        s = (same or products[label][1:])[0]
        foreign = [i for i in poses[pose(q)] if train[i].label != label]
        if not foreign:
            foreign = [i for i in ordered if train[i].label != label]
        d = foreign[
            int.from_bytes(hashlib.sha256(names[q].encode()).digest()[:8], "big") % len(foreign)
        ]
        rows = (q, s, d)
        b = [boxes[names[i]] for i in rows]
        pixels = []
        for i in rows:
            with Image.open(train[i].image_path) as image:
                if image.size != (256, 256):
                    raise ValueError("context-swap image dimensions differ")
                pixels.append(np.asarray(image.convert("RGB")))
        mask = context_mask(b, (256, 256))
        yy, xx = np.indices(mask.shape)
        independent = np.ones_like(mask)
        for x1, y1, x2, y2 in b:
            independent &= ~((xx >= x1 - 1) & (xx < x2) & (yy >= y1 - 1) & (yy < y2))
        assert np.array_equal(mask, independent)
        for donor, donor_box in zip(pixels[1:], b[1:], strict=True):
            composite = np.where(mask[..., None], donor, pixels[0])
            assert np.array_equal(composite[~mask], pixels[0][~mask])
            assert not np.any(mask & ~context_mask([donor_box], (256, 256)))
        naive = context_mask([b[0]], (256, 256))
        donor_garment = ~context_mask([b[2]], (256, 256))
        records.append(
            {
                "train_rows": rows,
                "names": [names[i] for i in rows],
                "image_sha256": [sha256(train[i].image_path) for i in rows],
                "boxes": b,
                "same_pose": [pose(q) == pose(s), pose(q) == pose(d)],
                "replaceable_fraction": float(mask.mean()),
                "naive_foreign_garment_fraction": float((naive & donor_garment).mean()),
            }
        )
        if time.perf_counter() - started > 120:
            raise ValueError("context-swap CPU budget exceeded")
    sufficient = sum(r["replaceable_fraction"] >= 0.2 for r in records)
    criteria = {
        "coverage": sufficient / 512 >= 0.9,
        "cpu_budget": time.perf_counter() - started <= 120,
    }
    result = {
        "schema": "sfora-inshop-context-swap-geometry-v1",
        "claim_eligible": False,
        "decision": "GO_TRAIN_FIT_DIAGNOSTIC" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "split": "officialTRAIN fit only",
        "records": records,
        "sufficient_context_count": sufficient,
        "partition_sha256": PARTITION_SHA,
        "bbox_sha256": BBOX_SHA,
        "fit_sha256": digest_rows(fit),
        "source_sha256": sha256(Path(__file__)),
        "cpu_wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
