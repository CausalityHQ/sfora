#!/usr/bin/env python3
"""Frozen TRAIN-only garment-box crop screen on archived In-Shop misses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from score_inshop_crop_view_pair import GALLERY_SHA, PARTITION_SHA, QUERY_SHA, roles, sha256
from train_sop_siglip2_compact import export_all

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_inshop import parse_inshop_partition

CHECKPOINT_SHA = "ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089"
MISSES_SHA = "8dfe23577547b188ae37aa0b83f564bd1dd8c411edb066f474724ccd2e89667c"
BBOX_SHA = "b1a67bf2b1bb22ce57b63238dfabfd987f21ef85e93448f01735148f5848f440"
HELD_SHA = "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"


def boxes_from_file(path: Path) -> dict[str, tuple[int, int, int, int]]:
    rows = path.read_text().splitlines()
    if len(rows) != 52_714 or rows[1].split() != [
        "image_name",
        "clothes_type",
        "pose_type",
        "x_1",
        "y_1",
        "x_2",
        "y_2",
    ]:
        raise ValueError("In-Shop box inventory differs")
    boxes = {}
    for line in rows[2:]:
        fields = line.split()
        if len(fields) != 7 or fields[0] in boxes:
            raise ValueError("In-Shop box row differs")
        boxes[fields[0]] = tuple(map(int, fields[-4:]))
    return boxes


def boxed_image(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 < image.width and 0 <= y1 < y2 < image.height):
        raise ValueError("In-Shop garment box differs")
    crop = image.convert("RGB").crop((x1, y1, x2 + 1, y2 + 1))
    side = max(crop.size)
    square = Image.new("RGB", (side, side), (127, 127, 127))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square


def scores(
    values: torch.Tensor, triples: list[tuple[int, int, int]], positions: dict[int, int]
) -> list[float]:
    packed = pack_int8_unit_embeddings(values)
    codes = packed.codes.float()
    norms = packed.inverse_norms.float()
    margins = []
    for q, p, n in triples:
        qi, pi, ni = positions[q], positions[p], positions[n]
        positive = torch.dot(codes[qi], codes[pi]) * norms[qi] * norms[pi]
        negative = torch.dot(codes[qi], codes[ni]) * norms[qi] * norms[ni]
        margins.append(float(positive - negative))
    return margins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "model-snapshot", "checkpoint", "misses", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    bbox_file = args.dataset_root / "Anno/list_bbox_inshop.txt"
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.misses) != MISSES_SHA
        or sha256(bbox_file) != BBOX_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("box-context source authority differs")
    archive = json.loads(args.misses.read_text())
    if archive.get("checkpoint_sha256") != CHECKPOINT_SHA or archive.get("miss_count") != 151:
        raise ValueError("archived misses differ")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if len(held) != 12_599 or digest_rows(held) != HELD_SHA:
        raise ValueError("TRAIN held split differs")
    paths = tuple(train[row].image_path for row in held)
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    for rows, digest in ((query, QUERY_SHA), (gallery, GALLERY_SHA)):
        if hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != digest:
            raise ValueError("TRAIN held role digest differs")
    triples = []
    for item in archive["misses"]:
        q, p, n = (
            int(item[key])
            for key in ("query_held_row", "best_positive_held_row", "best_impostor_held_row")
        )
        if q not in query or p not in gallery or n not in gallery:
            raise ValueError("archived miss role differs")
        if labels[q] != labels[p] or labels[q] == labels[n]:
            raise ValueError("archived miss identity differs")
        triples.append((q, p, n))
    if len(triples) != 151 or len({q for q, _, _ in triples}) != 151:
        raise ValueError("archived miss inventory differs")
    selected = sorted({row for triple in triples for row in triple})
    positions = {row: place for place, row in enumerate(selected)}
    boxes = boxes_from_file(bbox_file)
    for row in selected:
        key = str(paths[row].relative_to(args.dataset_root / "Img"))
        if key not in boxes:
            raise ValueError("selected image lacks garment box")

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = torch.nn.Linear(1024, 128).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vision.load_state_dict(state["vision"], strict=True)
    head.load_state_dict(state["head"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    original = export_all(
        vision,
        head,
        tuple(paths[row] for row in selected),
        tuple(selected),
        processor,
        workers=4,
        batch_size=64,
    )
    original_margins = scores(original, triples, positions)
    error = max(
        abs(actual - item["packed_margin"])
        for actual, item in zip(original_margins, archive["misses"], strict=True)
    )
    if error > 1e-5 or any(margin >= 0 for margin in original_margins):
        raise ValueError(f"archived packed triple replay differs: {error}")
    cropped = []
    with torch.inference_mode():
        for start in range(0, len(selected), 32):
            images = []
            for row in selected[start : start + 32]:
                with Image.open(paths[row]) as image:
                    key = str(paths[row].relative_to(args.dataset_root / "Img"))
                    images.append(boxed_image(image, boxes[key]))
            batch = processor(images=images, return_tensors="pt")
            pixels = batch["pixel_values"].cuda()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pooled = vision(pixel_values=pixels).pooler_output
            features = compact_head_features(pooled, head)
            cropped.append(torch.nn.functional.normalize(features, dim=1).cpu())
    cropped_values = torch.cat(cropped)
    if cropped_values.shape != original.shape or not bool(torch.isfinite(cropped_values).all()):
        raise ValueError("cropped export geometry differs")
    margins = scores(cropped_values, triples, positions)
    wins = sum(value > 0 for value in margins)
    median = float(np.median(margins))
    report = {
        "schema": "sfora-inshop-bbox-context-falsifier-v1",
        "source_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "misses_sha256": MISSES_SHA,
        "bbox_sha256": BBOX_SHA,
        "split": "official TRAIN held product-disjoint fixed roles",
        "distinct_images": len(selected),
        "misses": len(triples),
        "original_margin_max_abs_replay_error": error,
        "cropped_positive_wins": wins,
        "cropped_margin_median": median,
        "advance": wins >= 90 and median >= 0.02,
        "cropped_margins": margins,
        "export_and_score_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "claim_eligible": False,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "original_margin_max_abs_replay_error",
                    "cropped_positive_wins",
                    "cropped_margin_median",
                    "advance",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
