#!/usr/bin/env python3
"""Measure packed In-Shop TRAIN miss rate as unseen impostor products increase."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import GALLERY_SHA, PARTITION_SHA, QUERY_SHA, roles, sha256
from verify_inshop_siglip2_public_checkpoint import CHECKPOINT_SHA, RECEIPT_SHA

from sfora.siglip2_compact_serving import Siglip2CompactEncoder
from sfora.unicom_inshop import parse_inshop_partition

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DRAWS = 16
SEED = 179027


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "model-snapshot", "training-receipt", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.training_receipt) != RECEIPT_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
    ):
        raise ValueError("In-Shop scaling authority differs")
    training = json.loads(args.training_receipt.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    labels = tuple(train[row].label for row in held)
    paths = tuple(train[row].image_path for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (
        (len(query), len(gallery)) != (6354, 6245)
        or training.get("checkpoint_sha256") != CHECKPOINT_SHA
        or training.get("seed") != 179026
        or any(
            hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != digest
            for rows, digest in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
        )
    ):
        raise ValueError("In-Shop scaling roles differ")
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = Siglip2CompactEncoder.from_checkpoint(
        model_snapshot=args.model_snapshot,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        model_file_sha256=training["model_file_sha256"],
        precision="fp16_native",
        device=torch.device("cuda:0"),
    )
    started = time.perf_counter()
    code, inverse = [], []
    for start in range(0, len(held), 32):
        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)) for path in paths[start : start + 32]]
            packed = encoder.encode_images(images)
        code.append(packed.codes)
        inverse.append(packed.inverse_norms)
    torch.cuda.synchronize()
    export_wall = time.perf_counter() - started
    codes = torch.cat(code).float().cuda()
    inverses = torch.cat(inverse).float().cuda()
    if codes.shape != (12599, 128) or not bool(torch.isfinite(inverses).all()):
        raise ValueError("In-Shop scaling packed geometry differs")
    products = {label: n for n, label in enumerate(sorted(set(labels)))}
    qproduct = np.asarray([products[labels[row]] for row in query], dtype=np.int32)
    gproduct = np.asarray([products[labels[row]] for row in gallery], dtype=np.int32)
    rng = np.random.default_rng(SEED)
    selections = {
        fraction: [
            np.sort(rng.choice(len(products), size=round(fraction * len(products)), replace=False))
            for _ in range(DRAWS if fraction < 1 else 1)
        ]
        for fraction in FRACTIONS
    }
    # A sampled product is an impostor only for queries from other products;
    # each query always keeps every same-product positive in the fixed gallery.
    hits = {fraction: [[] for _ in draws] for fraction, draws in selections.items()}
    started = time.perf_counter()
    gclass = torch.tensor(gproduct, device="cuda")
    masks = {
        fraction: [
            torch.isin(gclass, torch.tensor(selected, device="cuda")) for selected in draws
        ]
        for fraction, draws in selections.items()
    }
    gcodes = codes[gallery]
    ginverse = inverses[gallery]
    for start in range(0, len(query), 32):
        rows = query[start : start + 32]
        classes = torch.tensor(qproduct[start : start + len(rows)], device="cuda")
        score = (codes[rows] @ gcodes.T) * inverses[rows, None] * ginverse[None, :]
        positive = classes[:, None] == gclass[None, :]
        best_positive, positive_index = score.masked_fill(~positive, -torch.inf).max(dim=1)
        if not bool(torch.isfinite(best_positive).all()):
            raise ValueError("In-Shop scaling query lacks a positive")
        for fraction, draws in masks.items():
            for draw, admitted in enumerate(draws):
                best_negative, negative_index = score.masked_fill(
                    positive | ~admitted[None, :], -torch.inf
                ).max(dim=1)
                hit = (best_positive > best_negative) | (
                    (best_positive == best_negative) & (positive_index < negative_index)
                )
                hits[fraction][draw].extend(hit.cpu().tolist())
    torch.cuda.synchronize()
    score_wall = time.perf_counter() - started
    if len(hits[1.0][0]) != 6354 or sum(hits[1.0][0]) != 6203:
        raise ValueError("In-Shop scaling public full-gallery parity differs")
    result = {
        "schema": "sfora-inshop-impostor-scaling-train-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "training_receipt_sha256": RECEIPT_SHA,
        "partition_sha256": PARTITION_SHA,
        "query_rows_sha256": QUERY_SHA,
        "gallery_rows_sha256": GALLERY_SHA,
        "query_count": len(query),
        "gallery_count": len(gallery),
        "product_count": len(products),
        "seed": SEED,
        "draws": DRAWS,
        "export_wall_seconds": export_wall,
        "score_wall_seconds": score_wall,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "fractions": {
            str(fraction): {
                "impostor_product_count": len(draws[0]),
                "miss_counts": [len(hit) - sum(hit) for hit in hits[fraction]],
            }
            for fraction, draws in selections.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"fractions": result["fractions"], "export_wall_seconds": export_wall}))


if __name__ == "__main__":
    main()
