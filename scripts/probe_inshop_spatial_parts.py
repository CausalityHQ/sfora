#!/usr/bin/env python3
"""Frozen TRAIN-only part correspondence screen on archived In-Shop misses."""

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
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from probe_inshop_bbox_context import scores
from score_inshop_crop_view_pair import GALLERY_SHA, PARTITION_SHA, QUERY_SHA, roles, sha256
from train_sop_siglip2_compact import export_all

from sfora.sop_compact_training import compact_head_features
from sfora.unicom_inshop import parse_inshop_partition

CHECKPOINT_SHA = "ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089"
MISSES_SHA = "8dfe23577547b188ae37aa0b83f564bd1dd8c411edb066f474724ccd2e89667c"
HELD_SHA = "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"


def part_score(query: torch.Tensor, gallery: torch.Tensor) -> float:
    if query.shape != gallery.shape or query.shape != (4, 128):
        raise ValueError("part geometry differs")
    return float((query @ gallery.T).amax(dim=1).mean())


def token_score(query: torch.Tensor, gallery: torch.Tensor) -> float:
    """Symmetric nearest-token cosine; token positions need not correspond."""
    if query.shape != gallery.shape or query.shape != (256, 1024):
        raise ValueError("raw token geometry differs")
    cosine = query @ gallery.T
    return float((cosine.amax(1).mean() + cosine.amax(0).mean()) / 2)


def main() -> None:
    whole_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "model-snapshot", "checkpoint", "misses", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--raw-token-match", action="store_true")
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.misses) != MISSES_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("spatial-part source authority differs")
    misses = json.loads(args.misses.read_text())
    if misses.get("checkpoint_sha256") != CHECKPOINT_SHA or misses.get("miss_count") != 151:
        raise ValueError("archived miss authority differs")
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
    for row in misses["misses"]:
        q, p, n = (
            int(row[key])
            for key in ("query_held_row", "best_positive_held_row", "best_impostor_held_row")
        )
        if q not in query or p not in gallery or n not in gallery:
            raise ValueError("miss role differs")
        if labels[q] != labels[p] or labels[q] == labels[n]:
            raise ValueError("miss identity differs")
        triples.append((q, p, n))
    if len(triples) != 151 or len({q for q, _, _ in triples}) != 151:
        raise ValueError("miss inventory differs")
    selected = sorted({row for triple in triples for row in triple})

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
    pooled_parts: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        tokens = output.last_hidden_state
        if tokens is None or tokens.ndim != 3 or tokens.shape[1:] != (256, 1024):
            raise ValueError("final patch grid differs")
        if args.raw_token_match:
            pooled_parts.append(tokens.float().cpu())
        else:
            blocks = tokens.float().reshape(-1, 2, 8, 2, 8, 1024).mean(dim=(2, 4))
            pooled_parts.append(blocks.reshape(-1, 4, 1024).cpu())

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    hook = vision.register_forward_hook(capture)
    try:
        original = export_all(
            vision,
            head,
            tuple(paths[row] for row in selected),
            tuple(selected),
            processor,
            workers=4,
            batch_size=64,
        )
    finally:
        hook.remove()
    source = torch.cat(pooled_parts)
    parts_count = 256 if args.raw_token_match else 4
    if source.shape != (len(selected), parts_count, 1024) or not bool(torch.isfinite(source).all()):
        raise ValueError("part source geometry differs")
    parts = []
    with torch.inference_mode():
        for chunk in source.split(64):
            if args.raw_token_match:
                if bool((chunk.norm(dim=2) == 0).any()):
                    raise ValueError("raw token norm is zero")
                parts.append(torch.nn.functional.normalize(chunk.cuda(), dim=2))
            else:
                projected = compact_head_features(chunk.reshape(-1, 1024).cuda(), head)
                parts.append(
                    torch.nn.functional.normalize(projected, dim=1).reshape(-1, 4, 128).cpu()
                )
    projected = torch.cat(parts)
    positions = {row: place for place, row in enumerate(selected)}
    replay_error = None
    if args.raw_token_match:
        original_margins = scores(original, triples, positions)
        replay_error = max(
            abs(actual - row["packed_margin"])
            for actual, row in zip(original_margins, misses["misses"], strict=True)
        )
        if replay_error > 0.005 or any(value >= 0 for value in original_margins):
            raise ValueError("original packed miss authority differs")
    score = token_score if args.raw_token_match else part_score
    margins = [
        score(projected[positions[q]], projected[positions[p]])
        - score(projected[positions[q]], projected[positions[n]])
        for q, p, n in triples
    ]
    wins = sum(value > 0 for value in margins)
    median = float(np.median(margins))
    report = {
        "schema": "sfora-inshop-raw-token-match-v1"
        if args.raw_token_match
        else "sfora-inshop-spatial-parts-falsifier-v1",
        "raw_token_match": args.raw_token_match,
        "original_margin_max_abs_replay_error": replay_error,
        "source_sha256": sha256(Path(__file__)),
        "source_files_sha256": {
            str(Path(function.__code__.co_filename)): sha256(Path(function.__code__.co_filename))
            for function in (export_all, scores, compact_head_features, parse_inshop_partition)
        },
        "partition_sha256": PARTITION_SHA,
        "held_rows_sha256": HELD_SHA,
        "query_rows_sha256": QUERY_SHA,
        "gallery_rows_sha256": GALLERY_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "misses_sha256": MISSES_SHA,
        "split": "official TRAIN held product-disjoint fixed roles",
        "distinct_images": len(selected),
        "misses": len(triples),
        "part_positive_wins": wins,
        "part_margin_median": median,
        "advance": wins >= 99 and median >= 0.02,
        "part_margins": margins,
        "export_and_score_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "claim_eligible": False,
    }
    report["whole_process_wall_seconds"] = time.perf_counter() - whole_started
    if args.raw_token_match:
        report["advance"] = (
            report["advance"]
            and report["whole_process_wall_seconds"] <= 60
            and report["peak_cuda_allocated_bytes"] <= 4_000_000_000
        )
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {key: report[key] for key in ("part_positive_wins", "part_margin_median", "advance")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
