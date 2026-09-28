#!/usr/bin/env python3
"""TRAIN-only source screen for SOP-trained SigLIP2 transfer to In-Shop."""

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
from score_inshop_crop_view_pair import (
    GALLERY_SHA,
    PARTITION_SHA,
    QUERY_SHA,
    bootstrap_lower,
    roles,
    sha256,
)
from torch.utils.data import DataLoader
from train_sop_siglip2_compact import ImageRows, make_collate

from sfora.unicom_inshop import parse_inshop_partition

CHECKPOINT_SHA = "2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172"
CHECKPOINT_RECEIPT_SHA = "07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5"
SOURCE_CACHE_SHA = "f232584491bf4ed1cf75daa7fcc03f218e7db5df797f4d57dd2bf034ac110885"
SOURCE_RANK_SHA = "77ce284af0b15259127ef33fe8d046554040a977772e84c04e13aaeca6e29eb8"
HELD_SHA = "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"


@torch.inference_mode()
def score(
    values: torch.Tensor,
    labels: tuple[str, ...],
    query: list[int],
    gallery: list[int],
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    if values.ndim != 2 or values.shape[0] != len(labels) or not bool(torch.isfinite(values).all()):
        raise ValueError("source feature geometry differs")
    device = device or torch.device("cuda")
    code = torch.nn.functional.normalize(values.float(), dim=1).to(device)
    names = {name: idx for idx, name in enumerate(sorted(set(labels)))}
    classes = torch.tensor([names[name] for name in labels], device=device)
    gallery_ids = classes[gallery]
    relevant = torch.bincount(gallery_ids)[classes[query]]
    if int(relevant.min()) < 1:
        raise ValueError("In-Shop source positive inventory differs")
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device=device)
    per_hit: list[int] = []
    per_ap: list[float] = []
    for start in range(0, len(query), 128):
        rows = query[start : start + 128]
        similarity = code[rows] @ code[gallery].T
        ranked = torch.argsort(similarity, dim=1, descending=True, stable=True)[:, :width]
        matches = gallery_ids[ranked] == classes[rows, None]
        count = relevant[start : start + len(rows)]
        precision = matches.cumsum(dim=1) / ranks[None, :]
        ap = (precision * matches * (ranks[None, :] <= count[:, None])).sum(dim=1) / count
        per_hit.extend(int(value) for value in matches[:, 0].cpu().tolist())
        per_ap.extend(float(value) for value in ap.cpu().tolist())
    return {
        "recall_at_1": float(np.mean(per_hit)),
        "map_at_r": float(np.mean(per_ap)),
        "per_query_r1": per_hit,
        "per_query_ap": per_ap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "checkpoint",
        "checkpoint-receipt",
        "source-cache",
        "source-rank-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.checkpoint_receipt) != CHECKPOINT_RECEIPT_SHA
        or sha256(args.source_cache) != SOURCE_CACHE_SHA
        or sha256(args.source_rank_receipt) != SOURCE_RANK_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("SOP prior screen source authority differs")
    checkpoint_receipt = json.loads(args.checkpoint_receipt.read_text())
    archived = json.loads(args.source_rank_receipt.read_text())
    if (
        checkpoint_receipt.get("checkpoint_sha256") != CHECKPOINT_SHA
        or checkpoint_receipt.get("seed") != 179024
        or checkpoint_receipt.get("updates") != 1000
        or len(archived.get("pretrained_source_per_query_r1", ())) != 6_354
    ):
        raise ValueError("SOP prior screen receipt authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    if len(train) != 25_882 or len(held) != 12_599 or digest_rows(held) != HELD_SHA:
        raise ValueError("In-Shop TRAIN held split differs")
    paths = tuple(train[row].image_path for row in held)
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    for rows, digest in ((query, QUERY_SHA), (gallery, GALLERY_SHA)):
        if hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != digest:
            raise ValueError("In-Shop TRAIN role digest differs")
    source = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if source.shape != (len(train), 1024) or source.dtype != np.float32:
        raise ValueError("pretrained source cache differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    started = time.perf_counter()
    control = score(torch.from_numpy(np.asarray(source[list(held)]).copy()), labels, query, gallery)
    if control["per_query_r1"] != archived["pretrained_source_per_query_r1"]:
        raise ValueError("pretrained source R@1 does not replay")

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vision.load_state_dict(state["vision"], strict=True)
    loader = DataLoader(
        ImageRows(paths, tuple(range(len(paths))), augment=False),
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=make_collate(processor),
    )
    torch.cuda.reset_peak_memory_stats()
    export_started = time.perf_counter()
    chunks = []
    with torch.inference_mode():
        for batch, _ in loader:
            pixels = batch["pixel_values"].cuda(non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pooled = vision(pixel_values=pixels).pooler_output
            if pooled is None or pooled.shape != (len(pixels), 1024):
                raise ValueError("SOP product-prior source geometry differs")
            chunks.append(pooled.float().cpu())
    torch.cuda.synchronize()
    export_s = time.perf_counter() - export_started
    treatment = score(torch.cat(chunks), labels, query, gallery)
    delta = np.asarray(treatment["per_query_r1"], dtype=np.float64) - np.asarray(
        control["per_query_r1"], dtype=np.float64
    )
    label_array = np.asarray([labels[row] for row in query])
    lower = bootstrap_lower(delta, label_array)
    upper = -bootstrap_lower(-delta, label_array)
    gain = float(np.mean(delta))
    report = {
        "schema": "sfora-inshop-sop-product-prior-source-v1",
        "source_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "source_cache_sha256": SOURCE_CACHE_SHA,
        "source_rank_receipt_sha256": SOURCE_RANK_SHA,
        "split": "official TRAIN held products; 6354 query / 6245 gallery",
        "baseline": control,
        "sop_product_prior": treatment,
        "r1_gain_percentage_points": 100 * gain,
        "r1_paired_product_bootstrap_95_pp": [100 * lower, 100 * upper],
        "source_gate_passed": gain >= 0.01
        and lower > 0
        and treatment["map_at_r"] >= control["map_at_r"],
        "export_wall_seconds": export_s,
        "total_wall_seconds": time.perf_counter() - started,
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
                    "r1_gain_percentage_points",
                    "r1_paired_product_bootstrap_95_pp",
                    "source_gate_passed",
                    "export_wall_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
