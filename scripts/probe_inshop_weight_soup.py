#!/usr/bin/env python3
"""One frozen checkpoint-average screen on product-disjoint In-Shop TRAIN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from evaluate_unicom_checkpoint_soup import average_model_states
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from train_sop_siglip2_compact import export_all, score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

SEEDS = (179024, 179026, 179027)
CHECKPOINT_SHAS = (
    "dc5025998a3ea1902cb8251dedb1a4d4fd2659ffc20413c3425178b3e311f376",
    "ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089",
    "4816912a52ed939e4461ba566abfcc0f3e4994b839ddc90e9c68355c554869a0",
)
FIT_SHA = "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be"
HELD_SHA = "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"
PCA_SHA = "f387aae10a5fe080ebbc9a8ec0e77ff9e86b4812b6e1066048de1cf7cf1e4ea8"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"


def asymmetric_rows(paths: tuple[Path, ...], labels: tuple[str, ...], root: Path) -> tuple[list[int], list[int]]:
    grouped: dict[str, list[int]] = {}
    for row, label in enumerate(labels):
        grouped.setdefault(label, []).append(row)
    query: list[int] = []
    gallery: list[int] = []
    for label in sorted(grouped):
        rows = sorted(
            grouped[label],
            key=lambda row: hashlib.sha256(str(paths[row].relative_to(root)).encode()).digest(),
        )
        count = max(1, min(len(rows) - 1, round(len(rows) / 2)))
        gallery.extend(rows[:count])
        query.extend(rows[count:])
    query.sort()
    gallery.sort()
    if (
        hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != QUERY_SHA
        or hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest() != GALLERY_SHA
    ):
        raise ValueError("fixed asymmetric roles differ")
    return query, gallery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "model-snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or len(args.checkpoint) != 3
        or len(args.receipt) != 3
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items())
    ):
        raise ValueError("soup input authority differs")
    receipts = [json.loads(path.read_text()) for path in args.receipt]
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    if len(fit) != 13_283 or len(held) != 12_599 or digest_rows(fit) != FIT_SHA or digest_rows(held) != HELD_SHA:
        raise ValueError("soup TRAIN split differs")
    for seed, checkpoint_path, receipt, expected_sha in zip(
        SEEDS, args.checkpoint, receipts, CHECKPOINT_SHAS, strict=True
    ):
        if (
            sha256(checkpoint_path) != expected_sha
            or receipt["checkpoint_sha256"] != expected_sha
            or receipt["seed"] != seed
            or receipt["arm"] != "freeze_emb"
            or receipt["updates"] != 1_000
            or receipt["pca_sha256"] != PCA_SHA
            or receipt["fit_rows_sha256"] != FIT_SHA
            or receipt["held_rows_sha256"] != HELD_SHA
            or receipt["model_file_sha256"] != MODEL_HASHES
            or receipt.get("recovered_rank_updates", 0) != 0
        ):
            raise ValueError("soup checkpoint or training receipt differs")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in args.checkpoint]
    if any(checkpoint.get("seed") != seed or checkpoint.get("arm") != "freeze_emb" for seed, checkpoint in zip(SEEDS, checkpoints, strict=True)):
        raise ValueError("soup checkpoint metadata differs")
    frozen = tuple(
        key for key in checkpoints[0]["vision"]
        if "embeddings." in key
        or ((match := re.search(r"(?:^|\.)encoder\.layers\.(\d+)\.", key)) and int(match[1]) < 12)
    )
    if len(frozen) < 192 or any(
        not torch.equal(checkpoints[0]["vision"][key], checkpoint["vision"][key])
        for checkpoint in checkpoints[1:] for key in frozen
    ):
        raise ValueError("soup frozen coordinates differ")
    vision_state = average_model_states(tuple(checkpoint["vision"] for checkpoint in checkpoints))
    head_state = average_model_states(tuple(checkpoint["head"] for checkpoint in checkpoints))
    del checkpoints
    if any(not bool(torch.isfinite(value).all()) for value in (*vision_state.values(), *head_state.values())):
        raise ValueError("soup has nonfinite weights")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.model_snapshot, local_files_only=True, backend="torchvision")
    full = AutoModel.from_pretrained(args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16)
    vision = full.vision_model.float().cuda().eval()
    del full
    head = torch.nn.Linear(1024, 128).cuda().eval()
    vision.load_state_dict(vision_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    del vision_state, head_state
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = tuple(train[row].image_path for row in held)
    labels = tuple(train[row].label for row in held)
    classes = {label: index for index, label in enumerate(sorted(set(labels)))}
    label_ids = torch.tensor([classes[label] for label in labels])
    query, gallery = asymmetric_rows(paths, labels, args.dataset_root)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = export_all(vision, head, paths, tuple(range(len(held))), processor, workers=4, batch_size=64)
    torch.cuda.synchronize()
    export_seconds = time.perf_counter() - started
    packed = pack_int8_unit_embeddings(values)
    quality = score_packed_full_gallery(
        packed.codes.float(), packed.inverse_norms, label_ids, torch.arange(len(held)), device=torch.device("cuda")
    )
    codes = packed.codes.float().cuda()
    inverse = packed.inverse_norms.float().cuda()
    gpu_labels = label_ids.cuda()
    gallery_mask = torch.zeros(len(held), dtype=torch.bool, device="cuda")
    gallery_mask[gallery] = True
    asymmetric_hits: list[float] = []
    for block in torch.tensor(query, device="cuda").split(64):
        scores = (codes[block] @ codes.T) * inverse[block, None] * inverse[None, :]
        scores[:, ~gallery_mask] = -torch.inf
        winners = scores.argmax(dim=1)
        asymmetric_hits.extend((gpu_labels[winners] == gpu_labels[block]).float().cpu().tolist())
    old_r1 = np.stack([receipt["quality"]["per_query_r1"] for receipt in receipts]).mean(axis=0)
    paired = product_bootstrap(np.asarray(quality["per_query_r1"]) - old_r1, np.asarray(labels))
    asymmetric_r1 = float(np.mean(asymmetric_hits))
    gates = {
        "symmetric_best_plus_0_1pp": quality["recall_at_1"] >= max(receipt["quality"]["recall_at_1"] for receipt in receipts) + 0.001,
        "paired_lower_positive": paired["lower_95"] > 0,
        "mapr_no_regression": quality["map_at_r"] >= float(np.mean([receipt["quality"]["map_at_r"] for receipt in receipts])),
        "asymmetric_no_regression": asymmetric_r1 >= 6203 / 6354,
    }
    report = {
        "schema": "sfora-inshop-weight-soup-train-preflight-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": CHECKPOINT_SHAS,
        "receipt_sha256": [sha256(path) for path in args.receipt],
        "held_rows_sha256": HELD_SHA,
        "query_rows_sha256": QUERY_SHA,
        "gallery_rows_sha256": GALLERY_SHA,
        "frozen_tensors_verified": len(frozen),
        "quality": quality,
        "asymmetric_r1": asymmetric_r1,
        "paired_vs_three_seed_mean": paired,
        "gates": gates,
        "advance": all(gates.values()),
        "export_seconds": export_seconds,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    if sha256(Path(__file__)) != report["source_sha256"]:
        raise ValueError("soup source changed during run")
    payload = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"r1": quality["recall_at_1"], "mapr": quality["map_at_r"], "asymmetric_r1": asymmetric_r1, "advance": report["advance"]}), flush=True)


if __name__ == "__main__":
    main()
