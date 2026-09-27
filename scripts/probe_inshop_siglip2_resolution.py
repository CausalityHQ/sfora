#!/usr/bin/env python3
"""TRAIN-only packed-quality feasibility screen for 192-pixel SigLIP2 inference."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, sha256, split
from train_sop_siglip2_compact import export_all, score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

SEED = 179026
PREFLIGHT_SHA = "4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254"
TRAINER_SHA = "76e20c328df6e632b387486e4761fb26994b74eff4c0a944c23400a40c4985cc"


class ResizedVision(torch.nn.Module):
    def __init__(self, vision: torch.nn.Module) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, **batch: torch.Tensor) -> object:
        return self.vision(**batch, interpolate_pos_encoding=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "checkpoint",
        "baseline-receipt",
        "preflight",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.baseline_receipt.read_text())
    preflight = json.loads(args.preflight.read_text())
    if (
        args.output.exists()
        or sha256(args.preflight) != PREFLIGHT_SHA
        or sha256(args.checkpoint) != receipt.get("checkpoint_sha256")
        or receipt.get("seed") != SEED
        or receipt.get("arm") != "freeze_emb"
        or receipt.get("source_sha256") != TRAINER_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
        or not torch.cuda.is_available()
    ):
        raise ValueError("In-Shop resolution probe authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    if (
        preflight["fit_sha256"] != digest_rows(fit)
        or preflight["held_sha256"] != digest_rows(held)
        or receipt["held_rows_sha256"] != digest_rows(held)
        or len(held) != 12_599
    ):
        raise ValueError("In-Shop resolution probe held rows differ")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    if processor.size != {"height": 256, "width": 256}:
        raise ValueError("In-Shop resolution probe processor differs")
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = torch.nn.Linear(1024, 128).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != SEED or checkpoint.get("arm") != "freeze_emb":
        raise ValueError("In-Shop resolution probe checkpoint identity differs")
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = tuple(train[index].image_path for index in held)
    labels = tuple(train[index].label for index in held)
    classes = {label: index for index, label in enumerate(sorted(set(labels)))}
    label_ids = torch.tensor([classes[label] for label in labels])
    quality = {}
    export_wall = {}
    for size in (256, 192):
        processor.size = {"height": size, "width": size}
        started = time.perf_counter()
        values = export_all(
            vision if size == 256 else ResizedVision(vision),
            head,
            paths,
            tuple(range(len(held))),
            processor,
            workers=4,
            batch_size=64,
        )
        export_wall[str(size)] = time.perf_counter() - started
        packed = pack_int8_unit_embeddings(values)
        quality[str(size)] = score_packed_full_gallery(
            packed.codes.float(),
            packed.inverse_norms,
            label_ids,
            torch.arange(len(held)),
            device=torch.device("cuda"),
        )
        if size == 256 and quality["256"]["per_query_r1"] != receipt["quality"]["per_query_r1"]:
            raise ValueError("In-Shop resolution probe baseline packed R@1 differs")
        print(json.dumps({"size": size, "r1": quality[str(size)]["recall_at_1"]}), flush=True)
    report = {
        "schema": "sfora-inshop-train-only-resolution-probe-v1",
        "claim_eligible": False,
        "seed": SEED,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "baseline_receipt_sha256": sha256(args.baseline_receipt),
        "preflight_sha256": PREFLIGHT_SHA,
        "probe_source_sha256": sha256(Path(__file__)),
        "held_queries_and_gallery": len(held),
        "quality": quality,
        "export_wall_seconds": export_wall,
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
