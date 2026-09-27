#!/usr/bin/env python3
"""TRAIN-only packed-quality preflight for 320-pixel SigLIP2 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
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
RECEIPT_SHA = "76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"


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
        or sha256(args.baseline_receipt) != RECEIPT_SHA
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
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    query: list[int] = []
    gallery: list[int] = []
    for label in sorted(grouped):
        rows = sorted(
            grouped[label],
            key=lambda index: hashlib.sha256(
                str(paths[index].relative_to(args.dataset_root)).encode()
            ).digest(),
        )
        count = max(1, min(len(rows) - 1, round(len(rows) / 2)))
        gallery.extend(rows[:count])
        query.extend(rows[count:])
    query.sort()
    gallery.sort()
    if (
        len(query) != 6_354
        or len(gallery) != 6_245
        or hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != QUERY_SHA
        or hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest() != GALLERY_SHA
    ):
        raise ValueError("In-Shop 320 asymmetric roles differ")
    classes = {label: index for index, label in enumerate(sorted(set(labels)))}
    label_ids = torch.tensor([classes[label] for label in labels])
    quality = {}
    asymmetric = {}
    export_wall = {}
    for size in (256, 320):
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
        codes = packed.codes.float().cuda()
        inverse = packed.inverse_norms.float().cuda()
        gallery_rows = torch.tensor(gallery, device="cuda")
        held_labels = label_ids.cuda()
        hits: list[bool] = []
        for block in torch.tensor(query, device="cuda").split(64):
            scores = (
                (codes[block] @ codes[gallery_rows].T)
                * inverse[block, None]
                * inverse[gallery_rows][None, :]
            )
            top = torch.argsort(scores, dim=1, descending=True, stable=True)[:, 0]
            hits.extend((held_labels[gallery_rows[top]] == held_labels[block]).cpu().tolist())
        asymmetric[str(size)] = {"hits": sum(hits), "queries": len(hits), "per_query_r1": hits}
        if size == 256 and (
            quality["256"]["per_query_r1"] != receipt["quality"]["per_query_r1"]
            or abs(quality["256"]["map_at_r"] - receipt["quality"]["map_at_r"]) > 1e-8
            or sum(hits) != 6_203
        ):
            raise ValueError("In-Shop 320 probe baseline packed parity differs")
        print(
            json.dumps(
                {
                    "size": size,
                    "symmetric_r1": quality[str(size)]["recall_at_1"],
                    "asymmetric_hits": sum(hits),
                }
            ),
            flush=True,
        )
    report = {
        "schema": "sfora-inshop-train-only-resolution-320-preflight-v1",
        "claim_eligible": False,
        "seed": SEED,
        "candidate_size": 320,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "baseline_receipt_sha256": sha256(args.baseline_receipt),
        "preflight_sha256": PREFLIGHT_SHA,
        "probe_source_sha256": sha256(Path(__file__)),
        "held_queries_and_gallery": len(held),
        "quality": quality,
        "asymmetric": asymmetric,
        "quality_gate_passed": asymmetric["320"]["hits"] >= 6_172,
        "query_rows_sha256": QUERY_SHA,
        "gallery_rows_sha256": GALLERY_SHA,
        "export_wall_seconds": export_wall,
    }
    if sha256(Path(__file__)) != report["probe_source_sha256"]:
        raise ValueError("In-Shop 320 probe source changed during execution")
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
