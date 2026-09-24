#!/usr/bin/env python3
"""Exploratory CUB class-disjoint transfer of matched SOP TRAIN tail controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from evaluate_sop_cub_transfer import VerifiedCubImages, encode_cub
from export_unicom_cub_embeddings import (
    CUB_ARCHIVE_SHA256,
    ordered_record_sha256,
    parse_cub_records,
    verify_extracted_cub_matches_archive,
)
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.utils.data import DataLoader
from train_sop_compact_backbone import CHECKPOINT_SHA256

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_evaluation import score_symmetric

ORIGINAL_RECEIPT_SHA256 = "331b8055b970704b1269b3229e0b2114f594033bfe1690c9af736b4631b46aa1"
CANDIDATE_RECEIPT_SHA256 = "cb3b03d814f08e0acf9e992f1569bc099566d77409b6895cdb2d3ba763e354cb"
CHECKPOINTS = {
    "head_arcface": "9eaf0cdbd757f1f83b00e5415349b4608a62769a56ace81983d67ea2b87781fd",
    "tail_arcface_eval": "9c07e1b0c0b171a9f72cafdc2c9a7b4d21248d28de5a4510180fb699b8300918",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--cub-archive", type=Path, required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--original-receipt", type=Path, required=True)
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--tail-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-sop-tail-cub-transfer", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or args.workers < 0:
        raise ValueError("SOP tail CUB output/workers differ")
    started = time.perf_counter()
    paths = {
        "original_receipt": (args.original_receipt, ORIGINAL_RECEIPT_SHA256),
        "candidate_receipt": (args.candidate_receipt, CANDIDATE_RECEIPT_SHA256),
        "source_checkpoint": (args.source_checkpoint, CHECKPOINT_SHA256),
        "cub_archive": (args.cub_archive, CUB_ARCHIVE_SHA256),
        "head_checkpoint": (args.head_checkpoint, CHECKPOINTS["head_arcface"]),
        "tail_checkpoint": (args.tail_checkpoint, CHECKPOINTS["tail_arcface_eval"]),
    }
    if any(sha256(path) != expected for path, expected in paths.values()):
        raise ValueError("SOP tail CUB source digest differs")
    original = json.loads(args.original_receipt.read_text())
    candidate = json.loads(args.candidate_receipt.read_text())
    if (
        original["schema"] != "sfora-sop-cached-teacher-tail-screen-v1"
        or candidate["schema"] != "sfora-sop-tail-eval-control-screen-v1"
        or original["results"]["head_arcface"]["checkpoint_sha256"] != CHECKPOINTS["head_arcface"]
        or candidate["results"]["tail_arcface_eval"]["checkpoint_sha256"] != CHECKPOINTS["tail_arcface_eval"]
    ):
        raise ValueError("SOP tail CUB training receipt differs")
    records = parse_cub_records(args.cub_root)
    content_sha256 = verify_extracted_cub_matches_archive(
        args.cub_archive, args.cub_root, records
    )
    selected = tuple(record for record in records if record.split == "test")
    if len(selected) != 5_924:
        raise ValueError("SOP tail CUB test inventory differs")
    manifest = b"".join(
        hashlib.sha256(record.image_path.read_bytes()).digest() for record in selected
    )
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "rows": len(selected)}), flush=True)
        return
    if not torch.cuda.is_available():
        raise ValueError("SOP tail CUB CUDA unavailable")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = authenticated.encoder.cuda().eval()
    loader = DataLoader(
        VerifiedCubImages(selected, authenticated.transform, manifest),
        batch_size=64,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    labels = torch.tensor([record.label for record in selected], dtype=torch.int64, device="cuda")
    results = {}
    for name, path in (("head_arcface", args.head_checkpoint), ("tail_arcface_eval", args.tail_checkpoint)):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("arm") != name or state.get("source_checkpoint_sha256") != CHECKPOINT_SHA256:
            raise ValueError("SOP tail CUB checkpoint metadata differs")
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        head = nn.Linear(768, 128)
        head.load_state_dict(state["head"], strict=True)
        head = head.cuda().eval()
        values, encode_seconds = encode_cub(model, head, loader)
        packed = pack_int8_unit_embeddings(values)
        score_started = time.perf_counter()
        scores = {
            "float": score_symmetric(values.cuda(), labels),
            "packed": score_symmetric(
                packed.codes.float().cuda(), labels,
                inverse_norms=packed.inverse_norms.cuda(),
            ),
        }
        results[name] = {
            "encode_seconds": encode_seconds,
            "score_seconds": time.perf_counter() - score_started,
            "features_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
            **scores,
        }
        print(json.dumps({"arm": name, "packed_r1": scores["packed"]["recall_at_1"]}), flush=True)
    receipt = {
        "schema": "sfora-sop-tail-cub-transfer-v1",
        "claim_eligible": False,
        "protocol": "CUB-200-2011 classes 101-200 test self retrieval; previously inspected transfer panel; no CUB fitting",
        "rows": len(selected),
        "labels": [record.label for record in selected],
        "image_ids": [record.image_id for record in selected],
        "gallery_bytes_per_item": 130,
        "source_sha256": {key: expected for key, (_path, expected) in paths.items()},
        "cub_content_sha256": content_sha256,
        "cub_ordered_records_sha256": ordered_record_sha256(records),
        "selected_image_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "script_sha256": sha256(Path(__file__)),
        "results": results,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"receipt": str(args.output), "seconds": receipt["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
