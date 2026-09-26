#!/usr/bin/env python3
"""Score every SOP TRAIN holdout image through the public packed query path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.siglip2_compact_serving import Siglip2CompactIndex

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed", type=int, choices=(179024, 179026, 179027), required=True)
    parser.add_argument("--arm", choices=("control", "freeze"), required=True)
    parser.add_argument("--precision", choices=("fp32_autocast", "fp16_native"), required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.native_library) != NATIVE_SHA
    ):
        raise ValueError("SOP public TRAIN source or scorer differs")
    decision = json.loads(args.decision.read_text())
    receipt_path = args.training_run / "receipt.json"
    receipt_sha = sha256(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (
        decision.get("schema") != "sfora-sop-true-freeze-paired-train-only-v1"
        or decision.get("seed") != args.seed
        or decision.get("arms", {}).get(args.arm, {}).get("receipt_sha256") != receipt_sha
        or receipt.get("seed") != args.seed
        or receipt.get("arm") != "float_rank_member_bank"
        or receipt.get("freeze_lower_stack") is not (args.arm == "freeze")
    ):
        raise ValueError("SOP public TRAIN checkpoint decision differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
    if labels.shape != (59_551,) or ids.shape != labels.shape or relatives.shape != labels.shape:
        raise ValueError("SOP public TRAIN inventory differs")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    held = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if (
        len(held) != 5_851
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != receipt["query_image_ids_sha256"]
    ):
        raise ValueError("SOP public TRAIN holdout differs")
    image_digest = hashlib.sha256()
    r1: list[float] = []
    ap: list[float] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with Siglip2CompactIndex.from_artifacts(
        model_snapshot=args.model_snapshot,
        training_receipt=receipt_path,
        training_checkpoint=args.training_run / "checkpoint.pt",
        train_embeddings=args.training_run / "train_embeddings.npy",
        native_library=args.native_library,
        expected_receipt_sha256=receipt_sha,
        precision=args.precision,
    ) as index:
        assert index.encoder is not None and index.gallery is not None
        values = np.load(
            args.training_run / "train_embeddings.npy", mmap_mode="r", allow_pickle=False
        )
        gallery = pack_int8_unit_embeddings(torch.from_numpy(np.asarray(values).copy()))
        gallery_code = gallery.codes.float().cuda()
        gallery_inverse = gallery.inverse_norms.float().cuda()
        gallery_labels = torch.from_numpy(labels.copy()).cuda()
        counts = torch.bincount(gallery_labels)
        max_relevant = int((counts[gallery_labels[torch.from_numpy(held.copy()).cuda()]] - 1).max())
        ranks = torch.arange(1, max_relevant + 1, device="cuda")
        for start in range(0, len(held), 32):
            rows = held[start : start + 32]
            images = []
            for row in rows:
                relative = PurePosixPath(str(relatives[row]))
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise ValueError("SOP public TRAIN image path differs")
                path = args.dataset_root.joinpath(*relative.parts)
                if not path.is_file() or path.is_symlink():
                    raise ValueError("SOP public TRAIN image missing")
                raw = path.read_bytes()
                image_digest.update(hashlib.sha256(raw).digest())
                with Image.open(BytesIO(raw)) as image:
                    images.append(image.convert("RGB"))
            packed = index.encoder.encode_images(images)
            native_ordinals, _ = index.gallery.search_packed(packed)
            query_code = packed.codes.float().cuda()
            query_inverse = packed.inverse_norms.float().cuda()
            scores = (
                (query_code @ gallery_code.T) * query_inverse[:, None] * gallery_inverse[None, :]
            )
            row_tensor = torch.from_numpy(rows.copy()).cuda()
            scores[torch.arange(len(rows), device="cuda"), row_tensor] = -torch.inf
            ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :max_relevant]
            # Native top-10 includes self; compare against the unmasked oracle separately.
            native_scores = (
                (query_code @ gallery_code.T) * query_inverse[:, None] * gallery_inverse[None, :]
            )
            native_expected = torch.argsort(native_scores, dim=1, descending=True, stable=True)[
                :, :10
            ]
            if not np.array_equal(native_ordinals, native_expected.cpu().numpy()):
                raise ValueError("SOP public TRAIN native top-10 differs")
            relevant = counts[gallery_labels[row_tensor]] - 1
            matches = gallery_labels[ranked] == gallery_labels[row_tensor, None]
            precision = matches.cumsum(dim=1) / ranks[None, :]
            block_ap = (precision * matches * (ranks[None, :] <= relevant[:, None])).sum(
                dim=1
            ) / relevant
            r1.extend(float(value) for value in matches[:, 0].cpu().tolist())
            ap.extend(float(value) for value in block_ap.cpu().tolist())
            if start % 1024 == 0:
                print(json.dumps({"queries_done": start + len(rows)}), flush=True)
    result = {
        "schema": "sfora-sop-true-freeze-public-train-v1",
        "claim_eligible": False,
        "seed": args.seed,
        "arm": args.arm,
        "precision": args.precision,
        "queries": len(held),
        "gallery_images": len(labels),
        "training_receipt_sha256": receipt_sha,
        "source_archive_sha256": ARCHIVE_SHA,
        "source_sha256": sha256(Path(__file__)),
        "query_image_manifest_sha256": image_digest.hexdigest(),
        "native_top10_exact": True,
        "packed_r1": float(np.mean(r1)),
        "packed_map_at_r": float(np.mean(ap)),
        "per_query_r1": r1,
        "per_query_ap": ap,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "seed": args.seed,
                "arm": args.arm,
                "precision": args.precision,
                "packed_r1": result["packed_r1"],
                "packed_map_at_r": result["packed_map_at_r"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
