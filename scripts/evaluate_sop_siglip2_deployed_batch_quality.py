#!/usr/bin/env python3
"""Score serving batch-1/32 queries against the frozen SOP batch-64 gallery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from evaluate_sop_siglip2_official import VerifiedRows, export_verified, sha256
from torch import nn
from train_sop_siglip2_compact import paths_from_archive

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
MANIFEST_SHA256 = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
DECISION_SHA256 = "ccfdb7055643f2a16124ecab62b30d31a2b025f380319e950cb4a481066f1852"
OFFICIAL_SHA256 = "86b5840728218364ccf1584965f019c462f224dc491deb6172e16aa8bf252665"
NATIVE_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"


@torch.inference_mode()
def packed_cross_r1(query, gallery, labels: np.ndarray, native_library: Path) -> dict:
    if query.codes.shape != gallery.codes.shape or query.codes.shape != (60_502, 128):
        raise ValueError("deployed batch query/gallery geometry differs")
    device = torch.device("cuda")
    codes = gallery.codes.float().to(device)
    query_codes = query.codes.float().to(device)
    inverse = gallery.inverse_norms.float().to(device)
    query_inverse = query.inverse_norms.float().to(device)
    label_gpu = torch.from_numpy(labels.copy()).to(device)
    nearest = []
    correct = []
    for start in range(0, len(labels), 64):
        stop = min(start + 64, len(labels))
        scores = (
            (query_codes[start:stop] @ codes.T) * query_inverse[start:stop, None] * inverse[None, :]
        )
        scores[
            torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)
        ] = -torch.inf
        winners = torch.argmax(scores, dim=1)
        nearest.extend(int(x) for x in winners.cpu().tolist())
        correct.extend(int(x) for x in (label_gpu[winners] == label_gpu[start:stop]).cpu().tolist())
    with CutilePackedInt8Gallery.open_packed(native_library, gallery) as live:
        ordinals, scores = live.search_packed(
            type(query)(query.codes[:32].contiguous(), query.inverse_norms[:32].contiguous())
        )
        oracle = (query_codes[:32] @ codes.T) * query_inverse[:32, None] * inverse[None, :]
        expected = torch.argsort(oracle, dim=1, descending=True, stable=True)[:, :10].cpu().numpy()
        if not np.array_equal(ordinals, expected) or not np.isfinite(scores).all():
            raise ValueError("deployed batch native top-10 differs from packed oracle")
        nonself = [
            next(int(x) for x in row if int(x) != index) for index, row in enumerate(ordinals)
        ]
        if nonself != nearest[:32]:
            raise ValueError("deployed batch native nonself top-1 differs")
    return {
        "recall_at_1": sum(correct) / len(correct),
        "per_query_r1": correct,
        "nearest_ordinals_sha256": hashlib.sha256(
            np.asarray(nearest, dtype="<i4").tobytes()
        ).hexdigest(),
        "native_first32_top10_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "archive",
        "manifest",
        "dataset-root",
        "decision",
        "training-receipt",
        "checkpoint",
        "training-source-root",
        "official-receipt",
        "official-embeddings",
        "model-snapshot",
        "native-library",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists() or args.workers != 2 or not torch.cuda.is_available():
        raise ValueError("deployed batch evaluation invocation differs")
    for path, digest in (
        (args.archive, ARCHIVE_SHA256),
        (args.manifest, MANIFEST_SHA256),
        (args.decision, DECISION_SHA256),
        (args.official_receipt, OFFICIAL_SHA256),
        (args.native_library, NATIVE_SHA256),
    ):
        if sha256(path) != digest:
            raise ValueError(f"deployed batch source differs: {path}")
    decision = json.loads(args.decision.read_text())
    train = json.loads(args.training_receipt.read_text())
    official = json.loads(args.official_receipt.read_text())
    if (
        decision.get("replication_gate_pass") is not True
        or sha256(args.training_receipt) != decision["arms"]["179023"]["bank"]["receipt_sha256"]
        or train.get("seed") != 179023
        or train.get("arm") != "float_rank_member_bank"
        or sha256(args.checkpoint) != train.get("checkpoint_sha256")
        or official.get("training_receipt_sha256") != sha256(args.training_receipt)
        or sha256(args.official_embeddings) != official.get("test_embeddings_sha256")
        or official.get("native_top10_exact") is not True
        or any(
            sha256(args.training_source_root / name) != digest
            for name, digest in train["source_files_sha256"].items()
        )
        or any(
            sha256(args.model_snapshot / name) != digest
            for name, digest in train["model_file_sha256"].items()
        )
    ):
        raise ValueError("deployed batch training authority differs")
    with np.load(args.archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["test_labels"], dtype=np.int64)
        ids = np.asarray(archive["test_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["test_relative_paths"])
    if labels.shape != (60_502,) or ids.shape != labels.shape or len(np.unique(labels)) != 11_316:
        raise ValueError("deployed batch TEST inventory differs")
    paths = paths_from_archive(args.dataset_root, relatives)
    manifest = args.manifest.read_bytes()
    if len(manifest) != 32 * len(paths):
        raise ValueError("deployed batch image manifest differs")
    dataset = VerifiedRows(paths, tuple(map(int, labels)), manifest)
    gallery_values = np.load(args.official_embeddings, mmap_mode="r")
    if gallery_values.shape != (60_502, 128):
        raise ValueError("deployed batch gallery differs")
    gallery = pack_int8_unit_embeddings(torch.from_numpy(np.asarray(gallery_values).copy()))
    import transformers

    processor = transformers.AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = transformers.AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = nn.Linear(1024, 128).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != 179023 or checkpoint.get("arm") != "float_rank_member_bank":
        raise ValueError("deployed batch checkpoint identity differs")
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original = np.asarray(official["packed_quality"]["per_query_r1"], dtype=np.float64)
    if (
        original.shape != labels.shape
        or abs(float(original.mean()) - official["packed_quality"]["recall_at_1"]) > 1e-6
    ):
        raise ValueError("deployed batch reference quality differs")
    rows = {}
    for batch_size in (1, 32):
        started = time.perf_counter()
        query_values = export_verified(
            dataset, processor, vision, head, workers=args.workers, batch_size=batch_size
        )
        export_seconds = time.perf_counter() - started
        output = args.output_dir / f"query_batch{batch_size}.npy"
        np.save(output, query_values.numpy(), allow_pickle=False)
        quality = packed_cross_r1(
            pack_int8_unit_embeddings(query_values), gallery, labels, args.native_library
        )
        delta = np.asarray(quality["per_query_r1"], dtype=np.float64) - original
        rows[str(batch_size)] = {
            "query_embeddings_sha256": sha256(output),
            "export_seconds": export_seconds,
            "packed_recall_at_1": quality["recall_at_1"],
            "packed_per_query_r1": quality["per_query_r1"],
            "native_first32_top10_exact": quality["native_first32_top10_exact"],
            "nearest_ordinals_sha256": quality["nearest_ordinals_sha256"],
            "delta_vs_batch64": float(delta.mean()),
            "paired_product_bootstrap_delta": product_bootstrap(delta, labels),
        }
        print(
            json.dumps(
                {
                    "batch": batch_size,
                    "r1": quality["recall_at_1"],
                    "export_seconds": export_seconds,
                }
            ),
            flush=True,
        )
    result = {
        "schema": "sfora-sop-siglip2-deployed-batch-quality-v1",
        "claim_eligible": False,
        "split": "already-observed SOP official TEST queries; fixed batch64 gallery; self excluded",
        "seed": 179023,
        "query_count": len(labels),
        "gallery_rows": len(labels),
        "batch64_recall_at_1": float(original.mean()),
        "rows": rows,
        "source_sha256": sha256(Path(__file__)),
        "export_source_sha256": sha256(
            Path(__file__).with_name("evaluate_sop_siglip2_official.py")
        ),
        "inputs": {
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "decision_sha256": DECISION_SHA256,
            "training_receipt_sha256": sha256(args.training_receipt),
            "checkpoint_sha256": sha256(args.checkpoint),
            "official_receipt_sha256": OFFICIAL_SHA256,
            "official_embeddings_sha256": sha256(args.official_embeddings),
            "native_library_sha256": NATIVE_SHA256,
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu": torch.cuda.get_device_name(),
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
