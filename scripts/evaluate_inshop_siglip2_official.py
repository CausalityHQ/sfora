#!/usr/bin/env python3
"""Evaluate a frozen In-Shop SigLIP2 checkpoint on official query/gallery."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import train_inshop_siglip2_compact as train_source
import transformers
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from torch import nn
from train_inshop_siglip2_compact import FIT_SHA256, HELD_SHA256, SEEDS
from train_sop_siglip2_compact import NATIVE_SHA256, TILEIRAS_SHA256, export_all, sha256

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"


def loaded_helper_paths() -> tuple[Path, ...]:
    return tuple(
        Path(inspect.getfile(inspect.unwrap(helper))).resolve()
        for helper in (train_source, export_all, pack_int8_unit_embeddings, parse_inshop_partition)
    )


def verify_loaded_helper_sources(sources: dict[str, str]) -> None:
    for path in loaded_helper_paths():
        if sources.get(str(path)) != sha256(path):
            raise ValueError(f"In-Shop loaded helper source differs: {path}")


@torch.inference_mode()
def score_asymmetric(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_labels: tuple[str, ...],
    gallery_labels: tuple[str, ...],
    *,
    query_inverse: torch.Tensor | None = None,
    gallery_inverse: torch.Tensor | None = None,
) -> dict[str, object]:
    if (
        query.ndim != 2
        or gallery.ndim != 2
        or query.shape[1] != gallery.shape[1]
        or len(query) != len(query_labels)
        or len(gallery) != len(gallery_labels)
        or not query.is_floating_point()
        or not gallery.is_floating_point()
        or query.device != gallery.device
        or (query_inverse is None) != (gallery_inverse is None)
    ):
        raise ValueError("In-Shop query/gallery scorer geometry differs")
    counts = Counter(gallery_labels)
    relevant = [counts[label] for label in query_labels]
    if min(relevant) < 1:
        raise ValueError("In-Shop query has no gallery positive")
    labels = {name: index for index, name in enumerate(sorted(counts))}
    gallery_ids = torch.tensor([labels[name] for name in gallery_labels], device=query.device)
    query_ids = torch.tensor([labels[name] for name in query_labels], device=query.device)
    max_rank = min(len(gallery), max(10, max(relevant)))
    ranks = torch.arange(1, max_rank + 1, device=query.device)
    r1: list[float] = []
    ap_at_r: list[float] = []
    for start in range(0, len(query), 64):
        end = min(start + 64, len(query))
        scores = query[start:end] @ gallery.T
        if query_inverse is not None and gallery_inverse is not None:
            if query_inverse.shape != (len(query),) or gallery_inverse.shape != (len(gallery),):
                raise ValueError("In-Shop packed inverse norm geometry differs")
            scores = scores * query_inverse[start:end, None] * gallery_inverse[None, :]
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :max_rank]
        matches = gallery_ids[ranked] == query_ids[start:end, None]
        block_relevant = torch.tensor(relevant[start:end], device=query.device)
        precision = matches.cumsum(dim=1) / ranks[None, :]
        ap = (precision * matches * (ranks[None, :] <= block_relevant[:, None])).sum(
            dim=1
        ) / block_relevant
        r1.extend(float(value) for value in matches[:, 0].cpu().tolist())
        ap_at_r.extend(float(value) for value in ap.cpu().tolist())
    return {
        "recall_at_1": float(np.mean(r1)),
        "map_at_r": float(np.mean(ap_at_r)),
        "per_query_r1": r1,
        "per_query_ap": ap_at_r,
    }


@torch.inference_mode()
def verify_native(
    query: PackedInt8Embeddings, gallery: PackedInt8Embeddings, native_library: Path
) -> dict[str, object]:
    qcode = query.codes.float().cuda()
    qinv = query.inverse_norms.cuda()
    gcode = gallery.codes.float().cuda()
    ginv = gallery.inverse_norms.cuda()
    max_delta = 0.0
    with CutilePackedInt8Gallery.open_packed(native_library, gallery) as native:
        for start in range(0, len(qcode), 32):
            end = min(start + 32, len(qcode))
            block = type(query)(
                query.codes[start:end].contiguous(), query.inverse_norms[start:end].contiguous()
            )
            ordinals, native_scores = native.search_packed(block)
            scores = (qcode[start:end] @ gcode.T) * qinv[start:end, None] * ginv[None, :]
            expected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
            if not np.array_equal(ordinals, expected.cpu().numpy()):
                raise ValueError("In-Shop native top10 differs from packed oracle")
            selected = scores.gather(1, expected).cpu().numpy()
            delta = float(np.max(np.abs(native_scores - selected)))
            max_delta = max(max_delta, delta)
            if delta > 1e-5:
                raise ValueError("In-Shop native scores differ from packed oracle")
    return {"native_top10_exact": True, "native_top10_max_score_abs_delta": max_delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--arm", choices=("bank", "float"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    evaluator_source_sha = sha256(Path(__file__))
    training_receipt_path = args.training_dir / "receipt.json"
    checkpoint_path = args.training_dir / "checkpoint.pt"
    tileiras_path = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        args.output_dir.exists()
        or args.workers < 0
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
        or sha256(args.native_library) != NATIVE_SHA256
        or not tileiras_path
        or sha256(Path(tileiras_path)) != TILEIRAS_SHA256
    ):
        raise ValueError("In-Shop official evaluation authority differs")
    receipt = json.loads(training_receipt_path.read_text())
    decision = json.loads(args.decision.read_text())
    if (
        decision.get("schema") != "sfora-inshop-siglip2-paired-holdout-v1"
        or decision.get("selected_arm") != args.arm
        or tuple(decision.get("seeds", ())) != SEEDS
        or decision.get("arms", {}).get(str(args.seed), {}).get(args.arm, {}).get("receipt_sha256")
        != sha256(training_receipt_path)
        or receipt.get("schema") != "sfora-inshop-siglip2-compact-paired-train-v1"
        or receipt.get("seed") != args.seed
        or receipt.get("arm") != args.arm
        or receipt.get("updates") != 1000
        or receipt.get("fit_rows_sha256") != FIT_SHA256
        or receipt.get("held_rows_sha256") != HELD_SHA256
        or receipt.get("checkpoint_sha256") != sha256(checkpoint_path)
        or receipt.get("quality", {}).get("recall_at_1") is None
        or receipt.get("hardware", {}).get("torch") != torch.__version__
        or receipt.get("hardware", {}).get("transformers") != transformers.__version__
        or any(
            sha256(Path(name)) != digest for name, digest in receipt["source_files_sha256"].items()
        )
    ):
        raise ValueError("In-Shop official training receipt differs")
    verify_loaded_helper_sources(receipt["source_files_sha256"])
    records = parse_inshop_partition(args.dataset_root)
    queries = tuple(row for row in records if row.split == "query")
    gallery = tuple(row for row in records if row.split == "gallery")
    if len(queries) != 14_218 or len(gallery) != 12_612:
        raise ValueError("In-Shop official query/gallery inventory differs")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = nn.Linear(1024, 128).cuda().eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("seed") != args.seed
        or checkpoint.get("arm") != args.arm
        or checkpoint.get("updates") != 1000
    ):
        raise ValueError("In-Shop official checkpoint identity differs")
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    q_values = export_all(
        vision,
        head,
        tuple(row.image_path for row in queries),
        tuple(range(len(queries))),
        processor,
        workers=args.workers,
        batch_size=64,
    )
    g_values = export_all(
        vision,
        head,
        tuple(row.image_path for row in gallery),
        tuple(range(len(gallery))),
        processor,
        workers=args.workers,
        batch_size=64,
    )
    export_seconds = time.perf_counter() - started
    q_labels = tuple(row.label for row in queries)
    g_labels = tuple(row.label for row in gallery)
    float_quality = score_asymmetric(q_values.cuda(), g_values.cuda(), q_labels, g_labels)
    q_packed = pack_int8_unit_embeddings(q_values)
    g_packed = pack_int8_unit_embeddings(g_values)
    packed_quality = score_asymmetric(
        q_packed.codes.float().cuda(),
        g_packed.codes.float().cuda(),
        q_labels,
        g_labels,
        query_inverse=q_packed.inverse_norms.cuda(),
        gallery_inverse=g_packed.inverse_norms.cuda(),
    )
    native = verify_native(q_packed, g_packed, args.native_library)
    if sha256(Path(__file__)) != evaluator_source_sha:
        raise ValueError("In-Shop official evaluator source changed during execution")
    result = {
        "schema": "sfora-inshop-siglip2-official-query-gallery-v1",
        "claim_eligible": False,
        "status": (
            "exploratory official In-Shop query/gallery read after train-only paired selection"
        ),
        "split": "official In-Shop asymmetric query/gallery",
        "seed": args.seed,
        "arm": args.arm,
        "queries": len(queries),
        "gallery_images": len(gallery),
        "gallery_wire_bytes_per_row": 130,
        "source_sha256": evaluator_source_sha,
        "partition_sha256": PARTITION_SHA256,
        "training_receipt_sha256": sha256(training_receipt_path),
        "decision_sha256": sha256(args.decision),
        "checkpoint_sha256": sha256(checkpoint_path),
        "native_library_sha256": NATIVE_SHA256,
        "tileiras_sha256": TILEIRAS_SHA256,
        "float_quality": float_quality,
        "packed_quality": packed_quality,
        **native,
        "export_seconds": export_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {"seed": args.seed, "arm": args.arm, "packed_r1": packed_quality["recall_at_1"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
