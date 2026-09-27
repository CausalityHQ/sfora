#!/usr/bin/env python3
"""TRAIN-only public checkpoint/custom-gallery quality and latency gate."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import (
    GALLERY_SHA,
    NATIVE_SHA,
    PARTITION_SHA,
    QUERY_SHA,
    bootstrap_lower,
    roles,
    sha256,
)

from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
from sfora.unicom_inshop import parse_inshop_partition

CHECKPOINT_SHA = "ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089"
RECEIPT_SHA = "76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc"
BASELINE_SHA = "fc48eebc65e291a3dcb365d350af4fb0e8d48982b38ae1ae38d8f4138d814387"


def percentiles(values: list[int]) -> dict[str, float]:
    ms = np.asarray(values, dtype=np.float64) / 1e6
    return {f"p{p}_ms": float(np.percentile(ms, p)) for p in (50, 95, 99)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "training-receipt",
        "checkpoint",
        "native-library",
        "baseline-seed-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.training_receipt) != RECEIPT_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.native_library) != NATIVE_SHA
        or sha256(args.baseline_seed_receipt) != BASELINE_SHA
    ):
        raise ValueError("In-Shop public checkpoint gate authority differs")
    training = json.loads(args.training_receipt.read_text())
    baseline = json.loads(args.baseline_seed_receipt.read_text())
    if (
        training.get("schema") != "sfora-inshop-siglip2-unseen-gallery-train-v1"
        or training.get("checkpoint_sha256") != CHECKPOINT_SHA
        or training.get("seed") != 179026
        or training.get("arm") != "freeze_emb"
        or baseline.get("checkpoint_sha256") != CHECKPOINT_SHA
    ):
        raise ValueError("In-Shop public checkpoint provenance differs")
    original_hits = np.asarray(baseline["scores"]["base"]["asymmetric_hits"], dtype=np.int8)
    if original_hits.shape != (6_354,) or int(original_hits.sum()) != 6_203:
        raise ValueError("In-Shop public checkpoint baseline differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    paths = tuple(train[row].image_path for row in held)
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (len(query), len(gallery)) != (6_354, 6_245):
        raise ValueError("In-Shop public checkpoint role geometry differs")
    import hashlib

    if any(
        hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != digest
        for rows, digest in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
    ):
        raise ValueError("In-Shop public checkpoint role digest differs")
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    loaded_at = time.perf_counter()
    encoder = Siglip2CompactEncoder.from_checkpoint(
        model_snapshot=args.model_snapshot,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        model_file_sha256=training["model_file_sha256"],
        precision="fp16_native",
        device=torch.device("cuda:0"),
    )
    torch.cuda.synchronize()
    load_wall = time.perf_counter() - loaded_at
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with Siglip2CompactIndex.from_image_paths(
        encoder=encoder,
        native_library=args.native_library,
        image_paths=[paths[row] for row in gallery],
        expected_native_library_sha256=NATIVE_SHA,
    ) as index:
        torch.cuda.synchronize()
        build_wall = time.perf_counter() - started
        hits = []
        query_started = time.perf_counter()
        for begin in range(0, len(query), 32):
            rows = query[begin : begin + 32]
            with ExitStack() as stack:
                images = [stack.enter_context(Image.open(paths[row])) for row in rows]
                ordinals, scores = index.search_images(images)
            if (
                ordinals.shape != (len(rows), 10)
                or scores.shape != (len(rows), 10)
                or np.any(ordinals < 0)
                or np.any(ordinals >= len(gallery))
                or not np.isfinite(scores).all()
                or np.any(scores[:, 1:] > scores[:, :-1])
                or np.any((scores[:, 1:] == scores[:, :-1]) & (ordinals[:, 1:] < ordinals[:, :-1]))
            ):
                raise ValueError("In-Shop public top-10 order differs")
            hits.extend(
                int(labels[row] == labels[gallery[int(winner)]])
                for row, winner in zip(rows, ordinals[:, 0], strict=True)
            )
        torch.cuda.synchronize()
        query_wall = time.perf_counter() - query_started
        unique = []
        seen = set()
        for row in query:
            digest = sha256(paths[row])
            if digest not in seen:
                seen.add(digest)
                unique.append(paths[row])
            if len(unique) == 1_000:
                break
        if len(unique) != 1_000:
            raise ValueError("In-Shop public latency inventory differs")
        for path in unique[:32]:
            with Image.open(path) as image:
                index.search_images([image])
        torch.cuda.synchronize()
        latency_ns = []
        for path in unique:
            torch.cuda.synchronize()
            call_started = time.perf_counter_ns()
            with Image.open(path) as image:
                index.search_images([image])
            torch.cuda.synchronize()
            latency_ns.append(time.perf_counter_ns() - call_started)
        peak_cuda = torch.cuda.max_memory_allocated()
    public_hits = np.asarray(hits, dtype=np.int8)
    delta = public_hits.astype(np.float64) - original_hits.astype(np.float64)
    r1 = float(public_hits.mean())
    lower_pp = 100 * bootstrap_lower(delta, np.asarray(labels)[query])
    timing = percentiles(latency_ns)
    gates = {
        "quality_point": 100 * (r1 - float(original_hits.mean())) >= -0.15,
        "quality_lower": lower_pp > -0.25,
        "p50": timing["p50_ms"] <= 20,
        "p99": timing["p99_ms"] <= 30,
        "build": build_wall <= 180,
        "peak_cuda": peak_cuda < 3_000_000_000,
    }
    result = {
        "schema": "sfora-inshop-public-checkpoint-train-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "serving_source_sha256": sha256(
            Path(__import__("sfora.siglip2_compact_serving", fromlist=["x"]).__file__)
        ),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "training_receipt_sha256": RECEIPT_SHA,
        "baseline_seed_receipt_sha256": BASELINE_SHA,
        "native_library_sha256": NATIVE_SHA,
        "query_rows_sha256": QUERY_SHA,
        "gallery_rows_sha256": GALLERY_SHA,
        "quality": {
            "public_hits": hits,
            "public_r1": r1,
            "training_export_r1": float(original_hits.mean()),
            "delta_pp": 100 * float(delta.mean()),
            "bootstrap_lower_pp": lower_pp,
        },
        "resource": {
            "model_load_wall_seconds": load_wall,
            "gallery_build_wall_seconds": build_wall,
            "full_query_wall_seconds": query_wall,
            "full_query_images_per_second": len(query) / query_wall,
            "gallery_wire_bytes": 130 * len(gallery),
            "peak_cuda_allocated_bytes": peak_cuda,
        },
        "latency": {
            "unique_image_hashes": len(unique),
            "calls": len(latency_ns),
            "timing": timing,
            "raw_ns": latency_ns,
        },
        "gates": gates,
        "advance": all(gates.values()),
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {"r1": r1, "delta_pp": result["quality"]["delta_pp"], "timing": timing, "gates": gates}
        ),
        flush=True,
    )
    if not result["advance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
