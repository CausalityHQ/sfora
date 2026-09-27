#!/usr/bin/env python3
"""Certify paired In-Shop TRAIN public image-to-top-10 latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import paired_latency_certification as latency
import torch
from certify_inshop_siglip2_direct_parity import CHECKPOINT_SHA, RECEIPT_SHA
from certify_sop_siglip2_serving_threads_p99 import (
    decode,
    digest_result,
    sha256,
    summary,
    timed_call,
)
from preflight_inshop_siglip2_unseen_gallery import split
from score_inshop_crop_view_pair import GALLERY_SHA, NATIVE_SHA, PARTITION_SHA, QUERY_SHA, roles

from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
from sfora.unicom_inshop import parse_inshop_partition

PARITY_SHA = "09a426dd90911afe271e61eb6b026dd5fbbee98ed02ef48f110beffb54f9055d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "training-receipt",
        "checkpoint",
        "native-library",
        "parity-receipt",
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
        or sha256(args.parity_receipt) != PARITY_SHA
        or torch.get_num_threads() != 20
    ):
        raise ValueError("In-Shop direct p99 authority differs")
    parity = json.loads(args.parity_receipt.read_text())
    if parity.get("exact_packed_and_top10") is not True or parity.get("gallery_rows") != 6_245:
        raise ValueError("In-Shop direct p99 parity authority differs")
    training = json.loads(args.training_receipt.read_text())
    if training.get("seed") != 179026 or training.get("arm") != "freeze_emb":
        raise ValueError("In-Shop direct p99 checkpoint differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    _, held = split(tuple(row.label for row in train))
    paths = tuple(train[row].image_path for row in held)
    labels = tuple(train[row].label for row in held)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (len(query), len(gallery)) != (6_354, 6_245):
        raise ValueError("In-Shop direct p99 role geometry differs")
    for rows, digest in ((query, QUERY_SHA), (gallery, GALLERY_SHA)):
        if hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != digest:
            raise ValueError("In-Shop direct p99 role digest differs")
    selected = np.linspace(0, len(query) - 1, 640, dtype=np.int64)
    query_rows = [int(query[row]) for row in selected]
    query_paths = [paths[row] for row in query_rows]
    if (
        query_rows != parity["query_rows"]
        or [sha256(path) for path in query_paths] != parity["query_image_sha256"]
    ):
        raise ValueError("In-Shop direct p99 query bytes differ")

    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    encoder = Siglip2CompactEncoder.from_checkpoint(
        model_snapshot=args.model_snapshot,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        model_file_sha256=training["model_file_sha256"],
        precision="fp16_native",
        device=torch.device("cuda:0"),
    )
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - started
    if not encoder._use_direct_preprocess:
        raise ValueError("In-Shop direct p99 processor was not selected")
    started = time.perf_counter()
    raw = {label: [] for label in ("baseline", "direct")}
    expected = {}
    with Siglip2CompactIndex.from_image_paths(
        encoder=encoder,
        native_library=args.native_library,
        image_paths=[paths[row] for row in gallery],
        expected_native_library_sha256=NATIVE_SHA,
    ) as index:
        torch.cuda.synchronize()
        gallery_build_s = time.perf_counter() - started
        encoder._use_direct_preprocess = False
        for block in range(20):
            for position in range(32):
                image_paths = query_paths[block * 32 + position : block * 32 + position + 1]
                result = index.search_images(decode(image_paths))
                expected[block, position] = digest_result(result)
        torch.cuda.reset_peak_memory_stats()
        for block in range(20):
            block_paths = query_paths[block * 32 : (block + 1) * 32]
            order = (
                ("baseline", "direct", "direct", "baseline")
                if block % 2 == 0
                else ("direct", "baseline", "baseline", "direct")
            )
            samples = {label: [] for label in raw}
            for label in order:
                encoder._use_direct_preprocess = label == "direct"
                for position in range(5):
                    timed_call(
                        index, block_paths[position : position + 1], expected[block, position]
                    )
                for position in range(250):
                    actual = position % 32
                    samples[label].append(
                        timed_call(index, block_paths[actual : actual + 1], expected[block, actual])
                    )
            for label in raw:
                raw[label].append(samples[label])
            print(json.dumps({"batch": 1, "blocks_done": block + 1}), flush=True)
        certification = latency.block_bootstrap_p99_ratio(raw["direct"], raw["baseline"])
        statistics = {label: summary(raw[label], 1) for label in raw}
        passed = (
            certification["point_p50_ratio"] <= 0.95
            and certification["point_ratio"] <= 1.0
            and certification["ci95_upper"] <= 1.05
        )
        batch32 = {label: [] for label in raw}
        for block in range(20):
            block_paths = query_paths[block * 32 : (block + 1) * 32]
            encoder._use_direct_preprocess = False
            expected_digest = digest_result(index.search_images(decode(block_paths)))
            for label in ("baseline", "direct") if block % 2 == 0 else ("direct", "baseline"):
                encoder._use_direct_preprocess = label == "direct"
                batch32[label].append(
                    [timed_call(index, block_paths, expected_digest) for _ in range(10)]
                )
        batch32_stats = {label: summary(batch32[label], 32) for label in batch32}
        passed = (
            passed
            and batch32_stats["direct"]["p95_ms"] <= 1.10 * batch32_stats["baseline"]["p95_ms"]
        )
        report = {
            "schema": "sfora-inshop-direct-public-paired-v1",
            "source_sha256": sha256(Path(__file__)),
            "serving_source_sha256": sha256(
                Path(__import__("sfora.siglip2_compact_serving", fromlist=["x"]).__file__)
            ),
            "helper_source_sha256": sha256(Path(latency.__file__)),
            "parity_receipt_sha256": PARITY_SHA,
            "training_receipt_sha256": RECEIPT_SHA,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "native_library_sha256": NATIVE_SHA,
            "query_roles_sha256": QUERY_SHA,
            "gallery_roles_sha256": GALLERY_SHA,
            "model_load_wall_seconds": model_load_s,
            "gallery_build_wall_seconds": gallery_build_s,
            "gallery_wire_bytes": 6_245 * 130,
            "batch1_rule": "p50 ratio <= .95, p99 ratio <= 1, paired bootstrap upper95 <= 1.05",
            "batch32_rule": "p95 direct <= 1.10 baseline",
            "public_gate_passed": passed,
            "gpu": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "statistics": statistics,
            "certification": certification,
            "batch32_statistics": batch32_stats,
            "raw_ns": raw,
            "batch32_raw_ns": batch32,
        }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "passed": passed,
                "statistics": statistics,
                "certification": certification,
                "batch32_statistics": batch32_stats,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
