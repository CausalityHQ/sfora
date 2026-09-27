#!/usr/bin/env python3
"""Paired exact public image-to-top-10 gate for the pinned direct processor."""

from __future__ import annotations

import argparse
import json
import os
import resource
from pathlib import Path, PurePosixPath

import numpy as np
import paired_latency_certification as latency
import torch
from certify_sop_siglip2_serving_threads_p99 import (
    ARCHIVE_SHA,
    DECISION_SHA,
    NATIVE_SHA,
    decode,
    digest_result,
    sha256,
    summary,
    timed_call,
)

import sfora.siglip2_compact_serving as serving
from sfora.siglip2_compact_serving import Siglip2CompactIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "decision",
        "runs",
        "source-archive",
        "dataset-root",
        "model-snapshot",
        "native-library",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.decision) != DECISION_SHA
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.native_library) != NATIVE_SHA
        or not torch.cuda.is_available()
        or torch.get_num_threads() != 20
    ):
        raise ValueError("SOP direct public authority differs")
    decision = json.loads(args.decision.read_text())
    run = args.runs / "sfora-sop-true-freeze-freeze-179024-1000-v1"
    receipt_path = run / "receipt.json"
    receipt_sha = sha256(receipt_path)
    if (
        decision.get("schema") != "sfora-sop-true-freeze-paired-train-only-v1"
        or decision.get("seed") != 179024
        or decision["arms"]["freeze"]["receipt_sha256"] != receipt_sha
    ):
        raise ValueError("SOP direct public checkpoint differs")
    with np.load(args.source_archive, allow_pickle=False) as source:
        ids = np.asarray(source["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(source["train_relative_paths"]).astype(str)
    if ids.shape != (59_551,) or relatives.shape != ids.shape:
        raise ValueError("SOP direct public TRAIN inventory differs")
    rows = np.linspace(0, 59_550, 640, dtype=np.int64)
    if len(set(rows.tolist())) != 640:
        raise ValueError("SOP direct public query rows repeat")
    paths = []
    for row in rows:
        relative = PurePosixPath(str(relatives[row]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("SOP direct public image path differs")
        path = args.dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP direct public image missing")
        paths.append(path)

    raw = {label: [] for label in ("baseline", "direct")}
    expected = {}
    with Siglip2CompactIndex.from_artifacts(
        model_snapshot=args.model_snapshot,
        training_receipt=receipt_path,
        training_checkpoint=run / "checkpoint.pt",
        train_embeddings=run / "train_embeddings.npy",
        native_library=args.native_library,
        expected_receipt_sha256=receipt_sha,
        precision="fp16_native",
    ) as index:
        encoder = index.encoder
        gallery = index.gallery
        assert encoder is not None and gallery is not None
        if not encoder._use_direct_preprocess:
            raise ValueError("pinned direct processor was not selected")
        for block in range(20):
            images = decode(paths[block * 32 : (block + 1) * 32])
            for position, image in enumerate(images):
                selected = [image]
                encoder._use_direct_preprocess = False
                baseline = encoder.encode_images(selected)
                baseline_result = gallery.search_packed(baseline)
                encoder._use_direct_preprocess = True
                direct = encoder.encode_images(selected)
                direct_result = gallery.search_packed(direct)
                if (
                    not torch.equal(baseline.codes, direct.codes)
                    or not torch.equal(baseline.inverse_norms, direct.inverse_norms)
                    or not np.array_equal(baseline_result[0], direct_result[0])
                    or not np.array_equal(baseline_result[1], direct_result[1])
                ):
                    raise ValueError("SOP direct packed or top-10 parity differs")
                expected[block, position] = digest_result(baseline_result)
            encoder._use_direct_preprocess = False
            baseline = encoder.encode_images(images)
            baseline_result = gallery.search_packed(baseline)
            encoder._use_direct_preprocess = True
            direct = encoder.encode_images(images)
            direct_result = gallery.search_packed(direct)
            if (
                not torch.equal(baseline.codes, direct.codes)
                or not torch.equal(baseline.inverse_norms, direct.inverse_norms)
                or not np.array_equal(baseline_result[0], direct_result[0])
                or not np.array_equal(baseline_result[1], direct_result[1])
            ):
                raise ValueError("SOP direct batch32 parity differs")
            expected[block, 32] = digest_result(baseline_result)
        torch.cuda.reset_peak_memory_stats()
        for block in range(20):
            block_paths = paths[block * 32 : (block + 1) * 32]
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
        # Batch 32 uses the same processor path in both arms. Confirm whole-call
        # parity and nonregression with a short paired diagnostic.
        batch32 = {label: [] for label in raw}
        for block in range(20):
            block_paths = paths[block * 32 : (block + 1) * 32]
            for label in ("baseline", "direct") if block % 2 == 0 else ("direct", "baseline"):
                encoder._use_direct_preprocess = label == "direct"
                batch32[label].append(
                    [timed_call(index, block_paths, expected[block, 32]) for _ in range(10)]
                )
        batch32_stats = {label: summary(batch32[label], 32) for label in batch32}
        passed = (
            passed
            and batch32_stats["direct"]["p95_ms"] <= 1.10 * batch32_stats["baseline"]["p95_ms"]
        )
        report = {
            "schema": "sfora-sop-direct-public-paired-v1",
            "source_sha256": sha256(Path(__file__)),
            "serving_source_sha256": sha256(Path(serving.__file__)),
            "helper_source_sha256": sha256(Path(latency.__file__)),
            "source_archive_sha256": ARCHIVE_SHA,
            "decision_sha256": DECISION_SHA,
            "training_receipt_sha256": receipt_sha,
            "native_library_sha256": NATIVE_SHA,
            "query_image_ids": ids[rows].tolist(),
            "query_image_sha256": [sha256(path) for path in paths],
            "packed_and_top10_exact": True,
            "batch1_gate_rule": (
                "p50 ratio <= 0.95; p99 point ratio <= 1.0; "
                "paired-block bootstrap p99 95% upper <= 1.05"
            ),
            "batch32_gate_rule": "paired batch32 p95 direct <= 1.10 baseline",
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
        )
    )


if __name__ == "__main__":
    main()
