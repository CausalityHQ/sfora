#!/usr/bin/env python3
"""Certify paired varied-image SOP public top-10 p99 at 20 versus 1 CPU thread."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
import paired_latency_certification as latency
import torch
from PIL import Image

import sfora.siglip2_compact_serving as serving
from sfora.siglip2_compact_serving import Siglip2CompactIndex

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
DECISION_SHA = "31ebbe4ae71c5f1c7922ef9140721957ccc408eab165df1570678a8843988ed5"
SERVING_SHA = "17deaca3b7a07c30664d52bcd3ef2abae56eb3a81458f4371530f00ca68d8079"
HELPER_SHA = "ada676f60096f90551450d463bfeeba168bab8689c1f433b58aed43bbb85c325"
SCREEN_SHA = "1d42c2928db1bb2e106f8cda61b79de39459d421713e8307997616d8419fd8b5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_result(result: tuple[np.ndarray, np.ndarray]) -> str:
    ordinals, scores = result
    return hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest()


def decode(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(BytesIO(path.read_bytes())) as opened:
            images.append(opened.convert("RGB"))
    return images


def summary(blocks: list[list[int]], batch: int) -> dict[str, float]:
    ms = np.asarray(blocks, dtype=np.float64).reshape(-1) / 1e6
    return {
        "p50_ms": float(np.median(ms)),
        "p95_ms": float(np.quantile(ms, 0.95)),
        "p99_ms": float(np.quantile(ms, 0.99, method="higher")),
        "mean_ms": float(ms.mean()),
        "images_per_second": float(batch * 1000 / ms.mean()),
    }


def timed_call(
    index: Siglip2CompactIndex,
    paths: list[Path],
    expected_digest: str,
) -> int:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    result = index.search_images(decode(paths))
    torch.cuda.synchronize()
    elapsed = time.perf_counter_ns() - started
    if digest_result(result) != expected_digest:
        raise ValueError("SOP serving thread p99 repeated top-10 differs")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "decision",
        "runs",
        "source-archive",
        "dataset-root",
        "model-snapshot",
        "native-library",
        "screen-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.decision) != DECISION_SHA
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.native_library) != NATIVE_SHA
        or sha256(args.screen_receipt) != SCREEN_SHA
        or sha256(Path(serving.__file__)) != SERVING_SHA
        or sha256(Path(latency.__file__)) != HELPER_SHA
        or not torch.cuda.is_available()
        or torch.get_num_threads() != 20
    ):
        raise ValueError("SOP serving thread p99 authority differs")
    decision = json.loads(args.decision.read_text())
    screen = json.loads(args.screen_receipt.read_text())
    if (
        decision.get("schema") != "sfora-sop-true-freeze-paired-train-only-v1"
        or decision.get("seed") != 179024
        or screen.get("advance_p99_certification") is not True
    ):
        raise ValueError("SOP serving thread p99 decision differs")
    run = args.runs / "sfora-sop-true-freeze-freeze-179024-1000-v1"
    receipt_path = run / "receipt.json"
    receipt_sha = sha256(receipt_path)
    if decision["arms"]["freeze"]["receipt_sha256"] != receipt_sha:
        raise ValueError("SOP serving thread checkpoint differs")
    with np.load(args.source_archive, allow_pickle=False) as source:
        ids = np.asarray(source["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(source["train_relative_paths"]).astype(str)
    if ids.shape != (59_551,) or relatives.shape != ids.shape:
        raise ValueError("SOP serving thread TRAIN inventory differs")
    rows = np.linspace(0, 59_550, 640, dtype=np.int64)
    if len(set(rows.tolist())) != 640:
        raise ValueError("SOP serving thread p99 query rows repeat")
    paths = []
    for row in rows:
        relative = PurePosixPath(str(relatives[row]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("SOP serving thread image path differs")
        path = args.dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP serving thread image missing")
        paths.append(path)
    initial_threads = torch.get_num_threads()
    if initial_threads != 20:
        raise ValueError("SOP serving thread baseline is not 20")
    raw: dict[str, dict[str, list[list[int]]]] = {
        str(batch): {"baseline": [], "single": []} for batch in (1, 32)
    }
    statistics = {}
    certified = {}
    expected: dict[tuple[int, int, int], str] = {}
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
        try:
            for block in range(20):
                images = decode(paths[block * 32 : (block + 1) * 32])
                for batch in (1, 32):
                    for position in range(32 if batch == 1 else 1):
                        selected = images[position : position + 1] if batch == 1 else images
                        prior = None
                        for label, threads in (("baseline", 20), ("single", 1)):
                            torch.set_num_threads(threads)
                            packed = encoder.encode_images(selected)
                            result = gallery.search_packed(packed)
                            if prior is not None and (
                                not torch.equal(prior[0].codes, packed.codes)
                                or not torch.equal(prior[0].inverse_norms, packed.inverse_norms)
                                or not np.array_equal(prior[1][0], result[0])
                                or not np.array_equal(prior[1][1], result[1])
                            ):
                                raise ValueError(
                                    "SOP serving thread p99 packed/top-10 parity differs"
                                )
                            if label == "baseline":
                                prior = (packed, result)
                                expected[(batch, block, position)] = digest_result(result)
            torch.cuda.reset_peak_memory_stats()
            for batch in (1, 32):
                key = str(batch)
                for block in range(20):
                    block_paths = paths[block * 32 : (block + 1) * 32]
                    order = (
                        ("baseline", "single", "single", "baseline")
                        if block % 2 == 0
                        else ("single", "baseline", "baseline", "single")
                    )
                    samples = {"baseline": [], "single": []}

                    for label in order:
                        torch.set_num_threads(20 if label == "baseline" else 1)
                        for position in range(5):
                            actual = position if batch == 1 else 0
                            selected = (
                                block_paths[actual : actual + 1] if batch == 1 else block_paths
                            )
                            timed_call(index, selected, expected[(batch, block, actual)])
                        for position in range(250):
                            actual = position % 32 if batch == 1 else 0
                            selected = (
                                block_paths[actual : actual + 1] if batch == 1 else block_paths
                            )
                            samples[label].append(
                                timed_call(index, selected, expected[(batch, block, actual)])
                            )
                    for label in ("baseline", "single"):
                        if len(samples[label]) != 500:
                            raise ValueError("SOP serving thread p99 timing count differs")
                        raw[key][label].append(samples[label])
                    print(json.dumps({"batch": batch, "blocks_done": block + 1}), flush=True)
                statistics[key] = {
                    label: summary(raw[key][label], batch) for label in ("baseline", "single")
                }
                certified[key] = latency.block_bootstrap_p99_ratio(
                    raw[key]["single"], raw[key]["baseline"]
                )
                passed = (
                    certified[key]["latency_gate_passed"]
                    and statistics[key]["single"]["p95_ms"] <= statistics[key]["baseline"]["p95_ms"]
                    and statistics[key]["single"]["images_per_second"]
                    >= statistics[key]["baseline"]["images_per_second"]
                )
                print(json.dumps({"batch": batch, "certification_passed": passed}), flush=True)
                if not passed:
                    break
        finally:
            torch.set_num_threads(initial_threads)
        report = {
            "schema": "sfora-sop-siglip2-serving-threads-p99-v1",
            "claim_eligible": False,
            "single_purpose_process_config_supported": len(certified) == 2
            and all(
                certified[str(batch)]["latency_gate_passed"]
                and statistics[str(batch)]["single"]["p95_ms"]
                <= statistics[str(batch)]["baseline"]["p95_ms"]
                and statistics[str(batch)]["single"]["images_per_second"]
                >= statistics[str(batch)]["baseline"]["images_per_second"]
                for batch in (1, 32)
            ),
            "source_sha256": sha256(Path(__file__)),
            "serving_source_sha256": SERVING_SHA,
            "helper_source_sha256": HELPER_SHA,
            "screen_receipt_sha256": SCREEN_SHA,
            "source_archive_sha256": ARCHIVE_SHA,
            "decision_sha256": DECISION_SHA,
            "training_receipt_sha256": receipt_sha,
            "native_library_sha256": NATIVE_SHA,
            "query_image_ids": ids[rows].tolist(),
            "query_image_sha256": [sha256(path) for path in paths],
            "packed_and_top10_exact": True,
            "gpu": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "statistics": statistics,
            "certification": certified,
            "raw_ns": raw,
        }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"certification": certified, "statistics": statistics}, sort_keys=True))


if __name__ == "__main__":
    main()
