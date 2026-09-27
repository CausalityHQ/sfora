#!/usr/bin/env python3
"""Paired public SOP image-to-top-10 timing for PyTorch intra-op threads 20/1."""

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
import torch
from PIL import Image

import sfora.siglip2_compact_serving as serving
from sfora.siglip2_compact_serving import Siglip2CompactIndex

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
DECISION_SHA = "31ebbe4ae71c5f1c7922ef9140721957ccc408eab165df1570678a8843988ed5"
SERVING_SHA = "17deaca3b7a07c30664d52bcd3ef2abae56eb3a81458f4371530f00ca68d8079"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summary(samples: list[int], batch: int) -> dict[str, float]:
    ms = np.asarray(samples, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.median(ms)),
        "p95_ms": float(np.quantile(ms, 0.95)),
        "mean_ms": float(ms.mean()),
        "images_per_second": float(batch * 1000 / ms.mean()),
    }


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
        or sha256(Path(serving.__file__)) != SERVING_SHA
        or not torch.cuda.is_available()
    ):
        raise ValueError("SOP serving thread screen authority differs")
    decision = json.loads(args.decision.read_text())
    if (
        decision.get("schema") != "sfora-sop-true-freeze-paired-train-only-v1"
        or decision.get("seed") != 179024
    ):
        raise ValueError("SOP serving thread decision differs")
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
    rows = np.linspace(0, 59_550, 32, dtype=np.int64)
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
        all_images = []
        for path in paths:
            with Image.open(BytesIO(path.read_bytes())) as opened:
                all_images.append(opened.convert("RGB"))
        packed = {}
        top10 = {}
        for label, threads in (("baseline", 20), ("single", 1)):
            torch.set_num_threads(threads)
            packed[label] = encoder.encode_images(all_images)
            top10[label] = gallery.search_packed(packed[label])
        if (
            not torch.equal(packed["baseline"].codes, packed["single"].codes)
            or not torch.equal(packed["baseline"].inverse_norms, packed["single"].inverse_norms)
            or not np.array_equal(top10["baseline"][0], top10["single"][0])
            or not np.array_equal(top10["baseline"][1], top10["single"][1])
        ):
            raise ValueError("SOP serving thread packed/top-10 parity differs")
        raw: dict[str, dict[str, list[int]]] = {
            str(batch): {"baseline": [], "single": []} for batch in (1, 32)
        }
        stable: dict[str, str] = {}

        def call(batch: int, label: str, *, record: bool) -> None:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            images = []
            for path in paths[:batch]:
                with Image.open(BytesIO(path.read_bytes())) as opened:
                    images.append(opened.convert("RGB"))
            ordinals, scores = index.search_images(images)
            torch.cuda.synchronize()
            elapsed = time.perf_counter_ns() - started
            digest = hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest()
            key = str(batch)
            if key in stable and stable[key] != digest:
                raise ValueError("SOP serving thread repeated top-10 differs")
            stable[key] = digest
            if record:
                raw[key][label].append(elapsed)

        torch.cuda.reset_peak_memory_stats()
        try:
            for batch in (1, 32):
                for block in range(20):
                    order = (
                        ("baseline", "single", "single", "baseline")
                        if block % 2 == 0
                        else ("single", "baseline", "baseline", "single")
                    )
                    for label in order:
                        torch.set_num_threads(20 if label == "baseline" else 1)
                        for _ in range(5):
                            call(batch, label, record=False)
                        for _ in range(10):
                            call(batch, label, record=True)
                    print(json.dumps({"batch": batch, "blocks_done": block + 1}), flush=True)
        finally:
            torch.set_num_threads(initial_threads)
        if any(len(values) != 400 for arms in raw.values() for values in arms.values()):
            raise ValueError("SOP serving thread timing count differs")
        timing = {
            label: {arm: summary(raw[label][arm], int(label)) for arm in ("baseline", "single")}
            for label in ("1", "32")
        }
        advance = all(
            timing[label]["single"]["p95_ms"] <= 0.95 * timing[label]["baseline"]["p95_ms"]
            and timing[label]["single"]["images_per_second"]
            >= timing[label]["baseline"]["images_per_second"]
            for label in ("1", "32")
        )
        report = {
            "schema": "sfora-sop-siglip2-serving-threads-screen-v1",
            "claim_eligible": False,
            "advance_p99_certification": advance,
            "source_sha256": sha256(Path(__file__)),
            "serving_source_sha256": SERVING_SHA,
            "source_archive_sha256": ARCHIVE_SHA,
            "decision_sha256": DECISION_SHA,
            "training_receipt_sha256": receipt_sha,
            "native_library_sha256": NATIVE_SHA,
            "query_image_ids": ids[rows].tolist(),
            "query_image_sha256": [sha256(path) for path in paths],
            "packed_32_exact": True,
            "top10_sha256": stable,
            "gpu": torch.cuda.get_device_name(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "timing": timing,
            "raw_ns": raw,
        }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"advance_p99_certification": advance, "timing": timing}, sort_keys=True))


if __name__ == "__main__":
    main()
