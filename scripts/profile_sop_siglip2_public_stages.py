#!/usr/bin/env python3
"""Attribute one fixed SOP TRAIN image-to-top-10 call to its public stages."""

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
from torch.nn import functional as F

import sfora.siglip2_compact_serving as serving
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.siglip2_compact_serving import Siglip2CompactIndex
from sfora.sop_compact_training import compact_head_features

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
DECISION_SHA = "31ebbe4ae71c5f1c7922ef9140721957ccc408eab165df1570678a8843988ed5"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summary(samples: list[int]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(values.mean()),
    }


@torch.inference_mode()
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
    ):
        raise ValueError("SOP public stage profile authority differs")
    decision = json.loads(args.decision.read_text())
    run = args.runs / "sfora-sop-true-freeze-freeze-179024-1000-v1"
    receipt_path = run / "receipt.json"
    receipt_sha = sha256(receipt_path)
    if decision["arms"]["freeze"]["receipt_sha256"] != receipt_sha:
        raise ValueError("SOP public stage profile checkpoint differs")
    with np.load(args.source_archive, allow_pickle=False) as source:
        relative = PurePosixPath(str(source["train_relative_paths"][0]))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("SOP public stage profile image path differs")
    image_path = args.dataset_root.joinpath(*relative.parts)
    if not image_path.is_file() or image_path.is_symlink():
        raise ValueError("SOP public stage profile image missing")
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
        samples: dict[str, list[int]] = {
            name: []
            for name in (
                "decode",
                "processor",
                "transfer",
                "vision",
                "head_pack",
                "search",
                "whole",
            )
        }
        expected_digest = None
        torch.cuda.reset_peak_memory_stats()
        for step in range(105):
            torch.cuda.synchronize()
            times = [time.perf_counter_ns()]
            with Image.open(BytesIO(image_path.read_bytes())) as opened:
                image = opened.convert("RGB")
            times.append(time.perf_counter_ns())
            batch = encoder.processor(images=[image.convert("RGB")], return_tensors="pt")
            if set(batch) != {"pixel_values"}:
                raise ValueError("SOP public stage profile processor differs")
            times.append(time.perf_counter_ns())
            pixels = batch["pixel_values"].to(device=encoder.device, dtype=torch.float16)
            torch.cuda.synchronize()
            times.append(time.perf_counter_ns())
            pooled = encoder.vision(pixel_values=pixels).pooler_output
            torch.cuda.synchronize()
            times.append(time.perf_counter_ns())
            if pooled is None or pooled.shape != (1, 1024):
                raise ValueError("SOP public stage profile encoder differs")
            features = F.normalize(compact_head_features(pooled, encoder.head), dim=1).cpu()
            packed = pack_int8_unit_embeddings(features)
            torch.cuda.synchronize()
            times.append(time.perf_counter_ns())
            ordinals, scores = gallery.search_packed(packed)
            torch.cuda.synchronize()
            times.append(time.perf_counter_ns())
            digest = hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest()
            public_ordinals, public_scores = index.search_images([image])
            public_digest = hashlib.sha256(
                public_ordinals.tobytes() + public_scores.tobytes()
            ).hexdigest()
            if digest != public_digest or (
                expected_digest is not None and digest != expected_digest
            ):
                raise ValueError("SOP public stage profile top-10 differs")
            expected_digest = digest
            if step >= 5:
                for name, start, stop in zip(
                    ("decode", "processor", "transfer", "vision", "head_pack", "search"),
                    times,
                    times[1:],
                    strict=True,
                ):
                    samples[name].append(stop - start)
                samples["whole"].append(times[-1] - times[0])
        report = {
            "schema": "sfora-sop-siglip2-public-stage-profile-v1",
            "claim_eligible": False,
            "source_sha256": sha256(Path(__file__)),
            "serving_source_sha256": sha256(Path(serving.__file__)),
            "decision_sha256": DECISION_SHA,
            "training_receipt_sha256": receipt_sha,
            "native_library_sha256": NATIVE_SHA,
            "image_sha256": sha256(image_path),
            "top10_sha256": expected_digest,
            "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
            "samples_per_stage": 100,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "stage": {name: summary(values) for name, values in samples.items()},
        }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"stage": report["stage"], "top10_sha256": expected_digest}, sort_keys=True))


if __name__ == "__main__":
    main()
