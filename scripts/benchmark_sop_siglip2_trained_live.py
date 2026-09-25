#!/usr/bin/env python3
"""Measure authenticated trained SigLIP2 SOP TRAIN image-to-native-top-10 latency."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.packed_int8_search import CpuPackedInt8Gallery
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
TILEIRAS_SHA256 = "df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae"
TRAINER_SHA256 = "95de5b80fda488db18715308588a3d56e53ecf6e4df99313877426274346b1c8"
MODEL_HASHES = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}
STAGES = (
    "host_preprocess_ns",
    "host_to_device_ns",
    "encode_transfer_ns",
    "pack_ns",
    "native_search_ns",
    "image_to_top10_ns",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summary(raw: list[int]) -> dict[str, float]:
    values = np.asarray(raw, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "mean_ms": float(values.mean()),
    }


def first_nonself(ordinals: np.ndarray, row: int) -> int:
    for ordinal in ordinals[0]:
        if int(ordinal) != row:
            return int(ordinal)
    raise ValueError("trained SigLIP2 live nonself top-10 is missing")


def paired_order(repeat: int) -> tuple[str, str]:
    """AB/BA/BA/AB blocks balance position within each group of four."""
    return (
        ("fp32_autocast", "fp16_native")
        if repeat % 4 in (0, 3)
        else ("fp16_native", "fp32_autocast")
    )


@torch.inference_mode()
def pooled_output(
    vision: torch.nn.Module, pixel_values: torch.Tensor, precision: str
) -> torch.Tensor:
    if precision not in ("fp32_autocast", "fp16_native"):
        raise ValueError("trained SigLIP2 live inference precision differs")
    pixels = pixel_values.to(dtype=torch.float16) if precision == "fp16_native" else pixel_values
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=precision == "fp32_autocast"):
        source = vision(pixel_values=pixels).pooler_output
    if source is None:
        raise ValueError("trained SigLIP2 live pooler missing")
    return source


@torch.inference_mode()
def one_call(
    paths: list[Path],
    processor: Any,
    vision: torch.nn.Module,
    head: torch.nn.Linear,
    gallery: CutilePackedInt8Gallery,
    precision: str,
) -> tuple[dict[str, int], str]:
    started = time.perf_counter_ns()
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    batch = processor(images=images, return_tensors="pt")
    if set(batch) != {"pixel_values"}:
        raise ValueError("trained SigLIP2 live processor geometry differs")
    host_ready = time.perf_counter_ns()
    tensors = {key: value.cuda(non_blocking=False) for key, value in batch.items()}
    device_ready = time.perf_counter_ns()
    source = pooled_output(vision, tensors["pixel_values"], precision)
    features = F.normalize(compact_head_features(source, head), dim=1).cpu()
    encoded = time.perf_counter_ns()
    if features.shape != (len(paths), 128) or not bool(torch.isfinite(features).all()):
        raise ValueError("trained SigLIP2 live feature geometry differs")
    packed = pack_int8_unit_embeddings(features)
    packed_at = time.perf_counter_ns()
    ordinals, scores = gallery.search_packed(packed)
    torch.cuda.synchronize()
    finished = time.perf_counter_ns()
    if ordinals.shape != (len(paths), 10) or scores.shape != ordinals.shape:
        raise ValueError("trained SigLIP2 live top-10 geometry differs")
    return (
        dict(
            zip(
                STAGES,
                (
                    host_ready - started,
                    device_ready - host_ready,
                    encoded - device_ready,
                    packed_at - encoded,
                    finished - packed_at,
                    finished - started,
                ),
                strict=True,
            )
        ),
        hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest(),
    )


@torch.inference_mode()
def verify_native(
    paths: list[Path],
    processor: Any,
    vision: torch.nn.Module,
    head: torch.nn.Linear,
    gallery: CutilePackedInt8Gallery,
    scalar: CpuPackedInt8Gallery,
    precision: str,
) -> None:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    batch = processor(images=images, return_tensors="pt")
    if set(batch) != {"pixel_values"}:
        raise ValueError("trained SigLIP2 live processor geometry differs")
    source = pooled_output(vision, batch["pixel_values"].cuda(), precision)
    features = F.normalize(compact_head_features(source, head), dim=1).cpu()
    packed = pack_int8_unit_embeddings(features)
    actual_ordinals, actual_scores = gallery.search_packed(packed)
    expected_ordinals, expected_scores = scalar.search_packed(packed)
    if not np.array_equal(actual_ordinals, expected_ordinals) or not np.allclose(
        actual_scores, expected_scores, rtol=0, atol=1e-5
    ):
        raise ValueError("trained SigLIP2 live native top-10 differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b1-blocks", type=int, default=1000)
    parser.add_argument("--b32-blocks", type=int, default=200)
    parser.add_argument(
        "--inference-precision",
        choices=("fp32_autocast", "fp16_native"),
        default="fp32_autocast",
    )
    parser.add_argument("--paired-precision", action="store_true")
    args = parser.parse_args()
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.b1_blocks < 1000
        or args.b32_blocks < 200
        or not torch.cuda.is_available()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.native_library) != NATIVE_SHA256
        or sha256(args.training_receipt) != args.expected_training_receipt_sha256
        or not tileiras
        or not Path(tileiras).is_file()
        or sha256(Path(tileiras)) != TILEIRAS_SHA256
        or any(sha256(args.model_snapshot / name) != value for name, value in MODEL_HASHES.items())
    ):
        raise ValueError("trained SigLIP2 live timing authority differs")
    training = json.loads(args.training_receipt.read_text())
    if (
        training.get("schema") != "sfora-sop-siglip2-compact-full-backbone-v1"
        or training.get("updates") != 1_000
        or training.get("seed") != 179019
        or training.get("arm") not in ("arcface", "packed_rank", "float_rank")
        or training.get("grad_scaler_initial_scale") != 128.0
        or training.get("source_archive_sha256") != ARCHIVE_SHA256
        or training.get("native_library_sha256") != NATIVE_SHA256
        or training.get("tileiras_sha256") != TILEIRAS_SHA256
        or training.get("quality", {}).get("native_top10_exact") is not True
        or training.get("quality", {}).get("gallery_wire_bytes_per_row") != 130
        or training.get("source_files_sha256", {}).get("scripts/train_sop_siglip2_compact.py")
        != TRAINER_SHA256
        or training.get("checkpoint_sha256") != sha256(args.training_checkpoint)
        or training.get("train_embeddings_sha256") != sha256(args.train_embeddings)
    ):
        raise ValueError("trained SigLIP2 live training receipt differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
    if labels.shape != (59_551,) or ids.shape != labels.shape:
        raise ValueError("trained SigLIP2 live TRAIN inventory differs")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    held = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if hashlib.sha256(ids[held].tobytes()).hexdigest() != training["query_image_ids_sha256"]:
        raise ValueError("trained SigLIP2 live heldout query inventory differs")
    query_rows = held[np.linspace(0, len(held) - 1, 32, dtype=int)]
    paths = []
    for row in query_rows:
        relative = PurePosixPath(str(relatives[row]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("trained SigLIP2 live path differs")
        path = args.dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("trained SigLIP2 live image differs")
        paths.append(path)
    import PIL
    import torchvision
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    if (
        transformers.__version__ != training["hardware"]["transformers"]
        or torchvision.__version__ != training["hardware"]["torchvision"]
        or PIL.__version__ != training["hardware"]["pillow"]
    ):
        raise ValueError("trained SigLIP2 live image stack differs")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    if (
        type(processor).__name__ != "SiglipImageProcessor"
        or processor.size["height"] != 256
        or processor.size["width"] != 256
        or processor.resample != 2
    ):
        raise ValueError("trained SigLIP2 live processor differs")
    full_model = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full_model.vision_model
    del full_model
    vision = vision.float().cuda().eval()
    head = torch.nn.Linear(1024, 128).cuda().eval()
    checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("seed") != 179019
        or checkpoint.get("updates") != 1_000
        or checkpoint.get("arm") != training["arm"]
    ):
        raise ValueError("trained SigLIP2 live checkpoint differs")
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    if args.paired_precision and args.inference_precision != "fp32_autocast":
        raise ValueError("paired precision requires fp32 autocast base")
    if args.inference_precision == "fp16_native":
        vision.half()
    values = np.load(args.train_embeddings, mmap_mode="r")
    if values.shape != (59_551, 128) or values.dtype != np.float32:
        raise ValueError("trained SigLIP2 live gallery geometry differs")
    packed_gallery = pack_int8_unit_embeddings(torch.from_numpy(np.asarray(values).copy()))
    scalar = CpuPackedInt8Gallery.open_packed(packed_gallery)
    raw: dict[str, Any] = {}
    cached_r1 = np.asarray(training["quality"]["per_query_r1"], dtype=np.bool_)
    if (
        cached_r1.shape != (5_851,)
        or abs(float(cached_r1.mean()) - training["quality"]["recall_at_1"]) > 1e-6
    ):
        raise ValueError("trained SigLIP2 cached query quality differs")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with CutilePackedInt8Gallery.open_packed(args.native_library, packed_gallery) as gallery:
        verify_native(paths[:1], processor, vision, head, gallery, scalar, args.inference_precision)
        verify_native(paths, processor, vision, head, gallery, scalar, args.inference_precision)
        if args.paired_precision:
            half_vision = copy.deepcopy(vision).half().eval()
            verify_native(paths, processor, half_vision, head, gallery, scalar, "fp16_native")
            towers = {"fp32_autocast": vision, "fp16_native": half_vision}
            for batch_size, blocks in ((1, args.b1_blocks), (32, args.b32_blocks)):
                selected = paths[:batch_size]
                for _ in range(5):
                    for mode in paired_order(_):
                        one_call(selected, processor, towers[mode], head, gallery, mode)
                samples = {mode: {stage: [] for stage in STAGES} for mode in towers}
                digests: dict[str, str] = {}
                for repeat in range(blocks):
                    for mode in paired_order(repeat):
                        durations, digest = one_call(
                            selected, processor, towers[mode], head, gallery, mode
                        )
                        if mode in digests and digest != digests[mode]:
                            raise ValueError("paired trained timing result changed during replay")
                        digests[mode] = digest
                        for stage in STAGES:
                            samples[mode][stage].append(durations[stage])
                    if repeat % 50 == 49:
                        print(
                            json.dumps({"paired_batch": batch_size, "blocks": repeat + 1}),
                            flush=True,
                        )
                raw[str(batch_size)] = {
                    mode: {
                        "raw_ns": samples[mode],
                        "summary": {stage: summary(samples[mode][stage]) for stage in STAGES},
                        "result_sha256": digests[mode],
                        "throughput_queries_per_second": batch_size
                        * 1e9
                        / float(np.mean(samples[mode]["image_to_top10_ns"])),
                    }
                    for mode in towers
                }
                raw[str(batch_size)]["fp16_minus_fp32_ms"] = summary(
                    [
                        half - full
                        for half, full in zip(
                            samples["fp16_native"]["image_to_top10_ns"],
                            samples["fp32_autocast"]["image_to_top10_ns"],
                            strict=True,
                        )
                    ]
                )
                raw[str(batch_size)]["blocks"] = blocks
            receipt = {
                "schema": "sfora-sop-siglip2-trained-live-paired-precision-v1",
                "claim_eligible": False,
                "split": "SOP official TRAIN product-disjoint holdout query images only",
                "arm": training["arm"],
                "seed": 179019,
                "order": "AB/BA/BA/AB repeated",
                "timing_query_policy": (
                    "fixed first holdout query at batch 1; fixed 32 queries at batch 32"
                ),
                "query_image_ids": ids[query_rows].tolist(),
                "query_image_sha256": [sha256(path) for path in paths],
                "source_archive_sha256": ARCHIVE_SHA256,
                "training_receipt_sha256": sha256(args.training_receipt),
                "training_checkpoint_sha256": sha256(args.training_checkpoint),
                "train_embeddings_sha256": sha256(args.train_embeddings),
                "gallery_export_precision": training["precision"],
                "gallery_rows": 59_551,
                "gallery_wire_bytes_per_row": 130,
                "native_top10_exact_on_32_selected_queries_both_modes": True,
                "native_library_sha256": NATIVE_SHA256,
                "tileiras_sha256": TILEIRAS_SHA256,
                "source_sha256": sha256(Path(__file__)),
                "timing": raw,
                "total_wall_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "hardware": {
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "python": platform.python_version(),
                    "transformers": transformers.__version__,
                    "torchvision": torchvision.__version__,
                    "pillow": PIL.__version__,
                },
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as stream:
                stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())
            print(
                json.dumps(
                    {"paired_precision": {k: v["fp16_minus_fp32_ms"] for k, v in raw.items()}}
                ),
                flush=True,
            )
            return
        for batch_size, blocks in ((1, args.b1_blocks), (32, args.b32_blocks)):
            selected = paths[:batch_size]
            for _ in range(5):
                one_call(selected, processor, vision, head, gallery, args.inference_precision)
            samples = {stage: [] for stage in STAGES}
            digest: str | None = None
            for repeat in range(blocks):
                durations, actual_digest = one_call(
                    selected, processor, vision, head, gallery, args.inference_precision
                )
                if digest is None:
                    digest = actual_digest
                elif actual_digest != digest:
                    raise ValueError("trained SigLIP2 live timing result changed during replay")
                for stage in STAGES:
                    samples[stage].append(durations[stage])
                if repeat % 50 == 49:
                    print(json.dumps({"batch": batch_size, "blocks": repeat + 1}), flush=True)
            raw[str(batch_size)] = {
                "raw_ns": samples,
                "summary": {stage: summary(samples[stage]) for stage in STAGES},
                "result_sha256": digest,
                "warmups": 5,
                "blocks": blocks,
                "throughput_queries_per_second": batch_size
                * 1e9
                / float(np.mean(samples["image_to_top10_ns"])),
            }
        live_r1: list[bool] = []
        changed_codes = 0
        changed_norms = 0
        changed_top10 = 0
        max_score_delta = 0.0
        for index, row in enumerate(held, start=1):
            relative = PurePosixPath(str(relatives[row]))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("trained SigLIP2 live query path differs")
            path = args.dataset_root.joinpath(*relative.parts)
            if not path.is_file() or path.is_symlink():
                raise ValueError("trained SigLIP2 live query image missing")
            with Image.open(path) as image:
                processor_batch = processor(images=[image.convert("RGB")], return_tensors="pt")
            if set(processor_batch) != {"pixel_values"}:
                raise ValueError("trained SigLIP2 live query processor differs")
            source = pooled_output(
                vision, processor_batch["pixel_values"].cuda(), args.inference_precision
            )
            with torch.inference_mode():
                feature = F.normalize(compact_head_features(source, head), dim=1).cpu()
                query = pack_int8_unit_embeddings(feature)
            cached = PackedInt8Embeddings(
                packed_gallery.codes[row : row + 1].contiguous(),
                packed_gallery.inverse_norms[row : row + 1].contiguous(),
            )
            changed_codes += int(not torch.equal(query.codes, cached.codes))
            changed_norms += int(not torch.equal(query.inverse_norms, cached.inverse_norms))
            ordinals, scores = gallery.search_packed(query)
            cached_ordinals, cached_scores = gallery.search_packed(cached)
            cached_top1 = first_nonself(cached_ordinals, int(row))
            if bool(labels[cached_top1] == labels[row]) != bool(cached_r1[index - 1]):
                raise ValueError(f"trained SigLIP2 cached quality differs at query {index}")
            changed_top10 += int(not np.array_equal(ordinals, cached_ordinals))
            max_score_delta = max(max_score_delta, float(np.max(np.abs(scores - cached_scores))))
            live_top1 = first_nonself(ordinals, int(row))
            live_r1.append(bool(labels[live_top1] == labels[row]))
            if index % 500 == 0:
                print(json.dumps({"live_queries": index}), flush=True)
    observed_r1 = np.asarray(live_r1, dtype=np.bool_)
    live_quality = {
        "live_batch1_recall_at_1": float(observed_r1.mean()),
        "cached_export_recall_at_1": float(cached_r1.mean()),
        "live_minus_cached_percentage_points": float((observed_r1.mean() - cached_r1.mean()) * 100),
        "per_query_live_r1": observed_r1.astype(np.uint8).tolist(),
        "per_query_cached_r1": cached_r1.astype(np.uint8).tolist(),
        "r1_changed_queries": int(np.count_nonzero(observed_r1 != cached_r1)),
        "code_changed_queries": changed_codes,
        "inverse_norm_changed_queries": changed_norms,
        "top10_changed_queries": changed_top10,
        "maximum_top10_score_absolute_delta": max_score_delta,
    }
    receipt = {
        "schema": "sfora-sop-siglip2-trained-live-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout query images only",
        "arm": training["arm"],
        "inference_precision": args.inference_precision,
        "seed": 179019,
        "query_image_ids": ids[query_rows].tolist(),
        "query_image_sha256": [sha256(path) for path in paths],
        "source_archive_sha256": ARCHIVE_SHA256,
        "training_receipt_sha256": sha256(args.training_receipt),
        "gallery_export_precision": training["precision"],
        "timing_query_policy": "fixed first holdout query at batch 1; fixed 32 queries at batch 32",
        "training_checkpoint_sha256": sha256(args.training_checkpoint),
        "train_embeddings_sha256": sha256(args.train_embeddings),
        "model_file_sha256": MODEL_HASHES,
        "native_library_sha256": NATIVE_SHA256,
        "tileiras_sha256": sha256(Path(tileiras)),
        "source_sha256": sha256(Path(__file__)),
        "gallery_rows": 59_551,
        "gallery_wire_bytes_per_row": 130,
        "native_top10_exact_on_32_selected_queries": True,
        "timing": raw,
        "live_batch1_quality": live_quality,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "torchvision": torchvision.__version__,
            "pillow": PIL.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "arm": training["arm"],
                "live_batch1_r1": live_quality["live_batch1_recall_at_1"],
                "timing": {batch: raw[batch]["summary"]["image_to_top10_ns"] for batch in raw},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
