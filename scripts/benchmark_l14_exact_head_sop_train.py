#!/usr/bin/env python3
"""Paired full image-to-top-k screen of original and exact-folded L/14 queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from evaluate_sop_cub_transfer import gpu_compute_pids
from evaluate_sop_reference_checkpoint import EXPECTED_SOP_RECORD_SHA256
from export_unicom_sop_embeddings import (
    load_sop_embedding_archive,
    ordered_record_sha256,
    parse_sop_records,
)
from PIL import Image
from probe_l14_parallel_fold import assert_no_foreign_gpu_processes, load_authenticated_l14
from torch import nn
from torch.nn import functional as F
from train_sop_compact_backbone import FIT_FRACTION, SPLIT_SEED, publish_file_noreplace, sha256

import sfora.cutile_int8 as cutile_int8_module
from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.inference_head import fold_eval_affine_head
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca

SOP_ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_LIBRARY_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
NATIVE_API_SHA256 = "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409"
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 179019
STAGES = (
    "host_decode_preprocess_ns",
    "host_to_device_ns",
    "encoder_transfer_ns",
    "project_pack_ns",
    "native_search_ns",
    "image_to_topk_ns",
)


def gallery_rows_without_queries(count: int, query_rows: tuple[int, ...]) -> tuple[int, ...]:
    if (
        count < 2
        or not query_rows
        or len(set(query_rows)) != len(query_rows)
        or any(index < 0 or index >= count for index in query_rows)
    ):
        raise ValueError("SOP paired gallery rows differ")
    held_out = frozenset(query_rows)
    return tuple(index for index in range(count) if index not in held_out)


def scalar_packed_topk(
    query_code: np.ndarray,
    query_inverse: np.float16,
    gallery_codes: np.ndarray,
    gallery_inverse: np.ndarray,
    *,
    k: int = 10,
) -> np.ndarray:
    """Independent full-scan ordinal oracle for one packed query."""

    if (
        query_code.dtype != np.int8
        or gallery_codes.dtype != np.int8
        or query_code.shape != (128,)
        or gallery_codes.ndim != 2
        or gallery_codes.shape[1] != 128
        or gallery_inverse.shape != (len(gallery_codes),)
        or gallery_inverse.dtype != np.float16
        or len(gallery_codes) < k
    ):
        raise ValueError("SOP scalar packed oracle geometry differs")
    dots = gallery_codes.astype(np.int32) @ query_code.astype(np.int32)
    scores = dots.astype(np.float32) * np.float32(query_inverse)
    scores *= gallery_inverse.astype(np.float32)
    return np.argsort(-scores, kind="stable")[:k]


@torch.inference_mode()
def encode_query(
    model: nn.Module, transform: object, paths: tuple[Path, ...]
) -> tuple[torch.Tensor, int, int, int, int, int]:
    started = time.perf_counter_ns()
    images = []
    for path in paths:
        with Image.open(BytesIO(path.read_bytes())) as image:
            images.append(transform(image.convert("RGB")))
    host = torch.stack(images)
    host_ready = time.perf_counter_ns()
    device = host.cuda(non_blocking=False)
    device_ready = time.perf_counter_ns()
    values = model(device).float().cpu().contiguous()
    encoded = time.perf_counter_ns()
    return values, started, host_ready, device_ready, encoded, len(paths)


def call(
    model: nn.Module,
    transform: object,
    paths: tuple[Path, ...],
    pca: object,
    gallery: CutilePackedInt8Gallery,
) -> tuple[dict[str, int], str, np.ndarray]:
    values, started, host_ready, device_ready, encoded, _ = encode_query(
        model, transform, paths
    )
    projected = pca.apply(F.normalize(values, dim=1))
    packed = pack_int8_unit_embeddings(projected)
    packed_at = time.perf_counter_ns()
    ordinals, scores = gallery.search_packed(packed)
    finished = time.perf_counter_ns()
    timings = {
        "host_decode_preprocess_ns": host_ready - started,
        "host_to_device_ns": device_ready - host_ready,
        "encoder_transfer_ns": encoded - device_ready,
        "project_pack_ns": packed_at - encoded,
        "native_search_ns": finished - packed_at,
        "image_to_topk_ns": finished - started,
    }
    digest = hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest()
    return timings, digest, ordinals[:, 0].copy()


def timing_summary(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.quantile(array, 0.50) / 1e6),
        "p95_ms": float(np.quantile(array, 0.95) / 1e6),
        "p99_ms_diagnostic_only": float(np.quantile(array, 0.99) / 1e6),
        "mean_ms": float(np.mean(array) / 1e6),
    }


def paired_block_p50_ratio(original: list[int], fused: list[int]) -> dict[str, float | int]:
    """One-sided paired-block bootstrap for the exact-fused/original median ratio."""

    if len(original) != 100 or len(fused) != 100 or min(original + fused) <= 0:
        raise ValueError("L/14 paired timing schedule differs")
    baseline = np.asarray(original, dtype=np.int64).reshape(10, 10)
    candidate = np.asarray(fused, dtype=np.int64).reshape(10, 10)
    random = np.random.default_rng(BOOTSTRAP_SEED)
    ratios = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        selected = random.integers(0, 10, size=10)
        ratios[draw] = np.median(candidate[selected]) / np.median(baseline[selected])
    return {
        "point": float(np.median(candidate) / np.median(baseline)),
        "upper_95": float(np.quantile(ratios, 0.95, method="higher")),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "unicom-checkout", "checkpoint", "sop-root", "sop-archive", "native-library", "output"
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--calls-per-block", type=int, default=10)
    parser.add_argument("--execute-l14-exact-head-sop-train", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.blocks != 10
        or args.calls_per_block != 10
        or not torch.cuda.is_available()
        or gpu_compute_pids()
        or sha256(args.sop_archive) != SOP_ARCHIVE_SHA256
        or sha256(args.native_library) != NATIVE_LIBRARY_SHA256
        or sha256(Path(cutile_int8_module.__file__)) != NATIVE_API_SHA256
    ):
        raise ValueError("L/14 exact head paired benchmark invocation differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = parse_sop_records(args.sop_root)
    if ordered_record_sha256(records) != EXPECTED_SOP_RECORD_SHA256:
        raise ValueError("L/14 exact head SOP record authority differs")
    train = tuple(row for row in records if row.split == "train")
    archive = load_sop_embedding_archive(args.sop_archive)
    if (
        archive["metadata"].get("model_identifier") != "UNICOM-ViT-L/14@336px"
        or archive["train_embeddings"].shape != (59_551, 768)
        or not np.array_equal(archive["train_image_ids"], [row.image_id for row in train])
        or not np.array_equal(archive["train_labels"], [row.label for row in train])
    ):
        raise ValueError("L/14 exact head SOP archive inventory differs")
    labels = tuple(int(row.label) for row in train)
    partition = deterministic_class_partition(labels, fit_fraction=FIT_FRACTION, seed=SPLIT_SEED)
    selected_local = np.linspace(0, len(partition.validation_row_indexes) - 1, 32, dtype=int)
    query_rows = tuple(partition.validation_row_indexes[int(index)] for index in selected_local)
    gallery_rows = gallery_rows_without_queries(len(train), query_rows)
    query_records = tuple(train[index] for index in query_rows)
    query_paths = tuple(row.image_path for row in query_records)
    query_manifest = b"".join(hashlib.sha256(path.read_bytes()).digest() for path in query_paths)
    source = torch.from_numpy(np.ascontiguousarray(archive["train_embeddings"]).copy())
    fit = F.normalize(source[list(partition.fit_row_indexes)].contiguous(), dim=1)
    pca_started = time.perf_counter()
    pca = fit_centered_pca(fit, dimensions=128)
    pca_fit_seconds = time.perf_counter() - pca_started
    projected_gallery = pca.apply(F.normalize(source[list(gallery_rows)], dim=1))
    packed_gallery = pack_int8_unit_embeddings(projected_gallery)
    del projected_gallery, fit
    model, transform = load_authenticated_l14(args.unicom_checkout, args.checkpoint)
    model = model.cuda().eval()
    source_head = model.feature
    fused_head = fold_eval_affine_head(source_head)
    assert_no_foreign_gpu_processes()
    model.feature = source_head
    live_source = encode_query(model, transform, query_paths)[0]
    archive_query = source[list(query_rows)]
    source_parity_max_abs = float((live_source - archive_query).abs().max())
    if source_parity_max_abs > 1e-3:
        raise ValueError("L/14 live SOP source feature differs from gallery archive")
    model.feature = fused_head
    live_fused = encode_query(model, transform, query_paths)[0]
    fused_max_abs = float((live_source - live_fused).abs().max())
    source_codes = pack_int8_unit_embeddings(pca.apply(F.normalize(live_source, dim=1)))
    fused_codes = pack_int8_unit_embeddings(pca.apply(F.normalize(live_fused, dim=1)))
    query_code_changes = int((source_codes.codes != fused_codes.codes).sum())
    query_inverse_norm_changes = int(
        (source_codes.inverse_norms != fused_codes.inverse_norms).sum()
    )
    gallery_bytes = int(packed_gallery.codes.numel() + 2 * len(gallery_rows))
    if len(gallery_rows) < 10:
        raise ValueError("L/14 paired gallery inventory differs")
    with CutilePackedInt8Gallery.open_packed(args.native_library, packed_gallery) as gallery:
        gallery_torch_cuda_bytes = torch.cuda.memory_allocated()
        scalar_by_arm: dict[str, np.ndarray] = {}
        native_by_arm: dict[str, np.ndarray] = {}
        for arm, codes in (("original", source_codes), ("exact_fused", fused_codes)):
            scalar_by_arm[arm] = np.stack(
                [
                    scalar_packed_topk(
                        codes.codes.numpy()[row],
                        codes.inverse_norms.numpy()[row],
                        packed_gallery.codes.numpy(),
                        packed_gallery.inverse_norms.numpy(),
                    )
                    for row in range(32)
                ]
            )
            native_by_arm[arm] = gallery.search_packed(codes)[0]
            if not np.array_equal(native_by_arm[arm], scalar_by_arm[arm]):
                raise ValueError(f"L/14 {arm} native packed top-k differs from scalar oracle")
        top1_unchanged = bool(
            np.array_equal(native_by_arm["original"][:, 0], native_by_arm["exact_fused"][:, 0])
        )
        timing: dict[str, object] = {}
        for batch_size in (1, 32):
            paths = query_paths[:batch_size]
            for head in (source_head, fused_head):
                model.feature = head
                for _ in range(5):
                    call(model, transform, paths, pca, gallery)
            samples = {
                arm: {stage: [] for stage in STAGES} for arm in ("original", "exact_fused")
            }
            first_hash: dict[str, str] = {}
            order = []
            for block in range(args.blocks):
                arms = (("original", source_head), ("exact_fused", fused_head))
                if block % 2:
                    arms = tuple(reversed(arms))
                for name, head in arms:
                    order.append(name)
                    model.feature = head
                    for _ in range(args.calls_per_block):
                        durations, result_hash, top1 = call(model, transform, paths, pca, gallery)
                        if not np.array_equal(top1, native_by_arm[name][:batch_size, 0]):
                            raise ValueError("L/14 timed packed top-1 differs from oracle")
                        if name in first_hash and first_hash[name] != result_hash:
                            raise ValueError("L/14 packed top-k changed during replay")
                        first_hash[name] = result_hash
                        for stage, value in durations.items():
                            samples[name][stage].append(value)
            timing[str(batch_size)] = {
                "samples_ns": samples,
                "interleaved_order": order,
                "result_sha256": first_hash,
                "summary": {
                    name: {stage: timing_summary(values) for stage, values in columns.items()}
                    for name, columns in samples.items()
                },
                "paired_block_p50_ratio": paired_block_p50_ratio(
                    samples["original"]["image_to_topk_ns"],
                    samples["exact_fused"]["image_to_topk_ns"],
                ),
            }
            print(
                json.dumps(
                    {
                        "batch": batch_size,
                        "image_to_topk": {
                            name: timing[str(batch_size)]["summary"][name]["image_to_topk_ns"]
                            for name in samples
                        },
                    }
                ),
                flush=True,
            )
    final_query_manifest = b"".join(
        hashlib.sha256(path.read_bytes()).digest() for path in query_paths
    )
    if final_query_manifest != query_manifest:
        raise ValueError("L/14 SOP query image content changed")
    assert_no_foreign_gpu_processes()
    result = {
        "schema": "sfora-l14-exact-head-sop-train-image-to-topk-f0-v2",
        "claim_eligible": False,
        "p99_contract_satisfied": False,
        "split": (
            "SOP train identities, seed-179019 90/10 class holdout queries, "
            "shared train gallery excluding query images"
        ),
        "gallery_semantics": (
            "same original-encoder 130-byte gallery for both query heads; "
            "diagnostic asymmetric serving screen"
        ),
        "gallery_rows": len(gallery_rows),
        "gallery_bytes": gallery_bytes,
        "query_train_image_ids": [row.image_id for row in query_records],
        "query_image_manifest_sha256": hashlib.sha256(query_manifest).hexdigest(),
        "source_archive_live_max_abs_difference": source_parity_max_abs,
        "exact_fused_max_abs_difference": fused_max_abs,
        "query_code_coordinate_changes": query_code_changes,
        "query_inverse_norm_row_changes": query_inverse_norm_changes,
        "pca128_fit_seconds": pca_fit_seconds,
        "pca128_mean_sha256": hashlib.sha256(pca.mean.numpy().tobytes()).hexdigest(),
        "pca128_components_sha256": hashlib.sha256(pca.components.numpy().tobytes()).hexdigest(),
        "gallery_torch_cuda_allocated_bytes_excludes_native": gallery_torch_cuda_bytes,
        "scalar_topk_ordinals": {name: rows.tolist() for name, rows in scalar_by_arm.items()},
        "native_topk_ordinals": {name: rows.tolist() for name, rows in native_by_arm.items()},
        "top1_unchanged_32_queries": top1_unchanged,
        "advance": bool(
            top1_unchanged
            and all(
                timing[str(batch)]["paired_block_p50_ratio"]["upper_95"] < 1
                for batch in (1, 32)
            )
        ),
        "timing": timing,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "inputs": {
            "checkpoint_sha256": sha256(args.checkpoint),
            "sop_archive_sha256": SOP_ARCHIVE_SHA256,
            "native_library_sha256": NATIVE_LIBRARY_SHA256,
            "native_api_sha256": NATIVE_API_SHA256,
            "script_sha256": sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish_file_noreplace(
        args.output,
        lambda stream: stream.write(
            (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )
    print(json.dumps({"output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
