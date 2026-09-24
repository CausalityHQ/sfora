#!/usr/bin/env python3
"""Paired live SOP TRAIN image-to-top-10 timing for pretrained L-class substrates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image
from probe_l14_parallel_fold import load_authenticated_l14
from score_sop_pretrained_substrate import fit_project_pack, sha256
from torch.nn import functional as F

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.packed_int8_search import CpuPackedInt8Gallery
from sfora.representation_ceiling import deterministic_class_partition

ARMS = ("unicom_l14_336", "siglip2_l16_256")
ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
MODEL_HASHES = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}
STAGES = (
    "host_preprocess_ns",
    "host_to_device_ns",
    "encode_transfer_ns",
    "project_pack_ns",
    "native_search_ns",
    "image_to_top10_ns",
)


def alternating_pair_order(repeat: int) -> tuple[str, str]:
    if type(repeat) is not int or repeat < 0:
        raise ValueError("SOP paired timing repeat differs")
    return ARMS if repeat % 2 == 0 else ARMS[::-1]


def verify_live_features(live: torch.Tensor, cached: torch.Tensor) -> float:
    """Require live preprocessing/model output to reproduce cached features."""

    if (
        type(live) is not torch.Tensor
        or type(cached) is not torch.Tensor
        or live.dtype != torch.float32
        or cached.dtype != torch.float32
        or live.shape != cached.shape
        or live.ndim != 2
        or len(live) < 1
        or not bool(torch.isfinite(live).all())
        or not bool(torch.isfinite(cached).all())
    ):
        raise ValueError("SOP live feature parity differs")
    minimum = float(F.cosine_similarity(live, cached, dim=1).min())
    if minimum < 0.999:
        raise ValueError(f"SOP live feature parity differs: {minimum:.6f}")
    return minimum


def summary(raw: list[int]) -> dict[str, float]:
    values = np.asarray(raw, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(values.mean()),
    }


@torch.inference_mode()
def encode_live(arm: dict[str, Any], paths: list[Path]) -> tuple[torch.Tensor, int, int, int]:
    started = time.perf_counter_ns()
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if arm["name"] == "unicom_l14_336":
        pixels = torch.stack([arm["transform"](image) for image in images])
        host_ready = time.perf_counter_ns()
        device = pixels.cuda(non_blocking=False)
        device_ready = time.perf_counter_ns()
        features = arm["model"](device).float().cpu()
    else:
        batch = arm["processor"](images=images, return_tensors="pt")
        host_ready = time.perf_counter_ns()
        tensors = {key: value.cuda() for key, value in batch.items() if torch.is_tensor(value)}
        device_ready = time.perf_counter_ns()
        output = arm["model"].get_image_features(**tensors)
        features = (
            output.float().cpu()
            if isinstance(output, torch.Tensor)
            else output.pooler_output.float().cpu()
        )
    encoded = time.perf_counter_ns()
    if features.shape != (len(paths), arm["width"]) or not bool(torch.isfinite(features).all()):
        raise ValueError("SOP live encoder feature geometry differs")
    return features, host_ready - started, device_ready - host_ready, encoded - device_ready


def one_call(arm: dict[str, Any], paths: list[Path]) -> tuple[dict[str, int], str]:
    started = time.perf_counter_ns()
    features, host_ns, transfer_ns, encode_ns = encode_live(arm, paths)
    encoded = time.perf_counter_ns()
    projected = arm["pca"].apply(F.normalize(features, dim=1))
    packed = pack_int8_unit_embeddings(projected)
    packed_at = time.perf_counter_ns()
    ordinals, scores = arm["gallery"].search_packed(packed)
    torch.cuda.synchronize()
    finished = time.perf_counter_ns()
    if len(ordinals) != len(paths) or ordinals.shape[1] != 10:
        raise ValueError("SOP live packed top-10 geometry differs")
    return (
        dict(
            zip(
                STAGES,
                (
                    host_ns,
                    transfer_ns,
                    encode_ns,
                    packed_at - encoded,
                    finished - packed_at,
                    finished - started,
                ),
                strict=True,
            )
        ),
        hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest(),
    )


def verify_native_top10(
    arm: dict[str, Any], paths: list[Path], packed_gallery: PackedInt8Embeddings
) -> None:
    features, _a, _b, _c = encode_live(arm, paths)
    query = pack_int8_unit_embeddings(arm["pca"].apply(F.normalize(features, dim=1)))
    actual_ordinals, actual_scores = arm["gallery"].search_packed(query)
    reference = CpuPackedInt8Gallery.open_packed(packed_gallery)
    expected_ordinals, expected_scores = reference.search_packed(query)
    if not np.array_equal(actual_ordinals, expected_ordinals) or not np.allclose(
        actual_scores, expected_scores, rtol=0, atol=1e-5
    ):
        raise ValueError("SOP live native top-10 differs from scalar reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "unicom-checkout",
        "l14-checkpoint",
        "unicom-l14-archive",
        "candidate-dir",
        "score-dir",
        "model-snapshot",
        "dataset-root",
        "native-library",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--b1-blocks", type=int, default=50)
    parser.add_argument("--b32-blocks", type=int, default=20)
    args = parser.parse_args()
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.b1_blocks < 10
        or args.b32_blocks < 10
        or not torch.cuda.is_available()
        or sha256(args.unicom_l14_archive) != ARCHIVE_SHA256
        or sha256(args.native_library) != NATIVE_SHA256
        or not tileiras
        or not Path(tileiras).is_file()
    ):
        raise ValueError("SOP live substrate timing source authority differs")
    score = json.loads((args.score_dir / "receipt.json").read_text())
    export = json.loads((args.candidate_dir / "receipt.json").read_text())
    if (
        score.get("schema") != "sfora-sop-pretrained-substrate-screen-v1"
        or score.get("candidate_feature_sha256") != export.get("features_sha256")
        or score.get("source_archive_sha256") != ARCHIVE_SHA256
        or sha256(args.candidate_dir / "train_features.npy") != export.get("features_sha256")
        or export.get("model_revision") != "787800c8990e6f058423089178e718139608408c"
        or args.model_snapshot.resolve().name != export.get("model_revision")
        or export.get("model_file_sha256") != MODEL_HASHES
        or any(sha256(args.model_snapshot / name) != value for name, value in MODEL_HASHES.items())
    ):
        raise ValueError("SOP live substrate quality or model authority differs")
    with np.load(args.unicom_l14_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
        reference_features = np.asarray(archive["train_embeddings"], dtype=np.float32)
    candidate_features = np.load(args.candidate_dir / "train_features.npy", mmap_mode="r")
    if (
        labels.shape != (59_551,)
        or reference_features.shape != (59_551, 768)
        or candidate_features.shape != (59_551, 1024)
    ):
        raise ValueError("SOP live substrate TRAIN feature inventory differs")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    selected = np.linspace(0, len(held_rows) - 1, 32, dtype=int)
    query_rows = held_rows[selected]
    paths = []
    for row in query_rows:
        relative = PurePosixPath(str(relatives[row]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("SOP live substrate path differs")
        path = args.dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP live substrate image differs")
        paths.append(path)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    source = {
        "unicom_l14_336": reference_features,
        "siglip2_l16_256": candidate_features,
    }
    fitted = {name: fit_project_pack(values, fit_rows) for name, values in source.items()}
    for name in ARMS:
        arm_receipt = json.loads((args.score_dir / f"{name}.json").read_text())
        actual = hashlib.sha256(
            fitted[name].pca.mean.numpy().tobytes() + fitted[name].pca.components.numpy().tobytes()
        ).hexdigest()
        if actual != arm_receipt["pca_sha256"]:
            raise ValueError(f"SOP live {name} PCA differs from quality screen")
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    if transformers.__version__ != export["hardware"]["transformers"]:
        raise ValueError("SOP live SigLIP2 processor version differs from export")

    l14_model, l14_transform = load_authenticated_l14(args.unicom_checkout, args.l14_checkpoint)
    l14_model = l14_model.cuda().eval()
    candidate_processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    if (
        type(candidate_processor).__name__ != "SiglipImageProcessor"
        or candidate_processor.size["height"] != 256
        or candidate_processor.size["width"] != 256
        or candidate_processor.resample != 2
    ):
        raise ValueError("SOP live SigLIP2 processor differs from export")
    candidate_model = (
        AutoModel.from_pretrained(
            args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    raw: dict[str, Any] = {}
    parity: dict[str, float] = {}
    batch1_parity: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats()
    with ExitStack() as stack:
        galleries = {
            name: stack.enter_context(
                CutilePackedInt8Gallery.open_packed(args.native_library, fitted[name].packed)
            )
            for name in ARMS
        }
        arms: dict[str, dict[str, Any]] = {
            "unicom_l14_336": {
                "name": "unicom_l14_336",
                "model": l14_model,
                "transform": l14_transform,
                "pca": fitted["unicom_l14_336"].pca,
                "gallery": galleries["unicom_l14_336"],
                "width": 768,
            },
            "siglip2_l16_256": {
                "name": "siglip2_l16_256",
                "model": candidate_model,
                "processor": candidate_processor,
                "pca": fitted["siglip2_l16_256"].pca,
                "gallery": galleries["siglip2_l16_256"],
                "width": 1024,
            },
        }
        for name in ARMS:
            features, _a, _b, _c = encode_live(arms[name], paths)
            cached = torch.from_numpy(np.asarray(source[name][query_rows]).copy())
            parity[name] = verify_live_features(features, cached)
            live_packed = pack_int8_unit_embeddings(
                fitted[name].pca.apply(F.normalize(features, dim=1))
            )
            if not torch.equal(
                live_packed.codes, fitted[name].packed.codes[query_rows]
            ) or not torch.equal(
                live_packed.inverse_norms, fitted[name].packed.inverse_norms[query_rows]
            ):
                raise ValueError(f"SOP live {name} packed query differs from quality export")
            verify_native_top10(arms[name], paths, fitted[name].packed)
            one_feature, _a, _b, _c = encode_live(arms[name], paths[:1])
            one_cached = torch.from_numpy(np.asarray(source[name][query_rows[:1]]).copy())
            batch1_parity[name] = verify_live_features(one_feature, one_cached)
            one_packed = pack_int8_unit_embeddings(
                fitted[name].pca.apply(F.normalize(one_feature, dim=1))
            )
            if not torch.equal(
                one_packed.codes, fitted[name].packed.codes[query_rows[:1]]
            ) or not torch.equal(
                one_packed.inverse_norms, fitted[name].packed.inverse_norms[query_rows[:1]]
            ):
                raise ValueError(
                    f"SOP live {name} batch-1 packed query differs from quality export"
                )
            verify_native_top10(arms[name], paths[:1], fitted[name].packed)
        for batch_size, blocks in ((1, args.b1_blocks), (32, args.b32_blocks)):
            selected_paths = paths[:batch_size]
            for name in ARMS:
                for _ in range(3):
                    one_call(arms[name], selected_paths)
            samples: dict[str, dict[str, list[int]]] = {
                name: {stage: [] for stage in STAGES} for name in ARMS
            }
            hashes: dict[str, str | None] = {name: None for name in ARMS}
            for repeat in range(blocks):
                for name in alternating_pair_order(repeat):
                    durations, digest = one_call(arms[name], selected_paths)
                    if hashes[name] is None:
                        hashes[name] = digest
                    elif hashes[name] != digest:
                        raise ValueError(f"SOP live {name} result changed during replay")
                    for stage in STAGES:
                        samples[name][stage].append(durations[stage])
                if repeat % 10 == 9:
                    print(json.dumps({"batch": batch_size, "blocks": repeat + 1}), flush=True)
            raw[str(batch_size)] = {
                name: {
                    "raw_ns": samples[name],
                    "summary": {stage: summary(samples[name][stage]) for stage in STAGES},
                    "result_sha256": hashes[name],
                    "warmups": 3,
                    "blocks": blocks,
                    "throughput_queries_per_second": batch_size
                    * 1e9
                    / float(np.mean(samples[name]["image_to_top10_ns"])),
                }
                for name in ARMS
            }
    ratios = {
        batch: raw[batch]["siglip2_l16_256"]["summary"]["image_to_top10_ns"]["p50_ms"]
        / raw[batch]["unicom_l14_336"]["summary"]["image_to_top10_ns"]["p50_ms"]
        for batch in ("1", "32")
    }
    receipt = {
        "schema": "sfora-sop-siglip2-vs-unicom-live-timing-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout query images only",
        "query_image_ids": ids[query_rows].tolist(),
        "query_image_sha256": [sha256(path) for path in paths],
        "source_archive_sha256": ARCHIVE_SHA256,
        "candidate_export_receipt_sha256": sha256(args.candidate_dir / "receipt.json"),
        "quality_receipt_sha256": sha256(args.score_dir / "receipt.json"),
        "l14_checkpoint_sha256": sha256(args.l14_checkpoint),
        "model_file_sha256": export["model_file_sha256"],
        "processor": {
            "class": type(candidate_processor).__name__,
            "backend": "torchvision",
            "size": 256,
            "resample": int(candidate_processor.resample),
            "transformers": transformers.__version__,
        },
        "model_parameter_dtype": {
            "unicom_l14_336": sorted({str(p.dtype) for p in l14_model.parameters()}),
            "siglip2_l16_256": sorted({str(p.dtype) for p in candidate_model.parameters()}),
        },
        "native_library_sha256": sha256(args.native_library),
        "tileiras_sha256": sha256(Path(tileiras)),
        "source_sha256": {
            "benchmark": sha256(Path(__file__)),
            "cutile_api": sha256(Path(__file__).resolve().parents[1] / "src/sfora/cutile_int8.py"),
        },
        "gallery_rows": len(labels),
        "gallery_wire_bytes_per_row": 130,
        "live_feature_min_cosine": parity,
        "batch1_feature_min_cosine": batch1_parity,
        "timing": raw,
        "p50_candidate_to_reference_ratio": ratios,
        "p50_screen_gate_pass": all(ratio <= 0.85 for ratio in ratios.values()),
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"ratio": ratios, "gate_pass": receipt["p50_screen_gate_pass"]}), flush=True)


if __name__ == "__main__":
    main()
