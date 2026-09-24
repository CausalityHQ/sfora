#!/usr/bin/env python3
"""Paired SOP TRAIN image-to-top-10 cost screen for B/16 resolution variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from probe_l14_parallel_fold import load_authenticated_l14
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch.nn import functional as F
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.unicom_resolution_adapter import output_at_resolution

SOURCE_ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019
ARMS = ("b16_224", "b16_336_detail", "b16_336_upsampled", "l14_336")
STAGES = (
    "host_preprocess_ns",
    "host_to_device_ns",
    "encode_transfer_ns",
    "project_pack_ns",
    "search_ns",
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
        "mean_ms": float(values.mean()),
    }


def query_paths(
    archive: Path, root: Path
) -> tuple[list[Path], list[int], torch.Tensor, torch.Tensor]:
    if sha256(archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("SOP B/16 archive differs")
    with np.load(archive, allow_pickle=False) as data:
        labels = np.asarray(data["train_labels"], dtype=np.int64)
        ids = np.asarray(data["train_image_ids"], dtype=np.int64)
        relative = np.asarray(data["train_relative_paths"])
        embeddings = np.asarray(data["train_embeddings"], dtype=np.float32)
    if labels.shape != (59_551,) or embeddings.shape != (59_551, 768):
        raise ValueError("SOP B/16 TRAIN inventory differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    fit = torch.from_numpy(embeddings[list(partition.fit_row_indexes)].copy())
    gallery = torch.from_numpy(embeddings.copy())
    selected = np.linspace(0, len(partition.validation_row_indexes) - 1, 32, dtype=int)
    rows = [partition.validation_row_indexes[int(index)] for index in selected]
    paths = []
    for row in rows:
        pure = PurePosixPath(str(relative[row]))
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError("SOP B/16 TRAIN path differs")
        path = root.joinpath(*pure.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP B/16 TRAIN image is missing")
        paths.append(path)
    return paths, ids[rows].tolist(), fit, gallery


@torch.inference_mode()
def one_call(
    arm: str,
    paths: list[Path],
    source_model: torch.nn.Module,
    l14_model: torch.nn.Module,
    base_transform: object,
    detail_transform: object,
    l14_transform: object,
    pca: object,
    gallery: CutilePackedInt8Gallery,
) -> tuple[dict[str, int], str]:
    started = time.perf_counter_ns()
    transform = (
        base_transform
        if arm in ("b16_224", "b16_336_upsampled")
        else (l14_transform if arm == "l14_336" else detail_transform)
    )
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    host = torch.stack(tensors)
    host_ready = time.perf_counter_ns()
    images = host.cuda(non_blocking=False)
    if arm == "b16_336_upsampled":
        images = F.interpolate(images, size=(336, 336), mode="bicubic", align_corners=False)
    device_ready = time.perf_counter_ns()
    if arm == "l14_336":
        values = l14_model(images).float().cpu()
    else:
        values = output_at_resolution(source_model, images).float().cpu()
    encoded = time.perf_counter_ns()
    packed = pack_int8_unit_embeddings(pca.apply(F.normalize(values, dim=1)))
    packed_at = time.perf_counter_ns()
    ordinals, scores = gallery.search_packed(packed)
    finished = time.perf_counter_ns()
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "unicom-checkout",
        "b16-checkpoint",
        "l14-checkpoint",
        "source-archive",
        "sop-root",
        "native-library",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--b1-calls", type=int, default=30)
    parser.add_argument("--b32-calls", type=int, default=10)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or args.b1_calls < 10
        or args.b32_calls < 5
    ):
        raise ValueError("SOP B/16 resolution cost invocation differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths, ids, fit, gallery_source = query_paths(args.source_archive, args.sop_root)
    pca = fit_centered_pca(F.normalize(fit, dim=1), dimensions=128)
    packed_gallery = pack_int8_unit_embeddings(pca.apply(F.normalize(gallery_source, dim=1)))
    source = load_authenticated_source_model(args.unicom_checkout, args.b16_checkpoint)
    l14_model, l14_transform = load_authenticated_l14(args.unicom_checkout, args.l14_checkpoint)
    source_model = source.encoder.cuda().eval()
    l14_model = l14_model.cuda().eval()
    source_steps = source.transform.transforms
    if (
        len(source_steps) != 5
        or not isinstance(source_steps[0], Resize)
        or not isinstance(source_steps[1], CenterCrop)
    ):
        raise ValueError("UNICOM B/16 transform differs")
    detail_transform = Compose(
        [Resize(336, interpolation=InterpolationMode.BICUBIC), CenterCrop(336), *source_steps[2:]]
    )
    raw: dict[str, dict[str, dict[str, object]]] = {}
    torch.cuda.reset_peak_memory_stats()
    with CutilePackedInt8Gallery.open_packed(args.native_library, packed_gallery) as gallery:
        for batch_size, calls in ((1, args.b1_calls), (32, args.b32_calls)):
            selected_paths = paths[:batch_size]
            by_arm = {arm: {stage: [] for stage in STAGES} for arm in ARMS}
            hashes = {arm: [] for arm in ARMS}
            order = (*ARMS, *reversed(ARMS))
            for arm in ARMS:
                for _ in range(3):
                    one_call(
                        arm,
                        selected_paths,
                        source_model,
                        l14_model,
                        source.transform,
                        detail_transform,
                        l14_transform,
                        pca,
                        gallery,
                    )
            for repeat in range(calls):
                for arm in order[:4] if repeat % 2 == 0 else order[4:]:
                    durations, digest = one_call(
                        arm,
                        selected_paths,
                        source_model,
                        l14_model,
                        source.transform,
                        detail_transform,
                        l14_transform,
                        pca,
                        gallery,
                    )
                    for stage in STAGES:
                        by_arm[arm][stage].append(durations[stage])
                    hashes[arm].append(digest)
                if repeat % 5 == 4:
                    print(
                        json.dumps({"batch": batch_size, "calls_per_arm": repeat + 1}), flush=True
                    )
            raw[str(batch_size)] = {
                arm: {
                    "raw_ns": by_arm[arm],
                    "summary": {stage: summary(by_arm[arm][stage]) for stage in STAGES},
                    "result_hashes": hashes[arm],
                }
                for arm in ARMS
            }
    b1 = raw["1"]
    ratio = (
        b1["b16_336_detail"]["summary"]["image_to_top10_ns"]["p50_ms"]
        / b1["l14_336"]["summary"]["image_to_top10_ns"]["p50_ms"]
    )
    receipt = {
        "schema": "sfora-sop-b16-resolution-cost-screen-v1",
        "claim_eligible": False,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "b16_checkpoint_sha256": source.checkpoint_sha256,
        "l14_checkpoint_sha256": sha256(args.l14_checkpoint),
        "native_library_sha256": sha256(args.native_library),
        "script_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(Path(output_at_resolution.__code__.co_filename)),
        "sop_train_query_ids": ids,
        "sop_train_query_image_sha256": [sha256(path) for path in paths],
        "gallery_rows": len(gallery_source),
        "gallery_wire_bytes_per_row": 130,
        "gallery_source": "UNICOM B/16@224 archive, shared across arms for cost only",
        "projection_source": (
            "PCA-128 on 53,700 fit-only UNICOM B/16@224 descriptors; "
            "shared across arms for cost only"
        ),
        "b1_b16_detail_to_l14_ratio": ratio,
        "b1_cost_gate_pass": ratio <= 0.7,
        "timing": raw,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(
        json.dumps({"result": "completed", "ratio": ratio, "gate_pass": ratio <= 0.7}), flush=True
    )


if __name__ == "__main__":
    main()
