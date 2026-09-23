#!/usr/bin/env python3
"""Exploratory paired SOP image-to-top-10 timing for OML and pretrained UNICOM B/16."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import resource
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive
from PIL import Image
from torch.nn import functional as F

import sfora.cutile_int8 as cutile_int8_module
from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.model_profiles import load_oml_sop_compact_encoder
from sfora.representation_ceiling import fit_centered_pca

OML_CHECKPOINT_SHA256 = "2701830538f31bd2dabb06622475cc889b1095580fc57218d0293e7a6bce53a7"
OML_FEATURES_SHA256 = "8f565027b20e55923826a5240d97c564171428a5f1cc7290e680d2223fd28da4"
UNICOM_CHECKPOINT_SHA256 = "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef"
UNICOM_FEATURES_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
NATIVE_LIBRARY_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
NATIVE_API_SHA256 = "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409"
QUALITY_SCREEN_SHA256 = "a7c65b7b5dda1a8884f08ea384f150ef98ac20c77607c0b8b2ed2fbbe6c1053d"
TEST_IMAGE_MANIFEST_SHA256 = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def percentile(samples: list[int], fraction: float) -> int:
    return sorted(samples)[math.ceil(len(samples) * fraction) - 1]


def verify_query_images(paths: tuple[Path, ...], manifest: bytes) -> None:
    """Check the query pixels against official SOP metadata-order digests."""
    if len(manifest) != 60_502 * 32 or len(paths) != 32:
        raise ValueError("paired SOP query image manifest differs")
    for index, path in enumerate(paths):
        if hashlib.sha256(path.read_bytes()).digest() != manifest[index * 32 : (index + 1) * 32]:
            raise ValueError("paired SOP query image content differs")


def _load_unicom(checkout: Path, checkpoint: Path):
    package_root = checkout / "unicom"
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    model, transform = unicom.load("ViT-B/16", download_root=str(checkpoint.parent))
    return model.cuda().eval(), transform


def _load_oml(checkpoint: Path):
    from oml.models.vit_dino.extractor import ViTExtractor
    from oml.transforms.images.torchvision import get_normalisation_resize_hypvit

    model = ViTExtractor(
        weights=str(checkpoint), arch="vits16", normalise_features=True, use_multi_scale=False
    )
    return model.cuda().eval(), get_normalisation_resize_hypvit(im_size=224, crop_size=224)


def _call(arm: dict[str, object], paths: tuple[Path, ...], gallery: CutilePackedInt8Gallery):
    started = time.perf_counter_ns()
    tensors = []
    transform = arm["transform"]
    for path in paths:
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))  # type: ignore[operator]
    images = torch.stack(tensors).cuda(non_blocking=False)
    decoded = time.perf_counter_ns()
    with torch.inference_mode():
        features = arm["model"](images).float().cpu()  # type: ignore[operator]
    encoded = time.perf_counter_ns()
    if arm["name"] == "unicom_b16":
        projected = arm["head"].apply(F.normalize(features, dim=1))  # type: ignore[union-attr]
    else:
        projected = arm["head"].transform(features)  # type: ignore[union-attr]
    packed = pack_int8_unit_embeddings(projected)
    packed_at = time.perf_counter_ns()
    ordinals, scores = gallery.search(packed.codes.numpy(), packed.inverse_norms.numpy())
    finished = time.perf_counter_ns()
    return (
        {
            "decode_preprocess_ns": decoded - started,
            "encoder_transfer_ns": encoded - decoded,
            "project_pack_ns": packed_at - encoded,
            "native_search_ns": finished - packed_at,
            "image_to_topk_ns": finished - started,
        },
        hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest(),
    )


def _measure(
    arm: dict[str, object], paths: tuple[Path, ...], gallery: CutilePackedInt8Gallery, calls: int
) -> dict[str, object]:
    label = f"{arm['name']} batch={len(paths)}"
    print(
        f"{label}: warmup started (first call may compile CUDA kernels)",
        file=sys.stderr,
        flush=True,
    )
    for _ in range(5):
        _call(arm, paths, gallery)
    print(f"{label}: warmup completed; {calls} timed calls started", file=sys.stderr, flush=True)
    samples: dict[str, list[int]] = {
        stage: []
        for stage in (
            "decode_preprocess_ns",
            "encoder_transfer_ns",
            "project_pack_ns",
            "native_search_ns",
            "image_to_topk_ns",
        )
    }
    first_hash = None
    for _ in range(calls):
        durations, result_hash = _call(arm, paths, gallery)
        if first_hash is None:
            first_hash = result_hash
        elif result_hash != first_hash:
            raise ValueError("SOP image-to-top-k result changed during paired replay")
        for stage, elapsed in durations.items():
            samples[stage].append(elapsed)
    print(f"{label}: timed calls completed", file=sys.stderr, flush=True)
    return {
        "name": arm["name"],
        "batch": len(paths),
        "calls": calls,
        "warmups": 5,
        "first_result_sha256": first_hash,
        "samples_ns": samples,
        "p50_ns": {name: percentile(series, 0.5) for name, series in samples.items()},
        "p95_ns": {name: percentile(series, 0.95) for name, series in samples.items()},
        "p99_ns": {name: percentile(series, 0.99) for name, series in samples.items()},
        "queries_per_second": len(paths) * 1e9 / (sum(samples["image_to_topk_ns"]) / calls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--oml-features", required=True, type=Path)
    parser.add_argument("--oml-checkpoint", required=True, type=Path)
    parser.add_argument("--unicom-features", required=True, type=Path)
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--unicom-checkpoint", required=True, type=Path)
    parser.add_argument("--native-library", required=True, type=Path)
    parser.add_argument("--quality-screen", required=True, type=Path)
    parser.add_argument("--test-image-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calls", type=int, default=200)
    parser.add_argument("--execute-paired-replay", action="store_true", required=True)
    args = parser.parse_args()
    if args.calls < 50 or args.output.exists() or not torch.cuda.is_available():
        raise ValueError("SOP image-to-top-k benchmark authority differs")
    expected_files = (
        (args.oml_features, OML_FEATURES_SHA256),
        (args.oml_checkpoint, OML_CHECKPOINT_SHA256),
        (args.unicom_features, UNICOM_FEATURES_SHA256),
        (args.unicom_checkpoint, UNICOM_CHECKPOINT_SHA256),
        (args.native_library, NATIVE_LIBRARY_SHA256),
        (Path(cutile_int8_module.__file__), NATIVE_API_SHA256),
        (args.quality_screen, QUALITY_SCREEN_SHA256),
        (args.test_image_manifest, TEST_IMAGE_MANIFEST_SHA256),
    )
    for path, digest in expected_files:
        if sha256(path) != digest:
            raise ValueError(f"SOP benchmark input differs: {path}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    source = load_sop_embedding_archive(args.unicom_features)
    with np.load(args.oml_features, allow_pickle=False) as archive:
        oml_test = torch.from_numpy(np.ascontiguousarray(archive["test_features"]))
        oml_ids = np.ascontiguousarray(archive["test_ids"])
    if not np.array_equal(oml_ids, source["test_image_ids"]):
        raise ValueError("paired SOP image row identities differ")
    test_paths = tuple(
        args.dataset_root / relative for relative in source["test_relative_paths"][:32]
    )
    verify_query_images(test_paths, args.test_image_manifest.read_bytes())
    oml_head = load_oml_sop_compact_encoder()
    oml_gallery = pack_int8_unit_embeddings(oml_head.transform(oml_test))
    train = F.normalize(torch.from_numpy(source["train_embeddings"]).float(), dim=1)
    pca = fit_centered_pca(train.contiguous(), dimensions=128)
    b16_test = F.normalize(torch.from_numpy(source["test_embeddings"]).float(), dim=1)
    b16_gallery = pack_int8_unit_embeddings(pca.apply(b16_test.contiguous()))
    screen = json.loads(args.quality_screen.read_text())
    if (
        hashlib.sha256(b16_gallery.codes.numpy().tobytes()).hexdigest()
        != screen["pca_packed"]["gallery_code_sha256"]
        or hashlib.sha256(b16_gallery.inverse_norms.numpy().tobytes()).hexdigest()
        != screen["pca_packed"]["gallery_inverse_norm_sha256"]
    ):
        raise ValueError("B/16 paired gallery differs from authenticated quality screen")
    oml_model, oml_transform = _load_oml(args.oml_checkpoint)
    b16_model, b16_transform = _load_unicom(args.unicom_checkout, args.unicom_checkpoint)
    arms = {
        "oml": {"name": "oml", "model": oml_model, "transform": oml_transform, "head": oml_head},
        "unicom_b16": {
            "name": "unicom_b16",
            "model": b16_model,
            "transform": b16_transform,
            "head": pca,
        },
    }
    results = []
    with ExitStack() as stack:
        galleries = {
            "oml": stack.enter_context(
                CutilePackedInt8Gallery.open(
                    args.native_library,
                    oml_gallery.codes.numpy()[32:].copy(),
                    oml_gallery.inverse_norms.numpy()[32:].copy(),
                )
            ),
            "unicom_b16": stack.enter_context(
                CutilePackedInt8Gallery.open(
                    args.native_library,
                    b16_gallery.codes.numpy()[32:].copy(),
                    b16_gallery.inverse_norms.numpy()[32:].copy(),
                )
            ),
        }
        for pair, order in enumerate((("oml", "unicom_b16"), ("unicom_b16", "oml")), 1):
            for name in order:
                for batch in (1, 32):
                    results.append(
                        {
                            "pair": pair,
                            **_measure(arms[name], test_paths[:batch], galleries[name], args.calls),
                        }
                    )
    receipt = {
        "schema": "sfora-sop-image-to-topk-pair-v1",
        "claim_eligible": False,
        "split": "already-observed official SOP test; first 32 queries; gallery rows 32:60502",
        "gallery_rows": 60_470,
        "hardware": {
            "gpu": torch.cuda.get_device_name(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "inputs": {str(path): digest for path, digest in expected_files},
        "oml_head_sha256": oml_head.sha256,
        "script_sha256": sha256(Path(__file__)),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rows": results,
    }
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps([{"pair": r["pair"], "name": r["name"], "batch": r["batch"]} for r in results])
    )


if __name__ == "__main__":
    main()
