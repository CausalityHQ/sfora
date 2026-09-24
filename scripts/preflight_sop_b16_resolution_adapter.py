#!/usr/bin/env python3
"""Eight-image authenticated SOP TRAIN preflight for UNICOM B/16 resolution."""

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
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch.nn import functional as F
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

from sfora.unicom_resolution_adapter import output_at_resolution

SOURCE_ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def train_image_paths(archive: Path, root: Path) -> tuple[list[Path], np.ndarray, list[int]]:
    if sha256(archive) != SOURCE_ARCHIVE_SHA256:
        raise ValueError("SOP source archive differs")
    with np.load(archive, allow_pickle=False) as arrays:
        relative = arrays["train_relative_paths"][:8].tolist()
        ids = arrays["train_image_ids"][:8].tolist()
        reference = arrays["train_embeddings"][:8].copy()
    if reference.shape != (8, 768) or len(set(ids)) != 8:
        raise ValueError("SOP TRAIN preflight geometry differs")
    paths = []
    for item in relative:
        pure = PurePosixPath(item)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError("SOP TRAIN path differs")
        path = root.joinpath(*pure.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP TRAIN image is missing")
        paths.append(path)
    return paths, reference, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--sop-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not torch.cuda.is_available():
        raise ValueError("B/16 resolution preflight output or CUDA differs")
    started = time.perf_counter()
    paths, reference, ids = train_image_paths(args.source_archive, args.sop_root)
    loaded = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    source_transform = loaded.transform
    steps = source_transform.transforms
    if (
        len(steps) != 5
        or not isinstance(steps[0], Resize)
        or not isinstance(steps[1], CenterCrop)
        or steps[0].size != 224
        or steps[1].size != (224, 224)
        or steps[0].interpolation != InterpolationMode.BICUBIC
    ):
        raise ValueError("UNICOM source transform differs")
    detail_transform = Compose(
        [Resize(336, interpolation=InterpolationMode.BICUBIC), CenterCrop(336), *steps[2:]]
    )
    base_images = []
    detail_images = []
    image_sha = []
    for path in paths:
        image_sha.append(sha256(path))
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            base_images.append(source_transform(rgb))
            detail_images.append(detail_transform(rgb))
    native = torch.stack(base_images).cuda()
    detail = torch.stack(detail_images).cuda()
    upsampled = F.interpolate(native, size=(336, 336), mode="bicubic", align_corners=False)
    model = loaded.encoder.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        direct = model(native)
        adapted = output_at_resolution(model, native)
        resolved = output_at_resolution(model, detail)
        control = output_at_resolution(model, upsampled)
    if not torch.equal(direct, adapted):
        raise ValueError("224-pixel B/16 adapter parity differs")
    if any(not bool(torch.isfinite(value).all()) for value in (direct, resolved, control)):
        raise ValueError("B/16 resolution output is nonfinite")
    direct_np = direct.float().cpu().numpy()
    archive_error = float(np.max(np.abs(direct_np - reference)))
    receipt = {
        "schema": "sfora-sop-b16-resolution-preflight-v1",
        "claim_eligible": False,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_checkpoint_sha256": loaded.checkpoint_sha256,
        "source_revision": loaded.revision,
        "script_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(Path(output_at_resolution.__code__.co_filename)),
        "train_image_ids": ids,
        "train_image_sha256": image_sha,
        "source_transform": repr(source_transform),
        "detail_transform": repr(detail_transform),
        "control_transform": "F.interpolate(canonical224,336,bicubic,align_corners=False)",
        "exact_224_parity": True,
        "archive_max_abs_error": archive_error,
        "output_shape": list(direct.shape),
        "mean_cosine_224_vs_336": float(F.cosine_similarity(direct, resolved).mean().item()),
        "mean_cosine_224_vs_upsampled": float(F.cosine_similarity(direct, control).mean().item()),
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
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
