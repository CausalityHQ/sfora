#!/usr/bin/env python3
"""Extract B/16 resolution controls from authenticated SOP TRAIN images only."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import torch
from PIL import Image
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

from sfora.unicom_resolution_adapter import output_at_resolution

SOURCE_ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class SopResolutionImages(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        paths: list[Path],
        base_transform: Callable[[Image.Image], torch.Tensor],
        detail_transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.paths = paths
        self.base_transform = base_transform
        self.detail_transform = detail_transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB")
            return self.base_transform(rgb), self.detail_transform(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("unicom-checkout", "b16-checkpoint", "source-archive", "sop-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or args.output_dir.is_symlink()
        or args.batch_size != 32
        or args.workers != 4
        or not torch.cuda.is_available()
        or sha256(args.source_archive) != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("SOP B/16 resolution extract invocation differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.source_archive, allow_pickle=False) as archive:
        relatives = archive["train_relative_paths"].tolist()
        ids = archive["train_image_ids"].tolist()
        source_first = archive["train_embeddings"][:8].copy()
    if len(relatives) != 59_551 or len(ids) != len(relatives) or source_first.shape != (8, 768):
        raise ValueError("SOP TRAIN source inventory differs")
    paths: list[Path] = []
    root = args.sop_root.resolve()
    for item in relatives:
        pure = PurePosixPath(item)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise ValueError("SOP TRAIN image path differs")
        path = args.sop_root.joinpath(*pure.parts)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("SOP TRAIN image is missing")
        paths.append(path)
    source = load_authenticated_source_model(args.unicom_checkout, args.b16_checkpoint)
    steps = cast(Compose, source.transform).transforms
    if (
        len(steps) != 5
        or not isinstance(steps[0], Resize)
        or not isinstance(steps[1], CenterCrop)
        or steps[0].size != 224
        or steps[1].size != (224, 224)
        or steps[0].interpolation != InterpolationMode.BICUBIC
    ):
        raise ValueError("UNICOM B/16 source transform differs")
    detail_transform = Compose(
        [Resize(336, interpolation=InterpolationMode.BICUBIC), CenterCrop(336), *steps[2:]]
    )
    dataset = SopResolutionImages(paths, source.transform, detail_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detail_path = args.output_dir / "b16_336_detail.npy"
    control_path = args.output_dir / "b16_336_upsampled.npy"
    detail_array = np.lib.format.open_memmap(
        detail_path, mode="w+", dtype="float32", shape=(59_551, 768)
    )
    control_array = np.lib.format.open_memmap(
        control_path, mode="w+", dtype="float32", shape=(59_551, 768)
    )
    model = source.encoder.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    detail_seconds = 0.0
    control_seconds = 0.0
    source_parity_max_abs = None
    rows = 0
    with torch.inference_mode():
        for batch, (base, detail) in enumerate(loader, start=1):
            base_gpu = base.cuda(non_blocking=True)
            detail_gpu = detail.cuda(non_blocking=True)
            if batch == 1:
                replay = output_at_resolution(model, base_gpu[:8]).float().cpu().numpy()
                source_parity_max_abs = float(np.max(np.abs(replay - source_first)))
                if source_parity_max_abs > 1e-3:
                    raise ValueError("SOP TRAIN archived B/16 source parity differs")
            begin = time.perf_counter()
            resolved = output_at_resolution(model, detail_gpu).float().cpu().numpy()
            detail_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            control_gpu = F.interpolate(
                base_gpu, size=(336, 336), mode="bicubic", align_corners=False
            )
            control = output_at_resolution(model, control_gpu).float().cpu().numpy()
            control_seconds += time.perf_counter() - begin
            stop = rows + len(base)
            if resolved.shape != (len(base), 768) or control.shape != resolved.shape:
                raise ValueError("SOP B/16 resolution feature geometry differs")
            if not np.isfinite(resolved).all() or not np.isfinite(control).all():
                raise ValueError("SOP B/16 resolution feature nonfinite")
            detail_array[rows:stop] = resolved
            control_array[rows:stop] = control
            rows = stop
            if batch == 1 or batch % 100 == 0 or rows == len(dataset):
                print(
                    json.dumps(
                        {
                            "rows": rows,
                            "total": len(dataset),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    if rows != len(dataset) or source_parity_max_abs is None:
        raise ValueError("SOP B/16 resolution extraction incomplete")
    detail_array.flush()
    control_array.flush()
    del detail_array, control_array
    receipt = {
        "schema": "sfora-sop-b16-resolution-train-extract-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN only; 59,551 ordered images, 11,318 products",
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "source_revision": source.revision,
        "script_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(Path(output_at_resolution.__code__.co_filename)),
        "train_image_ids_sha256": hashlib.sha256(
            np.asarray(ids, dtype=np.int64).tobytes()
        ).hexdigest(),
        "source_224_parity_max_abs": source_parity_max_abs,
        "features": {
            name: {"file": path.name, "sha256": sha256(path), "rows": rows, "width": 768}
            for name, path in (("b16_336_detail", detail_path), ("b16_336_upsampled", control_path))
        },
        "detail_encode_transfer_seconds": detail_seconds,
        "control_upsample_encode_transfer_seconds": control_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    with (args.output_dir / "extract-v1.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"result": "completed", "rows": rows, "total_seconds": receipt["total_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
