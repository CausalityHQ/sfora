#!/usr/bin/env python3
"""Build an ordered float32 final-block-input cache from SOP train images only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch.utils.data import DataLoader, Dataset

from sfora.unicom_tail_adapter import last_block_input, output_from_last_block_input

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
ROWS = 59_551
TOKENS = 196
WIDTH = 768


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OrderedImages(Dataset):
    def __init__(self, root: Path, relatives: list[str], transform: object) -> None:
        if (
            not root.is_dir()
            or root.is_symlink()
            or len(relatives) != ROWS
            or not callable(transform)
        ):
            raise ValueError("SOP tail-cache image inventory differs")
        self.paths: list[Path] = []
        self.transform = transform
        for relative in relatives:
            pure = PurePosixPath(relative)
            if not pure.parts or pure.is_absolute() or ".." in pure.parts or str(pure) != relative:
                raise ValueError("SOP tail-cache image path differs")
            path = root.joinpath(*pure.parts)
            if not path.is_file() or path.is_symlink():
                raise ValueError("SOP tail-cache image path differs")
            self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--sop-root", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--execute-sop-tail-cache", action="store_true", required=True)
    args = parser.parse_args()
    partial = args.output_data.with_name(f".{args.output_data.name}.partial")
    if (
        args.output_data.exists()
        or args.output_data.is_symlink()
        or args.output_receipt.exists()
        or args.output_receipt.is_symlink()
        or partial.exists()
        or partial.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or not torch.cuda.is_available()
    ):
        raise ValueError("SOP tail-cache invocation differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.source_archive, allow_pickle=False) as archive:
        relatives = [str(x) for x in archive["train_relative_paths"]]
        source_embeddings = np.asarray(archive["train_embeddings"], dtype=np.float32)
    if source_embeddings.shape != (ROWS, WIDTH):
        raise ValueError("SOP tail-cache source geometry differs")
    source = load_authenticated_source_model(args.unicom_checkout, args.checkpoint)
    model = source.encoder.cuda().eval()
    dataset = OrderedImages(args.sop_root, relatives, source.transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(partial, mode="w+", dtype="<f4", shape=(ROWS, TOKENS, WIDTH))
    min_archive_cosine = 1.0
    max_archive_abs = 0.0
    processed = 0
    with torch.inference_mode():
        for batch, indexes in loader:
            index = np.asarray(indexes, dtype=np.int64)
            if not np.array_equal(index, np.arange(processed, processed + len(index))):
                raise ValueError("SOP tail-cache loader order differs")
            images = batch.cuda(non_blocking=True)
            tokens = last_block_input(model, images)
            if tokens.shape != (len(index), TOKENS, WIDTH) or tokens.dtype != torch.float32:
                raise ValueError("SOP tail-cache prefix geometry differs")
            outputs = output_from_last_block_input(model, tokens).float()
            expected = torch.from_numpy(source_embeddings[index].copy()).cuda()
            cosine = torch.nn.functional.cosine_similarity(outputs, expected, dim=1)
            min_archive_cosine = min(min_archive_cosine, float(cosine.min()))
            max_archive_abs = max(max_archive_abs, float((outputs - expected).abs().max()))
            cache[index] = tokens.cpu().numpy()
            processed += len(index)
            if processed % 6_400 == 0 or processed == ROWS:
                print(
                    json.dumps(
                        {"processed": processed, "elapsed_seconds": time.perf_counter() - started}
                    ),
                    flush=True,
                )
    if processed != ROWS or min_archive_cosine < 0.99999:
        raise ValueError("SOP tail-cache replay differs from authenticated source")
    cache.flush()
    del cache
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    data_sha256 = sha256(partial)
    os.link(partial, args.output_data)
    partial.unlink()
    receipt = {
        "schema": "sfora-sop-b16-final-block-input-cache-v1",
        "claim_eligible": False,
        "script_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/sfora/unicom_tail_adapter.py"
        ),
        "source_archive_sha256": ARCHIVE_SHA256,
        "checkpoint_sha256": source.checkpoint_sha256,
        "unicom_revision": source.revision,
        "rows": ROWS,
        "shape": [ROWS, TOKENS, WIDTH],
        "dtype": "float32",
        "data_path": str(args.output_data),
        "data_sha256": data_sha256,
        "data_bytes": args.output_data.stat().st_size,
        "min_archive_cosine": min_archive_cosine,
        "max_archive_abs": max_archive_abs,
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
    with args.output_receipt.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps({"data_sha256": data_sha256, "elapsed_seconds": receipt["elapsed_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
