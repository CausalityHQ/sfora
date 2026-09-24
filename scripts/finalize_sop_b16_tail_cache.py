#!/usr/bin/env python3
"""Verify and publish a complete SOP tail cache left by a terminated session."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from build_sop_b16_tail_cache import ARCHIVE_SHA256, ROWS, TOKENS, WIDTH, sha256
from sop_teacher_anchored_runtime import load_authenticated_source_model

from sfora.unicom_tail_adapter import output_from_last_block_input

BUILD_SCRIPT_SHA256 = "bd0a21e263cef60fcf4b1c133c244fa0dab3096c1c55a6a984faf9eb196ff3f1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--partial-data", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--execute-sop-tail-cache-finalize", action="store_true", required=True)
    args = parser.parse_args()
    if (
        not args.partial_data.is_file()
        or args.partial_data.is_symlink()
        or args.output_data.exists()
        or args.output_data.is_symlink()
        or args.output_receipt.exists()
        or args.output_receipt.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(Path(__file__).with_name("build_sop_b16_tail_cache.py")) != BUILD_SCRIPT_SHA256
        or not torch.cuda.is_available()
    ):
        raise ValueError("SOP tail-cache recovery authority differs")
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    cache = np.load(args.partial_data, mmap_mode="r", allow_pickle=False)
    if cache.shape != (ROWS, TOKENS, WIDTH) or cache.dtype != np.float32:
        raise ValueError("SOP tail-cache recovery geometry differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        expected = np.asarray(archive["train_embeddings"], dtype=np.float32)
    if expected.shape != (ROWS, WIDTH):
        raise ValueError("SOP tail-cache source geometry differs")
    source = load_authenticated_source_model(args.unicom_checkout, args.checkpoint)
    model = source.encoder.cuda().eval()
    min_cosine = 1.0
    max_abs = 0.0
    with torch.inference_mode():
        for start in range(0, ROWS, 64):
            stop = min(start + 64, ROWS)
            block = np.asarray(cache[start:stop], dtype=np.float32)
            if not np.isfinite(block).all() or not np.all(np.any(block != 0, axis=(1, 2))):
                raise ValueError(f"SOP tail-cache partial data incomplete at row {start}")
            tokens = torch.from_numpy(block.copy()).cuda()
            outputs = output_from_last_block_input(model, tokens).float()
            source_rows = torch.from_numpy(expected[start:stop].copy()).cuda()
            cosine = torch.nn.functional.cosine_similarity(outputs, source_rows, dim=1)
            min_cosine = min(min_cosine, float(cosine.min()))
            max_abs = max(max_abs, float((outputs - source_rows).abs().max()))
            if stop % 6_400 == 0 or stop == ROWS:
                print(
                    json.dumps(
                        {"verified_rows": stop, "elapsed_seconds": time.perf_counter() - started}
                    ),
                    flush=True,
                )
    if min_cosine < 0.99999:
        raise ValueError("SOP tail-cache recovery replay differs")
    data_sha256 = sha256(args.partial_data)
    os.link(args.partial_data, args.output_data)
    args.partial_data.unlink()
    receipt = {
        "schema": "sfora-sop-b16-final-block-input-cache-recovered-v1",
        "claim_eligible": False,
        "build_session_exit_code": 143,
        "build_script_sha256": BUILD_SCRIPT_SHA256,
        "finalize_script_sha256": sha256(Path(__file__)),
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
        "min_archive_cosine": min_cosine,
        "max_archive_abs": max_abs,
        "finalization_seconds": time.perf_counter() - started,
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
    print(json.dumps({"data_sha256": data_sha256, "min_archive_cosine": min_cosine}), flush=True)


if __name__ == "__main__":
    main()
