#!/usr/bin/env python3
"""Check that cached B/16 prefix tokens replay the authenticated train encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sop_teacher_anchored_runtime import load_authenticated_source_model

from sfora.unicom_tail_adapter import last_block_input, output_from_last_block_input

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
ROWS = 8


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--sop-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP token cache source/output authority differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    source = load_authenticated_source_model(args.unicom_checkout, args.checkpoint)
    model = source.encoder.cuda().eval()
    with np.load(args.source_archive, allow_pickle=False) as archive:
        paths = [str(x) for x in archive["train_relative_paths"][:ROWS]]
        archived = np.asarray(archive["train_embeddings"][:ROWS], dtype=np.float32)
    images = []
    for relative in paths:
        with Image.open(args.sop_root / relative) as image:
            images.append(source.transform(image.convert("RGB")))
    batch = torch.stack(images).cuda()
    with torch.inference_mode():
        original = model(batch)
        prefix = last_block_input(model, batch)
        replayed = output_from_last_block_input(model, prefix)
        half_replayed = output_from_last_block_input(model, prefix.half())
    reference = torch.from_numpy(archived).cuda()
    normalized_original = torch.nn.functional.normalize(original.float(), dim=1)
    normalized_reference = torch.nn.functional.normalize(reference, dim=1)
    receipt = {
        "schema": "sfora-sop-b16-tail-cache-preflight-v1",
        "claim_eligible": False,
        "source_archive_sha256": ARCHIVE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/sfora/unicom_tail_adapter.py"
        ),
        "checkpoint_sha256": source.checkpoint_sha256,
        "unicom_revision": source.revision,
        "train_image_paths": paths,
        "rows": ROWS,
        "prefix_shape": list(prefix.shape),
        "prefix_dtype": str(prefix.dtype),
        "prefix_bytes_all_sop_train_f16_estimate": 59_551 * prefix.shape[1] * prefix.shape[2] * 2,
        "original_vs_replay_max_abs": float((original - replayed).abs().max()),
        "original_vs_f16_replay_max_abs": float((original - half_replayed).abs().max()),
        "original_vs_archive_min_cosine": float(
            (normalized_original * normalized_reference).sum(dim=1).min()
        ),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
