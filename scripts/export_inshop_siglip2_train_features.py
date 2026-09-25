#!/usr/bin/env python3
"""Cache the pinned SigLIP2 pretrained image tower on official In-Shop TRAIN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from export_sop_siglip2_train import MODEL_FILES, MODEL_REVISION, export_features, sha256

from sfora.unicom_inshop import parse_inshop_partition

PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
MODEL_HASHES = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or tuple(MODEL_FILES) != tuple(MODEL_HASHES)
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
        or not torch.cuda.is_available()
    ):
        raise ValueError("In-Shop SigLIP2 train cache authority differs")
    records = parse_inshop_partition(args.dataset_root)
    rows = tuple(row for row in records if row.split == "train")
    if len(rows) != 25_882 or len({row.label for row in rows}) != 3_997:
        raise ValueError("In-Shop SigLIP2 TRAIN rows differ")
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoImageProcessor.from_pretrained(args.model_snapshot, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    @torch.inference_mode()
    def encode(batch) -> np.ndarray:
        images = []
        for row in batch:
            with Image.open(row.image_path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        tensors = {name: value.cuda() for name, value in inputs.items() if torch.is_tensor(value)}
        output = model.get_image_features(**tensors)
        features = output if isinstance(output, torch.Tensor) else output.pooler_output
        return features.float().cpu().numpy()

    export_features(rows, encode, args.output_dir / "train_features.npy", width=1024, batch_size=32)
    torch.cuda.synchronize()
    receipt = {
        "schema": "sfora-inshop-siglip2-train-feature-export-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN only; all 25,882 rows in partition order",
        "rows": len(rows),
        "products": len({row.label for row in rows}),
        "width": 1024,
        "batch_size": 32,
        "partition_sha256": PARTITION_SHA256,
        "model_revision": MODEL_REVISION,
        "model_file_sha256": MODEL_HASHES,
        "ordered_rows_sha256": hashlib.sha256(
            "\n".join(
                f"{row.label}\0{row.image_path.relative_to(args.dataset_root)}" for row in rows
            ).encode()
        ).hexdigest(),
        "features_sha256": sha256(args.output_dir / "train_features.npy"),
        "encode_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source_sha256": sha256(Path(__file__)),
        "export_helper_sha256": sha256(Path(export_features.__code__.co_filename)),
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"rows": len(rows), "seconds": receipt["encode_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
