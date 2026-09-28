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
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from siglip2_base_authority import BASE_HASHES, BASE_REVISION, TRANSFER_SMOKE_SHA

from sfora.unicom_inshop import parse_inshop_partition

PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
MODEL_HASHES = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}


def load_vision_init(vision: torch.nn.Module, checkpoint: Path, expected_sha256: str) -> None:
    if len(expected_sha256) != 64 or sha256(checkpoint) != expected_sha256:
        raise ValueError("In-Shop vision checkpoint SHA differs")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["vision"]
    vision.load_state_dict(state, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tail-blocks-to-drop", type=int, choices=(0, 2), default=0)
    parser.add_argument("--vision-init-checkpoint", type=Path)
    parser.add_argument("--vision-init-sha256")
    parser.add_argument("--teacher-transfer-smoke-receipt", type=Path)
    args = parser.parse_args()
    transfer = args.teacher_transfer_smoke_receipt is not None
    if transfer and (
        sha256(args.teacher_transfer_smoke_receipt) != TRANSFER_SMOKE_SHA
        or args.vision_init_checkpoint is not None
        or args.tail_blocks_to_drop
    ):
        raise ValueError("Base transfer smoke authority differs")
    revision, hashes, width, depth = (
        (BASE_REVISION, BASE_HASHES, 768, 12)
        if transfer
        else (MODEL_REVISION, MODEL_HASHES, 1024, 24)
    )
    if (
        args.output_dir.exists()
        or (args.vision_init_checkpoint is None) != (args.vision_init_sha256 is None)
        or args.model_snapshot.resolve().name != revision
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or tuple(MODEL_FILES) != tuple(MODEL_HASHES)
        or any(sha256(args.model_snapshot / name) != digest for name, digest in hashes.items())
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
    model = AutoModel.from_pretrained(
        args.model_snapshot,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32 if transfer else torch.float16,
    )
    if len(model.vision_model.encoder.layers) != depth:
        raise ValueError("In-Shop SigLIP2 encoder depth differs")
    if args.tail_blocks_to_drop:
        model.vision_model.encoder.layers = torch.nn.ModuleList(
            list(model.vision_model.encoder.layers[: -args.tail_blocks_to_drop])
        )
    if args.vision_init_checkpoint is not None:
        load_vision_init(model.vision_model, args.vision_init_checkpoint, args.vision_init_sha256)
    model = model.cuda().eval()
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
        with torch.autocast("cuda", dtype=torch.float16, enabled=transfer):
            output = model.get_image_features(**tensors)
        features = output if isinstance(output, torch.Tensor) else output.pooler_output
        return features.float().cpu().numpy()

    if transfer:
        fit, _ = split(tuple(row.label for row in rows))
        if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
            raise ValueError("transfer fit cache inventory differs")
        export_features(
            tuple(rows[row] for row in fit),
            encode,
            args.output_dir / "fit_features.npy",
            width=width,
            batch_size=32,
        )
        values = np.lib.format.open_memmap(
            args.output_dir / "train_features.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), width),
        )
        values[:] = 0
        values[list(fit)] = np.load(args.output_dir / "fit_features.npy", allow_pickle=False)
        values.flush()
        del values
    else:
        export_features(
            rows, encode, args.output_dir / "train_features.npy", width=width, batch_size=32
        )
    torch.cuda.synchronize()
    receipt = {
        "schema": "sfora-inshop-siglip2-train-feature-export-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN fit-only export; held zero placeholders"
        if transfer
        else "official In-Shop TRAIN only; all 25,882 rows in partition order",
        "exported_rows": len(fit) if transfer else len(rows),
        "rows": len(rows),
        "products": len({row.label for row in rows}),
        "width": width,
        "exported_fit_only": transfer,
        "fit_sha256": digest_rows(fit) if transfer else None,
        "tail_blocks_dropped": args.tail_blocks_to_drop,
        "vision_init_sha256": args.vision_init_sha256,
        "batch_size": 32,
        "partition_sha256": PARTITION_SHA256,
        "model_revision": revision,
        "model_file_sha256": hashes,
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
