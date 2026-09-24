#!/usr/bin/env python3
"""Export a pinned SigLIP2 image tower on authenticated SOP TRAIN rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from export_unicom_sop_embeddings import SopRecord, _parse_split

MODEL_REVISION = "787800c8990e6f058423089178e718139608408c"
UNICOM_TRAIN_ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
MODEL_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def authenticated_train_rows(
    archive: Mapping[str, Any], dataset_root: Path
) -> tuple[SopRecord, ...]:
    """Match an embedding archive's TRAIN identity to the official TRAIN file."""

    rows = _parse_split(dataset_root, "train")
    ids = np.asarray(archive["train_image_ids"])
    labels = np.asarray(archive["train_labels"])
    paths = np.asarray(archive["train_relative_paths"])
    if (
        ids.shape != (len(rows),)
        or labels.shape != ids.shape
        or paths.shape != ids.shape
        or not np.array_equal(ids, np.asarray([row.image_id for row in rows]))
        or not np.array_equal(labels, np.asarray([row.label for row in rows]))
        or not np.array_equal(paths.astype(str), np.asarray([row.relative_path for row in rows]))
    ):
        raise ValueError("SOP TRAIN row authority differs")
    return rows


def export_features(
    rows: Sequence[SopRecord],
    encode: Callable[[Sequence[SopRecord]], np.ndarray],
    output: Path,
    *,
    width: int,
    batch_size: int,
) -> None:
    """Write finite float32 descriptors in authority order, publishing atomically."""

    if not rows or width < 1 or batch_size < 1 or output.exists() or output.is_symlink():
        raise ValueError("SOP SigLIP2 feature export inventory differs")
    partial = output.with_name(output.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("SOP SigLIP2 partial feature output exists")
    try:
        features = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=(len(rows), width)
        )
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            values = encode(batch)
            if (
                type(values) is not np.ndarray
                or values.shape != (len(batch), width)
                or values.dtype != np.float32
                or not np.isfinite(values).all()
            ):
                raise ValueError("SOP SigLIP2 nonfinite or malformed features")
            features[start : start + len(batch)] = values
        features.flush()
        del features
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-train-archive", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-rows", type=int)
    mode.add_argument("--execute-full-train", action="store_true")
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or args.output_dir.is_symlink()
        or args.batch_size < 1
        or (args.preflight_rows is not None and not 1 <= args.preflight_rows <= 64)
        or not torch.cuda.is_available()
        or sha256(args.unicom_train_archive) != UNICOM_TRAIN_ARCHIVE_SHA256
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or any(not (args.model_snapshot / name).is_file() for name in MODEL_FILES)
    ):
        raise ValueError("SOP SigLIP2 source or invocation authority differs")
    model_hashes = {name: sha256(args.model_snapshot / name) for name in MODEL_FILES}
    with np.load(args.unicom_train_archive, allow_pickle=False) as archive:
        all_rows = authenticated_train_rows(archive, args.dataset_root)
    if len(all_rows) != 59_551:
        raise ValueError("SOP SigLIP2 TRAIN row count differs")
    rows = all_rows if args.execute_full_train else all_rows[: args.preflight_rows]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def encode(batch: Sequence[SopRecord]) -> np.ndarray:
        images = []
        for row in batch:
            with Image.open(row.image_path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        tensors = {name: value.cuda() for name, value in inputs.items() if torch.is_tensor(value)}
        output = model.get_image_features(**tensors)
        if isinstance(output, torch.Tensor):
            features = output
        elif hasattr(output, "pooler_output"):
            features = output.pooler_output
        else:
            raise ValueError("SOP SigLIP2 image feature output differs")
        return features.float().cpu().numpy()

    started = time.perf_counter()
    first = encode(rows[: min(len(rows), args.batch_size)])
    if first.ndim != 2 or first.shape[0] != min(len(rows), args.batch_size):
        raise ValueError("SOP SigLIP2 image feature width differs")
    width = int(first.shape[1])
    export_features(
        rows,
        encode,
        args.output_dir / "train_features.npy",
        width=width,
        batch_size=args.batch_size,
    )
    torch.cuda.synchronize()
    source = Path(__file__).resolve()
    authority_source = source.with_name("export_unicom_sop_embeddings.py")
    receipt = {
        "schema": "sfora-sop-siglip2-train-feature-export-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN rows only",
        "full_train": args.execute_full_train,
        "rows": len(rows),
        "dimension": width,
        "batch_size": args.batch_size,
        "model_id": "google/siglip2-large-patch16-256",
        "model_revision": MODEL_REVISION,
        "model_file_sha256": model_hashes,
        "unicom_train_archive_sha256": UNICOM_TRAIN_ARCHIVE_SHA256,
        "source_sha256": {
            source.name: sha256(source),
            authority_source.name: sha256(authority_source),
        },
        "ordered_rows_sha256": hashlib.sha256(
            "\n".join(f"{row.image_id}\0{row.label}\0{row.relative_path}" for row in rows).encode()
        ).hexdigest(),
        "features_sha256": sha256(args.output_dir / "train_features.npy"),
        "encode_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "python": platform.python_version(),
        },
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps({"rows": len(rows), "width": width, "seconds": receipt["encode_wall_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
