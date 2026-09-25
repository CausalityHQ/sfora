#!/usr/bin/env python3
"""Verify the public trained SigLIP2 serving API against a frozen DGX receipt."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

from sfora.representation_ceiling import deterministic_class_partition
from sfora.siglip2_compact_serving import Siglip2CompactIndex

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
PAIR_SHA = "fe0b73759e25e4032c5055f1c043c2b8162dc774bc9e8bcac9f7624b997e2360"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--paired-receipt", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32_autocast", "fp16_native"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.paired_receipt) != PAIR_SHA
    ):
        raise ValueError("trained SigLIP2 serving verification authority differs")
    pair = json.loads(args.paired_receipt.read_text())
    if any(
        sha256(path) != pair[key]
        for path, key in (
            (args.training_receipt, "training_receipt_sha256"),
            (args.training_checkpoint, "training_checkpoint_sha256"),
            (args.train_embeddings, "train_embeddings_sha256"),
            (args.native_library, "native_library_sha256"),
        )
    ):
        raise ValueError("trained SigLIP2 serving paired artifact identity differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    held = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    rows = held[np.linspace(0, len(held) - 1, 32, dtype=int)]
    if ids[rows].tolist() != pair["query_image_ids"]:
        raise ValueError("trained SigLIP2 serving query inventory differs")
    images = []
    for position, row in enumerate(rows):
        relative = PurePosixPath(str(relatives[row]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("trained SigLIP2 serving query path differs")
        path = args.dataset_root.joinpath(*relative.parts)
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256(path) != pair["query_image_sha256"][position]
        ):
            raise ValueError("trained SigLIP2 serving query image differs")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    digests = {}
    with Siglip2CompactIndex.from_artifacts(
        model_snapshot=args.model_snapshot,
        training_receipt=args.training_receipt,
        training_checkpoint=args.training_checkpoint,
        train_embeddings=args.train_embeddings,
        native_library=args.native_library,
        expected_receipt_sha256=pair["training_receipt_sha256"],
        precision=args.precision,
    ) as index:
        for batch_size in (1, 32):
            ordinals, scores = index.search_images(images[:batch_size])
            digest = hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest()
            expected = pair["timing"][str(batch_size)][args.precision]["result_sha256"]
            if digest != expected:
                raise ValueError("trained SigLIP2 public API differs from paired timing result")
            digests[str(batch_size)] = digest
    result = {
        "schema": "sfora-siglip2-compact-serving-parity-v1",
        "claim_eligible": False,
        "precision": args.precision,
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": ARCHIVE_SHA,
        "paired_receipt_sha256": PAIR_SHA,
        "training_receipt_sha256": sha256(args.training_receipt),
        "serving_module_sha256": sha256(Path(inspect.getfile(Siglip2CompactIndex))),
        "batch_result_sha256": digests,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "transformers": __import__("transformers").__version__,
            "pillow": __import__("PIL").__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
