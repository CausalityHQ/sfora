#!/usr/bin/env python3
"""Compare the public query encoder with its pinned SOP TRAIN export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.siglip2_compact_serving import Siglip2CompactIndex

ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--seed", type=int, choices=(179024, 179026, 179027), required=True)
    parser.add_argument("--arm", choices=("control", "freeze"), required=True)
    parser.add_argument("--precision", choices=("fp32_autocast", "fp16_native"), required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.source_archive) != ARCHIVE_SHA
        or sha256(args.native_library) != NATIVE_SHA
    ):
        raise ValueError("SOP serving source or scorer differs")
    decision = json.loads(args.decision.read_text())
    receipt_path = args.training_run / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        decision.get("schema") != "sfora-sop-true-freeze-paired-train-only-v1"
        or decision.get("seed") != args.seed
        or decision.get("arms", {}).get(args.arm, {}).get("receipt_sha256") != sha256(receipt_path)
        or receipt.get("seed") != args.seed
        or receipt.get("arm") != "float_rank_member_bank"
        or receipt.get("freeze_lower_stack") is not (args.arm == "freeze")
    ):
        raise ValueError("SOP serving training arm differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    values = np.load(args.training_run / "train_embeddings.npy", mmap_mode="r", allow_pickle=False)
    if (
        values.shape != (59_551, 128)
        or sha256(args.training_run / "train_embeddings.npy") != receipt["train_embeddings_sha256"]
    ):
        raise ValueError("SOP serving exported gallery differs")
    rows = np.linspace(0, 59_550, 32, dtype=np.int64)
    images = []
    image_sha = []
    for row in rows:
        relative = PurePosixPath(str(relatives[row]))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("SOP serving image path differs")
        path = args.dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP serving image missing")
        image_sha.append(sha256(path))
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    expected = pack_int8_unit_embeddings(torch.from_numpy(np.asarray(values[rows]).copy()))
    with Siglip2CompactIndex.from_artifacts(
        model_snapshot=args.model_snapshot,
        training_receipt=receipt_path,
        training_checkpoint=args.training_run / "checkpoint.pt",
        train_embeddings=args.training_run / "train_embeddings.npy",
        native_library=args.native_library,
        expected_receipt_sha256=sha256(receipt_path),
        precision=args.precision,
    ) as index:
        assert index.encoder is not None and index.gallery is not None
        actual = index.encoder.encode_images(images)
        expected_top, _ = index.gallery.search_packed(expected)
        actual_top, _ = index.gallery.search_packed(actual)
        code_equal = np.all(actual.codes.numpy() == expected.codes.numpy(), axis=1)
        norm_equal = actual.inverse_norms.numpy() == expected.inverse_norms.numpy()
        top_equal = np.all(actual_top == expected_top, axis=1)
    result = {
        "schema": "sfora-sop-true-freeze-public-serving-parity-v1",
        "claim_eligible": False,
        "seed": args.seed,
        "arm": args.arm,
        "precision": args.precision,
        "training_receipt_sha256": sha256(receipt_path),
        "source_archive_sha256": ARCHIVE_SHA,
        "source_sha256": sha256(Path(__file__)),
        "query_image_ids": ids[rows].tolist(),
        "query_image_sha256": image_sha,
        "gallery_wire_bytes_per_row": 130,
        "query_code_exact_rows": int(code_equal.sum()),
        "query_inverse_norm_exact_rows": int(norm_equal.sum()),
        "query_top10_exact_rows": int(top_equal.sum()),
        "query_top1_exact_rows": int((actual_top[:, 0] == expected_top[:, 0]).sum()),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key.endswith("rows")}, sort_keys=True
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
