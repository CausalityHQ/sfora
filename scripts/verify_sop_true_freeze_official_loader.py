#!/usr/bin/env python3
"""Check one public SOP official gallery through the production image-to-top-10 API."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sfora import (
    cutile_int8,
    joint_relational_compaction,
    siglip2_compact_serving,
    sop_compact_training,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.siglip2_compact_serving import Siglip2CompactIndex

SOURCE_HASHES = {
    "siglip2_compact_serving": "700fd6d1e045e135c5298fd8842f6275e68bdd3cec9a158790aa8dce9267c614",
    "joint_relational_compaction": (
        "4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67"
    ),
    "cutile_int8": "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409",
    "sop_compact_training": "12ccd15c3b943fa519ba37f8d09870b105939872d9279541fc0ab42f20a7ead9",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    root = Path("/home/riomus/runs")
    seed = 179024
    training_receipt_sha = "07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5"
    official_receipt_sha = "7f5edf6b8ecb526e8ba29944119035245725055787cb088bd2635cc734b0eb6c"
    training = root / f"sfora-sop-true-freeze-freeze-{seed}-1000-v1"
    official = root / f"sfora-sop-true-freeze-public-official-{seed}-freeze-v1"
    output = root / "sfora-sop-true-freeze-official-loader-179024-v2.json"
    if output.exists() or torch.cuda.is_available() is False:
        raise ValueError("official loader invocation differs")
    imported = {
        name: digest(Path(inspect.getfile(module)))
        for name, module in (
            ("siglip2_compact_serving", siglip2_compact_serving),
            ("joint_relational_compaction", joint_relational_compaction),
            ("cutile_int8", cutile_int8),
            ("sop_compact_training", sop_compact_training),
        )
    }
    if imported != SOURCE_HASHES:
        raise ValueError("official loader imported source differs")
    archive = Path("/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz")
    if digest(archive) != "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a":
        raise ValueError("official loader source archive differs")
    with np.load(archive, allow_pickle=False) as source:
        relatives = np.asarray(source["test_relative_paths"]).astype(str)[:32]
    from train_sop_siglip2_compact import paths_from_archive

    paths = paths_from_archive(Path("/home/riomus/datasets/Stanford_Online_Products"), relatives)
    manifest_path = root / "sfora-sop-reference-b8f85611-179019/sop-test-image-sha256.bin"
    if digest(manifest_path) != "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1":
        raise ValueError("official loader image manifest differs")
    manifest = manifest_path.read_bytes()
    images = []
    for row, path in enumerate(paths):
        raw = path.read_bytes()
        if hashlib.sha256(raw).digest() != manifest[row * 32 : (row + 1) * 32]:
            raise ValueError("official loader image manifest differs")
        with Image.open(BytesIO(raw)) as image:
            images.append(image.convert("RGB"))
    training_receipt = training / "receipt.json"
    official_receipt = official / "receipt.json"
    if (
        digest(training_receipt) != training_receipt_sha
        or digest(official_receipt) != official_receipt_sha
    ):
        raise ValueError("official loader pinned receipt differs")
    checkpoint = training / "checkpoint.pt"
    embeddings = official / "test_embeddings.npy"
    library = Path(
        "/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so"
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    with Siglip2CompactIndex.from_artifacts(
        model_snapshot=Path(
            "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/"
            "snapshots/787800c8990e6f058423089178e718139608408c"
        ),
        training_receipt=training_receipt,
        training_checkpoint=checkpoint,
        train_embeddings=training / "train_embeddings.npy",
        native_library=library,
        expected_receipt_sha256=training_receipt_sha,
        precision="fp16_native",
        official_gallery_receipt=official_receipt,
        official_gallery_embeddings=embeddings,
        expected_official_gallery_receipt_sha256=official_receipt_sha,
    ) as index:
        assert index.encoder is not None
        query = index.encoder.encode_images(images)
        ordinals, scores = index.search_images(images)
    gallery = pack_int8_unit_embeddings(
        torch.from_numpy(np.asarray(np.load(embeddings, mmap_mode="r", allow_pickle=False)).copy())
    )
    if not (
        torch.equal(query.codes, gallery.codes[:32])
        and torch.equal(query.inverse_norms, gallery.inverse_norms[:32])
    ):
        raise ValueError("official loader query codes differ")
    codes = gallery.codes.float().cuda()
    inverse = gallery.inverse_norms.float().cuda()
    exact = (codes[:32] @ codes.T) * inverse[:32, None] * inverse[None, :]
    expected = torch.argsort(exact, dim=1, descending=True, stable=True)[:, :10]
    expected_scores = exact.gather(1, expected).cpu().numpy()
    if scores.shape != expected_scores.shape or not np.isfinite(scores).all():
        raise ValueError("official loader top-10 scores differ")
    if not np.array_equal(ordinals, expected.cpu().numpy()):
        raise ValueError("official loader top-10 ordinals differ")
    max_delta = float(np.max(np.abs(scores - expected_scores)))
    if max_delta > 1e-5:
        raise ValueError("official loader top-10 scores differ")
    result = {
        "schema": "sfora-sop-true-freeze-official-loader-v2",
        "seed": seed,
        "arm": "freeze",
        "query_rows": 32,
        "gallery_rows": len(gallery.codes),
        "gallery_receipt_sha256": digest(official_receipt),
        "gallery_embeddings_sha256": digest(embeddings),
        "training_receipt_sha256": digest(training_receipt),
        "source_sha256": digest(Path(__file__)),
        "imported_source_sha256": imported,
        "source_archive_sha256": digest(archive),
        "image_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "query_packed_exact": True,
        "native_top10_exact": True,
        "max_score_abs_delta": max_delta,
    }
    with output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
