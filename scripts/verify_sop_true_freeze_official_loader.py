#!/usr/bin/env python3
"""Check one public SOP official gallery through the production image-to-top-10 API."""

from __future__ import annotations

import hashlib
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.siglip2_compact_serving import Siglip2CompactIndex


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    root = Path("/home/riomus/runs")
    seed = 179024
    training = root / f"sfora-sop-true-freeze-freeze-{seed}-1000-v1"
    official = root / f"sfora-sop-true-freeze-public-official-{seed}-freeze-v1"
    output = root / "sfora-sop-true-freeze-official-loader-179024-v1.json"
    if output.exists() or torch.cuda.is_available() is False:
        raise ValueError("official loader invocation differs")
    with np.load(
        "/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz", allow_pickle=False
    ) as source:
        relatives = np.asarray(source["test_relative_paths"]).astype(str)[:32]
    from train_sop_siglip2_compact import paths_from_archive

    paths = paths_from_archive(Path("/home/riomus/datasets/Stanford_Online_Products"), relatives)
    manifest = (root / "sfora-sop-reference-b8f85611-179019/sop-test-image-sha256.bin").read_bytes()
    images = []
    for row, path in enumerate(paths):
        raw = path.read_bytes()
        if hashlib.sha256(raw).digest() != manifest[row * 32 : (row + 1) * 32]:
            raise ValueError("official loader image manifest differs")
        with Image.open(BytesIO(raw)) as image:
            images.append(image.convert("RGB"))
    training_receipt = training / "receipt.json"
    official_receipt = official / "receipt.json"
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
        expected_receipt_sha256=digest(training_receipt),
        precision="fp16_native",
        official_gallery_receipt=official_receipt,
        official_gallery_embeddings=embeddings,
        expected_official_gallery_receipt_sha256=digest(official_receipt),
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
    if not np.array_equal(ordinals, expected.cpu().numpy()):
        raise ValueError("official loader top-10 ordinals differ")
    max_delta = float(np.max(np.abs(scores - expected_scores)))
    if max_delta > 1e-5:
        raise ValueError("official loader top-10 scores differ")
    result = {
        "schema": "sfora-sop-true-freeze-official-loader-v1",
        "seed": seed,
        "arm": "freeze",
        "query_rows": 32,
        "gallery_rows": len(gallery.codes),
        "gallery_receipt_sha256": digest(official_receipt),
        "gallery_embeddings_sha256": digest(embeddings),
        "training_receipt_sha256": digest(training_receipt),
        "source_sha256": digest(Path(__file__)),
        "query_packed_exact": True,
        "native_top10_exact": True,
        "max_score_abs_delta": max_delta,
    }
    with output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
