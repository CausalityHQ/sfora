#!/usr/bin/env python3
"""One exploratory SOP float/PCA-packed screen from authenticated UNICOM B/16 features."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.sop_evaluation import score_symmetric


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-exploratory-test", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available():
        raise ValueError("SOP B/16 screen authority differs")
    started = time.perf_counter()
    archive_sha256 = sha256(args.archive)
    archive = load_sop_embedding_archive(args.archive)
    metadata = archive["metadata"]
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_identifier") != "UNICOM-ViT-B/16"
        or metadata.get("checkpoint_sha256")
        != "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef"
    ):
        raise ValueError("SOP B/16 model authority differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    train = F.normalize(torch.from_numpy(archive["train_embeddings"]).float(), dim=1)
    test = F.normalize(torch.from_numpy(archive["test_embeddings"]).float(), dim=1)
    labels = torch.from_numpy(archive["test_labels"]).to(device)

    float_started = time.perf_counter()
    float_score = score_symmetric(test.to(device), labels)
    float_seconds = time.perf_counter() - float_started

    fit_started = time.perf_counter()
    pca = fit_centered_pca(train.contiguous(), dimensions=128)
    fit_seconds = time.perf_counter() - fit_started
    transform_started = time.perf_counter()
    compact = pca.apply(test.contiguous())
    transform_seconds = time.perf_counter() - transform_started
    pca_float_started = time.perf_counter()
    pca_float_score = score_symmetric(compact.to(device), labels)
    pca_float_seconds = time.perf_counter() - pca_float_started
    pack_started = time.perf_counter()
    packed = pack_int8_unit_embeddings(compact)
    pack_seconds = time.perf_counter() - pack_started
    packed_started = time.perf_counter()
    packed_score = score_symmetric(
        packed.codes.float().to(device),
        labels,
        inverse_norms=packed.inverse_norms.to(device),
    )
    packed_seconds = time.perf_counter() - packed_started

    result = {
        "schema": "sfora-unicom-b16-sop-pretrained-screen-v2",
        "claim_eligible": False,
        "split": "official SOP Ebay_train fit; already-observed Ebay_test self retrieval",
        "source_archive_sha256": archive_sha256,
        "source_metadata": metadata,
        "script_sha256": sha256(Path(__file__)),
        "scorer_sha256": sha256(Path(__file__).parents[1] / "src/sfora/sop_evaluation.py"),
        "representation_sha256": sha256(
            Path(__file__).parents[1] / "src/sfora/representation_ceiling.py"
        ),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "train_rows": len(train),
        "test_rows": len(test),
        "pca_dimensions": 128,
        "wire_bytes_per_gallery_item": packed.bytes_per_vector,
        "float": {"score": float_score, "score_seconds": float_seconds},
        "pca_float": {
            "score": pca_float_score,
            "train_only_transform_seconds": transform_seconds,
            "score_seconds": pca_float_seconds,
        },
        "pca_packed": {
            "score": packed_score,
            "pca_fit_seconds": fit_seconds,
            "test_pack_seconds": pack_seconds,
            "score_seconds": packed_seconds,
            "gallery_code_sha256": hashlib.sha256(packed.codes.numpy().tobytes()).hexdigest(),
            "gallery_inverse_norm_sha256": hashlib.sha256(
                packed.inverse_norms.numpy().tobytes()
            ).hexdigest(),
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "total_seconds": time.perf_counter() - started,
    }
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "float_r1": float_score["recall_at_1"],
                "pca_float_r1": pca_float_score["recall_at_1"],
                "packed_r1": packed_score["recall_at_1"],
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
