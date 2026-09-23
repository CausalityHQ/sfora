#!/usr/bin/env python3
"""Paired train-identity SOP holdout screen of pretrained B/16 and L/14."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.sop_evaluation import score_symmetric

ARCHIVES = {
    "b16": (
        "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f",
        "UNICOM-ViT-B/16",
    ),
    "l14_336": (
        "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a",
        "UNICOM-ViT-L/14@336px",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_train_alignment(first: dict, second: dict) -> None:
    """Require exact row identity before comparing architecture representations."""

    keys = ("train_image_ids", "train_labels", "train_relative_paths")
    if (
        any(
            key not in first
            or key not in second
            or not np.array_equal(first[key], second[key])
            for key in keys
        )
        or first["train_embeddings"].shape != second["train_embeddings"].shape
    ):
        raise ValueError("paired SOP train rows differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--b16-archive", type=Path, required=True)
    parser.add_argument("--l14-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-sop-architecture-screen", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available():
        raise ValueError("SOP architecture screen invocation differs")
    paths = {"b16": args.b16_archive, "l14_336": args.l14_archive}
    for name, path in paths.items():
        if sha256(path) != ARCHIVES[name][0]:
            raise ValueError("SOP architecture source archive differs")
    archives = {name: load_sop_embedding_archive(path) for name, path in paths.items()}
    for name, archive in archives.items():
        metadata = archive["metadata"]
        if (
            metadata.get("model_identifier") != ARCHIVES[name][1]
            or archive["train_embeddings"].shape != (59_551, 768)
        ):
            raise ValueError("SOP architecture model or inventory differs")
    assert_train_alignment(archives["b16"], archives["l14_336"])
    labels = tuple(int(value) for value in archives["b16"]["train_labels"])
    partition = deterministic_class_partition(labels, fit_fraction=0.9, seed=179019)
    validation_labels = torch.tensor(
        [labels[index] for index in partition.validation_row_indexes],
        dtype=torch.int64,
        device="cuda",
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results = {}
    for name, archive in archives.items():
        features = archive["train_embeddings"]
        fit = F.normalize(
            torch.from_numpy(
                np.ascontiguousarray(features[list(partition.fit_row_indexes)])
            ).float(),
            dim=1,
        )
        validation = F.normalize(
            torch.from_numpy(
                np.ascontiguousarray(features[list(partition.validation_row_indexes)])
            ).float(),
            dim=1,
        )
        pca_started = time.perf_counter()
        pca = fit_centered_pca(fit.contiguous(), dimensions=128)
        pca_seconds = time.perf_counter() - pca_started
        projected = pca.apply(validation.contiguous())
        packed = pack_int8_unit_embeddings(projected)
        score_started = time.perf_counter()
        scored = {}
        for arm, (values, inverse) in {
            "full_float": (validation, None),
            "pca_float": (projected, None),
            "pca_packed": (packed.codes.float(), packed.inverse_norms),
        }.items():
            scored[arm] = score_symmetric(
                values.cuda(),
                validation_labels,
                inverse_norms=None if inverse is None else inverse.cuda(),
            )
        results[name] = {
            "pca_fit_seconds": pca_seconds,
            "score_seconds": time.perf_counter() - score_started,
            "arms": scored,
        }
    receipt = {
        "schema": "sfora-sop-pretrained-architecture-train-holdout-v1",
        "claim_eligible": False,
        "split": "deterministic 90/10 class-disjoint SOP training identities only",
        "split_seed": 179019,
        "fit_images": len(partition.fit_row_indexes),
        "fit_classes": len(partition.fit_class_ids),
        "validation_images": len(partition.validation_row_indexes),
        "validation_classes": len(partition.validation_class_ids),
        "validation_image_ids": [
            int(archives["b16"]["train_image_ids"][index])
            for index in partition.validation_row_indexes
        ],
        "validation_labels": validation_labels.cpu().tolist(),
        "results": results,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "inputs": {
            "archives": {name: ARCHIVES[name][0] for name in ARCHIVES},
            "script_sha256": sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                name: {
                    arm: {key: value[key] for key in ("recall_at_1", "map_at_r")}
                    for arm, value in data["arms"].items()
                }
                for name, data in results.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
