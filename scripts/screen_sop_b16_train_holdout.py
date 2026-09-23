#!/usr/bin/env python3
"""Train-only class-disjoint baseline for the compact B/16 backbone experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.sop_evaluation import score_symmetric

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--features-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-train-holdout-screen", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.features_archive) != ARCHIVE_SHA256:
        raise ValueError("SOP train holdout source differs")
    torch.set_num_threads(16)
    source = load_sop_embedding_archive(args.features_archive)
    labels = tuple(int(value) for value in source["train_labels"])
    partition = deterministic_class_partition(labels, fit_fraction=0.9, seed=179019)
    fit = F.normalize(
        torch.from_numpy(
            np.ascontiguousarray(source["train_embeddings"][list(partition.fit_row_indexes)])
        ).float(),
        dim=1,
    )
    validation = F.normalize(
        torch.from_numpy(
            np.ascontiguousarray(source["train_embeddings"][list(partition.validation_row_indexes)])
        ).float(),
        dim=1,
    )
    validation_labels = torch.tensor(
        [labels[index] for index in partition.validation_row_indexes], dtype=torch.int64
    )
    started = time.perf_counter()
    pca = fit_centered_pca(fit.contiguous(), dimensions=128)
    fit_seconds = time.perf_counter() - started
    projected = pca.apply(validation.contiguous())
    packed = pack_int8_unit_embeddings(projected)
    arms = {
        "full_float": (validation, None),
        "pca_float": (projected, None),
        "pca_packed": (packed.codes.float(), packed.inverse_norms),
    }
    results = {}
    for name, (values, inverse) in arms.items():
        scored = score_symmetric(values, validation_labels, inverse_norms=inverse)
        results[name] = {
            "recall_at_1": scored["recall_at_1"],
            "map_at_r": scored["map_at_r"],
            "per_query_r1": scored["per_query_r1"],
            "per_query_ap": scored["per_query_ap"],
        }
    receipt = {
        "schema": "sfora-sop-b16-train-holdout-screen-v1",
        "claim_eligible": False,
        "archive_sha256": ARCHIVE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "split_seed": 179019,
        "fit_fraction": 0.9,
        "fit_images": len(partition.fit_row_indexes),
        "fit_classes": len(partition.fit_class_ids),
        "validation_images": len(partition.validation_row_indexes),
        "validation_classes": len(partition.validation_class_ids),
        "pca_fit_seconds": fit_seconds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                name: {key: result[key] for key in ("recall_at_1", "map_at_r")}
                for name, result in results.items()
            }
        )
    )


if __name__ == "__main__":
    main()
