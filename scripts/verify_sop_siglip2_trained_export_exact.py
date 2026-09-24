#!/usr/bin/env python3
"""Replay the trained SigLIP2 export through scalar and native packed top-10."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from train_sop_siglip2_compact import (
    ARCHIVE_SHA256,
    NATIVE_SHA256,
    TILEIRAS_SHA256,
    evaluate_packed,
    sha256,
)

from sfora.representation_ceiling import deterministic_class_partition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--expected-embeddings-sha256", type=str, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.embeddings) != args.expected_embeddings_sha256
        or sha256(args.native_library) != NATIVE_SHA256
        or not tileiras
        or sha256(Path(tileiras)) != TILEIRAS_SHA256
        or not torch.cuda.is_available()
    ):
        raise ValueError("SOP trained export exactness authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
    values = np.load(args.embeddings, allow_pickle=False)
    if labels.shape != (59_551,) or values.shape != (59_551, 128) or values.dtype != np.float32:
        raise ValueError("SOP trained export exactness geometry differs")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = evaluate_packed(
        torch.from_numpy(values.copy()), labels, fit_rows, held_rows, args.native_library
    )
    receipt = {
        "schema": "sfora-sop-siglip2-trained-export-exact-v1",
        "claim_eligible": False,
        "source_sha256": sha256(Path(__file__)),
        "trainer_source_sha256": sha256(Path(__file__).with_name("train_sop_siglip2_compact.py")),
        "source_archive_sha256": ARCHIVE_SHA256,
        "embeddings_sha256": args.expected_embeddings_sha256,
        "native_library_sha256": NATIVE_SHA256,
        "tileiras_sha256": TILEIRAS_SHA256,
        "quality": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "recall_at_1": result["recall_at_1"],
                "native_top10_exact": result["native_top10_exact"],
                "max_score_abs_delta": result["native_top10_max_score_abs_delta"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
