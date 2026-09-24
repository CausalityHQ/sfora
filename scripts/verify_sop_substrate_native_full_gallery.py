#!/usr/bin/env python3
"""Replay every SOP TRAIN holdout query through the released native scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from score_sop_pretrained_substrate import fit_project_pack, sha256

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings
from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
TILEIRAS_SHA256 = "df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae"
ARMS = ("unicom_l14_336", "siglip2_l16_256")


def first_nonself(ordinals: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    """Take the first ranked row other than the query's own gallery row."""

    if (
        not isinstance(ordinals, np.ndarray)
        or ordinals.dtype != np.int64
        or ordinals.ndim != 2
        or ordinals.shape[1] < 2
        or not isinstance(query_rows, np.ndarray)
        or query_rows.dtype != np.int64
        or query_rows.shape != (len(ordinals),)
    ):
        raise ValueError("SOP native nonself geometry differs")
    valid = ordinals != query_rows[:, None]
    if not bool(valid.any(axis=1).all()):
        raise ValueError("SOP native nonself result missing")
    return cast(np.ndarray, ordinals[np.arange(len(ordinals)), valid.argmax(axis=1)])


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def replay_arm(
    name: str,
    features: np.ndarray,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
    labels: np.ndarray,
    quality_dir: Path,
    native_library: Path,
) -> dict[str, Any]:
    receipt = json.loads((quality_dir / f"{name}.json").read_text())
    corpus = fit_project_pack(features, fit_rows)
    pca_sha256 = hashlib.sha256(
        corpus.pca.mean.numpy().tobytes() + corpus.pca.components.numpy().tobytes()
    ).hexdigest()
    if pca_sha256 != receipt["pca_sha256"]:
        raise ValueError(f"SOP native {name} PCA differs from quality receipt")
    query = PackedInt8Embeddings(
        corpus.packed.codes[held_rows].contiguous(),
        corpus.packed.inverse_norms[held_rows].contiguous(),
    )
    started = time.perf_counter()
    with CutilePackedInt8Gallery.open_packed(native_library, corpus.packed) as gallery:
        ordinals, scores = gallery.search_packed(query)
    search_seconds = time.perf_counter() - started
    if (
        ordinals.shape != (len(held_rows), 10)
        or scores.shape != ordinals.shape
        or not np.isfinite(scores).all()
        or not bool((ordinals >= 0).all())
        or not bool((ordinals < len(labels)).all())
    ):
        raise ValueError(f"SOP native {name} top-10 geometry differs")
    top1 = first_nonself(ordinals, held_rows)
    observed = labels[top1] == labels[held_rows]
    expected = np.asarray(
        receipt["score"]["packed_pca128"]["full_train_gallery"]["per_query_r1"],
        dtype=np.bool_,
    )
    if expected.shape != observed.shape or not np.array_equal(observed, expected):
        raise ValueError(f"SOP native {name} per-query Recall@1 differs from quality receipt")

    query_codes = query.codes.numpy().astype(np.int32)
    gallery_codes = corpus.packed.codes.numpy().astype(np.int32)
    dots = np.einsum("bd,bkd->bk", query_codes, gallery_codes[ordinals], optimize=True)
    scalar_scores = (
        dots.astype(np.float32)
        * query.inverse_norms.numpy().astype(np.float32)[:, None]
        * corpus.packed.inverse_norms.numpy().astype(np.float32)[ordinals]
    )
    max_score_delta = float(np.max(np.abs(scalar_scores - scores)))
    if max_score_delta > 1e-5:
        raise ValueError(f"SOP native {name} top-10 score arithmetic differs")
    return {
        "quality_arm_sha256": sha256(quality_dir / f"{name}.json"),
        "pca_sha256": pca_sha256,
        "queries": len(held_rows),
        "top1_correct": int(observed.sum()),
        "recall_at_1": float(observed.mean()),
        "per_query_r1": observed.astype(np.uint8).tolist(),
        "top1_ordinals": top1.tolist(),
        "top10_ordinals_sha256": digest(ordinals),
        "top10_scores_sha256": digest(scores),
        "max_top10_score_abs_delta": max_score_delta,
        "search_seconds_including_gallery_open": search_seconds,
        "exact_per_query_quality_parity": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-l14-archive", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.unicom_l14_archive) != ARCHIVE_SHA256
        or sha256(args.native_library) != NATIVE_SHA256
        or not tileiras
        or sha256(Path(tileiras)) != TILEIRAS_SHA256
    ):
        raise ValueError("SOP native substrate source authority differs")
    quality = json.loads((args.quality_dir / "receipt.json").read_text())
    export = json.loads((args.candidate_dir / "receipt.json").read_text())
    if (
        quality.get("schema") != "sfora-sop-pretrained-substrate-screen-v1"
        or quality.get("candidate_feature_sha256") != export.get("features_sha256")
        or quality.get("source_archive_sha256") != ARCHIVE_SHA256
        or sha256(args.candidate_dir / "train_features.npy") != export.get("features_sha256")
    ):
        raise ValueError("SOP native substrate quality authority differs")
    with np.load(args.unicom_l14_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        reference_features = np.asarray(archive["train_embeddings"], dtype=np.float32)
    candidate_features = np.load(args.candidate_dir / "train_features.npy", mmap_mode="r")
    if (
        labels.shape != (59_551,)
        or ids.shape != labels.shape
        or reference_features.shape != (59_551, 768)
        or candidate_features.shape != (59_551, 1024)
    ):
        raise ValueError("SOP native substrate TRAIN inventory differs")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit_rows) != 53_700 or len(held_rows) != 5_851:
        raise ValueError("SOP native substrate TRAIN partition differs")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    arms = {
        name: replay_arm(
            name,
            features,
            fit_rows,
            held_rows,
            labels,
            args.quality_dir,
            args.native_library,
        )
        for name, features in zip(ARMS, (reference_features, candidate_features), strict=True)
    }
    receipt = {
        "schema": "sfora-sop-substrate-native-full-gallery-parity-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout only; no TEST rows",
        "fit_images": len(fit_rows),
        "holdout_queries": len(held_rows),
        "full_gallery_images": len(labels),
        "seed": 179019,
        "source_archive_sha256": ARCHIVE_SHA256,
        "quality_receipt_sha256": sha256(args.quality_dir / "receipt.json"),
        "candidate_feature_sha256": export["features_sha256"],
        "native_library_sha256": NATIVE_SHA256,
        "tileiras_sha256": TILEIRAS_SHA256,
        "source_sha256": sha256(Path(__file__)),
        "query_image_ids_sha256": digest(ids[held_rows]),
        "arms": arms,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(args.output)
    summary = {
        name: {"r1": arm["recall_at_1"], "max_score_delta": arm["max_top10_score_abs_delta"]}
        for name, arm in arms.items()
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
