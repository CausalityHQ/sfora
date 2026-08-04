#!/usr/bin/env python3
"""Compare top-1 query errors from independently verified final embedding packs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _scalar_string(value: np.ndarray) -> str:
    if value.shape != ():
        raise ValueError("expected scalar string metadata")
    return str(value.item())


def _load_pack(path: Path, *, expected_split: str) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "embeddings",
            "labels",
            "example_ids",
            "source_paths",
            "artifact_selection",
            "split",
            "checkpoint_sha256",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        if _scalar_string(payload["artifact_selection"]) != "final_training_state":
            raise ValueError(f"{path} is not a final-state pack")
        if _scalar_string(payload["split"]) != expected_split:
            raise ValueError(f"{path} is not a {expected_split} pack")
        result: dict[str, np.ndarray | str] = {
            "embeddings": np.asarray(payload["embeddings"], dtype=np.float64),
            "labels": np.asarray(payload["labels"], dtype=np.int64),
            "example_ids": np.asarray(payload["example_ids"]),
            "source_paths": np.asarray(payload["source_paths"]),
            "checkpoint_sha256": _scalar_string(payload["checkpoint_sha256"]),
        }
    embeddings = np.asarray(result["embeddings"])
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{path} contains non-finite embeddings")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 0):
        raise ValueError(f"{path} contains zero-norm embeddings")
    result["embeddings"] = embeddings / norms[:, None]
    return result


def _assert_same_rows(left: dict[str, Any], right: dict[str, Any], *, split: str) -> None:
    for field in ("labels", "example_ids", "source_paths"):
        if not np.array_equal(left[field], right[field]):
            raise ValueError(f"{split} {field} differ across seeds")


def _nearest_gallery(query: np.ndarray, gallery: np.ndarray, *, chunk_size: int) -> np.ndarray:
    query_norms = np.linalg.norm(query, axis=1)
    gallery_norms = np.linalg.norm(gallery, axis=1)
    if np.any(query_norms <= 0) or np.any(gallery_norms <= 0):
        raise ValueError("cosine retrieval requires nonzero embeddings")
    query = query / query_norms[:, None]
    gallery = gallery / gallery_norms[:, None]
    nearest = np.empty(len(query), dtype=np.int64)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        nearest[start:stop] = np.argmax(query[start:stop] @ gallery.T, axis=1)
    return nearest


def measure(
    query_packs: list[dict[str, Any]],
    gallery_packs: list[dict[str, Any]],
    *,
    chunk_size: int = 512,
) -> dict[str, Any]:
    if len(query_packs) != 2 or len(gallery_packs) != 2:
        raise ValueError("cross-seed audit requires exactly two query/gallery pack pairs")
    _assert_same_rows(query_packs[0], query_packs[1], split="query")
    _assert_same_rows(gallery_packs[0], gallery_packs[1], split="gallery")
    query_labels = np.asarray(query_packs[0]["labels"], dtype=np.int64)
    gallery_labels = np.asarray(gallery_packs[0]["labels"], dtype=np.int64)

    nearest = [
        _nearest_gallery(
            np.asarray(query_packs[index]["embeddings"], dtype=np.float64),
            np.asarray(gallery_packs[index]["embeddings"], dtype=np.float64),
            chunk_size=chunk_size,
        )
        for index in range(2)
    ]
    predictions = [gallery_labels[rows] for rows in nearest]
    correct = [prediction == query_labels for prediction in predictions]
    errors = [~values for values in correct]
    both_wrong = errors[0] & errors[1]
    error_counts = [int(values.sum()) for values in errors]
    union_errors = errors[0] | errors[1]
    minimum_errors = min(error_counts)
    if minimum_errors == 0:
        raise ValueError("error-overlap coefficient is undefined with an error-free seed")
    both_wrong_count = int(both_wrong.sum())
    shared_wrong_identity = predictions[0][both_wrong] == predictions[1][both_wrong]
    return {
        "queries": len(query_labels),
        "gallery_rows": len(gallery_labels),
        "checkpoint_sha256": [pack["checkpoint_sha256"] for pack in query_packs],
        "recall_at_1": [float(values.mean()) for values in correct],
        "error_counts": error_counts,
        "top1_gallery_row_agreement": float(np.mean(nearest[0] == nearest[1])),
        "predicted_identity_agreement": float(np.mean(predictions[0] == predictions[1])),
        "both_correct": int((correct[0] & correct[1]).sum()),
        "seed0_only_correct": int((correct[0] & errors[1]).sum()),
        "seed1_only_correct": int((errors[0] & correct[1]).sum()),
        "both_wrong": both_wrong_count,
        "error_overlap_coefficient": both_wrong_count / minimum_errors,
        "error_set_jaccard": both_wrong_count / int(union_errors.sum()),
        "same_wrong_identity_given_both_wrong": (
            float(shared_wrong_identity.mean()) if both_wrong_count else None
        ),
        "oracle_either_seed_recall_at_1": float((correct[0] | correct[1]).mean()),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, action="append", required=True)
    parser.add_argument("--gallery", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    query_packs = [_load_pack(path, expected_split="query") for path in args.query]
    gallery_packs = [_load_pack(path, expected_split="gallery") for path in args.gallery]
    if [pack["checkpoint_sha256"] for pack in query_packs] != [
        pack["checkpoint_sha256"] for pack in gallery_packs
    ]:
        raise ValueError("query/gallery checkpoint digests do not pair by seed")
    result = measure(query_packs, gallery_packs, chunk_size=args.chunk_size)
    _write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
