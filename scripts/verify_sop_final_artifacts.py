#!/usr/bin/env python3
"""Fail-closed joint verification for final official SOP embedding artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED = {"train": (59_551, 11_318), "test": (60_502, 11_316)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(archive: Any, key: str) -> str:
    if key not in archive:
        raise ValueError(f"artifact lacks {key}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"artifact {key} is not scalar")
    return str(value.item())


def _load(path: Path, expected_split: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"embeddings", "labels", "example_ids", "source_paths"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{expected_split} artifact lacks {sorted(missing)}")
        if _scalar(archive, "artifact_selection") != "final_training_state":
            raise ValueError(f"{expected_split} artifact is not final_training_state")
        if _scalar(archive, "split") != expected_split:
            raise ValueError(f"artifact split does not equal {expected_split}")
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        example_ids = np.asarray(archive["example_ids"]).astype(str)
        source_paths = np.asarray(archive["source_paths"]).astype(str)
        checkpoint_hash = _scalar(archive, "checkpoint_sha256")
        report_hash = _scalar(archive, "report_sha256")

    expected_rows, expected_classes = EXPECTED[expected_split]
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(f"unexpected {expected_split} embedding shape {embeddings.shape}")
    if any(len(values) != expected_rows for values in (labels, example_ids, source_paths)):
        raise ValueError(f"{expected_split} artifact arrays differ in length")
    if len(np.unique(labels)) != expected_classes:
        raise ValueError(f"unexpected {expected_split} class count")
    if len(np.unique(example_ids)) != expected_rows:
        raise ValueError(f"duplicate {expected_split} example IDs")
    resolved_paths = np.asarray([str(Path(value).resolve()) for value in source_paths])
    if len(np.unique(resolved_paths)) != expected_rows:
        raise ValueError(f"duplicate {expected_split} source paths")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"non-finite {expected_split} embeddings")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError(f"{expected_split} embeddings are not unit normalized")
    return {
        "rows": expected_rows,
        "classes": expected_classes,
        "dimensions": int(embeddings.shape[1]),
        "labels": set(labels.tolist()),
        "example_ids": set(example_ids.tolist()),
        "source_paths": set(resolved_paths.tolist()),
        "checkpoint_sha256": checkpoint_hash,
        "report_sha256": report_hash,
    }


def verify(
    train_path: Path,
    test_path: Path,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    train = _load(train_path, "train")
    test = _load(test_path, "test")
    if train["dimensions"] != test["dimensions"]:
        raise ValueError("train/test embedding dimensions differ")
    for key, label in (
        ("labels", "class labels"),
        ("example_ids", "example IDs"),
        ("source_paths", "source paths"),
    ):
        if train[key] & test[key]:
            raise ValueError(f"SOP train/test {label} overlap")
    expected_checkpoint_hash = sha256(checkpoint_path)
    expected_report_hash = sha256(report_path)
    for split, payload in (("train", train), ("test", test)):
        if payload["checkpoint_sha256"] != expected_checkpoint_hash:
            raise ValueError(f"{split} checkpoint hash does not match checkpoint file")
        if payload["report_sha256"] != expected_report_hash:
            raise ValueError(f"{split} report hash does not match report file")
    return {
        "status": "verified",
        "artifact_selection": "final_training_state",
        "train_examples": train["rows"],
        "train_classes": train["classes"],
        "test_examples": test["rows"],
        "test_classes": test["classes"],
        "dimensions": train["dimensions"],
        "train_test_label_overlap": 0,
        "train_test_example_id_overlap": 0,
        "train_test_source_path_overlap": 0,
        "checkpoint_sha256": expected_checkpoint_hash,
        "report_sha256": expected_report_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.train, args.test, args.checkpoint, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
