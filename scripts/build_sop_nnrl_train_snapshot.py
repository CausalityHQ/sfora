#!/usr/bin/env python3
"""Extract an authenticated official-training-only NNRL teacher snapshot."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from sfora.atomic_publication import publish_large_writer_noreplace

_SOURCE_ARRAYS = {
    "train_embeddings",
    "train_labels",
    "train_image_ids",
    "train_relative_paths",
    "test_embeddings",
    "test_labels",
    "test_image_ids",
    "test_relative_paths",
}
_TRAIN_ARRAYS = tuple(sorted(name for name in _SOURCE_ARRAYS if name.startswith("train_")))
_TEST_ARRAYS = tuple(sorted(name for name in _SOURCE_ARRAYS if name.startswith("test_")))
_METADATA_KEYS = {
    "array_sha256",
    "checkpoint_sha256",
    "embedding_dimension",
    "model_identifier",
    "model_revision",
    "ordered_record_sha256",
    "schema",
    "split_classes",
    "split_counts",
    "transform",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _sha256(value: str) -> str:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256")
    return value


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the local-only, explicit snapshot boundary."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", required=True, type=_absolute_path)
    parser.add_argument("--source-archive-sha256", required=True, type=_sha256)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--execute-build-snapshot", required=True, action="store_true")
    return parser.parse_args(arguments)


def _validated_source(
    source: Path, source_sha256: str
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if (
        not isinstance(source, Path)
        or source.is_symlink()
        or not source.is_file()
        or type(source_sha256) is not str
        or len(source_sha256) != 64
        or set(source_sha256) - set("0123456789abcdef")
    ):
        raise ValueError("source archive authority differs")
    payload = source.read_bytes()
    if _sha256_bytes(payload) != source_sha256:
        raise ValueError("source archive digest differs")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != _SOURCE_ARRAYS | {"metadata_json"}:
            raise ValueError("source archive schema differs")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in _TRAIN_ARRAYS}
    if (
        type(metadata) is not dict
        or set(metadata) != _METADATA_KEYS
        or metadata.get("schema") != "sfora-unicom-sop-embeddings-v1"
        or type(metadata.get("array_sha256")) is not dict
        or set(cast(dict[str, object], metadata["array_sha256"])) != _SOURCE_ARRAYS
        or type(metadata.get("split_counts")) is not dict
        or type(metadata.get("split_classes")) is not dict
        or type(metadata.get("embedding_dimension")) is not int
    ):
        raise ValueError("source archive metadata differs")
    digests = cast(dict[str, object], metadata["array_sha256"])
    if any(
        _sha256_bytes(array.tobytes(order="C")) != digests[name] for name, array in arrays.items()
    ):
        raise ValueError("train array digest differs")
    embeddings = arrays["train_embeddings"]
    labels = arrays["train_labels"]
    image_ids = arrays["train_image_ids"]
    paths = arrays["train_relative_paths"]
    train_count = cast(dict[str, object], metadata["split_counts"]).get("train")
    train_classes = cast(dict[str, object], metadata["split_classes"]).get("train")
    dimensions = metadata["embedding_dimension"]
    if (
        type(train_count) is not int
        or type(train_classes) is not int
        or type(dimensions) is not int
        or train_count <= 0
        or train_classes <= 1
        or dimensions <= 0
        or embeddings.dtype != np.float32
        or embeddings.shape != (train_count, dimensions)
        or not embeddings.flags.c_contiguous
        or not np.isfinite(embeddings).all()
        or np.any(np.linalg.norm(embeddings.astype(np.float64), axis=1) == 0.0)
        or labels.dtype != np.int64
        or labels.shape != (train_count,)
        or len(set(labels.tolist())) != train_classes
        or image_ids.dtype != np.int64
        or image_ids.shape != (train_count,)
        or len(set(image_ids.tolist())) != train_count
        or paths.dtype.kind != "U"
        or paths.shape != (train_count,)
        or len(set(paths.tolist())) != train_count
    ):
        raise ValueError("train array authority differs")
    if any(
        type(digests[name]) is not str
        or len(cast(str, digests[name])) != 64
        or set(cast(str, digests[name])) - set("0123456789abcdef")
        for name in _TEST_ARRAYS
    ):
        raise ValueError("excluded test digest authority differs")
    return metadata, arrays


def build_train_snapshot(source: Path, source_sha256: str, output: Path) -> dict[str, object]:
    """Publish train members only, with the source archive as byte authority."""

    metadata, arrays = _validated_source(source, source_sha256)
    source_digests = cast(dict[str, object], metadata["array_sha256"])
    snapshot_metadata = {
        "schema": "sfora-nnrl-sop-train-snapshot-v1",
        "source_archive_sha256": source_sha256,
        "model_identifier": metadata["model_identifier"],
        "model_revision": metadata["model_revision"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "embedding_dimension": metadata["embedding_dimension"],
        "train_rows": cast(dict[str, object], metadata["split_counts"])["train"],
        "train_classes": cast(dict[str, object], metadata["split_classes"])["train"],
        "train_array_sha256": {name: source_digests[name] for name in _TRAIN_ARRAYS},
        "excluded_test_array_sha256": {name: source_digests[name] for name in _TEST_ARRAYS},
        "ordered_train_record_sha256": cast(dict[str, object], metadata["ordered_record_sha256"])[
            "train"
        ],
        "transform": metadata["transform"],
    }
    metadata_json = json.dumps(
        snapshot_metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

    def writer(descriptor: int) -> None:
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            np.savez(
                stream,
                metadata_json=np.asarray(metadata_json),
                **arrays,  # type: ignore[arg-type]  # NumPy stubs misclassify array kwargs.
            )
            stream.flush()

    def validator(descriptor: int, size: int) -> None:
        if size <= 0:
            raise ValueError("train snapshot is empty")
        with (
            os.fdopen(os.dup(descriptor), "rb") as stream,
            np.load(stream, allow_pickle=False) as snapshot,
        ):
            if set(snapshot.files) != set(_TRAIN_ARRAYS) | {"metadata_json"}:
                raise ValueError("train snapshot schema differs")
            if json.loads(str(snapshot["metadata_json"].item())) != snapshot_metadata:
                raise ValueError("train snapshot metadata differs")
            if any(not np.array_equal(snapshot[name], arrays[name]) for name in _TRAIN_ARRAYS):
                raise ValueError("train snapshot array differs")

    published = publish_large_writer_noreplace(output, writer, validator=validator)
    published.close()
    return {
        "schema": "sfora-nnrl-sop-train-snapshot-result-v1",
        "claim_eligible": False,
        "source_archive_sha256": source_sha256,
        "output": str(output.resolve()),
        "output_sha256": _sha256_bytes(output.read_bytes()),
        "output_bytes": output.stat().st_size,
        "train_rows": snapshot_metadata["train_rows"],
    }


def main(arguments: Sequence[str] | None = None) -> int:
    """Build one immutable local snapshot."""

    try:
        args = parse_args(arguments)
        result = build_train_snapshot(args.source_archive, args.source_archive_sha256, args.output)
    except Exception as error:
        print(f"SOP NNRL snapshot failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
