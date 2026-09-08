from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_sop_nnrl_train_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_sop_nnrl_train_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_train_snapshot = MODULE.build_train_snapshot
parse_args = MODULE.parse_args


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _source_archive(path: Path, *, corrupt_train_digest: bool = False) -> str:
    arrays = {
        "train_embeddings": np.arange(24, dtype=np.float32).reshape(6, 4) + 1,
        "train_labels": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        "train_image_ids": np.arange(10, 16, dtype=np.int64),
        "train_relative_paths": np.asarray([f"train/{index}.jpg" for index in range(6)]),
        "test_embeddings": np.arange(16, dtype=np.float32).reshape(4, 4) + 1,
        "test_labels": np.asarray([3, 3, 4, 4], dtype=np.int64),
        "test_image_ids": np.arange(20, 24, dtype=np.int64),
        "test_relative_paths": np.asarray([f"test/{index}.jpg" for index in range(4)]),
    }
    digests = {name: _digest(value) for name, value in arrays.items()}
    if corrupt_train_digest:
        digests["train_embeddings"] = "0" * 64
    metadata = {
        "schema": "sfora-unicom-sop-embeddings-v1",
        "model_identifier": "fixture-teacher",
        "embedding_dimension": 4,
        "split_counts": {"train": 6, "test": 4},
        "split_classes": {"train": 3, "test": 2},
        "array_sha256": digests,
        "checkpoint_sha256": "1" * 64,
        "model_revision": "2" * 40,
        "ordered_record_sha256": {"train": "3" * 64, "test": "4" * 64},
        "transform": "fixture canonical transform",
    }
    buffer = io.BytesIO()
    np.savez(
        buffer,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        **arrays,
    )
    payload = buffer.getvalue()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_snapshot_contains_only_authenticated_official_training_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    source_sha256 = _source_archive(source)
    output = tmp_path / "train-only.npz"

    result = build_train_snapshot(source, source_sha256, output)

    assert result["source_archive_sha256"] == source_sha256
    assert result["train_rows"] == 6
    with np.load(output, allow_pickle=False) as snapshot:
        assert set(snapshot.files) == {
            "metadata_json",
            "train_embeddings",
            "train_labels",
            "train_image_ids",
            "train_relative_paths",
        }
        metadata = json.loads(str(snapshot["metadata_json"].item()))
        assert metadata["schema"] == "sfora-nnrl-sop-train-snapshot-v1"
        assert metadata["source_archive_sha256"] == source_sha256
        assert set(metadata["excluded_test_array_sha256"]) == {
            "test_embeddings",
            "test_labels",
            "test_image_ids",
            "test_relative_paths",
        }
        assert np.array_equal(snapshot["train_labels"], np.asarray([0, 0, 1, 1, 2, 2]))


def test_snapshot_rejects_digest_schema_train_drift_and_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    source_sha256 = _source_archive(source)
    with pytest.raises(ValueError, match="source archive digest"):
        build_train_snapshot(source, "0" * 64, tmp_path / "wrong.npz")

    corrupt = tmp_path / "corrupt.npz"
    corrupt_sha256 = _source_archive(corrupt, corrupt_train_digest=True)
    with pytest.raises(ValueError, match="train array digest"):
        build_train_snapshot(corrupt, corrupt_sha256, tmp_path / "corrupt-output.npz")

    output = tmp_path / "exists.npz"
    output.write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        build_train_snapshot(source, source_sha256, output)


def test_snapshot_cli_is_explicit_local_only_and_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    args = parse_args(
        [
            "--source-archive",
            str(source.resolve()),
            "--source-archive-sha256",
            "a" * 64,
            "--output",
            str(output.resolve()),
            "--execute-build-snapshot",
        ]
    )
    assert args.source_archive == source.resolve()
    for forbidden in ("--test-archive", "--url", "--s3", "--class-names"):
        with pytest.raises(SystemExit):
            parse_args(
                [
                    "--source-archive",
                    str(source.resolve()),
                    "--source-archive-sha256",
                    "a" * 64,
                    "--output",
                    str(output.resolve()),
                    "--execute-build-snapshot",
                    forbidden,
                    "value",
                ]
            )
