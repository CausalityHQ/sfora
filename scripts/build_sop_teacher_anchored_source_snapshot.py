#!/usr/bin/env python3
"""Build the authenticated UNICOM B/16 official-training-only SOP snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import stat
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from sfora.atomic_publication import publish_large_writer_noreplace
from sfora.nested_rank_protocol import ordered_training_records_sha256

_SOURCE_ARCHIVE_SHA256 = "6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818"
_SOURCE_TRAIN_EMBEDDINGS_SHA256 = "d88e9d35f8419c7a661bd1358c901ecb2c64d4111ecd6ec311229d0d7c76dd74"
_SOURCE_TEST_EMBEDDINGS_SHA256 = "52b7b1fa8c2668468c9ac8983a8dd1d98d8c04ca0684bec79c2c31ee5666bfa9"
_TEACHER_SNAPSHOT_SHA256 = "b0da9f6097646ffad78c21c84970751ae9a7da003e0eb2979e28f72009785264"
_SOURCE_MODEL_IDENTIFIER = "UNICOM-ViT-B/16"
_SOURCE_TRANSFORM = "official UNICOM ViT-B/16 load_model_and_transform"
_TRAIN_ARRAYS = (
    "train_embeddings",
    "train_image_ids",
    "train_labels",
    "train_relative_paths",
)


@dataclass(frozen=True, slots=True)
class SourceSnapshotModel:
    """Process-local encoder plus immutable source-snapshot identity."""

    encoder: nn.Module
    transform: Callable[[Any], torch.Tensor]
    model_identifier: str
    model_revision: str
    checkpoint_sha256: str
    source_archive_sha256: str
    excluded_test_embeddings_sha256: str
    transform_name: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(size - offset, 1024 * 1024), offset)
        if not chunk:
            raise RuntimeError("source snapshot read was truncated")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _is_digest(value: object, length: int = 64) -> bool:
    return (
        type(value) is str and len(value) == length and not (set(value) - set("0123456789abcdef"))
    )


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the local-only train-snapshot construction boundary."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--teacher-snapshot", required=True, type=_absolute_path)
    parser.add_argument("--teacher-snapshot-sha256", required=True)
    parser.add_argument("--image-root", required=True, type=_absolute_path)
    parser.add_argument("--unicom-checkout", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint", required=True, type=_absolute_path)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--execute-build-source-snapshot", required=True, action="store_true")
    result = parser.parse_args(arguments)
    if result.teacher_snapshot_sha256 != _TEACHER_SNAPSHOT_SHA256:
        parser.error("teacher snapshot SHA-256 differs")
    if result.batch_size != 64:
        parser.error("batch size must equal 64")
    return result


def _load_snapshot(path: Path, sha256: str) -> Any:
    script = Path(__file__).resolve().parent / "train_sop_nested_neighborhood_rank.py"
    name = "sfora_source_snapshot_train_loader"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise ValueError("teacher snapshot authority differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module.load_train_snapshot(path, sha256)
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError("teacher snapshot authority differs") from error
    finally:
        sys.modules.pop(name, None)


def _load_source_model(checkout: Path, checkpoint: Path) -> Any:
    script = Path(__file__).resolve().parent / "sop_teacher_anchored_runtime.py"
    name = "sfora_source_snapshot_model_loader"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise ValueError("source model authority differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module.load_authenticated_source_model(checkout, checkpoint)
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError("source model authority differs") from error
    finally:
        sys.modules.pop(name, None)


def _read_training_image(root: Path, relative: str) -> tuple[bytes, int, bytes]:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or not pure.parts
        or "." in pure.parts
        or ".." in pure.parts
        or str(pure) != relative
    ):
        raise ValueError("training image authority differs")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if type(no_follow) is not int:
        raise ValueError("training image authority differs")
    directories: list[int] = []
    try:
        root_info = root.lstat()
        if root.is_symlink() or not stat.S_ISDIR(root_info.st_mode):
            raise ValueError("training image authority differs")
        directory = os.open(root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | no_follow)
        directories.append(directory)
        opened_root = os.fstat(directory)
        if (opened_root.st_dev, opened_root.st_ino) != (root_info.st_dev, root_info.st_ino):
            raise ValueError("training image authority differs")
        for component in pure.parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | no_follow,
                dir_fd=directory,
            )
            directories.append(child)
            child_info = os.fstat(child)
            if not stat.S_ISDIR(child_info.st_mode):
                raise ValueError("training image authority differs")
            directory = child
        filename = pure.parts[-1]
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_CLOEXEC | no_follow,
            dir_fd=directory,
        )
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            opened = os.fstat(stream.fileno())
            chunks: list[bytes] = []
            while chunk := stream.read(1024 * 1024):
                chunks.append(chunk)
            payload = b"".join(chunks)
            final_opened = os.fstat(stream.fileno())
        final_path = os.stat(filename, dir_fd=directory, follow_symlinks=False)
        final_root = root.lstat()
        identity = (opened.st_dev, opened.st_ino, opened.st_size)
        if (
            not stat.S_ISREG(opened.st_mode)
            or identity != (final_opened.st_dev, final_opened.st_ino, final_opened.st_size)
            or identity != (final_path.st_dev, final_path.st_ino, final_path.st_size)
            or (final_root.st_dev, final_root.st_ino) != (root_info.st_dev, root_info.st_ino)
        ):
            raise ValueError("training image authority differs")
    except OSError as error:
        raise ValueError("training image authority differs") from error
    finally:
        for directory_descriptor in reversed(directories):
            os.close(directory_descriptor)
    return payload, len(payload), hashlib.sha256(payload).digest()


def _validate_source_model(value: SourceSnapshotModel) -> None:
    if (
        type(value) is not SourceSnapshotModel
        or not isinstance(value.encoder, nn.Module)
        or not callable(value.transform)
        or not value.model_identifier
        or not _is_digest(value.model_revision, 40)
        or not _is_digest(value.checkpoint_sha256)
        or not _is_digest(value.source_archive_sha256)
        or not _is_digest(value.excluded_test_embeddings_sha256)
        or not value.transform_name
    ):
        raise ValueError("source model authority differs")


def build_source_snapshot(
    *,
    teacher_snapshot: Path,
    teacher_snapshot_sha256: str,
    image_root: Path,
    output: Path,
    source_model: SourceSnapshotModel,
    expected_train_embeddings_sha256: str,
    batch_size: int,
) -> dict[str, object]:
    """Encode only teacher-named training images and publish one immutable snapshot."""

    _validate_source_model(source_model)
    if (
        not isinstance(teacher_snapshot, Path)
        or not _is_digest(teacher_snapshot_sha256)
        or not isinstance(image_root, Path)
        or image_root.is_symlink()
        or not image_root.is_dir()
        or not isinstance(output, Path)
        or output.is_symlink()
        or output.exists()
        or output.parent.is_symlink()
        or not output.parent.is_dir()
        or not _is_digest(expected_train_embeddings_sha256)
        or type(batch_size) is not int
        or batch_size < 1
    ):
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        raise ValueError("source snapshot authority differs")
    teacher = _load_snapshot(teacher_snapshot, teacher_snapshot_sha256)
    metadata = cast(dict[str, object], teacher.metadata)
    paths = tuple(str(value) for value in teacher.relative_paths)
    if (
        metadata.get("model_identifier") != "UNICOM-ViT-L/14@336px"
        or metadata.get("model_revision") != source_model.model_revision
        or len(paths) != len(teacher.labels)
        or not paths
    ):
        raise ValueError("teacher snapshot authority differs")

    device = next(source_model.encoder.parameters(), torch.empty(0)).device
    source_model.encoder.eval()
    image_manifest = hashlib.sha256(b"sfora-teacher-anchored-image-tree-v1\x00")
    encoded: list[NDArray[np.float32]] = []
    for start in range(0, len(paths), batch_size):
        tensors: list[torch.Tensor] = []
        for relative in paths[start : start + batch_size]:
            payload, size, digest = _read_training_image(image_root, relative)
            relative_bytes = relative.encode("utf-8")
            image_manifest.update(struct.pack("<Q", len(relative_bytes)))
            image_manifest.update(relative_bytes)
            image_manifest.update(struct.pack("<Q", size))
            image_manifest.update(digest)
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    tensor = source_model.transform(image.convert("RGB"))
            except (OSError, ValueError) as error:
                raise ValueError("training image authority differs") from error
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise ValueError("training image transform differs")
            tensors.append(tensor)
        retained_rows = len(tensors)
        if retained_rows < batch_size:
            tensors.extend(tensors[-1] for _ in range(batch_size - retained_rows))
        batch = torch.stack(tensors).to(device=device, non_blocking=False)
        with torch.inference_mode():
            values = source_model.encoder(batch)
        array = np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32)
        if (
            array.ndim != 2
            or array.shape[0] != batch_size
            or array.shape[1] < 2
            or not np.isfinite(array).all()
            or np.any(np.linalg.norm(array.astype(np.float64), axis=1) == 0.0)
        ):
            raise ValueError("source embedding authority differs")
        encoded.append(np.ascontiguousarray(array[:retained_rows], dtype=np.float32))
    embeddings = np.ascontiguousarray(np.concatenate(encoded), dtype=np.float32)
    embedding_sha256 = _sha256_bytes(embeddings.tobytes(order="C"))
    if embedding_sha256 != expected_train_embeddings_sha256:
        raise ValueError("source embedding digest differs")

    arrays: dict[str, np.ndarray] = {
        "train_embeddings": embeddings,
        "train_labels": np.ascontiguousarray(teacher.labels, dtype=np.int64),
        "train_image_ids": np.ascontiguousarray(teacher.image_ids, dtype=np.int64),
        "train_relative_paths": np.ascontiguousarray(teacher.relative_paths),
    }
    teacher_test = cast(dict[str, object], metadata["excluded_test_array_sha256"])
    snapshot_metadata = {
        "schema": "sfora-nnrl-sop-train-snapshot-v1",
        "source_archive_sha256": source_model.source_archive_sha256,
        "model_identifier": source_model.model_identifier,
        "model_revision": source_model.model_revision,
        "checkpoint_sha256": source_model.checkpoint_sha256,
        "embedding_dimension": embeddings.shape[1],
        "train_rows": embeddings.shape[0],
        "train_classes": len(set(arrays["train_labels"].tolist())),
        "train_array_sha256": {
            name: _sha256_bytes(arrays[name].tobytes(order="C")) for name in _TRAIN_ARRAYS
        },
        "excluded_test_array_sha256": {
            "test_embeddings": source_model.excluded_test_embeddings_sha256,
            "test_labels": teacher_test["test_labels"],
            "test_image_ids": teacher_test["test_image_ids"],
            "test_relative_paths": teacher_test["test_relative_paths"],
        },
        "ordered_train_record_sha256": ordered_training_records_sha256(
            arrays["train_image_ids"],
            arrays["train_labels"],
            tuple(str(value) for value in arrays["train_relative_paths"]),
        ),
        "transform": source_model.transform_name,
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

    validated_output_sha256: str | None = None

    def validator(descriptor: int, size: int) -> None:
        nonlocal validated_output_sha256
        if size <= 0:
            raise ValueError("source snapshot is empty")
        with (
            os.fdopen(os.dup(descriptor), "rb") as stream,
            np.load(stream, allow_pickle=False) as snapshot,
        ):
            if set(snapshot.files) != set(_TRAIN_ARRAYS) | {"metadata_json"}:
                raise ValueError("source snapshot schema differs")
            if json.loads(str(snapshot["metadata_json"].item())) != snapshot_metadata:
                raise ValueError("source snapshot metadata differs")
            if any(not np.array_equal(snapshot[name], arrays[name]) for name in _TRAIN_ARRAYS):
                raise ValueError("source snapshot array differs")
        validated_output_sha256 = _sha256_descriptor(descriptor, size)

    published = publish_large_writer_noreplace(output, writer, validator=validator)
    try:
        current = output.lstat()
        if (current.st_dev, current.st_ino) != published.identity:
            raise ValueError("source snapshot publication ownership differs")
        if current.st_size != published.size:
            raise ValueError("source snapshot publication content differs")
        output_sha256 = _sha256_descriptor(published.descriptor, published.size)
        if validated_output_sha256 is None or output_sha256 != validated_output_sha256:
            raise ValueError("source snapshot publication content differs")
        output_bytes = published.size
        current = output.lstat()
        if (current.st_dev, current.st_ino) != published.identity:
            raise ValueError("source snapshot publication ownership differs")
        if current.st_size != published.size:
            raise ValueError("source snapshot publication content differs")
    finally:
        published.close()
    return {
        "claim_eligible": False,
        "image_manifest_sha256": image_manifest.hexdigest(),
        "output": str(output.resolve()),
        "output_bytes": output_bytes,
        "output_sha256": output_sha256,
        "schema": "sfora-teacher-anchored-source-snapshot-result-v1",
        "train_embeddings_sha256": embedding_sha256,
        "train_rows": embeddings.shape[0],
    }


def main(arguments: Sequence[str] | None = None) -> int:
    """Build one authenticated B/16 training-only snapshot."""

    try:
        args = parse_args(arguments)
        loaded = _load_source_model(args.unicom_checkout, args.checkpoint)
        loaded.encoder.cuda()
        model = SourceSnapshotModel(
            encoder=loaded.encoder,
            transform=loaded.transform,
            model_identifier=_SOURCE_MODEL_IDENTIFIER,
            model_revision=loaded.revision,
            checkpoint_sha256=loaded.checkpoint_sha256,
            source_archive_sha256=_SOURCE_ARCHIVE_SHA256,
            excluded_test_embeddings_sha256=_SOURCE_TEST_EMBEDDINGS_SHA256,
            transform_name=_SOURCE_TRANSFORM,
        )
        result = build_source_snapshot(
            teacher_snapshot=args.teacher_snapshot,
            teacher_snapshot_sha256=args.teacher_snapshot_sha256,
            image_root=args.image_root,
            output=args.output,
            source_model=model,
            expected_train_embeddings_sha256=_SOURCE_TRAIN_EMBEDDINGS_SHA256,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"SOP source snapshot failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
