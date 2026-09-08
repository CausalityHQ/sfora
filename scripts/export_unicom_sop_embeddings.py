#!/usr/bin/env python3
"""Export authenticated official UNICOM embeddings for Stanford Online Products."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

SOP_EXPECTED_COUNTS = (59_551, 60_502)
SOP_EXPECTED_CLASSES = (11_318, 11_316)
_SOP_HEADER = "image_id class_id super_class_id path"
UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"


@dataclass(frozen=True, slots=True)
class SopRecord:
    """One ordered record from an official SOP split file."""

    split: str
    image_id: int
    label: int
    super_class_id: int
    relative_path: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class ModelAuthority:
    """Frozen official UNICOM model identity."""

    identifier: str
    load_name: str
    checkpoint_filename: str
    checkpoint_sha256: str
    revision: str = UNICOM_REVISION


_MODELS = {
    "b16": ModelAuthority(
        identifier="UNICOM-ViT-B/16",
        load_name="ViT-B/16",
        checkpoint_filename="FP16-ViT-B-16.pt",
        checkpoint_sha256="c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef",
    ),
    "l14-336": ModelAuthority(
        identifier="UNICOM-ViT-L/14@336px",
        load_name="ViT-L/14@336px",
        checkpoint_filename="FP16-ViT-L-14-336px.pt",
        checkpoint_sha256="3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
    ),
}


def model_authority(name: str) -> ModelAuthority:
    """Resolve one of the two preregistered model identities."""

    if type(name) is not str or name not in _MODELS:
        raise ValueError("SOP UNICOM model differs")
    return _MODELS[name]


def _parse_split(dataset_root: Path, split: str) -> tuple[SopRecord, ...]:
    metadata = dataset_root / ("Ebay_train.txt" if split == "train" else "Ebay_test.txt")
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != _SOP_HEADER:
        raise ValueError(f"SOP {split} header differs")
    records = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"SOP {split} row {line_number} differs")
        try:
            image_id, label, super_class_id = (int(value) for value in fields[:3])
        except ValueError as error:
            raise ValueError(f"SOP {split} row {line_number} differs") from error
        relative = PurePosixPath(fields[3])
        if (
            image_id < 1
            or label < 1
            or super_class_id < 1
            or relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != fields[3]
        ):
            raise ValueError(f"SOP {split} row {line_number} differs")
        image_path = dataset_root.joinpath(*relative.parts)
        if not image_path.is_file() or image_path.is_symlink():
            raise ValueError(f"SOP image is not regular: {image_path}")
        records.append(
            SopRecord(
                split=split,
                image_id=image_id,
                label=label,
                super_class_id=super_class_id,
                relative_path=str(relative),
                image_path=image_path,
            )
        )
    return tuple(records)


def parse_sop_records(
    dataset_root: Path,
    *,
    expected_counts: tuple[int, int] = SOP_EXPECTED_COUNTS,
    expected_classes: tuple[int, int] = SOP_EXPECTED_CLASSES,
) -> tuple[SopRecord, ...]:
    """Parse the official train/test protocol without reordering records."""

    if (
        not isinstance(dataset_root, Path)
        or type(expected_counts) is not tuple
        or len(expected_counts) != 2
        or any(type(value) is not int or value < 1 for value in expected_counts)
        or type(expected_classes) is not tuple
        or len(expected_classes) != 2
        or any(type(value) is not int or value < 1 for value in expected_classes)
    ):
        raise ValueError("SOP parser authority differs")
    train = _parse_split(dataset_root, "train")
    test = _parse_split(dataset_root, "test")
    if (len(train), len(test)) != expected_counts:
        raise ValueError("SOP split count differs")
    train_labels = {record.label for record in train}
    test_labels = {record.label for record in test}
    if (len(train_labels), len(test_labels)) != expected_classes:
        raise ValueError("SOP class count differs")
    if train_labels & test_labels:
        raise ValueError("SOP train/test classes are not disjoint")
    image_ids = [record.image_id for record in train + test]
    relative_paths = [record.relative_path for record in train + test]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("SOP image ID is duplicate")
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("SOP image path is duplicate")
    return train + test


def ordered_record_sha256(records: tuple[SopRecord, ...]) -> str:
    """Hash exact ordered protocol identities independently of host paths."""

    if type(records) is not tuple or not records or any(type(x) is not SopRecord for x in records):
        raise ValueError("SOP record authority differs")
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.split.encode())
        digest.update(b"\0")
        digest.update(str(record.image_id).encode())
        digest.update(b"\0")
        digest.update(str(record.label).encode())
        digest.update(b"\0")
        digest.update(str(record.super_class_id).encode())
        digest.update(b"\0")
        digest.update(record.relative_path.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _validate_metadata_base(metadata: Mapping[str, object]) -> None:
    if type(metadata) is not dict or set(metadata) != {
        "model_identifier",
        "model_revision",
        "checkpoint_sha256",
        "transform",
    }:
        raise ValueError("SOP model metadata differs")
    if type(metadata["model_identifier"]) is not str or not metadata["model_identifier"]:
        raise ValueError("SOP model metadata differs")
    if (
        type(metadata["model_revision"]) is not str
        or len(metadata["model_revision"]) != 40
        or set(metadata["model_revision"]) - set("0123456789abcdef")
        or type(metadata["checkpoint_sha256"]) is not str
        or len(metadata["checkpoint_sha256"]) != 64
        or set(metadata["checkpoint_sha256"]) - set("0123456789abcdef")
        or type(metadata["transform"]) is not str
        or not metadata["transform"]
    ):
        raise ValueError("SOP model metadata differs")


def load_sop_embedding_archive(
    path: Path,
    *,
    expected_counts: tuple[int, int] = SOP_EXPECTED_COUNTS,
    expected_classes: tuple[int, int] = SOP_EXPECTED_CLASSES,
    expected_dimension: int = 768,
) -> dict[str, object]:
    """Load and fully validate one locally authenticated SOP archive."""

    required = {
        "metadata_json",
        "train_embeddings",
        "train_labels",
        "train_image_ids",
        "train_relative_paths",
        "test_embeddings",
        "test_labels",
        "test_image_ids",
        "test_relative_paths",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("SOP embedding archive keys differ")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in required - {"metadata_json"}}
    if (
        type(metadata) is not dict
        or metadata.get("schema") != "sfora-unicom-sop-embeddings-v1"
        or metadata.get("embedding_dimension") != expected_dimension
        or metadata.get("split_counts") != {"test": expected_counts[1], "train": expected_counts[0]}
        or metadata.get("split_classes")
        != {"test": expected_classes[1], "train": expected_classes[0]}
        or type(metadata.get("ordered_record_sha256")) is not str
        or len(metadata["ordered_record_sha256"]) != 64
        or type(metadata.get("array_sha256")) is not dict
        or set(metadata["array_sha256"]) != set(arrays)
    ):
        raise ValueError("SOP embedding metadata differs")
    _validate_metadata_base(
        {
            key: metadata.get(key)
            for key in ("model_identifier", "model_revision", "checkpoint_sha256", "transform")
        }
    )
    for name, values in arrays.items():
        if _array_sha256(values) != metadata["array_sha256"].get(name):
            raise ValueError("SOP embedding array digest differs")
    train = arrays["train_embeddings"]
    test = arrays["test_embeddings"]
    train_labels = arrays["train_labels"]
    test_labels = arrays["test_labels"]
    train_ids = arrays["train_image_ids"]
    test_ids = arrays["test_image_ids"]
    train_paths = arrays["train_relative_paths"]
    test_paths = arrays["test_relative_paths"]
    if (
        train.dtype != np.float32
        or test.dtype != np.float32
        or train.shape != (expected_counts[0], expected_dimension)
        or test.shape != (expected_counts[1], expected_dimension)
        or not train.flags.c_contiguous
        or not test.flags.c_contiguous
        or not np.isfinite(train).all()
        or not np.isfinite(test).all()
        or np.any(np.linalg.norm(train.astype(np.float64), axis=1) == 0.0)
        or np.any(np.linalg.norm(test.astype(np.float64), axis=1) == 0.0)
        or train_labels.dtype != np.int64
        or test_labels.dtype != np.int64
        or train_ids.dtype != np.int64
        or test_ids.dtype != np.int64
        or train_labels.shape != (expected_counts[0],)
        or test_labels.shape != (expected_counts[1],)
        or train_ids.shape != (expected_counts[0],)
        or test_ids.shape != (expected_counts[1],)
        or train_paths.dtype.kind != "U"
        or test_paths.dtype.kind != "U"
        or train_paths.shape != (expected_counts[0],)
        or test_paths.shape != (expected_counts[1],)
        or len(set(train_labels.tolist())) != expected_classes[0]
        or len(set(test_labels.tolist())) != expected_classes[1]
        or set(train_labels.tolist()) & set(test_labels.tolist())
        or len(set(train_ids.tolist() + test_ids.tolist())) != sum(expected_counts)
        or len(set(train_paths.tolist() + test_paths.tolist())) != sum(expected_counts)
    ):
        raise ValueError("SOP embedding array authority differs")
    return {"metadata": metadata, **arrays}


def export_sop_embeddings(
    records: tuple[SopRecord, ...],
    encode_batch: Callable[[tuple[Path, ...]], np.ndarray],
    metadata_base: Mapping[str, object],
    output: Path,
    *,
    batch_size: int = 64,
    expected_counts: tuple[int, int] = SOP_EXPECTED_COUNTS,
    expected_classes: tuple[int, int] = SOP_EXPECTED_CLASSES,
) -> None:
    """Encode official rows and exclusively publish one validated archive."""

    if (
        type(records) is not tuple
        or not records
        or any(type(record) is not SopRecord for record in records)
        or not callable(encode_batch)
        or not isinstance(output, Path)
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("SOP export authority differs")
    _validate_metadata_base(metadata_base)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent = output.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or output.parent.is_symlink():
        raise ValueError("SOP output parent is not a real directory")
    split_counts = tuple(
        sum(record.split == split for record in records) for split in ("train", "test")
    )
    split_classes = tuple(
        len({record.label for record in records if record.split == split})
        for split in ("train", "test")
    )
    if split_counts != expected_counts or split_classes != expected_classes:
        raise ValueError("SOP export protocol differs")
    parts = []
    dimension = None
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        values = encode_batch(tuple(record.image_path for record in batch))
        if (
            type(values) is not np.ndarray
            or values.dtype != np.float32
            or values.ndim != 2
            or values.shape[0] != len(batch)
            or values.shape[1] < 2
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
            or np.any(np.linalg.norm(values.astype(np.float64), axis=1) == 0.0)
            or (dimension is not None and values.shape[1] != dimension)
        ):
            raise ValueError("SOP encoded batch differs")
        dimension = values.shape[1]
        parts.append(values.copy())
    assert dimension is not None
    all_embeddings = np.ascontiguousarray(np.concatenate(parts))
    arrays: dict[str, np.ndarray] = {}
    for split in ("train", "test"):
        indexes = [index for index, record in enumerate(records) if record.split == split]
        width = max(len(records[index].relative_path) for index in indexes)
        arrays[f"{split}_embeddings"] = np.ascontiguousarray(all_embeddings[indexes])
        arrays[f"{split}_labels"] = np.asarray(
            [records[index].label for index in indexes], dtype=np.int64
        )
        arrays[f"{split}_image_ids"] = np.asarray(
            [records[index].image_id for index in indexes], dtype=np.int64
        )
        arrays[f"{split}_relative_paths"] = np.asarray(
            [records[index].relative_path for index in indexes], dtype=f"<U{width}"
        )
    metadata = {
        "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
        "checkpoint_sha256": metadata_base["checkpoint_sha256"],
        "embedding_dimension": dimension,
        "model_identifier": metadata_base["model_identifier"],
        "model_revision": metadata_base["model_revision"],
        "ordered_record_sha256": ordered_record_sha256(records),
        "schema": "sfora-unicom-sop-embeddings-v1",
        "split_classes": {"test": expected_classes[1], "train": expected_classes[0]},
        "split_counts": {"test": expected_counts[1], "train": expected_counts[0]},
        "transform": metadata_base["transform"],
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    descriptor = None
    owned = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            np.savez(
                stream,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
                train_embeddings=arrays["train_embeddings"],
                train_labels=arrays["train_labels"],
                train_image_ids=arrays["train_image_ids"],
                train_relative_paths=arrays["train_relative_paths"],
                test_embeddings=arrays["test_embeddings"],
                test_labels=arrays["test_labels"],
                test_image_ids=arrays["test_image_ids"],
                test_relative_paths=arrays["test_relative_paths"],
            )
            stream.flush()
            os.fsync(stream.fileno())
        load_sop_embedding_archive(
            temporary,
            expected_counts=expected_counts,
            expected_classes=expected_classes,
            expected_dimension=dimension,
        )
        os.link(temporary, output)
        temporary.unlink()
        load_sop_embedding_archive(
            output,
            expected_counts=expected_counts,
            expected_classes=expected_classes,
            expected_dimension=dimension,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            info = temporary.lstat()
        except FileNotFoundError:
            pass
        else:
            if owned is not None and (info.st_dev, info.st_ino) == owned:
                temporary.unlink()


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only exporter surface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-checkout", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint", required=True, type=_absolute_path)
    parser.add_argument("--dataset-root", required=True, type=_absolute_path)
    parser.add_argument("--model", required=True, choices=tuple(_MODELS))
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--execute-export", action="store_true", required=True)
    return parser.parse_args(arguments)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _official_encoder(
    checkout: Path, checkpoint: Path, authority: ModelAuthority
) -> Callable[[tuple[Path, ...]], np.ndarray]:
    if checkpoint.name != authority.checkpoint_filename:
        raise ValueError("SOP checkpoint filename differs")
    if _git_revision(checkout) != authority.revision:
        raise ValueError("SOP UNICOM revision differs")
    if _sha256_file(checkpoint) != authority.checkpoint_sha256:
        raise ValueError("SOP checkpoint digest differs")
    import torch
    from PIL import Image

    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    module_file = getattr(unicom, "__file__", None)
    if (
        type(module_file) is not str
        or Path(module_file).resolve().parent != package_root / "unicom"
    ):
        raise ValueError("SOP imported UNICOM package differs")
    model, transform = unicom.load(authority.load_name, download_root=str(checkpoint.parent))
    model = model.cuda().eval()

    def encode(paths: tuple[Path, ...]) -> np.ndarray:
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).cuda(non_blocking=False)
        with torch.inference_mode():
            values = model(batch)
        return np.ascontiguousarray(values.float().cpu().numpy(), dtype=np.float32)

    return encode


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one authenticated model export."""

    args = parse_args(arguments)
    try:
        authority = model_authority(args.model)
        records = parse_sop_records(args.dataset_root)
        encoder = _official_encoder(args.unicom_checkout, args.checkpoint, authority)
        export_sop_embeddings(
            records,
            encoder,
            {
                "model_identifier": authority.identifier,
                "model_revision": authority.revision,
                "checkpoint_sha256": authority.checkpoint_sha256,
                "transform": f"official UNICOM {authority.load_name} load_model_and_transform",
            },
            args.output,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"SOP export failed: {error}", file=sys.stderr)
        return 2
    print(f"SOP export complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
