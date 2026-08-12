#!/usr/bin/env python3
"""Export frozen official UNICOM embeddings for the In-Shop audit."""

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
from pathlib import Path

import numpy as np

from sfora.unicom_audit_io import load_embedding_bundle

UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
UNICOM_B16_SHA256 = "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef"
EXPECTED_COUNTS = (25_882, 14_218, 12_612)


@dataclass(frozen=True)
class InshopRecord:
    split: str
    image_path: Path
    label: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def parse_inshop_partition(
    dataset_root: Path,
    *,
    expected_counts: tuple[int, int, int] | None = EXPECTED_COUNTS,
) -> tuple[InshopRecord, ...]:
    dataset_root = Path(dataset_root)
    partition = dataset_root / "Eval" / "list_eval_partition.txt"
    lines = partition.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("In-Shop partition is truncated")
    try:
        declared = int(lines[0].strip())
    except ValueError as error:
        raise ValueError("In-Shop partition count is invalid") from error
    if lines[1].split() != ["image_name", "item_id", "evaluation_status"]:
        raise ValueError("In-Shop partition header differs")
    records: list[InshopRecord] = []
    counts = {"train": 0, "query": 0, "gallery": 0}
    for line_number, line in enumerate(lines[2:], start=3):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"In-Shop partition row {line_number} differs")
        image_name, label, split = fields
        if split not in counts or not label:
            raise ValueError(f"In-Shop partition row {line_number} differs")
        image_path = dataset_root / "Img" / image_name
        if not image_path.is_file() or image_path.is_symlink():
            raise ValueError(f"In-Shop image is missing or not regular: {image_path}")
        records.append(InshopRecord(split=split, image_path=image_path, label=label))
        counts[split] += 1
    if len(records) != declared:
        raise ValueError("In-Shop partition declared count differs")
    if expected_counts is not None and tuple(counts.values()) != expected_counts:
        raise ValueError("In-Shop split counts differ")
    query_labels = {record.label for record in records if record.split == "query"}
    gallery_labels = {record.label for record in records if record.split == "gallery"}
    if query_labels != gallery_labels:
        raise ValueError("In-Shop query/gallery identity membership differs")
    return tuple(records)


def _validate_metadata_base(metadata: Mapping[str, object]) -> None:
    keys = (
        "model_identifier",
        "model_revision",
        "checkpoint_sha256",
        "image_list_sha256",
        "transform",
    )
    if type(metadata) is not dict or tuple(metadata) != keys:
        raise ValueError("export metadata key order differs")
    if metadata["model_identifier"] != "UNICOM-ViT-B/16":
        raise ValueError("export model identifier differs")
    for key, length in (
        ("model_revision", 40),
        ("checkpoint_sha256", 64),
        ("image_list_sha256", 64),
    ):
        value = metadata[key]
        if (
            type(value) is not str
            or len(value) != length
            or not set(value) <= set("0123456789abcdef")
        ):
            raise ValueError(f"export {key} differs")
    if type(metadata["transform"]) is not str or not metadata["transform"]:
        raise TypeError("export transform must be a nonempty string")


def export_embeddings(
    records: Sequence[InshopRecord],
    encode_batch: Callable[[tuple[Path, ...]], np.ndarray],
    metadata_base: Mapping[str, object],
    output: Path,
    *,
    batch_size: int = 64,
    expected_counts: tuple[int, int, int] = EXPECTED_COUNTS,
) -> None:
    """Encode records in official order and exclusively publish one bundle."""

    if type(records) is not tuple or not records:
        raise TypeError("records must be a nonempty tuple")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive builtin integer")
    _validate_metadata_base(metadata_base)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    parent_info = output.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or output.parent.is_symlink():
        raise ValueError("output parent must be a real directory")

    encoded_rows: list[np.ndarray] = []
    dimension: int | None = None
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        paths = tuple(record.image_path for record in batch)
        values = encode_batch(paths)
        if (
            type(values) is not np.ndarray
            or values.dtype != np.float32
            or values.ndim != 2
            or values.shape[0] != len(batch)
            or values.shape[1] == 0
            or not values.flags.c_contiguous
            or not np.isfinite(values).all()
        ):
            raise ValueError("encoded batch must be finite C-contiguous FP32 with matching rows")
        if dimension is None:
            dimension = values.shape[1]
        elif values.shape[1] != dimension:
            raise ValueError("encoded batch dimension differs")
        if np.any(np.linalg.norm(values.astype(np.float64), axis=1) == 0.0):
            raise ValueError("encoded batch contains a zero-norm row")
        encoded_rows.append(values.copy())
    assert dimension is not None
    all_embeddings = np.ascontiguousarray(np.concatenate(encoded_rows, axis=0))

    split_embeddings: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    for split, expected_count in zip(
        ("train", "query", "gallery"), expected_counts, strict=True
    ):
        indices = [index for index, record in enumerate(records) if record.split == split]
        if len(indices) != expected_count:
            raise ValueError("record split count differs")
        split_embeddings[split] = np.ascontiguousarray(all_embeddings[indices])
        split_labels[split] = np.asarray([records[index].label for index in indices])
    if set(split_labels["query"].tolist()) != set(split_labels["gallery"].tolist()):
        raise ValueError("record query/gallery label membership differs")

    arrays = {
        "train_embeddings": split_embeddings["train"],
        "train_labels": split_labels["train"],
        "query_embeddings": split_embeddings["query"],
        "query_labels": split_labels["query"],
        "gallery_embeddings": split_embeddings["gallery"],
        "gallery_labels": split_labels["gallery"],
    }
    metadata = {
        "schema_version": 1,
        **metadata_base,
        "embedding_dimension": dimension,
        "split_counts": dict(zip(("train", "query", "gallery"), expected_counts, strict=True)),
        "array_sha256": {name: _sha256_array(values) for name, values in arrays.items()},
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    owned: tuple[int, int] | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            np.savez(
                handle,
                metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        load_embedding_bundle(
            temporary,
            expected_counts=expected_counts,
            expected_dimension=dimension,
        )
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary.unlink()
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        load_embedding_bundle(
            output,
            expected_counts=expected_counts,
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


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(arguments)


def _git_revision(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _official_encoder(checkout: Path, checkpoint: Path):
    import torch
    from PIL import Image

    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    if Path(unicom.__file__).resolve().parent != package_root / "unicom":
        raise ValueError("imported UNICOM package does not come from the pinned checkout")
    model, transform = unicom.load(
        "ViT-B/16", download_root=str(checkpoint.parent)
    )
    model = model.cuda().eval()

    def encode(paths: tuple[Path, ...]) -> np.ndarray:
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).cuda(non_blocking=False)
        with torch.inference_mode():
            output = model(batch)
        return np.ascontiguousarray(output.float().cpu().numpy(), dtype=np.float32)

    return encode


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    try:
        if _git_revision(args.unicom_checkout) != UNICOM_REVISION:
            raise ValueError("UNICOM checkout revision differs")
        if _sha256_file(args.checkpoint) != UNICOM_B16_SHA256:
            raise ValueError("UNICOM checkpoint SHA-256 differs")
        partition = args.dataset_root / "Eval" / "list_eval_partition.txt"
        records = parse_inshop_partition(args.dataset_root)
        encode = _official_encoder(args.unicom_checkout, args.checkpoint)
        export_embeddings(
            records,
            encode,
            {
                "model_identifier": "UNICOM-ViT-B/16",
                "model_revision": UNICOM_REVISION,
                "checkpoint_sha256": UNICOM_B16_SHA256,
                "image_list_sha256": _sha256_file(partition),
                "transform": "official UNICOM ViT-B/16 load_model_and_transform",
            },
            args.output,
            batch_size=args.batch_size,
        )
    except Exception as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 2
    print(f"export complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
