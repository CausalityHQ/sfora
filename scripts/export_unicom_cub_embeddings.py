#!/usr/bin/env python3
"""Export authenticated UNICOM embeddings for the CUB-200-2011 class split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

CUB_EXPECTED_COUNTS = (5_864, 5_924)
CUB_EXPECTED_CLASSES = (100, 100)
CUB_EXPECTED_TOTAL_CLASSES = 200
CUB_ARCHIVE_BYTES = 1_150_585_339
CUB_ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
CUB_ARCHIVE_SHA256 = "0c685df5597a8b24909f6a7c9db6d11e008733779a671760afef78feb49bf081"
CUB_PROTOCOL = "classes-001-100-train-101-200-test"


@dataclass(frozen=True, slots=True)
class CubRecord:
    """One image in the standard metric-learning class-disjoint protocol."""

    split: str
    image_id: int
    label: int
    class_name: str
    relative_path: str
    image_path: Path


def _parse_indexed_lines(path: Path, *, role: str) -> tuple[tuple[int, str], ...]:
    rows: list[tuple[int, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"CUB {role} row {line_number} differs")
        try:
            ordinal = int(fields[0])
        except ValueError as error:
            raise ValueError(f"CUB {role} row {line_number} differs") from error
        if ordinal < 1 or not fields[1]:
            raise ValueError(f"CUB {role} row {line_number} differs")
        rows.append((ordinal, fields[1]))
    if not rows or tuple(row[0] for row in rows) != tuple(range(1, len(rows) + 1)):
        raise ValueError(f"CUB {role} identity differs")
    return tuple(rows)


def parse_cub_records(
    dataset_root: Path,
    *,
    expected_counts: tuple[int, int] = CUB_EXPECTED_COUNTS,
    expected_classes: tuple[int, int] = CUB_EXPECTED_CLASSES,
    expected_total_classes: int = CUB_EXPECTED_TOTAL_CLASSES,
    train_class_max: int = 100,
) -> tuple[CubRecord, ...]:
    """Parse the frozen first-100/last-100 class-disjoint retrieval protocol."""

    if (
        not isinstance(dataset_root, Path)
        or type(expected_counts) is not tuple
        or len(expected_counts) != 2
        or any(type(value) is not int or value < 1 for value in expected_counts)
        or type(expected_classes) is not tuple
        or len(expected_classes) != 2
        or any(type(value) is not int or value < 1 for value in expected_classes)
        or type(expected_total_classes) is not int
        or expected_total_classes < 2
        or type(train_class_max) is not int
        or train_class_max < 1
    ):
        raise ValueError("CUB parser authority differs")
    images = _parse_indexed_lines(dataset_root / "images.txt", role="image")
    labels_raw = _parse_indexed_lines(dataset_root / "image_class_labels.txt", role="label")
    classes = dict(_parse_indexed_lines(dataset_root / "classes.txt", role="class"))
    if len(classes) != expected_total_classes:
        raise ValueError("CUB class count differs")
    if tuple(row[0] for row in images) != tuple(row[0] for row in labels_raw):
        raise ValueError("CUB image/label identity differs")
    records: list[CubRecord] = []
    for (image_id, path_text), (_, label_text) in zip(images, labels_raw, strict=True):
        try:
            label = int(label_text)
        except ValueError as error:
            raise ValueError("CUB label differs") from error
        relative = PurePosixPath(path_text)
        class_name = classes.get(label)
        if (
            class_name is None
            or relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != path_text
            or not relative.parts
            or relative.parts[0] != class_name
        ):
            raise ValueError("CUB class/path authority differs")
        image_path = dataset_root / "images" / Path(*relative.parts)
        if not image_path.is_file() or image_path.is_symlink():
            raise ValueError(f"CUB image is not regular: {image_path}")
        records.append(
            CubRecord(
                split="train" if label <= train_class_max else "test",
                image_id=image_id,
                label=label,
                class_name=class_name,
                relative_path=path_text,
                image_path=image_path,
            )
        )
    records.sort(key=lambda record: (record.split == "test", record.image_id))
    train = tuple(record for record in records if record.split == "train")
    test = tuple(record for record in records if record.split == "test")
    train_labels = {record.label for record in train}
    test_labels = {record.label for record in test}
    counts = {label: sum(record.label == label for record in records) for label in classes}
    if any(count < 2 for count in counts.values()):
        raise ValueError("CUB class cardinality differs")
    if (len(train), len(test)) != expected_counts:
        raise ValueError("CUB split count differs")
    if (len(train_labels), len(test_labels)) != expected_classes or train_labels & test_labels:
        raise ValueError("CUB split class count differs")
    paths = [record.relative_path for record in records]
    if len(set(paths)) != len(paths):
        raise ValueError("CUB image path is duplicate")
    return train + test


def ordered_record_sha256(records: tuple[CubRecord, ...]) -> str:
    """Hash the ordered protocol without host-dependent absolute paths."""

    if type(records) is not tuple or not records or any(type(x) is not CubRecord for x in records):
        raise ValueError("CUB record authority differs")
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.split.encode())
        digest.update(b"\0")
        digest.update(str(record.image_id).encode())
        digest.update(b"\0")
        digest.update(str(record.label).encode())
        digest.update(b"\0")
        digest.update(record.class_name.encode())
        digest.update(b"\0")
        digest.update(record.relative_path.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_extracted_cub_matches_archive(
    archive_path: Path,
    dataset_root: Path,
    records: tuple[CubRecord, ...],
) -> str:
    """Bind every consumed metadata/image byte to the authenticated tar archive."""

    if (
        not isinstance(archive_path, Path)
        or not isinstance(dataset_root, Path)
        or type(records) is not tuple
        or not records
        or any(type(record) is not CubRecord for record in records)
    ):
        raise ValueError("CUB content authority differs")
    relative_paths = {
        "classes.txt": dataset_root / "classes.txt",
        "image_class_labels.txt": dataset_root / "image_class_labels.txt",
        "images.txt": dataset_root / "images.txt",
        **{f"images/{record.relative_path}": record.image_path for record in records},
    }
    observed: dict[str, str] = {}
    prefix = f"{dataset_root.name}/"
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.name.startswith(prefix):
                continue
            relative = member.name[len(prefix) :]
            extracted_path = relative_paths.get(relative)
            if extracted_path is None:
                continue
            if relative in observed or not member.isfile():
                raise ValueError("CUB archive content differs")
            if not extracted_path.is_file() or extracted_path.is_symlink():
                raise ValueError("CUB extracted content differs")
            archive_stream = archive.extractfile(member)
            if archive_stream is None:
                raise ValueError("CUB archive content differs")
            archive_digest = hashlib.sha256()
            with archive_stream, extracted_path.open("rb") as extracted_stream:
                extracted_digest = hashlib.sha256()
                while True:
                    archive_chunk = archive_stream.read(1024 * 1024)
                    extracted_chunk = extracted_stream.read(1024 * 1024)
                    if archive_chunk != extracted_chunk:
                        raise ValueError("CUB extracted content differs")
                    if not archive_chunk:
                        break
                    archive_digest.update(archive_chunk)
                    extracted_digest.update(extracted_chunk)
            if archive_digest.digest() != extracted_digest.digest():
                raise ValueError("CUB extracted content differs")
            observed[relative] = archive_digest.hexdigest()
    if set(observed) != set(relative_paths):
        raise ValueError("CUB archive content differs")
    manifest = hashlib.sha256()
    for relative in sorted(observed):
        manifest.update(relative.encode())
        manifest.update(b"\0")
        manifest.update(observed[relative].encode())
        manifest.update(b"\n")
    return manifest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _ordered_array_record_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for image_id, label, class_name, relative_path in zip(
            arrays[f"{split}_image_ids"].tolist(),
            arrays[f"{split}_labels"].tolist(),
            arrays[f"{split}_class_names"].tolist(),
            arrays[f"{split}_relative_paths"].tolist(),
            strict=True,
        ):
            digest.update(str(split).encode())
            digest.update(b"\0")
            digest.update(str(image_id).encode())
            digest.update(b"\0")
            digest.update(str(label).encode())
            digest.update(b"\0")
            digest.update(str(class_name).encode())
            digest.update(b"\0")
            digest.update(str(relative_path).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _metadata_base(metadata: Mapping[str, object]) -> None:
    if type(metadata) is not dict or set(metadata) != {
        "model_identifier",
        "model_revision",
        "checkpoint_sha256",
        "transform",
        "dataset_archive_sha256",
        "dataset_archive_md5",
        "dataset_content_sha256",
        "cub_exporter_source_sha256",
        "sop_exporter_source_sha256",
    }:
        raise ValueError("CUB model metadata differs")
    if (
        type(metadata["model_identifier"]) is not str
        or not metadata["model_identifier"]
        or type(metadata["model_revision"]) is not str
        or len(metadata["model_revision"]) != 40
        or set(metadata["model_revision"]) - set("0123456789abcdef")
        or type(metadata["checkpoint_sha256"]) is not str
        or len(metadata["checkpoint_sha256"]) != 64
        or set(metadata["checkpoint_sha256"]) - set("0123456789abcdef")
        or type(metadata["transform"]) is not str
        or not metadata["transform"]
        or metadata["dataset_archive_sha256"] != CUB_ARCHIVE_SHA256
        or metadata["dataset_archive_md5"] != CUB_ARCHIVE_MD5
        or type(metadata["dataset_content_sha256"]) is not str
        or len(metadata["dataset_content_sha256"]) != 64
        or set(metadata["dataset_content_sha256"]) - set("0123456789abcdef")
        or type(metadata["cub_exporter_source_sha256"]) is not str
        or len(metadata["cub_exporter_source_sha256"]) != 64
        or set(metadata["cub_exporter_source_sha256"]) - set("0123456789abcdef")
        or type(metadata["sop_exporter_source_sha256"]) is not str
        or len(metadata["sop_exporter_source_sha256"]) != 64
        or set(metadata["sop_exporter_source_sha256"]) - set("0123456789abcdef")
    ):
        raise ValueError("CUB model metadata differs")


def export_cub_embeddings(
    records: tuple[CubRecord, ...],
    encode: Callable[[tuple[Path, ...]], np.ndarray],
    model_metadata: dict[str, object],
    output: Path,
    *,
    batch_size: int = 64,
    expected_counts: tuple[int, int] = CUB_EXPECTED_COUNTS,
    expected_classes: tuple[int, int] = CUB_EXPECTED_CLASSES,
) -> None:
    """Encode records and atomically publish one authenticated local archive."""

    if output.exists() or output.is_symlink():
        raise FileExistsError("CUB embedding output exists")
    _metadata_base(model_metadata)
    if (
        type(records) is not tuple
        or len(records) != sum(expected_counts)
        or any(type(record) is not CubRecord for record in records)
        or not callable(encode)
        or not isinstance(output, Path)
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("CUB export authority differs")
    partial = output.with_name(output.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError("CUB embedding partial output exists")
    with partial.open("xb"):
        pass
    try:
        batches: list[np.ndarray] = []
        for start in range(0, len(records), batch_size):
            batch = np.asarray(
                encode(tuple(x.image_path for x in records[start : start + batch_size]))
            )
            if (
                batch.dtype != np.float32
                or batch.ndim != 2
                or batch.shape[0] != min(batch_size, len(records) - start)
                or batch.shape[1] < 2
                or not batch.flags.c_contiguous
                or not np.isfinite(batch).all()
            ):
                raise ValueError("CUB encoded batch differs")
            norms = np.linalg.norm(batch.astype(np.float64), axis=1)
            if not np.all(np.abs(norms - 1.0) <= 2e-5):
                batch = batch / np.linalg.norm(batch, axis=1, keepdims=True)
            if not np.isfinite(batch).all():
                raise ValueError("CUB encoded batch differs")
            batches.append(batch.astype(np.float32, copy=False))
        embeddings = np.ascontiguousarray(np.concatenate(batches))
        train_count = expected_counts[0]
        split_records = (records[:train_count], records[train_count:])
        arrays: dict[str, np.ndarray] = {}
        for split, selected, values in zip(
            ("train", "test"),
            split_records,
            (embeddings[:train_count], embeddings[train_count:]),
            strict=True,
        ):
            arrays[f"{split}_embeddings"] = np.ascontiguousarray(values)
            arrays[f"{split}_labels"] = np.asarray([x.label for x in selected], dtype=np.int64)
            arrays[f"{split}_image_ids"] = np.asarray(
                [x.image_id for x in selected], dtype=np.int64
            )
            arrays[f"{split}_relative_paths"] = np.asarray([x.relative_path for x in selected])
            arrays[f"{split}_class_names"] = np.asarray([x.class_name for x in selected])
        metadata = {
            **model_metadata,
            "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
            "embedding_dimension": int(embeddings.shape[1]),
            "export_batch_size": batch_size,
            "ordered_record_sha256": ordered_record_sha256(records),
            "protocol": CUB_PROTOCOL,
            "schema": "sfora-unicom-cub-embeddings-v1",
            "split_classes": {"train": expected_classes[0], "test": expected_classes[1]},
            "split_counts": {"train": expected_counts[0], "test": expected_counts[1]},
        }
        with partial.open("wb") as stream:
            np.savez(
                stream,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
                train_embeddings=arrays["train_embeddings"],
                train_labels=arrays["train_labels"],
                train_image_ids=arrays["train_image_ids"],
                train_relative_paths=arrays["train_relative_paths"],
                train_class_names=arrays["train_class_names"],
                test_embeddings=arrays["test_embeddings"],
                test_labels=arrays["test_labels"],
                test_image_ids=arrays["test_image_ids"],
                test_relative_paths=arrays["test_relative_paths"],
                test_class_names=arrays["test_class_names"],
            )
            stream.flush()
            os.fsync(stream.fileno())
        partial.chmod(stat.S_IRUSR | stat.S_IWUSR)
        os.link(partial, output)
        partial.unlink()
    except BaseException:
        if partial.exists() and partial.is_file() and not partial.is_symlink():
            partial.unlink()
        raise


def load_cub_embedding_archive(
    path: Path,
    *,
    expected_counts: tuple[int, int] = CUB_EXPECTED_COUNTS,
    expected_classes: tuple[int, int] = CUB_EXPECTED_CLASSES,
    expected_dimension: int = 768,
) -> dict[str, object]:
    """Load and fully authenticate one CUB embedding archive."""

    array_names = {
        f"{split}_{role}"
        for split in ("train", "test")
        for role in ("embeddings", "labels", "image_ids", "relative_paths", "class_names")
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != array_names | {"metadata_json"}:
            raise ValueError("CUB embedding archive keys differ")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in array_names}
    if (
        type(metadata) is not dict
        or metadata.get("schema") != "sfora-unicom-cub-embeddings-v1"
        or metadata.get("protocol") != CUB_PROTOCOL
        or metadata.get("embedding_dimension") != expected_dimension
        or type(metadata.get("export_batch_size")) is not int
        or metadata["export_batch_size"] < 1
        or metadata.get("split_counts") != {"train": expected_counts[0], "test": expected_counts[1]}
        or metadata.get("split_classes")
        != {"train": expected_classes[0], "test": expected_classes[1]}
        or type(metadata.get("array_sha256")) is not dict
        or set(metadata["array_sha256"]) != array_names
    ):
        raise ValueError("CUB embedding metadata differs")
    _metadata_base(
        {
            key: metadata.get(key)
            for key in (
                "model_identifier",
                "model_revision",
                "checkpoint_sha256",
                "transform",
                "dataset_archive_sha256",
                "dataset_archive_md5",
                "dataset_content_sha256",
                "cub_exporter_source_sha256",
                "sop_exporter_source_sha256",
            )
        }
    )
    for name, values in arrays.items():
        if metadata["array_sha256"].get(name) != _array_sha256(values):
            raise ValueError("CUB embedding array digest differs")
    for split, count, classes in zip(
        ("train", "test"), expected_counts, expected_classes, strict=True
    ):
        embeddings = arrays[f"{split}_embeddings"]
        labels = arrays[f"{split}_labels"]
        ids = arrays[f"{split}_image_ids"]
        paths = arrays[f"{split}_relative_paths"]
        names = arrays[f"{split}_class_names"]
        if (
            embeddings.dtype != np.float32
            or embeddings.shape != (count, expected_dimension)
            or not embeddings.flags.c_contiguous
            or not np.isfinite(embeddings).all()
            or not np.all(
                np.abs(np.linalg.norm(embeddings.astype(np.float64), axis=1) - 1.0) <= 2e-5
            )
            or labels.dtype != np.int64
            or labels.shape != (count,)
            or len(set(labels.tolist())) != classes
            or ids.dtype != np.int64
            or ids.shape != (count,)
            or len(set(ids.tolist())) != count
            or paths.shape != (count,)
            or paths.dtype.kind != "U"
            or names.shape != (count,)
            or names.dtype.kind != "U"
        ):
            raise ValueError("CUB embedding array authority differs")
    train_labels = set(arrays["train_labels"].tolist())
    test_labels = set(arrays["test_labels"].tolist())
    if (
        train_labels != set(range(1, expected_classes[0] + 1))
        or test_labels
        != set(
            range(
                expected_classes[0] + 1,
                expected_classes[0] + expected_classes[1] + 1,
            )
        )
        or metadata.get("ordered_record_sha256") != _ordered_array_record_sha256(arrays)
    ):
        raise ValueError("CUB embedding class split differs")
    return {"metadata": metadata, **arrays}


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _digest(value: str) -> str:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only export surface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-checkout", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint", required=True, type=_absolute_path)
    parser.add_argument("--dataset-root", required=True, type=_absolute_path)
    parser.add_argument("--dataset-archive", required=True, type=_absolute_path)
    parser.add_argument("--dataset-archive-sha256", required=True, type=_digest)
    parser.add_argument("--model", required=True, choices=("b16", "l14-336"))
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--execute-export", action="store_true", required=True)
    return parser.parse_args(arguments)


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_clean_git_checkout(checkout: Path, revision: str) -> None:
    """Reject modified or untracked code that could shadow the frozen model package."""

    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != revision or status:
        raise ValueError("CUB UNICOM checkout authority differs")


def main(arguments: Sequence[str] | None = None) -> int:
    """Authenticate inputs, encode the frozen protocol, and publish one archive."""

    args = parse_args(arguments)
    if (
        args.dataset_archive.stat().st_size != CUB_ARCHIVE_BYTES
        or args.dataset_archive_sha256 != CUB_ARCHIVE_SHA256
        or _hash_file(args.dataset_archive, "sha256") != CUB_ARCHIVE_SHA256
        or _hash_file(args.dataset_archive, "md5") != CUB_ARCHIVE_MD5
    ):
        raise ValueError("CUB dataset archive authority differs")
    from export_unicom_sop_embeddings import (  # pylint: disable=import-outside-toplevel
        _official_encoder,
        model_authority,
    )

    authority = model_authority(args.model)
    _require_clean_git_checkout(args.unicom_checkout, authority.revision)
    encode = _official_encoder(args.unicom_checkout, args.checkpoint, authority)
    sop_exporter_file = getattr(sys.modules["export_unicom_sop_embeddings"], "__file__", None)
    if type(sop_exporter_file) is not str:
        raise ValueError("CUB SOP exporter source authority differs")
    records = parse_cub_records(args.dataset_root)
    content_sha256 = verify_extracted_cub_matches_archive(
        args.dataset_archive, args.dataset_root, records
    )
    export_cub_embeddings(
        records,
        encode,
        {
            "model_identifier": authority.identifier,
            "model_revision": authority.revision,
            "checkpoint_sha256": authority.checkpoint_sha256,
            "transform": f"official UNICOM {authority.load_name} load_model_and_transform",
            "dataset_archive_sha256": CUB_ARCHIVE_SHA256,
            "dataset_archive_md5": CUB_ARCHIVE_MD5,
            "dataset_content_sha256": content_sha256,
            "cub_exporter_source_sha256": _hash_file(Path(__file__), "sha256"),
            "sop_exporter_source_sha256": _hash_file(Path(sop_exporter_file), "sha256"),
        },
        args.output,
        batch_size=args.batch_size,
    )
    try:
        stable_content_sha256 = verify_extracted_cub_matches_archive(
            args.dataset_archive, args.dataset_root, records
        )
        if stable_content_sha256 != content_sha256:
            raise ValueError("CUB extracted content changed during export")
    except BaseException:
        if args.output.is_file() and not args.output.is_symlink():
            args.output.unlink()
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
