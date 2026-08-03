#!/usr/bin/env python3
"""Audit the exact CUB mirror used by SForA against Caltech's official archive.

This deliberately compares encoded image bytes, labels, bounding boxes, and the
first-100/last-100 zero-shot partition.  It does not infer corpus identity from row
counts alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

OFFICIAL_ARCHIVE_BYTES = 1_150_585_339
OFFICIAL_ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
HF_REPO = "bentrevett/caltech-ucsd-birds-200-2011"
HF_REVISION = "1ef09e021b0b65b40337f6f285909656f407f6e0"
EXPECTED_SOURCE_SPLITS = {"train": 5_994, "test": 5_794}
EXPECTED_DML_SPLITS = {"train": 5_864, "test": 5_924}


@dataclass(frozen=True)
class OfficialImage:
    image_id: int
    relative_path: str
    label: int
    bbox: tuple[float, float, float, float]
    encoded_sha256: str


def _digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_indexed_text(payload: bytes) -> dict[int, str]:
    rows: dict[int, str] = {}
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"invalid metadata row {line_number}: {line!r}")
        index = int(fields[0])
        if index in rows:
            raise ValueError(f"duplicate metadata id {index}")
        rows[index] = fields[1]
    return rows


def _read_official_archive(path: Path) -> dict[str, OfficialImage]:
    metadata: dict[str, bytes] = {}
    image_digests: dict[str, str] = {}
    wanted = {
        "CUB_200_2011/images.txt",
        "CUB_200_2011/image_class_labels.txt",
        "CUB_200_2011/bounding_boxes.txt",
    }
    prefix = "CUB_200_2011/images/"
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"could not read archive member {member.name}")
            payload = extracted.read()
            if member.name in wanted:
                metadata[member.name] = payload
            elif member.name.startswith(prefix):
                relative_path = member.name.removeprefix(prefix)
                if relative_path in image_digests:
                    raise ValueError(f"duplicate official image path {relative_path}")
                image_digests[relative_path] = hashlib.sha256(payload).hexdigest()

    missing_metadata = wanted - metadata.keys()
    if missing_metadata:
        raise ValueError(f"archive lacks metadata: {sorted(missing_metadata)}")
    paths = _parse_indexed_text(metadata["CUB_200_2011/images.txt"])
    labels = {
        index: int(value)
        for index, value in _parse_indexed_text(
            metadata["CUB_200_2011/image_class_labels.txt"]
        ).items()
    }
    boxes = {
        index: tuple(float(value) for value in values.split())
        for index, values in _parse_indexed_text(
            metadata["CUB_200_2011/bounding_boxes.txt"]
        ).items()
    }
    expected_ids = set(range(1, 11_789))
    if set(paths) != expected_ids or set(labels) != expected_ids or set(boxes) != expected_ids:
        raise ValueError("official metadata ids are not exactly 1..11788")
    if set(paths.values()) != set(image_digests):
        raise ValueError("official image paths and archive members differ")
    if any(len(box) != 4 for box in boxes.values()):
        raise ValueError("official bounding box is not x/y/width/height")

    result: dict[str, OfficialImage] = {}
    for image_id in sorted(paths):
        relative_path = paths[image_id]
        basename = PurePosixPath(relative_path).name
        if basename in result:
            raise ValueError(f"official image basename is not unique: {basename}")
        result[basename] = OfficialImage(
            image_id=image_id,
            relative_path=relative_path,
            label=labels[image_id] - 1,
            bbox=boxes[image_id],  # type: ignore[arg-type]
            encoded_sha256=image_digests[relative_path],
        )
    return result


def _raw_image(record: dict[str, Any], *, split: str, index: int) -> tuple[str, bytes]:
    image = record.get("image")
    if not isinstance(image, dict):
        raise ValueError(f"HF {split} row {index} has no raw image mapping")
    path = image.get("path")
    payload = image.get("bytes")
    if not isinstance(path, str) or not isinstance(payload, bytes):
        raise ValueError(f"HF {split} row {index} lacks path or embedded bytes")
    return PurePosixPath(path).name, payload


def audit(archive_path: Path) -> dict[str, Any]:
    if archive_path.stat().st_size != OFFICIAL_ARCHIVE_BYTES:
        raise ValueError(
            f"official archive size is {archive_path.stat().st_size}; "
            f"expected {OFFICIAL_ARCHIVE_BYTES}"
        )
    archive_md5 = _digest_file(archive_path, "md5")
    if archive_md5 != OFFICIAL_ARCHIVE_MD5:
        raise ValueError(f"official archive md5 mismatch: {archive_md5}")
    official = _read_official_archive(archive_path)
    if len(official) != 11_788:
        raise ValueError(f"official archive contains {len(official)} images")

    try:
        from datasets import Image, load_dataset
    except ImportError as error:
        raise RuntimeError("run with the research extra to audit Hugging Face") from error

    dataset = load_dataset(HF_REPO, revision=HF_REVISION)
    seen: dict[str, tuple[str, int, str]] = {}
    source_counts: dict[str, int] = {}
    bbox_mismatches = 0
    byte_mismatches: list[str] = []
    label_mismatches: list[str] = []
    for split in ("train", "test"):
        rows = dataset[split].cast_column("image", Image(decode=False))
        source_counts[split] = len(rows)
        for index, record in enumerate(rows):
            basename, payload = _raw_image(record, split=split, index=index)
            if basename in seen:
                raise ValueError(f"HF basename occurs twice: {basename}")
            reference = official.get(basename)
            if reference is None:
                raise ValueError(f"HF image is absent from official archive: {basename}")
            label = int(record["label"])
            digest = hashlib.sha256(payload).hexdigest()
            seen[basename] = (split, label, digest)
            if label != reference.label:
                label_mismatches.append(basename)
            if digest != reference.encoded_sha256:
                byte_mismatches.append(basename)
            bbox = tuple(float(value) for value in record["bbox"])
            x, y, width, height = reference.bbox
            # Caltech stores x/y/width/height.  The mirror intentionally exposes
            # PIL-ready x0/y0/x1/y1, which is also what sfora.data expects if bbox
            # cropping is enabled.
            expected_hf_bbox = (x, y, x + width, y + height)
            if bbox != expected_hf_bbox:
                bbox_mismatches += 1

    if source_counts != EXPECTED_SOURCE_SPLITS:
        raise ValueError(f"HF source split counts changed: {source_counts}")
    missing = set(official) - seen.keys()
    if missing:
        raise ValueError(f"HF mirror omits {len(missing)} official images")
    if label_mismatches or byte_mismatches or bbox_mismatches:
        raise ValueError(
            "HF mirror differs from official corpus: "
            f"labels={len(label_mismatches)}, encoded_bytes={len(byte_mismatches)}, "
            f"bboxes={bbox_mismatches}"
        )

    dml_counts = {
        "train": sum(label < 100 for _, label, _ in seen.values()),
        "test": sum(label >= 100 for _, label, _ in seen.values()),
    }
    if dml_counts != EXPECTED_DML_SPLITS:
        raise ValueError(f"DML class split counts changed: {dml_counts}")
    label_counts = Counter(label for _, label, _ in seen.values())
    if set(label_counts) != set(range(200)):
        raise ValueError("HF labels are not exactly 0..199")

    digest_to_rows: dict[str, list[tuple[str, str, int]]] = {}
    for basename, (_, label, digest) in seen.items():
        partition = "train" if label < 100 else "test"
        digest_to_rows.setdefault(digest, []).append((basename, partition, label))
    duplicate_groups = [rows for rows in digest_to_rows.values() if len(rows) > 1]
    cross_partition_duplicates = [
        rows
        for rows in duplicate_groups
        if len({partition for _, partition, _ in rows}) > 1
    ]
    cross_label_duplicates = [
        rows for rows in duplicate_groups if len({label for _, _, label in rows}) > 1
    ]
    if cross_partition_duplicates:
        raise ValueError(
            f"{len(cross_partition_duplicates)} encoded images cross the DML partition"
        )
    if cross_label_duplicates:
        raise ValueError(f"{len(cross_label_duplicates)} encoded images cross labels")

    return {
        "status": "pass",
        "official": {
            "archive": str(archive_path),
            "bytes": OFFICIAL_ARCHIVE_BYTES,
            "md5": archive_md5,
            "images": len(official),
        },
        "huggingface": {
            "repo": HF_REPO,
            "revision": HF_REVISION,
            "source_split_counts": source_counts,
            "exact_encoded_byte_matches": len(seen),
            "label_matches": len(seen),
            "bbox_matches": len(seen),
            "bbox_schema": "x0/y0/x1/y1 converted from official x/y/width/height",
        },
        "dml_partition": {
            "rule": "labels 0..99 train; labels 100..199 test",
            "counts": dml_counts,
            "classes": {"train": 100, "test": 100},
            "cross_partition_encoded_duplicates": len(cross_partition_duplicates),
            "same_label_duplicate_groups": duplicate_groups,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.archive)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
