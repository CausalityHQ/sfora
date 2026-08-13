"""Pandas-free In-Shop partition parsing for official UNICOM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

EXPECTED_COUNTS = (25_882, 14_218, 12_612)


@dataclass(frozen=True)
class InshopRecord:
    split: str
    image_path: Path
    label: str


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
