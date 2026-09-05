"""Authenticate Cars metadata before decoding only selected optimization images."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from PIL import Image

from sfora.data import ImageExample

RECOVERY_MANIFEST_SHA256 = "6c053b820202fb5deccfba06360e8506f201ce9eedbc6569384abd8fc30004ac"
CARS_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"


def load_optimization_images(
    *,
    limit: int | None = None,
    dataset_loader: Callable[..., Any] | None = None,
    expected_manifest: str = RECOVERY_MANIFEST_SHA256,
) -> tuple[ImageExample, ...]:
    """Read label columns first; only then fetch pixels for the sorted opt prefix.

    Source train/test are the HF storage splits, not the DML class halves.
    Global row indexes are preserved across both storage splits, exactly as in
    the historical control loader. Non-optimization rows are never fetched.
    """
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 3963):
        raise ValueError("optimization limit must be a positive concrete integer up to3963")
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset
    records: list[tuple[str, int, Any, int]] = []
    for source_split in ("train", "test"):
        dataset = dataset_loader(
            "tanganke/stanford_cars", split=source_split, revision=CARS_REVISION
        )
        # Fetching a column does not execute HF Image feature decoding.
        labels = dataset["label"]
        offset = len(records)
        for row, label in enumerate(labels):
            if type(label) is not int or not 0 <= label < 196:
                raise ValueError("Cars metadata label is not a concrete registered class")
            records.append((f"cars-train-{label}-{offset + row}", label, dataset, row))
    if len(records) != 16185 or {r[1] for r in records} != set(range(196)):
        raise ValueError("Cars full metadata cardinality or classes differ")
    ordered = sorted((r for r in records if r[1] < 98), key=lambda r: r[0])
    counts = (
        sum(r[1] < 49 for r in ordered),
        sum(49 <= r[1] < 82 for r in ordered),
        sum(82 <= r[1] < 98 for r in ordered),
        sum(r[1] >= 98 for r in records),
    )
    if counts != (3963, 2746, 1345, 8131):
        raise ValueError("Cars class-band cardinalities differ")
    manifest = {"examples": [{"example_id": r[0], "label": r[1]} for r in ordered]}
    raw = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    if hashlib.sha256(raw).hexdigest() != expected_manifest:
        raise ValueError("Cars ordered metadata digest differs before pixel access")
    selected = [r for r in ordered if r[1] < 49]
    if limit is not None:
        selected = selected[:limit]
    examples = []
    for example_id, label, dataset, row_index in selected:
        row = dataset[row_index]
        if type(row["label"]) is not int or row["label"] != label:
            raise ValueError("selected Cars row label differs from authenticated metadata")
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise ValueError("selected Cars image is not a decoded PIL image")
        examples.append(ImageExample(example_id, image.convert("RGB"), label))
    return tuple(examples)
