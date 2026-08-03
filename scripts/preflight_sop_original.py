#!/usr/bin/env python3
"""Materialize and validate the pinned original-resolution official SOP source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from sfora.data import load_image_retrieval_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = load_image_retrieval_examples(dataset_name="sop", split="train")
    test = load_image_retrieval_examples(dataset_name="sop", split="test")
    train_labels = {example.label for example in train}
    test_labels = {example.label for example in test}
    if train_labels & test_labels:
        raise ValueError("official SOP train/test product identities overlap")

    sampled = train[:: max(1, len(train) // 256)][:256]
    dimensions: list[tuple[int, int]] = []
    for example in sampled:
        path = Path(example.image)
        with Image.open(path) as image:
            dimensions.append(image.size)
    non_224_square = sum(size != (224, 224) for size in dimensions)
    if non_224_square == 0:
        raise ValueError("SOP source preflight found only pre-resized 224x224 images")

    payload = {
        "source_repo": "nyris/stanford-online-products-v1",
        "source_revision": "24a1b9b8ec6c0b1fc4dd324f24b2d829413a6c69",
        "train_images": len(train),
        "train_classes": len(train_labels),
        "test_images": len(test),
        "test_classes": len(test_labels),
        "product_overlap": len(train_labels & test_labels),
        "dimension_sample_count": len(dimensions),
        "dimension_sample_unique": sorted({f"{width}x{height}" for width, height in dimensions}),
        "dimension_sample_non_224_square": non_224_square,
    }
    if payload["train_images"] != 59_551 or payload["train_classes"] != 11_318:
        raise ValueError(f"unexpected official SOP training counts: {payload}")
    if payload["test_images"] != 60_502 or payload["test_classes"] != 11_316:
        raise ValueError(f"unexpected official SOP test counts: {payload}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
