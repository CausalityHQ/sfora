#!/usr/bin/env python3
"""Training-only structural audit for a corrected official SOP run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_superclasses(path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()[1:], start=2):
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            raise ValueError(f"invalid SOP metadata row {line_number}: {line!r}")
        product = int(Path(fields[3]).stem.rsplit("_", 1)[0])
        superclass = int(fields[2])
        previous = mapping.setdefault(product, superclass)
        if previous != superclass:
            raise ValueError(f"product {product} has inconsistent superclasses")
    return mapping


def _unit_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero-norm embedding")
    return array / norms


def analyze(
    embeddings: np.ndarray,
    labels: np.ndarray,
    superclasses: np.ndarray,
    *,
    chunk_size: int = 512,
) -> dict[str, Any]:
    vectors = _unit_rows(np.asarray(embeddings, dtype=np.float32))
    labels = np.asarray(labels, dtype=np.int64)
    superclasses = np.asarray(superclasses, dtype=np.int64)
    if len(vectors) != len(labels) or len(labels) != len(superclasses):
        raise ValueError("embedding, label, and superclass lengths differ")

    members: dict[int, np.ndarray] = {
        int(label): np.flatnonzero(labels == label) for label in np.unique(labels)
    }
    centroids = {
        label: _unit_rows(vectors[index].mean(axis=0, keepdims=True))[0]
        for label, index in members.items()
    }
    within_cosines: list[float] = []
    centroid_cosines: list[float] = []
    class_rows: list[dict[str, float | int]] = []
    for label, index in members.items():
        block = vectors[index] @ vectors[index].T
        upper = block[np.triu_indices(len(index), 1)]
        within_cosines.extend(upper.tolist())
        alignment = vectors[index] @ centroids[label]
        centroid_cosines.extend(alignment.tolist())
        class_rows.append(
            {
                "label": label,
                "size": len(index),
                "mean_pair_cosine": float(upper.mean()),
                "mean_centroid_cosine": float(alignment.mean()),
            }
        )

    nearest_correct = 0
    nearest_negative_same_superclass = 0
    nearest_negative_count = 0
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors_t = torch.as_tensor(vectors, device=device)
    labels_t = torch.as_tensor(labels, device=device)
    superclasses_t = torch.as_tensor(superclasses, device=device)
    for start in range(0, len(vectors), chunk_size):
        stop = min(start + chunk_size, len(vectors))
        similarity = vectors_t[start:stop] @ vectors_t.T
        similarity[
            torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)
        ] = -torch.inf
        nearest = similarity.argmax(dim=1)
        nearest_correct += int((labels_t[nearest] == labels_t[start:stop]).sum().item())
        different = labels_t[start:stop, None] != labels_t[None, :]
        similarity.masked_fill_(~different, -torch.inf)
        nearest_negative = similarity.argmax(dim=1)
        nearest_negative_same_superclass += int(
            (superclasses_t[nearest_negative] == superclasses_t[start:stop]).sum().item()
        )
        nearest_negative_count += stop - start

    by_size: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for row in class_rows:
        by_size[int(row["size"])].append(row)
    size_summary = {
        str(size): {
            "classes": len(rows),
            "mean_pair_cosine": float(np.mean([float(row["mean_pair_cosine"]) for row in rows])),
            "mean_centroid_cosine": float(
                np.mean([float(row["mean_centroid_cosine"]) for row in rows])
            ),
        }
        for size, rows in sorted(by_size.items())
    }
    return {
        "examples": len(vectors),
        "classes": len(members),
        "superclasses": int(len(np.unique(superclasses))),
        "training_leave_one_out_r1": nearest_correct / len(vectors),
        "nearest_negative_same_superclass_fraction": (
            nearest_negative_same_superclass / nearest_negative_count
        ),
        "mean_within_class_pair_cosine": float(np.mean(within_cosines)),
        "mean_sample_to_class_centroid_cosine": float(np.mean(centroid_cosines)),
        "by_class_size": size_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    archive = np.load(args.embeddings, allow_pickle=False)
    labels = archive["labels"].astype(np.int64)
    superclass_by_product = load_superclasses(args.metadata)
    missing = sorted(set(labels.tolist()) - superclass_by_product.keys())
    if missing:
        raise ValueError(f"metadata lacks {len(missing)} embedding labels; first={missing[0]}")
    superclasses = np.asarray([superclass_by_product[int(label)] for label in labels])
    result = analyze(archive["embeddings"], labels, superclasses, chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
