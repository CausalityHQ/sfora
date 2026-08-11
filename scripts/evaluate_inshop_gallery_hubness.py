#!/usr/bin/env python3
"""Tune and falsify gallery-hubness correction on frozen In-Shop embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from sfora.gallery_hubness import (
    corrected_cosine_top1,
    gallery_local_density,
    top1_hubness,
)
from sfora.reciprocal_reranking import cosine_topk


class EmbeddingBundle(NamedTuple):
    embeddings: np.ndarray
    labels: np.ndarray
    example_ids: np.ndarray


class TrainSplit(NamedTuple):
    query: EmbeddingBundle
    gallery: EmbeddingBundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_embedding_bundle(path: Path) -> EmbeddingBundle:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"embedding path is not a regular file: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = ("embeddings", "labels", "example_ids")
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f"embedding archive lacks {missing[0]}")
        embeddings = np.array(payload["embeddings"], copy=True)
        labels = np.array(payload["labels"], copy=True)
        example_ids = np.array(payload["example_ids"], copy=True)
    if embeddings.ndim != 2 or embeddings.dtype != np.float32:
        raise ValueError("embeddings must be a rank-2 float32 array")
    if labels.ndim != 1 or labels.dtype != np.int64:
        raise ValueError("labels must be a rank-1 int64 array")
    if example_ids.ndim != 1 or example_ids.dtype.kind != "U":
        raise ValueError("example_ids must be a rank-1 Unicode array")
    if not (embeddings.shape[0] == labels.size == example_ids.size):
        raise ValueError("embedding, label, and example ID row counts differ")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("embedding archive must be nonempty")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-4):
        raise ValueError("embeddings must be unit-normalized")
    if np.any(example_ids == "") or np.unique(example_ids).size != example_ids.size:
        raise ValueError("example_ids must be nonempty and unique")
    return EmbeddingBundle(embeddings, labels, example_ids)


def _subset(bundle: EmbeddingBundle, indices: np.ndarray) -> EmbeddingBundle:
    return EmbeddingBundle(
        bundle.embeddings[indices], bundle.labels[indices], bundle.example_ids[indices]
    )


def _label_sort_key(label: np.int64) -> tuple[bytes, int]:
    return hashlib.sha256(np.asarray(label, dtype=np.int64).tobytes()).digest(), int(
        label
    )


def _example_sort_key(example_id: np.str_) -> tuple[bytes, str]:
    value = str(example_id)
    return hashlib.sha256(value.encode("utf-8")).digest(), value


def select_train_split(bundle: EmbeddingBundle, max_classes: int = 1024) -> TrainSplit:
    """Select one deterministic pseudo-query per eligible train identity."""

    if type(max_classes) is not int or max_classes <= 0:
        raise ValueError("max_classes must be a positive integer")
    unique, counts = np.unique(bundle.labels, return_counts=True)
    eligible = [label for label, count in zip(unique, counts, strict=True) if count >= 2]
    selected = sorted(eligible, key=_label_sort_key)[:max_classes]
    if not selected:
        raise ValueError("training bundle has no identity with at least two examples")
    query_indices: list[int] = []
    for label in selected:
        indices = np.flatnonzero(bundle.labels == label)
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: _example_sort_key(bundle.example_ids[index]),
        )
        query_indices.append(ordered[0])
    query_index_set = set(query_indices)
    gallery_indices = [
        index for index in range(bundle.labels.size) if index not in query_index_set
    ]
    return TrainSplit(
        query=_subset(bundle, np.asarray(query_indices, dtype=np.int64)),
        gallery=_subset(bundle, np.asarray(gallery_indices, dtype=np.int64)),
    )


def _recall_at_one(
    query_labels: np.ndarray, gallery_labels: np.ndarray, indices: np.ndarray
) -> float:
    if indices.shape != (query_labels.size,):
        raise ValueError("indices must have one entry per query")
    return float(np.mean(gallery_labels[indices] == query_labels))


def _raw_top1(query: np.ndarray, gallery: np.ndarray, block_size: int) -> np.ndarray:
    _, indices = cosine_topk(query, gallery, 1, block_size)
    return indices[:, 0]


def _evaluate_for_density(
    query: EmbeddingBundle,
    gallery: EmbeddingBundle,
    density: np.ndarray,
    *,
    block_size: int,
) -> float:
    indices, _ = corrected_cosine_top1(
        query.embeddings, gallery.embeddings, density, block_size
    )
    return _recall_at_one(query.labels, gallery.labels, indices)


def tune_k(
    train: EmbeddingBundle,
    *,
    max_classes: int = 1024,
    k_values: tuple[int | str, ...] = (1, 5, 10, 50, "all"),
    block_size: int = 256,
) -> tuple[int | str, list[dict[str, int | str | float]]]:
    """Select k using only a deterministic train-identity pseudo split."""

    split = select_train_split(train, max_classes=max_classes)
    raw_indices = _raw_top1(split.query.embeddings, split.gallery.embeddings, block_size)
    raw = _recall_at_one(split.query.labels, split.gallery.labels, raw_indices)
    if raw > 0.99:
        raise ValueError(
            "train pseudo split is saturated; k selection is not informative"
        )
    rows: list[dict[str, int | str | float]] = []
    selected: int | str | None = None
    best = -1.0
    for k in k_values:
        density = gallery_local_density(split.gallery.embeddings, k, block_size)
        corrected = _evaluate_for_density(
            split.query, split.gallery, density, block_size=block_size
        )
        rows.append(
            {
                "k": k,
                "raw_recall_at_one": raw,
                "corrected_recall_at_one": corrected,
                "absolute_gain": corrected - raw,
            }
        )
        if corrected > best:
            best = corrected
            selected = k
    if selected is None:
        raise ValueError("k_values must not be empty")
    return selected, rows


def evaluate_pair(
    query: EmbeddingBundle,
    gallery: EmbeddingBundle,
    *,
    k: int | str,
    block_size: int,
    permutation_seed: int = 20260811,
) -> dict[str, Any]:
    if query.embeddings.shape[1] != gallery.embeddings.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    raw_indices = _raw_top1(query.embeddings, gallery.embeddings, block_size)
    raw = _recall_at_one(query.labels, gallery.labels, raw_indices)
    density = gallery_local_density(gallery.embeddings, k, block_size)
    corrected = _evaluate_for_density(query, gallery, density, block_size=block_size)
    permutation = np.random.Generator(np.random.PCG64(permutation_seed)).permutation(
        density.size
    )
    permuted = _evaluate_for_density(
        query, gallery, density[permutation], block_size=block_size
    )
    hubness = top1_hubness(query.embeddings, gallery.embeddings, block_size)
    return {
        "raw_recall_at_one": raw,
        "corrected_recall_at_one": corrected,
        "absolute_gain": corrected - raw,
        "permuted_density_control_recall_at_one": permuted,
        "hubness": {
            "incoming_count_sum": int(np.sum(hubness.counts)),
            "maximum_count": hubness.maximum_count,
            "population_skewness": hubness.skewness,
        },
    }


def build_report(
    *,
    train_path: Path,
    evaluations: dict[str, tuple[Path, Path]],
    max_classes: int = 1024,
    k_values: tuple[int | str, ...] = (1, 5, 10, 50, "all"),
    block_size: int = 256,
    minimum_gain: float = 0.001,
) -> dict[str, Any]:
    train = load_embedding_bundle(train_path)
    selected, grid = tune_k(
        train,
        max_classes=max_classes,
        k_values=k_values,
        block_size=block_size,
    )
    input_rows: dict[str, dict[str, str]] = {}
    evaluation_rows: dict[str, dict[str, Any]] = {}
    for name, (query_path, gallery_path) in evaluations.items():
        query = load_embedding_bundle(query_path)
        gallery = load_embedding_bundle(gallery_path)
        input_rows[name] = {
            "query_path": str(query_path),
            "query_sha256": _sha256(query_path),
            "gallery_path": str(gallery_path),
            "gallery_sha256": _sha256(gallery_path),
        }
        evaluation_rows[name] = evaluate_pair(
            query, gallery, k=selected, block_size=block_size
        )
    if "published" not in evaluation_rows:
        raise ValueError("evaluations must include the published checkpoint")
    published = evaluation_rows["published"]
    hubness = published["hubness"]
    hubness_kill = bool(
        hubness["population_skewness"] < 1.0 and hubness["maximum_count"] < 20
    )
    return {
        "schema_version": "inshop-gallery-hubness-v1",
        "inputs": {
            "train_path": str(train_path),
            "train_sha256": _sha256(train_path),
            "evaluations": input_rows,
        },
        "tuning": {
            "selection_source": "train-identities-only",
            "max_classes": max_classes,
            "selected_k": selected,
            "grid": grid,
        },
        "evaluations": evaluation_rows,
        "minimum_gain": minimum_gain,
        "hubness_kill": hubness_kill,
        "passes_falsifier": bool(
            not hubness_kill and published["absolute_gain"] >= minimum_gain
        ),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if not path.parent.is_dir() or path.is_symlink():
        raise ValueError("output parent must exist and output must not be a symlink")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--published-query", type=Path, required=True)
    parser.add_argument("--published-gallery", type=Path, required=True)
    parser.add_argument("--reproduced-query", type=Path, required=True)
    parser.add_argument("--reproduced-gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    report = build_report(
        train_path=arguments.train,
        evaluations={
            "published": (arguments.published_query, arguments.published_gallery),
            "reproduced_seed0": (
                arguments.reproduced_query,
                arguments.reproduced_gallery,
            ),
        },
        block_size=arguments.block_size,
    )
    write_json_atomic(arguments.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
