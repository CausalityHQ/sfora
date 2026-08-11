#!/usr/bin/env python3
"""Tune reciprocal re-ranking on train identities and evaluate frozen In-Shop embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from sfora.reciprocal_reranking import RerankResult, rerank_queries


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
    """Strictly load the three arrays needed by the re-ranking experiment."""

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
    exact = np.asarray(label, dtype=np.int64).tobytes()
    return hashlib.sha256(exact).digest(), int(label)


def _example_sort_key(example_id: np.str_) -> tuple[bytes, str]:
    exact = str(example_id).encode("utf-8")
    return hashlib.sha256(exact).digest(), str(example_id)


def select_train_split(bundle: EmbeddingBundle, max_classes: int = 1024) -> TrainSplit:
    """Select one pseudo-query per deterministic train identity."""

    if type(max_classes) is not int or max_classes <= 0:
        raise ValueError("max_classes must be a positive integer")
    unique, counts = np.unique(bundle.labels, return_counts=True)
    eligible = [label for label, count in zip(unique, counts, strict=True) if count >= 2]
    selected_labels = sorted(eligible, key=_label_sort_key)[:max_classes]
    if not selected_labels:
        raise ValueError("training bundle has no identity with at least two examples")

    query_indices: list[int] = []
    gallery_indices: list[int] = []
    for label in selected_labels:
        indices = np.flatnonzero(bundle.labels == label)
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: _example_sort_key(bundle.example_ids[index]),
        )
        query_indices.append(ordered[0])
        gallery_indices.extend(ordered[1:])
    return TrainSplit(
        query=_subset(bundle, np.asarray(query_indices, dtype=np.int64)),
        gallery=_subset(bundle, np.asarray(gallery_indices, dtype=np.int64)),
    )


def recall_at_one(
    query_labels: np.ndarray, gallery_labels: np.ndarray, top_indices: np.ndarray
) -> float:
    if top_indices.ndim != 2 or top_indices.shape != (query_labels.size, 1):
        raise ValueError("top_indices must have shape (query_count, 1)")
    return float(np.mean(gallery_labels[top_indices[:, 0]] == query_labels))


def _top_one_from_precomputed(
    result: RerankResult, *, candidate_depth: int, blend: float
) -> np.ndarray:
    top = np.empty((result.raw_indices.shape[0], 1), dtype=np.int64)
    for row in range(result.raw_indices.shape[0]):
        raw_indices = result.raw_indices[row, :candidate_depth]
        combined = (
            (1.0 - blend) * result.raw_scores[row, :candidate_depth]
            + blend * result.structural_scores[row, :candidate_depth]
        )
        order = np.lexsort((raw_indices, -combined))
        top[row, 0] = raw_indices[order[0]]
    return top


def tune_parameters(
    train: EmbeddingBundle,
    *,
    max_classes: int = 1024,
    k_values: tuple[int, ...] = (3, 5, 10, 20),
    candidate_depth_values: tuple[int, ...] = (20, 50, 100),
    blend_values: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75),
    block_size: int = 256,
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    """Select parameters using only a pseudo retrieval split of train identities."""

    split = select_train_split(train, max_classes=max_classes)
    max_depth = max(candidate_depth_values)
    if max_depth > split.gallery.embeddings.shape[0]:
        raise ValueError("candidate depth exceeds pseudo-gallery size")
    rows: list[dict[str, int | float]] = []
    best_score = -1.0
    selected: dict[str, int | float] | None = None
    for k in k_values:
        if k > min(candidate_depth_values):
            raise ValueError("every k must not exceed every candidate depth")
        prepared = rerank_queries(
            split.query.embeddings,
            split.gallery.embeddings,
            k=k,
            candidate_depth=max_depth,
            blend=0.0,
            block_size=block_size,
        )
        for candidate_depth in candidate_depth_values:
            for blend in blend_values:
                top = _top_one_from_precomputed(
                    prepared, candidate_depth=candidate_depth, blend=blend
                )
                score = recall_at_one(split.query.labels, split.gallery.labels, top)
                row: dict[str, int | float] = {
                    "k": k,
                    "candidate_depth": candidate_depth,
                    "blend": blend,
                    "recall_at_one": score,
                }
                rows.append(row)
                if score > best_score:
                    best_score = score
                    selected = {
                        "k": k,
                        "candidate_depth": candidate_depth,
                        "blend": blend,
                    }
    if selected is None:
        raise RuntimeError("parameter grid was empty")
    return selected, rows


def _evaluate(
    query: EmbeddingBundle,
    gallery: EmbeddingBundle,
    *,
    selected: dict[str, int | float],
    block_size: int,
) -> dict[str, float]:
    if query.embeddings.shape[1] != gallery.embeddings.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    k = int(selected["k"])
    candidate_depth = int(selected["candidate_depth"])
    blend = float(selected["blend"])
    result = rerank_queries(
        query.embeddings,
        gallery.embeddings,
        k=k,
        candidate_depth=candidate_depth,
        blend=blend,
        block_size=block_size,
    )
    raw = recall_at_one(query.labels, gallery.labels, result.raw_indices[:, :1])
    reranked = recall_at_one(
        query.labels, gallery.labels, result.reranked_indices[:, :1]
    )
    return {
        "raw_recall_at_one": raw,
        "reranked_recall_at_one": reranked,
        "absolute_gain": reranked - raw,
    }


def build_report(
    *,
    train_path: Path,
    evaluations: dict[str, tuple[Path, Path]],
    max_classes: int = 1024,
    k_values: tuple[int, ...] = (3, 5, 10, 20),
    candidate_depth_values: tuple[int, ...] = (20, 50, 100),
    blend_values: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75),
    block_size: int = 256,
    minimum_gain: float = 0.0015,
) -> dict[str, Any]:
    train = load_embedding_bundle(train_path)
    selected, grid = tune_parameters(
        train,
        max_classes=max_classes,
        k_values=k_values,
        candidate_depth_values=candidate_depth_values,
        blend_values=blend_values,
        block_size=block_size,
    )
    input_evaluations: dict[str, dict[str, str]] = {}
    evaluation_rows: dict[str, dict[str, float]] = {}
    for name, (query_path, gallery_path) in evaluations.items():
        query = load_embedding_bundle(query_path)
        gallery = load_embedding_bundle(gallery_path)
        input_evaluations[name] = {
            "query_path": str(query_path),
            "query_sha256": _sha256(query_path),
            "gallery_path": str(gallery_path),
            "gallery_sha256": _sha256(gallery_path),
        }
        evaluation_rows[name] = _evaluate(
            query, gallery, selected=selected, block_size=block_size
        )
    if "published" not in evaluation_rows:
        raise ValueError("evaluations must include the published checkpoint")
    return {
        "schema_version": "inshop-reciprocal-reranking-v1",
        "inputs": {
            "train_path": str(train_path),
            "train_sha256": _sha256(train_path),
            "evaluations": input_evaluations,
        },
        "tuning": {
            "selection_source": "train-identities-only",
            "max_classes": max_classes,
            "selected": selected,
            "grid": grid,
        },
        "evaluations": evaluation_rows,
        "minimum_gain": minimum_gain,
        "passes_falsifier": evaluation_rows["published"]["absolute_gain"]
        >= minimum_gain,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish canonical JSON without replacing an existing destination."""

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
    args = _parse_args()
    report = build_report(
        train_path=args.train,
        evaluations={
            "published": (args.published_query, args.published_gallery),
            "reproduced_seed0": (args.reproduced_query, args.reproduced_gallery),
        },
        block_size=args.block_size,
    )
    write_json_atomic(args.output, report)
    print(json.dumps({"output": str(args.output), "passes": report["passes_falsifier"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
