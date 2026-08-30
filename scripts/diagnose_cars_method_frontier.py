#!/usr/bin/env python3
"""Diagnose which Cars196 geometry can justify the next SFORA method."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

CORRECT = "correct"
LOCAL_DISPERSION = "local_within_class_dispersion"
CENTROID_OVERLAP = "between_class_centroid_overlap"

_PACK_KEYS = {
    "embeddings",
    "labels",
    "example_ids",
    "content_sha256",
    "artifact_selection",
    "split",
    "checkpoint_sha256",
    "report_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _scalar_string(payload: Any, name: str) -> str:
    value = np.asarray(payload[name])
    if value.ndim != 0 or value.dtype.kind not in "SU":
        raise ValueError(f"embedding pack {name} must be a scalar string")
    return str(value.item())


def _load_frontier_pack(path: Path, *, split: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        missing = _PACK_KEYS - set(payload.files)
        if missing:
            raise ValueError(f"embedding pack lacks required keys: {sorted(missing)}")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
        labels = np.asarray(payload["labels"])
        example_ids = np.asarray(payload["example_ids"])
        content_sha256 = np.asarray(payload["content_sha256"])
        observed_split = _scalar_string(payload, "split")
        artifact_selection = _scalar_string(payload, "artifact_selection")
        checkpoint_sha256 = _scalar_string(payload, "checkpoint_sha256")
        report_sha256 = _scalar_string(payload, "report_sha256")
    if observed_split != split:
        raise ValueError(f"embedding pack split differs: {observed_split} != {split}")
    if artifact_selection != "final_training_state":
        raise ValueError("embedding pack is not from the final training state")
    if embeddings.ndim != 2 or labels.ndim != 1 or example_ids.ndim != 1:
        raise ValueError("embedding pack arrays have invalid ranks")
    if content_sha256.ndim != 1 or not (
        len(embeddings) == len(labels) == len(example_ids) == len(content_sha256)
    ):
        raise ValueError("embedding pack arrays have incompatible lengths")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("embedding pack contains non-finite values")
    if len(set(example_ids.tolist())) != len(example_ids):
        raise ValueError("embedding pack contains duplicate example IDs")
    for name, digest in (
        ("checkpoint", checkpoint_sha256),
        ("report", report_sha256),
    ):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"embedding pack {name} SHA-256 is invalid")
    return {
        "embeddings": embeddings,
        "labels": labels,
        "example_ids": example_ids,
        "content_sha256": content_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "report_sha256": report_sha256,
    }


def _validated_unit_embeddings(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    target = np.asarray(labels)
    if matrix.ndim != 2 or target.ndim != 1 or len(matrix) != len(target):
        raise ValueError("embeddings and labels have incompatible shapes")
    if len(matrix) < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be a finite non-empty matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("embeddings must have nonzero finite norms")
    classes, counts = np.unique(target, return_counts=True)
    if len(classes) < 2:
        raise ValueError("retrieval requires at least two classes")
    if np.any(counts < 2):
        raise ValueError("every class requires at least two examples")
    return matrix / norms, target


def classify_retrieval_failures(
    embeddings: np.ndarray, labels: np.ndarray, *, split: str
) -> np.ndarray:
    """Classify final-state train errors using production ranking and LOO centroids."""
    if split != "train":
        raise ValueError("method selection requires the final-state train split")
    exported = np.asarray(embeddings, dtype=np.float64)
    matrix, target = _validated_unit_embeddings(embeddings, labels)
    squared_norms = np.sum(exported * exported, axis=1)
    distances = squared_norms[:, None] + squared_norms[None, :] - (2.0 * exported @ exported.T)
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)
    label_counts = dict(zip(*np.unique(target, return_counts=True), strict=True))
    max_relevant_count = max(int(label_counts[label]) - 1 for label in label_counts)
    top_k = min(len(exported) - 1, max(30, max_relevant_count))
    if top_k < len(exported):
        top_indices = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
        top_distances = np.take_along_axis(distances, top_indices, axis=1)
        top_order = np.argsort(top_distances, axis=1, kind="stable")
        nearest = np.take_along_axis(top_indices, top_order, axis=1)[:, 0]
    else:
        nearest = np.argsort(distances, axis=1, kind="stable")[:, 0]

    classes = np.unique(target)
    class_sums = np.stack([matrix[target == label].sum(axis=0) for label in classes])
    class_counts = np.asarray([np.sum(target == label) for label in classes])
    centroids = class_sums / class_counts[:, None]
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if np.any(centroid_norms <= 0.0):
        raise ValueError("a class centroid has zero norm")
    centroids /= centroid_norms
    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    centroid_similarities = matrix @ centroids.T

    signatures = np.full(len(matrix), CORRECT, dtype="<U32")
    for query in range(len(matrix)):
        if target[nearest[query]] == target[query]:
            continue
        own_index = class_to_index[target[query]]
        own_centroid = (class_sums[own_index] - matrix[query]) / (class_counts[own_index] - 1)
        own_norm = np.linalg.norm(own_centroid)
        if own_norm <= 0.0:
            raise ValueError("a leave-one-out class centroid has zero norm")
        own_similarity = float(matrix[query] @ (own_centroid / own_norm))
        foreign = centroid_similarities[query].copy()
        foreign[own_index] = -np.inf
        signatures[query] = (
            CENTROID_OVERLAP if own_similarity <= np.max(foreign) else LOCAL_DISPERSION
        )
    return signatures


def summarize_failure_signatures(signatures: Sequence[str] | np.ndarray) -> dict[str, float | int]:
    """Return exact counts and fractions for a failure-signature vector."""
    values = np.asarray(signatures)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("failure signatures must be a non-empty vector")
    allowed = {CORRECT, LOCAL_DISPERSION, CENTROID_OVERLAP}
    unexpected = set(values.tolist()) - allowed
    if unexpected:
        raise ValueError(f"unknown failure signatures: {sorted(unexpected)}")
    correct = int(np.sum(values == CORRECT))
    local = int(np.sum(values == LOCAL_DISPERSION))
    overlap = int(np.sum(values == CENTROID_OVERLAP))
    failures = local + overlap
    return {
        "queries": int(len(values)),
        "correct": correct,
        "failures": failures,
        "recall_at_1": correct / len(values),
        LOCAL_DISPERSION: local,
        CENTROID_OVERLAP: overlap,
        "local_fraction_of_failures": local / failures if failures else 0.0,
        "between_class_fraction_of_failures": overlap / failures if failures else 0.0,
    }


def analyze_train_head_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Measure the registered Cars failure condition without authorizing a method."""
    embeddings = np.asarray(pack["embeddings"])
    labels = np.asarray(pack["labels"])
    signatures = classify_retrieval_failures(embeddings, labels, split="train")
    decomposition = summarize_failure_signatures(signatures)
    local = int(decomposition[LOCAL_DISPERSION])
    overlap = int(decomposition[CENTROID_OVERLAP])
    ratio = None if local == 0 and overlap else overlap / local if local else 0.0
    interference_condition_met = overlap >= 2 * local if (local or overlap) else False
    return {
        "schema_version": "cars-method-frontier-train-v1",
        "dataset": "cars",
        "split": "train",
        "representation": "deployed_normalized_head",
        "embedding_dimensions": int(embeddings.shape[1]),
        "checkpoint_sha256": str(pack["checkpoint_sha256"]),
        "report_sha256": str(pack["report_sha256"]),
        "failure_decomposition": decomposition,
        "between_to_local_failure_ratio": ratio,
        "registered_interference_ratio_threshold": 2.0,
        "registered_interference_condition_met": interference_condition_met,
        "authorizes_method": False,
        "interpretation": "measurement_only_existing_interference_methods_occupied",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the frozen Cars train-split retrieval failure frontier"
    )
    parser.add_argument("--train-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    pack = _load_frontier_pack(arguments.train_head, split="train")
    result = analyze_train_head_pack(pack)
    result["train_head_pack_sha256"] = sha256(arguments.train_head)
    output = canonical_json_bytes(result)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(f".{arguments.output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(output)
        os.replace(temporary, arguments.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
