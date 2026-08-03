#!/usr/bin/env python3
"""Describe corrected In-Shop final-state training geometry without test data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_ROWS = 25_882
EXPECTED_CLASSES = 3_997


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar_string(value: np.ndarray) -> str:
    if value.shape != ():
        raise ValueError("expected scalar string metadata")
    return str(value.item())


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    observed = np.quantile(values, probabilities)
    return {
        f"q{int(probability * 100):02d}": float(value)
        for probability, value in zip(probabilities, observed, strict=True)
    }


def _is_fragmented(similarity: np.ndarray) -> bool:
    count = len(similarity)
    masked = similarity.copy()
    np.fill_diagonal(masked, -np.inf)
    neighbours = np.argmax(masked, axis=1)
    adjacency = np.zeros((count, count), dtype=bool)
    adjacency[np.arange(count), neighbours] = True
    adjacency |= adjacency.T
    seen = {0}
    pending = [0]
    while pending:
        source = pending.pop()
        for target in np.flatnonzero(adjacency[source]):
            index = int(target)
            if index not in seen:
                seen.add(index)
                pending.append(index)
    return len(seen) != count


def analyze(
    embeddings: np.ndarray,
    labels: np.ndarray,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
    *,
    chunk_size: int = 512,
) -> dict[str, Any]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    proxies = np.asarray(proxies, dtype=np.float64)
    proxy_labels = np.asarray(proxy_labels, dtype=np.int64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    proxies /= np.linalg.norm(proxies, axis=1, keepdims=True)

    nearest_positive = np.full(len(labels), -np.inf)
    nearest_foreign = np.full(len(labels), -np.inf)
    loo_correct = np.zeros(len(labels), dtype=bool)
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    count_by_label = dict(zip(unique_labels.tolist(), class_counts.tolist(), strict=True))
    positive_eligible = np.asarray([count_by_label[int(label)] >= 2 for label in labels])
    for start in range(0, len(labels), chunk_size):
        stop = min(start + chunk_size, len(labels))
        similarity = embeddings[start:stop] @ embeddings.T
        similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        same = labels[start:stop, None] == labels[None, :]
        nearest_positive[start:stop] = np.where(same, similarity, -np.inf).max(axis=1)
        nearest_foreign[start:stop] = np.where(~same, similarity, -np.inf).max(axis=1)
        nearest = np.argmax(similarity, axis=1)
        loo_correct[start:stop] = labels[nearest] == labels[start:stop]

    proxy_similarity = embeddings @ proxies.T
    same_proxy = labels[:, None] == proxy_labels[None, :]
    if not np.all(same_proxy.any(axis=1)):
        raise ValueError("some training labels have no matching proxy")
    true_proxy = np.where(same_proxy, proxy_similarity, -np.inf).max(axis=1)
    foreign_proxy = np.where(~same_proxy, proxy_similarity, -np.inf).max(axis=1)

    positive_cosines: list[np.ndarray] = []
    fragmented = 0
    eligible = 0
    class_sizes: list[int] = []
    for label in unique_labels:
        members = embeddings[labels == label]
        class_sizes.append(len(members))
        similarity = members @ members.T
        upper = np.triu_indices(len(members), 1)
        if len(members) >= 2:
            positive_cosines.append(similarity[upper])
        if len(members) >= 3:
            eligible += 1
            fragmented += int(_is_fragmented(similarity))

    pair_cosine = np.concatenate(positive_cosines)
    sample_margin = nearest_positive[positive_eligible] - nearest_foreign[positive_eligible]
    proxy_margin = true_proxy - foreign_proxy
    return {
        "rows": len(labels),
        "classes": len(unique_labels),
        "positive_eligible_rows": int(positive_eligible.sum()),
        "singleton_rows": int((~positive_eligible).sum()),
        "leave_one_out_recall_at_1": float(loo_correct[positive_eligible].mean()),
        "nearest_positive_cosine": _quantiles(nearest_positive[positive_eligible]),
        "nearest_foreign_cosine": _quantiles(nearest_foreign),
        "nearest_positive_minus_foreign_margin": _quantiles(sample_margin),
        "negative_sample_margin_fraction": float(np.mean(sample_margin < 0.0)),
        "true_proxy_cosine": _quantiles(true_proxy),
        "nearest_foreign_proxy_cosine": _quantiles(foreign_proxy),
        "true_minus_foreign_proxy_margin": _quantiles(proxy_margin),
        "negative_proxy_margin_fraction": float(np.mean(proxy_margin < 0.0)),
        "all_within_class_pair_cosine": _quantiles(pair_cosine),
        "within_class_pair_count": len(pair_cosine),
        "class_size": _quantiles(np.asarray(class_sizes, dtype=np.float64)),
        "one_nn_fragmentation_eligible_classes": eligible,
        "one_nn_fragmented_classes": fragmented,
        "one_nn_fragmented_fraction": fragmented / eligible,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    with np.load(args.pack, allow_pickle=False) as payload:
        required = {
            "embeddings",
            "labels",
            "artifact_selection",
            "split",
            "checkpoint_sha256",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"training pack is missing {sorted(missing)}")
        if _scalar_string(payload["artifact_selection"]) != "final_training_state":
            raise ValueError("training pack is not from the final training state")
        if _scalar_string(payload["split"]) != "train":
            raise ValueError("training geometry requires the train split")
        if _scalar_string(payload["checkpoint_sha256"]) != _sha256(args.checkpoint):
            raise ValueError("training pack/checkpoint digest mismatch")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int64)

    observed = (len(labels), len(np.unique(labels)))
    if observed != (EXPECTED_ROWS, EXPECTED_CLASSES):
        raise ValueError(f"unexpected corrected In-Shop train shape: {observed}")
    if not np.isfinite(embeddings).all():
        raise ValueError("training embeddings contain non-finite values")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=2e-5, rtol=2e-5):
        raise ValueError("training embeddings are not unit normalized")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not the final training state")
    state = checkpoint["state_dict"]
    result = analyze(
        embeddings,
        labels,
        state["metric_proxies"].detach().cpu().numpy(),
        state["metric_proxy_labels"].detach().cpu().numpy(),
    )
    result.update(
        {
            "artifact_selection": "final_training_state",
            "checkpoint_sha256": _sha256(args.checkpoint),
            "pack_sha256": _sha256(args.pack),
            "uses_test_data": False,
        }
    )
    _write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
