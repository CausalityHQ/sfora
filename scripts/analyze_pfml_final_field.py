#!/usr/bin/env python3
"""Fail-closed static field census for a verified final Cars196 PFML artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or np.any(~np.isfinite(vectors)):
        raise ValueError("vectors must be a finite rank-2 array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("vectors contain a zero-norm row")
    return vectors / norms


def _histogram_quantile(histogram: np.ndarray, edges: np.ndarray, q: float) -> float:
    count = int(histogram.sum())
    if count == 0:
        return float("nan")
    target = q * (count - 1)
    cumulative = np.cumsum(histogram)
    index = int(np.searchsorted(cumulative, target + 1, side="left"))
    previous = int(cumulative[index - 1]) if index else 0
    in_bin = int(histogram[index])
    fraction = 0.5 if in_bin <= 0 else (target - previous) / in_bin
    return float(edges[index] + np.clip(fraction, 0.0, 1.0) * (edges[index + 1] - edges[index]))


def pair_field_statistics(
    left: np.ndarray,
    left_labels: np.ndarray,
    right: np.ndarray,
    right_labels: np.ndarray,
    *,
    delta: float,
    alpha: float,
    self_pairs: bool,
    chunk_size: int = 512,
    histogram_bins: int = 4096,
) -> dict[str, Any]:
    """Stream relation counts, distance histograms, and active radial force."""
    left = _normalize(left)
    right = _normalize(right)
    left_labels = np.asarray(left_labels, dtype=np.int64)
    right_labels = np.asarray(right_labels, dtype=np.int64)
    if len(left) != len(left_labels) or len(right) != len(right_labels):
        raise ValueError("vector and label counts differ")
    if self_pairs and (len(left) != len(right) or not np.array_equal(left, right)):
        raise ValueError("self_pairs requires identical left and right arrays")

    edges = np.linspace(0.0, 2.0 + 1.0e-8, histogram_bins + 1)
    relation = {
        "same": {
            "count": 0,
            "active": 0,
            "force_sum": 0.0,
            "hist": np.zeros(histogram_bins, dtype=np.int64),
        },
        "different": {
            "count": 0,
            "active": 0,
            "force_sum": 0.0,
            "hist": np.zeros(histogram_bins, dtype=np.int64),
        },
    }
    right_indices = np.arange(len(right))
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        similarities = np.clip(left[start:stop] @ right.T, -1.0, 1.0)
        distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarities))
        same = left_labels[start:stop, None] == right_labels[None, :]
        valid = np.ones_like(same, dtype=np.bool_)
        if self_pairs:
            valid &= right_indices[None, :] > np.arange(start, stop)[:, None]
        for name, relation_mask in (("same", same), ("different", ~same)):
            values = distances[valid & relation_mask]
            bucket = relation[name]
            bucket["count"] += int(len(values))
            active_values = values[values >= delta] if name == "same" else values[values < delta]
            bucket["active"] += int(len(active_values))
            if len(active_values):
                clamped = np.maximum(active_values, 1.0e-4)
                bucket["force_sum"] += float(np.sum(alpha * clamped ** (-alpha - 1.0)))
            bucket["hist"] += np.histogram(np.clip(values, 0.0, 2.0), bins=edges)[0]

    output: dict[str, Any] = {}
    for name, bucket in relation.items():
        count = int(bucket["count"])
        active = int(bucket["active"])
        histogram = bucket["hist"]
        output[name] = {
            "pair_count": count,
            "active_pair_count": active,
            "active_pair_fraction": active / count if count else None,
            "active_radial_force_sum": float(bucket["force_sum"]),
            "active_radial_force_mean_per_pair": (
                float(bucket["force_sum"]) / count if count else None
            ),
            "distance_quantiles_approx": {
                "q10": _histogram_quantile(histogram, edges, 0.1),
                "q50": _histogram_quantile(histogram, edges, 0.5),
                "q90": _histogram_quantile(histogram, edges, 0.9),
                "histogram_bins": histogram_bins,
            },
        }
    return output


def proxy_occupancy_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
) -> dict[str, Any]:
    embeddings = _normalize(embeddings)
    proxies = _normalize(proxies)
    labels = np.asarray(labels, dtype=np.int64)
    proxy_labels = np.asarray(proxy_labels, dtype=np.int64)
    rows = []
    for label in np.unique(labels):
        class_embeddings = embeddings[labels == label]
        class_proxies = proxies[proxy_labels == label]
        if len(class_proxies) == 0:
            raise ValueError(f"class {int(label)} has no proxy")
        assignments = np.argmax(class_embeddings @ class_proxies.T, axis=1)
        counts = np.bincount(assignments, minlength=len(class_proxies))
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        normalized_entropy = entropy / np.log(len(class_proxies)) if len(class_proxies) > 1 else 0.0
        effective_count = float(np.exp(entropy))
        proxy_similarity = class_proxies @ class_proxies.T
        upper = proxy_similarity[np.triu_indices(len(class_proxies), k=1)]
        rows.append(
            {
                "label": int(label),
                "samples": int(len(class_embeddings)),
                "proxies": int(len(class_proxies)),
                "occupied_proxies": int(np.count_nonzero(counts)),
                "normalized_assignment_entropy": normalized_entropy,
                "effective_proxy_count": effective_count,
                "maximum_occupancy_share": float(counts.max() / counts.sum()),
                "proxy_cosine_q50": float(np.quantile(upper, 0.5)) if len(upper) else None,
                "proxy_cosine_q90": float(np.quantile(upper, 0.9)) if len(upper) else None,
                "proxy_cosine_max": float(np.max(upper)) if len(upper) else None,
            }
        )

    aggregate_keys = (
        "occupied_proxies",
        "normalized_assignment_entropy",
        "effective_proxy_count",
        "maximum_occupancy_share",
        "proxy_cosine_q50",
        "proxy_cosine_q90",
        "proxy_cosine_max",
    )
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in rows if row[key] is not None])),
            "median": float(np.median([row[key] for row in rows if row[key] is not None])),
            "min": float(np.min([row[key] for row in rows if row[key] is not None])),
            "max": float(np.max([row[key] for row in rows if row[key] is not None])),
        }
        for key in aggregate_keys
    }
    return {"per_class": rows, "class_macro": aggregate}


def _require_keys(payload: Iterable[str], required: set[str], source: str) -> None:
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{source} lacks required keys: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    import torch

    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = report.get("config", {})
    required_config = {
        "dataset_name": "cars",
        "objectives": ["pfml"],
        "backbone_name": "resnet50",
        "embedding_dimensions": 512,
        "potential_delta": 0.2,
        "potential_alpha": 3.0,
        "proxy_count_per_class": 15,
        "checkpoint_selection_interval": 0,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError(f"unexpected report config {key}: {config.get(key)!r} != {expected!r}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not a final training state")
    if checkpoint.get("training_step") != 16_200:
        raise ValueError("checkpoint is not the expected 16,200-step final state")
    if checkpoint.get("evaluation_model_source") != "student":
        raise ValueError("checkpoint is not the PFML student")
    if checkpoint.get("training_config") != config:
        raise ValueError("checkpoint and report training configs differ")

    with np.load(args.train_pack, allow_pickle=False) as pack:
        _require_keys(
            pack.files,
            {
                "embeddings",
                "labels",
                "artifact_selection",
                "split",
                "checkpoint_sha256",
                "report_sha256",
                "content_sha256",
            },
            "train pack",
        )
        if (
            str(pack["artifact_selection"]) != "final_training_state"
            or str(pack["split"]) != "train"
        ):
            raise ValueError("train pack is not an explicitly final training export")
        if str(pack["checkpoint_sha256"]) != sha256(args.checkpoint):
            raise ValueError("train pack checkpoint digest mismatch")
        if str(pack["report_sha256"]) != sha256(args.report):
            raise ValueError("train pack report digest mismatch")
        embeddings = np.asarray(pack["embeddings"], dtype=np.float64)
        labels = np.asarray(pack["labels"], dtype=np.int64)
        content_hashes = np.asarray(pack["content_sha256"])
    if embeddings.shape != (8_054, 512) or len(np.unique(labels)) != 98:
        raise ValueError("train pack does not have the official Cars196 shape")
    if content_hashes.shape != (8_054,) or any(len(str(value)) != 64 for value in content_hashes):
        raise ValueError("train pack lacks one SHA-256 image-content binding per row")

    state = checkpoint.get("state_dict", {})
    _require_keys(state, {"metric_proxies", "metric_proxy_labels"}, "checkpoint state")
    raw_proxies = np.asarray(state["metric_proxies"].detach().cpu(), dtype=np.float64)
    proxy_labels = np.asarray(state["metric_proxy_labels"].detach().cpu(), dtype=np.int64)
    if raw_proxies.shape != (98 * 15, 512):
        raise ValueError(f"unexpected PFML proxy shape: {raw_proxies.shape}")
    expected_proxy_labels = np.repeat(np.arange(98, dtype=np.int64), 15)
    if not np.array_equal(proxy_labels, expected_proxy_labels):
        raise ValueError("PFML proxy labels are not the expected ordered Cars train labels")

    normalized_proxies = _normalize(raw_proxies)
    delta = float(config["potential_delta"])
    alpha = float(config["potential_alpha"])
    payload = {
        "evidence_status": "static_gram_diagnostic_not_candidate_provenance",
        "pair_count_convention": (
            "Each distinct pair is counted once. Eq. 6 contains both directions, so its "
            "ordered energy/force totals are twice the reported pairwise sums."
        ),
        "checkpoint_sha256": sha256(args.checkpoint),
        "report_sha256": sha256(args.report),
        "train_pack_sha256": sha256(args.train_pack),
        "delta": delta,
        "alpha": alpha,
        "field_support": {
            "sample_sample": pair_field_statistics(
                embeddings,
                labels,
                embeddings,
                labels,
                delta=delta,
                alpha=alpha,
                self_pairs=True,
                chunk_size=args.chunk_size,
            ),
            "sample_proxy": pair_field_statistics(
                embeddings,
                labels,
                normalized_proxies,
                proxy_labels,
                delta=delta,
                alpha=alpha,
                self_pairs=False,
                chunk_size=args.chunk_size,
            ),
            "proxy_proxy": pair_field_statistics(
                normalized_proxies,
                proxy_labels,
                normalized_proxies,
                proxy_labels,
                delta=delta,
                alpha=alpha,
                self_pairs=True,
                chunk_size=args.chunk_size,
            ),
        },
        "proxy_occupancy": proxy_occupancy_statistics(
            embeddings, labels, raw_proxies, proxy_labels
        ),
        "raw_proxy_norms": {
            "min": float(np.linalg.norm(raw_proxies, axis=1).min()),
            "median": float(np.median(np.linalg.norm(raw_proxies, axis=1))),
            "max": float(np.linalg.norm(raw_proxies, axis=1).max()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
