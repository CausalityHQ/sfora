#!/usr/bin/env python3
"""Exploratory cross-class covariance stability diagnostic.

This is a mechanism probe, not a production API or a model selector.  It fits
within-class covariance independently on two deterministic, disjoint sets of
fit classes and compares the resulting geometry without reading evaluation
features or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf


def class_key(dataset: str, label: object) -> bytes:
    return hashlib.sha256(f"covariance-stability-v1:{dataset}:{label}".encode()).digest()


def load_fit(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if {"fit_embeddings", "fit_labels"} <= set(archive.files):
            return archive["fit_embeddings"].copy(), archive["fit_labels"].copy()
        if {"train_embeddings", "train_labels"} <= set(archive.files):
            return archive["train_embeddings"].copy(), archive["train_labels"].copy()
        raise ValueError(f"unsupported archive schema: {path}")


def normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("embedding authority differs")
    return values / norms


def within_residuals(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    residuals = np.empty_like(values)
    for label in np.unique(labels):
        mask = labels == label
        residuals[mask] = values[mask] - values[mask].mean(axis=0)
    return residuals


def covariance(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    residuals = within_residuals(values, labels)
    fit = LedoitWolf(assume_centered=True, store_precision=False).fit(residuals)
    result = np.asarray(fit.covariance_, dtype=np.float64)
    if result.shape != (values.shape[1], values.shape[1]) or not np.isfinite(result).all():
        raise ValueError("covariance authority differs")
    return result, float(fit.shrinkage_)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("zero comparison norm")
    return float(np.vdot(left, right).real / denominator)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    return cosine(left, right)


def diagnose(name: str, path: Path) -> dict[str, object]:
    started = time.monotonic()
    embeddings, labels = load_fit(path)
    if embeddings.ndim != 2 or embeddings.shape[1] != 768 or labels.shape != (len(embeddings),):
        raise ValueError("dataset authority differs")
    classes = sorted(np.unique(labels).tolist(), key=lambda label: class_key(name, label))
    if len(classes) < 4:
        raise ValueError("insufficient fit classes")
    split = (len(classes) + 1) // 2
    halves = (classes[:split], classes[split:])
    if not halves[1]:
        raise ValueError("empty class half")
    normalized = normalized_rows(embeddings)
    covariances: list[np.ndarray] = []
    shrinkages: list[float] = []
    rows: list[int] = []
    for selected in halves:
        mask = np.isin(labels, selected)
        fitted, shrinkage = covariance(normalized[mask], labels[mask])
        covariances.append(fitted)
        shrinkages.append(shrinkage)
        rows.append(int(mask.sum()))

    cov1, cov2 = covariances
    trace1, trace2 = float(np.trace(cov1)), float(np.trace(cov2))
    scaled1, scaled2 = cov1 / trace1, cov2 / trace2
    values1, vectors1 = np.linalg.eigh(cov1)
    values2, vectors2 = np.linalg.eigh(cov2)
    order1 = np.argsort(values1, kind="stable")[::-1]
    order2 = np.argsort(values2, kind="stable")[::-1]
    values1, vectors1 = values1[order1], vectors1[:, order1]
    values2, vectors2 = values2[order2], vectors2[:, order2]

    pooled = (cov1 + cov2) * 0.5
    pooled_values, pooled_vectors = np.linalg.eigh(pooled)
    pooled_order = np.argsort(pooled_values, kind="stable")[::-1]
    pooled_vectors = pooled_vectors[:, pooled_order]
    projected1 = np.einsum("ij,ij->j", pooled_vectors, cov1 @ pooled_vectors)
    projected2 = np.einsum("ij,ij->j", pooled_vectors, cov2 @ pooled_vectors)
    if np.any(projected1 <= 0.0) or np.any(projected2 <= 0.0):
        raise ValueError("nonpositive projected variance")

    subspaces: dict[str, float] = {}
    projected_log_correlations: dict[str, float] = {}
    for width in (32, 64, 128):
        overlap = vectors1[:, :width].T @ vectors2[:, :width]
        subspaces[str(width)] = float(np.square(overlap).sum() / width)
        projected_log_correlations[str(width)] = pearson(
            np.log(projected1[:width]), np.log(projected2[:width])
        )

    return {
        "classes": [len(halves[0]), len(halves[1])],
        "rows": rows,
        "shrinkage": shrinkages,
        "trace": [trace1, trace2],
        "trace_ratio": min(trace1, trace2) / max(trace1, trace2),
        "normalized_covariance_frobenius_cosine": cosine(scaled1, scaled2),
        "normalized_covariance_relative_distance": float(
            np.linalg.norm(scaled1 - scaled2)
            / math.sqrt(np.linalg.norm(scaled1) * np.linalg.norm(scaled2))
        ),
        "sorted_spectrum_cosine": cosine(values1 / values1.sum(), values2 / values2.sum()),
        "top_subspace_overlap": subspaces,
        "pooled_basis_log_variance_correlation": projected_log_correlations,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, object] = {
        "schema": "sfora-covariance-stability-diagnostic-v1",
        "claim_eligible": False,
        "class_partition": (
            "ascending SHA256(covariance-stability-v1:<dataset>:<label>), contiguous halves"
        ),
        "evaluation_data_read": False,
        "datasets": {},
    }
    datasets = result["datasets"]
    assert isinstance(datasets, dict)
    for item in args.dataset:
        name, separator, raw_path = item.partition("=")
        if not separator or not name or not raw_path or name in datasets:
            raise ValueError(f"invalid dataset: {item}")
        datasets[name] = diagnose(name, Path(raw_path))
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
