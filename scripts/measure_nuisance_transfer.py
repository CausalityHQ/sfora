#!/usr/bin/env python3
"""Measure preregistered cross-identity within-class subspace transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KS = (1, 2, 4, 8, 16, 32, 64)


def _scatter(z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    within = np.zeros((z.shape[1], z.shape[1]), dtype=np.float64)
    means = []
    counts = []
    within_df = 0
    for label in np.unique(y):
        rows = z[y == label]
        mean = rows.mean(axis=0)
        centered = rows - mean
        within += centered.T @ centered
        within_df += len(rows) - 1
        means.append(mean)
        counts.append(len(rows))
    within /= within_df
    means_array = np.asarray(means)
    counts_array = np.asarray(counts, dtype=np.float64)
    grand = np.average(means_array, axis=0, weights=counts_array)
    centered_means = means_array - grand
    between = (centered_means.T * counts_array) @ centered_means
    between /= counts_array.sum()
    return within, between


def _ratios(basis: np.ndarray, within: np.ndarray, between: np.ndarray) -> dict[str, float]:
    projected_within = np.square(basis.T @ np.linalg.cholesky(within + 1e-12 * np.eye(within.shape[0]))).sum()
    projected_between = np.square(basis.T @ np.linalg.cholesky(between + 1e-12 * np.eye(between.shape[0]))).sum()
    w = projected_within / np.trace(within)
    b = projected_between / np.trace(between)
    return {"within_fraction": float(w), "between_fraction": float(b), "rho": float(w / b)}


def _measure_direction(
    z: np.ndarray, y: np.ndarray, source_labels: np.ndarray, target_labels: np.ndarray, rng_seed: int
) -> dict[str, object]:
    source = np.isin(y, source_labels)
    target = np.isin(y, target_labels)
    source_within, _ = _scatter(z[source], y[source])
    target_within, target_between = _scatter(z[target], y[target])
    eigenvalues, eigenvectors = np.linalg.eigh(source_within)
    order = np.argsort(eigenvalues)[::-1]
    observed = {}
    for k in KS:
        observed[str(k)] = _ratios(eigenvectors[:, order[:k]], target_within, target_between)

    rng = np.random.default_rng(rng_seed)
    permuted_labels = y[source].copy()
    rng.shuffle(permuted_labels)
    perm_within, _ = _scatter(z[source], permuted_labels)
    perm_values, perm_vectors = np.linalg.eigh(perm_within)
    perm_order = np.argsort(perm_values)[::-1]
    permuted = {}
    for k in KS:
        permuted[str(k)] = _ratios(perm_vectors[:, perm_order[:k]], target_within, target_between)

    random_rhos = []
    dimension = z.shape[1]
    for _ in range(100):
        q, _ = np.linalg.qr(rng.normal(size=(dimension, 32)))
        random_rhos.append(_ratios(q, target_within, target_between)["rho"])
    return {
        "observed": observed,
        "label_permuted": permuted,
        "random_k32_rho_mean": float(np.mean(random_rhos)),
        "random_k32_rho_std": float(np.std(random_rhos, ddof=1)),
    }


def measure(pack: dict[str, np.ndarray], seed: int) -> dict[str, object]:
    z = np.asarray(pack["embeddings"], dtype=np.float64)
    y = np.asarray(pack["labels"])
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-norm embedding")
    z = z / norms
    labels, counts = np.unique(y, return_counts=True)
    eligible = labels[counts >= 3]
    keep = np.isin(y, eligible)
    z, y = z[keep], y[keep]
    labels = np.sort(eligible)
    if len(labels) < 4:
        raise ValueError("too few eligible classes")
    return {
        "seed": seed,
        "examples": int(len(y)),
        "eligible_classes": int(len(labels)),
        "a_to_b": _measure_direction(z, y, labels[::2], labels[1::2], 225000 + seed * 2),
        "b_to_a": _measure_direction(z, y, labels[1::2], labels[::2], 225001 + seed * 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("packs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"k_values": list(KS), "packs": []}
    for seed, path in enumerate(args.packs):
        with np.load(path) as payload:
            result["packs"].append(measure(dict(payload), seed))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
