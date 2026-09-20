#!/usr/bin/env python3
"""Fit-only cross-fitted power/shrinkage whitening screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sfora.compact_metric import CompactMetricEncoder, _score_compact_metric_codes
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDAS = (0.0, 0.01, 0.1, 1.0)
CELLS = tuple((alpha, 0.0) for alpha in ALPHAS[:1]) + tuple(
    (alpha, regularization) for alpha in ALPHAS[1:] for regularization in LAMBDAS
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_key(dataset: str, label: int) -> bytes:
    return hashlib.sha256(f"power-whitening-v1:{dataset}:{label}".encode()).digest()


def fit_cells(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> dict[tuple[float, float], CompactMetricEncoder]:
    normalized = torch.nn.functional.normalize(embeddings.double(), dim=1).numpy()
    label_array = labels.numpy()
    mean = normalized.mean(axis=0)
    centered = normalized - mean
    residuals = np.empty_like(normalized)
    for label in np.unique(label_array):
        mask = label_array == label
        residuals[mask] = normalized[mask] - normalized[mask].mean(axis=0)
    within = residuals.T @ residuals / len(residuals)
    eigenvalues, eigenvectors = np.linalg.eigh(within)
    if eigenvalues[0] <= 0.0 or not np.isfinite(eigenvalues).all():
        raise ValueError("power whitening covariance differs")
    scale = float(np.trace(within) / within.shape[0])
    result: dict[tuple[float, float], CompactMetricEncoder] = {}
    for alpha, regularization in CELLS:
        powers = np.power(eigenvalues + regularization * scale, -alpha)
        transform = (eigenvectors * powers) @ eigenvectors.T
        transformed = centered @ transform
        total = transformed.T @ transformed / len(transformed)
        values, vectors = np.linalg.eigh(total)
        order = np.argsort(values, kind="stable")[::-1]
        if values[order[127]] <= 0.0 or not np.isfinite(values).all():
            raise ValueError("power whitening projection differs")
        selected = vectors[:, order[:128]].copy()
        for direction in selected.T:
            pivot = int(np.argmax(np.abs(direction)))
            if direction[pivot] < 0.0:
                direction *= -1.0
        weight64 = selected.T @ transform
        weight = np.ascontiguousarray(weight64.astype(np.float32))
        bias = np.ascontiguousarray((-(weight64 @ mean)).astype(np.float32))
        result[(alpha, regularization)] = CompactMetricEncoder(
            weight=torch.from_numpy(weight), bias=torch.from_numpy(bias)
        )
    return result


def score(
    encoder: CompactMetricEncoder,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    scored = _score_compact_metric_codes(encoder.transform(embeddings), labels, device=device)
    return scored[0], scored[1]


def screen(name: str, path: Path, device: torch.device) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = torch.from_numpy(
            np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
        )
        labels = torch.from_numpy(np.ascontiguousarray(archive["fit_labels"], dtype=np.int64))
    classes = sorted(
        (int(value) for value in torch.unique(labels)), key=lambda x: class_key(name, x)
    )
    folds: list[dict[str, object]] = []
    pooled = {f"{alpha:.2f}:{reg:.2f}": [0.0, 0.0, 0] for alpha, reg in CELLS}
    for fold in range(2):
        training_classes = classes[fold::2]
        validation_classes = classes[1 - fold :: 2]
        training_mask = torch.isin(labels, torch.tensor(training_classes, dtype=torch.int64))
        validation_mask = torch.isin(labels, torch.tensor(validation_classes, dtype=torch.int64))
        training = embeddings[training_mask].contiguous()
        training_labels = labels[training_mask].contiguous()
        validation = embeddings[validation_mask].contiguous()
        validation_labels = labels[validation_mask].contiguous()
        encoders = fit_cells(training, training_labels)
        scores: dict[str, dict[str, float]] = {}
        for cell, encoder in encoders.items():
            key = f"{cell[0]:.2f}:{cell[1]:.2f}"
            map_at_r, recall = score(encoder, validation, validation_labels, device)
            scores[key] = {"map_at_r": map_at_r, "recall_at_1": recall}
            pooled[key][0] += len(validation) * map_at_r
            pooled[key][1] += len(validation) * recall
            pooled[key][2] += len(validation)
        folds.append(
            {
                "fold": fold,
                "training_classes": len(training_classes),
                "training_rows": len(training),
                "validation_classes": len(validation_classes),
                "validation_rows": len(validation),
                "scores": scores,
            }
        )
    combined = {
        key: {"map_at_r": value[0] / value[2], "recall_at_1": value[1] / value[2]}
        for key, value in pooled.items()
    }
    selected = max(
        combined, key=lambda key: (combined[key]["map_at_r"], combined[key]["recall_at_1"], key)
    )
    return {"folds": folds, "pooled": combined, "selected": selected}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset", action="append", nargs=3, metavar=("NAME", "PATH", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(Path(__file__)) != args.script_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or not args.dataset
    ):
        raise ValueError("power whitening authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration["schema"] != "sfora-power-whitening-preregistration-v1"
        or preregistration["script_sha256"] != args.script_sha256
        or preregistration["source_commit"] != args.source_commit
    ):
        raise ValueError("power whitening preregistration differs")
    supplied = {name: (Path(path), digest) for name, path, digest in args.dataset}
    if set(supplied) != set(preregistration["datasets"]):
        raise ValueError("power whitening dataset authority differs")
    for name, (path, digest) in supplied.items():
        if digest != preregistration["datasets"][name] or sha256(path) != digest:
            raise ValueError("power whitening dataset authority differs")
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    started = time.monotonic()
    datasets = {
        name: screen(name, path, device) for name, (path, _digest) in sorted(supplied.items())
    }
    pca_key = "0.00:0.00"
    gains = {
        name: float(row["pooled"][row["selected"]]["map_at_r"])
        - float(row["pooled"][pca_key]["map_at_r"])
        for name, row in datasets.items()
    }
    recall_gains = {
        name: float(row["pooled"][row["selected"]]["recall_at_1"])
        - float(row["pooled"][pca_key]["recall_at_1"])
        for name, row in datasets.items()
    }
    macro_gain = math.fsum(gains.values()) / len(gains)
    passed = (
        macro_gain >= 0.002
        and min(gains.values()) >= -0.003
        and min(recall_gains.values()) >= -0.003
    )
    result = {
        "schema": "sfora-power-whitening-screen-v1",
        "claim_eligible": False,
        "evaluation_data_read": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "cells": [f"{alpha:.2f}:{reg:.2f}" for alpha, reg in CELLS],
        "datasets": datasets,
        "decision": {
            "passed": passed,
            "macro_selected_minus_pca_map_at_r": macro_gain,
            "per_dataset_selected_minus_pca_map_at_r": gains,
            "per_dataset_selected_minus_pca_recall_at_1": recall_gains,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
