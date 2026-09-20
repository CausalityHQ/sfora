#!/usr/bin/env python3
"""Causal source controls for the power-whitening gain on Cars and Food101."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from sfora.compact_metric import CompactMetricEncoder, _score_compact_metric_codes
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

CELLS = ((0.0, 0.0),) + tuple(
    (alpha, regularization)
    for alpha in (0.25, 0.5, 0.75, 1.0)
    for regularization in (0.0, 0.01, 0.1, 1.0)
)
SOURCES = ("within", "shuffled", "total")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "little")


def class_key(dataset: str, label: int) -> bytes:
    return hashlib.sha256(f"power-whitening-v1:{dataset}:{label}".encode()).digest()


def covariance(
    normalized: np.ndarray, labels: np.ndarray, source: str, namespace: str
) -> np.ndarray:
    mean = normalized.mean(axis=0)
    centered = normalized - mean
    if source == "total":
        return centered.T @ centered / len(centered)
    used_labels = labels
    if source == "shuffled":
        used_labels = np.random.default_rng(seed(namespace)).permutation(labels)
    residuals = np.empty_like(normalized)
    for label in np.unique(used_labels):
        mask = used_labels == label
        residuals[mask] = normalized[mask] - normalized[mask].mean(axis=0)
    return residuals.T @ residuals / len(residuals)


def encoder(
    normalized: np.ndarray,
    source_covariance: np.ndarray,
    alpha: float,
    regularization: float,
) -> CompactMetricEncoder:
    mean = normalized.mean(axis=0)
    centered = normalized - mean
    if alpha == 0.0:
        transform = np.eye(normalized.shape[1], dtype=np.float64)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(source_covariance)
        scale = float(np.trace(source_covariance) / source_covariance.shape[0])
        adjusted = eigenvalues + regularization * scale
        if adjusted[0] <= 0.0 or not np.isfinite(adjusted).all():
            raise ValueError("power whitening source cell differs")
        transform = (eigenvectors * np.power(adjusted, -alpha)) @ eigenvectors.T
    transformed = centered @ transform
    total = transformed.T @ transformed / len(transformed)
    values, vectors = np.linalg.eigh(total)
    order = np.argsort(values, kind="stable")[::-1]
    if values[order[127]] <= 0.0 or not np.isfinite(values).all():
        raise ValueError("power whitening source projection differs")
    selected = vectors[:, order[:128]].copy()
    for direction in selected.T:
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction *= -1.0
    weight64 = selected.T @ transform
    weight = np.ascontiguousarray(weight64.astype(np.float32))
    bias = np.ascontiguousarray((-(weight64 @ mean)).astype(np.float32))
    return CompactMetricEncoder(weight=torch.from_numpy(weight), bias=torch.from_numpy(bias))


def fit_cells(
    embeddings: torch.Tensor, labels: torch.Tensor, source: str, namespace: str
) -> dict[str, CompactMetricEncoder]:
    normalized = torch.nn.functional.normalize(embeddings.double(), dim=1).numpy()
    label_array = labels.numpy()
    source_covariance = covariance(normalized, label_array, source, namespace)
    result = {}
    for alpha, regularization in CELLS:
        key = f"{alpha:.2f}:{regularization:.2f}"
        try:
            result[key] = encoder(normalized, source_covariance, alpha, regularization)
        except ValueError:
            if alpha == 0.0 or regularization > 0.0:
                raise
    return result


def fit_ledoit_wolf(embeddings: torch.Tensor, labels: torch.Tensor) -> CompactMetricEncoder:
    normalized = torch.nn.functional.normalize(embeddings.double(), dim=1).numpy()
    label_array = labels.numpy()
    residuals = np.empty_like(normalized)
    for label in np.unique(label_array):
        mask = label_array == label
        residuals[mask] = normalized[mask] - normalized[mask].mean(axis=0)
    within = np.asarray(
        LedoitWolf(assume_centered=True, store_precision=False).fit(residuals).covariance_,
        dtype=np.float64,
    )
    return encoder(normalized, within, 0.5, 0.0)


def score(
    fitted: CompactMetricEncoder,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    values = _score_compact_metric_codes(fitted.encode(embeddings), labels, device=device)
    return float(values[0]), float(values[1])


def load_cars(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key: torch.from_numpy(np.ascontiguousarray(archive[key]))
            for key in (
                "fit_embeddings",
                "fit_labels",
                "evaluation_embeddings",
                "evaluation_labels",
            )
        }


def load_food(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["features"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["labels"], dtype=np.int64)
    values = sorted(
        (int(value) for value in np.unique(labels)),
        key=lambda value: hashlib.sha256(
            f"food-whitening-selector-v1:{value}".encode()
        ).digest(),
    )
    fit_values = set(values[:50])
    fit_mask = np.isin(labels, tuple(fit_values))
    return {
        "fit_embeddings": torch.from_numpy(features[fit_mask].copy()),
        "fit_labels": torch.from_numpy(labels[fit_mask].copy()),
        "evaluation_embeddings": torch.from_numpy(features[~fit_mask].copy()),
        "evaluation_labels": torch.from_numpy(labels[~fit_mask].copy()),
    }


def run_dataset(
    name: str, data: dict[str, torch.Tensor], device: torch.device
) -> dict[str, object]:
    fit = data["fit_embeddings"]
    labels = data["fit_labels"]
    classes = sorted(
        (int(value) for value in torch.unique(labels)), key=lambda value: class_key(name, value)
    )
    pooled = {source: {} for source in SOURCES}
    folds = []
    for fold in range(2):
        training_mask = torch.isin(labels, torch.tensor(classes[fold::2], dtype=torch.int64))
        validation_mask = torch.isin(
            labels, torch.tensor(classes[1 - fold :: 2], dtype=torch.int64)
        )
        training = fit[training_mask].contiguous()
        training_labels = labels[training_mask].contiguous()
        validation = fit[validation_mask].contiguous()
        validation_labels = labels[validation_mask].contiguous()
        fold_scores = {}
        for source in SOURCES:
            fitted = fit_cells(
                training, training_labels, source, f"power-source-control-v1:{name}:{fold}"
            )
            source_scores = {}
            for key, cell in fitted.items():
                map_at_r, recall = score(cell, validation, validation_labels, device)
                source_scores[key] = {"map_at_r": map_at_r, "recall_at_1": recall}
                aggregate = pooled[source].setdefault(key, [0.0, 0.0, 0])
                aggregate[0] += len(validation) * map_at_r
                aggregate[1] += len(validation) * recall
                aggregate[2] += len(validation)
            fold_scores[source] = source_scores
        folds.append({"fold": fold, "validation_rows": len(validation), "scores": fold_scores})
    combined = {
        source: {
            key: {"map_at_r": row[0] / row[2], "recall_at_1": row[1] / row[2]}
            for key, row in rows.items()
        }
        for source, rows in pooled.items()
    }
    selected = {
        source: max(
            rows, key=lambda key: (rows[key]["map_at_r"], rows[key]["recall_at_1"], key)
        )
        for source, rows in combined.items()
    }
    outer = {}
    for source in SOURCES:
        full = fit_cells(fit, labels, source, f"power-source-control-v1:{name}:full")
        map_at_r, recall = score(
            full[selected[source]],
            data["evaluation_embeddings"],
            data["evaluation_labels"],
            device,
        )
        outer[source] = {
            "selected_cell": selected[source],
            "map_at_r": map_at_r,
            "recall_at_1": recall,
        }
    ledoit = fit_ledoit_wolf(fit, labels)
    ledoit_score = score(
        ledoit, data["evaluation_embeddings"], data["evaluation_labels"], device
    )
    outer["ledoit_wolf_alpha_0.5"] = {
        "map_at_r": ledoit_score[0],
        "recall_at_1": ledoit_score[1],
    }
    return {"folds": folds, "pooled": combined, "selected": selected, "outer": outer}


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
        raise ValueError("power whitening source control authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    supplied = {name: (Path(path), digest) for name, path, digest in args.dataset}
    if (
        preregistration["schema"] != "sfora-power-whitening-source-control-preregistration-v1"
        or preregistration["source_commit"] != args.source_commit
        or preregistration["script_sha256"] != args.script_sha256
        or set(supplied) != set(preregistration["datasets"])
    ):
        raise ValueError("power whitening source control authority differs")
    for name, (path, digest) in supplied.items():
        if preregistration["datasets"][name] != digest or sha256(path) != digest:
            raise ValueError("power whitening source control authority differs")
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    started = time.monotonic()
    datasets = {
        name: run_dataset(
            name, load_food(path) if name == "food101" else load_cars(path), device
        )
        for name, (path, _digest) in sorted(supplied.items())
    }
    total_minus_within = {
        name: row["outer"]["total"]["map_at_r"] - row["outer"]["within"]["map_at_r"]
        for name, row in datasets.items()
    }
    values = sorted(total_minus_within.values())
    median = math.fsum(values) / len(values)
    labels_decorative = median >= -0.005 and min(values) >= -0.015
    result = {
        "schema": "sfora-power-whitening-source-control-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "datasets": datasets,
        "decision": {
            "labels_decorative": labels_decorative,
            "total_minus_within_map_at_r": total_minus_within,
            "median_total_minus_within_map_at_r": median,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
