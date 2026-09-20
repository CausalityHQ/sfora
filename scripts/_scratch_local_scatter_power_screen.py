#!/usr/bin/env python3
"""Fit-only closed-form local-scatter falsifier for signed-int8 128-D codes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sfora.compact_metric import (
    CompactMetricEncoder,
    _compact_metric_lexicographic_topk,
    _score_compact_metric_codes,
)
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

ARMS = (
    "global_within_total",
    "local_within_total",
    "global_within_local_between",
    "local_within_local_between",
    "random_within_total",
)
LOCAL_ARMS = ARMS[1:4]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_key(dataset: str, label: int) -> bytes:
    return hashlib.sha256(f"power-whitening-v1:{dataset}:{label}".encode()).digest()


def neighbor_indices(
    normalized: np.ndarray, labels: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = torch.from_numpy(np.ascontiguousarray(normalized.astype(np.float32))).to(device)
    label_values = torch.from_numpy(labels).to(device)
    same_rows = []
    between_rows = []
    counts = np.asarray([np.count_nonzero(labels == label) - 1 for label in labels])
    if counts.min() < 1:
        raise ValueError("local scatter requires at least two rows per class")
    with torch.inference_mode():
        for start in range(0, len(values), 512):
            stop = min(start + 512, len(values))
            similarities = values[start:stop] @ values.T
            same_labels = label_values[start:stop, None] == label_values[None, :]
            same = same_labels.clone()
            local = torch.arange(stop - start, device=device)
            same[local, torch.arange(start, stop, device=device)] = False
            same_rows.append(
                _compact_metric_lexicographic_topk(similarities.masked_fill(~same, -torch.inf), 3)
                .cpu()
                .numpy()
            )
            between_rows.append(
                _compact_metric_lexicographic_topk(
                    similarities.masked_fill(same_labels, -torch.inf), 3
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(same_rows), np.concatenate(between_rows), np.minimum(counts, 3)


def random_same_indices(labels: np.ndarray, namespace: str) -> tuple[np.ndarray, np.ndarray]:
    result = np.zeros((len(labels), 3), dtype=np.int64)
    counts = np.zeros(len(labels), dtype=np.int64)
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        for row in rows:
            others = [int(value) for value in rows if value != row]
            others.sort(
                key=lambda value: hashlib.sha256(
                    f"{namespace}:{row}:{value}".encode()
                ).digest()
            )
            count = min(3, len(others))
            counts[row] = count
            result[row, :count] = others[:count]
    return result, counts


def pair_scatter(
    normalized: np.ndarray, neighbors: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    scatter = np.zeros((normalized.shape[1], normalized.shape[1]), dtype=np.float64)
    for column in range(3):
        valid = counts > column
        difference = normalized[valid] - normalized[neighbors[valid, column]]
        weights = 1.0 / counts[valid]
        scatter += difference.T @ (difference * weights[:, None])
    return scatter / len(normalized)


def encoder(
    normalized: np.ndarray,
    within: np.ndarray,
    projection_scatter: np.ndarray,
    alpha: float,
    regularization: float,
) -> CompactMetricEncoder:
    mean = normalized.mean(axis=0)
    values, vectors = np.linalg.eigh(within)
    scale = float(np.trace(within) / within.shape[0])
    adjusted = values + regularization * scale
    if adjusted[0] <= 0.0 or not np.isfinite(adjusted).all():
        raise ValueError("local scatter within covariance differs")
    transform = (vectors * np.power(adjusted, -alpha)) @ vectors.T
    transformed_scatter = transform @ projection_scatter @ transform
    projected_values, projected_vectors = np.linalg.eigh(transformed_scatter)
    order = np.argsort(projected_values, kind="stable")[::-1]
    if projected_values[order[127]] <= 0.0 or not np.isfinite(projected_values).all():
        raise ValueError("local scatter projection differs")
    selected = projected_vectors[:, order[:128]].copy()
    for direction in selected.T:
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction *= -1.0
    weight64 = selected.T @ transform
    return CompactMetricEncoder(
        weight=torch.from_numpy(np.ascontiguousarray(weight64.astype(np.float32))),
        bias=torch.from_numpy(np.ascontiguousarray((-(weight64 @ mean)).astype(np.float32))),
    )


def fit_arms(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    regularization: float,
    namespace: str,
    device: torch.device,
) -> dict[str, CompactMetricEncoder]:
    normalized = torch.nn.functional.normalize(embeddings.double(), dim=1).numpy()
    label_array = labels.numpy()
    mean = normalized.mean(axis=0)
    centered = normalized - mean
    residuals = np.empty_like(normalized)
    for label in np.unique(label_array):
        mask = label_array == label
        residuals[mask] = normalized[mask] - normalized[mask].mean(axis=0)
    global_within = residuals.T @ residuals / len(residuals)
    total = centered.T @ centered / len(centered)
    same, between, same_counts = neighbor_indices(normalized, label_array, device)
    local_within = pair_scatter(normalized, same, same_counts)
    local_between = pair_scatter(normalized, between, np.full(len(normalized), 3))
    random_rows, random_counts = random_same_indices(label_array, namespace)
    random_within = pair_scatter(normalized, random_rows, random_counts)
    definitions = {
        "global_within_total": (global_within, total),
        "local_within_total": (local_within, total),
        "global_within_local_between": (global_within, local_between),
        "local_within_local_between": (local_within, local_between),
        "random_within_total": (random_within, total),
    }
    return {
        name: encoder(normalized, within, projection, alpha, regularization)
        for name, (within, projection) in definitions.items()
    }


def run_dataset(
    name: str,
    path: Path,
    cell: tuple[float, float],
    device: torch.device,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = torch.from_numpy(
            np.ascontiguousarray(archive["embeddings"], dtype=np.float32)
        )
        labels = torch.from_numpy(np.ascontiguousarray(archive["labels"], dtype=np.int64))
    classes = sorted(
        (int(value) for value in torch.unique(labels)), key=lambda value: class_key(name, value)
    )
    pooled = {arm: [0.0, 0.0, 0] for arm in ARMS}
    folds = []
    for fold in range(2):
        training_classes = classes[fold::2]
        validation_classes = classes[1 - fold :: 2]
        training_mask = torch.isin(labels, torch.tensor(training_classes, dtype=torch.int64))
        validation_mask = torch.isin(labels, torch.tensor(validation_classes, dtype=torch.int64))
        training = embeddings[training_mask].contiguous()
        training_labels = labels[training_mask].contiguous()
        validation = embeddings[validation_mask].contiguous()
        validation_labels = labels[validation_mask].contiguous()
        fitted = fit_arms(
            training,
            training_labels,
            cell[0],
            cell[1],
            f"local-scatter-v1:{name}:{fold}",
            device,
        )
        scores = {}
        for arm, fitted_encoder in fitted.items():
            score = _score_compact_metric_codes(
                fitted_encoder.encode(validation), validation_labels, device=device
            )
            scores[arm] = {"map_at_r": float(score[0]), "recall_at_1": float(score[1])}
            pooled[arm][0] += len(validation) * float(score[0])
            pooled[arm][1] += len(validation) * float(score[1])
            pooled[arm][2] += len(validation)
        folds.append(
            {
                "fold": fold,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "scores": scores,
            }
        )
    combined = {
        arm: {"map_at_r": row[0] / row[2], "recall_at_1": row[1] / row[2]}
        for arm, row in pooled.items()
    }
    return {"cell": list(cell), "folds": folds, "pooled": combined}


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
        raise ValueError("local scatter screen authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    supplied = {name: (Path(path), digest) for name, path, digest in args.dataset}
    if (
        preregistration["schema"] != "sfora-local-scatter-power-preregistration-v1"
        or preregistration["source_commit"] != args.source_commit
        or preregistration["script_sha256"] != args.script_sha256
        or set(supplied) != set(preregistration["datasets"])
    ):
        raise ValueError("local scatter screen authority differs")
    for name, (path, digest) in supplied.items():
        if preregistration["datasets"][name] != digest or sha256(path) != digest:
            raise ValueError("local scatter screen authority differs")
    configure_deterministic_similarity_runtime(17, cpu_threads=8)
    device = torch.device("cuda")
    started = time.monotonic()
    datasets = {
        name: run_dataset(
            name,
            path,
            tuple(preregistration["selected_cells"][name]),
            device,
        )
        for name, (path, _digest) in sorted(supplied.items())
    }
    macro = {
        arm: math.fsum(row["pooled"][arm]["recall_at_1"] for row in datasets.values())
        / len(datasets)
        for arm in LOCAL_ARMS
    }
    selected = max(
        LOCAL_ARMS,
        key=lambda arm: (
            macro[arm],
            math.fsum(row["pooled"][arm]["map_at_r"] for row in datasets.values()),
            arm,
        ),
    )
    deltas = {
        name: {
            metric: row["pooled"][selected][metric]
            - row["pooled"]["global_within_total"][metric]
            for metric in ("map_at_r", "recall_at_1")
        }
        for name, row in datasets.items()
    }
    macro_recall_delta = math.fsum(
        row["recall_at_1"] for row in deltas.values()
    ) / len(deltas)
    random_gap = {
        name: row["pooled"][selected]["recall_at_1"]
        - row["pooled"]["random_within_total"]["recall_at_1"]
        for name, row in datasets.items()
    }
    causal_datasets = [
        name for name, row in deltas.items() if row["recall_at_1"] >= 0.003
    ]
    random_control_passed = bool(causal_datasets) and all(
        random_gap[name] > 0.001 for name in causal_datasets
    )
    passed = (
        macro_recall_delta >= 0.003
        and min(row["map_at_r"] for row in deltas.values()) >= -0.003
        and min(row["recall_at_1"] for row in deltas.values()) >= -0.001
        and random_control_passed
    )
    result = {
        "schema": "sfora-local-scatter-power-screen-v1",
        "claim_eligible": False,
        "evaluation_data_read": False,
        "source_commit": args.source_commit,
        "preregistration_sha256": args.preregistration_sha256,
        "datasets": datasets,
        "decision": {
            "passed": passed,
            "selected_local_arm": selected,
            "macro_selected_minus_incumbent_recall_at_1": macro_recall_delta,
            "per_dataset_selected_minus_incumbent": deltas,
            "per_dataset_selected_minus_random_recall_at_1": random_gap,
            "random_control_causal_datasets": causal_datasets,
            "random_control_passed": random_control_passed,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
