#!/usr/bin/env python3
"""Fit-only fixed label-informed PCA falsifier."""

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
from sfora.representation_ceiling import fit_centered_pca


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_key(dataset: str, label: int) -> bytes:
    return hashlib.sha256(f"label-informed-pca-v1:{dataset}:{label}".encode()).digest()


def encoder_from_scatter(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    coefficient: float,
) -> tuple[CompactMetricEncoder, dict[str, float]]:
    normalized = torch.nn.functional.normalize(embeddings.double(), dim=1).numpy()
    label_array = labels.numpy()
    mean = normalized.mean(axis=0)
    centered = normalized - mean
    total = centered.T @ centered / len(centered)
    residuals = np.empty_like(normalized)
    between = np.zeros_like(total)
    for label in np.unique(label_array):
        mask = label_array == label
        class_mean = normalized[mask].mean(axis=0)
        residuals[mask] = normalized[mask] - class_mean
        displacement = class_mean - mean
        between += mask.sum() * np.outer(displacement, displacement) / len(normalized)
    within = residuals.T @ residuals / len(residuals)
    decomposition_residual = float(
        np.linalg.norm(total - (between + within)) / np.linalg.norm(total)
    )
    if decomposition_residual > 1e-12:
        raise ValueError("scatter decomposition differs")
    scatter = total - coefficient * within
    values, vectors = np.linalg.eigh(scatter)
    order = np.argsort(values, kind="stable")[::-1]
    if values[order[127]] <= 0.0 or not np.isfinite(values).all():
        raise ValueError("label-informed scatter differs")
    selected = vectors[:, order[:128]].copy()
    for direction in selected.T:
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction *= -1.0
    weight = np.ascontiguousarray(selected.T.astype(np.float32))
    bias = np.ascontiguousarray((-(selected.T @ mean)).astype(np.float32))
    return CompactMetricEncoder(weight=torch.from_numpy(weight), bias=torch.from_numpy(bias)), {
        "decomposition_relative_residual": decomposition_residual,
        "retained_minimum_eigenvalue": float(values[order[127]]),
        "scatter_minimum_eigenvalue": float(values[order[-1]]),
    }


def pca_encoder(embeddings: torch.Tensor) -> CompactMetricEncoder:
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    fit = fit_centered_pca(normalized, dimensions=128)
    weight = fit.components.float().contiguous()
    bias = (-(weight.double() @ fit.mean.double())).float().contiguous()
    return CompactMetricEncoder(weight=weight, bias=bias)


def score(
    encoder: CompactMetricEncoder,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    result = _score_compact_metric_codes(encoder.encode(embeddings), labels, device=device)
    return {"map_at_r": result[0], "recall_at_1": result[1]}


def dataset_screen(
    name: str,
    path: Path,
    device: torch.device,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = torch.from_numpy(
            np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
        )
        labels = torch.from_numpy(np.ascontiguousarray(archive["fit_labels"], dtype=np.int64))
    classes = sorted(
        (int(value) for value in torch.unique(labels)), key=lambda x: class_key(name, x)
    )
    folds: list[dict[str, object]] = []
    for fold in range(2):
        training_classes = classes[fold::2]
        validation_classes = classes[1 - fold :: 2]
        training_mask = torch.isin(labels, torch.tensor(training_classes, dtype=torch.int64))
        validation_mask = torch.isin(labels, torch.tensor(validation_classes, dtype=torch.int64))
        training = embeddings[training_mask].contiguous()
        training_labels = labels[training_mask].contiguous()
        validation = embeddings[validation_mask].contiguous()
        validation_labels = labels[validation_mask].contiguous()

        pca = pca_encoder(training)
        candidate, evidence = encoder_from_scatter(training, training_labels, coefficient=0.5)
        generator = torch.Generator(device="cpu").manual_seed(fold)
        shuffled_labels = training_labels[torch.randperm(len(training_labels), generator=generator)]
        shuffled, shuffled_evidence = encoder_from_scatter(
            training, shuffled_labels.contiguous(), coefficient=0.5
        )
        zero, _ = encoder_from_scatter(training, training_labels, coefficient=0.0)
        projector_error = float(
            torch.linalg.matrix_norm(
                pca.weight.double().T @ pca.weight.double()
                - zero.weight.double().T @ zero.weight.double()
            )
        )
        if projector_error > 1e-5:
            raise ValueError("PCA control differs")
        folds.append(
            {
                "fold": fold,
                "training_classes": len(training_classes),
                "training_rows": len(training),
                "validation_classes": len(validation_classes),
                "validation_rows": len(validation),
                "pca": score(pca, validation, validation_labels, device),
                "candidate": score(candidate, validation, validation_labels, device),
                "shuffled_label_control": score(shuffled, validation, validation_labels, device),
                "candidate_fit": evidence,
                "shuffled_fit": shuffled_evidence,
                "pca_projector_error": projector_error,
            }
        )
    pooled: dict[str, dict[str, float]] = {}
    total_rows = sum(int(row["validation_rows"]) for row in folds)
    for arm in ("pca", "candidate", "shuffled_label_control"):
        pooled[arm] = {}
        for metric in ("map_at_r", "recall_at_1"):
            pooled[arm][metric] = (
                math.fsum(
                    int(row["validation_rows"]) * float(row[arm][metric])
                    for row in folds  # type: ignore[index]
                )
                / total_rows
            )
    return {"folds": folds, "pooled": pooled}


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
        raise ValueError("label-informed PCA authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration["schema"] != "sfora-label-informed-pca-preregistration-v1"
        or preregistration["script_sha256"] != args.script_sha256
        or preregistration["source_commit"] != args.source_commit
    ):
        raise ValueError("label-informed PCA preregistration differs")
    supplied = {name: (Path(path), digest) for name, path, digest in args.dataset}
    if set(supplied) != set(preregistration["datasets"]):
        raise ValueError("label-informed PCA dataset authority differs")
    for name, (path, digest) in supplied.items():
        if digest != preregistration["datasets"][name] or sha256(path) != digest:
            raise ValueError("label-informed PCA dataset authority differs")

    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    started = time.monotonic()
    datasets = {
        name: dataset_screen(name, path, device)
        for name, (path, _digest) in sorted(supplied.items())
    }
    macro: dict[str, dict[str, float]] = {}
    for arm in ("pca", "candidate", "shuffled_label_control"):
        macro[arm] = {
            metric: math.fsum(float(row["pooled"][arm][metric]) for row in datasets.values())
            / len(datasets)
            for metric in ("map_at_r", "recall_at_1")
        }
    candidate_map_gain = macro["candidate"]["map_at_r"] - macro["pca"]["map_at_r"]
    shuffled_map_gain = macro["candidate"]["map_at_r"] - macro["shuffled_label_control"]["map_at_r"]
    worst_map = min(
        float(row["pooled"]["candidate"]["map_at_r"]) - float(row["pooled"]["pca"]["map_at_r"])
        for row in datasets.values()
    )
    worst_recall = min(
        float(row["pooled"]["candidate"]["recall_at_1"])
        - float(row["pooled"]["pca"]["recall_at_1"])
        for row in datasets.values()
    )
    passed = (
        candidate_map_gain >= 0.002
        and shuffled_map_gain >= 0.002
        and worst_map >= -0.003
        and worst_recall >= -0.003
    )
    result = {
        "schema": "sfora-label-informed-pca-screen-v1",
        "claim_eligible": False,
        "evaluation_data_read": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "datasets": datasets,
        "macro": macro,
        "decision": {
            "passed": passed,
            "candidate_minus_pca_map_at_r": candidate_map_gain,
            "candidate_minus_shuffled_map_at_r": shuffled_map_gain,
            "worst_dataset_candidate_minus_pca_map_at_r": worst_map,
            "worst_dataset_candidate_minus_pca_recall_at_1": worst_recall,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
