#!/usr/bin/env python3
"""Prospective Flowers102 class-disjoint compact-metric selector probe."""

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
    _compact_metric_fold,
    _score_compact_metric_codes,
    fit_compact_metric_projection,
    fit_within_class_whitening_projection,
)
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.representation_ceiling import fit_centered_pca


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_order(label: int) -> bytes:
    return hashlib.sha256(f"flowers-whitening-selector-v1:{label}".encode()).digest()


def pca_encoder(embeddings: torch.Tensor) -> CompactMetricEncoder:
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    pca = fit_centered_pca(normalized, dimensions=128)
    weight = pca.components.float().contiguous()
    bias = (-(weight.double() @ pca.mean.double())).float().contiguous()
    return CompactMetricEncoder(weight=weight, bias=bias)


def fit_arms(
    embeddings: torch.Tensor, labels: torch.Tensor, *, device: torch.device
) -> dict[str, CompactMetricEncoder]:
    whitening = fit_within_class_whitening_projection(
        embeddings, labels, output_dimensions=128
    ).encoder
    return {
        "pca": pca_encoder(embeddings),
        "learned": fit_compact_metric_projection(embeddings, labels, device=device).encoder,
        "within_class_whitening": whitening,
        "whitening_initialized": fit_compact_metric_projection(
            embeddings,
            labels,
            initial_encoder=whitening,
            device=device,
        ).encoder,
    }


def pooled(rows: list[dict[str, float | int]], name: str, metric: str) -> float:
    total = sum(int(row["rows"]) for row in rows)
    return math.fsum(int(row["rows"]) * float(row[f"{name}_{metric}"]) for row in rows) / total


def select(cv: dict[str, dict[str, float]]) -> tuple[str, str]:
    incumbent = (
        "learned"
        if (
            cv["learned"]["map_at_r"] >= cv["pca"]["map_at_r"] + 0.003
            and cv["learned"]["recall_at_1"] >= cv["pca"]["recall_at_1"]
        )
        else "pca"
    )
    eligible = [
        name
        for name in ("within_class_whitening", "whitening_initialized")
        if cv[name]["map_at_r"] >= cv[incumbent]["map_at_r"] + 0.003
        and cv[name]["recall_at_1"] >= cv[incumbent]["recall_at_1"] - 0.003
    ]
    if not eligible:
        return incumbent, incumbent
    selected = max(
        eligible,
        key=lambda name: (cv[name]["map_at_r"], cv[name]["recall_at_1"], name),
    )
    return incumbent, selected


def class_bootstrap(
    selected: np.ndarray, incumbent: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    values = np.unique(labels)
    differences = np.asarray(
        [selected[labels == value].mean() - incumbent[labels == value].mean() for value in values],
        dtype=np.float64,
    )
    generator = np.random.default_rng(20260920)
    samples = differences[
        generator.integers(0, len(differences), size=(10_000, len(differences)))
    ].mean(axis=1)
    lower, median, upper = np.quantile(samples, [0.025, 0.5, 0.975])
    return {"lower": float(lower), "median": float(median), "upper": float(upper)}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.features) != args.features_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or sha256(Path(__file__)) != args.script_sha256
    ):
        raise ValueError("Flowers whitening authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration["schema"] != "sfora-flowers-whitening-selector-preregistration-v1"
        or preregistration["source_commit"] != args.source_commit
        or preregistration["features_sha256"] != args.features_sha256
        or preregistration["script_sha256"] != args.script_sha256
    ):
        raise ValueError("Flowers whitening preregistration differs")

    runtime = configure_deterministic_similarity_runtime(17, cpu_threads=2)
    with np.load(args.features, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["features"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["labels"], dtype=np.int64)
        metadata = json.loads(str(archive["metadata_json"]))
    values = sorted((int(value) for value in np.unique(labels)), key=class_order)
    fit_values = values[:51]
    evaluation_values = values[51:]
    fit_mask = np.isin(labels, fit_values)
    evaluation_mask = np.isin(labels, evaluation_values)
    if (
        features.shape != (6_149, 768)
        or len(values) != 102
        or len(fit_values) != 51
        or len(evaluation_values) != 51
        or np.any(fit_mask & evaluation_mask)
    ):
        raise ValueError("Flowers whitening split differs")
    fit = torch.from_numpy(features[fit_mask].copy())
    fit_labels = torch.from_numpy(labels[fit_mask].copy())
    evaluation = torch.from_numpy(features[evaluation_mask].copy())
    evaluation_labels = torch.from_numpy(labels[evaluation_mask].copy())
    device = torch.device("cuda")
    started = time.monotonic()

    fold_rows: list[dict[str, float | int]] = []
    for fold in range(3):
        validation_mask = np.asarray(
            [_compact_metric_fold(int(value)) == fold for value in fit_labels.numpy()]
        )
        training_mask = ~validation_mask
        training = fit[torch.from_numpy(training_mask)].contiguous()
        training_labels = fit_labels[torch.from_numpy(training_mask)].contiguous()
        validation = fit[torch.from_numpy(validation_mask)].contiguous()
        validation_labels = fit_labels[torch.from_numpy(validation_mask)].contiguous()
        arms = fit_arms(training, training_labels, device=device)
        row: dict[str, float | int] = {"fold": fold, "rows": len(validation)}
        for name, encoder in arms.items():
            score = _score_compact_metric_codes(
                encoder.encode(validation), validation_labels, device=device
            )
            row[f"{name}_map_at_r"] = score[0]
            row[f"{name}_recall_at_1"] = score[1]
        fold_rows.append(row)

    names = ("pca", "learned", "within_class_whitening", "whitening_initialized")
    cv = {
        name: {metric: pooled(fold_rows, name, metric) for metric in ("map_at_r", "recall_at_1")}
        for name in names
    }
    incumbent, selected = select(cv)
    full_arms = fit_arms(fit, fit_labels, device=device)
    scores: dict[str, dict[str, float]] = {}
    per_query: dict[str, np.ndarray] = {}
    for name, encoder in full_arms.items():
        score = _score_compact_metric_codes(
            encoder.encode(evaluation), evaluation_labels, device=device
        )
        scores[name] = {"map_at_r": score[0], "recall_at_1": score[1]}
        per_query[name] = np.asarray(score[2], dtype=np.float64)
    teacher = _score_compact_metric_codes(evaluation, evaluation_labels, device=device)
    scores["teacher_float_768"] = {"map_at_r": teacher[0], "recall_at_1": teacher[1]}

    result = {
        "schema": "sfora-flowers-whitening-selector-result-v1",
        "claim_eligible": False,
        "authorities": {
            "features_sha256": args.features_sha256,
            "preregistration_sha256": args.preregistration_sha256,
            "script_sha256": args.script_sha256,
            "source_commit": args.source_commit,
        },
        "dataset": {
            "metadata": metadata,
            "fit_classes": len(fit_values),
            "fit_rows": int(fit_mask.sum()),
            "evaluation_classes": len(evaluation_values),
            "evaluation_rows": int(evaluation_mask.sum()),
        },
        "fit_only": {
            "folds": fold_rows,
            "scores": cv,
            "incumbent": incumbent,
            "selected": selected,
        },
        "external": {
            "scores": scores,
            "selected_minus_incumbent": {
                "map_at_r": scores[selected]["map_at_r"] - scores[incumbent]["map_at_r"],
                "recall_at_1": scores[selected]["recall_at_1"] - scores[incumbent]["recall_at_1"],
                "class_bootstrap_map_delta_95": class_bootstrap(
                    per_query[selected], per_query[incumbent], evaluation_labels.numpy()
                ),
            },
        },
        "encoders": {name: encoder.sha256 for name, encoder in full_arms.items()},
        "runtime": runtime,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
