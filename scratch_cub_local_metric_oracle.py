#!/usr/bin/env python3
"""Throwaway oracle gate for query-local covariance metrics on frozen CUB features."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ALPHAS = (0.01, 0.1, 1.0, 10.0)
DIMENSIONS = 128
K = 1000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def split_rows(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
    gallery: list[int] = []
    query: list[int] = []
    for label in sorted(set(labels.tolist())):
        rows = torch.where(labels == label)[0].tolist()
        ranked = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(f"cub-local-oracle\0{label}\0{row}".encode()).digest(),
                row,
            ),
        )
        boundary = len(ranked) // 2
        gallery.extend(ranked[:boundary])
        query.extend(ranked[boundary:])
    gallery_rows = torch.tensor(sorted(gallery), dtype=torch.int64)
    query_rows = torch.tensor(sorted(query), dtype=torch.int64)
    encoded = (
        gallery_rows.numpy().astype("<i8", copy=False).tobytes()
        + query_rows.numpy().astype("<i8", copy=False).tobytes()
    )
    return gallery_rows, query_rows, hashlib.sha256(encoded).hexdigest()


def fit_pca(fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = F.normalize(fit.float(), dim=1).double()
    mean = values.mean(0)
    _u, _s, vh = torch.linalg.svd(values - mean, full_matrices=False)
    return mean.float().contiguous(), vh[:DIMENSIONS].float().contiguous()


def project(values: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    return F.normalize((F.normalize(values.float(), dim=1) - mean) @ components.T, dim=1)


def per_query_metrics(
    order: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    relevant = torch.bincount(gallery_labels)[query_labels]
    width = int(relevant.max())
    ranked = order[:, :width]
    hits = gallery_labels[ranked].eq(query_labels[:, None])
    positions = torch.arange(1, width + 1, device=order.device, dtype=torch.float64)
    precision = hits.cumsum(1).double() / positions
    mask = positions[None, :] <= relevant[:, None]
    ap = (precision * hits * mask).sum(1) / relevant
    return ap.cpu(), hits[:, 0].double().cpu()


def summarize(ap: torch.Tensor, r1: torch.Tensor) -> dict[str, float]:
    return {"map_at_r": float(ap.mean()), "recall_at_1": float(r1.mean())}


def class_bootstrap_lower(
    delta: torch.Tensor, labels: torch.Tensor, *, replicates: int = 10_000
) -> float:
    classes = torch.unique(labels, sorted=True)
    class_means = torch.stack([delta[labels == label].mean() for label in classes]).numpy()
    generator = np.random.Generator(np.random.PCG64(20260919))
    draws = generator.integers(0, len(class_means), size=(replicates, len(class_means)))
    estimates = class_means[draws].mean(axis=1)
    return float(np.quantile(estimates, 0.025))


def pooled_within_scatter(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    residual_parts = []
    for label in torch.unique(labels, sorted=True):
        group = values[labels == label]
        if len(group) >= 2:
            residual_parts.append(group - group.mean(0, keepdim=True))
    residuals = torch.cat(residual_parts)
    return residuals.T @ residuals / max(len(residuals) - len(residual_parts), 1)


def metric_scores(
    query: torch.Tensor,
    gallery: torch.Tensor,
    scatter: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    dimension = scatter.shape[0]
    ridge = alpha * torch.trace(scatter) / dimension
    inverse = torch.linalg.inv(scatter + ridge * torch.eye(dimension, device=scatter.device))
    transformed_query = query @ inverse
    numerator = transformed_query @ gallery.T
    query_norm = torch.sqrt((transformed_query * query).sum().clamp_min(1e-12))
    gallery_norm = torch.sqrt((gallery @ inverse * gallery).sum(1).clamp_min(1e-12))
    return numerator / (query_norm * gallery_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    archive = np.load(args.features, allow_pickle=False)
    fit = torch.from_numpy(archive["fit_embeddings"]).float()
    evaluation = torch.from_numpy(archive["evaluation_embeddings"]).float()
    labels = torch.from_numpy(archive["evaluation_labels"].astype(np.int64))
    mean, components = fit_pca(fit)
    projected = project(evaluation, mean, components)
    gallery_rows, query_rows, split_sha256 = split_rows(labels)
    gallery_float = projected[gallery_rows].cuda()
    query = projected[query_rows].cuda()
    gallery_labels = labels[gallery_rows].cuda()
    query_labels = labels[query_rows].cuda()
    gallery_codes = torch.round(gallery_float * 127.0).clamp(-127, 127).to(torch.int8)
    gallery = F.normalize(gallery_codes.float(), dim=1)

    baseline_scores = query @ gallery.T
    candidate_width = min(K, len(gallery))
    candidate_scores, candidate_indexes = torch.topk(
        baseline_scores, k=candidate_width, dim=1, largest=True, sorted=True
    )
    baseline_ap, baseline_r1 = per_query_metrics(
        candidate_indexes, query_labels, gallery_labels
    )
    arms: dict[str, object] = {"pca128_int8_asymmetric": summarize(baseline_ap, baseline_r1)}

    global_scatter = pooled_within_scatter(gallery, gallery_labels)
    for alpha in ALPHAS:
        scores = torch.stack(
            [metric_scores(row, gallery, global_scatter, alpha) for row in query]
        )
        order = torch.topk(scores, k=candidate_width, dim=1, largest=True, sorted=True).indices
        ap, r1 = per_query_metrics(order, query_labels, gallery_labels)
        arms[f"global_oracle_alpha_{alpha:g}"] = {
            **summarize(ap, r1),
            "map_delta": float((ap - baseline_ap).mean()),
            "map_delta_class_bootstrap_lower_95": class_bootstrap_lower(
                ap - baseline_ap, query_labels.cpu()
            ),
        }

    local_ap: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS}
    local_r1: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS}
    for query_index in range(len(query)):
        indexes = candidate_indexes[query_index]
        candidates = gallery[indexes]
        candidate_labels = gallery_labels[indexes]
        scatter = pooled_within_scatter(candidates, candidate_labels)
        for alpha in ALPHAS:
            scores = metric_scores(query[query_index], candidates, scatter, alpha)
            reranked = indexes[torch.argsort(scores, descending=True, stable=True)]
            ap, r1 = per_query_metrics(
                reranked[None, :], query_labels[query_index : query_index + 1], gallery_labels
            )
            local_ap[alpha].append(float(ap[0]))
            local_r1[alpha].append(float(r1[0]))
        if (query_index + 1) % 100 == 0:
            print(json.dumps({"queries_complete": query_index + 1}), flush=True)

    for alpha in ALPHAS:
        ap = torch.tensor(local_ap[alpha], dtype=torch.float64)
        r1 = torch.tensor(local_r1[alpha], dtype=torch.float64)
        arms[f"query_local_oracle_alpha_{alpha:g}"] = {
            **summarize(ap, r1),
            "map_delta": float((ap - baseline_ap).mean()),
            "map_delta_class_bootstrap_lower_95": class_bootstrap_lower(
                ap - baseline_ap, query_labels.cpu()
            ),
        }

    best_local = max(
        (value for key, value in arms.items() if key.startswith("query_local_oracle")),
        key=lambda value: value["map_at_r"],
    )
    result = {
        "schema": "scratch-cub-local-metric-oracle-v1",
        "claim_eligible": False,
        "features_sha256": file_sha256(args.features),
        "split_sha256": split_sha256,
        "fit_rows": len(fit),
        "gallery_rows": len(gallery),
        "query_rows": len(query),
        "classes": int(labels.unique().numel()),
        "dimensions": DIMENSIONS,
        "candidate_width": candidate_width,
        "alphas": ALPHAS,
        "arms": arms,
        "decision": {
            "oracle_map_gain_gate": 0.015,
            "best_local_map_gain": best_local["map_delta"],
            "best_local_lower_95": best_local["map_delta_class_bootstrap_lower_95"],
            "passed": best_local["map_delta"] >= 0.015
            and best_local["map_delta_class_bootstrap_lower_95"] > 0.0,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(wire)
    print(wire, end="")


if __name__ == "__main__":
    main()
