#!/usr/bin/env python3
"""Select ridge/CosFace score fusion on unseen identities or evaluate it officially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sfora.foundation_adapter import (
    AdapterConfig,
    NestedLinearAdapter,
    fit_ridge_stitch,
    fuse_normalized_embeddings,
    hardened_retrieval_folds,
    retrieval_map_at_r,
    retrieval_recall_at_1,
    select_fusion_weight,
)

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_cache(path: Path) -> tuple[torch.Tensor, tuple[str, ...], np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))
    return (
        torch.from_numpy(embeddings),
        tuple(str(value) for value in metadata["ids"]),
        np.asarray([int(value) for value in metadata["labels"]], dtype=np.int64),
        str(metadata["key"]["arm"]),
    )


def load_linear(path: Path, device: torch.device) -> NestedLinearAdapter:
    saved: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if saved["model"] != "linear" or saved["fixed_epoch"] != 20:
        raise ValueError("fusion requires a fixed-20-epoch linear checkpoint")
    model = NestedLinearAdapter(AdapterConfig(**saved["config"]))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def metrics(
    query: torch.Tensor,
    query_labels: np.ndarray,
    gallery: torch.Tensor,
    gallery_labels: np.ndarray,
) -> tuple[float, float]:
    return (
        retrieval_recall_at_1(query, query_labels, gallery, gallery_labels),
        retrieval_map_at_r(query, query_labels, gallery, gallery_labels),
    )


def screen(args: argparse.Namespace) -> dict[str, object]:
    source, source_ids, labels, source_arm = load_cache(args.source_train)
    target, target_ids, target_labels, target_arm = load_cache(args.target_train)
    if source_ids != target_ids or not np.array_equal(labels, target_labels):
        raise ValueError("source and target training rows differ")
    if len(args.checkpoint) != 4:
        raise ValueError("fusion screen requires exactly four paired-seed checkpoints")
    device = torch.device(args.device)
    source = source.to(device)
    target = target.to(device)
    folds = hardened_retrieval_folds(labels, seed=0)
    optimization = torch.from_numpy(folds[0].optimization).to(device)
    ridge = fit_ridge_stitch(source, target, optimization, regularization=0.1)
    with torch.no_grad():
        ridge_output = ridge.transform(source)
    rows: dict[float, list[tuple[float, float]]] = {weight: [] for weight in GRID}
    per_seed: list[dict[str, object]] = []
    for seed, checkpoint in enumerate(args.checkpoint):
        model = load_linear(checkpoint, device)
        with torch.no_grad():
            cosine_output = model(torch.nn.functional.normalize(source, dim=1))
        seed_rows: dict[str, list[dict[str, float]]] = {str(weight): [] for weight in GRID}
        for fold in folds:
            gallery = np.concatenate((fold.gallery, fold.distractor))
            for weight in GRID:
                fused = fuse_normalized_embeddings(
                    cosine_output, ridge_output, weight=weight
                )
                recall, map_at_r = metrics(
                    fused[fold.query], labels[fold.query], fused[gallery], labels[gallery]
                )
                rows[weight].append((recall, map_at_r))
                seed_rows[str(weight)].append(
                    {"recall_at_1": recall, "map_at_r": map_at_r}
                )
        per_seed.append({"seed": seed, "weights": seed_rows})
    immutable_rows = {weight: tuple(values) for weight, values in rows.items()}
    selected = select_fusion_weight(immutable_rows)
    summary = {
        str(weight): {
            "mean_recall_at_1": float(np.mean([pair[0] for pair in values])),
            "mean_map_at_r": float(np.mean([pair[1] for pair in values])),
        }
        for weight, values in rows.items()
    }
    return {
        "mode": "screen",
        "source_arm": source_arm,
        "target_arm": target_arm,
        "grid": list(GRID),
        "selection_rule": "mean_recall_at_1_then_mean_map_at_r_then_grid_order",
        "selected_weight": selected,
        "summary": summary,
        "per_seed": per_seed,
    }


def official(args: argparse.Namespace) -> dict[str, object]:
    if len(args.checkpoint) != 1 or args.weight is None:
        raise ValueError("official fusion requires one checkpoint and one selected weight")
    source, source_ids, labels, source_arm = load_cache(args.source_train)
    target, target_ids, target_labels, target_arm = load_cache(args.target_train)
    query, _, query_labels, query_arm = load_cache(args.source_query)
    gallery, _, gallery_labels, gallery_arm = load_cache(args.source_gallery)
    if source_ids != target_ids or not np.array_equal(labels, target_labels):
        raise ValueError("source and target training rows differ")
    if source_arm != query_arm or source_arm != gallery_arm:
        raise ValueError("source train/query/gallery arms differ")
    device = torch.device(args.device)
    source = source.to(device)
    target = target.to(device)
    query = query.to(device)
    gallery = gallery.to(device)
    optimization = torch.from_numpy(
        hardened_retrieval_folds(labels, seed=0)[0].optimization
    ).to(device)
    ridge = fit_ridge_stitch(source, target, optimization, regularization=0.1)
    model = load_linear(args.checkpoint[0], device)
    with torch.no_grad():
        cosine_query = model(torch.nn.functional.normalize(query, dim=1))
        cosine_gallery = model(torch.nn.functional.normalize(gallery, dim=1))
        ridge_query = ridge.transform(query)
        ridge_gallery = ridge.transform(gallery)
        fused_query = fuse_normalized_embeddings(
            cosine_query, ridge_query, weight=args.weight
        )
        fused_gallery = fuse_normalized_embeddings(
            cosine_gallery, ridge_gallery, weight=args.weight
        )
    recall, map_at_r = metrics(
        fused_query, query_labels, fused_gallery, gallery_labels
    )
    return {
        "mode": "official",
        "source_arm": source_arm,
        "target_arm": target_arm,
        "selected_weight": args.weight,
        "recall_at_1": recall,
        "map_at_r": map_at_r,
        "descriptor_dimensions": fused_query.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("screen", "official"), required=True)
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--target-train", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--source-query", type=Path)
    parser.add_argument("--source-gallery", type=Path)
    parser.add_argument("--weight", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "official" and (
        args.source_query is None or args.source_gallery is None
    ):
        raise ValueError("official fusion requires source query and gallery caches")
    payload = screen(args) if args.mode == "screen" else official(args)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload if args.mode == "official" else payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
