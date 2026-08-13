#!/usr/bin/env python3
"""Train a small nonlinear correction to the train-only ridge stitch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.foundation_adapter import (
    HardenedRetrievalFold,
    NonlinearRidgeStitch,
    fit_ridge_stitch,
    hardened_retrieval_folds,
    retrieval_map_at_r,
    retrieval_recall_at_1,
    trainable_parameter_count,
)


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


def fold_metrics(
    embeddings: torch.Tensor,
    labels: np.ndarray,
    folds: tuple[HardenedRetrievalFold, ...],
) -> list[dict[str, float]]:
    rows = []
    for fold in folds:
        gallery = np.concatenate((fold.gallery, fold.distractor))
        rows.append(
            {
                "recall_at_1": retrieval_recall_at_1(
                    embeddings[fold.query],
                    labels[fold.query],
                    embeddings[gallery],
                    labels[gallery],
                ),
                "map_at_r": retrieval_map_at_r(
                    embeddings[fold.query],
                    labels[fold.query],
                    embeddings[gallery],
                    labels[gallery],
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--target-train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=1_024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("nonlinear stitch output already exists")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    source, source_ids, labels, source_arm = load_cache(args.source_train)
    target, target_ids, target_labels, target_arm = load_cache(args.target_train)
    if source_ids != target_ids or not np.array_equal(labels, target_labels):
        raise ValueError("source and target training rows differ")
    device = torch.device(args.device)
    source = source.to(device)
    target = target.to(device)
    folds = hardened_retrieval_folds(labels, seed=0)
    optimization = torch.from_numpy(folds[0].optimization).to(device)
    ridge = fit_ridge_stitch(
        source, target, optimization, regularization=args.regularization
    )
    ridge_metrics = fold_metrics(ridge.transform(source), labels, folds)
    model = NonlinearRidgeStitch(ridge, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    started = time.perf_counter()
    for _epoch in range(args.epochs):
        order = optimization.detach().cpu()[torch.randperm(len(optimization), generator=generator)]
        model.train()
        for start in range(0, len(order), args.batch_size):
            indexes = order[start : start + args.batch_size].to(device)
            predicted = F.normalize(model(source[indexes]), p=2, dim=1)
            expected = F.normalize(target[indexes], p=2, dim=1)
            loss = (1.0 - (predicted * expected).sum(dim=1)).mean()
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        nonlinear_metrics = fold_metrics(model(source), labels, folds)
    means = {
        name: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("recall_at_1", "map_at_r")
        }
        for name, rows in (("ridge", ridge_metrics), ("nonlinear", nonlinear_metrics))
    }
    payload = {
        "source_arm": source_arm,
        "target_arm": target_arm,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "trainable_parameters": trainable_parameter_count(model),
        "training_seconds": elapsed,
        "ridge_folds": ridge_metrics,
        "nonlinear_folds": nonlinear_metrics,
        "means": means,
    }
    torch.save(model.state_dict(), args.checkpoint)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["means"], sort_keys=True))


if __name__ == "__main__":
    main()
