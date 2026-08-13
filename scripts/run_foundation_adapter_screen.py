#!/usr/bin/env python3
"""Train cheap adapters on a cached foundation embedding export."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sfora.data import ImageExample
from sfora.foundation_adapter import (
    AdapterConfig,
    NestedLinearAdapter,
    NestedResidualAdapter,
    cosine_margin_loss,
    hardened_retrieval_folds,
    identity_balanced_batches,
    initialize_residual_from_linear,
    nested_embeddings,
    retrieval_map_at_r,
    retrieval_recall_at_1,
    trainable_parameter_count,
)
from sfora.foundation_pareto import build_identity_disjoint_validation_split


def _load_cache(path: Path) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("cache must be a regular non-symlink file")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"embeddings", "metadata_json"}:
            raise ValueError("cache archive members differ")
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        metadata = json.loads(bytes(archive["metadata_json"]).decode("utf-8"))
    ids = tuple(metadata["ids"])
    labels = np.asarray([int(value) for value in metadata["labels"]], dtype=np.int64)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids) or len(ids) != labels.shape[0]:
        raise ValueError("cache embeddings, IDs, and labels differ")
    if len(ids) != len(set(ids)) or not np.isfinite(embeddings).all():
        raise ValueError("cache IDs must be unique and embeddings finite")
    return embeddings, ids, labels, metadata


def _split_indexes(
    ids: tuple[str, ...], labels: np.ndarray, *, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    examples = tuple(
        ImageExample(example_id=example_id, image=None, label=int(label))
        for example_id, label in zip(ids, labels, strict=True)
    )
    split = build_identity_disjoint_validation_split(examples, fraction=fraction, seed=seed)
    index = {example_id: position for position, example_id in enumerate(ids)}
    return tuple(
        np.asarray([index[row.example_id] for row in role], dtype=np.int64)
        for role in (split.optimization, split.query, split.gallery)
    )  # type: ignore[return-value]


def _evaluate(
    model: torch.nn.Module,
    query: torch.Tensor,
    query_labels: np.ndarray,
    gallery: torch.Tensor,
    gallery_labels: np.ndarray,
    prefixes: tuple[int, ...],
) -> dict[int, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        query_outputs = nested_embeddings(model(query), prefixes)
        gallery_outputs = nested_embeddings(model(gallery), prefixes)
    return {
        width: {
            "recall_at_1": retrieval_recall_at_1(
                query_outputs[width], query_labels, gallery_outputs[width], gallery_labels
            ),
            "map_at_r": retrieval_map_at_r(
                query_outputs[width], query_labels, gallery_outputs[width], gallery_labels
            ),
        }
        for width in prefixes
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    embeddings, ids, labels, metadata = _load_cache(args.train_cache)
    if args.hardened_fold is None:
        optimization, query, gallery = _split_indexes(
            ids, labels, fraction=args.validation_fraction, seed=args.split_seed
        )
    else:
        folds = hardened_retrieval_folds(labels, seed=args.split_seed)
        if not 0 <= args.hardened_fold < len(folds):
            raise ValueError("hardened fold index differs from the four registered folds")
        fold = folds[args.hardened_fold]
        optimization = fold.optimization
        query = fold.query
        gallery = np.concatenate((fold.gallery, fold.distractor))
    train_labels_original = labels[optimization]
    eligible = sorted(
        int(label)
        for label in np.unique(train_labels_original)
        if np.count_nonzero(train_labels_original == label) >= args.images_per_identity
    )
    label_index = {label: index for index, label in enumerate(eligible)}
    keep = np.asarray([int(label) in label_index for label in train_labels_original])
    optimization = optimization[keep]
    train_labels = np.asarray(
        [label_index[int(label)] for label in labels[optimization]], dtype=np.int64
    )
    config = AdapterConfig(
        input_dim=int(embeddings.shape[1]),
        hidden_dim=args.hidden_dim,
        class_count=len(eligible),
        margin=args.margin,
        scale=args.scale,
    )
    model_class = NestedResidualAdapter if args.model == "mlp" else NestedLinearAdapter
    model = model_class(config)
    if args.initialize_from_linear is not None:
        saved = torch.load(args.initialize_from_linear, map_location="cpu", weights_only=False)
        if saved["model"] != "linear" or saved["config"] != asdict(config):
            raise ValueError("linear warm-start checkpoint differs from this experiment")
        linear = NestedLinearAdapter(config)
        linear.load_state_dict(saved["state_dict"])
        if isinstance(model, NestedResidualAdapter):
            initialize_residual_from_linear(model, linear)
        else:
            model.load_state_dict(linear.state_dict())
    device = torch.device(args.device)
    model.to(device)
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    train = torch.from_numpy(normalized[optimization]).to(device)
    validation_query = torch.from_numpy(normalized[query]).to(device)
    validation_gallery = torch.from_numpy(normalized[gallery]).to(device)
    query_labels = labels[query]
    gallery_labels = labels[gallery]
    raw_recall = retrieval_recall_at_1(
        validation_query, query_labels, validation_gallery, gallery_labels
    )
    raw_map = retrieval_map_at_r(
        validation_query, query_labels, validation_gallery, gallery_labels
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(args.epochs + 1):
        recalls = _evaluate(
            model,
            validation_query,
            query_labels,
            validation_gallery,
            gallery_labels,
            config.prefixes,
        )
        history.append(
            {
                "epoch": epoch,
                "metrics": {str(width): recalls[width] for width in recalls},
            }
        )
        if epoch == args.epochs:
            break
        model.train()
        batches = identity_balanced_batches(
            train_labels,
            identities_per_batch=args.identities_per_batch,
            images_per_identity=args.images_per_identity,
            seed=args.seed * 100_000 + epoch,
        )
        for batch in batches:
            indexes = torch.from_numpy(batch).to(device)
            batch_labels = torch.from_numpy(train_labels[batch]).to(device)
            loss, _ = cosine_margin_loss(
                model(train[indexes]), batch_labels, model.class_proxies, config
            )
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
    elapsed = time.perf_counter() - started
    final_metrics = recalls
    fixed_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    checkpoint = {
        "config": asdict(config),
        "model": args.model,
        "state_dict": fixed_state,
        "fixed_epoch": args.epochs,
    }
    if args.checkpoint.exists():
        raise FileExistsError(args.checkpoint)
    torch.save(checkpoint, args.checkpoint)
    return {
        "schema_version": "foundation-adapter-screen-v1",
        "model": args.model,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "config": asdict(config),
        "source_arm": metadata["key"]["arm"],
        "train_rows": int(train.shape[0]),
        "train_identities": len(eligible),
        "validation_query_rows": int(validation_query.shape[0]),
        "validation_gallery_rows": int(validation_gallery.shape[0]),
        "raw_recall_at_1": raw_recall,
        "raw_map_at_r": raw_map,
        "fixed_epoch": args.epochs,
        "validation_metrics": {
            str(width): final_metrics[width] for width in config.prefixes
        },
        "trainable_parameters": trainable_parameter_count(model),
        "training_seconds": elapsed,
        "history": history,
        "checkpoint": str(args.checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=("linear", "mlp"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--hardened-fold", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument("--hidden-dim", type=int, default=1_024)
    parser.add_argument("--initialize-from-linear", type=Path)
    parser.add_argument("--identities-per-batch", type=int, default=32)
    parser.add_argument("--images-per-identity", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = run(args)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary_keys = (
        "model",
        "seed",
        "fixed_epoch",
        "validation_metrics",
        "training_seconds",
    )
    print(json.dumps({key: payload[key] for key in summary_keys}))


if __name__ == "__main__":
    main()
