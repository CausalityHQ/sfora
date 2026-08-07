#!/usr/bin/env python3
"""Measure seen/unseen impostor geometry before In-Shop training.

This is a one-time Gate-1 diagnostic. It uses only the official ImageNet
BN-Inception checkpoint and deterministic test transforms; it does not train or
fit anything. The test split is read only to estimate whether the observed
unseen crowding exists at initialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sfora.data import load_image_retrieval_bundle, materialize_image
from sfora.image_end_to_end import (
    ImageEndToEndConfig,
    _TorchImageDataset,
    _default_transform_factory,
)


def _encode(model, loader, device, torch):
    values = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            features = model.model.features(images)
            pooled = model.model.gap(features).flatten(1)
            values.append(pooled.cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(values), np.concatenate(labels)


def _nearest_foreign(values: np.ndarray, labels: np.ndarray, *, chunk: int = 256) -> float:
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    best = np.full(len(values), -np.inf, dtype=np.float32)
    for start in range(0, len(values), chunk):
        stop = min(start + chunk, len(values))
        sim = values[start:stop] @ values.T
        same = labels[start:stop, None] == labels[None, :]
        sim[same] = -np.inf
        best[start:stop] = sim.max(axis=1)
    return float(np.mean(best))


def _query_gallery_foreign(query, query_labels, gallery, gallery_labels, *, chunk=256):
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    best = np.full(len(query), -np.inf, dtype=np.float32)
    for start in range(0, len(query), chunk):
        stop = min(start + chunk, len(query))
        sim = query[start:stop] @ gallery.T
        sim[query_labels[start:stop, None] == gallery_labels[None, :]] = -np.inf
        best[start:stop] = sim.max(axis=1)
    return float(np.mean(best))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from sfora.bn_inception import build_bn_inception

    bundle = load_image_retrieval_bundle(
        dataset_name="inshop", dataset_root=args.dataset_root,
        train_min_per_class=None, evaluation_min_per_class=None, seed=0,
    )
    config = ImageEndToEndConfig(
        dataset_name="inshop", backbone_name="bn_inception",
        pretrained_weights="bn_inception_52deb4733", embedding_dimensions=512,
        head_pooling="avg_max", input_size=224, freeze_batch_norm_affine=True,
    )
    transform = _default_transform_factory(config, False)
    def loader(examples):
        dataset = _TorchImageDataset(examples, transform)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model = build_bn_inception(embedding_size=512, pretrained=True, add_gmp=True,
                               freeze_batch_norm_affine=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train, train_labels = _encode(model, loader(bundle.train), device, torch)
    query, query_labels = _encode(model, loader(bundle.query), device, torch)
    gallery, gallery_labels = _encode(model, loader(bundle.gallery), device, torch)
    result = {
        "checkpoint": "bn_inception-52deb4733.pth",
        "train_rows": int(len(train)), "query_rows": int(len(query)),
        "gallery_rows": int(len(gallery)),
        "train_nearest_foreign_cosine": _nearest_foreign(train, train_labels),
        "query_nearest_foreign_gallery_cosine": _query_gallery_foreign(
            query, query_labels, gallery, gallery_labels
        ),
        "device": str(device), "trained": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
