#!/usr/bin/env python3
"""Measure whether unseen In-Shop feature maps leave the train support envelope.

This is a training/checkpoint diagnostic only. It does not select a method or
read test labels for model selection: train channel extrema are fitted first,
then query/gallery activations are scored against that fixed envelope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sfora.data import load_image_retrieval_bundle
from sfora.image_end_to_end import (
    ImageEndToEndConfig,
    _TorchImageDataset,
    _default_transform_factory,
    _torchvision_model_factory,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    report = json.loads(args.report.read_text())
    raw = dict(report["config"])
    for key in ("cem_edges_path", "cem_margin", "cem_weight"):
        raw.pop(key, None)
    config = ImageEndToEndConfig.model_validate(raw)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _torchvision_model_factory(config)
    state = {
        k: v
        for k, v in checkpoint["state_dict"].items()
        if k not in {"metric_proxies", "metric_proxy_labels"}
    }
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = _default_transform_factory(config, False)
    bundle = load_image_retrieval_bundle(
        dataset_name="inshop", dataset_root=args.dataset_root, seed=config.seed
    )

    def batches(examples):
        loader = DataLoader(
            _TorchImageDataset(examples, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        with torch.inference_mode():
            for images, _labels in loader:
                features = model.model.features(images.to(device, non_blocking=True))
                # The reference head consumes GAP+GMP of this 1024-channel map.
                yield (features.mean((2, 3)) + features.amax((2, 3))).cpu().numpy()

    train_min = np.full(1024, np.inf, dtype=np.float64)
    train_max = np.full(1024, -np.inf, dtype=np.float64)
    for values in batches(bundle.train):
        train_min = np.minimum(train_min, values.min(axis=0))
        train_max = np.maximum(train_max, values.max(axis=0))

    splits = {}
    for name, examples in (("query", bundle.query), ("gallery", bundle.gallery)):
        outside_values = 0
        total_values = 0
        rows_outside = 0
        rows = 0
        for values in batches(examples):
            outside = (values < train_min) | (values > train_max)
            outside_values += int(outside.sum())
            total_values += int(outside.size)
            rows_outside += int(outside.any(axis=1).sum())
            rows += len(values)
        splits[name] = {
            "rows": rows,
            "outside_value_fraction": outside_values / total_values,
            "rows_with_any_outside_channel": rows_outside / rows,
        }

    result = {
        "checkpoint": str(args.checkpoint),
        "feature_dimensions": 1024,
        "train_channel_min_min": float(train_min.min()),
        "train_channel_max_max": float(train_max.max()),
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
