#!/usr/bin/env python3
"""Evaluate and benchmark the released OML ViT-S/16 In-Shop extractor."""

from __future__ import annotations

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sfora.data import ImageExample, load_image_retrieval_examples, materialize_image
from sfora.foundation_adapter import retrieval_map_at_r, retrieval_recall_at_1
from sfora.foundation_oml import load_oml_vit


class ImageRows(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, examples: list[ImageExample]) -> None:
        self.examples = examples
        self.transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.examples[index]
        return self.transform(materialize_image(row.image)), int(row.label)


def extract(
    model: torch.nn.Module,
    examples: list[ImageExample],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(
        ImageRows(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    outputs: list[torch.Tensor] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_labels in loader:
            outputs.append(model(images.to(device, non_blocking=True)).detach().cpu())
            labels.append(batch_labels.numpy())
    return torch.cat(outputs), np.concatenate(labels).astype(np.int64, copy=False)


def latency_ms(model: torch.nn.Module, *, device: torch.device, batch_size: int) -> float:
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        for _ in range(20):
            model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(100):
            model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) * 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    model = load_oml_vit(str(args.checkpoint), device=device)
    query = load_image_retrieval_examples(
        dataset_name="inshop", split="query", dataset_root=args.dataset_root
    )
    gallery = load_image_retrieval_examples(
        dataset_name="inshop", split="gallery", dataset_root=args.dataset_root
    )
    query_features, query_labels = extract(
        model, query, device=device, batch_size=args.batch_size, workers=args.workers
    )
    gallery_features, gallery_labels = extract(
        model, gallery, device=device, batch_size=args.batch_size, workers=args.workers
    )
    checkpoint_sha = sha256(args.checkpoint.read_bytes()).hexdigest()
    payload = {
        "model": "OML vits16_inshop",
        "checkpoint_sha256": checkpoint_sha,
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "embedding_dimensions": query_features.shape[1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "recall_at_1": retrieval_recall_at_1(
            query_features, query_labels, gallery_features, gallery_labels
        ),
        "map_at_r": retrieval_map_at_r(
            query_features, query_labels, gallery_features, gallery_labels
        ),
        "batch32_latency_ms": latency_ms(model, device=device, batch_size=32),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
