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

from sfora.data import ImageExample, materialize_image
from sfora.foundation_adapter import retrieval_map_at_r, retrieval_recall_at_1
from sfora.foundation_finetune import select_query_gallery_identity_subset
from sfora.foundation_oml import (
    configure_oml_input_size,
    load_oml_inshop_examples,
    load_oml_vit,
)

REGISTERED_SCREEN_BASELINE_RECALL = 0.961093585699264
REGISTERED_SCREEN_BASELINE_MAP = 0.791594927741186


class ImageRows(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, examples: list[ImageExample], *, input_size: int) -> None:
        self.examples = examples
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
    input_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(
        ImageRows(examples, input_size=input_size),
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


def latency_ms(
    model: torch.nn.Module, *, device: torch.device, batch_size: int, input_size: int
) -> float:
    images = torch.randn(batch_size, 3, input_size, input_size, device=device)
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
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="OML img_highres directory (the released model's evaluation corpus)",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--evaluation-fraction", type=float)
    parser.add_argument("--evaluation-seed", type=int, default=17)
    parser.add_argument("--evaluation-role", choices=("screen", "holdout"), default="screen")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    model = load_oml_vit(str(args.checkpoint), device=device)
    configure_oml_input_size(model, input_size=args.input_size)
    partition = args.dataset_root / "Eval" / "list_eval_partition.txt"
    query = load_oml_inshop_examples(partition, image_root=args.image_root, split="query")
    gallery = load_oml_inshop_examples(partition, image_root=args.image_root, split="gallery")
    if args.evaluation_fraction is not None:
        query, gallery = select_query_gallery_identity_subset(
            query,
            gallery,
            seed=args.evaluation_seed,
            fraction=args.evaluation_fraction,
            complement=args.evaluation_role == "holdout",
        )
    query_features, query_labels = extract(
        model,
        query,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        input_size=args.input_size,
    )
    gallery_features, gallery_labels = extract(
        model,
        gallery,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
        input_size=args.input_size,
    )
    checkpoint_sha = sha256(args.checkpoint.read_bytes()).hexdigest()
    recall = retrieval_recall_at_1(query_features, query_labels, gallery_features, gallery_labels)
    map_at_r = retrieval_map_at_r(query_features, query_labels, gallery_features, gallery_labels)
    registered_screen = (
        args.evaluation_fraction == 0.2
        and args.evaluation_seed == 17
        and args.evaluation_role == "screen"
    )
    registered_gate = args.input_size == 288 and registered_screen
    payload = {
        "model": "OML vits16_inshop",
        "protocol": "OML img_highres query/gallery identity subset"
        if args.evaluation_fraction is not None
        else "OML img_highres full query/gallery",
        "input_size": args.input_size,
        "tokens": (args.input_size // 16) ** 2 + 1,
        "evaluation_fraction": args.evaluation_fraction,
        "evaluation_seed": args.evaluation_seed,
        "evaluation_role": args.evaluation_role,
        "checkpoint_sha256": checkpoint_sha,
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "embedding_dimensions": query_features.shape[1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "recall_at_1": recall,
        "map_at_r": map_at_r,
        "recall_at_1_delta_from_registered_224_screen": recall - REGISTERED_SCREEN_BASELINE_RECALL
        if registered_screen
        else None,
        "map_at_r_delta_from_registered_224_screen": map_at_r - REGISTERED_SCREEN_BASELINE_MAP
        if registered_screen
        else None,
        "fixres_gate_passed": bool(
            map_at_r - REGISTERED_SCREEN_BASELINE_MAP >= 0.008
            and recall - REGISTERED_SCREEN_BASELINE_RECALL >= 0.0015
        )
        if registered_gate
        else None,
        "batch32_latency_ms": latency_ms(
            model, device=device, batch_size=32, input_size=args.input_size
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
