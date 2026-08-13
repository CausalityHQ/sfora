#!/usr/bin/env python3
"""Screen frozen and tail-tuned OML ViT encoders on unseen training identities."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sfora.data import ImageExample, materialize_image
from sfora.foundation_adapter import retrieval_map_at_r, retrieval_recall_at_1
from sfora.foundation_finetune import (
    CosFaceHead,
    IdentityNeck,
    configure_vit_trainable_layers,
    select_query_gallery_identity_subset,
)
from sfora.foundation_oml import load_oml_inshop_examples, load_oml_vit


class ImageIdentityRows(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, examples: list[ImageExample], *, label_map: dict[int, int], training: bool
    ) -> None:
        self.examples = examples
        self.label_map = label_map
        if training:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.examples[index]
        return self.transform(materialize_image(row.image)), self.label_map[row.label]


def extract(
    model: nn.Module,
    neck: nn.Module,
    examples: list[ImageExample],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, np.ndarray]:
    labels = {label: label for label in {row.label for row in examples}}
    loader = DataLoader(
        ImageIdentityRows(examples, label_map=labels, training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features: list[torch.Tensor] = []
    observed_labels: list[np.ndarray] = []
    model.eval()
    neck.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
            output = model(images.to(device, non_blocking=True))
            features.append(F.normalize(neck(output.float()), dim=1).cpu())
            observed_labels.append(batch_labels.numpy().astype(np.int64, copy=False))
    return torch.cat(features), np.concatenate(observed_labels)


def evaluate(
    model: nn.Module,
    neck: nn.Module,
    query: list[ImageExample],
    gallery: list[ImageExample],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, float]:
    query_features, query_labels = extract(
        model, neck, query, device=device, batch_size=batch_size, workers=workers
    )
    gallery_features, gallery_labels = extract(
        model, neck, gallery, device=device, batch_size=batch_size, workers=workers
    )
    return {
        "recall_at_1": retrieval_recall_at_1(
            query_features, query_labels, gallery_features, gallery_labels
        ),
        "map_at_r": retrieval_map_at_r(
            query_features, query_labels, gallery_features, gallery_labels
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-seed", type=int, default=17)
    parser.add_argument("--evaluation-role", choices=("screen", "holdout"), default="screen")
    parser.add_argument("--trainable-blocks", type=int, choices=(0, 4), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.output_checkpoint.exists():
        raise FileExistsError("output or checkpoint already exists")
    if args.epochs <= 0 or not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("epochs must be positive and warmup must end before the final epoch")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimization = load_oml_inshop_examples(
        args.partition, image_root=args.image_root, split="train"
    )
    official_query = load_oml_inshop_examples(
        args.partition, image_root=args.image_root, split="query"
    )
    official_gallery = load_oml_inshop_examples(
        args.partition, image_root=args.image_root, split="gallery"
    )
    query, gallery = select_query_gallery_identity_subset(
        official_query,
        official_gallery,
        seed=args.evaluation_seed,
        fraction=args.evaluation_fraction,
        complement=args.evaluation_role == "holdout",
    )
    optimization_labels = sorted({row.label for row in optimization})
    label_map = {label: index for index, label in enumerate(optimization_labels)}
    counts = Counter(row.label for row in optimization)
    weights = torch.tensor([1.0 / counts[row.label] for row in optimization], dtype=torch.double)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(optimization), replacement=True, generator=generator
    )
    loader = DataLoader(
        ImageIdentityRows(optimization, label_map=label_map, training=True),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    model = load_oml_vit(str(args.checkpoint), device=device)
    trainable_names = configure_vit_trainable_layers(model, trainable_blocks=args.trainable_blocks)
    embedding_dim = int(model.num_features)
    neck = IdentityNeck(embedding_dim).to(device)
    head = CosFaceHead(
        embedding_dim=embedding_dim,
        class_count=len(optimization_labels),
        margin=0.2,
        scale=32.0,
    ).to(device)
    parameter_groups: list[dict[str, object]] = [
        {
            "params": [*neck.parameters(), *head.parameters()],
            "lr": 1.0e-3,
            "weight_decay": 1.0e-4,
        }
    ]
    encoder_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": 1.0e-5, "weight_decay": 1.0e-4}
        )
    optimizer = torch.optim.AdamW(parameter_groups, fused=device.type == "cuda")
    initial = evaluate(
        model,
        neck,
        query,
        gallery,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    epochs: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        encoder_active = bool(encoder_parameters) and epoch > args.warmup_epochs
        model.train(encoder_active)
        neck.train()
        head.train()
        loss_sum = 0.0
        examples_seen = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                if encoder_active:
                    embeddings = model(images)
                else:
                    with torch.no_grad():
                        embeddings = model(images)
            with torch.autocast(device_type=device.type, enabled=False):
                projected = neck(embeddings.float())
                logits = head(projected, labels)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * labels.shape[0]
            examples_seen += labels.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - train_started
        training_peak_memory = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
        eval_started = time.perf_counter()
        metrics = evaluate(
            model,
            neck,
            query,
            gallery,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        eval_seconds = time.perf_counter() - eval_started
        row: dict[str, float | int] = {
            "epoch": epoch,
            "loss": loss_sum / examples_seen,
            **metrics,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "train_images_per_second": examples_seen / train_seconds,
            "peak_gpu_memory_bytes": training_peak_memory,
        }
        epochs.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    final = (
        initial
        if not epochs
        else {
            "recall_at_1": float(epochs[-1]["recall_at_1"]),
            "map_at_r": float(epochs[-1]["map_at_r"]),
        }
    )
    torch.save(
        {
            "model": model.state_dict(),
            "neck": neck.state_dict(),
            "head": head.state_dict(),
            "epoch": args.epochs,
            "validation": final,
        },
        args.output_checkpoint,
    )
    payload = {
        "method": "OML ViT-S/16 CosFace tail fine-tuning",
        "seed": args.seed,
        "trainable_blocks": args.trainable_blocks,
        "warmup_epochs": args.warmup_epochs,
        "evaluation_role": args.evaluation_role,
        "evaluation_seed": args.evaluation_seed,
        "evaluation_fraction": args.evaluation_fraction,
        "trainable_modules": trainable_names,
        "optimization_rows": len(optimization),
        "evaluation_identities": len({row.label for row in query}),
        "validation_query_rows": len(query),
        "validation_gallery_rows": len(gallery),
        "initial": initial,
        "epochs": epochs,
        "final": final,
        "final_recall_delta": final["recall_at_1"] - initial["recall_at_1"],
        "final_map_at_r_delta": final["map_at_r"] - initial["map_at_r"],
        "peak_gpu_memory_bytes": max(int(row["peak_gpu_memory_bytes"]) for row in epochs),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
