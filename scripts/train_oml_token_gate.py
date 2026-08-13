#!/usr/bin/env python3
"""Fit a tiny patch-token gate on a frozen released OML ViT extractor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sfora.data import ImageExample, materialize_image
from sfora.foundation_adapter import retrieval_map_at_r, retrieval_recall_at_1
from sfora.foundation_finetune import (
    IdentityBalancedBatchSampler,
    TokenResidualGate,
    batch_hard_soft_triplet,
    select_query_gallery_identity_subset,
    split_cls_patch_tokens,
)
from sfora.foundation_oml import load_oml_inshop_examples, load_oml_vit


class EvaluationImages(Dataset[tuple[torch.Tensor, int]]):
    """Deterministic OML evaluation pixels and labels."""

    def __init__(self, examples: list[ImageExample]) -> None:
        self.examples = examples
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
        return self.transform(materialize_image(row.image)), row.label


def extract_token_pairs(
    model: nn.Module,
    examples: list[ImageExample],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        EvaluationImages(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    all_cls: list[torch.Tensor] = []
    all_patches: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            tokens = model.forward_features(images.to(device, non_blocking=True)).float()
            cls, patches = split_cls_patch_tokens(tokens)
            all_cls.append(cls.cpu())
            all_patches.append(patches.cpu())
            all_labels.append(labels.to(torch.int64))
    return torch.cat(all_cls), torch.cat(all_patches), torch.cat(all_labels)


def apply_gate(
    gate: TokenResidualGate,
    cls: torch.Tensor,
    patches: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    gate.eval()
    with torch.inference_mode():
        for start in range(0, cls.shape[0], batch_size):
            stop = start + batch_size
            output.append(
                gate(
                    cls[start:stop].to(device, non_blocking=True),
                    patches[start:stop].to(device, non_blocking=True),
                ).cpu()
            )
    return torch.cat(output)


def retrieval_metrics(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> dict[str, float]:
    return {
        "recall_at_1": retrieval_recall_at_1(
            query_features, query_labels.numpy(), gallery_features, gallery_labels.numpy()
        ),
        "map_at_r": retrieval_map_at_r(
            query_features, query_labels.numpy(), gallery_features, gallery_labels.numpy()
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--geometry-weight", type=float, default=0.25)
    parser.add_argument("--gate-weight", type=float, default=1.0e-4)
    parser.add_argument("--labels-per-batch", type=int, default=30)
    parser.add_argument("--instances-per-label", type=int, default=4)
    parser.add_argument("--extraction-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-seed", type=int, default=17)
    parser.add_argument("--evaluation-role", choices=("screen", "holdout"), default="screen")
    args = parser.parse_args()
    if args.output.exists() or args.output_checkpoint.exists():
        raise FileExistsError("output or checkpoint already exists")
    if args.epochs <= 0 or args.learning_rate <= 0.0 or args.geometry_weight < 0.0:
        raise ValueError("training configuration differs")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
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
    model = load_oml_vit(str(args.checkpoint), device=device)
    model.requires_grad_(False)
    extraction_started = time.perf_counter()
    train_cls, train_patches, train_labels = extract_token_pairs(
        model,
        optimization,
        device=device,
        batch_size=args.extraction_batch_size,
        workers=args.workers,
    )
    query_cls, query_patches, query_labels = extract_token_pairs(
        model,
        query,
        device=device,
        batch_size=args.extraction_batch_size,
        workers=args.workers,
    )
    gallery_cls, gallery_patches, gallery_labels = extract_token_pairs(
        model,
        gallery,
        device=device,
        batch_size=args.extraction_batch_size,
        workers=args.workers,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    extraction_seconds = time.perf_counter() - extraction_started
    initial = retrieval_metrics(
        F.normalize(query_cls, dim=1),
        query_labels,
        F.normalize(gallery_cls, dim=1),
        gallery_labels,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gate = TokenResidualGate(train_cls.shape[1]).to(device)
    sampler = IdentityBalancedBatchSampler(
        train_labels.tolist(),
        labels_per_batch=args.labels_per_batch,
        instances_per_label=args.instances_per_label,
        seed=args.seed,
    )
    loader = DataLoader(
        TensorDataset(train_cls, train_patches, train_labels),
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.learning_rate, weight_decay=0.0)
    epoch_rows: list[dict[str, float | int]] = []
    training_started = time.perf_counter()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        gate.train()
        loss_sum = 0.0
        metric_sum = 0.0
        geometry_sum = 0.0
        examples_seen = 0
        epoch_started = time.perf_counter()
        for cls, patches, labels in loader:
            cls = cls.to(device, non_blocking=True)
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = gate(cls, patches)
            metric_loss = batch_hard_soft_triplet(output, labels)
            geometry_loss = (1.0 - (output * F.normalize(cls, dim=1)).sum(dim=1)).mean()
            gate_loss = torch.tanh(gate.gate).square().mean()
            loss = metric_loss + args.geometry_weight * geometry_loss + args.gate_weight * gate_loss
            loss.backward()
            optimizer.step()
            batch_rows = labels.numel()
            loss_sum += float(loss.detach()) * batch_rows
            metric_sum += float(metric_loss.detach()) * batch_rows
            geometry_sum += float(geometry_loss.detach()) * batch_rows
            examples_seen += batch_rows
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - epoch_started
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "loss": loss_sum / examples_seen,
            "metric_loss": metric_sum / examples_seen,
            "geometry_loss": geometry_sum / examples_seen,
            "seconds": seconds,
            "images_per_second": examples_seen / seconds,
            "gate_mean_abs": float(torch.tanh(gate.gate).detach().abs().mean()),
            "gate_max_abs": float(torch.tanh(gate.gate).detach().abs().max()),
        }
        epoch_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    training_seconds = time.perf_counter() - training_started
    final = retrieval_metrics(
        apply_gate(
            gate,
            query_cls,
            query_patches,
            device=device,
            batch_size=args.extraction_batch_size,
        ),
        query_labels,
        apply_gate(
            gate,
            gallery_cls,
            gallery_patches,
            device=device,
            batch_size=args.extraction_batch_size,
        ),
        gallery_labels,
    )
    torch.save(
        {
            "gate": gate.state_dict(),
            "base_checkpoint": str(args.checkpoint),
            "embedding_dim": train_cls.shape[1],
        },
        args.output_checkpoint,
    )
    payload = {
        "method": "OML frozen ViT-S/16 diagonal residual token gate",
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "geometry_weight": args.geometry_weight,
        "gate_weight": args.gate_weight,
        "labels_per_batch": args.labels_per_batch,
        "instances_per_label": args.instances_per_label,
        "evaluation_role": args.evaluation_role,
        "evaluation_seed": args.evaluation_seed,
        "evaluation_fraction": args.evaluation_fraction,
        "optimization_rows": len(optimization),
        "evaluation_identities": len({row.label for row in query}),
        "trainable_parameters": sum(parameter.numel() for parameter in gate.parameters()),
        "initial": initial,
        "epochs_detail": epoch_rows,
        "final": final,
        "final_recall_delta": final["recall_at_1"] - initial["recall_at_1"],
        "final_map_at_r_delta": final["map_at_r"] - initial["map_at_r"],
        "extraction_seconds": extraction_seconds,
        "training_seconds": training_seconds,
        "peak_gate_training_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
