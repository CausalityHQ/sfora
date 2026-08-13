#!/usr/bin/env python3
"""Continue a released OML ViT with head-free triplets and a feature trust region."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sfora.data import ImageExample, materialize_image
from sfora.foundation_finetune import (
    IdentityBalancedBatchSampler,
    IdentityNeck,
    TrainableParameterEMA,
    batch_hard_soft_triplet,
    configure_vit_trainable_layers,
    normalized_feature_anchor,
    paired_retrieval_statistics,
    retrieval_query_values,
    select_query_gallery_identity_subset,
    warmup_cosine_learning_rate_factor,
)
from sfora.foundation_oml import (
    configure_oml_input_size,
    load_oml_inshop_examples,
    load_oml_vit,
)

REGISTERED_SCREEN_BASELINE_RECALL = 0.961093585699264
REGISTERED_FIXRES_SCREEN_BASELINE_RECALL = 0.9674027339642481
REGISTERED_MINIMUM_MAP_DELTA = 0.004
REGISTERED_MAXIMUM_RECALL_DROP = 0.0007


def resolved_teacher_input_size(*, student_input_size: int, requested: int | None) -> int:
    return student_input_size if requested is None else requested


def registered_screen_baseline_recall(*, student_input_size: int) -> float:
    if student_input_size == 288:
        return REGISTERED_FIXRES_SCREEN_BASELINE_RECALL
    if student_input_size == 224:
        return REGISTERED_SCREEN_BASELINE_RECALL
    raise ValueError("unsupported registered student input size")


def validate_registered_initial(
    *,
    seed: int,
    evaluation_role: str,
    student_input_size: int,
    initial: dict[str, float],
) -> None:
    """Fail before training when a registered screen does not reproduce its baseline."""

    if seed != 0 or evaluation_role != "screen":
        return
    baseline = registered_screen_baseline_recall(student_input_size=student_input_size)
    if abs(initial["recall_at_1"] - baseline) > 1.0e-12:
        raise ValueError("measured screen baseline differs")


def registered_seed0_screen_decision(
    *,
    seed: int,
    evaluation_role: str,
    initial: dict[str, float],
    final: dict[str, float],
    paired: dict[str, float | int],
    registered_baseline_recall: float = REGISTERED_SCREEN_BASELINE_RECALL,
) -> bool | None:
    """Evaluate only the promotion screen for which the thresholds were chosen."""

    if seed != 0 or evaluation_role != "screen":
        return None
    if abs(initial["recall_at_1"] - registered_baseline_recall) > 1.0e-12:
        raise ValueError("measured screen baseline differs")
    return bool(
        paired["map_at_r_delta"] >= REGISTERED_MINIMUM_MAP_DELTA
        and paired["map_at_r_delta_ci95_lower"] > 0.0
        and final["recall_at_1"] >= registered_baseline_recall - REGISTERED_MAXIMUM_RECALL_DROP
    )


class OMLImages(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, examples: list[ImageExample], *, training: bool, input_size: int = 224
    ) -> None:
        self.examples = examples
        if training:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        input_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
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
        return self.transform(materialize_image(row.image)), row.label


class OMLCrossResolutionTrainingViews(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Produce one augmented view at student and teacher resolutions."""

    def __init__(
        self,
        examples: list[ImageExample],
        *,
        student_input_size: int,
        teacher_input_size: int,
    ) -> None:
        self.examples = examples
        self.teacher_input_size = teacher_input_size
        self.student_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    student_input_size,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.examples[index]
        student = self.student_transform(materialize_image(row.image))
        if self.teacher_input_size == student.shape[-1]:
            return student, student, row.label
        teacher = F.interpolate(
            student[None],
            size=(self.teacher_input_size, self.teacher_input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0]
        return student, teacher, row.label


def extract(
    model: nn.Module,
    neck: nn.Module,
    examples: list[ImageExample],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    input_size: int,
) -> tuple[torch.Tensor, np.ndarray]:
    loader = DataLoader(
        OMLImages(examples, training=False, input_size=input_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    features: list[torch.Tensor] = []
    labels: list[np.ndarray] = []
    model.eval()
    neck.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            output = model(images.to(device, non_blocking=True)).float()
            features.append(F.normalize(neck(output), dim=1).cpu())
            labels.append(batch_labels.numpy().astype(np.int64, copy=False))
    return torch.cat(features), np.concatenate(labels)


def evaluation_values(
    model: nn.Module,
    neck: nn.Module,
    query: list[ImageExample],
    gallery: list[ImageExample],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    input_size: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    query_features, query_labels = extract(
        model,
        neck,
        query,
        device=device,
        batch_size=batch_size,
        workers=workers,
        input_size=input_size,
    )
    gallery_features, gallery_labels = extract(
        model,
        neck,
        gallery,
        device=device,
        batch_size=batch_size,
        workers=workers,
        input_size=input_size,
    )
    hits, average_precision = retrieval_query_values(
        query_features, query_labels, gallery_features, gallery_labels
    )
    return (
        {"recall_at_1": float(hits.mean()), "map_at_r": float(average_precision.mean())},
        hits,
        average_precision,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--trainable-blocks", type=int, default=4)
    parser.add_argument("--student-input-size", type=int, default=224)
    parser.add_argument("--teacher-input-size", type=int)
    parser.add_argument("--labels-per-batch", type=int, default=32)
    parser.add_argument("--instances-per-label", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    parser.add_argument("--evaluation-seed", type=int, default=17)
    parser.add_argument("--evaluation-role", choices=("screen", "holdout"), default="screen")
    args = parser.parse_args()
    args.teacher_input_size = resolved_teacher_input_size(
        student_input_size=args.student_input_size, requested=args.teacher_input_size
    )
    if args.output.exists() or args.output_checkpoint.exists():
        raise FileExistsError("output or checkpoint already exists")
    if (
        args.epochs <= 0
        or args.learning_rate <= 0.0
        or args.anchor_weight < 0.0
        or args.weight_decay < 0.0
        or not 0.0 < args.ema_decay < 1.0
        or args.warmup_steps < 0
        or args.trainable_blocks <= 0
        or args.student_input_size not in (224, 288)
        or args.teacher_input_size not in (224, args.student_input_size)
    ):
        raise ValueError("anchored continuation configuration differs")
    total_started = time.perf_counter()
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
    student = load_oml_vit(str(args.checkpoint), device=device)
    teacher = load_oml_vit(str(args.checkpoint), device=device)
    configure_oml_input_size(student, input_size=args.student_input_size)
    configure_oml_input_size(teacher, input_size=args.teacher_input_size)
    teacher.requires_grad_(False).eval()
    trainable_names = configure_vit_trainable_layers(
        student, trainable_blocks=args.trainable_blocks
    )
    neck = IdentityNeck(int(student.num_features)).to(device)
    trainable = nn.ModuleDict({"student": student, "neck": neck})
    initial, initial_hits, initial_ap = evaluation_values(
        student,
        neck,
        query,
        gallery,
        device=device,
        batch_size=args.evaluation_batch_size,
        workers=args.workers,
        input_size=args.student_input_size,
    )
    validate_registered_initial(
        seed=args.seed,
        evaluation_role=args.evaluation_role,
        student_input_size=args.student_input_size,
        initial=initial,
    )
    sampler = IdentityBalancedBatchSampler(
        [row.label for row in optimization],
        labels_per_batch=args.labels_per_batch,
        instances_per_label=args.instances_per_label,
        seed=args.seed,
    )
    loader = DataLoader(
        OMLCrossResolutionTrainingViews(
            optimization,
            student_input_size=args.student_input_size,
            teacher_input_size=args.teacher_input_size,
        ),
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in trainable.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    total_steps = args.epochs * len(loader)
    if args.warmup_steps >= total_steps:
        raise ValueError("warmup must finish before training")

    def learning_rate_factor(step: int) -> float:
        return warmup_cosine_learning_rate_factor(
            step, warmup_steps=args.warmup_steps, total_steps=total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    ema = TrainableParameterEMA(trainable, decay=args.ema_decay)
    epochs: list[dict[str, float | int]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        student.train()
        neck.train()
        loss_sum = 0.0
        metric_sum = 0.0
        anchor_sum = 0.0
        rows_seen = 0
        started = time.perf_counter()
        last_update_learning_rate = 0.0
        for student_images, teacher_images, labels in loader:
            student_images = student_images.to(device, non_blocking=True)
            teacher_images = teacher_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                student_raw = student(student_images)
                with torch.no_grad():
                    teacher_raw = teacher(teacher_images)
            with torch.autocast(device_type=device.type, enabled=False):
                student_features = neck(student_raw.float())
                teacher_features = teacher_raw.float()
                metric_loss = batch_hard_soft_triplet(F.normalize(student_features, dim=1), labels)
                anchor_loss = normalized_feature_anchor(student_features, teacher_features)
                loss = metric_loss + args.anchor_weight * anchor_loss
            loss.backward()
            last_update_learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            ema.update(trainable)
            global_step += 1
            batch_rows = labels.numel()
            loss_sum += float(loss.detach()) * batch_rows
            metric_sum += float(metric_loss.detach()) * batch_rows
            anchor_sum += float(anchor_loss.detach()) * batch_rows
            rows_seen += batch_rows
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "loss": loss_sum / rows_seen,
            "metric_loss": metric_sum / rows_seen,
            "anchor_loss": anchor_sum / rows_seen,
            "train_seconds": seconds,
            "train_images_per_second": rows_seen / seconds,
            "final_update_learning_rate": last_update_learning_rate,
        }
        epochs.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if global_step != total_steps:
        raise RuntimeError("training step count differs")
    training_seconds = time.perf_counter() - training_started
    training_peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    raw_final, raw_hits, raw_ap = evaluation_values(
        student,
        neck,
        query,
        gallery,
        device=device,
        batch_size=args.evaluation_batch_size,
        workers=args.workers,
        input_size=args.student_input_size,
    )
    paired_raw = paired_retrieval_statistics(
        initial_hits=initial_hits,
        final_hits=raw_hits,
        initial_ap=initial_ap,
        final_ap=raw_ap,
        seed=args.seed + 20_260_814,
        bootstrap_replicates=10_000,
    )
    ema.apply(trainable)
    final, final_hits, final_ap = evaluation_values(
        student,
        neck,
        query,
        gallery,
        device=device,
        batch_size=args.evaluation_batch_size,
        workers=args.workers,
        input_size=args.student_input_size,
    )
    paired = paired_retrieval_statistics(
        initial_hits=initial_hits,
        final_hits=final_hits,
        initial_ap=initial_ap,
        final_ap=final_ap,
        seed=args.seed + 20_260_813,
        bootstrap_replicates=10_000,
    )
    torch.save(
        {
            "model": student.state_dict(),
            "neck": neck.state_dict(),
            "base_checkpoint": str(args.checkpoint),
            "student_input_size": args.student_input_size,
            "teacher_input_size": args.teacher_input_size,
            "epoch": args.epochs,
            "ema_decay": args.ema_decay,
        },
        args.output_checkpoint,
    )
    payload = {
        "method": "OML head-free anchored batch-hard triplet continuation",
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "anchor_weight": args.anchor_weight,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "warmup_steps": args.warmup_steps,
        "trainable_blocks": args.trainable_blocks,
        "student_input_size": args.student_input_size,
        "teacher_input_size": args.teacher_input_size,
        "trainable_modules": trainable_names,
        "labels_per_batch": args.labels_per_batch,
        "instances_per_label": args.instances_per_label,
        "evaluation_role": args.evaluation_role,
        "evaluation_seed": args.evaluation_seed,
        "evaluation_fraction": args.evaluation_fraction,
        "workers": args.workers,
        "torch_version": str(torch.__version__),
        "timm_version": importlib.metadata.version("timm"),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "determinism": "fixed seeds; CUDA deterministic algorithms are not forced",
        "optimization_rows": len(optimization),
        "evaluation_identities": len({row.label for row in query}),
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "trainable_parameters": sum(
            parameter.numel() for parameter in trainable.parameters() if parameter.requires_grad
        ),
        "initial": initial,
        "epochs_detail": epochs,
        "final_raw_diagnostic_only": raw_final,
        "paired_statistics_raw_diagnostic_only": paired_raw,
        "final_ema": final,
        "final_recall_delta": final["recall_at_1"] - initial["recall_at_1"],
        "final_map_at_r_delta": final["map_at_r"] - initial["map_at_r"],
        "paired_statistics": paired,
        "seed0_screen_passed": registered_seed0_screen_decision(
            seed=args.seed,
            evaluation_role=args.evaluation_role,
            initial=initial,
            final=final,
            paired=paired,
            registered_baseline_recall=registered_screen_baseline_recall(
                student_input_size=args.student_input_size
            ),
        ),
        "registered_minimum_map_delta": REGISTERED_MINIMUM_MAP_DELTA,
        "registered_maximum_recall_drop": REGISTERED_MAXIMUM_RECALL_DROP,
        "training_seconds": training_seconds,
        "total_wall_seconds": time.perf_counter() - total_started,
        "peak_training_gpu_memory_bytes": training_peak_memory,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
