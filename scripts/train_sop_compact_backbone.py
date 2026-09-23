#!/usr/bin/env python3
"""One full-backbone SOP compact arm with a train-only identity holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive, parse_sop_records
from PIL import Image
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import __version__ as torchvision_version
from torchvision import transforms

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.sop_compact_training import (
    RANK_COEFFICIENT,
    CompactTrainingArm,
    compact_head_features,
    compact_training_terms,
)
from sfora.sop_evaluation import score_symmetric
from sfora.unicom_rank_finish import identity_balanced_batches

CHECKPOINT_SHA256 = "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef"
ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
FIT_FRACTION = 0.9
SPLIT_SEED = 179019
BATCH_SIZE = 128
IMAGES_PER_IDENTITY = 4
EVAL_BATCH_SIZE = 64
SOURCE_RELATIVES = (
    "scripts/train_sop_compact_backbone.py",
    "scripts/export_unicom_sop_embeddings.py",
    "scripts/sop_teacher_anchored_runtime.py",
    "src/sfora/sop_compact_training.py",
    "src/sfora/deployed_code_rank.py",
    "src/sfora/unicom_training.py",
    "src/sfora/unicom_rank_finish.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/sop_evaluation.py",
    "src/sfora/joint_relational_compaction.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {relative: sha256(root / relative) for relative in SOURCE_RELATIVES}


def publish_file_noreplace(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Durably publish one completed file without exposing a partial target."""

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary_name)


class IndexedImages(Dataset):
    def __init__(self, paths: tuple[Path, ...], labels: tuple[int, ...], transform) -> None:
        if not paths or len(paths) != len(labels):
            raise ValueError("SOP compact image inventory differs")
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


class FixedBatches(Sampler[list[int]]):
    def __init__(self, batches: tuple[tuple[int, ...], ...]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield list(batch)


def make_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def initialize_head_and_classifier(
    fit_features: torch.Tensor, fit_labels: tuple[int, ...]
) -> tuple[nn.Linear, nn.Parameter]:
    """Initialize every arm from the same fit-only PCA and class means."""

    if (
        fit_features.device.type != "cpu"
        or fit_features.dtype != torch.float32
        or fit_features.shape != (len(fit_labels), 768)
        or len(set(fit_labels)) < 2
    ):
        raise ValueError("SOP compact initialization inventory differs")
    source = F.normalize(fit_features.contiguous(), dim=1)
    pca = fit_centered_pca(source, dimensions=128)
    head = nn.Linear(768, 128)
    with torch.no_grad():
        head.weight.copy_(pca.components)
        head.bias.copy_(-(pca.components @ pca.mean))
    projected = pca.apply(source)
    class_names = tuple(sorted(set(fit_labels)))
    class_index = {label: index for index, label in enumerate(class_names)}
    class_sum = torch.zeros(len(class_names), 128)
    for row, label in enumerate(fit_labels):
        class_sum[class_index[label]] += projected[row]
    classifier = nn.Parameter(F.normalize(class_sum, dim=1))
    return head, classifier


@torch.inference_mode()
def evaluate_validation(
    model: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    labels: tuple[int, ...],
) -> dict[str, object]:
    model.eval()
    head.eval()
    outputs = []
    for images, _ in loader:
        source = model(images.cuda(non_blocking=True))
        features = compact_head_features(source, head)
        outputs.append(F.normalize(features, dim=1).cpu())
    values = torch.cat(outputs).contiguous()
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    float_result = score_symmetric(values, label_tensor)
    packed = pack_int8_unit_embeddings(values)
    packed_result = score_symmetric(
        packed.codes.float(), label_tensor, inverse_norms=packed.inverse_norms
    )
    return {
        name: {
            "recall_at_1": float(scored["recall_at_1"]),
            "map_at_r": float(scored["map_at_r"]),
            "per_query_r1": scored["per_query_r1"],
            "per_query_ap": scored["per_query_ap"],
        }
        for name, scored in (("float", float_result), ("packed", packed_result))
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-archive", type=Path, required=True)
    parser.add_argument(
        "--arm", type=CompactTrainingArm, choices=tuple(CompactTrainingArm), required=True
    )
    parser.add_argument("--seed", type=int, default=179019)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--execute-sop-compact-training", action="store_true", required=True)
    args = parser.parse_args()
    if (
        args.seed < 0
        or args.updates < 1
        or args.workers < 0
        or args.checkpoint_output.exists()
        or args.receipt_output.exists()
    ):
        parser.error("SOP compact training invocation differs")
    return args


def main() -> None:
    args = parse_args()
    initial_source_manifest = source_manifest()
    for destination in (args.checkpoint_output, args.receipt_output):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=destination.parent):
            pass
    if not torch.cuda.is_available():
        raise ValueError("SOP compact training requires CUDA")
    if (
        sha256(args.checkpoint) != CHECKPOINT_SHA256
        or sha256(args.features_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP compact training source differs")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(16)
    records = parse_sop_records(args.dataset_root)
    train_records = tuple(record for record in records if record.split == "train")
    archive = load_sop_embedding_archive(args.features_archive)
    if tuple(record.image_id for record in train_records) != tuple(
        int(value) for value in archive["train_image_ids"]
    ) or tuple(record.label for record in train_records) != tuple(
        int(value) for value in archive["train_labels"]
    ):
        raise ValueError("SOP compact training archive rows differ")
    partition = deterministic_class_partition(
        tuple(record.label for record in train_records),
        fit_fraction=FIT_FRACTION,
        seed=SPLIT_SEED,
    )
    fit_records = tuple(train_records[index] for index in partition.fit_row_indexes)
    validation_records = tuple(train_records[index] for index in partition.validation_row_indexes)
    fit_labels = tuple(record.label for record in fit_records)
    validation_labels = tuple(record.label for record in validation_records)
    class_names = tuple(sorted(set(fit_labels)))
    class_index = {name: index for index, name in enumerate(class_names)}
    batches = identity_balanced_batches(
        tuple(str(label) for label in fit_labels),
        batch_size=BATCH_SIZE,
        images_per_identity=IMAGES_PER_IDENTITY,
        seed=args.seed,
        epoch=1,
        steps=args.updates,
    )
    schedule_sha256 = hashlib.sha256(
        np.asarray(batches, dtype="<i4").tobytes(order="C")
    ).hexdigest()
    fit_features = torch.from_numpy(
        np.ascontiguousarray(archive["train_embeddings"][list(partition.fit_row_indexes)])
    ).float()
    head, classifier = initialize_head_and_classifier(fit_features, fit_labels)
    initial_head_sha256 = hashlib.sha256(
        head.weight.detach().numpy().tobytes(order="C")
        + head.bias.detach().numpy().tobytes(order="C")
    ).hexdigest()
    initial_classifier_sha256 = hashlib.sha256(
        classifier.detach().numpy().tobytes(order="C")
    ).hexdigest()
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.checkpoint)
    model = authenticated.encoder.cuda()
    head = head.cuda()
    classifier = nn.Parameter(classifier.cuda())
    train_dataset = IndexedImages(
        tuple(record.image_path for record in fit_records),
        tuple(class_index[label] for label in fit_labels),
        make_train_transform(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=FixedBatches(batches),
        num_workers=args.workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_dataset = IndexedImages(
        tuple(record.image_path for record in validation_records),
        validation_labels,
        authenticated.transform,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    initial_validation_started = time.perf_counter()
    initial_validation = evaluate_validation(
        model, head, validation_loader, validation_labels
    )
    initial_validation_seconds = time.perf_counter() - initial_validation_started
    initial_float = initial_validation["float"]
    initial_packed = initial_validation["packed"]
    if (
        abs(initial_float["recall_at_1"] - 0.8207144077935395) > 0.002
        or abs(initial_float["map_at_r"] - 0.5647995976409332) > 0.002
        or abs(initial_packed["recall_at_1"] - 0.8205434968381473) > 0.002
        or abs(initial_packed["map_at_r"] - 0.5653424049482333) > 0.002
    ):
        raise ValueError("SOP compact step-zero image/archive parity differs")
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=1000)
    masks = torch.arange(128, device="cuda", dtype=torch.int64).unsqueeze(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step_seconds = []
    first_loss = None
    last_loss = None
    first_control = None
    last_control = None
    first_rank = None
    last_rank = None
    model.train()
    head.train()
    for step, (images, labels) in enumerate(train_loader, start=1):
        step_started = time.perf_counter()
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        source = model(images)
        features = compact_head_features(source, head)
        control, rank = compact_training_terms(
            features, classifier, labels, masks, arm=args.arm
        )
        loss = control + RANK_COEFFICIENT * rank
        if not bool(torch.isfinite(loss)):
            raise ValueError("SOP compact training loss is nonfinite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(head.parameters()) + [classifier],
            1.0,
            error_if_nonfinite=True,
        )
        observed_steps = 0

        def count_step(_optimizer, _args, _kwargs) -> None:
            nonlocal observed_steps
            observed_steps += 1

        hook = optimizer.register_step_post_hook(count_step)
        try:
            scaler.step(optimizer)
        finally:
            hook.remove()
        scaler.update()
        if observed_steps != 1:
            raise ValueError("SOP compact GradScaler skipped an optimizer step")
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_started)
        if first_loss is None:
            first_loss = float(loss.detach())
            first_control = float(control.detach())
            first_rank = float(rank.detach())
        last_loss = float(loss.detach())
        last_control = float(control.detach())
        last_rank = float(rank.detach())
        if step % 50 == 0 or step == args.updates:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": last_loss,
                        "arcface": last_control,
                        "rank": last_rank,
                    }
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    training_peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated()
    publish_file_noreplace(
        args.checkpoint_output,
        lambda stream: torch.save(
            {
                "model": model.state_dict(),
                "head": head.state_dict(),
                "classifier": classifier.detach().cpu(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "arm": args.arm.value,
                "seed": args.seed,
                "updates": args.updates,
                "schedule_sha256": schedule_sha256,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all(),
                "python_rng_state": random.getstate(),
            },
            stream,
        ),
    )
    validation_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    validation = evaluate_validation(model, head, validation_loader, validation_labels)
    validation_seconds = time.perf_counter() - validation_started
    if source_manifest() != initial_source_manifest:
        raise ValueError("SOP compact source changed during training")
    receipt = {
        "schema": "sfora-sop-compact-full-backbone-v1",
        "claim_eligible": False,
        "arm": args.arm.value,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": BATCH_SIZE,
        "images_per_identity": IMAGES_PER_IDENTITY,
        "fit_images": len(fit_records),
        "fit_classes": len(class_names),
        "validation_images": len(validation_records),
        "validation_classes": len(set(validation_labels)),
        "validation_image_ids": [record.image_id for record in validation_records],
        "validation_labels": list(validation_labels),
        "split_seed": SPLIT_SEED,
        "schedule_sha256": schedule_sha256,
        "fit_row_indexes_sha256": hashlib.sha256(
            np.asarray(partition.fit_row_indexes, dtype="<i4").tobytes(order="C")
        ).hexdigest(),
        "validation_row_indexes_sha256": hashlib.sha256(
            np.asarray(partition.validation_row_indexes, dtype="<i4").tobytes(order="C")
        ).hexdigest(),
        "initial_validation_seconds": initial_validation_seconds,
        "initial_validation": initial_validation,
        "protocol": {
            "backbone": "authenticated UNICOM ViT-B/16@224 full fine-tune",
            "head": "fit-only normalized PCA 768-to-128 affine initialization",
            "classifier": "fit-only 128-dimensional class-mean imprint",
            "augmentation": (
                "RandomResizedCrop224 scale 0.8:1.0 bilinear; "
                "RandomHorizontalFlip; UNICOM normalization"
            ),
            "objective": (
                "ArcFace margin 0.3 scale 64; rank arms add "
                "2.0 SmoothAP temperature 0.01"
            ),
            "optimizer": (
                "AdamW weight_decay 0.05; backbone lr 1e-5; "
                "head and classifier lr 1e-4; gradient clip 1.0"
            ),
            "precision": (
                "official internal fp16 backbone autocast with GradScaler; "
                "float32 source normalization, head, and objective"
            ),
            "selection": "train-identity holdout packed mAP@R; official test excluded",
        },
        "first_loss": first_loss,
        "last_loss": last_loss,
        "first_arcface_loss": first_control,
        "last_arcface_loss": last_control,
        "first_rank_loss": first_rank,
        "last_rank_loss": last_rank,
        "rank_coefficient": RANK_COEFFICIENT,
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "updates_per_second": args.updates / training_seconds,
        "step_seconds_p50": float(np.median(step_seconds)),
        "step_seconds_p95": float(np.quantile(step_seconds, 0.95)),
        "training_peak_cuda_allocated_bytes": training_peak_cuda_allocated_bytes,
        "validation_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "validation": validation,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision_version,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
        },
        "inputs": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "features_archive_sha256": ARCHIVE_SHA256,
            "sop_train_metadata_sha256": sha256(args.dataset_root / "Ebay_train.txt"),
            "source_sha256": initial_source_manifest,
            "initial_head_sha256": initial_head_sha256,
            "initial_classifier_sha256": initial_classifier_sha256,
            "checkpoint_output_sha256": sha256(args.checkpoint_output),
        },
        "argv": sys.argv,
    }
    publish_file_noreplace(
        args.receipt_output,
        lambda stream: stream.write(
            (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )
    print(json.dumps({"validation": validation, "receipt": str(args.receipt_output)}), flush=True)


if __name__ == "__main__":
    main()
