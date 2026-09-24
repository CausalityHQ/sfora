#!/usr/bin/env python3
"""Matched SOP TRAIN continuation of one verified B/16 ArcFace checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import resource
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from export_unicom_sop_embeddings import _parse_split
from PIL import Image
from score_sop_b16_resolution_train import compare, sha256
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import (
    CompactTrainingArm,
    compact_head_features,
    compact_training_terms,
)
from sfora.sop_evaluation import score_gallery_r1, score_symmetric
from sfora.unicom_rank_finish import identity_balanced_batches
from sfora.unicom_resolution_adapter import output_at_resolution, train_output_at_resolution

START_SHA256 = "a2568c671336b2c5a97587634ed8312c026625e172eadcba0a0df57abf799672"
ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019
UPDATES = 200
ARMS = ("b16_224", "b16_336_detail", "b16_336_upsampled")
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def augmentation_seed(occurrence: int) -> int:
    return SEED * 1_000_003 + occurrence


def augmented_image(image: Image.Image, occurrence: int, arm: str) -> torch.Tensor:
    """Share each original-image crop and flip across all arms."""

    if arm not in ARMS:
        raise ValueError("unknown continuation arm")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(augmentation_seed(occurrence))
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(0.75, 4.0 / 3.0)
        )
        flip = bool(torch.rand(()) < 0.5)
    size = 336 if arm == "b16_336_detail" else 224
    cropped = TF.resized_crop(image, i, j, h, w, (size, size), InterpolationMode.BILINEAR)
    if flip:
        cropped = TF.hflip(cropped)
    tensor = TF.normalize(TF.to_tensor(cropped), MEAN, STD)
    if arm == "b16_336_upsampled":
        tensor = F.interpolate(
            tensor.unsqueeze(0), size=(336, 336), mode="bicubic", align_corners=False
        ).squeeze(0)
    return cast(torch.Tensor, tensor)


class OccurrenceImages(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, paths: tuple[Path, ...], labels: tuple[int, ...], rows: tuple[int, ...], arm: str
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.rows = rows
        self.arm = arm

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, occurrence: int) -> tuple[torch.Tensor, int]:
        row = self.rows[occurrence]
        with Image.open(self.paths[row]) as image:
            return augmented_image(image.convert("RGB"), occurrence, self.arm), self.labels[row]


class FixedBatches(Sampler[list[int]]):
    def __init__(self, steps: int, batch_size: int):
        self.steps = steps
        self.batch_size = batch_size

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.steps):
            yield list(range(step * self.batch_size, (step + 1) * self.batch_size))


class EvaluationImages(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: tuple[Path, ...], source_transform: Any) -> None:
        self.paths = paths
        self.source_transform = source_transform
        steps = source_transform.transforms
        if (
            len(steps) != 5
            or not isinstance(steps[0], transforms.Resize)
            or not isinstance(steps[1], transforms.CenterCrop)
            or steps[0].size != 224
            or steps[0].interpolation != InterpolationMode.BICUBIC
            or steps[1].size != (224, 224)
        ):
            raise ValueError("source B/16 evaluation transform differs")
        self.detail_transform = transforms.Compose(
            [
                transforms.Resize(336, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(336),
                *steps[2:],
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[row]) as image:
            rgb = image.convert("RGB")
            return self.source_transform(rgb), self.detail_transform(rgb)


@torch.inference_mode()
def evaluate_arm(
    model: nn.Module,
    head: nn.Linear,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    arm: str,
) -> torch.Tensor:
    model.eval()
    head.eval()
    chunks = []
    for base, detail in loader:
        images = detail if arm == "b16_336_detail" else base
        images = images.cuda(non_blocking=True)
        if arm == "b16_336_upsampled":
            images = F.interpolate(images, size=(336, 336), mode="bicubic", align_corners=False)
        source = output_at_resolution(model, images)
        features = compact_head_features(source, head)
        chunks.append(F.normalize(features, dim=1).cpu())
    values = torch.cat(chunks)
    if values.shape != (59_551, 128) or not bool(torch.isfinite(values).all()):
        raise ValueError("continuation evaluation feature inventory differs")
    return values


def score_arm(values: torch.Tensor, labels: torch.Tensor, held_rows: np.ndarray) -> dict[str, Any]:
    held_index = torch.from_numpy(held_rows).cuda()
    labels_gpu = labels.cuda()
    held_labels = labels_gpu[held_index]
    values_gpu = values.cuda()
    packed = pack_int8_unit_embeddings(values)
    codes_gpu = packed.codes.float().cuda()
    inverse_gpu = packed.inverse_norms.cuda()
    return {
        "float": {
            "holdout_only": score_symmetric(values_gpu[held_index], held_labels),
            "full_train_gallery": score_gallery_r1(values_gpu, labels_gpu, held_index),
        },
        "packed_130b": {
            "holdout_only": score_symmetric(
                codes_gpu[held_index], held_labels, inverse_norms=inverse_gpu[held_index]
            ),
            "full_train_gallery": score_gallery_r1(
                codes_gpu, labels_gpu, held_index, inverse_norms=inverse_gpu
            ),
        },
    }


def optimizer_step_range(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    """Audit that every trainable tensor received the intended updates."""

    steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
    if not steps or len(steps) != sum(len(group["params"]) for group in optimizer.param_groups):
        raise ValueError("continuation optimizer state inventory differs")
    return min(steps), max(steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "sop-root",
        "unicom-checkout",
        "source-checkpoint",
        "start-checkpoint",
        "source-archive",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or args.workers < 0
        or not torch.cuda.is_available()
        or sha256(args.start_checkpoint) != START_SHA256
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("continuation invocation or checkpoint differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    records = _parse_split(args.sop_root, "train")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        archive_train_ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        archive_train_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    if len(records) != 59_551 or tuple(r.image_id for r in records) != tuple(
        map(int, archive_train_ids)
    ):
        raise ValueError("SOP TRAIN row identity differs")
    labels = tuple(r.label for r in records)
    if labels != tuple(map(int, archive_train_labels)):
        raise ValueError("SOP TRAIN labels differ")
    partition = deterministic_class_partition(labels, fit_fraction=0.9, seed=SEED)
    fit_rows = partition.fit_row_indexes
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    fit_labels = tuple(labels[row] for row in fit_rows)
    if len(fit_rows) != 53_700 or len(held_rows) != 5_851:
        raise ValueError("SOP TRAIN partition differs")
    class_index = {name: index for index, name in enumerate(sorted(set(fit_labels)))}
    batches = identity_balanced_batches(
        tuple(str(label) for label in fit_labels),
        batch_size=128,
        images_per_identity=4,
        seed=SEED,
        epoch=2,
        steps=UPDATES,
    )
    schedule_sha = hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest()
    occurrence_rows = tuple(row for batch in batches for row in batch)
    source = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = source.encoder.cuda()
    head = nn.Linear(768, 128).cuda()
    classifier = nn.Parameter(torch.empty(len(class_index), 128, device="cuda"))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    scaler = cast(Any, torch.amp).GradScaler("cuda", init_scale=1024.0, growth_interval=1000)
    checkpoint = torch.load(args.start_checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("arm") != "arcface"
        or checkpoint.get("seed") != SEED
        or checkpoint.get("updates") != 1000
        or checkpoint["head"]["weight"].shape != (128, 768)
        or checkpoint["classifier"].shape != (len(class_index), 128)
    ):
        raise ValueError("continuation start state differs")
    eval_loader = DataLoader(
        EvaluationImages(tuple(r.image_path for r in records), source.transform),
        batch_size=32,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arm_results = {}
    checkpoints = {}
    masks = torch.arange(128, device="cuda", dtype=torch.int64).unsqueeze(0)
    for arm in ARMS:
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(checkpoint["head"])
        with torch.no_grad():
            classifier.copy_(checkpoint["classifier"].cuda())
        # AdamW's CPU step tensors can alias the input state dict. A deep copy
        # prevents the previous arm from advancing the next arm's clock.
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        scaler.load_state_dict(copy.deepcopy(checkpoint["scaler"]))
        if optimizer_step_range(optimizer) != (1000, 1000):
            raise ValueError("continuation start optimizer clock differs")
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        random.setstate(checkpoint["python_rng_state"])
        train_loader = DataLoader(
            OccurrenceImages(
                tuple(records[row].image_path for row in fit_rows),
                tuple(class_index[label] for label in fit_labels),
                occurrence_rows,
                arm,
            ),
            batch_sampler=FixedBatches(UPDATES, 128),
            num_workers=args.workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(SEED),
        )
        model.train()
        head.train()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        first_loss = None
        last_loss = None
        for step, (images, y) in enumerate(train_loader, 1):
            images = images.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            source_features = train_output_at_resolution(model, images)
            features = compact_head_features(source_features, head)
            control, rank = compact_training_terms(
                features, classifier, y, masks, arm=CompactTrainingArm.ARCFACE
            )
            loss = control + 2.0 * rank
            if not bool(torch.isfinite(loss)):
                raise ValueError("continuation loss is nonfinite")
            cast(torch.Tensor, scaler.scale(loss)).backward()  # type: ignore[no-untyped-call]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()) + [classifier],
                1.0,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            first_loss = float(loss.detach()) if first_loss is None else first_loss
            last_loss = float(loss.detach())
            if step % 25 == 0:
                print(json.dumps({"arm": arm, "step": step, "loss": last_loss}), flush=True)
        train_seconds = time.perf_counter() - begin
        if optimizer_step_range(optimizer) != (1200, 1200):
            raise ValueError("continuation did not complete 200 optimizer updates")
        train_peak = torch.cuda.max_memory_allocated()
        save_path = args.output_dir / f"{arm}-1200.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "head": head.state_dict(),
                "classifier": classifier.detach().cpu(),
                "arm": arm,
                "seed": SEED,
                "updates": 1200,
                "schedule_sha256": schedule_sha,
                "start_sha256": START_SHA256,
            },
            save_path,
        )
        checkpoints[arm] = {"path": save_path.name, "sha256": sha256(save_path)}
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        values = evaluate_arm(model, head, eval_loader, arm)
        encode_seconds = time.perf_counter() - begin
        encode_peak = torch.cuda.max_memory_allocated()
        begin = time.perf_counter()
        scored = score_arm(values, torch.tensor(labels, dtype=torch.int64), held_rows)
        score_seconds = time.perf_counter() - begin
        arm_results[arm] = {
            **scored,
            "train_seconds": train_seconds,
            "encode_seconds": encode_seconds,
            "score_seconds": score_seconds,
            "train_peak_cuda_allocated_bytes": train_peak,
            "encode_peak_cuda_allocated_bytes": encode_peak,
            "first_loss": first_loss,
            "last_loss": last_loss,
            "optimizer_step_range": optimizer_step_range(optimizer),
        }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "train_seconds": train_seconds,
                    "packed_full_r1": scored["packed_130b"]["full_train_gallery"]["recall_at_1"],
                }
            ),
            flush=True,
        )
        del train_loader, values, scored
        torch.cuda.empty_cache()
    held_labels = np.asarray(labels, dtype=np.int64)[held_rows]
    paired = {}
    for challenger, baseline_arm in ((ARMS[1], ARMS[0]), (ARMS[1], ARMS[2])):
        paired[f"{challenger}_minus_{baseline_arm}"] = {
            representation: {
                gallery: compare(
                    arm_results[challenger][representation][gallery],
                    arm_results[baseline_arm][representation][gallery],
                    held_labels,
                )
                for gallery in ("holdout_only", "full_train_gallery")
            }
            for representation in ("float", "packed_130b")
        }
    receipt = {
        "schema": "sfora-sop-b16-resolution-continuation-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint fit/holdout; no TEST rows read",
        "fit_images": 53_700,
        "holdout_queries": 5_851,
        "full_train_gallery_images": 59_551,
        "seed": SEED,
        "continuation_updates": UPDATES,
        "source_sha256": {
            "script": sha256(Path(__file__)),
            "adapter": sha256(Path(output_at_resolution.__code__.co_filename)),
        },
        "start_checkpoint_sha256": START_SHA256,
        "source_archive_sha256": ARCHIVE_SHA256,
        "source_checkpoint_sha256": source.checkpoint_sha256,
        "schedule_sha256": schedule_sha,
        "checkpoints": checkpoints,
        "arms": arm_results,
        "paired_product_bootstrap": paired,
        "total_seconds": time.perf_counter() - started,
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    with (args.output_dir / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps({"result": "completed", "total_seconds": receipt["total_seconds"]}), flush=True
    )


if __name__ == "__main__":
    main()
