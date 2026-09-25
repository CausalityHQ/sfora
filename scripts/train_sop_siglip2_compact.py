#!/usr/bin/env python3
"""Matched SigLIP2 SOP TRAIN compact ArcFace and deployed-code rank arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.deployed_code_rank import smooth_ap_bank_loss
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.live_head_bank import live_head_bank_loss
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.sop_compact_training import (
    CompactTrainingArm,
    compact_head_features,
    compact_training_terms,
)
from sfora.unicom_rank_finish import identity_balanced_batches

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
TILEIRAS_SHA256 = "df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae"
MODEL_HASHES = {
    "config.json": "172e39dbf0143b8fe22d2f08921730eb8c397967e58cb37d133161e28aa34104",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
    "model.safetensors": "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
}
FIT_SEED = 179019
WIDTH = 1024
OUTPUT_WIDTH = 128
IMAGES_PER_IDENTITY = 4
SIGLIP2_RANK_COEFFICIENT = 8.0
GRAD_SCALER_INITIAL_SCALE = 128.0
MEMBER_BANK_PREFLIGHT_SHA256 = "54e806715e76b2caa7e877d55b0fecbdc921718386bce845e04367164828624c"
MEMBER_BANK_COST_SHA256 = "8f6867ff4de768ffd3bfa108cb86d7537b913d6ae5cb4f8cb16a43bc87f741a9"
LIVE_HEAD_COST_BENCHMARK_SHA256 = "bfcf51e3e6e9fee5ba4a0c45cf144177e8571aa00681236ebaeff9350ec09fb3"
SOURCE_RELATIVES = (
    "scripts/train_sop_siglip2_compact.py",
    "src/sfora/cutile_int8.py",
    "src/sfora/deployed_code_rank.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/live_head_bank.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/sop_compact_training.py",
    "src/sfora/unicom_rank_finish.py",
    "src/sfora/unicom_training.py",
)


def training_precision(
    choice: str, *, device: str
) -> tuple[torch.dtype, GradScaler]:
    """Choose vision autocast and loss scaling as one training recipe."""

    if choice not in {"fp16", "bf16"}:
        raise ValueError("unsupported SOP SigLIP2 training vision dtype")
    dtype = torch.float16 if choice == "fp16" else torch.bfloat16
    scaler = GradScaler(
        device,
        init_scale=GRAD_SCALER_INITIAL_SCALE,
        enabled=choice == "fp16",
    )
    return dtype, scaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_rows_sha256(ids: np.ndarray, labels: np.ndarray, relatives: np.ndarray) -> str:
    if ids.shape != (59_551,) or labels.shape != ids.shape or relatives.shape != ids.shape:
        raise ValueError("SOP SigLIP2 ordered TRAIN row inventory differs")
    return hashlib.sha256(
        "\n".join(
            f"{int(image_id)}\0{int(label)}\0{relative}"
            for image_id, label, relative in zip(ids, labels, relatives.astype(str), strict=True)
        ).encode()
    ).hexdigest()


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {relative: sha256(root / relative) for relative in SOURCE_RELATIVES}


def assert_source_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in SOURCE_RELATIVES:
        path = Path(relative)
        module = (
            sys.modules[__name__]
            if path.parts[0] == "scripts"
            else sys.modules.get(f"sfora.{path.stem}")
        )
        loaded_path = getattr(module, "__file__", None)
        if (
            module is None
            or not isinstance(loaded_path, str)
            or Path(loaded_path).resolve() != (root / path).resolve()
        ):
            raise ValueError("executed SOP SigLIP2 training source differs from receipt")


class ImageRows(Dataset):  # type: ignore[misc]
    def __init__(
        self,
        paths: tuple[Path, ...],
        labels: tuple[int, ...],
        *,
        augment: bool,
    ) -> None:
        if not paths or len(paths) != len(labels):
            raise ValueError("SOP SigLIP2 image rows differ")
        self.paths = paths
        self.labels = labels
        self.augment = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        with Image.open(self.paths[index]) as image:
            rgb = image.convert("RGB")
        return (self.augment(rgb) if self.augment is not None else rgb), self.labels[index]


class FixedBatches(Sampler[list[int]]):  # type: ignore[misc]
    def __init__(self, batches: tuple[tuple[int, ...], ...]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        for batch in self.batches:
            yield list(batch)


def make_collate(
    processor: Any,
) -> Callable[[list[tuple[Image.Image, int]]], tuple[dict[str, torch.Tensor], torch.Tensor]]:
    def collate(
        rows: list[tuple[Image.Image, int]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        images, labels = zip(*rows, strict=True)
        batch = processor(images=list(images), return_tensors="pt")
        tensors = {key: value for key, value in batch.items() if torch.is_tensor(value)}
        if tensors.get("pixel_values") is None or tensors["pixel_values"].shape[0] != len(rows):
            raise ValueError("SOP SigLIP2 processor batch geometry differs")
        return tensors, torch.tensor(labels, dtype=torch.int64)

    return collate


def paths_from_archive(dataset_root: Path, relatives: np.ndarray) -> tuple[Path, ...]:
    paths = []
    for relative_text in relatives.astype(str):
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("SOP SigLIP2 path inventory differs")
        path = dataset_root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError("SOP SigLIP2 image missing")
        paths.append(path)
    return tuple(paths)


def initialize_head_and_classifier(
    fit_features: torch.Tensor, fit_labels: tuple[int, ...]
) -> tuple[nn.Linear, nn.Parameter, str]:
    if (
        fit_features.shape != (len(fit_labels), WIDTH)
        or fit_features.device.type != "cpu"
        or fit_features.dtype != torch.float32
        or not bool(torch.isfinite(fit_features).all())
    ):
        raise ValueError("SOP SigLIP2 initialization inventory differs")
    normalized = F.normalize(fit_features, dim=1)
    pca = fit_centered_pca(normalized, dimensions=OUTPUT_WIDTH)
    head = nn.Linear(WIDTH, OUTPUT_WIDTH)
    with torch.no_grad():
        head.weight.copy_(pca.components)
        head.bias.copy_(-(pca.components @ pca.mean))
    projected = pca.apply(normalized)
    names = tuple(sorted(set(fit_labels)))
    indexes = {label: index for index, label in enumerate(names)}
    sums = torch.zeros(len(names), OUTPUT_WIDTH)
    counts = torch.zeros(len(names), dtype=torch.int64)
    for row, label in enumerate(fit_labels):
        sums[indexes[label]] += projected[row]
        counts[indexes[label]] += 1
    if int(counts.min()) < 2:
        raise ValueError("SOP SigLIP2 class proxy inventory differs")
    classifier = nn.Parameter(F.normalize(sums, dim=1))
    pca_sha = hashlib.sha256(
        pca.mean.numpy().tobytes() + pca.components.numpy().tobytes()
    ).hexdigest()
    return head, classifier, pca_sha


def member_bank_positive_ordinals(class_ids: np.ndarray) -> torch.Tensor:
    """Padded fit ordinals of every other member of each product."""

    if class_ids.ndim != 1 or class_ids.dtype != np.int64 or len(class_ids) < 2:
        raise ValueError("SOP member-bank classes differ")
    members: dict[int, list[int]] = {}
    for row, class_id in enumerate(class_ids):
        members.setdefault(int(class_id), []).append(row)
    if any(len(rows) < 2 for rows in members.values()):
        raise ValueError("SOP member-bank product has no positive")
    width = max(map(len, members.values())) - 1
    table = torch.full((len(class_ids), width), -1, dtype=torch.long)
    for rows in members.values():
        for row in rows:
            others = [other for other in rows if other != row]
            table[row, : len(others)] = torch.tensor(others, dtype=torch.long)
    return table


def member_bank_refresh_rows(
    batch_ordinals: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Refresh each fit member once, using its last augmented batch view."""

    if not batch_ordinals or any(type(row) is not int or row < 0 for row in batch_ordinals):
        raise ValueError("SOP member-bank refresh batch differs")
    last_position = {row: position for position, row in enumerate(batch_ordinals)}
    rows = tuple(sorted(last_position))
    return rows, tuple(last_position[row] for row in rows)


def member_bank_initial_values(
    source: torch.Tensor, head: nn.Linear, *, live_head: bool
) -> torch.Tensor:
    """Initialize cached rows in the geometry selected for this bank arm."""

    with torch.no_grad():
        return (
            F.normalize(source.float(), dim=1)
            if live_head
            else F.normalize(compact_head_features(source, head), dim=1)
        )


def member_bank_refresh_values(
    source: torch.Tensor,
    projected: torch.Tensor,
    positions: torch.Tensor,
    *,
    live_head: bool,
) -> torch.Tensor:
    """Write the last augmented view after its optimizer step, detached."""

    return F.normalize((source if live_head else projected).detach().float()[positions], dim=1)


def member_bank_rank_loss(
    anchors: torch.Tensor,
    bank: torch.Tensor,
    head: nn.Linear,
    positive_ordinals: torch.Tensor,
    self_ordinals: torch.Tensor,
    *,
    live_head: bool,
) -> torch.Tensor:
    unit_anchors = F.normalize(anchors.float(), dim=1)
    return (
        live_head_bank_loss(unit_anchors, bank, head, positive_ordinals, self_ordinals)
        if live_head
        else smooth_ap_bank_loss(unit_anchors, bank, positive_ordinals, self_ordinals)
    )


def validate_live_head_cost_receipt(receipt: dict[str, Any]) -> None:
    """Require a bounded isolated GPU cost before the live-head training arm."""

    try:
        detached = receipt["arms"]["detached_bank"]
        live = receipt["arms"]["live_head"]
        sources = receipt["source_sha256"]
        valid = (
            receipt["schema"] == "sfora-sop-siglip2-live-head-isolated-cost-v1"
            and receipt["claim_eligible"] is False
            and receipt["device"] == "cuda"
            and "GB10" in receipt["hardware"]
            and receipt["torch_version"] == torch.__version__
            and receipt["cuda_version"] == torch.version.cuda
            and receipt["matmul_allow_tf32"] is False
            and receipt["rows"] == 53_700
            and receipt["anchors"] == 64
            and receipt["positives_per_anchor"] == 11
            and receipt["timed_blocks"] >= 4
            and receipt["calls_per_block_per_arm"] >= 10
            and detached["calls"] >= 40
            and live["calls"] >= 40
            and math.isfinite(float(receipt["forward_loss_difference"]))
            and abs(float(receipt["forward_loss_difference"])) <= 1e-6
            and math.isfinite(float(detached["p95_ms"]))
            and float(detached["p95_ms"]) > 0
            and math.isfinite(float(live["p95_ms"]))
            and 0 < float(live["p95_ms"]) <= 150.0
            and 0 <= int(live["max_incremental_allocated_bytes"]) <= 2_000_000_000
            and receipt["persistent_bank_bytes"]["detached_bank"] == 53_700 * 128 * 4
            and receipt["persistent_bank_bytes"]["live_head"] == 53_700 * 1024 * 4
            and sources["script"] == LIVE_HEAD_COST_BENCHMARK_SHA256
            and sources["detached_loss"] == sha256(Path(smooth_ap_bank_loss.__code__.co_filename))
            and sources["live_loss"] == sha256(Path(live_head_bank_loss.__code__.co_filename))
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("live-head cost gate receipt malformed") from exc
    if not valid:
        raise ValueError("live-head cost gate failed")


@torch.inference_mode()  # type: ignore[untyped-decorator]
def export_all(
    vision: nn.Module,
    head: nn.Linear,
    paths: tuple[Path, ...],
    labels: tuple[int, ...],
    processor: Any,
    *,
    workers: int,
    batch_size: int,
) -> torch.Tensor:
    dataset = ImageRows(paths, labels, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=make_collate(processor),
    )
    vision.eval()
    head.eval()
    outputs = []
    for index, (batch, _labels) in enumerate(loader, start=1):
        tensors = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            source = vision(**tensors).pooler_output
        if source is None:
            raise ValueError("SOP SigLIP2 export pooler missing")
        values = compact_head_features(source, head)
        outputs.append(F.normalize(values, dim=1).cpu())
        if index % 100 == 0:
            print(json.dumps({"export_batches": index}), flush=True)
    result = torch.cat(outputs).contiguous()
    if result.shape != (len(paths), OUTPUT_WIDTH) or not bool(torch.isfinite(result).all()):
        raise ValueError("SOP SigLIP2 export geometry differs")
    return result


@torch.inference_mode()  # type: ignore[untyped-decorator]
def score_packed_full_gallery(
    codes: torch.Tensor,
    inverse_norms: torch.Tensor,
    labels: torch.Tensor,
    held_rows: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    if (
        codes.ndim != 2
        or codes.shape[1] != 128
        or inverse_norms.shape != (len(codes),)
        or labels.shape != (len(codes),)
        or held_rows.ndim != 1
        or len(held_rows) < 1
    ):
        raise ValueError("SOP SigLIP2 packed scoring geometry differs")
    code = codes.float().to(device)
    inverse = inverse_norms.float().to(device)
    gallery_labels = labels.to(device)
    query_rows = held_rows.to(device)
    relevant = torch.bincount(gallery_labels)[gallery_labels[query_rows]] - 1
    if int(relevant.min()) < 1:
        raise ValueError("SOP SigLIP2 packed scoring positive inventory differs")
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device=device)
    per_r1: list[float] = []
    per_ap: list[float] = []
    for block in query_rows.split(64):
        scores = (code[block] @ code.T) * inverse[block, None] * inverse[None, :]
        scores[torch.arange(len(block), device=device), block] = -torch.inf
        query_labels = gallery_labels[block]
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = gallery_labels[ranked] == query_labels[:, None]
        block_relevant = relevant[len(per_r1) : len(per_r1) + len(block)]
        precision = matches.cumsum(dim=1) / ranks[None, :]
        ap = (precision * matches * (ranks[None, :] <= block_relevant[:, None])).sum(
            dim=1
        ) / block_relevant
        per_r1.extend(float(value) for value in matches[:, 0].cpu().tolist())
        per_ap.extend(float(value) for value in ap.cpu().tolist())
    return {
        "recall_at_1": float(np.mean(per_r1)),
        "map_at_r": float(np.mean(per_ap)),
        "per_query_r1": per_r1,
        "per_query_ap": per_ap,
    }


def evaluate_packed(
    values: torch.Tensor,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
    native_library: Path,
) -> dict[str, Any]:
    packed = pack_int8_unit_embeddings(values)
    if len(fit_rows) != 53_700 or len(held_rows) != 5_851:
        raise ValueError("SOP SigLIP2 packed evaluation partition differs")
    scored = score_packed_full_gallery(
        packed.codes.float(),
        packed.inverse_norms,
        torch.from_numpy(labels.copy()),
        torch.from_numpy(held_rows.copy()),
        device=torch.device("cuda"),
    )
    native_r1 = []
    max_native_score_abs_delta = 0.0
    code_gpu = packed.codes.float().cuda()
    norm_gpu = packed.inverse_norms.float().cuda()
    with CutilePackedInt8Gallery.open_packed(native_library, packed) as gallery:
        for block in np.array_split(held_rows, range(32, len(held_rows), 32)):
            block_gpu = torch.from_numpy(block.copy()).cuda()
            query = type(packed)(
                packed.codes[block].contiguous(), packed.inverse_norms[block].contiguous()
            )
            ordinals, native_scores = gallery.search_packed(query)
            if (
                ordinals.shape != (len(block), 10)
                or native_scores.shape != ordinals.shape
                or not np.isfinite(native_scores).all()
            ):
                raise ValueError("SOP SigLIP2 trained native top-10 geometry differs")
            oracle_scores = (
                (code_gpu[block_gpu] @ code_gpu.T) * norm_gpu[block_gpu, None] * norm_gpu[None, :]
            )
            oracle_ordinals = torch.argsort(oracle_scores, dim=1, descending=True, stable=True)[
                :, :10
            ]
            selected_scores = oracle_scores.gather(1, oracle_ordinals)
            if not np.array_equal(ordinals, oracle_ordinals.cpu().numpy()):
                raise ValueError("SOP SigLIP2 trained native top-10 differs from packed oracle")
            score_delta = float(np.max(np.abs(native_scores - selected_scores.cpu().numpy())))
            max_native_score_abs_delta = max(max_native_score_abs_delta, score_delta)
            if score_delta > 1e-5:
                raise ValueError("SOP SigLIP2 trained native scores differ from packed oracle")
            for row, ranked in zip(block, ordinals, strict=True):
                top1 = next(int(ordinal) for ordinal in ranked if int(ordinal) != int(row))
                native_r1.append(bool(labels[top1] == labels[row]))
    scalar_r1 = np.asarray(scored["per_query_r1"], dtype=np.bool_)
    if not np.array_equal(np.asarray(native_r1, dtype=np.bool_), scalar_r1):
        raise ValueError("SOP SigLIP2 trained native quality differs from scalar packed score")
    return {
        "recall_at_1": scored["recall_at_1"],
        "map_at_r": scored["map_at_r"],
        "per_query_r1": scored["per_query_r1"],
        "per_query_ap": scored["per_query_ap"],
        "native_per_query_r1_equal": True,
        "native_top10_exact": True,
        "native_top10_max_score_abs_delta": max_native_score_abs_delta,
        "gallery_wire_bytes_per_row": 130,
    }


def main() -> None:
    program_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--unicom-l14-archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arm", type=CompactTrainingArm, choices=tuple(CompactTrainingArm), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gradient-diagnostic-steps", type=int, default=0)
    parser.add_argument("--train-vision-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--member-bank", action="store_true")
    parser.add_argument("--member-bank-preflight", type=Path)
    parser.add_argument("--expected-member-bank-preflight-sha256")
    parser.add_argument("--member-bank-cost-receipt", type=Path)
    parser.add_argument("--expected-member-bank-cost-sha256")
    parser.add_argument("--live-head-bank", action="store_true")
    parser.add_argument("--live-head-cost-receipt", type=Path)
    parser.add_argument("--expected-live-head-cost-sha256")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    arm_name = (
        "float_rank_live_head_member_bank"
        if args.live_head_bank
        else "float_rank_member_bank"
        if args.member_bank
        else args.arm.value
    )
    assert_source_imports()
    initial_source_manifest = source_manifest()
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        args.output_dir.exists()
        or args.output_dir.is_symlink()
        or args.updates < 0
        or (args.updates == 0 and (not args.evaluate or args.arm is not CompactTrainingArm.ARCFACE))
        or args.batch_size != 64
        or args.workers < 0
        or args.gradient_diagnostic_steps < 0
        or args.gradient_diagnostic_steps > args.updates
        or (args.arm is CompactTrainingArm.ARCFACE and args.gradient_diagnostic_steps)
        or (args.live_head_bank and args.gradient_diagnostic_steps)
        or (args.member_bank and args.arm is not CompactTrainingArm.FLOAT_RANK)
        or (args.live_head_bank and not args.member_bank)
        or (args.live_head_bank != (args.live_head_cost_receipt is not None))
        or (args.live_head_bank != (args.expected_live_head_cost_sha256 is not None))
        or (args.member_bank != (args.member_bank_preflight is not None))
        or (args.member_bank != (args.expected_member_bank_preflight_sha256 is not None))
        or (args.member_bank != (args.member_bank_cost_receipt is not None))
        or (args.member_bank != (args.expected_member_bank_cost_sha256 is not None))
        or (
            args.member_bank
            and (
                args.expected_member_bank_preflight_sha256 != MEMBER_BANK_PREFLIGHT_SHA256
                or args.expected_member_bank_cost_sha256 != MEMBER_BANK_COST_SHA256
            )
        )
        or not torch.cuda.is_available()
        or (args.train_vision_dtype == "bf16" and not torch.cuda.is_bf16_supported())
        or sha256(args.unicom_l14_archive) != ARCHIVE_SHA256
        or sha256(args.native_library) != NATIVE_SHA256
        or not tileiras
        or sha256(Path(tileiras)) != TILEIRAS_SHA256
        or any(sha256(args.model_snapshot / name) != value for name, value in MODEL_HASHES.items())
    ):
        raise ValueError("SOP SigLIP2 training authority differs")
    member_bank_preflight = None
    if args.member_bank:
        if sha256(args.member_bank_preflight) != args.expected_member_bank_preflight_sha256:
            raise ValueError("SOP member-bank preflight hash differs")
        if sha256(args.member_bank_cost_receipt) != args.expected_member_bank_cost_sha256:
            raise ValueError("SOP member-bank timing hash differs")
        member_bank_preflight = json.loads(args.member_bank_preflight.read_text())
        bank_cost = json.loads(args.member_bank_cost_receipt.read_text())
        if (
            member_bank_preflight.get("schema") != "sfora-sop-siglip2-member-bank-preflight-v1"
            or member_bank_preflight.get("decision") != "raw_bank_allowed"
            or member_bank_preflight.get("split")
            != "SOP official TRAIN fit products only, seed 179019"
        ):
            raise ValueError("SOP member-bank branch differs")
        if (
            bank_cost.get("schema") != "sfora-sop-siglip2-member-bank-step-cost-v1"
            or bank_cost.get("preflight_sha256") != args.expected_member_bank_preflight_sha256
            or bank_cost.get("loss_source_sha256")
            != initial_source_manifest["src/sfora/deployed_code_rank.py"]
            or bank_cost.get("gate_pass") is not True
            or bank_cost.get("median_wall_seconds", float("inf")) > 0.06
            or bank_cost.get("rows") != 53_700
            or bank_cost.get("anchors") != 64
            or bank_cost.get("positives_per_anchor") != 11
        ):
            raise ValueError("SOP member-bank timing branch differs")
    if args.live_head_bank:
        if sha256(args.live_head_cost_receipt) != args.expected_live_head_cost_sha256:
            raise ValueError("SOP live-head cost receipt hash differs")
        validate_live_head_cost_receipt(json.loads(args.live_head_cost_receipt.read_text()))
    export_receipt = json.loads((args.candidate_dir / "receipt.json").read_text())
    if (
        export_receipt.get("schema") != "sfora-sop-siglip2-train-feature-export-v1"
        or export_receipt.get("full_train") is not True
        or export_receipt.get("rows") != 59_551
        or export_receipt.get("unicom_train_archive_sha256") != ARCHIVE_SHA256
        or export_receipt.get("model_revision") != "787800c8990e6f058423089178e718139608408c"
        or export_receipt.get("model_file_sha256") != MODEL_HASHES
        or sha256(args.candidate_dir / "train_features.npy")
        != export_receipt.get("features_sha256")
    ):
        raise ValueError("SOP SigLIP2 source features differ")
    with np.load(args.unicom_l14_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"])
    if ordered_rows_sha256(ids, labels, relatives) != export_receipt.get("ordered_rows_sha256"):
        raise ValueError("SOP SigLIP2 export rows differ from authoritative TRAIN archive")
    source = np.load(args.candidate_dir / "train_features.npy", mmap_mode="r")
    if labels.shape != (59_551,) or source.shape != (59_551, WIDTH):
        raise ValueError("SOP SigLIP2 TRAIN inventory differs")
    paths = paths_from_archive(args.dataset_root, relatives)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=FIT_SEED
    )
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit_rows) != 53_700 or len(held_rows) != 5_851:
        raise ValueError("SOP SigLIP2 class partition differs")
    fit_labels = tuple(map(int, labels[fit_rows]))
    names = tuple(sorted(set(fit_labels)))
    class_index = {name: index for index, name in enumerate(names)}
    fit_class_ids = np.asarray([class_index[label] for label in fit_labels], dtype=np.int64)
    schedule = (
        identity_balanced_batches(
            tuple(map(str, fit_labels)),
            batch_size=args.batch_size,
            images_per_identity=IMAGES_PER_IDENTITY,
            seed=args.seed,
            epoch=1,
            steps=args.updates,
        )
        if args.updates
        else ()
    )
    schedule_sha = hashlib.sha256(np.asarray(schedule, dtype="<i4").tobytes()).hexdigest()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    fit_features = torch.from_numpy(np.asarray(source[fit_rows]).copy()).float()
    head, classifier, pca_sha = initialize_head_and_classifier(fit_features, fit_labels)
    initial_head_sha = hashlib.sha256(
        head.weight.detach().numpy().tobytes() + head.bias.detach().numpy().tobytes()
    ).hexdigest()
    initial_classifier_sha = hashlib.sha256(classifier.detach().numpy().tobytes()).hexdigest()
    if args.member_bank and (
        member_bank_preflight.get("fit_rows_sha256")
        != hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest()
        or member_bank_preflight.get("source_pca_sha256") != pca_sha
        or member_bank_preflight.get("initial_classifier_sha256") != initial_classifier_sha
    ):
        raise ValueError("SOP member-bank geometry differs from preflight")
    bank_init_started = time.perf_counter()
    bank_values = (
        member_bank_initial_values(fit_features, head, live_head=args.live_head_bank)
        if args.member_bank
        else None
    )
    bank_init_cpu_seconds = time.perf_counter() - bank_init_started if args.member_bank else 0.0
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    if (
        type(processor).__name__ != "SiglipImageProcessor"
        or processor.size["height"] != 256
        or processor.size["width"] != 256
        or processor.resample != 2
    ):
        raise ValueError("SOP SigLIP2 processor differs")
    full_model = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    model_type = full_model.config.model_type
    vision = full_model.vision_model
    del full_model
    vision = vision.float().cuda().train()
    head = head.cuda().train()
    classifier = nn.Parameter(classifier.cuda())
    bank_init_gpu_started = time.perf_counter()
    if args.member_bank:
        bank = bank_values.cuda()
        positive_table = member_bank_positive_ordinals(fit_class_ids).cuda()
        if positive_table.shape[1] != bank_cost["positives_per_anchor"]:
            raise ValueError("SOP member-bank positive width differs from timing gate")
        schedule_ordinals = torch.tensor(schedule, dtype=torch.long, device="cuda")
        fit_class_ids_gpu = torch.from_numpy(fit_class_ids).cuda()
    else:
        bank = None
        positive_table = None
        schedule_ordinals = None
        fit_class_ids_gpu = None
    bank_init_seconds = (
        bank_init_cpu_seconds + time.perf_counter() - bank_init_gpu_started
        if args.member_bank
        else 0.0
    )
    train_dataset = ImageRows(
        tuple(paths[row] for row in fit_rows),
        tuple(class_index[label] for label in fit_labels),
        augment=True,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=FixedBatches(schedule),
        num_workers=args.workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collate(processor),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": vision.parameters(), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    train_vision_dtype, scaler = training_precision(args.train_vision_dtype, device="cuda")
    masks = torch.arange(OUTPUT_WIDTH, device="cuda", dtype=torch.int64).unsqueeze(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step_seconds = []
    losses = []
    control_losses = []
    rank_losses = []
    first_input_batch_sha256 = []
    rank_to_arcface_head_gradient_ratio = []
    for step, (batch, target) in enumerate(loader, start=1):
        step_started = time.perf_counter()
        if step <= 10:
            input_digest = hashlib.sha256()
            for key in sorted(batch):
                input_digest.update(key.encode())
                input_digest.update(batch[key].contiguous().numpy().tobytes())
            input_digest.update(target.contiguous().numpy().tobytes())
            first_input_batch_sha256.append(input_digest.hexdigest())
        tensors = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=train_vision_dtype):
            source_features = vision(**tensors).pooler_output
        if source_features is None:
            raise ValueError("SOP SigLIP2 train pooler missing")
        features = compact_head_features(source_features, head)
        control, rank = compact_training_terms(
            features,
            classifier,
            target,
            masks,
            arm=CompactTrainingArm.ARCFACE if args.member_bank else args.arm,
            arcface_margin=0.3,
            arcface_scale=64.0,
        )
        if args.member_bank:
            query_ordinals = schedule_ordinals[step - 1]
            if not torch.equal(target, fit_class_ids_gpu[query_ordinals]):
                raise ValueError("SOP member-bank schedule labels differ")
            rank = member_bank_rank_loss(
                features,
                bank,
                head,
                positive_table[query_ordinals],
                query_ordinals,
                live_head=args.live_head_bank,
            )
        loss = control + SIGLIP2_RANK_COEFFICIENT * rank
        if not bool(torch.isfinite(loss)):
            raise ValueError("SOP SigLIP2 train loss is nonfinite")
        if step <= args.gradient_diagnostic_steps:
            control_gradient = torch.autograd.grad(control, features, retain_graph=True)[0]
            rank_gradient = torch.autograd.grad(rank, features, retain_graph=True)[0]
            control_norm = float(control_gradient.norm())
            rank_norm = float(rank_gradient.norm())
            if not np.isfinite(control_norm) or not np.isfinite(rank_norm) or control_norm <= 0:
                raise ValueError("SOP SigLIP2 training gradient diagnostic differs")
            rank_to_arcface_head_gradient_ratio.append(
                SIGLIP2_RANK_COEFFICIENT * rank_norm / control_norm
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(vision.parameters()) + list(head.parameters()) + [classifier],
            1.0,
            error_if_nonfinite=True,
        )
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            raise ValueError("SOP SigLIP2 optimizer step skipped")
        if args.member_bank:
            refresh_rows, refresh_positions = member_bank_refresh_rows(schedule[step - 1])
            bank[torch.tensor(refresh_rows, dtype=torch.long, device="cuda")] = (
                member_bank_refresh_values(
                    source_features,
                    features,
                    torch.tensor(refresh_positions, dtype=torch.long, device="cuda"),
                    live_head=args.live_head_bank,
                )
            )
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_started)
        losses.append(float(loss.detach()))
        control_losses.append(float(control.detach()))
        rank_losses.append(float(rank.detach()))
        if step % 50 == 0 or step == args.updates:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses[-1],
                        "arcface": control_losses[-1],
                        "rank": rank_losses[-1],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    training_wall = time.perf_counter() - started
    training_peak_cuda = torch.cuda.max_memory_allocated()
    if source_manifest() != initial_source_manifest:
        raise ValueError("SOP SigLIP2 training source changed during execution")
    checkpoint_path = args.output_dir / "checkpoint.pt"
    checkpoint_started = time.perf_counter()
    torch.save(
        {
            "vision": {key: value.detach().cpu() for key, value in vision.state_dict().items()},
            "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "classifier": classifier.detach().cpu(),
            "seed": args.seed,
            "updates": args.updates,
            "arm": arm_name,
            "member_bank": args.member_bank,
            "live_head_bank": args.live_head_bank,
            "train_vision_dtype": args.train_vision_dtype,
        },
        checkpoint_path,
    )
    checkpoint_write_seconds = time.perf_counter() - checkpoint_started
    quality = None
    export_seconds = None
    score_seconds = None
    if args.evaluate:
        export_started = time.perf_counter()
        values = export_all(
            vision,
            head,
            paths,
            tuple(map(int, labels)),
            processor,
            workers=args.workers,
            batch_size=args.batch_size,
        )
        export_seconds = time.perf_counter() - export_started
        np.save(args.output_dir / "train_embeddings.npy", values.numpy(), allow_pickle=False)
        score_started = time.perf_counter()
        quality = evaluate_packed(values, labels, fit_rows, held_rows, args.native_library)
        score_seconds = time.perf_counter() - score_started
        if args.updates == 0 and (
            abs(float(quality["recall_at_1"]) - 0.7834558195180311) > 0.002
            or abs(float(quality["map_at_r"]) - 0.5081678917820923) > 0.003
        ):
            raise ValueError("SOP SigLIP2 step-zero archive quality parity differs")
    if source_manifest() != initial_source_manifest:
        raise ValueError("SOP SigLIP2 source changed during export or scoring")
    receipt = {
        "schema": "sfora-sop-siglip2-compact-full-backbone-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint fit/holdout; no TEST rows",
        "arm": arm_name,
        "member_bank_preflight_sha256": args.expected_member_bank_preflight_sha256,
        "member_bank_cost_sha256": args.expected_member_bank_cost_sha256,
        "live_head_cost_sha256": args.expected_live_head_cost_sha256,
        "member_bank_init_seconds": bank_init_seconds,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "images_per_identity": IMAGES_PER_IDENTITY,
        "fit_images": len(fit_rows),
        "fit_products": len(names),
        "holdout_queries": len(held_rows),
        "gallery_images": len(labels),
        "source_sha256": initial_source_manifest["scripts/train_sop_siglip2_compact.py"],
        "source_files_sha256": initial_source_manifest,
        "model_file_sha256": MODEL_HASHES,
        "model_type": model_type,
        "vision_class": type(vision).__name__,
        "vision_parameters": sum(parameter.numel() for parameter in vision.parameters()),
        "processor_class": type(processor).__name__,
        "source_archive_sha256": ARCHIVE_SHA256,
        "source_features_sha256": export_receipt["features_sha256"],
        "source_export_receipt_sha256": sha256(args.candidate_dir / "receipt.json"),
        "ordered_rows_sha256": export_receipt["ordered_rows_sha256"],
        "native_library_sha256": NATIVE_SHA256,
        "tileiras_sha256": TILEIRAS_SHA256,
        "source_pca_sha256": pca_sha,
        "initial_head_sha256": initial_head_sha,
        "initial_classifier_sha256": initial_classifier_sha,
        "schedule_sha256": schedule_sha,
        "first_input_batch_sha256": first_input_batch_sha256,
        "rank_to_arcface_head_gradient_ratio": rank_to_arcface_head_gradient_ratio,
        "query_image_ids_sha256": hashlib.sha256(ids[held_rows].tobytes()).hexdigest(),
        "training_objective": (
            "ArcFace margin 0.3 scale 64 plus 8.0 full-fit SmoothAP with live-head source bank"
            if args.live_head_bank
            else "ArcFace margin 0.3 scale 64 plus 8.0 full-fit float SmoothAP member bank"
            if args.member_bank
            else "ArcFace margin 0.3 scale 64 plus 8.0 deployed-code SmoothAP"
            if args.arm is CompactTrainingArm.PACKED_RANK
            else "ArcFace margin 0.3 scale 64"
            if args.arm is CompactTrainingArm.ARCFACE
            else "ArcFace margin 0.3 scale 64 plus 8.0 float SmoothAP"
        ),
        "optimizer": (
            "AdamW weight_decay 0.05; vision lr 1e-5; head/classifier lr 1e-4; grad clip 1.0"
        ),
        "precision": (
            "fp32 parameters, bf16 vision autocast, fp32 objective, no loss scaling"
            if args.train_vision_dtype == "bf16"
            else "fp32 parameters, fp16 vision autocast, fp32 objective, GradScaler"
        ),
        "train_vision_dtype": args.train_vision_dtype,
        "augmentation": (
            "random resized crop 256 scale 0.8..1.0; random horizontal flip; pinned processor"
        ),
        "training_wall_seconds": training_wall,
        "training_wall_including_member_bank_init_seconds": training_wall + bank_init_seconds,
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "step_seconds": step_seconds,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "first_arcface_loss": control_losses[0] if control_losses else None,
        "last_arcface_loss": control_losses[-1] if control_losses else None,
        "first_rank_loss": rank_losses[0] if rank_losses else None,
        "last_rank_loss": rank_losses[-1] if rank_losses else None,
        "rank_coefficient": SIGLIP2_RANK_COEFFICIENT,
        "grad_scaler_initial_scale": (
            GRAD_SCALER_INITIAL_SCALE if args.train_vision_dtype == "fp16" else 1.0
        ),
        "training_peak_cuda_allocated_bytes": training_peak_cuda,
        "export_seconds": export_seconds,
        "score_seconds": score_seconds,
        "total_wall_seconds_before_receipt": time.perf_counter() - program_started,
        "quality": quality,
        "checkpoint_sha256": sha256(checkpoint_path),
        "train_embeddings_sha256": (
            sha256(args.output_dir / "train_embeddings.npy") if args.evaluate else None
        ),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "pillow": PIL.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
        },
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "arm": arm_name,
                "training_seconds": training_wall,
                "quality_r1": quality["recall_at_1"] if quality else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
