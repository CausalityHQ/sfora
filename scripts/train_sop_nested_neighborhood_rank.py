#!/usr/bin/env python3
"""Train nested-neighborhood rank learning from authenticated local inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
import random
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sfora.atomic_publication import publish_bytes_noreplace, publish_large_writer_noreplace
from sfora.nested_neighborhood_rank import (
    NestedRankConfig,
    NestedRankHead,
    asymmetric_neighborhood_loss,
    nested_proxy_anchor_loss,
)
from sfora.nested_rank_protocol import (
    class_disjoint_fold,
    identity_balanced_schedule,
    ordered_training_records_sha256,
    shared_optimization_rows,
)

_ARMS = (
    "proxy-anchor",
    "neighborhood",
    "combined",
    "proxy-anchor-768",
    "s2sd-768-to-128",
)
_SPLIT_SEEDS = (17, 1729, 65537)
_GRAD_SCALER_TYPE = importlib.import_module("torch.amp").GradScaler
_SNAPSHOT_ARRAYS = {
    "train_embeddings",
    "train_labels",
    "train_image_ids",
    "train_relative_paths",
}
_SNAPSHOT_METADATA = {
    "checkpoint_sha256",
    "embedding_dimension",
    "excluded_test_array_sha256",
    "model_identifier",
    "model_revision",
    "ordered_train_record_sha256",
    "schema",
    "source_archive_sha256",
    "train_array_sha256",
    "train_classes",
    "train_rows",
    "transform",
}
_TEST_ARRAYS = {
    "test_embeddings",
    "test_labels",
    "test_image_ids",
    "test_relative_paths",
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _SopRecord(Protocol):
    image_id: int
    label: int
    relative_path: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class TrainSnapshot:
    """Authenticated official-training-only teacher rows and identities."""

    embeddings: NDArray[np.float32]
    labels: NDArray[np.int64]
    image_ids: NDArray[np.int64]
    relative_paths: NDArray[np.str_]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class NeighborhoodPreflight:
    """Entropy evidence and the preregistered neighborhood temperature."""

    temperature: float | None
    redundant: bool
    batch_count: int
    metrics: dict[float, dict[str, float | int]]


class _UniqueStore(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, values)


class _UniqueTrue(argparse.Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        *,
        default: object = False,
        required: bool = False,
        help: str | None = None,
        **kwargs: object,
    ) -> None:
        if kwargs:
            raise TypeError("unsupported boolean action configuration")
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        if getattr(namespace, self.dest, False):
            parser.error(f"{option_string} may be supplied only once")
        setattr(namespace, self.dest, True)


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _hex_digest(value: str, length: int) -> str:
    if len(value) != length or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError(f"value must be a lowercase {length}-digit digest")
    return value


def _sha256_arg(value: str) -> str:
    return _hex_digest(value, 64)


def _commit_arg(value: str) -> str:
    return _hex_digest(value, 40)


def _split_seed(value: str) -> int:
    parsed = int(value)
    if parsed not in _SPLIT_SEEDS:
        raise argparse.ArgumentTypeError("split seed is not preregistered")
    return parsed


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only training boundary."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.set_defaults(
        dataset_root=None,
        train_snapshot=None,
        train_snapshot_sha256=None,
        unicom_checkout=None,
        unicom_revision=None,
        unicom_checkpoint=None,
        unicom_checkpoint_sha256=None,
        source_commit=None,
        arm=None,
        split_seed=None,
        output_dir=None,
    )
    parser.add_argument("--dataset-root", required=True, type=_absolute_path, action=_UniqueStore)
    parser.add_argument("--train-snapshot", required=True, type=_absolute_path, action=_UniqueStore)
    parser.add_argument(
        "--train-snapshot-sha256", required=True, type=_sha256_arg, action=_UniqueStore
    )
    parser.add_argument(
        "--unicom-checkout", required=True, type=_absolute_path, action=_UniqueStore
    )
    parser.add_argument("--unicom-revision", required=True, type=_commit_arg, action=_UniqueStore)
    parser.add_argument(
        "--unicom-checkpoint", required=True, type=_absolute_path, action=_UniqueStore
    )
    parser.add_argument(
        "--unicom-checkpoint-sha256", required=True, type=_sha256_arg, action=_UniqueStore
    )
    parser.add_argument("--source-commit", required=True, type=_commit_arg, action=_UniqueStore)
    parser.add_argument("--arm", required=True, choices=_ARMS, action=_UniqueStore)
    parser.add_argument("--split-seed", required=True, type=_split_seed, action=_UniqueStore)
    parser.add_argument("--output-dir", required=True, type=_absolute_path, action=_UniqueStore)
    parser.add_argument("--execute-nnrl", required=True, action=_UniqueTrue, default=False)
    return parser.parse_args(arguments)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_field(value: object, length: int) -> bool:
    return (
        type(value) is str and len(value) == length and not (set(value) - set("0123456789abcdef"))
    )


def load_train_snapshot(path: Path, expected_sha256: str) -> TrainSnapshot:
    """Authenticate and load a train-only teacher snapshot without test members."""

    if (
        not isinstance(path, Path)
        or path.is_symlink()
        or not path.is_file()
        or not _digest_field(expected_sha256, 64)
    ):
        raise ValueError("train snapshot authority differs")
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ValueError("train snapshot digest differs")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != _SNAPSHOT_ARRAYS | {"metadata_json"}:
            raise ValueError("train snapshot schema differs")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in _SNAPSHOT_ARRAYS}
    if (
        type(metadata) is not dict
        or set(metadata) != _SNAPSHOT_METADATA
        or metadata.get("schema") != "sfora-nnrl-sop-train-snapshot-v1"
        or type(metadata.get("train_array_sha256")) is not dict
        or set(cast(dict[str, object], metadata["train_array_sha256"])) != _SNAPSHOT_ARRAYS
        or type(metadata.get("excluded_test_array_sha256")) is not dict
        or set(cast(dict[str, object], metadata["excluded_test_array_sha256"])) != _TEST_ARRAYS
    ):
        raise ValueError("train snapshot metadata differs")
    train_rows = metadata.get("train_rows")
    train_classes = metadata.get("train_classes")
    dimensions = metadata.get("embedding_dimension")
    digests = cast(dict[str, object], metadata["train_array_sha256"])
    if any(
        not _digest_field(digests[name], 64)
        or _sha256(arrays[name].tobytes(order="C")) != digests[name]
        for name in _SNAPSHOT_ARRAYS
    ):
        raise ValueError("train array digest differs")
    embeddings = arrays["train_embeddings"]
    labels = arrays["train_labels"]
    image_ids = arrays["train_image_ids"]
    relative_paths = arrays["train_relative_paths"]
    if (
        type(train_rows) is not int
        or type(train_classes) is not int
        or type(dimensions) is not int
        or train_rows <= 0
        or train_classes <= 1
        or dimensions <= 0
        or embeddings.dtype != np.float32
        or embeddings.shape != (train_rows, dimensions)
        or not embeddings.flags.c_contiguous
        or not np.isfinite(embeddings).all()
        or np.any(np.linalg.norm(embeddings.astype(np.float64), axis=1) == 0.0)
        or labels.dtype != np.int64
        or labels.shape != (train_rows,)
        or len(set(labels.tolist())) != train_classes
        or image_ids.dtype != np.int64
        or image_ids.shape != (train_rows,)
        or len(set(image_ids.tolist())) != train_rows
        or relative_paths.dtype.kind != "U"
        or relative_paths.shape != (train_rows,)
        or len(set(relative_paths.tolist())) != train_rows
    ):
        raise ValueError("train snapshot array authority differs")
    for key, length in (
        ("source_archive_sha256", 64),
        ("model_revision", 40),
        ("checkpoint_sha256", 64),
        ("ordered_train_record_sha256", 64),
    ):
        if not _digest_field(metadata.get(key), length):
            raise ValueError("train snapshot metadata differs")
    if any(
        not _digest_field(value, 64)
        for value in cast(dict[str, object], metadata["excluded_test_array_sha256"]).values()
    ):
        raise ValueError("train snapshot metadata differs")
    return TrainSnapshot(
        embeddings=cast(NDArray[np.float32], embeddings),
        labels=cast(NDArray[np.int64], labels),
        image_ids=cast(NDArray[np.int64], image_ids),
        relative_paths=cast(NDArray[np.str_], relative_paths),
        metadata=cast(dict[str, object], metadata),
    )


def bind_training_records(
    snapshot: TrainSnapshot,
    image_ids: NDArray[np.int64],
    labels: NDArray[np.int64],
    relative_paths: tuple[str, ...],
) -> dict[int, int]:
    """Bind an ordered official-training parser to the teacher snapshot."""

    if (
        not isinstance(snapshot, TrainSnapshot)
        or type(image_ids) is not np.ndarray
        or image_ids.dtype != np.int64
        or image_ids.shape != snapshot.image_ids.shape
        or type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.shape != snapshot.labels.shape
        or type(relative_paths) is not tuple
        or len(relative_paths) != snapshot.labels.size
        or any(type(path) is not str or not path for path in relative_paths)
        or not np.array_equal(image_ids, snapshot.image_ids)
        or not np.array_equal(labels, snapshot.labels)
        or relative_paths != tuple(str(path) for path in snapshot.relative_paths)
        or ordered_training_records_sha256(image_ids, labels, relative_paths)
        != snapshot.metadata["ordered_train_record_sha256"]
    ):
        raise ValueError("NNRL training record binding differs")
    return {int(image_id): row for row, image_id in enumerate(image_ids)}


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status_porcelain(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def validate_execution_authority(
    arguments: argparse.Namespace, snapshot: TrainSnapshot
) -> dict[str, object]:
    """Bind local filesystem objects to the explicit source/model authority."""

    if not isinstance(snapshot, TrainSnapshot):
        raise ValueError("NNRL execution authority differs")
    dataset_root = arguments.dataset_root
    checkout = arguments.unicom_checkout
    checkpoint = arguments.unicom_checkpoint
    output = arguments.output_dir
    if (
        not isinstance(dataset_root, Path)
        or dataset_root.is_symlink()
        or not dataset_root.is_dir()
        or not isinstance(checkout, Path)
        or checkout.is_symlink()
        or not checkout.is_dir()
        or not isinstance(checkpoint, Path)
        or checkpoint.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.name != "FP16-ViT-B-16.pt"
        or not isinstance(output, Path)
        or output.is_symlink()
        or output.exists()
        or output.parent.is_symlink()
        or not output.parent.is_dir()
    ):
        if isinstance(output, Path) and (output.exists() or output.is_symlink()):
            raise FileExistsError(output)
        raise ValueError("NNRL execution path authority differs")
    checkpoint_payload = checkpoint.read_bytes()
    if _sha256(checkpoint_payload) != arguments.unicom_checkpoint_sha256:
        raise ValueError("NNRL student checkpoint digest differs")
    if (
        _git_head(_REPOSITORY_ROOT) != arguments.source_commit
        or _git_head(checkout) != arguments.unicom_revision
        or _git_status_porcelain(_REPOSITORY_ROOT)
        or _git_status_porcelain(checkout)
        or snapshot.metadata["model_revision"] != arguments.unicom_revision
    ):
        raise ValueError("NNRL source revision differs")
    return {
        "source_commit": arguments.source_commit,
        "unicom_revision": arguments.unicom_revision,
        "student_checkpoint_sha256": arguments.unicom_checkpoint_sha256,
        "teacher_checkpoint_sha256": snapshot.metadata["checkpoint_sha256"],
        "train_snapshot_sha256": arguments.train_snapshot_sha256,
    }


def preflight_neighborhood_temperature(
    teacher_rows: NDArray[np.float32],
    labels: NDArray[np.int64],
    sample_ids: NDArray[np.int64],
    batches: tuple[tuple[int, ...], ...],
    *,
    optimization_rows: tuple[int, ...],
    temperatures: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> NeighborhoodPreflight:
    """Select the smallest temperature with a nondegenerate off-class target."""

    if (
        type(teacher_rows) is not np.ndarray
        or teacher_rows.dtype != np.float32
        or teacher_rows.ndim != 2
        or teacher_rows.shape[0] == 0
        or not np.isfinite(teacher_rows).all()
        or np.any(np.linalg.norm(teacher_rows.astype(np.float64), axis=1) == 0.0)
        or type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.shape != (teacher_rows.shape[0],)
        or type(sample_ids) is not np.ndarray
        or sample_ids.dtype != np.int64
        or sample_ids.shape != labels.shape
        or type(temperatures) is not tuple
        or temperatures != tuple(sorted(set(temperatures)))
        or any(
            type(value) is not float or not np.isfinite(value) or value <= 0.0
            for value in temperatures
        )
    ):
        raise ValueError("neighborhood preflight authority differs")
    if type(batches) is not tuple or len(batches) != 1_000:
        raise ValueError("neighborhood preflight schedule differs")
    if (
        type(optimization_rows) is not tuple
        or not optimization_rows
        or len(set(optimization_rows)) != len(optimization_rows)
        or any(
            type(row) is not int or not 0 <= row < teacher_rows.shape[0]
            for row in optimization_rows
        )
    ):
        raise ValueError("neighborhood preflight schedule differs")
    allowed_rows = set(optimization_rows)
    supports: dict[float, list[float]] = {value: [] for value in temperatures}
    minimum_valid = teacher_rows.shape[0]
    normalized = teacher_rows.astype(np.float64)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
    for batch in batches:
        if (
            type(batch) is not tuple
            or len(batch) < 2
            or any(type(row) is not int or not 0 <= row < teacher_rows.shape[0] for row in batch)
            or not set(batch) <= allowed_rows
        ):
            raise ValueError("neighborhood preflight schedule differs")
        selected = np.asarray(batch, dtype=np.int64)
        selected_labels = labels[selected]
        selected_ids = sample_ids[selected]
        valid = (selected_labels[:, None] != selected_labels[None, :]) & (
            selected_ids[:, None] != selected_ids[None, :]
        )
        valid_counts = valid.sum(axis=1)
        if np.any(valid_counts == 0):
            raise ValueError("neighborhood preflight batch inventory differs")
        minimum_valid = min(minimum_valid, int(valid_counts.min()))
        similarity = normalized[selected] @ normalized[selected].T
        for temperature in temperatures:
            logits = np.where(valid, similarity / temperature, -np.inf)
            maximum = np.max(logits, axis=1, keepdims=True)
            weights = np.where(valid, np.exp(logits - maximum), 0.0)
            probability = weights / weights.sum(axis=1, keepdims=True)
            log_probability = np.zeros_like(probability)
            np.log(probability, out=log_probability, where=probability > 0.0)
            entropy = -np.sum(probability * log_probability, axis=1)
            supports[temperature].extend(np.exp(entropy).tolist())
    metrics: dict[float, dict[str, float | int]] = {}
    selected_temperature = None
    for temperature in temperatures:
        values = np.sort(np.asarray(supports[temperature], dtype=np.float64))
        p05 = float(values[max(0, int(np.ceil(0.05 * values.size)) - 1)])
        median = float(np.median(values))
        metrics[temperature] = {
            "median_effective_support": median,
            "p05_effective_support": p05,
            "minimum_valid_keys": minimum_valid,
            "query_count": int(values.size),
        }
        if selected_temperature is None and median >= 4.0 and p05 >= 2.0:
            selected_temperature = temperature
    return NeighborhoodPreflight(
        temperature=selected_temperature,
        redundant=selected_temperature is None,
        batch_count=len(batches),
        metrics=metrics,
    )


def arm_loss(
    arm: str,
    embeddings: dict[int, torch.Tensor],
    dense: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    teacher_rows: torch.Tensor,
    raw_proxies: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Evaluate one preregistered phase-one objective without hidden weights."""

    if arm not in _ARMS:
        raise ValueError("NNRL arm differs")
    proxy_values = raw_proxies + 0.0
    if arm == "proxy-anchor-768":
        if tuple(embeddings) != (768,):
            raise ValueError("PA-768 head authority differs")
        return nested_proxy_anchor_loss({768: embeddings[768]}, labels, proxy_values)
    if arm == "neighborhood":
        return asymmetric_neighborhood_loss(
            embeddings[128],
            embeddings[128],
            teacher_rows,
            labels,
            sample_ids,
            temperature=temperature,
        )
    proxy = nested_proxy_anchor_loss(
        embeddings,
        labels,
        proxy_values,
        width_weights={32: 0.25, 128: 1.0},
    )
    if arm == "proxy-anchor":
        return proxy
    if arm == "s2sd-768-to-128":
        return proxy + asymmetric_neighborhood_loss(
            embeddings[128],
            embeddings[128],
            dense.detach(),
            labels,
            sample_ids,
            temperature=temperature,
        )
    neighborhood = asymmetric_neighborhood_loss(
        embeddings[128],
        embeddings[128],
        teacher_rows,
        labels,
        sample_ids,
        temperature=temperature,
    )
    return proxy + neighborhood


def _parameter_groups(
    module: nn.Module, *, prefix: str, learning_rate: float
) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    normalization = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
    )
    seen: set[int] = set()
    for child in module.modules():
        for name, parameter in child.named_parameters(recurse=False):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            target = (
                no_decay
                if name == "bias" or parameter.ndim < 2 or isinstance(child, normalization)
                else decay
            )
            target.append(parameter)
    if not seen or len(seen) != len(tuple(module.parameters())):
        raise ValueError("NNRL optimizer parameter inventory differs")
    groups: list[dict[str, object]] = []
    if decay:
        groups.append(
            {
                "name": f"{prefix}-decay",
                "params": decay,
                "lr": learning_rate,
                "weight_decay": 1e-4,
            }
        )
    if no_decay:
        groups.append(
            {
                "name": f"{prefix}-no-decay",
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
            }
        )
    return groups


def build_optimizer(
    encoder: nn.Module,
    head: nn.Module,
    raw_proxies: nn.Parameter,
    *,
    fused: bool,
) -> torch.optim.AdamW:
    """Build the fixed two-rate AdamW groups with explicit decay exclusions."""

    if (
        not isinstance(encoder, nn.Module)
        or not isinstance(head, nn.Module)
        or type(raw_proxies) is not nn.Parameter
        or type(fused) is not bool
    ):
        raise ValueError("NNRL optimizer authority differs")
    groups = [
        *_parameter_groups(encoder, prefix="encoder", learning_rate=1e-5),
        *_parameter_groups(head, prefix="head", learning_rate=1e-3),
        {
            "name": "proxies",
            "params": [raw_proxies],
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        groups,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fused=fused,
    )


def run_training_epoch(
    encoder: nn.Module,
    head: nn.Module,
    raw_proxies: nn.Parameter,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
    temperature: float,
    device: torch.device,
    fp16: bool,
    expected_steps: int,
    expected_batch_size: int = 128,
    scaler: Any | None = None,
) -> dict[str, float | int]:
    """Execute one fixed-count phase-one epoch with one encoder call per batch."""

    if (
        not isinstance(encoder, nn.Module)
        or not isinstance(head, nn.Module)
        or type(raw_proxies) is not nn.Parameter
        or not isinstance(optimizer, torch.optim.Optimizer)
        or arm not in _ARMS
        or type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or type(device) is not torch.device
        or type(fp16) is not bool
        or (fp16 and device.type == "cuda" and scaler is None)
        or (scaler is not None and not isinstance(scaler, _GRAD_SCALER_TYPE))
        or type(expected_steps) is not int
        or expected_steps <= 0
        or type(expected_batch_size) is not int
        or expected_batch_size <= 1
    ):
        raise ValueError("NNRL training authority differs")
    if isinstance(batches, Sized) and len(batches) != expected_steps:
        raise ValueError("NNRL training step inventory differs")
    encoder.train()
    for module in encoder.modules():
        if isinstance(
            module,
            (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm),
        ):
            module.eval()
    head.train()
    losses: list[float] = []
    attempted_steps = 0
    successful_updates = 0
    for images, labels, sample_ids, teacher_rows in batches:
        if len(losses) == expected_steps:
            raise ValueError("NNRL training step inventory differs")
        if (
            type(images) is not torch.Tensor
            or images.shape[0] != expected_batch_size
            or type(labels) is not torch.Tensor
            or labels.dtype != torch.int64
            or labels.shape != (expected_batch_size,)
            or type(sample_ids) is not torch.Tensor
            or sample_ids.dtype != torch.int64
            or sample_ids.shape != labels.shape
            or type(teacher_rows) is not torch.Tensor
            or teacher_rows.dtype != torch.float32
            or teacher_rows.ndim != 2
            or teacher_rows.shape[0] != expected_batch_size
        ):
            raise ValueError("NNRL training batch differs")
        images = images.to(device)
        labels = labels.to(device)
        sample_ids = sample_ids.to(device)
        teacher_rows = teacher_rows.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16,
        ):
            encoded = encoder(images)
        dense = torch.nn.functional.normalize(encoded.float(), dim=1)
        embeddings = head(dense)
        loss = arm_loss(
            arm,
            embeddings,
            dense,
            labels,
            sample_ids,
            teacher_rows,
            raw_proxies,
            temperature,
        )
        if not torch.isfinite(loss):
            raise ValueError("NNRL training loss is nonfinite")
        attempted_steps += 1
        if scaler is None:
            loss.backward()  # type: ignore[no-untyped-call]
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ):
            raise ValueError("NNRL training gradient is nonfinite")
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        if any(
            not torch.isfinite(parameter).all()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ):
            raise ValueError("NNRL training parameter is nonfinite")
        successful_updates += 1
        losses.append(float(loss.detach()))
    if len(losses) != expected_steps:
        raise ValueError("NNRL training step inventory differs")
    return {
        "steps": successful_updates,
        "attempted_steps": attempted_steps,
        "skipped_updates": attempted_steps - successful_updates,
        "mean_loss": math.fsum(losses) / len(losses),
    }


def build_grad_scaler(device: torch.device, *, fp16: bool) -> Any:
    """Build the fixed-scale FP16 guard validated by the real UNICOM canary."""

    if type(device) is not torch.device or type(fp16) is not bool:
        raise ValueError("NNRL gradient scaler authority differs")
    return _GRAD_SCALER_TYPE(
        "cuda",
        enabled=fp16 and device.type == "cuda",
        init_scale=1024.0,
        growth_interval=2**31 - 1,
    )


def resolved_training_recipe(
    *,
    arm: str,
    split_seed: int,
    temperature: float,
    steps_per_epoch: int,
    device: torch.device,
) -> dict[str, object]:
    """Return the exact numeric recipe and runtime bound to a training result."""

    if (
        arm not in _ARMS
        or split_seed not in _SPLIT_SEEDS
        or type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or type(steps_per_epoch) is not int
        or steps_per_epoch <= 0
        or type(device) is not torch.device
        or device.type != "cuda"
    ):
        raise ValueError("NNRL resolved recipe authority differs")
    output_dimensions = [768] if arm == "proxy-anchor-768" else [32, 128]
    capability = torch.cuda.get_device_capability(device)
    cudnn_version = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
    return {
        "schema": "sfora-nnrl-sop-training-recipe-v1",
        "objective": {
            "arm": arm,
            "temperature": temperature,
            "output_dimensions": output_dimensions,
        },
        "schedule": {
            "split_seed": split_seed,
            "epochs": 10,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": 128,
        },
        "optimizer": {
            "name": "AdamW",
            "encoder_learning_rate": 1e-5,
            "head_learning_rate": 1e-3,
            "proxy_learning_rate": 1e-3,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "decay": 1e-4,
        },
        "precision": {
            "autocast": "float16",
            "gradient_scaler_initial_scale": 1024.0,
            "gradient_scaler_growth_interval": 2**31 - 1,
            "fail_on_nonfinite": True,
        },
        "data": {
            "workers": 8,
            "pin_memory": True,
            "frozen_batch_norm": True,
            "transform": (
                "resize-256-bicubic/random-crop-224/random-horizontal-flip/unicom-normalize"
            ),
        },
        "runtime": {
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": cudnn_version,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": [int(capability[0]), int(capability[1])],
        },
    }


def run_phase_one(
    encoder: nn.Module,
    head: nn.Module,
    raw_proxies: nn.Parameter,
    epochs: tuple[Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], ...],
    *,
    arm: str,
    temperature: float,
    device: torch.device,
    fp16: bool,
    fused: bool,
    expected_batch_size: int = 128,
) -> list[dict[str, object]]:
    """Run the fixed ten-epoch anchored-representation phase."""

    if type(epochs) is not tuple or len(epochs) != 10:
        raise ValueError("NNRL phase-one epoch inventory differs")
    optimizer = build_optimizer(encoder, head, raw_proxies, fused=fused)
    scaler = build_grad_scaler(device, fp16=fp16)
    history: list[dict[str, object]] = []
    expected_steps = None
    for epoch_index, batches in enumerate(epochs, start=1):
        if not isinstance(batches, Sized) or len(batches) <= 0:
            raise ValueError("NNRL phase-one step inventory differs")
        if expected_steps is None:
            expected_steps = len(batches)
        elif len(batches) != expected_steps:
            raise ValueError("NNRL phase-one step inventory differs")
        row = run_training_epoch(
            encoder,
            head,
            raw_proxies,
            batches,
            optimizer,
            arm=arm,
            temperature=temperature,
            device=device,
            fp16=fp16,
            expected_steps=expected_steps,
            expected_batch_size=expected_batch_size,
            scaler=scaler if scaler.is_enabled() else None,
        )
        history.append({"epoch": epoch_index, **row})
    return history


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha256_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise RuntimeError("NNRL published artifact is truncated")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def publish_training_artifacts(
    output: Path,
    state: dict[str, object],
    *,
    authority: dict[str, object],
    arm: str,
    split_seed: int,
    temperature: float,
    optimization_rows: int,
    optimization_classes: int,
    history: list[dict[str, object]],
) -> dict[str, object]:
    """Publish a checkpoint, receipt, and terminal result without replacement."""

    if (
        not isinstance(output, Path)
        or output.exists()
        or output.is_symlink()
        or output.parent.is_symlink()
        or not output.parent.is_dir()
    ):
        if isinstance(output, Path) and (output.exists() or output.is_symlink()):
            raise FileExistsError(output)
        raise ValueError("NNRL publication path differs")
    if (
        type(state) is not dict
        or set(state) != {"encoder", "head", "raw_proxies"}
        or type(authority) is not dict
        or arm not in _ARMS
        or split_seed not in _SPLIT_SEEDS
        or type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or type(optimization_rows) is not int
        or optimization_rows <= 0
        or type(optimization_classes) is not int
        or optimization_classes <= 1
        or type(history) is not list
        or not history
    ):
        raise ValueError("NNRL publication authority differs")
    output.mkdir(mode=0o700)
    model_path = output / "model.pt"

    def write_model(descriptor: int) -> None:
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            torch.save(state, stream)
            stream.flush()

    def validate_model(descriptor: int, size: int) -> None:
        if size <= 0:
            raise ValueError("NNRL checkpoint is empty")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            restored = torch.load(stream, map_location="cpu", weights_only=True)
        if type(restored) is not dict or set(restored) != set(state):
            raise ValueError("NNRL checkpoint schema differs")

    with publish_large_writer_noreplace(
        model_path, write_model, validator=validate_model
    ) as published_model:
        model_sha256 = _sha256_descriptor(published_model.descriptor, published_model.size)
        model_bytes = published_model.size
    receipt: dict[str, object] = {
        "schema": "sfora-nnrl-sop-training-run-receipt-v1",
        "claim_eligible": False,
        "arm": arm,
        "split_seed": split_seed,
        "temperature": temperature,
        "optimization_rows": optimization_rows,
        "optimization_classes": optimization_classes,
        "epochs": len(history),
        "history": history,
        "authority": authority,
        "model_artifact": {
            "path": "model.pt",
            "sha256": model_sha256,
            "bytes": model_bytes,
        },
    }
    receipt_payload = _canonical_json_bytes(receipt)
    with publish_bytes_noreplace(
        output / "run-receipt.json",
        receipt_payload,
        validator=lambda payload: (
            None
            if payload == receipt_payload and json.loads(payload) == receipt
            else (_ for _ in ()).throw(ValueError("NNRL receipt differs"))
        ),
    ):
        pass
    result = {
        "schema": "sfora-nnrl-sop-training-result-v1",
        "claim_eligible": False,
        "status": "COMPLETE",
        **{key: value for key, value in receipt.items() if key not in {"schema", "claim_eligible"}},
        "run_receipt": {
            "path": "run-receipt.json",
            "sha256": _sha256(receipt_payload),
            "bytes": len(receipt_payload),
        },
    }
    result_payload = _canonical_json_bytes(result)
    with publish_bytes_noreplace(
        output / "RESULT_COMPLETE.json",
        result_payload,
        validator=lambda payload: (
            None
            if payload == result_payload and json.loads(payload) == result
            else (_ for _ in ()).throw(ValueError("NNRL result differs"))
        ),
    ):
        pass
    return result


class _SopOptimizationDataset(
    torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self,
        *,
        image_paths: tuple[Path, ...],
        labels: NDArray[np.int64],
        image_ids: NDArray[np.int64],
        teacher_rows: NDArray[np.float32],
        optimization_rows: tuple[int, ...],
        label_indexes: dict[int, int],
        transform: Callable[[object], object],
    ) -> None:
        self._image_paths = image_paths
        self._labels = labels
        self._image_ids = image_ids
        self._teacher_rows = teacher_rows
        self._optimization_rows = set(optimization_rows)
        self._label_indexes = label_indexes
        self._transform = transform

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index not in self._optimization_rows:
            raise ValueError("NNRL dataset accessed a validation row")
        from PIL import Image

        path = self._image_paths[index]
        with Image.open(path) as image:
            tensor = self._transform(image.convert("RGB"))
        if type(tensor) is not torch.Tensor:
            raise ValueError("NNRL transform output differs")
        return (
            tensor,
            torch.tensor(self._label_indexes[int(self._labels[index])], dtype=torch.int64),
            torch.tensor(int(self._image_ids[index]), dtype=torch.int64),
            torch.from_numpy(self._teacher_rows[index].copy()),
        )


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _build_train_transform() -> Callable[[object], object]:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    transform = transforms.Compose(
        (
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        )
    )
    return cast(Callable[[object], object], transform)


def _load_student_model(checkout: Path, checkpoint: Path) -> nn.Module:
    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    module_file = getattr(unicom, "__file__", None)
    if (
        type(module_file) is not str
        or Path(module_file).resolve().parent != package_root / "unicom"
    ):
        raise ValueError("NNRL imported UNICOM package differs")
    model, _evaluation_transform = unicom.load("ViT-B/16", download_root=str(checkpoint.parent))
    if not isinstance(model, nn.Module):
        raise ValueError("NNRL student model differs")
    return model


def _load_sop_training_records(dataset_root: Path) -> tuple[_SopRecord, ...]:
    exporter = importlib.import_module("export_unicom_sop_embeddings")
    training = tuple(exporter._parse_split(dataset_root, "train"))
    if len(training) != 59_551:
        raise ValueError("NNRL SOP training record count differs")
    return cast(tuple[_SopRecord, ...], training)


def _source_file_sha256() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        (_REPOSITORY_ROOT / "scripts/export_unicom_sop_embeddings.py").resolve(),
        (_REPOSITORY_ROOT / "src/sfora/atomic_publication.py").resolve(),
        (_REPOSITORY_ROOT / "src/sfora/nested_neighborhood_rank.py").resolve(),
        (_REPOSITORY_ROOT / "src/sfora/nested_rank_protocol.py").resolve(),
        (_REPOSITORY_ROOT / "src/sfora/representation_ceiling.py").resolve(),
    )
    return {str(path.relative_to(_REPOSITORY_ROOT)): _sha256(path.read_bytes()) for path in paths}


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def run_experiment(arguments: argparse.Namespace) -> dict[str, object]:
    """Run one authenticated ten-epoch SOP NNRL arm on CUDA."""

    snapshot = load_train_snapshot(arguments.train_snapshot, arguments.train_snapshot_sha256)
    authority = validate_execution_authority(arguments, snapshot)
    authority["source_files"] = _source_file_sha256()
    records = _load_sop_training_records(arguments.dataset_root)
    image_ids = np.asarray([record.image_id for record in records], dtype=np.int64)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    relative_paths = tuple(record.relative_path for record in records)
    bind_training_records(snapshot, image_ids, labels, relative_paths)
    image_paths = tuple(record.image_path for record in records)
    sample_ids = tuple(int(value) for value in image_ids)

    preflight_rows = shared_optimization_rows(sample_ids, labels, seeds=_SPLIT_SEEDS)
    preflight_schedule = identity_balanced_schedule(
        labels,
        snapshot.embeddings,
        preflight_rows,
        seed=17,
        steps=1_000,
    )
    preflight = preflight_neighborhood_temperature(
        snapshot.embeddings,
        labels,
        image_ids,
        preflight_schedule,
        optimization_rows=preflight_rows,
    )
    if preflight.redundant and arguments.arm in {
        "neighborhood",
        "combined",
        "s2sd-768-to-128",
    }:
        raise ValueError("NNRL neighborhood objective is redundant")
    temperature = 0.1 if preflight.temperature is None else preflight.temperature

    fold = class_disjoint_fold(sample_ids, labels, seed=arguments.split_seed)
    steps_per_epoch = max(1, len(fold.optimization) // 128)
    schedule = identity_balanced_schedule(
        labels,
        snapshot.embeddings,
        fold.optimization,
        seed=arguments.split_seed,
        steps=10 * steps_per_epoch,
    )
    label_values = tuple(sorted({int(labels[row]) for row in fold.optimization}))
    label_indexes = {label: index for index, label in enumerate(label_values)}
    dataset = _SopOptimizationDataset(
        image_paths=image_paths,
        labels=labels,
        image_ids=image_ids,
        teacher_rows=snapshot.embeddings,
        optimization_rows=fold.optimization,
        label_indexes=label_indexes,
        transform=_build_train_transform(),
    )
    epoch_loaders = []
    for epoch in range(10):
        epoch_batches = schedule[epoch * steps_per_epoch : (epoch + 1) * steps_per_epoch]
        epoch_loaders.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_sampler=[list(batch) for batch in epoch_batches],
                num_workers=8,
                pin_memory=True,
                worker_init_fn=_seed_worker,
                generator=torch.Generator().manual_seed(arguments.split_seed + epoch),
            )
        )

    random.seed(arguments.split_seed)
    np.random.seed(arguments.split_seed % 2**32)
    torch.manual_seed(arguments.split_seed)
    torch.cuda.manual_seed_all(arguments.split_seed)
    device = torch.device("cuda")
    authority["resolved_recipe"] = resolved_training_recipe(
        arm=arguments.arm,
        split_seed=arguments.split_seed,
        temperature=temperature,
        steps_per_epoch=steps_per_epoch,
        device=device,
    )
    encoder = _load_student_model(arguments.unicom_checkout, arguments.unicom_checkpoint).to(device)
    class_count = len(label_values)
    if arguments.arm == "proxy-anchor-768":
        config = NestedRankConfig(
            input_dim=768,
            hidden_dim=1024,
            output_dim=768,
            widths=(768,),
            class_count=class_count,
        )
    else:
        config = NestedRankConfig(input_dim=768, hidden_dim=1024, class_count=class_count)
    head = NestedRankHead(config).to(device)
    raw_proxies = nn.Parameter(torch.empty(class_count, config.output_dim, device=device))
    torch.nn.init.normal_(raw_proxies, std=0.01)
    history = run_phase_one(
        encoder,
        head,
        raw_proxies,
        tuple(epoch_loaders),
        arm=arguments.arm,
        temperature=temperature,
        device=device,
        fp16=True,
        fused=True,
    )
    authority["ordered_train_record_sha256"] = snapshot.metadata["ordered_train_record_sha256"]
    authority["preflight"] = {
        "temperature": preflight.temperature,
        "redundant": preflight.redundant,
        "batch_count": preflight.batch_count,
        "metrics": {str(key): value for key, value in preflight.metrics.items()},
    }
    return publish_training_artifacts(
        arguments.output_dir,
        {
            "encoder": _cpu_state_dict(encoder),
            "head": _cpu_state_dict(head),
            "raw_proxies": raw_proxies.detach().cpu(),
        },
        authority=authority,
        arm=arguments.arm,
        split_seed=arguments.split_seed,
        temperature=temperature,
        optimization_rows=len(fold.optimization),
        optimization_classes=class_count,
        history=history,
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one explicit authenticated NNRL training arm."""

    result = run_experiment(parse_args(arguments))
    print(_canonical_json_bytes(result).decode(), end="")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
