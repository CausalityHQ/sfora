#!/usr/bin/env python3
"""Build strict paired UniCOM imprint replication evidence."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import secrets
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REGISTERED_EPOCHS = (4, 8, 12, 16)
REGISTERED_CELLS = ("random_raw", "imprinted_raw")
TRAINING_PROTOCOL_FIELDS = (
    "protocol",
    "trainer_sha256",
    "unicom_revision",
    "initial_checkpoint_sha256",
    "partition_sha256",
    "seed",
    "epochs",
    "batch_size",
    "workers",
    "learning_rate",
    "classifier_learning_rate",
    "margin",
    "scale",
    "objective",
    "selected_features",
    "holdout_seed",
    "holdout_fraction",
    "eval_every",
    "checkpoint_every",
    "max_steps",
    "bf16",
    "compile",
    "fused",
    "classifier_init",
    "ema_decay",
    "ema_update",
)
REPORT_KEYS = (
    "schema_version",
    "seed",
    "selected_cell",
    "registered_epochs",
    "random_training_protocol",
    "imprinted_training_protocol",
    "random_raw",
    "imprinted_raw",
    "inference_latency",
    "evidence",
)
ARM_KEYS = (
    "checkpoint_sha256_by_epoch",
    "training_history_sha256_by_epoch",
    "epoch_metrics",
    "training_seconds",
    "peak_gpu_mib",
    "checkpoint_storage_bytes",
    "deployment_storage_bytes",
    "measurement_receipt_sha256",
    "profile",
)
MEASUREMENT_KEYS = (
    "schema_version",
    "seed",
    "classifier_init",
    "training_seconds",
    "peak_gpu_mib",
    "step_wall_seconds",
    "fusible_non_backbone_seconds",
    "training_log_sha256",
    "training_protocol_sha256",
    "checkpoint_sha256_by_epoch",
    "training_history_sha256_by_epoch",
)


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


def validate_training_protocol(value: object, *, seed: int, classifier_init: str) -> None:
    if type(value) is not dict or tuple(value) != TRAINING_PROTOCOL_FIELDS:
        raise ValueError("replication training protocol schema differs")
    expected = {
        "protocol": "unicom-inshop-official-single-device-v1",
        "seed": seed,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 1e-5,
        "classifier_learning_rate": 1e-4,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 512,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "eval_every": 4,
        "checkpoint_every": 4,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
        "classifier_init": classifier_init,
        "ema_decay": 0.999,
        "ema_update": "optimizer-step-post-hook-trainable-parameters-only",
    }
    if any(
        type(value[key]) is not type(expected_value) or value[key] != expected_value
        for key, expected_value in expected.items()
    ):
        raise ValueError("replication training protocol differs")
    if (
        not _sha256(value["trainer_sha256"])
        or type(value["unicom_revision"]) is not str
        or len(value["unicom_revision"]) != 40
        or not _sha256(value["initial_checkpoint_sha256"])
        or not _sha256(value["partition_sha256"])
    ):
        raise ValueError("replication training protocol binding differs")


def validate_runtime_bindings(
    protocols: Mapping[str, Mapping[str, object]],
    *,
    trainer_sha256: str,
    unicom_revision: str,
    initial_checkpoint_sha256: str,
    partition_sha256: str,
) -> None:
    expected = {
        "trainer_sha256": trainer_sha256,
        "unicom_revision": unicom_revision,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "partition_sha256": partition_sha256,
    }
    if tuple(protocols) != REGISTERED_CELLS or any(
        protocol.get(key) != value
        for protocol in protocols.values()
        for key, value in expected.items()
    ):
        raise ValueError("replication runtime/partition binding differs")


def _validated_evidence(value: object, *, name: str) -> dict[str, list[object]]:
    if type(value) is not dict or tuple(value) != ("top1_correct", "average_precision"):
        raise ValueError(f"{name} evidence schema differs")
    average_precision = value["average_precision"]
    top1_correct = value["top1_correct"]
    if (
        type(average_precision) is not list
        or type(top1_correct) is not list
        or not average_precision
        or len(average_precision) != len(top1_correct)
        or any(not _finite_float(item) or not 0.0 <= item <= 1.0 for item in average_precision)
        or any(type(item) is not bool for item in top1_correct)
    ):
        raise ValueError(f"{name} evidence differs")
    return value


def _metric_from_evidence(value: dict[str, list[object]]) -> dict[str, float]:
    return {
        "map_at_r": float(np.mean(np.asarray(value["average_precision"], dtype=np.float64))),
        "recall_at_1": float(np.mean(np.asarray(value["top1_correct"], dtype=np.float64))),
    }


def _validate_costs(arm: Mapping[str, Any], *, name: str) -> None:
    if not _finite_float(arm["training_seconds"]) or arm["training_seconds"] <= 0.0:
        raise ValueError(f"{name} training cost differs")
    for key in ("peak_gpu_mib", "checkpoint_storage_bytes", "deployment_storage_bytes"):
        if type(arm[key]) is not int or arm[key] <= 0:
            raise ValueError(f"{name} cost differs")
    if not _sha256(arm["measurement_receipt_sha256"]):
        raise ValueError(f"{name} measurement receipt differs")
    profile = arm["profile"]
    if type(profile) is not dict or tuple(profile) != (
        "step_wall_seconds",
        "fusible_non_backbone_seconds",
    ):
        raise ValueError(f"{name} profile schema differs")
    if (
        not _finite_float(profile["step_wall_seconds"])
        or profile["step_wall_seconds"] <= 0.0
        or not _finite_float(profile["fusible_non_backbone_seconds"])
        or not 0.0 <= profile["fusible_non_backbone_seconds"] <= profile["step_wall_seconds"]
    ):
        raise ValueError(f"{name} profile cost differs")


def validate_replication_pair(report: object) -> None:
    if type(report) is not dict or tuple(report) != REPORT_KEYS:
        raise ValueError("replication pair schema differs")
    if (
        report["schema_version"] != "unicom-ema-imprint-replication-pair-v1"
        or type(report["seed"]) is not int
        or report["seed"] not in range(1, 7)
        or report["selected_cell"] != "imprinted_raw"
        or report["registered_epochs"] != list(REGISTERED_EPOCHS)
    ):
        raise ValueError("replication pair registration differs")
    seed = report["seed"]
    validate_training_protocol(
        report["random_training_protocol"], seed=seed, classifier_init="random"
    )
    validate_training_protocol(
        report["imprinted_training_protocol"], seed=seed, classifier_init="imprinted"
    )
    normalized_random = dict(report["random_training_protocol"])
    normalized_imprinted = dict(report["imprinted_training_protocol"])
    normalized_random.pop("classifier_init")
    normalized_imprinted.pop("classifier_init")
    if normalized_random != normalized_imprinted:
        raise ValueError("paired replication protocols differ")
    evidence = report["evidence"]
    if type(evidence) is not dict or tuple(evidence) != REGISTERED_CELLS:
        raise ValueError("replication evidence schema differs")
    all_hashes: list[str] = []
    all_history_hashes: list[str] = []
    expected_query_count: int | None = None
    for cell in REGISTERED_CELLS:
        arm = report[cell]
        if type(arm) is not dict or tuple(arm) != ARM_KEYS:
            raise ValueError(f"{cell} schema differs")
        for key in ("checkpoint_sha256_by_epoch", "training_history_sha256_by_epoch"):
            values = arm[key]
            if type(values) is not list or len(values) != 4 or any(not _sha256(v) for v in values):
                raise ValueError(f"{cell} checkpoint evidence differs")
            if key == "checkpoint_sha256_by_epoch":
                all_hashes.extend(values)
            else:
                all_history_hashes.extend(values)
        rows = arm["epoch_metrics"]
        cell_evidence = evidence[cell]
        if (
            type(rows) is not list
            or type(cell_evidence) is not list
            or len(rows) != 4
            or len(cell_evidence) != 4
        ):
            raise ValueError(f"{cell} metric evidence differs")
        for epoch, row, raw_evidence in zip(REGISTERED_EPOCHS, rows, cell_evidence, strict=True):
            if (
                type(row) is not dict
                or tuple(row) != ("epoch", "map_at_r", "recall_at_1")
                or row["epoch"] != epoch
            ):
                raise ValueError(f"{cell} metric row order differs")
            entry = _validated_evidence(raw_evidence, name=cell)
            query_count = len(entry["top1_correct"])
            if expected_query_count is None:
                expected_query_count = query_count
            elif query_count != expected_query_count:
                raise ValueError("replication query population differs")
            if row != {"epoch": epoch, **_metric_from_evidence(entry)}:
                raise ValueError(f"{cell} metric differs from query evidence")
        _validate_costs(arm, name=cell)
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("replication checkpoint evidence is reused")
    if len(set(all_history_hashes)) != len(all_history_hashes):
        raise ValueError("replication training history evidence is reused")
    latency = report["inference_latency"]
    if type(latency) is not dict or tuple(latency) != (
        "warmup_repetitions",
        "measured_repetitions",
        "batch_size",
        "milliseconds_per_image",
    ):
        raise ValueError("replication latency schema differs")
    if (
        latency["warmup_repetitions"] != 10
        or latency["measured_repetitions"] != 50
        or latency["batch_size"] != 128
        or not _finite_float(latency["milliseconds_per_image"])
        or latency["milliseconds_per_image"] <= 0.0
    ):
        raise ValueError("replication latency differs")


def build_replication_pair(
    *,
    seed: int,
    rows: Sequence[Mapping[str, Any]],
    random_protocol: Mapping[str, Any],
    imprinted_protocol: Mapping[str, Any],
    random_training_seconds: float,
    imprinted_training_seconds: float,
    random_peak_gpu_mib: int,
    imprinted_peak_gpu_mib: int,
    random_checkpoint_storage_bytes: int,
    imprinted_checkpoint_storage_bytes: int,
    deployment_storage_bytes: int,
    inference_latency_ms_per_image: float,
    random_measurement_receipt_sha256: str,
    imprinted_measurement_receipt_sha256: str,
    random_profile: tuple[float, float],
    imprinted_profile: tuple[float, float],
) -> dict[str, object]:
    expected_order = tuple(
        (cell, epoch) for cell in REGISTERED_CELLS for epoch in REGISTERED_EPOCHS
    )
    if tuple((row.get("cell"), row.get("epoch")) for row in rows) != expected_order:
        raise ValueError("replication row order differs")

    def arm(
        cell: str,
        seconds: float,
        peak: int,
        storage: int,
        measurement_sha256: str,
        profile: tuple[float, float],
    ):
        selected = [row for row in rows if row["cell"] == cell]
        return {
            "checkpoint_sha256_by_epoch": [row["checkpoint_sha256"] for row in selected],
            "training_history_sha256_by_epoch": [
                row["training_history_sha256"] for row in selected
            ],
            "epoch_metrics": [
                {"epoch": row["epoch"], **_metric_from_evidence(row["query_evidence"])}
                for row in selected
            ],
            "training_seconds": seconds,
            "peak_gpu_mib": peak,
            "checkpoint_storage_bytes": storage,
            "deployment_storage_bytes": deployment_storage_bytes,
            "measurement_receipt_sha256": measurement_sha256,
            "profile": {
                "step_wall_seconds": profile[0],
                "fusible_non_backbone_seconds": profile[1],
            },
        }

    report = {
        "schema_version": "unicom-ema-imprint-replication-pair-v1",
        "seed": seed,
        "selected_cell": "imprinted_raw",
        "registered_epochs": list(REGISTERED_EPOCHS),
        "random_training_protocol": dict(random_protocol),
        "imprinted_training_protocol": dict(imprinted_protocol),
        "random_raw": arm(
            "random_raw",
            random_training_seconds,
            random_peak_gpu_mib,
            random_checkpoint_storage_bytes,
            random_measurement_receipt_sha256,
            random_profile,
        ),
        "imprinted_raw": arm(
            "imprinted_raw",
            imprinted_training_seconds,
            imprinted_peak_gpu_mib,
            imprinted_checkpoint_storage_bytes,
            imprinted_measurement_receipt_sha256,
            imprinted_profile,
        ),
        "inference_latency": {
            "warmup_repetitions": 10,
            "measured_repetitions": 50,
            "batch_size": 128,
            "milliseconds_per_image": inference_latency_ms_per_image,
        },
        "evidence": {
            cell: [dict(row["query_evidence"]) for row in rows if row["cell"] == cell]
            for cell in REGISTERED_CELLS
        },
    }
    validate_replication_pair(report)
    return report


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact_bytes(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    size = os.fstat(descriptor).st_size
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise RuntimeError("snapshot payload is truncated")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _snapshot_file(source: Path, destination: Path) -> tuple[str, int]:
    source_descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    destination_descriptor: int | None = None
    try:
        source_info = os.fstat(source_descriptor)
        destination_descriptor = os.open(
            destination, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
        )
        try:
            fcntl.ioctl(destination_descriptor, 0x40049409, source_descriptor)
        except OSError as error:
            if error.errno not in {
                errno.EINVAL,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.EXDEV,
            }:
                raise
            os.ftruncate(destination_descriptor, 0)
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            os.lseek(destination_descriptor, 0, os.SEEK_SET)
            remaining = source_info.st_size
            copy_file_range = getattr(os, "copy_file_range", None)
            while remaining:
                if copy_file_range is not None:
                    copied = copy_file_range(
                        source_descriptor,
                        destination_descriptor,
                        min(16 * 1024 * 1024, remaining),
                    )
                else:
                    chunk = os.read(
                        source_descriptor, min(16 * 1024 * 1024, remaining)
                    )
                    copied = len(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_descriptor, view)
                        if written == 0:
                            raise RuntimeError("snapshot write is truncated") from None
                        view = view[written:]
                if copied == 0:
                    raise RuntimeError("snapshot copy is truncated") from None
                remaining -= copied
        os.fsync(destination_descriptor)
        size = os.fstat(destination_descriptor).st_size
        if size != source_info.st_size:
            raise RuntimeError("snapshot size differs")
        return _sha256_descriptor(destination_descriptor), size
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        os.close(source_descriptor)


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _checkpoint_paths(directory: Path) -> tuple[Path, ...]:
    paths = tuple(directory / f"epoch-{epoch:04d}.pt" for epoch in REGISTERED_EPOCHS)
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("replication checkpoint path differs")
    return paths


def _load_checkpoint(path: Path, *, seed: int, epoch: int, classifier_init: str):
    import torch

    with tempfile.TemporaryDirectory(prefix=".checkpoint-snapshot-", dir=path.parent) as directory:
        snapshot = Path(directory) / path.name
        digest, size = _snapshot_file(path, snapshot)
        checkpoint = torch.load(
            snapshot, map_location="cpu", weights_only=False, mmap=True
        )
    expected = (
        "epoch",
        "model",
        "classifier",
        "ema",
        "optimizer",
        "scheduler",
        "scaler",
        "mask_generator",
        "torch_rng_state",
        "cuda_rng_states",
        "selection_holdout",
        "training_protocol",
        "history",
    )
    if type(checkpoint) is not dict or tuple(checkpoint) != expected:
        raise ValueError("replication checkpoint schema differs")
    if checkpoint["epoch"] != epoch or checkpoint["selection_holdout"] != {
        "seed": 0,
        "fraction": 0.2,
    }:
        raise ValueError("replication checkpoint registration differs")
    validate_training_protocol(
        checkpoint["training_protocol"], seed=seed, classifier_init=classifier_init
    )
    history = checkpoint["history"]
    if (
        type(history) is not list
        or len(history) != epoch
        or any(
            type(row) is not dict or row.get("epoch") != expected_epoch
            for expected_epoch, row in enumerate(history, 1)
        )
    ):
        raise ValueError("replication checkpoint history differs")
    return checkpoint, digest, size


def _load_official_model_snapshot(
    trainer,
    *,
    checkout: Path,
    checkpoint: Path,
    expected_sha256: str | None = None,
):
    with tempfile.TemporaryDirectory(
        prefix=".unicom-initial-snapshot-", dir=checkpoint.parent
    ) as directory:
        snapshot = Path(directory) / checkpoint.name
        digest, _size = _snapshot_file(checkpoint, snapshot)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("UNICOM initial checkpoint differs")
        loaded = trainer._load_official_model(checkout, snapshot)
    return loaded, digest


def _parse_partition_bytes(
    dataset_root: Path,
    payload: bytes,
    *,
    expected_counts: tuple[int, int, int] | None = (25_882, 14_218, 12_612),
):
    from sfora.unicom_inshop import InshopRecord

    if type(payload) is not bytes:
        raise TypeError("partition payload must be bytes")
    lines = payload.decode("utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("In-Shop partition is truncated")
    try:
        declared = int(lines[0].strip())
    except ValueError as error:
        raise ValueError("In-Shop partition count is invalid") from error
    if lines[1].split() != ["image_name", "item_id", "evaluation_status"]:
        raise ValueError("In-Shop partition header differs")
    records = []
    counts = {"train": 0, "query": 0, "gallery": 0}
    for line_number, line in enumerate(lines[2:], start=3):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"In-Shop partition row {line_number} differs")
        image_name, label, split = fields
        if split not in counts or not label:
            raise ValueError(f"In-Shop partition row {line_number} differs")
        image_path = dataset_root / "Img" / image_name
        if not image_path.is_file() or image_path.is_symlink():
            raise ValueError(f"In-Shop image is missing or not regular: {image_path}")
        records.append(InshopRecord(split=split, image_path=image_path, label=label))
        counts[split] += 1
    if len(records) != declared:
        raise ValueError("In-Shop partition declared count differs")
    if expected_counts is not None and tuple(counts.values()) != expected_counts:
        raise ValueError("In-Shop split counts differ")
    query_labels = {record.label for record in records if record.split == "query"}
    gallery_labels = {record.label for record in records if record.split == "gallery"}
    if query_labels != gallery_labels:
        raise ValueError("In-Shop query/gallery identity membership differs")
    return tuple(records), hashlib.sha256(payload).hexdigest()


def load_replication_rows(args: argparse.Namespace):
    import torch

    from sfora.unicom_training import identity_holdout

    factorial = _load_module(
        Path(__file__).resolve().with_name("evaluate_unicom_ema_imprint_factorial.py"),
        "_unicom_factorial_for_replication",
    )
    soup = _load_module(
        Path(__file__).resolve().with_name("evaluate_unicom_checkpoint_soup.py"),
        "_unicom_soup_for_imprint_replication",
    )
    trainer = soup._load_trainer()
    if trainer._git_revision(args.unicom_checkout) != trainer.UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for replication evaluation")
    paths = {
        "random_raw": _checkpoint_paths(args.random_run),
        "imprinted_raw": _checkpoint_paths(args.imprinted_run),
    }
    if set(paths["random_raw"]) & set(paths["imprinted_raw"]):
        raise ValueError("replication checkpoints overlap")

    partition_payload, partition_sha256 = _read_exact_bytes(
        args.dataset_root / "Eval" / "list_eval_partition.txt"
    )
    records, parsed_partition_sha256 = _parse_partition_bytes(
        args.dataset_root, partition_payload
    )
    if parsed_partition_sha256 != partition_sha256:
        raise AssertionError("partition digest differs")
    train_records = tuple(row for row in records if row.split == "train")
    optimization, query, gallery, _labels = identity_holdout(
        train_records, fraction=0.2, seed=0
    )
    if not optimization or not query or not gallery:
        raise ValueError("replication train-only split differs")
    (model, transform), loaded_initial_sha256 = _load_official_model_snapshot(
        trainer,
        checkout=args.unicom_checkout,
        checkpoint=args.initial_checkpoint,
        expected_sha256=trainer.UNICOM_L14_336_SHA256,
    )
    initial_checkpoint_sha256 = loaded_initial_sha256
    device = torch.device("cuda")
    model = model.to(device)
    calibration_loader = torch.utils.data.DataLoader(
        trainer.InshopEvalDataset(optimization, transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    rows: list[dict[str, object]] = []
    protocols: dict[str, dict[str, object]] = {}
    checkpoint_sizes = {cell: [] for cell in REGISTERED_CELLS}
    deployment_sizes: list[int] = []
    for cell, classifier_init in (("random_raw", "random"), ("imprinted_raw", "imprinted")):
        for epoch, path in zip(REGISTERED_EPOCHS, paths[cell], strict=True):
            checkpoint, checkpoint_sha256, checkpoint_size = _load_checkpoint(
                path, seed=args.seed, epoch=epoch, classifier_init=classifier_init
            )
            checkpoint_sizes[cell].append(checkpoint_size)
            protocol = checkpoint["training_protocol"]
            if cell in protocols and protocols[cell] != protocol:
                raise ValueError("replication checkpoint protocols differ")
            protocols[cell] = protocol
            model.load_state_dict(checkpoint["model"], strict=True)
            soup.recalibrate_batch_norm(model, calibration_loader, device=device)
            metrics, evidence = soup.evaluate_holdout_with_query_evidence(
                trainer,
                model,
                query,
                gallery,
                transform,
                device=device,
                batch_size=128,
                workers=4,
                selected_features=512,
            )
            rows.append(
                {
                    "cell": cell,
                    "epoch": epoch,
                    "checkpoint_sha256": checkpoint_sha256,
                    "training_history_sha256": _json_sha256(checkpoint["history"]),
                    "metrics": metrics,
                    "query_evidence": evidence,
                }
            )
            if epoch == 16:
                deployment_sizes.append(
                    sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in checkpoint["model"].values()
                    )
                )
            del checkpoint
    if deployment_sizes[0] != deployment_sizes[1]:
        raise ValueError("replication deployment storage differs")
    validate_runtime_bindings(
        protocols,
        trainer_sha256=_sha256_file(Path(trainer.__file__)),
        unicom_revision=trainer._git_revision(args.unicom_checkout),
        initial_checkpoint_sha256=initial_checkpoint_sha256,
        partition_sha256=partition_sha256,
    )
    latency = factorial._benchmark_inference_latency(model, calibration_loader, device=device)
    storage = {cell: sum(checkpoint_sizes[cell]) for cell in REGISTERED_CELLS}
    return rows, protocols, storage, deployment_sizes[0], latency


def load_measurement_receipt(
    path: Path,
    *,
    log_path: Path,
    seed: int,
    classifier_init: str,
    training_protocol: Mapping[str, object],
    checkpoint_sha256_by_epoch: Sequence[str],
    training_history_sha256_by_epoch: Sequence[str],
) -> tuple[dict[str, object], str]:
    payload, digest = _read_exact_bytes(path)
    receipt = strict_json_object(payload)
    if type(receipt) is not dict or tuple(receipt) != MEASUREMENT_KEYS:
        raise ValueError("training measurement receipt schema differs")
    if (
        receipt["schema_version"] != "unicom-training-measurement-v1"
        or receipt["seed"] != seed
        or type(receipt["seed"]) is not int
        or receipt["classifier_init"] != classifier_init
        or not _finite_float(receipt["training_seconds"])
        or receipt["training_seconds"] <= 0.0
        or type(receipt["peak_gpu_mib"]) is not int
        or receipt["peak_gpu_mib"] <= 0
        or not _finite_float(receipt["step_wall_seconds"])
        or receipt["step_wall_seconds"] <= 0.0
        or not _finite_float(receipt["fusible_non_backbone_seconds"])
        or not 0.0
        <= receipt["fusible_non_backbone_seconds"]
        <= receipt["step_wall_seconds"]
        or not _sha256(receipt["training_log_sha256"])
        or _sha256_file(log_path) != receipt["training_log_sha256"]
        or receipt["training_protocol_sha256"] != _json_sha256(training_protocol)
        or receipt["checkpoint_sha256_by_epoch"] != list(checkpoint_sha256_by_epoch)
        or receipt["training_history_sha256_by_epoch"]
        != list(training_history_sha256_by_epoch)
    ):
        raise ValueError("training measurement receipt differs")
    return receipt, digest


def run(args: argparse.Namespace) -> dict[str, object]:
    rows, protocols, storage, deployment_storage, latency = load_replication_rows(args)
    random_measurement, random_measurement_sha256 = load_measurement_receipt(
        args.random_measurement_receipt,
        log_path=args.random_training_log,
        seed=args.seed,
        classifier_init="random",
        training_protocol=protocols["random_raw"],
        checkpoint_sha256_by_epoch=[
            row["checkpoint_sha256"] for row in rows if row["cell"] == "random_raw"
        ],
        training_history_sha256_by_epoch=[
            row["training_history_sha256"] for row in rows if row["cell"] == "random_raw"
        ],
    )
    imprinted_measurement, imprinted_measurement_sha256 = load_measurement_receipt(
        args.imprinted_measurement_receipt,
        log_path=args.imprinted_training_log,
        seed=args.seed,
        classifier_init="imprinted",
        training_protocol=protocols["imprinted_raw"],
        checkpoint_sha256_by_epoch=[
            row["checkpoint_sha256"] for row in rows if row["cell"] == "imprinted_raw"
        ],
        training_history_sha256_by_epoch=[
            row["training_history_sha256"]
            for row in rows
            if row["cell"] == "imprinted_raw"
        ],
    )
    return build_replication_pair(
        seed=args.seed,
        rows=rows,
        random_protocol=protocols["random_raw"],
        imprinted_protocol=protocols["imprinted_raw"],
        random_training_seconds=random_measurement["training_seconds"],
        imprinted_training_seconds=imprinted_measurement["training_seconds"],
        random_peak_gpu_mib=random_measurement["peak_gpu_mib"],
        imprinted_peak_gpu_mib=imprinted_measurement["peak_gpu_mib"],
        random_checkpoint_storage_bytes=storage["random_raw"],
        imprinted_checkpoint_storage_bytes=storage["imprinted_raw"],
        deployment_storage_bytes=deployment_storage,
        inference_latency_ms_per_image=latency,
        random_measurement_receipt_sha256=random_measurement_sha256,
        imprinted_measurement_receipt_sha256=imprinted_measurement_sha256,
        random_profile=(
            random_measurement["step_wall_seconds"],
            random_measurement["fusible_non_backbone_seconds"],
        ),
        imprinted_profile=(
            imprinted_measurement["step_wall_seconds"],
            imprinted_measurement["fusible_non_backbone_seconds"],
        ),
    )


def strict_json_object(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TypeError("JSON payload must be bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def exact_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=exact_pairs)
    if type(value) is not dict:
        raise ValueError("JSON root must be an object")
    return value


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for no-clobber rollback")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        if error == errno.ENOENT:
            raise FileNotFoundError(error, os.strerror(error), source)
        raise OSError(error, os.strerror(error), source, destination)


def _link_fd_noreplace(
    descriptor: int, destination: Path, directory_descriptor: int
) -> None:
    linkat = getattr(ctypes.CDLL(None, use_errno=True), "linkat", None)
    if linkat is None:
        raise RuntimeError("linkat is required for descriptor publication")
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    result = linkat(descriptor, b"", directory_descriptor, os.fsencode(destination.name), 0x1000)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def _read_descriptor(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    payload = bytearray()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise RuntimeError("descriptor payload is truncated")
        payload.extend(chunk)
        offset += len(chunk)
    return bytes(payload)


def _rollback_published_link(
    output: Path, *, owned: tuple[int, int], directory_descriptor: int
) -> None:
    for _attempt in range(16):
        rollback = output.with_name(
            f".{output.name}.{os.getpid()}.{secrets.token_hex(12)}.rollback"
        )
        try:
            _rename_noreplace(output, rollback)
        except FileExistsError:
            continue
        except FileNotFoundError:
            return
        break
    else:
        raise RuntimeError("cannot allocate an exclusive rollback path")
    info = rollback.lstat()
    if (info.st_dev, info.st_ino) == owned:
        pass
    else:
        try:
            _rename_noreplace(rollback, output)
        except FileExistsError:
            os.fsync(directory_descriptor)
            return
    os.fsync(directory_descriptor)


def write_report_atomic(report: dict[str, object], output: Path) -> None:
    validate_replication_pair(report)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    descriptor: int | None = None
    directory_descriptor: int | None = None
    owned: tuple[int, int] | None = None
    published = False
    completed = False
    try:
        directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        descriptor = os.open(output.parent, os.O_RDWR | os.O_TMPFILE, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        persisted_payload = _read_descriptor(descriptor)
        if persisted_payload != payload:
            raise RuntimeError("persisted report bytes differ")
        persisted = strict_json_object(persisted_payload)
        validate_replication_pair(persisted)
        _link_fd_noreplace(descriptor, output, directory_descriptor)
        published = True
        os.fsync(directory_descriptor)
        output_info = output.lstat()
        if (output_info.st_dev, output_info.st_ino) != owned:
            raise RuntimeError("published report inode differs")
        published_payload = _read_descriptor(descriptor)
        if published_payload != payload:
            raise RuntimeError("published report bytes differ")
        validate_replication_pair(strict_json_object(published_payload))
        completed = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if published and not completed and owned is not None:
            if directory_descriptor is None:
                directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            _rollback_published_link(
                output, owned=owned, directory_descriptor=directory_descriptor
            )
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True, type=int, choices=range(1, 7))
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--random-run", required=True, type=Path)
    parser.add_argument("--imprinted-run", required=True, type=Path)
    parser.add_argument("--random-measurement-receipt", required=True, type=Path)
    parser.add_argument("--imprinted-measurement-receipt", required=True, type=Path)
    parser.add_argument("--random-training-log", required=True, type=Path)
    parser.add_argument("--imprinted-training-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        report = run(args)
        write_report_atomic(report, args.output)
    except Exception as error:
        print(f"replication evaluation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
