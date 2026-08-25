#!/usr/bin/env python3
"""Train the pinned UNICOM ViT-L/14@336 In-Shop recipe on one device."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_retrieval_audit import retrieval_view
from sfora.unicom_training import (
    experiment_stream_seed,
    identity_holdout,
    padded_epoch_indices,
    sample_shard_masks,
    sharded_mask_arcface_loss,
)

UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
UNICOM_L14_336_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
UNICOM_MEAN = (0.48145466, 0.4578275, 0.40821073)
UNICOM_STD = (0.26862954, 0.26130258, 0.27577711)
EMA_DECAY = 0.999
INITIALIZATION_RECEIPT_KEYS = (
    "schema_version",
    "seed",
    "classifier_init",
    "trainer_sha256",
    "algorithm",
    "classifier_tensor_sha256",
    "classifier_shape",
    "classifier_dtype",
    "optimizer_steps_per_epoch",
    "initialization_seconds",
    "post_initialization_rng",
)
TRAINING_RUN_RECEIPT_KEYS = (
    "schema_version",
    "source_commit",
    "trainer_sha256",
    "config_path",
    "config_sha256",
    "seed",
    "arm",
    "protocol",
    "command",
    "started_unix_ns",
    "finished_unix_ns",
    "elapsed_seconds",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "exit_status",
    "history",
    "checkpoints",
    "runtime",
)
FULL_WIDTH_ARM_PROTOCOLS = {
    "sampled_512": ("official-eight-mask", 512, 768),
    "full_768": ("official-eight-mask", 768, 768),
}


def _lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_binding(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("training run evidence path differs")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def training_run_receipt(
    *,
    source_commit: str,
    config_path: str,
    config_sha256: str,
    seed: int,
    arm: str,
    objective: str,
    selected_features: int,
    evaluation_features: int,
    command: list[str],
    started_unix_ns: int,
    finished_unix_ns: int,
    elapsed_seconds: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    exit_status: int,
    history_path: Path,
    checkpoint_paths: tuple[Path, ...],
    runtime: dict[str, str],
) -> dict[str, object]:
    """Build one source- and byte-bound prospective training receipt."""

    checkpoints = []
    if type(checkpoint_paths) is not tuple or len(checkpoint_paths) != 4:
        raise ValueError("training run checkpoint paths differ")
    for epoch, path in zip((4, 8, 12, 16), checkpoint_paths, strict=True):
        checkpoints.append({"epoch": epoch, **_file_binding(path)})
    value = {
        "schema_version": "unicom-full-width-training-run-v1",
        "source_commit": source_commit,
        "trainer_sha256": _sha256_file(Path(__file__)),
        "config_path": config_path,
        "config_sha256": config_sha256,
        "seed": seed,
        "arm": arm,
        "protocol": {
            "objective": objective,
            "selected_features": selected_features,
            "evaluation_features": evaluation_features,
        },
        "command": command,
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": finished_unix_ns,
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "exit_status": exit_status,
        "history": _file_binding(history_path),
        "checkpoints": checkpoints,
        "runtime": runtime,
    }
    validate_training_run_receipt(value)
    return value


def validate_training_run_receipt(value: object) -> None:
    """Strictly validate one prospective training run receipt."""

    if type(value) is not dict or tuple(value) != TRAINING_RUN_RECEIPT_KEYS:
        raise ValueError("training run receipt schema differs")
    arm = value["arm"]
    protocol = value["protocol"]
    if (
        value["schema_version"] != "unicom-full-width-training-run-v1"
        or not _lower_hex(value["source_commit"], 40)
        or not _lower_hex(value["trainer_sha256"], 64)
        or type(value["config_path"]) is not str
        or not value["config_path"].endswith(".json")
        or not _lower_hex(value["config_sha256"], 64)
        or type(value["seed"]) is not int
        or value["seed"] not in (0, 2, 3, 4, 5, 6)
        or type(arm) is not str
        or arm not in FULL_WIDTH_ARM_PROTOCOLS
        or type(protocol) is not dict
        or tuple(protocol) != ("objective", "selected_features", "evaluation_features")
        or tuple(protocol.values()) != FULL_WIDTH_ARM_PROTOCOLS[arm]
        or any(
            type(item) is not type(reference)
            for item, reference in zip(
                protocol.values(), FULL_WIDTH_ARM_PROTOCOLS[arm], strict=True
            )
        )
    ):
        raise ValueError("training run receipt binding differs")
    command = value["command"]
    if (
        type(command) is not list
        or not command
        or any(type(token) is not str or not token for token in command)
    ):
        raise ValueError("training run command differs")
    started = value["started_unix_ns"]
    finished = value["finished_unix_ns"]
    elapsed = value["elapsed_seconds"]
    allocated = value["peak_allocated_bytes"]
    reserved = value["peak_reserved_bytes"]
    exit_status = value["exit_status"]
    if (
        type(started) is not int
        or type(finished) is not int
        or started <= 0
        or finished <= started
        or type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed <= 0.0
        or type(allocated) is not int
        or type(reserved) is not int
        or allocated < 0
        or reserved < allocated
        or type(exit_status) is not int
        or exit_status != 0
    ):
        raise ValueError("training run timing or memory differs")
    history = value["history"]
    if (
        type(history) is not dict
        or tuple(history) != ("path", "sha256", "bytes")
        or type(history["path"]) is not str
        or not history["path"]
        or not _lower_hex(history["sha256"], 64)
        or type(history["bytes"]) is not int
        or history["bytes"] <= 0
    ):
        raise ValueError("training run history binding differs")
    checkpoints = value["checkpoints"]
    if type(checkpoints) is not list or len(checkpoints) != 4:
        raise ValueError("training run checkpoints differ")
    paths: set[str] = set()
    for epoch, row in zip((4, 8, 12, 16), checkpoints, strict=True):
        if (
            type(row) is not dict
            or tuple(row) != ("epoch", "path", "sha256", "bytes")
            or row["epoch"] != epoch
            or type(row["path"]) is not str
            or not row["path"]
            or row["path"] in paths
            or not _lower_hex(row["sha256"], 64)
            or type(row["bytes"]) is not int
            or row["bytes"] <= 0
        ):
            raise ValueError("training run checkpoint binding differs")
        paths.add(row["path"])
    runtime = value["runtime"]
    if (
        type(runtime) is not dict
        or tuple(runtime) != ("python", "torch", "cuda")
        or any(type(item) is not str or not item for item in runtime.values())
    ):
        raise ValueError("training run runtime differs")


def write_training_run_receipt_atomic(
    receipt: dict[str, object], output: Path
) -> None:
    """Publish one validated run receipt without replacing an existing path."""

    validate_training_run_receipt(receipt)
    if not isinstance(output, Path):
        raise TypeError("training run receipt output must be a Path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    descriptor: int | None = None
    published = False
    completed = False
    owned: tuple[int, int] | None = None
    try:
        descriptor = os.open(output.parent, os.O_RDWR | os.O_TMPFILE, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        _write_descriptor(descriptor, payload)
        os.fsync(descriptor)
        persisted = _read_descriptor(descriptor)
        if persisted != payload:
            raise RuntimeError("persisted training run receipt bytes differ")
        validate_training_run_receipt(strict_json_object(persisted))
        _link_receipt_fd_noreplace(descriptor, output, directory_descriptor)
        published = True
        os.fsync(directory_descriptor)
        output_info = output.lstat()
        if (output_info.st_dev, output_info.st_ino) != owned:
            raise RuntimeError("published training run receipt inode differs")
        published_payload = output.read_bytes()
        if published_payload != payload:
            raise RuntimeError("published training run receipt bytes differ")
        validate_training_run_receipt(strict_json_object(published_payload))
        completed = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if published and not completed and owned is not None:
            try:
                info = output.lstat()
            except FileNotFoundError:
                pass
            else:
                if (info.st_dev, info.st_ino) == owned:
                    output.unlink()
                    os.fsync(directory_descriptor)
        os.close(directory_descriptor)


def _state_digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\0" + pickle.dumps(value, protocol=5)).hexdigest()


def _tensor_state_digest(domain: bytes, value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise ValueError("RNG tensor state differs")
    payload = bytes(value.detach().cpu().contiguous().tolist())
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def rng_state_hashes() -> dict[str, object]:
    """Hash all registered RNG domains without advancing any stream."""
    import numpy as np

    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "python_sha256": _state_digest(b"python-random-v1", random.getstate()),
        "numpy_sha256": _state_digest(b"numpy-random-v1", np.random.get_state()),
        "torch_cpu_sha256": _tensor_state_digest(
            b"torch-cpu-random-v1", torch.get_rng_state()
        ),
        "torch_cuda_sha256_by_device": [
            _tensor_state_digest(f"torch-cuda-random-v1:{index}".encode(), state)
            for index, state in enumerate(cuda_states)
        ],
    }


def _lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_initialization_receipt(
    value: object, *, expected_shape: list[int]
) -> None:
    if type(value) is not dict or tuple(value) != INITIALIZATION_RECEIPT_KEYS:
        raise ValueError("initialization receipt schema differs")
    expected_algorithms = {
        "random": "torch-normal-std-0.01-rng-balanced",
        "imprinted": "normalized-class-means-norm-matched-rng-restored",
    }
    mode = value["classifier_init"]
    if (
        value["schema_version"] != "unicom-classifier-initialization-v1"
        or type(value["seed"]) is not int
        or value["seed"] not in range(2, 7)
        or type(mode) is not str
        or mode not in expected_algorithms
        or value["algorithm"] != expected_algorithms[mode]
        or not _lower_sha256(value["trainer_sha256"])
        or not _lower_sha256(value["classifier_tensor_sha256"])
        or type(value["classifier_shape"]) is not list
        or value["classifier_shape"] != expected_shape
        or any(type(item) is not int for item in value["classifier_shape"])
        or value["classifier_dtype"] != "torch.float32"
        or type(value["optimizer_steps_per_epoch"]) is not int
        or value["optimizer_steps_per_epoch"] <= 0
        or type(value["initialization_seconds"]) is not float
        or not math.isfinite(value["initialization_seconds"])
        or value["initialization_seconds"] <= 0.0
    ):
        raise ValueError("initialization receipt values differ")
    rng = value["post_initialization_rng"]
    if type(rng) is not dict or tuple(rng) != (
        "python_sha256",
        "numpy_sha256",
        "torch_cpu_sha256",
        "torch_cuda_sha256_by_device",
    ):
        raise ValueError("initialization receipt RNG schema differs")
    cuda = rng["torch_cuda_sha256_by_device"]
    if (
        any(not _lower_sha256(rng[key]) for key in tuple(rng)[:3])
        or type(cuda) is not list
        or not cuda
        or any(not _lower_sha256(item) for item in cuda)
    ):
        raise ValueError("initialization receipt RNG differs")


def classifier_initialization_receipt(
    *,
    seed: int,
    classifier_init: str,
    classifier: torch.Tensor,
    optimizer_steps_per_epoch: int,
    initialization_seconds: float,
    trainer_sha256: str,
) -> dict[str, object]:
    if not isinstance(classifier, torch.Tensor) or classifier.dtype != torch.float32:
        raise ValueError("initialization classifier differs")
    contiguous = classifier.detach().cpu().contiguous()
    receipt = {
        "schema_version": "unicom-classifier-initialization-v1",
        "seed": seed,
        "classifier_init": classifier_init,
        "trainer_sha256": trainer_sha256,
        "algorithm": {
            "random": "torch-normal-std-0.01-rng-balanced",
            "imprinted": "normalized-class-means-norm-matched-rng-restored",
        }.get(classifier_init),
        "classifier_tensor_sha256": hashlib.sha256(
            contiguous.numpy().tobytes(order="C")
        ).hexdigest(),
        "classifier_shape": list(contiguous.shape),
        "classifier_dtype": str(contiguous.dtype),
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "initialization_seconds": initialization_seconds,
        "post_initialization_rng": rng_state_hashes(),
    }
    validate_initialization_receipt(receipt, expected_shape=list(contiguous.shape))
    return receipt


def strict_json_object(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("JSON payload must be bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def exact_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(payload, parse_constant=reject_constant, object_pairs_hook=exact_pairs)
    if type(value) is not dict:
        raise ValueError("JSON root must be an object")
    return value


def _link_receipt_fd_noreplace(
    descriptor: int, destination: Path, directory_descriptor: int
) -> None:
    linkat = getattr(ctypes.CDLL(None, use_errno=True), "linkat", None)
    if linkat is None:
        raise RuntimeError("linkat is required for receipt publication")
    linkat.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    )
    linkat.restype = ctypes.c_int
    if linkat(descriptor, b"", directory_descriptor, os.fsencode(destination.name), 0x1000):
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def _read_descriptor(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    payload = bytearray()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1 << 20, size - offset), offset)
        if not chunk:
            raise RuntimeError("receipt descriptor is truncated")
        payload.extend(chunk)
        offset += len(chunk)
    return bytes(payload)


def _write_descriptor(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise RuntimeError("initialization receipt write made no progress")
        offset += written


def write_initialization_receipt_atomic(
    receipt: dict[str, object], output: Path, *, expected_shape: list[int]
) -> None:
    validate_initialization_receipt(receipt, expected_shape=expected_shape)
    if not isinstance(output, Path):
        raise TypeError("receipt output must be a Path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    descriptor: int | None = None
    published = False
    completed = False
    owned: tuple[int, int] | None = None
    try:
        descriptor = os.open(output.parent, os.O_RDWR | os.O_TMPFILE, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        _write_descriptor(descriptor, payload)
        os.fsync(descriptor)
        persisted = _read_descriptor(descriptor)
        if persisted != payload:
            raise RuntimeError("persisted initialization receipt bytes differ")
        validate_initialization_receipt(
            strict_json_object(persisted), expected_shape=expected_shape
        )
        _link_receipt_fd_noreplace(descriptor, output, directory_descriptor)
        published = True
        os.fsync(directory_descriptor)
        output_info = output.lstat()
        if (output_info.st_dev, output_info.st_ino) != owned:
            raise RuntimeError("published initialization receipt inode differs")
        published_payload = output.read_bytes()
        if published_payload != payload:
            raise RuntimeError("published initialization receipt bytes differ")
        validate_initialization_receipt(
            strict_json_object(published_payload), expected_shape=expected_shape
        )
        completed = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if published and not completed and owned is not None:
            try:
                info = output.lstat()
            except FileNotFoundError:
                pass
            else:
                if (info.st_dev, info.st_ino) == owned:
                    output.unlink()
                    os.fsync(directory_descriptor)
        os.close(directory_descriptor)


def bind_initialization_receipt(
    *,
    output: Path,
    resume: bool,
    seed: int,
    classifier_init: str,
    classifier: torch.Tensor,
    optimizer_steps_per_epoch: int,
    initialization_seconds: float | None,
    trainer_sha256: str,
    expected_shape: list[int],
) -> dict[str, object]:
    if list(classifier.shape) != expected_shape:
        raise ValueError("initialization classifier shape differs")
    if resume:
        if not output.is_file() or output.is_symlink():
            raise ValueError("resume initialization receipt is absent")
        recorded = strict_json_object(output.read_bytes())
        validate_initialization_receipt(recorded, expected_shape=expected_shape)
        expected = classifier_initialization_receipt(
            seed=seed,
            classifier_init=classifier_init,
            classifier=classifier,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,
            initialization_seconds=recorded["initialization_seconds"],
            trainer_sha256=trainer_sha256,
        )
        if recorded != expected:
            raise ValueError("resume initialization receipt differs")
        return recorded
    if type(initialization_seconds) is not float:
        raise TypeError("fresh initialization duration differs")
    receipt = classifier_initialization_receipt(
        seed=seed,
        classifier_init=classifier_init,
        classifier=classifier,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        initialization_seconds=initialization_seconds,
        trainer_sha256=trainer_sha256,
    )
    write_initialization_receipt_atomic(receipt, output, expected_shape=expected_shape)
    return receipt


def registered_classifier_shape(labels: Mapping[str, int]) -> list[int]:
    """Derive the live classifier shape and enforce the frozen partition cross-check."""
    if (
        type(labels) is not dict
        or tuple(labels.values()) != tuple(range(len(labels)))
        or len(labels) != 3200
    ):
        raise ValueError("registered classifier shape differs")
    return [len(labels), 768]


def classifier_shape_for_run(
    labels: Mapping[str, int],
    *,
    record_initialization: bool,
    selected_features: int | None = None,
    evaluation_features: int = 768,
) -> list[int]:
    """Derive a general training shape, applying the frozen count only to receipt runs."""
    if (
        type(record_initialization) is not bool
        or (selected_features is not None and type(selected_features) is not int)
        or (selected_features is not None and selected_features <= 0)
        or type(evaluation_features) is not int
        or evaluation_features <= 0
        or type(labels) is not dict
        or not labels
        or tuple(labels.values()) != tuple(range(len(labels)))
    ):
        raise ValueError("classifier shape differs")
    if record_initialization:
        registered_classifier_shape(labels)
    return [len(labels), 768]


class StepEMA:
    """Same-device FP32 exponential average of trainable retrieval weights."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        classifier: torch.nn.Parameter,
        *,
        decay: float = EMA_DECAY,
    ) -> None:
        if type(decay) is not float or decay != EMA_DECAY:
            raise ValueError("EMA decay differs from the frozen protocol")
        parameters = dict(backbone.named_parameters())
        if not parameters or any(
            parameter.dtype != torch.float32 for parameter in parameters.values()
        ):
            raise ValueError("EMA backbone parameters must be nonempty FP32 tensors")
        if classifier.dtype != torch.float32:
            raise ValueError("EMA classifier must be FP32")
        self._backbone = backbone
        self._classifier_source = classifier
        self._parameter_names = tuple(parameters)
        self._shadow = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        self._classifier = classifier.detach().clone()
        self._updates = 0
        self._hook = None

    @torch.no_grad()
    def update(self) -> None:
        parameters = dict(self._backbone.named_parameters())
        if tuple(parameters) != self._parameter_names:
            raise ValueError("EMA backbone parameter order differs")
        for name in self._parameter_names:
            source = parameters[name]
            shadow = self._shadow[name]
            if (
                source.dtype != torch.float32
                or source.shape != shadow.shape
                or source.device != shadow.device
            ):
                raise ValueError("EMA backbone parameter contract differs")
            shadow.mul_(EMA_DECAY).add_(source, alpha=1.0 - EMA_DECAY)
        source_classifier = self._classifier_source
        if (
            source_classifier.dtype != torch.float32
            or source_classifier.shape != self._classifier.shape
            or source_classifier.device != self._classifier.device
        ):
            raise ValueError("EMA classifier contract differs")
        self._classifier.mul_(EMA_DECAY).add_(source_classifier, alpha=1.0 - EMA_DECAY)
        self._updates += 1

    def register_step_hook(self, optimizer: torch.optim.Optimizer):
        if self._hook is not None:
            raise RuntimeError("EMA optimizer hook is already registered")
        self._hook = optimizer.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: self.update()
        )
        return self._hook

    def release_step_hook(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": EMA_DECAY,
            "updates": self._updates,
            "backbone": {
                name: self._shadow[name].detach().cpu().clone() for name in self._parameter_names
            },
            "classifier": self._classifier.detach().cpu().clone(),
        }

    def load_state_dict(self, state: object) -> None:
        if type(state) is not dict or tuple(state) != (
            "decay",
            "updates",
            "backbone",
            "classifier",
        ):
            raise ValueError("EMA state schema differs")
        if type(state["decay"]) is not float or state["decay"] != EMA_DECAY:
            raise ValueError("EMA state decay differs")
        updates = state["updates"]
        if type(updates) is not int or updates < 0:
            raise TypeError("EMA state update count differs")
        backbone = state["backbone"]
        if type(backbone) is not dict or tuple(backbone) != self._parameter_names:
            raise ValueError("EMA state parameter order differs")
        for name in self._parameter_names:
            value = backbone[name]
            target = self._shadow[name]
            if (
                type(value) is not torch.Tensor
                or value.dtype != torch.float32
                or value.shape != target.shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError("EMA state parameter differs")
        classifier = state["classifier"]
        if (
            type(classifier) is not torch.Tensor
            or classifier.dtype != torch.float32
            or classifier.shape != self._classifier.shape
            or not torch.isfinite(classifier).all()
        ):
            raise ValueError("EMA state classifier differs")
        with torch.no_grad():
            for name in self._parameter_names:
                self._shadow[name].copy_(backbone[name])
            self._classifier.copy_(classifier)
        self._updates = updates

    def materialize_backbone_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: value.detach().cpu().clone()
            for name, value in self._backbone.state_dict().items()
        }
        for name in self._parameter_names:
            state[name] = self._shadow[name].detach().cpu().clone()
        return state


class InshopTrainDataset(Dataset[tuple[torch.Tensor, int]]):
    """Optimization rows with a train-only contiguous identity mapping."""

    def __init__(
        self,
        records: tuple[InshopRecord, ...],
        label_indices: Mapping[str, int],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self._records = records
        self._label_indices = dict(label_indices)
        self._transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self._records[index]
        with Image.open(record.image_path) as image:
            tensor = self._transform(image.convert("RGB"))
        return tensor, self._label_indices[record.label]

    def __len__(self) -> int:
        return len(self._records)


class InshopEvalDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(
        self,
        records: tuple[InshopRecord, ...],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self._records = records
        self._transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self._records[index]
        with Image.open(record.image_path) as image:
            tensor = self._transform(image.convert("RGB"))
        return tensor, record.label

    def __len__(self) -> int:
        return len(self._records)


class PaddedEpochSampler(Sampler[int]):
    """Single-process equivalent of UNICOM's padded global sampler."""

    def __init__(self, *, size: int, batch_size: int, seed: int) -> None:
        self._size = size
        self._batch_size = batch_size
        self._seed = seed
        self._epoch = 0
        self._indices = padded_epoch_indices(
            size=size,
            global_batch=batch_size,
            epoch=self._epoch,
            seed=seed,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._indices = padded_epoch_indices(
            size=self._size,
            global_batch=self._batch_size,
            epoch=epoch,
            seed=self._seed,
        )

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def objective_masks(
    objective: str,
    *,
    dimension: int,
    selected: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Build the registered UNICOM feature mask or its ablation controls."""

    if objective == "official-eight-mask":
        shards = 8
    elif objective == "official-one-mask":
        shards = 1
    elif objective == "prefix-512":
        if dimension <= 0 or not 0 < selected <= dimension:
            raise ValueError("mask dimensions differ")
        return torch.arange(selected, device=device, dtype=torch.int64)[None]
    else:
        raise ValueError(f"unsupported objective: {objective}")
    return sample_shard_masks(
        dimension=dimension,
        selected=selected,
        shards=shards,
        generator=generator,
        device=device,
    )


def build_train_transform(image_size: int):
    """Return the augmentation pipeline from UNICOM's retrieval trainer."""

    from timm.data import create_transform

    return create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=UNICOM_MEAN,
        std=UNICOM_STD,
    )


def build_optimizer(
    backbone: torch.nn.Module,
    classifier: torch.nn.Parameter,
    *,
    learning_rate: float,
    classifier_learning_rate: float,
    fused: bool,
) -> torch.optim.AdamW:
    """Build the two-rate zero-decay AdamW optimizer used by UNICOM."""

    return torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": learning_rate},
            {"params": [classifier], "lr": classifier_learning_rate},
        ],
        lr=learning_rate,
        weight_decay=0.0,
        fused=fused,
    )


@torch.inference_mode()
def imprinted_classifier_values(
    model: torch.nn.Module,
    records: tuple[InshopRecord, ...],
    labels: Mapping[str, int],
    transform: Callable[[Image.Image], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    """Build norm-matched class means without perturbing later training streams."""

    import numpy as np

    if (
        type(records) is not tuple
        or not records
        or type(labels) is not dict
        or not labels
        or tuple(labels.values()) != tuple(range(len(labels)))
        or any(type(label) is not str or not label for label in labels)
    ):
        raise ValueError("imprinted classifier label mapping differs")
    if type(batch_size) is not int or batch_size <= 0 or type(workers) is not int or workers < 0:
        raise ValueError("imprinted classifier loader configuration differs")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    sums = [torch.zeros(768, dtype=torch.float64) for _label in labels]
    counts = [0 for _label in labels]
    try:
        loader = torch.utils.data.DataLoader(
            InshopEvalDataset(records, transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            generator=torch.Generator().manual_seed(experiment_stream_seed(0, 4_000)),
        )
        model.eval()
        for images, batch_labels in loader:
            embeddings = model(images.to(device)).float()
            if (
                embeddings.ndim != 2
                or embeddings.shape != (len(batch_labels), 768)
                or not torch.isfinite(embeddings).all()
            ):
                raise ValueError("imprinted classifier embeddings differ")
            norms = torch.linalg.vector_norm(embeddings, dim=1)
            if torch.any(norms == 0.0):
                raise ValueError("imprinted classifier embedding has zero norm")
            normalized = (embeddings / norms[:, None]).cpu()
            for value, label in zip(normalized, batch_labels, strict=True):
                if label not in labels:
                    raise ValueError("imprinted classifier record label differs")
                index = labels[label]
                sums[index].add_(value.double())
                counts[index] += 1
    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    if any(count <= 0 for count in counts):
        raise ValueError("imprinted classifier class is empty")
    rows = []
    for total, count in zip(sums, counts, strict=True):
        mean = total / count
        norm = torch.linalg.vector_norm(mean)
        if not torch.isfinite(norm) or norm == 0.0:
            raise ValueError("imprinted classifier class mean differs")
        rows.append((mean / norm).float())
    return torch.stack(rows) * (0.01 * math.sqrt(768.0))


def initialize_classifier_values(
    *,
    labels: int,
    mode: str,
    imprinted: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Consume the official random-init stream in both factorial arms."""

    if type(labels) is not int or labels <= 0 or mode not in ("random", "imprinted"):
        raise ValueError("classifier initialization configuration differs")
    values = torch.empty(labels, 768, dtype=torch.float32)
    torch.nn.init.normal_(values, std=0.01)
    if mode == "imprinted":
        replacement = imprinted()
        if (
            type(replacement) is not torch.Tensor
            or replacement.dtype != torch.float32
            or replacement.device.type != "cpu"
            or replacement.shape != values.shape
            or not torch.isfinite(replacement).all()
        ):
            raise ValueError("imprinted classifier values differ")
        values.copy_(replacement)
    return values


def run_training_epoch(
    backbone: torch.nn.Module,
    classifier: torch.nn.Parameter,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler,
    mask_generator: torch.Generator,
    device: torch.device,
    objective: str,
    selected_features: int,
    margin: float,
    scale: float,
    max_steps: int | None,
    bf16: bool,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float | int]:
    """Run one train epoch with the registered single-device loss."""

    backbone.train()
    losses: list[float] = []
    for images, labels in loader:
        if max_steps is not None and len(losses) >= max_steps:
            break
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bf16,
        ):
            embeddings = backbone(images)
        embeddings = embeddings.float()
        masks = objective_masks(
            objective,
            dimension=embeddings.shape[1],
            selected=selected_features,
            generator=mask_generator,
            device=device,
        )
        loss = sharded_mask_arcface_loss(
            embeddings,
            classifier,
            labels,
            masks,
            margin=margin,
            scale=scale,
        )
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("training loader produced no steps")
    return {"steps": len(losses), "mean_loss": math.fsum(losses) / len(losses)}


@torch.inference_mode()
def _encode_records(
    model: torch.nn.Module,
    records: tuple[InshopRecord, ...],
    transform: Callable[[Image.Image], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[object, object]:
    import numpy as np

    loader = torch.utils.data.DataLoader(
        InshopEvalDataset(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    chunks: list[np.ndarray] = []
    labels: list[str] = []
    model.eval()
    for images, batch_labels in loader:
        values = model(images.to(device)).float().cpu().numpy()
        chunks.append(np.ascontiguousarray(values, dtype=np.float32))
        labels.extend(batch_labels)
    return np.ascontiguousarray(np.concatenate(chunks)), np.asarray(labels)


def resolve_evaluation_features(
    selected_features: int, evaluation_features: int | None
) -> int:
    """Resolve an explicit evaluation width without changing legacy runs."""

    if type(selected_features) is not int:
        raise TypeError("selected feature width must be a builtin integer")
    if not 0 < selected_features <= 768:
        raise ValueError("selected feature width differs")
    if evaluation_features is None:
        return selected_features
    if type(evaluation_features) is not int:
        raise TypeError("evaluation feature width must be a builtin integer")
    if not 0 < evaluation_features <= 768:
        raise ValueError("evaluation feature width differs")
    return evaluation_features


def evaluate_holdout(
    model: torch.nn.Module,
    query: tuple[InshopRecord, ...],
    gallery: tuple[InshopRecord, ...],
    transform: Callable[[Image.Image], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    evaluation_features: int,
) -> dict[str, float]:
    """Evaluate the identity-disjoint holdout with official deployment geometry."""

    import numpy as np

    query_values, query_labels = _encode_records(
        model, query, transform, device=device, batch_size=batch_size, workers=workers
    )
    gallery_values, gallery_labels = _encode_records(
        model, gallery, transform, device=device, batch_size=batch_size, workers=workers
    )
    result = retrieval_view(
        query_values,
        gallery_values,
        query_labels,
        gallery_labels,
        coordinates=np.arange(evaluation_features),
        normalize_before=True,
    )
    return {
        "recall_at_1": result.recall[1],
        "recall_at_10": result.recall[10],
        "recall_at_20": result.recall[20],
        "recall_at_30": result.recall[30],
        "map_at_r": result.map_at_r,
    }


def fit_model(
    *,
    raw_model: torch.nn.Module,
    train_model: torch.nn.Module,
    classifier: torch.nn.Parameter,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    sampler,
    mask_generator: torch.Generator,
    device: torch.device,
    epochs: int,
    start_epoch: int,
    objective: str,
    selected_features: int,
    margin: float,
    scale: float,
    max_steps: int | None,
    bf16: bool,
    scaler: torch.amp.GradScaler | None,
    eval_every: int,
    checkpoint_every: int,
    output_dir: Path,
    evaluate: Callable[[int], dict[str, float]],
    selection_holdout: dict[str, int | float],
    training_protocol: dict[str, object],
    history: list[dict[str, object]] | None = None,
    step_ema: StepEMA | None = None,
) -> list[dict[str, object]]:
    """Fit and persist sparse raw-model checkpoints for later trajectory soups."""

    output_dir.mkdir(parents=True, exist_ok=True)
    history = [] if history is None else list(history)
    if step_ema is not None:
        step_ema.register_step_hook(optimizer)
    try:
        for epoch in range(start_epoch, epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            _seed_training_loader(loader, seed=int(training_protocol["seed"]), epoch=epoch)
            train_result = run_training_epoch(
                train_model,
                classifier,
                loader,
                optimizer,
                scheduler=scheduler,
                mask_generator=mask_generator,
                device=device,
                objective=objective,
                selected_features=selected_features,
                margin=margin,
                scale=scale,
                max_steps=max_steps,
                bf16=bf16,
                scaler=scaler,
            )
            completed_epoch = epoch + 1
            metrics = (
                evaluate(completed_epoch)
                if eval_every > 0 and completed_epoch % eval_every == 0
                else None
            )
            row = {"epoch": completed_epoch, "train": train_result, "metrics": metrics}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if (
                completed_epoch % checkpoint_every == 0
                or metrics is not None
                or completed_epoch == epochs
            ):
                save_training_checkpoint(
                    output_dir / f"epoch-{completed_epoch:04d}.pt",
                    epoch=completed_epoch,
                    raw_model=raw_model,
                    classifier=classifier,
                    step_ema=step_ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    mask_generator=mask_generator,
                    selection_holdout=selection_holdout,
                    training_protocol=training_protocol,
                    history=history,
                )
    finally:
        if step_ema is not None:
            step_ema.release_step_hook()
    return history


def _seed_training_loader(loader, *, seed: int, epoch: int) -> None:
    data_generator = getattr(loader, "generator", None)
    if type(data_generator) is not torch.Generator:
        raise ValueError("training loader must expose its dedicated generator")
    data_generator.manual_seed(experiment_stream_seed(seed, 2_000 + epoch))


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    raw_model: torch.nn.Module,
    classifier: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    mask_generator: torch.Generator,
    selection_holdout: dict[str, int | float],
    training_protocol: dict[str, object],
    history: list[dict[str, object]],
    step_ema: StepEMA | None = None,
) -> None:
    """Atomically persist all mutable state needed to resume an epoch boundary."""

    payload = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "classifier": classifier.detach(),
        "ema": None if step_ema is None else step_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "mask_generator": mask_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "selection_holdout": selection_holdout,
        "training_protocol": training_protocol,
        "history": history,
    }
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"checkpoint temporary already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_training_checkpoint(
    path: Path,
    *,
    raw_model: torch.nn.Module,
    classifier: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    mask_generator: torch.Generator,
    device: torch.device,
    selection_holdout: dict[str, int | float],
    training_protocol: dict[str, object],
    step_ema: StepEMA | None = None,
) -> tuple[int, list[dict[str, object]]]:
    """Restore an epoch-boundary checkpoint and return its epoch and history."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
        raise ValueError("training checkpoint schema differs")
    if checkpoint["selection_holdout"] != selection_holdout:
        raise ValueError("training checkpoint selection holdout differs")
    if checkpoint["training_protocol"] != training_protocol:
        raise ValueError("training checkpoint training protocol differs")
    registered_ema = training_protocol.get("ema_decay")
    if registered_ema is None:
        if checkpoint["ema"] is not None or step_ema is not None:
            raise ValueError("training checkpoint EMA state differs")
    elif registered_ema != EMA_DECAY or step_ema is None:
        raise ValueError("training checkpoint EMA protocol differs")
    else:
        step_ema.load_state_dict(checkpoint["ema"])
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint["classifier"].shape != classifier.shape:
        raise ValueError("training checkpoint classifier shape differs")
    with torch.no_grad():
        classifier.copy_(checkpoint["classifier"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is None:
        if checkpoint["scheduler"] is not None:
            raise ValueError("training checkpoint scheduler differs")
    else:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is None:
        if checkpoint["scaler"] is not None:
            raise ValueError("training checkpoint scaler differs")
    else:
        scaler.load_state_dict(checkpoint["scaler"])
    mask_generator.set_state(checkpoint["mask_generator"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_rng_states = checkpoint["cuda_rng_states"]
    if cuda_rng_states is not None:
        if not torch.cuda.is_available() or device.type != "cuda":
            raise ValueError("training checkpoint CUDA RNG state differs")
        torch.cuda.set_rng_state_all(cuda_rng_states)
    epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    if type(epoch) is not int or epoch < 1 or type(history) is not list:
        raise ValueError("training checkpoint progress differs")
    return epoch, history


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_official_model(checkout: Path, checkpoint: Path):
    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    if Path(unicom.__file__).resolve().parent != package_root / "unicom":
        raise ValueError("imported UNICOM package does not come from the pinned checkout")
    return unicom.load("ViT-L/14@336px", download_root=str(checkpoint.parent))


def _seed_process(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    from sfora.unicom_inshop import parse_inshop_partition

    if _git_revision(args.unicom_checkout) != UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if args.checkpoint.name != "FP16-ViT-L-14-336px.pt":
        raise ValueError("UNICOM checkpoint filename differs")
    if _sha256_file(args.checkpoint) != UNICOM_L14_336_SHA256:
        raise ValueError("UNICOM checkpoint SHA-256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UNICOM training")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch must be positive and workers nonnegative")
    if args.eval_every < 0 or args.checkpoint_every <= 0:
        raise ValueError("evaluation cadence must be nonnegative and checkpoint cadence positive")
    if args.holdout_fraction == 0.0 and args.eval_every != 0:
        raise ValueError("full-train mode requires --eval-every 0 to avoid test leakage")

    evaluation_features = resolve_evaluation_features(
        args.selected_features, args.evaluation_features
    )
    _seed_process(args.seed)
    device = torch.device("cuda")
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(row for row in records if row.split == "train")
    optimization, query, gallery, labels = identity_holdout(
        train_records,
        fraction=args.holdout_fraction,
        seed=args.holdout_seed,
    )
    if args.run_receipt is not None and len(labels) != 3_200:
        raise ValueError("registered full-width class count differs")
    record_initialization = args.seed in range(2, 7)
    classifier_shape = classifier_shape_for_run(
        labels,
        record_initialization=record_initialization,
        selected_features=args.selected_features,
        evaluation_features=evaluation_features,
    )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=args.run_receipt is None,
    )
    raw_model, eval_transform = _load_official_model(args.unicom_checkout, args.checkpoint)
    raw_model = raw_model.to(device)
    train_model = torch.compile(raw_model, mode="reduce-overhead") if args.compile else raw_model
    if record_initialization and args.resume is None:
        torch.cuda.synchronize()
        initialization_started = time.perf_counter()
    classifier_values = initialize_classifier_values(
        labels=classifier_shape[0],
        mode=args.classifier_init,
        imprinted=lambda: imprinted_classifier_values(
            raw_model,
            optimization,
            labels,
            eval_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        ),
    )
    initialization_seconds = None
    if record_initialization and args.resume is None:
        torch.cuda.synchronize()
        initialization_seconds = float(time.perf_counter() - initialization_started)
    classifier = torch.nn.Parameter(classifier_values.to(device))
    step_ema = StepEMA(raw_model, classifier)
    sampler = PaddedEpochSampler(size=len(optimization), batch_size=args.batch_size, seed=args.seed)
    data_generator = torch.Generator().manual_seed(experiment_stream_seed(args.seed, 2_000))
    loader = torch.utils.data.DataLoader(
        InshopTrainDataset(optimization, labels, build_train_transform(336)),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=data_generator,
    )
    if record_initialization:
        bind_initialization_receipt(
            output=args.output_dir / "initialization-receipt.json",
            resume=args.resume is not None,
            seed=args.seed,
            classifier_init=args.classifier_init,
            classifier=classifier,
            optimizer_steps_per_epoch=len(loader),
            initialization_seconds=initialization_seconds,
            trainer_sha256=_sha256_file(Path(__file__)),
            expected_shape=classifier_shape,
        )
    optimizer = build_optimizer(
        raw_model,
        classifier,
        learning_rate=args.learning_rate,
        classifier_learning_rate=args.classifier_learning_rate,
        fused=args.fused,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.learning_rate, args.classifier_learning_rate],
        steps_per_epoch=len(loader),
        epochs=args.epochs,
        pct_start=0.1,
    )
    mask_generator = torch.Generator(device=device).manual_seed(
        experiment_stream_seed(args.seed, 3_000)
    )
    scaler = None if args.bf16 else torch.amp.GradScaler("cuda", growth_interval=200)
    start_epoch = 0
    history: list[dict[str, object]] = []
    selection_holdout = {
        "seed": args.holdout_seed,
        "fraction": args.holdout_fraction,
    }
    training_protocol: dict[str, object] = {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": _sha256_file(Path(__file__)),
        "unicom_revision": UNICOM_REVISION,
        "initial_checkpoint_sha256": UNICOM_L14_336_SHA256,
        "partition_sha256": _sha256_file(args.dataset_root / "Eval" / "list_eval_partition.txt"),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "margin": args.margin,
        "scale": args.scale,
        "objective": args.objective,
        "selected_features": args.selected_features,
        "evaluation_features": evaluation_features,
        "holdout_seed": args.holdout_seed,
        "holdout_fraction": args.holdout_fraction,
        "eval_every": args.eval_every,
        "checkpoint_every": args.checkpoint_every,
        "max_steps": args.max_steps,
        "bf16": args.bf16,
        "compile": args.compile,
        "fused": args.fused,
        "classifier_init": args.classifier_init,
        "ema_decay": EMA_DECAY,
        "ema_update": "optimizer-step-post-hook-trainable-parameters-only",
    }
    if args.resume is not None:
        start_epoch, history = restore_training_checkpoint(
            args.resume,
            raw_model=raw_model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            mask_generator=mask_generator,
            device=device,
            selection_holdout=selection_holdout,
            training_protocol=training_protocol,
            step_ema=step_ema,
        )
        if start_epoch >= args.epochs:
            raise ValueError("resume checkpoint already reached requested epochs")
    if args.run_receipt is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started_unix_ns = time.time_ns()
        started_counter_ns = time.perf_counter_ns()
    result = fit_model(
        raw_model=raw_model,
        train_model=train_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        mask_generator=mask_generator,
        device=device,
        epochs=args.epochs,
        start_epoch=start_epoch,
        objective=args.objective,
        selected_features=args.selected_features,
        margin=args.margin,
        scale=args.scale,
        max_steps=args.max_steps,
        bf16=args.bf16,
        scaler=scaler,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        evaluate=lambda _epoch: evaluate_holdout(
            raw_model,
            query,
            gallery,
            eval_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            evaluation_features=evaluation_features,
        ),
        history=history,
        selection_holdout=selection_holdout,
        training_protocol=training_protocol,
        step_ema=step_ema,
    )
    if args.run_receipt is not None:
        torch.cuda.synchronize()
        args._training_run_measurement = {
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": time.time_ns(),
            "elapsed_seconds": float(
                (time.perf_counter_ns() - started_counter_ns) / 1_000_000_000
            ),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    return result


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    import numpy as np

    np.random.seed(worker_seed)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument(
        "--objective",
        choices=("official-eight-mask", "official-one-mask", "prefix-512"),
        default="official-eight-mask",
    )
    parser.add_argument("--selected-features", type=int, default=512)
    parser.add_argument("--evaluation-features", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--classifier-init", choices=("random", "imprinted"), default="random")
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--run-arm", choices=tuple(FULL_WIDTH_ARM_PROTOCOLS))
    parser.add_argument("--run-receipt", type=Path)
    return parser.parse_args(arguments)


def _validate_run_receipt_request(args: argparse.Namespace) -> None:
    values = (args.run_config, args.run_arm, args.run_receipt)
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError("run receipt arguments must be supplied together")
    if not args.run_config.is_file() or args.run_config.is_symlink():
        raise ValueError("run configuration must be a real file")
    if args.run_receipt.exists() or args.run_receipt.is_symlink():
        raise FileExistsError(args.run_receipt)
    if not args.run_receipt.parent.is_dir() or args.run_receipt.parent.is_symlink():
        raise ValueError("run receipt parent must be a real directory")
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    if not args.output_dir.parent.is_dir() or args.output_dir.parent.is_symlink():
        raise ValueError("training output parent must be a real directory")
    expected = FULL_WIDTH_ARM_PROTOCOLS[args.run_arm]
    observed = (args.objective, args.selected_features, args.evaluation_features)
    if observed != expected or any(
        type(value) is not type(reference)
        for value, reference in zip(observed, expected, strict=True)
    ):
        raise ValueError("run arm protocol differs")
    if (
        args.classifier_init != "imprinted"
        or args.epochs != 16
        or args.eval_every != 4
        or args.checkpoint_every != 4
        or args.resume is not None
    ):
        raise ValueError("prospective run receipt execution differs")


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        _validate_run_receipt_request(args)
        history = run(args)
    except Exception as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 2
    summary = args.output_dir / "history.json"
    summary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    if args.run_receipt is not None:
        try:
            measurement = args._training_run_measurement
            command = (
                list(sys.orig_argv)
                if arguments is None
                else [sys.executable, str(Path(__file__).resolve()), *arguments]
            )
            receipt = training_run_receipt(
                source_commit=_git_revision(Path(__file__).resolve().parents[1]),
                config_path=str(args.run_config),
                config_sha256=_sha256_file(args.run_config),
                seed=args.seed,
                arm=args.run_arm,
                objective=args.objective,
                selected_features=args.selected_features,
                evaluation_features=args.evaluation_features,
                command=command,
                started_unix_ns=measurement["started_unix_ns"],
                finished_unix_ns=measurement["finished_unix_ns"],
                elapsed_seconds=measurement["elapsed_seconds"],
                peak_allocated_bytes=measurement["peak_allocated_bytes"],
                peak_reserved_bytes=measurement["peak_reserved_bytes"],
                exit_status=0,
                history_path=summary,
                checkpoint_paths=tuple(
                    args.output_dir / f"epoch-{epoch:04d}.pt"
                    for epoch in (4, 8, 12, 16)
                ),
                runtime={
                    "python": sys.version.split()[0],
                    "torch": str(torch.__version__),
                    "cuda": str(torch.version.cuda),
                },
            )
            write_training_run_receipt_atomic(receipt, args.run_receipt)
        except Exception as error:
            print(f"training receipt failed: {error}", file=sys.stderr)
            return 2
    print(f"training complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
