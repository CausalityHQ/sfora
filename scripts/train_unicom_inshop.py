#!/usr/bin/env python3
"""Train the pinned UNICOM ViT-L/14@336 In-Shop recipe on one device."""

from __future__ import annotations

import argparse
import errno
import hashlib
import importlib
import io
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

if __name__ == "__main__":
    _bootstrap_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if _bootstrap_workspace not in (None, ":4096:8"):
        print(
            "training failed: CUBLAS deterministic workspace authority differs",
            file=sys.stderr,
        )
        raise SystemExit(2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from sfora.atomic_publication import (
    BudgetedPublisher,
    publish_bytes_noreplace,
    publish_writer_noreplace,
)
from sfora.cuda_authority import canonical_cuda_device_uuid
from sfora.unicom_fepf import (
    FepfExpectedProvenance,
    InitializationRngAudit,
    build_fepf_cache,
    canonical_initialization_receipt_v2_sha256,
    fit_fepf_head,
    initialization_receipt_v2,
    prepare_registered_fepf_evidence,
    validate_initialization_receipt_v2,
)
from sfora.unicom_inshop import InshopRecord
from sfora.unicom_retrieval_audit import (
    l2_normalize,
    retrieval_view,
    strict_typed_equal,
    validate_evaluation_evidence,
    write_evaluation_evidence,
)
from sfora.unicom_training import (
    experiment_stream_seed,
    identity_holdout,
    padded_epoch_indices,
    sample_shard_masks,
    sharded_mask_arcface_loss,
)

UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
UNICOM_L14_336_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
INSHOP_PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
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
INFERENCE_OPERATIONS = (
    "official_forward",
    "full768_l2",
    "prefix512",
    "squared_euclidean",
)
FEPF_TRAINING_PROTOCOL_KEYS = (
    "protocol", "trainer_sha256", "unicom_revision", "initial_checkpoint_sha256",
    "partition_sha256", "seed", "epochs", "batch_size", "workers", "learning_rate",
    "classifier_learning_rate", "margin", "scale", "objective", "selected_features",
    "evaluation_features", "holdout_seed", "holdout_fraction", "eval_every",
    "checkpoint_every", "max_steps", "bf16", "compile", "fused", "classifier_init",
    "ema_decay", "ema_update", "initialization_receipt_sha256",
    "environment", "environment_sha256",
)
REGISTERED_ENVIRONMENT_KEYS = (
    "python_vv", "torch", "torchvision", "timm", "numpy", "cuda", "cudnn",
    "compile", "device_uuid", "gpu_inventory", "pyproject_sha256",
    "uv_lock_sha256", "deterministic_execution",
)
TRAINING_CHECKPOINT_KEYS = (
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
TORCH_DTYPE_ELEMENT_SIZES = {
    "torch.bool": 1,
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.int16": 2,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.float32": 4,
    "torch.int64": 8,
    "torch.float64": 8,
    "torch.complex64": 8,
    "torch.complex128": 16,
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


def _canonical_tensor_bytes(value: torch.Tensor) -> bytes:
    if not isinstance(value, torch.Tensor):
        raise ValueError("inference tensor differs")
    return (
        value.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )


def build_inference_signature(
    raw_model: torch.nn.Module, *, descriptor: torch.Tensor
) -> dict[str, object]:
    """Bind the deployed raw-backbone inventory and one descriptor payload."""

    if not isinstance(raw_model, torch.nn.Module) or not isinstance(descriptor, torch.Tensor):
        raise TypeError("inference signature input differs")
    if (
        descriptor.dtype != torch.float32
        or descriptor.ndim != 2
        or descriptor.shape[1] != 512
        or not torch.isfinite(descriptor).all()
    ):
        raise ValueError("inference descriptor differs")
    signature = _build_inference_signature_unchecked(raw_model, descriptor=descriptor)
    validate_inference_signature(signature, raw_model=raw_model, descriptor=descriptor)
    return signature


def validate_inference_signature(
    signature: object,
    *,
    raw_model: torch.nn.Module,
    descriptor: torch.Tensor,
) -> None:
    """Rehash one same-arm deployed model and descriptor exactly."""

    if type(signature) is not dict or tuple(signature) != (
        "schema",
        "tensors",
        "total_bytes",
        "aggregate_sha256",
        "descriptor_dtype",
        "descriptor_dimension",
        "descriptor_sha256",
        "operations",
    ):
        raise ValueError("inference signature authenticity differs")
    if signature["schema"] != "unicom-inference-signature-v1":
        raise ValueError("inference signature authenticity differs")
    expected = _build_inference_signature_unchecked(raw_model, descriptor=descriptor)
    if signature != expected:
        raise ValueError("inference signature authenticity differs")


def _build_inference_signature_unchecked(
    raw_model: torch.nn.Module, *, descriptor: torch.Tensor
) -> dict[str, object]:
    """Implementation split used to avoid recursive public validation."""

    parameters = dict(raw_model.named_parameters())
    buffers = dict(raw_model.named_buffers())
    state = raw_model.state_dict()
    if set(parameters) & set(buffers) or set(state) != set(parameters) | set(buffers):
        raise ValueError("inference state inventory differs")
    aggregate = hashlib.sha256()
    tensors = []
    total_bytes = 0
    for name in sorted(state):
        value = state[name]
        payload = _canonical_tensor_bytes(value)
        row = {
            "name": name,
            "kind": "parameter" if name in parameters else "buffer",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "element_size": value.element_size(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        metadata = json.dumps(
            {key: row[key] for key in tuple(row)[:-1]}, separators=(",", ":")
        ).encode()
        aggregate.update(len(metadata).to_bytes(8, "big"))
        aggregate.update(metadata)
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
        total_bytes += len(payload)
        tensors.append(row)
    descriptor_payload = _canonical_tensor_bytes(descriptor)
    return {
        "schema": "unicom-inference-signature-v1",
        "tensors": tensors,
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "descriptor_dtype": str(descriptor.dtype),
        "descriptor_dimension": descriptor.shape[1],
        "descriptor_sha256": hashlib.sha256(descriptor_payload).hexdigest(),
        "operations": list(INFERENCE_OPERATIONS),
    }

def require_cross_arm_inference_equality(left: object, right: object) -> None:
    """Compare deployment structure while intentionally ignoring trained values."""

    _validate_inference_signature_object(left)
    _validate_inference_signature_object(right)

    def structure(signature: object) -> tuple[object, ...]:
        if type(signature) is not dict or tuple(signature) != (
            "schema",
            "tensors",
            "total_bytes",
            "aggregate_sha256",
            "descriptor_dtype",
            "descriptor_dimension",
            "descriptor_sha256",
            "operations",
        ):
            raise ValueError("cross-arm inference signature differs")
        tensors = signature["tensors"]
        if type(tensors) is not list:
            raise ValueError("cross-arm inference signature differs")
        rows = []
        for row in tensors:
            if type(row) is not dict or tuple(row) != (
                "name",
                "kind",
                "shape",
                "dtype",
                "numel",
                "element_size",
                "bytes",
                "sha256",
            ):
                raise ValueError("cross-arm inference signature differs")
            rows.append(tuple(row[key] for key in tuple(row)[:-1]))
        return (
            signature["schema"],
            tuple(rows),
            signature["total_bytes"],
            signature["descriptor_dtype"],
            signature["descriptor_dimension"],
            tuple(signature["operations"]),
        )

    if structure(left) != structure(right):
        raise ValueError("cross-arm inference signature differs")


def raw_backbone_state_sha256(raw_model: torch.nn.Module) -> str:
    """Hash exact sorted parameter-and-buffer metadata and tensor bytes."""

    descriptor = torch.zeros((1, 512), dtype=torch.float32)
    return _build_inference_signature_unchecked(
        raw_model, descriptor=descriptor
    )["aggregate_sha256"]


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


def _rooted_file_binding(path: Path, *, root_name: str, root: Path) -> dict[str, object]:
    if root_name not in {"current", "parent"}:
        raise ValueError("FEPF evidence root differs")
    root = root.resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("FEPF evidence path differs") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("FEPF evidence path differs")
    binding = _file_binding(path)
    return {
        "root": root_name,
        "path": relative.as_posix(),
        "sha256": binding["sha256"],
        "bytes": binding["bytes"],
    }


def training_run_receipt_v2(
    *,
    output_dir: Path,
    mode: str,
    training_seed: int,
    holdout_fraction: float,
    holdout_seed: int,
    training_protocol: dict[str, object],
    stop_after_epoch: int,
    initialization_receipt_path: Path,
    history_path: Path,
    checkpoint_paths: Mapping[int, Path],
    evaluation_receipt_paths: Mapping[int, Path],
    raw_backbone_pre_initialization_sha256: str,
    raw_backbone_pre_training_sha256: str,
    inference_signature: dict[str, object],
    parent_run_receipt_path: Path | None = None,
    parent_checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Build one authenticated fresh or continuation FEPF run receipt."""

    if not isinstance(output_dir, Path) or not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("FEPF evidence root differs")
    current_root = output_dir.resolve()
    continuation = parent_run_receipt_path is not None or parent_checkpoint_path is not None
    if continuation != (parent_run_receipt_path is not None and parent_checkpoint_path is not None):
        raise ValueError("FEPF parent evidence differs")
    if continuation:
        if stop_after_epoch != 16:
            raise ValueError("FEPF continuation stop differs")
        parent_root = parent_run_receipt_path.parent.resolve()
        if (
            parent_checkpoint_path.parent.resolve() != parent_root
            or initialization_receipt_path.parent.resolve() != parent_root
            or parent_root.parent != current_root.parent
            or parent_root == current_root
        ):
            raise ValueError("FEPF parent evidence root differs")
        relative_parent = Path(os.path.relpath(parent_root, current_root))
        if relative_parent.is_absolute() or relative_parent == Path("."):
            raise ValueError("FEPF parent evidence root differs")
        parent_evidence_root: dict[str, object] | None = {
            "kind": "relative",
            "path": relative_parent.as_posix(),
        }
        parent_run = _rooted_file_binding(
            parent_run_receipt_path, root_name="parent", root=parent_root
        )
        parent_checkpoint = _rooted_file_binding(
            parent_checkpoint_path, root_name="parent", root=parent_root
        )
        initialization_root_name = "parent"
        initialization_root = parent_root
    else:
        parent_root = None
        parent_evidence_root = None
        parent_run = None
        parent_checkpoint = None
        initialization_root_name = "current"
        initialization_root = current_root
    initialization_object = strict_json_object(initialization_receipt_path.read_bytes())
    initialization_digest = canonical_initialization_receipt_v2_sha256(initialization_object)
    expected_epochs = (4,) if stop_after_epoch == 4 else (4, 8, 12, 16)
    if (
        type(checkpoint_paths) is not dict
        or tuple(checkpoint_paths) != expected_epochs
        or (continuation and checkpoint_paths[4].resolve() != parent_checkpoint_path.resolve())
    ):
        raise ValueError("FEPF checkpoint inventory differs")
    if (
        type(evaluation_receipt_paths) is not dict
        or tuple(evaluation_receipt_paths) != expected_epochs
    ):
        raise ValueError("FEPF evaluation inventory differs")
    checkpoints = []
    for epoch, path in checkpoint_paths.items():
        root_name = "parent" if continuation and epoch == 4 else "current"
        root = parent_root if root_name == "parent" else current_root
        checkpoints.append(
            {
                "epoch": epoch,
                **_rooted_file_binding(path, root_name=root_name, root=root),
            }
        )
    evaluations = []
    for epoch, path in evaluation_receipt_paths.items():
        root_name = "parent" if continuation and epoch == 4 else "current"
        root = parent_root if root_name == "parent" else current_root
        evaluations.append(
            {
                "epoch": epoch,
                **_rooted_file_binding(path, root_name=root_name, root=root),
            }
        )
    receipt = {
        "schema": "unicom-fepf-training-run-receipt-v2",
        "mode": mode,
        "training_seed": training_seed,
        "holdout_fraction": holdout_fraction,
        "holdout_seed": holdout_seed,
        "training_protocol": training_protocol,
        "stop_after_epoch": stop_after_epoch,
        "initialization_receipt_sha256": initialization_digest,
        "raw_backbone_pre_initialization_sha256": raw_backbone_pre_initialization_sha256,
        "raw_backbone_pre_training_sha256": raw_backbone_pre_training_sha256,
        "parent_evidence_root": parent_evidence_root,
        "parent_run_receipt": parent_run,
        "parent_checkpoint": parent_checkpoint,
        "initialization_receipt": _rooted_file_binding(
            initialization_receipt_path,
            root_name=initialization_root_name,
            root=initialization_root,
        ),
        "history": _rooted_file_binding(
            history_path, root_name="current", root=current_root
        ),
        "checkpoints": checkpoints,
        "evaluations": evaluations,
        "inference_signature": inference_signature,
    }
    validate_training_run_receipt_v2(receipt, evidence_root=output_dir)
    return receipt


def _resolve_receipt_binding(
    binding: object, *, current_root: Path, parent_root: Path | None
) -> Path:
    if type(binding) is not dict or tuple(binding) != (
        "root",
        "path",
        "sha256",
        "bytes",
    ):
        raise ValueError("FEPF artifact binding differs")
    root_name = binding["root"]
    if root_name == "current":
        root = current_root
    elif root_name == "parent" and parent_root is not None:
        root = parent_root
    else:
        raise ValueError("FEPF artifact root differs")
    relative = Path(binding["path"])
    if (
        type(binding["path"]) is not str
        or not binding["path"]
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("FEPF artifact path differs")
    unresolved = root
    for part in relative.parts:
        unresolved = unresolved / part
        if unresolved.is_symlink():
            raise ValueError("FEPF artifact path differs")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("FEPF artifact path differs") from error
    if (
        not resolved.is_file()
        or not _lower_sha256(binding["sha256"])
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
        or resolved.stat().st_size != binding["bytes"]
        or _sha256_file(resolved) != binding["sha256"]
    ):
        raise ValueError("FEPF artifact bytes differ")
    return resolved


def _validate_inference_signature_object(signature: object) -> None:
    if type(signature) is not dict or tuple(signature) != (
        "schema", "tensors", "total_bytes", "aggregate_sha256",
        "descriptor_dtype", "descriptor_dimension", "descriptor_sha256", "operations",
    ):
        raise ValueError("FEPF inference signature differs")
    if (
        signature["schema"] != "unicom-inference-signature-v1"
        or not _lower_sha256(signature["aggregate_sha256"])
        or not _lower_sha256(signature["descriptor_sha256"])
        or signature["descriptor_dtype"] != "torch.float32"
        or signature["descriptor_dimension"] != 512
        or signature["operations"] != list(INFERENCE_OPERATIONS)
    ):
        raise ValueError("FEPF inference signature differs")
    tensors = signature["tensors"]
    if type(tensors) is not list or any(
        type(row) is not dict
        or tuple(row) != (
            "name", "kind", "shape", "dtype", "numel", "element_size", "bytes", "sha256"
        )
        for row in tensors
    ):
        raise ValueError("FEPF inference inventory differs")
    names = [row["name"] for row in tensors]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("FEPF inference inventory differs")
    total = 0
    for row in tensors:
        if (
            type(row["name"]) is not str
            or not row["name"]
            or row["kind"] not in {"parameter", "buffer"}
            or type(row["dtype"]) is not str
            or TORCH_DTYPE_ELEMENT_SIZES.get(row["dtype"]) != row["element_size"]
            or type(row["shape"]) is not list
            or any(type(value) is not int or value < 0 for value in row["shape"])
            or type(row["numel"]) is not int
            or type(row["element_size"]) is not int
            or type(row["bytes"]) is not int
            or row["numel"] < 0
            or math.prod(row["shape"]) != row["numel"]
            or row["element_size"] <= 0
            or row["bytes"] != row["numel"] * row["element_size"]
            or not _lower_sha256(row["sha256"])
        ):
            raise ValueError("FEPF inference inventory differs")
        total += row["bytes"]
    if signature["total_bytes"] != total:
        raise ValueError("FEPF inference byte count differs")


def _validate_evaluation_signature_against_inference(
    evaluation_receipt: Mapping[str, object],
    inference_signature: object,
    *,
    require_descriptor: bool,
) -> None:
    _validate_inference_signature_object(inference_signature)
    evaluation_signature = evaluation_receipt.get("evaluation_signature")
    if (
        type(evaluation_signature) is not dict
        or evaluation_signature.get("descriptor_dtype") != "float32"
        or evaluation_signature.get("descriptor_dimension")
        != inference_signature["descriptor_dimension"]
        or evaluation_signature.get("operations") != inference_signature["operations"]
        or inference_signature["descriptor_dtype"] != "torch.float32"
        or (
            require_descriptor
            and evaluation_signature.get("descriptor_sha256")
            != inference_signature["descriptor_sha256"]
        )
    ):
        raise ValueError("FEPF evaluation inference signature differs")


def validate_fepf_training_protocol(
    protocol: object, *, receipt: Mapping[str, object]
) -> None:
    """Validate the complete checkpoint-bound registered training matrix."""

    if type(protocol) is not dict or tuple(protocol) != FEPF_TRAINING_PROTOCOL_KEYS:
        raise ValueError("FEPF checkpoint protocol differs")
    runtime = (protocol.get("compile"), protocol.get("fused"))
    if runtime == (False, False):
        expected_ema = (EMA_DECAY, "optimizer-step-post-hook-trainable-parameters-only")
    elif runtime == (True, True):
        expected_ema = (None, None)
    else:
        raise ValueError("FEPF checkpoint runtime protocol differs")
    expected = (
        "unicom-inshop-official-single-device-v1", UNICOM_REVISION,
        UNICOM_L14_336_SHA256, receipt["training_seed"], 16, 128, 4, 1e-5, 1e-4,
        0.25, 32.0, "official-eight-mask", 512, 512, receipt["holdout_seed"],
        receipt["holdout_fraction"], 4, 4, None, False, receipt["mode"], *expected_ema,
        receipt["initialization_receipt_sha256"],
    )
    observed = (
        protocol["protocol"], protocol["unicom_revision"],
        protocol["initial_checkpoint_sha256"], protocol["seed"], protocol["epochs"],
        protocol["batch_size"], protocol["workers"], protocol["learning_rate"],
        protocol["classifier_learning_rate"], protocol["margin"], protocol["scale"],
        protocol["objective"], protocol["selected_features"],
        protocol["evaluation_features"], protocol["holdout_seed"],
        protocol["holdout_fraction"], protocol["eval_every"],
        protocol["checkpoint_every"], protocol["max_steps"], protocol["bf16"],
        protocol["classifier_init"], protocol["ema_decay"], protocol["ema_update"],
        protocol["initialization_receipt_sha256"],
    )
    if observed != expected or any(
        type(value) is not type(reference)
        for value, reference in zip(observed, expected, strict=True)
    ):
        raise ValueError("FEPF checkpoint protocol differs")
    if (
        protocol["trainer_sha256"] != _sha256_file(Path(__file__))
        or protocol["partition_sha256"] != INSHOP_PARTITION_SHA256
        or type(protocol["compile"]) is not bool
        or type(protocol["fused"]) is not bool
    ):
        raise ValueError("FEPF checkpoint protocol differs")
    validate_registered_environment_payload(protocol["environment"])
    if protocol["environment_sha256"] != hashlib.sha256(
        (json.dumps(protocol["environment"], indent=2, allow_nan=False) + "\n").encode()
    ).hexdigest():
        raise ValueError("FEPF checkpoint environment differs")


def _load_checkpoint_training_protocol(
    path: Path, *, expected_epoch: int
) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("FEPF checkpoint payload differs") from error
    if type(checkpoint) is not dict or tuple(checkpoint) != TRAINING_CHECKPOINT_KEYS:
        raise ValueError("FEPF checkpoint schema differs")
    if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] != expected_epoch:
        raise ValueError("FEPF checkpoint epoch differs")
    if type(checkpoint["training_protocol"]) is not dict:
        raise ValueError("FEPF checkpoint protocol differs")
    return checkpoint["training_protocol"]


def validate_training_run_receipt_v2(
    receipt: object, *, evidence_root: Path
) -> None:
    """Rehash every current/parent artifact in one FEPF evidence chain."""

    keys = (
        "schema",
        "mode",
        "training_seed",
        "holdout_fraction",
        "holdout_seed",
        "training_protocol",
        "stop_after_epoch",
        "initialization_receipt_sha256",
        "raw_backbone_pre_initialization_sha256",
        "raw_backbone_pre_training_sha256",
        "parent_evidence_root",
        "parent_run_receipt",
        "parent_checkpoint",
        "initialization_receipt",
        "history",
        "checkpoints",
        "evaluations",
        "inference_signature",
    )
    if (
        type(receipt) is not dict
        or tuple(receipt) != keys
        or receipt["schema"] != "unicom-fepf-training-run-receipt-v2"
        or receipt["mode"] not in {"imprinted", "fepf_mean", "fepf_random"}
        or type(receipt["training_seed"]) is not int
        or receipt["training_seed"] < 0
        or (receipt["mode"] == "fepf_random" and receipt["training_seed"] != 0)
        or type(receipt["holdout_fraction"]) is not float
        or not math.isfinite(receipt["holdout_fraction"])
        or type(receipt["holdout_seed"]) is not int
        or receipt["holdout_seed"] < 0
        or receipt["stop_after_epoch"] not in {4, 16}
        or not _lower_sha256(receipt["initialization_receipt_sha256"])
        or not _lower_sha256(receipt["raw_backbone_pre_initialization_sha256"])
        or receipt["raw_backbone_pre_initialization_sha256"]
        != receipt["raw_backbone_pre_training_sha256"]
    ):
        raise ValueError("FEPF run receipt differs")
    if not isinstance(evidence_root, Path):
        raise ValueError("FEPF evidence root differs")
    absolute_root = evidence_root.absolute()
    current_root = evidence_root.resolve()
    if absolute_root != current_root or not current_root.is_dir() or current_root.is_symlink():
        raise ValueError("FEPF evidence root differs")
    parent_spec = receipt["parent_evidence_root"]
    continuation = parent_spec is not None
    if continuation:
        if (
            type(parent_spec) is not dict
            or tuple(parent_spec) != ("kind", "path")
            or parent_spec["kind"] != "relative"
            or type(parent_spec["path"]) is not str
        ):
            raise ValueError("FEPF parent evidence root differs")
        relative_parent = Path(parent_spec["path"])
        if relative_parent.is_absolute() or relative_parent == Path("."):
            raise ValueError("FEPF parent evidence root differs")
        unresolved_parent = current_root / relative_parent
        parent_root = unresolved_parent.resolve()
        if (
            not parent_root.is_dir()
            or parent_root.is_symlink()
            or parent_root.parent != current_root.parent
            or parent_root == current_root
            or relative_parent != Path("..") / parent_root.name
        ):
            raise ValueError("FEPF parent evidence root differs")
    else:
        parent_root = None
        if receipt["parent_run_receipt"] is not None or receipt["parent_checkpoint"] is not None:
            raise ValueError("FEPF parent evidence differs")
    protocol = receipt["training_protocol"]
    validate_fepf_training_protocol(protocol, receipt=receipt)
    parent_receipt: dict[str, object] | None = None
    if continuation:
        if (
            type(receipt["parent_run_receipt"]) is not dict
            or receipt["parent_run_receipt"].get("root") != "parent"
            or receipt["parent_run_receipt"].get("path") != "run-receipt.json"
        ):
            raise ValueError("FEPF parent artifact root differs")
        parent_run_path = _resolve_receipt_binding(
            receipt["parent_run_receipt"],
            current_root=current_root,
            parent_root=parent_root,
        )
        parent_receipt = strict_json_object(parent_run_path.read_bytes())
        validate_training_run_receipt_v2(parent_receipt, evidence_root=parent_root)
        expected_initialization = dict(parent_receipt["initialization_receipt"])
        expected_initialization["root"] = "parent"
        terminal = parent_receipt["checkpoints"][-1]
        expected_checkpoint = {
            "root": "parent",
            "path": terminal["path"],
            "sha256": terminal["sha256"],
            "bytes": terminal["bytes"],
        }
        child_first = receipt["checkpoints"][0] if type(receipt["checkpoints"]) is list else None
        terminal_evaluation = parent_receipt["evaluations"][-1]
        expected_evaluation = {
            "root": "parent",
            "path": terminal_evaluation["path"],
            "sha256": terminal_evaluation["sha256"],
            "bytes": terminal_evaluation["bytes"],
        }
        child_first_evaluation = (
            receipt["evaluations"][0]
            if type(receipt["evaluations"]) is list and receipt["evaluations"]
            else None
        )
        if (
            parent_receipt["mode"] != receipt["mode"]
            or parent_receipt["training_seed"] != receipt["training_seed"]
            or parent_receipt["holdout_fraction"] != receipt["holdout_fraction"]
            or parent_receipt["holdout_seed"] != receipt["holdout_seed"]
            or parent_receipt["training_protocol"] != protocol
            or parent_receipt["stop_after_epoch"] != 4
            or receipt["stop_after_epoch"] != 16
            or receipt["initialization_receipt"] != expected_initialization
            or receipt["parent_checkpoint"] != expected_checkpoint
            or type(child_first) is not dict
            or {key: child_first[key] for key in tuple(child_first)[1:]}
            != expected_checkpoint
            or type(child_first_evaluation) is not dict
            or {
                key: child_first_evaluation[key]
                for key in tuple(child_first_evaluation)[1:]
            }
            != expected_evaluation
        ):
            raise ValueError("FEPF parent run substitution differs")
    initialization_path = _resolve_receipt_binding(
        receipt["initialization_receipt"],
        current_root=current_root,
        parent_root=parent_root,
    )
    if receipt["initialization_receipt"]["root"] != (
        "parent" if continuation else "current"
    ) or receipt["initialization_receipt"].get("path") != "initialization-receipt.json":
        raise ValueError("FEPF initialization root differs")
    initialization_object = strict_json_object(initialization_path.read_bytes())
    if (
        canonical_initialization_receipt_v2_sha256(initialization_object)
        != receipt["initialization_receipt_sha256"]
    ):
        raise ValueError("FEPF initialization digest differs")
    if (
        receipt["history"].get("root") != "current"
        or receipt["history"].get("path") != "history.json"
    ):
        raise ValueError("FEPF history root differs")
    _resolve_receipt_binding(receipt["history"], current_root=current_root, parent_root=parent_root)
    checkpoints = receipt["checkpoints"]
    expected_epochs = (4, 8, 12, 16) if continuation else (
        (4,) if receipt["stop_after_epoch"] == 4 else (4, 8, 12, 16)
    )
    if (
        type(checkpoints) is not list
        or tuple(row.get("epoch") for row in checkpoints) != expected_epochs
    ):
        raise ValueError("FEPF checkpoint inventory differs")
    checkpoint_paths = []
    for index, row in enumerate(checkpoints):
        if type(row) is not dict or tuple(row) != ("epoch", "root", "path", "sha256", "bytes"):
            raise ValueError("FEPF checkpoint binding differs")
        expected_root = "parent" if continuation and index == 0 else "current"
        if row["root"] != expected_root or row["path"] != f"epoch-{row['epoch']:04d}.pt":
            raise ValueError("FEPF checkpoint root differs")
        checkpoint_paths.append(
            _resolve_receipt_binding(
                {key: row[key] for key in tuple(row)[1:]},
                current_root=current_root,
                parent_root=parent_root,
            )
        )
        if (
            _load_checkpoint_training_protocol(
                checkpoint_paths[-1], expected_epoch=row["epoch"]
            )
            != protocol
        ):
            raise ValueError("FEPF checkpoint protocol differs")
    evaluations = receipt["evaluations"]
    if (
        type(evaluations) is not list
        or tuple(row.get("epoch") for row in evaluations) != expected_epochs
    ):
        raise ValueError("FEPF evaluation inventory differs")
    for index, row in enumerate(evaluations):
        if type(row) is not dict or tuple(row) != (
            "epoch",
            "root",
            "path",
            "sha256",
            "bytes",
        ):
            raise ValueError("FEPF evaluation binding differs")
        expected_root = "parent" if continuation and index == 0 else "current"
        if (
            row["root"] != expected_root
            or row["path"] != f"evaluation-epoch-{row['epoch']:04d}.json"
        ):
            raise ValueError("FEPF evaluation root differs")
        evaluation_path = _resolve_receipt_binding(
            {key: row[key] for key in tuple(row)[1:]},
            current_root=current_root,
            parent_root=parent_root,
        )
        evaluation_receipt = strict_json_object(evaluation_path.read_bytes())
        validate_evaluation_evidence(
            evaluation_receipt,
            parent_root if expected_root == "parent" else current_root,
        )
        if evaluation_receipt["epoch"] != row["epoch"]:
            raise ValueError("FEPF evaluation epoch differs")
        _validate_evaluation_signature_against_inference(
            evaluation_receipt,
            receipt["inference_signature"],
            require_descriptor=row["epoch"] == receipt["stop_after_epoch"],
        )
    if continuation:
        if (
            type(receipt["parent_checkpoint"]) is not dict
            or receipt["parent_checkpoint"].get("root") != "parent"
        ):
            raise ValueError("FEPF parent artifact root differs")
        parent_checkpoint_path = _resolve_receipt_binding(
            receipt["parent_checkpoint"],
            current_root=current_root,
            parent_root=parent_root,
        )
        if checkpoint_paths[0] != parent_checkpoint_path:
            raise ValueError("FEPF parent checkpoint differs")
        require_cross_arm_inference_equality(
            parent_receipt["inference_signature"], receipt["inference_signature"]
        )
        if (
            parent_receipt["mode"] != receipt["mode"]
            or parent_receipt["training_seed"] != receipt["training_seed"]
            or parent_receipt["holdout_fraction"] != receipt["holdout_fraction"]
            or parent_receipt["holdout_seed"] != receipt["holdout_seed"]
            or parent_receipt["training_protocol"] != receipt["training_protocol"]
            or parent_receipt["stop_after_epoch"] != 4
            or parent_receipt["initialization_receipt_sha256"]
            != receipt["initialization_receipt_sha256"]
            or parent_receipt["raw_backbone_pre_initialization_sha256"]
            != receipt["raw_backbone_pre_initialization_sha256"]
            or parent_receipt["raw_backbone_pre_training_sha256"]
            != receipt["raw_backbone_pre_training_sha256"]
            or parent_receipt["checkpoints"][-1]["sha256"]
            != receipt["parent_checkpoint"]["sha256"]
        ):
            raise ValueError("FEPF parent run substitution differs")
    _validate_inference_signature_object(receipt["inference_signature"])


def validate_fepf_result(result: object, evidence_root: Path) -> None:
    """Validate a result only from the committed run/evaluation evidence roots."""

    if not isinstance(evidence_root, Path):
        raise ValueError("FEPF evidence root differs")
    absolute_root = evidence_root.absolute()
    current_root = evidence_root.resolve()
    run_receipt_path = current_root / "run-receipt.json"
    if (
        absolute_root != current_root
        or not current_root.is_dir()
        or current_root.is_symlink()
        or not run_receipt_path.is_file()
        or run_receipt_path.is_symlink()
    ):
        raise ValueError("FEPF result evidence root differs")
    receipt = strict_json_object(run_receipt_path.read_bytes())
    validate_training_run_receipt_v2(receipt, evidence_root=current_root)
    history_path = _resolve_receipt_binding(
        receipt["history"], current_root=current_root, parent_root=None
    )
    try:
        history = json.loads(history_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("FEPF result history differs") from error
    if (
        type(history) is not list
        or type(result) is not list
        or not strict_typed_equal(result, history)
    ):
        raise ValueError("FEPF result history differs")
    parent_root = None
    if receipt["parent_evidence_root"] is not None:
        parent_root = (
            current_root / Path(receipt["parent_evidence_root"]["path"])
        ).resolve()
    expected_metrics: dict[int, object] = {}
    for binding in receipt["evaluations"]:
        evaluation_path = _resolve_receipt_binding(
            {key: binding[key] for key in tuple(binding)[1:]},
            current_root=current_root,
            parent_root=parent_root,
        )
        evaluation = strict_json_object(evaluation_path.read_bytes())
        expected_metrics[binding["epoch"]] = evaluation["metrics"]
    observed_metrics: dict[int, object] = {}
    for row in history:
        if (
            type(row) is not dict
            or type(row.get("epoch")) is not int
            or "metrics" not in row
        ):
            raise ValueError("FEPF result history differs")
        if row["metrics"] is None:
            continue
        if row["epoch"] in observed_metrics:
            raise ValueError("FEPF result evaluation metrics differ")
        observed_metrics[row["epoch"]] = row["metrics"]
    if not strict_typed_equal(observed_metrics, expected_metrics):
        raise ValueError("FEPF result evaluation metrics differ")


def write_training_run_receipt_atomic(
    receipt: dict[str, object], output: Path, *, evidence_root: Path | None = None,
    publication_guard: Callable[[bytes], None] = lambda _payload: None,
) -> None:
    """Publish one validated run receipt without replacing an existing path."""

    if not isinstance(output, Path):
        raise TypeError("training run receipt output must be a Path")
    is_fepf = receipt.get("schema") == "unicom-fepf-training-run-receipt-v2"
    if is_fepf:
        if evidence_root is None:
            raise ValueError("FEPF evidence root is required")
        if output.name != "run-receipt.json":
            raise ValueError("FEPF run receipt path differs")
        validate_training_run_receipt_v2(receipt, evidence_root=evidence_root)
    else:
        validate_training_run_receipt(receipt)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    publication_guard(payload)
    def validate(persisted: bytes) -> None:
        if persisted != payload:
            raise RuntimeError("persisted training run receipt bytes differ")
        persisted_object = strict_json_object(persisted)
        if is_fepf:
            validate_training_run_receipt_v2(persisted_object, evidence_root=evidence_root)
        else:
            validate_training_run_receipt(persisted_object)

    published = publish_bytes_noreplace(output, payload, validator=validate)
    published.close()


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


def _global_rng_snapshot() -> tuple[object, object, torch.Tensor, tuple[torch.Tensor, ...]]:
    import numpy as np

    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else (),
    )


def _restore_global_rng_snapshot(
    snapshot: tuple[object, object, torch.Tensor, tuple[torch.Tensor, ...]],
) -> None:
    import numpy as np

    python_state, numpy_state, torch_state, cuda_states = snapshot
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("FEPF CUDA RNG inventory differs")
        torch.cuda.set_rng_state_all(list(cuda_states))


def _fepf_rng_audit(
    entry: tuple[object, object, torch.Tensor, tuple[torch.Tensor, ...]],
    post_draw: tuple[object, object, torch.Tensor, tuple[torch.Tensor, ...]],
) -> InitializationRngAudit:
    restored = _global_rng_snapshot()

    def hashes(snapshot):
        python_state, numpy_state, torch_state, cuda_states = snapshot
        return (
            _state_digest(b"python-random-v1", python_state),
            _state_digest(b"numpy-random-v1", numpy_state),
            _tensor_state_digest(b"torch-cpu-random-v1", torch_state),
            tuple(
                _tensor_state_digest(f"torch-cuda-random-v1:{index}".encode(), state)
                for index, state in enumerate(cuda_states)
            ),
        )

    entry_hashes = hashes(entry)
    post_hashes = hashes(post_draw)
    restored_hashes = hashes(restored)
    return InitializationRngAudit(
        python_rng_entry_sha256=entry_hashes[0],
        python_rng_post_draw_sha256=post_hashes[0],
        python_rng_restored_sha256=restored_hashes[0],
        numpy_rng_entry_sha256=entry_hashes[1],
        numpy_rng_post_draw_sha256=post_hashes[1],
        numpy_rng_restored_sha256=restored_hashes[1],
        torch_cpu_rng_entry_sha256=entry_hashes[2],
        torch_cpu_rng_post_draw_sha256=post_hashes[2],
        torch_cpu_rng_restored_sha256=restored_hashes[2],
        torch_cuda_rng_entry_sha256=entry_hashes[3],
        torch_cuda_rng_post_draw_sha256=post_hashes[3],
        torch_cuda_rng_restored_sha256=restored_hashes[3],
    )


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


def validate_registered_environment_payload(value: object) -> None:
    if (
        type(value) is not dict
        or tuple(value) != REGISTERED_ENVIRONMENT_KEYS
        or any(
            type(value[key]) is not str or not value[key]
            for key in (
                "python_vv", "torch", "torchvision", "timm", "numpy", "cuda",
                "cudnn", "device_uuid",
            )
        )
        or not value["device_uuid"].startswith("GPU-")
        or type(value["compile"]) is not dict
        or tuple(value["compile"]) != ("available", "inductor")
        or any(type(item) is not str or not item for item in value["compile"].values())
        or type(value["gpu_inventory"]) is not list
        or not value["gpu_inventory"]
        or any(type(item) is not str or not item for item in value["gpu_inventory"])
        or not _lower_sha256(value["pyproject_sha256"])
        or not _lower_sha256(value["uv_lock_sha256"])
        or value["deterministic_execution"]
        != {
            "deterministic_algorithms": True,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        }
    ):
        raise ValueError("registered training environment differs")


def load_registered_environment_authority(
    path: Path, expected_sha256: str
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or not _lower_sha256(expected_sha256):
        raise ValueError("registered training environment authority differs")
    payload = path.read_bytes()
    value = strict_json_object(payload)
    if (
        payload != (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("registered training environment authority differs")
    validate_registered_environment_payload(value)
    return value


def _configured_file_authority(
    config_path: Path, key: str, path: Path, expected_sha256: str
) -> tuple[dict[str, object], bytes]:
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or path.is_symlink()
        or not path.is_file()
        or not _lower_sha256(expected_sha256)
    ):
        raise ValueError(f"configured {key} authority differs")
    config_payload = config_path.read_bytes()
    config = strict_json_object(config_payload)
    payload = path.read_bytes()
    authority = config.get(key)
    expected = {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "bytes": len(payload),
    }
    if (
        config_payload != (json.dumps(config, indent=2, allow_nan=False) + "\n").encode()
        or authority != expected
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError(f"configured {key} authority differs")
    return config, payload


def load_configured_environment_authority(
    config_path: Path, path: Path, expected_sha256: str
) -> dict[str, object]:
    """Load the sole environment named by the authenticated run configuration."""

    config_payload = config_path.read_bytes()
    config = strict_json_object(config_payload)
    registered = config.get("cuda_canary_environment")
    if type(registered) is dict and tuple(registered) == ("path",):
        if registered["path"] != str(path.resolve()) or not _lower_sha256(
            expected_sha256
        ):
            raise ValueError("configured environment authority differs")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("configured environment authority differs")
    else:
        _config, payload = _configured_file_authority(
            config_path, "cuda_canary_environment", path, expected_sha256
        )
    value = strict_json_object(payload)
    if payload != (json.dumps(value, indent=2, allow_nan=False) + "\n").encode():
        raise ValueError("configured environment authority differs")
    validate_registered_environment_payload(value)
    return value


def establish_registered_deterministic_execution(
    registered: object, *, torch_module=torch
) -> dict[str, object]:
    expected = {
        "deterministic_algorithms": True,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
    }
    if registered != expected:
        raise ValueError("registered training deterministic environment differs")
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current not in (None, expected["cublas_workspace_config"]):
        raise ValueError("registered training deterministic environment differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected["cublas_workspace_config"]
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    return expected


def registered_runtime_environment(
    device: torch.device, *, torch_module=torch
) -> dict[str, object]:
    import numpy as np
    import timm
    import torchvision

    repository = Path(__file__).resolve().parents[1]
    return {
        "python_vv": subprocess.run(
            [sys.executable, "-VV"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "torch": str(torch_module.__version__),
        "torchvision": str(torchvision.__version__),
        "timm": str(timm.__version__),
        "numpy": str(np.__version__),
        "cuda": str(torch_module.version.cuda),
        "cudnn": str(torch_module.backends.cudnn.version()),
        "compile": {
            "available": str(hasattr(torch_module, "compile")),
            "inductor": str(getattr(torch_module.version, "git_version", "unknown")),
        },
        "device_uuid": canonical_cuda_device_uuid(
            torch_module.cuda.get_device_properties(device).uuid
        ),
        "gpu_inventory": subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines(),
        "pyproject_sha256": _sha256_file(repository / "pyproject.toml"),
        "uv_lock_sha256": _sha256_file(repository / "uv.lock"),
        "deterministic_execution": establish_registered_deterministic_execution(
            {
                "deterministic_algorithms": True,
                "cuda_matmul_tf32": False,
                "cudnn_tf32": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": ":4096:8",
            },
            torch_module=torch_module,
        ),
    }


def load_publication_budget_authority(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("publication budget authority differs")
    payload = path.read_bytes()
    value = strict_json_object(payload)
    if payload != (json.dumps(value, indent=2, allow_nan=False) + "\n").encode():
        raise ValueError("publication budget authority differs")
    if (
        tuple(value) != ("schema", "publications")
        or value["schema"] != "unicom-fepf-publication-budget-v1"
        or type(value["publications"]) is not list
        or not value["publications"]
    ):
        raise ValueError("publication budget authority differs")
    names = []
    for row in value["publications"]:
        if (
            type(row) is not dict
            or tuple(row) != (
                "name", "path", "persistent_bytes", "temporary_bytes",
                "persistent_inodes", "temporary_inodes",
            )
            or type(row["name"]) is not str
            or not row["name"]
            or type(row["path"]) is not str
            or not row["path"]
            or Path(row["path"]).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(row["path"]).parts)
            or any(type(row[key]) is not int or row[key] < 0 for key in tuple(row)[2:])
        ):
            raise ValueError("publication budget row differs")
        names.append(row["name"])
    if len(names) != len(set(names)):
        raise ValueError("publication budget names differ")
    return value


def load_configured_publication_budget(
    config_path: Path, path: Path, expected_sha256: str, *, external: bool = True
) -> dict[str, object]:
    """Reload the publication inventory rooted by the run configuration."""

    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("configured publication budget authority differs")
    config_payload = config_path.read_bytes()
    config = strict_json_object(config_payload)
    if type(config.get("publication_budget_path")) is not str:
        raise ValueError("configured publication budget authority differs")
    if external:
        builder_path = Path(__file__).with_name("build_unicom_fepf_run_config.py")
        specification = importlib.util.spec_from_file_location(
            "training_exact_budget_builder", builder_path
        )
        if specification is None or specification.loader is None:
            raise ValueError("configured publication budget validator differs")
        builder = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(builder)
        builder.validate_external_exact_publication_budget(
            config, config.get("publication_budget")
        )
    expected_path = (
        Path(config["artifact_root"]) / config["publication_budget_path"]
    ).resolve()
    if (
        path.resolve() != expected_path
        or expected_sha256 != config.get("publication_budget_sha256")
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("configured publication budget authority differs")
    payload = path.read_bytes()
    if (
        hashlib.sha256(payload).hexdigest() != expected_sha256
        or payload
        != (json.dumps(config["publication_budget"], indent=2, allow_nan=False)
            + "\n").encode()
    ):
        raise ValueError("configured publication budget authority differs")
    value = strict_json_object(payload)
    if payload != (json.dumps(value, indent=2, allow_nan=False) + "\n").encode():
        raise ValueError("configured publication budget authority differs")
    # Keep the schema validator single-sourced while avoiding a second file read.
    temporary_value = value
    if (
        tuple(temporary_value) != ("schema", "publications")
        or temporary_value["schema"] != "unicom-fepf-publication-budget-v1"
        or type(temporary_value["publications"]) is not list
        or not temporary_value["publications"]
    ):
        raise ValueError("publication budget authority differs")
    names = []
    for row in temporary_value["publications"]:
        if (
            type(row) is not dict
            or tuple(row)
            != (
                "name", "path", "persistent_bytes", "temporary_bytes",
                "persistent_inodes", "temporary_inodes",
            )
            or type(row["name"]) is not str
            or not row["name"]
            or type(row["path"]) is not str
            or not row["path"]
            or Path(row["path"]).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(row["path"]).parts)
            or any(type(row[key]) is not int or row[key] < 0 for key in tuple(row)[2:])
        ):
            raise ValueError("publication budget row differs")
        names.append(row["name"])
    if len(names) != len(set(names)):
        raise ValueError("publication budget names differ")
    return temporary_value


def require_named_publication_capacity(
    budget: Mapping[str, object], name: str, root: Path,
    *, statvfs: Callable[[Path], object] = os.statvfs,
) -> None:
    rows = [row for row in budget["publications"] if row["name"] == name]
    if len(rows) != 1 or root.is_symlink() or not root.is_dir():
        raise ValueError("named publication capacity differs")
    row = rows[0]
    statistics = statvfs(root)
    required_bytes = row["persistent_bytes"] + row["temporary_bytes"]
    required_inodes = row["persistent_inodes"] + row["temporary_inodes"]
    if (
        statistics.f_bavail * statistics.f_frsize < required_bytes
        or statistics.f_favail < required_inodes
    ):
        raise OSError(errno.ENOSPC, "named publication capacity differs", root)


def require_configured_publication_capacity(
    config_path: Path,
    budget_path: Path,
    budget_sha256: str,
    name: str,
    destination: Path,
    root: Path,
    *,
    payload: bytes | None = None,
    external: bool = True,
    statvfs: Callable[[Path], object] = os.statvfs,
) -> None:
    budget = load_configured_publication_budget(
        config_path, budget_path, budget_sha256, external=external
    )
    rows = [row for row in budget["publications"] if row["name"] == name]
    try:
        relative = destination.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("publication destination path differs") from error
    if len(rows) != 1 or rows[0]["path"] != relative:
        raise ValueError("publication destination path differs")
    if payload is not None and len(payload) > rows[0]["persistent_bytes"]:
        raise OSError(errno.EFBIG, "publication payload bytes exceed budget", destination)
    config = strict_json_object(config_path.read_bytes())
    BudgetedPublisher(
        campaign_root=root,
        budget_path=budget_path,
        budget_sha256=budget_sha256,
        exact_budget=config["publication_budget"],
        statvfs=statvfs,
        physical_admission=external,
    ).validate_payload(
        name=name,
        destination=destination,
        payload=b"" if payload is None else payload,
    )


def registered_training_publication_destinations(
    output_dir: Path, *, epochs: tuple[int, ...] = (4, 8, 12, 16)
) -> dict[str, Path]:
    destinations = {
        "initialization-receipt": output_dir / "initialization-receipt.json",
        "history": output_dir / "history.json",
        "run-receipt": output_dir / "run-receipt.json",
    }
    for epoch in epochs:
        stem = f"evaluation-epoch-{epoch:04d}"
        destinations[f"checkpoint-epoch-{epoch:04d}"] = (
            output_dir / f"epoch-{epoch:04d}.pt"
        )
        destinations[f"{stem}-query"] = output_dir / f"{stem}-query.npy"
        destinations[f"{stem}-gallery"] = output_dir / f"{stem}-gallery.npy"
        destinations[f"{stem}-ranked-prefix"] = (
            output_dir / f"{stem}-ranked-prefix.json"
        )
        destinations[stem] = output_dir / f"{stem}.json"
    return destinations


def _require_cli_publication_capacity(
    args: argparse.Namespace,
    name: str,
    *,
    capacity_validator: Callable[..., None] = require_configured_publication_capacity,
) -> None:
    if args.publication_budget is None:
        return
    if args.run_config is None or args.publication_budget_sha256 is None:
        raise ValueError("configured publication budget authority is required")
    stage = getattr(args, "publication_stage", None)
    root = getattr(args, "campaign_root", None) or args.output_dir
    qualified_name = f"{stage}:{name}" if stage else name
    destinations = registered_training_publication_destinations(args.output_dir)
    destination = destinations.get(name)
    if not isinstance(destination, Path):
        raise ValueError("publication destination path differs")
    capacity_validator(
        args.run_config,
        args.publication_budget,
        args.publication_budget_sha256,
        qualified_name,
        destination,
        root,
    )


def require_cli_publication_capacity(
    args: argparse.Namespace,
    name: str,
    *,
    capacity_validator: Callable[..., None] = require_configured_publication_capacity,
) -> None:
    _require_cli_publication_capacity(
        args, name, capacity_validator=capacity_validator
    )


CHECKPOINT_PUBLICATION_KEYS = (
    "epoch", "model", "classifier", "ema", "optimizer", "scheduler", "scaler",
    "mask_generator", "torch_rng_state", "cuda_rng_states", "selection_holdout",
    "training_protocol", "history",
)


def validate_checkpoint_publication(
    value: object, *, expected: Mapping[str, object]
) -> None:
    if type(value) is not dict or tuple(value) != CHECKPOINT_PUBLICATION_KEYS:
        raise ValueError("checkpoint publication schema differs")
    if tuple(expected) != CHECKPOINT_PUBLICATION_KEYS:
        raise ValueError("checkpoint expected semantic schema differs")
    def exact(observed: object, reference: object) -> bool:
        if isinstance(reference, torch.Tensor):
            return (
                isinstance(observed, torch.Tensor)
                and observed.dtype == reference.dtype
                and tuple(observed.shape) == tuple(reference.shape)
                and torch.equal(observed, reference)
            )
        if isinstance(reference, Mapping):
            return (
                isinstance(observed, Mapping)
                and tuple(observed) == tuple(reference)
                and all(exact(observed[key], reference[key]) for key in reference)
            )
        if type(reference) in {list, tuple}:
            return (
                type(observed) is type(reference)
                and len(observed) == len(reference)
                and all(
                    exact(left, right)
                    for left, right in zip(observed, reference, strict=True)
                )
            )
        return type(observed) is type(reference) and observed == reference

    if not all(exact(value[key], expected[key]) for key in CHECKPOINT_PUBLICATION_KEYS):
        raise ValueError("checkpoint publication semantic differs")


def write_history_atomic_noreplace(
    history: object, output: Path,
    *, publication_guard: Callable[[bytes], None] = lambda _payload: None,
) -> None:
    """Publish canonical history bytes without replacing any existing entry."""

    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(history, indent=2, allow_nan=False) + "\n").encode()
    publication_guard(payload)
    def validate(persisted: bytes) -> None:
        if persisted != payload:
            raise RuntimeError("persisted training history bytes differ")

    published = publish_bytes_noreplace(output, payload, validator=validate)
    published.close()


def write_initialization_receipt_atomic(
    receipt: dict[str, object], output: Path, *, expected_shape: list[int]
) -> None:
    validate_initialization_receipt(receipt, expected_shape=expected_shape)
    if not isinstance(output, Path):
        raise TypeError("receipt output must be a Path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    def validate(persisted: bytes) -> None:
        if persisted != payload:
            raise RuntimeError("persisted initialization receipt bytes differ")
        validate_initialization_receipt(
            strict_json_object(persisted), expected_shape=expected_shape
        )

    published = publish_bytes_noreplace(output, payload, validator=validate)
    published.close()


def write_initialization_receipt_v2_atomic(
    receipt: dict[str, object], output: Path,
    *, publication_guard: Callable[[bytes], None] = lambda _payload: None,
) -> None:
    """Publish immutable Task 1 receipt bytes without re-encoding on resume."""

    if type(receipt) is not dict or receipt.get("schema") != "initialization-receipt-v2":
        raise ValueError("FEPF initialization receipt differs")
    if not isinstance(output, Path) or not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("FEPF initialization receipt output differs")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
    publication_guard(payload)
    def validate(persisted: bytes) -> None:
        if persisted != payload or strict_json_object(persisted) != receipt:
            raise RuntimeError("persisted FEPF initialization receipt differs")

    published = publish_bytes_noreplace(output, payload, validator=validate)
    published.close()


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


@torch.inference_mode()
def build_registered_fepf_cache(
    *,
    raw_model: torch.nn.Module,
    optimization: tuple[InshopRecord, ...],
    labels: Mapping[str, int],
    eval_transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    batch_size: int,
    workers: int,
):
    """Encode the frozen optimization inventory once for Task 1."""

    loader = torch.utils.data.DataLoader(
        InshopEvalDataset(optimization, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        generator=torch.Generator().manual_seed(experiment_stream_seed(0, 4_000)),
    )
    chunks: list[torch.Tensor] = []
    observed_labels: list[str] = []
    raw_model.eval()
    for images, batch_labels in loader:
        embeddings = raw_model(images.to(device)).float().detach().cpu().contiguous()
        if (
            embeddings.ndim != 2
            or embeddings.shape != (len(batch_labels), 768)
            or not torch.isfinite(embeddings).all()
        ):
            raise ValueError("FEPF cache embeddings differ")
        chunks.append(embeddings)
        observed_labels.extend(batch_labels)
    if tuple(observed_labels) != tuple(record.label for record in optimization):
        raise ValueError("FEPF cache order differs")
    features = torch.cat(chunks).contiguous() if chunks else torch.empty((0, 768))
    inventory = tuple((record.label, str(record.image_path)) for record in optimization)
    return build_fepf_cache(inventory, features, labels)


def _fepf_config_sha256(args: argparse.Namespace) -> str:
    path = getattr(args, "run_config", None)
    if path is None:
        return hashlib.sha256(b"").hexdigest()
    return _sha256_file(path)


def _fepf_schedule_sha256(args: argparse.Namespace) -> str:
    payload = (
        args.epochs,
        args.batch_size,
        args.workers,
        args.learning_rate,
        args.classifier_learning_rate,
        args.margin,
        args.scale,
        args.objective,
        args.selected_features,
        resolve_evaluation_features(args.selected_features, args.evaluation_features),
        args.eval_every,
        args.checkpoint_every,
        args.max_steps,
        args.bf16,
        args.compile,
        args.fused,
        args.no_ema,
    )
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def initialize_registered_classifier(
    *,
    args: argparse.Namespace,
    raw_model: torch.nn.Module,
    optimization: tuple[InshopRecord, ...],
    labels: Mapping[str, int],
    eval_transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Run the sole fresh registered initializer and restore its side effects."""

    if args.classifier_init not in {"imprinted", "fepf_mean", "fepf_random"}:
        raise ValueError("FEPF initialization mode differs")
    entry = _global_rng_snapshot()
    was_training = raw_model.training
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_started = time.perf_counter()
    official_random_head = torch.empty(len(labels), 768, dtype=torch.float32)
    torch.nn.init.normal_(official_random_head, std=0.01)
    post_draw = _global_rng_snapshot()
    cache = evidence = fit = None
    initialization_seconds: float | None = None
    try:
        cache = build_registered_fepf_cache(
            raw_model=raw_model,
            optimization=optimization,
            labels=labels,
            eval_transform=eval_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        evidence = prepare_registered_fepf_evidence(
            cache,
            official_random_head,
            mode=args.classifier_init,
            training_seed=args.seed,
            device=device,
        )
        if args.classifier_init != "imprinted":
            fit = fit_fepf_head(
                cache,
                evidence,
                training_seed=args.seed,
                device=device,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initialization_seconds = float(time.perf_counter() - initialization_started)
    finally:
        raw_model.train(was_training)
        _restore_global_rng_snapshot(post_draw)
    if cache is None or evidence is None or initialization_seconds is None:
        raise RuntimeError("FEPF initialization evidence is absent")
    rng_audit = _fepf_rng_audit(entry, post_draw)
    receipt = initialization_receipt_v2(
        mode=args.classifier_init,
        training_seed=args.seed,
        holdout_fraction=float(args.holdout_fraction),
        holdout_seed=args.holdout_seed,
        source_sha256=_sha256_file(Path(__file__)),
        checkpoint_sha256=UNICOM_L14_336_SHA256,
        config_sha256=_fepf_config_sha256(args),
        schedule_sha256=_fepf_schedule_sha256(args),
        official_random_head=official_random_head,
        evidence=evidence,
        initialization_seconds=initialization_seconds,
        cache=cache,
        rng_audit=rng_audit,
        fit=fit,
        device=device,
    )
    values = evidence.prepared_start_head if fit is None else fit.head.detach().cpu()
    return values.detach().clone().contiguous(), receipt


def load_and_validate_parent_run_receipt(
    *,
    path: Path,
    checkpoint: Path,
    initialization_receipt: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Load the authenticated parent run receipt for one continuation."""

    if not path.is_file() or path.is_symlink():
        raise ValueError("FEPF parent run receipt differs")
    parent = strict_json_object(path.read_bytes())
    validate_training_run_receipt_v2(parent, evidence_root=path.parent)
    current_root = path.parent.resolve()
    bound_checkpoint = _resolve_receipt_binding(
        {
            key: parent["checkpoints"][-1][key]
            for key in ("root", "path", "sha256", "bytes")
        },
        current_root=current_root,
        parent_root=None,
    )
    bound_initialization = _resolve_receipt_binding(
        parent["initialization_receipt"],
        current_root=current_root,
        parent_root=None,
    )
    if (
        parent.get("initialization_receipt_sha256") is None
        or parent["mode"] != args.classifier_init
        or parent["training_seed"] != args.seed
        or parent["holdout_fraction"] != args.holdout_fraction
        or parent["holdout_seed"] != args.holdout_seed
        or parent["stop_after_epoch"] != 4
        or bound_checkpoint != checkpoint.resolve()
        or bound_initialization != initialization_receipt.resolve()
    ):
        raise ValueError("FEPF parent initialization binding differs")
    return parent


def load_and_validate_parent_initialization_receipt(
    *,
    path: Path,
    args: argparse.Namespace,
    resume_checkpoint: Path,
    expected_sha256: str,
) -> dict[str, object]:
    """Authenticate the original receipt against trusted parent provenance."""

    if not path.is_file() or path.is_symlink():
        raise ValueError("FEPF parent initialization receipt differs")
    receipt = strict_json_object(path.read_bytes())
    expected = FepfExpectedProvenance(
        mode=args.classifier_init,
        training_seed=args.seed,
        holdout_fraction=float(args.holdout_fraction),
        holdout_seed=args.holdout_seed,
        source_sha256=_sha256_file(Path(__file__)),
        checkpoint_sha256=UNICOM_L14_336_SHA256,
        config_sha256=_fepf_config_sha256(args),
        schedule_sha256=_fepf_schedule_sha256(args),
        receipt_sha256=expected_sha256,
    )
    validate_initialization_receipt_v2(receipt, expected=expected, device=torch.device("cuda"))
    return receipt


def resolve_registered_classifier_initialization(
    *,
    args: argparse.Namespace,
    fresh: Callable[[], tuple[torch.Tensor, dict[str, object]]],
) -> tuple[torch.Tensor, dict[str, object], dict[str, object] | None]:
    """Choose exactly one fresh-or-resume initialization evidence path."""

    if args.resume is None:
        values, receipt = fresh()
        return values, receipt, None
    parent = load_and_validate_parent_run_receipt(
        path=args.parent_run_receipt,
        checkpoint=args.resume,
        initialization_receipt=args.parent_initialization_receipt,
        args=args,
    )
    receipt = load_and_validate_parent_initialization_receipt(
        path=args.parent_initialization_receipt,
        args=args,
        resume_checkpoint=args.resume,
        expected_sha256=parent["initialization_receipt_sha256"],
    )
    values = torch.empty(receipt["classifier_shape"], dtype=torch.float32)
    return values, receipt, parent


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
    descriptor_sink: Callable[[torch.Tensor], None] | None = None,
    dataset_root: Path | None = None,
    evidence_root: Path | None = None,
    epoch: int | None = None,
    publication_guard: Callable[[str, Path, bytes], None] = (
        lambda _component, _destination, _payload: None
    ),
) -> dict[str, float]:
    """Evaluate the identity-disjoint holdout with official deployment geometry."""

    import numpy as np

    query_values, query_labels = _encode_records(
        model, query, transform, device=device, batch_size=batch_size, workers=workers
    )
    gallery_values, gallery_labels = _encode_records(
        model, gallery, transform, device=device, batch_size=batch_size, workers=workers
    )
    if descriptor_sink is not None:
        if (
            evaluation_features != 512
        ):
            raise ValueError("FEPF inference descriptor differs")
        descriptor_sink(
            torch.from_numpy(
                np.ascontiguousarray(
                    np.concatenate(
                        (l2_normalize(query_values), l2_normalize(gallery_values))
                    )[:, :512]
                )
            )
        )
    evidence_values = (dataset_root, evidence_root, epoch)
    if any(value is not None for value in evidence_values):
        if any(value is None for value in evidence_values):
            raise ValueError("evaluation evidence arguments differ")
        receipt = write_evaluation_evidence(
            query_values=query_values,
            gallery_values=gallery_values,
            query_records=query,
            gallery_records=gallery,
            dataset_root=dataset_root,
            coordinates=np.arange(evaluation_features, dtype=np.int64),
            normalize_before=True,
            epoch=epoch,
            evidence_root=evidence_root,
            publication_guard=publication_guard,
        )
        return dict(receipt["metrics"])
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
    stop_after_epoch: int | None = None,
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
    publication_guard: Callable[[str, Path, bytes], None] = (
        lambda _name, _destination, _payload: None
    ),
) -> list[dict[str, object]]:
    """Fit and persist sparse raw-model checkpoints for later trajectory soups."""

    output_dir.mkdir(parents=True, exist_ok=True)
    history = [] if history is None else list(history)
    terminal_epoch = epochs if stop_after_epoch is None else stop_after_epoch
    if (
        type(epochs) is not int
        or epochs <= 0
        or type(terminal_epoch) is not int
        or not start_epoch < terminal_epoch <= epochs
    ):
        raise ValueError("training stop boundary differs")
    if step_ema is not None:
        step_ema.register_step_hook(optimizer)
    try:
        for epoch in range(start_epoch, terminal_epoch):
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
                or completed_epoch == terminal_epoch
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
                    publication_guard=lambda payload, epoch=completed_epoch: publication_guard(
                        f"checkpoint-epoch-{epoch:04d}",
                        output_dir / f"epoch-{epoch:04d}.pt",
                        payload,
                    ),
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
    publication_guard: Callable[[bytes], None] = lambda _payload: None,
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
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"checkpoint already exists: {path}")
    def writer(descriptor: int) -> None:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            torch.save(payload, handle)
            handle.flush()

    def validate_checkpoint(encoded: bytes) -> None:
        if not encoded:
            raise ValueError("checkpoint publication is empty")
        publication_guard(encoded)
        restored = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=False)
        validate_checkpoint_publication(restored, expected=payload)

    published = publish_writer_noreplace(path, writer, validator=validate_checkpoint)
    published.close()


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
    if type(checkpoint) is not dict or tuple(checkpoint) != TRAINING_CHECKPOINT_KEYS:
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


def _git_text(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def registered_source_commit(run_config: Path, checkout: Path) -> str:
    """Authenticate a detached config-only handoff and return its source parent."""

    if not isinstance(run_config, Path) or not isinstance(checkout, Path):
        raise TypeError("config-only handoff paths must be pathlib.Path values")
    if not run_config.is_file() or run_config.is_symlink():
        raise ValueError("config-only handoff differs")
    checkout = checkout.resolve()
    try:
        relative = run_config.resolve().relative_to(checkout)
    except ValueError as error:
        raise ValueError("config-only handoff differs") from error
    relative_text = relative.as_posix()
    config = strict_json_object(run_config.read_bytes())
    source = config.get("source")
    handoff = config.get("handoff")
    if type(source) is not dict or type(handoff) is not dict:
        raise ValueError("config-only handoff differs")
    source_commit = source.get("commit")
    if (
        type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or handoff.get("config_parent") != source_commit
        or handoff.get("config_commit_paths") != [relative_text]
        or handoff.get("execution_checkout") != "config_commit_detached_clean"
    ):
        raise ValueError("config-only handoff differs")
    head = _git_revision(checkout)
    changed_paths = _git_text(
        checkout, "diff-tree", "--no-commit-id", "--name-only", "-r", head
    ).splitlines()
    committed_config = subprocess.run(
        ["git", "-C", str(checkout), "show", f"{head}:{relative_text}"],
        check=True,
        capture_output=True,
    ).stdout
    branch = subprocess.run(
        ["git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        _git_text(checkout, "rev-parse", f"{head}^") != source_commit
        or changed_paths != [relative_text]
        or committed_config != run_config.read_bytes()
        or _git_text(checkout, "status", "--porcelain=v1")
        or branch.returncode == 0
    ):
        raise ValueError("config-only handoff differs")
    return source_commit


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


def _is_fepf_request(args: argparse.Namespace) -> bool:
    return (
        args.classifier_init in {"fepf_mean", "fepf_random"}
        or args.stop_after_epoch is not None
        or args.parent_initialization_receipt is not None
        or args.parent_run_receipt is not None
    )


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    from sfora.unicom_inshop import parse_inshop_partition

    fepf_request = _is_fepf_request(args)
    if fepf_request:
        validate_fepf_recipe(args)
    campaign_authority = bool(
        getattr(args, "_fepf_campaign_authorities_validated", False)
    )
    registered_environment = (
        args._fepf_registered_environment
        if fepf_request and campaign_authority
        else None
    )
    if registered_environment is not None:
        establish_registered_deterministic_execution(
            registered_environment.get("deterministic_execution")
        )
    if _git_revision(args.unicom_checkout) != UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if args.checkpoint.name != "FP16-ViT-L-14-336px.pt":
        raise ValueError("UNICOM checkpoint filename differs")
    if _sha256_file(args.checkpoint) != UNICOM_L14_336_SHA256:
        raise ValueError("UNICOM checkpoint SHA-256 differs")
    partition_path = args.dataset_root / "Eval" / "list_eval_partition.txt"
    if fepf_request and _sha256_file(partition_path) != INSHOP_PARTITION_SHA256:
        raise ValueError("In-Shop partition SHA-256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UNICOM training")
    def publication_guard(
        name: str, destination: Path | None = None, payload: bytes | None = None
    ) -> None:
        if fepf_request and campaign_authority:
            if destination is None:
                require_cli_publication_capacity(args, name)
            else:
                require_configured_publication_capacity(
                    args.run_config,
                    args.publication_budget,
                    args.publication_budget_sha256,
                    f"{args.publication_stage}:{name}",
                    destination,
                    args.campaign_root or args.output_dir,
                    payload=payload,
                )

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
    if campaign_authority and registered_runtime_environment(device) != registered_environment:
        raise ValueError("live training environment differs")
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(row for row in records if row.split == "train")
    optimization, query, gallery, labels = identity_holdout(
        train_records,
        fraction=args.holdout_fraction,
        seed=args.holdout_seed,
    )
    if args.run_receipt is not None and len(labels) != 3_200:
        raise ValueError("registered full-width class count differs")
    record_initialization = args.seed in range(2, 7) and not fepf_request
    classifier_shape = classifier_shape_for_run(
        labels,
        record_initialization=record_initialization,
        selected_features=args.selected_features,
        evaluation_features=evaluation_features,
    )
    args.output_dir.mkdir(
        parents=True,
        exist_ok=args.run_receipt is None and not fepf_request,
    )
    raw_model, eval_transform = _load_official_model(args.unicom_checkout, args.checkpoint)
    raw_model = raw_model.to(device)
    raw_backbone_pre_initialization_sha256 = (
        raw_backbone_state_sha256(raw_model) if fepf_request else None
    )
    train_model = torch.compile(raw_model, mode="reduce-overhead") if args.compile else raw_model
    parent_run: dict[str, object] | None = None
    initialization_receipt: dict[str, object] | None = None
    initialization_receipt_path: Path | None = None
    if fepf_request:
        classifier_values, initialization_receipt, parent_run = (
            resolve_registered_classifier_initialization(
                args=args,
                fresh=lambda: initialize_registered_classifier(
                    args=args,
                    raw_model=raw_model,
                    optimization=optimization,
                    labels=labels,
                    eval_transform=eval_transform,
                    device=device,
                ),
            )
        )
        if args.resume is None:
            initialization_receipt_path = args.output_dir / "initialization-receipt.json"
            write_initialization_receipt_v2_atomic(
                initialization_receipt,
                initialization_receipt_path,
                publication_guard=lambda payload: publication_guard(
                    "initialization-receipt", initialization_receipt_path, payload
                ),
            )
        else:
            initialization_receipt_path = args.parent_initialization_receipt
        initialization_receipt_sha256 = canonical_initialization_receipt_v2_sha256(
            initialization_receipt
        )
        if parent_run is not None and (
            initialization_receipt_sha256
            != parent_run["initialization_receipt_sha256"]
        ):
            raise ValueError("FEPF trusted initialization digest differs")
    else:
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
        initialization_receipt_sha256 = None
    classifier = torch.nn.Parameter(classifier_values.to(device))
    ema_decay, ema_update = (
        resolve_registered_ema_protocol(args)
        if fepf_request
        else (EMA_DECAY, "optimizer-step-post-hook-trainable-parameters-only")
    )
    step_ema = StepEMA(raw_model, classifier) if ema_decay is not None else None
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
        "partition_sha256": _sha256_file(partition_path),
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
        "ema_decay": ema_decay,
        "ema_update": ema_update,
    }
    if fepf_request:
        training_protocol["initialization_receipt_sha256"] = (
            initialization_receipt_sha256
        )
        training_protocol["environment"] = registered_environment
        training_protocol["environment_sha256"] = args.environment_sha256
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
    if fepf_request and args.resume is None:
        raw_backbone_pre_training_sha256 = raw_backbone_state_sha256(raw_model)
        if (
            raw_backbone_pre_training_sha256
            != raw_backbone_pre_initialization_sha256
        ):
            raise ValueError("FEPF raw backbone changed during initialization")
    elif fepf_request:
        raw_backbone_pre_initialization_sha256 = parent_run[
            "raw_backbone_pre_initialization_sha256"
        ]
        raw_backbone_pre_training_sha256 = parent_run[
            "raw_backbone_pre_training_sha256"
        ]
    else:
        raw_backbone_pre_training_sha256 = None
    if args.run_receipt is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started_unix_ns = time.time_ns()
        started_counter_ns = time.perf_counter_ns()
    final_descriptor: torch.Tensor | None = None

    def capture_descriptor(value: torch.Tensor) -> None:
        nonlocal final_descriptor
        final_descriptor = value.detach().cpu().clone().contiguous()

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
        stop_after_epoch=args.stop_after_epoch,
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
            descriptor_sink=capture_descriptor if fepf_request else None,
            dataset_root=args.dataset_root if fepf_request else None,
            evidence_root=args.output_dir if fepf_request else None,
            epoch=_epoch if fepf_request else None,
            publication_guard=(
                (
                    lambda component, destination, payload: publication_guard(
                        f"evaluation-epoch-{_epoch:04d}"
                        + ("" if component == "receipt" else f"-{component}"),
                        destination,
                        payload,
                    )
                ) if fepf_request
                else (lambda _component, _destination, _payload: None)
            ),
        ),
        history=history,
        selection_holdout=selection_holdout,
        training_protocol=training_protocol,
        step_ema=step_ema,
        publication_guard=publication_guard,
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
    if fepf_request:
        if final_descriptor is None:
            raise ValueError("FEPF final inference descriptor is absent")
        args._fepf_run_evidence = {
            "initialization_receipt": initialization_receipt,
            "initialization_receipt_path": initialization_receipt_path,
            "raw_backbone_pre_initialization_sha256": raw_backbone_pre_initialization_sha256,
            "raw_backbone_pre_training_sha256": raw_backbone_pre_training_sha256,
            "inference_signature": build_inference_signature(
                raw_model, descriptor=final_descriptor
            ),
            "training_protocol": training_protocol,
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
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--classifier-init",
        choices=("random", "imprinted", "fepf_mean", "fepf_random"),
        default="random",
    )
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--parent-initialization-receipt", type=Path)
    parser.add_argument("--parent-run-receipt", type=Path)
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--environment-authority", type=Path)
    parser.add_argument("--environment-sha256")
    parser.add_argument("--publication-budget", type=Path)
    parser.add_argument("--publication-budget-sha256")
    parser.add_argument("--run-arm")
    parser.add_argument("--run-receipt", type=Path)
    parser.add_argument("--publication-stage")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--authority-preflight-only", action="store_true")
    return parser.parse_args(arguments)


def resolve_registered_ema_protocol(args: argparse.Namespace) -> tuple[float | None, str | None]:
    """Bind EMA presence to the registered current/composed runtime substrate."""

    runtime = (args.compile, args.fused, args.no_ema)
    if runtime == (False, False, False):
        return EMA_DECAY, "optimizer-step-post-hook-trainable-parameters-only"
    if runtime == (True, True, True):
        return None, None
    raise ValueError("registered FEPF EMA/runtime authority differs")


def validate_fepf_recipe(args: argparse.Namespace) -> None:
    """Fail closed unless a registered FEPF run uses the frozen recipe."""

    if args.stop_after_epoch not in (4, 16):
        raise ValueError("FEPF stop boundary differs")
    if args.classifier_init not in {"imprinted", "fepf_mean", "fepf_random"}:
        raise ValueError("FEPF mode differs")
    resolve_registered_ema_protocol(args)
    if args.classifier_init == "fepf_random" and args.seed != 0:
        raise ValueError("FEPF random seed differs")
    if args.resume is not None and args.stop_after_epoch != 16:
        raise ValueError("FEPF continuation stop differs")
    expected = (
        16,
        128,
        4,
        1e-5,
        1e-4,
        0.25,
        32.0,
        "official-eight-mask",
        512,
        512,
        4,
        4,
        None,
        False,
    )
    observed = (
        args.epochs,
        args.batch_size,
        args.workers,
        args.learning_rate,
        args.classifier_learning_rate,
        args.margin,
        args.scale,
        args.objective,
        args.selected_features,
        resolve_evaluation_features(args.selected_features, args.evaluation_features),
        args.eval_every,
        args.checkpoint_every,
        args.max_steps,
        args.bf16,
    )
    if observed != expected or any(
        type(value) is not type(reference)
        for value, reference in zip(observed, expected, strict=True)
    ):
        raise ValueError("FEPF recipe differs")
    parent_receipts = (
        args.parent_initialization_receipt,
        args.parent_run_receipt,
    )
    if args.resume is None:
        if any(value is not None for value in parent_receipts):
            raise ValueError("FEPF parent receipts differ")
    elif any(value is None for value in parent_receipts):
        raise ValueError("FEPF parent receipts differ")
    elif args.output_dir.resolve() == args.resume.parent.resolve():
        raise ValueError("FEPF continuation output differs")


def _validate_evaluation_dataset_root(dataset_root: Path) -> None:
    if not isinstance(dataset_root, Path):
        raise ValueError("FEPF dataset root differs")
    absolute_root = dataset_root.absolute()
    resolved_root = dataset_root.resolve()
    image_root = resolved_root / "Img"
    if (
        absolute_root != resolved_root
        or not resolved_root.is_dir()
        or resolved_root.is_symlink()
        or not image_root.is_dir()
        or image_root.is_symlink()
    ):
        raise ValueError("FEPF dataset root differs")


def _validate_run_receipt_request(args: argparse.Namespace) -> None:
    if _is_fepf_request(args):
        validate_fepf_recipe(args)
        _validate_evaluation_dataset_root(args.dataset_root)
        if (
            args.run_config is None
            or args.run_receipt is None
            or args.run_arm is not None
        ):
            raise ValueError("FEPF run receipt arguments differ")
        if args.run_receipt.name != "run-receipt.json" or (
            args.resume is not None
            and args.parent_run_receipt.name != "run-receipt.json"
        ):
            raise ValueError("FEPF run receipt path differs")
        if not args.run_config.is_file() or args.run_config.is_symlink():
            raise ValueError("FEPF run configuration differs")
        if args.run_receipt.exists() or args.run_receipt.is_symlink():
            raise FileExistsError(args.run_receipt)
        if args.run_receipt.parent.resolve() != args.output_dir.resolve():
            raise ValueError("FEPF run receipt root differs")
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise FileExistsError(args.output_dir)
        if not args.output_dir.parent.is_dir() or args.output_dir.parent.is_symlink():
            raise ValueError("FEPF output parent differs")
        return
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


def validate_fepf_cli_campaign_authorities(args: argparse.Namespace) -> None:
    """Authenticate campaign-owned authorities before public FEPF execution."""

    if not _is_fepf_request(args):
        return
    if (
        args.run_config is None
        or args.environment_authority is None
        or args.environment_sha256 is None
        or args.publication_budget is None
        or args.publication_budget_sha256 is None
        or args.publication_stage is None
        or args.campaign_root is None
    ):
        raise ValueError("registered environment/publication authority is required")
    config = strict_json_object(args.run_config.read_bytes())
    if Path(config.get("artifact_root", "")).resolve() != args.campaign_root.resolve():
        raise ValueError("registered campaign root differs")
    environment = load_configured_environment_authority(
        args.run_config, args.environment_authority, args.environment_sha256
    )
    establish_registered_deterministic_execution(
        environment["deterministic_execution"]
    )
    load_configured_publication_budget(
        args.run_config,
        args.publication_budget,
        args.publication_budget_sha256,
        external=True,
    )
    args._fepf_registered_environment = environment
    args._fepf_campaign_authorities_validated = True


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        if args.authority_preflight_only:
            stage = args.publication_stage or args.run_arm
            root = args.campaign_root
            if stage is None or root is None:
                if args.run_config is None:
                    raise ValueError("publication stage authority is required")
                config_value = strict_json_object(args.run_config.read_bytes())
                root = Path(config_value["artifact_root"])
            if args.publication_budget is None or args.publication_budget_sha256 is None:
                raise ValueError("publication budget authority is required")
            load_configured_publication_budget(
                args.run_config, args.publication_budget, args.publication_budget_sha256,
                external=False,
            )
            name = f"{stage}:initialization-receipt"
            destination = root / str(stage) / "initialization-receipt.json"
            require_configured_publication_capacity(
                args.run_config,
                args.publication_budget,
                args.publication_budget_sha256,
                name,
                destination,
                root,
                external=False,
            )
            if args.environment_authority is None or args.environment_sha256 is None:
                raise ValueError("environment authority is required")
            environment = load_configured_environment_authority(
                args.run_config, args.environment_authority, args.environment_sha256
            )
            establish_registered_deterministic_execution(
                environment["deterministic_execution"]
            )
            return 0
        validate_fepf_cli_campaign_authorities(args)
        _validate_run_receipt_request(args)
        source_commit = None
        if args.run_receipt is not None:
            source_commit = registered_source_commit(
                args.run_config, Path(__file__).resolve().parents[1]
            )
        history = run(args)
    except Exception as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 2
    summary = args.output_dir / "history.json"
    write_history_atomic_noreplace(
        history,
        summary,
        publication_guard=lambda payload: (
            require_configured_publication_capacity(
                args.run_config,
                args.publication_budget,
                args.publication_budget_sha256,
                f"{args.publication_stage}:history",
                summary,
                args.campaign_root or args.output_dir,
                payload=payload,
            )
            if _is_fepf_request(args)
            else None
        ),
    )
    if args.run_receipt is not None:
        try:
            if _is_fepf_request(args):
                evidence = args._fepf_run_evidence
                stop_after_epoch = args.stop_after_epoch or args.epochs
                if args.resume is None:
                    checkpoint_paths = {
                        epoch: args.output_dir / f"epoch-{epoch:04d}.pt"
                        for epoch in (4, 8, 12, 16)
                        if epoch <= stop_after_epoch
                    }
                    evaluation_receipt_paths = {
                        epoch: args.output_dir / f"evaluation-epoch-{epoch:04d}.json"
                        for epoch in (4, 8, 12, 16)
                        if epoch <= stop_after_epoch
                    }
                else:
                    checkpoint_paths = {
                        4: args.resume,
                        **{
                            epoch: args.output_dir / f"epoch-{epoch:04d}.pt"
                            for epoch in (8, 12, 16)
                            if epoch <= stop_after_epoch
                        },
                    }
                    evaluation_receipt_paths = {
                        4: args.parent_run_receipt.parent
                        / "evaluation-epoch-0004.json",
                        **{
                            epoch: args.output_dir
                            / f"evaluation-epoch-{epoch:04d}.json"
                            for epoch in (8, 12, 16)
                            if epoch <= stop_after_epoch
                        },
                    }
                receipt = training_run_receipt_v2(
                    output_dir=args.output_dir,
                    mode=args.classifier_init,
                    training_seed=args.seed,
                    holdout_fraction=float(args.holdout_fraction),
                    holdout_seed=args.holdout_seed,
                    training_protocol=evidence["training_protocol"],
                    stop_after_epoch=stop_after_epoch,
                    initialization_receipt_path=evidence[
                        "initialization_receipt_path"
                    ],
                    history_path=summary,
                    checkpoint_paths=checkpoint_paths,
                    evaluation_receipt_paths=evaluation_receipt_paths,
                    raw_backbone_pre_initialization_sha256=evidence[
                        "raw_backbone_pre_initialization_sha256"
                    ],
                    raw_backbone_pre_training_sha256=evidence[
                        "raw_backbone_pre_training_sha256"
                    ],
                    inference_signature=evidence["inference_signature"],
                    parent_run_receipt_path=args.parent_run_receipt,
                    parent_checkpoint_path=args.resume,
                )
                write_training_run_receipt_atomic(
                    receipt,
                    args.run_receipt,
                    evidence_root=args.output_dir,
                    publication_guard=lambda payload: require_configured_publication_capacity(
                        args.run_config,
                        args.publication_budget,
                        args.publication_budget_sha256,
                        f"{args.publication_stage}:run-receipt",
                        args.run_receipt,
                        args.campaign_root or args.output_dir,
                        payload=payload,
                    ),
                )
                print(f"training complete: {summary}")
                return 0
            measurement = args._training_run_measurement
            command = (
                list(sys.orig_argv)
                if arguments is None
                else [sys.executable, str(Path(__file__).resolve()), *arguments]
            )
            receipt = training_run_receipt(
                source_commit=source_commit,
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
