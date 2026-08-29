#!/usr/bin/env python3
"""Profile the registered UniCOM training step without mutating its checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from sfora.atomic_publication import BudgetedPublisher, publish_bytes_noreplace

WARMUP_STEPS = 20
MEASURE_STEPS = 50
PROFILER_STEPS = 10
BOOTSTRAP_SEED = 20_016
BOOTSTRAP_REPLICATES = 10_000
KERNEL_GATE_THRESHOLD = 0.1
CLOCK_DOMAIN_REL_TOL = 1e-2
OBJECTIVE_MARKER = "unicom_objective_profile_step"
PARENT_TRAINER_COMMIT = "70c760e57e6c27dec1473eecd4765e0a8cd4cf6b"
PARENT_TRAINER_PATH = "scripts/train_unicom_inshop.py"
PARENT_TRAINER_SOURCE = f"{PARENT_TRAINER_COMMIT}:{PARENT_TRAINER_PATH}"
PARENT_TRAINER_SHA256 = (
    "6eea2dab88ff9e4c5a547f9fe326ebf56879882784c5a80c8e136f6d02b52170"
)


class RuntimeProtocol(NamedTuple):
    compile: bool
    fused: bool
    ema: bool


RUNTIME_PROTOCOLS = {
    "current": RuntimeProtocol(compile=False, fused=False, ema=True),
    "composed": RuntimeProtocol(compile=True, fused=True, ema=False),
}
COMPONENT_KEYS = (
    "h2d_seconds",
    "zero_grad_seconds",
    "backbone_forward_seconds",
    "objective_forward_seconds",
    "head_backward_seconds",
    "backbone_backward_seconds",
    "update_seconds",
    "tail_seconds",
)
TIMING_KEYS = (
    "step_wall_seconds",
    "cuda_step_seconds",
    *COMPONENT_KEYS,
)
_GEMM_NAMES = {
    "aten::addmm",
    "aten::baddbmm",
    "aten::bmm",
    "aten::linear",
    "aten::matmul",
    "aten::mm",
}
PROFILE_KEYS = (
    "schema_version",
    "run_checkpoint",
    "run_checkpoint_sha256",
    "trainer_sha256",
    "objective_sha256",
    "profiler_sha256",
    "checkpoint_epoch",
    "started_unix_ns",
    "finished_unix_ns",
    "classifier_init",
    "warmup_steps",
    "measure_steps",
    "profiler_steps",
    "timing_samples",
    "fusible_samples",
    "summary",
    "runtime",
)
PROFILE_V2_KEYS = (
    "schema",
    "profile_kind",
    "runtime_mode",
    "parent_trainer_source",
    "parent_trainer_sha256",
    "live_trainer_sha256",
    "profiler_sha256",
    "checkpoint",
    "run_receipt",
    "config",
    "checkpoint_epoch",
    "checkpoint_protocol",
    "inference_signature",
    "runtime_overrides",
    "warmup_steps",
    "measure_steps",
    "objective_steps",
    "optimizer_step_count",
    "objective_call_count",
    "timing_synchronized",
    "peak_reset",
    "started_unix_ns",
    "finished_unix_ns",
    "losses",
    "unscaled_gradients_finite",
    "scaler_decisions",
    "timing_samples",
    "objective_samples",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "parameter_schema",
    "optimizer_schema",
    "environment_sha256",
    "environment",
)
REGISTERED_ENVIRONMENT_KEYS = (
    "python_vv", "torch", "torchvision", "timm", "numpy", "cuda", "cudnn",
    "compile", "device_uuid", "gpu_inventory", "pyproject_sha256",
    "uv_lock_sha256", "deterministic_execution",
)
INFERENCE_SIGNATURE_KEYS = (
    "schema",
    "tensors",
    "total_bytes",
    "aggregate_sha256",
    "descriptor_dtype",
    "descriptor_dimension",
    "descriptor_sha256",
    "operations",
)
INFERENCE_TENSOR_KEYS = (
    "name",
    "kind",
    "shape",
    "dtype",
    "numel",
    "element_size",
    "bytes",
    "sha256",
)
INFERENCE_OPERATIONS = (
    "official_forward",
    "full768_l2",
    "prefix512",
    "squared_euclidean",
)
RUNTIME_POOLED_RATIO_MAX = 0.8695652173913043
RUNTIME_PAIR_RATIO_MAX = 0.90
RUNTIME_LOSS_REL_L2_MAX = 2e-4
RUNTIME_LOSS_MAX_ABS = 2e-3
RUNTIME_MEMORY_RATIO_MAX = 1.02


def _finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validate_timing_sample(sample: object, index: int) -> dict[str, float]:
    if type(sample) is not dict or tuple(sample) != TIMING_KEYS:
        raise ValueError(f"timing sample {index} schema differs")
    validated = {
        key: _finite_float(sample[key], f"timing sample {index} {key}", nonnegative=True)
        for key in TIMING_KEYS
    }
    if validated["step_wall_seconds"] <= 0.0 or validated["cuda_step_seconds"] <= 0.0:
        raise ValueError("timing step durations must be positive")
    contiguous = math.fsum(validated[key] for key in COMPONENT_KEYS)
    if not math.isclose(
        contiguous,
        validated["cuda_step_seconds"],
        rel_tol=1e-9,
        abs_tol=1e-5,
    ):
        raise ValueError("timing contiguous component sum differs")
    # CUDA events and perf_counter use independent oscillators. One percent is
    # conservative for clock skew while retaining a structural timeline check.
    if validated["cuda_step_seconds"] > validated["step_wall_seconds"] * (
        1.0 + CLOCK_DOMAIN_REL_TOL
    ):
        raise ValueError(
            f"timing sample {index} CUDA step exceeds wall step: "
            f"cuda={validated['cuda_step_seconds']!r} "
            f"wall={validated['step_wall_seconds']!r}"
        )
    return validated


def summarize_timing_samples(samples: object) -> dict[str, object]:
    """Validate and aggregate unprofiled contiguous CUDA-event samples."""

    if type(samples) is not tuple or not samples:
        raise ValueError("timing samples must be a nonempty tuple")
    rows = tuple(_validate_timing_sample(sample, index) for index, sample in enumerate(samples))
    wall = np.asarray([row["step_wall_seconds"] for row in rows], dtype=np.float64)
    component_means = {
        key: math.fsum(row[key] for row in rows) / len(rows) for key in COMPONENT_KEYS
    }
    objective = (
        component_means["objective_forward_seconds"]
        + component_means["head_backward_seconds"]
    )
    wall_mean = float(wall.mean(dtype=np.float64))
    return {
        "step_wall_seconds": float(np.median(wall)),
        "step_wall_p10_seconds": float(np.quantile(wall, 0.1)),
        "step_wall_p90_seconds": float(np.quantile(wall, 0.9)),
        "objective_ceiling_seconds": objective,
        "objective_ceiling_fraction": objective / wall_mean,
        "component_mean_seconds": component_means,
    }


def _event_children(event: object) -> tuple[object, ...]:
    children = getattr(event, "cpu_children", None)
    if type(children) is not list:
        raise TypeError("profiler event children differ")
    return tuple(children)


def _descendants(event: object):
    for child in _event_children(event):
        yield child
        yield from _descendants(child)


def fusible_nonbackbone_seconds(events: object) -> float:
    """Sum objective CUDA ops across caller and autograd threads, except GEMMs."""

    if type(events) is not tuple:
        raise TypeError("profiler events must be a tuple")
    markers = tuple(
        event
        for event in events
        if getattr(event, "name", None) == OBJECTIVE_MARKER
        and getattr(getattr(event, "device_type", None), "name", None) == "CPU"
    )
    if len(markers) != 1:
        raise ValueError("profiler objective marker count differs")
    marker_descendants = tuple(_descendants(markers[0]))
    candidates = marker_descendants + tuple(
        event
        for event in events
        if getattr(getattr(event, "device_type", None), "name", None) == "CPU"
        and getattr(event, "name", "").startswith("aten::")
        and all(event is not descendant for descendant in marker_descendants)
    )
    total_microseconds = 0.0
    for event in candidates:
        name = getattr(event, "name", None)
        value = getattr(event, "self_device_time_total", None)
        if type(name) is not str or type(value) not in (float, int):
            raise TypeError("profiler event fields differ")
        duration = float(value)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("profiler event device time differs")
        if name not in _GEMM_NAMES:
            total_microseconds += duration
    return total_microseconds / 1_000_000.0


def summarize_profile(timing_samples: object, fusible_samples: object) -> dict[str, object]:
    """Combine unprofiled wall samples with separate objective-only profiler samples."""

    timing = summarize_timing_samples(timing_samples)
    if type(fusible_samples) is not tuple or not fusible_samples:
        raise ValueError("fusible samples must be a nonempty tuple")
    fusible = np.asarray(
        [
            _finite_float(value, f"fusible sample {index}", nonnegative=True)
            for index, value in enumerate(fusible_samples)
        ],
        dtype=np.float64,
    )
    wall = np.asarray(
        [
            _validate_timing_sample(sample, index)["step_wall_seconds"]
            for index, sample in enumerate(timing_samples)
        ],
        dtype=np.float64,
    )
    if float(fusible.max()) > float(wall.max()) * (1.0 + 1e-9):
        raise ValueError("fusible objective time exceeds the measured step")
    fusible_mean = float(fusible.mean(dtype=np.float64))
    wall_mean = float(wall.mean(dtype=np.float64))
    fraction = fusible_mean / wall_mean
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    wall_draws = wall[generator.integers(0, wall.size, size=(BOOTSTRAP_REPLICATES, wall.size))]
    fusible_draws = fusible[
        generator.integers(
            0,
            fusible.size,
            size=(BOOTSTRAP_REPLICATES, fusible.size),
        )
    ]
    bootstrap = fusible_draws.mean(axis=1) / wall_draws.mean(axis=1)
    lower = float(np.quantile(bootstrap, 0.025))
    return {
        **timing,
        "fusible_non_backbone_seconds": fusible_mean,
        "fusible_non_backbone_fraction": fraction,
        "fusible_fraction_bootstrap_lower_95": lower,
        "kernel_gate_threshold": KERNEL_GATE_THRESHOLD,
        "kernel_gate_passed": bool(lower >= KERNEL_GATE_THRESHOLD),
    }


def _lower_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        raise ValueError("registered profile environment differs")


def load_registered_environment_authority(
    path: Path, expected_sha256: str
) -> dict[str, object]:
    value, payload = _strict_json_object_payload(path)
    if (
        not _lower_sha256(expected_sha256)
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("registered profile environment authority differs")
    validate_registered_environment_payload(value)
    return value


def load_configured_environment_authority(
    config_path: Path, path: Path, expected_sha256: str
) -> dict[str, object]:
    """Load one environment through the immutable authority in the run config."""

    config = _strict_json_object(config_path)
    value, payload = _strict_json_object_payload(path)
    expected = {
        "path": str(path.resolve()),
        "sha256": expected_sha256,
        "bytes": len(payload),
    }
    registered = config.get("cuda_canary_environment")
    authority_matches = (
        registered == expected
        or (
            type(registered) is dict
            and tuple(registered) == ("path",)
            and registered["path"] == str(path.resolve())
        )
    )
    if (
        not authority_matches
        or not _lower_sha256(expected_sha256)
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("configured profile environment authority differs")
    validate_registered_environment_payload(value)
    return value


def _validate_file_authority(value: object, name: str) -> None:
    if (
        type(value) is not dict
        or tuple(value) != ("path", "sha256", "bytes")
        or type(value["path"]) is not str
        or not Path(value["path"]).is_absolute()
        or not _lower_sha256(value["sha256"])
        or type(value["bytes"]) is not int
        or value["bytes"] <= 0
    ):
        raise ValueError(f"profile {name} authority differs")


def _validate_inference_signature(signature: object) -> None:
    if (
        type(signature) is not dict
        or tuple(signature) != INFERENCE_SIGNATURE_KEYS
        or signature["schema"] != "unicom-inference-signature-v1"
        or type(signature["tensors"]) is not list
        or not signature["tensors"]
        or type(signature["total_bytes"]) is not int
        or signature["total_bytes"] <= 0
        or not _lower_sha256(signature["aggregate_sha256"])
        or signature["descriptor_dtype"] != "torch.float32"
        or signature["descriptor_dimension"] != 512
        or not _lower_sha256(signature["descriptor_sha256"])
        or signature["operations"] != list(INFERENCE_OPERATIONS)
    ):
        raise ValueError("profile inference signature differs")
    names: list[str] = []
    total_bytes = 0
    for row in signature["tensors"]:
        if (
            type(row) is not dict
            or tuple(row) != INFERENCE_TENSOR_KEYS
            or type(row["name"]) is not str
            or not row["name"]
            or row["kind"] not in {"parameter", "buffer"}
            or type(row["shape"]) is not list
            or any(type(size) is not int or size < 0 for size in row["shape"])
            or type(row["dtype"]) is not str
            or not row["dtype"].startswith("torch.")
            or type(row["numel"]) is not int
            or row["numel"] < 0
            or type(row["element_size"]) is not int
            or row["element_size"] <= 0
            or type(row["bytes"]) is not int
            or row["bytes"] != row["numel"] * row["element_size"]
            or not _lower_sha256(row["sha256"])
        ):
            raise ValueError("profile inference tensor differs")
        names.append(row["name"])
        total_bytes += row["bytes"]
    if names != sorted(names) or len(names) != len(set(names)) or total_bytes != signature[
        "total_bytes"
    ]:
        raise ValueError("profile inference tensor inventory differs")


def _validate_scaler_decision(value: object) -> None:
    if (
        type(value) is not dict
        or tuple(value) != ("enabled", "scale_before", "scale_after", "skipped")
        or type(value["enabled"]) is not bool
        or type(value["skipped"]) is not bool
        or value["skipped"]
    ):
        raise ValueError("profile scaler decision differs")
    if value["enabled"]:
        _finite_float(value["scale_before"], "profile scaler before")
        _finite_float(value["scale_after"], "profile scaler after")
        if value["scale_before"] <= 0.0 or value["scale_after"] <= 0.0:
            raise ValueError("profile scaler decision differs")
    elif value["scale_before"] is not None or value["scale_after"] is not None:
        raise ValueError("profile scaler decision differs")


def _validate_optimizer_value_schema(value: object) -> None:
    if type(value) is not dict or "kind" not in value:
        raise ValueError("profile optimizer schema differs")
    kind = value["kind"]
    if kind == "tensor":
        if (
            tuple(value) != ("kind", "shape", "dtype")
            or type(value["shape"]) is not list
            or any(type(size) is not int or size < 0 for size in value["shape"])
            or type(value["dtype"]) is not str
            or not value["dtype"].startswith("torch.")
        ):
            raise ValueError("profile optimizer schema differs")
    elif kind in {"bool", "int", "float", "str", "none"}:
        if tuple(value) != ("kind",):
            raise ValueError("profile optimizer schema differs")
    elif kind in {"tuple", "list"}:
        if tuple(value) != ("kind", "items") or type(value["items"]) is not list:
            raise ValueError("profile optimizer schema differs")
        for item in value["items"]:
            _validate_optimizer_value_schema(item)
    else:
        raise ValueError("profile optimizer schema differs")


def _validate_parameter_schema_object(value: object) -> None:
    if type(value) is not list or not value:
        raise ValueError("profile parameter schema differs")
    names = []
    for row in value:
        if (
            type(row) is not dict
            or tuple(row) != ("name", "shape", "dtype")
            or type(row["name"]) is not str
            or not row["name"]
            or type(row["shape"]) is not list
            or any(type(size) is not int or size < 0 for size in row["shape"])
            or type(row["dtype"]) is not str
            or not row["dtype"].startswith("torch.")
        ):
            raise ValueError("profile parameter schema differs")
        names.append(row["name"])
    raw_names = names[:-1]
    if (
        names[-1] != "classifier"
        or raw_names != sorted(raw_names)
        or any(not name.startswith("raw_model.") for name in raw_names)
        or len(names) != len(set(names))
    ):
        raise ValueError("profile parameter schema differs")


def _validate_optimizer_schema_object(value: object) -> None:
    if (
        type(value) is not dict
        or tuple(value) != ("param_groups", "state")
        or type(value["param_groups"]) is not list
        or not value["param_groups"]
        or type(value["state"]) is not list
    ):
        raise ValueError("profile optimizer schema differs")
    for group in value["param_groups"]:
        if (
            type(group) is not dict
            or tuple(group) != ("parameter_count", "fields")
            or type(group["parameter_count"]) is not int
            or group["parameter_count"] <= 0
            or type(group["fields"]) is not dict
            or not group["fields"]
            or tuple(group["fields"]) != tuple(sorted(group["fields"]))
        ):
            raise ValueError("profile optimizer schema differs")
        for descriptor in group["fields"].values():
            _validate_optimizer_value_schema(descriptor)
    parameters = []
    for row in value["state"]:
        if (
            type(row) is not dict
            or tuple(row) != ("parameter", "fields")
            or type(row["parameter"]) is not int
            or row["parameter"] < 0
            or type(row["fields"]) is not dict
            or tuple(row["fields"]) != tuple(sorted(row["fields"]))
        ):
            raise ValueError("profile optimizer schema differs")
        parameters.append(row["parameter"])
        for descriptor in row["fields"].values():
            _validate_optimizer_value_schema(descriptor)
    if parameters != sorted(parameters) or len(parameters) != len(set(parameters)):
        raise ValueError("profile optimizer schema differs")


def _validate_profile_v2(receipt: object, *, expected_kind: str | None = None) -> None:
    if (
        type(receipt) is not dict
        or tuple(receipt) != PROFILE_V2_KEYS
        or receipt["schema"] != "unicom-training-step-profile-v2"
        or receipt["profile_kind"] not in {"runtime", "quality"}
        or receipt["runtime_mode"] not in RUNTIME_PROTOCOLS
        or (expected_kind is not None and receipt["profile_kind"] != expected_kind)
        or receipt["parent_trainer_source"] != PARENT_TRAINER_SOURCE
        or receipt["parent_trainer_sha256"] != PARENT_TRAINER_SHA256
        or not _lower_sha256(receipt["live_trainer_sha256"])
        or not _lower_sha256(receipt["profiler_sha256"])
        or len(
            {
                receipt["parent_trainer_sha256"],
                receipt["live_trainer_sha256"],
                receipt["profiler_sha256"],
            }
        )
        != 3
    ):
        raise ValueError("profile receipt binding differs")
    for key in ("checkpoint", "run_receipt", "config"):
        _validate_file_authority(receipt[key], key)
    if (
        type(receipt["checkpoint_epoch"]) is not int
        or receipt["checkpoint_epoch"] < 1
        or type(receipt["checkpoint_protocol"]) is not dict
    ):
        raise ValueError("profile checkpoint authority differs")
    _validate_checkpoint_authority_for_profile(
        {"training_protocol": receipt["checkpoint_protocol"]},
        profile_kind=receipt["profile_kind"],
        live_trainer_sha256=receipt["live_trainer_sha256"],
    )
    _validate_inference_signature(receipt["inference_signature"])
    runtime = RUNTIME_PROTOCOLS[receipt["runtime_mode"]]
    if receipt["runtime_overrides"] != {
        "compile": runtime.compile,
        "fused": runtime.fused,
        "ema": runtime.ema,
    }:
        raise ValueError("profile runtime override differs")
    objective_steps = PROFILER_STEPS if receipt["profile_kind"] == "runtime" else 0
    if (
        receipt["warmup_steps"] != WARMUP_STEPS
        or receipt["measure_steps"] != MEASURE_STEPS
        or receipt["objective_steps"] != objective_steps
        or receipt["optimizer_step_count"] != WARMUP_STEPS + MEASURE_STEPS
        or receipt["objective_call_count"] != objective_steps
        or receipt["timing_synchronized"] is not True
        or receipt["peak_reset"]
        != {"after_warmup": True, "before_measurement": True, "empty_cache": False}
    ):
        raise ValueError("profile step evidence differs")
    started = receipt["started_unix_ns"]
    finished = receipt["finished_unix_ns"]
    if type(started) is not int or type(finished) is not int or started <= 0 or finished <= started:
        raise ValueError("profile process interval differs")
    losses = receipt["losses"]
    gradients = receipt["unscaled_gradients_finite"]
    decisions = receipt["scaler_decisions"]
    timings = receipt["timing_samples"]
    objective = receipt["objective_samples"]
    if (
        type(losses) is not list
        or len(losses) != MEASURE_STEPS
        or type(gradients) is not list
        or gradients != [True] * MEASURE_STEPS
        or type(decisions) is not list
        or len(decisions) != MEASURE_STEPS
        or type(timings) is not list
        or len(timings) != MEASURE_STEPS
        or type(objective) is not list
        or len(objective) != objective_steps
    ):
        raise ValueError("profile measured evidence differs")
    for index, loss in enumerate(losses):
        _finite_float(loss, f"profile loss {index}")
    for decision in decisions:
        _validate_scaler_decision(decision)
    for index, sample in enumerate(timings):
        _validate_timing_sample(sample, index)
    for index, sample in enumerate(objective):
        _finite_float(sample, f"profile objective sample {index}", nonnegative=True)
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if type(receipt[key]) is not int or receipt[key] <= 0:
            raise ValueError("profile peak memory differs")
    if (
        type(receipt["parameter_schema"]) is not list
        or not receipt["parameter_schema"]
        or type(receipt["optimizer_schema"]) is not dict
        or not receipt["optimizer_schema"]
    ):
        raise ValueError("profile schema or environment differs")
    validate_registered_environment_payload(receipt["environment"])
    if receipt["environment_sha256"] != hashlib.sha256(
        (json.dumps(receipt["environment"], indent=2, allow_nan=False) + "\n").encode()
    ).hexdigest():
        raise ValueError("profile environment hash differs")


def validate_quality_profile(receipt: object) -> None:
    """Strictly validate one resource-only quality-arm profile receipt."""

    _validate_profile_v2(receipt, expected_kind="quality")
    if receipt["checkpoint_epoch"] != 16:
        raise ValueError("quality profile checkpoint must be epoch 16")
    _validate_parameter_schema_object(receipt["parameter_schema"])
    _validate_optimizer_schema_object(receipt["optimizer_schema"])
    _validate_quality_authority_chain(receipt)


def _require_persisted_authority(value: dict[str, object], name: str) -> Path:
    _validate_file_authority(value, name)
    path = Path(value["path"])
    if path != path.resolve() or path.is_symlink() or not path.is_file():
        raise ValueError(f"quality profile {name} authority differs")
    if _file_authority(path) != value:
        raise ValueError(f"quality profile {name} authority is stale")
    return path


def _checkpoint_parameter_schema(
    checkpoint: dict[str, object], inference_signature: dict[str, object]
) -> list[dict[str, object]]:
    model = checkpoint["model"]
    classifier = checkpoint["classifier"]
    if not isinstance(model, dict):
        raise ValueError("quality profile checkpoint parameter schema differs")
    rows = []
    for inference_row in inference_signature["tensors"]:
        if inference_row["kind"] != "parameter":
            continue
        name = inference_row["name"]
        value = model.get(name)
        if (
            value is None
            or list(value.shape) != inference_row["shape"]
            or str(value.dtype) != inference_row["dtype"]
        ):
            raise ValueError("quality profile checkpoint parameter schema differs")
        rows.append(
            {
                "name": f"raw_model.{name}",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )
    rows.sort(key=lambda row: row["name"])
    rows.append(
        {
            "name": "classifier",
            "shape": list(classifier.shape),
            "dtype": str(classifier.dtype),
        }
    )
    _validate_parameter_schema_object(rows)
    return rows


def _rebuild_checkpoint_inference_signature(
    checkpoint: dict[str, object], external_signature: dict[str, object]
) -> dict[str, object]:
    """Rehash the complete checkpoint model inventory under external structure authority."""

    import torch

    _validate_inference_signature(external_signature)
    model = checkpoint.get("model")
    if not isinstance(model, dict):
        raise ValueError("profile checkpoint inference signature differs")
    external_rows = external_signature["tensors"]
    kinds = {row["name"]: row["kind"] for row in external_rows}
    if set(model) != set(kinds):
        raise ValueError("profile checkpoint inference inventory differs")
    aggregate = hashlib.sha256()
    rows = []
    total_bytes = 0
    for name in sorted(model):
        value = model[name]
        try:
            payload = (
                value.detach()
                .cpu()
                .contiguous()
                .reshape(-1)
                .view(torch.uint8)
                .numpy()
                .tobytes(order="C")
            )
        except Exception as error:
            raise ValueError("profile checkpoint inference tensor differs") from error
        row = {
            "name": name,
            "kind": kinds[name],
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
        rows.append(row)
        total_bytes += len(payload)
    return {
        "schema": "unicom-inference-signature-v1",
        "tensors": rows,
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "descriptor_dtype": external_signature["descriptor_dtype"],
        "descriptor_dimension": external_signature["descriptor_dimension"],
        "descriptor_sha256": external_signature["descriptor_sha256"],
        "operations": external_signature["operations"],
    }


def _require_quality_signature_structure(
    current: dict[str, object], historical: dict[str, object]
) -> None:
    """Bind current quality values to the registered tensor/operation structure."""

    _validate_inference_signature(current)
    _validate_inference_signature(historical)
    structural_keys = (
        "name", "kind", "shape", "dtype", "numel", "element_size", "bytes"
    )
    current_structure = [
        {key: row[key] for key in structural_keys} for row in current["tensors"]
    ]
    historical_structure = [
        {key: row[key] for key in structural_keys}
        for row in historical["tensors"]
    ]
    if (
        current_structure != historical_structure
        or current["operations"] != historical["operations"]
        or current["descriptor_dtype"] != historical["descriptor_dtype"]
        or current["descriptor_dimension"] != historical["descriptor_dimension"]
    ):
        raise ValueError("quality profile inference structure differs")


def _validate_quality_authority_chain(receipt: dict[str, object]) -> None:
    checkpoint_path = _require_persisted_authority(receipt["checkpoint"], "checkpoint")
    run_receipt_path = _require_persisted_authority(receipt["run_receipt"], "run_receipt")
    config_path = _require_persisted_authority(receipt["config"], "config")
    if run_receipt_path.name != "run-receipt.json":
        raise ValueError("quality profile run receipt authority differs")
    config = _strict_json_object(config_path)
    run_receipt = _strict_json_object(run_receipt_path)
    required_config = {
        "parent_trainer_commit": PARENT_TRAINER_COMMIT,
        "parent_trainer_path": PARENT_TRAINER_PATH,
        "parent_trainer_sha256": PARENT_TRAINER_SHA256,
        "live_trainer_sha256": receipt["live_trainer_sha256"],
        "profiler_sha256": receipt["profiler_sha256"],
    }
    if any(config.get(key) != value for key, value in required_config.items()):
        raise ValueError("quality profile config authority differs")
    environment_authority = config.get("cuda_canary_environment")
    if type(environment_authority) is not dict:
        raise ValueError("quality profile environment config authority differs")
    environment = load_configured_environment_authority(
        config_path,
        Path(environment_authority.get("path", "")),
        environment_authority.get("sha256"),
    )
    historical_signature = config.get("runtime_inference_signature")
    current_signature = run_receipt.get("inference_signature")
    _require_quality_signature_structure(current_signature, historical_signature)
    repository = Path(__file__).resolve().parents[1]
    trainer = _load_authenticated_live_trainer(repository, config)
    profiler_source = Path(__file__).resolve()
    source_commit = config.get("source_commit")
    if (
        _sha256_file(profiler_source) != receipt["profiler_sha256"]
        or hashlib.sha256(
            _git_blob_bytes(
                repository,
                f"{source_commit}:scripts/profile_unicom_training_step.py",
            )
        ).hexdigest()
        != receipt["profiler_sha256"]
    ):
        raise ValueError("quality profile profiler source authority differs")
    validator = getattr(trainer, "validate_training_run_receipt_v2", None)
    if not callable(validator):
        raise ValueError("quality profile live run receipt validator differs")
    validator(run_receipt, evidence_root=run_receipt_path.parent)
    checkpoints = run_receipt.get("checkpoints")
    if type(checkpoints) is not list or not checkpoints:
        raise ValueError("quality profile run checkpoint chain differs")
    terminal = checkpoints[-1]
    if (
        type(terminal) is not dict
        or terminal.get("epoch") != 16
        or terminal.get("path") != "epoch-0016.pt"
        or terminal.get("sha256") != receipt["checkpoint"]["sha256"]
        or terminal.get("bytes") != receipt["checkpoint"]["bytes"]
        or checkpoint_path != run_receipt_path.parent / terminal["path"]
    ):
        raise ValueError("quality profile terminal checkpoint authority differs")
    checkpoint = _load_checkpoint(checkpoint_path)
    _validate_checkpoint_authority_for_profile(
        checkpoint,
        profile_kind="quality",
        live_trainer_sha256=receipt["live_trainer_sha256"],
    )
    if (
        receipt["checkpoint_protocol"].get("environment") != environment
        or receipt["checkpoint_protocol"].get("environment_sha256")
        != environment_authority["sha256"]
        or receipt["environment"] != environment
        or receipt["environment_sha256"] != environment_authority["sha256"]
    ):
        raise ValueError("quality profile environment authority differs")
    if (
        receipt["inference_signature"] != current_signature
        or _rebuild_checkpoint_inference_signature(checkpoint, current_signature)
        != current_signature
    ):
        raise ValueError("quality profile inference authority differs")
    if (
        checkpoint["epoch"] != 16
        or checkpoint["training_protocol"] != receipt["checkpoint_protocol"]
        or run_receipt.get("training_protocol") != receipt["checkpoint_protocol"]
    ):
        raise ValueError("quality profile checkpoint protocol chain differs")
    expected_parameters = _checkpoint_parameter_schema(
        checkpoint, current_signature
    )
    expected_optimizer = _optimizer_state_dict_schema(checkpoint["optimizer"])
    if (
        receipt["parameter_schema"] != expected_parameters
        or receipt["optimizer_schema"] != expected_optimizer
    ):
        raise ValueError("quality profile registered schema differs")


def validate_runtime_profile(
    receipt: object,
    *,
    expected_mode: str,
    checkpoint: Path,
    run_receipt: Path,
    config: Path,
    expected_environment: object,
    expected_environment_sha256: str | None = None,
) -> None:
    """Validate one runtime receipt against caller-owned external authorities."""

    _validate_profile_v2(receipt, expected_kind="runtime")
    _validate_runtime_authority_chain(
        receipt,
        checkpoint=checkpoint,
        run_receipt=run_receipt,
        config=config,
    )
    if expected_environment_sha256 is None and type(expected_environment) is dict:
        expected_environment_sha256 = hashlib.sha256(
            (json.dumps(expected_environment, indent=2, allow_nan=False) + "\n").encode()
        ).hexdigest()
    if (
        type(receipt) is not dict
        or expected_mode not in RUNTIME_PROTOCOLS
        or receipt["runtime_mode"] != expected_mode
        or receipt["checkpoint"] != _file_authority(checkpoint)
        or receipt["run_receipt"] != _file_authority(run_receipt)
        or receipt["config"] != _file_authority(config)
        or type(expected_environment) is not dict
        or receipt["environment"] != expected_environment
        or receipt["environment_sha256"] != expected_environment_sha256
        or any(
            type(receipt["environment"].get(key)) is not type(value)
            or receipt["environment"].get(key) != value
            for key, value in expected_environment.items()
        )
    ):
        raise ValueError("runtime profile external authority differs")


def _validate_runtime_authority_chain(
    receipt: dict[str, object], *, checkpoint: Path, run_receipt: Path, config: Path
) -> None:
    checkpoint_path = _require_persisted_authority(receipt["checkpoint"], "checkpoint")
    run_receipt_path = _require_persisted_authority(receipt["run_receipt"], "run_receipt")
    config_path = _require_persisted_authority(receipt["config"], "config")
    if (
        checkpoint_path != checkpoint.resolve()
        or run_receipt_path != run_receipt.resolve()
        or config_path != config.resolve()
    ):
        raise ValueError("runtime profile external authority differs")
    config_object = _strict_json_object(config_path)
    run_object = _strict_json_object(run_receipt_path)
    repository = Path(__file__).resolve().parents[1]
    if (
        config_object.get("parent_trainer_commit") != PARENT_TRAINER_COMMIT
        or config_object.get("parent_trainer_path") != PARENT_TRAINER_PATH
        or config_object.get("parent_trainer_sha256") != PARENT_TRAINER_SHA256
        or config_object.get("profiler_sha256") != receipt["profiler_sha256"]
        or _sha256_file(Path(__file__).resolve()) != receipt["profiler_sha256"]
        or hashlib.sha256(
            _git_blob_bytes(
                repository,
                f"{config_object.get('source_commit')}:scripts/profile_unicom_training_step.py",
            )
        ).hexdigest()
        != receipt["profiler_sha256"]
    ):
        raise ValueError("runtime profile config/source authority differs")
    trainer = _load_authenticated_parent_trainer(repository, PARENT_TRAINER_SOURCE)
    validator = getattr(trainer, "validate_training_run_receipt", None)
    if not callable(validator):
        raise ValueError("runtime profile run validator differs")
    validator(run_object)
    external_run = reload_normal_legacy_runtime_authority(config_path, run_receipt_path)
    if external_run != run_object:
        raise ValueError("legacy runtime receipt authority differs")
    validate_registered_legacy_runtime_chain(
        run_object, authority=config_object.get("legacy_runtime_authority")
    )
    history_path = Path(run_object["history"]["path"])
    if _file_authority(history_path) != run_object["history"]:
        raise ValueError("legacy runtime history authority differs")
    checkpoints = run_object.get("checkpoints")
    for row in checkpoints:
        if _file_authority(Path(row["path"])) != {
            "path": str(Path(row["path"]).resolve()),
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }:
            raise ValueError("legacy runtime checkpoint authority differs")
    terminal = checkpoints[-1] if type(checkpoints) is list and checkpoints else None
    if (
        type(terminal) is not dict
        or terminal.get("sha256") != receipt["checkpoint"]["sha256"]
        or terminal.get("bytes") != receipt["checkpoint"]["bytes"]
        or Path(terminal.get("path", "")).resolve() != checkpoint_path
    ):
        raise ValueError("runtime profile checkpoint chain differs")
    checkpoint_object = _load_checkpoint(checkpoint_path)
    _validate_checkpoint_authority_for_profile(
        checkpoint_object,
        profile_kind="runtime",
        live_trainer_sha256=receipt["live_trainer_sha256"],
    )
    external_signature = config_object.get("runtime_inference_signature")
    _validate_inference_signature(external_signature)
    if (
        receipt["inference_signature"] != external_signature
        or _rebuild_checkpoint_inference_signature(
            checkpoint_object, external_signature
        )
        != external_signature
    ):
        raise ValueError("runtime profile inference authority differs")
    if (
        checkpoint_object.get("epoch") != receipt["checkpoint_epoch"]
        or checkpoint_object.get("training_protocol") != receipt["checkpoint_protocol"]
        or receipt["parameter_schema"]
        != _checkpoint_parameter_schema(
            checkpoint_object, external_signature
        )
        or receipt["optimizer_schema"]
        != _optimizer_state_dict_schema(checkpoint_object.get("optimizer"))
    ):
        raise ValueError("runtime profile rebuilt authority differs")


def validate_registered_legacy_runtime_chain(
    receipt: object, *, authority: object | None = None
) -> None:
    """Require the exact completed historical seed-2 runtime authority."""

    if (
        type(receipt) is not dict
        or receipt.get("seed") != 2
        or receipt.get("arm") != "sampled_512"
        or receipt.get("protocol")
        != {
            "objective": "official-eight-mask",
            "selected_features": 512,
            "evaluation_features": 768,
        }
        or receipt.get("exit_status") != 0
        or [row.get("epoch") for row in receipt.get("checkpoints", [])]
        != [4, 8, 12, 16]
        or type(receipt.get("history")) is not dict
        or "--classifier-init" not in receipt.get("command", [])
        or receipt["command"][receipt["command"].index("--classifier-init") + 1]
        != "imprinted"
    ):
        raise ValueError("registered legacy seed2 runtime chain differs")
    if authority is None:
        return
    if (
        type(authority) is not dict
        or tuple(authority) != ("run_receipt", "config", "history", "checkpoints")
        or receipt.get("history") != authority.get("history")
        or receipt.get("checkpoints") != authority.get("checkpoints")
        or receipt.get("config_path") != authority.get("config", {}).get("path")
        or receipt.get("config_sha256") != authority.get("config", {}).get("sha256")
    ):
        raise ValueError("registered legacy external authority differs")
    persisted = {}
    for label, binding in (
        ("config", authority["config"]),
        ("history", authority["history"]),
        *((f"checkpoint-{epoch}", binding) for epoch, binding in zip(
            (4, 8, 12, 16), authority["checkpoints"], strict=True
        )),
    ):
        path = Path(binding["path"])
        payload = path.read_bytes()
        if (
            path.is_symlink()
            or hashlib.sha256(payload).hexdigest() != binding["sha256"]
            or len(payload) != binding["bytes"]
        ):
            raise ValueError("registered legacy external authority differs")
        persisted[label] = payload
    try:
        legacy_config = json.loads(persisted["config"])
        history = json.loads(persisted["history"])
    except (TypeError, ValueError) as error:
        raise ValueError("registered legacy semantic authority differs") from error
    if type(legacy_config) is not dict or type(history) is not list:
        raise ValueError("registered legacy semantic authority differs")
    for row in history:
        if type(row) is not dict or any(
            key in row and (type(row[key]) not in (int, float) or not math.isfinite(row[key]))
            for key in ("loss", "map", "rank1")
        ):
            raise ValueError("registered legacy history authority differs")


def reload_normal_legacy_runtime_authority(
    config_path: Path, run_receipt_path: Path
) -> dict[str, object]:
    config = _strict_json_object(config_path)
    authority = config.get("legacy_runtime_authority")
    binding = authority.get("run_receipt") if type(authority) is dict else None
    _validate_file_authority(binding, "legacy run receipt")
    receipt, payload = _strict_json_object_payload(run_receipt_path)
    if (
        run_receipt_path.is_symlink()
        or str(run_receipt_path.resolve()) != binding["path"]
        or hashlib.sha256(payload).hexdigest() != binding["sha256"]
        or len(payload) != binding["bytes"]
    ):
        raise ValueError("legacy run receipt external authority differs")
    validate_registered_legacy_runtime_chain(receipt, authority=authority)
    return receipt


def validate_live_runtime_inference_authority(
    checkpoint: dict[str, object], *, model: object, descriptor: object,
    external: dict[str, object],
) -> None:
    import torch

    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("live inference checkpoint differs")
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    state = model.state_dict()
    if set(state) != set(parameters) | set(buffers):
        raise ValueError("live inference tensor inventory differs")
    if set(checkpoint["model"]) != set(state):
        raise ValueError("live inference checkpoint inventory differs")
    aggregate = hashlib.sha256()
    rows = []
    total_bytes = 0
    for name in sorted(state):
        value = state[name]
        checkpoint_value = checkpoint["model"][name]
        if not torch.equal(value.detach().cpu(), checkpoint_value.detach().cpu()):
            raise ValueError("live inference checkpoint tensor differs")
        payload = (
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes(order="C")
        )
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
        rows.append(row)
        total_bytes += len(payload)
    if (
        not isinstance(descriptor, torch.Tensor)
        or descriptor.dtype != torch.float32
        or descriptor.ndim != 2
        or descriptor.shape[1] != 512
        or not torch.isfinite(descriptor).all()
    ):
        raise ValueError("live inference descriptor differs")
    descriptor_payload = (
        descriptor.detach()
        .cpu()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .numpy()
        .tobytes(order="C")
    )
    expected = {
        "schema": "unicom-inference-signature-v1",
        "tensors": rows,
        "total_bytes": total_bytes,
        "aggregate_sha256": aggregate.hexdigest(),
        "descriptor_dtype": str(descriptor.dtype),
        "descriptor_dimension": descriptor.shape[1],
        "descriptor_sha256": hashlib.sha256(descriptor_payload).hexdigest(),
        "operations": list(INFERENCE_OPERATIONS),
    }
    if external != expected:
        raise ValueError("live inference authority differs")


def validate_runtime_descriptor_authority(
    signature: object, descriptor_path: Path
) -> None:
    import io

    if type(signature) is not dict or descriptor_path.is_symlink():
        raise ValueError("runtime descriptor authority differs")
    payload = descriptor_path.read_bytes()
    try:
        descriptor = np.load(io.BytesIO(payload), allow_pickle=False)
        canonical = io.BytesIO()
        np.save(canonical, descriptor, allow_pickle=False)
    except Exception as error:
        raise ValueError("runtime descriptor authority differs") from error
    if (
        canonical.getvalue() != payload
        or descriptor.dtype != np.float32
        or descriptor.shape != (1, 512)
        or signature.get("descriptor_dtype") != "torch.float32"
        or signature.get("descriptor_dimension") != 512
        or signature.get("descriptor_sha256")
        != hashlib.sha256(descriptor.tobytes(order="C")).hexdigest()
        or signature.get("operations") != list(INFERENCE_OPERATIONS)
    ):
        raise ValueError("runtime descriptor authority differs")


def _process_median_wall(receipt: dict[str, object]) -> float:
    return float(
        np.median(
            np.asarray(
                [sample["step_wall_seconds"] for sample in receipt["timing_samples"]],
                dtype=np.float64,
            )
        )
    )


def _aligned_losses_pass(current: dict[str, object], composed: dict[str, object]) -> bool:
    control = tuple(current["losses"])
    candidate = tuple(composed["losses"])
    differences = tuple(right - left for left, right in zip(control, candidate, strict=True))
    denominator = max(math.sqrt(math.fsum(value * value for value in control)), 1e-12)
    relative = math.sqrt(math.fsum(value * value for value in differences)) / denominator
    maximum = max(abs(value) for value in differences)
    return relative <= RUNTIME_LOSS_REL_L2_MAX and maximum <= RUNTIME_LOSS_MAX_ABS


def compare_runtime_smoke(receipts: object) -> str:
    """Select a runtime from eight fresh A-B-B-A-A-B-B-A process receipts."""

    if type(receipts) is not tuple or len(receipts) != 8:
        return "INVALID"
    try:
        for receipt in receipts:
            _validate_profile_v2(receipt, expected_kind="runtime")
        expected_modes = ("current", "composed", "composed", "current") * 2
        if tuple(receipt["runtime_mode"] for receipt in receipts) != expected_modes:
            return "INVALID"
        if any(
            left["finished_unix_ns"] > right["started_unix_ns"]
            for left, right in zip(receipts, receipts[1:], strict=False)
        ):
            return "INVALID"
        shared_keys = (
            "parent_trainer_source",
            "parent_trainer_sha256",
            "live_trainer_sha256",
            "profiler_sha256",
            "checkpoint",
            "run_receipt",
            "config",
            "checkpoint_epoch",
            "checkpoint_protocol",
            "inference_signature",
            "parameter_schema",
            "optimizer_schema",
            "environment_sha256",
            "environment",
        )
        authority = tuple(receipts[0][key] for key in shared_keys)
        if any(tuple(receipt[key] for key in shared_keys) != authority for receipt in receipts[1:]):
            return "INVALID"

        current_indices = (0, 3, 4, 7)
        composed_indices = (1, 2, 5, 6)
        nearest_pairs = ((0, 1), (3, 2), (4, 5), (7, 6))
        pair_ratios = tuple(
            _process_median_wall(receipts[composed_index])
            / _process_median_wall(receipts[current_index])
            for current_index, composed_index in nearest_pairs
        )
        current_wall = np.asarray(
            [
                sample["step_wall_seconds"]
                for index in current_indices
                for sample in receipts[index]["timing_samples"]
            ],
            dtype=np.float64,
        )
        composed_wall = np.asarray(
            [
                sample["step_wall_seconds"]
                for index in composed_indices
                for sample in receipts[index]["timing_samples"]
            ],
            dtype=np.float64,
        )
        pooled_ratio = float(np.median(composed_wall) / np.median(current_wall))
        losses_pass = all(
            _aligned_losses_pass(receipts[current_index], receipts[composed_index])
            for current_index, composed_index in nearest_pairs
        )
        allocated_ratio = max(
            receipts[index]["peak_allocated_bytes"] for index in composed_indices
        ) / max(receipts[index]["peak_allocated_bytes"] for index in current_indices)
        reserved_ratio = max(
            receipts[index]["peak_reserved_bytes"] for index in composed_indices
        ) / max(receipts[index]["peak_reserved_bytes"] for index in current_indices)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return "INVALID"
    if (
        pooled_ratio <= RUNTIME_POOLED_RATIO_MAX
        and all(ratio <= RUNTIME_PAIR_RATIO_MAX for ratio in pair_ratios)
        and losses_pass
        and allocated_ratio <= RUNTIME_MEMORY_RATIO_MAX
        and reserved_ratio <= RUNTIME_MEMORY_RATIO_MAX
    ):
        return "PASS_COMPOSED"
    return "PASS_CURRENT"


def _profile_samples(
    payload: object, expected_init: str
) -> tuple[tuple, tuple, tuple, tuple[int, int]]:
    if type(payload) is not dict:
        raise TypeError("ABBA profile must be an object")
    if tuple(payload) != PROFILE_KEYS:
        raise ValueError("ABBA profile schema differs")
    if (
        payload["schema_version"] != "unicom-training-step-profile-v1"
        or payload["classifier_init"] != expected_init
        or payload["warmup_steps"] != WARMUP_STEPS
        or payload["measure_steps"] != MEASURE_STEPS
        or payload["profiler_steps"] != PROFILER_STEPS
    ):
        raise ValueError("ABBA profile binding differs")
    metadata = (
        payload["run_checkpoint_sha256"],
        payload["trainer_sha256"],
        payload["objective_sha256"],
        payload["profiler_sha256"],
        payload["checkpoint_epoch"],
        payload["runtime"],
    )
    if (
        any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in metadata[:4]
        )
        or type(metadata[4]) is not int
        or metadata[4] < 1
        or type(metadata[5]) is not dict
    ):
        raise ValueError("ABBA profile provenance differs")
    started = payload["started_unix_ns"]
    finished = payload["finished_unix_ns"]
    if (
        type(started) is not int
        or type(finished) is not int
        or started <= 0
        or finished <= started
    ):
        raise ValueError("ABBA profile order differs")
    timing = payload["timing_samples"]
    fusible = payload["fusible_samples"]
    if (
        type(timing) is not list
        or len(timing) != MEASURE_STEPS
        or type(fusible) is not list
        or len(fusible) != PROFILER_STEPS
    ):
        raise ValueError("ABBA profile sample counts differ")
    timing_tuple = tuple(timing)
    fusible_tuple = tuple(fusible)
    recomputed = summarize_profile(timing_tuple, fusible_tuple)
    if payload["summary"] != recomputed:
        raise ValueError("ABBA profile summary differs")
    return timing_tuple, fusible_tuple, metadata, (started, finished)


def aggregate_abba_profiles(profiles: object) -> dict[str, dict[str, object]]:
    """Pool two fresh checkpoint reloads per arm in fixed A-B-B-A order."""

    if type(profiles) is not tuple or len(profiles) != 4:
        raise ValueError("ABBA requires exactly four profiles")
    expected = ("random", "imprinted", "imprinted", "random")
    validated = tuple(
        _profile_samples(profile, classifier_init)
        for profile, classifier_init in zip(profiles, expected, strict=True)
    )
    shared = tuple(row[2][1:] for row in validated)
    if any(value != shared[0] for value in shared[1:]):
        raise ValueError("ABBA profile provenance differs")
    if validated[0][2][0] != validated[3][2][0] or validated[1][2][0] != validated[2][2][0]:
        raise ValueError("ABBA checkpoint provenance differs")
    if any(
        earlier[3][1] > later[3][0]
        for earlier, later in zip(validated, validated[1:], strict=False)
    ):
        raise ValueError("ABBA profile order differs")
    result: dict[str, dict[str, object]] = {}
    for classifier_init, indices in (("random", (0, 3)), ("imprinted", (1, 2))):
        timing = validated[indices[0]][0] + validated[indices[1]][0]
        fusible = validated[indices[0]][1] + validated[indices[1]][1]
        result[classifier_init] = summarize_profile(timing, fusible)
    return result


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-checkpoint", required=True, type=Path)
    parser.add_argument("--run-receipt", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--runtime-mode", default="current", choices=tuple(RUNTIME_PROTOCOLS)
    )
    parser.add_argument(
        "--profile-kind", default="runtime", choices=("runtime", "quality")
    )
    parser.add_argument("--parent-trainer-source", default=PARENT_TRAINER_SOURCE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--environment-authority", type=Path)
    parser.add_argument("--environment-sha256")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=MEASURE_STEPS)
    parser.add_argument("--profiler-steps", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--publication-stage")
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--authority-preflight-only", action="store_true")
    args = parser.parse_args(arguments)
    if args.profiler_steps is None:
        args.profiler_steps = PROFILER_STEPS if args.profile_kind == "runtime" else 0
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trainer(source: Path):
    source = source.resolve()
    spec = importlib.util.spec_from_file_location("_unicom_profile_trainer", source)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the UniCOM trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != source:
        raise ValueError("loaded trainer source differs")
    return module


def _git_blob_bytes(repository: Path, source: str) -> bytes:
    repository = repository.resolve()
    if not repository.is_dir() or repository.is_symlink():
        raise ValueError("parent trainer repository differs")
    result = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "blob", source],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError("parent trainer Git blob differs")
    return result.stdout


def _load_authenticated_parent_trainer(repository: Path, source: str):
    if source != PARENT_TRAINER_SOURCE:
        raise ValueError("parent trainer source differs")
    payload = _git_blob_bytes(repository, source)
    if hashlib.sha256(payload).hexdigest() != PARENT_TRAINER_SHA256:
        raise ValueError("parent trainer blob hash differs")
    temporary_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix=".unicom-parent-trainer-") as directory:
            temporary_path = Path(directory) / "_authenticated_parent_trainer.py"
            with temporary_path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            trainer = _load_trainer(temporary_path)
            trainer.__profile_source_sha256__ = PARENT_TRAINER_SHA256
            trainer.__profile_source_spec__ = PARENT_TRAINER_SOURCE
            return trainer
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_parent_checkpoint_authority(checkpoint: object) -> None:
    if (
        type(checkpoint) is not dict
        or type(checkpoint.get("training_protocol")) is not dict
        or checkpoint["training_protocol"].get("trainer_sha256")
        != PARENT_TRAINER_SHA256
    ):
        raise ValueError("checkpoint parent trainer source differs")


def _validate_checkpoint_authority_for_profile(
    checkpoint: object,
    *,
    profile_kind: str,
    live_trainer_sha256: str,
) -> None:
    if profile_kind == "runtime":
        _validate_parent_checkpoint_authority(checkpoint)
        return
    if (
        profile_kind != "quality"
        or not _lower_sha256(live_trainer_sha256)
        or type(checkpoint) is not dict
        or type(checkpoint.get("training_protocol")) is not dict
        or checkpoint["training_protocol"].get("trainer_sha256")
        != live_trainer_sha256
    ):
        raise ValueError("checkpoint live trainer source differs")


def _load_authenticated_live_trainer(repository: Path, config: dict[str, object]):
    source_commit = config.get("source_commit")
    expected_sha256 = config.get("live_trainer_sha256")
    if (
        type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or not _lower_sha256(expected_sha256)
    ):
        raise ValueError("live trainer config authority differs")
    source = repository.resolve() / PARENT_TRAINER_PATH
    if source.is_symlink() or not source.is_file() or _sha256_file(source) != expected_sha256:
        raise ValueError("live trainer file authority differs")
    blob = _git_blob_bytes(repository, f"{source_commit}:{PARENT_TRAINER_PATH}")
    if hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError("live trainer Git authority differs")
    trainer = _load_trainer(source)
    trainer.__profile_source_sha256__ = expected_sha256
    trainer.__profile_source_spec__ = f"{source_commit}:{PARENT_TRAINER_PATH}"
    return trainer


def _load_replay_trainer(
    args: argparse.Namespace,
    *,
    repository: Path,
    config: dict[str, object],
):
    if args.profile_kind == "runtime":
        return _load_authenticated_parent_trainer(
            repository, args.parent_trainer_source
        )
    if args.profile_kind == "quality":
        return _load_authenticated_live_trainer(repository, config)
    raise ValueError("profile kind differs")


def _construct_runtime(
    raw_model,
    classifier,
    *,
    protocol: dict[str, object],
    trainer,
    runtime_mode: str,
):
    import torch

    if runtime_mode not in RUNTIME_PROTOCOLS:
        raise ValueError("runtime override differs")
    runtime = RUNTIME_PROTOCOLS[runtime_mode]
    train_model = (
        torch.compile(raw_model, mode="reduce-overhead") if runtime.compile else raw_model
    )
    optimizer = trainer.build_optimizer(
        raw_model,
        classifier,
        learning_rate=protocol["learning_rate"],
        classifier_learning_rate=protocol["classifier_learning_rate"],
        fused=runtime.fused,
    )
    step_ema = trainer.StepEMA(raw_model, classifier) if runtime.ema else None
    return train_model, optimizer, step_ema


def _validate_counts(args: argparse.Namespace) -> None:
    expected_profiler_steps = PROFILER_STEPS if args.profile_kind == "runtime" else 0
    if (
        type(args.warmup_steps) is not int
        or type(args.measure_steps) is not int
        or type(args.profiler_steps) is not int
        or (args.warmup_steps, args.measure_steps, args.profiler_steps)
        != (WARMUP_STEPS, MEASURE_STEPS, expected_profiler_steps)
    ):
        raise ValueError("step counts differ from the registered profiler")
    if type(args.bootstrap_seed) is not int or args.bootstrap_seed != BOOTSTRAP_SEED:
        raise ValueError("bootstrap seed differs from the registered profiler")


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
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
    if type(payload) is not dict or tuple(payload) != expected:
        raise ValueError("training checkpoint schema differs")
    if type(payload["training_protocol"]) is not dict:
        raise ValueError("training checkpoint protocol differs")
    if type(payload["selection_holdout"]) is not dict:
        raise ValueError("training checkpoint holdout differs")
    return payload


def _restore_checkpoint_payload(payload: dict[str, Any], state: dict[str, Any]) -> int:
    import torch

    protocol = state["protocol"]
    raw_model = state["raw_model"]
    classifier = state["classifier"]
    step_ema = state["step_ema"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    scaler = state["scaler"]
    mask_generator = state["mask_generator"]
    if payload["selection_holdout"] != state["holdout"]:
        raise ValueError("training checkpoint holdout differs")
    if payload["training_protocol"] != protocol:
        raise ValueError("training checkpoint protocol differs")
    raw_model.load_state_dict(payload["model"], strict=True)
    value = payload["classifier"]
    if type(value) is not torch.Tensor or value.shape != classifier.shape:
        raise ValueError("training checkpoint classifier differs")
    with torch.no_grad():
        classifier.copy_(value)
    if step_ema is None:
        if type(payload["ema"]) is not dict:
            raise ValueError("training checkpoint EMA differs")
    else:
        step_ema.load_state_dict(payload["ema"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is None:
        if payload["scaler"] is not None:
            raise ValueError("training checkpoint scaler differs")
    else:
        scaler.load_state_dict(payload["scaler"])
    mask_generator.set_state(payload["mask_generator"])
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_states = payload["cuda_rng_states"]
    if type(cuda_states) is not list or not cuda_states:
        raise ValueError("training checkpoint CUDA RNG state differs")
    torch.cuda.set_rng_state_all(cuda_states)
    epoch = payload["epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("training checkpoint epoch differs")
    return epoch


def _build_replay_state(
    args: argparse.Namespace,
    trainer,
    *,
    live_trainer_sha256: str,
):
    import torch
    from PIL import Image

    from sfora.unicom_inshop import parse_inshop_partition
    from sfora.unicom_training import experiment_stream_seed, identity_holdout

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UniCOM step replay")
    checkpoint = _load_checkpoint(args.run_checkpoint)
    _validate_checkpoint_authority_for_profile(
        checkpoint,
        profile_kind=args.profile_kind,
        live_trainer_sha256=live_trainer_sha256,
    )
    protocol = dict(checkpoint["training_protocol"])
    holdout = dict(checkpoint["selection_holdout"])
    if protocol.get("initial_checkpoint_sha256") != _sha256_file(args.initial_checkpoint):
        raise ValueError("initial UniCOM checkpoint differs")
    if protocol.get("unicom_revision") != trainer._git_revision(args.unicom_checkout):
        raise ValueError("UniCOM checkout revision differs")

    seed = protocol["seed"]
    if type(seed) is not int:
        raise TypeError("training seed differs")
    trainer._seed_process(seed)
    device = torch.device("cuda")
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(row for row in records if row.split == "train")
    optimization, _query, _gallery, labels = identity_holdout(
        train_records,
        fraction=protocol["holdout_fraction"],
        seed=protocol["holdout_seed"],
    )
    raw_model, eval_transform = trainer._load_official_model(
        args.unicom_checkout,
        args.initial_checkpoint,
    )
    raw_model = raw_model.to(device)
    descriptor_record = next((row for row in records if row.split == "query"), None)
    if descriptor_record is None:
        raise ValueError("registered runtime descriptor record differs")
    checkpoint_shape = checkpoint["classifier"].shape
    classifier = torch.nn.Parameter(
        torch.empty(checkpoint_shape, device=device, dtype=torch.float32)
    )
    train_model, optimizer, step_ema = _construct_runtime(
        raw_model,
        classifier,
        protocol=protocol,
        trainer=trainer,
        runtime_mode=args.runtime_mode,
    )
    sampler = trainer.PaddedEpochSampler(
        size=len(optimization),
        batch_size=protocol["batch_size"],
        seed=seed,
    )
    data_generator = torch.Generator().manual_seed(experiment_stream_seed(seed, 2_000))
    loader = torch.utils.data.DataLoader(
        trainer.InshopTrainDataset(
            optimization,
            labels,
            trainer.build_train_transform(336),
        ),
        batch_size=protocol["batch_size"],
        sampler=sampler,
        num_workers=protocol["workers"],
        pin_memory=True,
        drop_last=True,
        worker_init_fn=trainer._seed_worker,
        generator=data_generator,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[protocol["learning_rate"], protocol["classifier_learning_rate"]],
        steps_per_epoch=len(loader),
        epochs=protocol["epochs"],
        pct_start=0.1,
    )
    mask_generator = torch.Generator(device=device).manual_seed(
        experiment_stream_seed(seed, 3_000)
    )
    scaler = (
        None
        if protocol["bf16"]
        else torch.amp.GradScaler("cuda", growth_interval=200)
    )
    state = {
        "raw_model": raw_model,
        "train_model": train_model,
        "classifier": classifier,
        "step_ema": step_ema,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loader": loader,
        "mask_generator": mask_generator,
        "scaler": scaler,
        "device": device,
        "protocol": protocol,
        "holdout": holdout,
        "trainable_parameters": tuple(raw_model.parameters()) + (classifier,),
    }
    start_epoch = _restore_checkpoint_payload(checkpoint, state)
    del checkpoint
    raw_model.eval()
    with Image.open(descriptor_record.image_path) as image, torch.no_grad():
        full_descriptor = raw_model(
            eval_transform(image.convert("RGB")).unsqueeze(0).to(device)
        ).float()
        full_descriptor = torch.nn.functional.normalize(full_descriptor, dim=1)
        state["authority_descriptor"] = (
            full_descriptor[:, :512].detach().cpu().contiguous()
        )
    sampler.set_epoch(start_epoch)
    trainer._seed_training_loader(loader, seed=seed, epoch=start_epoch)
    if step_ema is not None:
        step_ema.register_step_hook(optimizer)
    raw_model.train()
    state["checkpoint_epoch"] = start_epoch
    return state


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _cuda_event(torch):
    return torch.cuda.Event(enable_timing=True)


def _timing_row(
    wall: float,
    component_spans: Sequence[float],
    cuda_step: float,
) -> dict[str, float]:
    spans = tuple(component_spans)
    if len(spans) != len(COMPONENT_KEYS):
        raise ValueError("timing component count differs")
    return {
        "step_wall_seconds": wall,
        "cuda_step_seconds": cuda_step,
        **dict(zip(COMPONENT_KEYS, spans, strict=True)),
    }


def _event_timing_row(wall: float, boundaries: Sequence[object]) -> dict[str, float]:
    events = tuple(boundaries)
    if len(events) != len(COMPONENT_KEYS) + 1:
        raise ValueError("timing boundary count differs")
    component_spans = tuple(
        float(events[index].elapsed_time(events[index + 1])) / 1_000.0
        for index in range(len(COMPONENT_KEYS))
    )
    cuda_step = float(events[0].elapsed_time(events[-1])) / 1_000.0
    return _timing_row(wall, component_spans, cuda_step)


def _step_replay_scheduler(scheduler: object) -> None:
    if scheduler.last_epoch == scheduler.total_steps:
        return
    scheduler.step()


def _gradients_are_finite(parameters: Sequence[object]) -> bool:
    import torch

    values = tuple(parameters)
    return bool(
        values
        and all(
            isinstance(getattr(parameter, "grad", None), torch.Tensor)
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in values
        )
    )


def _optimizer_parameters(optimizer: object) -> tuple[object, ...]:
    return tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def _optimizer_step(state: dict[str, Any]) -> dict[str, object]:
    optimizer = state["optimizer"]
    scaler = state["scaler"]
    parameters = tuple(state.get("trainable_parameters", _optimizer_parameters(optimizer)))
    scale_before: float | None = None
    if scaler is None:
        if not _gradients_are_finite(parameters):
            raise ValueError("unscaled gradient is nonfinite")
        optimizer.step()
        skipped = False
    else:
        scale_before = float(scaler.get_scale())
        if not math.isfinite(scale_before) or scale_before <= 0.0:
            raise ValueError("GradScaler scale differs")
        scaler.unscale_(optimizer)
        if not _gradients_are_finite(parameters):
            raise ValueError("unscaled gradient is nonfinite")
        observed_steps = 0

        def observe_step(_optimizer, _args, _kwargs):
            nonlocal observed_steps
            observed_steps += 1

        hook = optimizer.register_step_post_hook(observe_step)
        try:
            scaler.step(optimizer)
        finally:
            hook.remove()
        scaler.update()
        skipped = observed_steps != 1
        if skipped:
            raise ValueError("GradScaler skipped an optimizer step")
    _step_replay_scheduler(state["scheduler"])
    scale_after = None if scaler is None else float(scaler.get_scale())
    if scale_after is not None and (not math.isfinite(scale_after) or scale_after <= 0.0):
        raise ValueError("GradScaler scale differs")
    return {
        "enabled": scaler is not None,
        "scale_before": scale_before,
        "scale_after": scale_after,
        "skipped": skipped,
    }


def _training_step(state: dict[str, Any], batch, *, measured: bool) -> dict[str, object] | None:
    import torch

    protocol = state["protocol"]
    train_model = state["train_model"]
    optimizer = state["optimizer"]
    scaler = state["scaler"]
    device = state["device"]
    boundaries = [_cuda_event(torch) for _ in range(len(COMPONENT_KEYS) + 1)]
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    boundaries[0].record()
    images, labels = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device=device, dtype=torch.int64, non_blocking=True)
    boundaries[1].record()
    optimizer.zero_grad(set_to_none=True)
    boundaries[2].record()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=protocol["bf16"]):
        embeddings = train_model(images)
    boundaries[3].record()
    embeddings = embeddings.float()
    masks = trainer_objective_masks(
        state,
        dimension=embeddings.shape[1],
    )
    loss = trainer_loss(state, embeddings, labels, masks)
    boundaries[4].record()

    head_boundary_recorded = False

    def mark_head_backward(gradient):
        nonlocal head_boundary_recorded
        if not head_boundary_recorded:
            boundaries[5].record()
            head_boundary_recorded = True
        return gradient

    hook = embeddings.register_hook(mark_head_backward)
    try:
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
    finally:
        hook.remove()
    if not head_boundary_recorded:
        raise RuntimeError("embedding-gradient boundary was not observed")
    boundaries[6].record()
    scaler_decision = _optimizer_step(state)
    boundaries[7].record()
    loss_value = float(loss.detach())
    if not math.isfinite(loss_value):
        raise ValueError("measured loss is nonfinite")
    boundaries[8].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start
    if not measured:
        return None
    return {
        "timing": _event_timing_row(float(wall), boundaries),
        "loss": loss_value,
        "gradients_finite": True,
        "scaler": scaler_decision,
    }


def trainer_objective_masks(state: dict[str, Any], *, dimension: int):
    trainer = state["trainer"]
    protocol = state["protocol"]
    return trainer.objective_masks(
        protocol["objective"],
        dimension=dimension,
        selected=protocol["selected_features"],
        generator=state["mask_generator"],
        device=state["device"],
    )


def trainer_loss(state: dict[str, Any], embeddings, labels, masks):
    trainer = state["trainer"]
    protocol = state["protocol"]
    return trainer.sharded_mask_arcface_loss(
        embeddings,
        state["classifier"],
        labels,
        masks,
        margin=protocol["margin"],
        scale=protocol["scale"],
    )


def _profile_objective(state: dict[str, Any], embeddings, labels) -> float:
    import torch

    detached = embeddings.detach().requires_grad_(True)
    classifier = state["classifier"].detach().requires_grad_(True)
    profile_state = dict(state)
    profile_state["classifier"] = classifier
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof,
        torch.profiler.record_function(OBJECTIVE_MARKER),
    ):
        masks = trainer_objective_masks(profile_state, dimension=detached.shape[1])
        loss = trainer_loss(profile_state, detached, labels, masks)
        loss.backward()
    torch.cuda.synchronize()
    return fusible_nonbackbone_seconds(tuple(prof.events()))


def _reset_cuda_peaks() -> None:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def _objective_profile_step(state: dict[str, Any], batch) -> float:
    import torch

    images, labels = batch
    torch.cuda.synchronize()
    images = images.to(state["device"], non_blocking=True)
    labels = labels.to(state["device"], dtype=torch.int64, non_blocking=True)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=state["protocol"]["bf16"],
    ):
        embeddings = state["train_model"](images).float()
    return _profile_objective(state, embeddings, labels)


def _execute_profile_phases(
    state: dict[str, Any], *, profile_kind: str
) -> dict[str, object]:
    if profile_kind not in {"runtime", "quality"}:
        raise ValueError("profile kind differs")
    iterator = iter(state["loader"])
    for _index in range(WARMUP_STEPS):
        batch, iterator = _next_batch(iterator, state["loader"])
        if _training_step(state, batch, measured=False) is not None:
            raise ValueError("warmup step emitted measured evidence")
    _reset_cuda_peaks()
    timing_rows: list[dict[str, float]] = []
    losses: list[float] = []
    gradients: list[bool] = []
    scaler_decisions: list[dict[str, object]] = []
    for _index in range(MEASURE_STEPS):
        batch, iterator = _next_batch(iterator, state["loader"])
        row = _training_step(state, batch, measured=True)
        if type(row) is not dict or tuple(row) != (
            "timing",
            "loss",
            "gradients_finite",
            "scaler",
        ):
            raise ValueError("measured step evidence differs")
        timing_rows.append(_validate_timing_sample(row["timing"], len(timing_rows)))
        losses.append(_finite_float(row["loss"], "measured loss"))
        if row["gradients_finite"] is not True:
            raise ValueError("unscaled gradient is nonfinite")
        gradients.append(True)
        decision = row["scaler"]
        if type(decision) is not dict or decision.get("skipped") is not False:
            raise ValueError("GradScaler decision differs")
        scaler_decisions.append(decision)
    objective_samples: list[float] = []
    if profile_kind == "runtime":
        for _index in range(PROFILER_STEPS):
            batch, iterator = _next_batch(iterator, state["loader"])
            objective_samples.append(
                _finite_float(
                    _objective_profile_step(state, batch),
                    "objective profiler sample",
                    nonnegative=True,
                )
            )
    return {
        "optimizer_step_count": WARMUP_STEPS + MEASURE_STEPS,
        "objective_call_count": len(objective_samples),
        "losses": losses,
        "unscaled_gradients_finite": gradients,
        "scaler_decisions": scaler_decisions,
        "timing_samples": timing_rows,
        "objective_samples": objective_samples,
    }


def _file_authority(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError("profile authority path differs")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError("profile authority path differs")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _strict_json_object_payload(path: Path) -> tuple[dict[str, object], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("profile JSON authority differs")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("profile JSON authority has duplicate keys")
            result[key] = value
        return result

    try:
        payload = path.read_bytes()
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("profile JSON authority has nonfinite value")
            ),
        )
    except Exception as error:
        raise ValueError("profile JSON authority differs") from error
    if type(value) is not dict:
        raise ValueError("profile JSON authority differs")
    canonical = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if payload != canonical:
        raise ValueError("profile JSON authority is noncanonical")
    return value, payload


def _strict_json_object(path: Path) -> dict[str, object]:
    return _strict_json_object_payload(path)[0]


def _load_profile_authorities(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    run_receipt = _strict_json_object(args.run_receipt)
    config = _strict_json_object(args.config)
    expected_parent = (
        PARENT_TRAINER_COMMIT,
        PARENT_TRAINER_PATH,
        PARENT_TRAINER_SHA256,
    )
    observed_parent = (
        config.get("parent_trainer_commit"),
        config.get("parent_trainer_path"),
        config.get("parent_trainer_sha256"),
    )
    if observed_parent != expected_parent:
        raise ValueError("profile config parent trainer authority differs")
    historical_signature = config.get("runtime_inference_signature")
    _validate_inference_signature(historical_signature)
    if args.profile_kind == "quality":
        inference_signature = run_receipt.get("inference_signature")
        _require_quality_signature_structure(
            inference_signature, historical_signature
        )
    elif args.profile_kind == "runtime":
        inference_signature = historical_signature
    else:
        raise ValueError("profile kind differs")
    checkpoint = _load_checkpoint(args.run_checkpoint)
    if (
        _rebuild_checkpoint_inference_signature(checkpoint, inference_signature)
        != inference_signature
    ):
        raise ValueError("profile config/checkpoint inference authority differs")
    run_signature = run_receipt.get("inference_signature")
    if run_signature is not None and run_signature != inference_signature:
        raise ValueError("profile run receipt inference authority differs")
    checkpoint_authority = _file_authority(args.run_checkpoint)
    checkpoints = run_receipt.get("checkpoints")
    if (
        type(checkpoints) is not list
        or not any(
            type(row) is dict
            and row.get("sha256") == checkpoint_authority["sha256"]
            and row.get("bytes") == checkpoint_authority["bytes"]
            for row in checkpoints
        )
    ):
        raise ValueError("profile run receipt checkpoint authority differs")
    return run_receipt, config, inference_signature


def _validate_quality_replay_inputs(
    args: argparse.Namespace,
    *,
    trainer,
    run_receipt: dict[str, object],
    config: dict[str, object],
    inference_signature: dict[str, object],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    profiler_sha256 = _sha256_file(Path(__file__).resolve())
    source_commit = config.get("source_commit")
    if (
        config.get("profiler_sha256") != profiler_sha256
        or hashlib.sha256(
            _git_blob_bytes(
                repository,
                f"{source_commit}:scripts/profile_unicom_training_step.py",
            )
        ).hexdigest()
        != profiler_sha256
    ):
        raise ValueError("quality profile profiler source authority differs")
    validator = getattr(trainer, "validate_training_run_receipt_v2", None)
    if not callable(validator):
        raise ValueError("quality profile live run receipt validator differs")
    validator(run_receipt, evidence_root=args.run_receipt.resolve().parent)
    checkpoint = _load_checkpoint(args.run_checkpoint)
    live_sha256 = config.get("live_trainer_sha256")
    _validate_checkpoint_authority_for_profile(
        checkpoint,
        profile_kind="quality",
        live_trainer_sha256=live_sha256,
    )
    checkpoints = run_receipt.get("checkpoints")
    terminal = checkpoints[-1] if type(checkpoints) is list and checkpoints else None
    checkpoint_authority = _file_authority(args.run_checkpoint)
    if (
        type(terminal) is not dict
        or terminal.get("epoch") != 16
        or terminal.get("path") != "epoch-0016.pt"
        or terminal.get("sha256") != checkpoint_authority["sha256"]
        or terminal.get("bytes") != checkpoint_authority["bytes"]
        or args.run_checkpoint.resolve()
        != args.run_receipt.resolve().parent / "epoch-0016.pt"
        or checkpoint["epoch"] != 16
        or checkpoint["training_protocol"] != run_receipt.get("training_protocol")
        or run_receipt.get("inference_signature") != inference_signature
    ):
        raise ValueError("quality profile live checkpoint chain differs")


def _parameter_schema(raw_model: object, classifier: object) -> list[dict[str, object]]:
    rows = [
        {
            "name": f"raw_model.{name}",
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in sorted(raw_model.named_parameters())
    ]
    rows.append(
        {
            "name": "classifier",
            "shape": list(classifier.shape),
            "dtype": str(classifier.dtype),
        }
    )
    if not rows or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("profile parameter schema differs")
    return rows


def _optimizer_value_schema(value: object) -> dict[str, object]:
    import torch

    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if type(value) in (bool, int, float, str) or value is None:
        return {"kind": type(value).__name__ if value is not None else "none"}
    if type(value) in (tuple, list):
        return {
            "kind": type(value).__name__,
            "items": [_optimizer_value_schema(item) for item in value],
        }
    raise ValueError("profile optimizer state schema differs")


def _optimizer_schema(optimizer: object) -> dict[str, object]:
    return _optimizer_state_dict_schema(optimizer.state_dict())


def _optimizer_state_dict_schema(state: object) -> dict[str, object]:
    if type(state) is not dict or tuple(state) != ("state", "param_groups"):
        raise ValueError("profile optimizer schema differs")
    groups = []
    for group in state["param_groups"]:
        if type(group) is not dict or type(group.get("params")) is not list:
            raise ValueError("profile optimizer group schema differs")
        groups.append(
            {
                "parameter_count": len(group["params"]),
                "fields": {
                    key: _optimizer_value_schema(value)
                    for key, value in sorted(group.items())
                    if key != "params"
                },
            }
        )
    states = []
    for parameter, values in sorted(state["state"].items()):
        if type(parameter) is not int or type(values) is not dict:
            raise ValueError("profile optimizer state schema differs")
        states.append(
            {
                "parameter": parameter,
                "fields": {
                    key: _optimizer_value_schema(value) for key, value in sorted(values.items())
                },
            }
        )
    return {"param_groups": groups, "state": states}


def _establish_registered_deterministic_execution(
    torch, registered: object
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
        raise ValueError("registered profile deterministic environment differs")
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current not in (None, expected["cublas_workspace_config"]):
        raise ValueError("registered profile deterministic environment differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return expected


def _runtime_environment(torch, device: object) -> dict[str, object]:
    import timm
    import torchvision

    repository = Path(__file__).resolve().parents[1]
    return {
        "python_vv": subprocess.run(
            [sys.executable, "-VV"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "timm": str(timm.__version__),
        "numpy": str(np.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "compile": {
            "available": str(hasattr(torch, "compile")),
            "inductor": str(getattr(torch.version, "git_version", "unknown")),
        },
        "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "gpu_inventory": subprocess.run(
            [
                "nvidia-smi", "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines(),
        "pyproject_sha256": _sha256_file(repository / "pyproject.toml"),
        "uv_lock_sha256": _sha256_file(repository / "uv.lock"),
        "deterministic_execution": _establish_registered_deterministic_execution(
            torch,
            {
                "deterministic_algorithms": True,
                "cuda_matmul_tf32": False,
                "cudnn_tf32": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": ":4096:8",
            },
        ),
    }


def replay_profile(args: argparse.Namespace) -> dict[str, object]:
    started_unix_ns = time.time_ns()
    _validate_counts(args)
    if args.parent_trainer_source != PARENT_TRAINER_SOURCE:
        raise ValueError("parent trainer source differs")
    if args.profile_kind == "runtime":
        reload_normal_legacy_runtime_authority(args.config, args.run_receipt)
    repository = Path(__file__).resolve().parents[1]
    if args.environment_authority is None or args.environment_sha256 is None:
        raise ValueError("registered profile environment authority is required")
    environment = load_configured_environment_authority(
        args.config, args.environment_authority, args.environment_sha256
    )
    deterministic = environment.get("deterministic_execution")
    if (
        type(deterministic) is not dict
        or deterministic.get("cublas_workspace_config") != ":4096:8"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (None, ":4096:8")
    ):
        raise ValueError("registered profile deterministic environment differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    _establish_registered_deterministic_execution(
        torch, deterministic
    )
    registered_device = torch.device("cuda")
    if _runtime_environment(torch, registered_device) != environment:
        raise ValueError("live profile environment differs")
    run_receipt, config, inference_signature = _load_profile_authorities(args)
    trainer = _load_replay_trainer(
        args,
        repository=repository,
        config=config,
    )
    live_trainer_source = Path(__file__).with_name("train_unicom_inshop.py")
    live_trainer_sha256 = _sha256_file(live_trainer_source)
    if args.profile_kind == "quality":
        _validate_quality_replay_inputs(
            args,
            trainer=trainer,
            run_receipt=run_receipt,
            config=config,
            inference_signature=inference_signature,
        )
        if live_trainer_sha256 != config.get("live_trainer_sha256"):
            raise ValueError("quality profile live trainer hash differs")
    state = _build_replay_state(
        args,
        trainer,
        live_trainer_sha256=live_trainer_sha256,
    )
    state["trainer"] = trainer
    validate_live_runtime_inference_authority(
        _load_checkpoint(args.run_checkpoint),
        model=state["raw_model"],
        descriptor=state["authority_descriptor"],
        external=inference_signature,
    )
    try:
        phase = _execute_profile_phases(state, profile_kind=args.profile_kind)
        torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated(state["device"]))
        peak_reserved = int(torch.cuda.max_memory_reserved(state["device"]))
    finally:
        if state["step_ema"] is not None:
            state["step_ema"].release_step_hook()
    finished_unix_ns = time.time_ns()
    runtime = RUNTIME_PROTOCOLS[args.runtime_mode]
    payload = {
        "schema": "unicom-training-step-profile-v2",
        "profile_kind": args.profile_kind,
        "runtime_mode": args.runtime_mode,
        "parent_trainer_source": PARENT_TRAINER_SOURCE,
        "parent_trainer_sha256": PARENT_TRAINER_SHA256,
        "live_trainer_sha256": live_trainer_sha256,
        "profiler_sha256": _sha256_file(Path(__file__)),
        "checkpoint": _file_authority(args.run_checkpoint),
        "run_receipt": _file_authority(args.run_receipt),
        "config": _file_authority(args.config),
        "checkpoint_epoch": state["checkpoint_epoch"],
        "checkpoint_protocol": state["protocol"],
        "inference_signature": inference_signature,
        "runtime_overrides": {
            "compile": runtime.compile,
            "fused": runtime.fused,
            "ema": runtime.ema,
        },
        "warmup_steps": WARMUP_STEPS,
        "measure_steps": MEASURE_STEPS,
        "objective_steps": PROFILER_STEPS if args.profile_kind == "runtime" else 0,
        "optimizer_step_count": phase["optimizer_step_count"],
        "objective_call_count": phase["objective_call_count"],
        "timing_synchronized": True,
        "peak_reset": {
            "after_warmup": True,
            "before_measurement": True,
            "empty_cache": False,
        },
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": finished_unix_ns,
        "losses": phase["losses"],
        "unscaled_gradients_finite": phase["unscaled_gradients_finite"],
        "scaler_decisions": phase["scaler_decisions"],
        "timing_samples": phase["timing_samples"],
        "objective_samples": phase["objective_samples"],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "parameter_schema": _parameter_schema(state["raw_model"], state["classifier"]),
        "optimizer_schema": _optimizer_schema(state["optimizer"]),
        "environment_sha256": args.environment_sha256,
        "environment": environment,
    }
    if args.profile_kind == "quality":
        validate_quality_profile(payload)
    else:
        _validate_profile_v2(payload, expected_kind="runtime")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"profile output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()

    def validate(persisted: bytes) -> None:
        if persisted != encoded or json.loads(persisted) != payload:
            raise RuntimeError("persisted profile bytes differ")

    published = publish_bytes_noreplace(
        path,
        encoded,
        validator=validate,
    )
    published.close()


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        config = _strict_json_object(args.config)
        if args.environment_authority is None or args.environment_sha256 is None:
            raise ValueError("registered profile environment authority is required")
        environment = load_configured_environment_authority(
            args.config, args.environment_authority, args.environment_sha256
        )
        deterministic = environment["deterministic_execution"]
        if deterministic.get("cublas_workspace_config") != ":4096:8":
            raise ValueError("registered profile deterministic environment differs")
        current_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if current_workspace not in (None, ":4096:8"):
            raise ValueError("registered profile deterministic environment differs")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        builder_path = Path(__file__).with_name("build_unicom_fepf_run_config.py")
        specification = importlib.util.spec_from_file_location(
            "profile_exact_budget_builder", builder_path
        )
        if specification is None or specification.loader is None:
            raise ValueError("profile publication budget validator differs")
        builder = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(builder)
        budget_validator = (
            builder.validate_exact_publication_budget
            if args.authority_preflight_only
            else builder.validate_external_exact_publication_budget
        )
        budget_validator(config, config.get("publication_budget"))
        root = Path(config.get("artifact_root", ""))
        if (
            args.publication_stage is None
            or args.campaign_root is None
            or args.campaign_root.resolve() != root.resolve()
        ):
            raise ValueError("profile publication stage authority differs")
        budget = config.get("publication_budget")
        budget_payload = (json.dumps(budget, indent=2, allow_nan=False) + "\n").encode()
        rows = budget.get("publications") if type(budget) is dict else None
        expected_name = f"{args.publication_stage}:terminal"
        expected_path = args.output.resolve().relative_to(root.resolve()).as_posix()
        matching = [
            row for row in rows or []
            if row.get("name") == expected_name and row.get("path") == expected_path
        ]
        if (
            hashlib.sha256(budget_payload).hexdigest()
            != config.get("publication_budget_sha256")
            or len(matching) != 1
        ):
            raise ValueError("profile publication budget authority differs")
        row = matching[0]
        publisher = BudgetedPublisher(
            campaign_root=root,
            budget_path=root / config["publication_budget_path"],
            budget_sha256=config["publication_budget_sha256"],
            exact_budget=config["publication_budget"],
            physical_admission=not args.authority_preflight_only,
        )
        publisher.validate_payload(
            name=expected_name,
            destination=args.output,
            payload=b"",
        )
        if not args.authority_preflight_only:
            available = os.statvfs(root)
            if (
                available.f_bavail * available.f_frsize
                < row["persistent_bytes"] + row["temporary_bytes"]
                or available.f_favail
                < row["persistent_inodes"] + row["temporary_inodes"]
            ):
                raise OSError("profile publication capacity is insufficient")
        if args.authority_preflight_only:
            if args.profile_kind == "runtime":
                reload_normal_legacy_runtime_authority(args.config, args.run_receipt)
                signature = config.get("runtime_inference_signature")
                _validate_inference_signature(signature)
                checkpoint = _load_checkpoint(args.run_checkpoint)
                if (
                    _rebuild_checkpoint_inference_signature(checkpoint, signature)
                    != signature
                ):
                    raise ValueError("profile live inference authority differs")
            else:
                run_receipt, loaded_config, signature = _load_profile_authorities(args)
                trainer = _load_replay_trainer(
                    args,
                    repository=Path(__file__).resolve().parents[1],
                    config=loaded_config,
                )
                _validate_quality_replay_inputs(
                    args,
                    trainer=trainer,
                    run_receipt=run_receipt,
                    config=loaded_config,
                    inference_signature=signature,
                )
            return 0
        payload = replay_profile(args)
        encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
        if len(encoded) > row["persistent_bytes"]:
            raise OSError("profile publication bytes exceed budget")
        publisher.validate_payload(
            name=expected_name,
            destination=args.output,
            payload=encoded,
        )
        write_json_atomic(args.output, payload)
    except Exception as error:
        print(f"profiling failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile_kind": payload["profile_kind"],
                "runtime_mode": payload["runtime_mode"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
