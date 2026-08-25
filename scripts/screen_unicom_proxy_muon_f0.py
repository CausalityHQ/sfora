#!/usr/bin/env python3
"""Strict result contract for the preregistered UniCOM ProxyMuon F0 screen."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import NoReturn

import torch

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_probe import (
    PROBE_SELECTED_FEATURES,
    _fit_probe_trajectory_with_optimizer,
    class_mean_head,
    diagnostic_panel_losses,
    evaluate_probe_head,
    fit_proxy_muon_trajectory,
    fit_spherical_probe,
    split_probe_records,
)
from sfora.unicom_proxy_muon import (
    LR_GRID,
    OPTIMIZERS,
    PHASE1_SEEDS,
    PHASE2_SEEDS,
    RETAINED_STEPS,
    VALIDATION_STEPS,
    accuracy_noninferior,
    build_head_optimizer,
    compute_reach_step,
    decide_proxy_muon_f0,
    select_adamw_reference,
    select_learning_rate,
)
from sfora.unicom_training import identity_holdout

SCIENTIFIC_SCHEMA_VERSION = "unicom-proxy-muon-f0-v1"
FAILURE_SCHEMA_VERSION = "unicom-proxy-muon-f0-failure-v1"
RUN_CONFIG_SCHEMA_VERSION = "unicom-proxy-muon-f0-run-v1"
PARENT_MODULE_NAME = "_sfora_proxy_muon_parent_features"
PHASE2_VARIANTS = (
    "adamw_selected",
    "adamw_anchor",
    "proxy_muon",
    "proxy_muon_fp32",
)
TOP_KEYS = (
    "schema_version",
    "status",
    "authority",
    "runtime",
    "protocol",
    "initializer",
    "phase1",
    "selected_learning_rates",
    "phase2",
    "comparisons",
    "predicates",
    "process",
)
FAILURE_KEYS = (
    "schema_version",
    "status",
    "authority",
    "runtime",
    "protocol",
    "completed_cells",
    "completed_cell_sha256s",
    "error",
    "process",
)
AUTHORITY_KEYS = ("source_commit", "handoff_commit", "sources", "inputs")
SOURCE_HASH_KEYS = (
    "runner",
    "decision",
    "probe",
    "training",
    "inshop",
    "parent_spherical",
)
INPUT_HASH_KEYS = (
    "run_config",
    "final_report",
    "spherical_parent_result",
    "cap_closure_receipt",
    "checkpoint",
    "partition",
    "imprinted_head",
    "adamw_fitted_head_seed_0",
    "adamw_fitted_head_seed_1",
    "adamw_fitted_head_seed_2",
)
RUNTIME_KEYS = (
    "python_version",
    "torch_version",
    "numpy_version",
    "sklearn_version",
    "cuda_version",
    "gpu_name",
    "cuda_available",
    "cuda_device_count",
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "deterministic_algorithms",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "muon_signature",
    "observed_update_dtype",
)
PROTOCOL_KEYS = (
    "learning_rate_grid",
    "phase1_seeds",
    "phase2_seeds",
    "phase1_steps",
    "phase2_steps",
    "retained_steps",
    "validation_steps",
    "batch_size",
    "diagnostic_batches",
    "diagnostic_masks",
    "elapsed_limit_seconds",
    "peak_limit_bytes",
)
INITIALIZER_KEYS = (
    "kind",
    "feature_sha256",
    "label_sha256",
    "initial_head_sha256",
    "validation_feature_sha256",
)
PHASE1_ROW_KEYS = (
    "optimizer",
    "learning_rate",
    "fit_seed",
    "steps",
    "initial_head_sha256",
    "final_head_sha256",
    "diagnostic_step0",
    "diagnostic_step64",
)
PANEL_KEYS = ("components", "mean")
SELECTION_KEYS = ("learning_rate", "mean_final_loss", "interior", "tie_lrs")
RETAINED_ROW_KEYS = (
    "step",
    "head_sha256",
    "diagnostic",
    "validation_accuracy",
    "update_dtype",
    "polar_factor_residual",
)
PHASE2_ROW_KEYS = ("fit_seed", "variant", "learning_rate", "retained")
COMPARISON_KEYS = (
    "fit_seed",
    "adamw_reference_variant",
    "adamw_reference_learning_rate",
    "adamw_reference_step512_loss",
    "adamw_reference_step512_accuracy",
    "proxy_muon_reach_step",
    "proxy_muon_accuracy_at_reach",
    "proxy_muon_accuracy_delta",
    "proxy_muon_step512_accuracy_delta",
    "proxy_muon_fp32_reach_step",
    "proxy_muon_fp32_accuracy_at_reach",
    "proxy_muon_fp32_accuracy_delta",
    "proxy_muon_fp32_step512_accuracy_delta",
)
PREDICATE_KEYS = (
    "adamw_lr_interior",
    "proxy_muon_lr_interior",
    "all_reach_by_307",
    "all_reach_noninferior",
    "all_step512_noninferior",
    "any_bf16_accuracy_loss",
    "fp32_sensitivity_supported",
)
PROCESS_KEYS = (
    "command",
    "elapsed_seconds",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "completed_cells",
)
ERROR_KEYS = ("class", "message")
CONFIG_TOP_KEYS = (
    "schema_version",
    "source",
    "handoff",
    "environment",
    "inputs",
    "parent",
    "protocol",
    "result",
)
CONFIG_SOURCE_KEYS = ("commit", "files")
CONFIG_SOURCE_PATHS = (
    "scripts/screen_unicom_proxy_muon_f0.py",
    "src/sfora/unicom_proxy_muon.py",
    "src/sfora/unicom_probe.py",
    "src/sfora/unicom_training.py",
    "src/sfora/unicom_inshop.py",
    "scripts/screen_unicom_spherical_probe.py",
)
EXECUTING_SOURCE_MODULES = (
    ("decision", "sfora.unicom_proxy_muon", CONFIG_SOURCE_PATHS[1]),
    ("probe", "sfora.unicom_probe", CONFIG_SOURCE_PATHS[2]),
    ("training", "sfora.unicom_training", CONFIG_SOURCE_PATHS[3]),
    ("inshop", "sfora.unicom_inshop", CONFIG_SOURCE_PATHS[4]),
)
CONFIG_SOURCE_NAMES = SOURCE_HASH_KEYS
CONFIG_FILE_KEYS = ("path", "sha256")
CONFIG_HANDOFF_KEYS = ("parent_commit", "sole_path", "detached_clean")
CONFIG_INPUT_KEYS = (
    "unicom_checkout",
    "unicom_revision",
    "checkpoint",
    "checkpoint_sha256",
    "dataset_root",
    "partition",
    "partition_sha256",
)
CONFIG_PARENT_KEYS = (
    "final_report",
    "spherical_parent_result",
    "cap_closure_receipt",
    "parent_tensors",
)
CONFIG_PARENT_TENSOR_KEYS = (
    "imprinted_head",
    "adamw_fitted_head_seed_0",
    "adamw_fitted_head_seed_1",
    "adamw_fitted_head_seed_2",
)
CONFIG_RESULT_KEYS = ("relative_path", "failure_relative_path")
FILE_REFERENCE_KEYS = ("path", "sha256")
RECONSTRUCTED_PARENT_KEYS = (
    "fitting_features",
    "fitting_labels",
    "validation_features",
    "validation_labels",
    "imprinted_head",
    "adamw_fitted_head_seed_0",
    "adamw_fitted_head_seed_1",
    "adamw_fitted_head_seed_2",
)


class ExecutionBudget:
    """Fail-fast elapsed/GPU-memory observer shared by every optimizer step."""

    def __init__(
        self,
        *,
        elapsed_limit_seconds: float,
        peak_limit_bytes: int,
        clock: Callable[[], float] = time.monotonic,
        allocated: Callable[[], int] | None = None,
        reserved: Callable[[], int] | None = None,
    ) -> None:
        if (
            type(elapsed_limit_seconds) is not float
            or elapsed_limit_seconds <= 0.0
            or type(peak_limit_bytes) is not int
            or peak_limit_bytes <= 0
            or not callable(clock)
            or (allocated is not None and not callable(allocated))
            or (reserved is not None and not callable(reserved))
        ):
            raise ValueError("ProxyMuon execution budget differs")
        self._elapsed_limit_seconds = elapsed_limit_seconds
        self._peak_limit_bytes = peak_limit_bytes
        self._clock = clock
        self._allocated = allocated or self._cuda_peak_allocated
        self._reserved = reserved or self._cuda_peak_reserved
        self._started_at = float(clock())
        self._elapsed_seconds = 0.0
        self._peak_allocated_bytes = 0
        self._peak_reserved_bytes = 0

    @staticmethod
    def _cuda_peak_allocated() -> int:
        return int(torch.cuda.max_memory_allocated(0)) if torch.cuda.is_available() else 0

    @staticmethod
    def _cuda_peak_reserved() -> int:
        return int(torch.cuda.max_memory_reserved(0)) if torch.cuda.is_available() else 0

    def observe(self, completed_step: int) -> None:
        if type(completed_step) is not int or completed_step <= 0:
            raise ValueError("ProxyMuon completed step differs")
        elapsed = float(self._clock()) - self._started_at
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("ProxyMuon elapsed observation differs")
        self._elapsed_seconds = elapsed
        if elapsed > self._elapsed_limit_seconds:
            raise RuntimeError("ProxyMuon elapsed limit exceeded")
        allocated = self._allocated()
        reserved = self._reserved()
        if (
            type(allocated) is not int
            or allocated < 0
            or type(reserved) is not int
            or reserved < 0
        ):
            raise RuntimeError("ProxyMuon memory observation differs")
        self._peak_allocated_bytes = max(self._peak_allocated_bytes, allocated)
        self._peak_reserved_bytes = max(self._peak_reserved_bytes, reserved)
        if allocated > self._peak_limit_bytes:
            raise RuntimeError("ProxyMuon peak memory limit exceeded")

    def snapshot(self) -> dict[str, object]:
        return {
            "elapsed_seconds": self._elapsed_seconds,
            "peak_allocated_bytes": self._peak_allocated_bytes,
            "peak_reserved_bytes": self._peak_reserved_bytes,
        }


@contextmanager
def preserve_process_state():
    """Restore process-global state touched by parent reconstruction or F0."""

    import numpy as np

    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    thread_count = torch.get_num_threads()
    interop_thread_count = torch.get_num_interop_threads()
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.current_device() if cuda_available else None
    cuda_rng = torch.cuda.get_rng_state_all() if cuda_available else None
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        if torch.get_num_threads() != thread_count:
            torch.set_num_threads(thread_count)
        if torch.get_num_interop_threads() != interop_thread_count:
            torch.set_num_interop_threads(interop_thread_count)
        if cuda_available:
            assert cuda_device is not None and cuda_rng is not None
            torch.cuda.set_device(cuda_device)
            torch.cuda.set_rng_state_all(cuda_rng)


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_object(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("JSON payload must be bytes")
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise TypeError("JSON payload must be an object")
    return value


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args(arguments)


def load_run_config(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise FileNotFoundError("ProxyMuon run config differs")
    config = strict_json_object(path.read_bytes())
    _validate_run_config(config)
    return config


def _relative_path(value: object, name: str) -> str:
    text = _string(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError(f"{name} must be a safe relative path")
    return text


def _file_reference(value: object, name: str) -> dict[str, object]:
    reference = _object(value, FILE_REFERENCE_KEYS, name)
    _string(reference["path"], f"{name}.path")
    _sha256(reference["sha256"], f"{name}.sha256")
    return reference


def _validate_run_config(value: object) -> dict[str, object]:
    config = _object(value, CONFIG_TOP_KEYS, "run config")
    if (
        type(config["schema_version"]) is not str
        or config["schema_version"] != RUN_CONFIG_SCHEMA_VERSION
    ):
        raise ValueError("ProxyMuon run config version differs")
    source = _object(config["source"], CONFIG_SOURCE_KEYS, "run config source")
    _commit(source["commit"], "run config source commit")
    files = _list(source["files"], "run config source files")
    if len(files) != len(CONFIG_SOURCE_PATHS):
        raise ValueError("run config source file count differs")
    for raw, expected_path in zip(files, CONFIG_SOURCE_PATHS, strict=True):
        reference = _object(raw, CONFIG_FILE_KEYS, "run config source file")
        if type(reference["path"]) is not str or reference["path"] != expected_path:
            raise ValueError("run config source file order differs")
        _sha256(reference["sha256"], "run config source file SHA-256")
    handoff = _object(config["handoff"], CONFIG_HANDOFF_KEYS, "run config handoff")
    if (
        _commit(handoff["parent_commit"], "run config parent commit")
        != source["commit"]
        or _relative_path(handoff["sole_path"], "run config sole path")
        != "docs/unicom_proxy_muon_f0_run_config.json"
        or handoff["detached_clean"] is not True
    ):
        raise ValueError("run config handoff differs")
    _validate_runtime(config["environment"])
    inputs = _object(config["inputs"], CONFIG_INPUT_KEYS, "run config inputs")
    for key in ("unicom_checkout", "checkpoint", "dataset_root", "partition"):
        _string(inputs[key], f"run config inputs.{key}")
    _commit(inputs["unicom_revision"], "run config UniCOM revision")
    _sha256(inputs["checkpoint_sha256"], "run config checkpoint SHA-256")
    _sha256(inputs["partition_sha256"], "run config partition SHA-256")
    parent = _object(config["parent"], CONFIG_PARENT_KEYS, "run config parent")
    for key in CONFIG_PARENT_KEYS[:3]:
        _file_reference(parent[key], f"run config parent.{key}")
    tensors = _object(
        parent["parent_tensors"],
        CONFIG_PARENT_TENSOR_KEYS,
        "run config parent tensors",
    )
    for key in CONFIG_PARENT_TENSOR_KEYS:
        _sha256(tensors[key], f"run config parent tensor.{key}")
    _validate_protocol(config["protocol"])
    result = _object(config["result"], CONFIG_RESULT_KEYS, "run config result")
    output = _relative_path(result["relative_path"], "run config result path")
    failure = _relative_path(
        result["failure_relative_path"], "run config failure path"
    )
    if output == failure:
        raise ValueError("run config result paths differ")
    return config


def _git_bytes(repo_root: Path, revision: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{relative_path}"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Git source differs: {relative_path}") from error


def load_spherical_feature_module(
    repo_root: Path, expected_git_revision: str, expected_sha256: str
) -> object:
    if not isinstance(repo_root, Path) or not repo_root.is_dir() or repo_root.is_symlink():
        raise FileNotFoundError("repository root differs")
    _commit(expected_git_revision, "parent feature Git revision")
    _sha256(expected_sha256, "parent feature SHA-256")
    source = repo_root / "scripts" / "screen_unicom_spherical_probe.py"
    if (
        not source.is_file()
        or source.is_symlink()
        or source.resolve().parent != (repo_root / "scripts").resolve()
    ):
        raise FileNotFoundError("parent feature source differs")
    if PARENT_MODULE_NAME in sys.modules:
        raise ValueError("parent feature module imported before source authentication")
    worktree_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    git_digest = hashlib.sha256(
        _git_bytes(
            repo_root,
            expected_git_revision,
            "scripts/screen_unicom_spherical_probe.py",
        )
    ).hexdigest()
    if worktree_digest != expected_sha256 or git_digest != expected_sha256:
        raise ValueError("parent feature SHA-256 differs")
    spec = importlib.util.spec_from_file_location(PARENT_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise ImportError("parent feature loader differs")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[PARENT_MODULE_NAME] = loaded
    try:
        spec.loader.exec_module(loaded)
        if (
            Path(loaded.__file__).resolve() != source.resolve()
            or loaded._load_official_model.__module__ != PARENT_MODULE_NAME
            or loaded._encode_feature_sets.__module__ != PARENT_MODULE_NAME
        ):
            raise ValueError("parent feature callable differs")
        return loaded
    except BaseException:
        sys.modules.pop(PARENT_MODULE_NAME, None)
        raise


def load_training_only_records(
    partition: Path, dataset_root: Path
) -> tuple[InshopRecord, ...]:
    if (
        not isinstance(partition, Path)
        or not isinstance(dataset_root, Path)
        or not partition.is_file()
        or partition.is_symlink()
        or not dataset_root.is_dir()
        or dataset_root.is_symlink()
        or partition.resolve()
        != (dataset_root / "Eval" / "list_eval_partition.txt").resolve()
    ):
        raise FileNotFoundError("In-Shop training partition differs")
    lines = partition.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError("In-Shop partition is truncated")
    try:
        declared = int(lines[0].strip())
    except ValueError as error:
        raise ValueError("In-Shop partition count is invalid") from error
    if lines[1].split() != ["image_name", "item_id", "evaluation_status"]:
        raise ValueError("In-Shop partition header differs")
    records: list[InshopRecord] = []
    for line_number, line in enumerate(lines[2:], start=3):
        fields = line.split()
        if len(fields) != 3 or fields[2] not in ("train", "query", "gallery"):
            raise ValueError(f"In-Shop partition row {line_number} differs")
        image_name, label, split = fields
        if not image_name or not label:
            raise ValueError(f"In-Shop partition row {line_number} differs")
        if split != "train":
            continue
        image_path = dataset_root / "Img" / image_name
        if not image_path.is_file() or image_path.is_symlink():
            raise ValueError(f"In-Shop train image differs: {image_path}")
        records.append(InshopRecord(split="train", image_path=image_path, label=label))
    if len(lines) - 2 != declared:
        raise ValueError("In-Shop partition declared count differs")
    if not records:
        raise ValueError("In-Shop training inventory is empty")
    return tuple(records)


def authenticate_loaded_source_origins(repo_root: Path) -> None:
    if not isinstance(repo_root, Path):
        raise TypeError("ProxyMuon source-origin root differs")
    origins = (("runner", __file__, CONFIG_SOURCE_PATHS[0]),)
    for label, module_name, relative_path in EXECUTING_SOURCE_MODULES:
        loaded = sys.modules.get(module_name)
        loaded_file = getattr(loaded, "__file__", None)
        origins += ((label, loaded_file, relative_path),)
    for label, loaded_file, relative_path in origins:
        if type(loaded_file) is not str or (
            Path(loaded_file).resolve() != (repo_root / relative_path).resolve()
        ):
            raise ValueError(f"ProxyMuon loaded source origin differs: {label}")


def authenticate_source_and_inputs(
    config: Mapping[str, object], repo_root: Path
) -> dict[str, object]:
    if type(config) is not dict or not isinstance(repo_root, Path):
        raise TypeError("ProxyMuon authority arguments differ")
    validated = _validate_run_config(config)
    if (
        not repo_root.is_dir()
        or repo_root.is_symlink()
        or Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=repo_root, text=True
            ).strip()
        ).resolve()
        != repo_root.resolve()
    ):
        raise ValueError("ProxyMuon repository root differs")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    source_commit = validated["source"]["commit"]
    parent = subprocess.check_output(
        ["git", "rev-parse", "HEAD^"], cwd=repo_root, text=True
    ).strip()
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if head == source_commit or parent != source_commit or symbolic.returncode == 0:
        raise ValueError("ProxyMuon detached handoff differs")
    sole_path = validated["handoff"]["sole_path"]
    changed = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    if changed != [sole_path]:
        raise ValueError("ProxyMuon handoff scope differs")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ):
        raise ValueError("ProxyMuon checkout is dirty")
    config_path = repo_root / sole_path
    if (
        not config_path.is_file()
        or config_path.is_symlink()
        or not _same_concrete(strict_json_object(config_path.read_bytes()), validated)
    ):
        raise ValueError("ProxyMuon committed config differs")

    source_hashes: dict[str, str] = {}
    source_identities: set[tuple[int, int]] = set()
    for name, expected_path, raw in zip(
        CONFIG_SOURCE_NAMES,
        CONFIG_SOURCE_PATHS,
        validated["source"]["files"],
        strict=True,
    ):
        expected_digest = raw["sha256"]
        worktree_path = repo_root / expected_path
        if not worktree_path.is_file() or worktree_path.is_symlink():
            raise FileNotFoundError(f"ProxyMuon source differs: {expected_path}")
        identity = (worktree_path.stat().st_dev, worktree_path.stat().st_ino)
        if identity in source_identities:
            raise ValueError(f"ProxyMuon source alias differs: {expected_path}")
        source_identities.add(identity)
        if (
            hashlib.sha256(worktree_path.read_bytes()).hexdigest() != expected_digest
            or hashlib.sha256(
                _git_bytes(repo_root, source_commit, expected_path)
            ).hexdigest()
            != expected_digest
        ):
            raise ValueError(f"ProxyMuon source SHA-256 differs: {expected_path}")
        source_hashes[name] = expected_digest
    authenticate_loaded_source_origins(repo_root)

    inputs = validated["inputs"]
    unicom_checkout = Path(inputs["unicom_checkout"])
    dataset_root = Path(inputs["dataset_root"])
    checkpoint = Path(inputs["checkpoint"])
    partition = Path(inputs["partition"])
    for directory, name in (
        (unicom_checkout, "UniCOM checkout"),
        (dataset_root, "dataset root"),
    ):
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError(f"{name} differs")
    unicom_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=unicom_checkout, text=True
    ).strip()
    if (
        unicom_revision != inputs["unicom_revision"]
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=unicom_checkout,
            text=True,
        )
    ):
        raise ValueError("UniCOM revision differs")
    if partition.resolve() != (
        dataset_root / "Eval" / "list_eval_partition.txt"
    ).resolve():
        raise ValueError("In-Shop partition path differs")
    external_identities: set[tuple[int, int]] = set()
    for path, digest, name in (
        (checkpoint, inputs["checkpoint_sha256"], "checkpoint"),
        (partition, inputs["partition_sha256"], "partition"),
    ):
        if (
            not path.is_file()
            or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"ProxyMuon {name} differs")
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in external_identities:
            raise ValueError(f"ProxyMuon {name} alias differs")
        external_identities.add(identity)

    parent_config = validated["parent"]
    input_hashes = {
        "run_config": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "final_report": "",
        "spherical_parent_result": "",
        "cap_closure_receipt": "",
        "checkpoint": inputs["checkpoint_sha256"],
        "partition": inputs["partition_sha256"],
        **parent_config["parent_tensors"],
    }
    for config_name, authority_name in (
        ("final_report", "final_report"),
        ("spherical_parent_result", "spherical_parent_result"),
        ("cap_closure_receipt", "cap_closure_receipt"),
    ):
        reference = parent_config[config_name]
        path = Path(reference["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]
        ):
            raise ValueError(f"ProxyMuon parent {config_name} differs")
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in external_identities:
            raise ValueError(f"ProxyMuon parent {config_name} alias differs")
        external_identities.add(identity)
        input_hashes[authority_name] = reference["sha256"]
    for key in INPUT_HASH_KEYS:
        _sha256(input_hashes[key], f"authenticated input.{key}")
    for relative in validated["result"].values():
        path = repo_root / relative
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
            raise FileExistsError(path)
    authority = {
        "source_commit": source_commit,
        "handoff_commit": head,
        "sources": source_hashes,
        "inputs": input_hashes,
    }
    _validate_authority(authority)
    return authority


def observe_runtime() -> dict[str, object]:
    import numpy as np
    import sklearn
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "sklearn_version": str(sklearn.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else "unavailable",
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_memory_allocated_bytes": (
            torch.cuda.memory_allocated(0) if cuda_available else 0
        ),
        "cuda_memory_reserved_bytes": (
            torch.cuda.memory_reserved(0) if cuda_available else 0
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "muon_signature": str(__import__("inspect").signature(torch.optim.Muon)),
        "observed_update_dtype": "torch.bfloat16",
    }


def validate_observed_runtime(
    expected: object, observed: object
) -> dict[str, object]:
    _validate_runtime(expected)
    runtime = _validate_runtime(observed)
    if not _same_concrete(runtime, expected):
        raise ValueError("observed runtime differs")
    if (
        runtime["cuda_available"] is not True
        or runtime["cuda_device_count"] != 1
        or runtime["cuda_memory_allocated_bytes"] != 0
        or runtime["cuda_memory_reserved_bytes"] != 0
    ):
        raise ValueError("observed CUDA device is not exclusively idle")
    return runtime


def _object(value: object, keys: tuple[str, ...], name: str) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError(f"{name} schema differs")
    return value


def _list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a list")
    return value


def _string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not value and not allow_empty):
        raise TypeError(f"{name} must be a string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be a float")
    number = value
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} differs")
    return number


def _sha256(value: object, name: str) -> str:
    digest = _string(value, name)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def _commit(value: object, name: str) -> str:
    commit = _string(value, name)
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return commit


def _same_concrete(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return tuple(left) == tuple(right) and all(
            _same_concrete(left[key], right[key]) for key in right
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _same_concrete(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _validate_named_hashes(
    value: object, keys: tuple[str, ...], name: str
) -> None:
    mapping = _object(value, keys, name)
    for key, digest in mapping.items():
        _sha256(digest, f"{name}.{key}")


def _validate_authority(value: object) -> None:
    authority = _object(value, AUTHORITY_KEYS, "authority")
    _commit(authority["source_commit"], "authority.source_commit")
    _commit(authority["handoff_commit"], "authority.handoff_commit")
    _validate_named_hashes(authority["sources"], SOURCE_HASH_KEYS, "authority.sources")
    _validate_named_hashes(authority["inputs"], INPUT_HASH_KEYS, "authority.inputs")


def _validate_runtime(value: object) -> dict[str, object]:
    runtime = _object(value, RUNTIME_KEYS, "runtime")
    for key in RUNTIME_KEYS[:6]:
        _string(runtime[key], f"runtime.{key}")
    if type(runtime["cuda_available"]) is not bool:
        raise TypeError("runtime.cuda_available must be a bool")
    _integer(runtime["cuda_device_count"], "runtime.cuda_device_count")
    _integer(
        runtime["cuda_memory_allocated_bytes"],
        "runtime.cuda_memory_allocated_bytes",
    )
    _integer(
        runtime["cuda_memory_reserved_bytes"],
        "runtime.cuda_memory_reserved_bytes",
    )
    for key in (
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
    ):
        if type(runtime[key]) is not bool:
            raise TypeError(f"runtime.{key} must be a bool")
    _string(runtime["muon_signature"], "runtime.muon_signature")
    if runtime["observed_update_dtype"] != "torch.bfloat16":
        raise ValueError("runtime observed update dtype differs")
    return runtime


def _validate_protocol(value: object) -> None:
    protocol = _object(value, PROTOCOL_KEYS, "protocol")
    expected = {
        "learning_rate_grid": list(LR_GRID),
        "phase1_seeds": list(PHASE1_SEEDS),
        "phase2_seeds": list(PHASE2_SEEDS),
        "phase1_steps": 64,
        "phase2_steps": 512,
        "retained_steps": list(RETAINED_STEPS),
        "validation_steps": list(VALIDATION_STEPS),
        "batch_size": 128,
        "diagnostic_batches": 4,
        "diagnostic_masks": 4,
        "elapsed_limit_seconds": 2700.0,
        "peak_limit_bytes": 8 * 1024**3,
    }
    if not _same_concrete(protocol, expected):
        raise ValueError("protocol differs")


def _validate_initializer(value: object) -> dict[str, object]:
    initializer = _object(value, INITIALIZER_KEYS, "initializer")
    if type(initializer["kind"]) is not str or initializer["kind"] != "imprinted":
        raise ValueError("initializer kind differs")
    for key in INITIALIZER_KEYS[1:]:
        _sha256(initializer[key], f"initializer.{key}")
    return initializer


def _validate_panel(value: object, name: str) -> tuple[tuple[float, ...], float]:
    panel = _object(value, PANEL_KEYS, name)
    components = _list(panel["components"], f"{name}.components")
    if len(components) != 16:
        raise ValueError(f"{name} component count differs")
    numbers = tuple(_number(item, f"{name}.components") for item in components)
    mean = _number(panel["mean"], f"{name}.mean")
    if mean != math.fsum(numbers) / 16:
        raise ValueError(f"{name} mean differs")
    return numbers, mean


def _validate_phase1(
    value: object, *, initial_head_sha256: str
) -> list[dict[str, object]]:
    rows = _list(value, "phase1")
    expected_order = [
        (optimizer, learning_rate, seed)
        for optimizer in OPTIMIZERS
        for learning_rate in LR_GRID
        for seed in PHASE1_SEEDS
    ]
    if len(rows) != len(expected_order):
        raise ValueError("phase1 row count differs")
    validated: list[dict[str, object]] = []
    for index, (raw, expected) in enumerate(zip(rows, expected_order, strict=True)):
        row = _object(raw, PHASE1_ROW_KEYS, f"phase1[{index}]")
        if (
            type(row["optimizer"]) is not str
            or (row["optimizer"], row["learning_rate"], row["fit_seed"])
            != expected
            or type(row["learning_rate"]) is not float
            or row["steps"] != 64
            or type(row["steps"]) is not int
        ):
            raise ValueError("phase1 row order differs")
        if (
            _sha256(row["initial_head_sha256"], "phase1 initial head")
            != initial_head_sha256
        ):
            raise ValueError("phase1 initial head differs")
        _sha256(row["final_head_sha256"], "phase1 final head")
        _validate_panel(row["diagnostic_step0"], "phase1 diagnostic step0")
        _validate_panel(row["diagnostic_step64"], "phase1 diagnostic step64")
        validated.append(row)
    for seed in PHASE1_SEEDS:
        seed_rows = [row for row in validated if row["fit_seed"] == seed]
        first = seed_rows[0]["diagnostic_step0"]
        if any(not _same_concrete(row["diagnostic_step0"], first) for row in seed_rows[1:]):
            raise ValueError("phase1 initial diagnostic differs")
    return validated


def _validate_selections(
    value: object, phase1: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    selections = _object(value, OPTIMIZERS, "selected learning rates")
    for optimizer in OPTIMIZERS:
        block = _object(selections[optimizer], SELECTION_KEYS, f"{optimizer} selection")
        decision_rows = [
            {
                "optimizer": row["optimizer"],
                "learning_rate": row["learning_rate"],
                "fit_seed": row["fit_seed"],
                "step_64_diagnostic_mean": _object(
                    row["diagnostic_step64"], PANEL_KEYS, "panel"
                )["mean"],
            }
            for row in phase1
        ]
        selected = select_learning_rate(decision_rows, optimizer=optimizer)
        means = {
            learning_rate: math.fsum(
                float(row["step_64_diagnostic_mean"])
                for row in decision_rows
                if row["optimizer"] == optimizer
                and row["learning_rate"] == learning_rate
            )
            / len(PHASE1_SEEDS)
            for learning_rate in LR_GRID
        }
        expected = {
            "learning_rate": selected.learning_rate,
            "mean_final_loss": selected.mean_step_64_loss,
            "interior": selected.interior,
            "tie_lrs": [
                learning_rate
                for learning_rate in LR_GRID
                if means[learning_rate] == selected.mean_step_64_loss
            ],
        }
        if not _same_concrete(block, expected):
            raise ValueError(f"{optimizer} selection differs")
    return selections


def _expected_phase2_variants(adamw_lr: float) -> tuple[str, ...]:
    if adamw_lr == 0.0001:
        return ("adamw_selected", "proxy_muon", "proxy_muon_fp32")
    return PHASE2_VARIANTS


def _validate_retained(value: object, variant: str, name: str) -> list[dict[str, object]]:
    rows = _list(value, name)
    if len(rows) != len(RETAINED_STEPS):
        raise ValueError(f"{name} count differs")
    result: list[dict[str, object]] = []
    for raw, step in zip(rows, RETAINED_STEPS, strict=True):
        row = _object(raw, RETAINED_ROW_KEYS, name)
        if type(row["step"]) is not int or row["step"] != step:
            raise ValueError(f"{name} order differs")
        _sha256(row["head_sha256"], f"{name} head")
        _validate_panel(row["diagnostic"], f"{name} diagnostic")
        accuracy = row["validation_accuracy"]
        if step in VALIDATION_STEPS:
            number = _number(accuracy, f"{name} validation accuracy")
            if number < 0.0 or number > 1.0:
                raise ValueError(f"{name} validation accuracy differs")
        elif accuracy is not None:
            raise ValueError(f"{name} validation step differs")
        update_dtype = row["update_dtype"]
        residual = row["polar_factor_residual"]
        if variant.startswith("adamw") or step == 0:
            if update_dtype is not None or residual is not None:
                raise ValueError(f"{name} trace differs")
        else:
            expected_dtype = (
                "torch.float32" if variant == "proxy_muon_fp32" else "torch.bfloat16"
            )
            if type(update_dtype) is not str or update_dtype != expected_dtype:
                raise ValueError(f"{name} trace dtype differs")
            _number(residual, f"{name} trace residual", minimum=0.0)
        result.append(row)
    return result


def _validate_phase2(
    value: object,
    selections: Mapping[str, object],
    *,
    initial_head_sha256: str,
) -> list[dict[str, object]]:
    rows = _list(value, "phase2")
    adamw_lr = float(selections["adamw"]["learning_rate"])
    proxy_lr = float(selections["proxy_muon"]["learning_rate"])
    variants = _expected_phase2_variants(adamw_lr)
    expected_order = [(seed, variant) for seed in PHASE2_SEEDS for variant in variants]
    if len(rows) != len(expected_order):
        raise ValueError("phase2 row count differs")
    result: list[dict[str, object]] = []
    for index, (raw, (seed, variant)) in enumerate(
        zip(rows, expected_order, strict=True)
    ):
        row = _object(raw, PHASE2_ROW_KEYS, f"phase2[{index}]")
        expected_lr = (
            proxy_lr
            if variant.startswith("proxy_muon")
            else 0.0001
            if variant == "adamw_anchor"
            else adamw_lr
        )
        if (
            row["fit_seed"] != seed
            or type(row["fit_seed"]) is not int
            or type(row["variant"]) is not str
            or row["variant"] != variant
            or row["learning_rate"] != expected_lr
            or type(row["learning_rate"]) is not float
        ):
            raise ValueError("phase2 row order differs")
        retained = _validate_retained(
            row["retained"], variant, f"phase2[{index}].retained"
        )
        if retained[0]["head_sha256"] != initial_head_sha256:
            raise ValueError("phase2 initial head differs")
        result.append(row)
    for seed in PHASE2_SEEDS:
        seed_rows = [row for row in result if row["fit_seed"] == seed]
        first = _retained_at(seed_rows[0], 0)["diagnostic"]
        if any(
            not _same_concrete(_retained_at(row, 0)["diagnostic"], first)
            for row in seed_rows[1:]
        ):
            raise ValueError("phase2 initial diagnostic differs")
    return result


def _retained_at(row: Mapping[str, object], step: int) -> Mapping[str, object]:
    retained = row["retained"]
    assert type(retained) is list
    return next(item for item in retained if item["step"] == step)


def _validate_comparisons(
    value: object,
    phase2: Sequence[Mapping[str, object]],
    selections: Mapping[str, object],
) -> list[dict[str, object]]:
    comparisons = _list(value, "comparisons")
    if len(comparisons) != len(PHASE2_SEEDS):
        raise ValueError("comparison count differs")
    result: list[dict[str, object]] = []
    for raw, seed in zip(comparisons, PHASE2_SEEDS, strict=True):
        comparison = _object(raw, COMPARISON_KEYS, f"comparison[{seed}]")
        per_seed = {row["variant"]: row for row in phase2 if row["fit_seed"] == seed}
        reference_rows = [
            {
                "variant": variant,
                "learning_rate": float(per_seed[variant]["learning_rate"]),
                "fit_seed": seed,
                "step_512_diagnostic_mean": float(
                    _retained_at(per_seed[variant], 512)["diagnostic"]["mean"]
                ),
                "step_512_accuracy": float(
                    _retained_at(per_seed[variant], 512)["validation_accuracy"]
                ),
            }
            for variant in (
                ("adamw_selected",)
                if "adamw_anchor" not in per_seed
                else ("adamw_selected", "adamw_anchor")
            )
        ]
        reference = select_adamw_reference(
            reference_rows,
            selected_learning_rate=float(selections["adamw"]["learning_rate"]),
            fit_seed=seed,
        )
        proxy = per_seed["proxy_muon"]
        fp32 = per_seed["proxy_muon_fp32"]
        proxy_losses = {
            step: float(_retained_at(proxy, step)["diagnostic"]["mean"])
            for step in VALIDATION_STEPS
        }
        reach = compute_reach_step(proxy_losses, reference.step_512_diagnostic_mean)
        proxy_at_reach = (
            None
            if reach == ">512"
            else float(_retained_at(proxy, int(reach))["validation_accuracy"])
        )
        proxy_512_accuracy = float(_retained_at(proxy, 512)["validation_accuracy"])
        fp32_512_accuracy = float(_retained_at(fp32, 512)["validation_accuracy"])
        fp32_losses = {
            step: float(_retained_at(fp32, step)["diagnostic"]["mean"])
            for step in VALIDATION_STEPS
        }
        fp32_reach = compute_reach_step(fp32_losses, reference.step_512_diagnostic_mean)
        fp32_at_reach = (
            None
            if fp32_reach == ">512"
            else float(_retained_at(fp32, int(fp32_reach))["validation_accuracy"])
        )
        expected = {
            "fit_seed": seed,
            "adamw_reference_variant": (
                "adamw_selected"
                if reference.learning_rate
                == float(selections["adamw"]["learning_rate"])
                else "adamw_anchor"
            ),
            "adamw_reference_learning_rate": reference.learning_rate,
            "adamw_reference_step512_loss": reference.step_512_diagnostic_mean,
            "adamw_reference_step512_accuracy": reference.step_512_accuracy,
            "proxy_muon_reach_step": reach,
            "proxy_muon_accuracy_at_reach": proxy_at_reach,
            "proxy_muon_accuracy_delta": (
                None
                if proxy_at_reach is None
                else proxy_at_reach - reference.step_512_accuracy
            ),
            "proxy_muon_step512_accuracy_delta": proxy_512_accuracy
            - reference.step_512_accuracy,
            "proxy_muon_fp32_reach_step": fp32_reach,
            "proxy_muon_fp32_accuracy_at_reach": fp32_at_reach,
            "proxy_muon_fp32_accuracy_delta": (
                None
                if fp32_at_reach is None
                else fp32_at_reach - reference.step_512_accuracy
            ),
            "proxy_muon_fp32_step512_accuracy_delta": fp32_512_accuracy
            - reference.step_512_accuracy,
        }
        if not _same_concrete(comparison, expected):
            raise ValueError(f"comparison[{seed}] differs")
        result.append(comparison)
    return result


def _expected_predicates(
    selections: Mapping[str, object], comparisons: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    deltas = [comparison["proxy_muon_accuracy_delta"] for comparison in comparisons]
    step512 = [
        float(comparison["proxy_muon_step512_accuracy_delta"])
        for comparison in comparisons
    ]
    fp32 = [
        float(comparison["proxy_muon_fp32_step512_accuracy_delta"])
        for comparison in comparisons
    ]
    fp32_reach_deltas = [
        comparison["proxy_muon_fp32_accuracy_delta"] for comparison in comparisons
    ]
    proxy_route = all(
        comparison["proxy_muon_reach_step"] == 307
        and comparison["proxy_muon_accuracy_delta"] is not None
        and float(comparison["proxy_muon_accuracy_delta"]) >= -0.002
        and float(comparison["proxy_muon_step512_accuracy_delta"]) >= -0.002
        for comparison in comparisons
    )
    fp32_route = all(
        comparison["proxy_muon_fp32_reach_step"] == 307
        and comparison["proxy_muon_fp32_accuracy_delta"] is not None
        and float(comparison["proxy_muon_fp32_accuracy_delta"]) >= -0.002
        and float(comparison["proxy_muon_fp32_step512_accuracy_delta"]) >= -0.002
        for comparison in comparisons
    )
    return {
        "adamw_lr_interior": selections["adamw"]["interior"],
        "proxy_muon_lr_interior": selections["proxy_muon"]["interior"],
        "all_reach_by_307": all(
            comparison["proxy_muon_reach_step"] == 307 for comparison in comparisons
        ),
        "all_reach_noninferior": all(delta is not None and delta >= -0.002 for delta in deltas),
        "all_step512_noninferior": all(delta >= -0.002 for delta in step512),
        "any_bf16_accuracy_loss": any(
            delta is not None and float(delta) < -0.002 for delta in deltas
        )
        or any(delta < -0.002 for delta in step512),
        "fp32_sensitivity_supported": (not proxy_route)
        and fp32_route
        and all(
            delta is not None and float(delta) >= -0.002
            for delta in fp32_reach_deltas
        )
        and all(delta >= -0.002 for delta in fp32),
    }


def _validate_process(value: object, *, expected_cells: int) -> dict[str, object]:
    process = _object(value, PROCESS_KEYS, "process")
    command = _list(process["command"], "process.command")
    if not command or any(type(item) is not str or not item for item in command):
        raise ValueError("process command differs")
    elapsed = _number(process["elapsed_seconds"], "process.elapsed_seconds", minimum=0.0)
    allocated = _integer(process["peak_allocated_bytes"], "process.peak_allocated_bytes")
    _integer(process["peak_reserved_bytes"], "process.peak_reserved_bytes")
    completed = _integer(process["completed_cells"], "process.completed_cells")
    if completed != expected_cells or elapsed > 2700.0 or allocated > 8 * 1024**3:
        raise ValueError("process limits differ")
    return process


def validate_scientific_result(value: object) -> dict[str, object]:
    result = _object(value, TOP_KEYS, "scientific result")
    if (
        type(result["schema_version"]) is not str
        or result["schema_version"] != SCIENTIFIC_SCHEMA_VERSION
        or type(result["status"]) is not str
    ):
        raise ValueError("scientific result version differs")
    _validate_authority(result["authority"])
    _validate_runtime(result["runtime"])
    _validate_protocol(result["protocol"])
    initializer = _validate_initializer(result["initializer"])
    authority = result["authority"]
    if initializer["initial_head_sha256"] != authority["inputs"]["imprinted_head"]:
        raise ValueError("initializer authority differs")
    phase1 = _validate_phase1(
        result["phase1"], initial_head_sha256=initializer["initial_head_sha256"]
    )
    selections = _validate_selections(result["selected_learning_rates"], phase1)
    phase2 = _validate_phase2(
        result["phase2"],
        selections,
        initial_head_sha256=initializer["initial_head_sha256"],
    )
    comparisons = _validate_comparisons(result["comparisons"], phase2, selections)
    predicates = _object(result["predicates"], PREDICATE_KEYS, "predicates")
    expected_predicates = _expected_predicates(selections, comparisons)
    if not _same_concrete(predicates, expected_predicates):
        raise ValueError("predicates differ")
    _validate_process(result["process"], expected_cells=30 + len(phase2))
    references = {
        int(comparison["fit_seed"]): float(
            comparison["adamw_reference_step512_accuracy"]
        )
        for comparison in comparisons
    }
    decision = decide_proxy_muon_f0(
        {
            "structural_valid": True,
            "adamw_selected_lr_interior": predicates["adamw_lr_interior"],
            "proxy_muon_selected_lr_interior": predicates["proxy_muon_lr_interior"],
            "proxy_muon_reach_steps": {
                int(comparison["fit_seed"]): comparison["proxy_muon_reach_step"]
                for comparison in comparisons
            },
            "proxy_muon_noninferior_at_reach": {
                int(comparison["fit_seed"]): (
                    comparison["proxy_muon_accuracy_at_reach"] is not None
                    and accuracy_noninferior(
                        float(comparison["proxy_muon_accuracy_at_reach"]),
                        references[int(comparison["fit_seed"])],
                    )
                )
                for comparison in comparisons
            },
            "proxy_muon_step512_noninferior": {
                int(comparison["fit_seed"]): float(
                    comparison["proxy_muon_step512_accuracy_delta"]
                )
                >= -0.002
                for comparison in comparisons
            },
            "fp32_reach_steps": {
                int(comparison["fit_seed"]): comparison[
                    "proxy_muon_fp32_reach_step"
                ]
                for comparison in comparisons
            },
            "fp32_noninferior_at_reach": {
                int(comparison["fit_seed"]): (
                    comparison["proxy_muon_fp32_accuracy_at_reach"] is not None
                    and accuracy_noninferior(
                        float(comparison["proxy_muon_fp32_accuracy_at_reach"]),
                        references[int(comparison["fit_seed"])],
                    )
                )
                for comparison in comparisons
            },
            "fp32_step512_noninferior": {
                int(comparison["fit_seed"]): float(
                    comparison["proxy_muon_fp32_step512_accuracy_delta"]
                )
                >= -0.002
                for comparison in comparisons
            },
        }
    )
    if result["status"] != decision:
        raise ValueError("scientific status differs")
    return result


def validate_failure_receipt(value: object) -> dict[str, object]:
    receipt = _object(value, FAILURE_KEYS, "failure receipt")
    if (
        type(receipt["schema_version"]) is not str
        or receipt["schema_version"] != FAILURE_SCHEMA_VERSION
        or type(receipt["status"]) is not str
        or receipt["status"] != "STRUCTURAL_FAILURE"
    ):
        raise ValueError("failure receipt identity differs")
    _validate_authority(receipt["authority"])
    if receipt["runtime"] is not None:
        _validate_runtime(receipt["runtime"])
    _validate_protocol(receipt["protocol"])
    completed = _integer(receipt["completed_cells"], "failure completed_cells")
    hashes = _list(receipt["completed_cell_sha256s"], "failure cell hashes")
    if len(hashes) != completed or completed > 42:
        raise ValueError("failure completed cell evidence differs")
    for digest in hashes:
        _sha256(digest, "failure cell hash")
    error = _object(receipt["error"], ERROR_KEYS, "failure error")
    _string(error["class"], "failure error.class")
    _string(error["message"], "failure error.message", allow_empty=True)
    process = _object(receipt["process"], PROCESS_KEYS, "failure process")
    command = _list(process["command"], "failure process.command")
    if not command or any(type(item) is not str or not item for item in command):
        raise ValueError("failure process command differs")
    _number(process["elapsed_seconds"], "failure elapsed", minimum=0.0)
    _integer(process["peak_allocated_bytes"], "failure peak allocated")
    _integer(process["peak_reserved_bytes"], "failure peak reserved")
    if process["completed_cells"] != completed or type(process["completed_cells"]) is not int:
        raise ValueError("failure process completed cells differs")
    return receipt


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable")
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
        raise OSError(error, os.strerror(error), destination)


def _write_temp_exclusive(path: Path, encoded: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            os.fchmod(handle.fileno(), 0o600)
            if handle.write(encoded) != len(encoded):
                raise OSError(errno.EIO, "short publication write")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_persisted(path: Path) -> bytes:
    return path.read_bytes()


def publish_result_exclusive(
    path: Path,
    payload: object,
    validator: Callable[[object], object],
) -> bytes:
    if not isinstance(path, Path) or not callable(validator):
        raise TypeError("publication arguments differ")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=False, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    validator(payload)
    encoded = canonical_json_bytes(payload)
    try:
        _write_temp_exclusive(temporary, encoded)
        _rename_noreplace(temporary, path)
        _fsync_directory(path.parent)
        persisted = _read_persisted(path)
        if persisted != encoded:
            raise ValueError("published result bytes differ")
        reloaded = strict_json_object(persisted)
        validator(reloaded)
        return persisted
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def run_proxy_muon_f0(
    *,
    protocol: object,
    authority: object,
    runtime: object,
    evidence: Mapping[str, object],
    budget: object,
    completed_cell_sha256s: list[str],
    command: object,
    phase1_runner: Callable[..., list[dict[str, object]]] | None = None,
    phase2_runner: Callable[..., list[dict[str, object]]] | None = None,
    selected_features: int = PROBE_SELECTED_FEATURES,
) -> dict[str, object]:
    """Execute both registered phases after authority/runtime authentication."""

    _validate_protocol(protocol)
    _validate_authority(authority)
    _validate_runtime(runtime)
    if (
        type(evidence) is not dict
        or tuple(evidence)
        != (
            "fitting_features",
            "fitting_labels",
            "validation_features",
            "validation_labels",
            "validation_group_represented",
            "imprinted_head",
        )
        or type(completed_cell_sha256s) is not list
        or completed_cell_sha256s
        or not callable(getattr(budget, "observe", None))
        or not callable(getattr(budget, "snapshot", None))
    ):
        raise ValueError("ProxyMuon authenticated execution arguments differ")
    phase1_runner = run_phase1_cells if phase1_runner is None else phase1_runner
    phase2_runner = run_phase2_cells if phase2_runner is None else phase2_runner
    if not callable(phase1_runner) or not callable(phase2_runner):
        raise TypeError("ProxyMuon scientific phase runner differs")
    phase1 = phase1_runner(
        evidence["fitting_features"],
        evidence["fitting_labels"],
        evidence["imprinted_head"],
        completed_cell_sha256s,
        progress_callback=budget.observe,
        selected_features=selected_features,
    )
    selections = {
        optimizer: _selection_payload(phase1, optimizer)
        for optimizer in OPTIMIZERS
    }
    phase2 = phase2_runner(
        evidence["fitting_features"],
        evidence["fitting_labels"],
        evidence["validation_features"],
        evidence["validation_labels"],
        evidence["validation_group_represented"],
        evidence["imprinted_head"],
        adamw_learning_rate=float(selections["adamw"]["learning_rate"]),
        proxy_muon_learning_rate=float(
            selections["proxy_muon"]["learning_rate"]
        ),
        completed_cell_sha256s=completed_cell_sha256s,
        progress_callback=budget.observe,
        selected_features=selected_features,
    )
    process = {
        "command": command,
        **budget.snapshot(),
        "completed_cells": len(completed_cell_sha256s),
    }
    initializer = {
        "kind": "imprinted",
        "feature_sha256": tensor_sha256(evidence["fitting_features"]),
        "label_sha256": tensor_sha256(evidence["fitting_labels"]),
        "initial_head_sha256": tensor_sha256(evidence["imprinted_head"]),
        "validation_feature_sha256": tensor_sha256(
            evidence["validation_features"]
        ),
    }
    return assemble_scientific_result(
        authority=authority,
        runtime=runtime,
        protocol=protocol,
        initializer=initializer,
        phase1=phase1,
        phase2=phase2,
        process=process,
    )


def phase1_cell_schedule() -> tuple[tuple[str, float, int], ...]:
    """Return the frozen Phase-1 optimizer/LR/seed order."""

    return tuple(
        (optimizer, learning_rate, fit_seed)
        for optimizer in OPTIMIZERS
        for learning_rate in LR_GRID
        for fit_seed in PHASE1_SEEDS
    )


def phase2_cell_schedule(
    adamw_learning_rate: float, proxy_muon_learning_rate: float
) -> tuple[tuple[int, str, float], ...]:
    """Return the frozen Phase-2 order with the AdamW anchor deduplicated."""

    if (
        type(adamw_learning_rate) is not float
        or adamw_learning_rate not in LR_GRID
        or type(proxy_muon_learning_rate) is not float
        or proxy_muon_learning_rate not in LR_GRID
    ):
        raise ValueError("ProxyMuon selected learning rate differs")
    variants = _expected_phase2_variants(adamw_learning_rate)
    return tuple(
        (
            fit_seed,
            variant,
            (
                proxy_muon_learning_rate
                if variant.startswith("proxy_muon")
                else 0.0001
                if variant == "adamw_anchor"
                else adamw_learning_rate
            ),
        )
        for fit_seed in PHASE2_SEEDS
        for variant in variants
    )


def run_phase1_cells(
    fitting_features: object,
    fitting_labels: object,
    initial_head: object,
    completed_cell_sha256s: list[str],
    *,
    cell_runner: Callable[..., dict[str, object]] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    selected_features: int = PROBE_SELECTED_FEATURES,
) -> list[dict[str, object]]:
    """Run Phase 1 in frozen order and append only completed row hashes."""

    if type(completed_cell_sha256s) is not list or any(
        type(value) is not str for value in completed_cell_sha256s
    ):
        raise TypeError("ProxyMuon completed prefix differs")
    runner = _run_phase1_cell if cell_runner is None else cell_runner
    if not callable(runner):
        raise TypeError("ProxyMuon Phase-1 cell runner differs")
    rows: list[dict[str, object]] = []
    for optimizer, learning_rate, fit_seed in phase1_cell_schedule():
        runner_kwargs = {
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "fit_seed": fit_seed,
        }
        if cell_runner is None:
            runner_kwargs["progress_callback"] = progress_callback
            runner_kwargs["selected_features"] = selected_features
        row = runner(
            fitting_features,
            fitting_labels,
            initial_head,
            **runner_kwargs,
        )
        if (
            type(row) is not dict
            or tuple(row) != PHASE1_ROW_KEYS
            or (row["optimizer"], row["learning_rate"], row["fit_seed"])
            != (optimizer, learning_rate, fit_seed)
        ):
            raise ValueError("ProxyMuon Phase-1 cell result differs")
        _sha256(row["initial_head_sha256"], "Phase-1 initial head")
        _sha256(row["final_head_sha256"], "Phase-1 final head")
        _validate_panel(row["diagnostic_step0"], "Phase-1 diagnostic step0")
        _validate_panel(row["diagnostic_step64"], "Phase-1 diagnostic step64")
        rows.append(row)
        completed_cell_sha256s.append(
            hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        )
        if progress_callback is not None:
            progress_callback(64)
    return rows


def _run_phase1_cell(
    fitting_features: object,
    fitting_labels: object,
    initial_head: object,
    *,
    optimizer: str,
    learning_rate: float,
    fit_seed: int,
    fit_trajectory: Callable[..., Mapping[int, torch.Tensor]] | None = None,
    panel_runner: Callable[..., Sequence[float]] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    selected_features: int = PROBE_SELECTED_FEATURES,
) -> dict[str, object]:
    fit = _fit_phase1_trajectory if fit_trajectory is None else fit_trajectory
    panel = diagnostic_panel_losses if panel_runner is None else panel_runner
    if not callable(fit) or not callable(panel):
        raise TypeError("ProxyMuon Phase-1 callback differs")
    fit_kwargs = {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "fit_seed": fit_seed,
        "progress_callback": progress_callback,
    }
    if fit_trajectory is None:
        fit_kwargs["selected_features"] = selected_features
    snapshots = fit(
        fitting_features,
        fitting_labels,
        initial_head,
        **fit_kwargs,
    )
    if type(snapshots) is not dict or tuple(snapshots) != (0, 64):
        raise ValueError("ProxyMuon Phase-1 snapshots differ")
    initial = snapshots[0]
    final = snapshots[64]
    if (
        type(initial) is not torch.Tensor
        or type(final) is not torch.Tensor
        or initial.dtype != torch.float32
        or final.dtype != torch.float32
        or initial.shape != final.shape
        or not initial.is_contiguous()
        or not final.is_contiguous()
        or not torch.isfinite(initial).all()
        or not torch.isfinite(final).all()
    ):
        raise ValueError("ProxyMuon Phase-1 snapshot tensor differs")
    live_initial = _head_on_feature_device(initial, fitting_features)
    step0 = tuple(
        float(value)
        for value in panel(
            fitting_features,
            fitting_labels,
            live_initial,
            fit_seed=fit_seed,
            **(
                {"selected_features": selected_features}
                if panel_runner is None
                else {}
            ),
        )
    )
    del live_initial
    live_final = _head_on_feature_device(final, fitting_features)
    step64 = tuple(
        float(value)
        for value in panel(
            fitting_features,
            fitting_labels,
            live_final,
            fit_seed=fit_seed,
            **(
                {"selected_features": selected_features}
                if panel_runner is None
                else {}
            ),
        )
    )
    del live_final
    return {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "fit_seed": fit_seed,
        "steps": 64,
        "initial_head_sha256": tensor_sha256(initial),
        "final_head_sha256": tensor_sha256(final),
        "diagnostic_step0": panel_payload(step0),
        "diagnostic_step64": panel_payload(step64),
    }


def _fit_phase1_trajectory(
    fitting_features: torch.Tensor,
    fitting_labels: torch.Tensor,
    initial_head: torch.Tensor,
    *,
    optimizer: str,
    learning_rate: float,
    fit_seed: int,
    progress_callback: Callable[[int], None] | None = None,
    selected_features: int = PROBE_SELECTED_FEATURES,
) -> dict[int, torch.Tensor]:
    def optimizer_factory(head: torch.nn.Parameter) -> torch.optim.Optimizer:
        return build_head_optimizer(head, optimizer, learning_rate)

    _fit, snapshots, _traces = _fit_probe_trajectory_with_optimizer(
        fitting_features,
        fitting_labels,
        initial_head,
        snapshot_steps=(0, 64),
        optimizer_factory=optimizer_factory,
        trace_factory=None,
        steps=64,
        fit_seed=fit_seed,
        progress_callback=progress_callback,
        selected_features=selected_features,
    )
    return {
        step: snapshot.detach().cpu().contiguous()
        for step, snapshot in snapshots.items()
    }


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash concrete contiguous tensor bytes in C order."""

    if (
        type(value) is not torch.Tensor
        or not value.is_contiguous()
        or not torch.isfinite(value).all()
    ):
        raise ValueError("ProxyMuon tensor bytes differ")
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _head_on_feature_device(head: torch.Tensor, features: object) -> torch.Tensor:
    if type(features) is torch.Tensor and head.device != features.device:
        return head.to(device=features.device)
    return head


def verify_parent_tensor_hashes(
    tensors: Mapping[str, torch.Tensor], expected: Mapping[str, str]
) -> dict[str, str]:
    """Bind reconstructed features, labels, imprint, and three parent fits."""

    if (
        type(tensors) is not dict
        or tuple(tensors) != RECONSTRUCTED_PARENT_KEYS
        or type(expected) is not dict
        or tuple(expected) != RECONSTRUCTED_PARENT_KEYS
    ):
        raise ValueError("ProxyMuon reconstructed parent inventory differs")
    observed: dict[str, str] = {}
    for name in RECONSTRUCTED_PARENT_KEYS:
        digest = tensor_sha256(tensors[name])
        if _sha256(expected[name], f"expected parent tensor {name}") != digest:
            raise ValueError(f"ProxyMuon reconstructed parent tensor differs: {name}")
        observed[name] = digest
    return observed


def reconstruct_parent_evidence(
    config: Mapping[str, object],
    authority: Mapping[str, object],
    repo_root: Path,
) -> dict[str, object]:
    """Reconstruct the authenticated spherical-parent training-only tensors."""

    validated = _validate_run_config(config)
    _validate_authority(authority)
    inputs = validated["inputs"]
    records = load_training_only_records(
        Path(inputs["partition"]), Path(inputs["dataset_root"])
    )
    optimization, _heldout_query, _heldout_gallery, labels = identity_holdout(
        records, fraction=0.2, seed=0
    )
    split = split_probe_records(optimization, labels)
    if (
        len(optimization) != 20_650
        or len(labels) != 3_200
        or len(split.fitting) != 14_330
        or len(split.validation) != 3_188
        or split.validation_class_count != 3_188
        or split.singleton_class_count != 12
        or split.excluded_same_series_count != 3_132
    ):
        raise ValueError("ProxyMuon parent split differs")

    parent_digest = str(authority["sources"]["parent_spherical"])
    parent_module = load_spherical_feature_module(
        repo_root, str(authority["source_commit"]), parent_digest
    )
    parent_result = strict_json_object(
        Path(validated["parent"]["spherical_parent_result"]["path"]).read_bytes()
    )
    try:
        parent_module.validate_result(parent_result)
        arguments = argparse.Namespace(
            unicom_checkout=Path(inputs["unicom_checkout"]),
            checkpoint=Path(inputs["checkpoint"]),
            batch_size=128,
            workers=4,
        )
        fitting_features, validation_features = parent_module._encode_feature_sets(
            arguments, split.fitting, split.validation
        )
    finally:
        sys.modules.pop(PARENT_MODULE_NAME, None)
        for name in tuple(sys.modules):
            if name == "unicom" or name.startswith("unicom."):
                sys.modules.pop(name, None)
    device = fitting_features.device
    fitting_labels = torch.tensor(
        [labels[row.label] for row in split.fitting],
        dtype=torch.int64,
        device=device,
    ).contiguous()
    validation_labels = torch.tensor(
        [labels[row.label] for row in split.validation],
        dtype=torch.int64,
        device=device,
    ).contiguous()
    imprinted_head = class_mean_head(
        fitting_features, fitting_labels, class_count=len(labels)
    ).contiguous()
    parent_fits = {
        seed: fit_spherical_probe(
            fitting_features, fitting_labels, imprinted_head, fit_seed=seed
        ).head.contiguous()
        for seed in PHASE1_SEEDS
    }
    probes = parent_result["probes"]
    if (
        type(probes) is not list
        or len(probes) != 3
        or [probe["fit_seed"] for probe in probes] != list(PHASE1_SEEDS)
    ):
        raise ValueError("ProxyMuon spherical parent probes differ")
    expected = {
        "fitting_features": tensor_sha256(fitting_features),
        "fitting_labels": tensor_sha256(fitting_labels),
        "validation_features": tensor_sha256(validation_features),
        "validation_labels": tensor_sha256(validation_labels),
        "imprinted_head": str(authority["inputs"]["imprinted_head"]),
        "adamw_fitted_head_seed_0": str(
            authority["inputs"]["adamw_fitted_head_seed_0"]
        ),
        "adamw_fitted_head_seed_1": str(
            authority["inputs"]["adamw_fitted_head_seed_1"]
        ),
        "adamw_fitted_head_seed_2": str(
            authority["inputs"]["adamw_fitted_head_seed_2"]
        ),
    }
    tensors = {
        "fitting_features": fitting_features,
        "fitting_labels": fitting_labels,
        "validation_features": validation_features,
        "validation_labels": validation_labels,
        "imprinted_head": imprinted_head,
        "adamw_fitted_head_seed_0": parent_fits[0],
        "adamw_fitted_head_seed_1": parent_fits[1],
        "adamw_fitted_head_seed_2": parent_fits[2],
    }
    verify_parent_tensor_hashes(tensors, expected)
    if (
        parent_result["class_mean"]["sha256"] != expected["imprinted_head"]
        or [probe["sha256"] for probe in probes]
        != [expected[f"adamw_fitted_head_seed_{seed}"] for seed in PHASE1_SEEDS]
    ):
        raise ValueError("ProxyMuon spherical parent imprint differs")
    del parent_fits
    return {
        "fitting_features": fitting_features,
        "fitting_labels": fitting_labels,
        "validation_features": validation_features,
        "validation_labels": validation_labels,
        "validation_group_represented": split.validation_group_represented,
        "imprinted_head": imprinted_head,
    }


def panel_payload(components: Sequence[float]) -> dict[str, object]:
    """Reduce one registered 16-cell panel without changing its order."""

    if type(components) not in (tuple, list) or len(components) != 16:
        raise ValueError("ProxyMuon diagnostic panel differs")
    values = [float(value) for value in components]
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("ProxyMuon diagnostic panel value differs")
    return {"components": values, "mean": math.fsum(values) / 16}


def run_phase2_cells(
    fitting_features: object,
    fitting_labels: object,
    validation_features: object,
    validation_labels: object,
    validation_group_represented: object,
    initial_head: object,
    adamw_learning_rate: float,
    proxy_muon_learning_rate: float,
    completed_cell_sha256s: list[str],
    *,
    cell_runner: Callable[..., dict[str, object]] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    selected_features: int = PROBE_SELECTED_FEATURES,
) -> list[dict[str, object]]:
    """Run Phase 2 in seed-major order and retain an exact completed prefix."""

    if type(completed_cell_sha256s) is not list or any(
        type(value) is not str for value in completed_cell_sha256s
    ):
        raise TypeError("ProxyMuon completed prefix differs")
    runner = _run_phase2_cell if cell_runner is None else cell_runner
    if not callable(runner):
        raise TypeError("ProxyMuon Phase-2 cell runner differs")
    rows: list[dict[str, object]] = []
    for fit_seed, variant, learning_rate in phase2_cell_schedule(
        adamw_learning_rate, proxy_muon_learning_rate
    ):
        runner_kwargs = {
            "fit_seed": fit_seed,
            "variant": variant,
            "learning_rate": learning_rate,
        }
        if cell_runner is None:
            runner_kwargs["progress_callback"] = progress_callback
            runner_kwargs["selected_features"] = selected_features
        row = runner(
            fitting_features,
            fitting_labels,
            validation_features,
            validation_labels,
            validation_group_represented,
            initial_head,
            **runner_kwargs,
        )
        if (
            type(row) is not dict
            or tuple(row) != PHASE2_ROW_KEYS
            or (row["fit_seed"], row["variant"], row["learning_rate"])
            != (fit_seed, variant, learning_rate)
        ):
            raise ValueError("ProxyMuon Phase-2 cell result differs")
        _validate_retained(row["retained"], variant, "Phase-2 retained")
        rows.append(row)
        completed_cell_sha256s.append(
            hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        )
        if progress_callback is not None:
            progress_callback(512)
    return rows


def _run_phase2_cell(
    fitting_features: object,
    fitting_labels: object,
    validation_features: object,
    validation_labels: object,
    validation_group_represented: object,
    initial_head: object,
    *,
    fit_seed: int,
    variant: str,
    learning_rate: float,
    fit_trajectory: Callable[..., tuple[Mapping[int, torch.Tensor], Mapping[int, object | None]]]
    | None = None,
    panel_runner: Callable[..., Sequence[float]] | None = None,
    accuracy_runner: Callable[..., float] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    selected_features: int = PROBE_SELECTED_FEATURES,
) -> dict[str, object]:
    fit = _fit_phase2_trajectory if fit_trajectory is None else fit_trajectory
    panel = diagnostic_panel_losses if panel_runner is None else panel_runner
    accuracy = _evaluate_accuracy if accuracy_runner is None else accuracy_runner
    if not callable(fit) or not callable(panel) or not callable(accuracy):
        raise TypeError("ProxyMuon Phase-2 callback differs")
    fit_kwargs = {
        "fit_seed": fit_seed,
        "variant": variant,
        "learning_rate": learning_rate,
        "progress_callback": progress_callback,
    }
    if fit_trajectory is None:
        fit_kwargs["selected_features"] = selected_features
    snapshots, traces = fit(
        fitting_features,
        fitting_labels,
        initial_head,
        **fit_kwargs,
    )
    if (
        type(snapshots) is not dict
        or tuple(snapshots) != RETAINED_STEPS
        or type(traces) is not dict
        or tuple(traces) != RETAINED_STEPS
    ):
        raise ValueError("ProxyMuon Phase-2 trajectory differs")
    retained: list[dict[str, object]] = []
    for step in RETAINED_STEPS:
        head = snapshots[step]
        if (
            type(head) is not torch.Tensor
            or head.dtype != torch.float32
            or not head.is_contiguous()
            or not torch.isfinite(head).all()
        ):
            raise ValueError("ProxyMuon Phase-2 snapshot differs")
        live_head = _head_on_feature_device(head, fitting_features)
        components = tuple(
            float(value)
            for value in panel(
                fitting_features,
                fitting_labels,
                live_head,
                fit_seed=fit_seed,
                **(
                    {"selected_features": selected_features}
                    if panel_runner is None
                    else {}
                ),
            )
        )
        validation_accuracy = (
            float(
                accuracy(
                    validation_features,
                    validation_labels,
                    _head_on_feature_device(head, validation_features),
                    represented=validation_group_represented,
                    **(
                        {"selected_features": selected_features}
                        if accuracy_runner is None
                        else {}
                    ),
                )
            )
            if step in VALIDATION_STEPS
            else None
        )
        if validation_accuracy is not None and not 0.0 <= validation_accuracy <= 1.0:
            raise ValueError("ProxyMuon validation accuracy differs")
        trace = traces[step]
        if variant.startswith("adamw") or step == 0:
            if trace is not None:
                raise ValueError("ProxyMuon Phase-2 trace differs")
            update_dtype = None
            residual = None
        else:
            expected_dtype = (
                "torch.float32" if variant == "proxy_muon_fp32" else "torch.bfloat16"
            )
            if (
                trace is None
                or getattr(trace, "update_dtype", None) != expected_dtype
                or type(getattr(trace, "polar_factor_residual", None)) is not float
                or not math.isfinite(trace.polar_factor_residual)
                or trace.polar_factor_residual < 0.0
            ):
                raise ValueError("ProxyMuon Phase-2 trace differs")
            update_dtype = trace.update_dtype
            residual = trace.polar_factor_residual
        retained.append(
            {
                "step": step,
                "head_sha256": tensor_sha256(head),
                "diagnostic": panel_payload(components),
                "validation_accuracy": validation_accuracy,
                "update_dtype": update_dtype,
                "polar_factor_residual": residual,
            }
        )
        del live_head
    return {
        "fit_seed": fit_seed,
        "variant": variant,
        "learning_rate": learning_rate,
        "retained": retained,
    }


def _fit_phase2_trajectory(
    fitting_features: torch.Tensor,
    fitting_labels: torch.Tensor,
    initial_head: torch.Tensor,
    *,
    fit_seed: int,
    variant: str,
    learning_rate: float,
    progress_callback: Callable[[int], None] | None = None,
    selected_features: int = PROBE_SELECTED_FEATURES,
) -> tuple[dict[int, torch.Tensor], dict[int, object | None]]:
    if variant.startswith("adamw"):
        def optimizer_factory(head: torch.nn.Parameter) -> torch.optim.Optimizer:
            return build_head_optimizer(head, "adamw", learning_rate)

        _fit, snapshots, traces = _fit_probe_trajectory_with_optimizer(
            fitting_features,
            fitting_labels,
            initial_head,
            snapshot_steps=RETAINED_STEPS,
            optimizer_factory=optimizer_factory,
            trace_factory=None,
            steps=512,
            fit_seed=fit_seed,
            progress_callback=progress_callback,
            selected_features=selected_features,
        )
    elif variant in ("proxy_muon", "proxy_muon_fp32"):
        _fit, snapshots, traces = fit_proxy_muon_trajectory(
            fitting_features,
            fitting_labels,
            initial_head,
            learning_rate=learning_rate,
            ns_dtype=(
                torch.float32 if variant == "proxy_muon_fp32" else torch.bfloat16
            ),
            snapshot_steps=RETAINED_STEPS,
            steps=512,
            fit_seed=fit_seed,
            progress_callback=progress_callback,
            selected_features=selected_features,
        )
    else:
        raise ValueError("ProxyMuon Phase-2 variant differs")
    return (
        {
            step: snapshot.detach().cpu().contiguous()
            for step, snapshot in snapshots.items()
        },
        dict(traces),
    )


def _evaluate_accuracy(
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    head: torch.Tensor,
    *,
    represented: tuple[bool, ...],
    selected_features: int = PROBE_SELECTED_FEATURES,
) -> float:
    device_head = head.to(device=validation_features.device)
    try:
        return float(
            evaluate_probe_head(
                validation_features,
                validation_labels,
                device_head,
                validation_group_represented=represented,
                selected_features=selected_features,
            ).accuracy
        )
    finally:
        del device_head


def _selection_payload(
    phase1: list[dict[str, object]], optimizer: str
) -> dict[str, object]:
    decision_rows = [
        {
            "optimizer": row["optimizer"],
            "learning_rate": row["learning_rate"],
            "fit_seed": row["fit_seed"],
            "step_64_diagnostic_mean": row["diagnostic_step64"]["mean"],
        }
        for row in phase1
    ]
    selected = select_learning_rate(decision_rows, optimizer=optimizer)
    means = {
        learning_rate: math.fsum(
            float(row["step_64_diagnostic_mean"])
            for row in decision_rows
            if row["optimizer"] == optimizer
            and row["learning_rate"] == learning_rate
        )
        / len(PHASE1_SEEDS)
        for learning_rate in LR_GRID
    }
    return {
        "learning_rate": selected.learning_rate,
        "mean_final_loss": selected.mean_step_64_loss,
        "interior": selected.interior,
        "tie_lrs": [
            learning_rate
            for learning_rate in LR_GRID
            if means[learning_rate] == selected.mean_step_64_loss
        ],
    }


def _comparison_payloads(
    phase2: list[dict[str, object]], selections: Mapping[str, object]
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for seed in PHASE2_SEEDS:
        per_seed = {
            str(row["variant"]): row
            for row in phase2
            if row["fit_seed"] == seed
        }
        reference_variants = (
            ("adamw_selected",)
            if "adamw_anchor" not in per_seed
            else ("adamw_selected", "adamw_anchor")
        )
        reference_rows = [
            {
                "variant": variant,
                "learning_rate": float(per_seed[variant]["learning_rate"]),
                "fit_seed": seed,
                "step_512_diagnostic_mean": float(
                    _retained_at(per_seed[variant], 512)["diagnostic"]["mean"]
                ),
                "step_512_accuracy": float(
                    _retained_at(per_seed[variant], 512)["validation_accuracy"]
                ),
            }
            for variant in reference_variants
        ]
        reference = select_adamw_reference(
            reference_rows,
            selected_learning_rate=float(selections["adamw"]["learning_rate"]),
            fit_seed=seed,
        )

        def summarize(
            variant: str,
            per_seed: Mapping[str, object] = per_seed,
            reference: object = reference,
        ) -> tuple[int | str, float | None, float]:
            row = per_seed[variant]
            losses = {
                step: float(_retained_at(row, step)["diagnostic"]["mean"])
                for step in VALIDATION_STEPS
            }
            reach = compute_reach_step(losses, reference.step_512_diagnostic_mean)
            accuracy_at_reach = (
                None
                if reach == ">512"
                else float(_retained_at(row, int(reach))["validation_accuracy"])
            )
            accuracy_512 = float(_retained_at(row, 512)["validation_accuracy"])
            return reach, accuracy_at_reach, accuracy_512

        proxy_reach, proxy_at_reach, proxy_512 = summarize("proxy_muon")
        fp32_reach, fp32_at_reach, fp32_512 = summarize("proxy_muon_fp32")
        selected_adamw_lr = float(selections["adamw"]["learning_rate"])
        comparisons.append(
            {
                "fit_seed": seed,
                "adamw_reference_variant": (
                    "adamw_selected"
                    if reference.learning_rate == selected_adamw_lr
                    else "adamw_anchor"
                ),
                "adamw_reference_learning_rate": reference.learning_rate,
                "adamw_reference_step512_loss": reference.step_512_diagnostic_mean,
                "adamw_reference_step512_accuracy": reference.step_512_accuracy,
                "proxy_muon_reach_step": proxy_reach,
                "proxy_muon_accuracy_at_reach": proxy_at_reach,
                "proxy_muon_accuracy_delta": (
                    None
                    if proxy_at_reach is None
                    else proxy_at_reach - reference.step_512_accuracy
                ),
                "proxy_muon_step512_accuracy_delta": proxy_512
                - reference.step_512_accuracy,
                "proxy_muon_fp32_reach_step": fp32_reach,
                "proxy_muon_fp32_accuracy_at_reach": fp32_at_reach,
                "proxy_muon_fp32_accuracy_delta": (
                    None
                    if fp32_at_reach is None
                    else fp32_at_reach - reference.step_512_accuracy
                ),
                "proxy_muon_fp32_step512_accuracy_delta": fp32_512
                - reference.step_512_accuracy,
            }
        )
    return comparisons


def _decision_from_result_parts(
    selections: Mapping[str, object], comparisons: Sequence[Mapping[str, object]]
) -> str:
    references = {
        int(comparison["fit_seed"]): float(
            comparison["adamw_reference_step512_accuracy"]
        )
        for comparison in comparisons
    }
    return decide_proxy_muon_f0(
        {
            "structural_valid": True,
            "adamw_selected_lr_interior": selections["adamw"]["interior"],
            "proxy_muon_selected_lr_interior": selections["proxy_muon"]["interior"],
            "proxy_muon_reach_steps": {
                int(row["fit_seed"]): row["proxy_muon_reach_step"]
                for row in comparisons
            },
            "proxy_muon_noninferior_at_reach": {
                int(row["fit_seed"]): (
                    row["proxy_muon_accuracy_at_reach"] is not None
                    and accuracy_noninferior(
                        float(row["proxy_muon_accuracy_at_reach"]),
                        references[int(row["fit_seed"])],
                    )
                )
                for row in comparisons
            },
            "proxy_muon_step512_noninferior": {
                int(row["fit_seed"]): float(
                    row["proxy_muon_step512_accuracy_delta"]
                )
                >= -0.002
                for row in comparisons
            },
            "fp32_reach_steps": {
                int(row["fit_seed"]): row["proxy_muon_fp32_reach_step"]
                for row in comparisons
            },
            "fp32_noninferior_at_reach": {
                int(row["fit_seed"]): (
                    row["proxy_muon_fp32_accuracy_at_reach"] is not None
                    and accuracy_noninferior(
                        float(row["proxy_muon_fp32_accuracy_at_reach"]),
                        references[int(row["fit_seed"])],
                    )
                )
                for row in comparisons
            },
            "fp32_step512_noninferior": {
                int(row["fit_seed"]): float(
                    row["proxy_muon_fp32_step512_accuracy_delta"]
                )
                >= -0.002
                for row in comparisons
            },
        }
    )


def assemble_scientific_result(
    *,
    authority: object,
    runtime: object,
    protocol: object,
    initializer: object,
    phase1: object,
    phase2: object,
    process: object,
) -> dict[str, object]:
    """Recompute and validate a complete scientific result."""

    if type(phase1) is not list or type(phase2) is not list:
        raise TypeError("ProxyMuon scientific rows differ")
    selections = {
        optimizer: _selection_payload(phase1, optimizer)
        for optimizer in OPTIMIZERS
    }
    comparisons = _comparison_payloads(phase2, selections)
    predicates = _expected_predicates(selections, comparisons)
    result = {
        "schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "status": _decision_from_result_parts(selections, comparisons),
        "authority": authority,
        "runtime": runtime,
        "protocol": protocol,
        "initializer": initializer,
        "phase1": phase1,
        "selected_learning_rates": selections,
        "phase2": phase2,
        "comparisons": comparisons,
        "predicates": predicates,
        "process": process,
    }
    validate_scientific_result(result)
    return result


def assemble_failure_receipt(
    *,
    authority: object,
    runtime: object,
    protocol: object,
    completed_cell_sha256s: object,
    error: BaseException,
    process: object,
) -> dict[str, object]:
    """Assemble and validate the reduced structural-failure receipt."""

    if type(completed_cell_sha256s) is not list or not isinstance(error, BaseException):
        raise TypeError("ProxyMuon failure evidence differs")
    receipt = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "status": "STRUCTURAL_FAILURE",
        "authority": authority,
        "runtime": runtime,
        "protocol": protocol,
        "completed_cells": len(completed_cell_sha256s),
        "completed_cell_sha256s": completed_cell_sha256s,
        "error": {"class": type(error).__name__, "message": str(error)},
        "process": process,
    }
    validate_failure_receipt(receipt)
    return receipt


def main(arguments: Sequence[str] | None = None) -> int:
    """Authenticate, execute, and publish exactly one ProxyMuon screen."""

    parsed = parse_args(arguments)
    if not sys.flags.isolated or not sys.flags.dont_write_bytecode:
        return 2
    repo_root = Path.cwd().resolve()
    try:
        config = load_run_config(parsed.config)
        authority = authenticate_source_and_inputs(config, repo_root)
    except Exception:
        return 2

    command = [
        ".venv/bin/python",
        "-I",
        "-B",
        "scripts/screen_unicom_proxy_muon_f0.py",
        "--config",
        str(config["handoff"]["sole_path"]),
    ]
    completed: list[str] = []
    runtime: dict[str, object] | None = None
    budget: ExecutionBudget | None = None
    try:
        with preserve_process_state():
            runtime = validate_observed_runtime(
                config["environment"], observe_runtime()
            )
            evidence = reconstruct_parent_evidence(config, authority, repo_root)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(0)
            budget = ExecutionBudget(
                elapsed_limit_seconds=float(
                    config["protocol"]["elapsed_limit_seconds"]
                ),
                peak_limit_bytes=int(config["protocol"]["peak_limit_bytes"]),
            )
            result = run_proxy_muon_f0(
                protocol=config["protocol"],
                authority=authority,
                runtime=runtime,
                evidence=evidence,
                budget=budget,
                completed_cell_sha256s=completed,
                command=command,
            )
    except Exception as error:
        observed = (
            budget.snapshot()
            if budget is not None
            else {
                "elapsed_seconds": 0.0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            }
        )
        process = {
            "command": command,
            **observed,
            "completed_cells": len(completed),
        }
        try:
            receipt = assemble_failure_receipt(
                authority=authority,
                runtime=runtime,
                protocol=config["protocol"],
                completed_cell_sha256s=completed,
                error=error,
                process=process,
            )
            failure = repo_root / str(config["result"]["failure_relative_path"])
            publish_result_exclusive(failure, receipt, validate_failure_receipt)
        except Exception:
            pass
        return 2
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = repo_root / str(config["result"]["relative_path"])
    try:
        publish_result_exclusive(output, result, validate_scientific_result)
    except Exception:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
