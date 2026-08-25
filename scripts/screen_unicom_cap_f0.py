#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NoReturn

import numpy as np

_TOP_KEYS = (
    "schema_version",
    "spec",
    "parent",
    "environment",
    "inputs",
    "protocol",
    "source",
    "handoff",
    "result",
)
_SPEC_KEYS = ("path", "sha256", "commit")
_PARENT_KEYS = ("path", "sha256", "source_commit")
_ENVIRONMENT_KEYS = (
    "python",
    "torch",
    "numpy",
    "sklearn",
    "cuda",
    "device",
    "model_dtype",
    "reduction_dtype",
)
_INPUT_KEYS = (
    "unicom_checkout",
    "unicom_revision",
    "checkpoint",
    "checkpoint_sha256",
    "dataset_root",
    "partition",
    "partition_sha256",
)
_PROTOCOL_KEYS = (
    "holdout_fraction",
    "holdout_seed",
    "split_seed",
    "fit_seeds",
    "fit_steps",
    "snapshot_steps",
    "batch_size",
    "batch_seed",
    "mask_seed",
    "evaluation_mask_seed",
    "diagnostic_seed",
    "gradient_seed",
    "covariance_mask_seed",
    "evaluation_mask_sets",
    "covariance_mask_sets",
    "shards",
    "selected_features",
    "feature_count",
    "margin",
    "accuracy_margin",
    "scale",
    "optimizer",
    "row_norm",
    "paired_t_critical_df63",
    "paired_t_critical_df3187",
    "loss_delta_minimum",
    "accuracy_delta_minimum",
    "non_worse_mask_minimum",
    "head_cosine_mean_minimum",
    "step_equivalence_minimum",
)
_SOURCE_KEYS = ("commit", "files")
_SOURCE_ROW_KEYS = ("path", "sha256")
_HANDOFF_KEYS = ("parent_commit", "sole_path", "detached_clean")
_RESULT_KEYS = ("relative_path", "schema_version")
_SOURCE_PATHS = (
    "pyproject.toml",
    "scripts/screen_unicom_cap_f0.py",
    "src/sfora/unicom_cap.py",
    "src/sfora/unicom_probe.py",
    "src/sfora/unicom_training.py",
    "src/sfora/unicom_inshop.py",
    "tests/test_screen_unicom_cap_f0.py",
    "tests/test_unicom_cap.py",
    "tests/test_unicom_probe.py",
    "tests/test_unicom_cap_f0_run_config.py",
)
_FROZEN_SPEC = {
    "path": "docs/superpowers/specs/2026-08-25-unicom-cap-f0-design.md",
    "sha256": "cf6994c9bda0677a714cd0a12dcca459af0fe610d28a9869091992724a4e880a",
    "commit": "cfd2ebf18b4d3a2c80c3b96957d777e23224a4cc",
}
_FROZEN_PARENT = {
    "path": "reports/generated/unicom-spherical-probe-ed2e789.json",
    "sha256": "d1a52703849acb96f359c2c7f209942fcbf6fa770eeaa0ed41d947780d714ddf",
    "source_commit": "ed2e7893b05d3b5105ff992691efccc5b13ad5a0",
}
_FROZEN_ENVIRONMENT = {
    "python": "3.13.9",
    "torch": "2.12.1+cu130",
    "numpy": "2.5.0",
    "sklearn": "1.9.0",
    "cuda": "13.0",
    "device": "NVIDIA GB10",
    "model_dtype": "float32",
    "reduction_dtype": "float64",
}
_FROZEN_INPUTS = {
    "unicom_checkout": "/home/riomus/UniCOM",
    "unicom_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
    "checkpoint": "/home/riomus/checkpoints/FP16-ViT-L-14-336px.pt",
    "checkpoint_sha256": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
    "dataset_root": "/home/riomus/datasets/In-shop Clothes Retrieval Benchmark",
    "partition": (
        "/home/riomus/datasets/In-shop Clothes Retrieval Benchmark/"
        "Eval/list_eval_partition.txt"
    ),
    "partition_sha256": "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c",
}
_FROZEN_PROTOCOL = {
    "holdout_fraction": 0.2,
    "holdout_seed": 0,
    "split_seed": 23_000,
    "fit_seeds": [0, 1, 2],
    "fit_steps": 512,
    "snapshot_steps": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    "batch_size": 128,
    "batch_seed": 23_001,
    "mask_seed": 23_002,
    "evaluation_mask_seed": 23_003,
    "diagnostic_seed": 23_004,
    "gradient_seed": 23_005,
    "covariance_mask_seed": 23_006,
    "evaluation_mask_sets": 64,
    "covariance_mask_sets": 8,
    "shards": 8,
    "selected_features": 512,
    "feature_count": 768,
    "margin": 0.25,
    "accuracy_margin": 0.0,
    "scale": 32.0,
    "optimizer": "AdamW(lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0)",
    "row_norm": 0.27712812921102037,
    "paired_t_critical_df63": 1.998340542520741,
    "paired_t_critical_df3187": 1.9607086212236648,
    "loss_delta_minimum": 0.0501203852609845,
    "accuracy_delta_minimum": 0.006380126646800488,
    "non_worse_mask_minimum": 60,
    "head_cosine_mean_minimum": 0.95,
    "step_equivalence_minimum": 64,
}
_RESULT_TOP_KEYS = (
    "schema_version",
    "authority",
    "runtime",
    "dataset",
    "protocol",
    "covariance",
    "class_mean",
    "cap_metrics",
    "seeds",
    "decision",
    "candidate_values_computed",
)
_RUNTIME_RESULT_KEYS = (*_ENVIRONMENT_KEYS, "elapsed_seconds", "peak_gpu_mib")
_COVARIANCE_KEYS = (
    "sample_count",
    "feature_count",
    "shrinkage",
    "matrix_fp64_le_base64",
    "trace",
    "cholesky_diagonal_min",
    "cholesky_diagonal_max",
    "sha256",
    "condition_number",
    "effective_rank",
    "construction_mask_sha256",
    "mismatch",
)
_SUMMARY_KEYS = ("minimum", "p05", "median", "mean")
_MISMATCH_KEYS = ("row_cosines", "summary")
_METRIC_KEYS = (
    "mean_loss",
    "accuracy",
    "correct_count",
    "observation_count",
    "per_mask_mean_losses",
    "per_mask_represented_mean_losses",
    "per_mask_unrepresented_mean_losses",
    "per_image_mean_losses",
    "represented_mean_loss",
    "unrepresented_mean_loss",
)
_CLASS_MEAN_KEYS = (
    "sha256",
    "row_norms",
    "row_norm_min",
    "row_norm_max",
    "validation",
)
_CAP_METRIC_KEYS = ("validation", "statistics", "predicates")
_STATISTIC_KEYS = (
    "loss_delta",
    "accuracy_delta",
    "non_worse_mask_count",
    "unrepresented_loss_delta",
    "mask_paired_mean_delta",
    "mask_paired_95_lower_bound",
    "identity_paired_mean_delta",
    "identity_paired_95_lower_bound",
)
_STATIC_PREDICATE_KEYS = (
    "loss_delta_at_least_0_0501203852609845",
    "accuracy_delta_at_least_0_006380126646800488",
    "mask_and_stratum_consistent",
    "paired_95_lower_bound_positive",
    "identity_95_lower_bound_positive",
)
_SEED_KEYS = (
    "fit_seed",
    "fitted_target",
    "trajectory",
    "cap_to_target",
    "step_equivalence",
    "predicates",
)
_TARGET_KEYS = ("sha256", "row_norm_min", "row_norm_max", "validation")
_TRAJECTORY_KEYS = ("step", "sha256", "validation")
_COSINE_KEYS = ("row_cosines", "summary")
_SEED_PREDICATE_KEYS = (
    "head_cosine_at_least_0_95",
    "step_equivalence_at_least_64",
)
_DECISION_KEYS = ("per_variant", "selected_variant", "status")
_DECISION_VARIANT_KEYS = (
    "statistics",
    "predicates",
    "passes_static",
    "passes_all",
    "decision_level",
    "min_step_equivalence",
)
_CAP_VARIANTS = ("cap_centered", "cap_uncentered")


@dataclass(frozen=True)
class CapExecutionInventory:
    result: dict[str, object]
    fitting: tuple[object, ...]
    validation: tuple[object, ...]
    validation_group_represented: tuple[bool, ...]
    labels: dict[str, int]
    class_mean_sha256: str
    target_sha256_by_seed: dict[int, str]
    fit_steps: int
    batch_size: int
    peak_gpu_mib: int
    parent_class_mean_metric_sha256: str | None = None
    parent_target_metric_sha256_by_seed: dict[int, str] | None = None


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
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise TypeError("JSON payload must be an object")
    return value


def _object(value: object, keys: tuple[str, ...], name: str) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError(f"{name} schema differs")
    return value


def _string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    return value


def _sha256(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _commit(value: object, name: str) -> str:
    text = _string(value, name)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a full lowercase Git commit")
    return text


def _same_concrete(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return tuple(actual) == tuple(expected) and all(
            _same_concrete(actual[key], expected[key]) for key in expected
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _same_concrete(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def validate_run_config(value: object) -> None:
    config = _object(value, _TOP_KEYS, "run config")
    if config["schema_version"] != "unicom-cap-f0-run-v1":
        raise ValueError("run config version differs")

    spec = _object(config["spec"], _SPEC_KEYS, "spec")
    _string(spec["path"], "spec.path")
    _sha256(spec["sha256"], "spec.sha256")
    _commit(spec["commit"], "spec.commit")
    if not _same_concrete(spec, _FROZEN_SPEC):
        raise ValueError("spec authority differs")

    parent = _object(config["parent"], _PARENT_KEYS, "parent")
    _string(parent["path"], "parent.path")
    _sha256(parent["sha256"], "parent.sha256")
    _commit(parent["source_commit"], "parent.source_commit")
    if not _same_concrete(parent, _FROZEN_PARENT):
        raise ValueError("parent authority differs")

    environment = _object(config["environment"], _ENVIRONMENT_KEYS, "environment")
    for key in _ENVIRONMENT_KEYS:
        _string(environment[key], f"environment.{key}")
    if not _same_concrete(environment, _FROZEN_ENVIRONMENT):
        raise ValueError("environment authority differs")

    inputs = _object(config["inputs"], _INPUT_KEYS, "inputs")
    for key in ("unicom_checkout", "checkpoint", "dataset_root", "partition"):
        _string(inputs[key], f"inputs.{key}")
    _commit(inputs["unicom_revision"], "inputs.unicom_revision")
    _sha256(inputs["checkpoint_sha256"], "inputs.checkpoint_sha256")
    _sha256(inputs["partition_sha256"], "inputs.partition_sha256")
    if not _same_concrete(inputs, _FROZEN_INPUTS):
        raise ValueError("input authority differs")

    protocol = _object(config["protocol"], _PROTOCOL_KEYS, "protocol")
    if type(protocol["fit_seeds"]) is not list or protocol["fit_seeds"] != [0, 1, 2]:
        raise ValueError("protocol.fit_seeds differs")
    if type(protocol["snapshot_steps"]) is not list or protocol["snapshot_steps"] != [
        0,
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]:
        raise ValueError("protocol.snapshot_steps differs")
    for key in _PROTOCOL_KEYS:
        item = protocol[key]
        if key in {"fit_seeds", "snapshot_steps", "optimizer"}:
            continue
        if type(item) not in {int, float} or isinstance(item, bool) or not np.isfinite(item):
            raise TypeError(f"protocol.{key} must be a finite concrete number")
    _string(protocol["optimizer"], "protocol.optimizer")
    if not _same_concrete(protocol, _FROZEN_PROTOCOL):
        raise ValueError("protocol authority differs")

    source = _object(config["source"], _SOURCE_KEYS, "source")
    source_commit = _commit(source["commit"], "source.commit")
    if type(source["files"]) is not list or len(source["files"]) != len(_SOURCE_PATHS):
        raise ValueError("source.files differs")
    for expected_path, raw_row in zip(_SOURCE_PATHS, source["files"], strict=True):
        row = _object(raw_row, _SOURCE_ROW_KEYS, "source row")
        if row["path"] != expected_path:
            raise ValueError("source file order differs")
        _sha256(row["sha256"], f"source[{expected_path}].sha256")

    handoff = _object(config["handoff"], _HANDOFF_KEYS, "handoff")
    if _commit(handoff["parent_commit"], "handoff.parent_commit") != source_commit:
        raise ValueError("handoff parent differs from source commit")
    if handoff["sole_path"] != "docs/unicom_cap_f0_run_config.json":
        raise ValueError("handoff path differs")
    if handoff["detached_clean"] is not True:
        raise ValueError("handoff must require a clean detached checkout")

    result = _object(config["result"], _RESULT_KEYS, "result")
    if result["relative_path"] != f"reports/generated/unicom-cap-f0-{source_commit[:7]}.json":
        raise ValueError("result path differs from source commit")
    if result["schema_version"] != "unicom-cap-f0-v1":
        raise ValueError("result schema version differs")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _real_file(path: Path, name: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{name} is absent") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"{name} is not a real regular file")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise ValueError(f"{name} path differs")
    return resolved


def _real_directory(path: Path, name: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{name} is absent") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"{name} is not a real directory")
    resolved = path.resolve(strict=True)
    if resolved != path.absolute():
        raise ValueError(f"{name} path differs")
    return resolved


def authenticate_run(args: argparse.Namespace) -> dict[str, object]:
    repo_root = Path(__file__).resolve(strict=True).parents[1]
    config_path = _real_file(Path(args.config), "config")
    config = strict_json_object(config_path.read_bytes())
    validate_run_config(config)
    handoff = config["handoff"]
    source = config["source"]
    inputs = config["inputs"]
    parent = config["parent"]
    result = config["result"]

    if config_path != repo_root / handoff["sole_path"]:
        raise ValueError("config path differs")
    head = _git(repo_root, "rev-parse", "HEAD").decode().strip()
    parent_commit = _git(repo_root, "rev-parse", "HEAD^").decode().strip()
    if parent_commit != source["commit"] or parent_commit != handoff["parent_commit"]:
        raise ValueError("config handoff parent differs")
    detached = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if detached.returncode == 0:
        raise ValueError("checkout is not detached")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("checkout is not clean")
    changed = tuple(
        line
        for line in _git(
            repo_root, "diff-tree", "--no-commit-id", "--name-only", "-r", head
        )
        .decode()
        .splitlines()
        if line
    )
    if changed != (handoff["sole_path"],):
        raise ValueError("handoff edge differs")
    if _git(repo_root, "show", f"{head}:{handoff['sole_path']}") != config_path.read_bytes():
        raise ValueError("config worktree and Git bytes differ")

    for row in source["files"]:
        relative = row["path"]
        worktree = _real_file(repo_root / relative, f"source {relative}")
        payload = worktree.read_bytes()
        if _sha256_bytes(payload) != row["sha256"]:
            raise ValueError(f"source {relative} digest differs")
        if _git(repo_root, "show", f"{source['commit']}:{relative}") != payload:
            raise ValueError(f"source {relative} Git bytes differ")

    spec = config["spec"]
    spec_path = _real_file(repo_root / spec["path"], "spec")
    if _sha256_bytes(spec_path.read_bytes()) != spec["sha256"]:
        raise ValueError("spec digest differs")
    if _git(repo_root, "show", f"{spec['commit']}:{spec['path']}") != spec_path.read_bytes():
        raise ValueError("spec Git bytes differ")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", spec["commit"], source["commit"]],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    parent_path = _real_file(repo_root / parent["path"], "parent result")
    if parent_path != Path(args.parent_result).resolve(strict=True):
        raise ValueError("parent result flag differs")
    if _sha256_bytes(parent_path.read_bytes()) != parent["sha256"]:
        raise ValueError("parent result digest differs")
    parent_git_bytes = _git(
        repo_root, "show", f"{parent['source_commit']}:{parent['path']}"
    )
    if parent_git_bytes != parent_path.read_bytes():
        raise ValueError("parent result Git bytes differ")
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            parent["source_commit"],
            source["commit"],
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    unicom = _real_directory(Path(inputs["unicom_checkout"]), "UniCOM checkout")
    if unicom != Path(args.unicom_checkout).resolve(strict=True):
        raise ValueError("UniCOM checkout flag differs")
    if _git(unicom, "rev-parse", "HEAD").decode().strip() != inputs["unicom_revision"]:
        raise ValueError("UniCOM revision differs")
    checkpoint = _real_file(Path(inputs["checkpoint"]), "checkpoint")
    if checkpoint != Path(args.checkpoint).resolve(strict=True):
        raise ValueError("checkpoint flag differs")
    if _sha256_bytes(checkpoint.read_bytes()) != inputs["checkpoint_sha256"]:
        raise ValueError("checkpoint digest differs")
    dataset_root = _real_directory(Path(inputs["dataset_root"]), "dataset root")
    if dataset_root != Path(args.dataset_root).resolve(strict=True):
        raise ValueError("dataset root flag differs")
    partition = _real_file(Path(inputs["partition"]), "partition")
    if _sha256_bytes(partition.read_bytes()) != inputs["partition_sha256"]:
        raise ValueError("partition digest differs")

    output = Path(args.output).absolute()
    if output != repo_root / result["relative_path"]:
        raise ValueError("output flag differs")
    _real_directory(output.parent, "output parent")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    return {
        "config": config,
        "repo_root": repo_root,
        "source_commit": source["commit"],
        "head": head,
    }


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} differs")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} differs")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} differs")
    return value


def _float_list(value: object, count: int, name: str) -> tuple[float, ...]:
    if type(value) is not list or len(value) != count:
        raise ValueError(f"{name} count differs")
    return tuple(_finite_float(item, name) for item in value)


def _summary_from_rows(value: object, count: int, name: str) -> tuple[object, ...]:
    block = _object(value, _MISMATCH_KEYS, name)
    rows = _float_list(block["row_cosines"], count, f"{name} rows")
    if any(not -1.0 <= item <= 1.0 for item in rows):
        raise ValueError(f"{name} row cosine differs")
    summary = _object(block["summary"], _SUMMARY_KEYS, f"{name} summary")
    array = np.asarray(rows, dtype=np.float64)
    expected = (
        float(array.min()),
        float(np.quantile(array, 0.05, method="linear")),
        float(np.quantile(array, 0.5, method="linear")),
        float(array.mean()),
    )
    if tuple(_finite_float(summary[key], f"{name} summary") for key in _SUMMARY_KEYS) != expected:
        raise ValueError(f"{name} summary differs")
    return rows, summary


def _metric_from_json(
    value: object,
    *,
    mask_count: int,
    image_count: int,
    name: str,
) -> object:
    from sfora.unicom_probe import ProbeMetrics

    metric = _object(value, _METRIC_KEYS, name)
    per_mask = _float_list(metric["per_mask_mean_losses"], mask_count, name)
    per_mask_represented = _float_list(
        metric["per_mask_represented_mean_losses"], mask_count, name
    )
    per_mask_unrepresented = _float_list(
        metric["per_mask_unrepresented_mean_losses"], mask_count, name
    )
    per_image = _float_list(metric["per_image_mean_losses"], image_count, name)
    mean_loss = _finite_float(metric["mean_loss"], f"{name} mean loss")
    accuracy = _finite_float(metric["accuracy"], f"{name} accuracy")
    correct = metric["correct_count"]
    observations = metric["observation_count"]
    if (
        type(correct) is not int
        or type(observations) is not int
        or observations != image_count * mask_count
        or not 0 <= correct <= observations
        or accuracy != correct / observations
        or mean_loss != math.fsum(per_mask) / mask_count
        or mean_loss != math.fsum(per_image) / image_count
    ):
        raise ValueError(f"{name} aggregate differs")
    represented = _finite_float(metric["represented_mean_loss"], name)
    unrepresented = _finite_float(metric["unrepresented_mean_loss"], name)
    if (
        represented != math.fsum(per_mask_represented) / mask_count
        or unrepresented != math.fsum(per_mask_unrepresented) / mask_count
    ):
        raise ValueError(f"{name} stratum aggregate differs")
    return ProbeMetrics(
        mean_loss=mean_loss,
        accuracy=accuracy,
        correct_count=correct,
        observation_count=observations,
        per_mask_mean_losses=per_mask,
        per_mask_represented_mean_losses=per_mask_represented,
        per_mask_unrepresented_mean_losses=per_mask_unrepresented,
        per_image_mean_losses=per_image,
        represented_mean_loss=represented,
        unrepresented_mean_loss=unrepresented,
    )


def validate_result(value: object, *, inventory: object) -> None:
    from sfora.unicom_cap import CapCosineSummary, cap_decision

    result = _object(value, _RESULT_TOP_KEYS, "result")
    trusted = _object(
        inventory, ("authority", "runtime", "dataset", "protocol"), "inventory"
    )
    if result["schema_version"] != "unicom-cap-f0-v1":
        raise ValueError("result schema version differs")
    for section in ("authority", "dataset", "protocol"):
        if not _same_concrete(result[section], trusted[section]):
            raise ValueError(f"result {section} differs")

    runtime_keys = tuple(trusted["runtime"])
    if runtime_keys != _ENVIRONMENT_KEYS:
        raise ValueError("runtime inventory differs")
    runtime = _object(result["runtime"], _RUNTIME_RESULT_KEYS, "runtime")
    if not _same_concrete(
        {key: runtime[key] for key in _ENVIRONMENT_KEYS}, trusted["runtime"]
    ):
        raise ValueError("runtime authority differs")
    if _finite_float(runtime["elapsed_seconds"], "elapsed seconds") <= 0.0:
        raise ValueError("runtime observation differs")
    _nonnegative_int(runtime["peak_gpu_mib"], "peak GPU MiB")

    dataset = trusted["dataset"]
    protocol = trusted["protocol"]
    fitting_count = _positive_int(dataset["fitting_image_count"], "fitting count")
    class_count = _positive_int(
        dataset["optimization_identity_count"], "optimization identity count"
    )
    image_count = _positive_int(
        dataset["validation_image_count"], "validation image count"
    )
    feature_count = _positive_int(protocol["feature_count"], "feature count")
    target_row_norm = _finite_float(protocol["row_norm"], "target row norm")
    if target_row_norm <= 0.0:
        raise ValueError("target row norm differs")
    mask_count = _positive_int(
        protocol["evaluation_mask_sets"], "evaluation mask count"
    )
    construction_sets = _positive_int(
        protocol["covariance_mask_sets"], "construction mask count"
    )
    shards = _positive_int(protocol["shards"], "shard count")

    covariance = _object(result["covariance"], _COVARIANCE_KEYS, "covariance")
    if covariance["sample_count"] != fitting_count or covariance["feature_count"] != feature_count:
        raise ValueError("covariance shape differs")
    shrinkage = _finite_float(covariance["shrinkage"], "covariance shrinkage")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance shrinkage differs")
    encoded = _string(covariance["matrix_fp64_le_base64"], "covariance matrix")
    matrix_bytes = base64.b64decode(encoded, validate=True)
    if base64.b64encode(matrix_bytes).decode("ascii") != encoded:
        raise ValueError("covariance matrix encoding differs")
    if len(matrix_bytes) != feature_count * feature_count * 8:
        raise ValueError("covariance matrix byte count differs")
    matrix = np.frombuffer(matrix_bytes, dtype="<f8").reshape(feature_count, feature_count)
    if not np.isfinite(matrix).all() or not np.allclose(
        matrix, matrix.T, rtol=1e-12, atol=1e-14
    ):
        raise ValueError("covariance matrix differs")
    try:
        cholesky = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance matrix is not positive definite") from error
    eigenvalues = np.linalg.eigvalsh(matrix)
    probabilities = eigenvalues / eigenvalues.sum()
    diagonal = np.diag(cholesky)
    expected_covariance = {
        "trace": float(np.trace(matrix)),
        "cholesky_diagonal_min": float(diagonal.min()),
        "cholesky_diagonal_max": float(diagonal.max()),
        "sha256": _sha256_bytes(matrix_bytes),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "effective_rank": float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        ),
    }
    for key, expected in expected_covariance.items():
        actual = covariance[key]
        if key == "sha256":
            _sha256(actual, "covariance sha256")
        else:
            _finite_float(actual, f"covariance {key}")
        if actual != expected:
            raise ValueError(f"covariance {key} differs")
    mask_hashes = covariance["construction_mask_sha256"]
    if type(mask_hashes) is not list or len(mask_hashes) != construction_sets:
        raise ValueError("construction mask set count differs")
    for mask_set in mask_hashes:
        if type(mask_set) is not list or len(mask_set) != shards:
            raise ValueError("construction mask shard count differs")
        for digest in mask_set:
            _sha256(digest, "construction mask sha256")
    mismatch = _object(covariance["mismatch"], _CAP_VARIANTS, "mismatch")
    for variant in _CAP_VARIANTS:
        _summary_from_rows(
            mismatch[variant], construction_sets * class_count, f"{variant} mismatch"
        )

    class_mean = _object(result["class_mean"], _CLASS_MEAN_KEYS, "class mean")
    _sha256(class_mean["sha256"], "class mean sha256")
    row_norms = _float_list(class_mean["row_norms"], class_count, "class mean row norms")
    if any(
        not math.isclose(
            value, target_row_norm, rel_tol=2e-6, abs_tol=2e-7
        )
        for value in row_norms
    ):
        raise ValueError("class mean row norm differs")
    if (
        _finite_float(class_mean["row_norm_min"], "class mean row norm") != min(row_norms)
        or _finite_float(class_mean["row_norm_max"], "class mean row norm")
        != max(row_norms)
    ):
        raise ValueError("class mean row norm summary differs")
    class_metric = _metric_from_json(
        class_mean["validation"],
        mask_count=mask_count,
        image_count=image_count,
        name="class mean metric",
    )

    cap_metrics_block = _object(result["cap_metrics"], _CAP_VARIANTS, "CAP metrics")
    cap_metrics: dict[str, object] = {}
    for variant in _CAP_VARIANTS:
        block = _object(cap_metrics_block[variant], _CAP_METRIC_KEYS, variant)
        cap_metrics[variant] = _metric_from_json(
            block["validation"],
            mask_count=mask_count,
            image_count=image_count,
            name=f"{variant} metric",
        )
        _object(block["statistics"], _STATISTIC_KEYS, f"{variant} statistics")
        _object(block["predicates"], _STATIC_PREDICATE_KEYS, f"{variant} predicates")

    fit_seeds = tuple(protocol["fit_seeds"])
    snapshot_steps = tuple(protocol["snapshot_steps"])
    if type(result["seeds"]) is not list or len(result["seeds"]) != len(fit_seeds):
        raise ValueError("seed count differs")
    target_heads: dict[int, dict[str, object]] = {}
    trajectories: dict[int, dict[int, float]] = {}
    seed_blocks: dict[int, dict[str, object]] = {}
    for expected_seed, raw_seed in zip(fit_seeds, result["seeds"], strict=True):
        seed = _object(raw_seed, _SEED_KEYS, "seed")
        if type(seed["fit_seed"]) is not int or seed["fit_seed"] != expected_seed:
            raise ValueError("fit seed differs")
        fitted = _object(seed["fitted_target"], _TARGET_KEYS, "fitted target")
        _sha256(fitted["sha256"], "fitted target sha256")
        for key in ("row_norm_min", "row_norm_max"):
            if not math.isclose(
                _finite_float(fitted[key], f"fitted target {key}"),
                target_row_norm,
                rel_tol=2e-6,
                abs_tol=2e-7,
            ):
                raise ValueError("fitted target row norm differs")
        _metric_from_json(
            fitted["validation"],
            mask_count=mask_count,
            image_count=image_count,
            name="fitted target metric",
        )
        if type(seed["trajectory"]) is not list or len(seed["trajectory"]) != len(
            snapshot_steps
        ):
            raise ValueError("trajectory count differs")
        trajectory_losses: dict[int, float] = {}
        for expected_step, raw_snapshot in zip(
            snapshot_steps, seed["trajectory"], strict=True
        ):
            snapshot = _object(raw_snapshot, _TRAJECTORY_KEYS, "trajectory snapshot")
            if type(snapshot["step"]) is not int or snapshot["step"] != expected_step:
                raise ValueError("trajectory step differs")
            _sha256(snapshot["sha256"], "trajectory sha256")
            metric = _metric_from_json(
                snapshot["validation"],
                mask_count=mask_count,
                image_count=image_count,
                name="trajectory metric",
            )
            trajectory_losses[expected_step] = metric.mean_loss
            if expected_step == 0 and snapshot["sha256"] != class_mean["sha256"]:
                raise ValueError("trajectory initial head differs")
            if expected_step == snapshot_steps[-1] and (
                snapshot["sha256"] != fitted["sha256"]
                or snapshot["validation"] != fitted["validation"]
            ):
                raise ValueError("trajectory final target differs")
        cosines = _object(seed["cap_to_target"], _CAP_VARIANTS, "CAP target cosine")
        target_heads[expected_seed] = {}
        for variant in _CAP_VARIANTS:
            _rows, summary = _summary_from_rows(
                cosines[variant], class_count, f"{variant} target cosine"
            )
            target_heads[expected_seed][variant] = CapCosineSummary(
                *(_finite_float(summary[key], "target cosine") for key in _SUMMARY_KEYS)
            )
        _object(seed["step_equivalence"], _CAP_VARIANTS, "step equivalence")
        predicates = _object(seed["predicates"], _CAP_VARIANTS, "seed predicates")
        for variant in _CAP_VARIANTS:
            predicate = _object(
                predicates[variant], _SEED_PREDICATE_KEYS, f"{variant} seed predicates"
            )
            if any(type(predicate[key]) is not bool for key in _SEED_PREDICATE_KEYS):
                raise ValueError("seed predicate type differs")
        trajectories[expected_seed] = trajectory_losses
        seed_blocks[expected_seed] = seed

    recomputed = cap_decision(
        class_mean=class_metric,
        cap_metrics=cap_metrics,
        target_heads=target_heads,
        trajectories=trajectories,
        expected_mask_count=mask_count,
        expected_image_count=image_count,
    )
    for variant in _CAP_VARIANTS:
        expected_variant = recomputed.per_variant[variant]
        block = cap_metrics_block[variant]
        if block["statistics"] != vars(expected_variant.statistics):
            raise ValueError(f"{variant} statistics differ")
        if block["predicates"] != expected_variant.seed_invariant_predicates:
            raise ValueError(f"{variant} static predicates differ")
        for seed in fit_seeds:
            evidence = seed_blocks[seed]
            if (
                evidence["step_equivalence"][variant]
                != expected_variant.per_seed_step_equivalence[seed]
            ):
                raise ValueError(f"{variant} step equivalence differs")
            if evidence["predicates"][variant] != expected_variant.per_seed_predicates[seed]:
                raise ValueError(f"{variant} seed predicates differ")

    decision = _object(result["decision"], _DECISION_KEYS, "decision")
    per_variant = _object(decision["per_variant"], _CAP_VARIANTS, "decision variants")
    for variant in _CAP_VARIANTS:
        expected_variant = recomputed.per_variant[variant]
        expected_block = {
            "statistics": vars(expected_variant.statistics),
            "predicates": expected_variant.seed_invariant_predicates,
            "passes_static": expected_variant.passes_static,
            "passes_all": expected_variant.passes_all,
            "decision_level": expected_variant.decision_level,
            "min_step_equivalence": expected_variant.min_step_equivalence,
        }
        block = _object(per_variant[variant], _DECISION_VARIANT_KEYS, variant)
        if block != expected_block:
            raise ValueError(f"{variant} decision differs")
    if (
        decision["selected_variant"] != recomputed.selected_variant
        or decision["status"] != recomputed.status
        or result["candidate_values_computed"] is not True
    ):
        raise ValueError("CAP decision differs")


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def write_result_atomic(
    value: dict[str, object],
    output: Path,
    *,
    validator: Callable[[object], None],
) -> None:
    validator(value)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = _canonical_bytes(value)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    descriptor = -1
    directory_descriptor = -1
    linked = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("atomic result write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        persisted = temporary.read_bytes()
        if persisted != payload:
            raise ValueError("temporary result bytes differ")
        validator(strict_json_object(persisted))

        directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.link(temporary, output)
        linked = True
        os.fsync(directory_descriptor)
    except BaseException:
        if linked:
            try:
                if output.stat().st_ino == temporary.stat().st_ino:
                    output.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-replay-only", action="store_true")
    return parser.parse_args(argv)


def _encode_feature_sets(
    args: argparse.Namespace,
    fitting: tuple[object, ...],
    validation: tuple[object, ...],
    *,
    loader_workers: int = 4,
) -> tuple[object, object]:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from sfora.unicom_inshop import InshopRecord

    if (
        type(fitting) is not tuple
        or not fitting
        or type(validation) is not tuple
        or not validation
        or any(type(row) is not InshopRecord for row in fitting + validation)
        or type(loader_workers) is not int
        or loader_workers < 0
    ):
        raise ValueError("CAP feature row inventory differs")
    model, transform = _load_official_model(args.unicom_checkout, args.checkpoint)
    model = model.cuda().eval()
    combined = fitting + validation

    class EvaluationDataset(Dataset[object]):
        def __len__(self) -> int:
            return len(combined)

        def __getitem__(self, index: int) -> object:
            with Image.open(combined[index].image_path) as image:
                value = transform(image.convert("RGB"))
            if type(value) is not torch.Tensor or value.dtype != torch.float32:
                raise ValueError("CAP evaluation transform differs")
            return value

    loader = DataLoader(
        EvaluationDataset(),
        batch_size=128,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=True,
        drop_last=False,
    )
    chunks: list[object] = []
    for batch in loader:
        if type(batch) is not torch.Tensor or batch.dtype != torch.float32:
            raise ValueError("CAP feature batch differs")
        with torch.inference_mode():
            output = model(batch.cuda(non_blocking=False))
        if (
            type(output) is not torch.Tensor
            or output.dtype not in (torch.float16, torch.float32)
            or output.ndim != 2
            or output.shape[1] != 768
            or not torch.isfinite(output).all()
        ):
            raise ValueError("CAP feature output differs")
        chunks.append(output.float().detach().cpu())
    features = torch.cat(chunks, dim=0).contiguous()
    if features.shape != (len(combined), 768):
        raise ValueError("CAP encoded feature inventory differs")
    return (
        features[: len(fitting)].contiguous().cuda(),
        features[len(fitting) :].contiguous().cuda(),
    )


def _load_official_model(checkout: Path, checkpoint: Path) -> tuple[object, object]:
    import importlib

    package_root = (checkout / "unicom").resolve()
    if any(name == "unicom" or name.startswith("unicom.") for name in sys.modules):
        raise ValueError("UNICOM package was imported before source authentication")
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    if Path(unicom.__file__).resolve().parent != package_root / "unicom":
        raise ValueError("imported UNICOM package differs")
    return unicom.load("ViT-L/14@336px", download_root=str(checkpoint.parent))


def _validate_runtime(expected: object) -> None:
    import sklearn
    import torch

    authority = _object(expected, _ENVIRONMENT_KEYS, "runtime authority")
    if not torch.cuda.is_available():
        raise ValueError("CAP CUDA runtime differs")
    observed = {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "sklearn": str(sklearn.__version__),
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0),
        "model_dtype": "float32",
        "reduction_dtype": "float64",
    }
    if not _same_concrete(observed, authority):
        raise ValueError("CAP runtime differs")


def _build_execution_inventory(
    args: argparse.Namespace, authenticated: object
) -> CapExecutionInventory:
    from sfora.unicom_inshop import parse_inshop_partition
    from sfora.unicom_probe import split_probe_records
    from sfora.unicom_training import identity_holdout

    trusted = _object(
        authenticated,
        ("config", "repo_root", "source_commit", "head"),
        "authenticated run",
    )
    config = trusted["config"]
    if type(config) is not dict:
        raise ValueError("authenticated config differs")
    parent = strict_json_object(Path(args.parent_result).read_bytes())
    parent_dataset = parent.get("dataset")
    parent_class_mean = parent.get("class_mean")
    parent_probes = parent.get("probes")
    if (
        type(parent_dataset) is not dict
        or type(parent_class_mean) is not dict
        or type(parent_probes) is not list
        or len(parent_probes) != 3
        or tuple(row.get("fit_seed") for row in parent_probes if type(row) is dict)
        != (0, 1, 2)
    ):
        raise ValueError("parent primitive schema differs")
    class_mean_sha256 = _sha256(
        parent_class_mean.get("sha256"), "parent class mean sha256"
    )
    class_mean_validation = parent_class_mean.get("validation")
    if type(class_mean_validation) is not dict:
        raise ValueError("parent class mean metric differs")
    target_sha256_by_seed: dict[int, str] = {}
    target_metric_sha256_by_seed: dict[int, str] = {}
    for expected_seed, row in enumerate(parent_probes):
        if type(row) is not dict or row.get("fit_seed") != expected_seed:
            raise ValueError("parent probe order differs")
        target_sha256_by_seed[expected_seed] = _sha256(
            row.get("sha256"), f"parent target {expected_seed} sha256"
        )
        validation = row.get("validation")
        if type(validation) is not dict:
            raise ValueError(f"parent target {expected_seed} metric differs")
        target_metric_sha256_by_seed[expected_seed] = _sha256_bytes(
            _canonical_bytes(validation)
        )

    records = parse_inshop_partition(Path(args.dataset_root))
    train_records = tuple(row for row in records if row.split == "train")
    optimization, _query, _gallery, labels = identity_holdout(
        train_records,
        fraction=config["protocol"]["holdout_fraction"],
        seed=config["protocol"]["holdout_seed"],
    )
    split = split_probe_records(
        optimization, labels, seed=config["protocol"]["split_seed"]
    )
    observed_dataset = {
        "partition_sha256": config["inputs"]["partition_sha256"],
        "optimization_identity_count": len(labels),
        "optimization_image_count": len(optimization),
        "fitting_image_count": len(split.fitting),
        "validation_image_count": len(split.validation),
        "validation_class_count": split.validation_class_count,
        "singleton_class_count": split.singleton_class_count,
        "excluded_same_series_count": split.excluded_same_series_count,
        "represented_validation_count": sum(split.validation_group_represented),
        "unrepresented_validation_count": sum(
            not value for value in split.validation_group_represented
        ),
    }
    if not _same_concrete(observed_dataset, parent_dataset):
        raise ValueError("parent dataset reproduction differs")
    protocol = config["protocol"]
    result_protocol = {
        "fit_seeds": protocol["fit_seeds"],
        "snapshot_steps": protocol["snapshot_steps"],
        "evaluation_mask_sets": protocol["evaluation_mask_sets"],
        "covariance_mask_sets": protocol["covariance_mask_sets"],
        "shards": protocol["shards"],
        "feature_count": protocol["feature_count"],
        "row_norm": protocol["row_norm"],
        "paired_t_critical_df63": protocol["paired_t_critical_df63"],
        "paired_t_critical_df3187": protocol["paired_t_critical_df3187"],
        "loss_delta_minimum": protocol["loss_delta_minimum"],
        "accuracy_delta_minimum": protocol["accuracy_delta_minimum"],
        "non_worse_mask_minimum": protocol["non_worse_mask_minimum"],
        "head_cosine_mean_minimum": protocol["head_cosine_mean_minimum"],
        "step_equivalence_minimum": protocol["step_equivalence_minimum"],
    }
    result_inventory = {
        "authority": {
            "spec_path": config["spec"]["path"],
            "spec_sha256": config["spec"]["sha256"],
            "spec_commit": config["spec"]["commit"],
            "parent_path": config["parent"]["path"],
            "parent_sha256": config["parent"]["sha256"],
            "parent_source_commit": config["parent"]["source_commit"],
            "source_commit": trusted["source_commit"],
            "handoff_commit": trusted["head"],
            "unicom_revision": config["inputs"]["unicom_revision"],
            "checkpoint_sha256": config["inputs"]["checkpoint_sha256"],
            "partition_sha256": config["inputs"]["partition_sha256"],
        },
        "runtime": config["environment"],
        "dataset": observed_dataset,
        "protocol": result_protocol,
    }
    return CapExecutionInventory(
        result=result_inventory,
        fitting=split.fitting,
        validation=split.validation,
        validation_group_represented=split.validation_group_represented,
        labels=labels,
        class_mean_sha256=class_mean_sha256,
        target_sha256_by_seed=target_sha256_by_seed,
        fit_steps=protocol["fit_steps"],
        batch_size=protocol["batch_size"],
        peak_gpu_mib=0,
        parent_class_mean_metric_sha256=_sha256_bytes(
            _canonical_bytes(class_mean_validation)
        ),
        parent_target_metric_sha256_by_seed=target_metric_sha256_by_seed,
    )


def _tensor_sha256(value: object) -> str:
    import torch

    if type(value) is not torch.Tensor:
        raise TypeError("CAP tensor differs")
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _metric_payload(value: object) -> dict[str, object]:
    from sfora.unicom_probe import ProbeMetrics

    if type(value) is not ProbeMetrics:
        raise TypeError("CAP metric differs")
    return {
        "mean_loss": value.mean_loss,
        "accuracy": value.accuracy,
        "correct_count": value.correct_count,
        "observation_count": value.observation_count,
        "per_mask_mean_losses": list(value.per_mask_mean_losses),
        "per_mask_represented_mean_losses": list(
            value.per_mask_represented_mean_losses
        ),
        "per_mask_unrepresented_mean_losses": list(
            value.per_mask_unrepresented_mean_losses
        ),
        "per_image_mean_losses": list(value.per_image_mean_losses),
        "represented_mean_loss": value.represented_mean_loss,
        "unrepresented_mean_loss": value.unrepresented_mean_loss,
    }


def _row_cosine_payload(left: object, right: object) -> dict[str, object]:
    import torch

    if type(left) is not torch.Tensor or type(right) is not torch.Tensor:
        raise TypeError("CAP cosine tensor differs")
    values = (
        torch.nn.functional.cosine_similarity(left, right, dim=1)
        .clamp(-1.0, 1.0)
        .double()
        .cpu()
        .numpy()
    )
    if not np.isfinite(values).all():
        raise ValueError("CAP cosine differs")
    return {
        "row_cosines": values.tolist(),
        "summary": {
            "minimum": float(values.min()),
            "p05": float(np.quantile(values, 0.05, method="linear")),
            "median": float(np.quantile(values, 0.5, method="linear")),
            "mean": float(values.mean()),
        },
    }


def execute_screen(
    args: argparse.Namespace, inventory: CapExecutionInventory
) -> dict[str, object]:
    import torch

    from sfora.unicom_cap import (
        CAP_VARIANTS,
        CapCosineSummary,
        build_cap_heads,
        cap_decision,
        covariance_mask_mismatch,
    )
    from sfora.unicom_inshop import InshopRecord
    from sfora.unicom_probe import (
        class_mean_head,
        evaluate_probe_head,
        fit_spherical_probe_trajectory,
    )

    if type(inventory) is not CapExecutionInventory:
        raise TypeError("CAP execution inventory differs")
    trusted = _object(
        inventory.result, ("authority", "runtime", "dataset", "protocol"), "inventory"
    )
    dataset = trusted["dataset"]
    protocol = trusted["protocol"]
    class_count = _positive_int(
        dataset["optimization_identity_count"], "optimization identity count"
    )
    fitting_count = _positive_int(dataset["fitting_image_count"], "fitting count")
    validation_count = _positive_int(
        dataset["validation_image_count"], "validation count"
    )
    feature_count = _positive_int(protocol["feature_count"], "feature count")
    mask_count = _positive_int(protocol["evaluation_mask_sets"], "mask count")
    covariance_mask_count = _positive_int(
        protocol["covariance_mask_sets"], "covariance mask count"
    )
    row_norm = _finite_float(protocol["row_norm"], "row norm")
    if (
        type(inventory.fitting) is not tuple
        or len(inventory.fitting) != fitting_count
        or type(inventory.validation) is not tuple
        or len(inventory.validation) != validation_count
        or any(type(row) is not InshopRecord for row in inventory.fitting)
        or any(type(row) is not InshopRecord for row in inventory.validation)
        or type(inventory.labels) is not dict
        or tuple(inventory.labels.values()) != tuple(range(class_count))
        or type(inventory.validation_group_represented) is not tuple
        or len(inventory.validation_group_represented) != validation_count
        or any(
            type(value) is not bool
            for value in inventory.validation_group_represented
        )
        or not any(inventory.validation_group_represented)
        or all(inventory.validation_group_represented)
        or tuple(inventory.target_sha256_by_seed) != tuple(protocol["fit_seeds"])
        or type(inventory.fit_steps) is not int
        or inventory.fit_steps <= 0
        or type(inventory.batch_size) is not int
        or inventory.batch_size <= 0
        or type(inventory.peak_gpu_mib) is not int
        or inventory.peak_gpu_mib < 0
    ):
        raise ValueError("CAP execution inventory differs")
    _sha256(inventory.class_mean_sha256, "registered class mean")
    for digest in inventory.target_sha256_by_seed.values():
        _sha256(digest, "registered target")
    if inventory.parent_class_mean_metric_sha256 is not None:
        _sha256(
            inventory.parent_class_mean_metric_sha256,
            "registered parent class mean metric",
        )
    if inventory.parent_target_metric_sha256_by_seed is not None:
        if tuple(inventory.parent_target_metric_sha256_by_seed) != tuple(
            protocol["fit_seeds"]
        ):
            raise ValueError("registered parent target metric seeds differ")
        for digest in inventory.parent_target_metric_sha256_by_seed.values():
            _sha256(digest, "registered parent target metric")

    started = time.perf_counter()
    fitting_features, validation_features = _encode_feature_sets(
        args, inventory.fitting, inventory.validation
    )
    if (
        type(fitting_features) is not torch.Tensor
        or fitting_features.dtype != torch.float32
        or fitting_features.shape != (fitting_count, feature_count)
        or not fitting_features.is_contiguous()
        or type(validation_features) is not torch.Tensor
        or validation_features.dtype != torch.float32
        or validation_features.shape != (validation_count, feature_count)
        or not validation_features.is_contiguous()
        or fitting_features.device != validation_features.device
    ):
        raise ValueError("CAP feature inventory differs")
    device = fitting_features.device
    fitting_labels = torch.tensor(
        [inventory.labels[row.label] for row in inventory.fitting],
        dtype=torch.int64,
        device=device,
    )
    validation_labels = torch.tensor(
        [inventory.labels[row.label] for row in inventory.validation],
        dtype=torch.int64,
        device=device,
    )
    class_mean = class_mean_head(fitting_features, fitting_labels, class_count)
    if _tensor_sha256(class_mean) != inventory.class_mean_sha256:
        raise ValueError("registered class mean differs")
    class_metric = evaluate_probe_head(
        validation_features,
        validation_labels,
        class_mean,
        validation_group_represented=inventory.validation_group_represented,
        mask_sets=mask_count,
    )
    if (
        inventory.parent_class_mean_metric_sha256 is not None
        and _sha256_bytes(_canonical_bytes(_metric_payload(class_metric)))
        != inventory.parent_class_mean_metric_sha256
    ):
        raise ValueError("parent class mean metric differs")
    trajectories: dict[int, dict[int, float]] = {}
    seed_primitives: list[dict[str, object]] = []
    fitted_targets: dict[int, object] = {}
    snapshot_steps = tuple(protocol["snapshot_steps"])
    for seed in tuple(protocol["fit_seeds"]):
        fit, snapshots = fit_spherical_probe_trajectory(
            fitting_features,
            fitting_labels,
            class_mean,
            snapshot_steps=snapshot_steps,
            steps=inventory.fit_steps,
            batch_size=inventory.batch_size,
            fit_seed=seed,
        )
        target_sha256 = _tensor_sha256(fit.head)
        if target_sha256 != inventory.target_sha256_by_seed[seed]:
            raise ValueError("registered fitted target differs")
        target_metric = evaluate_probe_head(
            validation_features,
            validation_labels,
            fit.head,
            validation_group_represented=inventory.validation_group_represented,
            mask_sets=mask_count,
        )
        if (
            inventory.parent_target_metric_sha256_by_seed is not None
            and _sha256_bytes(_canonical_bytes(_metric_payload(target_metric)))
            != inventory.parent_target_metric_sha256_by_seed[seed]
        ):
            raise ValueError(f"parent seed {seed} metric differs")
        trajectory_rows: list[dict[str, object]] = []
        trajectory_losses: dict[int, float] = {}
        for step in snapshot_steps:
            snapshot = snapshots[step]
            metric = evaluate_probe_head(
                validation_features,
                validation_labels,
                snapshot,
                validation_group_represented=inventory.validation_group_represented,
                mask_sets=mask_count,
            )
            trajectory_rows.append(
                {
                    "step": step,
                    "sha256": _tensor_sha256(snapshot),
                    "validation": _metric_payload(metric),
                }
            )
            trajectory_losses[step] = metric.mean_loss
        trajectories[seed] = trajectory_losses
        target_norms = torch.linalg.vector_norm(fit.head, dim=1).double().cpu()
        fitted_targets[seed] = fit.head
        seed_primitives.append(
            {
                "fit_seed": seed,
                "fitted_target": {
                    "sha256": target_sha256,
                    "row_norm_min": float(target_norms.min()),
                    "row_norm_max": float(target_norms.max()),
                    "validation": _metric_payload(target_metric),
                },
                "trajectory": trajectory_rows,
            }
        )
        del snapshots, fit

    construction = build_cap_heads(fitting_features, fitting_labels, row_norm=row_norm)
    diagnostic = covariance_mask_mismatch(
        construction,
        seed=23_006,
        mask_sets=covariance_mask_count,
    )
    cap_metric_objects = {
        variant: evaluate_probe_head(
            validation_features,
            validation_labels,
            construction.heads[variant],
            validation_group_represented=inventory.validation_group_represented,
            mask_sets=mask_count,
        )
        for variant in CAP_VARIANTS
    }
    target_heads: dict[int, dict[str, CapCosineSummary]] = {}
    for seed, primitive in zip(
        tuple(protocol["fit_seeds"]), seed_primitives, strict=True
    ):
        cosine_payload = {
            variant: _row_cosine_payload(
                construction.heads[variant], fitted_targets[seed]
            )
            for variant in CAP_VARIANTS
        }
        primitive["cap_to_target"] = cosine_payload
        target_heads[seed] = {
            variant: CapCosineSummary(
                *(cosine_payload[variant]["summary"][key] for key in _SUMMARY_KEYS)
            )
            for variant in CAP_VARIANTS
        }

    decision = cap_decision(
        class_mean=class_metric,
        cap_metrics=cap_metric_objects,
        target_heads=target_heads,
        trajectories=trajectories,
        expected_mask_count=mask_count,
        expected_image_count=validation_count,
    )
    for seed, primitive in zip(tuple(protocol["fit_seeds"]), seed_primitives, strict=True):
        primitive["step_equivalence"] = {
            variant: decision.per_variant[variant].per_seed_step_equivalence[seed]
            for variant in CAP_VARIANTS
        }
        primitive["predicates"] = {
            variant: decision.per_variant[variant].per_seed_predicates[seed]
            for variant in CAP_VARIANTS
        }
    covariance = np.ascontiguousarray(construction.covariance, dtype="<f8")
    class_norms = torch.linalg.vector_norm(class_mean, dim=1).double().cpu().numpy()
    cap_metrics = {
        variant: {
            "validation": _metric_payload(cap_metric_objects[variant]),
            "statistics": vars(decision.per_variant[variant].statistics),
            "predicates": decision.per_variant[variant].seed_invariant_predicates,
        }
        for variant in CAP_VARIANTS
    }
    result = {
        "schema_version": "unicom-cap-f0-v1",
        "authority": trusted["authority"],
        "runtime": {
            **trusted["runtime"],
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_gpu_mib": inventory.peak_gpu_mib,
        },
        "dataset": trusted["dataset"],
        "protocol": trusted["protocol"],
        "covariance": {
            "sample_count": construction.sample_count,
            "feature_count": construction.feature_count,
            "shrinkage": construction.shrinkage,
            "matrix_fp64_le_base64": base64.b64encode(
                covariance.tobytes(order="C")
            ).decode("ascii"),
            "trace": construction.covariance_trace,
            "cholesky_diagonal_min": construction.cholesky_diagonal_min,
            "cholesky_diagonal_max": construction.cholesky_diagonal_max,
            "sha256": construction.covariance_sha256,
            "condition_number": construction.condition_number,
            "effective_rank": construction.effective_rank,
            "construction_mask_sha256": [
                list(mask_set) for mask_set in diagnostic["mask_sha256"]
            ],
            "mismatch": {
                variant: {
                    "row_cosines": diagnostic["cosines"][variant]["row_cosines"],
                    "summary": {
                        key: diagnostic["cosines"][variant][key]
                        for key in _SUMMARY_KEYS
                    },
                }
                for variant in CAP_VARIANTS
            },
        },
        "class_mean": {
            "sha256": inventory.class_mean_sha256,
            "row_norms": class_norms.tolist(),
            "row_norm_min": float(class_norms.min()),
            "row_norm_max": float(class_norms.max()),
            "validation": _metric_payload(class_metric),
        },
        "cap_metrics": cap_metrics,
        "seeds": seed_primitives,
        "decision": {
            "per_variant": {
                variant: {
                    "statistics": vars(decision.per_variant[variant].statistics),
                    "predicates": decision.per_variant[
                        variant
                    ].seed_invariant_predicates,
                    "passes_static": decision.per_variant[variant].passes_static,
                    "passes_all": decision.per_variant[variant].passes_all,
                    "decision_level": decision.per_variant[variant].decision_level,
                    "min_step_equivalence": decision.per_variant[
                        variant
                    ].min_step_equivalence,
                }
                for variant in CAP_VARIANTS
            },
            "selected_variant": decision.selected_variant,
            "status": decision.status,
        },
        "candidate_values_computed": True,
    }
    validate_result(result, inventory=trusted)
    del fitting_features, validation_features, class_mean, construction
    return result


def run(
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    authenticated = authenticate_run(args)
    import torch
    from threadpoolctl import threadpool_limits

    torch_state = torch.get_rng_state().clone()
    torch_threads = torch.get_num_threads()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            _validate_runtime(authenticated["config"]["environment"])
            inventory = _build_execution_inventory(args, authenticated)
            return _execute_with_runtime_observation(args, inventory)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(torch_threads)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _execute_with_runtime_observation(
    args: argparse.Namespace, inventory: CapExecutionInventory
) -> tuple[dict[str, object], CapExecutionInventory]:
    import torch

    if type(inventory) is not CapExecutionInventory:
        raise TypeError("CAP execution inventory differs")
    torch.cuda.reset_peak_memory_stats()
    value = execute_screen(args, inventory)
    torch.cuda.synchronize()
    peak_gpu_mib = math.ceil(torch.cuda.max_memory_allocated() / 1024**2)
    runtime = value.get("runtime")
    if type(runtime) is not dict:
        raise ValueError("CAP runtime result differs")
    runtime["peak_gpu_mib"] = peak_gpu_mib
    observed_inventory = replace(inventory, peak_gpu_mib=peak_gpu_mib)
    validate_result(value, inventory=observed_inventory.result)
    return value, observed_inventory


def run_parent_replay_preflight(_args: argparse.Namespace) -> dict[str, object]:
    args = _args
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    authenticated = authenticate_run(args)
    import torch
    from threadpoolctl import threadpool_limits

    torch_state = torch.get_rng_state().clone()
    torch_threads = torch.get_num_threads()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            _validate_runtime(authenticated["config"]["environment"])
            inventory = _build_execution_inventory(args, authenticated)
            return _compute_parent_replay(args, inventory)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.set_num_threads(torch_threads)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _compute_parent_replay(
    args: argparse.Namespace, inventory: CapExecutionInventory
) -> dict[str, object]:
    import torch

    from sfora.unicom_probe import class_mean_head, fit_spherical_probe_trajectory

    if type(inventory) is not CapExecutionInventory:
        raise TypeError("CAP execution inventory differs")
    trusted = _object(
        inventory.result, ("authority", "runtime", "dataset", "protocol"), "inventory"
    )
    protocol = trusted["protocol"]
    fitting_features, validation_features = _encode_feature_sets(
        args, inventory.fitting, inventory.validation
    )
    fitting_labels = torch.tensor(
        [inventory.labels[row.label] for row in inventory.fitting],
        dtype=torch.int64,
        device=fitting_features.device,
    )
    class_mean = class_mean_head(
        fitting_features, fitting_labels, len(inventory.labels)
    )
    class_sha256 = _tensor_sha256(class_mean)
    if class_sha256 != inventory.class_mean_sha256:
        raise ValueError("registered class mean differs")
    target_sha256_by_seed: dict[str, str] = {}
    for seed in tuple(protocol["fit_seeds"]):
        fit, snapshots = fit_spherical_probe_trajectory(
            fitting_features,
            fitting_labels,
            class_mean,
            snapshot_steps=tuple(protocol["snapshot_steps"]),
            steps=inventory.fit_steps,
            batch_size=inventory.batch_size,
            fit_seed=seed,
        )
        digest = _tensor_sha256(fit.head)
        if digest != inventory.target_sha256_by_seed[seed]:
            raise ValueError(f"registered fitted target {seed} differs")
        target_sha256_by_seed[str(seed)] = digest
        del snapshots, fit
    del fitting_features, validation_features, class_mean
    return {
        "class_mean_sha256": class_sha256,
        "target_sha256_by_seed": target_sha256_by_seed,
        "candidate_values_computed": False,
    }


def _validate_parent_replay(value: object) -> dict[str, object]:
    replay = _object(
        value,
        ("class_mean_sha256", "target_sha256_by_seed", "candidate_values_computed"),
        "parent replay",
    )
    _sha256(replay["class_mean_sha256"], "parent replay class mean")
    targets = _object(
        replay["target_sha256_by_seed"], ("0", "1", "2"), "parent replay targets"
    )
    for seed in ("0", "1", "2"):
        _sha256(targets[seed], f"parent replay target {seed}")
    if replay["candidate_values_computed"] is not False:
        raise ValueError("parent replay computed candidate values")
    return replay


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    if (
        args.output.exists()
        or args.output.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        return 2
    try:
        if args.parent_replay_only:
            replay = _validate_parent_replay(run_parent_replay_preflight(args))
            sys.stdout.write(_canonical_bytes(replay).decode("utf-8"))
            return 0
        value, inventory = run(args)

        def bound_validator(candidate: object) -> None:
            validate_result(candidate, inventory=inventory)

        write_result_atomic(value, args.output, validator=bound_validator)
    except Exception:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
