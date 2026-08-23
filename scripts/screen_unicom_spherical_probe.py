#!/usr/bin/env python3
"""Run the training-data-only UniCOM spherical-probe direction screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import stat
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sfora.unicom_inshop import InshopRecord, parse_inshop_partition
from sfora.unicom_probe import (
    ProbeFit,
    ProbeGradientMetrics,
    ProbeMetrics,
    ProbeSplit,
    class_mean_head,
    compare_probe_gradients,
    evaluate_probe_heads,
    fit_spherical_probe,
    probe_decision,
    split_probe_records,
)
from sfora.unicom_training import identity_holdout

TOP_KEYS = (
    "schema_version",
    "source",
    "model",
    "dataset",
    "protocol",
    "class_mean",
    "probes",
    "decision",
    "runtime",
)
SOURCE_KEYS = (
    "git_revision",
    "script_sha256",
    "module_sha256",
    "training_sha256",
    "inshop_sha256",
)
MODEL_KEYS = ("unicom_revision", "checkpoint_sha256")
DATASET_KEYS = (
    "partition_sha256",
    "optimization_identity_count",
    "optimization_image_count",
    "fitting_image_count",
    "validation_image_count",
    "validation_class_count",
    "singleton_class_count",
    "excluded_same_series_count",
    "represented_validation_count",
    "unrepresented_validation_count",
)
PROTOCOL_KEYS = (
    "holdout_fraction",
    "holdout_seed",
    "split_seed",
    "fit_seeds",
    "fit_steps",
    "batch_size",
    "batch_seed",
    "mask_seed",
    "evaluation_mask_seed",
    "diagnostic_seed",
    "gradient_seed",
    "evaluation_mask_sets",
    "shards",
    "selected_features",
    "margin",
    "accuracy_margin",
    "scale",
    "optimizer",
    "row_norm",
    "paired_t_critical_df63",
    "paired_t_critical_df3187",
    "positive_mask_minimum",
    "head_cosine_minimum",
    "head_cosine_mean_minimum",
    "gradient_median_cosine_maximum",
)
FIT_KEYS = ("initial_loss", "final_loss", "steps")
CLASS_MEAN_KEYS = ("sha256", "row_norm_min", "row_norm_max", "validation")
PROBE_KEYS = (
    "fit_seed",
    "sha256",
    "row_norm_min",
    "row_norm_max",
    "fit",
    "head_cosine",
    "validation",
    "gradient",
)
COSINE_KEYS = ("minimum", "p05", "median", "mean")
METRIC_KEYS = (
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
GRADIENT_KEYS = (
    "class_mean_l2",
    "spherical_probe_l2",
    "relative_l2_difference",
    "spherical_to_class_mean_l2_ratio",
    "cosine_min",
    "cosine_p05",
    "cosine_median",
    "cosine_mean",
    "zero_gradient_row_count",
)
DECISION_KEYS = ("status", "per_seed")
SEED_DECISION_KEYS = ("fit_seed", "statistics", "predicates")
STATISTIC_KEYS = (
    "mean_paired_loss_delta",
    "paired_95_lower_bound",
    "identity_95_lower_bound",
    "positive_mask_count",
    "unrepresented_paired_95_lower_bound",
    "unrepresented_positive_mask_count",
    "accuracy_delta",
    "represented_loss_delta",
    "unrepresented_loss_delta",
)
PREDICATE_KEYS = (
    "fit_loss_decreased",
    "head_cosine_minimum",
    "head_cosine_mean",
    "validation_loss_positive",
    "paired_95_lower_positive",
    "identity_95_lower_positive",
    "positive_mask_majority",
    "validation_accuracy_noninferior",
    "represented_loss_positive",
    "unrepresented_loss_positive",
    "unrepresented_paired_95_lower_positive",
    "unrepresented_positive_mask_majority",
    "gradient_median_cosine",
)
RUNTIME_KEYS = ("python", "torch", "numpy", "cuda", "elapsed_seconds", "peak_gpu_mib")
UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
OPTIMIZER_ID = "AdamW(lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0)"


@dataclass(frozen=True)
class ScreenInventory:
    partition_sha256: str
    optimization: tuple[InshopRecord, ...]
    labels: dict[str, int]
    split: ProbeSplit
    source_revision: str
    script_sha256: str
    module_sha256: str
    training_sha256: str
    inshop_sha256: str


def _validate_inventory(value: ScreenInventory) -> None:
    if (
        type(value) is not ScreenInventory
        or not _sha256(value.partition_sha256)
        or type(value.optimization) is not tuple
        or len(value.optimization) != 20_650
        or type(value.labels) is not dict
        or len(value.labels) != 3_200
        or tuple(value.labels.values()) != tuple(range(3_200))
        or type(value.split) is not ProbeSplit
        or len(value.split.fitting) != 14_330
        or len(value.split.validation) != 3_188
        or value.split.validation_class_count != 3_188
        or value.split.singleton_class_count != 12
        or value.split.excluded_same_series_count != 3_132
        or len(value.split.validation_group_represented) != 3_188
        or sum(value.split.validation_group_represented) != 2_162
        or not _sha256(value.script_sha256)
        or not _sha256(value.module_sha256)
        or not _sha256(value.training_sha256)
        or not _sha256(value.inshop_sha256)
        or type(value.source_revision) is not str
        or len(value.source_revision) != 40
        or any(
            type(row) is not InshopRecord or row.label not in value.labels
            for row in value.split.fitting + value.split.validation
        )
    ):
        raise ValueError("probe authenticated inventory differs")


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


def _source_binding() -> tuple[str, dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    revision = _git_revision(root)
    relative_paths = (
        "scripts/screen_unicom_spherical_probe.py",
        "src/sfora/unicom_probe.py",
        "src/sfora/unicom_training.py",
        "src/sfora/unicom_inshop.py",
    )
    digests: dict[str, str] = {}
    for relative in relative_paths:
        path = root / relative
        worktree = path.read_bytes()
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"{revision}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if worktree != blob:
            raise ValueError(f"executing probe source differs from Git blob: {relative}")
        digests[relative] = hashlib.sha256(worktree).hexdigest()
    return revision, digests


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def strict_json_object(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("JSON payload must be bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        payload,
        parse_constant=reject_constant,
        object_pairs_hook=reject_pairs,
    )
    if type(value) is not dict:
        raise ValueError("JSON top level must be an object")
    return value


def _metric(value: object, name: str) -> ProbeMetrics:
    if type(value) is not dict or tuple(value) != METRIC_KEYS:
        raise ValueError(f"{name} metric schema differs")
    result = ProbeMetrics(
        mean_loss=_finite_float(value["mean_loss"], f"{name} mean loss", positive=True),
        accuracy=_finite_float(value["accuracy"], f"{name} accuracy"),
        correct_count=value["correct_count"],
        observation_count=value["observation_count"],
        per_mask_mean_losses=tuple(value["per_mask_mean_losses"])
        if type(value["per_mask_mean_losses"]) is list
        else (),
        per_mask_represented_mean_losses=tuple(
            value["per_mask_represented_mean_losses"]
        )
        if type(value["per_mask_represented_mean_losses"]) is list
        else (),
        per_mask_unrepresented_mean_losses=tuple(
            value["per_mask_unrepresented_mean_losses"]
        )
        if type(value["per_mask_unrepresented_mean_losses"]) is list
        else (),
        per_image_mean_losses=tuple(value["per_image_mean_losses"])
        if type(value["per_image_mean_losses"]) is list
        else (),
        represented_mean_loss=_finite_float(
            value["represented_mean_loss"], f"{name} represented loss", positive=True
        ),
        unrepresented_mean_loss=_finite_float(
            value["unrepresented_mean_loss"],
            f"{name} unrepresented loss",
            positive=True,
        ),
    )
    if (
        type(result.correct_count) is not int
        or type(result.observation_count) is not int
        or result.observation_count <= 0
        or not 0 <= result.correct_count <= result.observation_count
        or not 0.0 <= result.accuracy <= 1.0
        or result.accuracy != result.correct_count / result.observation_count
        or len(result.per_mask_mean_losses) != 64
        or len(result.per_mask_represented_mean_losses) != 64
        or len(result.per_mask_unrepresented_mean_losses) != 64
        or len(result.per_image_mean_losses) * 64 != result.observation_count
        or any(
            type(item) is not float or not math.isfinite(item) or item <= 0.0
            for item in result.per_mask_mean_losses
        )
        or result.mean_loss != math.fsum(result.per_mask_mean_losses) / 64
        or result.represented_mean_loss
        != math.fsum(result.per_mask_represented_mean_losses) / 64
        or result.unrepresented_mean_loss
        != math.fsum(result.per_mask_unrepresented_mean_losses) / 64
    ):
        raise ValueError(f"{name} metric values differ")
    return result


def validate_result(value: object) -> None:
    if type(value) is not dict or tuple(value) != TOP_KEYS:
        raise ValueError("probe result schema differs")
    if value["schema_version"] != "unicom-spherical-probe-causal-screen-v3":
        raise ValueError("probe result version differs")

    source = value["source"]
    if (
        type(source) is not dict
        or tuple(source) != SOURCE_KEYS
        or type(source["git_revision"]) is not str
        or len(source["git_revision"]) != 40
        or not _sha256(source["script_sha256"])
        or not _sha256(source["module_sha256"])
        or not _sha256(source["training_sha256"])
        or not _sha256(source["inshop_sha256"])
    ):
        raise ValueError("probe source binding differs")
    model = value["model"]
    if (
        type(model) is not dict
        or tuple(model) != MODEL_KEYS
        or model["unicom_revision"] != UNICOM_REVISION
        or model["checkpoint_sha256"] != CHECKPOINT_SHA256
    ):
        raise ValueError("probe model binding differs")
    dataset = value["dataset"]
    if (
        type(dataset) is not dict
        or tuple(dataset) != DATASET_KEYS
        or not _sha256(dataset["partition_sha256"])
        or dataset["optimization_identity_count"] != 3200
        or dataset["optimization_image_count"] != 20_650
        or type(dataset["fitting_image_count"]) is not int
        or dataset["fitting_image_count"] != 14_330
        or type(dataset["validation_image_count"]) is not int
        or type(dataset["validation_class_count"]) is not int
        or type(dataset["singleton_class_count"]) is not int
        or dataset["validation_image_count"] != dataset["validation_class_count"]
        or dataset["validation_class_count"] + dataset["singleton_class_count"] != 3200
        or dataset["singleton_class_count"] != 12
        or dataset["validation_class_count"] != 3_188
        or dataset["excluded_same_series_count"] != 3_132
        or dataset["fitting_image_count"]
        + dataset["validation_image_count"]
        + dataset["excluded_same_series_count"]
        != dataset["optimization_image_count"]
        or type(dataset["represented_validation_count"]) is not int
        or type(dataset["unrepresented_validation_count"]) is not int
        or dataset["represented_validation_count"] != 2_162
        or dataset["unrepresented_validation_count"] != 1_026
        or dataset["represented_validation_count"]
        + dataset["unrepresented_validation_count"]
        != dataset["validation_image_count"]
    ):
        raise ValueError("probe dataset binding differs")

    target_norm = 0.01 * math.sqrt(768.0)
    protocol = value["protocol"]
    expected_protocol = {
        "holdout_fraction": 0.2,
        "holdout_seed": 0,
        "split_seed": 23_000,
        "fit_seeds": [0, 1, 2],
        "fit_steps": 512,
        "batch_size": 128,
        "batch_seed": 23_001,
        "mask_seed": 23_002,
        "evaluation_mask_seed": 23_003,
        "diagnostic_seed": 23_004,
        "gradient_seed": 23_005,
        "evaluation_mask_sets": 64,
        "shards": 8,
        "selected_features": 512,
        "margin": 0.25,
        "accuracy_margin": 0.0,
        "scale": 32.0,
        "optimizer": OPTIMIZER_ID,
        "row_norm": target_norm,
        "paired_t_critical_df63": 1.998340542520741,
        "paired_t_critical_df3187": 1.9607086212236648,
        "positive_mask_minimum": 48,
        "head_cosine_minimum": 0.8,
        "head_cosine_mean_minimum": 0.95,
        "gradient_median_cosine_maximum": 0.995,
    }
    if (
        type(protocol) is not dict
        or protocol != expected_protocol
        or tuple(protocol) != PROTOCOL_KEYS
    ):
        raise ValueError("probe protocol differs")

    class_mean = value["class_mean"]
    if (
        type(class_mean) is not dict
        or tuple(class_mean) != CLASS_MEAN_KEYS
        or not _sha256(class_mean["sha256"])
    ):
        raise ValueError("class_mean head schema differs")
    for name, head in (("class_mean", class_mean),):
        minimum = _finite_float(head["row_norm_min"], f"{name} row norm min", positive=True)
        maximum = _finite_float(head["row_norm_max"], f"{name} row norm max", positive=True)
        if maximum < minimum or not math.isclose(
            minimum, target_norm, rel_tol=2e-6, abs_tol=2e-7
        ) or not math.isclose(maximum, target_norm, rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError(f"{name} row norms differ")
    class_mean_metric = _metric(class_mean["validation"], "class_mean")
    expected_observations = dataset["validation_image_count"] * protocol["evaluation_mask_sets"]
    if class_mean_metric.observation_count != expected_observations:
        raise ValueError("probe observation count differs")

    probes = value["probes"]
    if type(probes) is not list or len(probes) != 3:
        raise ValueError("probe head list differs")
    probe_fits: dict[int, ProbeFit] = {}
    probe_metrics: dict[int, ProbeMetrics] = {}
    probe_gradients: dict[int, ProbeGradientMetrics] = {}
    for seed, probe in enumerate(probes):
        if (
            type(probe) is not dict
            or tuple(probe) != PROBE_KEYS
            or probe["fit_seed"] != seed
            or not _sha256(probe["sha256"])
        ):
            raise ValueError("probe head schema differs")
        minimum = _finite_float(probe["row_norm_min"], "probe row norm min", positive=True)
        maximum = _finite_float(probe["row_norm_max"], "probe row norm max", positive=True)
        if maximum < minimum or not math.isclose(
            minimum, target_norm, rel_tol=2e-6, abs_tol=2e-7
        ) or not math.isclose(maximum, target_norm, rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError("probe row norms differ")
        fit = probe["fit"]
        cosine = probe["head_cosine"]
        gradient = probe["gradient"]
        if type(fit) is not dict or tuple(fit) != FIT_KEYS or fit["steps"] != 512:
            raise ValueError("probe fit schema differs")
        if type(cosine) is not dict or tuple(cosine) != COSINE_KEYS:
            raise ValueError("probe head cosine schema differs")
        if type(gradient) is not dict or tuple(gradient) != GRADIENT_KEYS:
            raise ValueError("probe gradient schema differs")
        fit_values = tuple(
            _finite_float(value, f"probe seed {seed} fit value", positive=True)
            for value in (fit["initial_loss"], fit["final_loss"])
        )
        cosine_values = tuple(
            _finite_float(cosine[key], f"probe seed {seed} cosine")
            for key in COSINE_KEYS
        )
        if not all(-1.0 <= item <= 1.0 for item in cosine_values):
            raise ValueError("probe head cosine value differs")
        gradient_values = tuple(
            _finite_float(gradient[key], f"probe seed {seed} gradient")
            for key in GRADIENT_KEYS[:-1]
        )
        zero_gradient_row_count = gradient["zero_gradient_row_count"]
        if (
            type(zero_gradient_row_count) is not int
            or not 0 <= zero_gradient_row_count < 128
        ):
            raise ValueError("probe zero-gradient row count differs")
        probe_fits[seed] = ProbeFit(
            head=torch.empty(0),
            initial_loss=fit_values[0],
            final_loss=fit_values[1],
            steps=fit["steps"],
            row_cosine_min=cosine_values[0],
            row_cosine_p05=cosine_values[1],
            row_cosine_median=cosine_values[2],
            row_cosine_mean=cosine_values[3],
        )
        probe_metrics[seed] = _metric(probe["validation"], f"probe seed {seed}")
        if probe_metrics[seed].observation_count != expected_observations:
            raise ValueError("probe observation count differs")
        probe_gradients[seed] = ProbeGradientMetrics(
            *gradient_values, zero_gradient_row_count
        )

    decision = value["decision"]
    if (
        type(decision) is not dict
        or tuple(decision) != DECISION_KEYS
        or type(decision["per_seed"]) is not list
        or len(decision["per_seed"]) != 3
    ):
        raise ValueError("probe decision schema differs")
    for entry in decision["per_seed"]:
        if (
            type(entry) is not dict
            or tuple(entry) != SEED_DECISION_KEYS
            or type(entry["statistics"]) is not dict
            or tuple(entry["statistics"]) != STATISTIC_KEYS
            or type(entry["predicates"]) is not dict
            or tuple(entry["predicates"]) != PREDICATE_KEYS
        ):
            raise ValueError("probe decision schema differs")
    recomputed = probe_decision(
        class_mean=class_mean_metric,
        probe_fits=probe_fits,
        probe_metrics=probe_metrics,
        probe_gradients=probe_gradients,
    )
    expected_decision = {
        "status": recomputed.status,
        "per_seed": [
            {
                "fit_seed": seed,
                "statistics": vars(recomputed.per_seed_statistics[seed]),
                "predicates": recomputed.per_seed_predicates[seed],
            }
            for seed in range(3)
        ],
    }
    if decision != expected_decision:
        raise ValueError("probe decision differs")

    runtime = value["runtime"]
    if (
        type(runtime) is not dict
        or tuple(runtime) != RUNTIME_KEYS
        or any(type(runtime[key]) is not str or not runtime[key] for key in RUNTIME_KEYS[:4])
        or _finite_float(runtime["elapsed_seconds"], "probe elapsed seconds", positive=True) <= 0.0
        or type(runtime["peak_gpu_mib"]) is not int
        or runtime["peak_gpu_mib"] <= 0
    ):
        raise ValueError("probe runtime differs")


def write_result_atomic(value: dict[str, object], output: Path) -> None:
    validate_result(value)
    output = Path(output)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    payload = (json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    descriptor: int | None = None
    linked_identity: tuple[int, int] | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("probe result write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        persisted = strict_json_object(temporary.read_bytes())
        validate_result(persisted)
        if persisted != value:
            raise ValueError("persisted probe result differs")
        try:
            temporary_stat = temporary.stat()
            os.link(temporary, output)
            linked_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
            directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            temporary.unlink()
            if (
                not output.is_file()
                or output.is_symlink()
                or stat.S_IMODE(output.stat().st_mode) != 0o600
            ):
                raise ValueError("published probe result differs")
            validate_result(strict_json_object(output.read_bytes()))
        except Exception:
            if linked_identity is not None:
                try:
                    output_stat = output.lstat()
                except FileNotFoundError:
                    pass
                else:
                    if (
                        stat.S_ISREG(output_stat.st_mode)
                        and (output_stat.st_dev, output_stat.st_ino) == linked_identity
                    ):
                        output.unlink()
            raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(arguments)


def _authenticate_run(args: argparse.Namespace) -> ScreenInventory:
    if (
        type(args.batch_size) is not int
        or args.batch_size <= 0
        or type(args.workers) is not int
        or args.workers < 0
    ):
        raise ValueError("probe loader schedule differs")
    for path, name in (
        (args.unicom_checkout, "UNICOM checkout"),
        (args.dataset_root, "dataset root"),
        (args.output.parent, "output parent"),
    ):
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(f"{name} is not a real directory")
    if (
        not args.checkpoint.is_file()
        or args.checkpoint.is_symlink()
        or args.checkpoint.name != "FP16-ViT-L-14-336px.pt"
    ):
        raise FileNotFoundError("UNICOM checkpoint differs")
    partition = args.dataset_root / "Eval" / "list_eval_partition.txt"
    if not partition.is_file() or partition.is_symlink():
        raise FileNotFoundError("In-Shop partition differs")
    if _git_revision(args.unicom_checkout) != UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if _sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("UNICOM checkpoint SHA-256 differs")
    partition_sha256 = _sha256_file(partition)
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(row for row in records if row.split == "train")
    optimization, _query, _gallery, labels = identity_holdout(
        train_records, fraction=0.2, seed=0
    )
    split = split_probe_records(optimization, labels)
    revision, source_digests = _source_binding()
    result = ScreenInventory(
        partition_sha256=partition_sha256,
        optimization=optimization,
        labels=labels,
        split=split,
        source_revision=revision,
        script_sha256=source_digests["scripts/screen_unicom_spherical_probe.py"],
        module_sha256=source_digests["src/sfora/unicom_probe.py"],
        training_sha256=source_digests["src/sfora/unicom_training.py"],
        inshop_sha256=source_digests["src/sfora/unicom_inshop.py"],
    )
    _validate_inventory(result)
    return result


class _EvaluationDataset(Dataset[torch.Tensor]):
    def __init__(self, records: tuple[InshopRecord, ...], transform: object) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.records[index].image_path) as image:
            value = self.transform(image.convert("RGB"))  # type: ignore[operator]
        if type(value) is not torch.Tensor or value.dtype != torch.float32:
            raise ValueError("UNICOM evaluation transform differs")
        return value


def _load_official_model(checkout: Path, checkpoint: Path) -> tuple[object, object]:
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


def _encode_feature_sets(
    args: argparse.Namespace,
    fitting: tuple[InshopRecord, ...],
    validation: tuple[InshopRecord, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    model, transform = _load_official_model(args.unicom_checkout, args.checkpoint)
    model = model.cuda().eval()
    combined = fitting + validation
    loader = DataLoader(
        _EvaluationDataset(combined, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    chunks: list[torch.Tensor] = []
    for batch in loader:
        if type(batch) is not torch.Tensor or batch.dtype != torch.float32:
            raise ValueError("UNICOM feature batch differs")
        with torch.inference_mode():
            output = model(batch.cuda(non_blocking=False))
        if (
            type(output) is not torch.Tensor
            or output.dtype not in (torch.float16, torch.float32)
            or output.ndim != 2
            or output.shape[1] != 768
            or not torch.isfinite(output).all()
        ):
            raise ValueError("UNICOM feature output differs")
        chunks.append(output.float().detach().cpu())
    features = torch.cat(chunks, dim=0).contiguous()
    if features.shape != (len(combined), 768):
        raise ValueError("UNICOM feature inventory differs")
    fitting_features = features[: len(fitting)].contiguous().cuda()
    validation_features = features[len(fitting) :].contiguous().cuda()
    return fitting_features, validation_features


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _metric_payload(value: ProbeMetrics) -> dict[str, object]:
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


def _execute_screen(
    args: argparse.Namespace, inventory: ScreenInventory
) -> dict[str, object]:
    _validate_inventory(inventory)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA probe inventory differs")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    fitting_features, validation_features = _encode_feature_sets(
        args, inventory.split.fitting, inventory.split.validation
    )
    device = fitting_features.device
    fitting_labels = torch.tensor(
        [inventory.labels[row.label] for row in inventory.split.fitting],
        dtype=torch.int64,
        device=device,
    )
    validation_labels = torch.tensor(
        [inventory.labels[row.label] for row in inventory.split.validation],
        dtype=torch.int64,
        device=device,
    )
    class_mean = class_mean_head(
        fitting_features, fitting_labels, class_count=len(inventory.labels)
    )
    fits: dict[int, ProbeFit] = {}
    probe_metrics: dict[int, ProbeMetrics] = {}
    gradients: dict[int, ProbeGradientMetrics] = {}
    class_mean_metric: ProbeMetrics | None = None
    for seed in range(3):
        fit = fit_spherical_probe(
            fitting_features, fitting_labels, class_mean, fit_seed=seed
        )
        metrics = evaluate_probe_heads(
            validation_features,
            validation_labels,
            {"class_mean": class_mean, "spherical_probe": fit.head},
            validation_group_represented=inventory.split.validation_group_represented,
        )
        if class_mean_metric is None:
            class_mean_metric = metrics["class_mean"]
        elif metrics["class_mean"] != class_mean_metric:
            raise ValueError("class-mean validation replay differs")
        fits[seed] = fit
        probe_metrics[seed] = metrics["spherical_probe"]
        gradients[seed] = compare_probe_gradients(
            fitting_features,
            fitting_labels,
            {"class_mean": class_mean, "spherical_probe": fit.head},
            fit_seed=seed,
        )
    if class_mean_metric is None:
        raise AssertionError("unreachable empty probe seed schedule")
    decision = probe_decision(
        class_mean=class_mean_metric,
        probe_fits=fits,
        probe_metrics=probe_metrics,
        probe_gradients=gradients,
    )
    target_norm = 0.01 * math.sqrt(768.0)
    class_norms = torch.linalg.vector_norm(class_mean, dim=1)
    probes: list[dict[str, object]] = []
    for seed in range(3):
        fit = fits[seed]
        norms = torch.linalg.vector_norm(fit.head, dim=1)
        probes.append(
            {
                "fit_seed": seed,
                "sha256": _tensor_sha256(fit.head),
                "row_norm_min": float(norms.min()),
                "row_norm_max": float(norms.max()),
                "fit": {
                    "initial_loss": fit.initial_loss,
                    "final_loss": fit.final_loss,
                    "steps": fit.steps,
                },
                "head_cosine": {
                    "minimum": fit.row_cosine_min,
                    "p05": fit.row_cosine_p05,
                    "median": fit.row_cosine_median,
                    "mean": fit.row_cosine_mean,
                },
                "validation": _metric_payload(probe_metrics[seed]),
                "gradient": vars(gradients[seed]),
            }
        )
    result: dict[str, object] = {
        "schema_version": "unicom-spherical-probe-causal-screen-v3",
        "source": {
            "git_revision": inventory.source_revision,
            "script_sha256": inventory.script_sha256,
            "module_sha256": inventory.module_sha256,
            "training_sha256": inventory.training_sha256,
            "inshop_sha256": inventory.inshop_sha256,
        },
        "model": {
            "unicom_revision": UNICOM_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "dataset": {
            "partition_sha256": inventory.partition_sha256,
            "optimization_identity_count": len(inventory.labels),
            "optimization_image_count": len(inventory.optimization),
            "fitting_image_count": len(inventory.split.fitting),
            "validation_image_count": len(inventory.split.validation),
            "validation_class_count": inventory.split.validation_class_count,
            "singleton_class_count": inventory.split.singleton_class_count,
            "excluded_same_series_count": (
                inventory.split.excluded_same_series_count
            ),
            "represented_validation_count": sum(
                inventory.split.validation_group_represented
            ),
            "unrepresented_validation_count": sum(
                not value for value in inventory.split.validation_group_represented
            ),
        },
        "protocol": {
            "holdout_fraction": 0.2,
            "holdout_seed": 0,
            "split_seed": 23_000,
            "fit_seeds": [0, 1, 2],
            "fit_steps": 512,
            "batch_size": 128,
            "batch_seed": 23_001,
            "mask_seed": 23_002,
            "evaluation_mask_seed": 23_003,
            "diagnostic_seed": 23_004,
            "gradient_seed": 23_005,
            "evaluation_mask_sets": 64,
            "shards": 8,
            "selected_features": 512,
            "margin": 0.25,
            "accuracy_margin": 0.0,
            "scale": 32.0,
            "optimizer": OPTIMIZER_ID,
            "row_norm": target_norm,
            "paired_t_critical_df63": 1.998340542520741,
            "paired_t_critical_df3187": 1.9607086212236648,
            "positive_mask_minimum": 48,
            "head_cosine_minimum": 0.8,
            "head_cosine_mean_minimum": 0.95,
            "gradient_median_cosine_maximum": 0.995,
        },
        "class_mean": {
            "sha256": _tensor_sha256(class_mean),
            "row_norm_min": float(class_norms.min()),
            "row_norm_max": float(class_norms.max()),
            "validation": _metric_payload(class_mean_metric),
        },
        "probes": probes,
        "decision": {
            "status": decision.status,
            "per_seed": [
                {
                    "fit_seed": seed,
                    "statistics": vars(decision.per_seed_statistics[seed]),
                    "predicates": decision.per_seed_predicates[seed],
                }
                for seed in range(3)
            ],
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "cuda": str(torch.version.cuda),
            "elapsed_seconds": float(time.perf_counter() - started),
            "peak_gpu_mib": math.ceil(torch.cuda.max_memory_allocated() / 1024**2),
        },
    }
    validate_result(result)
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        inventory = _authenticate_run(args)
        return _execute_screen(args, inventory)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        result = run(args)
        write_result_atomic(result, args.output)
    except Exception as error:
        print(f"probe screen failed: {error}", file=sys.stderr)
        return 2
    print(f"probe screen complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
