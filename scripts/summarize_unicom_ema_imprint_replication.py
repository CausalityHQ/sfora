#!/usr/bin/env python3
"""Summarize the preregistered six-seed UniCOM imprint replication."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.util
import json
import math
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REGISTERED_SEEDS = (1, 2, 3, 4, 5, 6)
SELECTED_CELL = "imprinted_raw"
PAIR_SCHEMA_V1 = "unicom-ema-imprint-replication-pair-v1"
PAIR_SCHEMA_V2 = "unicom-ema-imprint-replication-pair-v2"
SUMMARY_SCHEMA = "unicom-ema-imprint-replication-summary-v2"
SELECTION_REPORT_PATH = "reports/generated/unicom_ema_imprint_factorial_88604a4_seed0.json"
SELECTION_REPORT_SHA256 = "c0666a68e70990115d80e8dc06a9f94efe83156a3fddd50f36bdbf2b3b8cd217"
SELECTION_RECORDING_COMMIT = "81f3f48c374d14b5a91bbeba7a1fec2fb0a4a2d6"
HISTORICAL_SEED1_REPORT_PATH = (
    "reports/generated/unicom_ema_imprint_replication_c83cd96_seed1.json"
)
HISTORICAL_SEED1_REPORT_SHA256 = (
    "0cfb888fdbc0e409048943a8d6e47635e571dec5830b86260e0730d80ebf4ab8"
)
HISTORICAL_SEED1_RECORDING_COMMIT = "5a08b95266c7d40d57ec7fd747999969147a04e6"
SELECTION_AUTHORITY = {
    "path": SELECTION_REPORT_PATH,
    "sha256": SELECTION_REPORT_SHA256,
    "recording_commit": SELECTION_RECORDING_COMMIT,
    "selected_cell": SELECTED_CELL,
    "decision": "PROMOTE",
}
STUDENT_T_CRITICAL_TWO_SIDED_95_DF5 = 2.5705818356363146
RECALL_AT_1_DELTA_GUARD = -0.00125
_TOP_LEVEL_KEYS = (
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
_ARM_KEYS = (
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
_ARM_V2_KEYS = _ARM_KEYS + (
    "optimizer_steps_per_epoch",
    "initialization_seconds",
    "initialization_receipt_sha256",
    "post_initialization_rng_sha256",
)
_METRIC_KEYS = ("epoch", "map_at_r", "recall_at_1")
_PROFILE_KEYS = ("step_wall_seconds", "fusible_non_backbone_seconds")
_LATENCY_KEYS = (
    "warmup_repetitions",
    "measured_repetitions",
    "batch_size",
    "milliseconds_per_image",
)
_SUMMARY_KEYS = (
    "schema_version",
    "training_seeds",
    "selected_cell",
    "selection_authority",
    "reports",
    "initialization_evidence",
    "map_deltas",
    "mean_map_delta",
    "map_delta_sample_standard_deviation",
    "student_t_critical_two_sided_95_df5",
    "map_delta_paired_student_t_95_interval",
    "exact_two_sided_sign_p_value",
    "recall_at_1_deltas",
    "recall_at_1_delta_guard",
    "all_map_deltas_positive",
    "nondegenerate_training_seed_variation",
    "all_recall_at_1_deltas_above_guard",
    "quality_claim_supported",
    "first_quality_epochs",
    "costs",
    "fixed_epoch_pareto_nondominated",
    "all_first_quality_epochs_noninferior",
    "all_future_iso_quality_profiled_compute_noninferior",
    "per_seed_resource_noninferior",
    "claim_supported",
)
_COST_KEYS = (
    "training_seconds",
    "first_quality_epochs",
    "fixed_epoch_profiled_compute_seconds",
    "fixed_epoch_profiled_compute_overhead_seconds",
    "iso_quality_profiled_compute_seconds",
    "peak_gpu_mib",
    "inference_latency_protocol",
    "inference_latency_ms_per_image",
    "checkpoint_storage_bytes",
    "deployment_storage_bytes",
    "profile_fusible_non_backbone_fraction",
    "kernel_profile_threshold",
    "kernel_eligible",
    "historical_cost_limitations",
)


def _exact_object(value: object, keys: tuple[str, ...], name: str) -> dict[str, Any]:
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError(f"{name} schema differs")
    return value


def _exact_ordered_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return tuple(left) == tuple(right) and all(
            _exact_ordered_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_ordered_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _validate_replication_pair(value: object) -> None:
    path = Path(__file__).resolve().with_name("evaluate_unicom_ema_imprint_replication.py")
    name = "_unicom_ema_imprint_replication_validator"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("replication pair validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.validate_replication_pair(value)
    finally:
        sys.modules.pop(name, None)


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} cost must be positive")
    return value


def _metric(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise TypeError(f"{name} checkpoint digest differs")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} checkpoint digest differs") from exc
    if value != value.lower():
        raise ValueError(f"{name} checkpoint digest differs")
    return value


def _validate_selection_authority(value: object) -> dict[str, str]:
    authority = _exact_object(
        value,
        ("path", "sha256", "recording_commit", "selected_cell", "decision"),
        "selection authority",
    )
    if not _exact_ordered_equal(authority, SELECTION_AUTHORITY):
        raise ValueError("selection authority differs")
    return authority


def load_selection_authority(path: Path) -> dict[str, str]:
    """Authenticate the immutable seed-0 selection report against Git and worktree bytes."""
    if not isinstance(path, Path):
        raise TypeError("selection report must be a Path")
    repository = Path(__file__).resolve().parents[1]
    expected = repository / SELECTION_REPORT_PATH
    if (
        Path(os.path.abspath(path)) != expected
        or path.is_symlink()
        or not expected.is_file()
        or expected.is_symlink()
    ):
        raise ValueError("selection authority path differs")
    payload = expected.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SELECTION_REPORT_SHA256:
        raise ValueError("selection authority SHA-256 differs")
    report = strict_json_object(payload)
    validator_path = Path(__file__).resolve().with_name(
        "evaluate_unicom_ema_imprint_factorial.py"
    )
    name = "_unicom_ema_imprint_factorial_validator"
    spec = importlib.util.spec_from_file_location(name, validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("selection authority validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        module.validate_factorial_report(report)
    finally:
        sys.modules.pop(name, None)
    gate = report["gate"]
    if (
        gate.get("selected_cell") != SELECTED_CELL
        or gate.get("decision") != "PROMOTE"
        or gate.get("promoted") is not True
    ):
        raise ValueError("selection authority gate differs")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SELECTION_RECORDING_COMMIT, "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("selection authority recording commit is not an ancestor")
    recorded = subprocess.run(
        ["git", "show", f"{SELECTION_RECORDING_COMMIT}:{SELECTION_REPORT_PATH}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if recorded != payload:
        raise ValueError("selection authority Git blob differs")
    return dict(SELECTION_AUTHORITY)


def load_historical_seed1_report(path: Path) -> dict[str, Any]:
    """Load the irreplaceable seed-1 report only from its registered Git-backed path."""
    if not isinstance(path, Path):
        raise TypeError("historical seed-1 report must be a Path")
    repository = Path(__file__).resolve().parents[1]
    expected = repository / HISTORICAL_SEED1_REPORT_PATH
    if (
        Path(os.path.abspath(path)) != expected
        or path.is_symlink()
        or not expected.is_file()
        or expected.is_symlink()
    ):
        raise ValueError("historical seed-1 report path differs")
    payload = expected.read_bytes()
    if hashlib.sha256(payload).hexdigest() != HISTORICAL_SEED1_REPORT_SHA256:
        raise ValueError("historical seed-1 report SHA-256 differs")
    recorded_value = strict_json_object(payload)
    _validate_replication_pair(recorded_value)
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            HISTORICAL_SEED1_RECORDING_COMMIT,
            "HEAD",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("historical seed-1 recording commit is not an ancestor")
    recorded_payload = subprocess.run(
        [
            "git",
            "show",
            f"{HISTORICAL_SEED1_RECORDING_COMMIT}:{HISTORICAL_SEED1_REPORT_PATH}",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if recorded_payload != payload:
        raise ValueError("historical seed-1 Git blob differs")
    return recorded_value


def authenticate_historical_seed1(value: object) -> None:
    """Bind the irreplaceable seed-1 semantic object to its exact Git-backed bytes."""
    repository = Path(__file__).resolve().parents[1]
    recorded_value = load_historical_seed1_report(
        repository / HISTORICAL_SEED1_REPORT_PATH
    )
    if not _exact_ordered_equal(value, recorded_value):
        raise ValueError("historical seed-1 report differs")


def _arm(value: object, name: str, *, future: bool) -> dict[str, Any]:
    arm = _exact_object(value, _ARM_V2_KEYS if future else _ARM_KEYS, name)
    hashes = arm["checkpoint_sha256_by_epoch"]
    if type(hashes) is not list or len(hashes) != 4:
        raise ValueError(f"{name} checkpoint evidence differs")
    for digest in hashes:
        _sha256(digest, name)
    history_hashes = arm["training_history_sha256_by_epoch"]
    if type(history_hashes) is not list or len(history_hashes) != 4:
        raise ValueError(f"{name} training history evidence differs")
    for digest in history_hashes:
        _sha256(digest, name)
    rows = arm["epoch_metrics"]
    if type(rows) is not list or len(rows) != 4:
        raise ValueError(f"{name} epoch metrics differ")
    for expected_epoch, raw_row in zip((4, 8, 12, 16), rows, strict=True):
        row = _exact_object(raw_row, _METRIC_KEYS, f"{name} epoch metric")
        if type(row["epoch"]) is not int or row["epoch"] != expected_epoch:
            raise ValueError(f"{name} epoch metrics differ")
        _metric(row["map_at_r"], f"{name}.map_at_r")
        _metric(row["recall_at_1"], f"{name}.recall_at_1")
    _finite_float(arm["training_seconds"], f"{name}.training_seconds cost", positive=True)
    peak = arm["peak_gpu_mib"]
    if type(peak) is not int or peak <= 0:
        raise ValueError(f"{name}.peak_gpu_mib cost must be a positive integer")
    storage = arm["checkpoint_storage_bytes"]
    if type(storage) is not int or storage <= 0:
        raise ValueError(f"{name}.checkpoint_storage_bytes cost must be a positive integer")
    deployment = arm["deployment_storage_bytes"]
    if type(deployment) is not int or deployment <= 0:
        raise ValueError(f"{name}.deployment_storage_bytes cost must be a positive integer")
    _sha256(arm["measurement_receipt_sha256"], f"{name} measurement receipt")
    profile = _exact_object(arm["profile"], _PROFILE_KEYS, f"{name} profile")
    step = _finite_float(profile["step_wall_seconds"], f"{name} step cost", positive=True)
    fusible = _finite_float(
        profile["fusible_non_backbone_seconds"], f"{name} fusible cost"
    )
    if fusible < 0.0 or fusible > step:
        raise ValueError(f"{name} profile cost differs")
    if future:
        steps = arm["optimizer_steps_per_epoch"]
        if type(steps) is not int or steps <= 0:
            raise ValueError(f"{name} optimizer steps differ")
        _finite_float(
            arm["initialization_seconds"],
            f"{name}.initialization_seconds",
            positive=True,
        )
        _sha256(arm["initialization_receipt_sha256"], f"{name} initialization receipt")
        _sha256(arm["post_initialization_rng_sha256"], f"{name} post-init RNG")
    return arm


def _latency(value: object) -> dict[str, Any]:
    latency = _exact_object(value, _LATENCY_KEYS, "inference latency")
    expected = {"warmup_repetitions": 10, "measured_repetitions": 50, "batch_size": 128}
    for key, expected_value in expected.items():
        if type(latency[key]) is not int or latency[key] != expected_value:
            raise ValueError("inference latency protocol differs")
    _finite_float(latency["milliseconds_per_image"], "inference latency cost", positive=True)
    return latency


def _first_epoch_reaching(arm: dict[str, Any], target: float) -> int | None:
    for row in arm["epoch_metrics"]:
        if row["map_at_r"] >= target:
            return row["epoch"]
    return None


def summarize_replications(
    reports: list[object], *, selection_authority: object
) -> dict[str, object]:
    """Validate and summarize exactly the six preregistered paired reports."""
    if type(reports) is not list or len(reports) != len(REGISTERED_SEEDS):
        raise ValueError("training seeds differ")
    authenticate_historical_seed1(reports[0])
    authority = _validate_selection_authority(selection_authority)

    map_deltas: list[float] = []
    recall_deltas: list[float] = []
    checkpoint_hashes: list[str] = []
    validated: list[dict[str, Any]] = []
    first_quality_epochs: list[dict[str, object]] = []
    initialization_evidence: list[dict[str, object]] = []
    training_costs: list[dict[str, object]] = []
    fixed_epoch_costs: list[dict[str, object]] = []
    fixed_epoch_overheads: list[dict[str, object]] = []
    iso_quality_costs: list[dict[str, object]] = []
    peak_costs: list[dict[str, object]] = []
    checkpoint_costs: list[dict[str, object]] = []
    deployment_costs: list[dict[str, object]] = []
    latency_costs: list[dict[str, object]] = []
    profile_ratios: list[dict[str, object]] = []
    expected_protocol: dict[str, object] | None = None
    expected_future_trainer_sha256: str | None = None
    expected_future_optimizer_steps: int | None = None
    expected_query_count: int | None = None
    history_hashes: list[str] = []
    measurement_hashes: list[str] = []
    initialization_hashes: list[str] = []
    initialization_rng_hashes: list[str] = []
    for expected_seed, raw_report in zip(REGISTERED_SEEDS, reports, strict=True):
        report = _exact_object(raw_report, _TOP_LEVEL_KEYS, "replication pair")
        _validate_replication_pair(report)
        expected_schema = PAIR_SCHEMA_V1 if expected_seed == 1 else PAIR_SCHEMA_V2
        if report["schema_version"] != expected_schema:
            raise ValueError("replication pair schema version differs")
        if type(report["seed"]) is not int or report["seed"] != expected_seed:
            raise ValueError("training seeds differ")
        if report["selected_cell"] != SELECTED_CELL:
            raise ValueError("selected cell differs")
        if report["registered_epochs"] != [4, 8, 12, 16] or any(
            type(epoch) is not int for epoch in report["registered_epochs"]
        ):
            raise ValueError("registered epochs differ")
        future = expected_seed != 1
        random_arm = _arm(report["random_raw"], "random_raw", future=future)
        imprinted_arm = _arm(report["imprinted_raw"], "imprinted_raw", future=future)
        if future and (
            random_arm["post_initialization_rng_sha256"]
            != imprinted_arm["post_initialization_rng_sha256"]
        ):
            raise ValueError("post-initialization RNG differs")
        normalized_protocol = dict(report["random_training_protocol"])
        normalized_protocol.pop("seed")
        normalized_protocol.pop("classifier_init")
        normalized_protocol.pop("trainer_sha256")
        if expected_protocol is None:
            expected_protocol = normalized_protocol
        elif normalized_protocol != expected_protocol:
            raise ValueError("paired training protocol differs across seeds")
        if future:
            trainer_sha256 = report["random_training_protocol"]["trainer_sha256"]
            if expected_future_trainer_sha256 is None:
                expected_future_trainer_sha256 = trainer_sha256
            elif trainer_sha256 != expected_future_trainer_sha256:
                raise ValueError("prospective trainer differs across seeds")
            optimizer_steps = random_arm["optimizer_steps_per_epoch"]
            if expected_future_optimizer_steps is None:
                expected_future_optimizer_steps = optimizer_steps
            elif optimizer_steps != expected_future_optimizer_steps:
                raise ValueError("optimizer steps differ across seeds")
        query_count = len(report["evidence"]["random_raw"][0]["top1_correct"])
        if expected_query_count is None:
            expected_query_count = query_count
        elif query_count != expected_query_count:
            raise ValueError("paired query population differs across seeds")
        latency = _latency(report["inference_latency"])
        checkpoint_hashes.extend(
            (
                *random_arm["checkpoint_sha256_by_epoch"],
                *imprinted_arm["checkpoint_sha256_by_epoch"],
            )
        )
        history_hashes.extend(
            (
                *random_arm["training_history_sha256_by_epoch"],
                *imprinted_arm["training_history_sha256_by_epoch"],
            )
        )
        measurement_hashes.extend(
            (
                random_arm["measurement_receipt_sha256"],
                imprinted_arm["measurement_receipt_sha256"],
            )
        )
        random_endpoint = random_arm["epoch_metrics"][-1]
        imprinted_endpoint = imprinted_arm["epoch_metrics"][-1]
        map_deltas.append(imprinted_endpoint["map_at_r"] - random_endpoint["map_at_r"])
        recall_deltas.append(
            imprinted_endpoint["recall_at_1"] - random_endpoint["recall_at_1"]
        )
        target = random_endpoint["map_at_r"]
        random_epoch = _first_epoch_reaching(random_arm, target)
        imprinted_epoch = _first_epoch_reaching(imprinted_arm, target)
        speedup = (
            None
            if random_epoch is None or imprinted_epoch is None
            else float(random_epoch / imprinted_epoch)
        )
        first_quality_epochs.append(
            {
                "seed": expected_seed,
                "random_raw": random_epoch,
                "imprinted_raw": imprinted_epoch,
                "speedup": speedup,
            }
        )
        if not future:
            initialization_evidence.append(
                {
                    "seed": expected_seed,
                    "status": "historical_initialization_receipt_unavailable",
                    "random_raw": None,
                    "imprinted_raw": None,
                    "post_initialization_rng_equal": None,
                }
            )
            fixed_epoch_costs.append(
                {"seed": expected_seed, "random_raw": None, "imprinted_raw": None}
            )
            fixed_epoch_overheads.append(
                {"seed": expected_seed, "imprinted_minus_random": None}
            )
            iso_quality_costs.append(
                {"seed": expected_seed, "random_raw": None, "imprinted_raw": None}
            )
        else:
            initialization_hashes.extend(
                (
                    random_arm["initialization_receipt_sha256"],
                    imprinted_arm["initialization_receipt_sha256"],
                )
            )
            initialization_rng_hashes.append(
                random_arm["post_initialization_rng_sha256"]
            )
            initialization_evidence.append(
                {
                    "seed": expected_seed,
                    "status": "prospective_authenticated",
                    "random_raw": {
                        "initialization_receipt_sha256": random_arm[
                            "initialization_receipt_sha256"
                        ],
                        "optimizer_steps_per_epoch": random_arm[
                            "optimizer_steps_per_epoch"
                        ],
                        "initialization_seconds": random_arm["initialization_seconds"],
                        "post_initialization_rng_sha256": random_arm[
                            "post_initialization_rng_sha256"
                        ],
                    },
                    "imprinted_raw": {
                        "initialization_receipt_sha256": imprinted_arm[
                            "initialization_receipt_sha256"
                        ],
                        "optimizer_steps_per_epoch": imprinted_arm[
                            "optimizer_steps_per_epoch"
                        ],
                        "initialization_seconds": imprinted_arm[
                            "initialization_seconds"
                        ],
                        "post_initialization_rng_sha256": imprinted_arm[
                            "post_initialization_rng_sha256"
                        ],
                    },
                    "post_initialization_rng_equal": True,
                }
            )
            random_fixed = (
                16
                * random_arm["optimizer_steps_per_epoch"]
                * random_arm["profile"]["step_wall_seconds"]
                + random_arm["initialization_seconds"]
            )
            imprinted_fixed = (
                16
                * imprinted_arm["optimizer_steps_per_epoch"]
                * imprinted_arm["profile"]["step_wall_seconds"]
                + imprinted_arm["initialization_seconds"]
            )
            fixed_epoch_costs.append(
                {
                    "seed": expected_seed,
                    "random_raw": random_fixed,
                    "imprinted_raw": imprinted_fixed,
                }
            )
            fixed_epoch_overheads.append(
                {
                    "seed": expected_seed,
                    "imprinted_minus_random": imprinted_fixed - random_fixed,
                }
            )
            random_iso = (
                None
                if random_epoch is None
                else random_epoch
                * random_arm["optimizer_steps_per_epoch"]
                * random_arm["profile"]["step_wall_seconds"]
                + random_arm["initialization_seconds"]
            )
            imprinted_iso = (
                None
                if imprinted_epoch is None
                else imprinted_epoch
                * imprinted_arm["optimizer_steps_per_epoch"]
                * imprinted_arm["profile"]["step_wall_seconds"]
                + imprinted_arm["initialization_seconds"]
            )
            iso_quality_costs.append(
                {
                    "seed": expected_seed,
                    "random_raw": random_iso,
                    "imprinted_raw": imprinted_iso,
                }
            )
        training_costs.append(
            {
                "seed": expected_seed,
                "random_raw": random_arm["training_seconds"],
                "imprinted_raw": imprinted_arm["training_seconds"],
            }
        )
        peak_costs.append(
            {
                "seed": expected_seed,
                "random_raw": random_arm["peak_gpu_mib"],
                "imprinted_raw": imprinted_arm["peak_gpu_mib"],
            }
        )
        checkpoint_costs.append(
            {
                "seed": expected_seed,
                "random_raw": random_arm["checkpoint_storage_bytes"],
                "imprinted_raw": imprinted_arm["checkpoint_storage_bytes"],
            }
        )
        deployment_costs.append(
            {
                "seed": expected_seed,
                "random_raw": random_arm["deployment_storage_bytes"],
                "imprinted_raw": imprinted_arm["deployment_storage_bytes"],
            }
        )
        latency_costs.append(
            {"seed": expected_seed, "milliseconds_per_image": latency["milliseconds_per_image"]}
        )
        profile_ratios.append(
            {
                "seed": expected_seed,
                "random_raw": random_arm["profile"]["fusible_non_backbone_seconds"]
                / random_arm["profile"]["step_wall_seconds"],
                "imprinted_raw": imprinted_arm["profile"]["fusible_non_backbone_seconds"]
                / imprinted_arm["profile"]["step_wall_seconds"],
            }
        )
        validated.append(report)

    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("checkpoint evidence is reused")
    if len(set(history_hashes)) != len(history_hashes):
        raise ValueError("training history evidence is reused")
    if len(set(measurement_hashes)) != len(measurement_hashes):
        raise ValueError("training measurement evidence is reused")
    if len(set(initialization_hashes)) != len(initialization_hashes):
        raise ValueError("initialization receipt evidence is reused")
    if len(set(initialization_rng_hashes)) != len(initialization_rng_hashes):
        raise ValueError("post-initialization RNG evidence is reused")

    mean_delta = float(np.mean(np.asarray(map_deltas, dtype=np.float64)))
    sample_sd = float(np.std(np.asarray(map_deltas, dtype=np.float64), ddof=1))
    half_width = STUDENT_T_CRITICAL_TWO_SIDED_95_DF5 * sample_sd / math.sqrt(6.0)
    interval = [mean_delta - half_width, mean_delta + half_width]
    positive_count = sum(delta > 0.0 for delta in map_deltas)
    negative_count = sum(delta < 0.0 for delta in map_deltas)
    nonzero_count = positive_count + negative_count
    if nonzero_count == 0:
        sign_p = 1.0
    else:
        tail = min(positive_count, negative_count)
        sign_p = min(
            1.0,
            2.0
            * sum(math.comb(nonzero_count, k) for k in range(tail + 1))
            / (2**nonzero_count),
        )

    all_positive = all(delta > 0.0 for delta in map_deltas)
    recall_guard = all(delta >= RECALL_AT_1_DELTA_GUARD for delta in recall_deltas)
    nondegenerate = sample_sd > 0.0
    quality_supported = (
        all_positive
        and nondegenerate
        and interval[0] > 0.0
        and sign_p <= 0.05
        and recall_guard
    )
    kernel_eligible = any(
        max(row["random_raw"], row["imprinted_raw"]) >= 0.1 for row in profile_ratios
    )
    costs = {
        "training_seconds": training_costs,
        "first_quality_epochs": first_quality_epochs,
        "fixed_epoch_profiled_compute_seconds": fixed_epoch_costs,
        "fixed_epoch_profiled_compute_overhead_seconds": fixed_epoch_overheads,
        "iso_quality_profiled_compute_seconds": iso_quality_costs,
        "peak_gpu_mib": peak_costs,
        "inference_latency_protocol": {
            "warmup_repetitions": 10,
            "measured_repetitions": 50,
            "batch_size": 128,
        },
        "inference_latency_ms_per_image": latency_costs,
        "checkpoint_storage_bytes": checkpoint_costs,
        "deployment_storage_bytes": deployment_costs,
        "profile_fusible_non_backbone_fraction": profile_ratios,
        "kernel_profile_threshold": 0.1,
        "kernel_eligible": kernel_eligible,
        "historical_cost_limitations": [
            {
                "seed": 1,
                "status": "historical_initialization_receipt_unavailable",
            }
        ],
    }
    if tuple(costs) != _COST_KEYS:
        raise RuntimeError("replication summary cost construction differs")
    all_first_quality_noninferior = all(
        row["imprinted_raw"] is not None
        and row["random_raw"] is not None
        and row["imprinted_raw"] <= row["random_raw"]
        for row in first_quality_epochs
    )
    future_iso_noninferior = all(
        row["imprinted_raw"] is not None
        and row["random_raw"] is not None
        and row["imprinted_raw"] <= row["random_raw"]
        for row in iso_quality_costs
        if row["seed"] != 1
    )
    resource_noninferior = all(
        peak["imprinted_raw"] <= peak["random_raw"]
        and checkpoint["imprinted_raw"] <= checkpoint["random_raw"]
        and deployment["imprinted_raw"] == deployment["random_raw"]
        for peak, checkpoint, deployment in zip(
            peak_costs, checkpoint_costs, deployment_costs, strict=True
        )
    )
    pareto_nondominated = all(
        not (
            delta <= 0.0
            and peak["random_raw"] <= peak["imprinted_raw"]
            and checkpoint["random_raw"] <= checkpoint["imprinted_raw"]
            and deployment["random_raw"] <= deployment["imprinted_raw"]
            and (
                fixed["random_raw"] is None
                or fixed["random_raw"] <= fixed["imprinted_raw"]
            )
        )
        for delta, fixed, peak, checkpoint, deployment in zip(
            map_deltas,
            fixed_epoch_costs,
            peak_costs,
            checkpoint_costs,
            deployment_costs,
            strict=True,
        )
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "training_seeds": list(REGISTERED_SEEDS),
        "selected_cell": SELECTED_CELL,
        "selection_authority": dict(authority),
        "reports": validated,
        "initialization_evidence": initialization_evidence,
        "map_deltas": map_deltas,
        "mean_map_delta": mean_delta,
        "map_delta_sample_standard_deviation": sample_sd,
        "student_t_critical_two_sided_95_df5": STUDENT_T_CRITICAL_TWO_SIDED_95_DF5,
        "map_delta_paired_student_t_95_interval": interval,
        "exact_two_sided_sign_p_value": sign_p,
        "recall_at_1_deltas": recall_deltas,
        "recall_at_1_delta_guard": RECALL_AT_1_DELTA_GUARD,
        "all_map_deltas_positive": all_positive,
        "nondegenerate_training_seed_variation": nondegenerate,
        "all_recall_at_1_deltas_above_guard": recall_guard,
        "quality_claim_supported": quality_supported,
        "first_quality_epochs": first_quality_epochs,
        "costs": costs,
        "fixed_epoch_pareto_nondominated": pareto_nondominated,
        "all_first_quality_epochs_noninferior": all_first_quality_noninferior,
        "all_future_iso_quality_profiled_compute_noninferior": future_iso_noninferior,
        "per_seed_resource_noninferior": resource_noninferior,
        "claim_supported": (
            quality_supported
            and all_first_quality_noninferior
            and future_iso_noninferior
            and resource_noninferior
        ),
    }
    return summary


def validate_summary(value: object) -> None:
    summary = _exact_object(value, _SUMMARY_KEYS, "replication summary")
    reports = summary["reports"]
    if type(reports) is not list:
        raise ValueError("replication summary reports differ")
    expected = summarize_replications(
        reports, selection_authority=summary["selection_authority"]
    )
    if not _exact_ordered_equal(summary, expected):
        raise ValueError("replication summary derived values differ")


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


def write_summary_atomic(summary: dict[str, object], output: Path) -> None:
    validate_summary(summary)
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode()
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
            raise RuntimeError("persisted summary bytes differ")
        validate_summary(strict_json_object(persisted_payload))
        _link_fd_noreplace(descriptor, output, directory_descriptor)
        published = True
        os.fsync(directory_descriptor)
        output_info = output.lstat()
        if (output_info.st_dev, output_info.st_ino) != owned:
            raise RuntimeError("published summary inode differs")
        published_payload = _read_descriptor(descriptor)
        if published_payload != payload:
            raise RuntimeError("published summary bytes differ")
        validate_summary(strict_json_object(published_payload))
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


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs=6, type=Path)
    parser.add_argument("--selection-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(arguments)
    reports = [
        load_historical_seed1_report(args.reports[0]),
        *(strict_json_object(path.read_bytes()) for path in args.reports[1:]),
    ]
    authority = load_selection_authority(args.selection_report)
    summary = summarize_replications(reports, selection_authority=authority)
    write_summary_atomic(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
