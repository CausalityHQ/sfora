#!/usr/bin/env python3
"""Evaluate the frozen UniCOM step-EMA × imprinted-head factorial."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

REGISTERED_CELLS = ("random_raw", "random_ema", "imprinted_raw", "imprinted_ema")
REGISTERED_CANDIDATES = ("random_ema", "imprinted_raw", "imprinted_ema")
REGISTERED_EPOCHS = (4, 8, 12, 16)
ARCHIVED_MAP_AT_R = 0.8716329439260202
ARCHIVED_RECALL_AT_1 = 0.972396486825596
REGISTERED_LEGACY_CONTROL_SHA256 = (
    "9af6f811eb162a4054d9c86bc117c2e261951750da207670841e058521c077fa"
)
REGISTERED_RANDOM_E16_CHECKPOINT_SHA256 = (
    "df53194c44e4131a6d89832639cedcd10b78d27b18b3c2a802997e12abd85e55"
)
REGISTERED_RANDOM_HISTORY_SHA256 = (
    "de62f014367fdc492f84dc96d05f02d6c29bd711d46813e642181c476c37ef96"
)
ARCHIVED_FTRA_REPORT_SHA256 = "c8bb65dcff33b09c40f602f1d243336d83a8d29f93c3c8c2e75236fed375140a"
ARCHIVED_RAW_CHECKPOINT_SHA256 = "210d0113b40d2a5ef3bb836f818ed2d632d046f3e818b1bb5049e25ba845f0a5"
ARCHIVED_TRAINER_SHA256 = "b2cfdaed33d46ec445141bb40b1a3f28aed0d3ca859101843ddf825866640bb1"
ARCHIVED_INITIAL_CHECKPOINT_SHA256 = (
    "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
)
ARCHIVED_PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
ARCHIVED_RECOMPUTATION_REPORT = "reports/unicom_archived_raw_current_evaluator_2026-08-14.json"
ARCHIVED_RECOMPUTATION_REPORT_SHA256 = (
    "b64bd3df5d9abe4ee7d1a1f4332148e1cc8adfc40ba2421e5fb5d2b2be6cdfe7"
)
ARCHIVED_RECOMPUTATION_EVALUATOR_REVISION = "d4f5f3e5029b2f29fe11d725545a56a3cf904b63"
ARCHIVED_RECOMPUTATION_EVALUATOR_SHA256 = (
    "2adf842db58574fd6c8487743f2dab341881aad1d5ca5224e647022a4f168a4d"
)
ARCHIVED_RECOMPUTATION_INTERPRETATION = (
    "the evaluator reproduces the archived checkpoint; the new raw-control difference is a "
    "training-lineage difference"
)
CONTROL_REPAIR_REASON = (
    "cross-lineage-absolute-gate-invalid-after-nondeterministic-training-trajectory"
)
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


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    return value


def _registered_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], Mapping[str, object]]:
    if type(rows) is not list or len(rows) != len(REGISTERED_CELLS) * len(REGISTERED_EPOCHS):
        raise ValueError("factorial rows differ")
    expected = tuple((cell, epoch) for cell in REGISTERED_CELLS for epoch in REGISTERED_EPOCHS)
    observed = tuple((row.get("cell"), row.get("epoch")) for row in rows)
    if observed != expected:
        raise ValueError("factorial row order differs")
    result = {}
    for key, row in zip(expected, rows, strict=True):
        metrics = row.get("metrics")
        if type(metrics) is not dict:
            raise ValueError("factorial row metrics differ")
        _finite_float(metrics.get("map_at_r"), name="mAP@R")
        _finite_float(metrics.get("recall_at_1"), name="Recall@1")
        result[key] = row
    return result


def select_candidate(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    by_key = _registered_rows(rows)
    candidates = [by_key[(cell, 16)] for cell in REGISTERED_CANDIDATES]
    return max(
        candidates,
        key=lambda row: (
            row["metrics"]["map_at_r"],
            row["metrics"]["recall_at_1"],
            -REGISTERED_CANDIDATES.index(row["cell"]),
        ),
    )


def time_to_quality(
    rows: Sequence[Mapping[str, object]], *, cell: str, target: float
) -> dict[str, object]:
    by_key = _registered_rows(rows)
    if cell not in REGISTERED_CELLS:
        raise ValueError("time-to-quality cell differs")
    target = _finite_float(target, name="time-to-quality target")
    first_epoch = next(
        (
            epoch
            for epoch in REGISTERED_EPOCHS
            if by_key[(cell, epoch)]["metrics"]["map_at_r"] >= target
        ),
        None,
    )
    return {
        "target_map_at_r": target,
        "first_epoch": first_epoch,
        "speedup": None if first_epoch is None else 16.0 / first_epoch,
    }


def paired_map_bootstrap_interval(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    seed: int = 0,
    samples: int = 10_000,
    chunk_size: int = 64,
) -> tuple[float, float]:
    try:
        baseline_values = np.asarray(
            baseline["query_evidence"]["average_precision"], dtype=np.float64
        )
        candidate_values = np.asarray(
            candidate["query_evidence"]["average_precision"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("paired mAP evidence differs") from error
    if (
        baseline_values.ndim != 1
        or not baseline_values.size
        or baseline_values.shape != candidate_values.shape
        or not np.isfinite(baseline_values).all()
        or not np.isfinite(candidate_values).all()
        or type(seed) is not int
        or type(samples) is not int
        or samples <= 0
        or type(chunk_size) is not int
        or chunk_size <= 0
    ):
        raise ValueError("paired mAP evidence differs")
    differences = candidate_values - baseline_values
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = generator.integers(0, differences.size, size=(stop - start, differences.size))
        means[start:stop] = differences[indices].mean(axis=1, dtype=np.float64)
    bounds = np.percentile(means, [2.5, 97.5])
    return float(bounds[0]), float(bounds[1])


def factorial_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_interval: tuple[float, float],
) -> dict[str, object]:
    by_key = _registered_rows(rows)
    if (
        type(bootstrap_interval) is not tuple
        or len(bootstrap_interval) != 2
        or any(type(value) is not float or not math.isfinite(value) for value in bootstrap_interval)
        or bootstrap_interval[0] > bootstrap_interval[1]
    ):
        raise ValueError("factorial bootstrap interval differs")
    baseline = by_key[("random_raw", 16)]
    selected = select_candidate(rows)
    baseline_map = baseline["metrics"]["map_at_r"]
    baseline_recall = baseline["metrics"]["recall_at_1"]
    selected_map = selected["metrics"]["map_at_r"]
    selected_recall = selected["metrics"]["recall_at_1"]
    instrument_reproduced = (
        baseline_map >= ARCHIVED_MAP_AT_R - 0.002
        and baseline_recall >= ARCHIVED_RECALL_AT_1 - 0.002
    )
    map_gain = selected_map - baseline_map
    recall_delta = selected_recall - baseline_recall
    ema_deltas = {
        "random": by_key[("random_ema", 16)]["metrics"]["map_at_r"]
        - by_key[("random_raw", 16)]["metrics"]["map_at_r"],
        "imprinted": by_key[("imprinted_ema", 16)]["metrics"]["map_at_r"]
        - by_key[("imprinted_raw", 16)]["metrics"]["map_at_r"],
    }
    imprint_delta = by_key[("imprinted_raw", 16)]["metrics"]["map_at_r"] - baseline_map
    imprint_time = time_to_quality(rows, cell="imprinted_raw", target=baseline_map)
    imprint_speedup = imprint_time["speedup"]
    promoted = (
        instrument_reproduced
        and map_gain >= 0.003
        and recall_delta >= -0.00125
        and bootstrap_interval[0] > 0.0
    )
    return {
        "instrument_map_tolerance": 0.002,
        "instrument_recall_at_1_tolerance": 0.002,
        "instrument_reproduced": instrument_reproduced,
        "selected_cell": selected["cell"],
        "selected_map_gain": map_gain,
        "selected_recall_at_1_delta": recall_delta,
        "map_delta_query_bootstrap_95_interval": list(bootstrap_interval),
        "minimum_map_gain": 0.003,
        "recall_at_1_guard": -0.00125,
        "ema_map_deltas": ema_deltas,
        "ema_closed": all(delta < 0.0015 for delta in ema_deltas.values()),
        "imprint_raw_map_gain": imprint_delta,
        "imprint_time_to_quality": imprint_time,
        "imprint_closed": (
            imprint_delta < 0.0015 and (imprint_speedup is None or imprint_speedup < 1.5)
        ),
        "decision": "INVALID"
        if not instrument_reproduced
        else ("PROMOTE" if promoted else "CLOSE"),
        "promoted": promoted,
    }


def control_gate(row: Mapping[str, object]) -> dict[str, object]:
    if type(row) is not dict or row.get("cell") != "random_raw" or row.get("epoch") != 16:
        raise ValueError("control row differs")
    metrics = row.get("metrics")
    if type(metrics) is not dict:
        raise ValueError("control metrics differ")
    map_at_r = _finite_float(metrics.get("map_at_r"), name="control mAP@R")
    recall_at_1 = _finite_float(metrics.get("recall_at_1"), name="control Recall@1")
    reproduced = (
        abs(map_at_r - ARCHIVED_MAP_AT_R) <= 0.002
        and abs(recall_at_1 - ARCHIVED_RECALL_AT_1) <= 0.002
    )
    return {
        "instrument_map_tolerance": 0.002,
        "instrument_recall_at_1_tolerance": 0.002,
        "instrument_reproduced": reproduced,
        "decision": "CONTINUE" if reproduced else "INVALID",
    }


def repaired_control_gate(row: Mapping[str, object]) -> dict[str, object]:
    if type(row) is not dict or row.get("cell") != "random_raw" or row.get("epoch") != 16:
        raise ValueError("repaired control row differs")
    metrics = row.get("metrics")
    if type(metrics) is not dict:
        raise ValueError("repaired control metrics differ")
    map_at_r = _finite_float(metrics.get("map_at_r"), name="repaired control mAP@R")
    recall_at_1 = _finite_float(metrics.get("recall_at_1"), name="repaired control Recall@1")
    noninferior = (
        map_at_r >= ARCHIVED_MAP_AT_R - 0.002 and recall_at_1 >= ARCHIVED_RECALL_AT_1 - 0.002
    )
    return {
        "archived_map_at_r": ARCHIVED_MAP_AT_R,
        "archived_recall_at_1": ARCHIVED_RECALL_AT_1,
        "map_noninferiority_tolerance": 0.002,
        "recall_at_1_noninferiority_tolerance": 0.002,
        "lineage_noninferior": noninferior,
        "minimum_candidate_map_gain": 0.003,
        "candidate_recall_at_1_guard": -0.00125,
        "candidate_bootstrap_lower_must_be_positive": True,
        "decision": "CONTINUE" if noninferior else "INVALID",
    }


def _load_archived_recomputation_evidence() -> dict[str, object]:
    path = Path(__file__).parents[1] / ARCHIVED_RECOMPUTATION_REPORT
    if _sha256_file(path) != ARCHIVED_RECOMPUTATION_REPORT_SHA256:
        raise ValueError("archived recomputation report bytes differ")
    value = json.loads(path.read_text())
    if type(value) is not dict or tuple(value) != (
        "schema_version",
        "purpose",
        "archived_report",
        "archived_raw_checkpoint",
        "evaluator",
        "result",
        "disclosure",
    ):
        raise ValueError("archived recomputation report schema differs")
    if (
        value["schema_version"] != "unicom-archived-raw-current-evaluator-v1"
        or value["purpose"]
        != "locate the seed-0 control divergence without evaluating any candidate arm"
        or value["archived_report"]
        != {
            "path": "reports/archive/unicom_ftra_seed0_76f2290.remote.json",
            "sha256": ARCHIVED_FTRA_REPORT_SHA256,
        }
        or value["archived_raw_checkpoint"]
        != {
            "path": "/home/riomus/unicom-screen-84df706-seed0-fp32-e16/epoch-0016.pt",
            "sha256": ARCHIVED_RAW_CHECKPOINT_SHA256,
        }
        or value["evaluator"]
        != {
            "revision": ARCHIVED_RECOMPUTATION_EVALUATOR_REVISION,
            "path": "scripts/evaluate_unicom_ema_imprint_factorial.py",
            "sha256": ARCHIVED_RECOMPUTATION_EVALUATOR_SHA256,
        }
        or value["result"]
        != {
            "query_count": 797,
            "map_at_r": ARCHIVED_MAP_AT_R,
            "recall_at_1": ARCHIVED_RECALL_AT_1,
            "archived_map_exact": True,
            "archived_recall_at_1_exact": True,
        }
        or value["disclosure"]
        != {
            "candidate_values_computed": False,
            "artifact_created_after_original_control_invalidated": True,
            "interpretation": ARCHIVED_RECOMPUTATION_INTERPRETATION,
        }
    ):
        raise ValueError("archived recomputation report differs")
    return value


def materialize_checkpoint_state(checkpoint: object, *, use_ema: bool) -> dict[str, object]:
    import torch

    if type(use_ema) is not bool or type(checkpoint) is not dict:
        raise TypeError("checkpoint materialization arguments differ")
    raw = checkpoint.get("model")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("checkpoint raw model state differs")
    state = {}
    for name, value in raw.items():
        if type(name) is not str or not name or type(value) is not torch.Tensor:
            raise ValueError("checkpoint raw model state differs")
        state[name] = value.detach().cpu().clone()
    ema = checkpoint.get("ema")
    if type(ema) is not dict or tuple(ema) != (
        "decay",
        "updates",
        "backbone",
        "classifier",
    ):
        raise ValueError("checkpoint EMA state differs")
    if (
        type(ema["decay"]) is not float
        or ema["decay"] != 0.999
        or type(ema["updates"]) is not int
        or ema["updates"] < 0
        or type(ema["backbone"]) is not dict
        or not ema["backbone"]
        or type(ema["classifier"]) is not torch.Tensor
    ):
        raise ValueError("checkpoint EMA state differs")
    changed = False
    for name, value in ema["backbone"].items():
        if (
            type(name) is not str
            or name not in state
            or type(value) is not torch.Tensor
            or value.dtype != torch.float32
            or value.shape != state[name].shape
            or state[name].dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError("checkpoint EMA parameter differs")
        if use_ema:
            changed = changed or not torch.equal(value, state[name])
            state[name] = value.detach().cpu().clone()
    if use_ema and not changed:
        raise ValueError("checkpoint EMA state matches raw state")
    return state


def ema_parameter_audit(checkpoint: Mapping[str, object]) -> tuple[int, float, float]:
    import torch

    raw = checkpoint.get("model")
    ema = checkpoint.get("ema")
    if not isinstance(raw, Mapping) or type(ema) is not dict:
        raise ValueError("checkpoint EMA audit state differs")
    updates = ema.get("updates")
    backbone = ema.get("backbone")
    if type(updates) is not int or updates <= 0 or type(backbone) is not dict or not backbone:
        raise ValueError("checkpoint EMA audit state differs")
    squared_distance = 0.0
    for name, value in backbone.items():
        raw_value = raw.get(name)
        if (
            type(value) is not torch.Tensor
            or type(raw_value) is not torch.Tensor
            or value.shape != raw_value.shape
        ):
            raise ValueError("checkpoint EMA audit parameter differs")
        difference = value.detach().cpu() - raw_value.detach().cpu()
        squared_distance += float(torch.sum(difference.square(), dtype=torch.float64))
    distance = math.sqrt(squared_distance)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("checkpoint EMA state matches raw state")
    return updates, float(0.999**updates), distance


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_training_protocol(value: object, *, classifier_init: str) -> None:
    if type(value) is not dict or tuple(value) != TRAINING_PROTOCOL_FIELDS:
        raise ValueError("factorial training protocol schema differs")
    expected = {
        "protocol": "unicom-inshop-official-single-device-v1",
        "seed": 0,
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
    if any(value[name] != expected_value for name, expected_value in expected.items()):
        raise ValueError("factorial training protocol differs")
    if (
        not _sha256(value["trainer_sha256"])
        or type(value["unicom_revision"]) is not str
        or len(value["unicom_revision"]) != 40
        or not _sha256(value["initial_checkpoint_sha256"])
        or not _sha256(value["partition_sha256"])
    ):
        raise ValueError("factorial training protocol binding differs")


def validate_factorial_report(report: object) -> None:
    if type(report) is not dict or tuple(report) != (
        "schema_version",
        "provenance",
        "costs",
        "rows",
        "gate",
    ):
        raise ValueError("factorial report schema differs")
    if report["schema_version"] != "unicom-ema-imprint-factorial-v1":
        raise ValueError("factorial report version differs")
    provenance = report["provenance"]
    if type(provenance) is not dict or tuple(provenance) != (
        "unicom_revision",
        "initial_checkpoint_sha256",
        "partition_sha256",
        "holdout_seed",
        "holdout_fraction",
        "batch_norm_recalibration",
        "query_chunk_size",
        "training_history_metrics",
        "control_report",
        "control_report_sha256",
        "random_run",
        "imprinted_run",
        "random_training_protocol",
        "imprinted_training_protocol",
    ):
        raise ValueError("factorial provenance schema differs")
    if (
        type(provenance["unicom_revision"]) is not str
        or len(provenance["unicom_revision"]) != 40
        or not _sha256(provenance["initial_checkpoint_sha256"])
        or not _sha256(provenance["partition_sha256"])
        or provenance["holdout_seed"] != 0
        or provenance["holdout_fraction"] != 0.2
        or provenance["batch_norm_recalibration"] != "full-optimization-cumulative-batches-all-arms"
        or provenance["query_chunk_size"] != 256
        or provenance["training_history_metrics"]
        != "instrument-only-unhardened-not-used-for-selection"
        or type(provenance["control_report"]) is not str
        or not provenance["control_report"]
        or not _sha256(provenance["control_report_sha256"])
        or any(
            type(provenance[name]) is not str or not provenance[name]
            for name in ("random_run", "imprinted_run")
        )
    ):
        raise ValueError("factorial provenance differs")
    validate_training_protocol(provenance["random_training_protocol"], classifier_init="random")
    validate_training_protocol(
        provenance["imprinted_training_protocol"], classifier_init="imprinted"
    )
    random_protocol = provenance["random_training_protocol"]
    imprinted_protocol = provenance["imprinted_training_protocol"]
    for name in TRAINING_PROTOCOL_FIELDS:
        if name != "classifier_init" and random_protocol[name] != imprinted_protocol[name]:
            raise ValueError("factorial training protocols differ")
    if (
        random_protocol["unicom_revision"] != provenance["unicom_revision"]
        or random_protocol["initial_checkpoint_sha256"] != provenance["initial_checkpoint_sha256"]
        or random_protocol["partition_sha256"] != provenance["partition_sha256"]
    ):
        raise ValueError("factorial training provenance differs")
    costs = report["costs"]
    if type(costs) is not dict or tuple(costs) != (
        "random_training_seconds",
        "imprinted_training_seconds",
        "random_peak_gpu_mib",
        "imprinted_peak_gpu_mib",
        "checkpoint_storage_bytes",
        "architecture_inference_latency_ms_per_image",
        "evaluator_seconds",
    ):
        raise ValueError("factorial cost schema differs")
    for name in (
        "random_training_seconds",
        "imprinted_training_seconds",
        "architecture_inference_latency_ms_per_image",
        "evaluator_seconds",
    ):
        if type(costs[name]) is not float or not math.isfinite(costs[name]) or costs[name] < 0.0:
            raise ValueError("factorial cost differs")
    for name in ("random_peak_gpu_mib", "imprinted_peak_gpu_mib", "checkpoint_storage_bytes"):
        if type(costs[name]) is not int or costs[name] < 0:
            raise ValueError("factorial cost differs")

    rows = report["rows"]
    _registered_rows(rows)
    query_count = None
    for row in rows:
        if tuple(row) != (
            "cell",
            "epoch",
            "checkpoint",
            "checkpoint_sha256",
            "training_history_sha256",
            "ema_updates",
            "ema_initial_weight",
            "ema_parameter_l2_distance",
            "metrics",
            "query_evidence",
        ):
            raise ValueError("factorial row schema differs")
        if (
            type(row["checkpoint"]) is not str
            or not row["checkpoint"]
            or not _sha256(row["checkpoint_sha256"])
            or not _sha256(row["training_history_sha256"])
        ):
            raise ValueError("factorial checkpoint evidence differs")
        if row["cell"].endswith("_raw"):
            if any(
                row[name] is not None
                for name in (
                    "ema_updates",
                    "ema_initial_weight",
                    "ema_parameter_l2_distance",
                )
            ):
                raise ValueError("factorial raw EMA audit differs")
        elif (
            type(row["ema_updates"]) is not int
            or row["ema_updates"] <= 0
            or type(row["ema_initial_weight"]) is not float
            or row["ema_initial_weight"] != float(0.999 ** row["ema_updates"])
            or type(row["ema_parameter_l2_distance"]) is not float
            or not math.isfinite(row["ema_parameter_l2_distance"])
            or row["ema_parameter_l2_distance"] <= 0.0
        ):
            raise ValueError("factorial EMA audit differs")
        metrics = row["metrics"]
        if tuple(metrics) != (
            "recall_at_1",
            "recall_at_10",
            "recall_at_20",
            "recall_at_30",
            "map_at_r",
        ):
            raise ValueError("factorial metric schema differs")
        if any(
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in metrics.values()
        ):
            raise ValueError("factorial metric differs")
        evidence = row["query_evidence"]
        if type(evidence) is not dict or tuple(evidence) != (
            "top1_correct",
            "average_precision",
        ):
            raise ValueError("factorial query evidence schema differs")
        top1 = evidence["top1_correct"]
        average_precision = evidence["average_precision"]
        if (
            type(top1) is not list
            or type(average_precision) is not list
            or not top1
            or len(top1) != len(average_precision)
            or any(type(value) is not bool for value in top1)
            or any(
                type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in average_precision
            )
        ):
            raise ValueError("factorial query evidence differs")
        if query_count is None:
            query_count = len(top1)
        elif query_count != len(top1):
            raise ValueError("factorial query evidence count differs")
        expected_recall = sum(top1) / len(top1)
        expected_map = math.fsum(average_precision) / len(average_precision)
        if (
            abs(metrics["recall_at_1"] - expected_recall) > 1e-15
            or abs(metrics["map_at_r"] - expected_map) > 1e-12
        ):
            raise ValueError("factorial metric evidence differs")
    selected = select_candidate(rows)
    baseline = next(row for row in rows if row["cell"] == "random_raw" and row["epoch"] == 16)
    interval = paired_map_bootstrap_interval(baseline, selected)
    expected_gate = factorial_gate(rows, bootstrap_interval=interval)
    if report["gate"] != expected_gate:
        raise ValueError("factorial gate differs")


def validate_control_report(report: object) -> None:
    if type(report) is not dict or tuple(report) != (
        "schema_version",
        "provenance",
        "costs",
        "row",
        "gate",
    ):
        raise ValueError("control report schema differs")
    if report["schema_version"] != "unicom-ema-imprint-control-v1":
        raise ValueError("control report version differs")
    provenance = report["provenance"]
    if type(provenance) is not dict or tuple(provenance) != (
        "unicom_revision",
        "initial_checkpoint_sha256",
        "partition_sha256",
        "holdout_seed",
        "holdout_fraction",
        "batch_norm_recalibration",
        "query_chunk_size",
        "training_history_metrics",
        "random_run",
        "random_training_protocol",
    ):
        raise ValueError("control provenance schema differs")
    if (
        type(provenance["unicom_revision"]) is not str
        or len(provenance["unicom_revision"]) != 40
        or not _sha256(provenance["initial_checkpoint_sha256"])
        or not _sha256(provenance["partition_sha256"])
        or provenance["holdout_seed"] != 0
        or provenance["holdout_fraction"] != 0.2
        or provenance["batch_norm_recalibration"] != "full-optimization-cumulative-batches"
        or provenance["query_chunk_size"] != 256
        or provenance["training_history_metrics"]
        != "instrument-only-unhardened-not-used-for-selection"
        or type(provenance["random_run"]) is not str
        or not provenance["random_run"]
    ):
        raise ValueError("control provenance differs")
    protocol = provenance["random_training_protocol"]
    validate_training_protocol(protocol, classifier_init="random")
    if (
        protocol["unicom_revision"] != provenance["unicom_revision"]
        or protocol["initial_checkpoint_sha256"] != provenance["initial_checkpoint_sha256"]
        or protocol["partition_sha256"] != provenance["partition_sha256"]
    ):
        raise ValueError("control training provenance differs")
    costs = report["costs"]
    if type(costs) is not dict or tuple(costs) != (
        "random_training_seconds",
        "random_peak_gpu_mib",
        "checkpoint_storage_bytes",
        "architecture_inference_latency_ms_per_image",
        "evaluator_seconds",
    ):
        raise ValueError("control cost schema differs")
    for name in (
        "random_training_seconds",
        "architecture_inference_latency_ms_per_image",
        "evaluator_seconds",
    ):
        if type(costs[name]) is not float or not math.isfinite(costs[name]) or costs[name] < 0:
            raise ValueError("control cost differs")
    for name in ("random_peak_gpu_mib", "checkpoint_storage_bytes"):
        if type(costs[name]) is not int or costs[name] < 0:
            raise ValueError("control cost differs")
    row = report["row"]
    if type(row) is not dict or tuple(row) != (
        "cell",
        "epoch",
        "checkpoint",
        "checkpoint_sha256",
        "training_history_sha256",
        "ema_updates",
        "ema_initial_weight",
        "ema_parameter_l2_distance",
        "metrics",
        "query_evidence",
    ):
        raise ValueError("control row schema differs")
    if row["cell"] != "random_raw" or row["epoch"] != 16:
        raise ValueError("control row differs")
    if any(
        row[name] is not None
        for name in ("ema_updates", "ema_initial_weight", "ema_parameter_l2_distance")
    ):
        raise ValueError("control raw EMA audit differs")
    if (
        type(row["checkpoint"]) is not str
        or not row["checkpoint"]
        or not _sha256(row["checkpoint_sha256"])
        or not _sha256(row["training_history_sha256"])
    ):
        raise ValueError("control row binding differs")
    _validate_control_row_evidence(row)
    if report["gate"] != control_gate(row):
        raise ValueError("control gate differs")


def build_repaired_control_report(
    legacy_report: object,
    *,
    legacy_control_report: str,
    legacy_control_report_sha256: str,
) -> dict[str, object]:
    validate_control_report(legacy_report)
    recomputation = _load_archived_recomputation_evidence()
    if (
        type(legacy_control_report) is not str
        or not legacy_control_report
        or legacy_control_report_sha256 != REGISTERED_LEGACY_CONTROL_SHA256
        or legacy_report["gate"]["decision"] != "INVALID"
        or legacy_report["row"]["checkpoint_sha256"] != REGISTERED_RANDOM_E16_CHECKPOINT_SHA256
        or legacy_report["row"]["training_history_sha256"] != REGISTERED_RANDOM_HISTORY_SHA256
        or legacy_report["provenance"]["initial_checkpoint_sha256"]
        != ARCHIVED_INITIAL_CHECKPOINT_SHA256
        or legacy_report["provenance"]["partition_sha256"] != ARCHIVED_PARTITION_SHA256
    ):
        raise ValueError("legacy control repair binding differs")
    repaired = {
        "schema_version": "unicom-ema-imprint-control-v2",
        "repair": {
            "reason": CONTROL_REPAIR_REASON,
            "legacy_control_report": legacy_control_report,
            "legacy_control_report_sha256": legacy_control_report_sha256,
            "archived_ftra_report_sha256": ARCHIVED_FTRA_REPORT_SHA256,
            "archived_raw_checkpoint_sha256": ARCHIVED_RAW_CHECKPOINT_SHA256,
            "archived_trainer_sha256": ARCHIVED_TRAINER_SHA256,
            "archived_recomputation_report": ARCHIVED_RECOMPUTATION_REPORT,
            "archived_recomputation_report_sha256": ARCHIVED_RECOMPUTATION_REPORT_SHA256,
            "archived_recomputation_evaluator_revision": (
                ARCHIVED_RECOMPUTATION_EVALUATOR_REVISION
            ),
            "archived_recomputation_evaluator_sha256": (ARCHIVED_RECOMPUTATION_EVALUATOR_SHA256),
            "archived_recomputed_map_at_r": recomputation["result"]["map_at_r"],
            "archived_recomputed_recall_at_1": recomputation["result"]["recall_at_1"],
            "candidate_values_observed": False,
            "promotion_thresholds_unchanged": True,
        },
        "provenance": legacy_report["provenance"],
        "costs": legacy_report["costs"],
        "row": legacy_report["row"],
        "legacy_gate": legacy_report["gate"],
        "gate": repaired_control_gate(legacy_report["row"]),
    }
    validate_repaired_control_report(repaired)
    return repaired


def validate_repaired_control_report(report: object) -> None:
    if type(report) is not dict or tuple(report) != (
        "schema_version",
        "repair",
        "provenance",
        "costs",
        "row",
        "legacy_gate",
        "gate",
    ):
        raise ValueError("repaired control report schema differs")
    if report["schema_version"] != "unicom-ema-imprint-control-v2":
        raise ValueError("repaired control report version differs")
    repair = report["repair"]
    if type(repair) is not dict or tuple(repair) != (
        "reason",
        "legacy_control_report",
        "legacy_control_report_sha256",
        "archived_ftra_report_sha256",
        "archived_raw_checkpoint_sha256",
        "archived_trainer_sha256",
        "archived_recomputation_report",
        "archived_recomputation_report_sha256",
        "archived_recomputation_evaluator_revision",
        "archived_recomputation_evaluator_sha256",
        "archived_recomputed_map_at_r",
        "archived_recomputed_recall_at_1",
        "candidate_values_observed",
        "promotion_thresholds_unchanged",
    ):
        raise ValueError("repaired control repair schema differs")
    recomputation = _load_archived_recomputation_evidence()
    if (
        repair["reason"] != CONTROL_REPAIR_REASON
        or type(repair["legacy_control_report"]) is not str
        or not repair["legacy_control_report"]
        or repair["legacy_control_report_sha256"] != REGISTERED_LEGACY_CONTROL_SHA256
        or repair["archived_ftra_report_sha256"] != ARCHIVED_FTRA_REPORT_SHA256
        or repair["archived_raw_checkpoint_sha256"] != ARCHIVED_RAW_CHECKPOINT_SHA256
        or repair["archived_trainer_sha256"] != ARCHIVED_TRAINER_SHA256
        or repair["archived_recomputation_report"] != ARCHIVED_RECOMPUTATION_REPORT
        or repair["archived_recomputation_report_sha256"] != ARCHIVED_RECOMPUTATION_REPORT_SHA256
        or repair["archived_recomputation_evaluator_revision"]
        != recomputation["evaluator"]["revision"]
        or repair["archived_recomputation_evaluator_sha256"] != recomputation["evaluator"]["sha256"]
        or repair["archived_recomputed_map_at_r"] != recomputation["result"]["map_at_r"]
        or repair["archived_recomputed_recall_at_1"] != recomputation["result"]["recall_at_1"]
        or repair["candidate_values_observed"] is not False
        or repair["promotion_thresholds_unchanged"] is not True
    ):
        raise ValueError("repaired control repair differs")
    legacy = {
        "schema_version": "unicom-ema-imprint-control-v1",
        "provenance": report["provenance"],
        "costs": report["costs"],
        "row": report["row"],
        "gate": report["legacy_gate"],
    }
    validate_control_report(legacy)
    if (
        report["legacy_gate"]["decision"] != "INVALID"
        or report["row"]["checkpoint_sha256"] != REGISTERED_RANDOM_E16_CHECKPOINT_SHA256
        or report["row"]["training_history_sha256"] != REGISTERED_RANDOM_HISTORY_SHA256
        or report["provenance"]["initial_checkpoint_sha256"] != ARCHIVED_INITIAL_CHECKPOINT_SHA256
        or report["provenance"]["partition_sha256"] != ARCHIVED_PARTITION_SHA256
        or report["gate"] != repaired_control_gate(report["row"])
    ):
        raise ValueError("repaired control binding differs")


def _validate_control_row_evidence(row: Mapping[str, object]) -> None:
    metrics = row.get("metrics")
    evidence = row.get("query_evidence")
    if type(metrics) is not dict or tuple(metrics) != (
        "recall_at_1",
        "recall_at_10",
        "recall_at_20",
        "recall_at_30",
        "map_at_r",
    ):
        raise ValueError("control metric schema differs")
    if any(
        type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in metrics.values()
    ):
        raise ValueError("control metric differs")
    if type(evidence) is not dict or tuple(evidence) != ("top1_correct", "average_precision"):
        raise ValueError("control query evidence schema differs")
    top1 = evidence["top1_correct"]
    average_precision = evidence["average_precision"]
    if (
        type(top1) is not list
        or type(average_precision) is not list
        or not top1
        or len(top1) != len(average_precision)
        or any(type(value) is not bool for value in top1)
        or any(
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in average_precision
        )
    ):
        raise ValueError("control query evidence differs")
    if (
        abs(metrics["recall_at_1"] - sum(top1) / len(top1)) > 1e-15
        or abs(metrics["map_at_r"] - math.fsum(average_precision) / len(average_precision)) > 1e-12
    ):
        raise ValueError("control metric evidence differs")


def validate_control_binding(
    control_report: object,
    rows: Sequence[Mapping[str, object]],
    random_protocol: Mapping[str, object],
) -> None:
    if type(control_report) is not dict or control_report.get("schema_version") != (
        "unicom-ema-imprint-control-v2"
    ):
        raise ValueError("factorial requires repaired control report")
    validate_repaired_control_report(control_report)
    by_key = _registered_rows(rows)
    if (
        control_report["gate"]["decision"] != "CONTINUE"
        or control_report["row"]["checkpoint_sha256"]
        != by_key[("random_raw", 16)]["checkpoint_sha256"]
        or control_report["row"]["training_history_sha256"]
        != by_key[("random_raw", 16)]["training_history_sha256"]
        or control_report["row"]["metrics"] != by_key[("random_raw", 16)]["metrics"]
        or control_report["row"]["query_evidence"] != by_key[("random_raw", 16)]["query_evidence"]
        or control_report["provenance"]["random_training_protocol"] != random_protocol
    ):
        raise ValueError("factorial control report binding differs")


def write_report_atomic(report: dict[str, object], output: Path) -> None:
    validators = {
        "unicom-ema-imprint-control-v1": validate_control_report,
        "unicom-ema-imprint-control-v2": validate_repaired_control_report,
        "unicom-ema-imprint-factorial-v1": validate_factorial_report,
    }
    try:
        validator = validators[report.get("schema_version")]
    except (KeyError, TypeError) as error:
        raise ValueError("report version differs") from error
    validator(report)
    if not isinstance(output, Path):
        raise TypeError("output must be a Path")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if output.exists() or output.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(output)
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    descriptor = None
    owned = None
    directory_descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = json.loads(temporary.read_text())
        validator(persisted)
        os.link(temporary, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory_descriptor)
        temporary.unlink()
        os.fsync(directory_descriptor)
        validator(json.loads(output.read_text()))
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        try:
            info = temporary.lstat()
        except FileNotFoundError:
            pass
        else:
            if owned is not None and (info.st_dev, info.st_ino) == owned:
                temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_local_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _checkpoint_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("factorial run directory differs")
    paths = tuple(directory / f"epoch-{epoch:04d}.pt" for epoch in REGISTERED_EPOCHS)
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("factorial checkpoint path differs")
    return paths


def _load_registered_checkpoint(
    path: Path,
    *,
    epoch: int,
    classifier_init: str,
    expected_ema_names: tuple[str, ...] | None = None,
):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if type(checkpoint) is not dict or tuple(checkpoint) != (
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
    ):
        raise ValueError("factorial checkpoint schema differs")
    if checkpoint["epoch"] != epoch or checkpoint["selection_holdout"] != {
        "seed": 0,
        "fraction": 0.2,
    }:
        raise ValueError("factorial checkpoint registration differs")
    validate_training_protocol(checkpoint["training_protocol"], classifier_init=classifier_init)
    history = checkpoint["history"]
    if (
        type(history) is not list
        or len(history) != epoch
        or any(
            type(row) is not dict or row.get("epoch") != index
            for index, row in enumerate(history, 1)
        )
    ):
        raise ValueError("factorial checkpoint history differs")
    ema = checkpoint["ema"]
    if type(ema) is not dict or type(ema.get("updates")) is not int or ema["updates"] <= 0:
        raise ValueError("factorial checkpoint EMA updates differ")
    if expected_ema_names is not None and (
        type(expected_ema_names) is not tuple
        or tuple(ema.get("backbone", ())) != expected_ema_names
    ):
        raise ValueError("factorial checkpoint EMA parameter order differs")
    return checkpoint


def _benchmark_inference_latency(model, calibration_loader, *, device) -> float:
    import torch

    try:
        images, _labels = next(iter(calibration_loader))
    except StopIteration as error:
        raise ValueError("factorial latency batch is empty") from error
    images = images.to(device)
    model.eval()
    with torch.inference_mode():
        for _index in range(10):
            model(images)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _index in range(50):
            model(images)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return float(elapsed * 1_000.0 / (50 * images.shape[0]))


def load_factorial_rows(args: argparse.Namespace):
    import torch

    from sfora.unicom_inshop import parse_inshop_partition
    from sfora.unicom_training import identity_holdout

    soup = _load_local_module(
        Path(__file__).resolve().with_name("evaluate_unicom_checkpoint_soup.py"),
        "_unicom_soup_for_ema_imprint",
    )
    trainer = soup._load_trainer()
    if trainer._git_revision(args.unicom_checkout) != trainer.UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if _sha256_file(args.initial_checkpoint) != trainer.UNICOM_L14_336_SHA256:
        raise ValueError("UNICOM initial checkpoint SHA-256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the EMA imprint factorial")
    random_paths = _checkpoint_paths(args.random_run)
    imprinted_paths = _checkpoint_paths(args.imprinted_run)
    if set(random_paths) & set(imprinted_paths):
        raise ValueError("factorial run checkpoints overlap")

    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(record for record in records if record.split == "train")
    optimization, query, gallery, _labels = identity_holdout(train_records, fraction=0.2, seed=0)
    if not optimization or not query or not gallery:
        raise ValueError("factorial train-only split differs")
    model, transform = trainer._load_official_model(args.unicom_checkout, args.initial_checkpoint)
    device = torch.device("cuda")
    model = model.to(device)
    expected_ema_names = tuple(name for name, _parameter in model.named_parameters())
    calibration_loader = torch.utils.data.DataLoader(
        trainer.InshopEvalDataset(optimization, transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    path_by_cell = {
        "random_raw": random_paths,
        "random_ema": random_paths,
        "imprinted_raw": imprinted_paths,
        "imprinted_ema": imprinted_paths,
    }
    rows = []
    random_protocol = None
    imprinted_protocol = None
    prior_updates = {"random": 0, "imprinted": 0}
    for cell in REGISTERED_CELLS:
        classifier_init = "random" if cell.startswith("random_") else "imprinted"
        use_ema = cell.endswith("_ema")
        for epoch, path in zip(REGISTERED_EPOCHS, path_by_cell[cell], strict=True):
            checkpoint = _load_registered_checkpoint(
                path,
                epoch=epoch,
                classifier_init=classifier_init,
                expected_ema_names=expected_ema_names,
            )
            protocol = checkpoint["training_protocol"]
            if classifier_init == "random":
                random_protocol = protocol if random_protocol is None else random_protocol
                if protocol != random_protocol:
                    raise ValueError("random factorial checkpoint protocols differ")
            else:
                imprinted_protocol = protocol if imprinted_protocol is None else imprinted_protocol
                if protocol != imprinted_protocol:
                    raise ValueError("imprinted factorial checkpoint protocols differ")
            if use_ema:
                run_name = classifier_init
                updates = checkpoint["ema"]["updates"]
                if updates <= prior_updates[run_name]:
                    raise ValueError("factorial EMA update sequence differs")
                prior_updates[run_name] = updates
            state = materialize_checkpoint_state(checkpoint, use_ema=use_ema)
            if use_ema:
                ema_updates, ema_initial_weight, ema_distance = ema_parameter_audit(checkpoint)
            else:
                ema_updates = ema_initial_weight = ema_distance = None
            model.load_state_dict(state, strict=True)
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
                    "checkpoint": str(path),
                    "checkpoint_sha256": _sha256_file(path),
                    "training_history_sha256": _json_sha256(checkpoint["history"]),
                    "ema_updates": ema_updates,
                    "ema_initial_weight": ema_initial_weight,
                    "ema_parameter_l2_distance": ema_distance,
                    "metrics": metrics,
                    "query_evidence": evidence,
                }
            )
            del checkpoint, state
    assert random_protocol is not None and imprinted_protocol is not None
    trainer_sha256 = _sha256_file(Path(trainer.__file__))
    if (
        random_protocol["trainer_sha256"] != trainer_sha256
        or imprinted_protocol["trainer_sha256"] != trainer_sha256
    ):
        raise ValueError("factorial executing trainer source differs")
    for name in TRAINING_PROTOCOL_FIELDS:
        if name != "classifier_init" and random_protocol[name] != imprinted_protocol[name]:
            raise ValueError("factorial run protocols differ")
    storage = sum(path.stat().st_size for path in (*random_paths, *imprinted_paths))
    inference_latency = _benchmark_inference_latency(model, calibration_loader, device=device)
    return rows, random_protocol, imprinted_protocol, storage, inference_latency


def load_control_row(args: argparse.Namespace):
    import torch

    from sfora.unicom_inshop import parse_inshop_partition
    from sfora.unicom_training import identity_holdout

    soup = _load_local_module(
        Path(__file__).resolve().with_name("evaluate_unicom_checkpoint_soup.py"),
        "_unicom_soup_for_ema_imprint_control",
    )
    trainer = soup._load_trainer()
    if trainer._git_revision(args.unicom_checkout) != trainer.UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if _sha256_file(args.initial_checkpoint) != trainer.UNICOM_L14_336_SHA256:
        raise ValueError("UNICOM initial checkpoint SHA-256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the EMA imprint control")
    path = _checkpoint_paths(args.random_run)[-1]
    checkpoint = _load_registered_checkpoint(path, epoch=16, classifier_init="random")
    protocol = checkpoint["training_protocol"]
    if _sha256_file(Path(trainer.__file__)) != protocol["trainer_sha256"]:
        raise ValueError("control executing trainer source differs")
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(record for record in records if record.split == "train")
    optimization, query, gallery, _labels = identity_holdout(train_records, fraction=0.2, seed=0)
    if not optimization or not query or not gallery:
        raise ValueError("control train-only split differs")
    model, transform = trainer._load_official_model(args.unicom_checkout, args.initial_checkpoint)
    device = torch.device("cuda")
    model = model.to(device)
    if tuple(checkpoint["ema"]["backbone"]) != tuple(
        name for name, _parameter in model.named_parameters()
    ):
        raise ValueError("control checkpoint EMA parameter order differs")
    calibration_loader = torch.utils.data.DataLoader(
        trainer.InshopEvalDataset(optimization, transform),
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    state = materialize_checkpoint_state(checkpoint, use_ema=False)
    model.load_state_dict(state, strict=True)
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
    row = {
        "cell": "random_raw",
        "epoch": 16,
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "training_history_sha256": _json_sha256(checkpoint["history"]),
        "ema_updates": None,
        "ema_initial_weight": None,
        "ema_parameter_l2_distance": None,
        "metrics": metrics,
        "query_evidence": evidence,
    }
    latency = _benchmark_inference_latency(model, calibration_loader, device=device)
    storage = sum(path.stat().st_size for path in _checkpoint_paths(args.random_run))
    return row, protocol, storage, latency


def run_control(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    row, protocol, storage, inference_latency = load_control_row(args)
    report = {
        "schema_version": "unicom-ema-imprint-control-v1",
        "provenance": {
            "unicom_revision": protocol["unicom_revision"],
            "initial_checkpoint_sha256": protocol["initial_checkpoint_sha256"],
            "partition_sha256": protocol["partition_sha256"],
            "holdout_seed": 0,
            "holdout_fraction": 0.2,
            "batch_norm_recalibration": "full-optimization-cumulative-batches",
            "query_chunk_size": 256,
            "training_history_metrics": "instrument-only-unhardened-not-used-for-selection",
            "random_run": str(args.random_run),
            "random_training_protocol": protocol,
        },
        "costs": {
            "random_training_seconds": args.random_training_seconds,
            "random_peak_gpu_mib": args.random_peak_gpu_mib,
            "checkpoint_storage_bytes": storage,
            "architecture_inference_latency_ms_per_image": inference_latency,
            "evaluator_seconds": float(time.monotonic() - started),
        },
        "row": row,
        "gate": control_gate(row),
    }
    validate_control_report(report)
    return report


def run_repair_control(args: argparse.Namespace) -> dict[str, object]:
    payload = args.legacy_control_report.read_bytes()
    legacy = json.loads(payload)
    return build_repaired_control_report(
        legacy,
        legacy_control_report=str(args.legacy_control_report),
        legacy_control_report_sha256=hashlib.sha256(payload).hexdigest(),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    control_report = json.loads(args.control_report.read_text())
    validate_repaired_control_report(control_report)
    if control_report["gate"]["decision"] != "CONTINUE":
        raise ValueError("factorial control report did not authorize continuation")
    rows, random_protocol, imprinted_protocol, storage, inference_latency = load_factorial_rows(
        args
    )
    validate_control_binding(control_report, rows, random_protocol)
    selected = select_candidate(rows)
    baseline = next(row for row in rows if row["cell"] == "random_raw" and row["epoch"] == 16)
    interval = paired_map_bootstrap_interval(baseline, selected)
    report = {
        "schema_version": "unicom-ema-imprint-factorial-v1",
        "provenance": {
            "unicom_revision": random_protocol["unicom_revision"],
            "initial_checkpoint_sha256": random_protocol["initial_checkpoint_sha256"],
            "partition_sha256": random_protocol["partition_sha256"],
            "holdout_seed": 0,
            "holdout_fraction": 0.2,
            "batch_norm_recalibration": "full-optimization-cumulative-batches-all-arms",
            "query_chunk_size": 256,
            "training_history_metrics": "instrument-only-unhardened-not-used-for-selection",
            "control_report": str(args.control_report),
            "control_report_sha256": _sha256_file(args.control_report),
            "random_run": str(args.random_run),
            "imprinted_run": str(args.imprinted_run),
            "random_training_protocol": random_protocol,
            "imprinted_training_protocol": imprinted_protocol,
        },
        "costs": {
            "random_training_seconds": args.random_training_seconds,
            "imprinted_training_seconds": args.imprinted_training_seconds,
            "random_peak_gpu_mib": args.random_peak_gpu_mib,
            "imprinted_peak_gpu_mib": args.imprinted_peak_gpu_mib,
            "checkpoint_storage_bytes": storage,
            "architecture_inference_latency_ms_per_image": inference_latency,
            "evaluator_seconds": float(time.monotonic() - started),
        },
        "rows": rows,
        "gate": factorial_gate(rows, bootstrap_interval=interval),
    }
    validate_factorial_report(report)
    return report


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("control", "repair-control", "factorial"), default="factorial"
    )
    parser.add_argument("--unicom-checkout", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--random-run", type=Path)
    parser.add_argument("--imprinted-run", type=Path)
    parser.add_argument("--control-report", type=Path)
    parser.add_argument("--legacy-control-report", type=Path)
    parser.add_argument("--random-training-seconds", type=float)
    parser.add_argument("--imprinted-training-seconds", type=float)
    parser.add_argument("--random-peak-gpu-mib", type=int)
    parser.add_argument("--imprinted-peak-gpu-mib", type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(arguments)
    shared = (
        args.unicom_checkout,
        args.initial_checkpoint,
        args.dataset_root,
        args.random_run,
        args.random_training_seconds,
        args.random_peak_gpu_mib,
    )
    if args.mode in ("control", "factorial") and any(value is None for value in shared):
        parser.error("control and factorial modes require random run arguments")
    if args.mode == "repair-control" and args.legacy_control_report is None:
        parser.error("repair-control mode requires --legacy-control-report")
    if args.mode == "factorial" and any(
        value is None
        for value in (
            args.imprinted_run,
            args.control_report,
            args.imprinted_training_seconds,
            args.imprinted_peak_gpu_mib,
        )
    ):
        parser.error("factorial mode requires imprinted run and cost arguments")
    return args


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        runners = {
            "control": run_control,
            "repair-control": run_repair_control,
            "factorial": run,
        }
        report = runners[args.mode](args)
        write_report_atomic(report, args.output)
    except Exception as error:
        print(f"factorial evaluation failed: {error}", file=sys.stderr)
        return 2
    if args.mode == "control":
        succeeded = report["gate"]["instrument_reproduced"]
    elif args.mode == "repair-control":
        succeeded = report["gate"]["decision"] == "CONTINUE"
    else:
        succeeded = report["gate"]["promoted"]
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
