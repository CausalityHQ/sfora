#!/usr/bin/env python3
"""Select a UNICOM trajectory soup and WiSE interpolation on train identities only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REGISTERED_SCREEN_ALPHAS = (0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0)
REGISTERED_SCREEN_WINDOWS = ((16,), (12, 16), (8, 12, 16), (4, 8, 12, 16))
REGISTERED_REPLICATION_SEEDS = (1, 2, 3, 4, 5, 6)
REPORT_PROTOCOL = "unicom-train-identity-holdout-suffix-soup-wise-v3"
STUDENT_T_CRITICAL_TWO_SIDED_95_DF5 = 2.5705818356363146
REGISTERED_TRAINER_SHA256 = "b2cfdaed33d46ec445141bb40b1a3f28aed0d3ca859101843ddf825866640bb1"
REGISTERED_UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
REGISTERED_INITIAL_CHECKPOINT_SHA256 = (
    "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
)
REGISTERED_PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"


def suffix_windows(paths: tuple[Path, ...]) -> tuple[tuple[Path, ...], ...]:
    if not paths:
        raise ValueError("at least one trajectory checkpoint is required")
    return tuple(paths[index:] for index in range(len(paths) - 1, -1, -1))


def _validate_matching_states(states: tuple[Mapping[str, torch.Tensor], ...]) -> tuple[str, ...]:
    if not states:
        raise ValueError("at least one model state is required")
    keys = tuple(states[0])
    if not keys or any(tuple(state) != keys for state in states[1:]):
        raise ValueError("model state keys or order differ")
    for key in keys:
        first = states[0][key]
        if type(first) is not torch.Tensor:
            raise TypeError(f"model state value is not a tensor: {key}")
        if any(
            value.shape != first.shape or value.dtype != first.dtype
            for value in (state[key] for state in states[1:])
        ):
            raise ValueError(f"model state tensor metadata differs: {key}")
    return keys


def average_model_states(
    states: tuple[Mapping[str, torch.Tensor], ...],
) -> OrderedDict[str, torch.Tensor]:
    """Average floating tensors in FP64 and carry latest non-floating buffers."""

    keys = _validate_matching_states(states)
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        values = tuple(state[key].detach().cpu() for state in states)
        first = values[0]
        if first.is_floating_point():
            accumulator = torch.zeros_like(first, dtype=torch.float64)
            for value in values:
                accumulator.add_(value.to(torch.float64))
            result[key] = accumulator.div_(len(values)).to(first.dtype)
        else:
            result[key] = values[-1].clone()
    return result


def interpolate_model_states(
    initial: Mapping[str, torch.Tensor],
    soup: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> OrderedDict[str, torch.Tensor]:
    """Return initial + alpha * (soup - initial) with FP64 arithmetic."""

    if type(alpha) is not float or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("interpolation alpha must be a finite builtin float in [0, 1]")
    keys = _validate_matching_states((initial, soup))
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in keys:
        base = initial[key].detach().cpu()
        trained = soup[key].detach().cpu()
        if base.is_floating_point():
            mixed = base.to(torch.float64).add(
                trained.to(torch.float64).sub(base.to(torch.float64)), alpha=alpha
            )
            result[key] = mixed.to(base.dtype)
        else:
            result[key] = trained.clone()
    return result


def selected_model_payload(
    state: Mapping[str, torch.Tensor],
    selected: Mapping[str, object],
    training_protocol: Mapping[str, object],
) -> dict[str, object]:
    return {
        "model": state,
        "selection": selected,
        "training_protocol": training_protocol,
    }


def endpoint_model_payload(
    state: Mapping[str, torch.Tensor],
    endpoint: Mapping[str, object],
    training_protocol: Mapping[str, object],
) -> dict[str, object]:
    return {
        "model": state,
        "endpoint": endpoint,
        "training_protocol": training_protocol,
    }


def _epoch(path: Path) -> int:
    try:
        value = int(path.stem.removeprefix("epoch-"))
    except ValueError as error:
        raise ValueError(f"checkpoint name does not encode an epoch: {path}") from error
    if value <= 0 or path.stem != f"epoch-{value:04d}":
        raise ValueError(f"checkpoint name does not encode an epoch: {path}")
    return value


def _alpha_text(alpha: float) -> str:
    return format(alpha, ".12g")


def evaluate_grid(
    model: torch.nn.Module,
    initial: Mapping[str, torch.Tensor],
    checkpoints: tuple[tuple[Path, Mapping[str, torch.Tensor]], ...],
    *,
    alphas: tuple[float, ...],
    all_suffixes: bool = True,
    registered_specs: tuple[tuple[tuple[Path, ...], float], ...] | None = None,
    prepare: Callable[[], None] = lambda: None,
    evaluate: Callable[[], dict[str, float] | tuple[dict[str, float], dict[str, list[object]]]],
) -> list[dict[str, Any]]:
    """Evaluate every suffix soup and interpolation in a stable registered order."""

    if not checkpoints:
        raise ValueError("at least one trajectory checkpoint is required")
    paths = tuple(path for path, _state in checkpoints)
    epochs = tuple(_epoch(path) for path in paths)
    if tuple(sorted(paths, key=_epoch)) != paths or len(set(epochs)) != len(epochs):
        raise ValueError("trajectory checkpoints must have unique increasing epochs")
    if not alphas or any(
        type(alpha) is not float or not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
        for alpha in alphas
    ):
        raise ValueError("alphas must be finite builtin floats in [0, 1]")
    if tuple(sorted(set(alphas))) != alphas:
        raise ValueError("alphas must be unique increasing values")
    state_by_path = dict(checkpoints)
    candidates: list[dict[str, Any]] = []
    specs = (
        registered_specs
        if registered_specs is not None
        else tuple(
            (window, alpha)
            for window in (suffix_windows(paths) if all_suffixes else (paths,))
            for alpha in alphas
        )
    )
    for window, alpha in specs:
        epochs = tuple(_epoch(path) for path in window)
        soup = average_model_states(tuple(state_by_path[path] for path in window))
        state = interpolate_model_states(initial, soup, alpha=alpha)
        model.load_state_dict(state, strict=True)
        prepare()
        evaluation = evaluate()
        if isinstance(evaluation, tuple):
            metrics, query_evidence = evaluation
        else:
            metrics, query_evidence = evaluation, None
        if tuple(metrics) != (
            "recall_at_1",
            "recall_at_10",
            "recall_at_20",
            "recall_at_30",
            "map_at_r",
        ) and tuple(metrics) != ("recall_at_1", "map_at_r"):
            raise ValueError("candidate metric schema differs")
        candidate = {
            "name": (
                f"epochs-{'_'.join(str(epoch) for epoch in epochs)}-alpha-{_alpha_text(alpha)}"
            ),
            "epochs": list(epochs),
            "checkpoints": [str(path) for path in window],
            "alpha": alpha,
            "metrics": metrics,
        }
        if query_evidence is not None:
            _validate_query_evidence(query_evidence)
            candidate["query_evidence"] = query_evidence
        candidates.append(candidate)
    return candidates


def _validate_query_evidence(value: dict[str, list[object]]) -> None:
    if type(value) is not dict or tuple(value) != ("top1_correct", "average_precision"):
        raise ValueError("candidate query evidence schema differs")
    top1 = value["top1_correct"]
    average_precision = value["average_precision"]
    if (
        type(top1) is not list
        or type(average_precision) is not list
        or not top1
        or len(top1) != len(average_precision)
        or any(type(item) is not bool for item in top1)
        or any(
            type(item) is not float or not math.isfinite(item) or not 0.0 <= item <= 1.0
            for item in average_precision
        )
    ):
        raise ValueError("candidate query evidence differs")


def evaluate_holdout_with_query_evidence(
    trainer,
    model: torch.nn.Module,
    query,
    gallery,
    transform,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    selected_features: int,
) -> tuple[dict[str, float], dict[str, list[object]]]:
    query_values, query_labels = trainer._encode_records(
        model, query, transform, device=device, batch_size=batch_size, workers=workers
    )
    gallery_values, gallery_labels = trainer._encode_records(
        model, gallery, transform, device=device, batch_size=batch_size, workers=workers
    )
    result = trainer.retrieval_view(
        query_values,
        gallery_values,
        query_labels,
        gallery_labels,
        coordinates=np.arange(selected_features),
        normalize_before=True,
    )
    if result.average_precision is None:
        raise ValueError("retrieval evaluator omitted per-query average precision")
    metrics = {
        "recall_at_1": result.recall[1],
        "recall_at_10": result.recall[10],
        "recall_at_20": result.recall[20],
        "recall_at_30": result.recall[30],
        "map_at_r": result.map_at_r,
    }
    evidence = {
        "top1_correct": result.top1_correct.tolist(),
        "average_precision": result.average_precision.tolist(),
    }
    _validate_query_evidence(evidence)
    return metrics, evidence


def paired_candidate_gate(
    selected: dict[str, Any], endpoint: dict[str, Any], *, seed: int = 0, samples: int = 10_000
) -> dict[str, object]:
    for candidate in (selected, endpoint):
        _validate_query_evidence(candidate.get("query_evidence"))
    selected_evidence = selected["query_evidence"]
    endpoint_evidence = endpoint["query_evidence"]
    selected_ap = np.asarray(selected_evidence["average_precision"], dtype=np.float64)
    endpoint_ap = np.asarray(endpoint_evidence["average_precision"], dtype=np.float64)
    selected_top1 = np.asarray(selected_evidence["top1_correct"], dtype=np.float64)
    endpoint_top1 = np.asarray(endpoint_evidence["top1_correct"], dtype=np.float64)
    if selected_ap.shape != endpoint_ap.shape or selected_top1.shape != endpoint_top1.shape:
        raise ValueError("selected/endpoint query evidence differs")
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, selected_ap.size, size=(samples, selected_ap.size))
    map_replicates = (selected_ap - endpoint_ap)[indices].mean(axis=1)
    top1_replicates = (selected_top1 - endpoint_top1)[indices].mean(axis=1)
    map_delta = float((selected_ap - endpoint_ap).mean())
    top1_delta = float((selected_top1 - endpoint_top1).mean())
    return {
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
        "resampling_unit": "holdout_query_within_training_seed",
        "map_delta": map_delta,
        "map_delta_query_bootstrap_95_interval": [
            float(np.percentile(map_replicates, 2.5)),
            float(np.percentile(map_replicates, 97.5)),
        ],
        "recall_at_1_delta": top1_delta,
        "recall_at_1_delta_query_bootstrap_95_interval": [
            float(np.percentile(top1_replicates, 2.5)),
            float(np.percentile(top1_replicates, 97.5)),
        ],
        "promoted": map_delta >= 0.003 and top1_delta >= -0.00125,
    }


def summarize_training_seed_replications(
    screen_report_path: Path,
    report_paths: Sequence[Path],
) -> dict[str, object]:
    """Load and summarize six fixed replications independent of the seed-0 screen."""

    if not isinstance(screen_report_path, Path) or any(
        not isinstance(path, Path) for path in report_paths
    ):
        raise TypeError("training seed summary inputs must be paths")
    screen_report = json.loads(screen_report_path.read_text())
    reports = [json.loads(path.read_text()) for path in report_paths]
    return _summarize_training_seed_replication_payloads(
        screen_report,
        reports,
        screen_report_sha256=_sha256_file(screen_report_path),
    )


def _summarize_training_seed_replication_payloads(
    screen_report: Mapping[str, Any],
    reports: Sequence[Mapping[str, Any]],
    *,
    screen_report_sha256: str,
) -> dict[str, object]:
    """Summarize six fixed replications independent of the seed-0 screen."""

    if (
        type(screen_report_sha256) is not str
        or len(screen_report_sha256) != 64
        or any(character not in "0123456789abcdef" for character in screen_report_sha256)
    ):
        raise ValueError("seed-0 screen report SHA-256 differs")
    if len(reports) != len(REGISTERED_REPLICATION_SEEDS):
        raise ValueError("paired training seeds differ from frozen protocol")
    if any(type(report) is not dict for report in reports):
        raise ValueError("paired training seed report differs from frozen protocol")
    seeds = [report["training_seed"] for report in reports]
    if seeds != list(REGISTERED_REPLICATION_SEEDS):
        raise ValueError("paired training seeds differ from frozen protocol")
    if (
        type(screen_report) is not dict
        or screen_report.get("protocol") != REPORT_PROTOCOL
        or screen_report.get("mode") != "screen"
        or screen_report.get("training_seed") != 0
        or screen_report.get("alphas") != list(REGISTERED_SCREEN_ALPHAS)
        or screen_report.get("batch_size") != 128
        or screen_report.get("workers") != 4
        or screen_report.get("selected_features") != 512
        or screen_report.get("screen_report") is not None
        or screen_report.get("screen_report_sha256") is not None
        or screen_report.get("holdout_seed") != 0
        or screen_report.get("holdout_fraction") != 0.2
    ):
        raise ValueError("seed-0 screen report differs from frozen protocol")
    screen_training_protocol = screen_report.get("training_protocol")
    _validate_registered_training_protocol(screen_training_protocol, seed=0)
    expected_training_protocol = dict(screen_training_protocol)
    expected_training_protocol.pop("seed")
    screen_selected = screen_report.get("selected")
    screen_endpoint = screen_report.get("endpoint")
    screen_candidates = screen_report.get("candidates")
    screen_gate = screen_report.get("gate")
    if (
        type(screen_selected) is not dict
        or type(screen_endpoint) is not dict
        or type(screen_candidates) is not list
        or screen_selected not in screen_candidates
        or screen_endpoint not in screen_candidates
        or select_candidate(screen_candidates) != screen_selected
        or type(screen_gate) is not dict
        or screen_gate.get("resampling_unit") != "holdout_query_within_training_seed"
        or screen_gate.get("promoted") is not True
    ):
        raise ValueError("seed-0 screen winner differs")
    screen_specs = tuple(
        (tuple(_candidate_identity(candidate)["epochs"]), candidate["alpha"])
        for candidate in screen_candidates
    )
    expected_screen_specs = tuple(
        (epochs, alpha)
        for epochs in REGISTERED_SCREEN_WINDOWS
        for alpha in REGISTERED_SCREEN_ALPHAS
    )
    if screen_specs != expected_screen_specs:
        raise ValueError("seed-0 screen winner differs")
    expected_winner = _candidate_identity(screen_selected)
    if _candidate_identity(screen_endpoint) != {"epochs": [16], "alpha": 1.0}:
        raise ValueError("seed-0 screen winner differs")
    _validate_reported_delta(screen_selected, screen_endpoint, screen_gate)
    expected_query_count = len(screen_selected["query_evidence"]["top1_correct"])

    map_deltas: list[float] = []
    recall_deltas: list[float] = []
    expected_screen_path: str | None = None
    prior_seed_checkpoint_hashes: set[str] = set()
    for seed, report in zip(REGISTERED_REPLICATION_SEEDS, reports, strict=True):
        if (
            report.get("alphas") != [expected_winner["alpha"]]
            or report.get("batch_size") != 128
            or report.get("workers") != 4
            or report.get("selected_features") != 512
            or report.get("holdout_seed") != 0
            or report.get("holdout_fraction") != 0.2
        ):
            raise ValueError("paired training seed evaluation protocol differs")
        if (
            report.get("protocol") != REPORT_PROTOCOL
            or report.get("mode") != "fixed"
            or report.get("screen_report_sha256") != screen_report_sha256
            or type(report.get("screen_report")) is not str
            or not report["screen_report"]
        ):
            raise ValueError("paired training seed screen report differs")
        if expected_screen_path is None:
            expected_screen_path = report["screen_report"]
        elif report["screen_report"] != expected_screen_path:
            raise ValueError("paired training seed screen report differs")
        training_protocol = report.get("training_protocol")
        _validate_registered_training_protocol(training_protocol, seed=seed)
        normalized_protocol = dict(training_protocol)
        normalized_protocol.pop("seed")
        if normalized_protocol != expected_training_protocol:
            raise ValueError("paired training protocol differs")
        selected = report.get("selected")
        endpoint = report.get("endpoint")
        candidates = report.get("candidates")
        if (
            type(selected) is not dict
            or type(endpoint) is not dict
            or type(candidates) is not list
            or candidates != [selected, endpoint]
        ):
            raise ValueError("paired training seed winner differs")
        winner = _candidate_identity(selected)
        if winner != expected_winner or _candidate_identity(endpoint) != {
            "epochs": [16],
            "alpha": 1.0,
        }:
            raise ValueError("paired training seed winner differs")
        seed_checkpoints = set(_candidate_checkpoint_evidence(selected)) | set(
            _candidate_checkpoint_evidence(endpoint)
        )
        try:
            seed_checkpoint_hashes = {
                _sha256_file(Path(checkpoint)) for checkpoint in seed_checkpoints
            }
        except OSError as error:
            raise ValueError("paired training seed checkpoint evidence differs") from error
        if seed_checkpoint_hashes & prior_seed_checkpoint_hashes:
            raise ValueError("paired training seed checkpoint evidence differs")
        prior_seed_checkpoint_hashes.update(seed_checkpoint_hashes)
        gate = report.get("gate")
        if (
            type(gate) is not dict
            or gate.get("resampling_unit") != "holdout_query_within_training_seed"
        ):
            raise ValueError("paired training seed query evidence differs")
        map_delta, recall_delta = _validate_reported_delta(selected, endpoint, gate)
        if len(selected["query_evidence"]["top1_correct"]) != expected_query_count:
            raise ValueError("paired training seed query count differs")
        map_deltas.append(map_delta)
        recall_deltas.append(recall_delta)

    map_values = np.asarray(map_deltas, dtype=np.float64)
    mean = float(map_values.mean())
    sample_standard_deviation = float(map_values.std(ddof=1))
    half_width = STUDENT_T_CRITICAL_TWO_SIDED_95_DF5 * sample_standard_deviation / math.sqrt(6)
    lower = mean - half_width
    upper = mean + half_width
    positives = sum(delta > 0.0 for delta in map_deltas)
    negatives = sum(delta < 0.0 for delta in map_deltas)
    nonzero = positives + negatives
    tail = min(positives, negatives)
    sign_p_value = min(
        1.0,
        2.0 * sum(math.comb(nonzero, count) for count in range(tail + 1)) / (2**nonzero),
    )
    all_map_positive = positives == len(REGISTERED_REPLICATION_SEEDS)
    recall_guard = all(delta >= -0.00125 for delta in recall_deltas)
    nondegenerate_variation = sample_standard_deviation > 0.0
    claim_supported = (
        all_map_positive
        and lower > 0.0
        and sign_p_value <= 0.05
        and recall_guard
        and nondegenerate_variation
    )
    return {
        "training_seeds": list(REGISTERED_REPLICATION_SEEDS),
        "resampling_unit": "paired_training_seed",
        "winner": expected_winner,
        "map_deltas": map_deltas,
        "map_delta_mean": mean,
        "map_delta_sample_standard_deviation": sample_standard_deviation,
        "student_t_degrees_of_freedom": 5,
        "student_t_critical_two_sided_95": STUDENT_T_CRITICAL_TWO_SIDED_95_DF5,
        "map_delta_paired_student_t_95_interval": [lower, upper],
        "exact_two_sided_sign_p_value": sign_p_value,
        "recall_at_1_deltas": recall_deltas,
        "recall_at_1_guard": -0.00125,
        "all_map_deltas_positive": all_map_positive,
        "all_recall_at_1_deltas_above_guard": recall_guard,
        "nondegenerate_training_seed_variation": nondegenerate_variation,
        "claim_supported": claim_supported,
    }


def _registered_training_protocol(seed: int) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": REGISTERED_TRAINER_SHA256,
        "unicom_revision": REGISTERED_UNICOM_REVISION,
        "initial_checkpoint_sha256": REGISTERED_INITIAL_CHECKPOINT_SHA256,
        "partition_sha256": REGISTERED_PARTITION_SHA256,
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
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
    }


def _validate_registered_training_protocol(value: object, *, seed: int) -> None:
    expected = _registered_training_protocol(seed)
    if (
        type(value) is not dict
        or tuple(value) != tuple(expected)
        or any(
            type(value[key]) is not type(expected_value) for key, expected_value in expected.items()
        )
        or value != expected
    ):
        raise ValueError("paired training protocol differs")


def _candidate_identity(candidate: Mapping[str, Any]) -> dict[str, object]:
    epochs = candidate.get("epochs")
    alpha = candidate.get("alpha")
    if (
        type(epochs) is not list
        or not epochs
        or any(type(epoch) is not int for epoch in epochs)
        or type(alpha) is not float
        or not math.isfinite(alpha)
    ):
        raise ValueError("paired training seed winner differs")
    return {"epochs": epochs, "alpha": alpha}


def _candidate_checkpoint_evidence(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    epochs = candidate.get("epochs")
    checkpoints = candidate.get("checkpoints")
    if (
        type(epochs) is not list
        or type(checkpoints) is not list
        or len(checkpoints) != len(epochs)
        or any(type(path) is not str or not path for path in checkpoints)
        or len(set(checkpoints)) != len(checkpoints)
        or any(_epoch(Path(path)) != epoch for epoch, path in zip(epochs, checkpoints, strict=True))
    ):
        raise ValueError("paired training seed checkpoint evidence differs")
    return tuple(checkpoints)


def _validate_reported_delta(
    selected: dict[str, Any], endpoint: dict[str, Any], gate: dict[str, Any]
) -> tuple[float, float]:
    for candidate in (selected, endpoint):
        _validate_query_evidence(candidate.get("query_evidence"))
        metrics = candidate.get("metrics")
        if type(metrics) is not dict:
            raise ValueError("paired training seed candidate metrics differ")
        evidence = candidate["query_evidence"]
        expected_map = float(np.asarray(evidence["average_precision"], dtype=np.float64).mean())
        expected_recall = float(np.asarray(evidence["top1_correct"], dtype=np.float64).mean())
        if (
            type(metrics.get("map_at_r")) is not float
            or type(metrics.get("recall_at_1")) is not float
            or abs(metrics["map_at_r"] - expected_map) > 1e-12
            or abs(metrics["recall_at_1"] - expected_recall) > 1e-12
        ):
            raise ValueError("paired training seed candidate metrics differ")
    selected_evidence = selected["query_evidence"]
    endpoint_evidence = endpoint["query_evidence"]
    selected_ap = np.asarray(selected_evidence["average_precision"], dtype=np.float64)
    endpoint_ap = np.asarray(endpoint_evidence["average_precision"], dtype=np.float64)
    selected_top1 = np.asarray(selected_evidence["top1_correct"], dtype=np.float64)
    endpoint_top1 = np.asarray(endpoint_evidence["top1_correct"], dtype=np.float64)
    if selected_ap.shape != endpoint_ap.shape or selected_top1.shape != endpoint_top1.shape:
        raise ValueError("paired training seed query evidence differs")
    map_delta = float((selected_ap - endpoint_ap).mean())
    recall_delta = float((selected_top1 - endpoint_top1).mean())
    if (
        type(gate.get("map_delta")) is not float
        or type(gate.get("recall_at_1_delta")) is not float
        or abs(gate["map_delta"] - map_delta) > 1e-12
        or abs(gate["recall_at_1_delta"] - recall_delta) > 1e-12
        or gate.get("promoted") is not (map_delta >= 0.003 and recall_delta >= -0.00125)
    ):
        raise ValueError("paired training seed delta differs from recomputed evidence")
    return map_delta, recall_delta


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidate list is empty")
    return max(
        candidates,
        key=lambda row: (row["metrics"]["map_at_r"], row["metrics"]["recall_at_1"]),
    )


def _load_trainer():
    path = Path(__file__).resolve().with_name("train_unicom_inshop.py")
    spec = importlib.util.spec_from_file_location("_unicom_trainer_for_soup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load UNICOM trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_checkpoint_states(
    paths: tuple[Path, ...],
    *,
    holdout_seed: int,
    holdout_fraction: float,
) -> tuple[
    tuple[tuple[Path, Mapping[str, torch.Tensor]], ...],
    dict[str, object],
]:
    result: list[tuple[Path, Mapping[str, torch.Tensor]]] = []
    training_protocol: dict[str, object] | None = None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if type(checkpoint) is not dict or checkpoint.get("epoch") != _epoch(path):
            raise ValueError(f"training checkpoint epoch differs: {path}")
        if checkpoint.get("selection_holdout") != {
            "seed": holdout_seed,
            "fraction": holdout_fraction,
        }:
            raise ValueError(f"training checkpoint selection holdout differs: {path}")
        checkpoint_protocol = checkpoint.get("training_protocol")
        if type(checkpoint_protocol) is not dict or not checkpoint_protocol:
            raise ValueError(f"training checkpoint training protocol differs: {path}")
        if training_protocol is None:
            training_protocol = checkpoint_protocol
        elif checkpoint_protocol != training_protocol:
            raise ValueError(f"training checkpoint training protocol differs: {path}")
        state = checkpoint.get("model")
        if type(state) not in (dict, OrderedDict):
            raise ValueError(f"training checkpoint model state differs: {path}")
        result.append((path, state))
    if training_protocol is None:
        raise ValueError("at least one trajectory checkpoint is required")
    return tuple(result), training_protocol


def _expected_screen_training_protocol(trainer, args) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": trainer._sha256_file(Path(trainer.__file__)),
        "unicom_revision": trainer.UNICOM_REVISION,
        "initial_checkpoint_sha256": trainer.UNICOM_L14_336_SHA256,
        "partition_sha256": trainer._sha256_file(
            args.dataset_root / "Eval" / "list_eval_partition.txt"
        ),
        "seed": args.training_seed,
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
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
    }


def _validate_screen_training_protocol(training_protocol, trainer, args) -> None:
    if training_protocol != _expected_screen_training_protocol(trainer, args):
        raise ValueError("training checkpoint protocol differs from frozen screen")


def _validate_selection_args(args) -> None:
    if (
        args.holdout_seed != 0
        or args.holdout_fraction != 0.2
        or args.batch_size != 128
        or args.workers != 4
        or args.selected_features != 512
    ):
        raise ValueError("selection data/evaluation settings differ from frozen protocol")
    epochs = tuple(_epoch(path) for path in args.trajectory_checkpoints)
    alphas = tuple(args.alphas)
    if args.mode == "screen":
        if args.screen_report is not None:
            raise ValueError("screen mode must not consume a prior screen report")
        if args.training_seed != 0 or epochs != (4, 8, 12, 16):
            raise ValueError("screen seed/checkpoints differ from frozen protocol")
        if alphas != REGISTERED_SCREEN_ALPHAS:
            raise ValueError("screen alpha grid differs from frozen protocol")
    elif args.mode == "fixed":
        if (
            args.training_seed not in REGISTERED_REPLICATION_SEEDS
            or len(alphas) != 1
            or not isinstance(args.screen_report, Path)
            or epochs != (4, 8, 12, 16)
        ):
            raise ValueError("fixed replication seed/alpha differs from frozen protocol")
        screen = json.loads(args.screen_report.read_text())
        if (
            type(screen) is not dict
            or screen.get("protocol") != REPORT_PROTOCOL
            or screen.get("mode") != "screen"
            or screen.get("training_seed") != 0
            or screen.get("alphas") != list(REGISTERED_SCREEN_ALPHAS)
            or screen.get("batch_size") != 128
            or screen.get("workers") != 4
            or screen.get("selected_features") != 512
            or screen.get("holdout_seed") != 0
            or screen.get("holdout_fraction") != 0.2
            or type(screen.get("selected")) is not dict
            or type(screen.get("candidates")) is not list
            or not screen["candidates"]
            or type(screen.get("gate")) is not dict
            or screen["gate"].get("promoted") is not True
        ):
            raise ValueError("fixed replication screen report differs")
        selected = screen["selected"]
        if (
            selected not in screen["candidates"]
            or select_candidate(screen["candidates"]) != selected
        ):
            raise ValueError("fixed replication screen winner differs")
        valid_windows = ([16], [12, 16], [8, 12, 16], [4, 8, 12, 16])
        if selected.get("epochs") not in valid_windows or selected.get("alpha") != alphas[0]:
            raise ValueError("fixed replication differs from screen winner")
    else:
        raise ValueError("selection mode differs")


def recalibrate_batch_norm(model: torch.nn.Module, loader, *, device: torch.device) -> None:
    """Recompute candidate BatchNorm statistics on optimization identities only."""

    batch_norms = tuple(
        module
        for module in model.modules()
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        )
    )
    if not batch_norms:
        model.eval()
        return
    momenta = tuple(module.momentum for module in batch_norms)
    model.eval()
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    batches = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(images.to(device, non_blocking=True))
                batches += 1
        if batches == 0:
            raise ValueError("BatchNorm recalibration loader is empty")
    finally:
        for module, momentum in zip(batch_norms, momenta, strict=True):
            module.momentum = momentum
        model.eval()


def recalibrated_model_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    loader,
    *,
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    """Load one candidate and return its post-recalibration state on CPU."""

    model.load_state_dict(state, strict=True)
    recalibrate_batch_norm(model, loader, device=device)
    return OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
    )


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"selection output already exists: {path}")
    try:
        with temporary.open("xb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(value: object, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"selection output already exists: {path}")
    payload = (json.dumps(value, indent=2) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from sfora.unicom_inshop import parse_inshop_partition
    from sfora.unicom_training import identity_holdout

    _validate_selection_args(args)
    trainer = _load_trainer()
    model_path = args.output_dir / "selected-model.pt"
    endpoint_model_path = args.output_dir / "endpoint-model.pt"
    report_path = args.output_dir / "selection.json"
    for path in (model_path, endpoint_model_path, report_path):
        if path.exists() or path.with_name(f"{path.name}.tmp").exists():
            raise FileExistsError(f"selection output already exists: {path}")
    if trainer._git_revision(args.unicom_checkout) != trainer.UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if trainer._sha256_file(args.initial_checkpoint) != trainer.UNICOM_L14_336_SHA256:
        raise ValueError("UNICOM initial checkpoint SHA-256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UNICOM soup evaluation")
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(record for record in records if record.split == "train")
    optimization, query, gallery, _labels = identity_holdout(
        train_records, fraction=args.holdout_fraction, seed=args.holdout_seed
    )
    if not query or not gallery:
        raise ValueError("soup selection requires a nonempty train-only holdout")
    model, transform = trainer._load_official_model(args.unicom_checkout, args.initial_checkpoint)
    initial = OrderedDict(
        (name, value.detach().cpu().clone()) for name, value in model.state_dict().items()
    )
    device = torch.device("cuda")
    model = model.to(device)
    checkpoints, training_protocol = _load_checkpoint_states(
        tuple(args.trajectory_checkpoints),
        holdout_seed=args.holdout_seed,
        holdout_fraction=args.holdout_fraction,
    )
    _validate_screen_training_protocol(training_protocol, trainer, args)
    registered_specs = None
    if args.mode == "fixed":
        screen_selected = json.loads(args.screen_report.read_text())["selected"]
        path_by_epoch = {_epoch(path): path for path, _state in checkpoints}
        winner_paths = tuple(path_by_epoch[epoch] for epoch in screen_selected["epochs"])
        registered_specs = (
            (winner_paths, screen_selected["alpha"]),
            ((path_by_epoch[16],), 1.0),
        )
    calibration_loader = torch.utils.data.DataLoader(
        trainer.InshopEvalDataset(optimization, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    candidates = evaluate_grid(
        model,
        initial,
        checkpoints,
        alphas=tuple(args.alphas),
        all_suffixes=args.mode == "screen",
        registered_specs=registered_specs,
        prepare=lambda: recalibrate_batch_norm(model, calibration_loader, device=device),
        evaluate=lambda: evaluate_holdout_with_query_evidence(
            trainer,
            model,
            query,
            gallery,
            transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            selected_features=args.selected_features,
        ),
    )
    selected = select_candidate(candidates) if args.mode == "screen" else candidates[0]
    endpoint = next(
        (
            candidate
            for candidate in candidates
            if candidate["epochs"] == [16] and candidate["alpha"] == 1.0
        ),
        None,
    )
    if endpoint is None:
        raise ValueError("registered endpoint candidate is absent")
    gate = paired_candidate_gate(selected, endpoint)
    checkpoint_by_epoch = {_epoch(path): state for path, state in checkpoints}
    soup = average_model_states(tuple(checkpoint_by_epoch[epoch] for epoch in selected["epochs"]))
    selected_state = interpolate_model_states(initial, soup, alpha=selected["alpha"])
    selected_state = recalibrated_model_state(
        model, selected_state, calibration_loader, device=device
    )
    endpoint_state = recalibrated_model_state(
        model, checkpoint_by_epoch[16], calibration_loader, device=device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        selected_model_payload(selected_state, selected, training_protocol),
        model_path,
    )
    _atomic_torch_save(
        endpoint_model_payload(endpoint_state, endpoint, training_protocol),
        endpoint_model_path,
    )
    report = {
        "protocol": REPORT_PROTOCOL,
        "mode": args.mode,
        "training_seed": args.training_seed,
        "alphas": list(args.alphas),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "selected_features": args.selected_features,
        "screen_report": None if args.screen_report is None else str(args.screen_report),
        "screen_report_sha256": (
            None if args.screen_report is None else _sha256_file(args.screen_report)
        ),
        "holdout_seed": args.holdout_seed,
        "holdout_fraction": args.holdout_fraction,
        "training_protocol": training_protocol,
        "batch_norm_recalibration": "full-optimization-cumulative-batches",
        "selected": selected,
        "endpoint": endpoint,
        "gate": gate,
        "candidates": candidates,
        "model_path": str(model_path),
    }
    _atomic_json(report, report_path)
    return report


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trajectory-checkpoints", required=True, type=Path, nargs="+")
    parser.add_argument("--alphas", required=True, type=float, nargs="+")
    parser.add_argument("--mode", required=True, choices=("screen", "fixed"))
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--screen-report", type=Path)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selected-features", type=int, default=512)
    return parser.parse_args(arguments)


def parse_aggregate_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true", required=True)
    parser.add_argument("--screen-report", required=True, type=Path)
    parser.add_argument("--replication-reports", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if arguments is None else arguments)
    if values and values[0] == "--aggregate":
        try:
            aggregate_args = parse_aggregate_args(values)
            summary = summarize_training_seed_replications(
                aggregate_args.screen_report, aggregate_args.replication_reports
            )
            _atomic_json(summary, aggregate_args.output)
        except Exception as error:
            print(f"soup aggregation failed: {error}", file=sys.stderr)
            return 2
        print(json.dumps(summary, sort_keys=True))
        return 0
    try:
        report = run(parse_args(values))
    except Exception as error:
        print(f"soup evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report["selected"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
