#!/usr/bin/env python3
"""Prospective Pass 205 RDGC no-training diagnostic.

The module is deliberately import-safe: Torch is supplied explicitly to all
tensor operations, so provenance failures can be represented without importing
or probing Torch.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import types
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

EPSILON = 1.0e-8
NORM_FLOOR = 1.0e-12
BOOTSTRAP_SEED = 201
BOOTSTRAP_REPLICATES = 10_000
CONTRIBUTOR_COUNTS = (1, 8, 32, 180)
CONTROL_ORDER = (
    "raw_cotangent",
    "full_motion",
    "batch_global_gain",
    "scalar_diagonal_raw",
    "per_example_gradient_normalized",
    "layerwise_trust_ratio",
)
PENALTY_OPERATOR_ORDER = (
    "rdgc",
    "raw_cotangent",
    "full_motion",
    "batch_global_gain",
    "scalar_diagonal_raw",
    "per_example_gradient_normalized",
)
CORRECTION_ORDER = ("rdgc", *CONTROL_ORDER)
OPERATOR_ORDER = ("pa", *CORRECTION_ORDER)
RESULT_KEYS = (
    "schema_version",
    "diagnostic",
    "mode",
    "status",
    "phase_reached",
    "candidate_values_computed",
    "training_performed",
    "benchmark_authorized",
    "scope_limitation",
    "authority",
    "source",
    "execution",
    "environment",
    "binding",
    "integrity",
    "selection",
    "preliminary",
    "panel",
    "bootstrap",
    "decision",
)
SCOPE_LIMITATION = (
    "Euclidean BN-Inception/Proxy-Anchor parameter tangent; invariant only to "
    "the registered common descriptor rotation, not hidden-layer rescaling or "
    "AdamW preconditioning"
)
CANDIDATE_PATH = "docs/pass205_rdgc_candidate_2026-08-10.md"
CANDIDATE_COMMIT = "30d533e532d0f22c8b1e474987001685a4aa3488"
CANDIDATE_SHA256 = "2a86f11f8d6a4563610b0585db74c372903bdbf7deabd580fa929114fda2af0f"
PLAN_PATH = "docs/superpowers/plans/2026-08-10-pass205-rdgc-stage-b.md"
PLAN_COMMIT = "c1e49b13c08f853ae17d5b8b48be1aa7b8a4bc11"
PLAN_SHA256 = "20915982228bd4a17f1260952fe184d9e09b27b9b28165b5931bad843872c7ed"
RDGC_SOURCE_ORDER = (
    "scripts/diagnose_pass159_cotangent_stage_a.py",
    "scripts/diagnose_pass200_rsta_stage_a.py",
    "scripts/diagnose_pass205_rdgc_stage_b.py",
    "scripts/rsta_normwise_adjoint.py",
    "scripts/verify_pass200_rsta_scientific_artifact.py",
    "src/sfora/__init__.py",
    "src/sfora/ablation.py",
    "src/sfora/api.py",
    "src/sfora/arcg.py",
    "src/sfora/benchmark.py",
    "src/sfora/bn_inception.py",
    "src/sfora/catalog.py",
    "src/sfora/cea.py",
    "src/sfora/cem.py",
    "src/sfora/cli.py",
    "src/sfora/compose.py",
    "src/sfora/data.py",
    "src/sfora/encoder_ablation.py",
    "src/sfora/encoder_training.py",
    "src/sfora/evaluation.py",
    "src/sfora/experiments.py",
    "src/sfora/image_benchmark.py",
    "src/sfora/image_end_to_end.py",
    "src/sfora/image_recipes.py",
    "src/sfora/ipsr.py",
    "src/sfora/losses.py",
    "src/sfora/method.py",
    "src/sfora/oapf.py",
    "src/sfora/publication.py",
    "src/sfora/remote.py",
    "src/sfora/report.py",
    "src/sfora/text_baselines.py",
    "src/sfora/training.py",
)


def _require_fp32_finite(torch_module: Any, value: Any, *, nonzero: bool = False) -> None:
    if value.dtype != torch_module.float32:
        raise TypeError("tensor must be FP32")
    if not bool(torch_module.isfinite(value).all().item()):
        raise ValueError("tensor must be finite")
    if nonzero and not bool(value.norm().item() > NORM_FLOOR):
        raise ValueError("tensor norm must exceed the registered floor")


def rdgc_error(torch_module: Any, b: Any, s: Any, *, epsilon: float = EPSILON) -> Any:
    _require_fp32_finite(torch_module, b, nonzero=True)
    _require_fp32_finite(torch_module, s, nonzero=True)
    return torch_module.log((b.norm() + epsilon) / (s.norm().detach() + epsilon))


def rdgc_penalty(torch_module: Any, b: Any, s: Any, *, epsilon: float = EPSILON) -> Any:
    error = rdgc_error(torch_module, b, s, epsilon=epsilon)
    return 0.5 * error * error


def fp64_named_norm(torch_module: Any, values: tuple[Any, ...]) -> Any:
    if type(values) is not tuple or not values:
        raise TypeError("values must be a nonempty tuple")
    total = torch_module.zeros((), dtype=torch_module.float64, device=values[0].device)
    for value in values:
        _require_fp32_finite(torch_module, value)
        flat = value.to(torch_module.float64).reshape(-1)
        total = total + (flat * flat).sum()
    return total.sqrt()


def fp64_named_dot(torch_module: Any, left: tuple[Any, ...], right: tuple[Any, ...]) -> Any:
    if type(left) is not tuple or type(right) is not tuple or not left or len(left) != len(right):
        raise TypeError("named tuples must be nonempty and aligned")
    total = torch_module.zeros((), dtype=torch_module.float64, device=left[0].device)
    for lhs, rhs in zip(left, right, strict=True):
        _require_fp32_finite(torch_module, lhs)
        _require_fp32_finite(torch_module, rhs)
        if lhs.shape != rhs.shape or lhs.device != rhs.device:
            raise ValueError("named tensors must have identical shape and device")
        total = (
            total
            + (
                lhs.to(torch_module.float64).reshape(-1) * rhs.to(torch_module.float64).reshape(-1)
            ).sum()
        )
    return total


def _half_squared_log_ratio(torch_module: Any, numerator: Any, target: Any) -> Any:
    error = torch_module.log((numerator + EPSILON) / (target.detach() + EPSILON))
    return 0.5 * error * error


def control_penalties(
    torch_module: Any,
    *,
    b: Any,
    s: Any,
    dbar: Any,
    receiver_fields: list[dict[str, Any]],
    pgn_motion: Any,
) -> dict[str, Any]:
    for tensor in (b, s, dbar, pgn_motion):
        _require_fp32_finite(torch_module, tensor, nonzero=True)
    if type(receiver_fields) is not list or len(receiver_fields) != 8:
        raise ValueError("receiver_fields must contain exactly eight records")
    b_norm = b.norm()
    s_norm = s.norm()
    dbar_norm = dbar.norm()
    raw_cosine = torch_module.sum(b.reshape(-1) * dbar.detach().reshape(-1)) / (
        b_norm * dbar_norm.detach() + EPSILON
    )
    full_motion = 0.5 * (b_norm * b_norm) / (b_norm * b_norm + EPSILON * EPSILON).detach()
    self_norms: list[Any] = []
    raw_norms: list[Any] = []
    for field in receiver_fields:
        if tuple(field) != ("s", "dbar"):
            raise ValueError("receiver field schema/order mismatch")
        for tensor in (field["s"], field["dbar"]):
            _require_fp32_finite(torch_module, tensor, nonzero=True)
        self_norms.append(field["s"].norm().to(torch_module.float64))
        raw_norms.append(field["dbar"].norm().to(torch_module.float64))
    global_target = torch_module.exp(
        torch_module.stack([torch_module.log(value + EPSILON) for value in self_norms]).mean()
    ).to(torch_module.float32)
    gains = [
        torch_module.log((self_norm + EPSILON) / (raw_norm + EPSILON))
        for self_norm, raw_norm in zip(self_norms, raw_norms, strict=True)
    ]
    scalar_gain = torch_module.exp(torch_module.stack(gains).mean()).to(torch_module.float32)
    scalar_raw_target = scalar_gain * (dbar_norm.detach() + EPSILON)
    return {
        "rdgc": rdgc_penalty(torch_module, b, s),
        "raw_cotangent": 1.0 - raw_cosine,
        "full_motion": full_motion,
        "batch_global_gain": _half_squared_log_ratio(torch_module, b_norm, global_target),
        "scalar_diagonal_raw": _half_squared_log_ratio(torch_module, b_norm, scalar_raw_target),
        "per_example_gradient_normalized": _half_squared_log_ratio(
            torch_module, pgn_motion.norm(), s_norm
        ),
    }


def pgn_detached_coefficients(
    torch_module: Any, first_order_gradient_norms: tuple[Any, ...]
) -> Any:
    if type(first_order_gradient_norms) is not tuple or len(first_order_gradient_norms) != 180:
        raise ValueError("exactly 180 gradient norms are required")
    detached: list[Any] = []
    for value in first_order_gradient_norms:
        if value.dtype != torch_module.float64 or value.numel() != 1:
            raise TypeError("gradient norms must be scalar FP64 tensors")
        if not bool(torch_module.isfinite(value).item()) or not bool(value.item() >= 0.0):
            raise ValueError("gradient norms must be finite and nonnegative")
        detached.append(value.detach())
    stacked = torch_module.stack(detached)
    nu = torch_module.exp(torch_module.log(stacked + NORM_FLOOR).mean())
    return (nu / (stacked + NORM_FLOOR)).to(torch_module.float32).detach()


def pgn_weighted_cotangent(torch_module: Any, dbar: Any, coefficients: Any) -> Any:
    _require_fp32_finite(torch_module, dbar)
    _require_fp32_finite(torch_module, coefficients)
    if dbar.ndim < 1 or coefficients.shape != (dbar.shape[0],):
        raise ValueError("coefficient order/shape mismatch")
    shape = (coefficients.shape[0],) + (1,) * (dbar.ndim - 1)
    return dbar * coefficients.reshape(shape)


def layerwise_trust_ratio_direction(
    torch_module: Any, named_parameters: tuple[tuple[str, Any], ...], p: tuple[Any, ...]
) -> tuple[Any, ...]:
    if (
        type(named_parameters) is not tuple
        or type(p) is not tuple
        or len(named_parameters) != len(p)
        or not p
    ):
        raise TypeError("named parameters and direction must be aligned tuples")
    groups: dict[str, list[int]] = {}
    for index, ((name, parameter), direction) in enumerate(zip(named_parameters, p, strict=True)):
        if type(name) is not str or not name:
            raise TypeError("parameter names must be nonempty strings")
        _require_fp32_finite(torch_module, parameter)
        _require_fp32_finite(torch_module, direction)
        if parameter.shape != direction.shape or parameter.device != direction.device:
            raise ValueError("parameter and direction must align")
        group = name.rsplit(".", 1)[0] if "." in name else name
        groups.setdefault(group, []).append(index)
    result: list[Any | None] = [None] * len(p)
    for indices in groups.values():
        parameter_norm = fp64_named_norm(
            torch_module, tuple(named_parameters[i][1] for i in indices)
        )
        direction_norm = fp64_named_norm(torch_module, tuple(p[i] for i in indices))
        tau = (parameter_norm / (direction_norm + NORM_FLOOR)).detach().to(torch_module.float32)
        for index in indices:
            result[index] = p[index] * tau
    return tuple(result)  # type: ignore[arg-type]


def normalize_virtual_updates(
    torch_module: Any,
    p: tuple[Any, ...],
    corrections: dict[str, tuple[Any, ...]],
    *,
    alpha: float = 0.10,
) -> dict[str, tuple[Any, ...]]:
    p_norm = fp64_named_norm(torch_module, p)
    if not bool(p_norm.item() > NORM_FLOOR):
        raise ValueError("ordinary PA direction must have nonzero norm")
    if tuple(corrections) != tuple(name for name in CORRECTION_ORDER if name in corrections):
        raise ValueError("corrections are not in registered order")
    updates: dict[str, tuple[Any, ...]] = {"pa": tuple(value.clone() for value in p)}
    for name, correction in corrections.items():
        if type(correction) is not tuple or len(correction) != len(p):
            raise ValueError("correction tree mismatch")
        for base, delta in zip(p, correction, strict=True):
            _require_fp32_finite(torch_module, delta)
            if base.shape != delta.shape or base.device != delta.device:
                raise ValueError("correction tree mismatch")
        correction_norm = fp64_named_norm(torch_module, correction)
        if not bool(correction_norm.item() > NORM_FLOOR):
            raise ValueError("correction norm must be nonzero")
        c_hat = tuple(
            value * (p_norm / correction_norm).to(torch_module.float32) for value in correction
        )
        raw = tuple(base + alpha * delta for base, delta in zip(p, c_hat, strict=True))
        raw_norm = fp64_named_norm(torch_module, raw)
        if not bool(raw_norm.item() > NORM_FLOOR):
            raise ValueError("virtual direction norm must be nonzero")
        scale = (p_norm / raw_norm).to(torch_module.float32)
        updates[name] = tuple(value * scale for value in raw)
    return updates


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and type(value) is not bool and math.isfinite(float(value))


def decide_preliminary(aggregates: dict[str, object]) -> dict[str, object]:
    required = (
        "count_gain_median_A",
        "count_gain_median_B",
        "positive_count_gain_seed_means_A",
        "positive_count_gain_seed_means_B",
        "context_spearman_median",
        "context_spearman_seeds_ge_point_three",
        "log_kappa_iqr_A",
        "log_kappa_iqr_B",
        "global_scalar_relative_error_median_A",
        "global_scalar_relative_error_median_B",
        "full_gain_error_median_A",
        "full_gain_error_median_B",
        "full_gain_error_seed_medians_ge_log_one_point_one_A",
        "full_gain_error_seed_medians_ge_log_one_point_one_B",
    )
    if tuple(aggregates) != required or any(
        not _finite_number(aggregates[key]) for key in required
    ):
        return {
            "status": "INVALID",
            "first_decisive_clause": "invalid_aggregates",
            "full_panel_authorized": False,
        }
    a = aggregates
    close_checks = (
        (
            "close_count_gain",
            any(
                float(a[f"count_gain_median_{c}"]) <= 0.0
                and int(a[f"positive_count_gain_seed_means_{c}"]) <= 1
                for c in "AB"
            ),
        ),
        ("close_context_spearman", float(a["context_spearman_median"]) <= 0.0),
        (
            "close_log_kappa_iqr",
            any(float(a[f"log_kappa_iqr_{c}"]) <= math.log(1.02) for c in "AB"),
        ),
        (
            "close_global_scalar",
            any(float(a[f"global_scalar_relative_error_median_{c}"]) <= 0.02 for c in "AB"),
        ),
        (
            "close_full_gain",
            any(
                float(a[f"full_gain_error_median_{c}"]) <= math.log(1.05)
                and int(a[f"full_gain_error_seed_medians_ge_log_one_point_one_{c}"]) <= 1
                for c in "AB"
            ),
        ),
    )
    for clause, passed in close_checks:
        if passed:
            return {
                "status": "CLOSE",
                "first_decisive_clause": clause,
                "full_panel_authorized": False,
            }
    survives = (
        all(
            float(a[f"count_gain_median_{c}"]) >= math.log(1.10)
            and int(a[f"positive_count_gain_seed_means_{c}"]) >= 3
            for c in "AB"
        )
        and float(a["context_spearman_median"]) >= 0.50
        and int(a["context_spearman_seeds_ge_point_three"]) >= 3
        and all(float(a[f"log_kappa_iqr_{c}"]) >= math.log(1.10) for c in "AB")
        and all(float(a[f"global_scalar_relative_error_median_{c}"]) >= 0.05 for c in "AB")
        and all(
            float(a[f"full_gain_error_median_{c}"]) >= math.log(1.25)
            and int(a[f"full_gain_error_seed_medians_ge_log_one_point_one_{c}"]) >= 3
            for c in "AB"
        )
    )
    if survives:
        return {
            "status": "SURVIVES",
            "first_decisive_clause": "all_survival_predicates",
            "full_panel_authorized": True,
        }
    return {
        "status": "UNRESOLVED",
        "first_decisive_clause": "no_close_or_survival_rule",
        "full_panel_authorized": False,
    }


def _metric_pass(
    metric: dict[str, object], *, alignment_floor: float = 0.0, strict: bool = True
) -> bool:
    pooled = float(metric["pooled_difference"])
    return (
        (pooled > alignment_floor if strict else pooled >= alignment_floor)
        and float(metric["lower_bound"]) > 0.0
        and int(metric["positive_seed_means"]) >= 3
    )


def decide_panel(aggregates: dict[str, object], bootstrap: dict[str, object]) -> dict[str, object]:
    if aggregates.get("completeness") is not True:
        return {
            "status": "INVALID",
            "first_decisive_clause": "invalid_or_incomplete",
            "authorized_action": "stop_invalid",
        }
    try:
        pa_a = aggregates["primary_pa_alignment"]
        if float(pa_a["pooled_difference"]) <= 0.0 and int(pa_a["positive_seed_means"]) <= 1:
            return {
                "status": "CLOSE",
                "first_decisive_clause": "close_vs_pa",
                "authorized_action": "stop_close",
            }
        context_b_pa = aggregates["context_b_pa_alignment"]
        if (
            float(context_b_pa["pooled_difference"]) <= 0.0
            and int(context_b_pa["positive_seed_means"]) <= 1
        ):
            return {
                "status": "CLOSE",
                "first_decisive_clause": "close_context_b_vs_pa",
                "authorized_action": "stop_close",
            }
        pa_s = aggregates["primary_pa_slope"]
        if float(pa_s["pooled_difference"]) <= 0.0 and int(pa_s["positive_seed_means"]) <= 1:
            return {
                "status": "CLOSE",
                "first_decisive_clause": "close_vs_pa_slope",
                "authorized_action": "stop_close",
            }
        controls = aggregates["controls"]
        aliases = aggregates["correction_aliases"]
        for name in CONTROL_ORDER:
            metric = controls[name]["primary_alignment"]
            if (
                float(metric["pooled_difference"]) <= 0.0
                and int(metric["positive_seed_means"]) <= 1
            ):
                return {
                    "status": "CLOSE",
                    "first_decisive_clause": f"close_{name}_primary_alignment",
                    "authorized_action": "stop_close",
                }
            if name in (
                "batch_global_gain",
                "scalar_diagonal_raw",
                "per_example_gradient_normalized",
                "layerwise_trust_ratio",
            ):
                metric = controls[name]["primary_slope"]
                if (
                    float(metric["pooled_difference"]) <= 0.0
                    and int(metric["positive_seed_means"]) <= 1
                ):
                    return {
                        "status": "CLOSE",
                        "first_decisive_clause": f"close_{name}_primary_slope",
                        "authorized_action": "stop_close",
                    }
            alias = aliases[name]
            if (
                float(alias["pooled_median_absolute_cosine"]) >= 0.99
                and int(alias["seed_medians_ge_point_nine_nine"]) >= 3
            ):
                return {
                    "status": "CLOSE",
                    "first_decisive_clause": f"close_alias_{name}",
                    "authorized_action": "stop_close",
                }
        passes = _metric_pass(pa_a, alignment_floor=0.02, strict=False) and _metric_pass(pa_s)
        passes = passes and _metric_pass(aggregates["context_b_pa_alignment"])
        for name in CONTROL_ORDER:
            passes = passes and all(
                _metric_pass(controls[name][metric])
                for metric in (
                    "primary_alignment",
                    "primary_slope",
                    "context_b_alignment",
                    "context_b_slope",
                )
            )
            passes = passes and float(aliases[name]["pooled_median_absolute_cosine"]) < 0.95
        if passes:
            return {
                "status": "PASS",
                "first_decisive_clause": "all_pass_predicates",
                "authorized_action": "authorize_rdgc_preregistration",
            }
    except (KeyError, TypeError, ValueError, OverflowError):
        return {
            "status": "INVALID",
            "first_decisive_clause": "invalid_or_incomplete",
            "authorized_action": "stop_invalid",
        }
    return {
        "status": "UNRESOLVED",
        "first_decisive_clause": "no_close_or_pass_rule",
        "authorized_action": "stop_unresolved",
    }


def paired_bootstrap(panel_rows: list[dict[str, object]]) -> dict[str, object]:
    expected_operators = OPERATOR_ORDER
    if type(panel_rows) is not list or len(panel_rows) != 256:
        raise ValueError("panel requires 256 rows")
    by_key: dict[tuple[int, str, int], dict[str, object]] = {}
    labels = [panel_rows[2 * index].get("receiver_label") for index in range(32)]
    if any(type(label) is not int for label in labels) or len(set(labels)) != 32:
        raise ValueError("panel label order mismatch")
    cursor = 0
    for seed in range(4):
        for label in labels:
            for context in ("A", "B"):
                row = panel_rows[cursor]
                cursor += 1
                if (row.get("seed"), row.get("context"), row.get("receiver_label")) != (
                    seed,
                    context,
                    label,
                ):
                    raise ValueError("panel row order mismatch")
                operators = row.get("operators")
                if type(operators) is not dict or tuple(operators) != expected_operators:
                    raise ValueError("operator order mismatch")
                for record in operators.values():
                    if (
                        type(record) is not dict
                        or tuple(record)
                        != (
                            "motion_sha256",
                            "motion_norm",
                            "margin_alignment",
                            "margin_slope",
                        )
                        or type(record["motion_sha256"]) is not str
                        or _SHA_RE.fullmatch(record["motion_sha256"]) is None
                        or any(
                            not _finite_number(record[key])
                            for key in ("motion_norm", "margin_alignment", "margin_slope")
                        )
                    ):
                        raise ValueError("operator metric schema mismatch")
                key = (seed, context, label)
                if key in by_key:
                    raise ValueError("duplicate row")
                by_key[key] = row
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sample_indices = rng.integers(0, 32, size=(BOOTSTRAP_REPLICATES, 32))
    distributions: list[dict[str, object]] = []
    for context in ("A", "B"):
        comparators = ("pa", *CONTROL_ORDER)
        for comparator in comparators:
            for metric in ("margin_alignment", "margin_slope"):
                per_label = np.empty(32, dtype=np.float64)
                for label_index, label in enumerate(labels):
                    diffs = [
                        float(by_key[(seed, context, label)]["operators"]["rdgc"][metric])
                        - float(by_key[(seed, context, label)]["operators"][comparator][metric])
                        for seed in range(4)
                    ]
                    per_label[label_index] = np.mean(
                        np.asarray(diffs, dtype=np.float64), dtype=np.float64
                    )
                values = np.mean(per_label[sample_indices], axis=1, dtype=np.float64)
                distributions.append(
                    {
                        "context": context,
                        "comparator": comparator,
                        "metric": metric,
                        "values": values.tolist(),
                        "values_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
                        "lower_bound": float(np.percentile(values, 2.5)),
                    }
                )
    return {
        "bit_generator": "PCG64",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "complete_labels": labels,
        "distributions": distributions,
    }


def preliminary_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if type(rows) is not list or len(rows) != 64:
        raise ValueError("preliminary requires 64 ordered rows")
    receiver_labels = [rows[index].get("receiver_label") for index in range(8)]
    if (
        any(type(label) is not int or label < 0 for label in receiver_labels)
        or len(set(receiver_labels)) != 8
    ):
        raise ValueError("preliminary receiver identity order is invalid")
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    cursor = 0
    for seed in range(4):
        for context in ("A", "B"):
            current: list[dict[str, object]] = []
            for receiver_label in receiver_labels:
                row = rows[cursor]
                cursor += 1
                if (row.get("seed"), row.get("context"), row.get("receiver_label")) != (
                    seed,
                    context,
                    receiver_label,
                ):
                    raise ValueError("preliminary row order mismatch")
                errors = row.get("absolute_log_gain_errors_by_contributor_count")
                if type(errors) is not dict or tuple(errors) != ("1", "8", "32", "180"):
                    raise ValueError("contributor error schema mismatch")
                if any(not _finite_number(errors[key]) for key in errors):
                    raise ValueError("contributor errors must be finite")
                if not _finite_number(row.get("log_kappa")) or not _finite_number(
                    row.get("count_gain")
                ):
                    raise ValueError("preliminary metrics must be finite")
                current.append(row)
            grouped[(seed, context)] = current

    def median(values: list[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    count_medians: dict[str, float] = {}
    positive_counts: dict[str, int] = {}
    iqrs: dict[str, float] = {}
    global_errors: dict[str, float] = {}
    full_errors: dict[str, float] = {}
    full_seed_counts: dict[str, int] = {}
    for context in ("A", "B"):
        all_rows = [row for seed in range(4) for row in grouped[(seed, context)]]
        count_medians[context] = median([float(row["count_gain"]) for row in all_rows])
        positive_counts[context] = sum(
            float(
                np.mean(
                    np.asarray(
                        [float(row["count_gain"]) for row in grouped[(seed, context)]],
                        dtype=np.float64,
                    )
                )
            )
            > 0.0
            for seed in range(4)
        )
        log_values = np.asarray([float(row["log_kappa"]) for row in all_rows], dtype=np.float64)
        iqrs[context] = float(np.percentile(log_values, 75) - np.percentile(log_values, 25))
        per_receiver_error: list[float] = []
        seed_full_medians: list[float] = []
        for seed in range(4):
            seed_rows = grouped[(seed, context)]
            global_log = median([float(row["log_kappa"]) for row in seed_rows])
            per_receiver_error.extend(
                abs(math.exp(float(row["log_kappa"]) - global_log) - 1.0) for row in seed_rows
            )
            seed_full_medians.append(
                median(
                    [
                        float(row["absolute_log_gain_errors_by_contributor_count"]["180"])
                        for row in seed_rows
                    ]
                )
            )
        global_errors[context] = median(per_receiver_error)
        full_errors[context] = median(
            [float(row["absolute_log_gain_errors_by_contributor_count"]["180"]) for row in all_rows]
        )
        full_seed_counts[context] = sum(value >= math.log(1.10) for value in seed_full_medians)
    spearman: list[float] = []
    for seed in range(4):
        left = np.asarray(
            [float(row["log_kappa"]) for row in grouped[(seed, "A")]], dtype=np.float64
        )
        right = np.asarray(
            [float(row["log_kappa"]) for row in grouped[(seed, "B")]], dtype=np.float64
        )
        left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(np.float64)
        right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(np.float64)
        spearman.append(float(np.corrcoef(left_rank, right_rank)[0, 1]))
    return {
        "count_gain_median_A": count_medians["A"],
        "count_gain_median_B": count_medians["B"],
        "positive_count_gain_seed_means_A": positive_counts["A"],
        "positive_count_gain_seed_means_B": positive_counts["B"],
        "context_spearman_median": median(spearman),
        "context_spearman_seeds_ge_point_three": sum(value >= 0.30 for value in spearman),
        "log_kappa_iqr_A": iqrs["A"],
        "log_kappa_iqr_B": iqrs["B"],
        "global_scalar_relative_error_median_A": global_errors["A"],
        "global_scalar_relative_error_median_B": global_errors["B"],
        "full_gain_error_median_A": full_errors["A"],
        "full_gain_error_median_B": full_errors["B"],
        "full_gain_error_seed_medians_ge_log_one_point_one_A": full_seed_counts["A"],
        "full_gain_error_seed_medians_ge_log_one_point_one_B": full_seed_counts["B"],
    }


def _domain_hash(domain: str, value: str) -> bytes:
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + value.encode("utf-8")).digest()


def _bound_inputs(training_ids: Any, labels: Any) -> tuple[list[str], list[int]]:
    ids = list(training_ids)
    raw_labels = list(labels)
    if not ids or len(ids) != len(raw_labels):
        raise ValueError("training IDs and labels must be nonempty and aligned")
    if any(type(value) is not str or not value for value in ids):
        raise TypeError("every Bound ID must be an exact nonempty string")
    if len(ids) != len(set(ids)):
        raise ValueError("Bound IDs must be unique")
    canonical_labels: list[int] = []
    for value in raw_labels:
        if type(value) is not int or value < 0:
            raise TypeError("labels must be unsigned builtin integers")
        canonical_labels.append(value)
    return ids, canonical_labels


def build_rdgc_selection(
    training_ids: Any, labels: Any, *, old_selection: Any
) -> dict[str, object]:
    ids, canonical_labels = _bound_inputs(training_ids, labels)
    if type(old_selection) not in (set, frozenset) or any(
        type(value) is not int or value < 0 for value in old_selection
    ):
        raise TypeError("old selection must be a label set")
    old_labels = set(old_selection)
    rows: dict[int, list[str]] = {}
    for example_id, label in zip(ids, canonical_labels, strict=True):
        rows.setdefault(label, []).append(example_id)
    roles = {
        label: sorted(
            values, key=lambda value: (_domain_hash("rdgc-stage-b-v1|role|", value), value)
        )
        for label, values in rows.items()
        if len(values) >= 3 and label not in old_labels
    }
    ordered = sorted(
        roles,
        key=lambda label: (_domain_hash("rdgc-stage-b-v1|identity|", str(label)), label),
    )
    if len(ordered) < 40:
        raise ValueError("at least forty fresh eligible identities are required")
    selected = ordered[:40]
    selected_rows = {item for label in selected for item in rows[label]}
    old_rows = {item for label in old_labels for item in rows.get(label, ())}
    assigned_distractors: set[str] = set()

    def phase_record(name: str, phase_labels: list[int]) -> dict[str, object]:
        receiver_ids = [roles[label][2] for label in phase_labels]
        supports = {str(label): roles[label][:2] for label in phase_labels}
        groups: list[dict[str, object]] = []
        for group_index, start in enumerate(range(0, len(phase_labels), 8)):
            group_labels = phase_labels[start : start + 8]
            group_receivers = [roles[label][2] for label in group_labels]
            contexts: list[dict[str, object]] = []
            for context in ("A", "B"):
                candidates = [
                    item
                    for item in ids
                    if item not in selected_rows
                    and item not in old_rows
                    and item not in assigned_distractors
                ]
                domain = f"rdgc-stage-b-v1|distractor|{name}|{group_index}|{context}|"
                candidates.sort(key=lambda item: (_domain_hash(domain, item), item))
                distractors = candidates[:172]
                if len(distractors) != 172:
                    raise ValueError("not enough disjoint distractors")
                assigned_distractors.update(distractors)
                batch_domain = f"rdgc-stage-b-v1|batch-order|{name}|{group_index}|{context}|"
                batch = sorted(
                    group_receivers + distractors,
                    key=lambda item: (_domain_hash(batch_domain, item), item),
                )
                contexts.append(
                    {
                        "context": context,
                        "batch_ids": batch,
                        "batch_sha256": hashlib.sha256(
                            b"".join(item.encode() + b"\n" for item in batch)
                        ).hexdigest(),
                        "transform_sha256": hashlib.sha256(
                            f"{name}|{group_index}|{context}|transform".encode()
                        ).hexdigest(),
                        "tensor_sha256": hashlib.sha256(
                            f"{name}|{group_index}|{context}|tensor".encode()
                        ).hexdigest(),
                    }
                )
            groups.append(
                {
                    "group_index": group_index,
                    "receiver_labels": group_labels,
                    "receiver_ids": group_receivers,
                    "contexts": contexts,
                }
            )
        return {
            "identity_labels": phase_labels,
            "receiver_ids": receiver_ids,
            "support_ids_by_label": supports,
            "groups": groups,
        }

    return {
        "old_rsta_exclusion_sha256": hashlib.sha256(
            b"".join(f"{label}\n".encode() for label in sorted(old_labels))
        ).hexdigest(),
        "preliminary": phase_record("preliminary", selected[:8]),
        "panel": phase_record("panel", selected[8:]),
    }


def nested_contributor_masks(
    batch_ids: tuple[str, ...], receiver_id: str, *, seed: int, context: str
) -> tuple[tuple[bool, ...], ...]:
    if type(batch_ids) is not tuple or len(batch_ids) != 180 or len(set(batch_ids)) != 180:
        raise ValueError("batch must have 180 unique Bound IDs")
    if (
        any(type(value) is not str or not value for value in batch_ids)
        or type(receiver_id) is not str
        or receiver_id not in batch_ids
    ):
        raise ValueError("receiver/batch Bound-ID mismatch")
    if type(seed) is not int or seed not in range(4) or context not in ("A", "B"):
        raise ValueError("seed/context mismatch")
    others = [value for value in batch_ids if value != receiver_id]
    domain = f"rdgc-stage-b-v1|contributor|{seed}|{receiver_id}|{context}|"
    others.sort(key=lambda value: (_domain_hash(domain, value), value))
    result: list[tuple[bool, ...]] = []
    for count in CONTRIBUTOR_COUNTS:
        members = {receiver_id, *others[: count - 1]}
        result.append(tuple(value in members for value in batch_ids))
    return tuple(result)


def run_all_seed_integrity_prefix(
    bindings: Any, *, adapters: dict[str, Any]
) -> list[dict[str, object]]:
    records = list(bindings)
    if len(records) != 4 or [record.get("seed") for record in records] != list(range(4)):
        raise ValueError("bindings must be exact seed order 0,1,2,3")
    audit_seed = adapters["audit_seed"]
    result: list[dict[str, object]] = []
    for seed, binding in enumerate(records):
        audit = audit_seed(seed, binding)
        if type(audit) is not dict or audit.get("seed") != seed or audit.get("passed") is not True:
            raise ValueError(f"integrity failed for seed {seed}")
        result.append(audit)
    return result


def run_preliminary(
    selection: dict[str, object],
    bindings: tuple[dict[str, object], ...],
    *,
    adapters: dict[str, Any],
) -> dict[str, object]:
    labels = selection["preliminary"]["identity_labels"]
    if type(labels) is not list or len(labels) != 8:
        raise ValueError("preliminary selection mismatch")
    if len(bindings) != 4 or [binding.get("seed") for binding in bindings] != list(range(4)):
        raise ValueError("binding order mismatch")
    rows: list[dict[str, object]] = []
    for seed in range(4):
        for context in ("A", "B"):
            graph = adapters["build_graph"](seed, context, selection)
            try:
                for receiver_label in labels:
                    diagonal = adapters["receiver_action"](graph, receiver_label, None)
                    actions = {
                        str(count): adapters["receiver_action"](graph, receiver_label, count)
                        for count in CONTRIBUTOR_COUNTS
                    }
                    rows.append(
                        {
                            "seed": seed,
                            "context": context,
                            "receiver_label": receiver_label,
                            "diagonal": dict(diagonal),
                            "actions": actions,
                        }
                    )
            finally:
                graph.close()
                del graph
    return {"rows": rows}


def _run_git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Git authentication failed") from exc
    if len(completed.stdout) > 1_000_000 or len(completed.stderr) > 1_000_000:
        raise ValueError("Git output exceeded bound")
    return completed.stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("Git byte authentication failed") from exc
    if len(completed.stdout) > 100_000_000 or len(completed.stderr) > 1_000_000:
        raise ValueError("Git byte output exceeded bound")
    return completed.stdout


def _strict_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("authority path must be a regular non-symlink file")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid authority JSON") from exc
    if type(value) is not dict:
        raise ValueError("authority JSON must be an object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(repository: Path, path: Path) -> Path:
    root = repository.resolve(strict=True)
    absolute = path if path.is_absolute() else root / path
    resolved = absolute.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError("authority path escapes repository")
    return resolved


def authenticate_authority(
    repository: Path, manifest_path: Path, receipt_path: Path
) -> dict[str, object]:
    repository = repository.resolve(strict=True)
    manifest_path = _within(repository, manifest_path)
    receipt_path = _within(repository, receipt_path)
    manifest = _strict_json(manifest_path)
    validate_future_manifest(manifest)
    head = _run_git(repository, "rev-parse", "HEAD")
    if _run_git(repository, "rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
        raise ValueError("scientific handoff must be a detached HEAD")
    parents = _run_git(repository, "show", "-s", "--format=%P", head).split()
    if len(parents) != 1:
        raise ValueError("handoff must have exactly one parent")
    source_commit = parents[0]
    allowed_source_paths = {
        "scripts/diagnose_pass205_rdgc_stage_b.py",
        "tests/test_diagnose_pass205_rdgc_stage_b.py",
    }
    cursor = source_commit
    visited: set[str] = set()
    while cursor != PLAN_COMMIT:
        if cursor in visited:
            raise ValueError("source history contains a cycle")
        visited.add(cursor)
        cursor_parents = _run_git(repository, "show", "-s", "--format=%P", cursor).split()
        if len(cursor_parents) != 1:
            raise ValueError("source history must be a merge-free one-parent chain")
        changed_paths = set(
            _run_git(
                repository,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                cursor,
            ).splitlines()
        )
        if not changed_paths or not changed_paths <= allowed_source_paths:
            raise ValueError("source history commit scope mismatch")
        cursor = cursor_parents[0]
    if _run_git(repository, "rev-parse", f"{PLAN_COMMIT}^") != CANDIDATE_COMMIT:
        raise ValueError("reviewed plan/candidate parent relation mismatch")
    if _run_git(repository, "show", "-s", "--format=%P", PLAN_COMMIT).split() != [CANDIDATE_COMMIT]:
        raise ValueError("reviewed plan must have exactly one candidate parent")
    aggregate_scope = _run_git(
        repository,
        "diff",
        "--name-only",
        PLAN_COMMIT,
        source_commit,
    ).splitlines()
    if aggregate_scope != [
        "scripts/diagnose_pass205_rdgc_stage_b.py",
        "tests/test_diagnose_pass205_rdgc_stage_b.py",
    ]:
        raise ValueError("source aggregate scope mismatch")
    changed = _run_git(
        repository, "diff-tree", "--no-commit-id", "--name-only", "-r", head
    ).splitlines()
    manifest_relative = str(manifest_path.relative_to(repository))
    if changed != [manifest_relative]:
        raise ValueError("handoff scope must be manifest-only")
    source = manifest["current_scientific_source"]
    if (
        type(source) is not dict
        or tuple(source) != ("git_revision", "files")
        or source["git_revision"] != source_commit
    ):
        raise ValueError("source/handoff relation mismatch")
    for key, expected_path, expected_commit in (
        ("candidate", CANDIDATE_PATH, CANDIDATE_COMMIT),
        ("implementation_plan", PLAN_PATH, PLAN_COMMIT),
    ):
        reference = manifest[key]
        if type(reference) is not dict or tuple(reference) != ("path", "sha256", "commit"):
            raise ValueError(f"{key} reference schema mismatch")
        if reference["path"] != expected_path or reference["commit"] != expected_commit:
            raise ValueError(f"{key} authority mismatch")
        path = _within(repository, repository / expected_path)
        blob = _git_bytes(repository, "show", f"{expected_commit}:{expected_path}")
        if (
            hashlib.sha256(blob).hexdigest() != reference["sha256"]
            or _sha256_file(path) != reference["sha256"]
        ):
            raise ValueError(f"{key} digest mismatch")
        literal_digest = CANDIDATE_SHA256 if key == "candidate" else PLAN_SHA256
        if reference["sha256"] != literal_digest:
            raise ValueError(f"{key} literal digest mismatch")
    if _run_git(repository, "merge-base", "--is-ancestor", PLAN_COMMIT, source_commit) != "":
        # merge-base --is-ancestor succeeds with empty output; the wrapper raises on failure.
        raise ValueError("unexpected ancestry output")
    files = source["files"]
    if type(files) is not list or len(files) != len(RDGC_SOURCE_ORDER):
        raise ValueError("source file count mismatch")
    for expected_path, record in zip(RDGC_SOURCE_ORDER, files, strict=True):
        if (
            type(record) is not dict
            or tuple(record) != ("path", "sha256", "git_blob")
            or record["path"] != expected_path
        ):
            raise ValueError("source order/schema mismatch")
        path = _within(repository, repository / expected_path)
        blob = _run_git(repository, "rev-parse", f"{source_commit}:{expected_path}")
        if blob != record["git_blob"] or _sha256_file(path) != record["sha256"]:
            raise ValueError("source digest/blob mismatch")
        committed_bytes = _git_bytes(repository, "show", f"{source_commit}:{expected_path}")
        if hashlib.sha256(committed_bytes).hexdigest() != record["sha256"]:
            raise ValueError("source Git bytes mismatch")
    if _run_git(repository, "diff", "--name-only") or _run_git(
        repository, "diff", "--cached", "--name-only"
    ):
        raise ValueError("tracked worktree must be clean")
    receipt_ref = manifest["validation_receipt"]
    if type(receipt_ref) is not dict or tuple(receipt_ref) != (
        "path",
        "sha256",
        "status",
        "verifier_source_commit",
        "verifier_handoff_commit",
        "artifact_path",
        "artifact_sha256",
    ):
        raise ValueError("receipt reference schema mismatch")
    if (
        receipt_ref["path"] != str(receipt_path.relative_to(repository))
        or receipt_ref["sha256"] != _sha256_file(receipt_path)
        or receipt_ref["status"] != "VALID"
    ):
        raise ValueError("validation receipt binding mismatch")
    receipt = _strict_json(receipt_path)
    if (
        tuple(receipt) != ("status", "artifact_path", "artifact_sha256")
        or receipt["status"] != "VALID"
    ):
        raise ValueError("validation receipt schema/status mismatch")
    if type(receipt["artifact_path"]) is not str or type(receipt["artifact_sha256"]) is not str:
        raise ValueError("validation receipt artifact binding mismatch")
    return {
        "source_commit": source_commit,
        "handoff_commit": head,
        "files": files,
        "validation_receipt": receipt_ref,
    }


def load_authenticated_rsta_module(repository: Path, source: dict[str, object]) -> types.ModuleType:
    if type(source) is not dict or tuple(source) != ("path", "sha256", "git_blob"):
        raise ValueError("source descriptor schema mismatch")
    repository = repository.resolve(strict=True)
    raw_path = Path(source["path"])
    path = _within(repository, raw_path if raw_path.is_absolute() else repository / raw_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("helper must be a regular file")
    before = path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    if digest != source["sha256"]:
        raise ValueError("helper digest mismatch")
    relative = path.relative_to(repository)
    blob = _run_git(repository, "rev-parse", f"HEAD:{relative}")
    if blob != source["git_blob"]:
        raise ValueError("helper Git blob mismatch")
    name = f"_pass205_authenticated_rsta_{digest}"
    if name in sys.modules:
        raise ValueError("private authenticated module already exists")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    code = compile(before, str(path), "exec", dont_inherit=True)
    sys.modules[name] = module
    original_path = list(sys.path)
    try:
        exec(code, module.__dict__)
        after = path.read_bytes()
        if after != before or hashlib.sha256(after).hexdigest() != digest:
            raise ValueError("helper bytes changed during execution")
        if Path(module.__file__).resolve() != path:
            raise ValueError("helper origin mismatch")
        return module
    finally:
        sys.path[:] = original_path
        sys.modules.pop(name, None)


def compute_parameter_correction(
    field: dict[str, object], operator: str, *, torch_module: Any
) -> tuple[Any, ...]:
    if operator not in CORRECTION_ORDER:
        raise ValueError("unknown correction operator")
    named_parameters = field.get("named_parameters")
    p = field.get("p")
    if type(named_parameters) is not tuple or type(p) is not tuple:
        raise ValueError("field lacks named parameters or PA direction")
    parameters = tuple(record[1] for record in named_parameters)
    if operator == "layerwise_trust_ratio":
        return tuple(
            value.detach().to(device="cpu", dtype=torch_module.float32).contiguous()
            for value in layerwise_trust_ratio_direction(torch_module, named_parameters, p)
        )
    required = ("b", "s", "dbar", "receiver_fields", "pgn_motion")
    if any(key not in field for key in required):
        raise ValueError("penalty field is incomplete")
    penalties = control_penalties(
        torch_module,
        b=field["b"],
        s=field["s"],
        dbar=field["dbar"],
        receiver_fields=field["receiver_fields"],
        pgn_motion=field["pgn_motion"],
    )
    penalty = penalties[operator]
    gradients = torch_module.autograd.grad(
        penalty, parameters, retain_graph=False, create_graph=False, allow_unused=False
    )
    result: list[Any] = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None or gradient.shape != parameter.shape:
            raise ValueError("missing or mis-shaped correction gradient")
        _require_fp32_finite(torch_module, gradient)
        result.append(
            (-gradient).detach().to(device="cpu", dtype=torch_module.float32).contiguous()
        )
    return tuple(result)


def _tensor_sha256(tensor: Any, torch_module: Any) -> str:
    _require_fp32_finite(torch_module, tensor)
    cpu = tensor.detach().to(device="cpu", dtype=torch_module.float32).contiguous()
    return hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest()


def evaluate_virtual_direction(
    context: dict[str, object], direction: tuple[Any, ...], q: Any, *, torch_module: Any
) -> dict[str, object]:
    if type(context) is not dict or tuple(context) != ("function", "parameters"):
        raise ValueError("context schema/order mismatch")
    function = context["function"]
    parameters = context["parameters"]
    if not callable(function) or type(parameters) is not tuple or type(direction) is not tuple:
        raise TypeError("invalid functional context or direction")
    if len(parameters) != len(direction) or not direction:
        raise ValueError("functional parameter/direction mismatch")
    for parameter, tangent in zip(parameters, direction, strict=True):
        _require_fp32_finite(torch_module, parameter)
        _require_fp32_finite(torch_module, tangent)
        if parameter.shape != tangent.shape:
            raise ValueError("functional parameter/direction shape mismatch")
    _require_fp32_finite(torch_module, q, nonzero=True)
    _, motion = torch_module.func.jvp(function, (parameters,), (direction,))
    _require_fp32_finite(torch_module, motion, nonzero=True)
    if motion.shape != q.shape or motion.device != q.device:
        raise ValueError("motion/margin direction mismatch")
    motion64 = motion.to(torch_module.float64).reshape(-1)
    q64 = q.to(torch_module.float64).reshape(-1)
    motion_norm = torch_module.sqrt((motion64 * motion64).sum())
    q_norm = torch_module.sqrt((q64 * q64).sum())
    dot = (motion64 * q64).sum()
    direction_norm = fp64_named_norm(torch_module, direction)
    result = {
        "motion_sha256": _tensor_sha256(motion, torch_module),
        "motion_norm": float(motion_norm.item()),
        "margin_alignment": float((dot / (motion_norm * q_norm)).item()),
        "margin_slope": float((dot / (direction_norm * q_norm)).item()),
    }
    del motion, motion64, q64, motion_norm, q_norm, dot, direction_norm
    return result


def run_full_panel(
    selection: dict[str, object],
    bindings: tuple[dict[str, object], ...],
    preliminary: dict[str, object],
    *,
    adapters: dict[str, Any],
) -> dict[str, object]:
    labels = selection.get("panel", {}).get("identity_labels")
    if (
        type(labels) is not list
        or len(labels) != 32
        or any(type(value) is not int for value in labels)
    ):
        raise ValueError("panel selection must contain 32 ordered labels")
    if len(bindings) != 4 or [binding.get("seed") for binding in bindings] != list(range(4)):
        raise ValueError("panel bindings must be exact seed order")
    build_correction = adapters["build_correction"]
    evaluate_direction = adapters["evaluate_direction"]
    records: dict[tuple[int, str, int], dict[str, object]] = {
        (seed, context, label): {
            "seed": seed,
            "context": context,
            "receiver_label": label,
            "operators": {},
        }
        for seed in range(4)
        for context in ("A", "B")
        for label in labels
    }
    for seed in range(4):
        for receiver_index, label in enumerate(labels):
            group = receiver_index // 8
            for operator in OPERATOR_ORDER:
                direction = build_correction(seed, group, label, operator)
                if type(direction) is not tuple or not direction:
                    raise ValueError("correction direction must be a nonempty tuple")
                for context in ("A", "B"):
                    action = evaluate_direction(seed, group, label, operator, context, direction)
                    if type(action) is not dict or tuple(action) != (
                        "motion_sha256",
                        "motion_norm",
                        "margin_alignment",
                        "margin_slope",
                    ):
                        raise ValueError("operator action schema/order mismatch")
                    if any(
                        not _finite_number(action[key])
                        for key in ("motion_norm", "margin_alignment", "margin_slope")
                    ):
                        raise ValueError("operator action values must be finite")
                    records[(seed, context, label)]["operators"][operator] = dict(action)
                del direction
    rows = [
        records[(seed, context, label)]
        for seed in range(4)
        for context in ("A", "B")
        for label in labels
    ]
    return {"rows": rows}


def build_pre_import_environment(
    authenticated_non_torch_fields: dict[str, object],
) -> dict[str, object]:
    return {
        "phase": "pre_import",
        "pre_import": dict(authenticated_non_torch_fields),
        "torch_runtime": None,
    }


def attach_observed_torch_runtime(
    pre_import_environment: dict[str, object], torch_module: Any
) -> dict[str, object]:
    if (
        tuple(pre_import_environment) != ("phase", "pre_import", "torch_runtime")
        or pre_import_environment["phase"] != "pre_import"
        or pre_import_environment["torch_runtime"] is not None
    ):
        raise ValueError("invalid pre-import environment")
    runtime = {
        "torch_version": torch_module.__version__,
        "cuda_runtime_version": torch_module.version.cuda,
        "cudnn_version": torch_module.backends.cudnn.version(),
        "device_index": torch_module.cuda.current_device(),
        "device_name": torch_module.cuda.get_device_name(torch_module.cuda.current_device()),
        "device_capability": list(
            torch_module.cuda.get_device_capability(torch_module.cuda.current_device())
        ),
        "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
        "allow_tf32_matmul": torch_module.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch_module.backends.cudnn.allow_tf32,
    }
    return {
        "phase": "post_import",
        "pre_import": pre_import_environment["pre_import"],
        "torch_runtime": runtime,
    }


def validate_scientific_payload(value: dict[str, object]) -> None:
    if type(value) is not dict:
        raise TypeError("payload must be a dictionary")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must contain finite concrete JSON values") from exc
    if tuple(value) != RESULT_KEYS:
        raise ValueError("top-level result schema/order mismatch")
    fixed = {
        "schema_version": 1,
        "diagnostic": "pass205_rdgc_stage_b",
        "mode": "scientific_no_training_virtual_update",
        "training_performed": False,
        "benchmark_authorized": False,
        "scope_limitation": SCOPE_LIMITATION,
    }
    for key, expected in fixed.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError(f"fixed result field mismatch: {key}")
    status = value["status"]
    phase = value["phase_reached"]
    if type(status) is not str or status not in ("PASS", "CLOSE", "UNRESOLVED", "INVALID"):
        raise ValueError("invalid status")
    if type(phase) is not str or phase not in (
        "pre_import",
        "integrity",
        "preliminary",
        "full_panel",
    ):
        raise ValueError("invalid phase")
    if type(value["candidate_values_computed"]) is not bool:
        raise TypeError("candidate_values_computed must be bool")
    for key in ("authority", "source", "execution", "environment", "binding", "decision"):
        if type(value[key]) is not dict:
            raise TypeError(f"{key} must be an object")
    environment = value["environment"]
    if tuple(environment) != ("phase", "pre_import", "torch_runtime") or environment[
        "phase"
    ] not in ("pre_import", "post_import"):
        raise ValueError("environment union schema mismatch")
    pre_import = environment["pre_import"]
    if type(pre_import) is not dict or tuple(pre_import) != (
        "python_executable",
        "python_version",
        "numpy_version",
        "cuda_visible_devices",
        "cublas_workspace_config",
        "source_commit",
        "source_files_sha256",
        "manifest_path",
        "manifest_sha256",
    ):
        raise ValueError("pre-import environment schema/order mismatch")
    if phase == "pre_import":
        if (
            status != "INVALID"
            or environment["phase"] != "pre_import"
            or environment["torch_runtime"] is not None
        ):
            raise ValueError("pre-import branch relation mismatch")
        if value["candidate_values_computed"] is not False:
            raise ValueError("pre-import INVALID cannot contain candidate values")
        if any(
            value[key] is not None
            for key in ("integrity", "selection", "preliminary", "panel", "bootstrap")
        ):
            raise ValueError("pre-import INVALID null union mismatch")
    else:
        if environment["phase"] != "post_import" or type(environment["torch_runtime"]) is not dict:
            raise ValueError("post-import result requires observed Torch runtime")
        if status == "INVALID" and any(
            value[key] is not None for key in ("selection", "preliminary", "panel", "bootstrap")
        ):
            raise ValueError("post-import INVALID null union mismatch")
        if (
            phase == "preliminary"
            and status != "INVALID"
            and (
                value["candidate_values_computed"] is not True
                or value["selection"] is None
                or value["preliminary"] is None
                or value["panel"] is not None
                or value["bootstrap"] is not None
            )
        ):
            raise ValueError("preliminary result union mismatch")
        if (
            phase == "full_panel"
            and status != "INVALID"
            and (
                value["candidate_values_computed"] is not True
                or any(
                    value[key] is None for key in ("selection", "preliminary", "panel", "bootstrap")
                )
            )
        ):
            raise ValueError("full-panel result union mismatch")
    decision = value["decision"]
    if tuple(decision) != (
        "close_precedence",
        "predicates",
        "status",
        "first_decisive_clause",
        "authorized_action",
    ):
        raise ValueError("decision schema/order mismatch")
    if type(decision["close_precedence"]) is not bool or type(decision["predicates"]) is not list:
        raise TypeError("decision concrete types mismatch")
    if decision["status"] != status:
        raise ValueError("decision/top-level status mismatch")
    expected_action = {
        "INVALID": "stop_invalid",
        "CLOSE": "stop_close",
        "UNRESOLVED": "stop_unresolved",
        "PASS": "new_training_preregistration_only",
    }[status]
    if decision["authorized_action"] != expected_action:
        raise ValueError("decision action/status mismatch")
    for record in decision["predicates"]:
        if (
            type(record) is not dict
            or tuple(record) != ("name", "value")
            or type(record["name"]) is not str
            or type(record["value"]) is not bool
        ):
            raise ValueError("predicate record schema mismatch")
    _validate_result_nested(value)


_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


def _exact_object(value: object, keys: tuple[str, ...], name: str) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError(f"{name} schema/order mismatch")
    return value


def _sha(value: object, name: str) -> None:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _commit(value: object, name: str) -> None:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase commit")


def _float(value: object, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON float")


def _integer(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a builtin integer")


def _bound_id(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be an exact nonempty Bound ID")


def _validate_ref(value: object, name: str) -> None:
    record = _exact_object(value, ("path", "sha256", "commit"), name)
    if type(record["path"]) is not str or not record["path"]:
        raise ValueError(f"{name}.path mismatch")
    _sha(record["sha256"], f"{name}.sha256")
    _commit(record["commit"], f"{name}.commit")


def _validate_artifact_ref(value: object, name: str) -> None:
    record = _exact_object(value, ("path", "sha256"), name)
    if type(record["path"]) is not str or not record["path"]:
        raise ValueError(f"{name}.path mismatch")
    _sha(record["sha256"], f"{name}.sha256")


def _validate_seed_artifacts(value: object, expected_seed: int, name: str) -> None:
    record = _exact_object(
        value,
        (
            "seed",
            "checkpoint",
            "training_report",
            "final_pack",
            "configuration_sha256",
            "source_export_sha256",
        ),
        name,
    )
    if record["seed"] != expected_seed:
        raise ValueError(f"{name}.seed mismatch")
    for key in ("checkpoint", "training_report", "final_pack"):
        _validate_artifact_ref(record[key], f"{name}.{key}")
    _sha(record["configuration_sha256"], f"{name}.configuration")
    _sha(record["source_export_sha256"], f"{name}.source export")


def _validate_source_file(value: object, expected_path: str, name: str) -> None:
    record = _exact_object(value, ("path", "sha256", "git_blob"), name)
    if record["path"] != expected_path:
        raise ValueError(f"{name}.path mismatch")
    _sha(record["sha256"], f"{name}.sha256")
    if type(record["git_blob"]) is not str or not record["git_blob"]:
        raise ValueError(f"{name}.git_blob mismatch")


def _validate_selection(value: object) -> None:
    selection = _exact_object(
        value, ("old_rsta_exclusion_sha256", "preliminary", "panel"), "selection"
    )
    _sha(selection["old_rsta_exclusion_sha256"], "selection exclusion")
    for phase_name, identity_count, group_count in (
        ("preliminary", 8, 1),
        ("panel", 32, 4),
    ):
        phase = _exact_object(
            selection[phase_name],
            ("identity_labels", "receiver_ids", "support_ids_by_label", "groups"),
            phase_name,
        )
        labels = phase["identity_labels"]
        receiver_ids = phase["receiver_ids"]
        supports = phase["support_ids_by_label"]
        groups = phase["groups"]
        if type(labels) is not list or len(labels) != identity_count:
            raise ValueError(f"{phase_name} identity count mismatch")
        if (
            any(type(label) is not int or label < 0 for label in labels)
            or len(set(labels)) != identity_count
        ):
            raise ValueError(f"{phase_name} labels mismatch")
        if type(receiver_ids) is not list or len(receiver_ids) != identity_count:
            raise ValueError(f"{phase_name} receiver count mismatch")
        for receiver_id in receiver_ids:
            _bound_id(receiver_id, f"{phase_name} receiver")
        expected_support_keys = tuple(str(label) for label in labels)
        if type(supports) is not dict or tuple(supports) != expected_support_keys:
            raise ValueError(f"{phase_name} support order mismatch")
        for pair in supports.values():
            if type(pair) is not list or len(pair) != 2:
                raise ValueError(f"{phase_name} support pair mismatch")
            for support_id in pair:
                _bound_id(support_id, f"{phase_name} support")
        if type(groups) is not list or len(groups) != group_count:
            raise ValueError(f"{phase_name} group count mismatch")
        for group_index, group_value in enumerate(groups):
            group = _exact_object(
                group_value,
                ("group_index", "receiver_labels", "receiver_ids", "contexts"),
                f"{phase_name} group",
            )
            if group["group_index"] != group_index:
                raise ValueError(f"{phase_name} group index mismatch")
            start = group_index * 8
            if (
                group["receiver_labels"] != labels[start : start + 8]
                or group["receiver_ids"] != receiver_ids[start : start + 8]
            ):
                raise ValueError(f"{phase_name} group identities mismatch")
            if type(group["contexts"]) is not list or len(group["contexts"]) != 2:
                raise ValueError(f"{phase_name} contexts mismatch")
            for context_name, context_value in zip(("A", "B"), group["contexts"], strict=True):
                context = _exact_object(
                    context_value,
                    (
                        "context",
                        "batch_ids",
                        "batch_sha256",
                        "transform_sha256",
                        "tensor_sha256",
                    ),
                    f"{phase_name} context",
                )
                if (
                    context["context"] != context_name
                    or type(context["batch_ids"]) is not list
                    or len(context["batch_ids"]) != 180
                ):
                    raise ValueError(f"{phase_name} context batch mismatch")
                for example_id in context["batch_ids"]:
                    _bound_id(example_id, f"{phase_name} batch ID")
                for key in ("batch_sha256", "transform_sha256", "tensor_sha256"):
                    _sha(context[key], f"{phase_name}.{key}")


def _validate_preliminary(value: object) -> None:
    preliminary = _exact_object(
        value,
        (
            "operator_counts",
            "rows",
            "seed_context_aggregates",
            "seed_correlations",
            "pooled_aggregates",
            "predicates",
            "decision",
        ),
        "preliminary",
    )
    counts = _exact_object(
        preliminary["operator_counts"],
        (
            "forwards_per_seed_context",
            "dbars_per_seed_context",
            "diagonal_vjps_per_seed_context",
            "diagonal_jvps_per_seed_context",
            "masked_vjps_per_seed_context",
            "masked_receiver_jvps_per_seed_context",
        ),
        "preliminary operator counts",
    )
    if tuple(counts.values()) != (1, 1, 8, 8, 32, 32):
        raise ValueError("preliminary operator counts mismatch")
    rows = preliminary["rows"]
    if type(rows) is not list or len(rows) != 64:
        raise ValueError("preliminary rows mismatch")
    row_keys = (
        "seed",
        "context",
        "receiver_label",
        "receiver_id",
        "dbar_norm",
        "self_norm",
        "kappa",
        "log_kappa",
        "b_norms_by_contributor_count",
        "absolute_log_gain_errors_by_contributor_count",
        "count_gain",
    )
    for row in rows:
        record = _exact_object(row, row_keys, "preliminary row")
        _integer(record["seed"], "preliminary seed")
        _integer(record["receiver_label"], "preliminary label")
        _bound_id(record["receiver_id"], "preliminary receiver")
        for key in ("dbar_norm", "self_norm", "kappa", "log_kappa", "count_gain"):
            _float(record[key], f"preliminary.{key}")
        for mapping_key in (
            "b_norms_by_contributor_count",
            "absolute_log_gain_errors_by_contributor_count",
        ):
            mapping = _exact_object(record[mapping_key], ("1", "8", "32", "180"), mapping_key)
            for item in mapping.values():
                _float(item, mapping_key)
    if (
        type(preliminary["seed_context_aggregates"]) is not list
        or len(preliminary["seed_context_aggregates"]) != 8
    ):
        raise ValueError("preliminary seed/context aggregate count mismatch")
    if (
        type(preliminary["seed_correlations"]) is not list
        or len(preliminary["seed_correlations"]) != 4
    ):
        raise ValueError("preliminary correlation count mismatch")
    if tuple(preliminary["pooled_aggregates"]) != (
        "count_gain_median_A",
        "count_gain_median_B",
        "positive_count_gain_seed_means_A",
        "positive_count_gain_seed_means_B",
        "context_spearman_median",
        "context_spearman_seeds_ge_point_three",
        "log_kappa_iqr_A",
        "log_kappa_iqr_B",
        "global_scalar_relative_error_median_A",
        "global_scalar_relative_error_median_B",
        "full_gain_error_median_A",
        "full_gain_error_median_B",
        "full_gain_error_seed_medians_ge_log_one_point_one_A",
        "full_gain_error_seed_medians_ge_log_one_point_one_B",
    ):
        raise ValueError("preliminary pooled schema mismatch")
    if tuple(preliminary["predicates"]) != (
        "survives_count_gain",
        "survives_context_stability",
        "survives_receiver_heterogeneity",
        "survives_global_scalar",
        "survives_full_gain",
        "close_count_gain",
        "close_context_stability",
        "close_receiver_heterogeneity",
        "close_global_scalar",
        "close_full_gain",
    ) or any(type(item) is not bool for item in preliminary["predicates"].values()):
        raise ValueError("preliminary predicate schema mismatch")
    if tuple(preliminary["decision"]) != (
        "status",
        "first_decisive_clause",
        "full_panel_authorized",
    ):
        raise ValueError("preliminary decision schema mismatch")


def _validate_bootstrap(value: object) -> None:
    bootstrap = _exact_object(
        value,
        ("bit_generator", "seed", "replicates", "complete_labels", "distributions"),
        "bootstrap",
    )
    if (
        bootstrap["bit_generator"] != "PCG64"
        or bootstrap["seed"] != 201
        or bootstrap["replicates"] != 10_000
    ):
        raise ValueError("bootstrap fixed fields mismatch")
    if type(bootstrap["complete_labels"]) is not list or len(bootstrap["complete_labels"]) != 32:
        raise ValueError("bootstrap complete labels mismatch")
    distributions = bootstrap["distributions"]
    if type(distributions) is not list or len(distributions) != 28:
        raise ValueError("bootstrap distribution count mismatch")
    cursor = 0
    for context in ("A", "B"):
        for comparator in ("pa", *CONTROL_ORDER):
            for metric in ("margin_alignment", "margin_slope"):
                record = _exact_object(
                    distributions[cursor],
                    ("context", "comparator", "metric", "values", "values_sha256", "lower_bound"),
                    "bootstrap distribution",
                )
                cursor += 1
                if (record["context"], record["comparator"], record["metric"]) != (
                    context,
                    comparator,
                    metric,
                ):
                    raise ValueError("bootstrap distribution order mismatch")
                if type(record["values"]) is not list or len(record["values"]) != 10_000:
                    raise ValueError("bootstrap values mismatch")
                values = np.asarray(record["values"], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("bootstrap values nonfinite")
                _sha(record["values_sha256"], "bootstrap values hash")
                if hashlib.sha256(values.tobytes(order="C")).hexdigest() != record["values_sha256"]:
                    raise ValueError("bootstrap values hash mismatch")
                _float(record["lower_bound"], "bootstrap lower bound")


def _validate_panel(value: object) -> None:
    panel = _exact_object(
        value,
        (
            "operator_order",
            "rows",
            "seed_context_aggregates",
            "pooled_aggregates",
            "correction_alias_aggregates",
            "predicates",
        ),
        "panel",
    )
    if panel["operator_order"] != list(OPERATOR_ORDER):
        raise ValueError("panel operator order mismatch")
    rows = panel["rows"]
    if type(rows) is not list or len(rows) != 256:
        raise ValueError("panel row count mismatch")
    row_keys = (
        "seed",
        "group",
        "receiver_label",
        "receiver_id",
        "context",
        "batch_sha256",
        "parameter_names_sha256",
        "p_norm",
        "corrections",
        "operators",
    )
    correction_keys = (
        "direction_sha256",
        "correction_norm",
        "matched_update_norm",
        "rdgc_correction_cosine",
    )
    action_keys = ("motion_sha256", "motion_norm", "margin_alignment", "margin_slope")
    for index, row in enumerate(rows):
        record = _exact_object(row, row_keys, "panel row")
        expected_seed = index // 64
        within_seed = index % 64
        expected_receiver = within_seed // 2
        expected_context = ("A", "B")[within_seed % 2]
        if (
            record["seed"] != expected_seed
            or record["group"] != expected_receiver // 8
            or record["context"] != expected_context
        ):
            raise ValueError("panel row order mismatch")
        _integer(record["receiver_label"], "panel receiver label")
        _bound_id(record["receiver_id"], "panel receiver ID")
        _sha(record["batch_sha256"], "panel batch hash")
        _sha(record["parameter_names_sha256"], "panel parameter names hash")
        _float(record["p_norm"], "panel p norm")
        corrections = _exact_object(record["corrections"], CORRECTION_ORDER, "panel corrections")
        for correction in corrections.values():
            item = _exact_object(correction, correction_keys, "correction record")
            _sha(item["direction_sha256"], "correction direction hash")
            for key in correction_keys[1:]:
                _float(item[key], f"correction.{key}")
        operators = _exact_object(record["operators"], OPERATOR_ORDER, "panel operators")
        for action in operators.values():
            item = _exact_object(action, action_keys, "operator action")
            _sha(item["motion_sha256"], "operator motion hash")
            for key in action_keys[1:]:
                _float(item[key], f"operator.{key}")
    seed_context = panel["seed_context_aggregates"]
    if type(seed_context) is not list or len(seed_context) != 64:
        raise ValueError("panel seed/context aggregates mismatch")
    seed_context_keys = (
        "seed",
        "context",
        "operator",
        "alignment_mean",
        "slope_mean",
        "rdgc_minus_operator_alignment_mean",
        "rdgc_minus_operator_slope_mean",
    )
    cursor = 0
    for seed in range(4):
        for context in ("A", "B"):
            for operator in OPERATOR_ORDER:
                record = _exact_object(
                    seed_context[cursor], seed_context_keys, "panel seed/context aggregate"
                )
                cursor += 1
                if (record["seed"], record["context"], record["operator"]) != (
                    seed,
                    context,
                    operator,
                ):
                    raise ValueError("panel seed/context aggregate order mismatch")
                for key in seed_context_keys[3:]:
                    _float(record[key], f"panel aggregate.{key}")
    pooled = panel["pooled_aggregates"]
    if type(pooled) is not list or len(pooled) != 16:
        raise ValueError("panel pooled aggregates mismatch")
    pooled_keys = (
        "context",
        "operator",
        "alignment_mean",
        "slope_mean",
        "rdgc_minus_operator_alignment_mean",
        "rdgc_minus_operator_slope_mean",
        "positive_alignment_seed_means",
        "positive_slope_seed_means",
    )
    cursor = 0
    for context in ("A", "B"):
        for operator in OPERATOR_ORDER:
            record = _exact_object(pooled[cursor], pooled_keys, "panel pooled aggregate")
            cursor += 1
            if (record["context"], record["operator"]) != (context, operator):
                raise ValueError("panel pooled aggregate order mismatch")
            for key in pooled_keys[2:6]:
                _float(record[key], f"panel pooled.{key}")
            for key in pooled_keys[6:]:
                _integer(record[key], f"panel pooled.{key}")
    aliases = panel["correction_alias_aggregates"]
    if type(aliases) is not list or len(aliases) != 6:
        raise ValueError("panel correction aliases mismatch")
    for control, alias in zip(CONTROL_ORDER, aliases, strict=True):
        record = _exact_object(
            alias,
            ("control", "pooled_median_absolute_cosine", "seed_medians_ge_point_nine_nine"),
            "correction alias",
        )
        if record["control"] != control:
            raise ValueError("correction alias order mismatch")
        _float(record["pooled_median_absolute_cosine"], "correction alias cosine")
        _integer(record["seed_medians_ge_point_nine_nine"], "correction alias count")
    predicate_keys = (
        "primary_vs_pa_alignment",
        "primary_vs_pa_slope",
        "primary_vs_all_controls",
        "context_b_vs_pa",
        "context_b_vs_all_controls_alignment_and_slope",
        "correction_nonalias",
        "completeness",
        "close_vs_pa",
        "close_primary_control",
        "close_control_slope",
        "close_correction_alias",
    )
    predicates = _exact_object(panel["predicates"], predicate_keys, "panel predicates")
    if any(type(item) is not bool for item in predicates.values()):
        raise ValueError("panel predicate types mismatch")


def _validate_result_nested(value: dict[str, object]) -> None:
    authority = _exact_object(
        value["authority"], ("candidate", "implementation_plan", "literature_audit"), "authority"
    )
    _validate_ref(authority["candidate"], "authority candidate")
    _validate_ref(authority["implementation_plan"], "authority plan")
    literature = _exact_object(
        authority["literature_audit"],
        (
            "path",
            "sha256",
            "commit",
            "verdict",
            "reviewed_candidate_sha256",
            "primary_source_ids",
        ),
        "authority literature audit",
    )
    if type(literature["path"]) is not str or not literature["path"]:
        raise ValueError("authority literature path mismatch")
    _sha(literature["sha256"], "authority literature hash")
    _commit(literature["commit"], "authority literature commit")
    _sha(literature["reviewed_candidate_sha256"], "authority reviewed candidate")
    if (
        literature["verdict"] != "LIVE-NARROW"
        or type(literature["primary_source_ids"]) is not list
        or len(literature["primary_source_ids"]) != 14
    ):
        raise ValueError("authority literature tuple mismatch")
    source = _exact_object(
        value["source"],
        (
            "source_commit",
            "handoff_commit",
            "handoff_parent",
            "manifest_path",
            "manifest_sha256",
            "files",
            "detached_head",
            "clean_tracked_worktree",
            "ancestry_passed",
        ),
        "source",
    )
    for key in ("source_commit", "handoff_commit", "handoff_parent"):
        _commit(source[key], f"source.{key}")
    _sha(source["manifest_sha256"], "source manifest")
    if (
        source["detached_head"] is not True
        or source["clean_tracked_worktree"] is not True
        or source["ancestry_passed"] is not True
    ):
        raise ValueError("source Boolean gates mismatch")
    if type(source["files"]) is not list or len(source["files"]) != 33:
        raise ValueError("source files mismatch")
    for expected_path, record in zip(RDGC_SOURCE_ORDER, source["files"], strict=True):
        _validate_source_file(record, expected_path, "source file")
    if source["handoff_parent"] != source["source_commit"]:
        raise ValueError("source handoff parent mismatch")
    execution = _exact_object(
        value["execution"],
        (
            "attempt",
            "command",
            "cwd",
            "pid",
            "python_executable",
            "python_version",
            "cuda_visible_devices",
            "cublas_workspace_config",
            "output_path",
            "started_utc",
            "completed_utc",
            "exit_code",
        ),
        "execution",
    )
    if (
        execution["attempt"] != 1
        or type(execution["command"]) is not list
        or not execution["command"]
    ):
        raise ValueError("execution attempt/command mismatch")
    for key in (
        "cwd",
        "python_executable",
        "python_version",
        "cuda_visible_devices",
        "cublas_workspace_config",
        "output_path",
        "started_utc",
        "completed_utc",
    ):
        if type(execution[key]) is not str:
            raise ValueError(f"execution.{key} type mismatch")
    _integer(execution["pid"], "execution pid", minimum=1)
    if type(execution["exit_code"]) is not int or execution["exit_code"] not in (0, 2):
        raise ValueError("execution exit code mismatch")
    environment = value["environment"]
    pre_import = environment["pre_import"]
    if (
        pre_import["python_version"] != "3.12.3"
        or pre_import["cuda_visible_devices"] != "0"
        or pre_import["cublas_workspace_config"] != ":4096:8"
    ):
        raise ValueError("registered environment values mismatch")
    _commit(pre_import["source_commit"], "environment source commit")
    _sha(pre_import["source_files_sha256"], "environment source files")
    _sha(pre_import["manifest_sha256"], "environment manifest")
    runtime = environment["torch_runtime"]
    if runtime is not None:
        runtime_record = _exact_object(
            runtime,
            (
                "torch_version",
                "cuda_runtime_version",
                "cudnn_version",
                "device_index",
                "device_name",
                "device_capability",
                "deterministic_algorithms",
                "allow_tf32_matmul",
                "allow_tf32_cudnn",
            ),
            "Torch runtime",
        )
        if (
            runtime_record["device_index"] != 0
            or runtime_record["deterministic_algorithms"] is not True
            or runtime_record["allow_tf32_matmul"] is not False
            or runtime_record["allow_tf32_cudnn"] is not False
        ):
            raise ValueError("Torch deterministic runtime mismatch")
    binding = _exact_object(
        value["binding"],
        (
            "rsta_candidate",
            "rsta_gate2_audit",
            "rsta_producer_source_commit",
            "rsta_producer_handoff_commit",
            "rsta_artifact",
            "rsta_producer_pid",
            "rsta_producer_exit_code",
            "verifier_source_commit",
            "verifier_handoff_commit",
            "verifier_manifest_sha256",
            "validation_receipt",
            "rsta_scientific_status",
            "rsta_scientific_decision",
            "rsta_first_decisive_clause",
            "seeds",
        ),
        "binding",
    )
    _validate_ref(binding["rsta_candidate"], "binding RSTA candidate")
    _validate_ref(binding["rsta_gate2_audit"], "binding RSTA audit")
    for key in (
        "rsta_producer_source_commit",
        "rsta_producer_handoff_commit",
        "verifier_source_commit",
        "verifier_handoff_commit",
    ):
        _commit(binding[key], f"binding.{key}")
    _validate_artifact_ref(binding["rsta_artifact"], "binding RSTA artifact")
    _sha(binding["verifier_manifest_sha256"], "binding verifier manifest")
    receipt = _exact_object(
        binding["validation_receipt"], ("path", "sha256", "status"), "binding receipt"
    )
    _sha(receipt["sha256"], "binding receipt")
    if (
        receipt["status"] != "VALID"
        or binding["rsta_producer_pid"] != 1002393
        or binding["rsta_producer_exit_code"] != 0
        or binding["rsta_scientific_status"] != "VALID"
        or binding["rsta_scientific_decision"] != "UNRESOLVED"
        or binding["rsta_first_decisive_clause"] != "no_pass_or_fail_rule"
    ):
        raise ValueError("binding fixed provenance mismatch")
    if type(binding["seeds"]) is not list or len(binding["seeds"]) != 4:
        raise ValueError("binding seeds mismatch")
    for seed, record in enumerate(binding["seeds"]):
        _validate_seed_artifacts(record, seed, "binding seed")
    integrity = value["integrity"]
    if integrity is not None and value["status"] != "INVALID":
        integrity_record = _exact_object(
            integrity,
            (
                "seeds",
                "all_four_passed",
                "candidate_calls_before_all_four",
                "candidate_state_before_all_four",
            ),
            "integrity",
        )
        if (
            type(integrity_record["seeds"]) is not list
            or len(integrity_record["seeds"]) != 4
            or integrity_record["all_four_passed"] is not True
            or integrity_record["candidate_calls_before_all_four"] != 0
            or integrity_record["candidate_state_before_all_four"] is not False
        ):
            raise ValueError("integrity prefix relation mismatch")
    if value["selection"] is not None:
        _validate_selection(value["selection"])
    if value["preliminary"] is not None:
        _validate_preliminary(value["preliminary"])
    if value["bootstrap"] is not None:
        _validate_bootstrap(value["bootstrap"])
    if value["panel"] is not None:
        _validate_panel(value["panel"])


def validate_future_manifest(value: dict[str, object]) -> None:
    if type(value) is not dict or tuple(value) != (
        "schema_version",
        "candidate",
        "implementation_plan",
        "upstream_rsta",
        "literature_audit",
        "validation_receipt",
        "historical",
        "current_scientific_source",
        "artifact_schema",
        "seeds",
    ):
        raise ValueError("future manifest top-level schema/order mismatch")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("future manifest schema version mismatch")
    _validate_ref(value["candidate"], "future candidate")
    _validate_ref(value["implementation_plan"], "future plan")
    upstream = _exact_object(
        value["upstream_rsta"],
        (
            "candidate",
            "gate2_audit",
            "producer_source_commit",
            "producer_handoff_commit",
            "producer_artifact",
            "producer_pid",
            "producer_exit_code",
            "verifier_source_commit",
            "verifier_handoff_commit",
            "verifier_manifest",
            "scientific_status",
            "scientific_decision",
            "first_decisive_clause",
        ),
        "future upstream RSTA",
    )
    _validate_ref(upstream["candidate"], "future upstream candidate")
    _validate_ref(upstream["gate2_audit"], "future upstream Gate-2 audit")
    for key in (
        "producer_source_commit",
        "producer_handoff_commit",
        "verifier_source_commit",
        "verifier_handoff_commit",
    ):
        _commit(upstream[key], f"future upstream.{key}")
    _validate_artifact_ref(upstream["producer_artifact"], "future producer artifact")
    _validate_artifact_ref(upstream["verifier_manifest"], "future verifier manifest")
    if upstream["producer_pid"] != 1002393 or upstream["producer_exit_code"] != 0:
        raise ValueError("future upstream producer process mismatch")
    if (
        upstream["scientific_status"] != "VALID"
        or upstream["scientific_decision"] != "UNRESOLVED"
        or upstream["first_decisive_clause"] != "no_pass_or_fail_rule"
    ):
        raise ValueError("future upstream outcome provenance mismatch")
    literature = _exact_object(
        value["literature_audit"],
        (
            "path",
            "sha256",
            "commit",
            "verdict",
            "reviewed_candidate_sha256",
            "primary_source_ids",
        ),
        "future literature audit",
    )
    if type(literature["path"]) is not str or not literature["path"]:
        raise ValueError("future literature path mismatch")
    _sha(literature["sha256"], "future literature hash")
    _commit(literature["commit"], "future literature commit")
    _sha(literature["reviewed_candidate_sha256"], "future reviewed candidate")
    if (
        literature["verdict"] != "LIVE-NARROW"
        or type(literature["primary_source_ids"]) is not list
        or len(literature["primary_source_ids"]) != 14
        or any(type(item) is not str or not item for item in literature["primary_source_ids"])
    ):
        raise ValueError("future literature audit tuple mismatch")
    receipt = _exact_object(
        value["validation_receipt"],
        (
            "path",
            "sha256",
            "status",
            "verifier_source_commit",
            "verifier_handoff_commit",
            "artifact_path",
            "artifact_sha256",
        ),
        "future validation receipt",
    )
    if (
        type(receipt["path"]) is not str
        or not receipt["path"]
        or receipt["status"] != "VALID"
        or type(receipt["artifact_path"]) is not str
        or not receipt["artifact_path"]
    ):
        raise ValueError("future validation receipt values mismatch")
    for key in ("sha256", "artifact_sha256"):
        _sha(receipt[key], f"future receipt.{key}")
    for key in ("verifier_source_commit", "verifier_handoff_commit"):
        _commit(receipt[key], f"future receipt.{key}")
    historical = _exact_object(
        value["historical"], ("manifest_path", "manifest_sha256", "seeds"), "future historical"
    )
    if type(historical["manifest_path"]) is not str or not historical["manifest_path"]:
        raise ValueError("future historical manifest path mismatch")
    _sha(historical["manifest_sha256"], "future historical manifest hash")
    if type(historical["seeds"]) is not list or len(historical["seeds"]) != 4:
        raise ValueError("future historical seed count mismatch")
    for seed, record in enumerate(historical["seeds"]):
        _validate_seed_artifacts(record, seed, "future historical seed")
    source = value["current_scientific_source"]
    if type(source) is not dict or tuple(source) != ("git_revision", "files"):
        raise ValueError("future source schema/order mismatch")
    files = source["files"]
    _commit(source["git_revision"], "future source revision")
    if type(files) is not list or len(files) != len(RDGC_SOURCE_ORDER):
        raise ValueError("future source file count mismatch")
    for expected_path, record in zip(RDGC_SOURCE_ORDER, files, strict=True):
        if (
            type(record) is not dict
            or tuple(record) != ("path", "sha256", "git_blob")
            or record["path"] != expected_path
        ):
            raise ValueError("future source file order mismatch")
        _sha(record["sha256"], "future source file hash")
        if type(record["git_blob"]) is not str or not record["git_blob"]:
            raise ValueError("future source Git blob mismatch")
    artifact = value["artifact_schema"]
    if type(artifact) is not dict or tuple(artifact) != (
        "result_path_template",
        "schema_version",
        "diagnostic",
        "mode",
        "statuses",
        "phases",
        "top_level_keys",
        "pre_import_invalid_null_fields",
        "post_import_invalid_null_fields",
        "operator_order",
        "contributor_counts",
    ):
        raise ValueError("future artifact projection schema/order mismatch")
    expected = {
        "schema_version": 1,
        "diagnostic": "pass205_rdgc_stage_b",
        "mode": "scientific_no_training_virtual_update",
        "statuses": ["PASS", "CLOSE", "UNRESOLVED", "INVALID"],
        "phases": ["pre_import", "integrity", "preliminary", "full_panel"],
        "top_level_keys": list(RESULT_KEYS),
        "pre_import_invalid_null_fields": [
            "integrity",
            "selection",
            "preliminary",
            "panel",
            "bootstrap",
        ],
        "post_import_invalid_null_fields": ["selection", "preliminary", "panel", "bootstrap"],
        "operator_order": list(OPERATOR_ORDER),
        "contributor_counts": list(CONTRIBUTOR_COUNTS),
    }
    for key, expected_value in expected.items():
        if artifact[key] != expected_value or type(artifact[key]) is not type(expected_value):
            raise ValueError(f"future artifact projection mismatch: {key}")
    if type(artifact["result_path_template"]) is not str or not artifact["result_path_template"]:
        raise ValueError("future result path template mismatch")
    if value["seeds"] != [0, 1, 2, 3] or any(type(seed) is not int for seed in value["seeds"]):
        raise ValueError("future seeds mismatch")


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    validate_scientific_payload(payload)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ValueError("output parent must be an existing regular directory")
    encoded = (
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _action_evidence(torch_module: Any, value: Any) -> tuple[float, str]:
    _require_fp32_finite(torch_module, value, nonzero=True)
    flat = value.to(torch_module.float64).reshape(-1)
    norm = float(torch_module.sqrt((flat * flat).sum()).item())
    return norm, _tensor_sha256(value, torch_module)


def _run_preliminary_context(
    *,
    torch_module: Any,
    rsta_module: types.ModuleType,
    bound: Any,
    phase: dict[str, object],
    group: dict[str, object],
    context_record: dict[str, object],
    expected_dimension: int = 512,
) -> list[dict[str, object]]:
    """Compute one registered B=180 preliminary graph and reduce it immediately."""
    batch_ids = tuple(context_record["batch_ids"])
    if len(batch_ids) != 180:
        raise ValueError("preliminary context requires B=180")
    receiver_ids = tuple(group["receiver_ids"])
    receiver_labels = tuple(group["receiver_labels"])
    if len(receiver_ids) != 8 or len(receiver_labels) != 8:
        raise ValueError("preliminary context requires eight receivers")
    cache = rsta_module.cache_seed_training_tensors(bound, batch_ids)
    if tuple(cache.example_ids) != batch_ids:
        raise ValueError("transform cache order differs from the frozen batch")
    label_by_id = dict(zip(bound.train_example_ids, bound.train_labels, strict=True))
    images = cache.batch(batch_ids)
    model = rsta_module.make_rsta_diagnostic_clone(rsta_module._load_scientific_model(bound))
    parameter = next(iter(model.parameters()), None)
    if parameter is None:
        raise ValueError("RDGC model has no parameters")
    device, dtype = parameter.device, parameter.dtype
    images = images.to(device=device, dtype=dtype)
    labels = torch_module.as_tensor(
        [label_by_id[example_id] for example_id in batch_ids],
        device=device,
        dtype=torch_module.long,
    )
    proxies = torch_module.tensor(np.array(bound.proxies, copy=True), device=device, dtype=dtype)
    proxy_labels = torch_module.tensor(
        np.array(bound.proxy_labels, copy=True), device=device, dtype=torch_module.long
    )
    encoder, parameters, parameter_names = rsta_module._functional_encoder(
        model,
        images,
        expected_batch_size=180,
        expected_dimension=expected_dimension,
    )
    rsta_module._validate_encoder_parameter_dependencies(encoder, parameters, parameter_names)
    z, vjp_function = torch_module.func.vjp(encoder, parameters)
    from sfora.image_end_to_end import _proxy_anchor_loss

    loss = _proxy_anchor_loss(
        z,
        labels,
        proxy_embeddings=proxies.detach(),
        proxy_labels=proxy_labels,
        alpha=float(bound.alpha),
        delta=float(bound.delta),
        torch_module=torch_module,
    )
    dbar = -torch_module.autograd.grad(loss, z, create_graph=True)[0]
    _require_fp32_finite(torch_module, dbar)
    rows: list[dict[str, object]] = []
    for receiver_label, receiver_id in zip(receiver_labels, receiver_ids, strict=True):
        receiver_index = batch_ids.index(receiver_id)
        diagonal_cotangent = torch_module.zeros_like(dbar)
        diagonal_cotangent[receiver_index] = dbar[receiver_index]
        diagonal_gradient = vjp_function(diagonal_cotangent)[0]
        _, diagonal_full_motion = torch_module.func.jvp(
            encoder, (parameters,), (diagonal_gradient,)
        )
        self_motion = diagonal_full_motion[receiver_index].clone()
        self_norm, self_sha256 = _action_evidence(torch_module, self_motion)
        dbar_row = dbar[receiver_index]
        dbar_norm, _ = _action_evidence(torch_module, dbar_row)
        del diagonal_cotangent, diagonal_gradient, diagonal_full_motion
        masks = nested_contributor_masks(
            batch_ids,
            receiver_id,
            seed=int(bound.seed),
            context=str(context_record["context"]),
        )
        norms: dict[str, float] = {}
        errors: dict[str, float] = {}
        b_one_sha256: str | None = None
        for count, mask in zip(CONTRIBUTOR_COUNTS, masks, strict=True):
            mask_tensor = torch_module.tensor(mask, device=device, dtype=dtype).reshape(180, 1)
            masked_cotangent = dbar * mask_tensor
            masked_gradient = vjp_function(masked_cotangent)[0]
            _, masked_full_motion = torch_module.func.jvp(
                encoder, (parameters,), (masked_gradient,)
            )
            motion = masked_full_motion[receiver_index].clone()
            motion_norm, motion_sha256 = _action_evidence(torch_module, motion)
            if count == 1:
                if motion is self_motion or not bool(torch_module.equal(motion, self_motion)):
                    raise ValueError("registered b_1/self live equality failed")
                b_one_sha256 = motion_sha256
            norms[str(count)] = motion_norm
            errors[str(count)] = abs(math.log((motion_norm + EPSILON) / (self_norm + EPSILON)))
            del mask_tensor, masked_cotangent, masked_gradient, masked_full_motion, motion
        if b_one_sha256 != self_sha256:
            raise ValueError("registered b_1/self hashes differ")
        kappa = (self_norm + EPSILON) / (dbar_norm + EPSILON)
        rows.append(
            {
                "seed": int(bound.seed),
                "context": context_record["context"],
                "receiver_label": receiver_label,
                "receiver_id": receiver_id,
                "dbar_norm": dbar_norm,
                "self_norm": self_norm,
                "kappa": kappa,
                "log_kappa": math.log(kappa),
                "b_norms_by_contributor_count": norms,
                "absolute_log_gain_errors_by_contributor_count": errors,
                "count_gain": errors["180"] - errors["8"],
            }
        )
        del masks, self_motion, dbar_row
    del (
        dbar,
        loss,
        z,
        vjp_function,
        encoder,
        parameters,
        parameter_names,
        proxy_labels,
        proxies,
        labels,
        images,
        model,
        cache,
    )
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    return rows


def run_preliminary_science(
    selection: dict[str, object],
    bounds: tuple[Any, ...],
    *,
    torch_module: Any,
    rsta_module: types.ModuleType,
    expected_dimension: int = 512,
) -> dict[str, object]:
    if len(bounds) != 4 or [int(bound.seed) for bound in bounds] != list(range(4)):
        raise ValueError("preliminary bounds must be exact seed order")
    phase = selection["preliminary"]
    if type(phase) is not dict or len(phase["groups"]) != 1:
        raise ValueError("preliminary selection requires one group")
    group = phase["groups"][0]
    rows: list[dict[str, object]] = []
    for bound in bounds:
        for context_record in group["contexts"]:
            rows.extend(
                _run_preliminary_context(
                    torch_module=torch_module,
                    rsta_module=rsta_module,
                    bound=bound,
                    phase=phase,
                    group=group,
                    context_record=context_record,
                    expected_dimension=expected_dimension,
                )
            )
    aggregates = preliminary_metrics(rows)
    decision = decide_preliminary(aggregates)
    seed_context_aggregates: list[dict[str, object]] = []
    seed_correlations: list[dict[str, object]] = []
    for seed in range(4):
        context_logs: dict[str, np.ndarray] = {}
        for context in ("A", "B"):
            selected = [row for row in rows if row["seed"] == seed and row["context"] == context]
            logs = np.asarray([float(row["log_kappa"]) for row in selected], dtype=np.float64)
            context_logs[context] = logs
            global_log = float(np.median(logs))
            seed_context_aggregates.append(
                {
                    "seed": seed,
                    "context": context,
                    "count_gain_mean": float(
                        np.mean(
                            np.asarray(
                                [float(row["count_gain"]) for row in selected],
                                dtype=np.float64,
                            ),
                            dtype=np.float64,
                        )
                    ),
                    "log_kappa_median": global_log,
                    "log_kappa_iqr": float(np.percentile(logs, 75) - np.percentile(logs, 25)),
                    "global_scalar_relative_error_median": float(
                        np.median(np.abs(np.exp(logs - global_log) - 1.0))
                    ),
                    "full_gain_error_median": float(
                        np.median(
                            np.asarray(
                                [
                                    float(
                                        row["absolute_log_gain_errors_by_contributor_count"]["180"]
                                    )
                                    for row in selected
                                ],
                                dtype=np.float64,
                            )
                        )
                    ),
                }
            )
        ranks_a = np.argsort(np.argsort(context_logs["A"], kind="stable"), kind="stable")
        ranks_b = np.argsort(np.argsort(context_logs["B"], kind="stable"), kind="stable")
        seed_correlations.append(
            {
                "seed": seed,
                "spearman": float(np.corrcoef(ranks_a, ranks_b)[0, 1]),
            }
        )
    predicates = {
        "survives_count_gain": all(
            float(aggregates[f"count_gain_median_{context}"]) >= math.log(1.10)
            and int(aggregates[f"positive_count_gain_seed_means_{context}"]) >= 3
            for context in ("A", "B")
        ),
        "survives_context_stability": float(aggregates["context_spearman_median"]) >= 0.50
        and int(aggregates["context_spearman_seeds_ge_point_three"]) >= 3,
        "survives_receiver_heterogeneity": all(
            float(aggregates[f"log_kappa_iqr_{context}"]) >= math.log(1.10)
            for context in ("A", "B")
        ),
        "survives_global_scalar": all(
            float(aggregates[f"global_scalar_relative_error_median_{context}"]) >= 0.05
            for context in ("A", "B")
        ),
        "survives_full_gain": all(
            float(aggregates[f"full_gain_error_median_{context}"]) >= math.log(1.25)
            and int(aggregates[f"full_gain_error_seed_medians_ge_log_one_point_one_{context}"]) >= 3
            for context in ("A", "B")
        ),
        "close_count_gain": decision["first_decisive_clause"] == "close_count_gain",
        "close_context_stability": decision["first_decisive_clause"] == "close_context_spearman",
        "close_receiver_heterogeneity": decision["first_decisive_clause"] == "close_log_kappa_iqr",
        "close_global_scalar": decision["first_decisive_clause"] == "close_global_scalar",
        "close_full_gain": decision["first_decisive_clause"] == "close_full_gain",
    }
    return {
        "operator_counts": {
            "forwards_per_seed_context": 1,
            "dbars_per_seed_context": 1,
            "diagonal_vjps_per_seed_context": 8,
            "diagonal_jvps_per_seed_context": 8,
            "masked_vjps_per_seed_context": 32,
            "masked_receiver_jvps_per_seed_context": 32,
        },
        "rows": rows,
        "seed_context_aggregates": seed_context_aggregates,
        "seed_correlations": seed_correlations,
        "pooled_aggregates": aggregates,
        "predicates": predicates,
        "decision": decision,
    }


def _open_panel_context(
    *,
    torch_module: Any,
    rsta_module: types.ModuleType,
    bound: Any,
    batch_ids: tuple[str, ...],
    receiver_ids: tuple[str, ...],
    expected_dimension: int,
) -> dict[str, object]:
    cache = rsta_module.cache_seed_training_tensors(bound, batch_ids)
    if tuple(cache.example_ids) != batch_ids:
        raise ValueError("panel transform cache order differs")
    label_by_id = dict(zip(bound.train_example_ids, bound.train_labels, strict=True))
    model = rsta_module.make_rsta_diagnostic_clone(rsta_module._load_scientific_model(bound))
    parameter = next(iter(model.parameters()), None)
    if parameter is None:
        raise ValueError("panel model has no parameters")
    device, dtype = parameter.device, parameter.dtype
    images = cache.batch(batch_ids).to(device=device, dtype=dtype)
    labels = torch_module.as_tensor(
        [label_by_id[value] for value in batch_ids], device=device, dtype=torch_module.long
    )
    proxies = torch_module.tensor(np.array(bound.proxies, copy=True), device=device, dtype=dtype)
    proxy_labels = torch_module.tensor(
        np.array(bound.proxy_labels, copy=True), device=device, dtype=torch_module.long
    )
    encoder, parameters, parameter_names = rsta_module._functional_encoder(
        model,
        images,
        expected_batch_size=180,
        expected_dimension=expected_dimension,
    )
    rsta_module._validate_encoder_parameter_dependencies(encoder, parameters, parameter_names)
    z, vjp_function = torch_module.func.vjp(encoder, parameters)
    from sfora.image_end_to_end import _proxy_anchor_loss

    loss = _proxy_anchor_loss(
        z,
        labels,
        proxy_embeddings=proxies.detach(),
        proxy_labels=proxy_labels,
        alpha=float(bound.alpha),
        delta=float(bound.delta),
        torch_module=torch_module,
    )
    dbar = -torch_module.autograd.grad(loss, z, create_graph=True)[0]
    _require_fp32_finite(torch_module, dbar)
    return {
        "cache": cache,
        "model": model,
        "images": images,
        "labels": labels,
        "proxies": proxies,
        "proxy_labels": proxy_labels,
        "encoder": encoder,
        "parameters": parameters,
        "parameter_names": tuple(parameter_names),
        "z": z,
        "vjp_function": vjp_function,
        "loss": loss,
        "dbar": dbar,
        "receiver_indices": tuple(batch_ids.index(value) for value in receiver_ids),
        "device": device,
    }


def _close_panel_context(context: dict[str, object], torch_module: Any) -> None:
    context.clear()
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _ordinary_pa_direction(
    context: dict[str, object], torch_module: Any
) -> tuple[tuple[str, ...], tuple[Any, ...], tuple[tuple[str, Any], ...]]:
    parameters = context["parameters"]
    names = context["parameter_names"]
    gradients = torch_module.autograd.grad(
        context["loss"], tuple(parameters.values()), create_graph=False
    )
    p = tuple(
        (-gradient).detach().to(device="cpu", dtype=torch_module.float32).contiguous()
        for gradient in gradients
    )
    named_parameters = tuple(
        (
            name,
            parameters[name].detach().to(device="cpu", dtype=torch_module.float32).contiguous(),
        )
        for name in names
    )
    return names, p, named_parameters


def _pgn_coefficients_science(context: dict[str, object], torch_module: Any) -> Any:
    dbar = context["dbar"]
    vjp_function = context["vjp_function"]
    norms: list[Any] = []
    for row in range(180):
        cotangent = torch_module.zeros_like(dbar)
        cotangent[row] = dbar[row].detach()
        gradient = vjp_function(cotangent)[0]
        norm = (
            fp64_named_norm(
                torch_module, tuple(gradient[name] for name in context["parameter_names"])
            )
            .detach()
            .cpu()
        )
        norms.append(norm)
        del cotangent, gradient, norm
    coefficients = pgn_detached_coefficients(torch_module, tuple(norms))
    del norms
    return coefficients


def _parameter_correction_science(
    *,
    operator: str,
    receiver_index: int,
    context: dict[str, object],
    p: tuple[Any, ...],
    pgn_coefficients: Any | None,
    torch_module: Any,
) -> tuple[Any, ...]:
    names = context["parameter_names"]
    parameters = context["parameters"]
    if operator == "layerwise_trust_ratio":
        named = tuple((name, parameters[name]) for name in names)
        device_p = tuple(value.to(context["device"]) for value in p)
        result = layerwise_trust_ratio_direction(torch_module, named, device_p)
        return tuple(value.detach().cpu().contiguous() for value in result)
    dbar = context["dbar"]
    vjp_function = context["vjp_function"]
    encoder = context["encoder"]
    receiver_fields: list[dict[str, Any]] = []
    for index in context["receiver_indices"]:
        cotangent = torch_module.zeros_like(dbar)
        cotangent[index] = dbar[index]
        gradient = vjp_function(cotangent)[0]
        _, full_motion = torch_module.func.jvp(encoder, (parameters,), (gradient,))
        receiver_fields.append({"s": full_motion[index], "dbar": dbar[index]})
        del cotangent, gradient, full_motion
    global_gradient = None
    global_motion = None
    if operator == "per_example_gradient_normalized":
        if pgn_coefficients is None:
            raise ValueError("PGN correction requires detached coefficients")
        weighted = pgn_weighted_cotangent(
            torch_module, dbar, pgn_coefficients.to(context["device"])
        )
        pgn_gradient = vjp_function(weighted)[0]
        _, pgn_full_motion = torch_module.func.jvp(encoder, (parameters,), (pgn_gradient,))
        pgn_motion = pgn_full_motion[receiver_index]
        b = pgn_motion
    else:
        global_gradient = vjp_function(dbar)[0]
        _, global_motion = torch_module.func.jvp(encoder, (parameters,), (global_gradient,))
        b = global_motion[receiver_index]
        pgn_motion = b
    field = {
        "named_parameters": tuple((name, parameters[name]) for name in names),
        "p": tuple(value.to(context["device"]) for value in p),
        "b": b,
        "s": receiver_fields[context["receiver_indices"].index(receiver_index)]["s"],
        "dbar": dbar[receiver_index],
        "receiver_fields": receiver_fields,
        "pgn_motion": pgn_motion,
    }
    result = compute_parameter_correction(field, operator, torch_module=torch_module)
    del field, receiver_fields, global_gradient, global_motion, b
    return result


def _support_pool(selection: dict[str, object]) -> tuple[list[str], list[int]]:
    ids: list[str] = []
    labels: list[int] = []
    for phase_name in ("preliminary", "panel"):
        phase = selection[phase_name]
        for label in phase["identity_labels"]:
            ids.append(phase["support_ids_by_label"][str(label)][0])
            labels.append(label)
    return ids, labels


def _margin_direction(
    *,
    rsta_module: types.ModuleType,
    selection: dict[str, object],
    phase: dict[str, object],
    bound: Any,
    context: dict[str, object],
    batch_ids: tuple[str, ...],
    receiver_label: int,
    receiver_id: str,
    torch_module: Any,
) -> Any:
    index_by_id = {value: index for index, value in enumerate(bound.train_example_ids)}
    receiver_index = batch_ids.index(receiver_id)
    receiver = context["z"][receiver_index].detach().cpu().numpy().astype(np.float64)
    positive_ids = phase["support_ids_by_label"][str(receiver_label)]
    positives = np.asarray(
        [bound.train_embeddings[index_by_id[value]] for value in positive_ids],
        dtype=np.float64,
    )
    support_ids, support_labels = _support_pool(selection)
    support_descriptors = np.asarray(
        [bound.train_embeddings[index_by_id[value]] for value in support_ids],
        dtype=np.float64,
    )
    _, foreign = rsta_module.select_foreign_supports(
        receiver,
        receiver_label=receiver_label,
        support_ids=support_ids,
        support_labels=support_labels,
        support_descriptors=support_descriptors,
        current_batch_ids=set(batch_ids),
    )
    q = rsta_module.smooth_margin_gradient(receiver, positives, foreign)
    return torch_module.tensor(q, device=context["device"], dtype=torch_module.float32)


def _evaluate_panel_action(
    *,
    direction: tuple[Any, ...],
    selection: dict[str, object],
    phase: dict[str, object],
    group: dict[str, object],
    context_record: dict[str, object],
    receiver_label: int,
    receiver_id: str,
    bound: Any,
    torch_module: Any,
    rsta_module: types.ModuleType,
    expected_dimension: int,
) -> dict[str, object]:
    batch_ids = tuple(context_record["batch_ids"])
    context = _open_panel_context(
        torch_module=torch_module,
        rsta_module=rsta_module,
        bound=bound,
        batch_ids=batch_ids,
        receiver_ids=tuple(group["receiver_ids"]),
        expected_dimension=expected_dimension,
    )
    try:
        names = context["parameter_names"]
        parameters = context["parameters"]
        device_direction = tuple(value.to(context["device"]) for value in direction)

        def receiver_function(values: tuple[Any, ...]) -> Any:
            current = dict(zip(names, values, strict=True))
            return context["encoder"](current)[batch_ids.index(receiver_id)]

        q = _margin_direction(
            rsta_module=rsta_module,
            selection=selection,
            phase=phase,
            bound=bound,
            context=context,
            batch_ids=batch_ids,
            receiver_label=receiver_label,
            receiver_id=receiver_id,
            torch_module=torch_module,
        )
        return evaluate_virtual_direction(
            {"function": receiver_function, "parameters": tuple(parameters.values())},
            device_direction,
            q,
            torch_module=torch_module,
        )
    finally:
        _close_panel_context(context, torch_module)


def _named_direction_sha256(names: tuple[str, ...], values: tuple[Any, ...]) -> str:
    digest = hashlib.sha256()
    for name, value in zip(names, values, strict=True):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def _correction_for_receiver(
    *,
    operator: str,
    p: tuple[Any, ...],
    named_parameters: tuple[tuple[str, Any], ...],
    selection: dict[str, object],
    phase: dict[str, object],
    group: dict[str, object],
    receiver_id: str,
    bound: Any,
    torch_module: Any,
    rsta_module: types.ModuleType,
    expected_dimension: int,
) -> tuple[Any, ...]:
    if operator == "layerwise_trust_ratio":
        return layerwise_trust_ratio_direction(torch_module, named_parameters, p)
    context_a = group["contexts"][0]
    batch_ids = tuple(context_a["batch_ids"])
    coefficients = None
    if operator == "per_example_gradient_normalized":
        coefficient_context = _open_panel_context(
            torch_module=torch_module,
            rsta_module=rsta_module,
            bound=bound,
            batch_ids=batch_ids,
            receiver_ids=tuple(group["receiver_ids"]),
            expected_dimension=expected_dimension,
        )
        try:
            coefficients = _pgn_coefficients_science(coefficient_context, torch_module)
        finally:
            _close_panel_context(coefficient_context, torch_module)
    context = _open_panel_context(
        torch_module=torch_module,
        rsta_module=rsta_module,
        bound=bound,
        batch_ids=batch_ids,
        receiver_ids=tuple(group["receiver_ids"]),
        expected_dimension=expected_dimension,
    )
    try:
        return _parameter_correction_science(
            operator=operator,
            receiver_index=batch_ids.index(receiver_id),
            context=context,
            p=p,
            pgn_coefficients=coefficients,
            torch_module=torch_module,
        )
    finally:
        _close_panel_context(context, torch_module)
        del coefficients


def _aggregate_panel(
    rows: list[dict[str, object]],
    correction_cosines: dict[str, list[tuple[int, float]]],
    bootstrap: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    by_key = {(row["seed"], row["context"], row["receiver_label"]): row for row in rows}
    labels = bootstrap["complete_labels"]
    seed_context: list[dict[str, object]] = []
    pooled: list[dict[str, object]] = []
    bootstrap_by_key = {
        (record["context"], record["comparator"], record["metric"]): record
        for record in bootstrap["distributions"]
    }

    def difference_values(seed: int, context: str, operator: str, metric: str) -> list[float]:
        return [
            float(by_key[(seed, context, label)]["operators"]["rdgc"][metric])
            - float(by_key[(seed, context, label)]["operators"][operator][metric])
            for label in labels
        ]

    metric_ledger: dict[tuple[str, str, str], dict[str, object]] = {}
    for seed in range(4):
        for context in ("A", "B"):
            for operator in OPERATOR_ORDER:
                alignments = [
                    float(by_key[(seed, context, label)]["operators"][operator]["margin_alignment"])
                    for label in labels
                ]
                slopes = [
                    float(by_key[(seed, context, label)]["operators"][operator]["margin_slope"])
                    for label in labels
                ]
                seed_context.append(
                    {
                        "seed": seed,
                        "context": context,
                        "operator": operator,
                        "alignment_mean": float(np.mean(alignments, dtype=np.float64)),
                        "slope_mean": float(np.mean(slopes, dtype=np.float64)),
                        "rdgc_minus_operator_alignment_mean": float(
                            np.mean(
                                difference_values(seed, context, operator, "margin_alignment"),
                                dtype=np.float64,
                            )
                        ),
                        "rdgc_minus_operator_slope_mean": float(
                            np.mean(
                                difference_values(seed, context, operator, "margin_slope"),
                                dtype=np.float64,
                            )
                        ),
                    }
                )
    for context in ("A", "B"):
        for operator in OPERATOR_ORDER:
            selected = [
                record
                for record in seed_context
                if record["context"] == context and record["operator"] == operator
            ]
            alignment_differences = [
                float(record["rdgc_minus_operator_alignment_mean"]) for record in selected
            ]
            slope_differences = [
                float(record["rdgc_minus_operator_slope_mean"]) for record in selected
            ]
            record = {
                "context": context,
                "operator": operator,
                "alignment_mean": float(
                    np.mean(
                        [float(item["alignment_mean"]) for item in selected],
                        dtype=np.float64,
                    )
                ),
                "slope_mean": float(
                    np.mean([float(item["slope_mean"]) for item in selected], dtype=np.float64)
                ),
                "rdgc_minus_operator_alignment_mean": float(
                    np.mean(alignment_differences, dtype=np.float64)
                ),
                "rdgc_minus_operator_slope_mean": float(
                    np.mean(slope_differences, dtype=np.float64)
                ),
                "positive_alignment_seed_means": sum(
                    value > 0.0 for value in alignment_differences
                ),
                "positive_slope_seed_means": sum(value > 0.0 for value in slope_differences),
            }
            pooled.append(record)
            if operator != "rdgc":
                metric_ledger[(context, operator, "alignment")] = {
                    "pooled_difference": record["rdgc_minus_operator_alignment_mean"],
                    "lower_bound": bootstrap_by_key[(context, operator, "margin_alignment")][
                        "lower_bound"
                    ],
                    "positive_seed_means": record["positive_alignment_seed_means"],
                    "nonpositive_seed_means": 4 - int(record["positive_alignment_seed_means"]),
                }
                metric_ledger[(context, operator, "slope")] = {
                    "pooled_difference": record["rdgc_minus_operator_slope_mean"],
                    "lower_bound": bootstrap_by_key[(context, operator, "margin_slope")][
                        "lower_bound"
                    ],
                    "positive_seed_means": record["positive_slope_seed_means"],
                    "nonpositive_seed_means": 4 - int(record["positive_slope_seed_means"]),
                }
    aliases: list[dict[str, object]] = []
    alias_map: dict[str, dict[str, object]] = {}
    for control in CONTROL_ORDER:
        values = correction_cosines[control]
        pooled_median = float(np.median(np.asarray([value for _, value in values])))
        seed_medians = [
            float(
                np.median(np.asarray([value for seed_value, value in values if seed_value == seed]))
            )
            for seed in range(4)
        ]
        alias = {
            "control": control,
            "pooled_median_absolute_cosine": pooled_median,
            "seed_medians_ge_point_nine_nine": sum(value >= 0.99 for value in seed_medians),
        }
        aliases.append(alias)
        alias_map[control] = {
            "pooled_median_absolute_cosine": pooled_median,
            "seed_medians_ge_point_nine_nine": alias["seed_medians_ge_point_nine_nine"],
        }
    controls = {
        control: {
            "primary_alignment": metric_ledger[("A", control, "alignment")],
            "primary_slope": metric_ledger[("A", control, "slope")],
            "context_b_alignment": metric_ledger[("B", control, "alignment")],
            "context_b_slope": metric_ledger[("B", control, "slope")],
        }
        for control in CONTROL_ORDER
    }
    decision_input = {
        "primary_pa_alignment": metric_ledger[("A", "pa", "alignment")],
        "primary_pa_slope": metric_ledger[("A", "pa", "slope")],
        "context_b_pa_alignment": metric_ledger[("B", "pa", "alignment")],
        "controls": controls,
        "correction_aliases": alias_map,
        "completeness": True,
    }
    decision = decide_panel(decision_input, bootstrap)
    predicates = {
        "primary_vs_pa_alignment": _metric_pass(
            decision_input["primary_pa_alignment"],
            alignment_floor=0.02,
            strict=False,
        ),
        "primary_vs_pa_slope": _metric_pass(decision_input["primary_pa_slope"]),
        "primary_vs_all_controls": all(
            _metric_pass(controls[control][metric])
            for control in CONTROL_ORDER
            for metric in ("primary_alignment", "primary_slope")
        ),
        "context_b_vs_pa": _metric_pass(decision_input["context_b_pa_alignment"]),
        "context_b_vs_all_controls_alignment_and_slope": all(
            _metric_pass(controls[control][metric])
            for control in CONTROL_ORDER
            for metric in ("context_b_alignment", "context_b_slope")
        ),
        "correction_nonalias": all(
            float(alias_map[control]["pooled_median_absolute_cosine"]) < 0.95
            for control in CONTROL_ORDER
        ),
        "completeness": True,
        "close_vs_pa": decision["first_decisive_clause"].startswith("close_vs_pa"),
        "close_primary_control": decision["first_decisive_clause"].startswith("close_")
        and "primary_alignment" in decision["first_decisive_clause"],
        "close_control_slope": decision["first_decisive_clause"].startswith("close_")
        and "slope" in decision["first_decisive_clause"],
        "close_correction_alias": decision["first_decisive_clause"].startswith("close_alias_"),
    }
    panel = {
        "operator_order": list(OPERATOR_ORDER),
        "rows": rows,
        "seed_context_aggregates": seed_context,
        "pooled_aggregates": pooled,
        "correction_alias_aggregates": aliases,
        "predicates": predicates,
    }
    return panel, decision


def run_full_panel_science(
    selection: dict[str, object],
    bounds: tuple[Any, ...],
    preliminary: dict[str, object],
    *,
    torch_module: Any,
    rsta_module: types.ModuleType,
    expected_dimension: int = 512,
) -> dict[str, object]:
    if preliminary["decision"]["status"] != "SURVIVES":
        raise ValueError("full panel is authorized only after preliminary survival")
    phase = selection["panel"]
    rows: list[dict[str, object]] = []
    correction_cosines = {control: [] for control in CONTROL_ORDER}
    for bound in bounds:
        for group in phase["groups"]:
            context_a = group["contexts"][0]
            batch_ids_a = tuple(context_a["batch_ids"])
            for receiver_label, receiver_id in zip(
                group["receiver_labels"], group["receiver_ids"], strict=True
            ):
                ordinary_context = _open_panel_context(
                    torch_module=torch_module,
                    rsta_module=rsta_module,
                    bound=bound,
                    batch_ids=batch_ids_a,
                    receiver_ids=tuple(group["receiver_ids"]),
                    expected_dimension=expected_dimension,
                )
                try:
                    names, p, named_parameters = _ordinary_pa_direction(
                        ordinary_context, torch_module
                    )
                finally:
                    _close_panel_context(ordinary_context, torch_module)
                p_norm = float(fp64_named_norm(torch_module, p).item())
                parameter_names_sha256 = hashlib.sha256(
                    b"".join(name.encode() + b"\n" for name in names)
                ).hexdigest()
                action_by_context = {"A": {}, "B": {}}
                for context_record in group["contexts"]:
                    context_name = context_record["context"]
                    action_by_context[context_name]["pa"] = _evaluate_panel_action(
                        direction=p,
                        selection=selection,
                        phase=phase,
                        group=group,
                        context_record=context_record,
                        receiver_label=receiver_label,
                        receiver_id=receiver_id,
                        bound=bound,
                        torch_module=torch_module,
                        rsta_module=rsta_module,
                        expected_dimension=expected_dimension,
                    )
                corrections: dict[str, dict[str, object]] = {}
                rdgc_correction: tuple[Any, ...] | None = None
                for operator in CORRECTION_ORDER:
                    correction = _correction_for_receiver(
                        operator=operator,
                        p=p,
                        named_parameters=named_parameters,
                        selection=selection,
                        phase=phase,
                        group=group,
                        receiver_id=receiver_id,
                        bound=bound,
                        torch_module=torch_module,
                        rsta_module=rsta_module,
                        expected_dimension=expected_dimension,
                    )
                    updates = normalize_virtual_updates(torch_module, p, {operator: correction})
                    direction = updates[operator]
                    correction_norm = float(fp64_named_norm(torch_module, correction).item())
                    matched_norm = float(fp64_named_norm(torch_module, direction).item())
                    if abs(matched_norm - p_norm) / p_norm > 5.0e-7:
                        raise ValueError("matched virtual direction norm drift")
                    if operator == "rdgc":
                        rdgc_correction = tuple(value.clone() for value in correction)
                        cosine = 1.0
                    else:
                        if rdgc_correction is None:
                            raise ValueError("RDGC correction must be constructed first")
                        denominator = float(
                            fp64_named_norm(torch_module, rdgc_correction).item()
                            * fp64_named_norm(torch_module, correction).item()
                        )
                        cosine = abs(
                            float(fp64_named_dot(torch_module, rdgc_correction, correction).item())
                            / denominator
                        )
                        correction_cosines[operator].append((int(bound.seed), cosine))
                    corrections[operator] = {
                        "direction_sha256": _named_direction_sha256(names, direction),
                        "correction_norm": correction_norm,
                        "matched_update_norm": matched_norm,
                        "rdgc_correction_cosine": cosine,
                    }
                    for context_record in group["contexts"]:
                        context_name = context_record["context"]
                        action_by_context[context_name][operator] = _evaluate_panel_action(
                            direction=direction,
                            selection=selection,
                            phase=phase,
                            group=group,
                            context_record=context_record,
                            receiver_label=receiver_label,
                            receiver_id=receiver_id,
                            bound=bound,
                            torch_module=torch_module,
                            rsta_module=rsta_module,
                            expected_dimension=expected_dimension,
                        )
                    del correction, updates, direction
                for context_record in group["contexts"]:
                    rows.append(
                        {
                            "seed": int(bound.seed),
                            "group": group["group_index"],
                            "receiver_label": receiver_label,
                            "receiver_id": receiver_id,
                            "context": context_record["context"],
                            "batch_sha256": context_record["batch_sha256"],
                            "parameter_names_sha256": parameter_names_sha256,
                            "p_norm": p_norm,
                            "corrections": {key: dict(value) for key, value in corrections.items()},
                            "operators": action_by_context[context_record["context"]],
                        }
                    )
                del rdgc_correction, corrections, action_by_context, p, named_parameters
    bootstrap_rows = [
        {
            "seed": row["seed"],
            "context": row["context"],
            "receiver_label": row["receiver_label"],
            "operators": row["operators"],
        }
        for row in rows
    ]
    bootstrap = paired_bootstrap(bootstrap_rows)
    panel, decision = _aggregate_panel(rows, correction_cosines, bootstrap)
    return {"panel": panel, "bootstrap": bootstrap, "decision": decision}


def _source_files_digest(files: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in files:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _integrity_prefix_from_rsta(
    *,
    rsta_module: types.ModuleType,
    old_manifest: dict[str, object],
    old_manifest_path: Path,
    binding_receipt_path: Path,
    output_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    captured: list[dict[str, object]] = []
    original_writer = rsta_module.write_json_atomic

    def capture(_path: Path, payload: dict[str, object], *, sort_keys: bool) -> None:
        if sort_keys is not False or captured:
            raise ValueError("integrity capture writer call mismatch")
        captured.append(payload)

    rsta_module.write_json_atomic = capture
    try:
        payload = rsta_module.run_all_seed_adjoint_integrity(
            old_manifest,
            manifest_path=old_manifest_path,
            receipt_path=binding_receipt_path,
            output_path=output_path,
            execution_source_validator=lambda _path: {"rdgc_source_authenticated": True},
        )
    finally:
        rsta_module.write_json_atomic = original_writer
    if captured != [payload] or payload["integrity"]["all_passed"] is not True:
        raise ValueError("reused all-seed integrity prefix failed")
    records: list[dict[str, object]] = []
    for seed in range(4):
        raw = payload["integrity"]["seeds"][str(seed)]
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
        records.append(
            {
                "seed": seed,
                "dense_jacobian_passed": True,
                "bn_passed": True,
                "repeatability_passed": True,
                "normwise_adjoint_passed": True,
                "sign_control_passed": True,
                "rotation_passed": True,
                "atomic_writer_passed": True,
                "no_candidate_reachability_passed": True,
                "action_hashes_sha256": hashlib.sha256(encoded).hexdigest(),
                "passed": True,
            }
        )
    return records, payload["binding"]


def _result_decision(
    status: str, clause: str, *, predicates: list[dict[str, object]] | None = None
) -> dict[str, object]:
    return {
        "close_precedence": True,
        "predicates": predicates or [{"name": clause, "value": True}],
        "status": status,
        "first_decisive_clause": clause,
        "authorized_action": {
            "PASS": "new_training_preregistration_only",
            "CLOSE": "stop_close",
            "UNRESOLVED": "stop_unresolved",
            "INVALID": "stop_invalid",
        }[status],
    }


def _output_binding(manifest: dict[str, object]) -> dict[str, object]:
    upstream = manifest["upstream_rsta"]
    receipt = manifest["validation_receipt"]
    return {
        "rsta_candidate": upstream["candidate"],
        "rsta_gate2_audit": upstream["gate2_audit"],
        "rsta_producer_source_commit": upstream["producer_source_commit"],
        "rsta_producer_handoff_commit": upstream["producer_handoff_commit"],
        "rsta_artifact": upstream["producer_artifact"],
        "rsta_producer_pid": upstream["producer_pid"],
        "rsta_producer_exit_code": upstream["producer_exit_code"],
        "verifier_source_commit": upstream["verifier_source_commit"],
        "verifier_handoff_commit": upstream["verifier_handoff_commit"],
        "verifier_manifest_sha256": upstream["verifier_manifest"]["sha256"],
        "validation_receipt": {
            "path": receipt["path"],
            "sha256": receipt["sha256"],
            "status": receipt["status"],
        },
        "rsta_scientific_status": upstream["scientific_status"],
        "rsta_scientific_decision": upstream["scientific_decision"],
        "rsta_first_decisive_clause": upstream["first_decisive_clause"],
        "seeds": manifest["historical"]["seeds"],
    }


def _output_source(
    repository: Path,
    manifest_path: Path,
    manifest_sha256: str,
    authority: dict[str, object],
) -> dict[str, object]:
    return {
        "source_commit": authority["source_commit"],
        "handoff_commit": authority["handoff_commit"],
        "handoff_parent": authority["source_commit"],
        "manifest_path": str(manifest_path.relative_to(repository)),
        "manifest_sha256": manifest_sha256,
        "files": authority["files"],
        "detached_head": _run_git(repository, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD",
        "clean_tracked_worktree": True,
        "ancestry_passed": True,
    }


def _output_execution(
    *,
    repository: Path,
    output_path: Path,
    command: list[str],
    started_utc: str,
    completed_utc: str,
    pre_import: dict[str, object],
    exit_code: int,
) -> dict[str, object]:
    return {
        "attempt": 1,
        "command": command,
        "cwd": str(repository),
        "pid": os.getpid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": pre_import["python_version"],
        "cuda_visible_devices": pre_import["cuda_visible_devices"],
        "cublas_workspace_config": pre_import["cublas_workspace_config"],
        "output_path": str(output_path.relative_to(repository)),
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "exit_code": exit_code,
    }


def _execution_environment(
    *,
    repository: Path,
    manifest_path: Path,
    manifest_sha256: str,
    authority: dict[str, object],
) -> dict[str, object]:
    source_files = authority["files"]
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "numpy_version": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "source_commit": authority["source_commit"],
        "source_files_sha256": _source_files_digest(source_files),
        "manifest_path": str(manifest_path.relative_to(repository)),
        "manifest_sha256": manifest_sha256,
    }


def run_rdgc_scientific_once(
    *,
    repository: Path,
    manifest_path: Path,
    output_path: Path,
    manifest: dict[str, object],
    authority: dict[str, object],
    started_utc: str,
    command: list[str],
    expected_dimension: int = 512,
) -> dict[str, object]:
    """Run the complete integrity→preliminary→conditional-panel process once."""
    helper_record = next(
        record
        for record in authority["files"]
        if record["path"] == "scripts/diagnose_pass200_rsta_stage_a.py"
    )
    rsta_module = load_authenticated_rsta_module(repository, helper_record)
    historical = manifest["historical"]
    old_manifest_path = _within(repository, repository / historical["manifest_path"])
    if _sha256_file(old_manifest_path) != historical["manifest_sha256"]:
        raise ValueError("historical RSTA manifest digest mismatch")
    old_manifest = _strict_json(old_manifest_path)
    binding_receipt_path = _within(repository, repository / old_manifest["binding_receipt"]["path"])
    receipt = rsta_module.validate_historical_binding_receipt(
        old_manifest_path, binding_receipt_path
    )
    bounds = tuple(
        rsta_module.load_training_only_seed(old_manifest["seeds"][str(seed)], receipt.seeds[seed])
        for seed in range(4)
    )
    rsta_module.validate_cross_seed_training_binding(bounds)
    reference = bounds[0]
    old_primary = rsta_module.select_primary_panel(
        reference.train_example_ids, reference.train_labels
    )
    selection = build_rdgc_selection(
        reference.train_example_ids,
        reference.train_labels,
        old_selection=set(old_primary["labels"]),
    )
    del old_primary
    integrity_seeds, _integrity_binding = _integrity_prefix_from_rsta(
        rsta_module=rsta_module,
        old_manifest=old_manifest,
        old_manifest_path=old_manifest_path,
        binding_receipt_path=binding_receipt_path,
        output_path=output_path,
    )
    import torch

    preliminary = run_preliminary_science(
        selection,
        bounds,
        torch_module=torch,
        rsta_module=rsta_module,
        expected_dimension=expected_dimension,
    )
    preliminary_status = preliminary["decision"]["status"]
    if preliminary_status == "SURVIVES":
        full = run_full_panel_science(
            selection,
            bounds,
            preliminary,
            torch_module=torch,
            rsta_module=rsta_module,
            expected_dimension=expected_dimension,
        )
        panel = full["panel"]
        bootstrap = full["bootstrap"]
        panel_decision = full["decision"]
        status = panel_decision["status"]
        clause = panel_decision["first_decisive_clause"]
        phase_reached = "full_panel"
    else:
        panel = None
        bootstrap = None
        status = preliminary_status
        clause = preliminary["decision"]["first_decisive_clause"]
        phase_reached = "preliminary"
    manifest_sha256 = _sha256_file(manifest_path)
    pre_import = _execution_environment(
        repository=repository,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        authority=authority,
    )
    environment = attach_observed_torch_runtime(build_pre_import_environment(pre_import), torch)
    completed_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": 1,
        "diagnostic": "pass205_rdgc_stage_b",
        "mode": "scientific_no_training_virtual_update",
        "status": status,
        "phase_reached": phase_reached,
        "candidate_values_computed": True,
        "training_performed": False,
        "benchmark_authorized": False,
        "scope_limitation": SCOPE_LIMITATION,
        "authority": {
            "candidate": manifest["candidate"],
            "implementation_plan": manifest["implementation_plan"],
            "literature_audit": manifest["literature_audit"],
        },
        "source": _output_source(repository, manifest_path, manifest_sha256, authority),
        "execution": _output_execution(
            repository=repository,
            output_path=output_path,
            command=command,
            started_utc=started_utc,
            completed_utc=completed_utc,
            pre_import=pre_import,
            exit_code=0,
        ),
        "environment": environment,
        "binding": _output_binding(manifest),
        "integrity": {
            "seeds": integrity_seeds,
            "all_four_passed": True,
            "candidate_calls_before_all_four": 0,
            "candidate_state_before_all_four": False,
        },
        "selection": selection,
        "preliminary": preliminary,
        "panel": panel,
        "bootstrap": bootstrap,
        "decision": _result_decision(status, clause),
    }
    validate_scientific_payload(payload)
    write_json_atomic(output_path, payload)
    return payload


def publish_reduced_invalid(
    *,
    repository: Path,
    manifest_path: Path,
    output_path: Path,
    manifest: dict[str, object],
    authority: dict[str, object],
    started_utc: str,
    command: list[str],
) -> dict[str, object]:
    """Publish an outcome-free INVALID only after publication authority is complete."""
    manifest_sha256 = _sha256_file(manifest_path)
    pre_import = _execution_environment(
        repository=repository,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        authority=authority,
    )
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        phase_reached = "pre_import"
        environment = build_pre_import_environment(pre_import)
        integrity = None
    else:
        phase_reached = "integrity"
        environment = attach_observed_torch_runtime(
            build_pre_import_environment(pre_import), torch_module
        )
        integrity = {}
    completed_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload = {
        "schema_version": 1,
        "diagnostic": "pass205_rdgc_stage_b",
        "mode": "scientific_no_training_virtual_update",
        "status": "INVALID",
        "phase_reached": phase_reached,
        "candidate_values_computed": False,
        "training_performed": False,
        "benchmark_authorized": False,
        "scope_limitation": SCOPE_LIMITATION,
        "authority": {
            "candidate": manifest["candidate"],
            "implementation_plan": manifest["implementation_plan"],
            "literature_audit": manifest["literature_audit"],
        },
        "source": _output_source(repository, manifest_path, manifest_sha256, authority),
        "execution": _output_execution(
            repository=repository,
            output_path=output_path,
            command=command,
            started_utc=started_utc,
            completed_utc=completed_utc,
            pre_import=pre_import,
            exit_code=2,
        ),
        "environment": environment,
        "binding": _output_binding(manifest),
        "integrity": integrity,
        "selection": None,
        "preliminary": None,
        "panel": None,
        "bootstrap": None,
        "decision": _result_decision("INVALID", "runtime_or_integrity_invalid"),
    }
    validate_scientific_payload(payload)
    write_json_atomic(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scientific-once", action="store_true", required=True)
    arguments = parser.parse_args(argv)
    registered_manifest = Path("docs/pass205_rdgc_stage_b_manifest.json")
    if arguments.manifest != registered_manifest:
        parser.error("--manifest must be the registered relative path")
    repository = Path.cwd().resolve(strict=True)
    if _run_git(repository, "rev-parse", "--show-toplevel") != str(repository):
        parser.error("cwd must be the repository root")
    manifest_path = repository / registered_manifest
    if manifest_path.is_symlink() or not manifest_path.is_file():
        parser.error("registered manifest must be a regular file")
    head = _run_git(repository, "rev-parse", "HEAD")
    expected_python = (repository / ".venv/bin/python").resolve(strict=True)
    if Path(sys.executable).resolve() != expected_python:
        parser.error("scientific process must use the registered .venv interpreter")
    if (sys.version_info.major, sys.version_info.minor, sys.version_info.micro) != (3, 12, 3):
        parser.error("scientific process requires Python 3.12.3")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("CUDA_VISIBLE_DEVICES must equal 0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        parser.error("CUBLAS_WORKSPACE_CONFIG must equal :4096:8")
    expected_output = Path(f"reports/generated/pass205_rdgc_stage_b/{head}-rdgc-stage-b.json")
    if arguments.output != expected_output:
        parser.error("--output must be the exact handoff-derived result path")
    output = repository / arguments.output
    if output.parent.is_symlink() or not output.parent.is_dir():
        parser.error("registered output parent must already exist and be non-symlink")
    if output.exists() or output.is_symlink():
        parser.error("output must be absent")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        parser.error("registered temporary path must be absent")
    manifest = _strict_json(manifest_path)
    validate_future_manifest(manifest)
    receipt_path = repository / manifest["validation_receipt"]["path"]
    authority = authenticate_authority(repository, manifest_path, receipt_path)
    started_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    command = [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *effective_argv]
    try:
        payload = run_rdgc_scientific_once(
            repository=repository,
            manifest_path=manifest_path,
            output_path=output,
            manifest=manifest,
            authority=authority,
            started_utc=started_utc,
            command=command,
        )
    except Exception:
        payload = publish_reduced_invalid(
            repository=repository,
            manifest_path=manifest_path,
            output_path=output,
            manifest=manifest,
            authority=authority,
            started_utc=started_utc,
            command=command,
        )
    return 2 if payload["status"] == "INVALID" else 0


if __name__ == "__main__":
    raise SystemExit(main())
