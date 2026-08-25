#!/usr/bin/env python3
"""Strict result contract for the preregistered UniCOM ProxyMuon F0 screen."""

from __future__ import annotations

import ctypes
import errno
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import NoReturn

from sfora.unicom_proxy_muon import (
    LR_GRID,
    OPTIMIZERS,
    PHASE1_SEEDS,
    PHASE2_SEEDS,
    RETAINED_STEPS,
    VALIDATION_STEPS,
    accuracy_noninferior,
    compute_reach_step,
    decide_proxy_muon_f0,
    select_adamw_reference,
    select_learning_rate,
)

SCIENTIFIC_SCHEMA_VERSION = "unicom-proxy-muon-f0-v1"
FAILURE_SCHEMA_VERSION = "unicom-proxy-muon-f0-failure-v1"
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
    "fitting_features",
    "fitting_labels",
    "validation_features",
    "validation_labels",
    "imprinted_head",
)
RUNTIME_KEYS = (
    "python_version",
    "torch_version",
    "numpy_version",
    "sklearn_version",
    "cuda_version",
    "gpu_name",
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


def _validate_runtime(value: object) -> None:
    runtime = _object(value, RUNTIME_KEYS, "runtime")
    for key in RUNTIME_KEYS[:6]:
        _string(runtime[key], f"runtime.{key}")
    for key in RUNTIME_KEYS[6:9]:
        if type(runtime[key]) is not bool:
            raise TypeError(f"runtime.{key} must be a bool")
    _string(runtime["muon_signature"], "runtime.muon_signature")
    if runtime["observed_update_dtype"] != "torch.bfloat16":
        raise ValueError("runtime observed update dtype differs")


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
    reserved = _integer(process["peak_reserved_bytes"], "process.peak_reserved_bytes")
    completed = _integer(process["completed_cells"], "process.completed_cells")
    if completed != expected_cells or elapsed > 2700.0 or max(allocated, reserved) > 8 * 1024**3:
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
    if (
        initializer["feature_sha256"] != authority["inputs"]["fitting_features"]
        or initializer["label_sha256"] != authority["inputs"]["fitting_labels"]
        or initializer["validation_feature_sha256"]
        != authority["inputs"]["validation_features"]
        or initializer["initial_head_sha256"]
        != authority["inputs"]["imprinted_head"]
    ):
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
