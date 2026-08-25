from __future__ import annotations

import copy
import gc
import importlib.util
import json
import math
import subprocess
import sys
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_unicom_proxy_muon_f0.py"
SPEC = importlib.util.spec_from_file_location("screen_unicom_proxy_muon_f0", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_proxy_muon_result_contract_module_exists() -> None:
    assert callable(module.strict_json_object)
    assert callable(module.validate_scientific_result)
    assert callable(module.validate_failure_receipt)
    assert callable(module.canonical_json_bytes)
    assert callable(module.publish_result_exclusive)


def test_proxy_muon_scientific_orchestration_api_exists() -> None:
    """Catch a screen that validates receipts but cannot produce one."""

    assert callable(module.run_proxy_muon_f0)
    assert not hasattr(module, "_run_authenticated_proxy_muon")
    assert callable(module.assemble_scientific_result)
    assert callable(module.assemble_failure_receipt)
    assert callable(module.main)


def test_protocol_cell_order_keeps_seed_innermost_and_deduplicates_anchor() -> None:
    """Catch loop reordering or an accidental duplicate AdamW anchor fit."""

    phase1 = module.phase1_cell_schedule()
    assert phase1[:4] == (
        ("adamw", 0.000025, 0),
        ("adamw", 0.000025, 1),
        ("adamw", 0.000025, 2),
        ("adamw", 0.00005, 0),
    )
    assert phase1[-3:] == (
        ("proxy_muon", 0.0004, 0),
        ("proxy_muon", 0.0004, 1),
        ("proxy_muon", 0.0004, 2),
    )
    assert len(phase1) == 30
    assert module.phase2_cell_schedule(0.0001, 0.0002) == tuple(
        (seed, variant, 0.0002 if variant.startswith("proxy_muon") else 0.0001)
        for seed in (3, 4, 5)
        for variant in ("adamw_selected", "proxy_muon", "proxy_muon_fp32")
    )
    assert module.phase2_cell_schedule(0.0002, 0.0001)[:4] == (
        (3, "adamw_selected", 0.0002),
        (3, "adamw_anchor", 0.0001),
        (3, "proxy_muon", 0.0001),
        (3, "proxy_muon_fp32", 0.0001),
    )


def test_phase1_runner_is_fail_fast_and_retains_exact_completed_prefix() -> None:
    """Catch continuation after a failed cell or hashing a non-completed cell."""

    golden = _phase1_rows()
    completed: list[str] = []
    observed: list[tuple[str, float, int]] = []

    def cell_runner(
        _features: object,
        _labels: object,
        _initial: object,
        *,
        optimizer: str,
        learning_rate: float,
        fit_seed: int,
    ) -> dict[str, object]:
        observed.append((optimizer, learning_rate, fit_seed))
        if len(observed) == 8:
            raise RuntimeError("injected cell failure")
        return golden[len(observed) - 1]

    with pytest.raises(RuntimeError, match="injected cell failure"):
        module.run_phase1_cells(
            object(), object(), object(), completed, cell_runner=cell_runner
        )

    assert observed == list(module.phase1_cell_schedule()[:8])
    assert completed == [
        __import__("hashlib").sha256(module.canonical_json_bytes(row)).hexdigest()
        for row in golden[:7]
    ]


def test_phase1_runner_checks_budget_after_each_completed_cell() -> None:
    golden = _phase1_rows()
    completed: list[str] = []
    observed_cells = 0

    def boundary(_step: int) -> None:
        nonlocal observed_cells
        observed_cells += 1
        raise RuntimeError("registered execution budget exceeded")

    with pytest.raises(RuntimeError, match="registered execution budget exceeded"):
        module.run_phase1_cells(
            object(),
            object(),
            object(),
            completed,
            cell_runner=lambda *_args, **_kwargs: golden[0],
            progress_callback=boundary,
        )

    assert observed_cells == 1
    assert completed == [
        __import__("hashlib").sha256(
            module.canonical_json_bytes(golden[0])
        ).hexdigest()
    ]


def test_phase1_cell_reduces_only_registered_snapshot_bytes_and_panels() -> None:
    """Catch a cell row assembled from the wrong snapshot or panel order."""

    initial = torch.arange(16, dtype=torch.float32).reshape(4, 4).contiguous()
    final = (initial + 1.0).contiguous()

    def fit_trajectory(*_args, **_kwargs):
        return {0: initial.clone(), 64: final.clone()}

    def panel(_features, _labels, head, *, fit_seed: int):
        base = 1.0 if torch.equal(head, initial) else 0.5
        return tuple(base + fit_seed * 0.01 + index * 1e-6 for index in range(16))

    row = module._run_phase1_cell(
        object(),
        object(),
        initial,
        optimizer="adamw",
        learning_rate=0.0001,
        fit_seed=2,
        fit_trajectory=fit_trajectory,
        panel_runner=panel,
    )

    assert tuple(row) == module.PHASE1_ROW_KEYS
    assert row["steps"] == 64
    assert row["initial_head_sha256"] == module.tensor_sha256(initial)
    assert row["final_head_sha256"] == module.tensor_sha256(final)
    assert row["diagnostic_step0"]["mean"] == math.fsum(
        row["diagnostic_step0"]["components"]
    ) / 16
    assert row["diagnostic_step64"]["components"][0] == 0.52


def test_phase2_runner_is_seed_major_fail_fast_and_hashes_only_completed_rows() -> None:
    """Catch Phase-2 variant reordering, continuation, or premature prefix hashing."""

    golden = _scientific_fixture()["phase2"]
    completed: list[str] = []
    observed: list[tuple[int, str, float]] = []

    def cell_runner(
        _features: object,
        _labels: object,
        _validation_features: object,
        _validation_labels: object,
        _represented: object,
        _initial: object,
        *,
        fit_seed: int,
        variant: str,
        learning_rate: float,
    ) -> dict[str, object]:
        observed.append((fit_seed, variant, learning_rate))
        if len(observed) == 6:
            raise RuntimeError("injected Phase-2 failure")
        return golden[len(observed) - 1]

    with pytest.raises(RuntimeError, match="injected Phase-2 failure"):
        module.run_phase2_cells(
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            0.0001,
            0.0002,
            completed,
            cell_runner=cell_runner,
        )

    assert observed == list(module.phase2_cell_schedule(0.0001, 0.0002)[:6])
    assert completed == [
        __import__("hashlib").sha256(module.canonical_json_bytes(row)).hexdigest()
        for row in golden[:5]
    ]


@pytest.mark.parametrize(
    "variant,expected_dtype",
    (("adamw_selected", None), ("proxy_muon", "torch.bfloat16")),
)
def test_phase2_cell_reduces_registered_snapshots_traces_and_validation_only(
    variant: str, expected_dtype: str | None
) -> None:
    """Catch wrong retained-step traces or validation on unregistered steps."""

    initial = torch.arange(16, dtype=torch.float32).reshape(4, 4).contiguous()
    snapshots = {
        step: (initial if step == 0 else initial + float(index)).contiguous()
        for index, step in enumerate(module.RETAINED_STEPS)
    }
    traces = {
        step: (
            None
            if step == 0 or expected_dtype is None
            else SimpleNamespace(
                update_dtype=expected_dtype,
                polar_factor_residual=0.01 + index * 0.001,
            )
        )
        for index, step in enumerate(module.RETAINED_STEPS)
    }

    def fit_trajectory(*_args, **_kwargs):
        return snapshots, traces

    def panel(_features, _labels, head, *, fit_seed: int):
        value = float(head[0, 0]) + fit_seed
        return tuple(value + index * 1e-6 for index in range(16))

    evaluated_steps: list[int] = []

    def accuracy(_features, _labels, head, *, represented):
        assert represented == (True, False)
        step = next(key for key, value in snapshots.items() if torch.equal(value, head))
        evaluated_steps.append(step)
        return 0.8 + step / 10_000

    row = module._run_phase2_cell(
        object(),
        object(),
        object(),
        object(),
        (True, False),
        initial,
        fit_seed=3,
        variant=variant,
        learning_rate=0.0001,
        fit_trajectory=fit_trajectory,
        panel_runner=panel,
        accuracy_runner=accuracy,
    )

    assert evaluated_steps == [307, 435, 512]
    assert [item["step"] for item in row["retained"]] == list(module.RETAINED_STEPS)
    assert [
        item["validation_accuracy"]
        for item in row["retained"]
        if item["step"] not in module.VALIDATION_STEPS
    ] == [None] * 6
    assert row["retained"][0]["update_dtype"] is None
    assert row["retained"][1]["update_dtype"] == expected_dtype


def test_cell_reducers_move_retained_cpu_heads_to_live_feature_device() -> None:
    initial = torch.arange(16, dtype=torch.float32).reshape(4, 4).contiguous()
    features = torch.empty(1, 1, device="meta")
    observed_devices: list[torch.device] = []

    def panel(_features, _labels, head, *, fit_seed: int):
        del fit_seed
        observed_devices.append(head.device)
        return tuple(float(index) for index in range(16))

    module._run_phase1_cell(
        features,
        object(),
        initial,
        optimizer="adamw",
        learning_rate=0.0001,
        fit_seed=0,
        fit_trajectory=lambda *_args, **_kwargs: {0: initial, 64: initial},
        panel_runner=panel,
    )

    snapshots = {step: initial for step in module.RETAINED_STEPS}
    traces = {step: None for step in module.RETAINED_STEPS}

    def accuracy(_features, _labels, head, *, represented):
        del represented
        observed_devices.append(head.device)
        return 0.5

    module._run_phase2_cell(
        features,
        object(),
        features,
        object(),
        object(),
        initial,
        fit_seed=3,
        variant="adamw_selected",
        learning_rate=0.0001,
        fit_trajectory=lambda *_args, **_kwargs: (snapshots, traces),
        panel_runner=panel,
        accuracy_runner=accuracy,
    )

    assert observed_devices
    assert all(device.type == "meta" for device in observed_devices)


def test_reconstructed_parent_hash_gate_binds_features_labels_and_all_four_heads() -> None:
    """Catch a screen that authenticates metadata but not reconstructed tensors."""

    tensors = {
        name: (torch.arange(8, dtype=torch.float32) + index).contiguous()
        for index, name in enumerate(
            (
                "fitting_features",
                "fitting_labels",
                "validation_features",
                "validation_labels",
                "imprinted_head",
                "adamw_fitted_head_seed_0",
                "adamw_fitted_head_seed_1",
                "adamw_fitted_head_seed_2",
            )
        )
    }
    expected = {name: module.tensor_sha256(value) for name, value in tensors.items()}

    assert module.verify_parent_tensor_hashes(tensors, expected) == expected
    mutated = dict(tensors)
    mutated["adamw_fitted_head_seed_2"] = (
        mutated["adamw_fitted_head_seed_2"] + 1.0
    ).contiguous()
    with pytest.raises(ValueError, match="adamw_fitted_head_seed_2"):
        module.verify_parent_tensor_hashes(mutated, expected)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _panel(value: float) -> dict[str, object]:
    components = [float(value + offset * 1e-6) for offset in range(16)]
    return {"components": components, "mean": math.fsum(components) / 16}


def _protocol() -> dict[str, object]:
    return {
        "learning_rate_grid": [0.000025, 0.00005, 0.0001, 0.0002, 0.0004],
        "phase1_seeds": [0, 1, 2],
        "phase2_seeds": [3, 4, 5],
        "phase1_steps": 64,
        "phase2_steps": 512,
        "retained_steps": [0, 64, 128, 192, 256, 307, 384, 435, 512],
        "validation_steps": [307, 435, 512],
        "batch_size": 128,
        "diagnostic_batches": 4,
        "diagnostic_masks": 4,
        "elapsed_limit_seconds": 2700.0,
        "peak_limit_bytes": 8 * 1024**3,
    }


def _authority() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "handoff_commit": "b" * 40,
        "sources": {
            name: _digest(index + 1)
            for index, name in enumerate(module.SOURCE_HASH_KEYS)
        },
        "inputs": {
            "run_config": _digest(19),
            "final_report": _digest(20),
            "spherical_parent_result": _digest(21),
            "cap_closure_receipt": _digest(22),
            "checkpoint": _digest(23),
            "partition": _digest(24),
            "imprinted_head": _digest(40),
            "adamw_fitted_head_seed_0": _digest(41),
            "adamw_fitted_head_seed_1": _digest(42),
            "adamw_fitted_head_seed_2": _digest(43),
        },
    }


def _runtime() -> dict[str, object]:
    return {
        "python_version": "3.12.3",
        "torch_version": "2.12.1",
        "numpy_version": "2.5.0",
        "sklearn_version": "1.7.2",
        "cuda_version": "12.8",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_memory_allocated_bytes": 0,
        "cuda_memory_reserved_bytes": 0,
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "muon_signature": "pinned",
        "observed_update_dtype": "torch.bfloat16",
    }


def _phase1_rows(
    *, adamw_selected: float = 0.0001, proxy_selected: float = 0.0002
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    choices = {"adamw": adamw_selected, "proxy_muon": proxy_selected}
    grid = _protocol()["learning_rate_grid"]
    assert type(grid) is list
    for optimizer_index, optimizer in enumerate(("adamw", "proxy_muon")):
        selected_index = grid.index(choices[optimizer])
        for lr_index, learning_rate in enumerate(grid):
            for seed in (0, 1, 2):
                final_loss = (
                    0.5
                    + optimizer_index * 0.02
                    + abs(lr_index - selected_index) * 0.1
                    + seed * 0.001
                )
                rows.append(
                    {
                        "optimizer": optimizer,
                        "learning_rate": float(learning_rate),
                        "fit_seed": seed,
                        "steps": 64,
                        "initial_head_sha256": _digest(40),
                        "final_head_sha256": _digest(200 + len(rows)),
                        "diagnostic_step0": _panel(1.5 + seed * 0.01),
                        "diagnostic_step64": _panel(final_loss),
                    }
                )
    return rows


def _selection(
    rows: list[dict[str, object]], optimizer: str
) -> dict[str, object]:
    means = {}
    for learning_rate in _protocol()["learning_rate_grid"]:
        values = [
            row["diagnostic_step64"]["mean"]
            for row in rows
            if row["optimizer"] == optimizer and row["learning_rate"] == learning_rate
        ]
        means[learning_rate] = math.fsum(values) / 3
    mean, learning_rate = min((mean, lr) for lr, mean in means.items())
    return {
        "learning_rate": learning_rate,
        "mean_final_loss": mean,
        "interior": learning_rate not in (0.000025, 0.0004),
        "tie_lrs": [lr for lr in _protocol()["learning_rate_grid"] if means[lr] == mean],
    }


def _retained(variant: str, seed: int, *, reference_loss: float = 0.55) -> list[dict[str, object]]:
    rows = []
    for index, step in enumerate(_protocol()["retained_steps"]):
        if step == 0:
            loss = 1.0
            accuracy = None
        elif variant.startswith("proxy_muon"):
            loss = reference_loss + 0.1 - 0.0006 * step
            accuracy = 0.91 + seed * 0.001 if step in (307, 435, 512) else None
        else:
            loss = reference_loss + 0.2 - 0.000390625 * step
            accuracy = 0.90 + seed * 0.001 if step in (307, 435, 512) else None
        update_dtype = None
        residual = None
        if variant.startswith("proxy_muon") and step != 0:
            update_dtype = "torch.float32" if variant == "proxy_muon_fp32" else "torch.bfloat16"
            residual = 0.01 + index * 0.001
        rows.append(
            {
                "step": step,
                "head_sha256": (
                    _digest(40) if step == 0 else _digest(500 + seed * 100 + index)
                ),
                "diagnostic": _panel(loss),
                "validation_accuracy": accuracy,
                "update_dtype": update_dtype,
                "polar_factor_residual": residual,
            }
        )
    return rows


def _scientific_fixture(
    *, adamw_selected: float = 0.0001, proxy_selected: float = 0.0002
) -> dict[str, object]:
    phase1 = _phase1_rows(
        adamw_selected=adamw_selected, proxy_selected=proxy_selected
    )
    selections = {
        optimizer: _selection(phase1, optimizer)
        for optimizer in ("adamw", "proxy_muon")
    }
    variants = (
        ("adamw_selected", "proxy_muon", "proxy_muon_fp32")
        if adamw_selected == 0.0001
        else (
            "adamw_selected",
            "adamw_anchor",
            "proxy_muon",
            "proxy_muon_fp32",
        )
    )
    phase2 = []
    for seed in (3, 4, 5):
        for variant in variants:
            learning_rate = (
                proxy_selected
                if variant.startswith("proxy_muon")
                else 0.0001
                if variant == "adamw_anchor"
                else adamw_selected
            )
            phase2.append(
                {
                    "fit_seed": seed,
                    "variant": variant,
                    "learning_rate": learning_rate,
                    "retained": _retained(variant, seed),
                }
            )
    comparisons = []
    for seed in (3, 4, 5):
        comparisons.append(
            {
                "fit_seed": seed,
                "adamw_reference_variant": (
                    "adamw_selected" if adamw_selected == 0.0001 else "adamw_anchor"
                ),
                "adamw_reference_learning_rate": 0.0001,
                "adamw_reference_step512_loss": _panel(0.55)["mean"],
                "adamw_reference_step512_accuracy": 0.90 + seed * 0.001,
                "proxy_muon_reach_step": 307,
                "proxy_muon_accuracy_at_reach": 0.91 + seed * 0.001,
                "proxy_muon_accuracy_delta": 0.010000000000000009,
                "proxy_muon_step512_accuracy_delta": 0.010000000000000009,
                "proxy_muon_fp32_reach_step": 307,
                "proxy_muon_fp32_accuracy_at_reach": 0.91 + seed * 0.001,
                "proxy_muon_fp32_accuracy_delta": 0.010000000000000009,
                "proxy_muon_fp32_step512_accuracy_delta": 0.010000000000000009,
            }
        )
    predicates = {
        "adamw_lr_interior": True,
        "proxy_muon_lr_interior": True,
        "all_reach_by_307": True,
        "all_reach_noninferior": True,
        "all_step512_noninferior": True,
        "any_bf16_accuracy_loss": False,
        "fp32_sensitivity_supported": False,
    }
    return {
        "schema_version": module.SCIENTIFIC_SCHEMA_VERSION,
        "status": "PROCEED_TRAINING",
        "authority": _authority(),
        "runtime": _runtime(),
        "protocol": _protocol(),
        "initializer": {
            "kind": "imprinted",
            "feature_sha256": _digest(50),
            "label_sha256": _digest(51),
            "initial_head_sha256": _digest(40),
            "validation_feature_sha256": _digest(52),
        },
        "phase1": phase1,
        "selected_learning_rates": selections,
        "phase2": phase2,
        "comparisons": comparisons,
        "predicates": predicates,
        "process": {
            "command": [".venv/bin/python", "-I", "-B", "script", "--config", "config"],
            "elapsed_seconds": 10.0,
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
            "completed_cells": 30 + len(phase2),
        },
    }


def _failure_fixture() -> dict[str, object]:
    return {
        "schema_version": module.FAILURE_SCHEMA_VERSION,
        "status": "STRUCTURAL_FAILURE",
        "authority": _authority(),
        "runtime": None,
        "protocol": _protocol(),
        "completed_cells": 2,
        "completed_cell_sha256s": [_digest(900), _digest(901)],
        "error": {"class": "RuntimeError", "message": "boom"},
        "process": {
            "command": ["python", "script"],
            "elapsed_seconds": 1.0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "completed_cells": 2,
        },
    }


def test_assemblers_recompute_scientific_route_and_failure_prefix() -> None:
    """Catch assemblers that trust caller-supplied decisions or drop prefix evidence."""

    expected = _scientific_fixture()
    assembled = module.assemble_scientific_result(
        authority=expected["authority"],
        runtime=expected["runtime"],
        protocol=expected["protocol"],
        initializer=expected["initializer"],
        phase1=expected["phase1"],
        phase2=expected["phase2"],
        process=expected["process"],
    )
    assert assembled == expected

    failure = _failure_fixture()
    rebuilt_failure = module.assemble_failure_receipt(
        authority=failure["authority"],
        runtime=failure["runtime"],
        protocol=failure["protocol"],
        completed_cell_sha256s=failure["completed_cell_sha256s"],
        error=RuntimeError("boom"),
        process=failure["process"],
    )
    assert rebuilt_failure == failure


def test_authenticated_orchestrator_runs_both_phases_and_assembles_exact_result() -> None:
    expected = _scientific_fixture()
    features = torch.arange(8, dtype=torch.float32).reshape(2, 4).contiguous()
    labels = torch.tensor([0, 1], dtype=torch.int64)
    validation_features = (features + 1.0).contiguous()
    initial = torch.eye(4, dtype=torch.float32).contiguous()
    initial_sha256 = module.tensor_sha256(initial)
    expected["authority"]["inputs"]["imprinted_head"] = initial_sha256
    expected["initializer"] = {
        "kind": "imprinted",
        "feature_sha256": module.tensor_sha256(features),
        "label_sha256": module.tensor_sha256(labels),
        "initial_head_sha256": initial_sha256,
        "validation_feature_sha256": module.tensor_sha256(validation_features),
    }
    for row in expected["phase1"]:
        row["initial_head_sha256"] = initial_sha256
    for row in expected["phase2"]:
        row["retained"][0]["head_sha256"] = initial_sha256
    completed: list[str] = []
    callbacks: list[object] = []

    class Budget:
        def observe(self, _step: int) -> None:
            return None

        def snapshot(self) -> dict[str, object]:
            return {
                "elapsed_seconds": 10.0,
                "peak_allocated_bytes": 1024,
                "peak_reserved_bytes": 2048,
            }

    budget = Budget()

    def phase1_runner(*_args, progress_callback, **_kwargs):
        callbacks.append(progress_callback)
        rows = expected["phase1"]
        completed.extend(
            __import__("hashlib").sha256(module.canonical_json_bytes(row)).hexdigest()
            for row in rows
        )
        return rows

    def phase2_runner(
        *_args,
        adamw_learning_rate,
        proxy_muon_learning_rate,
        progress_callback,
        **_kwargs,
    ):
        assert adamw_learning_rate == 0.0001
        assert proxy_muon_learning_rate == 0.0002
        callbacks.append(progress_callback)
        rows = expected["phase2"]
        completed.extend(
            __import__("hashlib").sha256(module.canonical_json_bytes(row)).hexdigest()
            for row in rows
        )
        return rows

    evidence = {
        "fitting_features": features,
        "fitting_labels": labels,
        "validation_features": validation_features,
        "validation_labels": labels,
        "validation_group_represented": object(),
        "imprinted_head": initial,
    }
    actual = module.run_proxy_muon_f0(
        protocol=expected["protocol"],
        authority=expected["authority"],
        runtime=expected["runtime"],
        evidence=evidence,
        budget=budget,
        completed_cell_sha256s=completed,
        command=expected["process"]["command"],
        phase1_runner=phase1_runner,
        phase2_runner=phase2_runner,
    )

    assert actual == expected
    assert callbacks == [budget.observe, budget.observe]


def _set_validation_losses(
    payload: dict[str, object], variant: str, losses: tuple[float, float, float]
) -> None:
    for phase2_row in payload["phase2"]:
        if phase2_row["variant"] != variant:
            continue
        for step, loss in zip((307, 435, 512), losses, strict=True):
            retained = next(row for row in phase2_row["retained"] if row["step"] == step)
            retained["diagnostic"] = _panel(loss)


def _set_validation_accuracies(
    payload: dict[str, object], variant: str, accuracies: tuple[float, float, float]
) -> None:
    for phase2_row in payload["phase2"]:
        if phase2_row["variant"] != variant:
            continue
        for step, accuracy in zip((307, 435, 512), accuracies, strict=True):
            retained = next(row for row in phase2_row["retained"] if row["step"] == step)
            retained["validation_accuracy"] = accuracy + phase2_row["fit_seed"] * 0.001


def _rebuild_comparisons_and_predicates(payload: dict[str, object]) -> None:
    comparisons = []
    for seed in (3, 4, 5):
        per_seed = {
            row["variant"]: row
            for row in payload["phase2"]
            if row["fit_seed"] == seed
        }
        adamw_rows = [per_seed["adamw_selected"]]
        if "adamw_anchor" in per_seed:
            adamw_rows.append(per_seed["adamw_anchor"])

        def at(row: dict[str, object], step: int) -> dict[str, object]:
            return next(item for item in row["retained"] if item["step"] == step)

        reference_row = min(
            adamw_rows,
            key=lambda row: (at(row, 512)["diagnostic"]["mean"], row["learning_rate"]),
        )
        reference_loss = at(reference_row, 512)["diagnostic"]["mean"]
        reference_accuracy = at(reference_row, 512)["validation_accuracy"]

        def variant_summary(
            variant: str,
            rows: dict[str, dict[str, object]] = per_seed,
            reference_step512_loss: float = reference_loss,
            reference_step512_accuracy: float = reference_accuracy,
        ) -> tuple[int | str, float | None, float | None, float]:
            row = rows[variant]
            reach: int | str = ">512"
            for step in (307, 435, 512):
                if (
                    at(row, step)["diagnostic"]["mean"]
                    <= reference_step512_loss
                ):
                    reach = step
                    break
            accuracy_at_reach = (
                None if reach == ">512" else at(row, reach)["validation_accuracy"]
            )
            delta_at_reach = (
                None
                if accuracy_at_reach is None
                else accuracy_at_reach - reference_step512_accuracy
            )
            step512_delta = (
                at(row, 512)["validation_accuracy"] - reference_step512_accuracy
            )
            return reach, accuracy_at_reach, delta_at_reach, step512_delta

        proxy = variant_summary("proxy_muon")
        fp32 = variant_summary("proxy_muon_fp32")
        comparisons.append(
            {
                "fit_seed": seed,
                "adamw_reference_variant": reference_row["variant"],
                "adamw_reference_learning_rate": reference_row["learning_rate"],
                "adamw_reference_step512_loss": reference_loss,
                "adamw_reference_step512_accuracy": reference_accuracy,
                "proxy_muon_reach_step": proxy[0],
                "proxy_muon_accuracy_at_reach": proxy[1],
                "proxy_muon_accuracy_delta": proxy[2],
                "proxy_muon_step512_accuracy_delta": proxy[3],
                "proxy_muon_fp32_reach_step": fp32[0],
                "proxy_muon_fp32_accuracy_at_reach": fp32[1],
                "proxy_muon_fp32_accuracy_delta": fp32[2],
                "proxy_muon_fp32_step512_accuracy_delta": fp32[3],
            }
        )
    payload["comparisons"] = comparisons
    proxy_route = all(
        row["proxy_muon_reach_step"] == 307
        and row["proxy_muon_accuracy_delta"] is not None
        and row["proxy_muon_accuracy_delta"] >= -0.002
        and row["proxy_muon_step512_accuracy_delta"] >= -0.002
        for row in comparisons
    )
    fp32_route = all(
        row["proxy_muon_fp32_reach_step"] == 307
        and row["proxy_muon_fp32_accuracy_delta"] is not None
        and row["proxy_muon_fp32_accuracy_delta"] >= -0.002
        and row["proxy_muon_fp32_step512_accuracy_delta"] >= -0.002
        for row in comparisons
    )
    payload["predicates"] = {
        "adamw_lr_interior": payload["selected_learning_rates"]["adamw"]["interior"],
        "proxy_muon_lr_interior": payload["selected_learning_rates"]["proxy_muon"]["interior"],
        "all_reach_by_307": all(
            row["proxy_muon_reach_step"] == 307 for row in comparisons
        ),
        "all_reach_noninferior": all(
            row["proxy_muon_accuracy_delta"] is not None
            and row["proxy_muon_accuracy_delta"] >= -0.002
            for row in comparisons
        ),
        "all_step512_noninferior": all(
            row["proxy_muon_step512_accuracy_delta"] >= -0.002
            for row in comparisons
        ),
        "any_bf16_accuracy_loss": any(
            row["proxy_muon_accuracy_delta"] is not None
            and row["proxy_muon_accuracy_delta"] < -0.002
            for row in comparisons
        )
        or any(
            row["proxy_muon_step512_accuracy_delta"] < -0.002
            for row in comparisons
        ),
        "fp32_sensitivity_supported": (not proxy_route) and fp32_route,
    }


@pytest.mark.parametrize("adamw_selected,expected_rows", [(0.0001, 9), (0.0002, 12)])
def test_scientific_fixture_validates_both_phase2_cardinalities(
    adamw_selected: float, expected_rows: int
) -> None:
    payload = _scientific_fixture(adamw_selected=adamw_selected)
    assert len(payload["phase1"]) == 30
    assert len(payload["phase2"]) == expected_rows
    assert module.validate_scientific_result(payload) is payload


def test_scientific_validator_records_reserved_memory_without_capping_it() -> None:
    payload = _scientific_fixture()
    payload["process"]["peak_reserved_bytes"] = 8 * 1024**3 + 1

    assert module.validate_scientific_result(payload) is payload


@pytest.mark.parametrize(
    "scenario,status",
    (
        ("proceed", "PROCEED_TRAINING"),
        ("fp32", "ROUTE_FP32_ORTHOGONALIZATION"),
        ("fp32_step512_inferior", "CLOSE_PROXY_MUON"),
        ("matched", "ROUTE_MATCHED_LR"),
        ("close", "CLOSE_PROXY_MUON"),
        ("boundary", "UNRESOLVED_LR_BOUNDARY"),
    ),
)
def test_scientific_validator_accepts_every_recomputed_route(
    scenario: str, status: str
) -> None:
    payload = _scientific_fixture(
        adamw_selected=0.000025 if scenario == "boundary" else 0.0001
    )
    if scenario in ("fp32", "fp32_step512_inferior"):
        _set_validation_losses(payload, "proxy_muon", (0.9, 0.9, 0.9))
        if scenario == "fp32_step512_inferior":
            _set_validation_accuracies(
                payload, "proxy_muon_fp32", (0.91, 0.91, 0.89)
            )
    elif scenario == "matched":
        _set_validation_losses(payload, "proxy_muon", (0.9, 0.54, 0.53))
        _set_validation_losses(payload, "proxy_muon_fp32", (0.9, 0.54, 0.53))
    elif scenario == "close":
        _set_validation_losses(payload, "proxy_muon", (0.9, 0.9, 0.9))
        _set_validation_losses(payload, "proxy_muon_fp32", (0.9, 0.9, 0.9))
    _rebuild_comparisons_and_predicates(payload)
    payload["status"] = status

    assert module.validate_scientific_result(payload) is payload
    if scenario in ("fp32", "fp32_step512_inferior", "close"):
        assert payload["predicates"]["any_bf16_accuracy_loss"] is False
    if scenario == "fp32_step512_inferior":
        assert payload["comparisons"][0][
            "proxy_muon_fp32_step512_accuracy_delta"
        ] < -0.002
        assert payload["comparisons"][0]["proxy_muon_step512_accuracy_delta"] > 0.0


def test_failure_fixture_is_disjoint_and_valid() -> None:
    payload = _failure_fixture()
    assert module.validate_failure_receipt(payload) is payload
    with pytest.raises(ValueError):
        module.validate_scientific_result(payload)


def test_strict_json_rejects_duplicates_constants_and_nonobjects() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        module.strict_json_object(b'{"x":1,"x":2}')
    with pytest.raises(ValueError, match="non-finite"):
        module.strict_json_object(b'{"x":NaN}')
    with pytest.raises(TypeError, match="object"):
        module.strict_json_object(b"[]")


@pytest.mark.parametrize(
    "mutation",
    (
        "top_order",
        "bool_step",
        "bad_hash",
        "initializer_authority",
        "phase1_initial_head",
        "phase1_initial_diagnostic",
        "phase2_initial_head",
        "phase2_initial_diagnostic",
        "panel_component",
        "panel_mean",
        "selection",
        "phase1_order",
        "phase2_order",
        "trace_dtype",
        "validation_wrong_step",
        "comparison",
        "predicate",
        "status",
        "elapsed",
        "peak",
        "completed",
        "integer_float",
        "nested_extra",
    ),
)
def test_scientific_result_rejects_independent_or_dependent_drift(mutation: str) -> None:
    payload = _scientific_fixture()
    if mutation == "top_order":
        payload = {
            "status": payload["status"],
            **{key: value for key, value in payload.items() if key != "status"},
        }
    elif mutation == "bool_step":
        payload["phase1"][0]["steps"] = True
    elif mutation == "bad_hash":
        payload["phase1"][0]["final_head_sha256"] = "z" * 64
    elif mutation == "initializer_authority":
        payload["initializer"]["initial_head_sha256"] = _digest(99)
    elif mutation == "phase1_initial_head":
        payload["phase1"][0]["initial_head_sha256"] = _digest(99)
    elif mutation == "phase1_initial_diagnostic":
        payload["phase1"][3]["diagnostic_step0"] = _panel(9.0)
    elif mutation == "phase2_initial_head":
        payload["phase2"][0]["retained"][0]["head_sha256"] = _digest(99)
    elif mutation == "phase2_initial_diagnostic":
        payload["phase2"][1]["retained"][0]["diagnostic"] = _panel(9.0)
    elif mutation == "panel_component":
        payload["phase1"][0]["diagnostic_step64"]["components"][0] += 1.0
    elif mutation == "panel_mean":
        payload["phase1"][0]["diagnostic_step64"]["mean"] += 1.0
    elif mutation == "selection":
        payload["selected_learning_rates"]["proxy_muon"]["learning_rate"] = 0.0001
    elif mutation == "phase1_order":
        payload["phase1"][0], payload["phase1"][1] = payload["phase1"][1], payload["phase1"][0]
    elif mutation == "phase2_order":
        payload["phase2"][0], payload["phase2"][1] = payload["phase2"][1], payload["phase2"][0]
    elif mutation == "trace_dtype":
        payload["phase2"][1]["retained"][1]["update_dtype"] = "torch.float32"
    elif mutation == "validation_wrong_step":
        payload["phase2"][0]["retained"][1]["validation_accuracy"] = 0.9
    elif mutation == "comparison":
        payload["comparisons"][0]["proxy_muon_reach_step"] = 435
    elif mutation == "predicate":
        payload["predicates"]["all_reach_by_307"] = False
    elif mutation == "status":
        payload["status"] = "CLOSE_PROXY_MUON"
    elif mutation == "elapsed":
        payload["process"]["elapsed_seconds"] = 2700.1
    elif mutation == "peak":
        payload["process"]["peak_allocated_bytes"] = 8 * 1024**3 + 1
    elif mutation == "completed":
        payload["process"]["completed_cells"] -= 1
    elif mutation == "integer_float":
        payload["process"]["elapsed_seconds"] = 10
    elif mutation == "nested_extra":
        payload["phase1"][0]["diagnostic_step64"]["extra"] = 1
    with pytest.raises((TypeError, ValueError)):
        module.validate_scientific_result(payload)


@pytest.mark.parametrize("mutation", ("extra", "count", "hash", "runtime", "bool"))
def test_failure_receipt_rejects_schema_and_evidence_drift(mutation: str) -> None:
    payload = _failure_fixture()
    if mutation == "extra":
        payload["phase1"] = []
    elif mutation == "count":
        payload["completed_cells"] = 1
    elif mutation == "hash":
        payload["completed_cell_sha256s"][0] = "x" * 64
    elif mutation == "runtime":
        payload["runtime"] = {"python_version": "3.12.3"}
    elif mutation == "bool":
        payload["process"]["completed_cells"] = True
    with pytest.raises((TypeError, ValueError)):
        module.validate_failure_receipt(payload)


def test_dependent_route_rewrite_cannot_override_independent_phase2_rows() -> None:
    payload = _scientific_fixture()
    for comparison in payload["comparisons"]:
        comparison["proxy_muon_reach_step"] = 435
    payload["predicates"]["all_reach_by_307"] = False
    payload["status"] = "ROUTE_MATCHED_LR"

    with pytest.raises(ValueError, match=r"comparison\[3\] differs"):
        module.validate_scientific_result(payload)


def test_canonical_publication_reloads_distinct_object_and_never_clobbers(
    tmp_path: Path,
) -> None:
    payload = _failure_fixture()
    output = tmp_path / "result.json"
    seen: list[object] = []

    def validator(value: object) -> None:
        seen.append(value)
        module.validate_failure_receipt(value)

    persisted = module.publish_result_exclusive(output, payload, validator)

    assert persisted == module.canonical_json_bytes(payload)
    assert output.read_bytes() == persisted
    assert oct(output.stat().st_mode & 0o777) == "0o600"
    assert len(seen) == 2
    assert seen[0] is payload
    assert seen[1] is not payload
    assert seen[1] == payload
    assert not list(tmp_path.glob(".*.tmp.*"))
    with pytest.raises(FileExistsError):
        module.publish_result_exclusive(output, payload, validator)
    assert output.read_bytes() == persisted


@pytest.mark.parametrize(
    "operation",
    ("write", "rename", "directory_fsync", "reload", "prevalidate", "postvalidate"),
)
def test_publication_failure_is_never_retried_or_clobbered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    payload = _failure_fixture()
    output = tmp_path / "result.json"
    calls = 0

    def validator(value: object) -> None:
        nonlocal calls
        calls += 1
        if operation == "prevalidate" and calls == 1:
            raise ValueError("injected prevalidation failure")
        if operation == "postvalidate" and calls == 2:
            raise ValueError("injected postvalidation failure")
        module.validate_failure_receipt(value)

    helper = {
        "write": "_write_temp_exclusive",
        "rename": "_rename_noreplace",
        "directory_fsync": "_fsync_directory",
        "reload": "_read_persisted",
    }.get(operation)
    if helper is not None:
        def fail(*_args, **_kwargs):
            raise OSError(f"injected {operation} failure")

        monkeypatch.setattr(module, helper, fail)

    with pytest.raises((OSError, ValueError)):
        module.publish_result_exclusive(output, payload, validator)

    assert not list(tmp_path.glob(".*.tmp.*"))
    if operation in ("write", "rename", "prevalidate"):
        assert not output.exists()
    else:
        assert output.is_file()
        original = output.read_bytes()
        with pytest.raises(FileExistsError):
            module.publish_result_exclusive(output, payload, validator)
        assert output.read_bytes() == original


@pytest.mark.parametrize(
    "operation", ("open", "fdopen", "fchmod", "write", "short_write", "flush", "fsync")
)
def test_temp_writer_propagates_each_durability_operation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    encoded = b"payload\n"

    class FakeHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def fileno(self) -> int:
            return 17

        def write(self, value: bytes) -> int:
            if operation == "write":
                raise OSError("write")
            return len(value) - 1 if operation == "short_write" else len(value)

        def flush(self) -> None:
            if operation == "flush":
                raise OSError("flush")

    def fail_if(selected: str):
        if operation == selected:
            raise OSError(selected)

    monkeypatch.setattr(module.os, "open", lambda *_args: (fail_if("open"), 17)[1])
    monkeypatch.setattr(
        module.os,
        "fdopen",
        lambda *_args, **_kwargs: (fail_if("fdopen"), FakeHandle())[1],
    )
    monkeypatch.setattr(module.os, "fchmod", lambda *_args: fail_if("fchmod"))
    monkeypatch.setattr(module.os, "fsync", lambda *_args: fail_if("fsync"))
    monkeypatch.setattr(module.os, "close", lambda *_args: None)

    with pytest.raises(OSError):
        module._write_temp_exclusive(tmp_path / "temp", encoded)


def test_canonical_json_bytes_are_utf8_sorted_only_by_insertion_order() -> None:
    payload = {"z": "λ", "a": 1}
    assert module.canonical_json_bytes(payload) == '{"z":"λ","a":1}\n'.encode()
    assert json.loads(module.canonical_json_bytes(payload)) == payload


def test_cli_accepts_only_exact_config_argument(tmp_path: Path) -> None:
    config = tmp_path / "run.json"
    assert module.parse_args(["--config", str(config)]).config == config
    with pytest.raises(SystemExit):
        module.parse_args([])
    with pytest.raises(SystemExit):
        module.parse_args(["--config", str(config), "--output", "x"])


def test_main_publishes_only_reduced_receipt_after_authenticated_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "reports").mkdir()
    config = {
        "environment": _runtime(),
        "protocol": _protocol(),
        "handoff": {"sole_path": "docs/unicom_proxy_muon_f0_run_config.json"},
        "result": {
            "relative_path": "reports/result.json",
            "failure_relative_path": "reports/failure.json",
        },
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module.sys, "flags", SimpleNamespace(isolated=1, dont_write_bytecode=1)
    )
    monkeypatch.setattr(module, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        module, "authenticate_source_and_inputs", lambda *_args: _authority()
    )
    monkeypatch.setattr(module, "observe_runtime", _runtime)
    monkeypatch.setattr(module, "preserve_process_state", nullcontext)

    def fail_reconstruction(*_args):
        raise RuntimeError("injected reconstruction failure")

    monkeypatch.setattr(module, "reconstruct_parent_evidence", fail_reconstruction)

    assert module.main(["--config", "ignored.json"]) == 2
    assert not (tmp_path / "reports" / "result.json").exists()
    receipt = module.strict_json_object(
        (tmp_path / "reports" / "failure.json").read_bytes()
    )
    module.validate_failure_receipt(receipt)
    assert receipt["runtime"] == _runtime()
    assert receipt["completed_cells"] == 0
    assert receipt["error"] == {
        "class": "RuntimeError",
        "message": "injected reconstruction failure",
    }


def test_main_does_not_publish_failure_receipt_after_success_publication_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "reports").mkdir()
    config = {
        "environment": _runtime(),
        "protocol": _protocol(),
        "handoff": {"sole_path": "docs/unicom_proxy_muon_f0_run_config.json"},
        "result": {
            "relative_path": "reports/result.json",
            "failure_relative_path": "reports/failure.json",
        },
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module.sys, "flags", SimpleNamespace(isolated=1, dont_write_bytecode=1)
    )
    monkeypatch.setattr(module, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        module, "authenticate_source_and_inputs", lambda *_args: _authority()
    )
    monkeypatch.setattr(module, "observe_runtime", _runtime)
    monkeypatch.setattr(module, "preserve_process_state", nullcontext)
    monkeypatch.setattr(
        module,
        "reconstruct_parent_evidence",
        lambda *_args: {
            "fitting_features": object(),
            "fitting_labels": object(),
            "validation_features": object(),
            "validation_labels": object(),
            "validation_group_represented": object(),
            "imprinted_head": object(),
        },
    )
    monkeypatch.setattr(
        module,
        "run_proxy_muon_f0",
        lambda **_kwargs: _scientific_fixture(),
    )
    attempted: list[Path] = []

    def fail_publication(path: Path, *_args) -> None:
        attempted.append(path)
        raise OSError("injected publication failure")

    monkeypatch.setattr(module, "publish_result_exclusive", fail_publication)

    assert module.main(["--config", "ignored.json"]) == 2
    assert attempted == [tmp_path / "reports" / "result.json"]
    assert not (tmp_path / "reports" / "failure.json").exists()


def test_main_starts_registered_budget_after_parent_reconstruction_and_cache_clear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    config = {
        "environment": _runtime(),
        "protocol": _protocol(),
        "handoff": {"sole_path": "docs/unicom_proxy_muon_f0_run_config.json"},
        "result": {
            "relative_path": "reports/result.json",
            "failure_relative_path": "reports/failure.json",
        },
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module.sys, "flags", SimpleNamespace(isolated=1, dont_write_bytecode=1)
    )
    monkeypatch.setattr(module, "load_run_config", lambda _path: config)
    monkeypatch.setattr(
        module, "authenticate_source_and_inputs", lambda *_args: _authority()
    )
    monkeypatch.setattr(module, "observe_runtime", _runtime)
    monkeypatch.setattr(module, "preserve_process_state", nullcontext)
    monkeypatch.setattr(
        module,
        "reconstruct_parent_evidence",
        lambda *_args: (events.append("reconstruct"), {})[1],
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda, "empty_cache", lambda: events.append("empty_cache")
    )
    monkeypatch.setattr(
        module.torch.cuda,
        "reset_peak_memory_stats",
        lambda _device: events.append("reset_peak"),
    )

    class Budget:
        def __init__(self, **_kwargs):
            events.append("budget")

        def snapshot(self):
            return {
                "elapsed_seconds": 0.0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            }

    monkeypatch.setattr(module, "ExecutionBudget", Budget)
    monkeypatch.setattr(
        module,
        "run_proxy_muon_f0",
        lambda **_kwargs: (events.append("run"), _scientific_fixture())[1],
    )
    monkeypatch.setattr(module, "publish_result_exclusive", lambda *_args: None)

    assert module.main(["--config", "ignored.json"]) == 0
    assert events[:5] == [
        "reconstruct",
        "empty_cache",
        "reset_peak",
        "budget",
        "run",
    ]


def test_process_state_guard_restores_rng_determinism_and_backend_flags() -> None:
    import random

    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic

    with module.preserve_process_state():
        random.seed(91)
        np.random.seed(92)
        torch.manual_seed(93)
        torch.use_deterministic_algorithms(not deterministic)
        torch.backends.cudnn.benchmark = not cudnn_benchmark
        torch.backends.cudnn.deterministic = not cudnn_deterministic

    assert random.getstate() == python_state
    restored_numpy = np.random.get_state()
    assert restored_numpy[0] == numpy_state[0]
    assert np.array_equal(restored_numpy[1], numpy_state[1])
    assert restored_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    assert torch.backends.cudnn.benchmark is cudnn_benchmark
    assert torch.backends.cudnn.deterministic is cudnn_deterministic


def test_authenticated_parent_feature_module_loads_by_exact_git_bytes() -> None:
    repo_root = SCRIPT.parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    parent = repo_root / "scripts" / "screen_unicom_spherical_probe.py"
    expected_sha256 = __import__("hashlib").sha256(parent.read_bytes()).hexdigest()

    loaded = module.load_spherical_feature_module(
        repo_root, revision, expected_sha256
    )
    try:
        assert Path(loaded.__file__).resolve() == parent.resolve()
        assert loaded._load_official_model.__module__ == module.PARENT_MODULE_NAME
        assert loaded._encode_feature_sets.__module__ == module.PARENT_MODULE_NAME
    finally:
        sys.modules.pop(module.PARENT_MODULE_NAME, None)


def test_parent_feature_loader_rejects_wrong_hash_symlink_and_preimport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = SCRIPT.parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    parent = repo_root / "scripts" / "screen_unicom_spherical_probe.py"
    expected_sha256 = __import__("hashlib").sha256(parent.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        module.load_spherical_feature_module(repo_root, revision, "0" * 64)
    monkeypatch.setitem(sys.modules, module.PARENT_MODULE_NAME, object())
    with pytest.raises(ValueError, match="before source authentication"):
        module.load_spherical_feature_module(repo_root, revision, expected_sha256)
    monkeypatch.delitem(sys.modules, module.PARENT_MODULE_NAME)
    fake_root = tmp_path / "repo"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "scripts" / "screen_unicom_spherical_probe.py").symlink_to(parent)
    with pytest.raises((FileNotFoundError, ValueError)):
        module.load_spherical_feature_module(fake_root, revision, expected_sha256)


def test_parent_feature_loader_works_under_isolated_python() -> None:
    repo_root = SCRIPT.parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    parent = repo_root / "scripts" / "screen_unicom_spherical_probe.py"
    expected_sha256 = __import__("hashlib").sha256(parent.read_bytes()).hexdigest()
    code = """
import importlib.util, json, pathlib, sys
runner_path, repo_root, revision, digest = sys.argv[1:]
spec = importlib.util.spec_from_file_location("isolated_proxy_runner", runner_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
loaded = module.load_spherical_feature_module(pathlib.Path(repo_root), revision, digest)
print(json.dumps({
    "file": str(pathlib.Path(loaded.__file__).resolve()),
    "loader_module": loaded._load_official_model.__module__,
    "encoder_module": loaded._encode_feature_sets.__module__,
}))
sys.modules.pop(module.PARENT_MODULE_NAME, None)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(SCRIPT),
            str(repo_root),
            revision,
            expected_sha256,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(completed.stdout)
    assert evidence == {
        "file": str(parent.resolve()),
        "loader_module": module.PARENT_MODULE_NAME,
        "encoder_module": module.PARENT_MODULE_NAME,
    }


def test_training_only_partition_loader_never_opens_query_or_gallery(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    partition.parent.mkdir(parents=True)
    train_image = dataset / "Img" / "img" / "train" / "a.jpg"
    train_image.parent.mkdir(parents=True)
    train_image.write_bytes(b"not-opened-image-bytes")
    partition.write_text(
        "3\n"
        "image_name item_id evaluation_status\n"
        "img/train/a.jpg id_1 train\n"
        "img/query/poison.jpg id_2 query\n"
        "img/gallery/poison.jpg id_3 gallery\n",
        encoding="utf-8",
    )

    records = module.load_training_only_records(partition, dataset)

    assert len(records) == 1
    assert records[0].split == "train"
    assert records[0].label == "id_1"
    assert records[0].image_path == train_image


def test_execution_budget_observer_checks_every_step_and_tracks_exact_peaks() -> None:
    elapsed_values = iter((100.0, 101.5, 102.0, 2801.0))
    allocated_values = iter((100, 300, 250))
    reserved_values = iter((200, 400, 250))
    budget = module.ExecutionBudget(
        elapsed_limit_seconds=2700.0,
        peak_limit_bytes=8 * 1024**3,
        clock=lambda: next(elapsed_values),
        allocated=lambda: next(allocated_values),
        reserved=lambda: next(reserved_values),
    )

    budget.observe(1)
    budget.observe(2)
    assert budget.snapshot() == {
        "elapsed_seconds": 2.0,
        "peak_allocated_bytes": 300,
        "peak_reserved_bytes": 400,
    }
    with pytest.raises(RuntimeError, match="elapsed limit"):
        budget.observe(3)


def test_execution_budget_caps_allocated_memory_and_records_reserved_only() -> None:
    limit = 8 * 1024**3
    budget = module.ExecutionBudget(
        elapsed_limit_seconds=2700.0,
        peak_limit_bytes=limit,
        clock=iter((0.0, 1.0)).__next__,
        allocated=lambda: limit,
        reserved=lambda: limit + 1,
    )

    budget.observe(1)

    assert budget.snapshot()["peak_reserved_bytes"] == limit + 1


def test_real_cpu_optimizer_to_validated_receipt_integration(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(205)
    class_count = 32
    dimension = 16
    labels = torch.arange(class_count, dtype=torch.int64).repeat_interleave(16)
    centers = torch.randn(class_count, dimension, generator=generator)
    centers = torch.nn.functional.normalize(centers, dim=1)
    features = centers.index_select(0, labels) + 0.01 * torch.randn(
        labels.shape[0], dimension, generator=generator
    )
    features = torch.nn.functional.normalize(features, dim=1).float().contiguous()
    initial = module.class_mean_head(
        features, labels, class_count=class_count
    ).contiguous()
    represented = tuple(index % 2 == 0 for index in range(labels.numel()))

    authority = copy.deepcopy(_authority())
    authority["inputs"]["imprinted_head"] = module.tensor_sha256(initial)
    completed: list[str] = []

    class Budget:
        def __init__(self) -> None:
            self.observations = 0

        def observe(self, _step: int) -> None:
            self.observations += 1

        def snapshot(self) -> dict[str, object]:
            return {
                "elapsed_seconds": 1.0,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            }

    budget = Budget()
    feature_ref = weakref.ref(features)
    initial_ref = weakref.ref(initial)
    result = module.run_proxy_muon_f0(
        protocol=_protocol(),
        authority=authority,
        runtime=_runtime(),
        evidence={
            "fitting_features": features,
            "fitting_labels": labels,
            "validation_features": features,
            "validation_labels": labels,
            "validation_group_represented": represented,
            "imprinted_head": initial,
        },
        budget=budget,
        completed_cell_sha256s=completed,
        command=["python", "-I", "-B", "screen.py", "--config", "run.json"],
        selected_features=16,
    )
    assert module.validate_scientific_result(result) is result
    assert [
        (row["optimizer"], row["learning_rate"], row["fit_seed"])
        for row in result["phase1"]
    ] == list(module.phase1_cell_schedule())
    assert [
        (row["fit_seed"], row["variant"], row["learning_rate"])
        for row in result["phase2"]
    ] == list(
        module.phase2_cell_schedule(
            result["selected_learning_rates"]["adamw"]["learning_rate"],
            result["selected_learning_rates"]["proxy_muon"]["learning_rate"],
        )
    )
    assert len(completed) == 30 + len(result["phase2"])
    assert budget.observations == 1_920 + 30 + 512 * len(result["phase2"]) + len(
        result["phase2"]
    )

    output = tmp_path / "result.json"
    persisted = module.publish_result_exclusive(
        output, result, module.validate_scientific_result
    )
    assert module.strict_json_object(persisted) == result
    with pytest.raises(FileExistsError):
        module.publish_result_exclusive(
            output, result, module.validate_scientific_result
        )

    del features, initial
    gc.collect()
    assert feature_ref() is None
    assert initial_ref() is None
