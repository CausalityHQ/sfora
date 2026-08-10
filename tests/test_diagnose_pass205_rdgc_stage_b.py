"""Synthetic CPU tests for the prospective Pass 205 RDGC diagnostic."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import types
import weakref
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts/diagnose_pass205_rdgc_stage_b.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_pass205_rdgc_stage_b", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_module_import_is_torch_free_in_fresh_process() -> None:
    program = (
        "import importlib.util,sys;"
        f"p={str(_SCRIPT)!r};"
        "s=importlib.util.spec_from_file_location('rdgc_import_probe',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert 'torch' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=_ROOT, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def _finite_tensor(values: list[float], *, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, requires_grad=requires_grad)


def _preliminary_aggregates() -> dict[str, object]:
    return {
        "count_gain_median_A": math.log(1.20),
        "count_gain_median_B": math.log(1.20),
        "positive_count_gain_seed_means_A": 4,
        "positive_count_gain_seed_means_B": 4,
        "context_spearman_median": 0.70,
        "context_spearman_seeds_ge_point_three": 4,
        "log_kappa_iqr_A": math.log(1.20),
        "log_kappa_iqr_B": math.log(1.20),
        "global_scalar_relative_error_median_A": 0.08,
        "global_scalar_relative_error_median_B": 0.08,
        "full_gain_error_median_A": math.log(1.30),
        "full_gain_error_median_B": math.log(1.30),
        "full_gain_error_seed_medians_ge_log_one_point_one_A": 4,
        "full_gain_error_seed_medians_ge_log_one_point_one_B": 4,
    }


def _panel_aggregates() -> dict[str, object]:
    metric = lambda pooled=0.04, lower=0.01, positive=4: {  # noqa: E731
        "pooled_difference": pooled,
        "lower_bound": lower,
        "positive_seed_means": positive,
        "nonpositive_seed_means": 4 - positive,
    }
    controls = {
        name: {
            "primary_alignment": metric(),
            "primary_slope": metric(),
            "context_b_alignment": metric(),
            "context_b_slope": metric(),
        }
        for name in _MODULE.CONTROL_ORDER
    }
    return {
        "primary_pa_alignment": metric(pooled=0.03),
        "primary_pa_slope": metric(),
        "context_b_pa_alignment": metric(),
        "controls": controls,
        "correction_aliases": {
            name: {
                "pooled_median_absolute_cosine": 0.40,
                "seed_medians_ge_point_nine_nine": 0,
            }
            for name in _MODULE.CONTROL_ORDER
        },
        "completeness": True,
    }


def test_rdgc_is_exact_half_squared_log_gain_error() -> None:
    b = _finite_tensor([3.0, 4.0], requires_grad=True)
    s = _finite_tensor([0.0, 2.0], requires_grad=True)
    error = _MODULE.rdgc_error(torch, b, s)
    penalty = _MODULE.rdgc_penalty(torch, b, s)
    expected = math.log((5.0 + 1.0e-8) / (2.0 + 1.0e-8))
    assert error.item() == pytest.approx(expected, rel=1e-7)
    assert penalty.item() == pytest.approx(0.5 * expected * expected, rel=1e-7)


def test_rdgc_detaches_only_scalar_self_norm() -> None:
    b = _finite_tensor([1.0, 2.0], requires_grad=True)
    s = _finite_tensor([3.0, 1.0], requires_grad=True)
    penalty = _MODULE.rdgc_penalty(torch, b, s)
    penalty.backward()
    assert b.grad is not None and torch.count_nonzero(b.grad).item() > 0
    assert s.grad is None


def test_rdgc_has_no_angular_or_vector_self_target_reachability() -> None:
    b = _finite_tensor([1.5, -0.5], requires_grad=True)
    s = _finite_tensor([3.0, 4.0])
    rotated = _finite_tensor([-4.0, 3.0])
    first = _MODULE.rdgc_penalty(torch, b, s)
    second = _MODULE.rdgc_penalty(torch, b, rotated)
    first_gradient = torch.autograd.grad(first, b, retain_graph=True)[0]
    second_gradient = torch.autograd.grad(second, b)[0]
    assert torch.equal(first, second)
    assert torch.equal(first_gradient, second_gradient)


@pytest.mark.parametrize(
    "value",
    (
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor([float("inf")], dtype=torch.float32),
        torch.tensor([float("nan")], dtype=torch.float32),
    ),
)
def test_rdgc_requires_fp32_actions_and_finite_nonzero_norms(value: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        _MODULE.rdgc_penalty(torch, value, _finite_tensor([1.0]))


def test_control_order_and_formulas_are_literal() -> None:
    b = _finite_tensor([3.0, 4.0], requires_grad=True)
    s = _finite_tensor([0.0, 2.0])
    dbar = _finite_tensor([4.0, 0.0])
    receiver_fields = [
        {"s": _finite_tensor([float(index + 1), 0.0]), "dbar": _finite_tensor([2.0, 0.0])}
        for index in range(8)
    ]
    pgn = _finite_tensor([0.0, 6.0], requires_grad=True)
    controls = _MODULE.control_penalties(
        torch,
        b=b,
        s=s,
        dbar=dbar,
        receiver_fields=receiver_fields,
        pgn_motion=pgn,
    )
    assert tuple(controls) == _MODULE.PENALTY_OPERATOR_ORDER
    assert controls["raw_cotangent"].item() == pytest.approx(0.4, rel=1e-6)
    assert controls["full_motion"].item() == pytest.approx(0.5, rel=1e-6)
    assert controls["rdgc"].item() == pytest.approx(
        0.5 * math.log((5.0 + 1e-8) / (2.0 + 1e-8)) ** 2, rel=1e-6
    )
    assert controls["per_example_gradient_normalized"].item() == pytest.approx(
        0.5 * math.log((6.0 + 1e-8) / (2.0 + 1e-8)) ** 2, rel=1e-6
    )


def test_batch_global_gain_uses_eight_receiver_geometric_mean() -> None:
    fields = [
        {"s": _finite_tensor([float(2**index)]), "dbar": _finite_tensor([1.0])}
        for index in range(8)
    ]
    result = _MODULE.control_penalties(
        torch,
        b=_finite_tensor([8.0], requires_grad=True),
        s=fields[0]["s"],
        dbar=fields[0]["dbar"],
        receiver_fields=fields,
        pgn_motion=_finite_tensor([3.0], requires_grad=True),
    )
    target = math.exp(sum(math.log(float(2**index) + 1e-8) for index in range(8)) / 8)
    assert result["batch_global_gain"].item() == pytest.approx(
        0.5 * math.log((8.0 + 1e-8) / target) ** 2, rel=1e-6
    )


def test_scalar_diagonal_raw_uses_batch_gain_times_each_raw_norm() -> None:
    fields = [
        {"s": _finite_tensor([float(index + 2)]), "dbar": _finite_tensor([float(index + 1)])}
        for index in range(8)
    ]
    result = _MODULE.control_penalties(
        torch,
        b=_finite_tensor([5.0], requires_grad=True),
        s=fields[3]["s"],
        dbar=fields[3]["dbar"],
        receiver_fields=fields,
        pgn_motion=_finite_tensor([2.0], requires_grad=True),
    )
    gain = math.exp(
        sum(math.log((float(index + 2) + 1e-8) / (float(index + 1) + 1e-8)) for index in range(8))
        / 8
    )
    target = gain * (4.0 + 1e-8)
    assert result["scalar_diagonal_raw"].item() == pytest.approx(
        0.5 * math.log((5.0 + 1e-8) / target) ** 2, rel=1e-6
    )


def test_per_example_gradient_normalization_uses_all_180_in_row_order() -> None:
    norms = tuple(torch.tensor(float(index + 1), dtype=torch.float64) for index in range(180))
    coefficients = _MODULE.pgn_detached_coefficients(torch, norms)
    expected_nu = math.exp(sum(math.log(float(index + 1) + 1e-12) for index in range(180)) / 180)
    assert coefficients.shape == (180,)
    assert coefficients.dtype == torch.float32
    assert coefficients[0].item() == pytest.approx(expected_nu / (1.0 + 1e-12), rel=1e-6)
    assert coefficients[-1].item() == pytest.approx(expected_nu / (180.0 + 1e-12), rel=1e-6)


def test_pgn_coefficients_are_detached_before_one_weighted_global_vjp() -> None:
    norms = tuple(
        torch.tensor(float(index + 1), dtype=torch.float64, requires_grad=True)
        for index in range(180)
    )
    coefficients = _MODULE.pgn_detached_coefficients(torch, norms)
    assert not coefficients.requires_grad and coefficients.grad_fn is None
    dbar = torch.arange(360, dtype=torch.float32).reshape(180, 2)
    weighted = _MODULE.pgn_weighted_cotangent(torch, dbar, coefficients)
    assert weighted.shape == dbar.shape
    assert torch.equal(weighted[17], dbar[17] * coefficients[17])


def test_pgn_one_global_vjp_is_algebraically_equal_to_tiny_explicit_sum() -> None:
    matrix = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [4.0, -2.0]], dtype=torch.float32)
    parameters = torch.tensor([0.3, -0.7], dtype=torch.float32)

    def outputs(value: torch.Tensor) -> torch.Tensor:
        return matrix * value

    dbar = torch.tensor([[0.2, -0.4], [0.7, 0.3], [-0.1, 0.8]], dtype=torch.float32)
    coefficients = torch.tensor([2.0, 0.5, 1.25], dtype=torch.float32)
    _, vjp = torch.func.vjp(outputs, parameters)
    weighted_global = vjp(dbar * coefficients[:, None])[0]
    explicit = sum(
        (
            torch.func.vjp(outputs, parameters)[1](
                torch.nn.functional.one_hot(torch.tensor(index), num_classes=3).to(torch.float32)[
                    :, None
                ]
                * dbar[index]
            )[0]
            * coefficients[index]
        )
        for index in range(3)
    )
    assert torch.allclose(weighted_global, explicit, atol=1e-6, rtol=1e-6)


def test_full_motion_control_is_generic_normalized_damping() -> None:
    b = _finite_tensor([3.0, 4.0], requires_grad=True)
    result = _MODULE.control_penalties(
        torch,
        b=b,
        s=_finite_tensor([2.0]),
        dbar=_finite_tensor([1.0]),
        receiver_fields=[{"s": _finite_tensor([2.0]), "dbar": _finite_tensor([1.0])}] * 8,
        pgn_motion=_finite_tensor([2.0], requires_grad=True),
    )["full_motion"]
    assert result.item() == pytest.approx(0.5, rel=1e-6)
    assert torch.autograd.grad(result, b)[0].norm().item() > 0


def test_layerwise_trust_ratio_uses_exact_registered_groups_and_formula() -> None:
    named = (
        ("block.weight", _finite_tensor([3.0, 4.0])),
        ("block.bias", _finite_tensor([12.0])),
        ("head", _finite_tensor([5.0])),
    )
    p = (_finite_tensor([6.0, 8.0]), _finite_tensor([24.0]), _finite_tensor([10.0]))
    direction = _MODULE.layerwise_trust_ratio_direction(torch, named, p)
    block_tau = 13.0 / 26.0
    head_tau = 5.0 / 10.0
    assert tuple(t.dtype for t in direction) == (torch.float32,) * 3
    assert torch.allclose(direction[0], p[0] * block_tau)
    assert torch.allclose(direction[1], p[1] * block_tau)
    assert torch.allclose(direction[2], p[2] * head_tau)


def test_layerwise_trust_ratio_is_distinct_from_batch_global_gain() -> None:
    named = (("a.weight", _finite_tensor([2.0])), ("b.weight", _finite_tensor([8.0])))
    p = (_finite_tensor([4.0]), _finite_tensor([2.0]))
    direction = _MODULE.layerwise_trust_ratio_direction(torch, named, p)
    assert direction[0].item() == pytest.approx(2.0)
    assert direction[1].item() == pytest.approx(8.0)
    assert direction[0].item() / p[0].item() != direction[1].item() / p[1].item()


def test_virtual_updates_match_pa_parameter_norm_in_named_order() -> None:
    p = (_finite_tensor([3.0, 4.0]), _finite_tensor([12.0]))
    corrections = {
        name: (_finite_tensor([1.0, -2.0]), _finite_tensor([0.5]))
        for name in _MODULE.CORRECTION_ORDER
    }
    updates = _MODULE.normalize_virtual_updates(torch, p, corrections)
    expected_norm = 13.0
    assert tuple(updates) == _MODULE.OPERATOR_ORDER
    assert updates["pa"] is not p
    for values in updates.values():
        assert tuple(value.dtype for value in values) == (torch.float32, torch.float32)
        assert _MODULE.fp64_named_norm(torch, values).item() == pytest.approx(
            expected_norm, rel=5e-7
        )


def test_virtual_update_uses_alpha_point_one_before_final_normalization() -> None:
    p = (_finite_tensor([3.0, 4.0]),)
    correction = (_finite_tensor([-4.0, 3.0]),)
    updates = _MODULE.normalize_virtual_updates(torch, p, {"rdgc": correction})
    c_hat = np.asarray([-4.0, 3.0], dtype=np.float64)
    v = np.asarray([3.0, 4.0], dtype=np.float64) + 0.10 * c_hat
    expected = 5.0 * v / np.linalg.norm(v)
    assert np.allclose(updates["rdgc"][0].numpy(), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "p,correction",
    (
        ((_finite_tensor([0.0]),), (_finite_tensor([1.0]),)),
        ((_finite_tensor([1.0]).to(torch.float64),), (_finite_tensor([1.0]),)),
        ((_finite_tensor([1.0]),), (_finite_tensor([float("nan")]),)),
        ((_finite_tensor([1.0]), _finite_tensor([2.0])), (_finite_tensor([1.0]),)),
    ),
)
def test_virtual_update_rejects_zero_nonfinite_wrong_dtype_and_reordered_trees(
    p: tuple[torch.Tensor, ...], correction: tuple[torch.Tensor, ...]
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _MODULE.normalize_virtual_updates(torch, p, {"rdgc": correction})


def test_preliminary_survival_requires_every_literal_predicate() -> None:
    valid = _preliminary_aggregates()
    decision = _MODULE.decide_preliminary(valid)
    assert decision == {
        "status": "SURVIVES",
        "first_decisive_clause": "all_survival_predicates",
        "full_panel_authorized": True,
    }
    mutations = {
        "count_gain_median_A": math.nextafter(math.log(1.10), -math.inf),
        "context_spearman_median": math.nextafter(0.50, -math.inf),
        "log_kappa_iqr_A": math.nextafter(math.log(1.10), -math.inf),
        "global_scalar_relative_error_median_A": math.nextafter(0.05, -math.inf),
        "full_gain_error_median_A": math.nextafter(math.log(1.25), -math.inf),
    }
    for key, value in mutations.items():
        case = deepcopy(valid)
        case[key] = value
        assert _MODULE.decide_preliminary(case)["status"] == "UNRESOLVED", key


def test_preliminary_close_precedence_and_exact_boundaries() -> None:
    case = _preliminary_aggregates()
    case["count_gain_median_A"] = 0.0
    case["positive_count_gain_seed_means_A"] = 1
    case["context_spearman_median"] = 0.0
    decision = _MODULE.decide_preliminary(case)
    assert decision == {
        "status": "CLOSE",
        "first_decisive_clause": "close_count_gain",
        "full_panel_authorized": False,
    }
    for key, boundary in (
        ("context_spearman_median", 0.0),
        ("log_kappa_iqr_A", math.log(1.02)),
        ("global_scalar_relative_error_median_A", 0.02),
        ("full_gain_error_median_A", math.log(1.05)),
    ):
        exact = _preliminary_aggregates()
        exact[key] = boundary
        if key == "full_gain_error_median_A":
            exact["full_gain_error_seed_medians_ge_log_one_point_one_A"] = 1
        assert _MODULE.decide_preliminary(exact)["status"] == "CLOSE", key


def test_preliminary_middle_region_is_unresolved() -> None:
    case = _preliminary_aggregates()
    case["global_scalar_relative_error_median_A"] = 0.03
    assert _MODULE.decide_preliminary(case) == {
        "status": "UNRESOLVED",
        "first_decisive_clause": "no_close_or_survival_rule",
        "full_panel_authorized": False,
    }


def test_panel_pass_requires_every_pa_six_control_context_and_alias_predicate() -> None:
    valid = _panel_aggregates()
    assert _MODULE.decide_panel(valid, {})["status"] == "PASS"
    cases: list[dict[str, object]] = []
    case = deepcopy(valid)
    case["primary_pa_alignment"]["pooled_difference"] = 0.019
    cases.append(case)
    case = deepcopy(valid)
    case["controls"][_MODULE.CONTROL_ORDER[0]]["primary_slope"]["lower_bound"] = 0.0
    cases.append(case)
    case = deepcopy(valid)
    case["correction_aliases"][_MODULE.CONTROL_ORDER[0]]["pooled_median_absolute_cosine"] = 0.95
    cases.append(case)
    for mutant in cases:
        assert _MODULE.decide_panel(mutant, {})["status"] == "UNRESOLVED"


def test_context_b_each_control_requires_alignment_and_slope_pooled_lb_and_three_seeds() -> None:
    for control in _MODULE.CONTROL_ORDER:
        for metric in ("context_b_alignment", "context_b_slope"):
            for field, value in (
                ("pooled_difference", 0.0),
                ("lower_bound", 0.0),
                ("positive_seed_means", 2),
            ):
                case = _panel_aggregates()
                case["controls"][control][metric][field] = value
                assert _MODULE.decide_panel(case, {})["status"] != "PASS"


def test_panel_close_precedence_and_exact_boundaries() -> None:
    case = _panel_aggregates()
    case["primary_pa_alignment"]["pooled_difference"] = 0.0
    case["primary_pa_alignment"]["positive_seed_means"] = 1
    case["controls"][_MODULE.CONTROL_ORDER[0]]["primary_alignment"]["pooled_difference"] = -1.0
    decision = _MODULE.decide_panel(case, {})
    assert decision["status"] == "CLOSE"
    assert decision["first_decisive_clause"] == "close_vs_pa"
    assert decision["authorized_action"] == "stop_close"


def test_panel_middle_region_is_unresolved() -> None:
    case = _panel_aggregates()
    case["primary_pa_alignment"]["pooled_difference"] = 0.01
    decision = _MODULE.decide_panel(case, {})
    assert decision["status"] == "UNRESOLVED"
    assert decision["authorized_action"] == "stop_unresolved"


@pytest.mark.parametrize("fault", (False, float("nan"), float("inf"), None))
def test_any_integrity_schema_or_nonfinite_fault_is_invalid(fault: object) -> None:
    case = _panel_aggregates()
    case["completeness"] = fault
    decision = _MODULE.decide_panel(case, {})
    assert decision["status"] == "INVALID"
    assert decision["authorized_action"] == "stop_invalid"


def _bootstrap_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    operators = _MODULE.OPERATOR_ORDER
    for seed in range(4):
        for label in range(32):
            for context in ("A", "B"):
                records = {
                    operator: {
                        "motion_sha256": "a" * 64,
                        "motion_norm": 1.0,
                        "margin_alignment": float(label + seed + (10 if operator == "rdgc" else 0)),
                        "margin_slope": float(2 * label + seed + (5 if operator == "rdgc" else 0)),
                    }
                    for operator in operators
                }
                rows.append(
                    {
                        "seed": seed,
                        "context": context,
                        "receiver_label": label,
                        "operators": records,
                    }
                )
    return rows


def test_paired_bootstrap_uses_exact_32_labels_four_seeds_and_pcg64_201() -> None:
    result = _MODULE.paired_bootstrap(_bootstrap_rows())
    assert result["bit_generator"] == "PCG64"
    assert result["seed"] == 201
    assert result["replicates"] == 10_000
    assert result["complete_labels"] == list(range(32))
    assert len(result["distributions"]) == 28
    first = result["distributions"][0]
    assert tuple(first) == (
        "context",
        "comparator",
        "metric",
        "values",
        "values_sha256",
        "lower_bound",
    )
    assert len(first["values"]) == 10_000
    values = np.asarray(first["values"], dtype=np.float64)
    assert hashlib.sha256(values.tobytes(order="C")).hexdigest() == first["values_sha256"]
    assert first["lower_bound"] == pytest.approx(float(np.percentile(values, 2.5)))


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "reordered", "partial"))
def test_bootstrap_rejects_unpaired_missing_reordered_or_partial_rows(mutation: str) -> None:
    rows = _bootstrap_rows()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    else:
        rows[-1]["operators"].pop(_MODULE.CONTROL_ORDER[-1])
    with pytest.raises((TypeError, ValueError)):
        _MODULE.paired_bootstrap(rows)


def test_pre_import_environment_is_exact_non_torch_union() -> None:
    fields = {
        "python_executable": ".venv/bin/python",
        "python_version": "3.12.3",
        "numpy_version": "2.5.0",
        "cuda_visible_devices": "0",
        "cublas_workspace_config": ":4096:8",
        "source_commit": "a" * 40,
        "source_files_sha256": "b" * 64,
        "manifest_path": "docs/pass205_rdgc_stage_b_manifest.json",
        "manifest_sha256": "c" * 64,
    }
    value = _MODULE.build_pre_import_environment(fields)
    assert value == {"phase": "pre_import", "pre_import": fields, "torch_runtime": None}
    assert tuple(value) == ("phase", "pre_import", "torch_runtime")


def test_observed_torch_runtime_is_attached_without_fabricated_defaults() -> None:
    fields = {
        "python_executable": ".venv/bin/python",
        "python_version": "3.12.3",
        "numpy_version": "2.5.0",
        "cuda_visible_devices": "0",
        "cublas_workspace_config": ":4096:8",
        "source_commit": "a" * 40,
        "source_files_sha256": "b" * 64,
        "manifest_path": "docs/pass205_rdgc_stage_b_manifest.json",
        "manifest_sha256": "c" * 64,
    }
    pre = _MODULE.build_pre_import_environment(fields)
    fake = types.SimpleNamespace(
        __version__="2.5.1",
        version=types.SimpleNamespace(cuda="12.4"),
        backends=types.SimpleNamespace(
            cudnn=types.SimpleNamespace(version=lambda: 90100, allow_tf32=False),
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        ),
        are_deterministic_algorithms_enabled=lambda: True,
        cuda=types.SimpleNamespace(
            current_device=lambda: 0,
            get_device_name=lambda _index: "synthetic",
            get_device_capability=lambda _index: (9, 0),
        ),
    )
    value = _MODULE.attach_observed_torch_runtime(pre, fake)
    assert value["phase"] == "post_import"
    assert value["torch_runtime"]["torch_version"] == "2.5.1"
    assert value["torch_runtime"]["device_capability"] == [9, 0]


def _bound_rows(count: int = 600) -> tuple[list[str], list[int]]:
    ids: list[str] = []
    labels: list[int] = []
    for label in range(count):
        for row in range(4):
            ids.append(json.dumps({"label": label, "row": row}, separators=(",", ":")))
            labels.append(label)
    return ids, labels


def test_selection_recomputes_old_64_and_freezes_fresh_8_then_32() -> None:
    ids, labels = _bound_rows()
    old = set(range(64))
    value = _MODULE.build_rdgc_selection(ids, labels, old_selection=old)
    assert len(value["preliminary"]["identity_labels"]) == 8
    assert len(value["panel"]["identity_labels"]) == 32
    chosen = value["preliminary"]["identity_labels"] + value["panel"]["identity_labels"]
    assert not old.intersection(chosen)
    assert len(set(chosen)) == 40
    assert (
        value["old_rsta_exclusion_sha256"]
        == hashlib.sha256(b"".join(f"{label}\n".encode() for label in sorted(old))).hexdigest()
    )


def test_selection_bound_ids_are_exact_nonempty_unmodified_strings() -> None:
    ids, labels = _bound_rows()
    value = _MODULE.build_rdgc_selection(ids, labels, old_selection=set(range(64)))
    selected_ids = value["preliminary"]["receiver_ids"] + value["panel"]["receiver_ids"]
    assert all(item in ids for item in selected_ids)
    for mutant in (0, True, ""):
        bad_ids = list(ids)
        bad_ids[300] = mutant  # type: ignore[list-item]
        with pytest.raises((TypeError, ValueError)):
            _MODULE.build_rdgc_selection(bad_ids, labels, old_selection=set(range(64)))


def test_selection_support_receiver_roles_are_disjoint_and_deterministic() -> None:
    ids, labels = _bound_rows()
    first = _MODULE.build_rdgc_selection(ids, labels, old_selection=set(range(64)))
    second = _MODULE.build_rdgc_selection(ids, labels, old_selection=set(range(64)))
    assert first == second
    for phase in ("preliminary", "panel"):
        supports = {item for pair in first[phase]["support_ids_by_label"].values() for item in pair}
        receivers = set(first[phase]["receiver_ids"])
        assert supports.isdisjoint(receivers)


def test_selection_rejects_duplicate_length_and_insufficient_identity_rows() -> None:
    ids, labels = _bound_rows()
    for mutant_ids, mutant_labels in (
        (ids[:-1], labels),
        ([ids[0], *ids[1:-1], ids[0]], labels),
        (ids[: 100 * 4], labels[: 100 * 4]),
    ):
        with pytest.raises((TypeError, ValueError)):
            _MODULE.build_rdgc_selection(mutant_ids, mutant_labels, old_selection=set(range(64)))


def test_nested_masks_are_receiver_plus_exact_prefix_counts() -> None:
    batch_ids = tuple(f"id-{index}" for index in range(180))
    result = _MODULE.nested_contributor_masks(batch_ids, "id-17", seed=2, context="B")
    assert tuple(sum(mask) for mask in result) == (1, 8, 32, 180)
    assert all(mask[17] for mask in result)
    assert all(
        all(not left or right for left, right in zip(result[index], result[index + 1], strict=True))
        for index in range(3)
    )


def test_integrity_prefix_all_four_pass_before_candidate_calls() -> None:
    events: list[tuple[str, int]] = []

    def audit(seed: int, _binding: object) -> dict[str, object]:
        events.append(("audit", seed))
        return {"seed": seed, "passed": True}

    result = _MODULE.run_all_seed_integrity_prefix(
        ({"seed": seed} for seed in range(4)), adapters={"audit_seed": audit}
    )
    assert events == [("audit", seed) for seed in range(4)]
    assert result == [{"seed": seed, "passed": True} for seed in range(4)]


def test_later_seed_integrity_failure_stops_immediately() -> None:
    events: list[int] = []

    def audit(seed: int, _binding: object) -> dict[str, object]:
        events.append(seed)
        return {"seed": seed, "passed": seed != 2}

    with pytest.raises(ValueError, match="seed 2"):
        _MODULE.run_all_seed_integrity_prefix(
            ({"seed": seed} for seed in range(4)), adapters={"audit_seed": audit}
        )
    assert events == [0, 1, 2]


def test_preliminary_schedule_is_one_graph_then_diagonal_and_nested_counts() -> None:
    events: list[tuple[object, ...]] = []

    class Graph:
        def close(self) -> None:
            events.append(("close",))

    def build(seed: int, context: str, _selection: object) -> Graph:
        events.append(("build", seed, context))
        return Graph()

    def receiver(graph: Graph, label: int, count: int | None) -> dict[str, object]:
        events.append(("action", label, count, id(graph)))
        return {"motion_norm": float(label + (count or 0) + 1), "sha256": "a" * 64}

    selection = {"preliminary": {"identity_labels": list(range(8))}}
    rows = _MODULE.run_preliminary(
        selection,
        tuple({"seed": seed} for seed in range(4)),
        adapters={"build_graph": build, "receiver_action": receiver},
    )
    assert len(rows["rows"]) == 64
    assert sum(event[0] == "build" for event in events) == 8
    assert sum(event[0] == "action" for event in events) == 8 * 8 * 5
    assert sum(event[0] == "close" for event in events) == 8
    first_actions = [event[2] for event in events if event[0] == "action"][:5]
    assert first_actions == [None, 1, 8, 32, 180]


def test_preliminary_metrics_exact_row_order_and_scalar_aggregates() -> None:
    rows: list[dict[str, object]] = []
    for seed in range(4):
        for context in ("A", "B"):
            for label in range(8):
                values = {str(count): math.log(1 + count / 10) for count in (1, 8, 32, 180)}
                rows.append(
                    {
                        "seed": seed,
                        "context": context,
                        "receiver_label": label,
                        "log_kappa": 0.1 * label,
                        "absolute_log_gain_errors_by_contributor_count": values,
                        "count_gain": values["180"] - values["8"],
                    }
                )
    result = _MODULE.preliminary_metrics(rows)
    assert tuple(result) == tuple(_preliminary_aggregates())
    assert result["count_gain_median_A"] == pytest.approx(math.log(19) - math.log(1.8))
    bad = deepcopy(rows)
    bad[0], bad[1] = bad[1], bad[0]
    with pytest.raises(ValueError):
        _MODULE.preliminary_metrics(bad)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _authority_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    repository = tmp_path / "authority"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "rdgc@example.invalid")
    _git(repository, "config", "user.name", "RDGC Test")
    for source_path in _MODULE.RDGC_SOURCE_ORDER:
        if source_path == "scripts/diagnose_pass205_rdgc_stage_b.py":
            continue
        path = repository / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {source_path}\n")
    _git(
        repository,
        "add",
        *[
            path
            for path in _MODULE.RDGC_SOURCE_ORDER
            if path != "scripts/diagnose_pass205_rdgc_stage_b.py"
        ],
    )
    _git(repository, "commit", "-qm", "base sources")
    candidate_path = repository / _MODULE.CANDIDATE_PATH
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("candidate\n")
    _git(repository, "add", str(candidate_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "candidate")
    candidate_commit = _git(repository, "rev-parse", "HEAD")
    plan_path = repository / _MODULE.PLAN_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("plan\n")
    _git(repository, "add", str(plan_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "plan")
    plan_commit = _git(repository, "rev-parse", "HEAD")
    for source_path in _MODULE.RDGC_SOURCE_ORDER:
        path = repository / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {source_path}\n")
    test_path = repository / "tests/test_diagnose_pass205_rdgc_stage_b.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("# test\n")
    _git(repository, "add", *(_MODULE.RDGC_SOURCE_ORDER), str(test_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "source")
    source_commit = _git(repository, "rev-parse", "HEAD")
    receipt_path = repository / "reports/validation.json"
    receipt_path.parent.mkdir(parents=True)
    receipt = {
        "status": "VALID",
        "artifact_path": "forbidden-old-result.json",
        "artifact_sha256": "d" * 64,
    }
    receipt_path.write_text(json.dumps(receipt, separators=(",", ":")) + "\n")
    source_files = []
    for source_path in _MODULE.RDGC_SOURCE_ORDER:
        data = (repository / source_path).read_bytes()
        source_files.append(
            {
                "path": source_path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob": _git(repository, "rev-parse", f"HEAD:{source_path}"),
            }
        )
    manifest_path = repository / "docs/pass205_rdgc_stage_b_manifest.json"
    future = _future_manifest()
    manifest = {
        "schema_version": 1,
        "candidate": {
            "path": _MODULE.CANDIDATE_PATH,
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "commit": candidate_commit,
        },
        "implementation_plan": {
            "path": _MODULE.PLAN_PATH,
            "sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            "commit": plan_commit,
        },
        "upstream_rsta": future["upstream_rsta"],
        "literature_audit": future["literature_audit"],
        "validation_receipt": {
            "path": str(receipt_path.relative_to(repository)),
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "status": "VALID",
            "verifier_source_commit": "e" * 40,
            "verifier_handoff_commit": "f" * 40,
            "artifact_path": "forbidden-old-result.json",
            "artifact_sha256": "d" * 64,
        },
        "historical": future["historical"],
        "current_scientific_source": {"git_revision": source_commit, "files": source_files},
        "artifact_schema": future["artifact_schema"],
        "seeds": [0, 1, 2, 3],
    }
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    _git(repository, "add", str(manifest_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "handoff")
    _git(repository, "checkout", "--detach", "-q", "HEAD")
    monkeypatch.setattr(_MODULE, "CANDIDATE_COMMIT", candidate_commit)
    monkeypatch.setattr(
        _MODULE, "CANDIDATE_SHA256", hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(_MODULE, "PLAN_COMMIT", plan_commit)
    monkeypatch.setattr(_MODULE, "PLAN_SHA256", hashlib.sha256(plan_path.read_bytes()).hexdigest())
    return repository, manifest_path, receipt_path


def test_authority_binds_linear_handoff_sources_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest_path, receipt_path = _authority_repository(tmp_path, monkeypatch)
    value = _MODULE.authenticate_authority(repository, manifest_path, receipt_path)
    assert value["source_commit"] == _git(repository, "rev-parse", "HEAD^")
    assert value["handoff_commit"] == _git(repository, "rev-parse", "HEAD")
    assert value["validation_receipt"]["status"] == "VALID"
    assert len(value["files"]) == 33


def test_authority_rejects_dirty_source_digest_and_never_opens_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest_path, receipt_path = _authority_repository(tmp_path, monkeypatch)
    forbidden = repository / "forbidden-old-result.json"
    forbidden.symlink_to("/definitely/not-readable")
    _MODULE.authenticate_authority(repository, manifest_path, receipt_path)
    source = repository / _MODULE.RDGC_SOURCE_ORDER[0]
    source.write_text("dirty\n")
    with pytest.raises(ValueError, match="clean|digest"):
        _MODULE.authenticate_authority(repository, manifest_path, receipt_path)


def test_authenticated_rsta_loader_binds_file_bytes_and_cleans_private_module(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "rdgc@example.invalid")
    _git(tmp_path, "config", "user.name", "RDGC Test")
    source = tmp_path / "helper.py"
    source.write_text("VALUE = 17\n")
    _git(tmp_path, "add", "helper.py")
    _git(tmp_path, "commit", "-qm", "helper")
    descriptor = {
        "path": "helper.py",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "git_blob": _git(tmp_path, "rev-parse", "HEAD:helper.py"),
    }
    module = _MODULE.load_authenticated_rsta_module(tmp_path, descriptor)
    assert module.VALUE == 17
    assert Path(module.__file__).resolve() == source.resolve()
    assert not any(name.startswith("_pass205_authenticated_rsta_") for name in sys.modules)
    source.write_text("VALUE = 18\n")
    with pytest.raises(ValueError, match="digest"):
        _MODULE.load_authenticated_rsta_module(tmp_path, descriptor)


def test_compute_parameter_correction_uses_literal_penalty_and_named_order() -> None:
    theta = _finite_tensor([1.2, -0.4], requires_grad=True)
    b = theta * torch.tensor([2.0, 3.0])
    field = {
        "named_parameters": (("encoder.weight", theta),),
        "p": (_finite_tensor([0.5, -1.0]),),
        "b": b,
        "s": _finite_tensor([1.0, 2.0]),
        "dbar": _finite_tensor([0.3, -0.8]),
        "receiver_fields": [
            {"s": _finite_tensor([1.0 + i, 2.0]), "dbar": _finite_tensor([0.5, 1.0])}
            for i in range(8)
        ],
        "pgn_motion": theta * torch.tensor([1.0, -2.0]),
    }
    expected_theta = theta.detach().clone().requires_grad_(True)
    expected_b = expected_theta * torch.tensor([2.0, 3.0])
    expected = tuple(
        -value
        for value in torch.autograd.grad(
            _MODULE.rdgc_penalty(torch, expected_b, field["s"]), (expected_theta,)
        )
    )
    correction = _MODULE.compute_parameter_correction(field, "rdgc", torch_module=torch)
    assert len(correction) == 1
    assert torch.allclose(correction[0], expected[0])
    assert correction[0].dtype == torch.float32


def test_layerwise_correction_is_direct_and_has_no_candidate_graph() -> None:
    theta = _finite_tensor([3.0, 4.0], requires_grad=True)
    field = {
        "named_parameters": (("block.weight", theta),),
        "p": (_finite_tensor([6.0, 8.0]),),
    }
    value = _MODULE.compute_parameter_correction(field, "layerwise_trust_ratio", torch_module=torch)
    assert torch.allclose(value[0], _finite_tensor([3.0, 4.0]))
    assert not value[0].requires_grad


def test_virtual_jvp_uses_real_fresh_torch_func_and_scalar_evidence() -> None:
    parameters = (_finite_tensor([1.0, -2.0]),)

    def function(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.stack(
            (values[0][0] * values[0][0], values[0][1] + values[0][1] + values[0][1])
        )

    direction = (_finite_tensor([0.5, -1.0]),)
    q = _finite_tensor([2.0, -1.0])
    value = _MODULE.evaluate_virtual_direction(
        {"function": function, "parameters": parameters},
        direction,
        q,
        torch_module=torch,
    )
    expected = _finite_tensor([1.0, -3.0])
    assert value["motion_norm"] == pytest.approx(float(expected.double().norm()))
    assert value["margin_alignment"] == pytest.approx(
        float(torch.nn.functional.cosine_similarity(expected, q, dim=0))
    )
    assert tuple(value) == ("motion_sha256", "motion_norm", "margin_alignment", "margin_slope")


def test_panel_uses_exact_seed_group_receiver_operator_context_order() -> None:
    events: list[tuple[object, ...]] = []

    def correction(seed: int, group: int, receiver: int, operator: str) -> tuple[torch.Tensor, ...]:
        events.append(("correction", seed, group, receiver, operator))
        return (_finite_tensor([float(receiver + 1)]),)

    def evaluate(
        seed: int,
        group: int,
        receiver: int,
        operator: str,
        context: str,
        direction: tuple[torch.Tensor, ...],
    ) -> dict[str, object]:
        events.append(("evaluate", seed, group, receiver, operator, context))
        return {
            "motion_sha256": "a" * 64,
            "motion_norm": float(direction[0].item()),
            "margin_alignment": 0.1,
            "margin_slope": 0.2,
        }

    selection = {"panel": {"identity_labels": list(range(32))}}
    value = _MODULE.run_full_panel(
        selection,
        tuple({"seed": seed} for seed in range(4)),
        {},
        adapters={"build_correction": correction, "evaluate_direction": evaluate},
    )
    assert len(value["rows"]) == 256
    assert sum(event[0] == "correction" for event in events) == 4 * 32 * 8
    assert sum(event[0] == "evaluate" for event in events) == 4 * 32 * 8 * 2
    assert events[:3] == [
        ("correction", 0, 0, 0, "pa"),
        ("evaluate", 0, 0, 0, "pa", "A"),
        ("evaluate", 0, 0, 0, "pa", "B"),
    ]
    assert tuple(value["rows"][0]["operators"]) == _MODULE.OPERATOR_ORDER


def test_panel_fail_fast_does_not_build_later_corrections() -> None:
    events: list[str] = []

    def correction(*_args: object) -> tuple[torch.Tensor, ...]:
        events.append("correction")
        return (_finite_tensor([1.0]),)

    def evaluate(*args: object) -> dict[str, object]:
        events.append("evaluate")
        if events.count("evaluate") == 3:
            raise ValueError("synthetic nonfinite")
        return {
            "motion_sha256": "a" * 64,
            "motion_norm": 1.0,
            "margin_alignment": 0.1,
            "margin_slope": 0.2,
        }

    with pytest.raises(ValueError, match="synthetic"):
        _MODULE.run_full_panel(
            {"panel": {"identity_labels": list(range(32))}},
            tuple({"seed": seed} for seed in range(4)),
            {},
            adapters={"build_correction": correction, "evaluate_direction": evaluate},
        )
    assert events == ["correction", "evaluate", "evaluate", "correction", "evaluate"]


def test_atomic_writer_no_clobber_symlink_or_temporary(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    payload = _reduced_pre_import_invalid()
    _MODULE.write_json_atomic(output, payload)
    assert json.loads(output.read_text()) == payload
    with pytest.raises(FileExistsError):
        _MODULE.write_json_atomic(output, payload)
    assert not (tmp_path / ".result.json.tmp").exists()
    link = tmp_path / "link.json"
    link.symlink_to(output)
    with pytest.raises(FileExistsError):
        _MODULE.write_json_atomic(link, payload)


def test_cli_requires_exact_manifest_output_and_scientific_once() -> None:
    with pytest.raises(SystemExit):
        _MODULE.main(["--manifest", "x", "--output", "y"])
    with pytest.raises(SystemExit):
        _MODULE.main(["--manifest", "x", "--output", "y", "--scientific-once", "--retry"])


def test_real_cpu_torch_func_end_to_end_authenticated_pass200_helper_no_adapters(
    tmp_path: Path,
) -> None:
    helper = _ROOT / "scripts/rsta_normwise_adjoint.py"
    descriptor = {
        "path": str(helper),
        "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "git_blob": _git(_ROOT, "rev-parse", "HEAD:scripts/rsta_normwise_adjoint.py"),
    }
    loaded = _MODULE.load_authenticated_rsta_module(_ROOT, descriptor)
    assert loaded.__file__ == str(helper)
    p = (_finite_tensor([0.8, -0.4]),)

    def fresh_field() -> dict[str, object]:
        theta = _finite_tensor([1.1, -0.7], requires_grad=True)
        return {
            "named_parameters": (("encoder.weight", theta),),
            "p": p,
            "b": torch.stack((theta[0] * theta[0], 2.0 * theta[1])),
            "s": _finite_tensor([1.0, 1.0]),
            "dbar": _finite_tensor([0.2, -0.5]),
            "receiver_fields": [
                {"s": _finite_tensor([1.0 + i, 1.0]), "dbar": _finite_tensor([0.5, 0.25])}
                for i in range(8)
            ],
            "pgn_motion": torch.stack((theta[0], -theta[1])),
        }

    corrections = {
        name: _MODULE.compute_parameter_correction(fresh_field(), name, torch_module=torch)
        for name in _MODULE.CORRECTION_ORDER
    }
    updates = _MODULE.normalize_virtual_updates(torch, p, corrections)
    theta = _finite_tensor([1.1, -0.7])

    def function(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.stack((values[0][0] * values[0][0], values[0][1] + values[0][1]))

    records = {
        context: {
            name: _MODULE.evaluate_virtual_direction(
                {"function": function, "parameters": (theta.detach(),)},
                direction,
                _finite_tensor([1.0, -1.0]),
                torch_module=torch,
            )
            for name, direction in updates.items()
        }
        for context in ("A", "B")
    }
    assert tuple(records) == ("A", "B")
    assert all(tuple(records[context]) == _MODULE.OPERATOR_ORDER for context in ("A", "B"))
    payload = _synthetic_full_payload(records)
    _MODULE.validate_scientific_payload(payload)
    output = tmp_path / "receipt.json"
    _MODULE.write_json_atomic(output, payload)
    assert json.loads(output.read_text())["status"] == "UNRESOLVED"


def test_virtual_jvp_leaves_only_scalars_hashes_and_releases_action() -> None:
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    parameters = (_finite_tensor([1.0, 2.0]),)

    def function(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        output = values[0] * values[0]
        references.append(weakref.ref(output))
        return output

    result = _MODULE.evaluate_virtual_direction(
        {"function": function, "parameters": parameters},
        (_finite_tensor([0.5, -0.25]),),
        _finite_tensor([1.0, -1.0]),
        torch_module=torch,
    )
    gc.collect()
    assert all(reference() is None for reference in references)
    assert all(type(value) in (str, float) for value in result.values())


def _reduced_pre_import_invalid() -> dict[str, object]:
    future = _future_manifest()
    environment = {
        "phase": "pre_import",
        "pre_import": {
            "python_executable": ".venv/bin/python",
            "python_version": "3.12.3",
            "numpy_version": np.__version__,
            "cuda_visible_devices": "0",
            "cublas_workspace_config": ":4096:8",
            "source_commit": "a" * 40,
            "source_files_sha256": "b" * 64,
            "manifest_path": "docs/pass205_rdgc_stage_b_manifest.json",
            "manifest_sha256": "c" * 64,
        },
        "torch_runtime": None,
    }
    reference = {"path": "docs/value.md", "sha256": "d" * 64, "commit": "e" * 40}
    source = {
        "source_commit": "a" * 40,
        "handoff_commit": "f" * 40,
        "handoff_parent": "a" * 40,
        "manifest_path": "docs/pass205_rdgc_stage_b_manifest.json",
        "manifest_sha256": "c" * 64,
        "files": [
            {"path": path, "sha256": "b" * 64, "git_blob": "1" * 40}
            for path in _MODULE.RDGC_SOURCE_ORDER
        ],
        "detached_head": True,
        "clean_tracked_worktree": True,
        "ancestry_passed": True,
    }
    return {
        "schema_version": 1,
        "diagnostic": "pass205_rdgc_stage_b",
        "mode": "scientific_no_training_virtual_update",
        "status": "INVALID",
        "phase_reached": "pre_import",
        "candidate_values_computed": False,
        "training_performed": False,
        "benchmark_authorized": False,
        "scope_limitation": _MODULE.SCOPE_LIMITATION,
        "authority": {
            "candidate": dict(reference),
            "implementation_plan": dict(reference),
            "literature_audit": future["literature_audit"],
        },
        "source": source,
        "execution": {
            "attempt": 1,
            "command": ["python", "script"],
            "cwd": str(_ROOT),
            "pid": 1,
            "python_executable": ".venv/bin/python",
            "python_version": "3.12.3",
            "cuda_visible_devices": "0",
            "cublas_workspace_config": ":4096:8",
            "output_path": "reports/generated/pass205_rdgc_stage_b/result.json",
            "started_utc": "2026-08-10T00:00:00Z",
            "completed_utc": "2026-08-10T00:00:01Z",
            "exit_code": 2,
        },
        "environment": environment,
        "binding": {
            "rsta_candidate": future["upstream_rsta"]["candidate"],
            "rsta_gate2_audit": future["upstream_rsta"]["gate2_audit"],
            "rsta_producer_source_commit": future["upstream_rsta"]["producer_source_commit"],
            "rsta_producer_handoff_commit": future["upstream_rsta"]["producer_handoff_commit"],
            "rsta_artifact": future["upstream_rsta"]["producer_artifact"],
            "rsta_producer_pid": 1002393,
            "rsta_producer_exit_code": 0,
            "verifier_source_commit": future["upstream_rsta"]["verifier_source_commit"],
            "verifier_handoff_commit": future["upstream_rsta"]["verifier_handoff_commit"],
            "verifier_manifest_sha256": future["upstream_rsta"]["verifier_manifest"]["sha256"],
            "validation_receipt": {
                "path": future["validation_receipt"]["path"],
                "sha256": future["validation_receipt"]["sha256"],
                "status": "VALID",
            },
            "rsta_scientific_status": "VALID",
            "rsta_scientific_decision": "UNRESOLVED",
            "rsta_first_decisive_clause": "no_pass_or_fail_rule",
            "seeds": future["historical"]["seeds"],
        },
        "integrity": None,
        "selection": None,
        "preliminary": None,
        "panel": None,
        "bootstrap": None,
        "decision": {
            "close_precedence": True,
            "predicates": [{"name": "structural_invalid", "value": True}],
            "status": "INVALID",
            "first_decisive_clause": "structural_invalid",
            "authorized_action": "stop_invalid",
        },
    }


def test_result_union_rejects_order_phase_null_type_and_nonfinite_mutations() -> None:
    value = _reduced_pre_import_invalid()
    _MODULE.validate_scientific_payload(value)
    mutations: list[dict[str, object]] = []
    reordered = {key: value[key] for key in reversed(value)}
    mutations.append(reordered)
    for key, replacement in (
        ("training_performed", 0),
        ("candidate_values_computed", True),
        ("integrity", {}),
        ("phase_reached", "integrity"),
    ):
        mutant = deepcopy(value)
        mutant[key] = replacement
        mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["environment"]["torch_runtime"] = {}
    mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["decision"]["status"] = "PASS"
    mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["execution"] = {"peak": float("nan")}
    mutations.append(mutant)
    for mutant in mutations:
        with pytest.raises((TypeError, ValueError), match="."):
            _MODULE.validate_scientific_payload(mutant)


def _future_manifest() -> dict[str, object]:
    reference = {"path": "docs/value.md", "sha256": "d" * 64, "commit": "e" * 40}
    artifact = {"path": "artifacts/value.json", "sha256": "f" * 64}
    seeds = [
        {
            "seed": seed,
            "checkpoint": dict(artifact),
            "training_report": dict(artifact),
            "final_pack": dict(artifact),
            "configuration_sha256": "1" * 64,
            "source_export_sha256": "2" * 64,
        }
        for seed in range(4)
    ]
    files = [
        {"path": path, "sha256": "a" * 64, "git_blob": "b" * 40}
        for path in _MODULE.RDGC_SOURCE_ORDER
    ]
    return {
        "schema_version": 1,
        "candidate": dict(reference),
        "implementation_plan": dict(reference),
        "upstream_rsta": {
            "candidate": dict(reference),
            "gate2_audit": dict(reference),
            "producer_source_commit": "1" * 40,
            "producer_handoff_commit": "2" * 40,
            "producer_artifact": dict(artifact),
            "producer_pid": 1002393,
            "producer_exit_code": 0,
            "verifier_source_commit": "3" * 40,
            "verifier_handoff_commit": "4" * 40,
            "verifier_manifest": dict(artifact),
            "scientific_status": "VALID",
            "scientific_decision": "UNRESOLVED",
            "first_decisive_clause": "no_pass_or_fail_rule",
        },
        "literature_audit": {
            "path": "docs/audit.md",
            "sha256": "3" * 64,
            "commit": "5" * 40,
            "verdict": "LIVE-NARROW",
            "reviewed_candidate_sha256": "d" * 64,
            "primary_source_ids": [f"source-{index}" for index in range(14)],
        },
        "validation_receipt": {
            "path": "reports/validation.json",
            "sha256": "4" * 64,
            "status": "VALID",
            "verifier_source_commit": "3" * 40,
            "verifier_handoff_commit": "4" * 40,
            "artifact_path": "forbidden-old-result.json",
            "artifact_sha256": "5" * 64,
        },
        "historical": {
            "manifest_path": "docs/pass200_rsta_receipt_stage_a_manifest.json",
            "manifest_sha256": "6" * 64,
            "seeds": seeds,
        },
        "current_scientific_source": {"git_revision": "c" * 40, "files": files},
        "artifact_schema": {
            "result_path_template": (
                "reports/generated/pass205_rdgc_stage_b/{handoff_commit}-rdgc-stage-b.json"
            ),
            "schema_version": 1,
            "diagnostic": "pass205_rdgc_stage_b",
            "mode": "scientific_no_training_virtual_update",
            "statuses": ["PASS", "CLOSE", "UNRESOLVED", "INVALID"],
            "phases": ["pre_import", "integrity", "preliminary", "full_panel"],
            "top_level_keys": list(_MODULE.RESULT_KEYS),
            "pre_import_invalid_null_fields": [
                "integrity",
                "selection",
                "preliminary",
                "panel",
                "bootstrap",
            ],
            "post_import_invalid_null_fields": ["selection", "preliminary", "panel", "bootstrap"],
            "operator_order": list(_MODULE.OPERATOR_ORDER),
            "contributor_counts": list(_MODULE.CONTRIBUTOR_COUNTS),
        },
        "seeds": [0, 1, 2, 3],
    }


def _synthetic_full_payload(
    records: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    ids, labels = _bound_rows()
    selection = _MODULE.build_rdgc_selection(ids, labels, old_selection=set(range(64)))
    preliminary_labels = selection["preliminary"]["identity_labels"]
    preliminary_rows: list[dict[str, object]] = []
    for seed in range(4):
        for context in ("A", "B"):
            for receiver_label in preliminary_labels:
                errors = {
                    key: float(index + 1) for index, key in enumerate(("1", "8", "32", "180"))
                }
                preliminary_rows.append(
                    {
                        "seed": seed,
                        "context": context,
                        "receiver_label": receiver_label,
                        "receiver_id": selection["preliminary"]["receiver_ids"][
                            preliminary_labels.index(receiver_label)
                        ],
                        "dbar_norm": 1.0,
                        "self_norm": 2.0,
                        "kappa": 2.0,
                        "log_kappa": float(math.log(2.0)),
                        "b_norms_by_contributor_count": dict(errors),
                        "absolute_log_gain_errors_by_contributor_count": dict(errors),
                        "count_gain": 2.0,
                    }
                )
    preliminary = {
        "operator_counts": {
            "forwards_per_seed_context": 1,
            "dbars_per_seed_context": 1,
            "diagonal_vjps_per_seed_context": 8,
            "diagonal_jvps_per_seed_context": 8,
            "masked_vjps_per_seed_context": 32,
            "masked_receiver_jvps_per_seed_context": 32,
        },
        "rows": preliminary_rows,
        "seed_context_aggregates": [
            {"seed": seed, "context": context} for seed in range(4) for context in ("A", "B")
        ],
        "seed_correlations": [{"seed": seed, "spearman": 1.0} for seed in range(4)],
        "pooled_aggregates": _preliminary_aggregates(),
        "predicates": {
            "survives_count_gain": True,
            "survives_context_stability": True,
            "survives_receiver_heterogeneity": True,
            "survives_global_scalar": True,
            "survives_full_gain": True,
            "close_count_gain": False,
            "close_context_stability": False,
            "close_receiver_heterogeneity": False,
            "close_global_scalar": False,
            "close_full_gain": False,
        },
        "decision": {
            "status": "SURVIVES",
            "first_decisive_clause": "all_survival_predicates",
            "full_panel_authorized": True,
        },
    }
    panel_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    panel_labels = selection["panel"]["identity_labels"]
    for seed in range(4):
        for receiver_index, receiver_label in enumerate(panel_labels):
            for context in ("A", "B"):
                operators = {name: dict(records[context][name]) for name in _MODULE.OPERATOR_ORDER}
                panel_rows.append(
                    {
                        "seed": seed,
                        "group": receiver_index // 8,
                        "receiver_label": receiver_label,
                        "receiver_id": selection["panel"]["receiver_ids"][receiver_index],
                        "context": context,
                        "batch_sha256": "1" * 64,
                        "parameter_names_sha256": "2" * 64,
                        "p_norm": 1.0,
                        "corrections": {
                            name: {
                                "direction_sha256": "3" * 64,
                                "correction_norm": 1.0,
                                "matched_update_norm": 1.0,
                                "rdgc_correction_cosine": 0.5,
                            }
                            for name in _MODULE.CORRECTION_ORDER
                        },
                        "operators": operators,
                    }
                )
                bootstrap_rows.append(
                    {
                        "seed": seed,
                        "context": context,
                        "receiver_label": receiver_label,
                        "operators": operators,
                    }
                )
    bootstrap = _MODULE.paired_bootstrap(bootstrap_rows)
    seed_context_aggregates = [
        {
            "seed": seed,
            "context": context,
            "operator": operator,
            "alignment_mean": 0.1,
            "slope_mean": 0.2,
            "rdgc_minus_operator_alignment_mean": 0.0,
            "rdgc_minus_operator_slope_mean": 0.0,
        }
        for seed in range(4)
        for context in ("A", "B")
        for operator in _MODULE.OPERATOR_ORDER
    ]
    pooled_aggregates = [
        {
            "context": context,
            "operator": operator,
            "alignment_mean": 0.1,
            "slope_mean": 0.2,
            "rdgc_minus_operator_alignment_mean": 0.0,
            "rdgc_minus_operator_slope_mean": 0.0,
            "positive_alignment_seed_means": 0,
            "positive_slope_seed_means": 0,
        }
        for context in ("A", "B")
        for operator in _MODULE.OPERATOR_ORDER
    ]
    aliases = [
        {
            "control": control,
            "pooled_median_absolute_cosine": 0.5,
            "seed_medians_ge_point_nine_nine": 0,
        }
        for control in _MODULE.CONTROL_ORDER
    ]
    panel = {
        "operator_order": list(_MODULE.OPERATOR_ORDER),
        "rows": panel_rows,
        "seed_context_aggregates": seed_context_aggregates,
        "pooled_aggregates": pooled_aggregates,
        "correction_alias_aggregates": aliases,
        "predicates": {
            "primary_vs_pa_alignment": False,
            "primary_vs_pa_slope": False,
            "primary_vs_all_controls": False,
            "context_b_vs_pa": False,
            "context_b_vs_all_controls_alignment_and_slope": False,
            "correction_nonalias": True,
            "completeness": True,
            "close_vs_pa": False,
            "close_primary_control": False,
            "close_control_slope": False,
            "close_correction_alias": False,
        },
    }
    payload = _reduced_pre_import_invalid()
    payload.update(
        {
            "status": "UNRESOLVED",
            "phase_reached": "full_panel",
            "candidate_values_computed": True,
            "integrity": {},
            "selection": selection,
            "preliminary": preliminary,
            "panel": panel,
            "bootstrap": bootstrap,
        }
    )
    payload["environment"]["phase"] = "post_import"
    payload["environment"]["torch_runtime"] = {
        "torch_version": torch.__version__,
        "cuda_runtime_version": "synthetic",
        "cudnn_version": 1,
        "device_index": 0,
        "device_name": "synthetic",
        "device_capability": [9, 0],
        "deterministic_algorithms": True,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
    }
    payload["integrity"] = {
        "seeds": [
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
                "action_hashes_sha256": "7" * 64,
                "passed": True,
            }
            for seed in range(4)
        ],
        "all_four_passed": True,
        "candidate_calls_before_all_four": 0,
        "candidate_state_before_all_four": False,
    }
    payload["decision"].update(
        {
            "status": "UNRESOLVED",
            "first_decisive_clause": "no_close_or_pass_rule",
            "authorized_action": "stop_unresolved",
        }
    )
    return payload


def test_future_manifest_validator_freezes_source_and_projection_order() -> None:
    value = _future_manifest()
    _MODULE.validate_future_manifest(value)
    for path, replacement in (
        (("seeds",), [0, 1, 3, 2]),
        (
            ("current_scientific_source", "files"),
            list(reversed(value["current_scientific_source"]["files"])),
        ),
        (("artifact_schema", "operator_order"), list(reversed(_MODULE.OPERATOR_ORDER))),
        (("artifact_schema", "top_level_keys"), list(reversed(_MODULE.RESULT_KEYS))),
    ):
        mutant = deepcopy(value)
        target = mutant
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        with pytest.raises(ValueError):
            _MODULE.validate_future_manifest(mutant)
