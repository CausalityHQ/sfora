from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rsta_normwise_adjoint as normwise

_CALIBRATION_CLI_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/calibrate_pass200_rsta_normwise_adjoint.py"
)


def _load_calibration_cli() -> Any:
    name = f"calibrate_pass200_rsta_normwise_adjoint_test_{os.urandom(4).hex()}"
    spec = importlib.util.spec_from_file_location(name, _CALIBRATION_CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PROTOCOL_CORRECT_METADATA = json.loads(r"""{
  "zero_corner":{"fixture_id":"zero_corner","kind":"zero_linear","seeds":{"x":"0x4e4f524d00000001","output_direction":"0x4e4f524d00000002","parameter_direction":"0x4e4f524d00000003"},"dimensions":{"input":17,"output":17},"scales":{"operator":"0"}},
  "affine_scale_2m12":{"fixture_id":"affine_scale_2m12","kind":"affine_linear","seeds":{"matrix":"0x4e4f524d00000101","input":"0x4e4f524d00000102","output_direction":"0x4e4f524d00000103","parameter_direction":"0x4e4f524d00000104"},"dimensions":{"input":193,"output":257},"scales":{"operator":"2**-12"}},
  "affine_scale_1":{"fixture_id":"affine_scale_1","kind":"affine_linear","seeds":{"matrix":"0x4e4f524d00000101","input":"0x4e4f524d00000102","output_direction":"0x4e4f524d00000103","parameter_direction":"0x4e4f524d00000104"},"dimensions":{"input":193,"output":257},"scales":{"operator":"1"}},
  "affine_scale_2p12":{"fixture_id":"affine_scale_2p12","kind":"affine_linear","seeds":{"matrix":"0x4e4f524d00000101","input":"0x4e4f524d00000102","output_direction":"0x4e4f524d00000103","parameter_direction":"0x4e4f524d00000104"},"dimensions":{"input":193,"output":257},"scales":{"operator":"2**12"}},
  "smooth_parameter_tree":{"fixture_id":"smooth_parameter_tree","kind":"smooth_parameter_tree","seeds":{"w1":"0x4e4f524d00000201","b1":"0x4e4f524d00000202","w2":"0x4e4f524d00000203","b2":"0x4e4f524d00000204","input":"0x4e4f524d00000205","parameter_direction":"0x4e4f524d00000206","output_direction":"0x4e4f524d00000207"},"dimensions":{"batch":17,"input":11,"hidden":23,"output":19,"parameter_shapes":{"w1":[23,11],"b1":[23],"w2":[19,23],"b2":[19]}},"scales":{"weight":"2**-3","bias":"2**-4","input":"2**-2","normalization_eps":"1e-12"}},
  "paired_cancellation":{"fixture_id":"paired_cancellation","kind":"paired_cancellation_linear","seeds":{"input":"0x4e4f524d00000301","output_pair_base":"0x4e4f524d00000302","parameter_pair_base":"0x4e4f524d00000303"},"dimensions":{"input":8193,"output":8193,"pairs":4096},"scales":{"positive_pair":"2**10","negative_pair":"-2**10","final":"2**-10"}}
}""")

_PROTOCOL_FAULT_METADATA = json.loads(r"""{
  "zero_map_forward_injection":{"fixture_id":"zero_map_forward_injection","kind":"injected_forward_action","seeds":{"x":"0x4e4f524d00000001","output_direction":"0x4e4f524d00000002","parameter_direction":"0x4e4f524d00000003"},"dimensions":{"input":17,"output":17},"scales":{"operator":"0","forward_injection":"2**-10"}},
  "identity_reverse_scale_fault":{"fixture_id":"identity_reverse_scale_fault","kind":"injected_reverse_scale","seeds":{"shared_direction":"0x4e4f524d00000401"},"dimensions":{"input":4096,"output":4096},"scales":{"operator":"1","reverse_action":"255/256"}},
  "identity_reverse_pair_sign_fault":{"fixture_id":"identity_reverse_pair_sign_fault","kind":"injected_reverse_pair_sign","seeds":{"pair_base":"0x4e4f524d00000402"},"dimensions":{"input":4096,"output":4096,"pairs":2048},"scales":{"operator":"1","forward_pair":"[1,1]","reverse_pair":"[1,-1]"}}
}""")


def _protocol_literal_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return list(actual) == list(expected) and all(
            _protocol_literal_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _protocol_literal_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


@pytest.mark.parametrize("container", ["seeds", "dimensions", "parameter_shapes", "scales"])
def test_protocol_metadata_oracle_detects_coordinated_nested_reorder(
    container: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    literal = _PROTOCOL_CORRECT_METADATA["smooth_parameter_tree"]
    kind, seeds, dimensions, scales = deepcopy(normwise._FIXTURE_METADATA["smooth_parameter_tree"])
    containers = {"seeds": seeds, "dimensions": dimensions, "scales": scales}
    if container == "parameter_shapes":
        parent = dimensions
        parent[container] = dict(reversed(tuple(parent[container].items())))
    else:
        target = containers[container]
        containers[container] = dict(reversed(tuple(target.items())))
        seeds, dimensions, scales = (
            containers["seeds"],
            containers["dimensions"],
            containers["scales"],
        )
    monkeypatch.setitem(
        normwise._FIXTURE_METADATA,
        "smooth_parameter_tree",
        (kind, seeds, dimensions, scales),
    )
    coordinated = normwise.correct_fixture_specs()[4].metadata
    assert not _protocol_literal_equal(coordinated, literal)


def _reference(
    u: torch.Tensor,
    a: torch.Tensor,
    v: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    names: tuple[str, ...],
) -> dict[str, float]:
    ud, ad = u.numpy().astype(np.float64), a.numpy().astype(np.float64)
    lhs = float(np.sum(ud * ad, dtype=np.float64))
    rhs_terms = [
        np.sum(
            v[name].numpy().astype(np.float64) * b[name].numpy().astype(np.float64),
            dtype=np.float64,
        )
        for name in names
    ]
    rhs = float(np.sum(np.asarray(rhs_terms, dtype=np.float64), dtype=np.float64))
    un = float(np.sqrt(np.sum(ud * ud, dtype=np.float64)))
    an = float(np.sqrt(np.sum(ad * ad, dtype=np.float64)))
    vn = float(
        np.sqrt(
            np.sum(
                np.asarray(
                    [
                        np.sum(v[name].numpy().astype(np.float64) ** 2, dtype=np.float64)
                        for name in names
                    ]
                ),
                dtype=np.float64,
            )
        )
    )
    bn = float(
        np.sqrt(
            np.sum(
                np.asarray(
                    [
                        np.sum(b[name].numpy().astype(np.float64) ** 2, dtype=np.float64)
                        for name in names
                    ]
                ),
                dtype=np.float64,
            )
        )
    )
    error = abs(lhs - rhs)
    denominator = un * an + vn * bn
    eta = 0.0 if denominator == 0.0 and error == 0.0 else error / denominator
    return {
        "lhs": lhs,
        "rhs": rhs,
        "absolute_error": error,
        "legacy_denominator": max(abs(lhs), abs(rhs), np.float64(1.0e-12)),
        "output_direction_l2": un,
        "parameter_direction_l2": vn,
        "jvp_l2": an,
        "vjp_l2": bn,
        "normwise_denominator": denominator,
        "eta_norm": eta,
        "beta_norm": 2.0 * eta,
        "lhs_absolute_product_sum": float(np.sum(np.abs(ud * ad), dtype=np.float64)),
        "rhs_absolute_product_sum": float(
            np.sum(
                np.asarray(
                    [
                        np.sum(
                            np.abs(
                                v[name].numpy().astype(np.float64)
                                * b[name].numpy().astype(np.float64)
                            ),
                            dtype=np.float64,
                        )
                        for name in names
                    ]
                ),
                dtype=np.float64,
            )
        ),
    }


def test_normwise_metrics_cast_every_factor_before_product_and_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    u = torch.tensor([1.000000119], dtype=torch.float32)
    a = torch.tensor([0.99999994], dtype=torch.float32)
    names = ("residual", "large_positive", "large_negative")
    v = {
        "residual": torch.tensor([1.000000119], dtype=torch.float32),
        "large_positive": torch.tensor([100000008.0], dtype=torch.float32),
        "large_negative": torch.tensor([100000008.0], dtype=torch.float32),
    }
    b = {
        "residual": torch.tensor([0.99999994], dtype=torch.float32),
        "large_positive": torch.tensor([99999992.0], dtype=torch.float32),
        "large_negative": torch.tensor([-99999992.0], dtype=torch.float32),
    }
    expected = _reference(u, a, v, b, names)
    fp32_lhs = float((u * a).sum(dtype=torch.float32).double())
    fp32_rhs = float(
        torch.stack([(v[name] * b[name]).sum(dtype=torch.float32) for name in names])
        .sum(dtype=torch.float32)
        .double()
    )
    assert (fp32_lhs, fp32_rhs) != (expected["lhs"], expected["rhs"])
    rhs_terms = [
        float((v[name].double() * b[name].double()).sum(dtype=torch.float64)) for name in names
    ]
    forward_order = (rhs_terms[0] + rhs_terms[1]) + rhs_terms[2]
    reverse_order = (rhs_terms[1] + rhs_terms[2]) + rhs_terms[0]
    assert forward_order != reverse_order
    original_sum = torch.Tensor.sum

    def guarded_sum(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.dtype == torch.float32:
            raise AssertionError("FP32 reduction occurred before cast")
        return original_sum(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "sum", guarded_sum)
    monkeypatch.setattr(
        torch.Tensor,
        "float",
        lambda _self: (_ for _ in ()).throw(AssertionError("forbidden float conversion")),
    )
    actual = normwise.normwise_adjoint_metrics(u, a, v, b, names)
    for name, value in expected.items():
        assert actual[name] == value
    assert (
        actual["legacy_relative_error"]
        == expected["absolute_error"] / expected["legacy_denominator"]
    )
    assert actual["threshold"] == 5.0e-4


def test_normwise_zero_zero_corner_is_exact_zero() -> None:
    z = torch.zeros(4, dtype=torch.float32)
    result = normwise.normwise_adjoint_metrics(z, z, {"p": z}, {"p": z}, ("p",))
    assert result["normwise_denominator"] == 0.0
    assert result["eta_norm"] == result["beta_norm"] == 0.0
    assert result["lhs_cancellation_factor"] == result["rhs_cancellation_factor"] == 1.0
    assert result["passed"] is True


def test_normwise_zero_positive_corner_is_infinity_and_fails() -> None:
    one = torch.ones(1, dtype=torch.float32)
    zero = torch.zeros(1, dtype=torch.float32)
    result = normwise.normwise_adjoint_metrics(one, one, {"p": zero}, {"p": zero}, ("p",))
    assert result["normwise_denominator"] == 1.0
    assert result["rhs_cancellation_factor"] == 1.0
    value = _valid_result()
    entry = value["correct_fixtures"][normwise.CORRECT_FIXTURE_IDS[0]]
    entry.update(
        {
            "lhs": 1.0,
            "rhs": 0.0,
            "absolute_error": 1.0,
            "legacy_denominator": 1.0,
            "legacy_relative_error": 1.0,
            "output_direction_l2": 0.0,
            "parameter_direction_l2": 0.0,
            "jvp_l2": 0.0,
            "vjp_l2": 0.0,
            "normwise_denominator": 0.0,
            "eta_norm": "infinity",
            "beta_norm": "infinity",
            "lhs_absolute_product_sum": 1.0,
            "rhs_absolute_product_sum": 0.0,
            "lhs_cancellation_factor": 1.0,
            "rhs_cancellation_factor": 1.0,
            "passed": False,
        }
    )
    value["all_passed"] = False
    normwise.validate_calibration_result(value)


def test_cancellation_factor_corner_contract() -> None:
    q = torch.tensor([1.0, 1.0], dtype=torch.float32)
    signed = torch.tensor([1.0, -1.0], dtype=torch.float32)
    result = normwise.normwise_adjoint_metrics(q, signed, {"p": q}, {"p": signed}, ("p",))
    assert result["lhs"] == result["rhs"] == 0.0
    assert result["lhs_cancellation_factor"] == "infinity"
    assert result["rhs_cancellation_factor"] == "infinity"


def test_action_hashes_cover_actual_c_contiguous_fp32_bytes_in_named_order() -> None:
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32).t()
    tree = {"b": torch.tensor([5.0], dtype=torch.float32), "a": tensor}
    expected_tensor = hashlib.sha256(np.ascontiguousarray(tensor.numpy()).tobytes()).hexdigest()
    expected_tree = hashlib.sha256(
        np.ascontiguousarray(tree["a"].numpy()).tobytes()
        + np.ascontiguousarray(tree["b"].numpy()).tobytes()
    ).hexdigest()
    assert normwise.tensor_sha256(tensor) == expected_tensor
    assert normwise.parameter_tree_sha256(tree, ("a", "b")) == expected_tree


def _metric_entry() -> dict[str, Any]:
    one = torch.ones(1, dtype=torch.float32)
    metrics = normwise.normwise_adjoint_metrics(one, one, {"p": one}, {"p": one}, ("p",))
    threshold = metrics.pop("threshold")
    passed = metrics.pop("passed")
    return {
        "fixture_id": "x",
        "kind": "x",
        "seeds": {},
        "dimensions": {"input": 1},
        "scales": {"operator": "0"},
        **metrics,
        "jvp_sha256": normwise.tensor_sha256(one),
        "vjp_sha256": normwise.parameter_tree_sha256({"p": one}, ("p",)),
        "controls": {},
        "threshold": threshold,
        "passed": passed,
    }


def _valid_result() -> dict[str, Any]:
    entry = _metric_entry()
    fault_controls = {
        "unmodified": {
            "jvp_sha256": "3" * 64,
            "vjp_sha256": "4" * 64,
            "beta_norm": 0.0,
            "passed": True,
        }
    }
    correct = {}
    for name in normwise.CORRECT_FIXTURE_IDS:
        item = deepcopy(entry)
        item.update(normwise.fixture_metadata(name))
        item = {
            **normwise.fixture_metadata(name),
            **{
                key: value
                for key, value in item.items()
                if key not in ("fixture_id", "kind", "seeds", "dimensions", "scales")
            },
        }
        item["controls"] = {
            control_name: {
                "jvp_sha256": item["jvp_sha256"],
                "vjp_sha256": item["vjp_sha256"],
                "beta_norm": 0.0,
                (
                    "exact_action_hash_match"
                    if control_name in ("rebuild", "reversed_action_order")
                    else "exact_relation"
                ): True,
                "passed": True,
            }
            for control_name in (
                "rebuild",
                "reversed_action_order",
                "parameter_sign",
                "output_sign",
            )
        }
        correct[name] = item
    faults = {}
    for name in normwise.REGISTERED_FAULT_IDS:
        item = deepcopy(entry)
        item.update(normwise.fixture_metadata(name))
        item = {
            **normwise.fixture_metadata(name),
            **{
                key: value
                for key, value in item.items()
                if key not in ("fixture_id", "kind", "seeds", "dimensions", "scales")
            },
        }
        item["controls"] = deepcopy(fault_controls)
        item.update(
            {
                "lhs": 1.0,
                "rhs": 0.0,
                "absolute_error": 1.0,
                "legacy_denominator": 1.0,
                "legacy_relative_error": 1.0,
                "output_direction_l2": 1.0,
                "parameter_direction_l2": 1.0,
                "jvp_l2": 1.0,
                "vjp_l2": 1.0,
                "normwise_denominator": 2.0,
                "eta_norm": 0.5,
                "beta_norm": 1.0,
                "lhs_absolute_product_sum": 1.0,
                "rhs_absolute_product_sum": 0.0,
                "lhs_cancellation_factor": 1.0,
                "rhs_cancellation_factor": 1.0,
                "passed": True,
            }
        )
        faults[name] = item
    return {
        "schema_version": 1,
        "diagnostic": "pass200-rsta-normwise-adjoint-calibration",
        "mode": "cpu_synthetic_calibration",
        "candidate_values_computed": False,
        "stage_a_verdict": "NOT_COMPUTED",
        "uses_test_data": "synthetic_only",
        "protocol": {
            "path": "docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md",
            "sha256": "a" * 64,
            "commit": "b" * 40,
        },
        "execution_audit": {
            "executing_git_commit": "c" * 40,
            "calibration_source_commit": "d" * 40,
            "calibration_cli_path": "scripts/calibrate_pass200_rsta_normwise_adjoint.py",
            "calibration_cli_sha256": "1" * 64,
        },
        "source": {
            "git_revision": "d" * 40,
            "files": {
                "scripts/rsta_normwise_adjoint.py": "f" * 64,
                "scripts/calibrate_pass200_rsta_normwise_adjoint.py": "1" * 64,
                "tests/test_rsta_normwise_adjoint.py": "2" * 64,
            },
        },
        "environment": {
            "device": "cpu",
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "deterministic_algorithms": True,
            "autocast": False,
            "model_dtype": "torch.float32",
            "reduction_dtype": "torch.float64",
            "python_version": "3",
            "torch_version": "t",
            "numpy_version": "n",
        },
        "correct_fixtures": correct,
        "registered_faults": faults,
        "all_passed": True,
    }


@pytest.mark.parametrize(
    ("control_name", "hash_name"),
    (
        ("rebuild", "jvp_sha256"),
        ("rebuild", "vjp_sha256"),
        ("reversed_action_order", "jvp_sha256"),
        ("reversed_action_order", "vjp_sha256"),
        ("parameter_sign", "vjp_sha256"),
        ("output_sign", "jvp_sha256"),
    ),
)
def test_calibration_schema_rejects_valid_hex_control_hash_not_bound_to_baseline(
    control_name: str, hash_name: str
) -> None:
    value = _valid_result()
    entry = value["correct_fixtures"]["zero_corner"]
    assert entry["controls"][control_name][hash_name] == entry[hash_name]
    entry["controls"][control_name][hash_name] = "f" * 64

    with pytest.raises(ValueError, match="control.*hash"):
        normwise.validate_calibration_result(value)


def test_calibration_schema_rejects_every_recursive_mutation() -> None:
    value = _valid_result()
    normwise.validate_calibration_result(value)
    for mutation in (
        "missing",
        "extra",
        "fixed",
        "nan",
        "derived",
        "hash",
        "passed",
        "fixture_order",
        "metadata",
        "control",
    ):
        changed = deepcopy(value)
        if mutation == "missing":
            changed.pop("mode")
        elif mutation == "extra":
            changed["extra"] = None
        elif mutation == "fixed":
            changed["candidate_values_computed"] = 0
        elif mutation == "nan":
            changed["correct_fixtures"][normwise.CORRECT_FIXTURE_IDS[0]]["lhs"] = float("nan")
        elif mutation == "derived":
            changed["correct_fixtures"][normwise.CORRECT_FIXTURE_IDS[0]]["beta_norm"] = 1.0
        elif mutation == "hash":
            changed["correct_fixtures"][normwise.CORRECT_FIXTURE_IDS[0]]["jvp_sha256"] = "z" * 64
        elif mutation == "passed":
            changed["all_passed"] = False
        elif mutation == "fixture_order":
            changed["correct_fixtures"] = dict(reversed(tuple(changed["correct_fixtures"].items())))
        elif mutation == "metadata":
            changed["correct_fixtures"][normwise.CORRECT_FIXTURE_IDS[0]]["scales"]["operator"] = (
                "0.0"
            )
        else:
            changed["correct_fixtures"][normwise.CORRECT_FIXTURE_IDS[0]]["controls"]["extra"] = {}
        with pytest.raises((TypeError, ValueError)):
            normwise.validate_calibration_result(changed)

    encoded = json.dumps(value, allow_nan=False)
    assert json.loads(encoded) == value


def test_calibration_schema_rejects_impossible_scalar_and_control_payloads() -> None:
    base = _valid_result()
    fixture_id = normwise.CORRECT_FIXTURE_IDS[0]
    scalar_names = (
        "lhs",
        "rhs",
        "absolute_error",
        "legacy_denominator",
        "legacy_relative_error",
        "output_direction_l2",
        "parameter_direction_l2",
        "jvp_l2",
        "vjp_l2",
        "normwise_denominator",
        "eta_norm",
        "beta_norm",
        "lhs_absolute_product_sum",
        "rhs_absolute_product_sum",
        "lhs_cancellation_factor",
        "rhs_cancellation_factor",
        "threshold",
    )
    for name in scalar_names:
        changed = deepcopy(base)
        changed["correct_fixtures"][fixture_id][name] = 0
        with pytest.raises(ValueError):
            normwise.validate_calibration_result(changed)

    for name in (
        "absolute_error",
        "legacy_denominator",
        "legacy_relative_error",
        "output_direction_l2",
        "parameter_direction_l2",
        "jvp_l2",
        "vjp_l2",
        "normwise_denominator",
        "eta_norm",
        "beta_norm",
        "lhs_absolute_product_sum",
        "rhs_absolute_product_sum",
        "lhs_cancellation_factor",
        "rhs_cancellation_factor",
    ):
        changed = deepcopy(base)
        changed["correct_fixtures"][fixture_id][name] = -1.0
        with pytest.raises(ValueError):
            normwise.validate_calibration_result(changed)

    for control_name in ("rebuild", "reversed_action_order", "parameter_sign", "output_sign"):
        changed = deepcopy(base)
        control = changed["correct_fixtures"][fixture_id]["controls"][control_name]
        control["beta_norm"] = -1.0
        control["passed"] = True
        with pytest.raises(ValueError):
            normwise.validate_calibration_result(changed)

    reviewer_example = deepcopy(base)
    entry = reviewer_example["correct_fixtures"][fixture_id]
    entry.update(
        {
            "output_direction_l2": -1.0,
            "parameter_direction_l2": -1.0,
            "jvp_l2": -1.0,
            "vjp_l2": -1.0,
            "normwise_denominator": 2.0,
        }
    )
    with pytest.raises(ValueError):
        normwise.validate_calibration_result(reviewer_example)


def test_calibration_schema_exhaustively_rejects_recursive_mutations() -> None:
    base = _valid_result()
    dict_paths: set[tuple[str, ...]] = set()
    list_paths: set[tuple[str, ...]] = set()
    leaf_paths: set[tuple[str, ...]] = set()
    numeric_paths: set[tuple[str, ...]] = set()

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            dict_paths.add(path)
            for key, item in value.items():
                walk(item, (*path, key))
        elif isinstance(value, list):
            list_paths.add(path)
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))
        else:
            leaf_paths.add(path)
            if type(value) in (int, float):
                numeric_paths.add(path)

    walk(base)

    def parent_at(value: Any, path: tuple[str, ...]) -> Any:
        cursor = value
        for part in path:
            cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
        return cursor

    def reject(changed: dict[str, Any], label: tuple[str, ...]) -> None:
        try:
            normwise.validate_calibration_result(changed)
        except (TypeError, ValueError):
            covered.add(label)
            return
        pytest.fail(f"validator accepted isolated mutation: {label}")

    covered: set[tuple[str, ...]] = set()
    expected: set[tuple[str, ...]] = set()

    for path in dict_paths:
        mapping = parent_at(base, path)
        for key in mapping:
            label = ("remove", *path, key)
            expected.add(label)
            changed = deepcopy(base)
            del parent_at(changed, path)[key]
            reject(changed, label)
        label = ("extra", *path)
        expected.add(label)
        changed = deepcopy(base)
        parent_at(changed, path)["unexpected_schema_key"] = None
        reject(changed, label)
        if len(mapping) > 1:
            label = ("order", *path)
            expected.add(label)
            changed = deepcopy(base)
            target = parent_at(changed, path)
            reversed_items = tuple(reversed(tuple(target.items())))
            target.clear()
            target.update(reversed_items)
            reject(changed, label)

    for path in list_paths:
        label = ("list_type", *path)
        expected.add(label)
        changed = deepcopy(base)
        parent = parent_at(changed, path[:-1])
        key = path[-1]
        current = parent[int(key)] if isinstance(parent, list) else parent[key]
        if isinstance(parent, list):
            parent[int(key)] = tuple(current)
        else:
            parent[key] = tuple(current)
        reject(changed, label)
        if len(current) > 1:
            label = ("list_order", *path)
            expected.add(label)
            changed = deepcopy(base)
            parent = parent_at(changed, path[:-1])
            if isinstance(parent, list):
                parent[int(path[-1])] = list(reversed(current))
            else:
                parent[path[-1]] = list(reversed(current))
            reject(changed, label)

    for path in leaf_paths:
        label = ("leaf", *path)
        expected.add(label)
        changed = deepcopy(base)
        parent = parent_at(changed, path[:-1])
        key = path[-1]
        old = parent[int(key)] if isinstance(parent, list) else parent[key]
        if type(old) is bool:
            replacement: Any = not old
        elif type(old) is int:
            replacement = old + 1
        elif type(old) is float:
            replacement = old + 0.125
        elif type(old) is str:
            replacement = ""
        else:
            replacement = object()
        if isinstance(parent, list):
            parent[int(key)] = replacement
        else:
            parent[key] = replacement
        reject(changed, label)

    for path in numeric_paths:
        for nonfinite_name, nonfinite in (
            ("nan", float("nan")),
            ("posinf", float("inf")),
            ("neginf", -float("inf")),
        ):
            label = (nonfinite_name, *path)
            expected.add(label)
            changed = deepcopy(base)
            parent = parent_at(changed, path[:-1])
            key = path[-1]
            if isinstance(parent, list):
                parent[int(key)] = nonfinite
            else:
                parent[key] = nonfinite
            reject(changed, label)

    relational = deepcopy(base)
    relational["execution_audit"]["calibration_cli_sha256"] = "9" * 64
    label = ("relational", "execution_audit", "calibration_cli_sha256")
    expected.add(label)
    reject(relational, label)

    assert covered == expected
    assert {tuple(path) for kind, *path in covered if kind == "leaf"} == leaf_paths
    assert {tuple(path) for kind, *path in covered if kind == "remove"} >= {
        (*path, key) for path in dict_paths for key in parent_at(base, path)
    }


def _pcg(seed: str, shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
    values = np.random.Generator(np.random.PCG64(int(seed, 16))).standard_normal(shape)
    return torch.tensor(np.ascontiguousarray(values * scale), dtype=torch.float32)


def test_correct_fixture_construction_is_byte_exact() -> None:
    specs = normwise.correct_fixture_specs()
    assert tuple(spec.fixture_id for spec in specs) == tuple(_PROTOCOL_CORRECT_METADATA)
    for spec in specs:
        literal = _PROTOCOL_CORRECT_METADATA[spec.fixture_id]
        assert _protocol_literal_equal(spec.metadata, literal)
        built = normwise._construct_correct_fixture(spec)
        tensors = built["tensors"]
        seeds = literal["seeds"]
        if spec.fixture_id == "zero_corner":
            expected = {
                "x": _pcg(seeds["x"], (17,)),
                "u": _pcg(seeds["output_direction"], (17,)),
                "v": _pcg(seeds["parameter_direction"], (17,)),
            }
        elif literal["kind"] == "affine_linear":
            expected = {
                "matrix": _pcg(seeds["matrix"], (257, 193)),
                "x": _pcg(seeds["input"], (193,)),
                "u": _pcg(seeds["output_direction"], (257,)),
                "v": _pcg(seeds["parameter_direction"], (193,)),
            }
        elif spec.fixture_id == "smooth_parameter_tree":
            shapes = {"w1": (23, 11), "b1": (23,), "w2": (19, 23), "b2": (19,)}
            expected = {
                name: _pcg(seeds[name], shape, 2**-3 if name.startswith("w") else 2**-4)
                for name, shape in shapes.items()
            }
            expected["input"] = _pcg(seeds["input"], (17, 11), 2**-2)
            flat = _pcg(seeds["parameter_direction"], (732,))
            start = 0
            for name, shape in shapes.items():
                count = int(np.prod(shape))
                expected[f"v_{name}"] = flat[start : start + count].reshape(shape)
                start += count
            expected["u"] = _pcg(seeds["output_direction"], (17, 19))
        else:
            expected = {"x": _pcg(seeds["input"], (8193,))}
            q = _pcg(seeds["output_pair_base"], (4096,))
            p = _pcg(seeds["parameter_pair_base"], (4096,))
            expected["u"] = torch.cat((q.repeat_interleave(2), torch.ones(1)))
            expected["v"] = torch.cat((p.repeat_interleave(2), torch.ones(1)))
            expected["diagonal"] = torch.cat(
                (torch.tensor([2**10, -(2**10)]).repeat(4096), torch.tensor([2**-10]))
            ).to(torch.float32)
        assert list(tensors) == list(expected)
        for name, wanted in expected.items():
            assert tensors[name].dtype == torch.float32
            assert tensors[name].device.type == "cpu"
            assert tensors[name].is_contiguous()
            assert tensors[name].numpy().tobytes(order="C") == wanted.numpy().tobytes(order="C")

    affine = specs[1]
    for mutation in ("scale", "seed", "seed_order", "dimension_type", "extra"):
        metadata = deepcopy(affine.metadata)
        if mutation == "scale":
            metadata["scales"]["operator"] = "0.000244140625"
        elif mutation == "seed":
            metadata["seeds"]["matrix"] = "0X4E4F524D00000101"
        elif mutation == "seed_order":
            metadata["seeds"] = dict(reversed(tuple(metadata["seeds"].items())))
        elif mutation == "dimension_type":
            metadata["dimensions"]["input"] = 193.0
        else:
            metadata["scales"]["extra"] = "1"
        with pytest.raises(ValueError, match="metadata"):
            normwise._construct_correct_fixture(normwise.FixtureSpec(affine.fixture_id, metadata))


def test_correct_fixture_uses_torch_func_and_exact_action_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_jvp, original_vjp = torch.func.jvp, torch.func.vjp
    calls = {"jvp": 0, "vjp": 0}
    actual_jvp: list[torch.Tensor] = []
    actual_vjp: list[torch.Tensor | dict[str, torch.Tensor]] = []

    def jvp(*args: Any, **kwargs: Any) -> Any:
        calls["jvp"] += 1
        result = original_jvp(*args, **kwargs)
        actual_jvp.append(result[1].detach().clone())
        return result

    def vjp(*args: Any, **kwargs: Any) -> Any:
        calls["vjp"] += 1
        output, pullback = original_vjp(*args, **kwargs)

        def captured_pullback(*cotangents: Any) -> Any:
            result = pullback(*cotangents)
            action = result[0]
            actual_vjp.append(
                {name: value.detach().clone() for name, value in action.items()}
                if isinstance(action, dict)
                else action.detach().clone()
            )
            return result

        return output, captured_pullback

    monkeypatch.setattr(torch.func, "jvp", jvp)
    monkeypatch.setattr(torch.func, "vjp", vjp)
    results = [normwise.run_correct_fixture(spec) for spec in normwise.correct_fixture_specs()]
    assert calls == {"jvp": 6, "vjp": 6}
    for index, result in enumerate(results):
        fixture_id = tuple(_PROTOCOL_CORRECT_METADATA)[index]
        literal = _PROTOCOL_CORRECT_METADATA[fixture_id]
        seeds = literal["seeds"]
        assert result["beta_norm"] <= normwise.CORRECT_FIXTURE_CEILING
        assert result["passed"] is True
        jvp_bytes = np.ascontiguousarray(actual_jvp[index].numpy()).tobytes()
        assert result["jvp_sha256"] == hashlib.sha256(jvp_bytes).hexdigest()
        vjp = actual_vjp[index]
        if isinstance(vjp, dict):
            names = ("w1", "b1", "w2", "b2")
            shapes = ((23, 11), (23,), (19, 23), (19,))
            flat = _pcg(seeds["parameter_direction"], (732,))
            start = 0
            direction = {}
            for name, shape in zip(names, shapes, strict=True):
                count = math.prod(shape)
                direction[name] = flat[start : start + count].reshape(shape)
                start += count
            u = _pcg(seeds["output_direction"], (17, 19))
            vjp_bytes = b"".join(
                np.ascontiguousarray(vjp[name].numpy()).tobytes() for name in names
            )
        else:
            names = ("x",)
            if fixture_id == "zero_corner":
                direction = {"x": _pcg(seeds["parameter_direction"], (17,))}
                u = _pcg(seeds["output_direction"], (17,))
            elif literal["kind"] == "affine_linear":
                direction = {"x": _pcg(seeds["parameter_direction"], (193,))}
                u = _pcg(seeds["output_direction"], (257,))
            else:
                p = _pcg(seeds["parameter_pair_base"], (4096,))
                q = _pcg(seeds["output_pair_base"], (4096,))
                direction = {"x": torch.cat((p.repeat_interleave(2), torch.ones(1)))}
                u = torch.cat((q.repeat_interleave(2), torch.ones(1)))
            vjp_bytes = np.ascontiguousarray(vjp.numpy()).tobytes()
        assert result["vjp_sha256"] == hashlib.sha256(vjp_bytes).hexdigest()
        vjp_tree = vjp if isinstance(vjp, dict) else {"x": vjp}
        reference = _reference(u, actual_jvp[index], direction, vjp_tree, names)
        for name, expected in reference.items():
            assert result[name] == pytest.approx(expected, rel=1.0e-9, abs=1.0e-9)


def test_correct_fixture_oracle_rejects_fabricated_action_after_real_torch_func_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_jvp = torch.func.jvp
    captured: list[torch.Tensor] = []

    def substituted_jvp(*args: Any, **kwargs: Any) -> Any:
        output, action = original_jvp(*args, **kwargs)
        captured.append(action.detach().clone())
        return output, action + torch.ones_like(action)

    monkeypatch.setattr(torch.func, "jvp", substituted_jvp)
    result = normwise.run_correct_fixture(normwise.correct_fixture_specs()[1])
    actual_hash = hashlib.sha256(np.ascontiguousarray(captured[0].numpy()).tobytes()).hexdigest()
    with pytest.raises(AssertionError):
        assert result["jvp_sha256"] == actual_hash


def test_correct_fixture_calibration_band_and_paired_cancellation() -> None:
    results = {
        spec.fixture_id: normwise.run_correct_fixture(spec)
        for spec in normwise.correct_fixture_specs()
    }
    assert all(
        result["beta_norm"] <= 6.25e-5 and result["passed"] is True for result in results.values()
    )
    paired = results["paired_cancellation"]
    assert paired["lhs"] == paired["rhs"] == 2**-10
    assert paired["lhs_absolute_product_sum"] > 1.0
    assert paired["rhs_absolute_product_sum"] > 1.0


def test_rebuild_action_order_and_sign_controls_are_exact() -> None:
    for spec in normwise.correct_fixture_specs():
        result = normwise.run_fixture_controls(spec)
        controls = result["controls"]
        assert list(controls) == [
            "rebuild",
            "reversed_action_order",
            "parameter_sign",
            "output_sign",
        ]
        for name in ("rebuild", "reversed_action_order"):
            assert controls[name]["exact_action_hash_match"] is True
            assert controls[name]["jvp_sha256"] == result["jvp_sha256"]
            assert controls[name]["vjp_sha256"] == result["vjp_sha256"]
            assert controls[name]["beta_norm"] <= 6.25e-5
            assert controls[name]["passed"] is True
        for name in ("parameter_sign", "output_sign"):
            assert controls[name]["exact_relation"] is True
            assert controls[name]["beta_norm"] <= 6.25e-5
            assert controls[name]["passed"] is True
        assert result["passed"] is True


def test_rebuild_control_rejects_altered_later_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.func.jvp
    calls = 0

    def altered(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        output, action = original(*args, **kwargs)
        if calls == 2:
            action = action + torch.ones_like(action)
        return output, action

    monkeypatch.setattr(torch.func, "jvp", altered)
    result = normwise.run_fixture_controls(normwise.correct_fixture_specs()[1])
    assert result["controls"]["rebuild"]["exact_action_hash_match"] is False
    assert result["controls"]["rebuild"]["passed"] is False
    assert result["passed"] is False


def test_action_order_control_rejects_order_dependent_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_jvp = torch.func.jvp
    original_vjp = torch.func.vjp
    pullback_called = False

    def tracked_vjp(*args: Any, **kwargs: Any) -> Any:
        nonlocal pullback_called
        pullback_called = False
        output, pullback = original_vjp(*args, **kwargs)

        def tracked_pullback(*pullback_args: Any, **pullback_kwargs: Any) -> Any:
            nonlocal pullback_called
            pullback_called = True
            return pullback(*pullback_args, **pullback_kwargs)

        return output, tracked_pullback

    def order_dependent_jvp(*args: Any, **kwargs: Any) -> Any:
        output, action = original_jvp(*args, **kwargs)
        if pullback_called:
            action = action + torch.ones_like(action)
        return output, action

    monkeypatch.setattr(torch.func, "vjp", tracked_vjp)
    monkeypatch.setattr(torch.func, "jvp", order_dependent_jvp)
    result = normwise.run_fixture_controls(normwise.correct_fixture_specs()[1])
    assert result["controls"]["reversed_action_order"]["exact_action_hash_match"] is False
    assert result["controls"]["reversed_action_order"]["passed"] is False
    assert result["passed"] is False


def test_registered_fault_construction_and_separation_are_frozen() -> None:
    specs = normwise.registered_fault_specs()
    assert tuple(spec.fixture_id for spec in specs) == tuple(_PROTOCOL_FAULT_METADATA)
    expected_beta = {
        "zero_map_forward_injection": 2.0,
        "identity_reverse_scale_fault": 2.0 / 511.0,
        "identity_reverse_pair_sign_fault": 1.0,
    }
    for spec in specs:
        assert _protocol_literal_equal(spec.metadata, _PROTOCOL_FAULT_METADATA[spec.fixture_id])
        result = normwise.run_registered_fault(spec)
        seeds = _PROTOCOL_FAULT_METADATA[spec.fixture_id]["seeds"]
        if spec.fixture_id == "zero_map_forward_injection":
            u = _pcg(seeds["output_direction"], (17,))
            expected_jvp = torch.tensor(2**-10, dtype=torch.float32) * u
            expected_vjp = torch.tensor(0.0, dtype=torch.float32) * u
        elif spec.fixture_id == "identity_reverse_scale_fault":
            q = _pcg(seeds["shared_direction"], (4096,))
            expected_jvp = q
            expected_vjp = torch.tensor(255 / 256, dtype=torch.float32) * q
        else:
            base = _pcg(seeds["pair_base"], (2048,))
            expected_jvp = base.repeat_interleave(2)
            expected_vjp = torch.stack((base, -base), dim=1).reshape(-1)
        assert (
            result["jvp_sha256"]
            == hashlib.sha256(np.ascontiguousarray(expected_jvp.numpy()).tobytes()).hexdigest()
        )
        assert (
            result["vjp_sha256"]
            == hashlib.sha256(np.ascontiguousarray(expected_vjp.numpy()).tobytes()).hexdigest()
        )
        assert result["beta_norm"] == pytest.approx(
            expected_beta[spec.fixture_id], rel=2.0e-7, abs=2.0e-7
        )
        assert result["controls"]["unmodified"]["beta_norm"] <= 6.25e-5
        assert result["controls"]["unmodified"]["passed"] is True
        assert result["beta_norm"] >= 5.0e-4
        assert result["beta_norm"] - result["controls"]["unmodified"]["beta_norm"] >= 4.375e-4
        assert result["passed"] is True


def test_registered_fault_metadata_rejects_any_amplitude_seed_dimension_or_pair_drift() -> None:
    for fixture_id, literal in _PROTOCOL_FAULT_METADATA.items():
        for field in ("seeds", "dimensions", "scales"):
            for key in literal[field]:
                changed = deepcopy(literal)
                value = changed[field][key]
                changed[field][key] = value + 1 if type(value) is int else f"{value}-drift"
                with pytest.raises(ValueError, match="metadata"):
                    normwise.run_registered_fault(normwise.FaultSpec(fixture_id, changed))


def _run_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_calibration_source_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "source"
    protocol = repo / "docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md"
    paths = (
        repo / "scripts/rsta_normwise_adjoint.py",
        repo / "scripts/calibrate_pass200_rsta_normwise_adjoint.py",
        repo / "tests/test_rsta_normwise_adjoint.py",
    )
    protocol.parent.mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "tests").mkdir()
    protocol.write_bytes(b"frozen protocol\n")
    for index, path in enumerate(paths):
        path.write_bytes(f"source-{index}\n".encode())
    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "test@example.invalid")
    _run_git(repo, "config", "user.name", "Test")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-qm", "source")
    source = _run_git(repo, "rev-parse", "HEAD")
    output = (
        repo
        / "reports/generated/pass200_rsta_receipt"
        / f"{source}-normwise-adjoint-calibration.json"
    )
    output.parent.mkdir(parents=True)
    return repo, output, source


def test_candidate_free_cli_import_and_argument_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = {
        "exact_contextual_rsta_fields",
        "score_rsta_batch",
        "decide_stage_a",
        "joint_bootstrap",
        "scientific_payload",
        "serialize_receiver_rows",
        "load_model",
        "load_checkpoint",
        "load_data",
        "load_manifest",
    }

    class ForbiddenModule:
        def __getattr__(self, name: str) -> Any:
            if name in forbidden:
                raise AssertionError(f"forbidden candidate reachability: {name}")
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "diagnose_pass200_rsta_stage_a", ForbiddenModule())
    cli = _load_calibration_cli()
    assert forbidden.isdisjoint(vars(cli))
    assert cli._parse_args(["--output", "/tmp/out"]).output == Path("/tmp/out")
    with pytest.raises(SystemExit) as error:
        cli._parse_args(["--output", "/tmp/out", "--seed", "0"])
    assert error.value.code == 2


def test_candidate_free_fresh_process_import_closure_and_runtime_isolation() -> None:
    repository = Path(__file__).resolve().parents[1]
    script = r"""
import ast
import importlib.abc
import importlib.util
import sys
from pathlib import Path

repo = Path(sys.argv[1])
helper_path = repo / "scripts/rsta_normwise_adjoint.py"
cli_path = repo / "scripts/calibrate_pass200_rsta_normwise_adjoint.py"
forbidden_modules = (
    "diagnose_pass200_rsta_stage_a",
    "diagnose_pass159_cotangent_stage_a",
    "export_final_inshop_embeddings",
    "sfora.bn_inception",
    "sfora.data",
    "sfora.image_end_to_end",
)
forbidden_symbols = {
    "exact_contextual_rsta_fields", "score_rsta_batch", "decide_stage_a",
    "joint_bootstrap", "scientific_payload", "binding_only_payload",
    "_validate_receiver_audit_row", "_load_scientific_model",
    "_load_digest_bound_packs", "load_and_bind_seed", "load_training_only_seed",
    "load_checkpoint", "load_manifest", "load_data", "load_model", "DataLoader",
}
allowed_import_roots = {
    "__future__", "argparse", "collections", "contextlib", "copy", "dataclasses",
    "hashlib", "json", "math", "numpy", "os", "pathlib", "platform", "re",
    "rsta_normwise_adjoint", "stat", "subprocess", "sys", "torch", "typing",
}

class BlockForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden_modules):
            raise AssertionError("forbidden module import: " + fullname)
        return None

sys.meta_path.insert(0, BlockForbidden())
for source_path in (helper_path, cli_path):
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            assert roots <= allowed_import_roots, (source_path, roots)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert node.module.split(".", 1)[0] in allowed_import_roots, (source_path, node.module)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            symbol = node.id if isinstance(node, ast.Name) else node.attr
            assert symbol not in forbidden_symbols, (source_path, symbol)
    text = source_path.read_text()
    assert "torch.load" not in text

sys.path.insert(0, str(repo / "scripts"))
import rsta_normwise_adjoint as core
spec = importlib.util.spec_from_file_location("fresh_calibration_cli", cli_path)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)
provenance = {
    "protocol": {"path": str(cli.PROTOCOL_RELATIVE_PATH), "sha256": "a" * 64, "commit": "b" * 40},
    "execution_audit": {
        "executing_git_commit": "c" * 40,
        "calibration_source_commit": "d" * 40,
        "calibration_cli_path": cli.SOURCE_PATHS[1],
        "calibration_cli_sha256": "e" * 64,
    },
    "source": {
        "git_revision": "d" * 40,
        "files": {
            path: ("e" * 64 if path == cli.SOURCE_PATHS[1] else "f" * 64)
            for path in cli.SOURCE_PATHS
        },
    },
}
cli.authenticate_current_source = lambda path: provenance
payload = cli.calibration_payload(repo / cli.PROTOCOL_RELATIVE_PATH)
assert payload["candidate_values_computed"] is False
assert payload["stage_a_verdict"] == "NOT_COMPUTED"
assert not any(
    any(name == forbidden or name.startswith(forbidden + ".") for forbidden in forbidden_modules)
    for name in sys.modules
)
"""
    completed = subprocess.run(
        (sys.executable, "-I", "-c", script, str(repository)),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("phase", "exception_type"),
    (
        ("deterministic setup", RuntimeError),
        ("fixture construction", TypeError),
        ("payload validation", RuntimeError),
    ),
)
def test_cli_boundary_converts_ordinary_exception_to_structural_exit_without_output(
    phase: str,
    exception_type: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_calibration_cli()
    output = tmp_path / "result.json"
    temporary = tmp_path / f".{output.name}.tmp.{os.getpid()}"
    base = _valid_result()
    provenance = {name: deepcopy(base[name]) for name in ("protocol", "execution_audit", "source")}
    monkeypatch.setattr(cli, "authenticate_source", lambda *args: provenance)
    if phase == "deterministic setup":
        monkeypatch.setattr(
            cli,
            "_configure_cpu",
            lambda: (_ for _ in ()).throw(exception_type(f"injected {phase}")),
        )
    else:
        monkeypatch.setattr(cli, "_configure_cpu", lambda: None)
        monkeypatch.setattr(
            cli,
            "_build_payload",
            lambda value: (_ for _ in ()).throw(exception_type(f"injected {phase}")),
        )
    assert cli.run_cli(["--output", str(output)]) == 2
    assert capsys.readouterr().err == f"structural failure: injected {phase}\n"
    assert not output.exists()
    assert not temporary.exists()


def test_cli_boundary_does_not_catch_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_calibration_cli()
    monkeypatch.setattr(
        cli,
        "authenticate_source",
        lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        cli.run_cli(["--output", str(tmp_path / "result.json")])


def test_cli_boundary_publishes_valid_failed_payload_and_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_calibration_cli()
    output = tmp_path / "result.json"
    payload = _valid_result()
    control = payload["correct_fixtures"]["zero_corner"]["controls"]["parameter_sign"]
    control["exact_relation"] = False
    control["passed"] = False
    payload["correct_fixtures"]["zero_corner"]["passed"] = False
    payload["all_passed"] = False
    normwise.validate_calibration_result(payload)
    monkeypatch.setattr(cli, "authenticate_source", lambda *args: object())
    monkeypatch.setattr(cli, "_configure_cpu", lambda: None)
    monkeypatch.setattr(cli, "_build_payload", lambda value: payload)
    assert cli.run_cli(["--output", str(output)]) == 1
    assert json.loads(output.read_text()) == payload
    assert not (tmp_path / f".{output.name}.tmp.{os.getpid()}").exists()


def test_cli_source_authentication_binds_destination_git_and_worktree(tmp_path: Path) -> None:
    cli = _load_calibration_cli()
    repo, output, source = _make_calibration_source_repo(tmp_path)
    authenticated = cli.authenticate_source(repo, output)
    assert authenticated["execution_audit"]["executing_git_commit"] == source
    assert authenticated["execution_audit"]["calibration_source_commit"] == source
    assert list(authenticated["source"]["files"]) == [
        "scripts/rsta_normwise_adjoint.py",
        "scripts/calibrate_pass200_rsta_normwise_adjoint.py",
        "tests/test_rsta_normwise_adjoint.py",
    ]
    assert authenticated["protocol"]["commit"] == source
    assert (
        authenticated["execution_audit"]["calibration_cli_sha256"]
        == authenticated["source"]["files"]["scripts/calibrate_pass200_rsta_normwise_adjoint.py"]
    )

    wrong_outputs = (
        output.with_name(f"{'a' * 40}-normwise-adjoint-calibration.json"),
        output.with_name(f"x{source}-normwise-adjoint-calibration.json"),
        output.with_name(f"{source.upper()}-normwise-adjoint-calibration.json"),
        output.parent.parent / output.name,
        repo.parent / output.name,
    )
    for wrong in wrong_outputs:
        with pytest.raises(ValueError, match="output|source|commit"):
            cli.authenticate_source(repo, wrong)

    helper = repo / "scripts/rsta_normwise_adjoint.py"
    helper.write_bytes(b"dirty worktree\n")
    constructed = False
    with pytest.raises(ValueError, match="worktree|blob|source"):
        cli.authenticate_source(repo, output)
        constructed = True
    assert constructed is False


def test_cli_source_authentication_rejects_executing_blob_and_ancestry_drift(
    tmp_path: Path,
) -> None:
    cli = _load_calibration_cli()
    repo, output, source = _make_calibration_source_repo(tmp_path)
    helper = repo / "scripts/rsta_normwise_adjoint.py"
    original = helper.read_bytes()
    helper.write_bytes(b"later committed bytes\n")
    _run_git(repo, "add", str(helper.relative_to(repo)))
    _run_git(repo, "commit", "-qm", "later")
    helper.write_bytes(original)
    with pytest.raises(ValueError, match="executing|blob|source"):
        cli.authenticate_source(repo, output)

    unrelated = "f" * 40
    unrelated_output = output.with_name(f"{unrelated}-normwise-adjoint-calibration.json")
    with pytest.raises(ValueError, match="ancestor|commit|source"):
        cli.authenticate_source(repo, unrelated_output)


@pytest.mark.parametrize(
    "relative_path",
    (
        "scripts/rsta_normwise_adjoint.py",
        "scripts/calibrate_pass200_rsta_normwise_adjoint.py",
        "tests/test_rsta_normwise_adjoint.py",
        "docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md",
    ),
)
def test_cli_source_authentication_rejects_each_worktree_byte_drift(
    relative_path: str, tmp_path: Path
) -> None:
    cli = _load_calibration_cli()
    repo, output, _ = _make_calibration_source_repo(tmp_path)
    (repo / relative_path).write_bytes(b"independently drifted bytes\n")
    with pytest.raises(ValueError, match="worktree|blob|source|protocol"):
        cli.authenticate_source(repo, output)


def test_cli_authentication_failure_precedes_cpu_and_fixture_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_calibration_cli()
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "authenticate_source",
        lambda *args: (_ for _ in ()).throw(ValueError("source provenance")),
    )
    monkeypatch.setattr(cli, "_configure_cpu", lambda: calls.append("configured"))
    monkeypatch.setattr(cli.core, "correct_fixture_specs", lambda: calls.append("fixture"))
    assert cli.run_cli(["--output", str(tmp_path / "out.json")]) == 2
    assert calls == []


def test_cli_configures_cpu_before_fixture_construction_and_builds_exact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_calibration_cli()
    base = _valid_result()
    calls: list[str] = []
    monkeypatch.setattr(
        cli.torch, "set_num_threads", lambda value: calls.append(f"threads:{value}")
    )
    monkeypatch.setattr(
        cli.torch, "set_num_interop_threads", lambda value: calls.append(f"interop:{value}")
    )
    monkeypatch.setattr(
        cli.torch,
        "use_deterministic_algorithms",
        lambda enabled, warn_only=False: calls.append(f"deterministic:{enabled}:{warn_only}"),
    )
    monkeypatch.setattr(
        cli.torch,
        "set_autocast_enabled",
        lambda device, enabled: calls.append(f"autocast:{device}:{enabled}"),
    )
    monkeypatch.setattr(cli.torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(cli.torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(cli.torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(cli.torch, "is_autocast_enabled", lambda *args: False)
    monkeypatch.setattr(
        cli,
        "authenticate_current_source",
        lambda path: {
            "protocol": deepcopy(base["protocol"]),
            "execution_audit": deepcopy(base["execution_audit"]),
            "source": deepcopy(base["source"]),
        },
    )

    real_correct_specs = normwise.correct_fixture_specs

    def correct_specs() -> tuple[Any, ...]:
        assert calls[:4] == [
            "threads:1",
            "interop:1",
            "deterministic:True:False",
            "autocast:cpu:False",
        ]
        return real_correct_specs()

    monkeypatch.setattr(cli.core, "correct_fixture_specs", correct_specs)
    monkeypatch.setattr(
        cli.core,
        "run_fixture_controls",
        lambda spec: deepcopy(base["correct_fixtures"][spec.fixture_id]),
    )
    monkeypatch.setattr(cli.core, "registered_fault_specs", normwise.registered_fault_specs)
    monkeypatch.setattr(
        cli.core,
        "run_registered_fault",
        lambda spec: deepcopy(base["registered_faults"][spec.fixture_id]),
    )
    payload = cli.calibration_payload(tmp_path / "protocol.md")
    normwise.validate_calibration_result(payload)
    assert list(payload) == list(base)
    assert list(payload["correct_fixtures"]) == list(normwise.CORRECT_FIXTURE_IDS)
    assert list(payload["registered_faults"]) == list(normwise.REGISTERED_FAULT_IDS)
    assert payload["candidate_values_computed"] is False
    assert payload["stage_a_verdict"] == "NOT_COMPUTED"


def test_cli_publishes_finite_failed_calibration_artifact(tmp_path: Path) -> None:
    cli = _load_calibration_cli()
    payload = _valid_result()
    control = payload["correct_fixtures"]["zero_corner"]["controls"]["parameter_sign"]
    control["exact_relation"] = False
    control["passed"] = False
    payload["correct_fixtures"]["zero_corner"]["passed"] = False
    payload["all_passed"] = False
    normwise.validate_calibration_result(payload)
    destination = tmp_path / "failed.json"
    cli.publish_json_no_clobber(destination, payload)
    assert json.loads(destination.read_text()) == payload
    assert destination.read_bytes().endswith(b"\n")
    assert cli.calibration_exit_code(payload) == 1


def test_atomic_no_clobber_publication_exact_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_calibration_cli()
    destination = tmp_path / "result.json"
    payload = _valid_result()
    expected = json.dumps(payload, indent=2, allow_nan=False).encode() + b"\n"
    real_link = cli.os.link
    real_fsync = cli.os.fsync
    links: list[tuple[Path, Path, bool]] = []
    fsync_kinds: list[str] = []

    def tracked_link(source: Any, target: Any, *, follow_symlinks: bool = True) -> None:
        links.append((Path(source), Path(target), follow_symlinks))
        real_link(source, target, follow_symlinks=follow_symlinks)

    def tracked_fsync(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        fsync_kinds.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(cli.os, "link", tracked_link)
    monkeypatch.setattr(cli.os, "fsync", tracked_fsync)
    cli.publish_json_no_clobber(destination, payload)
    temporary = tmp_path / f".{destination.name}.tmp.{os.getpid()}"
    assert links == [(temporary, destination, False)]
    assert fsync_kinds == ["file", "directory", "directory"]
    assert destination.read_bytes() == expected
    assert destination.stat().st_mode & 0o777 == 0o600
    assert not temporary.exists()

    with pytest.raises(FileExistsError):
        cli.publish_json_no_clobber(destination, payload)
    assert destination.read_bytes() == expected


def test_atomic_no_clobber_rejects_foreign_exact_temp(tmp_path: Path) -> None:
    cli = _load_calibration_cli()
    destination = tmp_path / "result.json"
    temporary = tmp_path / f".{destination.name}.tmp.{os.getpid()}"
    temporary.write_bytes(b"foreign")
    with pytest.raises(FileExistsError):
        cli.publish_json_no_clobber(destination, _valid_result())
    assert temporary.read_bytes() == b"foreign"
    assert not destination.exists()


@pytest.mark.parametrize("foreign_name", ("destination", "temporary"))
def test_atomic_no_clobber_never_follows_foreign_symlink(foreign_name: str, tmp_path: Path) -> None:
    cli = _load_calibration_cli()
    destination = tmp_path / "result.json"
    temporary = tmp_path / f".{destination.name}.tmp.{os.getpid()}"
    target = tmp_path / "foreign"
    target.write_bytes(b"foreign")
    (destination if foreign_name == "destination" else temporary).symlink_to(target)
    with pytest.raises(FileExistsError):
        cli.publish_json_no_clobber(destination, _valid_result())
    assert target.read_bytes() == b"foreign"


def test_publication_rollback_preserves_replaced_destination_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_calibration_cli()
    destination = tmp_path / "result.json"
    real_fsync = cli.os.fsync
    fsync_count = 0

    def replace_then_fail(fd: int) -> None:
        nonlocal fsync_count
        fsync_count += 1
        if fsync_count == 2:
            destination.unlink()
            destination.write_bytes(b"foreign replacement")
            raise OSError("first_directory_fsync")
        real_fsync(fd)

    monkeypatch.setattr(cli.os, "fsync", replace_then_fail)
    with pytest.raises(OSError, match="first_directory_fsync"):
        cli.publish_json_no_clobber(destination, _valid_result())
    assert destination.read_bytes() == b"foreign replacement"
    assert not (tmp_path / f".{destination.name}.tmp.{os.getpid()}").exists()


def test_atomic_no_clobber_preserves_destination_racer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_calibration_cli()
    destination = tmp_path / "result.json"

    def racing_link(source: Any, target: Any, **kwargs: Any) -> None:
        Path(target).write_bytes(b"foreign destination")
        raise FileExistsError(target)

    monkeypatch.setattr(cli.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        cli.publish_json_no_clobber(destination, _valid_result())
    assert destination.read_bytes() == b"foreign destination"
    assert not (tmp_path / f".{destination.name}.tmp.{os.getpid()}").exists()


def test_publication_rollback_after_owned_open_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_calibration_cli()
    destination = tmp_path / "result.json"
    temporary = tmp_path / f".{destination.name}.tmp.{os.getpid()}"
    monkeypatch.setattr(
        cli.os,
        "fchmod",
        lambda *args: (_ for _ in ()).throw(OSError("fchmod")),
    )
    with pytest.raises(OSError, match="fchmod"):
        cli.publish_json_no_clobber(destination, _valid_result())
    assert not temporary.exists()
    assert not destination.exists()


@pytest.mark.parametrize(
    ("failure", "destination_survives"),
    (
        ("short_write", False),
        ("file_fsync", False),
        ("hard_link", False),
        ("first_directory_fsync", False),
        ("temp_unlink", False),
        ("second_directory_fsync", True),
    ),
)
def test_publication_rollback_is_owned_inode_checked(
    failure: str,
    destination_survives: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_calibration_cli()
    destination = tmp_path / "result.json"
    temporary = tmp_path / f".{destination.name}.tmp.{os.getpid()}"
    real_fsync = cli.os.fsync
    real_link = cli.os.link
    real_unlink = cli.os.unlink
    fsync_count = 0
    temp_unlink_failed = False

    if failure == "short_write":
        monkeypatch.setattr(cli, "_write_buffer", lambda stream, data: len(data) - 1)

    def failing_fsync(fd: int) -> None:
        nonlocal fsync_count
        fsync_count += 1
        wanted = {"file_fsync": 1, "first_directory_fsync": 2, "second_directory_fsync": 3}
        if failure in wanted and fsync_count == wanted[failure]:
            raise OSError(failure)
        real_fsync(fd)

    def failing_link(source: Any, target: Any, **kwargs: Any) -> None:
        if failure == "hard_link":
            raise OSError(failure)
        real_link(source, target, **kwargs)

    def failing_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal temp_unlink_failed
        if failure == "temp_unlink" and Path(path) == temporary and not temp_unlink_failed:
            temp_unlink_failed = True
            raise OSError(failure)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cli.os, "fsync", failing_fsync)
    monkeypatch.setattr(cli.os, "link", failing_link)
    monkeypatch.setattr(cli.os, "unlink", failing_unlink)
    with pytest.raises(OSError, match=failure.replace("_", ".*")):
        cli.publish_json_no_clobber(destination, _valid_result())
    assert destination.exists() is destination_survives
    assert not temporary.exists()
