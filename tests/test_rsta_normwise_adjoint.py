from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rsta_normwise_adjoint as normwise


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
        "eta_norm": error / denominator,
        "beta_norm": 2.0 * error / denominator,
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
    correct_controls = {
        name: {
            "jvp_sha256": "3" * 64,
            "vjp_sha256": "4" * 64,
            "beta_norm": 0.0,
            (
                "exact_action_hash_match"
                if name in ("rebuild", "reversed_action_order")
                else "exact_relation"
            ): True,
            "passed": True,
        }
        for name in ("rebuild", "reversed_action_order", "parameter_sign", "output_sign")
    }
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
        item["controls"] = deepcopy(correct_controls)
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
    assert tuple(spec.fixture_id for spec in specs) == normwise.CORRECT_FIXTURE_IDS
    for spec in specs:
        assert spec.metadata == normwise.fixture_metadata(spec.fixture_id)
        built = normwise._construct_correct_fixture(spec)
        tensors = built["tensors"]
        seeds = spec.metadata["seeds"]
        if spec.fixture_id == "zero_corner":
            expected = {
                "x": _pcg(seeds["x"], (17,)),
                "u": _pcg(seeds["output_direction"], (17,)),
                "v": _pcg(seeds["parameter_direction"], (17,)),
            }
        elif spec.kind == "affine_linear":
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

    def jvp(*args: Any, **kwargs: Any) -> Any:
        calls["jvp"] += 1
        return original_jvp(*args, **kwargs)

    def vjp(*args: Any, **kwargs: Any) -> Any:
        calls["vjp"] += 1
        return original_vjp(*args, **kwargs)

    monkeypatch.setattr(torch.func, "jvp", jvp)
    monkeypatch.setattr(torch.func, "vjp", vjp)
    results = [normwise.run_correct_fixture(spec) for spec in normwise.correct_fixture_specs()]
    assert calls == {"jvp": 6, "vjp": 6}
    for result in results:
        assert result["beta_norm"] <= normwise.CORRECT_FIXTURE_CEILING
        assert result["passed"] is True
        assert len(result["jvp_sha256"]) == len(result["vjp_sha256"]) == 64


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
