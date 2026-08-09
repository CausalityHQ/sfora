"""Candidate-independent normwise adjoint diagnostic arithmetic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

THRESHOLD = 5.0e-4
CORRECT_FIXTURE_CEILING = 6.25e-5
CORRECT_FIXTURE_IDS = (
    "zero_corner",
    "affine_scale_2m12",
    "affine_scale_1",
    "affine_scale_2p12",
    "smooth_parameter_tree",
    "paired_cancellation",
)
REGISTERED_FAULT_IDS = (
    "zero_map_forward_injection",
    "identity_reverse_scale_fault",
    "identity_reverse_pair_sign_fault",
)

_FIXTURE_METADATA: dict[str, tuple[str, dict[str, str], dict[str, Any], dict[str, str]]] = {
    "zero_corner": (
        "zero_linear",
        {
            "x": "0x4e4f524d00000001",
            "output_direction": "0x4e4f524d00000002",
            "parameter_direction": "0x4e4f524d00000003",
        },
        {"input": 17, "output": 17},
        {"operator": "0"},
    ),
    "affine_scale_2m12": (
        "affine_linear",
        {
            "matrix": "0x4e4f524d00000101",
            "input": "0x4e4f524d00000102",
            "output_direction": "0x4e4f524d00000103",
            "parameter_direction": "0x4e4f524d00000104",
        },
        {"input": 193, "output": 257},
        {"operator": "2**-12"},
    ),
    "affine_scale_1": (
        "affine_linear",
        {
            "matrix": "0x4e4f524d00000101",
            "input": "0x4e4f524d00000102",
            "output_direction": "0x4e4f524d00000103",
            "parameter_direction": "0x4e4f524d00000104",
        },
        {"input": 193, "output": 257},
        {"operator": "1"},
    ),
    "affine_scale_2p12": (
        "affine_linear",
        {
            "matrix": "0x4e4f524d00000101",
            "input": "0x4e4f524d00000102",
            "output_direction": "0x4e4f524d00000103",
            "parameter_direction": "0x4e4f524d00000104",
        },
        {"input": 193, "output": 257},
        {"operator": "2**12"},
    ),
    "smooth_parameter_tree": (
        "smooth_parameter_tree",
        {
            "w1": "0x4e4f524d00000201",
            "b1": "0x4e4f524d00000202",
            "w2": "0x4e4f524d00000203",
            "b2": "0x4e4f524d00000204",
            "input": "0x4e4f524d00000205",
            "parameter_direction": "0x4e4f524d00000206",
            "output_direction": "0x4e4f524d00000207",
        },
        {
            "batch": 17,
            "input": 11,
            "hidden": 23,
            "output": 19,
            "parameter_shapes": {"w1": [23, 11], "b1": [23], "w2": [19, 23], "b2": [19]},
        },
        {"weight": "2**-3", "bias": "2**-4", "input": "2**-2", "normalization_eps": "1e-12"},
    ),
    "paired_cancellation": (
        "paired_cancellation_linear",
        {
            "input": "0x4e4f524d00000301",
            "output_pair_base": "0x4e4f524d00000302",
            "parameter_pair_base": "0x4e4f524d00000303",
        },
        {"input": 8193, "output": 8193, "pairs": 4096},
        {"positive_pair": "2**10", "negative_pair": "-2**10", "final": "2**-10"},
    ),
    "zero_map_forward_injection": (
        "injected_forward_action",
        {
            "x": "0x4e4f524d00000001",
            "output_direction": "0x4e4f524d00000002",
            "parameter_direction": "0x4e4f524d00000003",
        },
        {"input": 17, "output": 17},
        {"operator": "0", "forward_injection": "2**-10"},
    ),
    "identity_reverse_scale_fault": (
        "injected_reverse_scale",
        {"shared_direction": "0x4e4f524d00000401"},
        {"input": 4096, "output": 4096},
        {"operator": "1", "reverse_action": "255/256"},
    ),
    "identity_reverse_pair_sign_fault": (
        "injected_reverse_pair_sign",
        {"pair_base": "0x4e4f524d00000402"},
        {"input": 4096, "output": 4096, "pairs": 2048},
        {"operator": "1", "forward_pair": "[1,1]", "reverse_pair": "[1,-1]"},
    ),
}


def fixture_metadata(fixture_id: str) -> dict[str, object]:
    """Return a detached copy of the registered literal metadata."""
    import copy

    kind, seeds, dimensions, scales = _FIXTURE_METADATA[fixture_id]
    return copy.deepcopy(
        {
            "fixture_id": fixture_id,
            "kind": kind,
            "seeds": seeds,
            "dimensions": dimensions,
            "scales": scales,
        }
    )


def _tensor(value: Any, *, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a finite CPU torch.float32 tensor")
    return value


def _stack_sum(values: list[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.zeros((), dtype=torch.float64)
    return torch.stack(values).sum(dtype=torch.float64)


def _cancellation_factor(product_sum: float, scalar: float) -> float | str:
    if scalar != 0.0:
        return product_sum / abs(scalar)
    if product_sum == 0.0:
        return 1.0
    return "infinity"


def normwise_adjoint_metrics(
    u: torch.Tensor,
    a: torch.Tensor,
    parameter_direction: Mapping[str, torch.Tensor],
    vjp_action: Mapping[str, torch.Tensor],
    parameter_names: Sequence[str],
) -> dict[str, object]:
    """Compute the frozen FP64 normwise adjoint metrics from FP32 actions."""
    u = _tensor(u, name="output direction")
    a = _tensor(a, name="JVP action")
    if u.shape != a.shape:
        raise ValueError("output direction and JVP action shapes differ")
    names = tuple(parameter_names)
    if not names or any(type(name) is not str or not name for name in names):
        raise ValueError("parameter names must be a nonempty ordered string sequence")
    if len(set(names)) != len(names) or list(parameter_direction) != list(names):
        raise ValueError("parameter direction topology differs from parameter names")
    if list(vjp_action) != list(names):
        raise ValueError("VJP action topology differs from parameter names")

    ud, ad = u.double(), a.double()
    lhs_tensor = (ud * ad).sum(dtype=torch.float64)
    lhs_absolute_tensor = (ud * ad).abs().sum(dtype=torch.float64)
    output_norm_tensor = (ud * ud).sum(dtype=torch.float64).sqrt()
    jvp_norm_tensor = (ad * ad).sum(dtype=torch.float64).sqrt()

    rhs_terms: list[torch.Tensor] = []
    rhs_absolute_terms: list[torch.Tensor] = []
    parameter_square_terms: list[torch.Tensor] = []
    vjp_square_terms: list[torch.Tensor] = []
    for name in names:
        direction = _tensor(parameter_direction[name], name=f"parameter direction {name}")
        action = _tensor(vjp_action[name], name=f"VJP action {name}")
        if direction.shape != action.shape:
            raise ValueError(f"parameter direction/VJP shape differs for {name}")
        direction64, action64 = direction.double(), action.double()
        product = direction64 * action64
        rhs_terms.append(product.sum(dtype=torch.float64))
        rhs_absolute_terms.append(product.abs().sum(dtype=torch.float64))
        parameter_square_terms.append((direction64 * direction64).sum(dtype=torch.float64))
        vjp_square_terms.append((action64 * action64).sum(dtype=torch.float64))

    rhs_tensor = _stack_sum(rhs_terms)
    rhs_absolute_tensor = _stack_sum(rhs_absolute_terms)
    parameter_norm_tensor = _stack_sum(parameter_square_terms).sqrt()
    vjp_norm_tensor = _stack_sum(vjp_square_terms).sqrt()
    values = [
        lhs_tensor,
        rhs_tensor,
        lhs_absolute_tensor,
        rhs_absolute_tensor,
        output_norm_tensor,
        parameter_norm_tensor,
        jvp_norm_tensor,
        vjp_norm_tensor,
    ]
    if not all(bool(torch.isfinite(value)) for value in values):
        raise ValueError("normwise adjoint reduction is nonfinite")
    lhs, rhs = float(lhs_tensor), float(rhs_tensor)
    absolute_error = abs(lhs - rhs)
    legacy_denominator = max(abs(lhs), abs(rhs), float(np.float64(1.0e-12)))
    legacy_relative_error = absolute_error / legacy_denominator
    output_norm, parameter_norm = float(output_norm_tensor), float(parameter_norm_tensor)
    jvp_norm, vjp_norm = float(jvp_norm_tensor), float(vjp_norm_tensor)
    normwise_denominator = output_norm * jvp_norm + parameter_norm * vjp_norm
    if normwise_denominator == 0.0:
        eta_norm: float | str = 0.0 if absolute_error == 0.0 else "infinity"
        beta_norm: float | str = 0.0 if absolute_error == 0.0 else "infinity"
    else:
        eta_norm = absolute_error / normwise_denominator
        beta_norm = 2.0 * eta_norm
    if any(
        not math.isfinite(value)
        for value in (
            absolute_error,
            legacy_denominator,
            legacy_relative_error,
            normwise_denominator,
        )
    ):
        raise ValueError("normwise adjoint derived scalar is nonfinite")
    lhs_absolute, rhs_absolute = float(lhs_absolute_tensor), float(rhs_absolute_tensor)
    return {
        "lhs": lhs,
        "rhs": rhs,
        "absolute_error": absolute_error,
        "legacy_denominator": legacy_denominator,
        "legacy_relative_error": legacy_relative_error,
        "output_direction_l2": output_norm,
        "parameter_direction_l2": parameter_norm,
        "jvp_l2": jvp_norm,
        "vjp_l2": vjp_norm,
        "normwise_denominator": normwise_denominator,
        "eta_norm": eta_norm,
        "beta_norm": beta_norm,
        "lhs_absolute_product_sum": lhs_absolute,
        "rhs_absolute_product_sum": rhs_absolute,
        "lhs_cancellation_factor": _cancellation_factor(lhs_absolute, lhs),
        "rhs_cancellation_factor": _cancellation_factor(rhs_absolute, rhs),
        "threshold": THRESHOLD,
        "passed": type(beta_norm) is float and beta_norm <= THRESHOLD,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = _tensor(tensor, name="hashed tensor")
    array = np.ascontiguousarray(value.detach().numpy(), dtype=np.float32)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def parameter_tree_sha256(tree: Mapping[str, torch.Tensor], names: Sequence[str]) -> str:
    ordered = tuple(names)
    if set(tree) != set(ordered) or len(tree) != len(ordered):
        raise ValueError("hashed parameter tree topology differs")
    digest = hashlib.sha256()
    for name in ordered:
        value = _tensor(tree[name], name=f"hashed parameter {name}")
        digest.update(np.ascontiguousarray(value.detach().numpy(), dtype=np.float32).tobytes())
    return digest.hexdigest()


def validate_calibration_result(value: object) -> None:
    top_keys = (
        "schema_version",
        "diagnostic",
        "mode",
        "candidate_values_computed",
        "stage_a_verdict",
        "uses_test_data",
        "protocol",
        "execution_audit",
        "source",
        "environment",
        "correct_fixtures",
        "registered_faults",
        "all_passed",
    )
    top = _exact_mapping(value, top_keys, name="calibration result")
    fixed = {
        "schema_version": 1,
        "diagnostic": "pass200-rsta-normwise-adjoint-calibration",
        "mode": "cpu_synthetic_calibration",
        "candidate_values_computed": False,
        "stage_a_verdict": "NOT_COMPUTED",
        "uses_test_data": "synthetic_only",
    }
    if any(
        top[name] != expected or type(top[name]) is not type(expected)
        for name, expected in fixed.items()
    ):
        raise ValueError("calibration fixed field differs")
    protocol = _exact_mapping(top["protocol"], ("path", "sha256", "commit"), name="protocol")
    if protocol["path"] != "docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md":
        raise ValueError("protocol path differs")
    _hex(protocol["sha256"], 64, name="protocol sha256")
    _hex(protocol["commit"], 40, name="protocol commit")
    execution = _exact_mapping(
        top["execution_audit"],
        (
            "executing_git_commit",
            "calibration_source_commit",
            "calibration_cli_path",
            "calibration_cli_sha256",
        ),
        name="execution audit",
    )
    _hex(execution["executing_git_commit"], 40, name="executing commit")
    _hex(execution["calibration_source_commit"], 40, name="source commit")
    if execution["calibration_cli_path"] != "scripts/calibrate_pass200_rsta_normwise_adjoint.py":
        raise ValueError("calibration CLI path differs")
    _hex(execution["calibration_cli_sha256"], 64, name="calibration CLI sha256")
    source = _exact_mapping(top["source"], ("git_revision", "files"), name="source")
    if source["git_revision"] != execution["calibration_source_commit"]:
        raise ValueError("source revision differs")
    files = _exact_mapping(
        source["files"],
        (
            "scripts/rsta_normwise_adjoint.py",
            "scripts/calibrate_pass200_rsta_normwise_adjoint.py",
            "tests/test_rsta_normwise_adjoint.py",
        ),
        name="source files",
    )
    for path, digest in files.items():
        _hex(digest, 64, name=f"source digest {path}")
    if (
        execution["calibration_cli_sha256"]
        != files["scripts/calibrate_pass200_rsta_normwise_adjoint.py"]
    ):
        raise ValueError("execution calibration CLI digest differs from source")
    environment = _exact_mapping(
        top["environment"],
        (
            "device",
            "torch_threads",
            "torch_interop_threads",
            "deterministic_algorithms",
            "autocast",
            "model_dtype",
            "reduction_dtype",
            "python_version",
            "torch_version",
            "numpy_version",
        ),
        name="environment",
    )
    expected_environment = {
        "device": "cpu",
        "torch_threads": 1,
        "torch_interop_threads": 1,
        "deterministic_algorithms": True,
        "autocast": False,
        "model_dtype": "torch.float32",
        "reduction_dtype": "torch.float64",
    }
    if any(
        environment[name] != expected or type(environment[name]) is not type(expected)
        for name, expected in expected_environment.items()
    ):
        raise ValueError("environment fixed field differs")
    if any(
        type(environment[name]) is not str or not environment[name]
        for name in ("python_version", "torch_version", "numpy_version")
    ):
        raise ValueError("environment version differs")
    correct = _exact_mapping(top["correct_fixtures"], CORRECT_FIXTURE_IDS, name="correct fixtures")
    faults = _exact_mapping(
        top["registered_faults"], REGISTERED_FAULT_IDS, name="registered faults"
    )
    entry_passes = []
    for fixture_id, entry in correct.items():
        entry_passes.append(_validate_entry(entry, fixture_id=fixture_id, fault=False))
    for fixture_id, entry in faults.items():
        entry_passes.append(_validate_entry(entry, fixture_id=fixture_id, fault=True))
    if type(top["all_passed"]) is not bool or top["all_passed"] is not all(entry_passes):
        raise ValueError("calibration all_passed differs")


def _exact_mapping(value: object, keys: Sequence[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or list(value) != list(keys):
        raise ValueError(f"{name} keys/order differ")
    return value


def _hex(value: object, length: int, *, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} differs")


def _literal_equal(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, Mapping):
        return list(value) == list(expected) and all(
            _literal_equal(value[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _literal_equal(item, wanted) for item, wanted in zip(value, expected, strict=True)
        )
    return value == expected


def _validate_control(value: object, *, relation: str | None) -> bool:
    relation_key = relation or "passed"
    keys = (
        ("jvp_sha256", "vjp_sha256", "beta_norm", relation_key, "passed")
        if relation
        else ("jvp_sha256", "vjp_sha256", "beta_norm", "passed")
    )
    control = _exact_mapping(value, keys, name="fixture control")
    _hex(control["jvp_sha256"], 64, name="control JVP hash")
    _hex(control["vjp_sha256"], 64, name="control VJP hash")
    if (
        type(control["beta_norm"]) is not float
        or not math.isfinite(control["beta_norm"])
        or control["beta_norm"] < 0.0
    ):
        raise ValueError("control beta_norm differs")
    if relation and type(control[relation]) is not bool:
        raise ValueError("control relation differs")
    expected_passed = control["beta_norm"] <= CORRECT_FIXTURE_CEILING and (
        not relation or control[relation] is True
    )
    if type(control["passed"]) is not bool or control["passed"] is not expected_passed:
        raise ValueError("control passed differs")
    return control["passed"]


def _validate_entry(value: object, *, fixture_id: str, fault: bool) -> bool:
    metric_keys = (
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
    )
    keys = (
        "fixture_id",
        "kind",
        "seeds",
        "dimensions",
        "scales",
        *metric_keys,
        "jvp_sha256",
        "vjp_sha256",
        "controls",
        "threshold",
        "passed",
    )
    entry = _exact_mapping(value, keys, name=f"fixture {fixture_id}")
    metadata = fixture_metadata(fixture_id)
    if any(not _literal_equal(entry[name], metadata[name]) for name in metadata):
        raise ValueError("fixture metadata differs")
    if any(
        type(entry[name]) is not float or not math.isfinite(entry[name]) for name in ("lhs", "rhs")
    ):
        raise ValueError("fixture scalar differs")
    nonnegative = (
        "absolute_error",
        "legacy_denominator",
        "legacy_relative_error",
        "output_direction_l2",
        "parameter_direction_l2",
        "jvp_l2",
        "vjp_l2",
        "normwise_denominator",
        "lhs_absolute_product_sum",
        "rhs_absolute_product_sum",
    )
    if any(
        type(entry[name]) is not float or not math.isfinite(entry[name]) or entry[name] < 0.0
        for name in nonnegative
    ):
        raise ValueError("fixture nonnegative scalar differs")
    for name in ("eta_norm", "beta_norm", "lhs_cancellation_factor", "rhs_cancellation_factor"):
        scalar = entry[name]
        if scalar != "infinity" and (
            type(scalar) is not float or not math.isfinite(scalar) or scalar < 0.0
        ):
            raise ValueError("fixture derived scalar type/sign differs")
    expected_error = abs(entry["lhs"] - entry["rhs"])
    expected_legacy = max(abs(entry["lhs"]), abs(entry["rhs"]), 1.0e-12)
    expected_normwise = (
        entry["output_direction_l2"] * entry["jvp_l2"]
        + entry["parameter_direction_l2"] * entry["vjp_l2"]
    )
    if expected_normwise == 0.0:
        eta: float | str = 0.0 if expected_error == 0.0 else "infinity"
        beta: float | str = 0.0 if expected_error == 0.0 else "infinity"
    else:
        eta, beta = expected_error / expected_normwise, 2.0 * expected_error / expected_normwise
    if (
        entry["absolute_error"] != expected_error
        or entry["legacy_denominator"] != expected_legacy
        or entry["legacy_relative_error"] != expected_error / expected_legacy
        or entry["normwise_denominator"] != expected_normwise
        or entry["eta_norm"] != eta
        or entry["beta_norm"] != beta
    ):
        raise ValueError("fixture derived metric differs")
    for prefix in ("lhs", "rhs"):
        expected = _cancellation_factor(entry[f"{prefix}_absolute_product_sum"], entry[prefix])
        if entry[f"{prefix}_cancellation_factor"] != expected:
            raise ValueError("fixture cancellation factor differs")
    _hex(entry["jvp_sha256"], 64, name="fixture JVP hash")
    _hex(entry["vjp_sha256"], 64, name="fixture VJP hash")
    if entry["threshold"] != THRESHOLD or type(entry["threshold"]) is not float:
        raise ValueError("fixture threshold differs")
    controls = entry["controls"]
    if fault:
        controls = _exact_mapping(controls, ("unmodified",), name="fault controls")
        controls_pass = _validate_control(controls["unmodified"], relation=None)
        control_beta = controls["unmodified"]["beta_norm"]
    else:
        controls = _exact_mapping(
            controls,
            ("rebuild", "reversed_action_order", "parameter_sign", "output_sign"),
            name="correct controls",
        )
        controls_pass = all(
            (
                _validate_control(controls["rebuild"], relation="exact_action_hash_match"),
                _validate_control(
                    controls["reversed_action_order"], relation="exact_action_hash_match"
                ),
                _validate_control(controls["parameter_sign"], relation="exact_relation"),
                _validate_control(controls["output_sign"], relation="exact_relation"),
            )
        )
    if fault:
        expected_passed = (
            type(beta) is float
            and beta >= THRESHOLD
            and beta - control_beta >= 7.0 * THRESHOLD / 8.0
            and controls_pass
        )
    else:
        expected_passed = type(beta) is float and beta <= CORRECT_FIXTURE_CEILING and controls_pass
    if type(entry["passed"]) is not bool or entry["passed"] is not expected_passed:
        raise ValueError("fixture passed differs")
    return entry["passed"]
