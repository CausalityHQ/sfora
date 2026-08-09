"""Candidate-independent normwise adjoint diagnostic arithmetic."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    metadata: dict[str, object]

    @property
    def kind(self) -> str:
        return str(self.metadata["kind"])


@dataclass(frozen=True)
class FaultSpec:
    fixture_id: str
    metadata: dict[str, object]

    @property
    def kind(self) -> str:
        return str(self.metadata["kind"])


def correct_fixture_specs() -> tuple[FixtureSpec, ...]:
    return tuple(FixtureSpec(name, fixture_metadata(name)) for name in CORRECT_FIXTURE_IDS)


def registered_fault_specs() -> tuple[FaultSpec, ...]:
    return tuple(FaultSpec(name, fixture_metadata(name)) for name in REGISTERED_FAULT_IDS)


def _draw(seed: str, shape: tuple[int, ...], *, scale: float = 1.0) -> torch.Tensor:
    array = np.random.Generator(np.random.PCG64(int(seed, 16))).standard_normal(shape)
    return torch.as_tensor(np.ascontiguousarray(array * scale), dtype=torch.float32)


def _construct_correct_fixture(spec: FixtureSpec) -> dict[str, object]:
    if spec.fixture_id not in CORRECT_FIXTURE_IDS or not _literal_equal(
        spec.metadata, fixture_metadata(spec.fixture_id)
    ):
        raise ValueError("correct fixture metadata differs from protocol")
    seeds = spec.metadata["seeds"]
    tensors: dict[str, torch.Tensor]
    if spec.fixture_id == "zero_corner":
        tensors = {
            "x": _draw(seeds["x"], (17,)),
            "u": _draw(seeds["output_direction"], (17,)),
            "v": _draw(seeds["parameter_direction"], (17,)),
        }
    elif spec.kind == "affine_linear":
        tensors = {
            "matrix": _draw(seeds["matrix"], (257, 193)),
            "x": _draw(seeds["input"], (193,)),
            "u": _draw(seeds["output_direction"], (257,)),
            "v": _draw(seeds["parameter_direction"], (193,)),
        }
    elif spec.fixture_id == "smooth_parameter_tree":
        shapes = {"w1": (23, 11), "b1": (23,), "w2": (19, 23), "b2": (19,)}
        tensors = {
            name: _draw(seeds[name], shape, scale=2**-3 if name.startswith("w") else 2**-4)
            for name, shape in shapes.items()
        }
        tensors["input"] = _draw(seeds["input"], (17, 11), scale=2**-2)
        flat = _draw(seeds["parameter_direction"], (732,))
        start = 0
        for name, shape in shapes.items():
            count = math.prod(shape)
            tensors[f"v_{name}"] = flat[start : start + count].reshape(shape)
            start += count
        tensors["u"] = _draw(seeds["output_direction"], (17, 19))
    else:
        q = _draw(seeds["output_pair_base"], (4096,))
        p = _draw(seeds["parameter_pair_base"], (4096,))
        tensors = {
            "x": _draw(seeds["input"], (8193,)),
            "u": torch.cat((q.repeat_interleave(2), torch.ones(1, dtype=torch.float32))),
            "v": torch.cat((p.repeat_interleave(2), torch.ones(1, dtype=torch.float32))),
            "diagonal": torch.cat(
                (
                    torch.tensor([2**10, -(2**10)], dtype=torch.float32).repeat(4096),
                    torch.tensor([2**-10], dtype=torch.float32),
                )
            ),
        }
    return {"spec": spec, "tensors": tensors}


def _run_fixture_trial(
    spec: FixtureSpec,
    *,
    parameter_sign: int = 1,
    output_sign: int = 1,
    reversed_action_order: bool = False,
) -> tuple[dict[str, object], torch.Tensor, Mapping[str, torch.Tensor], tuple[str, ...]]:
    built = _construct_correct_fixture(spec)
    tensors = built["tensors"]
    if spec.fixture_id == "smooth_parameter_tree":
        names = ("w1", "b1", "w2", "b2")
        primal = {name: tensors[name] for name in names}
        tangent = {name: parameter_sign * tensors[f"v_{name}"] for name in names}
        fixed_input = tensors["input"]

        def function(parameters: Mapping[str, torch.Tensor]) -> torch.Tensor:
            hidden = torch.tanh(fixed_input @ parameters["w1"].T + parameters["b1"])
            output = hidden @ parameters["w2"].T + parameters["b2"]
            return torch.nn.functional.normalize(output, dim=1, eps=1.0e-12)

        _, pullback = torch.func.vjp(function, primal)
        if reversed_action_order:
            (vjp_action,) = pullback(output_sign * tensors["u"])
            _, jvp_action = torch.func.jvp(function, (primal,), (tangent,))
        else:
            _, jvp_action = torch.func.jvp(function, (primal,), (tangent,))
            (vjp_action,) = pullback(output_sign * tensors["u"])
        parameter_direction = tangent
    else:
        names = ("x",)
        primal = tensors["x"]
        tangent_tensor = parameter_sign * tensors["v"]
        if spec.fixture_id == "zero_corner":

            def function(value: torch.Tensor) -> torch.Tensor:
                return value * torch.tensor(0.0, dtype=torch.float32)

        elif spec.kind == "affine_linear":
            scale = {
                "affine_scale_2m12": 2**-12,
                "affine_scale_1": 1.0,
                "affine_scale_2p12": 2**12,
            }[spec.fixture_id]
            matrix = tensors["matrix"]

            def function(value: torch.Tensor) -> torch.Tensor:
                return torch.tensor(scale, dtype=torch.float32) * (matrix @ value)

        else:
            diagonal = tensors["diagonal"]

            def function(value: torch.Tensor) -> torch.Tensor:
                return diagonal * value

        _, pullback = torch.func.vjp(function, primal)
        if reversed_action_order:
            (vjp_tensor,) = pullback(output_sign * tensors["u"])
            _, jvp_action = torch.func.jvp(function, (primal,), (tangent_tensor,))
        else:
            _, jvp_action = torch.func.jvp(function, (primal,), (tangent_tensor,))
            (vjp_tensor,) = pullback(output_sign * tensors["u"])
        parameter_direction = {"x": tangent_tensor}
        vjp_action = {"x": vjp_tensor}

    metrics = normwise_adjoint_metrics(
        output_sign * tensors["u"], jvp_action, parameter_direction, vjp_action, names
    )
    threshold = metrics.pop("threshold")
    metrics.pop("passed")
    result = {
        **fixture_metadata(spec.fixture_id),
        **metrics,
        "jvp_sha256": tensor_sha256(jvp_action),
        "vjp_sha256": parameter_tree_sha256(vjp_action, names),
        "controls": {},
        "threshold": threshold,
        "passed": type(metrics["beta_norm"]) is float
        and metrics["beta_norm"] <= CORRECT_FIXTURE_CEILING,
    }
    return result, jvp_action, vjp_action, names


def run_correct_fixture(spec: FixtureSpec) -> dict[str, object]:
    """Execute one frozen correct fixture through real torch.func actions."""
    result, _, _, _ = _run_fixture_trial(spec)
    return result


def _tree_equal(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], names: Sequence[str]
) -> bool:
    return all(torch.equal(left[name], right[name]) for name in names)


def run_fixture_controls(spec: FixtureSpec) -> dict[str, object]:
    """Run the frozen rebuild, action-order, and sign controls on fresh graphs."""
    baseline, baseline_jvp, baseline_vjp, names = _run_fixture_trial(spec)
    rebuild, rebuild_jvp, rebuild_vjp, _ = _run_fixture_trial(spec)
    reversed_trial, reversed_jvp, reversed_vjp, _ = _run_fixture_trial(
        spec, reversed_action_order=True
    )
    parameter_trial, parameter_jvp, parameter_vjp, _ = _run_fixture_trial(spec, parameter_sign=-1)
    output_trial, output_jvp, output_vjp, _ = _run_fixture_trial(spec, output_sign=-1)

    baseline_jvp_hash = tensor_sha256(baseline_jvp)
    baseline_vjp_hash = parameter_tree_sha256(baseline_vjp, names)

    def hash_control(
        trial: Mapping[str, object],
        jvp: torch.Tensor,
        vjp: Mapping[str, torch.Tensor],
    ) -> dict[str, object]:
        jvp_hash = tensor_sha256(jvp)
        vjp_hash = parameter_tree_sha256(vjp, names)
        exact = jvp_hash == baseline_jvp_hash and vjp_hash == baseline_vjp_hash
        return {
            "jvp_sha256": jvp_hash,
            "vjp_sha256": vjp_hash,
            "beta_norm": trial["beta_norm"],
            "exact_action_hash_match": exact,
            "passed": exact
            and type(trial["beta_norm"]) is float
            and trial["beta_norm"] <= CORRECT_FIXTURE_CEILING,
        }

    def sign_control(
        trial: Mapping[str, object],
        jvp: torch.Tensor,
        vjp: Mapping[str, torch.Tensor],
        *,
        relation: bool,
    ) -> dict[str, object]:
        return {
            "jvp_sha256": tensor_sha256(jvp),
            "vjp_sha256": parameter_tree_sha256(vjp, names),
            "beta_norm": trial["beta_norm"],
            "exact_relation": relation,
            "passed": relation
            and type(trial["beta_norm"]) is float
            and trial["beta_norm"] <= CORRECT_FIXTURE_CEILING,
        }

    controls = {
        "rebuild": hash_control(rebuild, rebuild_jvp, rebuild_vjp),
        "reversed_action_order": hash_control(reversed_trial, reversed_jvp, reversed_vjp),
        "parameter_sign": sign_control(
            parameter_trial,
            parameter_jvp,
            parameter_vjp,
            relation=torch.equal(parameter_jvp, -baseline_jvp)
            and _tree_equal(parameter_vjp, baseline_vjp, names),
        ),
        "output_sign": sign_control(
            output_trial,
            output_jvp,
            output_vjp,
            relation=torch.equal(output_jvp, baseline_jvp)
            and all(torch.equal(output_vjp[name], -baseline_vjp[name]) for name in names),
        ),
    }
    baseline["controls"] = controls
    baseline["passed"] = baseline["passed"] is True and all(
        control["passed"] is True for control in controls.values()
    )
    return baseline


def run_registered_fault(spec: FaultSpec) -> dict[str, object]:
    """Run one frozen fault after proving its unmodified action is adjoint-correct."""
    if spec.fixture_id not in REGISTERED_FAULT_IDS or not _literal_equal(
        spec.metadata, fixture_metadata(spec.fixture_id)
    ):
        raise ValueError("registered fault metadata differs from protocol")
    seeds = spec.metadata["seeds"]
    names = ("x",)

    if spec.fixture_id == "zero_map_forward_injection":
        primal = _draw(seeds["x"], (17,))
        u = _draw(seeds["output_direction"], (17,))
        v = _draw(seeds["parameter_direction"], (17,))

        def function(value: torch.Tensor) -> torch.Tensor:
            return value * torch.tensor(0.0, dtype=torch.float32)

        _, correct_jvp = torch.func.jvp(function, (primal,), (v,))
        _, pullback = torch.func.vjp(function, primal)
        (correct_vjp_tensor,) = pullback(u)
        fault_jvp = torch.tensor(2**-10, dtype=torch.float32) * u
        fault_vjp_tensor = correct_vjp_tensor
    else:
        q = _draw(
            seeds["shared_direction"]
            if spec.fixture_id == "identity_reverse_scale_fault"
            else seeds["pair_base"],
            (4096 if spec.fixture_id == "identity_reverse_scale_fault" else 2048,),
        )
        if spec.fixture_id == "identity_reverse_pair_sign_fault":
            q = q.repeat_interleave(2)
        primal = torch.zeros_like(q)
        u = q
        v = q

        def function(value: torch.Tensor) -> torch.Tensor:
            return value

        _, correct_jvp = torch.func.jvp(function, (primal,), (v,))
        _, pullback = torch.func.vjp(function, primal)
        (correct_vjp_tensor,) = pullback(u)
        fault_jvp = correct_jvp
        if spec.fixture_id == "identity_reverse_scale_fault":
            fault_vjp_tensor = torch.tensor(255 / 256, dtype=torch.float32) * q
        else:
            base = q[::2]
            fault_vjp_tensor = torch.stack((base, -base), dim=1).reshape(-1)

    direction = {"x": v}
    correct_vjp = {"x": correct_vjp_tensor}
    fault_vjp = {"x": fault_vjp_tensor}
    correct = normwise_adjoint_metrics(u, correct_jvp, direction, correct_vjp, names)
    fault = normwise_adjoint_metrics(u, fault_jvp, direction, fault_vjp, names)
    threshold = fault.pop("threshold")
    fault.pop("passed")
    control_beta = correct["beta_norm"]
    control_passed = type(control_beta) is float and control_beta <= CORRECT_FIXTURE_CEILING
    result = {
        **fixture_metadata(spec.fixture_id),
        **fault,
        "jvp_sha256": tensor_sha256(fault_jvp),
        "vjp_sha256": parameter_tree_sha256(fault_vjp, names),
        "controls": {
            "unmodified": {
                "jvp_sha256": tensor_sha256(correct_jvp),
                "vjp_sha256": parameter_tree_sha256(correct_vjp, names),
                "beta_norm": control_beta,
                "passed": control_passed,
            }
        },
        "threshold": threshold,
        "passed": type(fault["beta_norm"]) is float
        and fault["beta_norm"] >= THRESHOLD
        and fault["beta_norm"] - control_beta >= 7.0 * THRESHOLD / 8.0
        and control_passed,
    }
    return result


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
        if (
            any(
                controls[name][hash_name] != entry[hash_name]
                for name in ("rebuild", "reversed_action_order")
                for hash_name in ("jvp_sha256", "vjp_sha256")
            )
            or controls["parameter_sign"]["vjp_sha256"] != entry["vjp_sha256"]
            or controls["output_sign"]["jvp_sha256"] != entry["jvp_sha256"]
        ):
            raise ValueError("control action hash differs from baseline")
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
