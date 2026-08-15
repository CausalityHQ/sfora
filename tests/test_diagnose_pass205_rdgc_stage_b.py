"""Synthetic CPU tests for the prospective Pass 205 RDGC diagnostic."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
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
_VERIFIER_SCRIPT = _ROOT / "scripts/verify_pass200_rsta_scientific_artifact.py"
_VERIFIER_SPEC = importlib.util.spec_from_file_location(
    "verify_pass200_rsta_scientific_artifact_for_rdgc_test", _VERIFIER_SCRIPT
)
assert _VERIFIER_SPEC is not None and _VERIFIER_SPEC.loader is not None
_VERIFIER = importlib.util.module_from_spec(_VERIFIER_SPEC)
_VERIFIER_SPEC.loader.exec_module(_VERIFIER)
_RSTA_SCRIPT = _ROOT / "scripts/diagnose_pass200_rsta_stage_a.py"
_RSTA_SPEC = importlib.util.spec_from_file_location(
    "diagnose_pass200_rsta_stage_a_for_rdgc_test", _RSTA_SCRIPT
)
assert _RSTA_SPEC is not None and _RSTA_SPEC.loader is not None
_RSTA = importlib.util.module_from_spec(_RSTA_SPEC)
sys.modules[_RSTA_SPEC.name] = _RSTA
_RSTA_SPEC.loader.exec_module(_RSTA)


def test_repair_authority_literals_are_consumed_by_the_source_gate() -> None:
    assert (
        _MODULE.ORIGINAL_PLAN_PATH,
        _MODULE.ORIGINAL_PLAN_COMMIT,
        _MODULE.ORIGINAL_PLAN_SHA256,
    ) == (
        "docs/superpowers/plans/2026-08-10-pass205-rdgc-stage-b.md",
        "c1e49b13c08f853ae17d5b8b48be1aa7b8a4bc11",
        "20915982228bd4a17f1260952fe184d9e09b27b9b28165b5931bad843872c7ed",
    )
    assert (
        _MODULE.AUTHORITY_REPAIR_PLAN_PATH,
        _MODULE.AUTHORITY_REPAIR_PLAN_COMMIT,
        _MODULE.AUTHORITY_REPAIR_PLAN_SHA256,
    ) == (
        "docs/superpowers/plans/2026-08-11-pass205-rdgc-authority-repair.md",
        "4c72bc65e964cb863f9b4abf83bcdf0d38e7165a",
        "3893dd02f18afccd0bc3373789e6896fd4d1add53df3b324dfa1eb195ce13412",
    )
    assert (
        _MODULE.RUNTIME_AMENDMENT_PATH,
        _MODULE.RUNTIME_AMENDMENT_COMMIT,
        _MODULE.RUNTIME_AMENDMENT_SHA256,
    ) == (
        "docs/pass205_rdgc_dgx_runtime_amendment_2026-08-11.md",
        "29f0600d64d92d931ab2f57e04a59d9daba209d6",
        "eb18908fc8a514e5ac3f0deb67950eeaf2a256ad20e1ae594e4fbd6fb2f74df0",
    )
    assert (
        _MODULE.PLAN_PATH,
        _MODULE.PLAN_COMMIT,
        _MODULE.PLAN_SHA256,
    ) == (
        "docs/superpowers/plans/2026-08-11-pass205-rdgc-dgx-runtime-repair.md",
        "aa978a90a43bf2c8de25b001aadffcd44073e4e6",
        "668926b41389d90d87bf7f8717ce36903745ed146d46ae063b6d8b522d0d4cfd",
    )
    assert _MODULE.REPAIR_DESIGN_COMMIT == "2f2ea249a754a1fb4186ba55939d95c85de747a8"
    assert _MODULE.LITERATURE_AUDIT_COMMIT == "9ae137f3af0558728554c6af865fe96d6bf10060"
    assert _MODULE.AUTHORITY_AMENDMENT_COMMIT == "c7fae7683533e740660d7e860bd313be07a41014"


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


def test_entry_configures_runtime_before_artifact_hash_integrity_and_seed_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class StopAfterFirstSeed(RuntimeError):
        pass

    class RstaStub:
        @staticmethod
        def configure_deterministic_process() -> dict[str, object]:
            events.append("configure")
            return {}

        @staticmethod
        def load_training_only_seed(
            _entry: object,
            _receipt_seed: object,
            *,
            expected_dimension: int,
        ) -> object:
            assert expected_dimension == 17
            events.append("load_seed")
            raise StopAfterFirstSeed

    integrity_seeds = [
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
            "action_hashes_sha256": hashlib.sha256(f"seed-{seed}".encode()).hexdigest(),
            "passed": True,
        }
        for seed in range(4)
    ]

    def integrity_prefix(**_kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        events.append("integrity")
        return integrity_seeds, {}

    monkeypatch.setattr(_MODULE, "_integrity_prefix_from_rsta", integrity_prefix)

    def sha256_file(_path: Path) -> str:
        events.append("sha256_file")
        return "a" * 64

    monkeypatch.setattr(_MODULE, "_sha256_file", sha256_file)
    monkeypatch.setattr(
        _MODULE,
        "build_rdgc_selection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selection constructed before all-four integrity")
        ),
    )
    old_manifest = {
        "binding_receipt": {"path": "binding.json"},
        "seeds": {str(seed): {} for seed in range(4)},
    }
    receipt = types.SimpleNamespace(seeds=tuple(object() for _ in range(4)))
    manifest = {
        "historical": {
            "manifest_path": "old-manifest.json",
            "manifest_sha256": "a" * 64,
        }
    }
    authority = {
        "pass200_module": RstaStub,
        "validated_historical_manifest": old_manifest,
        "validated_binding_receipt": receipt,
    }
    (tmp_path / "old-manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "binding.json").write_text("{}", encoding="utf-8")

    with pytest.raises(StopAfterFirstSeed):
        _MODULE.run_rdgc_scientific_once(
            repository=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            output_path=tmp_path / "output.json",
            manifest=manifest,
            authority=authority,
            started_utc="2026-08-11T00:00:00Z",
            command=["registered"],
            expected_dimension=17,
        )
    assert events == ["configure", "sha256_file", "integrity", "load_seed"]


def test_real_historical_runtime_boundary_fails_fresh_then_passes_after_configuration() -> None:
    program = f"""
import importlib.util
import os
import sys
from pathlib import Path

path = Path({str(_RSTA_SCRIPT)!r})
spec = importlib.util.spec_from_file_location("rsta_runtime_boundary_probe", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
try:
    module._assert_deterministic_tf32_off()
except ValueError:
    pass
else:
    raise AssertionError("fresh process unexpectedly satisfied deterministic boundary")
module.configure_deterministic_process()
module._assert_deterministic_tf32_off()
"""
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_selection_state_is_constructed_only_after_all_four_integrity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class SelectionReached(RuntimeError):
        pass

    class RstaStub:
        @staticmethod
        def configure_deterministic_process() -> dict[str, object]:
            events.append("configure")
            return {}

        @staticmethod
        def load_training_only_seed(
            _entry: object,
            _receipt_seed: object,
            *,
            expected_dimension: int,
        ) -> object:
            assert expected_dimension == 17
            events.append("load_seed")
            return types.SimpleNamespace(
                train_example_ids=np.asarray(["id-0", "id-1"]),
                train_labels=np.asarray([0, 1]),
            )

        @staticmethod
        def validate_cross_seed_training_binding(_bounds: object) -> None:
            events.append("validate_binding")

        @staticmethod
        def select_primary_panel(_ids: object, _labels: object) -> dict[str, object]:
            events.append("old_selection")
            return {}

    integrity_seeds = [
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
            "action_hashes_sha256": hashlib.sha256(f"seed-{seed}".encode()).hexdigest(),
            "passed": True,
        }
        for seed in range(4)
    ]

    def integrity_prefix(**_kwargs: object) -> tuple[list[dict[str, object]], dict[str, object]]:
        events.append("integrity")
        return integrity_seeds, {}

    def build_selection(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("selection")
        raise SelectionReached

    monkeypatch.setattr(_MODULE, "_integrity_prefix_from_rsta", integrity_prefix)

    def sha256_file(_path: Path) -> str:
        events.append("sha256_file")
        return "a" * 64

    monkeypatch.setattr(_MODULE, "_sha256_file", sha256_file)
    monkeypatch.setattr(_MODULE, "build_rdgc_selection", build_selection)
    old_manifest = {
        "binding_receipt": {"path": "binding.json"},
        "seeds": {str(seed): {} for seed in range(4)},
    }
    authority = {
        "pass200_module": RstaStub,
        "validated_historical_manifest": old_manifest,
        "validated_binding_receipt": types.SimpleNamespace(
            seeds=tuple(object() for _ in range(4))
        ),
    }
    (tmp_path / "old-manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "binding.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SelectionReached):
        _MODULE.run_rdgc_scientific_once(
            repository=tmp_path,
            manifest_path=tmp_path / "manifest.json",
            output_path=tmp_path / "output.json",
            manifest={
                "historical": {
                    "manifest_path": "old-manifest.json",
                    "manifest_sha256": "a" * 64,
                }
            },
            authority=authority,
            started_utc="2026-08-11T00:00:00Z",
            command=["registered"],
            expected_dimension=17,
        )
    assert events == [
        "configure",
        "sha256_file",
        "integrity",
        "load_seed",
        "load_seed",
        "load_seed",
        "load_seed",
        "validate_binding",
        "old_selection",
        "selection",
    ]


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


def _preliminary_close_evidence(*, nonpositive_correlations: int = 0) -> dict[str, object]:
    return {
        "context_spearman_nonpositive_seed_count": nonpositive_correlations,
    }


def _preliminary_decision_aggregates() -> dict[str, object]:
    return {**_preliminary_aggregates(), "_close_evidence": _preliminary_close_evidence()}


def _panel_aggregates() -> dict[str, object]:
    metric = lambda pooled=0.04, lower=0.01, positive=4: {  # noqa: E731
        "pooled_difference": pooled,
        "lower_bound": lower,
        "positive_seed_means": positive,
        "nonpositive_seed_means": 4 - positive,
        "seed_means_ge_point_zero_one": positive,
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
    controls = {
        "rdgc": _MODULE.rdgc_penalty(torch, b, s),
        "raw_cotangent": _MODULE.raw_cotangent_penalty(torch, b, dbar),
        "full_motion": _MODULE.full_motion_penalty(torch, b),
        "batch_global_gain": _MODULE.batch_global_gain_penalty(torch, b, receiver_fields),
        "scalar_diagonal_raw": _MODULE.scalar_diagonal_raw_penalty(
            torch, b, dbar, receiver_fields
        ),
        "per_example_gradient_normalized": _MODULE.per_example_gradient_normalized_penalty(
            torch, pgn, s
        ),
    }
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
    result = _MODULE.batch_global_gain_penalty(
        torch, _finite_tensor([8.0], requires_grad=True), fields
    )
    target = math.exp(sum(math.log(float(2**index) + 1e-8) for index in range(8)) / 8)
    assert result.item() == pytest.approx(
        0.5 * math.log((8.0 + 1e-8) / target) ** 2, rel=1e-6
    )


def test_scalar_diagonal_raw_uses_batch_gain_times_each_raw_norm() -> None:
    fields = [
        {"s": _finite_tensor([float(index + 2)]), "dbar": _finite_tensor([float(index + 1)])}
        for index in range(8)
    ]
    result = _MODULE.scalar_diagonal_raw_penalty(
        torch, _finite_tensor([5.0], requires_grad=True), fields[3]["dbar"], fields
    )
    gain = math.exp(
        sum(math.log((float(index + 2) + 1e-8) / (float(index + 1) + 1e-8)) for index in range(8))
        / 8
    )
    target = gain * (4.0 + 1e-8)
    assert result.item() == pytest.approx(
        0.5 * math.log((5.0 + 1e-8) / target) ** 2, rel=1e-6
    )


def test_batch_and_scalar_targets_do_not_receive_an_unregistered_second_epsilon() -> None:
    fields = [
        {"s": _finite_tensor([2.0e-8]), "dbar": _finite_tensor([1.0e-8])}
        for _ in range(8)
    ]
    b = _finite_tensor([5.0e-8], requires_grad=True)
    target = 3.0e-8
    expected = 0.5 * math.log((5.0e-8 + 1.0e-8) / target) ** 2
    assert _MODULE.batch_global_gain_penalty(torch, b, fields).item() == pytest.approx(
        expected, rel=1e-6
    )
    assert _MODULE.scalar_diagonal_raw_penalty(
        torch, b, fields[0]["dbar"], fields
    ).item() == pytest.approx(expected, rel=1e-6)


def test_correction_dispatch_constructs_only_literal_requested_penalty_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names = {
        "rdgc": "rdgc_penalty",
        "raw_cotangent": "raw_cotangent_penalty",
        "full_motion": "full_motion_penalty",
        "batch_global_gain": "batch_global_gain_penalty",
        "scalar_diagonal_raw": "scalar_diagonal_raw_penalty",
        "per_example_gradient_normalized": "per_example_gradient_normalized_penalty",
    }
    events: list[str] = []
    theta = _finite_tensor([2.0], requires_grad=True)

    def constructor(name: str):
        def build(*_args: object, **_kwargs: object) -> torch.Tensor:
            events.append(name)
            return theta.square().sum()

        return build

    for operator, function_name in names.items():
        monkeypatch.setattr(_MODULE, function_name, constructor(operator))
    field = {
        "named_parameters": (("weight", theta),),
        "p": (_finite_tensor([1.0]),),
        "b": _finite_tensor([1.0]),
        "s": _finite_tensor([1.0]),
        "dbar": _finite_tensor([1.0]),
        "receiver_fields": [
            {"s": _finite_tensor([1.0]), "dbar": _finite_tensor([1.0])}
            for _ in range(8)
        ],
        "pgn_motion": _finite_tensor([1.0]),
    }
    for operator in names:
        events.clear()
        result = _MODULE.compute_parameter_correction(field, operator, torch_module=torch)
        assert len(result) == 1
        assert events == [operator]
    events.clear()
    minimal_rdgc = {
        "named_parameters": (("weight", theta),),
        "p": (_finite_tensor([1.0]),),
        "b": _finite_tensor([1.0]),
        "s": _finite_tensor([1.0]),
    }
    _MODULE.compute_parameter_correction(minimal_rdgc, "rdgc", torch_module=torch)
    assert events == ["rdgc"]


def test_scientific_rdgc_builds_only_global_and_target_self_actions() -> None:
    theta = _finite_tensor([1.5, -0.5], requires_grad=True)
    parameters = {"weight": theta}

    def encoder(values: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [values["weight"] * float(index + 1) for index in range(8)], dim=0
        )

    _z, actual_vjp = torch.func.vjp(encoder, parameters)
    vjp_calls = 0

    def counted_vjp(cotangent: torch.Tensor):
        nonlocal vjp_calls
        vjp_calls += 1
        return actual_vjp(cotangent)

    context = {
        "parameter_names": ("weight",),
        "parameters": parameters,
        "dbar": _finite_tensor([[0.1, 0.2]] * 8),
        "vjp_function": counted_vjp,
        "encoder": encoder,
        "receiver_indices": tuple(range(8)),
        "device": theta.device,
    }
    result = _MODULE._parameter_correction_science(
        operator="rdgc",
        receiver_index=0,
        context=context,
        p=(theta.detach(),),
        pgn_coefficients=None,
        torch_module=torch,
    )
    assert len(result) == 1 and vjp_calls == 2


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
    result = _MODULE.full_motion_penalty(torch, b)
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
    valid = _preliminary_decision_aggregates()
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
    }
    for key, value in mutations.items():
        case = deepcopy(valid)
        case[key] = value
        assert _MODULE.decide_preliminary(case)["status"] == "UNRESOLVED", key


def test_preliminary_decision_keeps_the_frozen_single_argument_interface() -> None:
    assert tuple(inspect.signature(_MODULE.decide_preliminary).parameters) == ("aggregates",)


def test_preliminary_close_precedence_and_exact_boundaries() -> None:
    case = _preliminary_decision_aggregates()
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
    ):
        exact = _preliminary_decision_aggregates()
        exact[key] = boundary
        exact["_close_evidence"] = _preliminary_close_evidence()
        assert _MODULE.decide_preliminary(exact)["status"] == "CLOSE", key


def test_preliminary_middle_region_is_unresolved() -> None:
    case = _preliminary_decision_aggregates()
    case["global_scalar_relative_error_median_A"] = 0.03
    assert _MODULE.decide_preliminary(case) == {
        "status": "UNRESOLVED",
        "first_decisive_clause": "no_close_or_survival_rule",
        "full_panel_authorized": False,
    }


def test_repair_removes_direct_full_gain_decisions_but_retains_descriptive_metrics() -> None:
    summary = _MODULE.summarize_preliminary_rows(
        _preliminary_decision_rows(full_gain_seed_values=(0.0,) * 4)
    )
    assert tuple(summary["predicates"]) == (
        "survives_count_gain",
        "survives_context_stability",
        "survives_receiver_heterogeneity",
        "survives_global_scalar",
        "close_count_gain",
        "close_context_stability",
        "close_receiver_heterogeneity",
        "close_global_scalar",
    )
    assert "full_gain_error_median_A" in summary["pooled_aggregates"]
    assert "full_gain_error_median_B" in summary["pooled_aggregates"]
    assert summary["decision"]["first_decisive_clause"] != "close_full_gain"


def _preliminary_decision_rows(
    *,
    count_gain: float = 1.0,
    log_kappa_step: float = 0.30,
    full_gain_seed_values: tuple[float, float, float, float] | None = None,
) -> list[dict[str, object]]:
    if full_gain_seed_values is None:
        full_gain_seed_values = (math.log(1.30),) * 4
    rows: list[dict[str, object]] = []
    for seed in range(4):
        for context in ("A", "B"):
            for receiver_label in range(8):
                rows.append(
                    {
                        "seed": seed,
                        "context": context,
                        "receiver_label": receiver_label,
                        "log_kappa": log_kappa_step * receiver_label,
                        "count_gain": count_gain,
                        "absolute_log_gain_errors_by_contributor_count": {
                            "1": 1.0,
                            "8": 1.0,
                            "32": 1.0,
                            "180": full_gain_seed_values[seed],
                        },
                    }
                )
    return rows


def test_preliminary_full_gain_endpoint_is_descriptive_only() -> None:
    rows = _preliminary_decision_rows(
        full_gain_seed_values=(0.0, math.log(1.04), math.log(1.06), math.log(1.08))
    )
    summary = _MODULE.summarize_preliminary_rows(rows)
    assert summary["pooled_aggregates"]["full_gain_error_median_A"] <= math.log(1.05)
    assert summary["decision"]["status"] == "SURVIVES"
    assert "survives_full_gain" not in summary["predicates"]
    assert "close_full_gain" not in summary["predicates"]


def test_preliminary_predicates_record_every_true_close_condition_not_only_first() -> None:
    summary = _MODULE.summarize_preliminary_rows(
        _preliminary_decision_rows(
            count_gain=-1.0,
            log_kappa_step=0.001,
            full_gain_seed_values=(0.0, 0.0, 0.0, 0.0),
        )
    )
    assert summary["decision"]["first_decisive_clause"] == "close_count_gain"
    assert summary["predicates"]["close_count_gain"] is True
    assert summary["predicates"]["close_receiver_heterogeneity"] is True
    assert summary["predicates"]["close_global_scalar"] is True
    assert "close_full_gain" not in summary["predicates"]


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


def test_panel_primary_alignment_requires_three_seed_means_at_point_zero_one() -> None:
    case = _panel_aggregates()
    case["primary_pa_alignment"]["seed_means_ge_point_zero_one"] = 2
    assert _MODULE.decide_panel(case, {})["status"] == "UNRESOLVED"


def test_panel_pass_action_is_the_frozen_training_preregistration_action() -> None:
    assert _MODULE.decide_panel(_panel_aggregates(), {})["authorized_action"] == (
        "new_training_preregistration_only"
    )


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


def test_panel_close_precedence_is_by_frozen_clause_before_control_order() -> None:
    case = _panel_aggregates()
    case["correction_aliases"]["raw_cotangent"].update(
        pooled_median_absolute_cosine=0.99,
        seed_medians_ge_point_nine_nine=3,
    )
    case["controls"]["layerwise_trust_ratio"]["primary_alignment"].update(
        pooled_difference=0.0,
        positive_seed_means=1,
        nonpositive_seed_means=3,
    )
    assert _MODULE.decide_panel(case, {})["first_decisive_clause"] == (
        "close_layerwise_trust_ratio_primary_alignment"
    )

    case = _panel_aggregates()
    case["controls"]["batch_global_gain"]["primary_slope"].update(
        pooled_difference=0.0,
        positive_seed_means=1,
        nonpositive_seed_means=3,
    )
    case["controls"]["layerwise_trust_ratio"]["primary_alignment"].update(
        pooled_difference=0.0,
        positive_seed_means=1,
        nonpositive_seed_means=3,
    )
    assert _MODULE.decide_panel(case, {})["first_decisive_clause"] == (
        "close_layerwise_trust_ratio_primary_alignment"
    )


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
        "python_version": "3.13.9",
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


def test_version_fields_remain_observational_builtin_strings() -> None:
    class VersionText(str):
        pass

    fields = {
        "python_executable": ".venv/bin/python",
        "python_version": "3.13.9",
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
        __version__=VersionText("2.5.1"),
        version=types.SimpleNamespace(cuda=VersionText("12.4")),
        backends=types.SimpleNamespace(
            cudnn=types.SimpleNamespace(version=lambda: 90100, allow_tf32=False),
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        ),
        are_deterministic_algorithms_enabled=lambda: True,
        cuda=types.SimpleNamespace(
            current_device=lambda: 0,
            get_device_name=lambda _index: VersionText("synthetic"),
            get_device_capability=lambda _index: (9, 0),
        ),
    )
    value = _MODULE.attach_observed_torch_runtime(pre, fake)
    assert value["phase"] == "post_import"
    assert type(value["pre_import"]["numpy_version"]) is str
    assert value["pre_import"]["numpy_version"] == "2.5.0"
    assert type(value["torch_runtime"]["torch_version"]) is str
    assert value["torch_runtime"]["torch_version"] == "2.5.1"
    assert type(value["torch_runtime"]["cuda_runtime_version"]) is str
    assert value["torch_runtime"]["cuda_runtime_version"] == "12.4"
    assert type(value["torch_runtime"]["device_name"]) is str
    assert value["torch_runtime"]["device_name"] == "synthetic"
    assert value["torch_runtime"]["device_capability"] == [9, 0]


def _bound_rows(count: int = 1_000) -> tuple[list[str], list[int]]:
    ids: list[str] = []
    labels: list[int] = []
    for label in range(count):
        for row in range(4):
            ids.append(json.dumps({"label": label, "row": row}, separators=(",", ":")))
            labels.append(label)
    return ids, labels


def _old_primary(ids: list[str], labels: list[int]) -> dict[str, object]:
    return _RSTA.select_primary_panel(ids, labels)


def test_selection_recomputes_old_64_and_freezes_fresh_8_then_32() -> None:
    ids, labels = _bound_rows()
    old = _old_primary(ids, labels)
    value = _MODULE.build_rdgc_selection(ids, labels, old_selection=old)
    assert len(value["preliminary"]["identity_labels"]) == 8
    assert len(value["panel"]["identity_labels"]) == 32
    chosen = value["preliminary"]["identity_labels"] + value["panel"]["identity_labels"]
    assert not set(old["labels"]).intersection(chosen)
    assert len(set(chosen)) == 40
    assert (
        value["old_rsta_exclusion_sha256"]
        == hashlib.sha256(
            b"".join(
                item.encode() + b"\n"
                for item in sorted(
                    {
                        *old["receiver_ids"],
                        *(
                            item
                            for label in old["labels"]
                            for item in old["support_ids_by_label"][label]
                        ),
                        *(item for block in old["distractor_blocks"] for item in block),
                    }
                )
            )
        ).hexdigest()
    )


def test_selection_bound_ids_are_exact_nonempty_unmodified_strings() -> None:
    ids, labels = _bound_rows()
    old = _old_primary(ids, labels)
    value = _MODULE.build_rdgc_selection(ids, labels, old_selection=old)
    selected_ids = value["preliminary"]["receiver_ids"] + value["panel"]["receiver_ids"]
    assert all(item in ids for item in selected_ids)
    for mutant in (0, True, ""):
        bad_ids = list(ids)
        bad_ids[300] = mutant  # type: ignore[list-item]
        with pytest.raises((TypeError, ValueError)):
            _MODULE.build_rdgc_selection(bad_ids, labels, old_selection=old)
    ndarray_value = _MODULE.build_rdgc_selection(
        np.asarray(ids),
        np.asarray(labels, dtype=np.int64),
        old_selection=old,
    )
    assert ndarray_value == value


def test_selection_support_receiver_roles_are_disjoint_and_deterministic() -> None:
    ids, labels = _bound_rows()
    old = _old_primary(ids, labels)
    first = _MODULE.build_rdgc_selection(ids, labels, old_selection=old)
    second = _MODULE.build_rdgc_selection(ids, labels, old_selection=old)
    assert first == second
    old_ids = {
        *old["receiver_ids"],
        *(item for label in old["labels"] for item in old["support_ids_by_label"][label]),
        *(item for block in old["distractor_blocks"] for item in block),
    }
    for phase in ("preliminary", "panel"):
        supports = {item for pair in first[phase]["support_ids_by_label"].values() for item in pair}
        receivers = set(first[phase]["receiver_ids"])
        distractors = {
            item
            for group in first[phase]["groups"]
            for context in group["contexts"]
            for item in context["batch_ids"]
            if item not in receivers
        }
        assert supports.isdisjoint(receivers)
        assert old_ids.isdisjoint(supports | receivers | distractors)


def test_selection_rejects_duplicate_length_and_insufficient_identity_rows() -> None:
    ids, labels = _bound_rows()
    old = _old_primary(ids, labels)
    for mutant_ids, mutant_labels in (
        (ids[:-1], labels),
        ([ids[0], *ids[1:-1], ids[0]], labels),
        (ids[: 100 * 4], labels[: 100 * 4]),
    ):
        with pytest.raises((TypeError, ValueError)):
            _MODULE.build_rdgc_selection(mutant_ids, mutant_labels, old_selection=old)


def test_selection_hash_plan_is_bound_to_authenticated_bytes_and_live_fp32_tensors() -> None:
    ids, labels = _bound_rows()
    selection = _MODULE.build_rdgc_selection(
        ids, labels, old_selection=_old_primary(ids, labels)
    )
    bounds = tuple(
        types.SimpleNamespace(
            seed=seed,
            checkpoint_bytes=f"checkpoint-{seed}".encode(),
            config={"seed": seed, "crop": 227},
            train_example_ids=ids,
            train_source_paths=[f"source/{index}.jpg" for index in range(len(ids))],
        )
        for seed in range(4)
    )

    class Cache:
        def __init__(self, bound: object, batch_ids: tuple[str, ...]) -> None:
            self.example_ids = batch_ids
            self.tensor_sha256 = {
                value: hashlib.sha256(
                    torch.tensor(
                        [float(bound.seed), float(ids.index(value))], dtype=torch.float32
                    )
                    .numpy()
                    .tobytes()
                ).hexdigest()
                for value in batch_ids
            }

    helper = types.SimpleNamespace(
        cache_seed_training_tensors=lambda bound, batch_ids: Cache(bound, tuple(batch_ids)),
        _json_ready=lambda value: value,
    )
    bound = _MODULE.bind_selection_context_hashes(selection, bounds, rsta_module=helper)
    hashes = [
        (context["transform_sha256"], context["tensor_sha256"])
        for phase in ("preliminary", "panel")
        for group in bound[phase]["groups"]
        for context in group["contexts"]
    ]
    assert len(hashes) == 10 and len(set(hashes)) == 10
    assert all(
        value not in {hashlib.sha256(b"transform").hexdigest(), "0" * 64}
        for pair in hashes
        for value in pair
    )
    changed_bounds = list(bounds)
    changed_bounds[0] = types.SimpleNamespace(**vars(bounds[0]))
    changed_bounds[0].checkpoint_bytes = b"changed-authenticated-checkpoint"
    changed = _MODULE.bind_selection_context_hashes(
        deepcopy(selection), tuple(changed_bounds), rsta_module=helper
    )
    assert changed["preliminary"]["groups"][0]["contexts"][0]["transform_sha256"] != hashes[0][0]


def test_pgn_fresh_graph_requires_exact_batch_input_descriptor_and_dbar_hashes() -> None:
    evidence = {
        "batch_sha256": "1" * 64,
        "input_sha256": "2" * 64,
        "descriptor_sha256": "3" * 64,
        "dbar_sha256": "4" * 64,
    }
    _MODULE.require_identical_pgn_graph_evidence(evidence, deepcopy(evidence))
    for key in evidence:
        mutant = deepcopy(evidence)
        mutant[key] = "f" * 64
        with pytest.raises(ValueError, match="PGN"):
            _MODULE.require_identical_pgn_graph_evidence(evidence, mutant)


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


def _real_roundtrip_receipt() -> dict[str, object]:
    value = {
        "schema_version": 1,
        "validation": "pass200-rsta-scientific-artifact-roundtrip",
        "mode": "offline_immutable_artifact",
        "attempt": 1,
        "status": "VALID",
        "outcome_disclosed": False,
        "artifact": {
            "path": _VERIFIER.ARTIFACT_PATH,
            "sha256": _VERIFIER.ARTIFACT_SHA256,
            "producer_pid": 1002393,
            "producer_exit_code": 0,
            "immutable": True,
        },
        "legacy_provenance": {
            "handoff_commit": _VERIFIER.LEGACY_HANDOFF_COMMIT,
            "source_commit": _VERIFIER.LEGACY_SOURCE_COMMIT,
            "manifest_path": _VERIFIER.LEGACY_MANIFEST_PATH,
            "manifest_sha256": _VERIFIER.LEGACY_MANIFEST_SHA256,
            "diagnostic_path": _VERIFIER.LEGACY_DIAGNOSTIC_PATH,
            "diagnostic_sha256": _VERIFIER.LEGACY_DIAGNOSTIC_SHA256,
        },
        "verifier_provenance": {
            "source_commit": "e" * 40,
            "handoff_commit": "f" * 40,
            "manifest_path": _VERIFIER.LEGACY_MANIFEST_PATH,
            "manifest_sha256": "a" * 64,
            "verifier_path": _VERIFIER.VERIFIER_PATH,
            "verifier_sha256": hashlib.sha256(_VERIFIER_SCRIPT.read_bytes()).hexdigest(),
            "amendment": {
                "path": _VERIFIER.RECOVERY_AMENDMENT_PATH,
                "sha256": _VERIFIER.RECOVERY_AMENDMENT_SHA256,
                "commit": _VERIFIER.RECOVERY_AMENDMENT_COMMIT,
            },
        },
        "process": {
            "parent_pid": 17,
            "child_pid": 19,
            "child_exit_code": 0,
            "python_executable": ".venv/bin/python",
            "python_version": "3.12.3",
            "numpy_version": str(np.__version__),
            "isolated": True,
            "child_head_commit": _VERIFIER.LEGACY_HANDOFF_COMMIT,
            "cuda_visible_devices": "",
        },
    }
    _VERIFIER.validate_roundtrip_receipt(value)
    return value


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
        if source_path == "scripts/verify_pass200_rsta_scientific_artifact.py":
            path.write_bytes(_VERIFIER_SCRIPT.read_bytes())
        else:
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
    original_plan_path = repository / _MODULE.ORIGINAL_PLAN_PATH
    original_plan_path.parent.mkdir(parents=True, exist_ok=True)
    original_plan_path.write_text("original plan\n")
    _git(repository, "add", str(original_plan_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "original plan")
    original_plan_commit = _git(repository, "rev-parse", "HEAD")
    future = _future_manifest()
    historical_receipt_path = repository / "reports/pass200-binding-receipt.json"
    historical_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    historical_receipt_path.write_text("{}\n")
    validated_historical_records = deepcopy(future["historical"]["seeds"])
    validated_historical_records[0]["checkpoint"]["path"] = (
        "artifacts/validator-selected-checkpoint.pt"
    )
    validated_historical_manifest = {
        "binding_receipt": {
            "path": str(historical_receipt_path.relative_to(repository)),
            "sha256": hashlib.sha256(historical_receipt_path.read_bytes()).hexdigest(),
        },
        "seeds": {
            str(seed): {
                "checkpoint_pt": dict(record["checkpoint"]),
                "report_json": dict(record["training_report"]),
                "retrieval_json": dict(record["retrieval_report"]),
                "train_npz": dict(record["train_final_pack"]),
            }
            for seed, record in enumerate(validated_historical_records)
        },
    }
    historical_manifest_path = repository / _MODULE.RSTA_MANIFEST_PATH
    historical_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    historical_manifest_path.write_text(
        json.dumps(validated_historical_manifest, separators=(",", ":")) + "\n"
    )
    historical_manifest_sha256 = hashlib.sha256(
        historical_manifest_path.read_bytes()
    ).hexdigest()
    for source_path in _MODULE.RDGC_SOURCE_ORDER:
        path = repository / source_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if source_path == "scripts/verify_pass200_rsta_scientific_artifact.py":
            path.write_bytes(_VERIFIER_SCRIPT.read_bytes())
        else:
            path.write_text(f"# {source_path}\n")
    test_path = repository / "tests/test_diagnose_pass205_rdgc_stage_b.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("# test\n")
    _git(
        repository,
        "add",
        *(_MODULE.RDGC_SOURCE_ORDER),
        str(test_path.relative_to(repository)),
        str(historical_manifest_path.relative_to(repository)),
        str(historical_receipt_path.relative_to(repository)),
    )
    _git(repository, "commit", "-qm", "initial source")
    initial_source_commit = _git(repository, "rev-parse", "HEAD")
    design_path = repository / _MODULE.REPAIR_DESIGN_PATH
    design_path.parent.mkdir(parents=True, exist_ok=True)
    design_path.write_text("repair design\n")
    _git(repository, "add", str(design_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "repair design")
    design_commit = _git(repository, "rev-parse", "HEAD")
    audit_path = repository / _MODULE.LITERATURE_AUDIT_PATH
    audit_path.write_text("literature audit\n")
    _git(repository, "add", str(audit_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "literature audit")
    audit_commit = _git(repository, "rev-parse", "HEAD")
    amendment_path = repository / _MODULE.AUTHORITY_AMENDMENT_PATH
    amendment_path.write_text("authority amendment\n")
    _git(repository, "add", str(amendment_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "authority amendment")
    amendment_commit = _git(repository, "rev-parse", "HEAD")
    authority_plan_path = repository / _MODULE.AUTHORITY_REPAIR_PLAN_PATH
    authority_plan_path.parent.mkdir(parents=True, exist_ok=True)
    authority_plan_path.write_text("authority repair plan\n")
    _git(repository, "add", str(authority_plan_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "authority repair plan")
    authority_plan_commit = _git(repository, "rev-parse", "HEAD")
    (repository / "scripts/diagnose_pass205_rdgc_stage_b.py").write_text(
        "# scripts/diagnose_pass205_rdgc_stage_b.py\n# repaired\n"
    )
    test_path.write_text("# test\n# repaired\n")
    _git(
        repository,
        "add",
        "scripts/diagnose_pass205_rdgc_stage_b.py",
        "tests/test_diagnose_pass205_rdgc_stage_b.py",
    )
    _git(repository, "commit", "-qm", "repaired source")
    reviewed_pre_runtime_source_commit = _git(repository, "rev-parse", "HEAD")
    manifest_path = repository / "docs/pass205_rdgc_stage_b_manifest.json"
    manifest_path.write_text("{}\n")
    _git(repository, "add", str(manifest_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "unexecuted pre-runtime handoff")
    unexecuted_pre_runtime_handoff_commit = _git(repository, "rev-parse", "HEAD")
    runtime_amendment_path = repository / _MODULE.RUNTIME_AMENDMENT_PATH
    runtime_amendment_path.write_text("runtime amendment\n")
    _git(repository, "add", str(runtime_amendment_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "runtime amendment")
    runtime_amendment_commit = _git(repository, "rev-parse", "HEAD")
    plan_path = repository / _MODULE.PLAN_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("runtime repair plan\n")
    _git(repository, "add", str(plan_path.relative_to(repository)))
    _git(repository, "commit", "-qm", "runtime repair plan")
    plan_commit = _git(repository, "rev-parse", "HEAD")
    (repository / "scripts/diagnose_pass205_rdgc_stage_b.py").write_text(
        "# scripts/diagnose_pass205_rdgc_stage_b.py\n# repaired\n# runtime repaired\n"
    )
    test_path.write_text("# test\n# repaired\n# runtime repaired\n")
    _git(
        repository,
        "add",
        "scripts/diagnose_pass205_rdgc_stage_b.py",
        "tests/test_diagnose_pass205_rdgc_stage_b.py",
    )
    _git(repository, "commit", "-qm", "runtime repaired source")
    source_commit = _git(repository, "rev-parse", "HEAD")
    receipt_path = repository / _MODULE.RSTA_VALIDATION_RECEIPT_PATH
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = _real_roundtrip_receipt()
    receipt["verifier_provenance"]["source_commit"] = original_plan_commit
    receipt["verifier_provenance"]["handoff_commit"] = initial_source_commit
    _VERIFIER.write_validation_receipt_atomic(receipt_path, receipt)
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
    historical = {
        "manifest_path": str(historical_manifest_path.relative_to(repository)),
        "manifest_sha256": historical_manifest_sha256,
        "seeds": validated_historical_records,
    }
    upstream = {
        "candidate": {
            "path": _MODULE.CANDIDATE_PATH,
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            "commit": candidate_commit,
        },
        "gate2_audit": {
            "path": _MODULE.LITERATURE_AUDIT_PATH,
            "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "commit": audit_commit,
        },
        "producer_source_commit": candidate_commit,
        "producer_handoff_commit": original_plan_commit,
        "producer_artifact": {
            "path": receipt["artifact"]["path"],
            "sha256": receipt["artifact"]["sha256"],
        },
        "producer_pid": receipt["artifact"]["producer_pid"],
        "producer_exit_code": receipt["artifact"]["producer_exit_code"],
        "verifier_source_commit": original_plan_commit,
        "verifier_handoff_commit": initial_source_commit,
        "verifier_manifest": {
            "path": str(historical_manifest_path.relative_to(repository)),
            "sha256": historical_manifest_sha256,
        },
        "scientific_status": "VALID",
        "scientific_decision": "UNRESOLVED",
        "first_decisive_clause": "no_pass_or_fail_rule",
    }
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
        "upstream_rsta": upstream,
        "literature_audit": {
            "path": _MODULE.LITERATURE_AUDIT_PATH,
            "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            "commit": audit_commit,
            "verdict": "LIVE-NARROW",
            "reviewed_candidate_sha256": hashlib.sha256(
                candidate_path.read_bytes()
            ).hexdigest(),
            "primary_source_ids": list(_MODULE.PRIMARY_SOURCE_IDS),
        },
        "validation_receipt": {
            "path": str(receipt_path.relative_to(repository)),
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "status": "VALID",
            "verifier_source_commit": original_plan_commit,
            "verifier_handoff_commit": initial_source_commit,
            "artifact_path": receipt["artifact"]["path"],
            "artifact_sha256": receipt["artifact"]["sha256"],
        },
        "historical": historical,
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
    monkeypatch.setattr(_MODULE, "ORIGINAL_PLAN_COMMIT", original_plan_commit)
    monkeypatch.setattr(
        _MODULE,
        "ORIGINAL_PLAN_SHA256",
        hashlib.sha256(original_plan_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(_MODULE, "REPAIR_DESIGN_COMMIT", design_commit)
    monkeypatch.setattr(
        _MODULE, "REPAIR_DESIGN_SHA256", hashlib.sha256(design_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(_MODULE, "LITERATURE_AUDIT_COMMIT", audit_commit)
    monkeypatch.setattr(
        _MODULE, "LITERATURE_AUDIT_SHA256", hashlib.sha256(audit_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(_MODULE, "AUTHORITY_AMENDMENT_COMMIT", amendment_commit)
    monkeypatch.setattr(
        _MODULE,
        "AUTHORITY_AMENDMENT_SHA256",
        hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(_MODULE, "AUTHORITY_REPAIR_PLAN_COMMIT", authority_plan_commit)
    monkeypatch.setattr(
        _MODULE,
        "AUTHORITY_REPAIR_PLAN_SHA256",
        hashlib.sha256(authority_plan_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        _MODULE, "REVIEWED_PRE_RUNTIME_SOURCE_COMMIT", reviewed_pre_runtime_source_commit
    )
    monkeypatch.setattr(
        _MODULE,
        "UNEXECUTED_PRE_RUNTIME_HANDOFF_COMMIT",
        unexecuted_pre_runtime_handoff_commit,
    )
    monkeypatch.setattr(_MODULE, "RUNTIME_AMENDMENT_COMMIT", runtime_amendment_commit)
    monkeypatch.setattr(
        _MODULE,
        "RUNTIME_AMENDMENT_SHA256",
        hashlib.sha256(runtime_amendment_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(_MODULE, "PLAN_COMMIT", plan_commit)
    monkeypatch.setattr(_MODULE, "PLAN_SHA256", hashlib.sha256(plan_path.read_bytes()).hexdigest())
    monkeypatch.setattr(_MODULE, "RSTA_CANDIDATE_PATH", _MODULE.CANDIDATE_PATH)
    monkeypatch.setattr(_MODULE, "RSTA_CANDIDATE_COMMIT", candidate_commit)
    monkeypatch.setattr(
        _MODULE,
        "RSTA_CANDIDATE_SHA256",
        hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(_MODULE, "RSTA_GATE2_AUDIT_PATH", _MODULE.LITERATURE_AUDIT_PATH)
    monkeypatch.setattr(_MODULE, "RSTA_GATE2_AUDIT_COMMIT", audit_commit)
    monkeypatch.setattr(
        _MODULE,
        "RSTA_GATE2_AUDIT_SHA256",
        hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(_MODULE, "RSTA_PRODUCER_SOURCE_COMMIT", candidate_commit)
    monkeypatch.setattr(_MODULE, "RSTA_PRODUCER_HANDOFF_COMMIT", original_plan_commit)
    monkeypatch.setattr(_MODULE, "RSTA_ARTIFACT_PATH", receipt["artifact"]["path"])
    monkeypatch.setattr(_MODULE, "RSTA_ARTIFACT_SHA256", receipt["artifact"]["sha256"])
    monkeypatch.setattr(_MODULE, "RSTA_PRODUCER_PID", receipt["artifact"]["producer_pid"])
    monkeypatch.setattr(_MODULE, "RSTA_VERIFIER_SOURCE_COMMIT", original_plan_commit)
    monkeypatch.setattr(_MODULE, "RSTA_VERIFIER_HANDOFF_COMMIT", initial_source_commit)
    monkeypatch.setattr(
        _MODULE, "RSTA_MANIFEST_PATH", str(historical_manifest_path.relative_to(repository))
    )
    monkeypatch.setattr(_MODULE, "RSTA_MANIFEST_SHA256", historical_manifest_sha256)
    monkeypatch.setattr(
        _MODULE, "RSTA_VALIDATION_RECEIPT_PATH", str(receipt_path.relative_to(repository))
    )
    monkeypatch.setattr(_MODULE, "REOPENED_SOURCE_COMMIT", initial_source_commit)
    monkeypatch.setattr(
        _MODULE,
        "REPAIR_DOCUMENT_CHAIN",
        (
            (_MODULE.REPAIR_DESIGN_PATH, design_commit),
            (_MODULE.LITERATURE_AUDIT_PATH, audit_commit),
            (_MODULE.AUTHORITY_AMENDMENT_PATH, amendment_commit),
            (_MODULE.AUTHORITY_REPAIR_PLAN_PATH, authority_plan_commit),
        ),
    )
    validator_calls: list[str] = []
    validated_receipt = types.SimpleNamespace(
        seeds=tuple(
            types.SimpleNamespace(
                seed=seed,
                train_source_export_sha256=record["train_source_export_sha256"],
            )
            for seed, record in enumerate(historical["seeds"])
        )
    )

    class FakePass200Module:
        @staticmethod
        def validate_scientific_execution_source(path: Path) -> dict[str, object]:
            assert path == historical_manifest_path.resolve()
            validator_calls.append("source")
            return {
                "executing_git_commit": "1" * 40,
                "diagnostic_path": "scripts/diagnose_pass200_rsta_stage_a.py",
                "diagnostic_sha256": "2" * 64,
                "frozen_source_revision": "3" * 40,
            }

        @staticmethod
        def validate_historical_binding_receipt(
            manifest_arg: Path, receipt_arg: Path
        ) -> object:
            assert manifest_arg == historical_manifest_path.resolve()
            assert receipt_arg == historical_receipt_path.resolve()
            validator_calls.append("receipt")
            return validated_receipt

    real_loader = _MODULE.load_authenticated_rsta_module
    real_derive = _MODULE.derive_rdgc_seed_artifacts

    def load_authority_module(
        repository_arg: Path, source: dict[str, object]
    ) -> object:
        if source["path"] == "scripts/diagnose_pass200_rsta_stage_a.py":
            validator_calls.append("load_pass200")
            return FakePass200Module()
        return real_loader(repository_arg, source)

    monkeypatch.setattr(_MODULE, "load_authenticated_rsta_module", load_authority_module)
    def recorded_derive(manifest_arg: object, receipt_arg: object) -> list[dict[str, object]]:
        validator_calls.append("derive")
        return real_derive(manifest_arg, receipt_arg)

    monkeypatch.setattr(_MODULE, "derive_rdgc_seed_artifacts", recorded_derive)
    monkeypatch.setattr(_MODULE, "_TEST_VALIDATOR_CALLS", validator_calls, raising=False)
    monkeypatch.setattr(
        _MODULE, "_TEST_INITIAL_SOURCE_COMMIT", initial_source_commit, raising=False
    )
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
    assert _MODULE._TEST_VALIDATOR_CALLS == ["load_pass200", "source", "receipt", "derive"]
    assert value["validated_historical_manifest"] == json.loads(
        (repository / _MODULE.RSTA_MANIFEST_PATH).read_text()
    )
    assert tuple(value["pass200_source_validation"]) == (
        "executing_git_commit",
        "diagnostic_path",
        "diagnostic_sha256",
        "frozen_source_revision",
    )


def test_authority_rejects_skipped_repair_chronology_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest_path, receipt_path = _authority_repository(tmp_path, monkeypatch)
    chain = _MODULE.REPAIR_DOCUMENT_CHAIN
    monkeypatch.setattr(_MODULE, "REPAIR_DOCUMENT_CHAIN", chain[:1] + chain[2:])
    with pytest.raises(ValueError, match="repair document history"):
        _MODULE.authenticate_authority(repository, manifest_path, receipt_path)
    assert _MODULE._TEST_VALIDATOR_CALLS == []


def test_authority_rejects_old_plan_as_repaired_source_chain_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest_path, receipt_path = _authority_repository(tmp_path, monkeypatch)
    monkeypatch.setattr(_MODULE, "REOPENED_SOURCE_COMMIT", _MODULE.ORIGINAL_PLAN_COMMIT)
    with pytest.raises(ValueError, match="repair document history"):
        _MODULE.authenticate_authority(repository, manifest_path, receipt_path)
    assert _MODULE._TEST_VALIDATOR_CALLS == []


def test_authority_uses_real_roundtrip_receipt_schema_and_nested_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest_path, receipt_path = _authority_repository(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    _VERIFIER.validate_roundtrip_receipt(receipt)
    for mutate in (
        lambda value: value.update({"outcome_disclosed": True}),
        lambda value: value["artifact"].update({"path": "alias.json"}),
        lambda value: value["artifact"].update({"sha256": "0" * 64}),
        lambda value: value["verifier_provenance"].update({"source_commit": "0" * 40}),
        lambda value: value["process"].update({"child_exit_code": 1}),
    ):
        mutant = deepcopy(receipt)
        mutate(mutant)
        receipt_path.write_text(json.dumps(mutant, separators=(",", ":")) + "\n")
        manifest["validation_receipt"]["sha256"] = hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
        _git(repository, "add", str(manifest_path.relative_to(repository)))
        _git(repository, "commit", "--amend", "--no-edit", "-q")
        with pytest.raises(ValueError, match="receipt"):
            _MODULE.authenticate_authority(repository, manifest_path, receipt_path)


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
    assert [
        (row["seed"], row["receiver_label"], row["context"]) for row in value["rows"]
    ] == [
        (seed, receiver_label, context)
        for seed in range(4)
        for receiver_label in range(32)
        for context in ("A", "B")
    ]
    assert len(_MODULE.paired_bootstrap(value["rows"])["distributions"]) == 28


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


def test_atomic_writer_strict_reloads_and_revalidates_published_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    payload = _reduced_pre_import_invalid()
    calls: list[dict[str, object]] = []
    original = _MODULE.validate_scientific_payload

    def record(value: dict[str, object]) -> None:
        calls.append(value)
        original(value)

    monkeypatch.setattr(_MODULE, "validate_scientific_payload", record)
    _MODULE.write_json_atomic(output, payload)
    assert calls == [payload, payload]
    assert calls[1] is not payload


def test_cli_requires_exact_manifest_output_and_scientific_once() -> None:
    with pytest.raises(SystemExit):
        _MODULE.main(["--manifest", "x", "--output", "y"])
    with pytest.raises(SystemExit):
        _MODULE.main(["--manifest", "x", "--output", "y", "--scientific-once", "--retry"])


def test_cli_requires_isolated_and_dont_write_bytecode_flags() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(_SCRIPT),
            "--manifest",
            "docs/pass205_rdgc_stage_b_manifest.json",
            "--output",
            "reports/generated/pass205_rdgc_stage_b/x-rdgc-stage-b.json",
            "--scientific-once",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "isolated" in completed.stderr


def test_dgx_python_runtime_rejects_local_cpu_version_without_launching() -> None:
    _MODULE.validate_rdgc_python_version((3, 13, 9))
    with pytest.raises(ValueError, match="scientific process requires Python 3.13.9"):
        _MODULE.validate_rdgc_python_version((3, 12, 3))
    with pytest.raises(TypeError, match="exact integer tuple"):
        _MODULE.validate_rdgc_python_version((3, 13, True))


def test_runtime_version_consistency_uses_one_exact_contract() -> None:
    assert _MODULE.RDGC_PYTHON_VERSION_INFO == (3, 13, 9)
    assert _MODULE.RDGC_PYTHON_VERSION == "3.13.9"
    assert (
        ".".join(str(component) for component in _MODULE.RDGC_PYTHON_VERSION_INFO)
        == _MODULE.RDGC_PYTHON_VERSION
    )
    source = _SCRIPT.read_text()
    main_source = source[source.index("def main(") :]
    validator_source = source[
        source.index("def _validate_result_nested(") : source.index("def main(")
    ]
    assert "validate_rdgc_python_version(" in main_source
    assert 'pre_import["python_version"] != RDGC_PYTHON_VERSION' in validator_source
    assert 'pre_import["python_version"] != "3.13.9"' not in validator_source


def test_execution_command_is_exact_nine_token_isolated_one_shot() -> None:
    value = _reduced_pre_import_invalid()
    value["execution"]["command"] = [
        ".venv/bin/python",
        "-I",
        "-B",
        "scripts/diagnose_pass205_rdgc_stage_b.py",
        "--manifest",
        "docs/pass205_rdgc_stage_b_manifest.json",
        "--output",
        "reports/generated/pass205_rdgc_stage_b/result.json",
        "--scientific-once",
    ]
    _MODULE.validate_scientific_payload(value)
    for mutant_command in (
        value["execution"]["command"][1:],
        [*value["execution"]["command"], "--retry"],
    ):
        mutant = deepcopy(value)
        mutant["execution"]["command"] = mutant_command
        with pytest.raises(ValueError, match="command"):
            _MODULE.validate_scientific_payload(mutant)


def test_real_cpu_torch_func_end_to_end_authenticated_pass200_helper_no_adapters(
    tmp_path: Path,
) -> None:
    helper = _ROOT / "scripts/diagnose_pass200_rsta_stage_a.py"
    descriptor = {
        "path": str(helper),
        "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "git_blob": _git(_ROOT, "rev-parse", "HEAD:scripts/diagnose_pass200_rsta_stage_a.py"),
    }
    loaded = _MODULE.load_authenticated_rsta_module(_ROOT, descriptor)
    assert loaded.__file__ == str(helper)
    assert loaded.domain_seed("rdgc-e2e|", "bound-id") == _RSTA.domain_seed(
        "rdgc-e2e|", "bound-id"
    )
    generator = torch.Generator().manual_seed(205)
    inputs_by_context = {
        "A": torch.randn((180, 3), generator=generator, dtype=torch.float32),
        "B": torch.randn((180, 3), generator=generator, dtype=torch.float32),
    }
    initial = {
        "encoder.weight": torch.randn((2, 3), generator=generator, dtype=torch.float32),
        "encoder.bias": torch.randn((2,), generator=generator, dtype=torch.float32),
    }
    graph_references: list[weakref.ReferenceType[torch.Tensor]] = []

    def open_context(context_name: str) -> tuple[dict[str, object], dict[str, int]]:
        inputs = inputs_by_context[context_name]
        parameters = {
            name: value.detach().clone().requires_grad_(True) for name, value in initial.items()
        }

        def encoder(values: dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.tanh(
                inputs @ values["encoder.weight"].transpose(0, 1) + values["encoder.bias"]
            )

        z, real_vjp = torch.func.vjp(encoder, parameters)
        loss = (z.square().sum(dim=1) * torch.linspace(0.5, 1.5, 180)).mean()
        dbar = -torch.autograd.grad(loss, z, create_graph=True)[0]
        calls = {"vjp": 0}

        def counted_vjp(cotangent: torch.Tensor):
            calls["vjp"] += 1
            return real_vjp(cotangent)

        graph_references.extend((weakref.ref(z), weakref.ref(dbar), weakref.ref(loss)))
        evidence = {
            "batch_sha256": hashlib.sha256(context_name.encode()).hexdigest(),
            "input_sha256": _MODULE._tensor_sha256(inputs, torch),
            "descriptor_sha256": _MODULE._tensor_sha256(z, torch),
            "dbar_sha256": _MODULE._tensor_sha256(dbar, torch),
        }
        return (
            {
                "parameter_names": tuple(parameters),
                "parameters": parameters,
                "dbar": dbar,
                "vjp_function": counted_vjp,
                "encoder": encoder,
                "receiver_indices": tuple(range(8)),
                "device": torch.device("cpu"),
                "graph_evidence": evidence,
                "loss": loss,
            },
            calls,
        )

    ordinary, _ = open_context("A")
    ordinary_gradients = torch.autograd.grad(
        ordinary["loss"], tuple(ordinary["parameters"].values()), retain_graph=False
    )
    p = tuple((-value).detach().contiguous() for value in ordinary_gradients)
    named_parameters = tuple(
        (name, value.detach().clone()) for name, value in ordinary["parameters"].items()
    )
    ordinary.clear()
    del ordinary, ordinary_gradients
    gc.collect()

    corrections: dict[str, tuple[torch.Tensor, ...]] = {}
    for operator in _MODULE.PENALTY_OPERATOR_ORDER:
        if operator == "per_example_gradient_normalized":
            coefficient_context, coefficient_calls = open_context("A")
            coefficients = _MODULE._pgn_coefficients_science(coefficient_context, torch)
            coefficient_evidence = dict(coefficient_context["graph_evidence"])
            assert coefficient_calls["vjp"] == 180
            coefficient_context.clear()
            del coefficient_context
            gc.collect()
            context, correction_calls = open_context("A")
            _MODULE.require_identical_pgn_graph_evidence(
                coefficient_evidence, context["graph_evidence"]
            )
            correction = _MODULE._parameter_correction_science(
                operator=operator,
                receiver_index=0,
                context=context,
                p=p,
                pgn_coefficients=coefficients,
                torch_module=torch,
            )
            assert correction_calls["vjp"] == 2
            del coefficients, coefficient_evidence
        else:
            context, _ = open_context("A")
            correction = _MODULE._parameter_correction_science(
                operator=operator,
                receiver_index=0,
                context=context,
                p=p,
                pgn_coefficients=None,
                torch_module=torch,
            )
        corrections[operator] = correction
        context.clear()
        del context, correction
        gc.collect()
    corrections["layerwise_trust_ratio"] = _MODULE.layerwise_trust_ratio_direction(
        torch, named_parameters, p
    )
    updates = _MODULE.normalize_virtual_updates(torch, p, corrections)
    records: dict[str, dict[str, dict[str, object]]] = {}
    for context_name in ("A", "B"):
        inputs = inputs_by_context[context_name]
        action_parameters = tuple(value.detach().clone() for value in initial.values())

        def receiver_function(
            values: tuple[torch.Tensor, ...], context_inputs: torch.Tensor = inputs
        ) -> torch.Tensor:
            weight, bias = values
            return torch.tanh(context_inputs[0] @ weight.transpose(0, 1) + bias)

        records[context_name] = {
            name: _MODULE.evaluate_virtual_direction(
                {"function": receiver_function, "parameters": action_parameters},
                direction,
                _finite_tensor([1.0, -1.0]),
                torch_module=torch,
            )
            for name, direction in updates.items()
        }
    assert tuple(records) == ("A", "B")
    assert all(tuple(records[context]) == _MODULE.OPERATOR_ORDER for context in ("A", "B"))
    del corrections, updates, named_parameters, p, action_parameters
    gc.collect()
    assert all(reference() is None for reference in graph_references)
    payload = _synthetic_full_payload(records)
    _MODULE.validate_scientific_payload(payload)
    output = tmp_path / "receipt.json"
    _MODULE.write_json_atomic(output, payload)
    persisted = json.loads(output.read_text())
    assert persisted == payload
    _MODULE.validate_scientific_payload(persisted)
    with pytest.raises(FileExistsError):
        _MODULE.write_json_atomic(output, payload)


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
            "python_version": "3.13.9",
            "numpy_version": str(np.__version__),
            "cuda_visible_devices": "0",
            "cublas_workspace_config": ":4096:8",
            "source_commit": "a" * 40,
            "source_files_sha256": "b" * 64,
            "manifest_path": "docs/pass205_rdgc_stage_b_manifest.json",
            "manifest_sha256": "c" * 64,
        },
        "torch_runtime": None,
    }
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
            "candidate": future["candidate"],
            "implementation_plan": future["implementation_plan"],
            "literature_audit": future["literature_audit"],
        },
        "source": source,
        "execution": {
            "attempt": 1,
            "command": [
                ".venv/bin/python",
                "-I",
                "-B",
                "scripts/diagnose_pass205_rdgc_stage_b.py",
                "--manifest",
                "docs/pass205_rdgc_stage_b_manifest.json",
                "--output",
                "reports/generated/pass205_rdgc_stage_b/result.json",
                "--scientific-once",
            ],
            "cwd": str(_ROOT),
            "pid": 1,
            "python_executable": ".venv/bin/python",
            "python_version": "3.13.9",
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
            "predicates": [{"name": "structural_integrity", "value": False}],
            "status": "INVALID",
            "first_decisive_clause": "structural_integrity",
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


def test_result_validator_rejects_valid_looking_fixed_authority_drift() -> None:
    value = _reduced_pre_import_invalid()
    mutations: list[dict[str, object]] = []
    for section, key, replacement in (
        (
            "authority",
            "candidate",
            {"path": "docs/x", "sha256": "1" * 64, "commit": "2" * 40},
        ),
        ("binding", "rsta_producer_source_commit", "3" * 40),
        ("binding", "rsta_producer_handoff_commit", "4" * 40),
        ("binding", "verifier_source_commit", "5" * 40),
        ("binding", "verifier_handoff_commit", "6" * 40),
        ("binding", "verifier_manifest_sha256", "7" * 64),
    ):
        mutant = deepcopy(value)
        mutant[section][key] = replacement
        mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["binding"]["validation_receipt"]["path"] = "reports/foreign-receipt.json"
    mutations.append(mutant)
    for mutant in mutations:
        with pytest.raises(ValueError, match="authority|binding|provenance"):
            _MODULE.validate_scientific_payload(mutant)


def test_post_import_invalid_requires_complete_actual_integrity_through_failure() -> None:
    value = _reduced_pre_import_invalid()
    value["phase_reached"] = "integrity"
    value["environment"]["phase"] = "post_import"
    value["environment"]["torch_runtime"] = {
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": "synthetic",
        "cudnn_version": 1,
        "device_index": 0,
        "device_name": "synthetic",
        "device_capability": [9, 0],
        "deterministic_algorithms": True,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
    }
    seeds = []
    for seed in range(4):
        passed = seed < 3
        seeds.append(
            {
                "seed": seed,
                "dense_jacobian_passed": passed,
                "bn_passed": passed,
                "repeatability_passed": passed,
                "normwise_adjoint_passed": passed,
                "sign_control_passed": passed,
                "rotation_passed": passed,
                "atomic_writer_passed": passed,
                "no_candidate_reachability_passed": passed,
                "action_hashes_sha256": hashlib.sha256(f"seed-{seed}".encode()).hexdigest(),
                "passed": passed,
            }
        )
    value["integrity"] = {
        "seeds": seeds,
        "all_four_passed": False,
        "candidate_calls_before_all_four": 0,
        "candidate_state_before_all_four": False,
    }
    value["decision"] = {
        "close_precedence": True,
        "predicates": [{"name": "structural_integrity", "value": False}],
        "status": "INVALID",
        "first_decisive_clause": "seed_3_integrity_failed",
        "authorized_action": "stop_invalid",
    }
    _MODULE.validate_scientific_payload(value)
    for mutation in ({}, {**value["integrity"], "all_four_passed": True}):
        mutant = deepcopy(value)
        mutant["integrity"] = mutation
        with pytest.raises(ValueError, match="integrity"):
            _MODULE.validate_scientific_payload(mutant)


def test_rsta_integrity_conversion_preserves_each_actual_gate_through_failure() -> None:
    seeds = {}
    for seed in range(4):
        controls = {
            "rebuild": {"passed": True},
            "reversed_action_order": {"passed": seed != 2},
            "parameter_sign": {"passed": True},
            "output_sign": {"passed": True},
        }
        seeds[str(seed)] = {
            "adjoint": {
                "normwise_passed": True,
                "integrity_passed": seed != 2,
                "jvp_sha256": hashlib.sha256(f"jvp-{seed}".encode()).hexdigest(),
                "vjp_sha256": hashlib.sha256(f"vjp-{seed}".encode()).hexdigest(),
                "controls": controls,
            }
        }
    raw = {
        "integrity": {
            "dense_fixture": {"passed": True},
            "bn_fixture": {"passed": True},
            "seeds": seeds,
            "all_passed": False,
        }
    }
    converted = _MODULE.convert_rsta_integrity_prefix(raw)
    assert converted["all_four_passed"] is False
    assert converted["seeds"][2]["rotation_passed"] is False
    assert converted["seeds"][2]["passed"] is False
    assert all(converted["seeds"][index]["passed"] for index in (0, 1, 3))


def _validated_historical_authorities() -> tuple[dict[str, object], object]:
    names = (
        "checkpoint_pt",
        "gallery_npz",
        "prehead_npz",
        "query_npz",
        "report_json",
        "retrieval_json",
        "train_npz",
    )
    manifest_seeds: dict[str, object] = {}
    receipt_seeds = []
    for seed in range(4):
        artifacts = {
            name: {
                "path": f"/registered/seed{seed}/{name}",
                "sha256": hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
            }
            for name in names
        }
        manifest_seeds[str(seed)] = artifacts
        receipt_seeds.append(
            types.SimpleNamespace(
                seed=seed,
                artifacts=artifacts,
                official_recall_at_1=0.5,
                train_row_count=180,
                train_identity_count=40,
                train_example_id_order_sha256="1" * 64,
                train_label_order_sha256="2" * 64,
                train_source_order_sha256="3" * 64,
                train_source_export_sha256=hashlib.sha256(
                    f"train-source:{seed}".encode()
                ).hexdigest(),
            )
        )
    return {"seeds": manifest_seeds}, types.SimpleNamespace(seeds=tuple(receipt_seeds))


def test_repaired_seed_schema_is_derived_from_validated_pass200_authorities() -> None:
    manifest, receipt = _validated_historical_authorities()
    records = _MODULE.derive_rdgc_seed_artifacts(manifest, receipt)
    assert [record["seed"] for record in records] == [0, 1, 2, 3]
    assert tuple(records[0]) == (
        "seed",
        "checkpoint",
        "training_report",
        "retrieval_report",
        "train_final_pack",
        "train_source_export_sha256",
    )
    assert records[0]["checkpoint"] == manifest["seeds"]["0"]["checkpoint_pt"]
    assert records[0]["training_report"] == manifest["seeds"]["0"]["report_json"]
    assert records[0]["retrieval_report"] == manifest["seeds"]["0"]["retrieval_json"]
    assert records[0]["train_final_pack"] == manifest["seeds"]["0"]["train_npz"]
    assert records[0]["train_source_export_sha256"] == (
        receipt.seeds[0].train_source_export_sha256
    )


def test_real_pass200_source_validator_rejects_later_worktree_drift() -> None:
    with pytest.raises(ValueError, match="source worktree differs"):
        _RSTA.validate_scientific_execution_source(
            _ROOT / _MODULE.RSTA_MANIFEST_PATH
        )


def test_repaired_seed_schema_rejects_recursive_shape_type_and_relation_drift() -> None:
    manifest, receipt = _validated_historical_authorities()
    baseline = _MODULE.derive_rdgc_seed_artifacts(manifest, receipt)[0]
    mutants: list[dict[str, object]] = []
    mutant = deepcopy(baseline)
    mutant["seed"] = False
    mutants.append(mutant)
    mutant = deepcopy(baseline)
    mutant["configuration_sha256"] = "0" * 64
    mutants.append(mutant)
    mutant = deepcopy(baseline)
    mutant["checkpoint"] = {
        "sha256": baseline["checkpoint"]["sha256"],
        "path": baseline["checkpoint"]["path"],
    }
    mutants.append(mutant)
    mutant = deepcopy(baseline)
    mutant["retrieval_report"]["sha256"] = "A" * 64
    mutants.append(mutant)
    mutant = deepcopy(baseline)
    mutant["train_final_pack"]["path"] = ""
    mutants.append(mutant)
    for mutant in mutants:
        with pytest.raises(ValueError):
            _MODULE._validate_seed_artifacts(mutant, 0, "mutant seed")


def _future_manifest() -> dict[str, object]:
    artifact = {"path": "artifacts/value.json", "sha256": "f" * 64}
    seeds = [
        {
            "seed": seed,
            "checkpoint": dict(artifact),
            "training_report": dict(artifact),
            "retrieval_report": dict(artifact),
            "train_final_pack": dict(artifact),
            "train_source_export_sha256": "2" * 64,
        }
        for seed in range(4)
    ]
    files = [
        {"path": path, "sha256": "a" * 64, "git_blob": "b" * 40}
        for path in _MODULE.RDGC_SOURCE_ORDER
    ]
    return {
        "schema_version": 1,
        "candidate": {
            "path": _MODULE.CANDIDATE_PATH,
            "sha256": _MODULE.CANDIDATE_SHA256,
            "commit": _MODULE.CANDIDATE_COMMIT,
        },
        "implementation_plan": {
            "path": _MODULE.PLAN_PATH,
            "sha256": _MODULE.PLAN_SHA256,
            "commit": _MODULE.PLAN_COMMIT,
        },
        "upstream_rsta": {
            "candidate": {
                "path": _MODULE.RSTA_CANDIDATE_PATH,
                "sha256": _MODULE.RSTA_CANDIDATE_SHA256,
                "commit": _MODULE.RSTA_CANDIDATE_COMMIT,
            },
            "gate2_audit": {
                "path": _MODULE.RSTA_GATE2_AUDIT_PATH,
                "sha256": _MODULE.RSTA_GATE2_AUDIT_SHA256,
                "commit": _MODULE.RSTA_GATE2_AUDIT_COMMIT,
            },
            "producer_source_commit": _MODULE.RSTA_PRODUCER_SOURCE_COMMIT,
            "producer_handoff_commit": _MODULE.RSTA_PRODUCER_HANDOFF_COMMIT,
            "producer_artifact": {
                "path": _MODULE.RSTA_ARTIFACT_PATH,
                "sha256": _MODULE.RSTA_ARTIFACT_SHA256,
            },
            "producer_pid": _MODULE.RSTA_PRODUCER_PID,
            "producer_exit_code": 0,
            "verifier_source_commit": _MODULE.RSTA_VERIFIER_SOURCE_COMMIT,
            "verifier_handoff_commit": _MODULE.RSTA_VERIFIER_HANDOFF_COMMIT,
            "verifier_manifest": {
                "path": _MODULE.RSTA_MANIFEST_PATH,
                "sha256": _MODULE.RSTA_MANIFEST_SHA256,
            },
            "scientific_status": "VALID",
            "scientific_decision": "UNRESOLVED",
            "first_decisive_clause": "no_pass_or_fail_rule",
        },
        "literature_audit": {
            "path": _MODULE.LITERATURE_AUDIT_PATH,
            "sha256": _MODULE.LITERATURE_AUDIT_SHA256,
            "commit": _MODULE.LITERATURE_AUDIT_COMMIT,
            "verdict": "LIVE-NARROW",
            "reviewed_candidate_sha256": _MODULE.CANDIDATE_SHA256,
            "primary_source_ids": [
                "pmlr-v130-zhou21a",
                "neurips-2022-67b0579a7298d9cf39c59404d867bdd7",
                "arxiv-2511.15487v2",
                "neurips-2019-c61f571dbd2fb949d3fe5ae1608dd48b",
                "pmlr-v80-chen18a",
                "pmlr-v37-martens15",
                "neurips-2025-4522de4178bddb36b49aa26efad537cf",
                "pmlr-v108-barshan20a",
                "neurips-2023-8249b30d877c91611fd8c7aa6ac2b5fe",
                "pmlr-v162-rame22a",
                "cvpr-2022-kim-adaface",
                "cvpr-2021-meng-magface",
                "cvpr-2019-zhang-adacos",
                "arxiv-1708.03888",
            ],
        },
        "validation_receipt": {
            "path": _MODULE.RSTA_VALIDATION_RECEIPT_PATH,
            "sha256": "4" * 64,
            "status": "VALID",
            "verifier_source_commit": _MODULE.RSTA_VERIFIER_SOURCE_COMMIT,
            "verifier_handoff_commit": _MODULE.RSTA_VERIFIER_HANDOFF_COMMIT,
            "artifact_path": _MODULE.RSTA_ARTIFACT_PATH,
            "artifact_sha256": _MODULE.RSTA_ARTIFACT_SHA256,
        },
        "historical": {
            "manifest_path": _MODULE.RSTA_MANIFEST_PATH,
            "manifest_sha256": _MODULE.RSTA_MANIFEST_SHA256,
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
    selection = _MODULE.build_rdgc_selection(
        ids, labels, old_selection=_old_primary(ids, labels)
    )
    preliminary_labels = selection["preliminary"]["identity_labels"]
    preliminary_rows: list[dict[str, object]] = []
    for seed in range(4):
        for context in ("A", "B"):
            for receiver_label in preliminary_labels:
                receiver_offset = preliminary_labels.index(receiver_label)
                kappa = 1.25 + 0.25 * receiver_offset
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
                        "self_norm": kappa,
                        "kappa": kappa,
                        "log_kappa": float(math.log(kappa)),
                        "b_norms_by_contributor_count": dict(errors),
                        "absolute_log_gain_errors_by_contributor_count": dict(errors),
                        "count_gain": 2.0,
                    }
                )
    preliminary_summary = _MODULE.summarize_preliminary_rows(preliminary_rows)
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
        **preliminary_summary,
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
                        "batch_sha256": selection["panel"]["groups"][receiver_index // 8][
                            "contexts"
                        ][("A", "B").index(context)]["batch_sha256"],
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
    correction_cosines = {
        control: [(int(row["seed"]), 0.5) for row in panel_rows if row["context"] == "A"]
        for control in _MODULE.CONTROL_ORDER
    }
    panel, panel_decision = _MODULE._aggregate_panel(
        panel_rows, correction_cosines, bootstrap
    )
    payload = _reduced_pre_import_invalid()
    payload.update(
        {
            "status": panel_decision["status"],
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
        "torch_version": str(torch.__version__),
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
            "predicates": [
                {"name": name, "value": value} for name, value in panel["predicates"].items()
            ],
            "status": panel_decision["status"],
            "first_decisive_clause": panel_decision["first_decisive_clause"],
            "authorized_action": {
                "PASS": "new_training_preregistration_only",
                "CLOSE": "stop_close",
                "UNRESOLVED": "stop_unresolved",
            }[panel_decision["status"]],
        }
    )
    return payload


def test_persisted_full_and_reduced_python_runtime_mutations_are_rejected() -> None:
    action = {
        "motion_sha256": "a" * 64,
        "motion_norm": 1.0,
        "margin_alignment": 0.25,
        "margin_slope": 0.5,
    }
    records = {
        context: {name: dict(action) for name in _MODULE.OPERATOR_ORDER}
        for context in ("A", "B")
    }
    payloads = (_reduced_pre_import_invalid(), _synthetic_full_payload(records))
    for payload in payloads:
        persisted = json.loads(json.dumps(payload, allow_nan=False))
        _MODULE.validate_scientific_payload(persisted)
        for invalid in ("3.12.3", 3139, ""):
            mutant = deepcopy(persisted)
            mutant["environment"]["pre_import"]["python_version"] = invalid
            with pytest.raises((TypeError, ValueError)):
                _MODULE.validate_scientific_payload(
                    json.loads(json.dumps(mutant, allow_nan=False))
                )

        class VersionText(str):
            pass

        subclass_mutant = deepcopy(persisted)
        subclass_mutant["environment"]["pre_import"]["python_version"] = VersionText(
            "3.13.9"
        )
        with pytest.raises((TypeError, ValueError)):
            _MODULE.validate_scientific_payload(subclass_mutant)


def test_panel_close_predicates_record_context_b_pa_failure_even_when_it_is_not_primary() -> None:
    records = {
        context: {
            name: {
                "motion_sha256": "a" * 64,
                "motion_norm": 1.0,
                "margin_alignment": 0.0,
                "margin_slope": 0.0,
            }
            for name in _MODULE.OPERATOR_ORDER
        }
        for context in ("A", "B")
    }
    records["A"]["rdgc"]["margin_alignment"] = 0.1
    records["A"]["rdgc"]["margin_slope"] = 0.1
    records["B"]["rdgc"]["margin_alignment"] = 0.0
    records["B"]["pa"]["margin_alignment"] = 0.1
    value = _synthetic_full_payload(records)
    assert value["decision"]["first_decisive_clause"] == "close_context_b_vs_pa"
    assert value["panel"]["predicates"]["close_vs_pa"] is True


def test_full_panel_result_decision_persists_all_ordered_panel_predicates() -> None:
    predicates = {
        name: False
        for name in (
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
    }
    predicates["completeness"] = True
    decision = _MODULE.result_decision_for_phase(
        "UNRESOLVED",
        "no_close_or_pass_rule",
        panel={"predicates": predicates},
    )
    assert decision["predicates"] == [
        {"name": name, "value": value} for name, value in predicates.items()
    ]


def test_full_validator_recomputes_rows_aggregates_predicates_decisions_and_bootstrap() -> None:
    action = {
        "motion_sha256": "a" * 64,
        "motion_norm": 1.0,
        "margin_alignment": 0.25,
        "margin_slope": 0.5,
    }
    records = {
        context: {name: dict(action) for name in _MODULE.OPERATOR_ORDER}
        for context in ("A", "B")
    }
    value = _synthetic_full_payload(records)
    _MODULE.validate_scientific_payload(value)
    mutations = []
    mutant = deepcopy(value)
    mutant["preliminary"]["rows"][0], mutant["preliminary"]["rows"][1] = (
        mutant["preliminary"]["rows"][1],
        mutant["preliminary"]["rows"][0],
    )
    mutations.append(mutant)
    for section, key in (
        ("preliminary", "pooled_aggregates"),
        ("preliminary", "predicates"),
        ("panel", "seed_context_aggregates"),
        ("panel", "pooled_aggregates"),
        ("panel", "predicates"),
    ):
        mutant = deepcopy(value)
        target = mutant[section][key]
        if type(target) is dict:
            first = next(iter(target))
            target[first] = not target[first]
        elif section == "preliminary":
            target[0]["count_gain_mean"] += 0.125
        else:
            numeric = "alignment_mean"
            target[0][numeric] += 0.125
        mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["panel"]["rows"][0]["receiver_label"] = mutant["panel"]["rows"][2][
        "receiver_label"
    ]
    mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["bootstrap"]["distributions"][0]["lower_bound"] += 0.125
    mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["bootstrap"]["complete_labels"][0], mutant["bootstrap"]["complete_labels"][1] = (
        mutant["bootstrap"]["complete_labels"][1],
        mutant["bootstrap"]["complete_labels"][0],
    )
    mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["decision"]["predicates"] = [{"name": "alias", "value": True}]
    mutations.append(mutant)
    for index, mutant in enumerate(mutations):
        try:
            _MODULE.validate_scientific_payload(mutant)
        except ValueError:
            pass
        else:
            pytest.fail(f"validator accepted relation mutation {index}")


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


def test_future_manifest_rejects_valid_looking_literature_authority_drift() -> None:
    value = _future_manifest()
    _MODULE.validate_future_manifest(value)
    mutations = []
    for key, replacement in (
        ("path", "docs/alias-audit.md"),
        ("sha256", "7" * 64),
        ("commit", "8" * 40),
        ("verdict", "UNRESOLVED"),
        ("reviewed_candidate_sha256", "9" * 64),
    ):
        mutant = deepcopy(value)
        mutant["literature_audit"][key] = replacement
        mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["literature_audit"]["primary_source_ids"][8] = "openreview-wrong-version"
    mutations.append(mutant)
    for mutant in mutations:
        with pytest.raises(ValueError, match="literature"):
            _MODULE.validate_future_manifest(mutant)


def test_future_manifest_rejects_original_plan_as_current_authority() -> None:
    value = _future_manifest()
    value["implementation_plan"] = {
        "path": _MODULE.ORIGINAL_PLAN_PATH,
        "sha256": _MODULE.ORIGINAL_PLAN_SHA256,
        "commit": _MODULE.ORIGINAL_PLAN_COMMIT,
    }
    with pytest.raises(ValueError, match="candidate|plan"):
        _MODULE.validate_future_manifest(value)


def test_future_manifest_rejects_valid_looking_upstream_and_historical_drift() -> None:
    value = _future_manifest()
    mutations: list[dict[str, object]] = []
    for section, key, replacement in (
        ("upstream_rsta", "producer_source_commit", "1" * 40),
        ("upstream_rsta", "producer_handoff_commit", "2" * 40),
        ("upstream_rsta", "verifier_source_commit", "3" * 40),
        ("upstream_rsta", "verifier_handoff_commit", "4" * 40),
        ("historical", "manifest_path", "docs/foreign-pass200.json"),
        ("historical", "manifest_sha256", "5" * 64),
        ("validation_receipt", "artifact_sha256", "6" * 64),
    ):
        mutant = deepcopy(value)
        mutant[section][key] = replacement
        mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["upstream_rsta"]["candidate"]["sha256"] = "7" * 64
    mutations.append(mutant)
    mutant = deepcopy(value)
    mutant["upstream_rsta"]["gate2_audit"]["commit"] = "8" * 40
    mutations.append(mutant)
    for mutant in mutations:
        with pytest.raises(ValueError, match="upstream|historical|receipt"):
            _MODULE.validate_future_manifest(mutant)
