from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "profile_unicom_training_step.py"
SPEC = importlib.util.spec_from_file_location("profile_unicom_training_step", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_runtime_override_protocols_are_exact_and_cli_requires_explicit_mode(
    tmp_path: Path,
) -> None:
    assert {
        "current": MODULE.RuntimeProtocol(compile=False, fused=False, ema=True),
        "composed": MODULE.RuntimeProtocol(compile=True, fused=True, ema=False),
    } == MODULE.RUNTIME_PROTOCOLS
    args = MODULE.parse_args(
        [
            "--run-checkpoint",
            str(tmp_path / "epoch-0016.pt"),
            "--run-receipt",
            str(tmp_path / "run-receipt.json"),
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--runtime-mode",
            "composed",
            "--profile-kind",
            "quality",
            "--parent-trainer-source",
            MODULE.PARENT_TRAINER_SOURCE,
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )

    assert args.runtime_mode == "composed"
    assert args.profile_kind == "quality"
    assert args.parent_trainer_source == MODULE.PARENT_TRAINER_SOURCE


def test_runtime_override_builds_only_registered_training_substrate(monkeypatch) -> None:
    import torch

    raw_model = torch.nn.Linear(3, 2)
    classifier = torch.nn.Parameter(torch.ones(4, 2))
    compiled = object()
    compile_calls: list[tuple[object, str]] = []

    def compile_model(model, *, mode):
        compile_calls.append((model, mode))
        return compiled

    monkeypatch.setattr(torch, "compile", compile_model)

    class _Trainer:
        class StepEMA:
            def __init__(self, model, head):
                self.inputs = (model, head)

        @staticmethod
        def build_optimizer(model, head, *, learning_rate, classifier_learning_rate, fused):
            return {
                "inputs": (model, head),
                "learning_rate": learning_rate,
                "classifier_learning_rate": classifier_learning_rate,
                "fused": fused,
            }

    protocol = {"learning_rate": 1e-5, "classifier_learning_rate": 1e-4}
    current = MODULE._construct_runtime(
        raw_model,
        classifier,
        protocol=protocol,
        trainer=_Trainer,
        runtime_mode="current",
    )
    composed = MODULE._construct_runtime(
        raw_model,
        classifier,
        protocol=protocol,
        trainer=_Trainer,
        runtime_mode="composed",
    )

    assert current[0] is raw_model
    assert current[1]["fused"] is False
    assert isinstance(current[2], _Trainer.StepEMA)
    assert composed[0] is compiled
    assert composed[1]["fused"] is True
    assert composed[2] is None
    assert compile_calls == [(raw_model, "reduce-overhead")]


def test_runtime_override_parent_trainer_loads_registered_git_blob_then_unlinks() -> None:
    trainer = MODULE._load_authenticated_parent_trainer(
        MODULE_PATH.parents[1], MODULE.PARENT_TRAINER_SOURCE
    )

    assert trainer.__profile_source_sha256__ == MODULE.PARENT_TRAINER_SHA256
    assert trainer.__profile_source_spec__ == MODULE.PARENT_TRAINER_SOURCE
    assert not Path(trainer.__file__).exists()


@pytest.mark.parametrize(
    "source,blob",
    [
        ("70c760e57e6c27dec1473eecd4765e0a8cd4cf6b:README.md", None),
        (
            "70c760e57e6c27dec1473eecd4765e0a8cd4cf6b:"
            "scripts/train_unicom_inshop.py",
            b"substituted trainer\n",
        ),
    ],
)
def test_runtime_override_parent_trainer_rejects_wrong_path_or_blob(
    source: str,
    blob: bytes | None,
    monkeypatch,
) -> None:
    if blob is not None:
        monkeypatch.setattr(MODULE, "_git_blob_bytes", lambda _repo, _source: blob)
    with pytest.raises(ValueError, match="parent trainer"):
        MODULE._load_authenticated_parent_trainer(MODULE_PATH.parents[1], source)


def test_runtime_override_checkpoint_uses_frozen_parent_hash_not_live_trainer() -> None:
    checkpoint = {"training_protocol": {"trainer_sha256": MODULE.PARENT_TRAINER_SHA256}}
    MODULE._validate_parent_checkpoint_authority(checkpoint)

    checkpoint["training_protocol"]["trainer_sha256"] = MODULE._sha256_file(
        MODULE_PATH.with_name("train_unicom_inshop.py")
    )
    with pytest.raises(ValueError, match="parent trainer"):
        MODULE._validate_parent_checkpoint_authority(checkpoint)


def test_runtime_override_registered_seed2_checkpoint_schema_loads(tmp_path: Path) -> None:
    import torch

    checkpoint_path = tmp_path / "seed-2" / "epoch-0016.pt"
    checkpoint_path.parent.mkdir()
    payload = {
        "epoch": 16,
        "model": {},
        "classifier": torch.zeros((4, 2), dtype=torch.float32),
        "ema": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": None,
        "mask_generator": torch.Generator().get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": [torch.Generator().get_state()],
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": {
            "seed": 2,
            "trainer_sha256": MODULE.PARENT_TRAINER_SHA256,
        },
        "history": [],
    }
    torch.save(payload, checkpoint_path)

    loaded = MODULE._load_checkpoint(checkpoint_path)
    MODULE._validate_parent_checkpoint_authority(loaded)

    assert loaded["epoch"] == 16
    assert loaded["training_protocol"]["seed"] == 2


def test_scaler_unscales_before_gradient_evidence_and_step(monkeypatch) -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(4.0)
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    calls: list[str] = []

    class _Scaler:
        def get_scale(self):
            return 8.0

        def unscale_(self, observed_optimizer):
            assert observed_optimizer is optimizer
            calls.append("unscale")
            parameter.grad.div_(8.0)

        def step(self, observed_optimizer):
            assert calls == ["unscale", "gradient"]
            calls.append("step")
            observed_optimizer.step()

        def update(self):
            calls.append("update")

    def inspect(parameters):
        assert tuple(parameters) == (parameter,)
        calls.append("gradient")
        return True

    monkeypatch.setattr(MODULE, "_gradients_are_finite", inspect)
    decision = MODULE._optimizer_step(
        {
            "optimizer": optimizer,
            "scheduler": types.SimpleNamespace(last_epoch=1, total_steps=1),
            "scaler": _Scaler(),
            "trainable_parameters": (parameter,),
        }
    )

    assert calls == ["unscale", "gradient", "step", "update"]
    assert decision == {
        "enabled": True,
        "scale_before": 8.0,
        "scale_after": 8.0,
        "skipped": False,
    }


def test_gradient_failure_is_closed_before_scaler_or_optimizer_step() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(float("nan"))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)

    class _Scaler:
        def get_scale(self):
            return 8.0

        def unscale_(self, _optimizer):
            return None

        def step(self, _optimizer):
            raise AssertionError("nonfinite gradients reached scaler.step")

        def update(self):
            raise AssertionError("nonfinite gradients reached scaler.update")

    with pytest.raises(ValueError, match="gradient is nonfinite"):
        MODULE._optimizer_step(
            {
                "optimizer": optimizer,
                "scheduler": types.SimpleNamespace(last_epoch=1, total_steps=1),
                "scaler": _Scaler(),
                "trainable_parameters": (parameter,),
            }
        )


def test_quality_profile_runs_exact_optimizer_phases_without_objective_only_calls(
    monkeypatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def step(_state, _batch, *, measured):
        calls.append(("step", measured))
        if not measured:
            return None
        return {
            "timing": _sample(1.0, 0.04, 0.03),
            "loss": 2.0,
            "gradients_finite": True,
            "scaler": {
                "enabled": False,
                "scale_before": None,
                "scale_after": None,
                "skipped": False,
            },
        }

    monkeypatch.setattr(MODULE, "_training_step", step)
    monkeypatch.setattr(MODULE, "_reset_cuda_peaks", lambda: calls.append(("reset", False)))
    monkeypatch.setattr(
        MODULE,
        "_objective_profile_step",
        lambda _state, _batch: calls.append(("objective", False)),
    )
    state = {"loader": (object(),)}

    evidence = MODULE._execute_profile_phases(state, profile_kind="quality")

    assert calls.count(("step", False)) == 20
    assert calls.count(("step", True)) == 50
    assert calls.count(("reset", False)) == 1
    assert ("objective", False) not in calls
    assert evidence["optimizer_step_count"] == 70
    assert evidence["objective_call_count"] == 0
    assert evidence["losses"] == [2.0] * 50
    assert evidence["unscaled_gradients_finite"] == [True] * 50


def _sample(step: float, objective_forward: float, head_backward: float) -> dict[str, float]:
    components = {
        "h2d_seconds": 0.01,
        "zero_grad_seconds": 0.01,
        "backbone_forward_seconds": 0.20,
        "objective_forward_seconds": objective_forward,
        "head_backward_seconds": head_backward,
        "backbone_backward_seconds": 0.40,
        "update_seconds": 0.20,
        "tail_seconds": 0.01,
    }
    cuda_step = math.fsum(components.values())
    return {
        "step_wall_seconds": step,
        "cuda_step_seconds": cuda_step,
        **components,
    }


def test_timing_summary_recomputes_contiguous_intervals_and_objective_ceiling() -> None:
    samples = (
        _sample(1.00, 0.04, 0.03),
        _sample(1.10, 0.05, 0.04),
        _sample(0.90, 0.03, 0.02),
    )

    summary = MODULE.summarize_timing_samples(samples)

    assert tuple(summary) == (
        "step_wall_seconds",
        "step_wall_p10_seconds",
        "step_wall_p90_seconds",
        "objective_ceiling_seconds",
        "objective_ceiling_fraction",
        "component_mean_seconds",
    )
    assert summary["step_wall_seconds"] == 1.0
    assert summary["step_wall_p10_seconds"] == pytest.approx(0.92)
    assert summary["step_wall_p90_seconds"] == pytest.approx(1.08)
    assert summary["objective_ceiling_seconds"] == pytest.approx(0.07)
    assert summary["objective_ceiling_fraction"] == pytest.approx(0.07)
    assert tuple(summary["component_mean_seconds"]) == MODULE.COMPONENT_KEYS

    invalid = dict(samples[0])
    invalid["cuda_step_seconds"] += 0.001
    with pytest.raises(ValueError, match="contiguous"):
        MODULE.summarize_timing_samples((invalid,))


@pytest.mark.parametrize("ppm", [1, 10, 50, 100])
def test_timing_summary_allows_cuda_event_clock_skew_above_cpu_wall(ppm: int) -> None:
    sample = _sample(1.0, 0.04, 0.03)
    sample["step_wall_seconds"] = sample["cuda_step_seconds"] / (1.0 + ppm * 1e-6)

    summary = MODULE.summarize_timing_samples((sample,))

    assert summary["step_wall_seconds"] == sample["step_wall_seconds"]
    assert summary["component_mean_seconds"] == {key: sample[key] for key in MODULE.COMPONENT_KEYS}


def test_clock_domain_tolerance_is_exact_registered_value() -> None:
    assert MODULE.CLOCK_DOMAIN_REL_TOL == 0.01


@pytest.mark.parametrize("factor", [1.05, 2.0, 1_000.0])
def test_timing_summary_rejects_structurally_impossible_cuda_span(factor: float) -> None:
    sample = _sample(1.0, 0.04, 0.03)
    sample["step_wall_seconds"] = sample["cuda_step_seconds"] / factor

    with pytest.raises(
        ValueError,
        match=r"timing sample 0 CUDA step exceeds wall step: cuda=.* wall=.*",
    ):
        MODULE.summarize_timing_samples((sample,))


def test_timing_summary_error_names_nonzero_index_and_exact_durations() -> None:
    valid = _sample(1.0, 0.04, 0.03)
    invalid = _sample(1.0, 0.04, 0.03)
    invalid["step_wall_seconds"] = 0.6

    with pytest.raises(ValueError) as caught:
        MODULE.summarize_timing_samples((valid, invalid))

    assert str(caught.value) == ("timing sample 1 CUDA step exceeds wall step: cuda=0.9 wall=0.6")


def test_timing_row_checks_independent_whole_span_against_components() -> None:
    consistent = MODULE._timing_row(
        5.3,
        (0.5,) * len(MODULE.COMPONENT_KEYS),
        4.0 + 2e-6,
    )
    MODULE.summarize_timing_samples((consistent,))

    inconsistent = MODULE._timing_row(
        5.3,
        (0.5,) * len(MODULE.COMPONENT_KEYS),
        4.5,
    )
    with pytest.raises(ValueError, match="contiguous"):
        MODULE.summarize_timing_samples((inconsistent,))


def test_event_timing_row_measures_components_and_whole_span_independently() -> None:
    calls: list[tuple[int, int]] = []

    class _Boundary:
        def __init__(self, index: int) -> None:
            self.index = index

        def elapsed_time(self, other: object) -> float:
            assert isinstance(other, _Boundary)
            calls.append((self.index, other.index))
            return float(other.index - self.index) * 100.0

    row = MODULE._event_timing_row(5.3, tuple(_Boundary(index) for index in range(9)))

    assert calls == [*(zip(range(8), range(1, 9), strict=True)), (0, 8)]
    assert row["cuda_step_seconds"] == pytest.approx(0.8)
    assert tuple(row[key] for key in MODULE.COMPONENT_KEYS) == pytest.approx((0.1,) * 8)


@dataclass
class _Event:
    name: str
    self_device_time_total: float = 0.0
    cpu_children: list[_Event] = field(default_factory=list)
    device_type: object = field(default_factory=lambda: types.SimpleNamespace(name="CPU"))


def test_fusible_classifier_includes_cuda_backward_on_the_autograd_thread() -> None:
    cpu_marker = _Event(
        MODULE.OBJECTIVE_MARKER,
        cpu_children=[
            _Event("aten::index_select", 120.0),
            _Event("aten::linalg_vector_norm", 80.0),
            _Event("aten::mm", 900.0),
            _Event("aten::linear", 0.0, [_Event("aten::matmul", 700.0)]),
            _Event("aten::_log_softmax", 100.0),
        ],
    )
    cuda_marker = _Event(
        MODULE.OBJECTIVE_MARKER,
        300_000.0,
        device_type=types.SimpleNamespace(name="CUDA"),
    )
    backward_on_autograd_thread = _Event("aten::_log_softmax_backward_data", 60.0)
    profiler_overhead = _Event("Activity Buffer Request", 50_000.0)

    seconds = MODULE.fusible_nonbackbone_seconds(
        (cpu_marker, cuda_marker, backward_on_autograd_thread, profiler_overhead)
    )

    assert seconds == pytest.approx(0.00036)


def test_profile_objective_requests_cpu_tree_and_cuda_durations(monkeypatch) -> None:
    observed: dict[str, object] = {}
    root = _Event(MODULE.OBJECTIVE_MARKER, cpu_children=[_Event("aten::index_select", 100.0)])

    class _Profile:
        def __init__(self, *, activities):
            observed["activities"] = activities

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def events(self):
            return [root]

    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class _Tensor:
        shape = (2, 3)

        def detach(self):
            return self

        def requires_grad_(self, _value):
            return self

    class _Loss:
        def backward(self):
            observed["backward"] = True

    def _record_function(name):
        observed["marker"] = name
        return _Context()

    fake_torch = types.SimpleNamespace(
        profiler=types.SimpleNamespace(
            ProfilerActivity=types.SimpleNamespace(CPU="cpu", CUDA="cuda"),
            profile=_Profile,
            record_function=_record_function,
        ),
        cuda=types.SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(MODULE, "trainer_objective_masks", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(MODULE, "trainer_loss", lambda *_args, **_kwargs: _Loss())

    seconds = MODULE._profile_objective({"classifier": _Tensor()}, _Tensor(), _Tensor())

    assert observed["activities"] == ["cpu", "cuda"]
    assert observed["marker"] == MODULE.OBJECTIVE_MARKER
    assert observed["backward"] is True
    assert seconds == pytest.approx(0.0001)


def test_profile_summary_uses_fixed_bootstrap_and_conservative_kernel_gate() -> None:
    timing = tuple(_sample(1.0, 0.08, 0.06) for _ in range(50))
    fusible = tuple(0.11 for _ in range(10))

    summary = MODULE.summarize_profile(timing, fusible)

    assert summary["step_wall_seconds"] == 1.0
    assert summary["fusible_non_backbone_seconds"] == pytest.approx(0.11)
    assert summary["fusible_non_backbone_fraction"] == pytest.approx(0.11)
    assert summary["fusible_fraction_bootstrap_lower_95"] == pytest.approx(0.11)
    assert summary["kernel_gate_threshold"] == 0.1
    assert summary["kernel_gate_passed"] is True

    below = MODULE.summarize_profile(timing, tuple(0.09 for _ in range(10)))
    assert below["kernel_gate_passed"] is False
    assert np.isfinite(below["fusible_fraction_bootstrap_lower_95"])


@pytest.mark.parametrize(
    "timing,fusible",
    [
        ((), (0.01,)),
        ((_sample(1.0, 0.04, 0.03),), ()),
        ((_sample(1.0, 0.04, 0.03),), (-0.01,)),
        ((_sample(1.0, 0.04, 0.03),), (1.01,)),
        ((_sample(1.0, 0.04, 0.03),), (float("nan"),)),
    ],
)
def test_profile_summary_rejects_invalid_or_impossible_samples(timing, fusible) -> None:
    with pytest.raises((TypeError, ValueError)):
        MODULE.summarize_profile(timing, fusible)


def test_cli_defaults_freeze_replay_counts_and_require_one_checkpoint(tmp_path: Path) -> None:
    args = MODULE.parse_args(
        [
            "--run-checkpoint",
            str(tmp_path / "epoch-0004.pt"),
            "--run-receipt",
            str(tmp_path / "run-receipt.json"),
            "--config",
            str(tmp_path / "config.json"),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--runtime-mode",
            "current",
            "--profile-kind",
            "runtime",
            "--parent-trainer-source",
            MODULE.PARENT_TRAINER_SOURCE,
            "--output",
            str(tmp_path / "profile.json"),
        ]
    )
    assert args.warmup_steps == 20
    assert args.measure_steps == 50
    assert args.profiler_steps == 10
    assert args.bootstrap_seed == 20_016
    assert args.output == tmp_path / "profile.json"

    args.measure_steps = 49
    with pytest.raises(ValueError, match="registered profiler"):
        MODULE._validate_counts(args)


def test_replay_scheduler_step_keeps_exhausted_onecycle_at_terminal_state() -> None:
    import torch

    parameters = (torch.nn.Parameter(torch.tensor(1.0)), torch.nn.Parameter(torch.tensor(2.0)))
    optimizer = torch.optim.SGD(
        ({"params": (parameters[0],), "lr": 1e-5}, {"params": (parameters[1],), "lr": 1e-4})
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-5, 1e-4],
        total_steps=2_576,
        pct_start=0.1,
    )
    for _ in range(2_576):
        optimizer.step()
        scheduler.step()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    restored_parameters = (
        torch.nn.Parameter(torch.tensor(1.0)),
        torch.nn.Parameter(torch.tensor(2.0)),
    )
    restored_optimizer = torch.optim.SGD(
        (
            {"params": (restored_parameters[0],), "lr": 1e-5},
            {"params": (restored_parameters[1],), "lr": 1e-4},
        )
    )
    restored_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        restored_optimizer,
        max_lr=[1e-5, 1e-4],
        total_steps=2_576,
        pct_start=0.1,
    )
    restored_optimizer.load_state_dict(optimizer_state)
    restored_scheduler.load_state_dict(scheduler_state)
    terminal_epoch = restored_scheduler.last_epoch
    terminal_lrs = tuple(restored_scheduler.get_last_lr())

    for _ in range(70):
        MODULE._step_replay_scheduler(restored_scheduler)

    assert terminal_epoch == restored_scheduler.total_steps == restored_scheduler.last_epoch
    assert tuple(restored_scheduler.get_last_lr()) == terminal_lrs


def test_replay_scheduler_step_advances_nonterminal_onecycle() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=3,
    )

    optimizer.step()
    MODULE._step_replay_scheduler(scheduler)

    assert scheduler.last_epoch == 1


def test_optimizer_step_updates_parameters_with_exhausted_replay_scheduler() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD((parameter,), lr=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        total_steps=1,
    )
    optimizer.step()
    scheduler.step()
    parameter.grad = torch.tensor(1.0)

    MODULE._optimizer_step({"optimizer": optimizer, "scheduler": scheduler, "scaler": None})

    assert parameter.item() < 1.0
    assert scheduler.last_epoch == scheduler.total_steps == 1


def test_atomic_writer_roundtrips_and_never_clobbers(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "profile.json"
    payload = {"schema_version": "test", "value": 1.25}

    MODULE.write_json_atomic(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert destination.read_bytes().endswith(b"\n")
    assert not tuple(destination.parent.glob(".*.tmp"))
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.write_json_atomic(destination, {"value": 9})
    assert destination.read_bytes() == original


def test_main_publishes_one_profile_and_reports_gate(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "profile.json"
    expected = {"profile_kind": "quality", "runtime_mode": "current"}
    monkeypatch.setattr(MODULE, "replay_profile", lambda _args: expected)
    arguments = [
        "--run-checkpoint",
        str(tmp_path / "epoch-0004.pt"),
        "--run-receipt",
        str(tmp_path / "run-receipt.json"),
        "--config",
        str(tmp_path / "config.json"),
        "--unicom-checkout",
        str(tmp_path / "unicom"),
        "--initial-checkpoint",
        str(tmp_path / "initial.pt"),
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--runtime-mode",
        "current",
        "--profile-kind",
        "quality",
        "--parent-trainer-source",
        MODULE.PARENT_TRAINER_SOURCE,
        "--output",
        str(output),
    ]

    assert MODULE.main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert json.loads(capsys.readouterr().out) == {
        "output": str(output),
        "profile_kind": "quality",
        "runtime_mode": "current",
    }
    assert MODULE.main(arguments) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_quality_profile_validator_requires_v2_evidence_and_no_objective_phase() -> None:
    receipt = _runtime_smoke_receipt(
        "current", started_unix_ns=1_000, wall=1.2
    )
    receipt["profile_kind"] = "quality"
    receipt["objective_steps"] = 0
    receipt["objective_call_count"] = 0
    receipt["objective_samples"] = []

    MODULE.validate_quality_profile(receipt)

    changed = copy.deepcopy(receipt)
    changed["objective_call_count"] = 10
    with pytest.raises(ValueError, match="step evidence"):
        MODULE.validate_quality_profile(changed)

    changed = copy.deepcopy(receipt)
    changed["inference_signature"]["operations"][2] = "normalize_prefix512"
    with pytest.raises(ValueError, match="inference signature"):
        MODULE.validate_quality_profile(changed)


def test_replay_profile_rejects_first_invalid_measured_row_before_next_step(
    monkeypatch,
) -> None:
    calls = 0

    def invalid_step(_state, _batch, *, measured):
        nonlocal calls
        calls += 1
        if not measured:
            return None
        sample = _sample(1.0, 0.04, 0.03)
        sample["step_wall_seconds"] = sample["cuda_step_seconds"] / 1.05
        return {
            "timing": sample,
            "loss": 2.0,
            "gradients_finite": True,
            "scaler": {
                "enabled": False,
                "scale_before": None,
                "scale_after": None,
                "skipped": False,
            },
        }

    monkeypatch.setattr(MODULE, "_training_step", invalid_step)
    monkeypatch.setattr(MODULE, "_reset_cuda_peaks", lambda: None)

    with pytest.raises(ValueError, match="timing sample 0 CUDA step exceeds wall step"):
        MODULE._execute_profile_phases({"loader": (object(),)}, profile_kind="quality")

    assert calls == 21


def _profile_payload(
    classifier_init: str,
    wall: float,
    fusible: float,
    *,
    started_unix_ns: int = 100,
    finished_unix_ns: int = 200,
) -> dict:
    timing = [_sample(wall, 0.08, 0.06) for _ in range(50)]
    fusible_samples = [fusible for _ in range(10)]
    return {
        "schema_version": "unicom-training-step-profile-v1",
        "run_checkpoint": f"/tmp/{classifier_init}.pt",
        "run_checkpoint_sha256": ("1" if classifier_init == "random" else "2") * 64,
        "trainer_sha256": "3" * 64,
        "objective_sha256": "5" * 64,
        "profiler_sha256": "4" * 64,
        "checkpoint_epoch": 4,
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": finished_unix_ns,
        "classifier_init": classifier_init,
        "warmup_steps": 20,
        "measure_steps": 50,
        "profiler_steps": 10,
        "timing_samples": timing,
        "fusible_samples": fusible_samples,
        "summary": MODULE.summarize_profile(tuple(timing), tuple(fusible_samples)),
        "runtime": {"torch_version": "test", "device_name": "test-gpu"},
    }


def test_abba_aggregation_pools_reloads_by_arm_and_recomputes_gate() -> None:
    profiles = (
        _profile_payload("random", 1.0, 0.08, started_unix_ns=100, finished_unix_ns=200),
        _profile_payload("imprinted", 1.0, 0.11, started_unix_ns=201, finished_unix_ns=300),
        _profile_payload("imprinted", 1.2, 0.13, started_unix_ns=301, finished_unix_ns=400),
        _profile_payload("random", 1.0, 0.10, started_unix_ns=401, finished_unix_ns=500),
    )

    result = MODULE.aggregate_abba_profiles(profiles)

    assert tuple(result) == ("random", "imprinted")
    assert result["random"]["step_wall_seconds"] == pytest.approx(1.0)
    assert result["random"]["fusible_non_backbone_seconds"] == pytest.approx(0.09)
    assert result["random"]["kernel_gate_passed"] is False
    assert result["imprinted"]["step_wall_seconds"] == pytest.approx(1.1)
    assert result["imprinted"]["fusible_non_backbone_seconds"] == pytest.approx(0.12)
    assert result["imprinted"]["kernel_gate_passed"] is True

    wrong_order = list(profiles)
    wrong_order[0] = _profile_payload("imprinted", 1.0, 0.08)
    with pytest.raises(ValueError, match="ABBA"):
        MODULE.aggregate_abba_profiles(tuple(wrong_order))


def test_abba_aggregation_rejects_overlapping_order_or_objective_drift() -> None:
    profiles = [
        _profile_payload("random", 1.0, 0.08, started_unix_ns=100, finished_unix_ns=200),
        _profile_payload("imprinted", 1.0, 0.08, started_unix_ns=201, finished_unix_ns=300),
        _profile_payload("imprinted", 1.0, 0.08, started_unix_ns=301, finished_unix_ns=400),
        _profile_payload("random", 1.0, 0.08, started_unix_ns=401, finished_unix_ns=500),
    ]

    profiles[2]["started_unix_ns"] = 299
    with pytest.raises(ValueError, match="order"):
        MODULE.aggregate_abba_profiles(tuple(profiles))

    profiles[2]["started_unix_ns"] = 301
    profiles[2]["objective_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="provenance"):
        MODULE.aggregate_abba_profiles(tuple(profiles))


def test_abba_aggregation_rejects_forged_summary() -> None:
    profiles = list(
        (
            _profile_payload("random", 1.0, 0.08),
            _profile_payload("imprinted", 1.0, 0.08),
            _profile_payload("imprinted", 1.0, 0.08),
            _profile_payload("random", 1.0, 0.08),
        )
    )
    profiles[2]["summary"] = dict(profiles[2]["summary"])
    profiles[2]["summary"]["kernel_gate_passed"] = True
    with pytest.raises(ValueError, match="summary"):
        MODULE.aggregate_abba_profiles(tuple(profiles))

    profiles[2] = _profile_payload("imprinted", 1.0, 0.08)
    profiles[2]["profiler_sha256"] = "5" * 64
    with pytest.raises(ValueError, match="provenance"):
        MODULE.aggregate_abba_profiles(tuple(profiles))


def _runtime_smoke_receipt(
    runtime_mode: str,
    *,
    started_unix_ns: int,
    wall: float,
) -> dict[str, object]:
    composed = runtime_mode == "composed"
    inference_signature = {
        "schema": "unicom-inference-signature-v1",
        "tensors": [
            {
                "name": "norm.running_mean",
                "kind": "buffer",
                "shape": [2],
                "dtype": "torch.float32",
                "numel": 2,
                "element_size": 4,
                "bytes": 8,
                "sha256": "9" * 64,
            }
        ],
        "total_bytes": 8,
        "aggregate_sha256": "8" * 64,
        "descriptor_dtype": "torch.float32",
        "descriptor_dimension": 512,
        "descriptor_sha256": "7" * 64,
        "operations": [
            "official_forward",
            "full768_l2",
            "prefix512",
            "squared_euclidean",
        ],
    }
    scaler = {
        "enabled": True,
        "scale_before": 65536.0,
        "scale_after": 65536.0,
        "skipped": False,
    }
    loss = 10.0001 if composed else 10.0
    return {
        "schema": "unicom-training-step-profile-v2",
        "profile_kind": "runtime",
        "runtime_mode": runtime_mode,
        "parent_trainer_source": (
            "70c760e57e6c27dec1473eecd4765e0a8cd4cf6b:"
            "scripts/train_unicom_inshop.py"
        ),
        "parent_trainer_sha256": (
            "6eea2dab88ff9e4c5a547f9fe326ebf56879882784c5a80c8e136f6d02b52170"
        ),
        "live_trainer_sha256": "6" * 64,
        "profiler_sha256": "5" * 64,
        "checkpoint": {
            "path": "/registered/seed-2/epoch-0016.pt",
            "sha256": "4" * 64,
            "bytes": 1_000,
        },
        "run_receipt": {
            "path": "/registered/seed-2/run-receipt.json",
            "sha256": "3" * 64,
            "bytes": 2_000,
        },
        "config": {
            "path": "/registered/config.json",
            "sha256": "2" * 64,
            "bytes": 3_000,
        },
        "checkpoint_epoch": 16,
        "checkpoint_protocol": {"trainer_sha256": MODULE.PARENT_TRAINER_SHA256},
        "inference_signature": inference_signature,
        "runtime_overrides": {
            "compile": composed,
            "fused": composed,
            "ema": not composed,
        },
        "warmup_steps": 20,
        "measure_steps": 50,
        "objective_steps": 10,
        "optimizer_step_count": 70,
        "objective_call_count": 10,
        "timing_synchronized": True,
        "peak_reset": {
            "after_warmup": True,
            "before_measurement": True,
            "empty_cache": False,
        },
        "started_unix_ns": started_unix_ns,
        "finished_unix_ns": started_unix_ns + 99,
        "losses": [loss] * 50,
        "unscaled_gradients_finite": [True] * 50,
        "scaler_decisions": [dict(scaler) for _ in range(50)],
        "timing_samples": [_sample(wall, 0.04, 0.03) for _ in range(50)],
        "objective_samples": [0.001] * 10,
        "peak_allocated_bytes": 1_005 if composed else 1_000,
        "peak_reserved_bytes": 2_010 if composed else 2_000,
        "parameter_schema": [
            {
                "name": "raw_model.weight",
                "shape": [2, 2],
                "dtype": "torch.float32",
            },
            {"name": "classifier", "shape": [4, 2], "dtype": "torch.float32"},
        ],
        "optimizer_schema": {
            "param_groups": [["amsgrad", "betas", "eps", "lr", "weight_decay"]],
            "state": [["exp_avg", "exp_avg_sq", "step"]],
        },
        "environment": {
            "python_version": "3.12.3",
            "torch_version": "2.6.0",
            "numpy_version": "2.1.3",
            "cuda_version": "12.4",
            "device_name": "NVIDIA H100 80GB HBM3",
        },
    }


def _passing_runtime_smoke_receipts() -> tuple[dict[str, object], ...]:
    modes = ("current", "composed", "composed", "current") * 2
    return tuple(
        _runtime_smoke_receipt(
            mode,
            started_unix_ns=1_000 + index * 100,
            wall=1.0 if mode == "composed" else 1.2,
        )
        for index, mode in enumerate(modes)
    )


def test_runtime_smoke_decision_passes_exact_abbaabba_evidence() -> None:
    receipts = _passing_runtime_smoke_receipts()

    assert MODULE.compare_runtime_smoke(receipts) == "PASS_COMPOSED"


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("order", "INVALID"),
        ("checkpoint", "INVALID"),
        ("environment", "INVALID"),
        ("overlap", "INVALID"),
        ("loss", "PASS_CURRENT"),
        ("gradient", "INVALID"),
        ("scaler", "INVALID"),
        ("parameter_schema", "INVALID"),
        ("optimizer_schema", "INVALID"),
        ("paired_time", "PASS_CURRENT"),
        ("pooled_time", "PASS_CURRENT"),
        ("allocated", "PASS_CURRENT"),
        ("reserved", "PASS_CURRENT"),
    ],
)
def test_runtime_smoke_decision_mutations_fail_closed(
    mutation: str,
    expected: str,
) -> None:
    receipts = list(copy.deepcopy(_passing_runtime_smoke_receipts()))
    if mutation == "order":
        receipts[0], receipts[1] = receipts[1], receipts[0]
    elif mutation == "checkpoint":
        receipts[2]["checkpoint"]["sha256"] = "1" * 64
    elif mutation == "environment":
        receipts[6]["environment"]["device_name"] = "substituted"
    elif mutation == "overlap":
        receipts[1]["started_unix_ns"] = receipts[0]["finished_unix_ns"] - 1
    elif mutation == "loss":
        receipts[1]["losses"][17] = 10.01
    elif mutation == "gradient":
        receipts[5]["unscaled_gradients_finite"][4] = False
    elif mutation == "scaler":
        receipts[2]["scaler_decisions"][9]["skipped"] = True
    elif mutation == "parameter_schema":
        receipts[6]["parameter_schema"][0]["shape"] = [4, 1]
    elif mutation == "optimizer_schema":
        receipts[1]["optimizer_schema"]["state"][0].append("substituted")
    elif mutation == "paired_time":
        receipts[1]["timing_samples"] = [_sample(1.09, 0.04, 0.03) for _ in range(50)]
    elif mutation == "pooled_time":
        for index in (1, 2, 5, 6):
            receipts[index]["timing_samples"] = [
                _sample(1.08, 0.04, 0.03) for _ in range(50)
            ]
    elif mutation == "allocated":
        receipts[2]["peak_allocated_bytes"] = 1_021
    elif mutation == "reserved":
        receipts[5]["peak_reserved_bytes"] = 2_041
    else:
        raise AssertionError(mutation)

    assert MODULE.compare_runtime_smoke(tuple(receipts)) == expected
