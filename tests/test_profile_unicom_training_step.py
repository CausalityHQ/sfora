from __future__ import annotations

import copy
import hashlib
import importlib
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


def _registered_environment() -> dict[str, object]:
    return {
        "python_vv": "Python 3.12.3",
        "torch": "2.6.0",
        "torchvision": "0.21.0",
        "timm": "1.0.0",
        "numpy": "2.1.3",
        "cuda": "12.4",
        "cudnn": "90100",
        "compile": {"available": "True", "inductor": "registered"},
        "device_uuid": "GPU-registered",
        "gpu_inventory": ["H100, GPU-registered, 550.54"],
        "pyproject_sha256": "1" * 64,
        "uv_lock_sha256": "2" * 64,
        "deterministic_execution": {
            "deterministic_algorithms": True,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        },
    }


def _assert_nested_state_equal(left, right) -> None:
    import torch

    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert tuple(left) == tuple(right)
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif type(left) in (list, tuple):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_state_equal(first, second)
    else:
        assert left == right


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


def test_review_quality_authority_selects_live_trainer_without_weakening_runtime(
    monkeypatch,
) -> None:
    parent = object()
    live = object()
    calls: list[str] = []
    monkeypatch.setattr(
        MODULE,
        "_load_authenticated_parent_trainer",
        lambda _repo, _source: calls.append("parent") or parent,
    )
    monkeypatch.setattr(
        MODULE,
        "_load_authenticated_live_trainer",
        lambda _repo, _config: calls.append("live") or live,
        raising=False,
    )
    repository = MODULE_PATH.parents[1]

    assert (
        MODULE._load_replay_trainer(
            types.SimpleNamespace(
                profile_kind="runtime",
                parent_trainer_source=MODULE.PARENT_TRAINER_SOURCE,
            ),
            repository=repository,
            config={},
        )
        is parent
    )
    assert (
        MODULE._load_replay_trainer(
            types.SimpleNamespace(
                profile_kind="quality",
                parent_trainer_source=MODULE.PARENT_TRAINER_SOURCE,
            ),
            repository=repository,
            config={"live_trainer_sha256": "a" * 64},
        )
        is live
    )
    assert calls == ["parent", "live"]


def test_review_quality_checkpoint_accepts_only_authenticated_live_hash() -> None:
    live_hash = "a" * 64
    checkpoint = {"training_protocol": {"trainer_sha256": live_hash}}

    MODULE._validate_checkpoint_authority_for_profile(
        checkpoint,
        profile_kind="quality",
        live_trainer_sha256=live_hash,
    )
    with pytest.raises(ValueError, match="live trainer"):
        MODULE._validate_checkpoint_authority_for_profile(
            checkpoint,
            profile_kind="quality",
            live_trainer_sha256="b" * 64,
        )
    with pytest.raises(ValueError, match="parent trainer"):
        MODULE._validate_checkpoint_authority_for_profile(
            checkpoint,
            profile_kind="runtime",
            live_trainer_sha256=live_hash,
        )


def test_review_runtime_seed2_full_registered_shape_restores_all_mutable_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import torch

    trainer = MODULE._load_authenticated_parent_trainer(
        MODULE_PATH.parents[1], MODULE.PARENT_TRAINER_SOURCE
    )
    source_model = torch.nn.Linear(2, 2)
    source_classifier = torch.nn.Parameter(torch.full((4, 2), 0.25))
    source_ema = trainer.StepEMA(source_model, source_classifier)
    source_optimizer = torch.optim.AdamW(
        (
            {"params": tuple(source_model.parameters()), "lr": 1e-5},
            {"params": (source_classifier,), "lr": 1e-4},
        ),
        weight_decay=0.0,
    )
    source_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        source_optimizer,
        max_lr=[1e-5, 1e-4],
        total_steps=80,
        pct_start=0.1,
    )
    source_optimizer.zero_grad(set_to_none=True)
    (source_model(torch.ones((1, 2))).sum() + source_classifier.sum()).backward()
    source_optimizer.step()
    source_scheduler.step()
    source_ema.update()
    mask_generator = torch.Generator().manual_seed(23_002)
    _ = torch.rand((3,), generator=mask_generator)
    protocol = {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": MODULE.PARENT_TRAINER_SHA256,
        "unicom_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "initial_checkpoint_sha256": (
            "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
        ),
        "partition_sha256": (
            "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
        ),
        "seed": 2,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 1e-5,
        "classifier_learning_rate": 1e-4,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 512,
        "evaluation_features": 512,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "eval_every": 4,
        "checkpoint_every": 4,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
        "classifier_init": "imprinted",
        "ema_decay": 0.999,
        "ema_update": "optimizer-step-post-hook-trainable-parameters-only",
    }
    checkpoint_path = tmp_path / "seed-2" / "epoch-0016.pt"
    checkpoint_path.parent.mkdir()
    payload = {
        "epoch": 16,
        "model": source_model.state_dict(),
        "classifier": source_classifier.detach().clone(),
        "ema": source_ema.state_dict(),
        "optimizer": source_optimizer.state_dict(),
        "scheduler": source_scheduler.state_dict(),
        "scaler": None,
        "mask_generator": mask_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": [torch.Generator().manual_seed(2).get_state()],
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": protocol,
        "history": [{"epoch": 16, "loss": 2.5}],
    }
    torch.save(payload, checkpoint_path)

    loaded = MODULE._load_checkpoint(checkpoint_path)
    MODULE._validate_parent_checkpoint_authority(loaded)
    restored_model = torch.nn.Linear(2, 2)
    restored_classifier = torch.nn.Parameter(torch.zeros((4, 2)))
    restored_ema = trainer.StepEMA(restored_model, restored_classifier)
    restored_optimizer = torch.optim.AdamW(
        (
            {"params": tuple(restored_model.parameters()), "lr": 1e-5},
            {"params": (restored_classifier,), "lr": 1e-4},
        ),
        weight_decay=0.0,
    )
    restored_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        restored_optimizer,
        max_lr=[1e-5, 1e-4],
        total_steps=80,
        pct_start=0.1,
    )
    restored_mask_generator = torch.Generator().manual_seed(0)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _states: None)
    epoch = MODULE._restore_checkpoint_payload(
        loaded,
        {
            "protocol": protocol,
            "raw_model": restored_model,
            "classifier": restored_classifier,
            "step_ema": restored_ema,
            "optimizer": restored_optimizer,
            "scheduler": restored_scheduler,
            "scaler": None,
            "mask_generator": restored_mask_generator,
            "holdout": {"seed": 0, "fraction": 0.2},
        },
    )

    assert epoch == 16
    _assert_nested_state_equal(restored_model.state_dict(), source_model.state_dict())
    assert torch.equal(restored_classifier, source_classifier)
    _assert_nested_state_equal(restored_ema.state_dict(), source_ema.state_dict())
    _assert_nested_state_equal(
        restored_optimizer.state_dict(), source_optimizer.state_dict()
    )
    _assert_nested_state_equal(
        restored_scheduler.state_dict(), source_scheduler.state_dict()
    )
    assert torch.equal(restored_mask_generator.get_state(), mask_generator.get_state())


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
    root = tmp_path / "campaign"
    root.mkdir()
    output = root / "quality-profile/terminal.json"
    environment_path = root / "preflight/cuda-environment.json"
    environment_path.parent.mkdir()
    environment_payload = (
        json.dumps(_registered_environment(), indent=2, allow_nan=False) + "\n"
    ).encode()
    environment_path.write_bytes(environment_payload)
    budget = {
        "schema": "unicom-fepf-publication-budget-v1",
        "publications": [{
            "name": "quality-profile:terminal",
            "path": "quality-profile/terminal.json",
            "persistent_bytes": 4096,
            "temporary_bytes": 4096,
            "persistent_inodes": 1,
            "temporary_inodes": 1,
        }, {
            "name": "cuda-canary:environment",
            "path": "preflight/cuda-environment.json",
            "persistent_bytes": 4096,
            "temporary_bytes": 4096,
            "persistent_inodes": 1,
            "temporary_inodes": 1,
        }],
    }
    budget_payload = (json.dumps(budget, indent=2, allow_nan=False) + "\n").encode()
    budget_path = root / "preflight/publication-budget.json"
    budget_path.write_bytes(budget_payload)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "artifact_root": str(root),
        "publication_budget": budget,
        "publication_budget_path": "preflight/publication-budget.json",
        "publication_budget_sha256": hashlib.sha256(budget_payload).hexdigest(),
        "cuda_canary_environment": {
            "path": str(environment_path.resolve()),
            "sha256": hashlib.sha256(environment_payload).hexdigest(),
            "bytes": len(environment_payload),
        },
    }, indent=2) + "\n")
    expected = {"profile_kind": "quality", "runtime_mode": "current"}
    monkeypatch.setattr(MODULE, "replay_profile", lambda _args: expected)

    class BudgetLoader:
        @staticmethod
        def exec_module(module) -> None:
            module.validate_external_exact_publication_budget = lambda *_args: budget

    specification = types.SimpleNamespace(loader=BudgetLoader())
    monkeypatch.setattr(
        MODULE.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: specification,
    )
    monkeypatch.setattr(
        MODULE.importlib.util,
        "module_from_spec",
        lambda _specification: types.SimpleNamespace(),
    )
    arguments = [
        "--run-checkpoint",
        str(tmp_path / "epoch-0004.pt"),
        "--run-receipt",
        str(tmp_path / "run-receipt.json"),
        "--config",
        str(config_path),
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
        "--publication-stage",
        "quality-profile",
        "--campaign-root",
        str(root),
        "--environment-authority",
        str(environment_path),
        "--environment-sha256",
        hashlib.sha256(environment_payload).hexdigest(),
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


def test_review_quality_profile_rejects_nonexistent_or_nonterminal_authority() -> None:
    receipt = _runtime_smoke_receipt(
        "current", started_unix_ns=1_000, wall=1.2
    )
    receipt["profile_kind"] = "quality"
    receipt["objective_steps"] = 0
    receipt["objective_call_count"] = 0
    receipt["objective_samples"] = []
    receipt["checkpoint_protocol"]["trainer_sha256"] = receipt[
        "live_trainer_sha256"
    ]

    with pytest.raises(ValueError, match="authority"):
        MODULE.validate_quality_profile(receipt)

    changed = copy.deepcopy(receipt)
    changed["checkpoint_epoch"] = 12
    with pytest.raises(ValueError, match="epoch 16"):
        MODULE.validate_quality_profile(changed)

    changed = copy.deepcopy(receipt)
    changed["objective_call_count"] = 10
    with pytest.raises(ValueError, match="step evidence"):
        MODULE.validate_quality_profile(changed)

    changed = copy.deepcopy(receipt)
    changed["inference_signature"]["operations"][2] = "normalize_prefix512"
    with pytest.raises(ValueError, match="inference signature"):
        MODULE.validate_quality_profile(changed)

    changed = copy.deepcopy(receipt)
    changed["parameter_schema"][0]["name"] = 7
    with pytest.raises(ValueError, match="parameter schema"):
        MODULE.validate_quality_profile(changed)

    changed = copy.deepcopy(receipt)
    changed["optimizer_schema"] = {"param_groups": "forged", "state": []}
    with pytest.raises(ValueError, match="optimizer schema"):
        MODULE.validate_quality_profile(changed)


def test_review4_quality_profile_reloads_canonical_live_chain_and_rejects_mutations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import torch

    receipt = _runtime_smoke_receipt(
        "current", started_unix_ns=1_000, wall=1.2
    )
    receipt["profile_kind"] = "quality"
    receipt["objective_steps"] = 0
    receipt["objective_call_count"] = 0
    receipt["objective_samples"] = []
    live_hash = receipt["live_trainer_sha256"]
    protocol = {
        "trainer_sha256": live_hash,
        "environment": copy.deepcopy(receipt["environment"]),
        "environment_sha256": receipt["environment_sha256"],
    }
    receipt["checkpoint_protocol"] = protocol
    trainer_path = MODULE_PATH.with_name("train_unicom_inshop.py")
    trainer_spec = importlib.util.spec_from_file_location(
        "review4_quality_trainer", trainer_path
    )
    assert trainer_spec is not None and trainer_spec.loader is not None
    trainer_module = importlib.util.module_from_spec(trainer_spec)
    trainer_spec.loader.exec_module(trainer_module)
    signature_model = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.zeros_(signature_model.weight)
    signature = trainer_module.build_inference_signature(
        signature_model, descriptor=torch.zeros((1, 512), dtype=torch.float32)
    )
    receipt["inference_signature"] = signature
    evidence_root = tmp_path / "quality"
    evidence_root.mkdir()
    checkpoint_path = evidence_root / "epoch-0016.pt"
    optimizer_state = {
        "state": {
            0: {
                "exp_avg": torch.zeros((2, 2)),
                "exp_avg_sq": torch.zeros((2, 2)),
                "step": torch.tensor(1.0),
            }
        },
        "param_groups": [
            {
                "params": [0, 1],
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "lr": 1e-4,
            }
        ],
    }
    torch.save(
        {
            "epoch": 16,
            "model": {"weight": torch.zeros((2, 2))},
            "classifier": torch.zeros((4, 2)),
            "ema": {},
            "optimizer": optimizer_state,
            "scheduler": {},
            "scaler": None,
            "mask_generator": torch.Generator().get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [torch.Generator().get_state()],
            "selection_holdout": {"seed": 0, "fraction": 0.2},
            "training_protocol": protocol,
            "history": [],
        },
        checkpoint_path,
    )
    checkpoint_authority = MODULE._file_authority(checkpoint_path)
    run_receipt_path = evidence_root / "run-receipt.json"
    run_receipt = {
        "schema": "unicom-fepf-training-run-receipt-v2",
        "training_protocol": protocol,
        "checkpoints": [
            {
                "epoch": 16,
                "root": "current",
                "path": "epoch-0016.pt",
                "sha256": checkpoint_authority["sha256"],
                "bytes": checkpoint_authority["bytes"],
            }
        ],
        "inference_signature": signature,
    }
    run_receipt_path.write_text(
        json.dumps(run_receipt, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    profiler_hash = MODULE._sha256_file(MODULE_PATH)
    receipt["profiler_sha256"] = profiler_hash
    config_path = tmp_path / "config.json"
    environment_path = tmp_path / "environment.json"
    environment_payload = (
        json.dumps(receipt["environment"], indent=2, allow_nan=False) + "\n"
    ).encode()
    environment_path.write_bytes(environment_payload)
    config = {
        "source_commit": "1" * 40,
        "parent_trainer_commit": MODULE.PARENT_TRAINER_COMMIT,
        "parent_trainer_path": MODULE.PARENT_TRAINER_PATH,
        "parent_trainer_sha256": MODULE.PARENT_TRAINER_SHA256,
        "live_trainer_sha256": live_hash,
        "profiler_sha256": profiler_hash,
        "cuda_canary_environment": {
            "path": str(environment_path.resolve()),
            "sha256": hashlib.sha256(environment_payload).hexdigest(),
            "bytes": len(environment_payload),
        },
        "runtime_inference_signature": signature,
    }
    config_path.write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    receipt["checkpoint"] = checkpoint_authority
    receipt["run_receipt"] = MODULE._file_authority(run_receipt_path)
    receipt["config"] = MODULE._file_authority(config_path)

    observed: list[tuple[dict, Path]] = []

    class _LiveTrainer:
        @staticmethod
        def validate_training_run_receipt_v2(value, *, evidence_root):
            observed.append((value, evidence_root))

    monkeypatch.setattr(
        MODULE,
        "_load_authenticated_live_trainer",
        lambda _repository, _config: _LiveTrainer,
    )
    monkeypatch.setattr(
        MODULE,
        "_git_blob_bytes",
        lambda _repository, source: (
            MODULE_PATH.read_bytes()
            if source.endswith(":scripts/profile_unicom_training_step.py")
            else b"unused"
        ),
    )

    MODULE.validate_quality_profile(receipt)

    assert observed == [(run_receipt, evidence_root)]

    changed = copy.deepcopy(receipt)
    symlink = tmp_path / "config-link.json"
    symlink.symlink_to(config_path)
    changed["config"] = {
        **changed["config"],
        "path": str(symlink),
    }
    with pytest.raises(ValueError, match="config authority"):
        MODULE.validate_quality_profile(changed)

    changed = copy.deepcopy(receipt)
    run_receipt_path.write_text(json.dumps(run_receipt) + "\n", encoding="utf-8")
    changed["run_receipt"] = MODULE._file_authority(run_receipt_path)
    with pytest.raises(ValueError, match="noncanonical"):
        MODULE.validate_quality_profile(changed)

    run_receipt_path.write_text(
        json.dumps(run_receipt, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    changed = copy.deepcopy(receipt)
    changed["environment"]["torch"] = "substituted"
    changed["environment_sha256"] = hashlib.sha256(
        (json.dumps(changed["environment"], indent=2) + "\n").encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="environment"):
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
    environment = _registered_environment()
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
            "param_groups": [
                {
                    "parameter_count": 2,
                    "fields": {
                        "betas": {
                            "kind": "tuple",
                            "items": [{"kind": "float"}, {"kind": "float"}],
                        },
                        "eps": {"kind": "float"},
                        "lr": {"kind": "float"},
                    },
                }
            ],
            "state": [
                {
                    "parameter": 0,
                    "fields": {
                        "exp_avg": {
                            "kind": "tensor",
                            "shape": [2, 2],
                            "dtype": "torch.float32",
                        },
                        "exp_avg_sq": {
                            "kind": "tensor",
                            "shape": [2, 2],
                            "dtype": "torch.float32",
                        },
                        "step": {
                            "kind": "tensor",
                            "shape": [],
                            "dtype": "torch.float32",
                        },
                    },
                }
            ],
        },
        "environment_sha256": hashlib.sha256(
            (json.dumps(environment, indent=2, allow_nan=False) + "\n").encode()
        ).hexdigest(),
        "environment": environment,
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


def test_review2_public_runtime_validator_reloads_external_authorities(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    run_receipt = tmp_path / "run-receipt.json"
    config = tmp_path / "config.json"
    checkpoint.write_bytes(b"checkpoint")
    run_receipt.write_bytes(b"run receipt")
    config.write_bytes(b"config")
    receipt = _runtime_smoke_receipt("current", started_unix_ns=1_000, wall=1.2)
    import hashlib

    for key, path in (
        ("checkpoint", checkpoint), ("run_receipt", run_receipt), ("config", config)
    ):
        receipt[key] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    with pytest.raises(ValueError, match="authority"):
        MODULE.validate_runtime_profile(
            receipt, expected_mode="current", checkpoint=checkpoint,
            run_receipt=run_receipt, config=config,
            expected_environment=receipt["environment"],
        )
    checkpoint.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="authority"):
        MODULE.validate_runtime_profile(
            receipt, expected_mode="current", checkpoint=checkpoint,
            run_receipt=run_receipt, config=config,
            expected_environment=receipt["environment"],
        )


def test_review3_profile_cli_binds_complete_environment_authority(tmp_path: Path) -> None:
    environment = _registered_environment()
    authority = tmp_path / "environment.json"
    payload = (json.dumps(environment, indent=2, allow_nan=False) + "\n").encode()
    authority.write_bytes(payload)
    args = MODULE.parse_args(
        [
            "--run-checkpoint", str(tmp_path / "epoch-0016.pt"),
            "--run-receipt", str(tmp_path / "run-receipt.json"),
            "--config", str(tmp_path / "config.json"),
            "--unicom-checkout", str(tmp_path / "unicom"),
            "--initial-checkpoint", str(tmp_path / "initial.pt"),
            "--dataset-root", str(tmp_path / "dataset"),
            "--runtime-mode", "current",
            "--profile-kind", "runtime",
            "--parent-trainer-source", MODULE.PARENT_TRAINER_SOURCE,
            "--environment-authority", str(authority),
            "--environment-sha256", hashlib.sha256(payload).hexdigest(),
            "--output", str(tmp_path / "profile.json"),
        ]
    )
    assert MODULE.load_registered_environment_authority(
        args.environment_authority, args.environment_sha256
    ) == environment


def test_review3_profile_publication_rejects_destination_inode_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "profile.json"
    publication = importlib.import_module("sfora.atomic_publication")
    original = publication._link_fd_noreplace

    def substitute(descriptor: int, directory_descriptor: int, name: str) -> None:
        original(descriptor, directory_descriptor, name)
        payload = destination.read_bytes()
        destination.unlink()
        destination.write_bytes(payload)

    monkeypatch.setattr(publication, "_link_fd_noreplace", substitute)
    with pytest.raises(RuntimeError, match="inode"):
        MODULE.write_json_atomic(destination, {"registered": True})
    assert destination.read_bytes() == b'{\n  "registered": true\n}\n'


def test_review4_synthesized_runtime_roots_minimal_run_and_partial_checkpoint_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    trainer_path = MODULE_PATH.with_name("train_unicom_inshop.py")
    spec = importlib.util.spec_from_file_location("review4_trainer", trainer_path)
    assert spec is not None and spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = torch.nn.Linear(2, 2, bias=False)
    descriptor = torch.zeros((1, 512), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters())
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    signature = trainer.build_inference_signature(model, descriptor=descriptor)
    checkpoint_protocol = {"trainer_sha256": MODULE.PARENT_TRAINER_SHA256}
    checkpoint_paths = []
    for epoch in (4, 8, 12, 16):
        path = tmp_path / f"epoch-{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": dict(model.state_dict()),
                "classifier": torch.zeros((4, 2)),
                "ema": {},
                "optimizer": optimizer.state_dict(),
                "scheduler": {},
                "scaler": None,
                "mask_generator": torch.Generator().get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": [torch.Generator().get_state()],
                "selection_holdout": {"seed": 0, "fraction": 0.2},
                "training_protocol": checkpoint_protocol,
                "history": [],
            },
            path,
        )
        checkpoint_paths.append(path)
    history = tmp_path / "history.json"
    history.write_text("[]\n")
    legacy_config_path = tmp_path / "legacy-config.json"
    profiler_hash = MODULE._sha256_file(MODULE_PATH)
    legacy_config = {
        "source_commit": "1" * 40,
        "parent_trainer_commit": MODULE.PARENT_TRAINER_COMMIT,
        "parent_trainer_path": MODULE.PARENT_TRAINER_PATH,
        "parent_trainer_sha256": MODULE.PARENT_TRAINER_SHA256,
        "profiler_sha256": profiler_hash,
        "runtime_inference_signature": signature,
    }
    legacy_config_path.write_text(json.dumps(legacy_config, indent=2) + "\n")
    run_object = trainer.training_run_receipt(
        source_commit="1" * 40,
        config_path=str(legacy_config_path),
        config_sha256=hashlib.sha256(legacy_config_path.read_bytes()).hexdigest(),
        seed=2,
        arm="sampled_512",
        objective="official-eight-mask",
        selected_features=512,
        evaluation_features=768,
        command=["python", "trainer.py", "--classifier-init", "imprinted"],
        started_unix_ns=1,
        finished_unix_ns=2,
        elapsed_seconds=1.0,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
        exit_status=0,
        history_path=history,
        checkpoint_paths=tuple(checkpoint_paths),
        runtime={"python": "3.12", "torch": "2.6", "cuda": "12.4"},
    )
    run_path = tmp_path / "run-receipt.json"
    run_path.write_text(json.dumps(run_object, indent=2) + "\n")

    def authority(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    config_path = tmp_path / "config.json"
    config = {
        **legacy_config,
        "legacy_runtime_authority": {
            "run_receipt": authority(run_path),
            "config": authority(legacy_config_path),
            "history": authority(history),
            "checkpoints": [
                {"epoch": epoch, **authority(path)}
                for epoch, path in zip((4, 8, 12, 16), checkpoint_paths, strict=True)
            ],
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    receipt = _runtime_smoke_receipt("current", started_unix_ns=10, wall=1.2)
    receipt["profiler_sha256"] = profiler_hash
    receipt["checkpoint_protocol"] = checkpoint_protocol
    receipt["inference_signature"] = signature
    receipt["checkpoint"] = MODULE._file_authority(checkpoint_paths[-1])
    receipt["run_receipt"] = MODULE._file_authority(run_path)
    receipt["config"] = MODULE._file_authority(config_path)
    checkpoint = MODULE._load_checkpoint(checkpoint_paths[-1])
    receipt["parameter_schema"] = MODULE._checkpoint_parameter_schema(
        checkpoint, signature
    )
    receipt["optimizer_schema"] = MODULE._optimizer_state_dict_schema(
        checkpoint["optimizer"]
    )
    monkeypatch.setattr(MODULE, "_git_blob_bytes", lambda *_args: MODULE_PATH.read_bytes())
    monkeypatch.setattr(
        MODULE,
        "_load_authenticated_parent_trainer",
        lambda *_args: trainer,
    )
    MODULE.validate_runtime_profile(
        receipt,
        expected_mode="current",
        checkpoint=checkpoint_paths[-1],
        run_receipt=run_path,
        config=config_path,
        expected_environment=receipt["environment"],
        expected_environment_sha256=receipt["environment_sha256"],
    )
    forged = copy.deepcopy(receipt)
    forged["inference_signature"]["tensors"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="inference"):
        MODULE.validate_runtime_profile(
            forged,
            expected_mode="current",
            checkpoint=checkpoint_paths[-1],
            run_receipt=run_path,
            config=config_path,
            expected_environment=receipt["environment"],
            expected_environment_sha256=receipt["environment_sha256"],
        )


def test_review4_environment_loader_is_one_read_and_config_rooted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _runtime_smoke_receipt(
        "current", started_unix_ns=1, wall=1.0
    )["environment"]
    path = tmp_path / "environment.json"
    first = (json.dumps(environment, indent=2) + "\n").encode()
    substituted = {**environment, "torch": "substituted"}
    second = (json.dumps(substituted, indent=2) + "\n").encode()
    path.write_bytes(first)
    config_path = tmp_path / "config.json"
    authority = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(first).hexdigest(),
        "bytes": len(first),
    }
    config_path.write_text(
        json.dumps({"cuda_canary_environment": authority}, indent=2) + "\n"
    )
    reads = iter((first, second))
    original = Path.read_bytes

    def swapping_read(candidate: Path) -> bytes:
        if candidate == path:
            return next(reads)
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", swapping_read)
    assert MODULE.load_configured_environment_authority(
        config_path, path, authority["sha256"]
    ) == environment
    assert next(reads) == second


def test_review4_profile_publication_rechecks_destination_after_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "profile.json"
    publication = importlib.import_module("sfora.atomic_publication")
    original = publication._pread_all
    reads = 0

    def substitute_after_reopen(descriptor: int) -> bytes:
        nonlocal reads
        payload = original(descriptor)
        reads += 1
        if reads == 2:
            destination.unlink()
            destination.write_bytes(payload)
        return payload

    monkeypatch.setattr(publication, "_pread_all", substitute_after_reopen)
    with pytest.raises(RuntimeError, match="inode"):
        MODULE.write_json_atomic(destination, {"registered": True})
    assert destination.is_file()


def test_review8_non_authentic_synthesized_legacy_runtime_uses_external_root(
    tmp_path: Path,
) -> None:
    import torch

    trainer_path = MODULE_PATH.with_name("train_unicom_inshop.py")
    spec = importlib.util.spec_from_file_location("review5_legacy_trainer", trainer_path)
    assert spec is not None and spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    model = torch.nn.Linear(2, 2, bias=False)
    signature = trainer.build_inference_signature(
        model, descriptor=torch.zeros((1, 512), dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoints = []
    for epoch in (4, 8, 12, 16):
        path = tmp_path / f"epoch-{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": dict(model.state_dict()),
                "classifier": torch.zeros((4, 2)),
                "ema": {},
                "optimizer": optimizer.state_dict(),
                "scheduler": {},
                "scaler": None,
                "mask_generator": torch.Generator().get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": [torch.Generator().get_state()],
                "selection_holdout": {"seed": 0, "fraction": 0.2},
                "training_protocol": {
                    "trainer_sha256": MODULE.PARENT_TRAINER_SHA256
                },
                "history": [],
            },
            path,
        )
        checkpoints.append(path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "parent_trainer_commit": MODULE.PARENT_TRAINER_COMMIT,
                "parent_trainer_path": MODULE.PARENT_TRAINER_PATH,
                "parent_trainer_sha256": MODULE.PARENT_TRAINER_SHA256,
                "runtime_inference_signature": signature,
            },
            indent=2,
        )
        + "\n"
    )
    history = tmp_path / "history.json"
    history.write_text("[]\n")
    legacy = trainer.training_run_receipt(
        source_commit="1" * 40,
        config_path=str(config_path),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        seed=2,
        arm="sampled_512",
        objective="official-eight-mask",
        selected_features=512,
        evaluation_features=768,
        command=["python", "trainer.py", "--classifier-init", "imprinted"],
        started_unix_ns=1,
        finished_unix_ns=2,
        elapsed_seconds=1.0,
        peak_allocated_bytes=1,
        peak_reserved_bytes=2,
        exit_status=0,
        history_path=history,
        checkpoint_paths=tuple(checkpoints),
        runtime={"python": "3.12", "torch": "2.6", "cuda": "12.4"},
    )
    run_receipt = tmp_path / "run-receipt.json"
    run_receipt.write_text(json.dumps(legacy, indent=2) + "\n")
    _run, _config, loaded = MODULE._load_profile_authorities(
        types.SimpleNamespace(
            run_receipt=run_receipt,
            config=config_path,
            run_checkpoint=checkpoints[-1],
            profile_kind="runtime",
        )
    )
    assert loaded == signature

    def authority(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    environment = tmp_path / "environment.json"
    environment_value = _registered_environment()
    environment.write_text(json.dumps(environment_value, indent=2) + "\n")
    campaign_root = tmp_path / "campaign"
    output = campaign_root / "runtime-00/terminal.json"
    output.parent.mkdir(parents=True)
    budget = {
        "schema": "unicom-fepf-publication-budget-v1",
        "publications": [{
            "name": "runtime-00:terminal",
            "path": "runtime-00/terminal.json",
            "persistent_bytes": 1 << 20,
            "temporary_bytes": 1 << 20,
            "persistent_inodes": 1,
            "temporary_inodes": 1,
        }],
    }
    budget_payload = (json.dumps(budget, indent=2) + "\n").encode()
    budget_path = campaign_root / "preflight/publication-budget.json"
    budget_path.parent.mkdir()
    budget_path.write_bytes(budget_payload)
    fepf_config = tmp_path / "fepf-config.json"
    fepf_config_value = {
        "artifact_root": str(campaign_root),
        "runtime_inference_signature": signature,
        "cuda_canary_environment": {
            "path": str(environment.resolve()),
            "sha256": hashlib.sha256(environment.read_bytes()).hexdigest(),
            "bytes": len(environment.read_bytes()),
        },
        "publication_budget_path": "preflight/publication-budget.json",
        "artifact_budget_inputs": {
            "raw_backbone_state_bytes": 16,
            "classifier_state_bytes": 16,
            "query_rows": 2,
            "gallery_rows": 2,
            "maximum_relevant_count": 1,
            "maximum_path_bytes": 64,
        },
        "legacy_runtime_authority": {
            "run_receipt": authority(run_receipt),
            "config": authority(config_path),
            "history": authority(history),
            "checkpoints": [
                {"epoch": epoch, **authority(path)}
                for epoch, path in zip((4, 8, 12, 16), checkpoints, strict=True)
            ],
        },
    }
    builder_spec = importlib.util.spec_from_file_location(
        "review8_exact_budget_builder",
        MODULE_PATH.with_name("build_unicom_fepf_run_config.py"),
    )
    assert builder_spec is not None and builder_spec.loader is not None
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    exact_budget = builder.exact_publication_budget(fepf_config_value)
    fepf_config_value["publication_budget"] = exact_budget
    fepf_config_value["publication_budget_sha256"] = hashlib.sha256(
        builder.canonical_json_bytes(exact_budget)
    ).hexdigest()
    fepf_config_value["artifact_budget_bytes"] = sum(
        row["persistent_bytes"] + row["temporary_bytes"]
        for row in exact_budget["publications"]
    )
    fepf_config_value["artifact_budget_inodes"] = sum(
        row["persistent_inodes"] + row["temporary_inodes"]
        for row in exact_budget["publications"]
    )
    budget_path.write_bytes(builder.canonical_json_bytes(exact_budget))
    fepf_config.write_text(json.dumps(fepf_config_value, indent=2) + "\n")
    assert MODULE.main([
        "--run-checkpoint", str(checkpoints[-1]),
        "--run-receipt", str(run_receipt),
        "--config", str(fepf_config),
        "--unicom-checkout", str(tmp_path / "unicom"),
        "--initial-checkpoint", str(tmp_path / "initial.pt"),
        "--dataset-root", str(tmp_path / "dataset"),
        "--runtime-mode", "current",
        "--profile-kind", "runtime",
        "--parent-trainer-source", MODULE.PARENT_TRAINER_SOURCE,
        "--output", str(output),
        "--environment-authority", str(environment),
        "--environment-sha256", hashlib.sha256(environment.read_bytes()).hexdigest(),
        "--publication-stage", "runtime-00",
        "--campaign-root", str(campaign_root),
        "--authority-preflight-only",
    ]) == 0


def test_review6_live_environment_is_checked_before_any_fepf_authority_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(MODULE, "_validate_counts", lambda _args: None)
    monkeypatch.setattr(
        MODULE,
        "load_registered_environment_authority",
        lambda *_args: {"unrooted": True},
    )

    def configured(*_args):
        seen.append("environment")
        raise ValueError("environment first")

    def artifacts(*_args):
        seen.append("artifacts")
        raise ValueError("artifacts read")

    monkeypatch.setattr(
        MODULE,
        "reload_normal_legacy_runtime_authority",
        lambda *_args: seen.append("legacy") or {},
    )
    monkeypatch.setattr(MODULE, "load_configured_environment_authority", configured)
    monkeypatch.setattr(MODULE, "_load_profile_authorities", artifacts)
    args = types.SimpleNamespace(
        parent_trainer_source=MODULE.PARENT_TRAINER_SOURCE,
        environment_authority=tmp_path / "environment.json",
        environment_sha256="1" * 64,
        config=tmp_path / "config.json",
        run_receipt=tmp_path / "non-authentic-run-receipt.json",
        profile_kind="runtime",
    )
    with pytest.raises(ValueError, match="environment first"):
        MODULE.replay_profile(args)
    assert seen == ["legacy", "environment"]


def test_review6_legacy_runtime_chain_is_exact_and_descriptor_is_recomputed(
    tmp_path: Path,
) -> None:
    legacy = {
        "seed": 2,
        "arm": "sampled_512",
        "protocol": {
            "objective": "official-eight-mask",
            "selected_features": 512,
            "evaluation_features": 768,
        },
        "exit_status": 0,
        "checkpoints": [{"epoch": epoch} for epoch in (4, 8, 12, 16)],
        "history": {"path": "/registered/history.json"},
        "command": ["python", "trainer.py", "--classifier-init", "imprinted"],
    }
    MODULE.validate_registered_legacy_runtime_chain(legacy)
    changed = copy.deepcopy(legacy)
    changed["seed"] = 3
    with pytest.raises(ValueError, match="seed|legacy"):
        MODULE.validate_registered_legacy_runtime_chain(changed)

    descriptor = np.zeros((1, 512), dtype=np.float32)
    path = tmp_path / "runtime-descriptor.npy"
    np.save(path, descriptor, allow_pickle=False)
    signature = {
        "descriptor_dtype": "torch.float32",
        "descriptor_dimension": 512,
        "descriptor_sha256": hashlib.sha256(descriptor.tobytes()).hexdigest(),
        "operations": list(MODULE.INFERENCE_OPERATIONS),
    }
    MODULE.validate_runtime_descriptor_authority(signature, path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="descriptor"):
        MODULE.validate_runtime_descriptor_authority(signature, path)


def test_review7_legacy_chain_is_rooted_outside_historical_receipt(
    tmp_path: Path,
) -> None:
    historical_config = tmp_path / "historical-config.json"
    history = tmp_path / "history.json"
    historical_config.write_text("{}\n")
    history.write_text("[]\n")
    checkpoints = []
    for epoch in (4, 8, 12, 16):
        path = tmp_path / f"epoch-{epoch:04d}.pt"
        path.write_bytes(f"checkpoint-{epoch}\n".encode())
        checkpoints.append({"epoch": epoch, **MODULE._file_authority(path)})
    receipt = {
        "seed": 2,
        "arm": "sampled_512",
        "protocol": {
            "objective": "official-eight-mask",
            "selected_features": 512,
            "evaluation_features": 768,
        },
        "exit_status": 0,
        "checkpoints": checkpoints,
        "history": MODULE._file_authority(history),
        "config_path": str(historical_config.resolve()),
        "config_sha256": MODULE._sha256_file(historical_config),
        "command": ["python", "trainer.py", "--classifier-init", "imprinted"],
    }
    authority = {
        "run_receipt": {"path": "/external/run.json", "sha256": "1" * 64, "bytes": 1},
        "config": MODULE._file_authority(historical_config),
        "history": MODULE._file_authority(history),
        "checkpoints": checkpoints,
    }
    MODULE.validate_registered_legacy_runtime_chain(receipt, authority=authority)
    changed = copy.deepcopy(authority)
    changed["checkpoints"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="legacy|external"):
        MODULE.validate_registered_legacy_runtime_chain(receipt, authority=changed)


def test_review8_profile_main_guards_budget_and_derives_live_tensor_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "publication_budget": {
            "schema": "unicom-fepf-publication-budget-v1", "publications": []
        },
        "runtime_inference_signature": {
            "tensors": [{"name": "running_mean", "kind": "parameter"}],
            "descriptor_dimension": 511,
        },
    }, indent=2) + "\n")
    reached_replay = False

    def forbidden(_args) -> object:
        nonlocal reached_replay
        reached_replay = True
        return {}

    monkeypatch.setattr(MODULE, "replay_profile", forbidden)
    result = MODULE.main([
        "--run-checkpoint", str(tmp_path / "epoch-0016.pt"),
        "--run-receipt", str(tmp_path / "run-receipt.json"),
        "--config", str(config),
        "--unicom-checkout", str(tmp_path / "unicom"),
        "--initial-checkpoint", str(tmp_path / "initial.pt"),
        "--dataset-root", str(tmp_path / "dataset"),
        "--runtime-mode", "current",
        "--profile-kind", "runtime",
        "--parent-trainer-source", MODULE.PARENT_TRAINER_SOURCE,
        "--output", str(tmp_path / "profile.json"),
        "--environment-authority", str(tmp_path / "environment.json"),
        "--environment-sha256", "1" * 64,
        "--authority-preflight-only",
    ])
    assert result == 2
    assert reached_replay is False


def test_review9_normal_runtime_replay_consumes_external_legacy_authority(
    tmp_path: Path,
) -> None:
    def authority(path: Path) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }

    legacy_config = tmp_path / "non-authentic-legacy-config.json"
    legacy_config.write_text('{"fixture":"non-authentic"}\n')
    history = tmp_path / "non-authentic-history.json"
    history.write_text("[]\n")
    checkpoints: list[dict[str, object]] = []
    for epoch in (4, 8, 12, 16):
        checkpoint = tmp_path / f"non-authentic-epoch-{epoch:04d}.pt"
        checkpoint.write_bytes(f"non-authentic checkpoint {epoch}\n".encode())
        checkpoints.append({"epoch": epoch, **authority(checkpoint)})
    receipt_object = {
        "seed": 2,
        "arm": "sampled_512",
        "protocol": {
            "objective": "official-eight-mask",
            "selected_features": 512,
            "evaluation_features": 768,
        },
        "command": [
            "python", "trainer.py", "--classifier-init", "imprinted"
        ],
        "exit_status": 0,
        "config_path": str(legacy_config.resolve()),
        "config_sha256": authority(legacy_config)["sha256"],
        "history": authority(history),
        "checkpoints": checkpoints,
    }
    receipt = tmp_path / "non-authentic-run-receipt.json"
    receipt.write_text(json.dumps(receipt_object, indent=2) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "legacy_runtime_authority": {
            "run_receipt": authority(receipt),
            "config": authority(legacy_config),
            "history": authority(history),
            "checkpoints": checkpoints,
        }
    }, indent=2) + "\n")
    observed = MODULE.reload_normal_legacy_runtime_authority(config, receipt)
    assert observed["seed"] == 2


def test_review10_public_runtime_replay_rejects_legacy_drift_before_model_or_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    environment = _registered_environment()
    monkeypatch.setattr(MODULE, "_validate_counts", lambda _args: None)
    monkeypatch.setattr(
        MODULE, "load_configured_environment_authority", lambda *_args: environment
    )
    monkeypatch.setattr(MODULE, "_runtime_environment", lambda *_args: environment)
    def reject_legacy(_config: Path, _receipt: Path) -> object:
        seen.append("legacy")
        raise ValueError("legacy external root differs")

    def forbidden_profile_read(_args) -> object:
        seen.append("model")
        raise AssertionError("model/checkpoint read preceded legacy validation")

    monkeypatch.setattr(
        MODULE, "reload_normal_legacy_runtime_authority", reject_legacy
    )
    monkeypatch.setattr(MODULE, "_load_profile_authorities", forbidden_profile_read)
    args = types.SimpleNamespace(
        parent_trainer_source=MODULE.PARENT_TRAINER_SOURCE,
        environment_authority=tmp_path / "environment.json",
        environment_sha256="1" * 64,
        config=tmp_path / "config.json",
        run_receipt=tmp_path / "legacy-run-receipt.json",
        profile_kind="runtime",
    )
    with pytest.raises(ValueError, match="legacy external root differs"):
        MODULE.replay_profile(args)
    assert seen == ["legacy"]


def test_review12_quality_replay_uses_v2_arm_not_legacy_seed2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _registered_environment()
    reached: list[str] = []
    monkeypatch.setattr(MODULE, "_validate_counts", lambda _args: None)
    monkeypatch.setattr(
        MODULE, "reload_normal_legacy_runtime_authority",
        lambda *_args: (_ for _ in ()).throw(AssertionError("quality touched legacy")),
    )
    monkeypatch.setattr(
        MODULE, "load_configured_environment_authority", lambda *_args: environment
    )
    monkeypatch.setattr(MODULE, "_runtime_environment", lambda *_args: environment)
    monkeypatch.setattr(
        MODULE, "_load_profile_authorities",
        lambda _args: (
            {"schema": "unicom-training-run-receipt-v2", "arm": "candidate"},
            {"live_trainer_sha256": "a" * 64},
            {"schema": "unicom-inference-signature-v1"},
        ),
    )
    monkeypatch.setattr(MODULE, "_load_replay_trainer", lambda *_a, **_k: object())

    def reached_quality(*_args, **_kwargs):
        reached.append("quality-v2")
        raise RuntimeError("quality boundary reached")

    monkeypatch.setattr(MODULE, "_validate_quality_replay_inputs", reached_quality)
    args = types.SimpleNamespace(
        parent_trainer_source=MODULE.PARENT_TRAINER_SOURCE,
        environment_authority=tmp_path / "environment.json",
        environment_sha256="1" * 64,
        config=tmp_path / "config.json",
        run_receipt=tmp_path / "candidate-run-receipt.json",
        profile_kind="quality",
    )
    with pytest.raises(RuntimeError, match="quality boundary reached"):
        MODULE.replay_profile(args)
    assert reached == ["quality-v2"]


@pytest.mark.parametrize("mutate_current", (False, True))
def test_review13_quality_authority_uses_own_v2_signature_not_historical_seed2(
    tmp_path: Path, mutate_current: bool
) -> None:
    import copy

    import torch

    checkpoint = {
        "epoch": 16,
        "model": {
            "weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)
        },
        "classifier": torch.ones((2, 2), dtype=torch.float32),
        "ema": {},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {},
        "scaler": None,
        "mask_generator": torch.Generator().manual_seed(17).get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": [],
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": {
            "trainer_sha256": MODULE.PARENT_TRAINER_SHA256,
            "environment": _registered_environment(),
        },
        "history": [],
    }
    checkpoint_path = tmp_path / "epoch-0016.pt"
    torch.save(checkpoint, checkpoint_path)
    seed_signature = {
        "schema": "unicom-inference-signature-v1",
        "tensors": [{
            "name": "weight", "kind": "parameter", "shape": [2, 2],
            "dtype": "torch.float32", "numel": 4, "element_size": 4,
            "bytes": 16, "sha256": "1" * 64,
        }],
        "total_bytes": 16,
        "aggregate_sha256": "2" * 64,
        "descriptor_dtype": "torch.float32",
        "descriptor_dimension": 512,
        "descriptor_sha256": "3" * 64,
        "operations": list(MODULE.INFERENCE_OPERATIONS),
    }
    current = MODULE._rebuild_checkpoint_inference_signature(
        checkpoint, seed_signature
    )
    historical = copy.deepcopy(current)
    historical["tensors"][0]["sha256"] = "4" * 64
    historical["aggregate_sha256"] = "5" * 64
    run_signature = copy.deepcopy(current)
    if mutate_current:
        run_signature["tensors"][0]["sha256"] = "6" * 64
        run_signature["aggregate_sha256"] = "7" * 64
    run_receipt_path = tmp_path / "run-receipt.json"
    run_receipt_path.write_text(json.dumps({
        "inference_signature": run_signature,
        "checkpoints": [{"epoch": 16, **MODULE._file_authority(checkpoint_path)}],
    }, indent=2) + "\n")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "parent_trainer_commit": MODULE.PARENT_TRAINER_COMMIT,
        "parent_trainer_path": MODULE.PARENT_TRAINER_PATH,
        "parent_trainer_sha256": MODULE.PARENT_TRAINER_SHA256,
        "runtime_inference_signature": historical,
    }, indent=2) + "\n")
    args = types.SimpleNamespace(
        profile_kind="quality",
        run_receipt=run_receipt_path,
        run_checkpoint=checkpoint_path,
        config=config_path,
    )
    if mutate_current:
        with pytest.raises(ValueError, match="quality|inference|signature"):
            MODULE._load_profile_authorities(args)
    else:
        _receipt, _config, observed = MODULE._load_profile_authorities(args)
        assert observed == current
        assert observed != historical


def test_profile_signature_accepts_ordered_state_with_scalar_buffer() -> None:
    import torch

    trainer_path = MODULE_PATH.with_name("train_unicom_inshop.py")
    spec = importlib.util.spec_from_file_location("scalar_signature_trainer", trainer_path)
    assert spec is not None and spec.loader is not None
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(3, 4, bias=False)
            self.register_buffer(
                "num_batches_tracked", torch.tensor(560_388, dtype=torch.int64)
            )

    model = Backbone()
    descriptor = torch.zeros((1, 512), dtype=torch.float32)
    checkpoint = {"model": model.state_dict()}
    external = trainer.build_inference_signature(model, descriptor=descriptor)

    rebuilt = MODULE._rebuild_checkpoint_inference_signature(checkpoint, external)

    assert rebuilt == external
    MODULE.validate_live_runtime_inference_authority(
        checkpoint, model=model, descriptor=descriptor, external=external
    )


def test_review12_profile_reestablishes_complete_deterministic_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled: list[bool] = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            stdout="H100, GPU-registered, 550.54\n"
        ),
    )
    fake = types.SimpleNamespace(
        __version__="2.6.0",
        version=types.SimpleNamespace(cuda="12.4", git_version="registered"),
        use_deterministic_algorithms=lambda value: enabled.append(value),
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),
            cudnn=types.SimpleNamespace(
                allow_tf32=True, benchmark=True, deterministic=False,
                version=lambda: 90100,
            ),
        ),
        cuda=types.SimpleNamespace(
            get_device_properties=lambda _device: types.SimpleNamespace(
                uuid="GPU-registered"
            )
        ),
    )
    environment = MODULE._runtime_environment(
        fake, types.SimpleNamespace(type="cuda")
    )
    assert environment["deterministic_execution"] == _registered_environment()[
        "deterministic_execution"
    ]
    assert enabled == [True]


def test_review12_runtime_legacy_chain_semantics_reject_before_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_config = tmp_path / "legacy-config.json"
    legacy_config.write_text('{"registered":true}\n')
    history = tmp_path / "history.json"
    history.write_text("[]\n")
    checkpoints = []
    for epoch in (4, 8, 12, 16):
        path = tmp_path / f"epoch-{epoch:04d}.pt"
        path.write_bytes(f"checkpoint-{epoch}\n".encode())
        checkpoints.append({"epoch": epoch, **MODULE._file_authority(path)})
    receipt = {
        "seed": 2, "arm": "sampled_512",
        "protocol": {
            "objective": "official-eight-mask", "selected_features": 512,
            "evaluation_features": 768,
        },
        "command": ["python", "trainer.py", "--classifier-init", "imprinted"],
        "exit_status": 0,
        "config_path": str(legacy_config.resolve()),
        "config_sha256": MODULE._sha256_file(legacy_config),
        "history": MODULE._file_authority(history),
        "checkpoints": checkpoints,
    }
    receipt_path = tmp_path / "run-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "legacy_runtime_authority": {
            "run_receipt": MODULE._file_authority(receipt_path),
            "config": MODULE._file_authority(legacy_config),
            "history": MODULE._file_authority(history),
            "checkpoints": checkpoints,
        }
    }, indent=2) + "\n")
    history.write_text('[{"loss":"semantically-invalid"}]\n')
    receipt["history"] = MODULE._file_authority(history)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    config_object = json.loads(config_path.read_bytes())
    config_object["legacy_runtime_authority"]["run_receipt"] = (
        MODULE._file_authority(receipt_path)
    )
    config_object["legacy_runtime_authority"]["history"] = MODULE._file_authority(
        history
    )
    config_path.write_text(json.dumps(config_object, indent=2) + "\n")
    reached_model = False

    def forbidden(*_args, **_kwargs):
        nonlocal reached_model
        reached_model = True
        raise AssertionError("model reached")

    monkeypatch.setattr(MODULE, "_load_profile_authorities", forbidden)
    monkeypatch.setattr(
        MODULE,
        "load_configured_environment_authority",
        lambda *_args: (_ for _ in ()).throw(AssertionError("environment reached")),
    )
    args = types.SimpleNamespace(
        parent_trainer_source=MODULE.PARENT_TRAINER_SOURCE,
        config=config_path, run_receipt=receipt_path, profile_kind="runtime",
        environment_authority=tmp_path / "environment.json",
        environment_sha256="1" * 64,
        warmup_steps=MODULE.WARMUP_STEPS,
        measure_steps=MODULE.MEASURE_STEPS,
        profiler_steps=MODULE.PROFILER_STEPS,
        bootstrap_seed=MODULE.BOOTSTRAP_SEED,
    )
    with pytest.raises(ValueError, match="legacy|history|authority"):
        MODULE.replay_profile(args)
    assert reached_model is False


def test_review9_live_kinds_and_descriptor_are_rebuilt_not_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    model = torch.nn.Linear(2, 2)
    checkpoint = {
        "epoch": 16,
        "model": dict(model.state_dict()),
        "classifier": torch.ones((3, 2)),
        "ema": {},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {},
        "scaler": None,
        "mask_generator": torch.Generator().get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": [torch.Generator().get_state()],
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": {"trainer_sha256": "1" * 64},
        "history": [],
    }
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    external = {
        "schema": "unicom-inference-signature-v1",
        "tensors": [
            {
                "name": name,
                "kind": "buffer",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
                "element_size": value.element_size(),
                "bytes": value.numel() * value.element_size(),
                "sha256": hashlib.sha256(
                    value.detach().cpu().contiguous().numpy().tobytes(order="C")
                ).hexdigest(),
            }
            for name, value in checkpoint["model"].items()
        ],
        "total_bytes": sum(value.numel() * value.element_size()
                           for value in checkpoint["model"].values()),
        "aggregate_sha256": "1" * 64,
        "descriptor_dtype": "torch.float32",
        "descriptor_dimension": 512,
        "descriptor_sha256": "2" * 64,
        "operations": list(MODULE.INFERENCE_OPERATIONS),
    }
    environment = _registered_environment()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(MODULE, "_validate_counts", lambda _args: None)
    monkeypatch.setattr(
        MODULE, "load_configured_environment_authority", lambda *_args: environment
    )
    monkeypatch.setattr(MODULE, "_runtime_environment", lambda *_args: environment)
    config_path = tmp_path / "config.json"
    run_receipt_path = tmp_path / "run-receipt.json"
    config_path.write_text(json.dumps({}, indent=2) + "\n")
    run_receipt_path.write_text(json.dumps({}, indent=2) + "\n")
    monkeypatch.setattr(
        MODULE, "reload_normal_legacy_runtime_authority", lambda *_args: {}
    )
    monkeypatch.setattr(
        MODULE,
        "_load_profile_authorities",
        lambda _args: ({}, {"live_trainer_sha256": MODULE._sha256_file(
            MODULE_PATH.with_name("train_unicom_inshop.py")
        )}, external),
    )
    monkeypatch.setattr(MODULE, "_load_replay_trainer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        MODULE,
        "_build_replay_state",
        lambda *_args, **_kwargs: {
            "raw_model": model,
            "classifier": torch.nn.Parameter(torch.ones((3, 2))),
            "authority_descriptor": torch.zeros((1, 512), dtype=torch.float32),
            "step_ema": None,
            "device": torch.device("cpu"),
        },
    )
    monkeypatch.setattr(
        MODULE,
        "_execute_profile_phases",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("normal replay advanced past live authority validation")
        ),
    )
    args = types.SimpleNamespace(
        parent_trainer_source=MODULE.PARENT_TRAINER_SOURCE,
        environment_authority=tmp_path / "environment.json",
        environment_sha256="3" * 64,
        config=config_path,
        run_receipt=run_receipt_path,
        run_checkpoint=checkpoint_path,
        profile_kind="runtime",
        runtime_mode="current",
    )
    with pytest.raises(ValueError, match="kind|descriptor|live"):
        MODULE.replay_profile(args)


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
        receipts[1]["optimizer_schema"]["state"][0]["fields"]["substituted"] = {
            "kind": "float"
        }
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
