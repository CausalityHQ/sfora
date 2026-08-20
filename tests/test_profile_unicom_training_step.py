from __future__ import annotations

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
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
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
    expected = {"summary": {"kernel_gate_passed": False}}
    monkeypatch.setattr(MODULE, "replay_profile", lambda _args: expected)
    arguments = [
        "--run-checkpoint",
        str(tmp_path / "epoch-0004.pt"),
        "--unicom-checkout",
        str(tmp_path / "unicom"),
        "--initial-checkpoint",
        str(tmp_path / "initial.pt"),
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--output",
        str(output),
    ]

    assert MODULE.main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert json.loads(capsys.readouterr().out)["kernel_gate_passed"] is False
    assert MODULE.main(arguments) == 2
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_replay_profile_validates_its_exact_produced_schema(tmp_path: Path, monkeypatch) -> None:
    import torch

    import sfora

    checkpoint = tmp_path / "epoch-0004.pt"
    checkpoint.write_bytes(b"checkpoint")
    objective_source = tmp_path / "objective.py"
    objective_source.write_text("# objective\n", encoding="utf-8")
    objective_module = types.ModuleType("sfora.unicom_training")
    objective_module.__file__ = str(objective_source)
    monkeypatch.setitem(sys.modules, "sfora.unicom_training", objective_module)
    monkeypatch.setattr(sfora, "unicom_training", objective_module, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "test-gpu")
    monkeypatch.setattr(MODULE, "_validate_counts", lambda _args: None)
    monkeypatch.setattr(MODULE, "_load_trainer", lambda _path: object())

    class _EMA:
        def release_step_hook(self):
            return None

    monkeypatch.setattr(
        MODULE,
        "_build_replay_state",
        lambda _args, _trainer: {
            "loader": (),
            "step_ema": _EMA(),
            "checkpoint_epoch": 4,
            "protocol": {"classifier_init": "random"},
            "device": "cuda:0",
        },
    )
    monkeypatch.setattr(MODULE, "summarize_profile", lambda _timing, _fusible: {"ok": True})
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        MODULE,
        "_profile_samples",
        lambda payload, expected_init: observed.update(
            {"payload": payload, "expected_init": expected_init}
        ),
    )
    args = types.SimpleNamespace(
        run_checkpoint=checkpoint,
        warmup_steps=0,
        measure_steps=0,
        profiler_steps=0,
    )

    payload = MODULE.replay_profile(args)

    assert observed == {"payload": payload, "expected_init": "random"}
    assert tuple(payload) == MODULE.PROFILE_KEYS


def test_replay_profile_rejects_first_invalid_measured_row_before_next_step(
    monkeypatch,
) -> None:
    class _EMA:
        released = False

        def release_step_hook(self):
            self.released = True

    ema = _EMA()
    monkeypatch.setattr(MODULE, "_validate_counts", lambda _args: None)
    monkeypatch.setattr(MODULE, "_load_trainer", lambda _path: object())
    monkeypatch.setattr(
        MODULE,
        "_build_replay_state",
        lambda _args, _trainer: {
            "loader": (object(), object()),
            "step_ema": ema,
            "checkpoint_epoch": 4,
            "protocol": {"classifier_init": "random"},
            "device": "cuda:0",
        },
    )
    calls = 0

    def invalid_step(_state, _batch, *, measured):
        nonlocal calls
        calls += 1
        assert measured is True
        sample = _sample(1.0, 0.04, 0.03)
        sample["step_wall_seconds"] = sample["cuda_step_seconds"] / 1.05
        return sample

    monkeypatch.setattr(MODULE, "_training_step", invalid_step)
    args = types.SimpleNamespace(
        warmup_steps=0,
        measure_steps=2,
        profiler_steps=0,
    )

    with pytest.raises(ValueError, match="timing sample 0 CUDA step exceeds wall step"):
        MODULE.replay_profile(args)

    assert calls == 1
    assert ema.released is True


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
