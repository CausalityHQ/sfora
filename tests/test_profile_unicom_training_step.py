from __future__ import annotations

import importlib.util
import json
import math
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
    assert cuda_step <= step
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


@dataclass
class _Event:
    name: str
    self_device_time_total: float = 0.0
    cpu_children: list[_Event] = field(default_factory=list)


def test_fusible_classifier_excludes_gemms_and_ignores_events_outside_marker() -> None:
    root = _Event(
        MODULE.OBJECTIVE_MARKER,
        cpu_children=[
            _Event("aten::index_select", 120.0),
            _Event("aten::linalg_vector_norm", 80.0),
            _Event("aten::mm", 900.0),
            _Event("aten::linear", 0.0, [_Event("aten::matmul", 700.0)]),
            _Event("aten::_log_softmax", 100.0),
        ],
    )
    outside = _Event("aten::index_select", 50_000.0)

    seconds = MODULE.fusible_nonbackbone_seconds((outside, root))

    assert seconds == pytest.approx(0.0003)


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


def _profile_payload(classifier_init: str, wall: float, fusible: float) -> dict:
    timing = [_sample(wall, 0.08, 0.06) for _ in range(50)]
    fusible_samples = [fusible for _ in range(10)]
    return {
        "schema_version": "unicom-training-step-profile-v1",
        "run_checkpoint_sha256": ("1" if classifier_init == "random" else "2") * 64,
        "trainer_sha256": "3" * 64,
        "profiler_sha256": "4" * 64,
        "checkpoint_epoch": 4,
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
        _profile_payload("random", 1.0, 0.08),
        _profile_payload("imprinted", 1.0, 0.11),
        _profile_payload("imprinted", 1.2, 0.13),
        _profile_payload("random", 1.0, 0.10),
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
