from __future__ import annotations

import importlib.util
import signal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_unicom_fepf_campaign", ROOT / "scripts/run_unicom_fepf_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeExecutor:
    def __init__(self, decisions: dict[str, str] | None = None, failure: str | None = None):
        self.decisions = decisions or {}
        self.failure = failure
        self.stages: list[dict[str, object]] = []

    def __call__(self, stage: dict[str, object]) -> dict[str, object]:
        self.stages.append(stage)
        if stage["name"] == self.failure:
            return {"exit_code": 19, "terminal": None}
        return {
            "exit_code": 0,
            "terminal": {
                "schema": "test-terminal-v1",
                "stage": stage["name"],
                "decision": self.decisions.get(stage["name"], "PASS"),
            },
        }


def _config() -> dict[str, object]:
    return {
        "schema": "unicom-fepf-run-config-v1",
        "source_commit": "a" * 40,
        "checkout_root": "/checkout",
        "artifact_root": "/artifacts",
        "runtime_order": [
            "current", "composed", "composed", "current",
            "current", "composed", "composed", "current",
        ],
        "exploratory": {
            "training_seed": 0, "holdout_seed": 0,
            "arms": ["imprinted", "fepf_mean", "fepf_random"],
        },
        "confirmation_pairs": [
            [7, 20_260_828], [8, 271_828], [9, 314_159],
            [10, 1_618_033], [11, 57_721],
        ],
        "commands": {
            "cuda_canary": ["python", "canary.py"],
            "runtime": [["python", "profile.py", mode] for mode in (
                "current", "composed", "composed", "current",
                "current", "composed", "composed", "current",
            )],
            "train": ["python", "train.py"],
            "profile_quality": ["python", "profile.py", "quality"],
            "evaluate": ["python", "evaluate.py"],
        },
    }


def _validate(stage: dict[str, object], terminal: object) -> None:
    assert type(terminal) is dict
    assert terminal["stage"] == stage["name"]


def test_controller_stops_before_epoch16_when_epoch4_fails() -> None:
    executor = FakeExecutor({"exploratory-epoch4-decision": "CLOSE_EPOCH4"})
    rc = MODULE.run_campaign(
        _config(), executor=executor, terminal_validator=_validate,
        through_stage="confirmation",
    )
    assert rc == 0
    assert [stage["name"] for stage in executor.stages] == [
        "cuda-canary",
        *[f"runtime-{index:02d}" for index in range(8)],
        "runtime-decision",
        "exploratory-control-stage4",
        "exploratory-candidate-stage4",
        "exploratory-epoch4-decision",
    ]


def test_controller_orders_profiles_and_runs_all_five_confirmation_pairs() -> None:
    executor = FakeExecutor({
        "exploratory-epoch4-decision": "PASS_TO_RESUME",
        "exploratory-decision": "PROMOTE",
    })
    markers: list[dict[str, object]] = []
    assert MODULE.run_campaign(
        _config(), executor=executor, terminal_validator=_validate,
        through_stage="confirmation", marker_writer=markers.append,
    ) == 0
    names = [stage["name"] for stage in executor.stages]
    assert names.index("runtime-decision") < names.index("exploratory-control-stage4")
    assert names.index("exploratory-random-stage16") > names.index("exploratory-decision")
    for pair_index in range(5):
        prefix = f"confirmation-{pair_index}"
        selected = [name for name in names if name.startswith(prefix)]
        assert selected == [
            f"{prefix}-control",
            f"{prefix}-candidate",
            f"{prefix}-profile-control-0",
            f"{prefix}-profile-candidate-0",
            f"{prefix}-profile-candidate-1",
            f"{prefix}-profile-control-1",
        ]
    assert names[-1] == "confirmation-decision"
    assert markers and markers[-1]["state"] == "complete"


def test_structural_or_process_failure_stops_and_preserves_exact_exit() -> None:
    executor = FakeExecutor(failure="runtime-02")
    assert MODULE.run_campaign(
        _config(), executor=executor, terminal_validator=_validate,
        through_stage="confirmation",
    ) == 19
    assert [stage["name"] for stage in executor.stages][-1] == "runtime-02"


def test_resume_skips_only_strictly_valid_terminal_stages() -> None:
    executor = FakeExecutor({"exploratory-epoch4-decision": "CLOSE_EPOCH4"})
    prior = {"cuda-canary": {"schema": "test-terminal-v1", "stage": "cuda-canary",
                              "decision": "PASS"}}
    assert MODULE.run_campaign(
        _config(), executor=executor, terminal_validator=_validate,
        through_stage="confirmation", prior_terminals=prior,
    ) == 0
    assert executor.stages[0]["name"] == "runtime-00"
    broken = {"cuda-canary": {"stage": "wrong"}}
    with pytest.raises(AssertionError):
        MODULE.run_campaign(
            _config(), executor=FakeExecutor(), terminal_validator=_validate,
            prior_terminals=broken,
        )


class FakeProcess:
    pid = 4312

    def __init__(self) -> None:
        self.polls = [None, None, 7]
        self.waited = False

    def poll(self):
        return self.polls.pop(0)

    def wait(self):
        self.waited = True
        return 7


def test_subprocess_runner_retains_original_pid_and_heartbeats(tmp_path: Path) -> None:
    process = FakeProcess()
    calls: list[tuple[object, ...]] = []
    markers: list[dict[str, object]] = []

    def popen(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return process

    runner = MODULE.SubprocessStageExecutor(
        checkout_root=tmp_path, marker_writer=markers.append,
        popen=popen, sleep=lambda _seconds: None,
        monotonic=iter((1.0, 2.0, 3.0, 4.0)).__next__,
    )
    result = runner({"name": "stage", "command": ["python", "work.py"]})
    assert result["exit_code"] == 7
    assert len(calls) == 1
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["start_new_session"] is True
    assert {row["pid"] for row in markers} == {4312}


def test_subprocess_runner_cancels_original_process_group(tmp_path: Path) -> None:
    process = FakeProcess()
    process.polls = [None]
    signals: list[tuple[int, int]] = []
    runner = MODULE.SubprocessStageExecutor(
        checkout_root=tmp_path, marker_writer=lambda _row: None,
        popen=lambda *_args, **_kwargs: process,
        sleep=lambda _seconds: None, monotonic=lambda: 1.0,
        cancelled=lambda: True,
        killpg=lambda pid, sig: signals.append((pid, sig)),
    )
    result = runner({"name": "stage", "command": ["python"]})
    assert signals == [(4312, signal.SIGTERM)]
    assert process.waited is True
    assert result["exit_code"] == 7

