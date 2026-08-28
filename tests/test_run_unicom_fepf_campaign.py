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

FAKE_CONTROLLER_AUTHORITIES = {
    "runtime_selector": lambda _receipts: "PASS_CURRENT",
    "source_publisher": lambda _root, name, _sources: {
        "path": f"/artifacts/{name}-sources.json", "sha256": "a" * 64, "bytes": 1
    },
}


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
        **FAKE_CONTROLLER_AUTHORITIES,
    )
    assert rc == 0
    assert [stage["name"] for stage in executor.stages] == [
        "cuda-canary",
        *[f"runtime-{index:02d}" for index in range(8)],
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
        **FAKE_CONTROLLER_AUTHORITIES,
    ) == 0
    names = [stage["name"] for stage in executor.stages]
    assert names.index("runtime-07") < names.index("exploratory-control-stage4")
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
        **FAKE_CONTROLLER_AUTHORITIES,
    ) == 19
    assert [stage["name"] for stage in executor.stages][-1] == "runtime-02"


def test_resume_skips_only_strictly_valid_terminal_stages() -> None:
    executor = FakeExecutor({"exploratory-epoch4-decision": "CLOSE_EPOCH4"})
    prior = {"cuda-canary": {"schema": "test-terminal-v1", "stage": "cuda-canary",
                              "decision": "PASS"}}
    assert MODULE.run_campaign(
        _config(), executor=executor, terminal_validator=_validate,
        through_stage="confirmation", prior_terminals=prior,
        **FAKE_CONTROLLER_AUTHORITIES,
    ) == 0
    assert executor.stages[0]["name"] == "runtime-00"
    broken = {"cuda-canary": {"stage": "wrong"}}
    with pytest.raises(AssertionError):
        MODULE.run_campaign(
            _config(), executor=FakeExecutor(), terminal_validator=_validate,
            prior_terminals=broken,
            **FAKE_CONTROLLER_AUTHORITIES,
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


def test_registered_commands_parse_through_real_task2_task4_task5_clis(tmp_path: Path) -> None:
    builder_spec = importlib.util.spec_from_file_location(
        "fepf_builder_for_cli_test", ROOT / "scripts/build_unicom_fepf_run_config.py"
    )
    assert builder_spec is not None and builder_spec.loader is not None
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    config = builder.build_run_config(
        repo=ROOT,
        checkout_root_template=str(tmp_path / "checkout-{config_commit}"),
        artifact_root=tmp_path / "artifacts",
        inference_structure={
            "schema": "unicom-fepf-structure-v1",
            "tensors": [{
                "name": "weight", "kind": "parameter", "shape": [1],
                "dtype": "torch.float32", "numel": 1, "element_size": 4, "bytes": 4,
            }],
            "classifier": {
                "shape": [1, 1], "dtype": "torch.float32", "numel": 1,
                "element_size": 4, "bytes": 4,
            },
            "operations": [
                "official_forward", "full768_l2", "prefix512", "squared_euclidean"
            ],
        },
        partition_inventory={
            "query_rows": 14_218, "gallery_rows": 12_612,
            "maximum_relevant_count": 64, "maximum_path_bytes": 120,
        },
        cuda_canary_authority={
            "device_uuid": "GPU-registered", "environment_sha256": "d" * 64,
        },
    )
    MODULE.validate_registered_command_vectors(config, checkout_root=ROOT)


def test_default_terminal_dispatch_uses_public_validators_and_runtime_comparator() -> None:
    assert hasattr(MODULE, "RegisteredTerminalValidator")
    assert hasattr(MODULE, "select_runtime_from_receipts")


def test_production_resume_and_storage_preflight_are_controller_owned(tmp_path: Path) -> None:
    assert hasattr(MODULE, "load_campaign_resume")
    assert hasattr(MODULE, "prepare_campaign_storage")


def test_selected_runtime_is_applied_to_training_and_quality_commands() -> None:
    executor = FakeExecutor({
        "exploratory-epoch4-decision": "PASS_TO_RESUME",
        "exploratory-decision": "PROMOTE",
    })
    authorities = dict(FAKE_CONTROLLER_AUTHORITIES)
    authorities["runtime_selector"] = lambda _receipts: "PASS_COMPOSED"
    assert MODULE.run_campaign(
        _config(), executor=executor, terminal_validator=_validate,
        **authorities,
    ) == 0
    train_commands = [
        stage["command"] for stage in executor.stages
        if "--classifier-init" in stage["command"]
    ]
    assert train_commands and all("--compile" in command and "--fused" in command
                                  for command in train_commands)
    profile_commands = [
        stage["command"] for stage in executor.stages if "-profile-" in stage["name"]
    ]
    assert profile_commands
    assert all(command[command.index("--runtime-mode") + 1] == "composed"
               for command in profile_commands)


def test_executor_terminates_child_when_polling_or_marker_raises(tmp_path: Path) -> None:
    process = FakeProcess()
    process.polls = [None]
    signals: list[tuple[int, int]] = []
    runner = MODULE.SubprocessStageExecutor(
        checkout_root=tmp_path,
        marker_writer=lambda _row: (_ for _ in ()).throw(RuntimeError("marker")),
        popen=lambda *_args, **_kwargs: process,
        killpg=lambda pid, sig: signals.append((pid, sig)),
        sleep=lambda _seconds: None,
    )
    with pytest.raises(RuntimeError, match="marker"):
        runner({"name": "stage", "command": ["python"]})
    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.waited


def test_review2_sources_cross_task5_canonical_loader(tmp_path: Path) -> None:
    authority = MODULE.publish_evaluation_sources(
        tmp_path, "epoch4", [{"registered": True}]
    )
    evaluator = MODULE._load_module(
        ROOT / "scripts/evaluate_unicom_fepf.py", "review2_evaluator_loader"
    )
    assert evaluator._strict_json_file(Path(authority["path"])) == [{"registered": True}]


def test_review2_resume_prevalidates_every_terminal_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Builder:
        @staticmethod
        def registered_stage_inventory(_config: object) -> tuple[str, ...]:
            return ("cuda-canary", "runtime-00", "runtime-01")

    monkeypatch.setattr(MODULE, "_load_module", lambda *_args: Builder)
    validated: list[str] = []
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.prevalidate_campaign_resume(
            _config(),
            {"cuda-canary": {"status": "PASS"}, "runtime-01": {"status": "PASS"}},
            terminal_validator=lambda stage, _terminal: validated.append(stage["name"]),
            checkout_root=ROOT,
        )
    assert validated == []


def test_review2_runtime_terminal_rehashes_external_authorities(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    receipt = tmp_path / "run-receipt.json"
    config = tmp_path / "config.json"
    for path in (checkpoint, receipt, config):
        path.write_bytes(path.name.encode())
    expected_environment = {"device_name": "registered"}
    calls: list[dict[str, object]] = []

    class Profiler:
        @staticmethod
        def validate_runtime_profile(terminal: object, **authorities: object) -> None:
            assert terminal == {"schema": "runtime"}
            calls.append(authorities)

    MODULE.validate_runtime_terminal(
        {
            "command": [
                "python", "profile.py", "--runtime-mode", "composed",
                "--run-checkpoint", str(checkpoint), "--run-receipt", str(receipt),
                "--config", str(config),
            ]
        },
        {"schema": "runtime"},
        profiler=Profiler,
        expected_environment=expected_environment,
    )
    assert calls == [{
        "expected_mode": "composed",
        "checkpoint": checkpoint,
        "run_receipt": receipt,
        "config": config,
        "expected_environment": expected_environment,
    }]


def test_review2_composed_runtime_disables_ema_and_joins_environment() -> None:
    command = MODULE.apply_runtime_selection(
        ["python", "trainer.py"], "PASS_COMPOSED", profile=False
    )
    assert command[-3:] == ["--compile", "--fused", "--no-ema"]
    environment = {"device_name": "registered"}
    MODULE.validate_profile_environment({"environment": environment}, environment)
    with pytest.raises(ValueError, match="environment"):
        MODULE.validate_profile_environment(
            {"environment": {"device_name": "substituted"}}, environment
        )


def test_review2_fresh_process_crosses_real_json_loaders(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "preflight").mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_bytes(MODULE._canonical_json({"registered": True}))
    MODULE.run_fresh_process_contract_preflight(
        checkout_root=ROOT, config_path=config, artifact_root=artifact_root
    )
    assert not (artifact_root / "preflight" / ".task6-loader-probe.json").exists()
