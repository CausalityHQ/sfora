from __future__ import annotations

import hashlib
import importlib.util
import json
import signal
import subprocess
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_unicom_fepf_campaign", ROOT / "scripts/run_unicom_fepf_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _committed_source_copy(tmp_path: Path, builder) -> Path:
    repo = tmp_path / "registered-source"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=repo,
        check=True,
    )
    for relative in builder.REGISTERED_SOURCE_PATHS:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "registered source"], cwd=repo, check=True)
    return repo


def test_review5_fresh_process_consumes_real_task2_task4_authorities(
    tmp_path: Path,
) -> None:
    result = MODULE.run_non_authentic_partial_cpu_preflight(
        source_root=ROOT, workspace=tmp_path, stop_before_cuda=True
    )
    assert all(run["returncode"] == 0 for run in result["public_runs"])
    environment_path = Path(result["pre_canary_absent_paths"][0])
    environment_path.write_bytes(environment_path.read_bytes() + b" ")
    train_run = next(
        run for run in result["public_runs"]
        if Path(run["argv"][3]).name == "train_unicom_inshop.py"
    )
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            train_run["argv"], cwd=Path(result["execution_checkout"]), check=True,
            capture_output=True,
        )


def test_review6_real_committed_cli_crossing_stops_before_cuda(tmp_path: Path) -> None:
    result = MODULE.run_non_authentic_partial_cpu_preflight(
        source_root=ROOT,
        workspace=tmp_path,
        stop_before_cuda=True,
    )
    assert result["normal_builder_returncode"] != 0
    assert Path(result["source_config_path"]).is_file()
    assert Path(result["execution_config_path"]).is_file()
    assert all(run["returncode"] == 0 for run in result["public_runs"])
    assert all(
        "--authority-preflight-only" in run["argv"]
        for run in result["public_runs"]
    )


def test_review7_committed_crossing_runs_public_mains_canary_first(tmp_path: Path) -> None:
    result = MODULE.run_non_authentic_partial_cpu_preflight(
        source_root=ROOT,
        workspace=tmp_path,
        stop_before_cuda=True,
    )
    executed_mains = [
        Path(run["argv"][3]).stem for run in result["public_runs"]
    ]
    assert executed_mains == [
        "run_unicom_fepf_campaign",
        "run_unicom_fepf_cuda_canary",
        "train_unicom_inshop",
        "profile_unicom_training_step",
        "evaluate_unicom_fepf",
        "run_unicom_fepf_campaign",
    ]
    assert len(result["pre_canary_absent_paths"]) == 1
    assert Path(result["pre_canary_absent_paths"][0]).is_file()


def test_review8_controller_main_first_launch_then_resume_uses_canary_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    environment_path = root / "preflight/cuda-environment.json"
    config_path = tmp_path / "config.json"
    config = {
        "artifact_root": str(root),
        "cuda_canary_environment": {"path": str(environment_path)},
        "publication_budget": {
            "schema": "unicom-fepf-publication-budget-v1", "publications": []
        },
        "publication_budget_path": "preflight/publication-budget.json",
        "publication_budget_sha256": hashlib.sha256(
            b'{\n  "schema": "unicom-fepf-publication-budget-v1",\n  "publications": []\n}\n'
        ).hexdigest(),
        "commands": {
            "runtime": [["python", "--environment-sha256", "{cuda_environment_sha256}"]],
            "train": ["python", "--environment-sha256", "{cuda_environment_sha256}"],
            "profile_quality": [
                "python", "--environment-sha256", "{cuda_environment_sha256}"
            ],
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    validations: list[tuple[str, bool]] = []
    launches: list[bool] = []

    class Builder:
        @staticmethod
        def validate_config_membership(_path: Path, _repo: Path) -> None:
            validations.append(("membership", root.exists()))

        @staticmethod
        def validate_first_launch_absence(_value: object) -> None:
            validations.append(("absence", root.exists()))

        @staticmethod
        def validate_exact_publication_budget(
            _config: object, _budget: object
        ) -> None:
            return None

        @staticmethod
        def validate_external_exact_publication_budget(
            _config: object, _budget: object
        ) -> None:
            validations.append(("external-budget", root.exists()))

    monkeypatch.setattr(MODULE, "_load_module", lambda *_args: Builder)
    monkeypatch.setattr(MODULE, "validate_registered_command_vectors", lambda *_a, **_k: None)
    monkeypatch.setattr(MODULE, "RegisteredTerminalValidator", lambda **_k: lambda *_a: None)
    monkeypatch.setattr(MODULE, "SubprocessStageExecutor", lambda **_k: object())
    monkeypatch.setattr(MODULE, "prevalidate_campaign_resume", lambda *_a, **_k: None)
    monkeypatch.setattr(MODULE, "require_campaign_remaining_capacity", lambda *_a: None)
    monkeypatch.setattr(MODULE, "run_fresh_process_contract_preflight", lambda **_k: None)
    monkeypatch.setattr(MODULE, "load_campaign_resume", lambda _config: {})

    def prepare(_config: object, *, physical_admission: bool = True) -> Path:
        assert physical_admission is True
        (root / "preflight").mkdir(parents=True, exist_ok=False)
        return root

    monkeypatch.setattr(MODULE, "prepare_campaign_storage", prepare)

    def first_run(value: dict[str, object], **_kwargs) -> int:
        launches.append(True)
        payload = b'{"device_uuid":"GPU-registered"}\n'
        environment_path.write_bytes(payload)
        resolver = MODULE.resolve_canary_environment_commands
        resolver(value, hashlib.sha256(payload).hexdigest())
        assert "{cuda_environment_sha256}" not in json.dumps(value["commands"])
        return 0

    monkeypatch.setattr(MODULE, "run_campaign", first_run)
    assert MODULE.main(["--config", str(config_path), "--through-stage", "runtime"]) == 0
    assert validations == [
        ("membership", False),
        ("external-budget", False),
        ("absence", False),
    ]
    assert launches == [True]

    validations.clear()
    def prepare_resume(_config: object, *, physical_admission: bool = True) -> Path:
        assert physical_admission is True
        return root

    monkeypatch.setattr(MODULE, "prepare_campaign_storage", prepare_resume)
    monkeypatch.setattr(MODULE, "run_campaign", lambda *_a, **_k: 0)
    assert MODULE.main(["--config", str(config_path), "--through-stage", "runtime"]) == 0
    assert validations == [("membership", True), ("external-budget", True)]


def test_review8_public_resume_rejects_coherent_canary_forgery_via_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    evidence = root / "preflight/canary-evidence"
    evidence.mkdir(parents=True)
    terminal = root / "preflight/cuda_canary_v1.json"
    manifest = evidence / "manifest.json"
    terminal.write_text('{"evidence_manifest_sha256":"1"}\n')
    manifest.write_text('{"objects":{}}\n')
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "artifact_root": str(root),
        "cuda_canary_environment": {
            "path": str(root / "preflight/cuda-environment.json")
        },
        "publication_budget": {
            "schema": "unicom-fepf-publication-budget-v1", "publications": []
        },
        "publication_budget_path": "preflight/publication-budget.json",
        "publication_budget_sha256": "0" * 64,
    }, indent=2) + "\n")
    launched = False

    def forbidden(**_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("resume launched child before validating canary")

    monkeypatch.setattr(MODULE, "SubprocessStageExecutor", forbidden)
    # The normal controller entrypoint must reject before launching any child;
    # a terminal-selected digest may not root the manifest being validated.
    assert MODULE.main([
        "--config", str(config_path), "--authority-preflight-only"
    ]) == 2
    assert launched is False


def test_review8_committed_crossing_executes_public_mains_not_help_or_private_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = MODULE.subprocess.run
    commands: list[list[str]] = []

    def record(command, *args, **kwargs):
        commands.append([str(item) for item in command])
        return original(command, *args, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", record)
    MODULE.run_non_authentic_partial_cpu_preflight(
        source_root=ROOT, workspace=tmp_path, stop_before_cuda=True
    )
    python_commands = [command for command in commands if "python" in Path(command[0]).name]
    assert python_commands
    assert all("--help" not in command and "-c" not in command for command in python_commands)
    executed = [Path(command[3]).name for command in python_commands if len(command) > 3]
    assert executed == [
        "build_unicom_fepf_run_config.py",
        "run_unicom_fepf_campaign.py",
        "run_unicom_fepf_cuda_canary.py",
        "train_unicom_inshop.py",
        "profile_unicom_training_step.py",
        "evaluate_unicom_fepf.py",
        "run_unicom_fepf_campaign.py",
    ]


def test_review9_committed_crossing_requires_zero_public_exit_and_observed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = MODULE.subprocess.run

    def fail_one(command, *args, **kwargs):
        if (
            len(command) > 3
            and Path(str(command[3])).name == "run_unicom_fepf_cuda_canary.py"
        ):
            return subprocess.CompletedProcess(command, 19, b"", b"failed")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", fail_one)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        MODULE.run_non_authentic_partial_cpu_preflight(
            source_root=ROOT, workspace=tmp_path, stop_before_cuda=True
        )
    assert raised.value.returncode == 19
    assert any(
        "run_unicom_fepf_cuda_canary.py" in str(item)
        for item in raised.value.cmd
    )


def test_review9_real_first_and_second_controller_invocations_observe_resume(
    tmp_path: Path,
) -> None:
    result = MODULE.run_non_authentic_partial_cpu_preflight(
        source_root=ROOT, workspace=tmp_path, stop_before_cuda=True
    )
    public_runs = result["public_runs"]
    assert Path(public_runs[0]["argv"][3]).name == "run_unicom_fepf_campaign.py"
    assert Path(public_runs[-1]["argv"][3]).name == "run_unicom_fepf_campaign.py"
    assert public_runs[0]["returncode"] == 0
    assert public_runs[-1]["returncode"] == 0
    assert all(Path(path).is_file() for path in result["observed_publications"])


def test_review10_non_authentic_partial_preflight_exposes_target_builder_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = MODULE.subprocess.run
    commands: list[list[str]] = []

    def record(command, *args, **kwargs):
        commands.append([str(item) for item in command])
        return original(command, *args, **kwargs)

    monkeypatch.setattr(MODULE.subprocess, "run", record)
    result = MODULE.run_non_authentic_partial_cpu_preflight(
        source_root=ROOT, workspace=tmp_path, stop_before_cuda=True
    )
    builder_commands = [
        command for command in commands
        if len(command) > 3
        and command[3] == "scripts/build_unicom_fepf_run_config.py"
    ]
    normal = [command for command in builder_commands if "--validate-handoff" not in command]
    assert len(normal) == 1
    assert normal[0][3:] == [
        "scripts/build_unicom_fepf_run_config.py",
        "--repo", result["source_checkout"],
        "--checkout-root-template", result["checkout_root_template"],
        "--artifact-root", result["artifact_root"],
        "--output", result["source_config_path"],
    ]
    assert Path(result["source_checkout"]).resolve() != Path(
        result["execution_checkout"]
    ).resolve()
    assert Path(result["execution_checkout"]).is_dir()
    assert Path(result["execution_config_path"]).is_file()
    assert subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=result["execution_checkout"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == result["config_commit"]
    assert subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=result["execution_checkout"],
        check=False, capture_output=True, text=True,
    ).returncode != 0
    assert result["normal_builder_returncode"] != 0
    assert all(run["returncode"] == 0 for run in result["public_runs"])
    assert all(Path(path).is_file() for path in result["observed_publications"])


def test_review10_controller_recomputes_exact_budget_on_first_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    config_path = tmp_path / "config.json"
    budget = {"schema": "unicom-fepf-publication-budget-v1", "publications": []}
    payload = (json.dumps(budget, indent=2) + "\n").encode()
    config = {
        "artifact_root": str(root),
        "artifact_budget_bytes": 1,
        "artifact_budget_inodes": 1,
        "cuda_canary_environment": {
            "path": str(root / "preflight/cuda-environment.json")
        },
        "publication_budget": budget,
        "publication_budget_path": "preflight/publication-budget.json",
        "publication_budget_sha256": hashlib.sha256(payload).hexdigest(),
        "commands": {"runtime": [], "train": [], "profile_quality": []},
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    recomputations: list[bool] = []

    class Builder:
        @staticmethod
        def validate_config_membership(_path: Path, _repo: Path) -> None:
            return None

        @staticmethod
        def validate_first_launch_absence(_value: object) -> None:
            return None

        @staticmethod
        def validate_exact_publication_budget(_config, _budget) -> None:
            recomputations.append(root.exists())

        @staticmethod
        def validate_external_exact_publication_budget(_config, _budget) -> None:
            recomputations.append(root.exists())

    monkeypatch.setattr(MODULE, "_load_module", lambda *_args: Builder)
    monkeypatch.setattr(MODULE, "validate_registered_command_vectors", lambda *_a, **_k: None)
    monkeypatch.setattr(MODULE, "prepare_campaign_storage", lambda *_a, **_k: (
        root.mkdir(parents=True, exist_ok=True) or root
    ))
    monkeypatch.setattr(MODULE, "RegisteredTerminalValidator", lambda **_k: lambda *_a: None)
    monkeypatch.setattr(MODULE, "load_campaign_resume", lambda _config: {})
    monkeypatch.setattr(MODULE, "prevalidate_campaign_resume", lambda *_a, **_k: None)
    monkeypatch.setattr(MODULE, "SubprocessStageExecutor", lambda **_k: object())
    monkeypatch.setattr(MODULE, "run_campaign", lambda *_a, **_k: 0)
    monkeypatch.setattr(MODULE, "require_campaign_remaining_capacity", lambda *_a: None)
    monkeypatch.setattr(MODULE, "run_fresh_process_contract_preflight", lambda **_k: None)
    assert MODULE.main(["--config", str(config_path), "--through-stage", "runtime"]) == 0
    assert MODULE.main(["--config", str(config_path), "--through-stage", "runtime"]) == 0
    assert recomputations == [False, True]


def test_review9_decision_source_prefix_is_adoptable_on_second_invocation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    authority = MODULE.publish_evaluation_sources(
        root, "exploratory-epoch4-decision", [{"registered": True}]
    )
    adopted = MODULE.publish_evaluation_sources(
        root, "exploratory-epoch4-decision", [{"registered": True}]
    )
    assert adopted == authority
    MODULE.validate_recoverable_publication_prefix(root)


def test_review6_controller_source_publisher_rechecks_owned_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_fsync = MODULE.os.fsync
    destination = tmp_path / "decision-sources.json"
    replaced = False

    def substitute_after_flush(descriptor: int) -> None:
        nonlocal replaced
        original_fsync(descriptor)
        if not replaced and destination.exists():
            payload = destination.read_bytes()
            destination.unlink()
            destination.write_bytes(payload)
            replaced = True

    monkeypatch.setattr(MODULE.os, "fsync", substitute_after_flush)
    with pytest.raises(RuntimeError, match="inode|ownership"):
        MODULE.publish_evaluation_sources(tmp_path, "decision", {"registered": True})

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


def test_review13_controller_supplies_canonical_canary_environment_to_children(
    tmp_path: Path,
) -> None:
    captured: list[dict[str, object]] = []

    class CompleteProcess:
        pid = 77

        @staticmethod
        def poll() -> int:
            return 0

    environment = {
        "deterministic_execution": {
            "cublas_workspace_config": ":4096:8",
        }
    }

    def popen(_command, **kwargs):
        captured.append(kwargs)
        return CompleteProcess()

    runner = MODULE.SubprocessStageExecutor(
        checkout_root=tmp_path,
        marker_writer=lambda _row: None,
        popen=popen,
        registered_environment=environment,
    )
    assert runner({"name": "runtime-00", "command": ["python", "profile.py"]})[
        "exit_code"
    ] == 0
    assert captured[0]["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_review13_controller_fresh_child_runs_structural_validation_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    evidence = root / "preflight/canary-evidence"
    evidence.mkdir(parents=True)
    manifest = evidence / "manifest.json"
    manifest.write_text('{"schema":"unicom-fepf-canary-evidence-v1"}\n')
    environment = {"device_uuid": "GPU-registered"}
    terminal = {
        "evidence_manifest_sha256": MODULE._sha256(manifest.read_bytes()),
        "environment": environment,
    }
    reconstructed: list[Path] = []

    class Canary:
        validate_cuda_canary_receipt = staticmethod(lambda *_a, **_k: None)
        validate_canary_evidence_manifest = staticmethod(lambda *_a, **_k: None)
        @staticmethod
        def reconstruct_canary_authority(_config, path, **_kwargs):
            reconstructed.append(path)
            return {}

        @staticmethod
        def validate_registered_canary_family(*_args, **_kwargs):
            raise AssertionError("fresh child was already fitted-validated")

    class Other:
        pass

    monkeypatch.setattr(
        MODULE,
        "_load_module",
        lambda path, _name: Canary if path.name == "run_unicom_fepf_cuda_canary.py" else Other,
    )
    validator = MODULE.RegisteredTerminalValidator(
        checkout_root=ROOT,
        config={
            "artifact_root": str(root),
            "cuda_canary_authority": {
                "device_uuid": "GPU-registered",
                "environment_sha256": "1" * 64,
            },
        },
    )
    stage = {"name": "cuda-canary", "command": [], "fresh_execution": True}
    validator(stage, terminal)
    validator(stage, terminal)
    assert reconstructed == [manifest]


def test_review15_controller_resume_deep_validates_in_registered_canary_child_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    evidence = root / "preflight/canary-evidence"
    evidence.mkdir(parents=True)
    manifest = evidence / "manifest.json"
    manifest.write_text('{"schema":"unicom-fepf-canary-evidence-v1"}\n')
    environment = {
        "device_uuid": "GPU-registered",
        "deterministic_execution": {"cublas_workspace_config": ":4096:8"},
    }
    terminal = {
        "evidence_manifest_sha256": MODULE._sha256(manifest.read_bytes()),
        "environment": environment,
    }
    command = ["python", "scripts/run_unicom_fepf_cuda_canary.py", "--config", "run.json"]
    config = {
        "artifact_root": str(root),
        "cuda_canary_receipt": "preflight/cuda_canary_v1.json",
        "cuda_canary_command": command,
        "cuda_canary_authority": {
            "device_uuid": "GPU-registered", "environment_sha256": "1" * 64,
        },
    }
    child_calls: list[tuple[list[str], dict[str, object]]] = []

    class Canary:
        validate_cuda_canary_receipt = staticmethod(lambda *_a, **_k: None)
        validate_canary_evidence_manifest = staticmethod(lambda *_a, **_k: None)
        reconstruct_canary_authority = staticmethod(lambda *_a, **_k: {})

        @staticmethod
        def validate_registered_canary_family(*_args, **_kwargs):
            raise AssertionError("controller must not import/run fitted CUDA validation")

    class Builder:
        @staticmethod
        def registered_stage_inventory(_config):
            return ("cuda-canary",)

    class Other:
        pass

    def load(path: Path, _name: str):
        if path.name == "run_unicom_fepf_cuda_canary.py":
            return Canary
        if path.name == "build_unicom_fepf_run_config.py":
            return Builder
        return Other

    def run_child(argv: list[str], **kwargs: object):
        child_calls.append((argv, kwargs))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(MODULE, "_load_module", load)
    monkeypatch.setattr(MODULE.subprocess, "run", run_child)
    validator = MODULE.RegisteredTerminalValidator(checkout_root=ROOT, config=config)
    prior = {"cuda-canary": terminal}
    MODULE.prevalidate_campaign_resume(
        config, prior, terminal_validator=validator, checkout_root=ROOT
    )
    code, resumed = MODULE._execute(
        MODULE._resume_stage(config, "cuda-canary"),
        executor=lambda _stage: (_ for _ in ()).throw(AssertionError("no rerun")),
        terminal_validator=validator,
        prior_terminals=prior,
        marker_writer=lambda _row: None,
    )
    assert (code, resumed) == (0, terminal)
    assert len(child_calls) == 1
    argv, kwargs = child_calls[0]
    assert argv == [
        *command, "--publication-stage", "cuda-canary",
        "--campaign-root", str(root),
    ]
    assert kwargs["cwd"] == ROOT
    assert kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


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
    source_repo = _committed_source_copy(tmp_path, builder)
    config = builder.build_run_config(
        repo=source_repo,
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
        cuda_canary_environment={
            "path": str((tmp_path / "artifacts/preflight/cuda-environment.json").resolve()),
            "sha256": "d" * 64,
            "bytes": 1024,
        },
        publication_budget={
            "path": str((tmp_path / "artifacts/preflight/publication-budget.json").resolve()),
            "sha256": "e" * 64,
            "bytes": 2048,
        },
        runtime_inference_signature={
            "schema": "unicom-inference-signature-v1",
            "tensors": [
                {
                    "name": "weight", "kind": "parameter", "shape": [1],
                    "dtype": "torch.float32", "numel": 1, "element_size": 4,
                    "bytes": 4, "sha256": "3" * 64,
                }
            ],
            "total_bytes": 4,
            "aggregate_sha256": "4" * 64,
            "descriptor_dtype": "torch.float32",
            "descriptor_dimension": 512,
            "descriptor_sha256": "5" * 64,
            "operations": [
                "official_forward", "full768_l2", "prefix512", "squared_euclidean"
            ],
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
    test_review5_fresh_process_consumes_real_task2_task4_authorities(tmp_path)


def test_review11_authentic_main_requires_external_budget_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifacts"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "artifact_root": str(root),
        "publication_budget": {
            "schema": "unicom-fepf-publication-budget-v1", "publications": []
        },
        "commands": {},
    }) + "\n")
    calls: list[str] = []

    class Builder:
        @staticmethod
        def validate_config_membership(_path: Path, _repo: Path) -> None:
            calls.append("membership")

        @staticmethod
        def validate_non_authentic_synthesized_handoff(
            _path: Path, _repo: Path
        ) -> None:
            calls.append("non-auth")

        @staticmethod
        def validate_exact_publication_budget(
            _config: object, _budget: object
        ) -> None:
            calls.append("internal-exact")

    monkeypatch.setattr(MODULE, "_load_module", lambda *_args: Builder)
    monkeypatch.setattr(
        MODULE, "validate_registered_command_vectors", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        MODULE, "prepare_campaign_storage",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("storage reached")),
    )
    assert MODULE.main(["--config", str(config_path)]) == 2
    assert calls == ["membership"]

    calls.clear()
    assert MODULE.main([
        "--config", str(config_path), "--authority-preflight-only",
        "--non-authentic-synthesized-authorities",
    ]) == 2
    assert calls == ["non-auth", "internal-exact"]


def test_review12_post_canary_guard_derives_environment_and_runs_public_preflights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    root = tmp_path / "campaign"
    (root / "preflight").mkdir(parents=True)
    environment = {"device_uuid": "GPU-registered", "deterministic_execution": {}}
    environment_path = root / "preflight/cuda-environment.json"
    environment_payload = MODULE._canonical_json(environment)
    environment_path.write_bytes(environment_payload)
    config_path = checkout / "config.json"
    config_path.write_bytes(MODULE._canonical_json({
        "cuda_canary_environment": {"path": str(environment_path.resolve())},
        "publication_budget_path": "preflight/publication-budget.json",
        "publication_budget_sha256": "a" * 64,
        "commands": {
            "runtime": [[
                "python", "-I", "-B", "scripts/profile_unicom_training_step.py",
                "--config", str(config_path),
                "--environment-authority", str(environment_path),
                "--environment-sha256", "{cuda_environment_sha256}",
                "--output", "{output}",
            ]]
        },
    }))
    commands: list[list[str]] = []

    def capture(command, **_kwargs):
        commands.append([str(item) for item in command])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(MODULE.subprocess, "run", capture)
    MODULE.run_fresh_process_contract_preflight(
        checkout_root=checkout, config_path=config_path, artifact_root=root
    )
    digest = hashlib.sha256(environment_payload).hexdigest()
    mains = [Path(command[3]).name for command in commands]
    assert mains == [
        "train_unicom_inshop.py", "profile_unicom_training_step.py",
        "evaluate_unicom_fepf.py",
    ]
    for command in commands[:2]:
        index = command.index("--environment-sha256")
        assert command[index + 1] == digest


def test_review12_controller_status_is_not_unbudgeted_campaign_evidence(
    tmp_path: Path,
) -> None:
    result = MODULE.run_non_authentic_partial_cpu_preflight(
        source_root=ROOT, workspace=tmp_path, stop_before_cuda=True
    )
    root = Path(result["artifact_root"])
    config = json.loads(Path(result["execution_config_path"]).read_bytes())
    rows = config["publication_budget"]["publications"]
    status = root / "controller-status.json"
    MODULE.write_status_marker_atomic(status, {"state": "running"})
    assert any(
        row["path"] == "controller-status.json" for row in rows
    )
