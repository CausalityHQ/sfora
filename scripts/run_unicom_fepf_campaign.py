#!/usr/bin/env python3
"""Run the authenticated UniCOM FEPF campaign one retained process at a time."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

RUNTIME_ORDER = ("current", "composed", "composed", "current") * 2
CONFIRMATION_PAIRS = (
    (7, 20_260_828), (8, 271_828), (9, 314_159),
    (10, 1_618_033), (11, 57_721),
)
QUALITY_PROFILE_ORDER = ("control", "candidate", "candidate", "control")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_registered_command_vectors(config: object, *, checkout_root: Path) -> None:
    value = _required_config(config)
    commands = value["commands"]
    trainer = _load_module(checkout_root / "scripts/train_unicom_inshop.py", "fepf_trainer_cli")
    profiler = _load_module(
        checkout_root / "scripts/profile_unicom_training_step.py", "fepf_profiler_cli"
    )
    evaluator = _load_module(
        checkout_root / "scripts/evaluate_unicom_fepf.py", "fepf_evaluator_cli"
    )
    runtime = ["/tmp/out.json" if item == "{output}" else item for item in commands["runtime"][0]]
    profiler.parse_args(runtime[4:])
    trainer.parse_args(
        _train_command(list(commands["train"]), mode="imprinted", training_seed=0,
                       holdout_seed=0, stop=4, output=Path("/tmp/run"))[4:]
    )
    evaluator.parse_args([
        "--phase", "epoch4", "--sources", "/tmp/sources.json",
        "--sources-sha256", "a" * 64, "--sources-bytes", "1",
        "--evidence-root", "/tmp/evidence", "--output", "/tmp/result.json",
        "--temporary", "/tmp/result.tmp",
    ])
    quality = [
        *commands["profile_quality"], "--runtime-mode", "current",
        "--run-checkpoint", "/tmp/checkpoint.pt", "--run-receipt",
        "/tmp/run-receipt.json", "--output", "/tmp/profile.json",
        "--config", "/tmp/config.json",
    ]
    profiler.parse_args(quality[4:])
    for phase in ("epoch4", "exploratory", "confirmation"):
        parsed = evaluator.parse_args([
            "--phase", phase, "--sources", "/tmp/sources.json",
            "--sources-sha256", "a" * 64, "--sources-bytes", "1",
            "--evidence-root", "/tmp/evidence", "--output", "/tmp/result.json",
            "--temporary", "/tmp/result.tmp",
        ])
        if parsed.phase != phase:
            raise ValueError("registered evaluator phase differs")


def select_runtime_from_receipts(receipts: object, *, checkout_root: Path) -> str:
    profiler = _load_module(
        checkout_root / "scripts/profile_unicom_training_step.py", "fepf_profiler_decision"
    )
    decision = profiler.compare_runtime_smoke(tuple(receipts))
    if decision not in {"PASS_CURRENT", "PASS_COMPOSED"}:
        raise ValueError("runtime smoke is structurally invalid")
    return decision


def apply_runtime_selection(command: list[str], decision: str, *, profile: bool) -> list[str]:
    if decision not in {"PASS_CURRENT", "PASS_COMPOSED"}:
        raise ValueError("runtime selection differs")
    result = list(command)
    if profile:
        index = result.index("--runtime-mode") + 1
        result[index] = "composed" if decision == "PASS_COMPOSED" else "current"
    elif decision == "PASS_COMPOSED":
        result.extend(("--compile", "--fused", "--no-ema"))
    return result


def validate_profile_environment(terminal: object, expected: object) -> None:
    if (
        type(terminal) is not dict
        or type(expected) is not dict
        or terminal.get("environment") != expected
    ):
        raise ValueError("profile environment differs from CUDA canary authority")


def _command_argument(command: object, option: str) -> str:
    if type(command) is not list or option not in command:
        raise ValueError("runtime command authority differs")
    index = command.index(option)
    if index + 1 >= len(command) or type(command[index + 1]) is not str:
        raise ValueError("runtime command authority differs")
    return command[index + 1]


def validate_runtime_terminal(
    stage: dict[str, object], terminal: object, *, profiler: object,
    expected_environment: object,
) -> None:
    command = stage["command"]
    validator = getattr(profiler, "validate_runtime_profile", None)
    if not callable(validator):
        raise ValueError("public runtime validator differs")
    validator(
        terminal,
        expected_mode=_command_argument(command, "--runtime-mode"),
        checkpoint=Path(_command_argument(command, "--run-checkpoint")),
        run_receipt=Path(_command_argument(command, "--run-receipt")),
        config=Path(_command_argument(command, "--config")),
        expected_environment=expected_environment,
    )


class RegisteredTerminalValidator:
    def __init__(self, *, checkout_root: Path, config: dict[str, object]) -> None:
        self.checkout_root = checkout_root
        self.config = config
        self.trainer = _load_module(checkout_root / "scripts/train_unicom_inshop.py", "fepf_tv")
        self.profiler = _load_module(
            checkout_root / "scripts/profile_unicom_training_step.py", "fepf_pv"
        )
        self.evaluator = _load_module(
            checkout_root / "scripts/evaluate_unicom_fepf.py", "fepf_ev"
        )
        self.canary = _load_module(
            checkout_root / "scripts/run_unicom_fepf_cuda_canary.py", "fepf_cv"
        )
        self.profile_environment: dict[str, object] | None = None

    def __call__(self, stage: dict[str, object], terminal: object) -> None:
        name = str(stage["name"])
        if name == "cuda-canary":
            authority = self.config["cuda_canary_authority"]
            self.canary.validate_cuda_canary_receipt(
                terminal,
                self.config,
                expected_device_uuid=authority["device_uuid"],
                expected_environment_sha256=authority["environment_sha256"],
            )
            environment = terminal.get("environment") if type(terminal) is dict else None
            profile = environment.get("profile") if type(environment) is dict else None
            if type(profile) is not dict:
                raise ValueError("canary profile environment authority differs")
            self.profile_environment = profile
        elif "profile" in name or name.startswith("runtime-"):
            if self.profile_environment is None:
                raise ValueError("canary environment must precede profiling")
            if "profile" in name:
                self.profiler.validate_quality_profile(terminal)
                validate_profile_environment(terminal, self.profile_environment)
            else:
                validate_runtime_terminal(
                    stage, terminal, profiler=self.profiler,
                    expected_environment=self.profile_environment,
                )
        elif "decision" in name:
            self.evaluator.validate_fepf_result(
                terminal,
                Path(stage["evidence_root"]),
                sources_authority=stage["sources_authority"],
            )
        else:
            self.trainer.validate_training_run_receipt_v2(
                terminal, evidence_root=Path(stage["destination"])
            )


def prepare_campaign_storage(config: dict[str, object]) -> Path:
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"), "fepf_builder_storage"
    )
    root = Path(config["artifact_root"])
    if not os.path.lexists(root):
        builder.prepare_artifact_root(
            root, required_bytes=config["artifact_budget_bytes"],
            required_inodes=config["artifact_budget_inodes"],
        )
    else:
        builder.require_remaining_capacity(
            root, total_budget_bytes=config["artifact_budget_bytes"],
            total_budget_inodes=config["artifact_budget_inodes"],
            consumed_bytes=0, consumed_inodes=0,
        )
    preflight = root / "preflight"
    if not os.path.lexists(preflight):
        preflight.mkdir(mode=0o700)
    return root


def require_campaign_remaining_capacity(config: dict[str, object], root: Path) -> None:
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"), "fepf_builder_capacity"
    )
    consumed_bytes = 0
    consumed_inodes = 1
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("campaign artifact symlink differs")
        consumed_inodes += 1
        if path.is_file():
            consumed_bytes += path.stat().st_size
    builder.require_remaining_capacity(
        root, total_budget_bytes=config["artifact_budget_bytes"],
        total_budget_inodes=config["artifact_budget_inodes"],
        consumed_bytes=consumed_bytes, consumed_inodes=consumed_inodes,
    )


def load_campaign_resume(config: dict[str, object]) -> dict[str, object]:
    root = Path(config["artifact_root"])
    if not root.exists():
        return {}
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"),
        "fepf_builder_resume_validation",
    )
    builder.validate_campaign_resume(
        config, root, terminal_validator=lambda path: json.loads(path.read_bytes())
    )
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"), "fepf_builder_resume"
    )
    inventory = set(builder.registered_stage_inventory(config))
    result: dict[str, object] = {}
    canary = root / config["cuda_canary_receipt"]
    if os.path.lexists(canary):
        if canary.is_symlink() or not canary.is_file():
            raise ValueError("campaign resume canary differs")
        result["cuda-canary"] = json.loads(canary.read_bytes())
    for child in root.iterdir():
        if child.name in {"preflight", "controller-status.json"}:
            continue
        if child.name.endswith("-sources.json") or child.name.endswith("-result.json"):
            continue
        if child.name not in inventory or child.is_symlink() or not child.is_dir():
            raise ValueError("campaign resume stage differs")
        candidates = tuple(
            path for path in (
                child / "terminal.json", child / "run-receipt.json"
            ) if path.is_file() and not path.is_symlink()
        )
        if len(candidates) != 1:
            raise ValueError("campaign resume terminal differs")
        result[child.name] = json.loads(candidates[0].read_bytes())
    for name in inventory:
        result_path = root / f"{name}-result.json"
        if result_path.is_file() and not result_path.is_symlink():
            result[name] = json.loads(result_path.read_bytes())
    return result


def _resume_stage(config: dict[str, object], name: str) -> dict[str, object]:
    root = Path(config["artifact_root"])
    if name == "cuda-canary":
        return _stage(
            name, list(config["cuda_canary_command"]), root,
            terminal_path=root / config["cuda_canary_receipt"],
        )
    if name.startswith("runtime-"):
        index = int(name.rsplit("-", 1)[1])
        output = root / name / "terminal.json"
        command = [
            str(output) if item == "{output}" else item
            for item in config["commands"]["runtime"][index]
        ]
        return _stage(name, command, root)
    if name.endswith("-decision"):
        sources = root / f"{name}-sources.json"
        payload = sources.read_bytes()
        stage = _stage(
            name, [], root, terminal_path=root / f"{name}-result.json"
        )
        stage["sources_authority"] = {
            "path": str(sources.resolve()), "sha256": _sha256(payload),
            "bytes": len(payload),
        }
        stage["evidence_root"] = root
        return stage
    terminal = (
        root / name / "terminal.json"
        if "profile" in name
        else root / name / "run-receipt.json"
    )
    return _stage(name, [], root, terminal_path=terminal)


def prevalidate_campaign_resume(
    config: dict[str, object], prior: Mapping[str, object],
    *, terminal_validator: Callable[[dict[str, object], object], None],
    checkout_root: Path,
) -> None:
    builder = _load_module(
        Path(__file__).with_name("build_unicom_fepf_run_config.py"),
        "fepf_builder_resume_order",
    )
    order = tuple(builder.registered_stage_inventory(config))
    unknown = set(prior) - set(order)
    if unknown:
        raise ValueError("campaign resume stage differs")
    indices = [index for index, name in enumerate(order) if name in prior]
    if indices and set(indices) != set(range(max(indices) + 1)):
        raise ValueError("campaign resume chain is incomplete")
    for name in order:
        if name in prior:
            terminal_validator(_resume_stage(config, name), prior[name])
    runtime_names = tuple(f"runtime-{index:02d}" for index in range(8))
    resumed_runtime = tuple(prior[name] for name in runtime_names if name in prior)
    if resumed_runtime and len(resumed_runtime) != 8:
        raise ValueError("runtime resume chain is incomplete")
    if resumed_runtime:
        select_runtime_from_receipts(resumed_runtime, checkout_root=checkout_root)


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish_evaluation_sources(root: Path, name: str, sources: object) -> dict[str, object]:
    path = root / f"{name}-sources.json"
    payload = _canonical_json(sources)
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(path)
    else:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    return {"path": str(path.resolve()), "sha256": _sha256(payload), "bytes": len(payload)}


def run_fresh_process_contract_preflight(
    *, checkout_root: Path, config_path: Path, artifact_root: Path
) -> None:
    probe = artifact_root / "preflight" / ".task6-loader-probe.json"
    payload = _canonical_json([{"registered": True}])
    with probe.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    program = """
import importlib.util
import sys
from pathlib import Path
root, config, probe = map(Path, sys.argv[1:])
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
profile = load(root / 'scripts/profile_unicom_training_step.py', 'task6_profile_probe')
evaluate = load(root / 'scripts/evaluate_unicom_fepf.py', 'task6_evaluate_probe')
profile._strict_json_object(config)
evaluate._strict_json_file(probe)
"""
    try:
        subprocess.run(
            [
                sys.executable, "-I", "-B", "-c", program,
                str(checkout_root), str(config_path), str(probe),
            ],
            cwd=checkout_root, check=True,
        )
    finally:
        if probe.is_file() and not probe.is_symlink():
            probe.unlink()


def _evaluation_stage(
    *, name: str, phase: str, base: list[str], root: Path, sources: object,
    source_publisher: Callable[[Path, str, object], dict[str, object]],
) -> dict[str, object]:
    authority = source_publisher(root, name, sources)
    output = root / f"{name}-result.json"
    temporary = root / f".{name}-result.json.tmp"
    command = [
        *base, "--phase", phase, "--sources", authority["path"],
        "--sources-sha256", authority["sha256"], "--sources-bytes",
        str(authority["bytes"]), "--evidence-root", str(root),
        "--output", str(output), "--temporary", str(temporary),
    ]
    stage = _stage(name, command, root, terminal_path=output)
    stage["sources_authority"] = authority
    stage["evidence_root"] = root
    return stage


def write_status_marker_atomic(path: Path, value: Mapping[str, object]) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("campaign status marker path differs")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if os.path.lexists(temporary):
        raise FileExistsError(temporary)
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json(dict(value)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


class SubprocessStageExecutor:
    """Own exactly one original Popen until it reaches a terminal state."""

    def __init__(
        self,
        *,
        checkout_root: Path,
        marker_writer: Callable[[dict[str, object]], None],
        popen: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        cancelled: Callable[[], bool] = lambda: False,
        killpg: Callable[[int, int], None] = os.killpg,
        poll_seconds: float = 5.0,
    ) -> None:
        if poll_seconds <= 0 or poll_seconds > 55:
            raise ValueError("campaign polling interval differs")
        self.checkout_root = checkout_root
        self.marker_writer = marker_writer
        self.popen = popen
        self.sleep = sleep
        self.monotonic = monotonic
        self.cancelled = cancelled
        self.killpg = killpg
        self.poll_seconds = poll_seconds

    def __call__(self, stage: dict[str, object]) -> dict[str, object]:
        command = stage["command"]
        if (
            type(command) is not list
            or not command
            or not all(type(item) is str for item in command)
        ):
            raise ValueError("campaign command differs")
        started = self.monotonic()
        process = self.popen(
            command, cwd=self.checkout_root, start_new_session=True
        )
        status: int | None = None
        try:
            while True:
                status = process.poll()
                elapsed = self.monotonic() - started
                self.marker_writer({
                    "state": "running", "stage": stage["name"], "pid": process.pid,
                    "elapsed_seconds": elapsed,
                    "last_child_progress": stage.get("progress_path"),
                })
                if status is not None:
                    break
                if self.cancelled():
                    self.killpg(process.pid, signal.SIGTERM)
                    status = process.wait()
                    break
                self.sleep(self.poll_seconds)
        except BaseException:
            if status is None:
                self.killpg(process.pid, signal.SIGTERM)
                process.wait()
            raise
        terminal: object = None
        terminal_path = stage.get("terminal_path")
        if status == 0 and isinstance(terminal_path, Path):
            raw = terminal_path.read_bytes()
            terminal = json.loads(raw)
        return {"exit_code": status, "terminal": terminal}


def _stage(
    name: str, command: list[str], root: Path, *, terminal_path: Path | None = None
) -> dict[str, object]:
    destination = root / name
    return {
        "name": name,
        "command": command,
        "destination": destination,
        "terminal_path": terminal_path or destination / "terminal.json",
        "progress_path": str(destination / "progress.json"),
    }


def _train_command(
    base: list[str], *, mode: str, training_seed: int, holdout_seed: int,
    stop: int, output: Path, resume: Path | None = None,
) -> list[str]:
    command = [
        *base, "--classifier-init", mode, "--seed", str(training_seed),
        "--holdout-seed", str(holdout_seed), "--holdout-fraction", "0.2",
        "--epochs", "16", "--stop-after-epoch", str(stop),
        "--output-dir", str(output), "--run-receipt", str(output / "run-receipt.json"),
    ]
    if resume is not None:
        command.extend([
            "--resume", str(resume / "epoch-0004.pt"),
            "--parent-run-receipt", str(resume / "run-receipt.json"),
            "--parent-initialization-receipt", str(resume / "initialization-receipt.json"),
        ])
    return command


def _profile_stages(
    *, prefix: str, root: Path, base: list[str], control: Path, candidate: Path,
    config_path: Path, runtime_decision: str,
) -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    for index, arm in enumerate(QUALITY_PROFILE_ORDER):
        arm_root = control if arm == "control" else candidate
        name = f"{prefix}-profile-{arm}-{0 if index < 2 else 1}"
        command = apply_runtime_selection([
            *base, "--runtime-mode", "current", "--run-checkpoint",
            str(arm_root / "epoch-0016.pt"), "--run-receipt",
            str(arm_root / "run-receipt.json"), "--output", str(root / name / "terminal.json"),
            "--config", str(config_path),
        ], runtime_decision, profile=True)
        stages.append(_stage(name, command, root))
    return stages


def _required_config(config: object) -> dict[str, object]:
    if (
        type(config) is not dict
        or config.get("schema") != "unicom-fepf-run-config-v1"
        or config.get("runtime_order") != list(RUNTIME_ORDER)
        or config.get("confirmation_pairs") != [list(pair) for pair in CONFIRMATION_PAIRS]
        or type(config.get("commands")) is not dict
    ):
        raise ValueError("campaign config differs")
    return config


def _execute(
    stage: dict[str, object], *, executor: Callable[[dict[str, object]], dict[str, object]],
    terminal_validator: Callable[[dict[str, object], object], None],
    prior_terminals: Mapping[str, object], marker_writer: Callable[[dict[str, object]], None],
) -> tuple[int, object]:
    name = str(stage["name"])
    marker_writer({"state": "starting", "stage": name})
    if name in prior_terminals:
        terminal = prior_terminals[name]
        terminal_validator(stage, terminal)
        marker_writer({"state": "resumed", "stage": name})
        return 0, terminal
    result = executor(stage)
    code = result["exit_code"]
    if type(code) is not int:
        raise ValueError("campaign exit code differs")
    if code != 0:
        marker_writer({"state": "failed", "stage": name, "exit_code": code})
        return code, None
    terminal = result["terminal"]
    terminal_validator(stage, terminal)
    marker_writer({"state": "terminal", "stage": name})
    return 0, terminal


def run_campaign(
    config: object,
    *,
    executor: Callable[[dict[str, object]], dict[str, object]],
    terminal_validator: Callable[[dict[str, object], object], None],
    through_stage: str = "confirmation",
    marker_writer: Callable[[dict[str, object]], None] = lambda _value: None,
    prior_terminals: Mapping[str, object] | None = None,
    runtime_selector: Callable[[object], str] | None = None,
    source_publisher: Callable[[Path, str, object], dict[str, object]] | None = None,
    config_path: Path | None = None,
    capacity_guard: Callable[[], None] = lambda: None,
) -> int:
    value = _required_config(config)
    if through_stage not in {"runtime", "exploratory", "confirmation"}:
        raise ValueError("campaign through-stage differs")
    prior = prior_terminals or {}
    root = Path(value["artifact_root"])
    commands = value["commands"]
    source_publisher = source_publisher or publish_evaluation_sources
    config_path = config_path or Path("docs/unicom_fepf_run_config.json").resolve()

    def run(stage: dict[str, object]) -> tuple[int, object]:
        capacity_guard()
        return _execute(
            stage, executor=executor, terminal_validator=terminal_validator,
            prior_terminals=prior, marker_writer=marker_writer,
        )

    canary_command = list(value.get("cuda_canary_command", commands.get("cuda_canary", [])))
    code, _ = run(_stage(
        "cuda-canary", canary_command, root,
        terminal_path=root / value.get("cuda_canary_receipt", "preflight/cuda_canary_v1.json"),
    ))
    if code:
        return code
    runtime_terminals = []
    for index, command in enumerate(commands["runtime"]):
        name = f"runtime-{index:02d}"
        output = root / name / "terminal.json"
        resolved = [str(output) if item == "{output}" else item for item in command]
        code, terminal = run(_stage(name, resolved, root))
        if code:
            return code
        runtime_terminals.append(terminal)
    decision = (
        runtime_selector(tuple(runtime_terminals))
        if runtime_selector is not None
        else select_runtime_from_receipts(tuple(runtime_terminals), checkout_root=Path.cwd())
    )
    if through_stage == "runtime":
        marker_writer({"state": "complete", "through_stage": "runtime"})
        return 0

    train = apply_runtime_selection(list(commands["train"]), decision, profile=False)
    control4 = root / "exploratory-control-stage4"
    candidate4 = root / "exploratory-candidate-stage4"
    for name, mode, destination in (
        ("exploratory-control-stage4", "imprinted", control4),
        ("exploratory-candidate-stage4", "fepf_mean", candidate4),
    ):
        command = _train_command(
            train, mode=mode, training_seed=0, holdout_seed=0, stop=4,
            output=destination,
        )
        code, _ = run(_stage(name, command, root, terminal_path=destination / "run-receipt.json"))
        if code:
            return code
    config_payload = config_path.read_bytes() if config_path.is_file() else _canonical_json(value)
    config_authority = {
        "path": str(config_path.resolve()), "sha256": _sha256(config_payload),
        "bytes": len(config_payload),
    }
    epoch4_sources = [{
        "training_seed": 0, "holdout_seed": 0,
        "control_root": control4.relative_to(root).as_posix(),
        "candidate_root": candidate4.relative_to(root).as_posix(),
        "quality_profiles": [], "config": config_authority,
    }]
    code, epoch4 = run(_evaluation_stage(
        name="exploratory-epoch4-decision", phase="epoch4",
        base=list(commands["evaluate"]), root=root, sources=epoch4_sources,
        source_publisher=source_publisher,
    ))
    if code:
        return code
    if epoch4["decision"] == "CLOSE_EPOCH4":
        marker_writer({"state": "complete", "decision": "CLOSE_EPOCH4"})
        return 0
    if epoch4["decision"] != "PASS_TO_RESUME":
        raise ValueError("epoch-4 decision differs")
    control16 = root / "exploratory-control-stage16"
    candidate16 = root / "exploratory-candidate-stage16"
    for name, mode, destination, parent in (
        ("exploratory-control-stage16", "imprinted", control16, control4),
        ("exploratory-candidate-stage16", "fepf_mean", candidate16, candidate4),
    ):
        code, _ = run(_stage(name, _train_command(
            train, mode=mode, training_seed=0, holdout_seed=0, stop=16,
            output=destination, resume=parent,
        ), root, terminal_path=destination / "run-receipt.json"))
        if code:
            return code
    for stage in _profile_stages(
        prefix="exploratory", root=root, base=list(commands["profile_quality"]),
        control=control16, candidate=candidate16, config_path=config_path,
        runtime_decision=decision,
    ):
        code, _ = run(stage)
        if code:
            return code
    exploratory_profiles = [
        f"exploratory-profile-{arm}-{0 if index < 2 else 1}/terminal.json"
        for index, arm in enumerate(QUALITY_PROFILE_ORDER)
    ]
    exploratory_sources = [{
        "training_seed": 0, "holdout_seed": 0,
        "control_root": control16.relative_to(root).as_posix(),
        "candidate_root": candidate16.relative_to(root).as_posix(),
        "quality_profiles": exploratory_profiles,
    }]
    code, exploratory_result = run(_evaluation_stage(
        name="exploratory-decision", phase="exploratory",
        base=list(commands["evaluate"]), root=root, sources=exploratory_sources,
        source_publisher=source_publisher,
    ))
    if code:
        return code
    if exploratory_result["decision"] != "PROMOTE":
        marker_writer({"state": "complete", "decision": exploratory_result["decision"]})
        return 0
    random_root = root / "exploratory-random-stage16"
    code, _ = run(_stage("exploratory-random-stage16", _train_command(
        train, mode="fepf_random", training_seed=0, holdout_seed=0, stop=16,
        output=random_root,
    ), root, terminal_path=random_root / "run-receipt.json"))
    if code or through_stage == "exploratory":
        marker_writer({"state": "complete", "through_stage": "exploratory"})
        return code

    for pair_index, (training_seed, holdout_seed) in enumerate(CONFIRMATION_PAIRS):
        prefix = f"confirmation-{pair_index}"
        control = root / f"{prefix}-control"
        candidate = root / f"{prefix}-candidate"
        for name, mode, destination in (
            (f"{prefix}-control", "imprinted", control),
            (f"{prefix}-candidate", "fepf_mean", candidate),
        ):
            code, _ = run(_stage(name, _train_command(
                train, mode=mode, training_seed=training_seed,
                holdout_seed=holdout_seed, stop=16, output=destination,
            ), root, terminal_path=destination / "run-receipt.json"))
            if code:
                return code
        for stage in _profile_stages(
            prefix=prefix, root=root, base=list(commands["profile_quality"]),
            control=control, candidate=candidate, config_path=config_path,
            runtime_decision=decision,
        ):
            code, _ = run(stage)
            if code:
                return code
    confirmation_sources = []
    for pair_index, (training_seed, holdout_seed) in enumerate(CONFIRMATION_PAIRS):
        prefix = f"confirmation-{pair_index}"
        confirmation_sources.append({
            "training_seed": training_seed, "holdout_seed": holdout_seed,
            "control_root": f"{prefix}-control", "candidate_root": f"{prefix}-candidate",
            "quality_profiles": [
                f"{prefix}-profile-{arm}-{0 if index < 2 else 1}/terminal.json"
                for index, arm in enumerate(QUALITY_PROFILE_ORDER)
            ],
        })
    code, _ = run(_evaluation_stage(
        name="confirmation-decision", phase="confirmation",
        base=list(commands["evaluate"]), root=root, sources=confirmation_sources,
        source_publisher=source_publisher,
    ))
    marker_writer({"state": "complete", "through_stage": "confirmation"})
    return code


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--through-stage", choices=("runtime", "exploratory", "confirmation"),
                        default="confirmation")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    cancelled = False

    def request_cancel(_signum: int, _frame: object) -> None:
        nonlocal cancelled
        cancelled = True

    previous_handlers = {
        signum: signal.signal(signum, request_cancel)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        config = json.loads(args.config.read_bytes())
        builder = _load_module(
            Path(__file__).with_name("build_unicom_fepf_run_config.py"),
            "fepf_builder_handoff",
        )
        builder.validate_config_handoff(args.config, Path.cwd())
        validate_registered_command_vectors(config, checkout_root=Path.cwd())
        root = prepare_campaign_storage(config)
        run_fresh_process_contract_preflight(
            checkout_root=Path.cwd(), config_path=args.config.resolve(), artifact_root=root,
        )
        marker_path = root / "controller-status.json"
        def marker(value: dict[str, object]) -> None:
            write_status_marker_atomic(marker_path, value)
        executor = SubprocessStageExecutor(
            checkout_root=Path.cwd(), marker_writer=marker,
            cancelled=lambda: cancelled,
        )

        validate = RegisteredTerminalValidator(checkout_root=Path.cwd(), config=config)
        prior = load_campaign_resume(config)
        prevalidate_campaign_resume(
            config, prior, terminal_validator=validate, checkout_root=Path.cwd()
        )

        return run_campaign(
            config, executor=executor, terminal_validator=validate,
            through_stage=args.through_stage, marker_writer=marker,
            prior_terminals=prior,
            runtime_selector=lambda receipts: select_runtime_from_receipts(
                receipts, checkout_root=Path.cwd()
            ),
            config_path=args.config.resolve(),
            capacity_guard=lambda: require_campaign_remaining_capacity(config, root),
            source_publisher=lambda source_root, name, sources: (
                require_campaign_remaining_capacity(config, root),
                publish_evaluation_sources(source_root, name, sources),
            )[1],
        )
    except Exception as error:
        print(f"FEPF campaign failed: {error}", file=sys.stderr)
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
