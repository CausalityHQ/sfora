#!/usr/bin/env python3
"""Run the authenticated UniCOM FEPF campaign one retained process at a time."""

from __future__ import annotations

import argparse
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


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


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
        terminal: object = None
        terminal_path = stage.get("terminal_path")
        if status == 0 and isinstance(terminal_path, Path):
            raw = terminal_path.read_bytes()
            terminal = json.loads(raw)
        return {"exit_code": status, "terminal": terminal}


def _stage(name: str, command: list[str], root: Path) -> dict[str, object]:
    destination = root / name
    return {
        "name": name,
        "command": command,
        "destination": destination,
        "terminal_path": destination / "terminal.json",
        "progress_path": str(destination / "progress.json"),
    }


def _train_command(
    base: list[str], *, mode: str, training_seed: int, holdout_seed: int,
    stop: int, output: Path, resume: Path | None = None,
) -> list[str]:
    command = [
        *base, "--classifier-init", mode, "--training-seed", str(training_seed),
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
) -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    for index, arm in enumerate(QUALITY_PROFILE_ORDER):
        arm_root = control if arm == "control" else candidate
        name = f"{prefix}-profile-{arm}-{0 if index < 2 else 1}"
        command = [
            *base, "--runtime-mode", "selected", "--checkpoint",
            str(arm_root / "epoch-0016.pt"), "--run-receipt",
            str(arm_root / "run-receipt.json"), "--output", str(root / name / "terminal.json"),
        ]
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
) -> int:
    value = _required_config(config)
    if through_stage not in {"runtime", "exploratory", "confirmation"}:
        raise ValueError("campaign through-stage differs")
    prior = prior_terminals or {}
    root = Path(value["artifact_root"])
    commands = value["commands"]

    def run(stage: dict[str, object]) -> tuple[int, object]:
        return _execute(
            stage, executor=executor, terminal_validator=terminal_validator,
            prior_terminals=prior, marker_writer=marker_writer,
        )

    canary_command = list(commands.get("cuda_canary", ["python", "canary.py"]))
    code, _ = run(_stage("cuda-canary", canary_command, root))
    if code:
        return code
    runtime_terminals = []
    for index, command in enumerate(commands["runtime"]):
        code, terminal = run(_stage(f"runtime-{index:02d}", list(command), root))
        if code:
            return code
        runtime_terminals.append(terminal)
    code, _runtime = run(_stage(
        "runtime-decision", [*commands["evaluate"], "--phase", "runtime"], root
    ))
    if code or through_stage == "runtime":
        marker_writer({"state": "complete", "through_stage": "runtime"})
        return code

    train = list(commands["train"])
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
        code, _ = run(_stage(name, command, root))
        if code:
            return code
    code, epoch4 = run(_stage(
        "exploratory-epoch4-decision",
        [*commands["evaluate"], "--phase", "epoch4"], root,
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
        ), root))
        if code:
            return code
    for stage in _profile_stages(
        prefix="exploratory", root=root, base=list(commands["profile_quality"]),
        control=control16, candidate=candidate16,
    ):
        code, _ = run(stage)
        if code:
            return code
    code, exploratory_result = run(_stage(
        "exploratory-decision", [*commands["evaluate"], "--phase", "exploratory"], root
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
    ), root))
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
            ), root))
            if code:
                return code
        for stage in _profile_stages(
            prefix=prefix, root=root, base=list(commands["profile_quality"]),
            control=control, candidate=candidate,
        ):
            code, _ = run(stage)
            if code:
                return code
    code, _ = run(_stage(
        "confirmation-decision", [*commands["evaluate"], "--phase", "confirmation"], root
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
    try:
        config = json.loads(args.config.read_bytes())
        root = Path(config["artifact_root"])
        marker_path = root / "controller-status.json"
        def marker(value: dict[str, object]) -> None:
            write_status_marker_atomic(marker_path, value)
        executor = SubprocessStageExecutor(checkout_root=Path.cwd(), marker_writer=marker)

        def validate(stage: dict[str, object], terminal: object) -> None:
            if type(terminal) is not dict:
                raise ValueError(f"terminal receipt missing for {stage['name']}")

        return run_campaign(
            config, executor=executor, terminal_validator=validate,
            through_stage=args.through_stage, marker_writer=marker,
        )
    except Exception as error:
        print(f"FEPF campaign failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
