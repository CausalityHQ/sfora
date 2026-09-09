#!/usr/bin/env python3
"""Pressure and progress authority for the local teacher-anchored SOP panel."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

_PROGRESS_VALIDATOR: Callable[[tuple[bytes, ...], str], object] | None = None
_UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
_TERMINATION_GRACE_SECONDS = 30.0
TEACHER_ANCHORED_ARMS = ("head-only", "base", "anchor", "symmetric", "complete")


class TeacherAnchoredPanelArguments(NamedTuple):
    """Authenticated local-only inputs shared by the five-arm panel."""

    source_checkpoint: Path
    source_checkpoint_sha256: str
    teacher_checkpoint: Path
    teacher_checkpoint_sha256: str
    unicom_checkout: Path
    source_snapshot: Path
    source_snapshot_sha256: str
    teacher_snapshot: Path
    teacher_snapshot_sha256: str
    schedule: Path
    schedule_sha256: str
    image_root: Path
    image_tree_sha256: str
    teacher_pca_sha256: str
    ceiling_receipt: Path
    ceiling_receipt_sha256: str
    source_revision: str
    seed: int
    output_directory: Path
    execute_teacher_anchored_panel: bool


class TeacherAnchoredPressureSample(NamedTuple):
    """One process-group and host-pressure observation."""

    rss_bytes: int
    psi_full_avg10: float
    swap_delta_bytes: int
    progress_age_seconds: float
    wall_seconds: float


class TeacherAnchoredProgressState(NamedTuple):
    """Last launch-bound progress cursor accepted by the watchdog."""

    sequence: int
    arm: str
    epoch: int
    update: int
    line_sha256: str


class TeacherAnchoredArmLaunch(NamedTuple):
    """One exact trainer process capability prepared by the panel."""

    arm: str
    receipt: Path
    receipt_sha256: str
    output: Path
    progress: Path
    command: tuple[str, ...]


class TeacherAnchoredArmTerminal(NamedTuple):
    """Terminal evidence retained from one original arm process."""

    returncode: int
    result_bytes: bytes
    progress: TeacherAnchoredProgressState
    completed: bool
    stop_receipt: bytes | None


def parse_teacher_anchored_panel_args(arguments: list[str]) -> TeacherAnchoredPanelArguments:
    """Parse the complete local-only five-arm panel capability."""

    if type(arguments) is not list or any(type(argument) is not str for argument in arguments):
        raise ValueError("unsupported argument")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
    input_names = (
        "source-checkpoint",
        "teacher-checkpoint",
        "source-snapshot",
        "teacher-snapshot",
        "schedule",
        "unicom-checkout",
        "image-root",
        "ceiling-receipt",
    )
    for name in input_names:
        parser.add_argument(f"--{name}")
    parser.add_argument("--output-directory")
    digest_names = (
        "source-checkpoint-sha256",
        "teacher-checkpoint-sha256",
        "source-snapshot-sha256",
        "teacher-snapshot-sha256",
        "schedule-sha256",
        "image-tree-sha256",
        "teacher-pca-sha256",
        "ceiling-receipt-sha256",
    )
    for name in (*digest_names, "source-revision", "seed"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--execute-teacher-anchored-panel", action="store_true")
    known_flags = {action.option_strings[0] for action in parser._actions if action.option_strings}
    presented_flags = [item.split("=", 1)[0] for item in arguments if item.startswith("--")]
    if any(presented_flags.count(flag) > 1 for flag in known_flags):
        raise ValueError("unsupported argument")
    try:
        parsed, unknown = parser.parse_known_args(arguments)
    except (argparse.ArgumentError, SystemExit) as error:
        raise ValueError("unsupported argument") from error
    if unknown:
        raise ValueError("unsupported argument")

    paths: dict[str, Path] = {}
    for name in input_names:
        attribute = name.replace("-", "_")
        raw = getattr(parsed, attribute)
        path = Path(raw) if type(raw) is str else Path()
        directory = name in ("unicom-checkout", "image-root")
        if (
            type(raw) is not str
            or not path.is_absolute()
            or not path.exists()
            or (directory and not path.is_dir())
            or (not directory and not path.is_file())
            or (name in ("ceiling-receipt", "unicom-checkout") and path.is_symlink())
        ):
            raise ValueError("absolute local input authority differs")
        paths[attribute] = path

    raw_output = parsed.output_directory
    output_directory = Path(raw_output) if type(raw_output) is str else Path()
    if (
        type(raw_output) is not str
        or not output_directory.is_absolute()
        or not output_directory.is_dir()
        or output_directory.is_symlink()
        or any(output_directory.iterdir())
    ):
        raise ValueError("absolute local output authority differs")

    digests: dict[str, str] = {}
    for name in digest_names:
        attribute = name.replace("-", "_")
        value = getattr(parsed, attribute)
        if not _is_sha256(value):
            raise ValueError("SHA-256 authority differs")
        digests[attribute] = value
    if parsed.source_revision != _UNICOM_REVISION:
        raise ValueError("revision authority differs")
    try:
        seed = int(parsed.seed)
    except (TypeError, ValueError) as error:
        raise ValueError("seed authority differs") from error
    if type(parsed.seed) is not str or str(seed) != parsed.seed or seed not in (17, 1729, 65537):
        raise ValueError("seed authority differs")
    if parsed.execute_teacher_anchored_panel is not True:
        raise ValueError("execution flag is required")
    return TeacherAnchoredPanelArguments(
        source_checkpoint=paths["source_checkpoint"],
        source_checkpoint_sha256=digests["source_checkpoint_sha256"],
        teacher_checkpoint=paths["teacher_checkpoint"],
        teacher_checkpoint_sha256=digests["teacher_checkpoint_sha256"],
        unicom_checkout=paths["unicom_checkout"],
        source_snapshot=paths["source_snapshot"],
        source_snapshot_sha256=digests["source_snapshot_sha256"],
        teacher_snapshot=paths["teacher_snapshot"],
        teacher_snapshot_sha256=digests["teacher_snapshot_sha256"],
        schedule=paths["schedule"],
        schedule_sha256=digests["schedule_sha256"],
        image_root=paths["image_root"],
        image_tree_sha256=digests["image_tree_sha256"],
        teacher_pca_sha256=digests["teacher_pca_sha256"],
        ceiling_receipt=paths["ceiling_receipt"],
        ceiling_receipt_sha256=digests["ceiling_receipt_sha256"],
        source_revision=parsed.source_revision,
        seed=seed,
        output_directory=output_directory,
        execute_teacher_anchored_panel=True,
    )


def canonical_teacher_anchored_launch_bytes(
    arguments: TeacherAnchoredPanelArguments, arm: str, output: Path
) -> bytes:
    """Build the exact launch authority consumed by the single-arm trainer."""

    if (
        type(arguments) is not TeacherAnchoredPanelArguments
        or arm not in TEACHER_ANCHORED_ARMS
        or not isinstance(output, Path)
        or not output.is_absolute()
        or output.exists()
    ):
        raise ValueError("teacher-anchored launch receipt differs")
    value = {
        "arm": arm,
        "claim_eligible": False,
        "inputs_sha256": {
            "ceiling_receipt": arguments.ceiling_receipt_sha256,
            "image_tree": arguments.image_tree_sha256,
            "schedule": arguments.schedule_sha256,
            "source_checkpoint": arguments.source_checkpoint_sha256,
            "source_snapshot": arguments.source_snapshot_sha256,
            "teacher_checkpoint": arguments.teacher_checkpoint_sha256,
            "teacher_snapshot": arguments.teacher_snapshot_sha256,
        },
        "output": str(output),
        "schema": "sfora-teacher-anchored-launch-v1",
        "seed": arguments.seed,
        "source_revision": arguments.source_revision,
    }
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )


def prepare_teacher_anchored_arm(
    arguments: TeacherAnchoredPanelArguments, arm: str
) -> TeacherAnchoredArmLaunch:
    """Create one exclusive launch receipt and the exact trainer invocation."""

    if type(arguments) is not TeacherAnchoredPanelArguments or arm not in TEACHER_ANCHORED_ARMS:
        raise ValueError("teacher-anchored arm launch differs")
    receipt = (arguments.output_directory / f"{arm}.launch.json").resolve()
    output = (arguments.output_directory / f"{arm}.result.json").resolve()
    progress = output.with_suffix(".progress.jsonl")
    if any(path.exists() for path in (receipt, output, progress)):
        raise ValueError("teacher-anchored arm output already exists")
    payload = canonical_teacher_anchored_launch_bytes(arguments, arm, output)
    try:
        with receipt.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError("teacher-anchored arm output already exists") from error
    receipt_sha256 = hashlib.sha256(payload).hexdigest()
    trainer = Path(__file__).resolve().parent / "train_sop_teacher_anchored_distillation.py"
    command = (
        sys.executable,
        str(trainer),
        "--source-checkpoint",
        str(arguments.source_checkpoint),
        "--teacher-checkpoint",
        str(arguments.teacher_checkpoint),
        "--source-snapshot",
        str(arguments.source_snapshot),
        "--teacher-snapshot",
        str(arguments.teacher_snapshot),
        "--schedule",
        str(arguments.schedule),
        "--unicom-checkout",
        str(arguments.unicom_checkout),
        "--image-root",
        str(arguments.image_root),
        "--launch-receipt",
        str(receipt),
        "--ceiling-receipt",
        str(arguments.ceiling_receipt),
        "--output",
        str(output),
        "--source-checkpoint-sha256",
        arguments.source_checkpoint_sha256,
        "--teacher-checkpoint-sha256",
        arguments.teacher_checkpoint_sha256,
        "--source-snapshot-sha256",
        arguments.source_snapshot_sha256,
        "--teacher-snapshot-sha256",
        arguments.teacher_snapshot_sha256,
        "--schedule-sha256",
        arguments.schedule_sha256,
        "--image-tree-sha256",
        arguments.image_tree_sha256,
        "--teacher-pca-sha256",
        arguments.teacher_pca_sha256,
        "--launch-receipt-sha256",
        receipt_sha256,
        "--ceiling-receipt-sha256",
        arguments.ceiling_receipt_sha256,
        "--source-revision",
        arguments.source_revision,
        "--seed",
        str(arguments.seed),
        "--arm",
        arm,
        "--execute-teacher-anchored",
    )
    return TeacherAnchoredArmLaunch(
        arm=arm,
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        output=output,
        progress=progress,
        command=command,
    )


def prepare_teacher_anchored_panel(
    arguments: TeacherAnchoredPanelArguments,
) -> tuple[TeacherAnchoredArmLaunch, ...]:
    """Prepare each registered arm once in the fixed causal order."""

    if type(arguments) is not TeacherAnchoredPanelArguments:
        raise ValueError("teacher-anchored panel launch differs")
    return tuple(prepare_teacher_anchored_arm(arguments, arm) for arm in TEACHER_ANCHORED_ARMS)


def canonical_teacher_anchored_panel_bytes(
    arguments: TeacherAnchoredPanelArguments,
    launches: tuple[TeacherAnchoredArmLaunch, ...],
    terminals: tuple[TeacherAnchoredArmTerminal, ...],
) -> bytes:
    """Bind the five successful arm results into one claim-ineligible panel receipt."""

    if (
        type(arguments) is not TeacherAnchoredPanelArguments
        or type(launches) is not tuple
        or type(terminals) is not tuple
        or tuple(launch.arm for launch in launches) != TEACHER_ANCHORED_ARMS
        or len(terminals) != len(launches)
    ):
        raise ValueError("teacher-anchored panel receipt differs")
    arms: dict[str, dict[str, str]] = {}
    for launch, terminal in zip(launches, terminals, strict=True):
        if (
            type(launch) is not TeacherAnchoredArmLaunch
            or type(terminal) is not TeacherAnchoredArmTerminal
            or terminal.returncode != 0
            or terminal.stop_receipt is not None
            or terminal.completed is not True
            or not terminal.result_bytes
            or terminal.result_bytes[-1:] != b"\n"
            or terminal.progress.arm != launch.arm
            or not _is_sha256(launch.receipt_sha256)
        ):
            raise ValueError("teacher-anchored panel receipt differs")
        _validate_teacher_anchored_arm_result(launch, terminal.result_bytes)
        arms[launch.arm] = {
            "launch_receipt_sha256": launch.receipt_sha256,
            "result_sha256": hashlib.sha256(terminal.result_bytes).hexdigest(),
        }
    value = {
        "arms": arms,
        "claim_eligible": False,
        "inputs_sha256": {
            "ceiling_receipt": arguments.ceiling_receipt_sha256,
            "image_tree": arguments.image_tree_sha256,
            "schedule": arguments.schedule_sha256,
            "source_checkpoint": arguments.source_checkpoint_sha256,
            "source_snapshot": arguments.source_snapshot_sha256,
            "teacher_checkpoint": arguments.teacher_checkpoint_sha256,
            "teacher_snapshot": arguments.teacher_snapshot_sha256,
        },
        "schema": "sfora-teacher-anchored-panel-v1",
        "seed": arguments.seed,
        "source_revision": arguments.source_revision,
    }
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )


def run_teacher_anchored_arm_process(
    launch: TeacherAnchoredArmLaunch, *, poll_seconds: float = 30.0
) -> TeacherAnchoredArmTerminal:
    """Run one original child in its own process group and authenticate completion."""

    if (
        type(launch) is not TeacherAnchoredArmLaunch
        or launch.arm not in TEACHER_ANCHORED_ARMS
        or not _is_sha256(launch.receipt_sha256)
        or type(launch.command) is not tuple
        or not launch.command
        or type(poll_seconds) is not float
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0.0
    ):
        raise ValueError("teacher-anchored process authority differs")
    stdout_path = launch.output.with_suffix(".stdout.partial")
    stdout_final = launch.output.with_suffix(".stdout.json")
    stderr_path = launch.output.with_suffix(".stderr.log")
    if stdout_path.exists() or stdout_final.exists() or stderr_path.exists():
        raise ValueError("teacher-anchored arm output already exists")
    process: subprocess.Popen[bytes] | None = None
    start = time.monotonic()
    last_progress_at = start
    previous: TeacherAnchoredProgressState | None = None
    samples: list[TeacherAnchoredPressureSample] = []
    stopped: tuple[str, bytes] | None = None
    swap_baseline = _swap_used_bytes()
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(
                launch.command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=teacher_anchored_child_environment(),
            )
            while process.poll() is None:
                now = time.monotonic()
                current = replay_teacher_anchored_progress_file(
                    launch.progress,
                    previous,
                    launch.receipt_sha256,
                    terminal=False,
                )
                if current is not None and (
                    previous is None or current.sequence > previous.sequence
                ):
                    previous = current
                    last_progress_at = now
                sample = _sample_teacher_anchored_pressure(
                    process.pid,
                    swap_baseline,
                    now - last_progress_at,
                    now - start,
                )
                samples.append(sample)
                reason = classify_teacher_anchored_stop(tuple(samples))
                if reason is not None:
                    if process.poll() is not None:
                        break
                    _terminate_teacher_anchored_process_group(process)
                    cursor = previous or TeacherAnchoredProgressState(
                        sequence=0,
                        arm=launch.arm,
                        epoch=0,
                        update=0,
                        line_sha256="0" * 64,
                    )
                    stop_payload = canonical_teacher_anchored_stop_bytes(
                        launch_receipt_sha256=launch.receipt_sha256,
                        last_progress_sha256=cursor.line_sha256,
                        last_arm=cursor.arm,
                        last_epoch=cursor.epoch,
                        last_update=cursor.update,
                        samples=tuple(samples),
                        reason=reason,
                    )
                    stop_path = launch.output.with_suffix(".stop.json")
                    with stop_path.open("xb") as stop_stream:
                        stop_stream.write(stop_payload)
                        stop_stream.flush()
                        os.fsync(stop_stream.fileno())
                    stopped = (reason, stop_payload)
                    break
                time.sleep(poll_seconds)
        returncode = process.wait()
        if stopped is not None:
            cursor = previous or TeacherAnchoredProgressState(
                sequence=0,
                arm=launch.arm,
                epoch=0,
                update=0,
                line_sha256="0" * 64,
            )
            return TeacherAnchoredArmTerminal(
                returncode=returncode,
                result_bytes=b"",
                progress=cursor,
                completed=False,
                stop_receipt=stopped[1],
            )
        if returncode != 0:
            raise ValueError("teacher-anchored arm process failed")
        if not launch.output.is_file():
            raise ValueError("teacher-anchored arm terminal differs")
        progress_payload = launch.progress.read_bytes()
        lines = tuple(progress_payload.splitlines(keepends=True))
        progress = replay_teacher_anchored_progress_file(
            launch.progress,
            previous,
            launch.receipt_sha256,
            terminal=True,
        )
        if progress is None:
            raise ValueError("teacher-anchored progress replay differs")
        last_scientific = next(
            (
                value
                for line in reversed(lines)
                if (value := json.loads(line))["stage"] != "heartbeat"
            ),
            None,
        )
        result_bytes = launch.output.read_bytes()
        if (
            not result_bytes
            or result_bytes[-1:] != b"\n"
            or stdout_path.read_bytes() != result_bytes
        ):
            raise ValueError("teacher-anchored arm terminal differs")
        try:
            os.link(stdout_path, stdout_final)
        except FileExistsError as error:
            raise ValueError("teacher-anchored arm output already exists") from error
        stdout_path.unlink()
        return TeacherAnchoredArmTerminal(
            returncode=returncode,
            result_bytes=result_bytes,
            progress=progress,
            completed=(
                type(last_scientific) is dict
                and last_scientific.get("stage") == "complete"
                and progress.epoch == 10
            ),
            stop_receipt=None,
        )
    except BaseException:
        if process is not None and process.poll() is None:
            _terminate_teacher_anchored_process_group(process)
        raise


def _swap_used_bytes() -> int:
    """Read current Linux swap use without granting any external capability."""

    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("SwapTotal:", "SwapFree:"):
            fields[parts[0]] = int(parts[1]) * 1024
    if set(fields) != {"SwapTotal:", "SwapFree:"}:
        raise ValueError("teacher-anchored pressure authority differs")
    return fields["SwapTotal:"] - fields["SwapFree:"]


def teacher_anchored_child_environment() -> dict[str, str]:
    """Return the explicit environment capability granted to a trainer child."""

    retained = (
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TMPDIR",
        "USER",
    )
    environment = {name: os.environ[name] for name in retained if name in os.environ}
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "MKL_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )
    return environment


def _sample_teacher_anchored_pressure(
    process_group: int,
    swap_baseline: int,
    progress_age_seconds: float,
    wall_seconds: float,
) -> TeacherAnchoredPressureSample:
    """Measure the registered process-group and host pressure signals."""

    completed = subprocess.run(
        ("ps", "-o", "rss=", "-g", str(process_group)),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise ValueError("teacher-anchored pressure authority differs")
    try:
        rss_bytes = sum(int(value) for value in completed.stdout.split()) * 1024
        psi_line = next(
            line
            for line in Path("/proc/pressure/memory").read_text(encoding="ascii").splitlines()
            if line.startswith("full ")
        )
        psi_full_avg10 = float(
            next(
                field.removeprefix("avg10=")
                for field in psi_line.split()
                if field.startswith("avg10=")
            )
        )
        swap_delta = _swap_used_bytes() - swap_baseline
    except (OSError, StopIteration, ValueError) as error:
        raise ValueError("teacher-anchored pressure authority differs") from error
    return TeacherAnchoredPressureSample(
        rss_bytes=rss_bytes,
        psi_full_avg10=psi_full_avg10,
        swap_delta_bytes=swap_delta,
        progress_age_seconds=float(progress_age_seconds),
        wall_seconds=float(wall_seconds),
    )


def _terminate_teacher_anchored_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate one original child group and wait for PID clearance."""

    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + _TERMINATION_GRACE_SECONDS
    while _teacher_anchored_process_group_active(process.pid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.01)
    if _teacher_anchored_process_group_active(process.pid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.wait()


def _teacher_anchored_process_group_active(process_group: int) -> bool:
    """Return whether a process group retains any non-zombie member."""

    completed = subprocess.run(
        ("ps", "-o", "stat=", "-g", str(process_group)),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise ValueError("teacher-anchored process authority differs")
    return any(state and not state.startswith("Z") for state in completed.stdout.split())


def _validate_teacher_anchored_arm_result(
    launch: TeacherAnchoredArmLaunch, payload: bytes
) -> str:
    """Authenticate one complete scientific endpoint before panel aggregation."""

    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("teacher-anchored arm result differs") from error
    keys = {
        "arm",
        "attempted_updates",
        "authority",
        "candidate_epoch",
        "checkpoint_path",
        "checkpoint_sha256",
        "claim_eligible",
        "completed_epochs",
        "diagnostics",
        "final_encoder_sha256",
        "final_frozen_sha256",
        "final_head_sha256",
        "initial_encoder_sha256",
        "initial_frozen_sha256",
        "initial_head_sha256",
        "optimizer_reset_epochs",
        "schedule_sha256",
        "schema",
        "stopped_reason",
        "successful_updates",
    }
    authority_keys = {
        "anchor_schedule_sha256",
        "arm",
        "batch_schedule_sha256",
        "fitting_probe_sha256",
        "head_replay_sha256",
        "inputs_sha256",
        "model_mode_sha256",
        "objective_sha256",
        "ridge_sha256",
        "runtime_sha256",
        "seed",
        "snapshot_replay_sha256",
        "source_revision",
        "split_sha256",
        "teacher_pca_sha256",
        "trainable_inventory_sha256",
    }
    authority = value.get("authority") if type(value) is dict else None
    inputs = authority.get("inputs_sha256") if type(authority) is dict else None
    checkpoint = launch.output.with_suffix(".pt")
    digest_fields = (
        "final_encoder_sha256",
        "final_frozen_sha256",
        "final_head_sha256",
        "initial_encoder_sha256",
        "initial_frozen_sha256",
        "initial_head_sha256",
        "schedule_sha256",
    )
    if (
        type(value) is not dict
        or set(value) != keys
        or payload
        != json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
        or value["schema"] != "sfora-teacher-anchored-arm-v1"
        or value["claim_eligible"] is not False
        or value["arm"] != launch.arm
        or value["candidate_epoch"] != 10
        or value["stopped_reason"] is not None
        or value["completed_epochs"] != list(range(1, 11))
        or value["optimizer_reset_epochs"] != [1, 2]
        or type(value["attempted_updates"]) is not int
        or value["attempted_updates"] <= 0
        or value["successful_updates"] != value["attempted_updates"]
        or type(value["diagnostics"]) is not list
        or len(value["diagnostics"]) != 11
        or type(authority) is not dict
        or set(authority) != authority_keys
        or authority["arm"] != launch.arm
        or type(inputs) is not dict
        or inputs.get("launch_receipt") != launch.receipt_sha256
        or value["schedule_sha256"] != authority["batch_schedule_sha256"]
        or value["checkpoint_path"] != checkpoint.name
        or not _is_sha256(value["checkpoint_sha256"])
        or not checkpoint.is_file()
        or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != value["checkpoint_sha256"]
        or any(not _is_sha256(value[key]) for key in digest_fields)
    ):
        raise ValueError("teacher-anchored arm result differs")
    return hashlib.sha256(payload).hexdigest()


def execute_teacher_anchored_panel(
    arguments: TeacherAnchoredPanelArguments,
    *,
    run_arm: Callable[[TeacherAnchoredArmLaunch], TeacherAnchoredArmTerminal] = (
        run_teacher_anchored_arm_process
    ),
) -> bytes:
    """Run the five causal arms serially and publish one terminal panel receipt."""

    if type(arguments) is not TeacherAnchoredPanelArguments or not callable(run_arm):
        raise ValueError("teacher-anchored panel authority differs")
    panel_path = arguments.output_directory / "panel.result.json"
    if panel_path.exists():
        raise ValueError("teacher-anchored panel output already exists")
    _load_progress_validator()
    launches: list[TeacherAnchoredArmLaunch] = []
    terminals: list[TeacherAnchoredArmTerminal] = []
    for arm in TEACHER_ANCHORED_ARMS:
        launch = prepare_teacher_anchored_arm(arguments, arm)
        launches.append(launch)
        terminal = run_arm(launch)
        if (
            type(terminal) is not TeacherAnchoredArmTerminal
            or terminal.returncode != 0
            or terminal.stop_receipt is not None
        ):
            raise ValueError("teacher-anchored panel arm stopped")
        terminals.append(terminal)
    payload = canonical_teacher_anchored_panel_bytes(
        arguments, tuple(launches), tuple(terminals)
    )
    try:
        with panel_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError("teacher-anchored panel output already exists") from error
    return payload


def _load_progress_validator() -> Callable[[tuple[bytes, ...], str], object]:
    global _PROGRESS_VALIDATOR
    if _PROGRESS_VALIDATOR is not None:
        return _PROGRESS_VALIDATOR
    path = Path(__file__).resolve().parent / "train_sop_teacher_anchored_distillation.py"
    spec = importlib.util.spec_from_file_location("sfora_teacher_anchored_progress", path)
    if spec is None or spec.loader is None:
        raise ValueError("teacher-anchored progress replay differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    validator = getattr(module, "validate_teacher_anchored_progress_chain", None)
    if not callable(validator):
        raise ValueError("teacher-anchored progress replay differs")
    _PROGRESS_VALIDATOR = cast(Callable[[tuple[bytes, ...], str], object], validator)
    return _PROGRESS_VALIDATOR


def advance_teacher_anchored_progress(
    previous: TeacherAnchoredProgressState | None,
    lines: tuple[bytes, ...],
    launch_receipt_sha256: str,
) -> TeacherAnchoredProgressState:
    """Advance only when the complete authenticated chain strictly extends."""

    if previous is not None and type(previous) is not TeacherAnchoredProgressState:
        raise ValueError("teacher-anchored progress replay differs")
    raw_state = cast(
        TeacherAnchoredProgressState,
        _load_progress_validator()(lines, launch_receipt_sha256),
    )
    try:
        current = TeacherAnchoredProgressState(
            sequence=raw_state.sequence,
            arm=raw_state.arm,
            epoch=raw_state.epoch,
            update=raw_state.update,
            line_sha256=raw_state.line_sha256,
        )
    except AttributeError as error:
        raise ValueError("teacher-anchored progress replay differs") from error
    if previous is not None and (
        current.sequence <= previous.sequence
        or current.arm != previous.arm
        or len(lines) < previous.sequence
        or hashlib.sha256(lines[previous.sequence - 1]).hexdigest() != previous.line_sha256
    ):
        raise ValueError("teacher-anchored progress replay differs")
    return current


def replay_teacher_anchored_progress_file(
    path: Path,
    previous: TeacherAnchoredProgressState | None,
    launch_receipt_sha256: str,
    *,
    terminal: bool,
) -> TeacherAnchoredProgressState | None:
    """Replay complete records while retaining an unfinished tail during execution."""

    if (
        not isinstance(path, Path)
        or (previous is not None and type(previous) is not TeacherAnchoredProgressState)
        or not _is_sha256(launch_receipt_sha256)
        or type(terminal) is not bool
    ):
        raise ValueError("teacher-anchored progress replay differs")
    if not path.exists():
        if terminal:
            raise ValueError("teacher-anchored progress replay differs")
        return previous
    payload = path.read_bytes()
    framed = payload.endswith(b"\n")
    if terminal and (not payload or not framed):
        raise ValueError("teacher-anchored progress replay differs")
    complete = payload if framed else payload[: payload.rfind(b"\n") + 1]
    if not complete:
        return previous
    lines = tuple(complete.splitlines(keepends=True))
    if previous is None:
        return advance_teacher_anchored_progress(None, lines, launch_receipt_sha256)
    if (
        len(lines) < previous.sequence
        or hashlib.sha256(lines[previous.sequence - 1]).hexdigest() != previous.line_sha256
    ):
        raise ValueError("teacher-anchored progress replay differs")
    if len(lines) == previous.sequence:
        return previous
    return advance_teacher_anchored_progress(previous, lines, launch_receipt_sha256)


def classify_teacher_anchored_stop(
    samples: tuple[TeacherAnchoredPressureSample, ...],
) -> str | None:
    """Return the first registered resource stop from ordered samples."""

    if type(samples) is not tuple or not samples:
        raise ValueError("teacher-anchored pressure authority differs")
    sustained = 0
    for sample in samples:
        if (
            type(sample) is not TeacherAnchoredPressureSample
            or type(sample.rss_bytes) is not int
            or sample.rss_bytes < 0
            or type(sample.psi_full_avg10) is not float
            or not math.isfinite(sample.psi_full_avg10)
            or sample.psi_full_avg10 < 0.0
            or type(sample.swap_delta_bytes) is not int
            or type(sample.progress_age_seconds) is not float
            or type(sample.wall_seconds) is not float
            or not math.isfinite(sample.progress_age_seconds)
            or not math.isfinite(sample.wall_seconds)
            or sample.progress_age_seconds < 0.0
            or sample.wall_seconds < 0.0
        ):
            raise ValueError("teacher-anchored pressure authority differs")
        if sample.wall_seconds >= 18 * 3600:
            return "wall-timeout"
        if sample.rss_bytes > 32 * 1024**3:
            return "rss-cap"
        if sample.psi_full_avg10 >= 0.79:
            return "psi-immediate"
        sustained = sustained + 1 if sample.psi_full_avg10 >= 0.50 else 0
        if sustained >= 3:
            return "psi-sustained"
        if sample.swap_delta_bytes > 2 * 1024**3:
            return "swap-delta"
        if sample.progress_age_seconds >= 900.0:
            return "progress-timeout"
    return None


def canonical_teacher_anchored_stop_bytes(
    *,
    launch_receipt_sha256: str,
    last_progress_sha256: str,
    last_arm: str,
    last_epoch: int,
    last_update: int,
    samples: tuple[TeacherAnchoredPressureSample, ...],
    reason: str,
) -> bytes:
    """Encode an outcome-blind canonical external-stop receipt."""

    classified = classify_teacher_anchored_stop(samples)
    if (
        not _is_sha256(launch_receipt_sha256)
        or not _is_sha256(last_progress_sha256)
        or last_arm not in ("head-only", "base", "anchor", "symmetric", "complete")
        or type(last_epoch) is not int
        or not 0 <= last_epoch <= 10
        or type(last_update) is not int
        or last_update < 0
        or reason != classified
    ):
        raise ValueError("teacher-anchored stop receipt differs")
    value = {
        "claim_eligible": False,
        "last_arm": last_arm,
        "last_epoch": last_epoch,
        "last_progress_sha256": last_progress_sha256,
        "last_update": last_update,
        "launch_receipt_sha256": launch_receipt_sha256,
        "reason": reason,
        "samples": [sample._asdict() for sample in samples],
        "schema": "sfora-teacher-anchored-stop-v1",
    }
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main(arguments: list[str] | None = None) -> int:
    """Execute one authenticated five-arm local panel and emit its canonical receipt."""

    parsed = parse_teacher_anchored_panel_args(sys.argv[1:] if arguments is None else arguments)
    payload = execute_teacher_anchored_panel(parsed)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    main()
