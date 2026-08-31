#!/usr/bin/env python3
"""Bounded lifecycle controller for the local SAGA GB10 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sfora.pass209_m4 import canonical_json_bytes
from sfora.saga_feasibility import validate_feasibility_result_bytes

CUDA_LIMIT_BYTES = 103_079_215_104
RSS_LIMIT_BYTES = 118_111_600_640
PSI_IMMEDIATE_PPM = 790_000
PSI_SUSTAINED_PPM = 500_000
PSI_SUSTAINED_SAMPLES = 3
SWAP_GROWTH_LIMIT_BYTES = 268_435_456
PROGRESS_LIMIT_NS = 300_000_000_000
WALL_LIMIT_NS = 7_200_000_000_000


@dataclass(frozen=True, slots=True)
class ControllerPaths:
    """All local inputs, scratch, and exclusive outputs."""

    model_root: Path
    snapshot_manifest: Path
    fixture: Path
    scientific_cli: Path
    scratch_root: Path
    result_output: Path
    terminal_output: Path


@dataclass(frozen=True, slots=True)
class ControllerIdentity:
    """Source and runtime identities passed verbatim to the child."""

    source_commit: str
    controller_commit: str
    binary_sha256: str
    environment_sha256: str
    host: str


@dataclass(frozen=True, slots=True)
class ResourceObservation:
    """One process-group resource/progress sample."""

    process_alive: bool
    elapsed_ns: int
    progress_age_ns: int
    cuda_reserved_bytes: int
    process_rss_bytes: int
    psi_full_avg10_ppm: int
    swap_growth_bytes: int

    def validated(self) -> ResourceObservation:
        if type(self.process_alive) is not bool:
            raise ValueError("SAGA controller process observation differs")
        for name in self.__dataclass_fields__:
            if name == "process_alive":
                continue
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("SAGA controller resource observation differs")
        return self


@dataclass(frozen=True, slots=True)
class ControllerTerminal:
    """In-memory terminal classification after PID and scratch clearance."""

    outcome: str
    reason: str
    result_published: bool
    restart_count: int
    process_cleared: bool
    scratch_cleared: bool


class ProcessRunner(Protocol):
    """Injectable process-group and resource sampler boundary."""

    def spawn(
        self,
        argv: tuple[str, ...],
        environment: dict[str, str],
        child_result_path: Path,
    ) -> object: ...

    def observe(self, process: object) -> ResourceObservation: ...

    def terminate(self, process: object) -> None: ...

    def wait(self, process: object) -> int: ...

    def is_alive(self, process: object) -> bool: ...


@dataclass(slots=True)
class _NativeState:
    process: subprocess.Popen[bytes]
    started_ns: int
    last_progress_ns: int
    last_progress_mtime_ns: int
    progress_path: Path
    swap_start_bytes: int


def _swap_used_bytes() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", maxsplit=1)
        if key in {"SwapTotal", "SwapFree"}:
            values[key] = int(raw.split()[0]) * 1024
    if set(values) != {"SwapTotal", "SwapFree"}:
        raise RuntimeError("SAGA controller cannot read swap authority")
    return values["SwapTotal"] - values["SwapFree"]


def _psi_full_avg10_ppm() -> int:
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        if line.startswith("full "):
            fields = dict(field.split("=", maxsplit=1) for field in line.split()[1:])
            return int(round(float(fields["avg10"]) * 1_000_000))
    raise RuntimeError("SAGA controller cannot read PSI authority")


def _process_group_rss_bytes(group_id: int) -> int:
    total_kib = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            tail = stat[stat.rfind(")") + 2 :].split()
            if int(tail[2]) != group_id:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total_kib * 1024


def _cuda_reserved_bytes_for_group(group_id: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    total_mib = 0
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        pid_raw, memory_raw = (part.strip() for part in line.split(",", maxsplit=1))
        pid = int(pid_raw)
        try:
            if os.getpgid(pid) == group_id:
                total_mib += int(memory_raw)
        except ProcessLookupError:
            continue
    return total_mib * 1024 * 1024


class NativeProcessRunner:
    """Linux process-group runner with bounded local resource sampling."""

    def __init__(self, *, sample_interval_seconds: float = 5.0) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self._states: dict[int, _NativeState] = {}

    def spawn(
        self,
        argv: tuple[str, ...],
        environment: dict[str, str],
        child_result_path: Path,
    ) -> object:
        try:
            progress_index = argv.index("--progress-output") + 1
            progress_path = Path(argv[progress_index])
        except (ValueError, IndexError) as error:
            raise ValueError("SAGA controller progress capability differs") from error
        if child_result_path.exists() or progress_path.exists():
            raise ValueError("SAGA controller child output already exists")
        process = subprocess.Popen(
            argv,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            start_new_session=True,
        )
        now = time.monotonic_ns()
        self._states[id(process)] = _NativeState(
            process=process,
            started_ns=now,
            last_progress_ns=now,
            last_progress_mtime_ns=0,
            progress_path=progress_path,
            swap_start_bytes=_swap_used_bytes(),
        )
        return process

    def _state(self, process: object) -> _NativeState:
        try:
            return self._states[id(process)]
        except KeyError as error:
            raise ValueError("SAGA controller process identity differs") from error

    def observe(self, process: object) -> ResourceObservation:
        state = self._state(process)
        if state.process.poll() is None and self.sample_interval_seconds > 0:
            time.sleep(self.sample_interval_seconds)
        now = time.monotonic_ns()
        if state.progress_path.exists():
            mtime_ns = state.progress_path.stat().st_mtime_ns
            if mtime_ns != state.last_progress_mtime_ns:
                state.last_progress_mtime_ns = mtime_ns
                state.last_progress_ns = now
        alive = state.process.poll() is None
        group_id = state.process.pid
        cuda_bytes = _cuda_reserved_bytes_for_group(group_id) if alive else 0
        return ResourceObservation(
            process_alive=alive,
            elapsed_ns=now - state.started_ns,
            progress_age_ns=now - state.last_progress_ns,
            cuda_reserved_bytes=cuda_bytes,
            process_rss_bytes=_process_group_rss_bytes(group_id),
            psi_full_avg10_ppm=_psi_full_avg10_ppm(),
            swap_growth_bytes=max(0, _swap_used_bytes() - state.swap_start_bytes),
        )

    def terminate(self, process: object) -> None:
        state = self._state(process)
        if state.process.poll() is None:
            os.killpg(state.process.pid, signal.SIGTERM)

    def wait(self, process: object) -> int:
        state = self._state(process)
        try:
            return state.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(state.process.pid, signal.SIGKILL)
            return state.process.wait(timeout=15)

    def is_alive(self, process: object) -> bool:
        return self._state(process).process.poll() is None


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.resolve(strict=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _lower_hex(value: str, width: int) -> bool:
    return len(value) == width and all(
        character in "0123456789abcdef" for character in value
    )


class FeasibilityController:
    """Own exactly one child process group and publish exactly one terminal."""

    def __init__(self, *, paths: ControllerPaths, identity: ControllerIdentity) -> None:
        self.paths = paths
        self.identity = identity

    def _validate(self) -> None:
        if type(self.paths) is not ControllerPaths or type(self.identity) is not ControllerIdentity:
            raise ValueError("SAGA controller authority differs")
        if not self.paths.model_root.is_dir():
            raise ValueError("SAGA controller model root differs")
        for path in (
            self.paths.snapshot_manifest,
            self.paths.fixture,
            self.paths.scientific_cli,
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError("SAGA controller input path differs")
        if self.paths.scratch_root.exists():
            raise ValueError("SAGA controller scratch root already exists")
        if self.paths.result_output.exists() or self.paths.terminal_output.exists():
            raise ValueError("SAGA controller output already exists")
        resolved_outputs = {
            self.paths.scratch_root.resolve(strict=False),
            self.paths.result_output.resolve(strict=False),
            self.paths.terminal_output.resolve(strict=False),
        }
        if len(resolved_outputs) != 3:
            raise ValueError("SAGA controller output paths overlap")
        if not _lower_hex(self.identity.source_commit, 40) or not _lower_hex(
            self.identity.controller_commit, 40
        ):
            raise ValueError("SAGA controller source identity differs")
        if not _lower_hex(self.identity.binary_sha256, 64) or not _lower_hex(
            self.identity.environment_sha256, 64
        ):
            raise ValueError("SAGA controller binary identity differs")
        if type(self.identity.host) is not str or not self.identity.host:
            raise ValueError("SAGA controller host identity differs")

    def _environment(self) -> dict[str, str]:
        forbidden = {
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "hf_token",
            "hugging_face_hub_token",
        }
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.lower() not in forbidden
        }
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            }
        )
        return environment

    def _child_argv(
        self, child_result: Path, progress_output: Path
    ) -> tuple[str, ...]:
        return (
            sys.executable,
            str(self.paths.scientific_cli),
            "--model-root",
            str(self.paths.model_root),
            "--snapshot-manifest",
            str(self.paths.snapshot_manifest),
            "--fixture",
            str(self.paths.fixture),
            "--result-output",
            str(child_result),
            "--progress-output",
            str(progress_output),
            "--source-commit",
            self.identity.source_commit,
            "--controller-commit",
            self.identity.controller_commit,
            "--binary-sha256",
            self.identity.binary_sha256,
            "--environment-sha256",
            self.identity.environment_sha256,
            "--host",
            self.identity.host,
            "--execute-feasibility",
        )

    @staticmethod
    def _stop_reason(
        observation: ResourceObservation, sustained_psi: int
    ) -> tuple[str, str] | None:
        if observation.cuda_reserved_bytes >= CUDA_LIMIT_BYTES:
            return "MEMORY_FAIL", "cuda-reserved-limit"
        if observation.process_rss_bytes >= RSS_LIMIT_BYTES:
            return "MEMORY_FAIL", "rss-limit"
        if observation.psi_full_avg10_ppm >= PSI_IMMEDIATE_PPM:
            return "MEMORY_FAIL", "psi-immediate"
        if sustained_psi >= PSI_SUSTAINED_SAMPLES:
            return "MEMORY_FAIL", "psi-sustained"
        if observation.swap_growth_bytes > SWAP_GROWTH_LIMIT_BYTES:
            return "MEMORY_FAIL", "swap-growth"
        if observation.progress_age_ns > PROGRESS_LIMIT_NS:
            return "TIME_BUDGET_FAIL", "progress-timeout"
        if observation.elapsed_ns > WALL_LIMIT_NS:
            return "TIME_BUDGET_FAIL", "wall-timeout"
        return None

    def _terminal_bytes(self, *, outcome: str, reason: str) -> bytes:
        payload: dict[str, object] = {
            "schema": "sfora-saga-gb10-feasibility-controller-terminal-v1",
            "claim_eligible": False,
            "outcome": outcome,
            "reason": reason,
            "restart_count": 0,
            "process_cleared": True,
            "scratch_cleared": True,
        }
        payload["terminal_sha256"] = hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest()
        return canonical_json_bytes(payload)

    def _cleanup_scratch(self, *paths: Path) -> None:
        for path in paths:
            if path.exists():
                path.unlink()
        self.paths.scratch_root.rmdir()

    def run(self, *, runner: ProcessRunner) -> ControllerTerminal:
        """Run and monitor the sole original child without restart."""

        self._validate()
        self.paths.scratch_root.mkdir(mode=0o700)
        child_result = self.paths.scratch_root / "scientific-result.json"
        progress_output = self.paths.scratch_root / "phase-progress.json"
        try:
            process = runner.spawn(
                self._child_argv(child_result, progress_output),
                self._environment(),
                child_result,
            )
        except (OSError, RuntimeError, ValueError):
            self._cleanup_scratch(child_result, progress_output)
            outcome = "BACKEND_INVALID"
            reason = "controller-spawn-failed"
            _write_new(
                self.paths.terminal_output,
                self._terminal_bytes(outcome=outcome, reason=reason),
            )
            return ControllerTerminal(
                outcome=outcome,
                reason=reason,
                result_published=False,
                restart_count=0,
                process_cleared=True,
                scratch_cleared=True,
            )
        sustained_psi = 0
        stop: tuple[str, str] | None = None
        result_published = False
        try:
            while True:
                observation = runner.observe(process).validated()
                sustained_psi = (
                    sustained_psi + 1
                    if observation.psi_full_avg10_ppm >= PSI_SUSTAINED_PPM
                    else 0
                )
                stop = self._stop_reason(observation, sustained_psi)
                if stop is not None:
                    runner.terminate(process)
                    break
                if not observation.process_alive:
                    break
            exit_code = runner.wait(process)
            if runner.is_alive(process):
                raise RuntimeError("SAGA controller process did not clear")

            if stop is None and exit_code == 0:
                raw = child_result.read_bytes()
                value = validate_feasibility_result_bytes(raw)
                outcome = value["outcome"]
                if type(outcome) is not str:
                    raise ValueError("SAGA scientific outcome differs")
                _write_new(self.paths.result_output, raw)
                result_published = True
                reason = "scientific-result"
            elif stop is not None:
                outcome, reason = stop
            else:
                outcome = "BACKEND_INVALID"
                reason = "scientific-child-exit"
        except (OSError, KeyError, RuntimeError, ValueError):
            outcome = "BACKEND_INVALID"
            reason = "controller-exception"
            if runner.is_alive(process):
                runner.terminate(process)
            runner.wait(process)

        self._cleanup_scratch(child_result, progress_output)
        if not result_published:
            _write_new(
                self.paths.terminal_output,
                self._terminal_bytes(outcome=outcome, reason=reason),
            )
        return ControllerTerminal(
            outcome=outcome,
            reason=reason,
            result_published=result_published,
            restart_count=0,
            process_cleared=True,
            scratch_cleared=True,
        )


def _reject_duplicate_options(argv: list[str]) -> None:
    options = [token for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise SystemExit("duplicate SAGA controller option")


def parse_controller_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local controller boundary and reject extra capability."""

    values = list(argv) if argv is not None else None
    if values is not None:
        _reject_duplicate_options(values)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--scientific-cli", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--result-output", required=True, type=Path)
    parser.add_argument("--terminal-output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--execute-controller", required=True, action="store_true")
    return parser.parse_args(values)


def main(argv: list[str] | None = None) -> int:
    """Run the bounded controller once and echo the published canonical object."""

    args = parse_controller_args(argv)
    if socket.gethostname() != args.host:
        raise SystemExit("SAGA controller host authority differs")
    scientific_bytes = args.scientific_cli.read_bytes()
    if hashlib.sha256(scientific_bytes).hexdigest() != args.binary_sha256:
        raise SystemExit("SAGA controller scientific binary digest differs")
    controller = FeasibilityController(
        paths=ControllerPaths(
            model_root=args.model_root,
            snapshot_manifest=args.snapshot_manifest,
            fixture=args.fixture,
            scientific_cli=args.scientific_cli,
            scratch_root=args.scratch_root,
            result_output=args.result_output,
            terminal_output=args.terminal_output,
        ),
        identity=ControllerIdentity(
            source_commit=args.source_commit,
            controller_commit=args.controller_commit,
            binary_sha256=args.binary_sha256,
            environment_sha256=args.environment_sha256,
            host=args.host,
        ),
    )
    terminal = controller.run(runner=NativeProcessRunner())
    published = (
        args.result_output if terminal.result_published else args.terminal_output
    )
    sys.stdout.buffer.write(published.read_bytes())
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
