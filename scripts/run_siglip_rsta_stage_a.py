#!/usr/bin/env python3
"""Project sealed control authority and execute one local-only RSTA Stage-A child."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from sfora.pass209_m4 import canonical_json_bytes
from sfora.siglip_rsta_stage_a import (
    RstaCheckpointBinding,
    RstaControlBinding,
    rsta_control_binding_bytes,
)


@dataclass(frozen=True)
class ProjectedStageAAuthority:
    """Outcome-blind bytes and checkpoint paths admitted to the scientific child."""

    control_binding_bytes: bytes
    optimization_manifest_bytes: bytes
    checkpoints: tuple[Path, Path, Path]


@dataclass(frozen=True)
class StageAProcessObservation:
    """One controller-side process/resource observation in exact base units."""

    rss_bytes: int
    cuda_reserved_bytes: int
    memory_psi_growth_ppm: int
    swap_growth_bytes: int
    elapsed_ns: int
    progress_age_ns: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in vars(self).values()):
            raise ValueError("RSTA process observation differs from authority")


def stage_a_process_stop_reason(observation: StageAProcessObservation) -> str | None:
    """Return the first registered lifecycle stop, or None while execution is healthy."""

    if type(observation) is not StageAProcessObservation:
        raise TypeError("RSTA process observation has the wrong concrete type")
    if observation.rss_bytes + observation.cuda_reserved_bytes >= 96 * 1024**3:
        return "memory-cap"
    if observation.memory_psi_growth_ppm > 0:
        return "memory-pressure"
    if observation.swap_growth_bytes > 0:
        return "swap-growth"
    if observation.elapsed_ns > 3_600_000_000_000:
        return "timeout"
    if observation.progress_age_ns > 300_000_000_000:
        return "progress"
    return None


class StageAChildFailure(RuntimeError):
    """One terminal child failure whose canonical receipt must be preserved."""

    def __init__(self, terminal_bytes: bytes) -> None:
        if type(terminal_bytes) is not bytes:
            raise TypeError("RSTA child terminal must be concrete bytes")
        _parse_canonical_object(terminal_bytes, role="RSTA child terminal")
        self.terminal_bytes = terminal_bytes
        super().__init__("RSTA scientific child failed")


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("controller paths must be normalized and absolute")
    return path


def _lower_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError(
            "source commit must be 40 lowercase hexadecimal characters"
        )
    return value


def _hostname(value: str) -> str:
    if not value or "/" in value or "\0" in value:
        raise argparse.ArgumentTypeError("expected hostname must be one nonempty name")
    return value


def parse_controller_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the controller's closed authority and local-execution capability surface."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--seed-receipt", action="append", type=_absolute_path, required=True)
    parser.add_argument("--aggregate-receipt", type=_absolute_path, required=True)
    parser.add_argument("--checkpoint", action="append", type=_absolute_path, required=True)
    parser.add_argument("--control-manifest", type=_absolute_path, required=True)
    parser.add_argument("--optimization-image-root", type=_absolute_path, required=True)
    parser.add_argument("--scientific-cli", type=_absolute_path, required=True)
    parser.add_argument("--scratch-root", type=_absolute_path, required=True)
    parser.add_argument("--result-output", type=_absolute_path, required=True)
    parser.add_argument("--terminal-output", type=_absolute_path, required=True)
    parser.add_argument("--expected-hostname", type=_hostname, required=True)
    parser.add_argument("--expected-source-commit", type=_lower_commit, required=True)
    parser.add_argument("--expected-controller-source-commit", type=_lower_commit, required=True)
    parser.add_argument("--execute-controller", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    scalar_flags = {
        "--aggregate-receipt",
        "--control-manifest",
        "--optimization-image-root",
        "--scientific-cli",
        "--scratch-root",
        "--result-output",
        "--terminal-output",
        "--expected-hostname",
        "--expected-source-commit",
        "--expected-controller-source-commit",
        "--execute-controller",
    }
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted(flag for flag in scalar_flags if flags.count(flag) > 1)
    if duplicates:
        parser.error(f"duplicate controller arguments are forbidden: {duplicates!r}")
    parsed = parser.parse_args(effective)
    if len(parsed.seed_receipt) != 3 or len(parsed.checkpoint) != 3:
        parser.error("controller requires exactly three seed receipts and checkpoints")
    return parsed


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _child_failure_bytes(*, reason: str, exit_code: int | None, stderr: bytes) -> bytes:
    if type(reason) is not str or not reason:
        raise ValueError("RSTA child failure reason differs")
    return canonical_json_bytes(
        {
            "schema": "rsta-terminal-v1",
            "claim_eligible": False,
            "reason": reason,
            "exit_code": exit_code,
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        }
    )


def _pre_science_invalid_result_bytes() -> bytes:
    return canonical_json_bytes(
        {
            "schema": "siglip-rsta-stage-a-result-v1",
            "claim_eligible": False,
            "verdict": "INVALID",
            "first_decisive_clause": "authority-mismatch",
        }
    )


def run_stage_a_child_process(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    sample: Callable[[int], StageAProcessObservation],
    popen_factory: Callable[..., object] = subprocess.Popen,
    terminate_group: Callable[[int], None] = lambda pid: os.killpg(pid, signal.SIGTERM),
    kill_group: Callable[[int], None] = lambda pid: os.killpg(pid, signal.SIGKILL),
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Own and monitor exactly one scientific child process group."""

    if (
        type(argv) is not tuple
        or not argv
        or any(type(value) is not str or not value for value in argv)
        or not isinstance(cwd, Path)
        or not cwd.is_absolute()
        or not callable(sample)
        or not callable(popen_factory)
        or not callable(terminate_group)
        or not callable(kill_group)
        or not callable(sleep)
    ):
        raise ValueError("RSTA child process authority differs")
    process = popen_factory(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        cwd=cwd,
        env={
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HUB_OFFLINE": "1",
            "HOME": os.environ.get("HOME", "/nonexistent"),
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTHONNOUSERSITE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    pid = getattr(process, "pid", None)
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("RSTA child process PID differs")

    def stop_child() -> tuple[bytes, bytes]:
        try:
            terminate_group(pid)
        except BaseException:
            kill_group(pid)
            return process.communicate()
        try:
            return process.communicate(timeout=30.0)
        except subprocess.TimeoutExpired:
            kill_group(pid)
            return process.communicate()

    while process.poll() is None:
        try:
            observation = sample(pid)
        except BaseException as error:
            _stdout, stderr = stop_child()
            raise StageAChildFailure(
                _child_failure_bytes(reason="monitor-error", exit_code=None, stderr=stderr)
            ) from error
        reason = stage_a_process_stop_reason(observation)
        if reason is not None:
            _stdout, stderr = stop_child()
            raise StageAChildFailure(
                _child_failure_bytes(reason=reason, exit_code=None, stderr=stderr)
            )
        sleep(1.0)
    stdout, stderr = process.communicate()
    exit_code = process.poll()
    if type(exit_code) is not int or exit_code != 0:
        raise StageAChildFailure(
            _child_failure_bytes(reason="child-exit", exit_code=exit_code, stderr=stderr)
        )
    _parse_canonical_object(stdout, role="RSTA child result")
    return stdout


def _validate_stage_a_child_result(raw: bytes) -> None:
    value = _parse_canonical_object(raw, role="RSTA child result")
    if value.get("claim_eligible") is not False:
        raise ValueError("RSTA child result claim authority differs")
    if value.get("schema") == "siglip-rsta-stage-a-result-v1":
        if (
            set(value) != {"schema", "claim_eligible", "verdict", "first_decisive_clause"}
            or value.get("verdict") != "INVALID"
            or value.get("first_decisive_clause")
            not in {
                "authority-mismatch",
                "backend-unavailable",
                "fixture-failure",
                "throughput-budget",
                "determinism-failure",
            }
        ):
            raise ValueError("RSTA child INVALID result differs")
        return
    result = value.get("result")
    if (
        value.get("schema") != "siglip-rsta-stage-a-scientific-result-v1"
        or set(value)
        != {
            "schema",
            "claim_eligible",
            "authority",
            "role_panel",
            "tensor_sha256_by_id",
            "parameter_authority",
            "backend_preflight",
            "execution",
            "repeatability",
            "result",
        }
        or type(result) is not dict
        or result.get("schema") != "siglip-rsta-stage-a-result-v1"
        or result.get("claim_eligible") is not False
        or result.get("verdict") not in {"PASS_ONWARD", "FAIL", "UNRESOLVED"}
    ):
        raise ValueError("RSTA child scientific result differs")


class StageAController:
    """One-shot local child controller with exact publication and cleanup semantics."""

    def __init__(self, *, scratch_root: Path, result_output: Path, terminal_output: Path) -> None:
        if any(
            not isinstance(path, Path) for path in (scratch_root, result_output, terminal_output)
        ):
            raise TypeError("RSTA controller paths have the wrong concrete type")
        self._scratch_root = scratch_root
        self._result_output = result_output
        self._terminal_output = terminal_output
        self._started = False

    def execute(
        self,
        authority: ProjectedStageAAuthority,
        *,
        optimization_image_root: Path,
        scientific_cli: Path,
        child_runner: Callable[[tuple[str, ...], Path], bytes],
    ) -> bytes:
        """Run the scientific child exactly once and publish only one complete terminal."""

        if self._started:
            raise RuntimeError("RSTA controller already started")
        self._started = True
        if (
            type(authority) is not ProjectedStageAAuthority
            or not isinstance(optimization_image_root, Path)
            or not optimization_image_root.is_dir()
            or not isinstance(scientific_cli, Path)
            or scientific_cli
            != Path(__file__).resolve().with_name("diagnose_siglip_rsta_stage_a.py")
            or scientific_cli.is_symlink()
            or not scientific_cli.is_file()
            or not callable(child_runner)
            or self._result_output.exists()
            or self._terminal_output.exists()
        ):
            raise ValueError("RSTA controller execution or scientific CLI authority differs")
        self._scratch_root.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="rsta-stage-a.", dir=self._scratch_root))
        binding_path = scratch / "control-binding.json"
        manifest_path = scratch / "optimization-manifest.json"
        try:
            _write_new(binding_path, authority.control_binding_bytes)
            _write_new(manifest_path, authority.optimization_manifest_bytes)
            binding_sha256 = hashlib.sha256(authority.control_binding_bytes).hexdigest()
            manifest_sha256 = hashlib.sha256(authority.optimization_manifest_bytes).hexdigest()
            argv = (
                sys.executable,
                str(scientific_cli),
                "--control-binding",
                str(binding_path),
                "--control-binding-sha256",
                binding_sha256,
                "--checkpoint-seed17",
                str(authority.checkpoints[0]),
                "--checkpoint-seed29",
                str(authority.checkpoints[1]),
                "--checkpoint-seed43",
                str(authority.checkpoints[2]),
                "--optimization-manifest",
                str(manifest_path),
                "--optimization-manifest-sha256",
                manifest_sha256,
                "--image-root",
                str(optimization_image_root),
                "--execute-stage-a",
            )
            try:
                result = child_runner(argv, scratch)
            except StageAChildFailure as error:
                _write_new(self._terminal_output, error.terminal_bytes)
                raise
            try:
                if type(result) is not bytes:
                    raise TypeError("RSTA child result must be concrete bytes")
                _validate_stage_a_child_result(result)
            except Exception as error:
                failure = StageAChildFailure(
                    _child_failure_bytes(reason="invalid-child-result", exit_code=0, stderr=b"")
                )
                _write_new(self._terminal_output, failure.terminal_bytes)
                raise failure from error
            _write_new(self._result_output, result)
            return result
        finally:
            shutil.rmtree(scratch)


def _memory_psi_full_avg10() -> float:
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "full":
            values = dict(field.split("=", 1) for field in fields[1:])
            return float(values["avg10"])
    raise RuntimeError("memory PSI full avg10 is unavailable")


def _swap_used_bytes() -> int:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        if key in {"SwapTotal", "SwapFree"}:
            values[key] = int(raw.split()[0]) * 1024
    if set(values) != {"SwapTotal", "SwapFree"}:
        raise RuntimeError("swap authority is unavailable")
    return values["SwapTotal"] - values["SwapFree"]


def _current_controller_source() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _lower_commit(commit)
    tracked_status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, tracked_status == ""


def _process_group_cpu_rss(pgid: int) -> tuple[int, int, set[int]]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    cpu_ticks = 0
    rss_bytes = 0
    pids: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            if int(fields[2]) != pgid:
                continue
            pid = int(entry.name)
            pids.add(pid)
            cpu_ticks += int(fields[11]) + int(fields[12])
            rss_bytes += int((entry / "statm").read_text().split()[1]) * page_size
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return cpu_ticks, rss_bytes, pids


def _cuda_used_bytes(pids: set[int]) -> int:
    if not pids:
        return 0
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total_mib = 0
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and int(fields[0]) in pids:
            total_mib += int(fields[1])
    return total_mib * 1024**2


class StageAProcessTracker:
    """Controller-side resource and forward-progress sampler for one child group."""

    def __init__(self, *, now_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._now_ns = now_ns
        self._started_ns = now_ns()
        self._last_progress_ns = self._started_ns
        self._last_cpu_ticks = -1
        self._baseline_psi = _memory_psi_full_avg10()
        self._baseline_swap = _swap_used_bytes()

    def sample(self, pgid: int) -> StageAProcessObservation:
        now = self._now_ns()
        cpu_ticks, rss_bytes, pids = _process_group_cpu_rss(pgid)
        if cpu_ticks > self._last_cpu_ticks:
            self._last_progress_ns = now
            self._last_cpu_ticks = cpu_ticks
        return StageAProcessObservation(
            rss_bytes=rss_bytes,
            cuda_reserved_bytes=_cuda_used_bytes(pids),
            memory_psi_growth_ppm=max(
                0, round((_memory_psi_full_avg10() - self._baseline_psi) * 1_000_000)
            ),
            swap_growth_bytes=max(0, _swap_used_bytes() - self._baseline_swap),
            elapsed_ns=now - self._started_ns,
            progress_age_ns=now - self._last_progress_ns,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Project authority and run one pressure-monitored local scientific child."""

    arguments = parse_controller_args(argv)
    try:
        if socket.gethostname() != arguments.expected_hostname:
            raise ValueError("RSTA controller host allowlist differs")
        controller_commit, tracked_clean = _current_controller_source()
        if (
            controller_commit != arguments.expected_controller_source_commit
            or tracked_clean is not True
        ):
            raise ValueError("RSTA controller source authority differs")
        projected = project_stage_a_authority(
            seed_receipts=tuple(arguments.seed_receipt),
            aggregate_receipt=arguments.aggregate_receipt,
            checkpoints=tuple(arguments.checkpoint),
            control_manifest=arguments.control_manifest,
        )
        binding = _parse_canonical_object(
            projected.control_binding_bytes, role="projected RSTA control binding"
        )
        if binding.get("source_commit") != arguments.expected_source_commit:
            raise ValueError("RSTA controller source allowlist differs")
        tracker = StageAProcessTracker()
        controller = StageAController(
            scratch_root=arguments.scratch_root,
            result_output=arguments.result_output,
            terminal_output=arguments.terminal_output,
        )
        controller.execute(
            projected,
            optimization_image_root=arguments.optimization_image_root,
            scientific_cli=arguments.scientific_cli,
            child_runner=lambda child_argv, child_cwd: run_stage_a_child_process(
                child_argv, cwd=child_cwd, sample=tracker.sample
            ),
        )
    except StageAChildFailure:
        return 1
    except Exception:
        _write_new(arguments.result_output, _pre_science_invalid_result_bytes())
        return 1
    return 0


def _control_module() -> ModuleType:
    path = Path(__file__).with_name("run_siglip_proxy_control.py")
    name = "_sfora_run_siglip_proxy_control_authority"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("control authority module is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_regular(path: Path, *, role: str) -> bytes:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be one regular file")
    return path.read_bytes()


def _parse_canonical_object(raw: bytes, *, role: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not valid JSON") from error
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise ValueError(f"{role} is not canonical JSON")
    return value


def _environment_projection(environment: dict[str, object]) -> dict[str, object]:
    names = (
        "torch_version",
        "transformers_version",
        "torchvision_version",
        "cuda_runtime",
        "device_name",
        "steps_per_epoch",
        "evaluation_batch_size",
        "query_block",
    )
    if set(environment) != {
        "source_revision",
        "source_tree_digest",
        "manifest_sha256",
        "torch_version",
        "transformers_version",
        "torchvision_version",
        "cuda_runtime",
        "device_name",
        "microbatch_size",
        "steps_per_epoch",
        "evaluation_batch_size",
        "query_block",
    }:
        raise ValueError("control environment authority differs")
    return {name: environment[name] for name in names}


def _validate_checkpoint_payload(
    raw: bytes,
    *,
    seed: int,
    config_sha256: str,
    run_authority_sha256: str,
    replay_score_tolerance: float,
) -> None:
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("control checkpoint payload is unreadable") from error
    expected = {
        "claim_eligible",
        "completed_epoch",
        "config_sha256",
        "cpu_rng_state",
        "cuda_rng_states",
        "final_objective",
        "initial_snapshot_sha256",
        "maximum_score_disagreement",
        "model_state",
        "optimizer_state",
        "run_authority_sha256",
        "sampler_cycles",
        "sampler_positions",
        "schema",
        "seed",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("control checkpoint payload schema differs")
    cycles = payload["sampler_cycles"]
    positions = payload["sampler_positions"]
    disagreement = payload["maximum_score_disagreement"]
    initial_sha256 = payload["initial_snapshot_sha256"]
    cpu_rng_state = payload["cpu_rng_state"]
    cuda_rng_states = payload["cuda_rng_states"]
    model_state = payload["model_state"]
    if (
        payload["schema"] != "sfora-siglip-proxy-checkpoint-payload-v1"
        or payload["claim_eligible"] is not False
        or type(payload["seed"]) is not int
        or payload["seed"] != seed
        or type(payload["completed_epoch"]) is not int
        or payload["completed_epoch"] != 60
        or payload["config_sha256"] != config_sha256
        or payload["run_authority_sha256"] != run_authority_sha256
        or type(payload["final_objective"]) is not float
        or not math.isfinite(payload["final_objective"])
        or type(disagreement) is not float
        or not math.isfinite(disagreement)
        or not 0.0 <= disagreement <= replay_score_tolerance
        or type(initial_sha256) is not str
        or len(initial_sha256) != 64
        or any(character not in "0123456789abcdef" for character in initial_sha256)
        or type(cycles) is not tuple
        or type(positions) is not tuple
        or len(cycles) != 49
        or len(positions) != 49
        or any(type(value) is not int or value < 0 for value in cycles + positions)
        or not isinstance(cpu_rng_state, torch.Tensor)
        or cpu_rng_state.dtype != torch.uint8
        or type(cuda_rng_states) is not tuple
        or any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
            for value in cuda_rng_states
        )
        or type(payload["optimizer_state"]) is not dict
        or type(model_state) is not OrderedDict
        or not model_state
        or any(type(name) is not str or not name for name in model_state)
        or any(not isinstance(value, torch.Tensor) for value in model_state.values())
    ):
        raise ValueError("control checkpoint payload authority differs")


def project_stage_a_authority(
    *,
    seed_receipts: tuple[Path, Path, Path],
    aggregate_receipt: Path,
    checkpoints: tuple[Path, Path, Path],
    control_manifest: Path,
) -> ProjectedStageAAuthority:
    """Authenticate sealed control artifacts and return only outcome-blind child inputs."""

    if (
        type(seed_receipts) is not tuple
        or len(seed_receipts) != 3
        or any(not isinstance(path, Path) for path in seed_receipts)
        or type(checkpoints) is not tuple
        or len(checkpoints) != 3
        or any(not isinstance(path, Path) for path in checkpoints)
    ):
        raise ValueError("RSTA authority paths differ")
    receipt_bytes = tuple(
        _read_regular(path, role="control seed receipt") for path in seed_receipts
    )
    recomputed_aggregate = _control_module().control_aggregate_receipt_bytes(receipt_bytes)
    observed_aggregate = _read_regular(aggregate_receipt, role="control aggregate receipt")
    if observed_aggregate != recomputed_aggregate:
        raise ValueError("control aggregate receipt differs from authenticated seeds")
    receipts = tuple(
        _parse_canonical_object(raw, role="control seed receipt") for raw in receipt_bytes
    )
    if tuple(value.get("seed") for value in receipts) != (17, 29, 43):
        raise ValueError("control seed receipt order differs")

    environments = tuple(value.get("environment") for value in receipts)
    if any(type(value) is not dict for value in environments):
        raise ValueError("control environment authority differs")
    environment_bytes = tuple(
        canonical_json_bytes(value) for value in environments if type(value) is dict
    )
    if len(environment_bytes) != 3 or len(set(environment_bytes)) != 1:
        raise ValueError("control environment authority differs between seeds")
    environment = environments[0]
    assert type(environment) is dict

    control = _control_module()
    config = control.SiglipProxyControlConfig()
    expected_config = control._json_compatible(vars(config))
    config_sha256 = control._config_sha256(config)
    if any(
        value.get("config") != expected_config or value.get("config_sha256") != config_sha256
        for value in receipts
    ):
        raise ValueError("control config authority differs")
    dataset_config_authority = receipts[0].get("dataset")
    model_config_authority = receipts[0].get("model")
    if (
        type(dataset_config_authority) is not dict
        or dataset_config_authority.get("name") != config.dataset_name
        or dataset_config_authority.get("revision") != config.dataset_revision
    ):
        raise ValueError("control dataset identity differs from frozen config")
    if (
        type(model_config_authority) is not dict
        or model_config_authority.get("name") != config.model_name
        or model_config_authority.get("revision") != config.model_revision
        or model_config_authority.get("resolved_revision") != config.model_revision
    ):
        raise ValueError("control model identity differs from frozen config")
    try:
        run_authority = control.ControlRunAuthority(**environment)
    except (TypeError, ValueError) as error:
        raise ValueError("control run authority differs") from error
    run_authority_sha256 = control._run_authority_sha256(run_authority)
    source_authority = receipts[0].get("source")
    dataset_authority = receipts[0].get("dataset")
    if (
        type(source_authority) is not dict
        or type(dataset_authority) is not dict
        or run_authority.source_revision != source_authority.get("revision")
        or run_authority.source_tree_digest != source_authority.get("tree_digest")
        or run_authority.manifest_sha256 != dataset_authority.get("manifest_sha256")
    ):
        raise ValueError("control run authority cross-binding differs")

    microbatches: list[object] = []
    for value in receipts:
        smoke = value.get("smoke")
        training = value.get("training")
        if type(smoke) is not dict or type(training) is not dict:
            raise ValueError("control microbatch authority differs")
        microbatches.extend(
            (
                smoke.get("selected_microbatch_size"),
                training.get("microbatch_size"),
            )
        )
    microbatches.append(environment.get("microbatch_size"))
    if (
        any(type(value) is not int for value in microbatches)
        or len(set(microbatches)) != 1
        or 120 % microbatches[0] != 0
    ):
        raise ValueError("control microbatch authority differs")

    checkpoint_bindings: list[RstaCheckpointBinding] = []
    for seed, value, path in zip((17, 29, 43), receipts, checkpoints, strict=True):
        checkpoint = value.get("checkpoint")
        if type(checkpoint) is not dict or set(checkpoint) != {
            "basename",
            "receipt_basename",
            "sha256",
            "bytes",
            "epoch",
        }:
            raise ValueError("control checkpoint receipt differs")
        raw = _read_regular(path, role="control checkpoint")
        if (
            checkpoint["basename"] != path.name
            or checkpoint["epoch"] != 60
            or checkpoint["bytes"] != len(raw)
            or checkpoint["sha256"] != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("control checkpoint authority differs")
        _validate_checkpoint_payload(
            raw,
            seed=seed,
            config_sha256=config_sha256,
            run_authority_sha256=run_authority_sha256,
            replay_score_tolerance=config.replay_score_tolerance,
        )
        checkpoint_bindings.append(
            RstaCheckpointBinding(
                seed=seed,
                sha256=checkpoint["sha256"],
                byte_length=checkpoint["bytes"],
            )
        )

    manifest_raw = _read_regular(control_manifest, role="control manifest")
    manifest = _parse_canonical_object(manifest_raw, role="control manifest")
    if (
        set(manifest)
        != {
            "schema",
            "claim_eligible",
            "dataset_id",
            "dataset_revision",
            "examples",
        }
        or manifest.get("schema") != "sfora-siglip-proxy-control-manifest-v1"
    ):
        raise ValueError("control manifest schema differs")
    if manifest.get("claim_eligible") is not False:
        raise ValueError("control manifest claim authority differs")
    examples = manifest.get("examples")
    if type(examples) is not list or not examples:
        raise ValueError("control manifest examples differ")
    if (
        environment.get("manifest_sha256")
        != hashlib.sha256(canonical_json_bytes({"examples": examples})).hexdigest()
    ):
        raise ValueError("control manifest digest differs")
    optimization_examples = []
    clean_validation_examples = 0
    burned_diagnostic_examples = 0
    for row in examples:
        if (
            type(row) is not dict
            or set(row) != {"example_id", "label"}
            or type(row["example_id"]) is not str
            or not row["example_id"]
            or type(row["label"]) is not int
            or not 0 <= row["label"] <= 97
        ):
            raise ValueError("control manifest row differs")
        if row["label"] < 49:
            optimization_examples.append(row)
        elif row["label"] < 82:
            clean_validation_examples += 1
        else:
            burned_diagnostic_examples += 1
    if not optimization_examples or len({row["example_id"] for row in examples}) != len(examples):
        raise ValueError("control optimization manifest differs")
    dataset = dataset_authority
    source = source_authority
    if manifest["dataset_id"] != dataset.get("name") or manifest["dataset_revision"] != dataset.get(
        "revision"
    ):
        raise ValueError("control dataset identity differs")
    if (
        dataset.get("optimization_examples") != len(optimization_examples)
        or dataset.get("clean_validation_examples") != clean_validation_examples
        or dataset.get("burned_diagnostic_examples") != burned_diagnostic_examples
    ):
        raise ValueError("control manifest band counts differ")
    optimization_manifest_bytes = canonical_json_bytes(
        {
            "schema": "rsta-optimization-manifest-v1",
            "claim_eligible": False,
            "dataset_id": manifest["dataset_id"],
            "dataset_revision": manifest["dataset_revision"],
            "examples": optimization_examples,
        }
    )
    optimization_manifest_sha256 = hashlib.sha256(optimization_manifest_bytes).hexdigest()
    environment_sha256 = hashlib.sha256(
        canonical_json_bytes(_environment_projection(environment))
    ).hexdigest()
    binding = RstaControlBinding(
        schema="rsta-control-binding-v1",
        claim_eligible=False,
        control_complete=True,
        source_commit=source["revision"],
        config_sha256=config_sha256,
        run_authority_sha256=run_authority_sha256,
        dataset_id=manifest["dataset_id"],
        dataset_revision=manifest["dataset_revision"],
        environment_sha256=environment_sha256,
        optimization_manifest_sha256=optimization_manifest_sha256,
        selected_microbatch_size=microbatches[0],
        checkpoints=tuple(checkpoint_bindings),
    )
    return ProjectedStageAAuthority(
        control_binding_bytes=rsta_control_binding_bytes(binding),
        optimization_manifest_bytes=optimization_manifest_bytes,
        checkpoints=checkpoints,
    )


if __name__ == "__main__":
    raise SystemExit(main())
