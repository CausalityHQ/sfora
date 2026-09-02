#!/usr/bin/env python3
"""Execute one capability-limited weight-space transfer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class ProjectedTransferCapabilities:
    """The exact local-only capability namespace admitted to one child."""

    argv: tuple[str, ...]
    child_output: Path
    manifest_bytes: bytes
    staged_files: tuple[Path, ...]


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("controller paths must be normalized and absolute")
    return path


def _lower_hex(value: str, *, length: int) -> str:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError(f"expected {length} lowercase hexadecimal characters")
    return value


def _sha256(value: str) -> str:
    return _lower_hex(value, length=64)


def _commit(value: str) -> str:
    return _lower_hex(value, length=40)


def _positive_bytes(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("byte length must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("byte length must be positive")
    return parsed


def _hostname(value: str) -> str:
    if not value or "/" in value or "\0" in value:
        raise argparse.ArgumentTypeError("hostname must be one nonempty name")
    return value


def parse_controller_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the controller's closed local authority and execution surface."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--burned-manifest", required=True, type=_absolute_path)
    parser.add_argument("--burned-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--burned-manifest-bytes", required=True, type=_positive_bytes)
    parser.add_argument("--burned-image-root", required=True, type=_absolute_path)
    parser.add_argument("--source-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--seed-result", required=True, action="append", type=_absolute_path)
    parser.add_argument("--seed-result-sha256", required=True, action="append", type=_sha256)
    parser.add_argument("--seed-result-bytes", required=True, action="append", type=_positive_bytes)
    parser.add_argument("--checkpoint", required=True, action="append", type=_absolute_path)
    parser.add_argument("--checkpoint-sha256", required=True, action="append", type=_sha256)
    parser.add_argument("--checkpoint-bytes", required=True, action="append", type=_positive_bytes)
    parser.add_argument("--diagnostic-cli", required=True, type=_absolute_path)
    parser.add_argument("--python", required=True, type=_absolute_path)
    parser.add_argument("--repository", required=True, type=_absolute_path)
    parser.add_argument("--scratch-root", required=True, type=_absolute_path)
    parser.add_argument("--result-output", required=True, type=_absolute_path)
    parser.add_argument("--terminal-output", required=True, type=_absolute_path)
    parser.add_argument("--expected-hostname", required=True, type=_hostname)
    parser.add_argument("--source-commit", required=True, type=_commit)
    parser.add_argument("--source-tree-digest", required=True, type=_sha256)
    parser.add_argument("--controller-source-commit", required=True, type=_commit)
    parser.add_argument("--spec", required=True, type=_absolute_path)
    parser.add_argument("--spec-sha256", required=True, type=_sha256)
    parser.add_argument("--spec-bytes", required=True, type=_positive_bytes)
    parser.add_argument("--execute-controller", required=True, action="store_true")
    effective = list(sys.argv[1:] if argv is None else argv)
    parsed = parser.parse_args(effective)
    lists = (
        parsed.seed_result,
        parsed.seed_result_sha256,
        parsed.seed_result_bytes,
        parsed.checkpoint,
        parsed.checkpoint_sha256,
        parsed.checkpoint_bytes,
    )
    if len(parsed.seed_result) not in (2, 3) or any(
        len(values) != len(parsed.seed_result) for values in lists
    ):
        parser.error("controller requires exactly two or three correlated seed inputs")
    return parsed


@dataclass(frozen=True)
class TransferProcessObservation:
    """One controller-side observation in exact base units."""

    rss_bytes: int
    cuda_reserved_bytes: int
    memory_psi_growth_ppm: int
    swap_growth_bytes: int
    elapsed_ns: int
    progress_age_ns: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in vars(self).values()):
            raise ValueError("transfer process observation differs")


def transfer_stop_reason(observation: TransferProcessObservation) -> str | None:
    """Return the first registered lifecycle stop, or None while healthy."""

    if type(observation) is not TransferProcessObservation:
        raise TypeError("transfer process observation has the wrong concrete type")
    if observation.rss_bytes >= 110 * 1024**3 or observation.cuda_reserved_bytes >= 96 * 1024**3:
        return "memory-cap"
    if observation.memory_psi_growth_ppm > 0:
        return "memory-pressure"
    if observation.swap_growth_bytes > 0:
        return "swap-growth"
    if observation.elapsed_ns > 5_400_000_000_000:
        return "timeout"
    if observation.progress_age_ns > 300_000_000_000:
        return "progress"
    return None


def _source_probe(repository: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, status == ""


def validate_controller_environment(
    arguments: argparse.Namespace,
    *,
    hostname: str = socket.gethostname(),
    source_probe: Callable[[Path], tuple[str, bool]] = _source_probe,
) -> None:
    """Require the exact clean controller source and registered local executable paths."""

    if not isinstance(arguments, argparse.Namespace) or not callable(source_probe):
        raise ValueError("controller environment interface differs")
    repository = arguments.repository.resolve(strict=True)
    diagnostic = arguments.diagnostic_cli.resolve(strict=True)
    specification = arguments.spec.resolve(strict=True)
    python = arguments.python.resolve(strict=True)
    expected_diagnostic = repository / "scripts" / "diagnose_weight_space_transfer.py"
    expected_specification = (
        repository
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-09-02-weight-space-transfer-interpolation-design.md"
    )
    if (
        hostname != arguments.expected_hostname
        or diagnostic != expected_diagnostic
        or specification != expected_specification
        or arguments.diagnostic_cli.is_symlink()
        or arguments.spec.is_symlink()
        or not python.is_file()
        or not os.access(python, os.X_OK)
    ):
        raise ValueError("controller environment path or host differs")
    commit, clean = source_probe(repository)
    if commit != arguments.controller_source_commit or clean is not True:
        raise ValueError("controller source authority differs")


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _canonical_object(raw: bytes, *, role: str) -> dict[str, object]:
    if type(raw) is not bytes:
        raise TypeError(f"{role} bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not valid JSON") from error
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError(f"{role} is not canonical JSON")
    return value


def canonical_controller_result_bytes(
    projected: ProjectedTransferCapabilities,
    child_result: bytes,
) -> bytes:
    """Bind one recomputed child result to its exact admitted capabilities."""

    if type(projected) is not ProjectedTransferCapabilities:
        raise TypeError("projected transfer capabilities differ")
    manifest = _canonical_object(projected.manifest_bytes, role="capability manifest")
    child = _canonical_object(child_result, role="child result")
    if (
        set(manifest)
        != {
            "claim_eligible",
            "controller_source_commit",
            "roles",
            "schema",
            "source_commit",
            "source_manifest_sha256",
            "source_tree_digest",
            "spec_bytes",
            "spec_sha256",
        }
        or manifest["claim_eligible"] is not False
        or manifest["schema"] != "sfora-weight-space-transfer-capabilities-v1"
        or type(manifest["roles"]) is not list
        or set(child) != {"claim_eligible", "curves", "decision", "schema"}
        or child["claim_eligible"] is not False
        or child["schema"] != "sfora-weight-space-transfer-result-v1"
        or type(child["curves"]) is not list
        or type(child["decision"]) is not dict
    ):
        raise ValueError("transfer result schema differs")
    if any(
        type(curve) is not dict or type(curve.get("seed")) is not int for curve in child["curves"]
    ):
        raise ValueError("transfer result curve authority differs")
    seeds = cast(tuple[int, ...], tuple(curve["seed"] for curve in child["curves"]))
    terminal = child["decision"].get("terminal_class")
    expected_terminal = {
        (17, 29): {
            "provisional-interior-benefit",
            "provisional-no-interior-benefit",
        },
        (17, 29, 43): {"interior-benefit", "no-interior-benefit"},
    }.get(seeds)
    if expected_terminal is None or terminal not in expected_terminal:
        raise ValueError("transfer result seed cardinality differs")
    return _canonical_bytes(
        {
            "capabilities": manifest,
            "capabilities_sha256": hashlib.sha256(projected.manifest_bytes).hexdigest(),
            "child_result_bytes": len(child_result),
            "child_result_sha256": hashlib.sha256(child_result).hexdigest(),
            "claim_eligible": False,
            "result": child,
            "schema": "sfora-weight-space-transfer-campaign-result-v1",
        }
    )


def _read_bound_file(path: Path, *, expected_sha256: str, expected_bytes: int, role: str) -> bytes:
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be one regular file")
    raw = path.read_bytes()
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{role} identity differs")
    return raw


def _copy_new(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination, follow_symlinks=False)


def project_transfer_capabilities(
    arguments: argparse.Namespace,
    scratch: Path,
) -> ProjectedTransferCapabilities:
    """Authenticate and copy only the burned pixels and correlated seed inputs."""

    if not isinstance(arguments, argparse.Namespace) or not isinstance(scratch, Path):
        raise ValueError("transfer capability projection interface differs")
    if scratch.exists() or scratch.is_symlink():
        raise FileExistsError(scratch)
    scratch.mkdir()
    staged: list[Path] = []
    try:
        burned_raw = _read_bound_file(
            arguments.burned_manifest,
            expected_sha256=arguments.burned_manifest_sha256,
            expected_bytes=arguments.burned_manifest_bytes,
            role="burned manifest",
        )
        burned_path = scratch / "burned.json"
        burned_path.write_bytes(burned_raw)
        staged.append(burned_path)
        _read_bound_file(
            arguments.spec,
            expected_sha256=arguments.spec_sha256,
            expected_bytes=arguments.spec_bytes,
            role="transfer specification",
        )

        if arguments.burned_image_root.is_symlink() or not arguments.burned_image_root.is_dir():
            raise ValueError("burned image namespace differs")
        images = scratch / "images"
        images.mkdir()
        entries = tuple(sorted(arguments.burned_image_root.iterdir(), key=lambda item: item.name))
        if not entries or any(entry.is_symlink() or not entry.is_file() for entry in entries):
            raise ValueError("burned image namespace differs")
        for entry in entries:
            destination = images / entry.name
            _copy_new(entry, destination)
            staged.append(destination)

        seed_result_paths: list[Path] = []
        checkpoint_paths: list[Path] = []
        role_rows: list[dict[str, object]] = [
            {
                "bytes": arguments.burned_manifest_bytes,
                "role": "burned-manifest",
                "sha256": arguments.burned_manifest_sha256,
            }
        ]
        for index, seed in enumerate((17, 29, 43)[: len(arguments.seed_result)]):
            result_raw = _read_bound_file(
                arguments.seed_result[index],
                expected_sha256=arguments.seed_result_sha256[index],
                expected_bytes=arguments.seed_result_bytes[index],
                role=f"seed {seed} result",
            )
            checkpoint_raw = _read_bound_file(
                arguments.checkpoint[index],
                expected_sha256=arguments.checkpoint_sha256[index],
                expected_bytes=arguments.checkpoint_bytes[index],
                role=f"seed {seed} checkpoint",
            )
            result_path = scratch / f"seed-{seed:03d}.json"
            checkpoint_path = scratch / arguments.checkpoint[index].name
            result_path.write_bytes(result_raw)
            checkpoint_path.write_bytes(checkpoint_raw)
            seed_result_paths.append(result_path)
            checkpoint_paths.append(checkpoint_path)
            staged.extend((result_path, checkpoint_path))
            role_rows.extend(
                (
                    {
                        "bytes": arguments.seed_result_bytes[index],
                        "role": f"seed-{seed:03d}-result",
                        "sha256": arguments.seed_result_sha256[index],
                    },
                    {
                        "bytes": arguments.checkpoint_bytes[index],
                        "role": f"seed-{seed:03d}-checkpoint",
                        "sha256": arguments.checkpoint_sha256[index],
                    },
                )
            )

        manifest_bytes = _canonical_bytes(
            {
                "claim_eligible": False,
                "controller_source_commit": arguments.controller_source_commit,
                "roles": role_rows,
                "schema": "sfora-weight-space-transfer-capabilities-v1",
                "source_commit": arguments.source_commit,
                "source_manifest_sha256": arguments.source_manifest_sha256,
                "source_tree_digest": arguments.source_tree_digest,
                "spec_bytes": arguments.spec_bytes,
                "spec_sha256": arguments.spec_sha256,
            }
        )
        manifest_path = scratch / "capabilities.json"
        manifest_path.write_bytes(manifest_bytes)
        staged.append(manifest_path)
        child_output = scratch / "child-result.json"
        argv: list[str] = [
            str(arguments.python),
            str(arguments.diagnostic_cli),
            "--burned-manifest",
            str(burned_path),
            "--burned-manifest-sha256",
            arguments.burned_manifest_sha256,
            "--burned-manifest-bytes",
            str(arguments.burned_manifest_bytes),
            "--burned-image-root",
            str(images),
            "--source-manifest-sha256",
            arguments.source_manifest_sha256,
            "--source-commit",
            arguments.source_commit,
            "--source-tree-digest",
            arguments.source_tree_digest,
        ]
        for index in range(len(seed_result_paths)):
            argv.extend(
                (
                    "--seed-result",
                    str(seed_result_paths[index]),
                    "--seed-result-sha256",
                    arguments.seed_result_sha256[index],
                    "--seed-result-bytes",
                    str(arguments.seed_result_bytes[index]),
                    "--checkpoint",
                    str(checkpoint_paths[index]),
                    "--checkpoint-sha256",
                    arguments.checkpoint_sha256[index],
                    "--checkpoint-bytes",
                    str(arguments.checkpoint_bytes[index]),
                )
            )
        argv.extend(
            (
                "--output",
                str(child_output),
                "--execute-weight-space-transfer",
            )
        )
        return ProjectedTransferCapabilities(
            argv=tuple(argv),
            child_output=child_output,
            manifest_bytes=manifest_bytes,
            staged_files=tuple(staged),
        )
    except BaseException:
        shutil.rmtree(scratch)
        raise


class TransferChildFailure(RuntimeError):
    """One terminal child failure with canonical controller evidence."""

    def __init__(self, terminal_bytes: bytes) -> None:
        if type(terminal_bytes) is not bytes or not terminal_bytes:
            raise TypeError("transfer child terminal differs")
        self.terminal_bytes = terminal_bytes
        super().__init__("weight-space transfer child failed")


def _publish_new(path: Path, payload: bytes) -> None:
    if not isinstance(path, Path) or type(payload) is not bytes or not payload:
        raise ValueError("controller publication interface differs")
    partial = path.with_name(f".{path.name}.partial")
    if path.exists() or path.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(path)
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, path, follow_symlinks=False)
    finally:
        partial.unlink(missing_ok=True)


def execute_transfer_controller(
    arguments: argparse.Namespace,
    *,
    projector: Callable[
        [argparse.Namespace, Path], ProjectedTransferCapabilities
    ] = project_transfer_capabilities,
    child_runner: Callable[[ProjectedTransferCapabilities, Path], bytes],
) -> bytes:
    """Own one scratch transaction, one child, and one retained campaign artifact."""

    if (
        not isinstance(arguments, argparse.Namespace)
        or not callable(projector)
        or not callable(child_runner)
        or not isinstance(arguments.scratch_root, Path)
        or arguments.scratch_root.is_symlink()
        or not arguments.scratch_root.is_dir()
        or not isinstance(arguments.repository, Path)
        or not arguments.repository.is_absolute()
    ):
        raise ValueError("transfer controller interface differs")
    if (
        arguments.result_output.exists()
        or arguments.result_output.is_symlink()
        or arguments.terminal_output.exists()
        or arguments.terminal_output.is_symlink()
    ):
        raise FileExistsError("transfer controller output already exists")
    scratch = Path(
        tempfile.mkdtemp(prefix="sfora-weight-space-transfer-", dir=arguments.scratch_root)
    )
    scratch.rmdir()
    try:
        projected = projector(arguments, scratch)
        if type(
            projected
        ) is not ProjectedTransferCapabilities or not projected.child_output.is_relative_to(
            scratch
        ):
            raise ValueError("projected transfer capability differs")
        try:
            receipt_raw = child_runner(projected, arguments.repository)
        except TransferChildFailure as error:
            _publish_new(arguments.terminal_output, error.terminal_bytes)
            raise
        receipt = _canonical_object(receipt_raw, role="diagnostic receipt")
        if (
            set(receipt)
            != {
                "claim_eligible",
                "result",
                "result_bytes",
                "result_sha256",
                "schema",
            }
            or receipt["claim_eligible"] is not False
            or receipt["schema"] != "sfora-weight-space-transfer-diagnostic-receipt-v1"
            or receipt["result"] != str(projected.child_output)
            or projected.child_output.is_symlink()
            or not projected.child_output.is_file()
        ):
            raise ValueError("diagnostic receipt differs")
        child_result = projected.child_output.read_bytes()
        if (
            type(receipt["result_bytes"]) is not int
            or receipt["result_bytes"] != len(child_result)
            or type(receipt["result_sha256"]) is not str
            or receipt["result_sha256"] != hashlib.sha256(child_result).hexdigest()
        ):
            raise ValueError("diagnostic result binding differs")
        result = canonical_controller_result_bytes(projected, child_result)
        _publish_new(arguments.result_output, result)
        return result
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def _failure_bytes(*, reason: str, exit_code: int | None, stderr: bytes) -> bytes:
    if type(reason) is not str or not reason or type(stderr) is not bytes:
        raise ValueError("transfer child failure evidence differs")
    return _canonical_bytes(
        {
            "claim_eligible": False,
            "exit_code": exit_code,
            "reason": reason,
            "schema": "sfora-weight-space-transfer-terminal-v1",
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "status": "failed",
        }
    )


def _wait_user_unit_ready(unit_name: str, launcher_poll: Callable[[], int | None]) -> None:
    deadline = time.monotonic() + 30.0
    while True:
        completed = subprocess.run(
            (
                "systemctl",
                "--user",
                "show",
                "--property=MainPID",
                "--value",
                unit_name,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            try:
                if int(completed.stdout.strip()) > 0:
                    return
            except ValueError:
                pass
        if launcher_poll() is not None:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("transfer child unit did not become ready")
        time.sleep(0.1)


def run_transfer_child_process(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    sample: object,
    popen_factory: object = subprocess.Popen,
    stop_unit: object = lambda unit: subprocess.run(
        ("systemctl", "--user", "kill", "--signal=TERM", "--kill-whom=all", unit),
        check=True,
        capture_output=True,
    ),
    kill_unit: object = lambda unit: subprocess.run(
        ("systemctl", "--user", "kill", "--signal=KILL", "--kill-whom=all", unit),
        check=True,
        capture_output=True,
    ),
    unit_name_factory: object = lambda: (
        f"sfora-weight-space-transfer-{os.getpid()}-{time.monotonic_ns()}"
    ),
    wait_unit_ready: object = _wait_user_unit_ready,
    sleep: object = time.sleep,
) -> bytes:
    """Run exactly one network-denied child in a named, monitored user unit."""

    if (
        type(argv) is not tuple
        or not argv
        or any(type(value) is not str or not value for value in argv)
        or not isinstance(cwd, Path)
        or not cwd.is_absolute()
        or not callable(sample)
        or not callable(popen_factory)
        or not callable(stop_unit)
        or not callable(kill_unit)
        or not callable(unit_name_factory)
        or not callable(wait_unit_ready)
        or not callable(sleep)
    ):
        raise ValueError("transfer child process interface differs")
    unit_name = unit_name_factory()
    if (
        type(unit_name) is not str
        or not unit_name
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."
            for character in unit_name
        )
    ):
        raise ValueError("transfer child unit name differs")
    child_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HOME": os.environ.get("HOME", "/nonexistent"),
        "PATH": os.environ.get("PATH", os.defpath),
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{cwd / 'src'}{os.pathsep}{cwd}",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    command = (
        "systemd-run",
        "--user",
        "--wait",
        "--pipe",
        "--quiet",
        "--collect",
        f"--unit={unit_name}",
        f"--working-directory={cwd}",
        "--property=SystemCallFilter=~@network-io",
        "--property=NoNewPrivileges=yes",
        "--property=MemoryMax=118111600640",
        "--property=RuntimeMaxSec=5400",
        *(f"--setenv={name}={value}" for name, value in child_environment.items()),
        "--",
        *argv,
    )
    launcher_environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"DBUS_SESSION_BUS_ADDRESS", "HOME", "PATH", "XDG_RUNTIME_DIR"}
    }
    with (
        tempfile.TemporaryFile(mode="w+b") as stdout_stream,
        tempfile.TemporaryFile(mode="w+b") as stderr_stream,
    ):
        process = popen_factory(
            command,
            stdout=stdout_stream,
            stderr=stderr_stream,
            start_new_session=True,
            cwd=cwd,
            env=launcher_environment,
        )
        pid = getattr(process, "pid", None)
        if type(pid) is not int or pid <= 0:
            raise RuntimeError("transfer child PID differs")

        def captured_output() -> tuple[bytes, bytes]:
            stdout_stream.flush()
            stderr_stream.flush()
            stdout_stream.seek(0)
            stderr_stream.seek(0)
            return stdout_stream.read(), stderr_stream.read()

        def stop_child() -> tuple[bytes, bytes]:
            try:
                stop_unit(unit_name)
                process.wait(timeout=30.0)
            except (BaseException, subprocess.TimeoutExpired):
                kill_unit(unit_name)
                process.wait()
            return captured_output()

        try:
            wait_unit_ready(unit_name, process.poll)
        except BaseException as error:
            exit_code = process.poll()
            if exit_code is not None:
                exit_code = process.wait()
                _stdout, stderr = captured_output()
                raise TransferChildFailure(
                    _failure_bytes(reason="child-exit", exit_code=exit_code, stderr=stderr)
                ) from error
            _stdout, stderr = stop_child()
            raise TransferChildFailure(
                _failure_bytes(reason="monitor-error", exit_code=None, stderr=stderr)
            ) from error

        while process.poll() is None:
            try:
                observation = sample(unit_name)
            except BaseException as error:
                if process.poll() is not None:
                    break
                _stdout, stderr = stop_child()
                raise TransferChildFailure(
                    _failure_bytes(reason="monitor-error", exit_code=None, stderr=stderr)
                ) from error
            reason = transfer_stop_reason(observation)
            if reason is not None:
                _stdout, stderr = stop_child()
                raise TransferChildFailure(
                    _failure_bytes(reason=reason, exit_code=None, stderr=stderr)
                )
            sleep(1.0)
        exit_code = process.wait()
        stdout, stderr = captured_output()
    if type(exit_code) is not int or exit_code != 0:
        raise TransferChildFailure(
            _failure_bytes(reason="child-exit", exit_code=exit_code, stderr=stderr)
        )
    if type(stdout) is not bytes or not stdout:
        raise TransferChildFailure(
            _failure_bytes(reason="child-output", exit_code=exit_code, stderr=stderr)
        )
    return stdout


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


def _process_group_cpu_rss(pgid: int) -> tuple[int, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    cpu_ticks = 0
    rss_bytes = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat_fields = (entry / "stat").read_text()
            fields = stat_fields[stat_fields.rfind(")") + 2 :].split()
            if int(fields[2]) != pgid:
                continue
            cpu_ticks += int(fields[11]) + int(fields[12])
            rss_bytes += int((entry / "statm").read_text().split()[1]) * page_size
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return cpu_ticks, rss_bytes


def _user_unit_cpu_rss(unit_name: str) -> tuple[int, int]:
    completed = subprocess.run(
        (
            "systemctl",
            "--user",
            "show",
            "--property=CPUUsageNSec",
            "--property=MemoryCurrent",
            unit_name,
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    values = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    if set(values) != {"CPUUsageNSec", "MemoryCurrent"}:
        raise RuntimeError("transfer child cgroup observation differs")
    try:
        return int(values["CPUUsageNSec"]), int(values["MemoryCurrent"])
    except ValueError as error:
        raise RuntimeError("transfer child cgroup counters differ") from error


def _gpu_observation() -> tuple[int, bool]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=used_gpu_memory",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    used = sum(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())
    utilization = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    active = any(int(line.strip()) > 0 for line in utilization.stdout.splitlines() if line.strip())
    return used * 1024**2, active


class TransferProcessTracker:
    """Resource and forward-progress sampler for the sole serialized GPU child."""

    def __init__(self, *, now_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._now_ns = now_ns
        self._started_ns = now_ns()
        self._last_progress_ns = self._started_ns
        self._last_cpu_ticks = -1
        self._baseline_psi = _memory_psi_full_avg10()
        self._baseline_swap = _swap_used_bytes()

    def sample(self, unit_name: str) -> TransferProcessObservation:
        if type(unit_name) is not str or not unit_name:
            raise ValueError("transfer child unit observation differs")
        now = self._now_ns()
        cpu_ticks, rss_bytes = _user_unit_cpu_rss(unit_name)
        cuda_bytes, gpu_active = _gpu_observation()
        if cpu_ticks > self._last_cpu_ticks or gpu_active:
            self._last_cpu_ticks = cpu_ticks
            self._last_progress_ns = now
        return TransferProcessObservation(
            rss_bytes=rss_bytes,
            cuda_reserved_bytes=cuda_bytes,
            memory_psi_growth_ppm=max(
                0,
                round((_memory_psi_full_avg10() - self._baseline_psi) * 1_000_000),
            ),
            swap_growth_bytes=max(0, _swap_used_bytes() - self._baseline_swap),
            elapsed_ns=now - self._started_ns,
            progress_age_ns=now - self._last_progress_ns,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate authority and execute one private-network diagnostic transaction."""

    arguments = parse_controller_arguments(argv)
    try:
        validate_controller_environment(arguments)
        tracker = TransferProcessTracker()
        execute_transfer_controller(
            arguments,
            child_runner=lambda projected, cwd: run_transfer_child_process(
                projected.argv,
                cwd=cwd,
                sample=tracker.sample,
            ),
        )
    except TransferChildFailure:
        return 1
    except Exception as error:
        if not arguments.terminal_output.exists() and not arguments.terminal_output.is_symlink():
            _publish_new(
                arguments.terminal_output,
                _failure_bytes(
                    reason="authority-failure",
                    exit_code=None,
                    stderr=str(error).encode(),
                ),
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
