#!/usr/bin/env python3
"""Run the capability-separated cross-seed denoising experiment serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scripts.run_weight_space_transfer import (
    TransferChildFailure,
    TransferProcessObservation,
    TransferProcessTracker,
    run_transfer_child_process,
)

Phase = Literal["prepare", "build", "evaluate"]


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("path must be normalized and absolute")
    return path


def _hex(value: str, length: int) -> str:
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("lowercase hexadecimal authority differs")
    return value


def _sha256(value: str) -> str:
    return _hex(value, 64)


def _commit(value: str) -> str:
    return _hex(value, 40)


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("byte length must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("byte length must be positive")
    return parsed


def parse_controller_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the closed controller authority and local executable paths."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source-commit", required=True, type=_commit)
    parser.add_argument("--source-tree-digest", required=True, type=_sha256)
    for name in ("seed-result", "checkpoint"):
        parser.add_argument(f"--{name}", required=True, action="append", type=_absolute_path)
        parser.add_argument(f"--{name}-sha256", required=True, action="append", type=_sha256)
        parser.add_argument(f"--{name}-bytes", required=True, action="append", type=_positive)
    parser.add_argument("--scalar-result", required=True, type=_absolute_path)
    parser.add_argument("--scalar-result-sha256", required=True, type=_sha256)
    parser.add_argument("--scalar-result-bytes", required=True, type=_positive)
    parser.add_argument("--burned-manifest", required=True, type=_absolute_path)
    parser.add_argument("--burned-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--burned-manifest-bytes", required=True, type=_positive)
    parser.add_argument("--burned-image-root", required=True, type=_absolute_path)
    parser.add_argument("--source-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--prepare-cli", required=True, type=_absolute_path)
    parser.add_argument("--build-cli", required=True, type=_absolute_path)
    parser.add_argument("--evaluate-cli", required=True, type=_absolute_path)
    parser.add_argument("--python", required=True, type=_absolute_path)
    parser.add_argument("--repository", required=True, type=_absolute_path)
    parser.add_argument("--scratch-root", required=True, type=_absolute_path)
    parser.add_argument("--prepared-output", required=True, type=_absolute_path)
    parser.add_argument("--candidate-output", required=True, type=_absolute_path)
    parser.add_argument("--result-output", required=True, type=_absolute_path)
    parser.add_argument("--terminal-output", required=True, type=_absolute_path)
    parser.add_argument("--execute-controller", required=True, action="store_true")
    arguments = parser.parse_args(argv)
    lists = (
        arguments.seed_result,
        arguments.seed_result_sha256,
        arguments.seed_result_bytes,
        arguments.checkpoint,
        arguments.checkpoint_sha256,
        arguments.checkpoint_bytes,
    )
    if any(type(values) is not list or len(values) != 3 for values in lists):
        parser.error("exactly three correlated seed inputs are required")
    return arguments


@dataclass(frozen=True)
class CrossSeedProcessObservation:
    """One phase observation in exact base units."""

    rss_bytes: int
    cuda_reserved_bytes: int
    memory_psi_growth_ppm: int
    swap_growth_bytes: int
    elapsed_ns: int
    progress_age_ns: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 0 for value in vars(self).values()):
            raise ValueError("cross-seed process observation differs")


@dataclass(frozen=True)
class CrossSeedSourceReceipt:
    """Clean checkout and exact executable-byte authority."""

    commit: str
    tree: str
    hostname: str
    file_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.commit) != 40
            or len(self.tree) != 40
            or not self.hostname
            or not self.file_sha256
            or any(len(value) != 64 for value in self.file_sha256)
        ):
            raise ValueError("source receipt differs")


def _source_checkout_receipt(
    repository: Path, files: tuple[Path, ...]
) -> CrossSeedSourceReceipt:
    if (
        not repository.is_absolute()
        or repository.is_symlink()
        or not repository.is_dir()
        or not files
    ):
        raise ValueError("source repository authority differs")
    resolved = repository.resolve()
    for path in files:
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(resolved)
        ):
            raise ValueError("source executable authority differs")
    status = subprocess.run(
        ("git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"),
        check=True,
        capture_output=True,
    ).stdout
    if status:
        raise ValueError("source checkout must be clean committed authority")
    commit = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD^{tree}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if any(len(value) != 40 for value in (commit, tree)):
        raise ValueError("source git identity differs")
    return CrossSeedSourceReceipt(
        commit=commit,
        tree=tree,
        hostname=socket.gethostname(),
        file_sha256=tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in files),
    )


def stop_reason(observation: CrossSeedProcessObservation) -> str | None:
    """Return the first registered controller stop or None while healthy."""

    if type(observation) is not CrossSeedProcessObservation:
        raise TypeError("cross-seed process observation type differs")
    if observation.rss_bytes >= 110 * 1024**3:
        return "rss-cap"
    if observation.cuda_reserved_bytes >= 96 * 1024**3:
        return "cuda-cap"
    if observation.memory_psi_growth_ppm > 0:
        return "memory-pressure"
    if observation.swap_growth_bytes > 0:
        return "swap-growth"
    if observation.progress_age_ns > 300_000_000_000:
        return "progress"
    if observation.elapsed_ns > 21_600_000_000_000:
        return "wall-cap"
    return None


def _identity(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"phase output is missing: {path}")
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def project_phase_argv(
    arguments: argparse.Namespace,
    phase: Phase,
    *,
    prepared_identity: tuple[str, int] | None = None,
    candidate_identity: tuple[str, int] | None = None,
) -> tuple[str, ...]:
    """Project only the capabilities admitted to one ordered child phase."""

    if not isinstance(arguments, argparse.Namespace) or phase not in (
        "prepare",
        "build",
        "evaluate",
    ):
        raise ValueError("controller phase differs")
    if phase == "prepare":
        argv = [
            str(arguments.python),
            str(arguments.prepare_cli),
            "--source-commit",
            arguments.source_commit,
            "--source-tree-digest",
            arguments.source_tree_digest,
        ]
        for index in range(3):
            argv.extend(
                (
                    "--seed-result",
                    str(arguments.seed_result[index]),
                    "--seed-result-sha256",
                    arguments.seed_result_sha256[index],
                    "--seed-result-bytes",
                    str(arguments.seed_result_bytes[index]),
                    "--checkpoint",
                    str(arguments.checkpoint[index]),
                    "--checkpoint-sha256",
                    arguments.checkpoint_sha256[index],
                    "--checkpoint-bytes",
                    str(arguments.checkpoint_bytes[index]),
                )
            )
        argv.extend(
            (
                "--output",
                str(arguments.prepared_output),
                "--execute-cross-seed-preparation",
            )
        )
        return tuple(argv)
    if prepared_identity is None:
        raise ValueError("prepared identity is required after preparation")
    prepared_manifest = arguments.prepared_output / "manifest.json"
    if phase == "build":
        builder_prepared = arguments.scratch_root / "builder-prepared"
        return (
            str(arguments.python),
            str(arguments.build_cli),
            "--prepared-root",
            str(builder_prepared),
            "--prepared-manifest",
            str(builder_prepared / "manifest.json"),
            "--prepared-manifest-sha256",
            prepared_identity[0],
            "--prepared-manifest-bytes",
            str(prepared_identity[1]),
            "--output",
            str(arguments.candidate_output),
            "--execute-cross-seed-builder",
        )
    if candidate_identity is None:
        raise ValueError("candidate identity is required after construction")
    return (
        str(arguments.python),
        str(arguments.evaluate_cli),
        "--prepared-root",
        str(arguments.prepared_output),
        "--prepared-manifest",
        str(prepared_manifest),
        "--prepared-manifest-sha256",
        prepared_identity[0],
        "--prepared-manifest-bytes",
        str(prepared_identity[1]),
        "--candidate-root",
        str(arguments.candidate_output),
        "--candidate-receipt",
        str(arguments.candidate_output / "receipt.json"),
        "--candidate-receipt-sha256",
        candidate_identity[0],
        "--candidate-receipt-bytes",
        str(candidate_identity[1]),
        "--scalar-result",
        str(arguments.scalar_result),
        "--scalar-result-sha256",
        arguments.scalar_result_sha256,
        "--scalar-result-bytes",
        str(arguments.scalar_result_bytes),
        "--burned-manifest",
        str(arguments.burned_manifest),
        "--burned-manifest-sha256",
        arguments.burned_manifest_sha256,
        "--burned-manifest-bytes",
        str(arguments.burned_manifest_bytes),
        "--burned-image-root",
        str(arguments.burned_image_root),
        "--source-manifest-sha256",
        arguments.source_manifest_sha256,
        "--output",
        str(arguments.result_output),
        "--execute-cross-seed-evaluation",
    )


def _publish_new(path: Path, raw: bytes) -> None:
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if path.exists() or path.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(path)
    try:
        with partial.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        partial.rename(path)
    finally:
        partial.unlink(missing_ok=True)


def _failure_terminal(
    *,
    phase: Phase,
    error: TransferChildFailure,
    phase_receipts: list[dict[str, object]],
    source_receipt: CrossSeedSourceReceipt,
) -> bytes:
    try:
        child = json.loads(error.terminal_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("child failure terminal is not valid JSON") from exc
    child_keys = {
        "claim_eligible",
        "exit_code",
        "reason",
        "schema",
        "status",
        "stderr_sha256",
    }
    if (
        type(child) is not dict
        or _canonical(child) != error.terminal_bytes
        or set(child) != child_keys
        or child.get("claim_eligible") is not False
        or child.get("schema") != "sfora-weight-space-transfer-terminal-v1"
        or child.get("status") != "failed"
        or type(child.get("reason")) is not str
        or not child.get("reason")
        or (
            child.get("exit_code") is not None
            and type(child.get("exit_code")) is not int
        )
        or type(child.get("stderr_sha256")) is not str
        or len(child["stderr_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in child["stderr_sha256"])
    ):
        raise ValueError("child failure terminal differs")
    reason = child.get("reason")
    resource_reasons = {
        "cuda-cap",
        "memory-cap",
        "memory-pressure",
        "progress",
        "rss-cap",
        "swap-growth",
        "timeout",
        "wall-cap",
    }
    if reason in resource_reasons:
        terminal_class = "resource-failure"
    elif phase == "build" and reason == "child-exit" and child["exit_code"] == 3:
        terminal_class = "numerical-failure"
    else:
        terminal_class = "authority-failure"
    return _canonical(
        {
            "child_terminal_bytes": len(error.terminal_bytes),
            "child_terminal_sha256": hashlib.sha256(error.terminal_bytes).hexdigest(),
            "claim_eligible": False,
            "failed_phase": phase,
            "phases": phase_receipts,
            "schema": "sfora-cross-seed-controller-terminal-v1",
            "source": {
                "commit": source_receipt.commit,
                "file_sha256": list(source_receipt.file_sha256),
                "hostname": source_receipt.hostname,
                "tree": source_receipt.tree,
            },
            "status": "failed",
            "terminal_class": terminal_class,
        }
    )


def _stage_builder_view(prepared_root: Path, scratch_root: Path) -> Path:
    target = scratch_root / "builder-prepared"
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    raw = prepared_root.joinpath("manifest.json").read_bytes()
    value = json.loads(raw)
    if type(value) is not dict or type(value.get("seeds")) is not list:
        raise ValueError("prepared builder projection differs")
    initial = value.get("initial_tower")
    if type(initial) is not dict or type(initial.get("directory")) is not str:
        raise ValueError("prepared builder projection differs")
    directories = [initial["directory"]]
    for row in value["seeds"]:
        if type(row) is not dict or type(row.get("tower_directory")) is not str:
            raise ValueError("prepared builder projection differs")
        directories.append(row["tower_directory"])
    if len(directories) != 4 or any(
        Path(directory).name != directory for directory in directories
    ):
        raise ValueError("prepared builder projection differs")
    target.mkdir()
    try:
        os.link(prepared_root / "manifest.json", target / "manifest.json")
        for directory in directories:
            shutil.copytree(
                prepared_root / directory,
                target / directory,
                copy_function=os.link,
            )
        return target
    except BaseException:
        shutil.rmtree(target)
        raise


def execute_cross_seed_controller(
    arguments: argparse.Namespace,
    *,
    run_phase: Callable[[str, tuple[str, ...]], bytes],
    source_receipt: CrossSeedSourceReceipt,
) -> bytes:
    """Run preparation, construction, and evaluation once each in strict order."""

    if (
        not isinstance(arguments, argparse.Namespace)
        or not callable(run_phase)
        or type(source_receipt) is not CrossSeedSourceReceipt
        or len(source_receipt.file_sha256) != 5
    ):
        raise ValueError("controller execution interface differs")
    for path in (
        arguments.prepared_output,
        arguments.candidate_output,
        arguments.result_output,
        arguments.terminal_output,
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    if not arguments.scratch_root.is_dir() or arguments.scratch_root.is_symlink():
        raise ValueError("controller scratch root differs")

    phase_receipts: list[dict[str, object]] = []

    def run_registered_phase(phase: Phase, argv: tuple[str, ...]) -> bytes:
        try:
            return run_phase(phase, argv)
        except TransferChildFailure as error:
            terminal = _failure_terminal(
                phase=phase,
                error=error,
                phase_receipts=phase_receipts,
                source_receipt=source_receipt,
            )
            _publish_new(arguments.terminal_output, terminal)
            raise

    prepare_argv = project_phase_argv(arguments, "prepare")
    prepare_stdout = run_registered_phase("prepare", prepare_argv)
    prepared_identity = _identity(arguments.prepared_output / "manifest.json")
    phase_receipts.append(
        {
            "phase": "prepare",
            "stdout_sha256": hashlib.sha256(prepare_stdout).hexdigest(),
        }
    )

    builder_view = _stage_builder_view(arguments.prepared_output, arguments.scratch_root)
    try:
        build_argv = project_phase_argv(
            arguments, "build", prepared_identity=prepared_identity
        )
        build_stdout = run_registered_phase("build", build_argv)
    finally:
        if builder_view.is_dir() and not builder_view.is_symlink():
            shutil.rmtree(builder_view)
    candidate_identity = _identity(arguments.candidate_output / "receipt.json")
    phase_receipts.append(
        {"phase": "build", "stdout_sha256": hashlib.sha256(build_stdout).hexdigest()}
    )

    evaluate_argv = project_phase_argv(
        arguments,
        "evaluate",
        prepared_identity=prepared_identity,
        candidate_identity=candidate_identity,
    )
    evaluate_stdout = run_registered_phase("evaluate", evaluate_argv)
    result_sha256, result_bytes = _identity(arguments.result_output)
    if evaluate_stdout != arguments.result_output.read_bytes():
        raise ValueError("evaluator stdout/result binding differs")
    phase_receipts.append(
        {
            "phase": "evaluate",
            "stdout_sha256": hashlib.sha256(evaluate_stdout).hexdigest(),
        }
    )
    terminal = _canonical(
        {
            "candidate_receipt_sha256": candidate_identity[0],
            "claim_eligible": False,
            "phases": phase_receipts,
            "prepared_manifest_sha256": prepared_identity[0],
            "result_bytes": result_bytes,
            "result_sha256": result_sha256,
            "schema": "sfora-cross-seed-controller-terminal-v1",
            "source": {
                "commit": source_receipt.commit,
                "file_sha256": list(source_receipt.file_sha256),
                "hostname": source_receipt.hostname,
                "tree": source_receipt.tree,
            },
            "status": "complete",
        }
    )
    _publish_new(arguments.terminal_output, terminal)
    return terminal


def _run_controller_child(
    arguments: argparse.Namespace,
    child_argv: tuple[str, ...],
    *,
    tracker_factory: Callable[[], object] = TransferProcessTracker,
    child_runner: Callable[..., bytes] = run_transfer_child_process,
) -> bytes:
    tracker = tracker_factory()
    sample = getattr(tracker, "sample", None)
    if not callable(sample):
        raise ValueError("controller tracker differs")

    def cross_seed_stop(observation: object) -> str | None:
        if type(observation) is not TransferProcessObservation:
            raise TypeError("cross-seed child observation differs")
        return stop_reason(CrossSeedProcessObservation(**vars(observation)))

    return child_runner(
        child_argv,
        cwd=arguments.repository,
        sample=sample,
        stop_decider=cross_seed_stop,
        runtime_max_sec=21_600,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run each child in a fresh local-socket-only named user unit."""

    arguments = parse_controller_arguments(argv)
    source_receipt = _source_checkout_receipt(
        arguments.repository,
        (
            arguments.prepare_cli,
            arguments.build_cli,
            arguments.evaluate_cli,
            Path(__file__).resolve(),
            arguments.repository
            / "docs/superpowers/specs"
            / "2026-09-02-cross-seed-spectral-task-vector-denoising-design.md",
        ),
    )

    def runner(_phase: str, child_argv: tuple[str, ...]) -> bytes:
        return _run_controller_child(arguments, child_argv)

    terminal = execute_cross_seed_controller(
        arguments, run_phase=runner, source_receipt=source_receipt
    )
    sys.stdout.buffer.write(terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
