#!/usr/bin/env python3
"""Run the capability-separated cross-seed denoising experiment serially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scripts.run_weight_space_transfer import (
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
) -> bytes:
    """Run preparation, construction, and evaluation once each in strict order."""

    if not isinstance(arguments, argparse.Namespace) or not callable(run_phase):
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
    prepare_argv = project_phase_argv(arguments, "prepare")
    prepare_stdout = run_phase("prepare", prepare_argv)
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
        build_stdout = run_phase("build", build_argv)
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
    evaluate_stdout = run_phase("evaluate", evaluate_argv)
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
            "status": "complete",
        }
    )
    _publish_new(arguments.terminal_output, terminal)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    """Run each child in a fresh network-denied named user unit."""

    arguments = parse_controller_arguments(argv)
    tracker = TransferProcessTracker()

    def runner(_phase: str, child_argv: tuple[str, ...]) -> bytes:
        return run_transfer_child_process(
            child_argv,
            cwd=arguments.repository,
            sample=tracker.sample,
        )

    terminal = execute_cross_seed_controller(arguments, run_phase=runner)
    sys.stdout.buffer.write(terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
