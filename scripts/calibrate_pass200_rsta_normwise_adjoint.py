#!/usr/bin/env python3
"""Candidate-free CPU calibration for the Pass 200 RSTA adjoint diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

import numpy as np
import rsta_normwise_adjoint as core
import torch

PROTOCOL_RELATIVE_PATH = Path(
    "docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md"
)
SOURCE_PATHS = (
    "scripts/rsta_normwise_adjoint.py",
    "scripts/calibrate_pass200_rsta_normwise_adjoint.py",
    "tests/test_rsta_normwise_adjoint.py",
)
OUTPUT_DIRECTORY = Path("reports/generated/pass200_rsta_receipt")
OUTPUT_SUFFIX = "-normwise-adjoint-calibration.json"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _git(repo_root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def _commit(repo_root: Path, *arguments: str) -> str:
    value = _git(repo_root, *arguments).decode("ascii").strip()
    if _COMMIT_RE.fullmatch(value) is None:
        raise ValueError("Git commit is not full lowercase 40-hex")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_worktree_bytes(repo_root: Path, relative_path: str) -> bytes:
    path = repo_root / relative_path
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"source worktree path is not a regular file: {relative_path}")
    return path.read_bytes()


def _git_blob(repo_root: Path, commit: str, relative_path: str) -> bytes:
    return _git(repo_root, "show", f"{commit}:{relative_path}")


def _authenticate_commit(repo_root: Path, source_commit: str) -> dict[str, object]:
    repo_root = repo_root.resolve(strict=True)
    executing_commit = _commit(repo_root, "rev-parse", "HEAD")
    try:
        _git(repo_root, "merge-base", "--is-ancestor", source_commit, executing_commit)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "calibration source commit is not an ancestor of executing commit"
        ) from error

    files: dict[str, str] = {}
    for relative_path in SOURCE_PATHS:
        worktree = _regular_worktree_bytes(repo_root, relative_path)
        try:
            reviewed_blob = _git_blob(repo_root, source_commit, relative_path)
            executing_blob = _git_blob(repo_root, executing_commit, relative_path)
        except subprocess.CalledProcessError as error:
            raise ValueError(f"source Git blob is missing: {relative_path}") from error
        if worktree != reviewed_blob or worktree != executing_blob:
            raise ValueError(f"source worktree/Git blob differs: {relative_path}")
        files[relative_path] = _sha256(worktree)

    protocol_relative = PROTOCOL_RELATIVE_PATH.as_posix()
    protocol_worktree = _regular_worktree_bytes(repo_root, protocol_relative)
    protocol_commit = _commit(repo_root, "log", "-1", "--format=%H", "--", protocol_relative)
    try:
        _git(repo_root, "merge-base", "--is-ancestor", protocol_commit, source_commit)
        protocol_blob = _git_blob(repo_root, protocol_commit, protocol_relative)
        source_protocol_blob = _git_blob(repo_root, source_commit, protocol_relative)
        executing_protocol_blob = _git_blob(repo_root, executing_commit, protocol_relative)
    except subprocess.CalledProcessError as error:
        raise ValueError("protocol commit/blob provenance differs") from error
    if not (protocol_worktree == protocol_blob == source_protocol_blob == executing_protocol_blob):
        raise ValueError("protocol worktree/Git blob differs")

    cli_digest = files[SOURCE_PATHS[1]]
    return {
        "protocol": {
            "path": protocol_relative,
            "sha256": _sha256(protocol_worktree),
            "commit": protocol_commit,
        },
        "execution_audit": {
            "executing_git_commit": executing_commit,
            "calibration_source_commit": source_commit,
            "calibration_cli_path": SOURCE_PATHS[1],
            "calibration_cli_sha256": cli_digest,
        },
        "source": {"git_revision": source_commit, "files": files},
    }


def authenticate_source(repo_root: Path, destination: Path) -> dict[str, object]:
    """Authenticate the output-derived reviewed source against Git and worktree bytes."""
    repo_root = repo_root.resolve(strict=True)
    if not destination.is_absolute():
        raise ValueError("output must be absolute")
    match = re.fullmatch(r"([0-9a-f]{40})" + re.escape(OUTPUT_SUFFIX), destination.name)
    if match is None:
        raise ValueError("output basename does not contain an exact source commit")
    source_commit = match.group(1)
    expected = (repo_root / OUTPUT_DIRECTORY / destination.name).resolve(strict=False)
    if destination != expected:
        raise ValueError("output path differs from the normalized registered destination")
    return _authenticate_commit(repo_root, source_commit)


def authenticate_current_source(protocol_path: Path) -> dict[str, object]:
    """Authenticate the current HEAD for direct library construction."""
    repo_root = Path(__file__).resolve().parents[1]
    expected_protocol = (repo_root / PROTOCOL_RELATIVE_PATH).resolve(strict=True)
    if protocol_path.resolve(strict=True) != expected_protocol:
        raise ValueError("protocol path differs from the registered path")
    return _authenticate_commit(repo_root, _commit(repo_root, "rev-parse", "HEAD"))


def _configure_cpu() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_autocast_enabled("cpu", False)


def _environment() -> dict[str, object]:
    return {
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast": torch.is_autocast_enabled("cpu"),
        "model_dtype": "torch.float32",
        "reduction_dtype": "torch.float64",
        "python_version": str(platform.python_version()),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }


def _same_literal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(right, Mapping):
        return list(left) == list(right) and all(
            _same_literal(left[key], right[key]) for key in right
        )
    if isinstance(right, list):
        return len(left) == len(right) and all(
            _same_literal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _build_payload(provenance: Mapping[str, object]) -> dict[str, object]:
    correct = {
        spec.fixture_id: core.run_fixture_controls(spec) for spec in core.correct_fixture_specs()
    }
    faults = {
        spec.fixture_id: core.run_registered_fault(spec) for spec in core.registered_fault_specs()
    }
    payload = {
        "schema_version": 1,
        "diagnostic": "pass200-rsta-normwise-adjoint-calibration",
        "mode": "cpu_synthetic_calibration",
        "candidate_values_computed": False,
        "stage_a_verdict": "NOT_COMPUTED",
        "uses_test_data": "synthetic_only",
        "protocol": provenance["protocol"],
        "execution_audit": provenance["execution_audit"],
        "source": provenance["source"],
        "environment": _environment(),
        "correct_fixtures": correct,
        "registered_faults": faults,
        "all_passed": all(
            entry["passed"] is True for entry in (*correct.values(), *faults.values())
        ),
    }
    core.validate_calibration_result(payload)
    if any(
        not _same_literal(payload[name], provenance[name])
        for name in ("protocol", "execution_audit", "source")
    ):
        raise ValueError("payload provenance differs from authenticated source")
    return payload


def calibration_payload(protocol_path: Path) -> dict[str, object]:
    """Construct the complete candidate-free calibration payload once on CPU."""
    _configure_cpu()
    return _build_payload(authenticate_current_source(protocol_path))


def _write_buffer(stream: BinaryIO, data: bytes) -> int:
    return stream.write(data)


def _same_inode(first: Path, second: Path, owned: tuple[int, int]) -> bool:
    try:
        left = first.lstat()
        right = second.lstat()
    except FileNotFoundError:
        return False
    return (
        (left.st_dev, left.st_ino) == owned
        and (right.st_dev, right.st_ino) == owned
        and stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
    )


def publish_json_no_clobber(destination: Path, payload: dict[str, object]) -> None:
    """Publish validated finite JSON using the frozen hard-link protocol."""
    core.validate_calibration_result(payload)
    encoded = json.dumps(payload, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output parent directory is missing: {parent}")
    if os.path.lexists(destination):
        raise FileExistsError(f"output already exists: {destination}")
    temporary = parent / f".{destination.name}.tmp.{os.getpid()}"
    owned: tuple[int, int] | None = None
    linked = False
    temp_unlinked = False
    directory_fd: int | None = None
    original_error: BaseException | None = None
    try:

        def opener(path: str, flags: int) -> int:
            return os.open(path, flags | os.O_NOFOLLOW, 0o600)

        with open(temporary, "xb", opener=opener) as stream:
            metadata = os.fstat(stream.fileno())
            owned = (metadata.st_dev, metadata.st_ino)
            os.fchmod(stream.fileno(), 0o600)
            if _write_buffer(stream, encoded) != len(encoded):
                raise OSError("short_write")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory_fd)
        os.unlink(temporary)
        temp_unlinked = True
        os.fsync(directory_fd)
        if destination.read_bytes() != encoded:
            raise OSError("published bytes differ from validated buffer")
        return
    except BaseException as error:
        original_error = error
        raise
    finally:
        if original_error is not None and owned is not None:
            if linked and not temp_unlinked and _same_inode(temporary, destination, owned):
                with suppress(OSError):
                    os.unlink(destination)
            try:
                metadata = temporary.lstat()
                if (metadata.st_dev, metadata.st_ino) == owned:
                    os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError:
                pass
            try:
                cleanup_fd = directory_fd
                if cleanup_fd is None:
                    cleanup_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
                os.fsync(cleanup_fd)
                if directory_fd is None:
                    os.close(cleanup_fd)
            except OSError:
                pass
        if directory_fd is not None:
            os.close(directory_fd)


def calibration_exit_code(payload: Mapping[str, object]) -> int:
    return 0 if payload["all_passed"] is True else 1


def run_cli(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parse_args(argv)
        repo_root = Path(__file__).resolve().parents[1]
        provenance = authenticate_source(repo_root, arguments.output)
        _configure_cpu()
        payload = _build_payload(provenance)
        publish_json_no_clobber(arguments.output, payload)
        return calibration_exit_code(payload)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"structural failure: {error}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
