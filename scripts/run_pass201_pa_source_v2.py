"""Capture and derive deterministic authority for the Pass201 ordinary-PA source."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NoReturn
from unittest.mock import patch

from pass201_pa_source_v2_contract import (
    TRAIN_MANIFEST_CALL_GRAPH,
    BoundCheckpointMetadata,
    CheckpointMetadata,
    DirectoryImportRoot,
    ExternalDirectoryImportTarget,
    ExternalFileBinding,
    ExternalFileImportTarget,
    ImportDirectoryBinding,
    MerkleBinding,
    NonexistentImportRoot,
    OutputEvidence,
    PrelaunchAuthority,
    PrivateChildFrame,
    RepoBlob,
    ZipImportRoot,
    bind_external_file,
    bind_import_roots,
    bind_merkle,
    bind_repo_blob,
    canonical_json_bytes,
    decode_checkpoint_binding_response,
    decode_checkpoint_metadata_response,
    decode_private_child_frame,
    encode_checkpoint_binding_request,
    encode_checkpoint_metadata_request,
    encode_private_child_frame,
    hash_open_regular,
    load_strict_json_bytes,
    load_strict_json_value_bytes,
    publish_new_file,
    validate_authorization_topology,
    validate_complete_receipt,
    validate_prelaunch,
    validate_train_manifest,
)
from typer.main import get_command

import sfora.cli
import sfora.image_end_to_end as image_end_to_end
from sfora.data import ImageExample
from sfora.image_end_to_end import ImageEndToEndConfig

RECIPE_ID = "proxy_anchor.inshop.official-51db570"
RECIPE_DIGEST = "97c0fe91ae527b5d3fb3be643e139524584981f5124d706f11341506be547361"
DATASET_ROOT = Path("/home/riomus/datasets/inshop_official_standard")
PRELAUNCH_PATH = PurePosixPath("docs/pass201_pa_source_v2_prelaunch.json")
RUN_DIRECTORY = PurePosixPath("reports/generated/pass201_source_v2/run-v2")
CONTROLLER_PATH = PurePosixPath("scripts/run_pass201_pa_source_v2.py")
SOURCE_PATHS = tuple(
    PurePosixPath(path)
    for path in (
        "scripts/pass201_pa_source_v2_contract.py",
        "src/sfora/__init__.py",
        "src/sfora/ablation.py",
        "src/sfora/api.py",
        "src/sfora/arcg.py",
        "src/sfora/benchmark.py",
        "src/sfora/bn_inception.py",
        "src/sfora/catalog.py",
        "src/sfora/cea.py",
        "src/sfora/cem.py",
        "src/sfora/cli.py",
        "src/sfora/compose.py",
        "src/sfora/data.py",
        "src/sfora/encoder_ablation.py",
        "src/sfora/encoder_training.py",
        "src/sfora/evaluation.py",
        "src/sfora/experiments.py",
        "src/sfora/image_benchmark.py",
        "src/sfora/image_end_to_end.py",
        "src/sfora/image_recipes.py",
        "src/sfora/ipsr.py",
        "src/sfora/losses.py",
        "src/sfora/method.py",
        "src/sfora/oapf.py",
        "src/sfora/publication.py",
        "src/sfora/remote.py",
        "src/sfora/report.py",
        "src/sfora/text_baselines.py",
        "src/sfora/training.py",
    )
)
OUTPUT_FILENAMES = {
    "report": "report.json",
    "checkpoint": "checkpoint.pt",
    "log": "training.log",
    "resolved_config": "resolved_config.json",
    "train_manifest": "train_manifest.json",
    "receipt": "receipt.json",
}
BN_INCEPTION_CHECKPOINT_FILENAME = "bn_inception-52deb4733.pth"
REPORT_NAME = "image-end-to-end-benchmark"
REPORT_DATASET_NAME = "inshop"
REPORT_PROTOCOL = "proxy-anchor-resnet50-512"
RFC3339_UTC = re.compile(r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\Z")
EXPECTED_REPORT_KEYS = frozenset(
    {
        "name",
        "dataset_name",
        "protocol",
        "config",
        "train_examples",
        "test_examples",
        "methods",
    }
)
SIDECAR_FRAME_MAGIC = b"pass201-sidecars-v1\0"


@dataclass(frozen=True)
class CapturedAuthority:
    config_bytes: bytes
    recipe_id: str
    recipe_digest: str
    train_count: int
    query_count: int
    gallery_count: int
    protocol: str
    protocol_name: str
    rows: tuple[tuple[int, str, int], ...]
    resolved_membership_sha256: str
    resolved_train_steps: int
    steps_per_epoch: int
    total_epochs: int


@dataclass(frozen=True)
class SidecarFrame:
    pid: int
    config_bytes: bytes
    manifest_bytes: bytes
    config_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class FreezeArgs:
    checkout_root: Path
    dataset_root: Path
    python_path: Path
    frozen_absence_checked_utc: str
    output_path: Path


@dataclass(frozen=True)
class AuthorizedSource:
    authority: PrelaunchAuthority
    authorization_commit: str
    manifest_path: Path
    manifest_bytes: bytes
    manifest_sha256: str
    manifest_git_blob: str
    runtime_bindings: Mapping[str, object]
    preflight_started_utc: str


@dataclass(frozen=True)
class RunningChild:
    process: subprocess.Popen[bytes]
    started_utc: str

    @property
    def pid(self) -> int:
        return self.process.pid


@dataclass(frozen=True)
class CompletedChild:
    pid: int
    started_utc: str
    ended_utc: str
    returncode: Literal[0]


@dataclass(frozen=True)
class PostflightEvidence:
    bindings: Mapping[str, object]
    ended_utc: str


@dataclass(frozen=True)
class ScientificOutputs:
    report: OutputEvidence
    checkpoint: OutputEvidence
    log: OutputEvidence


class _CaptureComplete(BaseException):
    """Controller-only unwind that bypasses the production CLI exception handlers."""


class _CaptureRejected(BaseException):
    """Carry a fail-closed capture error through the production CLI handlers."""


def _require(predicate: bool, message: str) -> None:
    if not predicate:
        raise ValueError(message)


def _repo_blob_payload(binding: RepoBlob) -> dict[str, object]:
    return {
        "path": binding.path.as_posix(),
        "git_mode": binding.git_mode,
        "bytes": binding.byte_count,
        "sha256": binding.sha256,
        "git_blob": binding.git_blob,
    }


def _external_file_payload(binding: ExternalFileBinding) -> dict[str, object]:
    return {
        "path": binding.path.as_posix(),
        "mode": binding.mode,
        "device": binding.device,
        "inode": binding.inode,
        "bytes": binding.byte_count,
        "sha256": binding.sha256,
    }


def _merkle_payload(binding: MerkleBinding, *, root: str | None = None) -> dict[str, object]:
    return {
        "root": root if root is not None else binding.root.as_posix(),
        "algorithm": binding.algorithm,
        "count": binding.count,
        "bytes": binding.byte_count,
        "root_sha256": binding.root_sha256,
    }


def _import_directory_payload(binding: ImportDirectoryBinding) -> dict[str, object]:
    targets: list[dict[str, object]] = []
    for target in binding.external_symlink_targets:
        base: dict[str, object] = {
            "link_relative_path": target.link_relative_path.as_posix(),
            "target_text": target.target_text,
            "resolved_path": target.resolved_path.as_posix(),
            "kind": target.kind,
        }
        if isinstance(target, ExternalFileImportTarget):
            base["file"] = _external_file_payload(target.file)
        elif isinstance(target, ExternalDirectoryImportTarget):
            base["directory"] = _import_directory_payload(target.directory)
        else:  # pragma: no cover - closed dataclass union
            raise TypeError("unknown import target")
        targets.append(base)
    return {
        "root": binding.root.as_posix(),
        "tree": {
            "algorithm": binding.tree.algorithm,
            "regular_count": binding.tree.regular_count,
            "symlink_count": binding.tree.symlink_count,
            "bytes": binding.tree.byte_count,
            "root_sha256": binding.tree.root_sha256,
        },
        "external_symlink_targets": targets,
    }


def _import_root_payload(binding: object) -> dict[str, object]:
    if isinstance(binding, NonexistentImportRoot):
        return {"entry": binding.entry, "status": binding.status}
    if isinstance(binding, ZipImportRoot):
        return {
            "entry": binding.entry,
            "status": binding.status,
            "file": _external_file_payload(binding.file),
        }
    if isinstance(binding, DirectoryImportRoot):
        return {
            "entry": binding.entry,
            "status": binding.status,
            "directory": _import_directory_payload(binding.directory),
        }
    raise TypeError("unknown import root")


def _run_checked(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    try:
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            env=None if environment is None else dict(environment),
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"command failed to start: {argv[0]}") from exc
    _require(result.returncode == 0, f"command failed: {argv[0]}")
    _require(not result.stderr, f"command emitted stderr: {argv[0]}")
    return result.stdout


def _git_path(environment: Mapping[str, str]) -> Path:
    candidate = shutil.which("git", path=environment["PATH"])
    _require(candidate is not None, "Git executable not found")
    return Path(candidate).resolve(strict=True)


def _source_commit(checkout: Path, git_path: Path) -> str:
    raw = _run_checked((str(git_path), "rev-parse", "HEAD"), cwd=checkout)
    try:
        revision = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Git HEAD is not ASCII") from exc
    _require(
        len(revision) == 40 and all(character in "0123456789abcdef" for character in revision),
        "invalid source commit",
    )
    status = _run_checked((str(git_path), "status", "--porcelain=v1", "-z"), cwd=checkout)
    _require(not status, "source checkout must be clean")
    return revision


def _bind_python_executable(path: Path) -> tuple[dict[str, object], str]:
    absolute = path.absolute()
    _require(absolute.is_absolute(), "Python path must be absolute")
    resolved = absolute.resolve(strict=True)
    binding = bind_external_file(resolved)
    payload = _external_file_payload(binding)
    payload["path"] = absolute.as_posix()
    return payload, resolved.as_posix()


def _canonical_package_bytes(
    interpreter: Path, checkout: Path, environment: Mapping[str, str]
) -> bytes:
    raw = _run_checked(
        (str(interpreter), "-m", "pip", "freeze", "--all"),
        cwd=checkout,
        environment=environment,
    )
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("pip freeze output is not UTF-8") from exc
    _require(bool(lines), "pip freeze returned an empty package authority")
    _require(all(line and "\x00" not in line for line in lines), "invalid package authority")
    _require(len(lines) == len(set(lines)), "duplicate package authority line")
    return ("\n".join(sorted(lines, key=lambda value: value.encode("utf-8"))) + "\n").encode(
        "utf-8"
    )


def _python_version(interpreter: Path, checkout: Path, environment: Mapping[str, str]) -> str:
    raw = _run_checked(
        (
            str(interpreter),
            "-c",
            "import platform;print(platform.python_version())",
        ),
        cwd=checkout,
        environment=environment,
    )
    try:
        version = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Python version is not ASCII") from exc
    _require(bool(re.fullmatch(r"\d+\.\d+\.\d+(?:[^\s]*)?", version)), "Python version drift")
    return version


def _partition_line_count(path: Path, expected: dict[str, object]) -> int:
    data = _read_immutable_regular(path)
    _require(len(data) == expected["bytes"], "partition byte count drift")
    _require(hashlib.sha256(data).hexdigest() == expected["sha256"], "partition hash drift")
    return len(data.splitlines())


def _bind_image_root_link(dataset_root: Path) -> dict[str, object]:
    link = dataset_root / "Img"
    before = os.lstat(link)
    _require(stat.S_ISLNK(before.st_mode), "In-Shop Img must be a symlink")
    target = os.readlink(link)
    after = os.lstat(link)
    _require(_stat_identity(before) == _stat_identity(after), "In-Shop Img symlink drift")
    _require(target == "img", "In-Shop Img symlink target drift")
    return {"path": link.as_posix(), "target": target, "lstat_mode": before.st_mode}


def _build_replacement_environment(args: FreezeArgs) -> dict[str, str]:
    home = os.environ.get("HOME") or "/home/riomus"
    xdg_cache = os.environ.get("XDG_CACHE_HOME") or f"{home}/.cache"
    torch_home = os.environ.get("TORCH_HOME") or f"{xdg_cache}/torch"
    values = {
        "HOME": home,
        "PATH": f"{args.python_path.absolute().parent.as_posix()}:/usr/bin:/bin",
        "PYTHONPATH": f"{args.checkout_root.resolve(strict=True).as_posix()}/src",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH") or "/usr/local/cuda/lib64",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES") or "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "LC_ALL": os.environ.get("LC_ALL") or "C.UTF-8",
        "LANG": os.environ.get("LANG") or "C.UTF-8",
        "TZ": "UTC",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS") or "1",
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS") or "1",
        "XDG_CACHE_HOME": xdg_cache,
        "TORCH_HOME": torch_home,
    }
    _require(all(type(value) is str and bool(value) for value in values.values()), "environment")
    return values


def _bind_freeze_runtime(args: FreezeArgs, environment: Mapping[str, str]) -> dict[str, object]:
    checkout = args.checkout_root.resolve(strict=True)
    _require(checkout == args.checkout_root.absolute(), "checkout root contains a symlink")
    git_path = _git_path(environment)
    source_commit = _source_commit(checkout, git_path)
    python_payload, python_realpath = _bind_python_executable(args.python_path)
    packages = _canonical_package_bytes(args.python_path.absolute(), checkout, environment)
    source_tree = bind_merkle(checkout / "src" / "sfora")
    partition_path = args.dataset_root / "Eval" / "list_eval_partition.txt"
    partition = _external_file_payload(bind_external_file(partition_path))
    declared_image_root = args.dataset_root / "Img" / "img"
    resolved_image_root = args.dataset_root / "img" / "img"
    _require(
        declared_image_root.resolve(strict=True) == resolved_image_root.resolve(strict=True),
        "resolved image root drift",
    )
    pretrained_path = (
        Path(environment["TORCH_HOME"]) / "hub" / "checkpoints" / BN_INCEPTION_CHECKPOINT_FILENAME
    )
    return {
        "source_commit": source_commit,
        "controller": _repo_blob_payload(bind_repo_blob(checkout, source_commit, CONTROLLER_PATH)),
        "source_files": [
            _repo_blob_payload(bind_repo_blob(checkout, source_commit, path))
            for path in SOURCE_PATHS
        ],
        "python_tree": _merkle_payload(source_tree, root="src/sfora"),
        "pyproject": _repo_blob_payload(
            bind_repo_blob(checkout, source_commit, PurePosixPath("pyproject.toml"))
        ),
        "lockfile": _repo_blob_payload(
            bind_repo_blob(checkout, source_commit, PurePosixPath("uv.lock"))
        ),
        "python": python_payload,
        "python_realpath": python_realpath,
        "python_version": _python_version(args.python_path.absolute(), checkout, environment),
        "git": _external_file_payload(bind_external_file(git_path)),
        "python_packages": {
            "bytes": len(packages),
            "sha256": hashlib.sha256(packages).hexdigest(),
        },
        "python_import_roots": [
            _import_root_payload(value)
            for value in bind_import_roots(args.python_path.absolute(), environment, checkout)
        ],
        "environment": dict(environment),
        "pretrained_checkpoint": _external_file_payload(bind_external_file(pretrained_path)),
        "partition": partition,
        "partition_lines": _partition_line_count(partition_path, partition),
        "image_root_link": _bind_image_root_link(args.dataset_root),
        "image_tree": _merkle_payload(bind_merkle(resolved_image_root)),
    }


def _frozen_argv(runtime: Mapping[str, object]) -> list[str]:
    return [
        str(runtime["python"]["path"]),  # type: ignore[index]
        "-m",
        "sfora.cli",
        "image-end-to-end",
        "--dataset-name",
        "inshop",
        "--dataset-root",
        DATASET_ROOT.as_posix(),
        "--objectives",
        "proxy_anchor",
        "--recipe",
        "auto",
        "--num-workers",
        "8",
        "--seed",
        "0",
        "--save-model-path",
        f"{RUN_DIRECTORY.as_posix()}/{OUTPUT_FILENAMES['checkpoint']}",
        "--output",
        f"{RUN_DIRECTORY.as_posix()}/{OUTPUT_FILENAMES['report']}",
    ]


def _run_freeze_capture_child(
    args: FreezeArgs,
    argv: list[str],
    environment: Mapping[str, str],
) -> CapturedAuthority:
    request = encode_capture_request(argv, args.dataset_root)
    try:
        result = subprocess.run(
            [
                str(args.python_path.absolute()),
                str(args.checkout_root / CONTROLLER_PATH),
                "capture-authority-child",
            ],
            cwd=args.checkout_root,
            env=dict(environment),
            input=request,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError("capture authority child failed to start") from exc
    _require(result.returncode == 0, "capture authority child failed")
    _require(not result.stderr, "capture authority child emitted stderr")
    return decode_capture_response(result.stdout)


def _validate_rfc3339_utc(value: str) -> None:
    _require(type(value) is str and RFC3339_UTC.fullmatch(value) is not None, "timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("invalid RFC3339 UTC timestamp") from exc
    _require(
        parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0,
        "timestamp",
    )


def _require_frozen_absence(args: FreezeArgs) -> None:
    checkout = args.checkout_root.resolve(strict=True)
    expected_output = checkout / PRELAUNCH_PATH
    _require(args.output_path.absolute() == expected_output, "alternative prelaunch output path")
    _require(not os.path.lexists(expected_output), "prelaunch output already exists")
    run_directory = checkout / RUN_DIRECTORY
    _require(not os.path.lexists(run_directory), "private run directory already exists")
    for filename in (*OUTPUT_FILENAMES.values(), "report.json.tmp"):
        _require(
            not os.path.lexists(run_directory / filename),
            f"frozen output already exists: {filename}",
        )


def _validate_freeze_capture(capture: CapturedAuthority) -> None:
    _require(capture.recipe_id == RECIPE_ID, "capture recipe ID drift")
    _require(capture.recipe_digest == RECIPE_DIGEST, "capture recipe digest drift")
    _require(capture.protocol == "query_gallery", "capture protocol drift")
    _require(capture.protocol_name == "deepfashion-inshop-official", "protocol name drift")
    _require(capture.train_count > 0, "empty training authority")
    _require(capture.query_count > 0 and capture.gallery_count > 0, "false query/gallery scope")
    _require(len(capture.rows) == capture.train_count, "optimization row count drift")
    _require(len({row[2] for row in capture.rows}) > 0, "empty identity authority")
    config = load_strict_json_bytes(capture.config_bytes)
    _require(canonical_json_bytes(config) == capture.config_bytes, "capture config encoding drift")
    _validate_operating_config(config)


def _build_prelaunch_payload(
    args: FreezeArgs,
    capture: CapturedAuthority,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    _validate_freeze_capture(capture)
    output_paths = {
        key: f"{RUN_DIRECTORY.as_posix()}/{filename}" for key, filename in OUTPUT_FILENAMES.items()
    }
    outputs = {key: {"path": path, "required_absent": True} for key, path in output_paths.items()}
    row_records = _capture_rows(capture)
    return {
        "schema_version": "pass201-pa-source-v2-prelaunch-v1",
        "status": "frozen",
        "purpose": "prospective ordinary Proxy Anchor seed-0 source authority",
        "source_commit": runtime["source_commit"],
        "authorization": {
            "manifest_path": PRELAUNCH_PATH.as_posix(),
            "required_parent_commit": runtime["source_commit"],
            "required_diff_paths": [PRELAUNCH_PATH.as_posix()],
            "required_diff_status": ["A"],
            "required_diff_modes": ["100644"],
            "clean_policy": "empty-porcelain-v1-z",
            "frozen_absence_checked_utc": args.frozen_absence_checked_utc,
            "frozen_absence": {key: "ENOENT" for key in ("run_directory", *OUTPUT_FILENAMES)},
        },
        "controller": runtime["controller"],
        "source": {
            "files": runtime["source_files"],
            "python_tree": runtime["python_tree"],
            "pyproject": runtime["pyproject"],
            "lockfile": runtime["lockfile"],
            "equivalence_test_id": (
                "tests/test_run_pass201_pa_source_v2.py::"
                "test_production_capture_uses_real_cli_boundary_without_training"
            ),
        },
        "execution": {
            "checkout_root": args.checkout_root.resolve(strict=True).as_posix(),
            "cwd": args.checkout_root.resolve(strict=True).as_posix(),
            "python": runtime["python"],
            "python_realpath": runtime["python_realpath"],
            "python_version": runtime["python_version"],
            "git": runtime["git"],
            "python_packages": runtime["python_packages"],
            "python_import_roots": runtime["python_import_roots"],
            "environment": runtime["environment"],
            "environment_policy": "replace",
            "argv": _frozen_argv(runtime),
            "objective": "proxy_anchor",
            "seed": 0,
            "expected_config_json": capture.config_bytes.decode("utf-8"),
            "expected_config_sha256": hashlib.sha256(capture.config_bytes).hexdigest(),
            "recipe_id": capture.recipe_id,
            "recipe_digest": capture.recipe_digest,
            "schedule": {
                "resolved_train_steps": capture.resolved_train_steps,
                "steps_per_epoch": capture.steps_per_epoch,
                "total_epochs": capture.total_epochs,
            },
            "pretrained_checkpoint": runtime["pretrained_checkpoint"],
        },
        "dataset": {
            "root": DATASET_ROOT.as_posix(),
            "partition": runtime["partition"],
            "partition_lines": runtime["partition_lines"],
            "bundle": {
                "train": capture.train_count,
                "query": capture.query_count,
                "gallery": capture.gallery_count,
                "protocol": capture.protocol,
                "protocol_name": capture.protocol_name,
            },
            "declared_image_root": f"{DATASET_ROOT.as_posix()}/Img/img",
            "resolved_image_root": f"{DATASET_ROOT.as_posix()}/img/img",
            "image_root_link": runtime["image_root_link"],
            "image_tree": runtime["image_tree"],
            "image_tree_leaf_base": "resolved_image_root",
            "image_tree_leaf_schema": "relative_path,size,sha256",
            "selection_policy": "full_official_partition",
            "optimization_authority": {
                "algorithm_id": "pass201-production-invocation-capture-v1",
                "row_count": len(row_records),
                "identity_count": len({row["label"] for row in row_records}),
                "ordered_row_sha256": _ordered_hash(row_records),
                "resolved_membership_sha256": capture.resolved_membership_sha256,
            },
        },
        "outputs": {
            "run_directory": RUN_DIRECTORY.as_posix(),
            "run_directory_required_absent": True,
            **outputs,
        },
        "sidecars": {
            "config_algorithm": "pass201-resolved-config-v2",
            "manifest_algorithm": "pass201-inshop-benchmark-row-suffix-v2",
            "schedule_algorithm": "pass201-inshop-completed-epoch-v1",
            "config_schema": "canonical-json-object-v1",
            "manifest_schema": "pass201-train-manifest-v1",
        },
        "postconditions": {
            "required_exit_code": 0,
            "require_source_equal": True,
            "require_partition_equal": True,
            "require_image_tree_equal": True,
            "require_two_process_sidecar_identity": True,
            "require_restricted_checkpoint_metadata": True,
            "require_complete_receipt": True,
        },
    }


def freeze_authority(args: FreezeArgs) -> bytes:
    _require(isinstance(args, FreezeArgs), "freeze arguments")
    _validate_rfc3339_utc(args.frozen_absence_checked_utc)
    _require(
        args.dataset_root.absolute() == DATASET_ROOT,
        "alternative dataset root",
    )
    _require_frozen_absence(args)
    environment = _build_replacement_environment(args)
    runtime = _bind_freeze_runtime(args, environment)
    argv = _frozen_argv(runtime)
    first = _run_freeze_capture_child(args, argv, environment)
    _require_frozen_absence(args)
    second = _run_freeze_capture_child(args, argv, environment)
    _require(first == second, "capture children disagree")
    _require_frozen_absence(args)
    payload = _build_prelaunch_payload(args, first, runtime)
    validate_prelaunch(payload)
    return canonical_json_bytes(payload)


def utc_now_rfc3339() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _ambient_environment() -> dict[str, str]:
    return dict(os.environ)


def _load_manifest_authority(
    manifest_path: Path,
) -> tuple[PrelaunchAuthority, Path, bytes]:
    exact_path = _exact_regular_path(manifest_path)
    manifest_bytes = _read_immutable_regular(exact_path)
    payload = load_strict_json_bytes(manifest_bytes)
    authority = validate_prelaunch(payload)
    _require(canonical_json_bytes(payload) == manifest_bytes, "manifest is not canonical")
    checkout = authority.checkout_root
    _require(
        checkout.is_absolute() and checkout.resolve(strict=True) == checkout.absolute(),
        "checkout root binding drift",
    )
    expected_path = checkout / authority.payload["authorization"]["manifest_path"]
    _require(exact_path == expected_path, "alternative manifest path")
    return authority, exact_path, manifest_bytes


def _require_authority_scope(authority: PrelaunchAuthority) -> None:
    execution = authority.payload["execution"]
    dataset = authority.payload["dataset"]
    bundle = dataset["bundle"]
    _require(
        type(bundle["query"]) is int
        and bundle["query"] > 0
        and type(bundle["gallery"]) is int
        and bundle["gallery"] > 0,
        "false query/gallery scope",
    )
    config = _validate_operating_config(load_strict_json_bytes(authority.expected_config_bytes))
    _require(config["recipe_id"] == execution["recipe_id"], "recipe ID authority drift")
    _require(
        config["recipe_digest"] == execution["recipe_digest"],
        "recipe digest authority drift",
    )


def _require_replacement_environment(authority: PrelaunchAuthority) -> None:
    expected = dict(authority.payload["execution"]["environment"])
    _require(_ambient_environment() == expected, "controller environment drift")


def _expected_runtime_bindings(authority: PrelaunchAuthority) -> dict[str, object]:
    payload = authority.payload
    execution = payload["execution"]
    source = payload["source"]
    dataset = payload["dataset"]
    return {
        "source_commit": authority.source_commit,
        "controller": _plain_json(payload["controller"]),
        "source_files": _plain_json(source["files"]),
        "python_tree": _plain_json(source["python_tree"]),
        "pyproject": _plain_json(source["pyproject"]),
        "lockfile": _plain_json(source["lockfile"]),
        "python": _plain_json(execution["python"]),
        "python_realpath": execution["python_realpath"],
        "python_version": execution["python_version"],
        "git": _plain_json(execution["git"]),
        "python_packages": _plain_json(execution["python_packages"]),
        "python_import_roots": _plain_json(execution["python_import_roots"]),
        "environment": _plain_json(execution["environment"]),
        "pretrained_checkpoint": _plain_json(execution["pretrained_checkpoint"]),
        "partition": _plain_json(dataset["partition"]),
        "partition_lines": dataset["partition_lines"],
        "image_root_link": _plain_json(dataset["image_root_link"]),
        "image_tree": _plain_json(dataset["image_tree"]),
    }


def _require_executing_python_identity(authority: PrelaunchAuthority) -> Path:
    executable = sys.executable
    _require(type(executable) is str and bool(executable), "executing controller Python path")
    actual = Path(executable)
    _require(actual.is_absolute(), "executing controller Python path")
    execution = authority.payload["execution"]
    _require(
        actual.as_posix() == execution["python"]["path"],
        "executing controller Python path drift",
    )
    resolved = actual.resolve(strict=True)
    _require(
        resolved.as_posix() == execution["python_realpath"],
        "executing controller Python realpath drift",
    )
    measured = _external_file_payload(bind_external_file(resolved))
    measured["path"] = actual.as_posix()
    _require(
        measured == _plain_json(execution["python"]),
        "executing controller Python file binding drift",
    )
    return actual


def _bind_runtime_after(authority: PrelaunchAuthority) -> dict[str, object]:
    execution = authority.payload["execution"]
    executing_python = _require_executing_python_identity(authority)
    args = FreezeArgs(
        checkout_root=authority.checkout_root,
        dataset_root=Path(authority.payload["dataset"]["root"]),
        python_path=executing_python,
        frozen_absence_checked_utc=authority.payload["authorization"]["frozen_absence_checked_utc"],
        output_path=authority.checkout_root / authority.payload["authorization"]["manifest_path"],
    )
    current = _bind_freeze_runtime(args, dict(execution["environment"]))
    current["source_commit"] = authority.source_commit
    return current


def _require_runtime_matches_authority(authority: PrelaunchAuthority, bindings: object) -> None:
    _require(
        type(bindings) is dict and bindings == _expected_runtime_bindings(authority),
        "runtime binding drift",
    )


def _record_preflight_absence(authority: PrelaunchAuthority) -> None:
    checkout = authority.checkout_root
    run_directory = checkout / authority.payload["outputs"]["run_directory"]
    try:
        parent_relative = run_directory.parent.relative_to(checkout)
    except ValueError as exc:
        raise ValueError("run directory escapes checkout") from exc
    current = checkout
    missing_parent = False
    for component in parent_relative.parts:
        current /= component
        if missing_parent or not os.path.lexists(current):
            missing_parent = True
            continue
        mode = os.lstat(current).st_mode
        _require(stat.S_ISDIR(mode) and not stat.S_ISLNK(mode), "run parent path drift")
    _require(not os.path.lexists(run_directory), "private run directory already exists")
    for key, filename in OUTPUT_FILENAMES.items():
        path = checkout / authority.payload["outputs"][key]["path"]
        _require(path.parent == run_directory, f"output escapes private run directory: {key}")
        _require(not os.path.lexists(path), f"output already exists: {filename}")
    _require(
        not os.path.lexists(run_directory / "report.json.tmp"),
        "report temporary already exists",
    )


def validate_runtime_preflight(manifest_path: Path) -> AuthorizedSource:
    authority, exact_path, manifest_bytes = _load_manifest_authority(manifest_path)
    authorization_commit = validate_authorization_topology(authority.checkout_root, authority)
    _require_authority_scope(authority)
    _require_replacement_environment(authority)
    runtime = _bind_runtime_after(authority)
    _require_runtime_matches_authority(authority, runtime)
    started_utc = utc_now_rfc3339()
    _record_preflight_absence(authority)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_git_blob = hashlib.sha1(
        b"blob " + str(len(manifest_bytes)).encode("ascii") + b"\0" + manifest_bytes
    ).hexdigest()
    return AuthorizedSource(
        authority,
        authorization_commit,
        exact_path,
        manifest_bytes,
        manifest_sha256,
        manifest_git_blob,
        runtime,
        started_utc,
    )


@contextlib.contextmanager
def create_and_lock_private_run_directory(
    authorized: AuthorizedSource,
):
    authority = authorized.authority
    run_directory = authority.checkout_root / authority.payload["outputs"]["run_directory"]
    try:
        run_directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        _require(
            run_directory.parent.resolve(strict=True) == run_directory.parent.absolute(),
            "run directory parent contains a symlink",
        )
        os.mkdir(run_directory, 0o700)
    except OSError as exc:
        raise ValueError("cannot create private run directory") from exc
    directory_fd = -1
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(run_directory, flags)
        opened = os.fstat(directory_fd)
        named = os.stat(run_directory, follow_symlinks=False)
        _require(
            stat.S_ISDIR(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino),
            "private run directory identity drift",
        )
        _require(stat.S_IMODE(opened.st_mode) == 0o700, "private run directory mode drift")
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("private run directory lock unavailable") from exc
        _require(not os.listdir(directory_fd), "private run directory is not empty")
        yield run_directory
    finally:
        if directory_fd >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(directory_fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(directory_fd)


def launch_once(authorized: AuthorizedSource, run_dir: Path) -> RunningChild:
    authority = authorized.authority
    execution = authority.payload["execution"]
    argv = list(execution["argv"])
    _require(
        argv and argv[0] == execution["python"]["path"],
        "training interpreter/argv drift",
    )
    log_path = authority.checkout_root / authority.payload["outputs"]["log"]["path"]
    _require(log_path.parent == run_dir, "training log escapes private run directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        log_fd = os.open(log_path, flags, 0o600)
    except OSError as exc:
        raise ValueError("training log already exists or cannot be created") from exc
    started_utc = utc_now_rfc3339()
    try:
        os.fchmod(log_fd, 0o600)
        process = subprocess.Popen(
            argv,
            cwd=authority.checkout_root,
            env=dict(execution["environment"]),
            stdin=subprocess.DEVNULL,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            shell=False,
            close_fds=True,
        )
    except OSError as exc:
        raise ValueError("ordinary PA child failed to start") from exc
    finally:
        os.close(log_fd)
    return RunningChild(process, started_utc)


def _complete_child(running: RunningChild) -> CompletedChild:
    returncode = running.process.wait()
    _require(returncode == 0, "ordinary PA child failed")
    return CompletedChild(
        pid=running.pid,
        started_utc=running.started_utc,
        ended_utc=utc_now_rfc3339(),
        returncode=0,
    )


def _require_pre_post_identity(authorized: AuthorizedSource) -> PostflightEvidence:
    _require_replacement_environment(authorized.authority)
    current = _bind_runtime_after(authorized.authority)
    _require_runtime_matches_authority(authorized.authority, current)
    _require(
        current == authorized.runtime_bindings,
        "postflight runtime binding differs from preflight",
    )
    return PostflightEvidence(current, utc_now_rfc3339())


def _require_run_entries(run_dir: Path, expected: set[str]) -> None:
    _require(
        run_dir.resolve(strict=True) == run_dir.absolute(),
        "private run directory path drift",
    )
    entries = list(os.scandir(run_dir))
    names = {entry.name for entry in entries}
    _require(len(entries) == len(names) and names == expected, "private run outputs drift")
    for entry in entries:
        _require(entry.is_file(follow_symlinks=False), f"non-regular run output: {entry.name}")


def _freeze_scientific_outputs(authorized: AuthorizedSource, run_dir: Path) -> ScientificOutputs:
    authority = authorized.authority
    _require_run_entries(run_dir, {"report.json", "checkpoint.pt", "training.log"})
    evidence: dict[str, OutputEvidence] = {}
    for key in ("report", "checkpoint", "log"):
        path = authority.checkout_root / authority.payload["outputs"][key]["path"]
        _require(path.parent == run_dir, f"{key} output escapes private run directory")
        evidence[key] = hash_open_regular(path)
    _require_run_entries(run_dir, {"report.json", "checkpoint.pt", "training.log"})
    return ScientificOutputs(
        evidence["report"],
        evidence["checkpoint"],
        evidence["log"],
    )


def _binding_matches_output(binding: ExternalFileBinding, evidence: OutputEvidence) -> bool:
    return (
        binding.path == Path(evidence.path)
        and binding.mode == evidence.mode
        and binding.byte_count == evidence.byte_count
        and binding.sha256 == evidence.sha256
    )


def _read_restricted_metadata(
    authorized: AuthorizedSource, scientific: ScientificOutputs
) -> BoundCheckpointMetadata:
    checkpoint_path = Path(scientific.checkpoint.path)
    result = _run_metadata_child(authorized.authority, checkpoint_path)
    _require(
        _binding_matches_output(result.binding, scientific.checkpoint),
        "restricted checkpoint binding differs from frozen output",
    )
    return result


def _run_sidecar_child(
    authorized: AuthorizedSource,
    scientific: ScientificOutputs,
    run_dir: Path,
) -> SidecarFrame:
    response = _run_private_child(
        authorized.authority,
        "run_pass201_pa_source_v2.py",
        (
            "derive-sidecars",
            "--manifest",
            str(authorized.manifest_path),
            "--report",
            str(scientific.report.path),
            "--checkpoint",
            str(scientific.checkpoint.path),
            "--output-dir",
            str(run_dir),
        ),
        b"",
    )
    return decode_sidecar_frame(response)


def _publish_sidecar_output(
    authorized: AuthorizedSource,
    run_dir: Path,
    output_name: str,
    data: bytes,
) -> OutputEvidence:
    _require(output_name in ("resolved_config", "train_manifest"), "sidecar output name")
    path = (
        authorized.authority.checkout_root
        / authorized.authority.payload["outputs"][output_name]["path"]
    )
    _require(path.parent == run_dir, "sidecar output escapes private run directory")
    return publish_new_file(path, data)


def _output_evidence_payload(evidence: OutputEvidence, checkout: Path) -> dict[str, object]:
    try:
        relative = Path(evidence.path).relative_to(checkout)
    except ValueError as exc:
        raise ValueError("output evidence escapes checkout") from exc
    return {
        "path": relative.as_posix(),
        "file_type": evidence.file_type,
        "mode": evidence.mode,
        "bytes": evidence.byte_count,
        "sha256": evidence.sha256,
    }


def _checkpoint_metadata_payload(metadata: CheckpointMetadata) -> dict[str, object]:
    arch = metadata.arch
    return {
        "literal_top_keys": list(metadata.top_keys),
        "artifact_selection": metadata.artifact_selection,
        "evaluation_model_source": metadata.evaluation_model_source,
        "arch": {
            "backbone_name": arch.backbone_name,
            "pretrained_weights": arch.pretrained_weights,
            "head_pooling": arch.head_pooling,
            "embedding_dimensions": arch.embedding_dimensions,
            "embedding_head_init": arch.embedding_head_init,
            "embedding_layer_norm": arch.embedding_layer_norm,
        },
        "training_step": metadata.training_step,
        "training_config_sha256": metadata.training_config_sha256,
        "state_dict_storage_materialized": metadata.state_dict_storage_materialized,
    }


def _require_unchanged_output(evidence: OutputEvidence) -> None:
    _require(
        hash_open_regular(Path(evidence.path)) == evidence,
        f"immutable output changed: {evidence.path}",
    )


def _build_complete_receipt(
    authorized: AuthorizedSource,
    process: CompletedChild,
    postflight: PostflightEvidence,
    scientific: ScientificOutputs,
    metadata: BoundCheckpointMetadata,
    frames: tuple[SidecarFrame, SidecarFrame],
    config_evidence: OutputEvidence,
    manifest_evidence: OutputEvidence,
) -> bytes:
    authority = authorized.authority
    _require(process.returncode == 0, "receipt process is not successful")
    _require_runtime_matches_authority(authority, dict(postflight.bindings))
    _require(
        dict(postflight.bindings) == authorized.runtime_bindings,
        "receipt postflight differs from preflight",
    )
    _require(
        _binding_matches_output(metadata.binding, scientific.checkpoint),
        "receipt checkpoint metadata binding drift",
    )
    first, second = frames
    config_bytes, manifest_bytes = validate_sidecar_identity(first, second)
    _require(
        config_evidence.byte_count == len(config_bytes)
        and config_evidence.sha256 == hashlib.sha256(config_bytes).hexdigest(),
        "published resolved config differs from sidecar",
    )
    _require(
        manifest_evidence.byte_count == len(manifest_bytes)
        and manifest_evidence.sha256 == hashlib.sha256(manifest_bytes).hexdigest(),
        "published train manifest differs from sidecar",
    )
    for evidence in (
        scientific.report,
        scientific.checkpoint,
        scientific.log,
        config_evidence,
        manifest_evidence,
    ):
        _require_unchanged_output(evidence)
    run_dir = authority.checkout_root / authority.payload["outputs"]["run_directory"]
    _require_run_entries(
        run_dir,
        {
            "report.json",
            "checkpoint.pt",
            "training.log",
            "resolved_config.json",
            "train_manifest.json",
        },
    )
    payload = _plain_json(authority.payload)
    measured_runtime = _plain_json(authorized.runtime_bindings)
    source = payload["source"]
    execution = payload["execution"]
    dataset = payload["dataset"]
    sidecars = payload["sidecars"]
    optimization = dataset["optimization_authority"]
    outputs = {
        "report": _output_evidence_payload(scientific.report, authority.checkout_root),
        "checkpoint": _output_evidence_payload(scientific.checkpoint, authority.checkout_root),
        "log": _output_evidence_payload(scientific.log, authority.checkout_root),
        "resolved_config": _output_evidence_payload(config_evidence, authority.checkout_root),
        "train_manifest": _output_evidence_payload(manifest_evidence, authority.checkout_root),
    }
    receipt = {
        "schema_version": "pass201-pa-source-v2-receipt-v1",
        "status": "complete",
        "candidate_values_computed": False,
        "authorization": {
            "authorization_commit": authorized.authorization_commit,
            "source_commit": authority.source_commit,
            "manifest_path": payload["authorization"]["manifest_path"],
            "manifest_bytes": len(authorized.manifest_bytes),
            "manifest_sha256": authorized.manifest_sha256,
            "manifest_git_blob": authorized.manifest_git_blob,
            "parent_verified": True,
            "single_addition_verified": True,
            "detached_head_verified": True,
            "clean_policy_verified": True,
        },
        "controller": {
            "file": measured_runtime["controller"],
            "python": measured_runtime["python"],
            "python_packages": measured_runtime["python_packages"],
            "source_tree": measured_runtime["python_tree"],
        },
        "command": {
            "cwd": execution["cwd"],
            "environment": execution["environment"],
            "argv": execution["argv"],
        },
        "preflight": {
            "started_utc": authorized.preflight_started_utc,
            "run_directory_absent": True,
            "source_tree": source["python_tree"],
            "partition": dataset["partition"],
            "image_tree": dataset["image_tree"],
            "pretrained_checkpoint": execution["pretrained_checkpoint"],
            "outputs_absent": {key: True for key in OUTPUT_FILENAMES},
        },
        "process": {
            "pid": process.pid,
            "started_utc": process.started_utc,
            "ended_utc": process.ended_utc,
            "exit_code": process.returncode,
        },
        "postflight": {
            "ended_utc": postflight.ended_utc,
            "source_tree": source["python_tree"],
            "partition": dataset["partition"],
            "image_tree": dataset["image_tree"],
            "pretrained_checkpoint": execution["pretrained_checkpoint"],
            "source_equal": True,
            "partition_equal": True,
            "image_tree_equal": True,
            "pretrained_checkpoint_equal": True,
        },
        "outputs": outputs,
        "checkpoint_metadata": _checkpoint_metadata_payload(metadata.metadata),
        "sidecar_derivation": {
            "config_algorithm": sidecars["config_algorithm"],
            "manifest_algorithm": sidecars["manifest_algorithm"],
            "schedule_algorithm": sidecars["schedule_algorithm"],
            "source_files": source["files"],
            "input_hashes": {
                "manifest": authorized.manifest_sha256,
                "source_tree": source["python_tree"]["root_sha256"],
                "partition": dataset["partition"]["sha256"],
                "image_tree": dataset["image_tree"]["root_sha256"],
                "pretrained_checkpoint": execution["pretrained_checkpoint"]["sha256"],
                "report": scientific.report.sha256,
                "checkpoint": scientific.checkpoint.sha256,
                "expected_config": authority.expected_config_sha256,
            },
            "child_processes": [
                {
                    "ordinal": index,
                    "pid": frame.pid,
                    "config_sha256": frame.config_sha256,
                    "manifest_sha256": frame.manifest_sha256,
                }
                for index, frame in enumerate(frames, start=1)
            ],
            "row_count": optimization["row_count"],
            "identity_count": optimization["identity_count"],
            "ordered_row_sha256": optimization["ordered_row_sha256"],
            "resolved_membership_count": optimization["row_count"],
            "resolved_membership_sha256": optimization["resolved_membership_sha256"],
            "membership_covered_by_preflight": True,
            "membership_covered_by_postflight": True,
        },
        "scope": {
            "ordinary_source_uses_official_query_gallery": True,
            "uses_pass201_operator_data": False,
            "pass201_candidate_paths_read": False,
            "authorized_action": "source_binding_only",
        },
    }
    validate_complete_receipt(receipt, authority)
    return canonical_json_bytes(receipt)


def _publish_complete_receipt(
    authorized: AuthorizedSource, run_dir: Path, data: bytes
) -> OutputEvidence:
    _require_run_entries(
        run_dir,
        {
            "report.json",
            "checkpoint.pt",
            "training.log",
            "resolved_config.json",
            "train_manifest.json",
        },
    )
    path = (
        authorized.authority.checkout_root
        / authorized.authority.payload["outputs"]["receipt"]["path"]
    )
    _require(path.parent == run_dir, "receipt escapes private run directory")
    return publish_new_file(path, data)


def publish_postflight(
    authorized: AuthorizedSource,
    process: CompletedChild,
    run_dir: Path,
) -> None:
    postflight = _require_pre_post_identity(authorized)
    scientific = _freeze_scientific_outputs(authorized, run_dir)
    metadata = _read_restricted_metadata(authorized, scientific)
    first = _run_sidecar_child(authorized, scientific, run_dir)
    second = _run_sidecar_child(authorized, scientific, run_dir)
    config_bytes, manifest_bytes = validate_sidecar_identity(first, second)
    config_evidence = _publish_sidecar_output(authorized, run_dir, "resolved_config", config_bytes)
    manifest_evidence = _publish_sidecar_output(
        authorized, run_dir, "train_manifest", manifest_bytes
    )
    receipt_bytes = _build_complete_receipt(
        authorized,
        process,
        postflight,
        scientific,
        metadata,
        (first, second),
        config_evidence,
        manifest_evidence,
    )
    _publish_complete_receipt(authorized, run_dir, receipt_bytes)


def run_authorized_source(manifest_path: Path) -> None:
    authorized = validate_runtime_preflight(manifest_path)
    with create_and_lock_private_run_directory(authorized) as run_dir:
        running = launch_once(authorized, run_dir)
        completed = _complete_child(running)
        publish_postflight(authorized, completed, run_dir)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_bound_image(path: Path, declared_root: Path, resolved_root: Path) -> dict[str, object]:
    _require(isinstance(path, Path), "optimization image must be a Path")
    lexical = Path(os.path.abspath(path))
    try:
        declared_relative = lexical.relative_to(declared_root)
    except ValueError as exc:
        raise ValueError("optimization image escapes declared image root") from exc
    current = declared_root
    for component in declared_relative.parts:
        current /= component
        _require(not stat.S_ISLNK(os.lstat(current).st_mode), "optimization image is a symlink")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("optimization image escapes resolved image root") from exc
    _require(relative == declared_relative, "optimization image path resolution drift")
    _require(relative.parts and ".." not in relative.parts, "invalid optimization image path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(resolved, flags)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), "optimization image is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    _require(
        _stat_identity(before) == _stat_identity(after),
        "optimization image changed during read",
    )
    data = b"".join(chunks)
    _require(len(data) == before.st_size, "optimization image size drift")
    return {
        "relative_path": relative.as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _ordered_hash(records: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for record in records:
        encoded = canonical_json_bytes(record)
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def _capture_boundary(
    captured: list[CapturedAuthority], dataset_root: Path
) -> Callable[..., NoReturn]:
    bound_root = dataset_root.resolve(strict=True)
    declared_image_root = bound_root / "Img" / "img"
    physical_image_root = (bound_root / "img" / "img").resolve(strict=True)
    _require((bound_root / "Img").is_symlink(), "In-Shop Img must be a symlink")
    _require(os.readlink(bound_root / "Img") == "img", "In-Shop Img symlink drift")
    _require(
        declared_image_root.resolve(strict=True) == physical_image_root,
        "In-Shop resolved image root drift",
    )

    def capture_impl(**kwargs: Any) -> NoReturn:
        _require(not captured, "production boundary count")
        _require(
            set(kwargs)
            == {
                "train_examples",
                "test_examples",
                "gallery_examples",
                "config",
                "progress_callback",
            },
            "production boundary arguments",
        )
        train = kwargs["train_examples"]
        query = kwargs["test_examples"]
        gallery = kwargs.get("gallery_examples")
        config = kwargs["config"]
        _require(type(train) is list and type(query) is list, "production bundle types")
        _require(type(gallery) is list, "official In-Shop gallery")
        _require(type(config) is ImageEndToEndConfig, "final ImageEndToEndConfig")
        _require(
            config.dataset_root is not None
            and Path(config.dataset_root).resolve(strict=True) == bound_root,
            "dataset root drift",
        )
        _require(config.dataset_name == "inshop", "dataset drift")
        _require(config.objectives == ("proxy_anchor",), "objective drift")
        _require(type(config.seed) is int and config.seed == 0, "seed drift")
        _require(config.recipe_id == RECIPE_ID, "recipe ID drift")
        _require(config.recipe_digest == RECIPE_DIGEST, "recipe digest drift")
        _require(config.proxy_count_per_class == 1, "proxy count drift")
        _require(config.backbone_name == "bn_inception", "backbone drift")
        _require(config.pretrained_weights == "bn_inception_52deb4733", "weights drift")
        _require(config.batch_size == 180, "batch size drift")
        _require(config.drop_last_train_batch is True, "drop-last drift")
        _require(config.train_epochs == 60, "epoch drift")
        _require(config.checkpoint_selection_interval == 0, "checkpoint split drift")
        _require(config.label_noise_fraction == 0.0, "label noise drift")
        _require(config.limit_per_class is None, "class limit drift")
        _require(config.max_classes is None, "class cap drift")
        _require(config.num_workers == 8, "worker count drift")
        _require(config.recipe_modified_fields == {}, "modified recipe drift")
        _require(
            config.dataset_selection_policy == "full_official_partition",
            "selection policy drift",
        )
        fraction = (
            config.checkpoint_selection_validation_fraction
            if config.checkpoint_selection_interval > 0
            else 0.0
        )
        optimization, _checkpoint = image_end_to_end._checkpoint_train_validation_split(
            train, fraction=fraction, seed=config.seed
        )
        optimization = image_end_to_end._apply_training_label_noise(
            optimization, fraction=config.label_noise_fraction, seed=config.seed
        )
        schedule = image_end_to_end._resolve_training_schedule(
            config,
            optimization_example_count=len(optimization),
            optimization_labels=[int(example.label) for example in optimization],
        )
        _require(
            type(schedule) is tuple
            and len(schedule) == 3
            and all(type(value) is int and value > 0 for value in schedule),
            "schedule type drift",
        )
        membership = [
            _read_bound_image(example.image, declared_image_root, physical_image_root)
            for example in optimization
        ]
        rows = tuple(
            (index, example.example_id, int(example.label))
            for index, example in enumerate(optimization)
        )
        _require(
            all(type(example) is ImageExample for example in (*train, *query, *gallery)),
            "production example type drift",
        )
        captured.append(
            CapturedAuthority(
                canonical_json_bytes(config.model_dump(mode="json")),
                config.recipe_id,
                config.recipe_digest,
                len(train),
                len(query),
                len(gallery),
                "query_gallery",
                "deepfashion-inshop-official",
                rows,
                _ordered_hash(membership),
                schedule[0],
                schedule[1],
                schedule[2],
            )
        )
        raise _CaptureComplete

    def capture_then_raise(**kwargs: Any) -> NoReturn:
        try:
            capture_impl(**kwargs)
        except ValueError as exc:
            raise _CaptureRejected(str(exc)) from exc

    return capture_then_raise


def _invoke_real_typer(argv: Sequence[str], dataset_root: Path) -> None:
    _require(type(argv) in (list, tuple), "argv sequence type")
    values = list(argv)
    _require(
        len(values) == 20
        and values[1:7]
        == [
            "-m",
            "sfora.cli",
            "image-end-to-end",
            "--dataset-name",
            "inshop",
            "--dataset-root",
        ]
        and values[8:17]
        == [
            "--objectives",
            "proxy_anchor",
            "--recipe",
            "auto",
            "--num-workers",
            "8",
            "--seed",
            "0",
            "--save-model-path",
        ]
        and values[18] == "--output",
        "frozen argv drift",
    )
    _require(
        Path(values[7]).resolve(strict=True) == dataset_root.resolve(strict=True),
        "frozen dataset root drift",
    )
    _require(bool(values[0]) and bool(values[17]) and bool(values[19]), "frozen argv path drift")
    command = get_command(sfora.cli.app)
    with contextlib.redirect_stdout(io.StringIO()):
        command.main(args=values[3:], prog_name=values[0], standalone_mode=False)


def capture_authority(argv: Sequence[str], dataset_root: Path) -> CapturedAuthority:
    captured: list[CapturedAuthority] = []
    try:
        with patch.object(
            sfora.cli,
            "run_image_end_to_end_benchmark",
            _capture_boundary(captured, dataset_root),
        ):
            _invoke_real_typer(argv, dataset_root)
    except _CaptureComplete:
        pass
    except _CaptureRejected as exc:
        raise ValueError(str(exc)) from None
    _require(len(captured) == 1, "production boundary count")
    return captured[0]


def encode_capture_request(argv: Sequence[str], dataset_root: Path) -> bytes:
    _require(type(argv) in (list, tuple), "capture request argv type")
    values = list(argv)
    _require(all(type(value) is str for value in values), "capture request argument type")
    payload = canonical_json_bytes(
        {"argv": values, "dataset_root": dataset_root.absolute().as_posix()}
    )
    return encode_private_child_frame(PrivateChildFrame("capture-request", os.getpid(), payload))


def _decode_capture_request(data: bytes) -> tuple[list[str], Path]:
    frame = decode_private_child_frame(data)
    _require(frame.role == "capture-request", "capture child request role")
    payload = load_strict_json_bytes(frame.payload)
    _require(set(payload) == {"argv", "dataset_root"}, "capture request keys")
    argv = payload["argv"]
    dataset_root = payload["dataset_root"]
    _require(
        type(argv) is list and all(type(value) is str for value in argv),
        "capture request argv type",
    )
    _require(type(dataset_root) is str and Path(dataset_root).is_absolute(), "dataset root type")
    return argv, Path(dataset_root)


def _capture_payload(capture: CapturedAuthority) -> bytes:
    return canonical_json_bytes(
        {
            "config_json": capture.config_bytes.decode("utf-8"),
            "recipe_id": capture.recipe_id,
            "recipe_digest": capture.recipe_digest,
            "train_count": capture.train_count,
            "query_count": capture.query_count,
            "gallery_count": capture.gallery_count,
            "protocol": capture.protocol,
            "protocol_name": capture.protocol_name,
            "rows": [list(row) for row in capture.rows],
            "resolved_membership_sha256": capture.resolved_membership_sha256,
            "resolved_train_steps": capture.resolved_train_steps,
            "steps_per_epoch": capture.steps_per_epoch,
            "total_epochs": capture.total_epochs,
        }
    )


def decode_capture_response(data: bytes) -> CapturedAuthority:
    frame = decode_private_child_frame(data)
    _require(frame.role == "capture-response", "capture child response role")
    payload = load_strict_json_bytes(frame.payload)
    expected_keys = {
        "config_json",
        "recipe_id",
        "recipe_digest",
        "train_count",
        "query_count",
        "gallery_count",
        "protocol",
        "protocol_name",
        "rows",
        "resolved_membership_sha256",
        "resolved_train_steps",
        "steps_per_epoch",
        "total_epochs",
    }
    _require(set(payload) == expected_keys, "capture response keys")
    config_text = payload["config_json"]
    _require(type(config_text) is str, "capture response config type")
    config_bytes = config_text.encode("utf-8")
    _require(
        canonical_json_bytes(load_strict_json_bytes(config_bytes)) == config_bytes,
        "capture response config is not canonical",
    )
    for key in (
        "recipe_id",
        "recipe_digest",
        "protocol",
        "protocol_name",
        "resolved_membership_sha256",
    ):
        _require(type(payload[key]) is str and bool(payload[key]), f"capture response {key} type")
    for key in (
        "train_count",
        "query_count",
        "gallery_count",
        "resolved_train_steps",
        "steps_per_epoch",
        "total_epochs",
    ):
        _require(type(payload[key]) is int and payload[key] >= 0, f"capture response {key} type")
    raw_rows = payload["rows"]
    _require(type(raw_rows) is list, "capture response rows type")
    rows: list[tuple[int, str, int]] = []
    for index, row in enumerate(raw_rows):
        _require(
            type(row) is list
            and len(row) == 3
            and type(row[0]) is int
            and row[0] == index
            and type(row[1]) is str
            and bool(row[1])
            and type(row[2]) is int,
            "capture response row type",
        )
        rows.append((row[0], row[1], row[2]))
    return CapturedAuthority(
        config_bytes,
        payload["recipe_id"],
        payload["recipe_digest"],
        payload["train_count"],
        payload["query_count"],
        payload["gallery_count"],
        payload["protocol"],
        payload["protocol_name"],
        tuple(rows),
        payload["resolved_membership_sha256"],
        payload["resolved_train_steps"],
        payload["steps_per_epoch"],
        payload["total_epochs"],
    )


def _capture_child_output() -> bytes:
    argv, dataset_root = _decode_capture_request(sys.stdin.buffer.read())
    capture = capture_authority(argv, dataset_root)
    return encode_private_child_frame(
        PrivateChildFrame("capture-response", os.getpid(), _capture_payload(capture))
    )


def _validate_operating_config(config: object) -> dict[str, Any]:
    _require(type(config) is dict, "resolved config must be an object")
    result = config
    _require(result.get("dataset_name") == REPORT_DATASET_NAME, "resolved dataset drift")
    _require(result.get("protocol") == REPORT_PROTOCOL, "resolved protocol drift")
    _require(result.get("objectives") == ["proxy_anchor"], "resolved objective drift")
    _require(type(result.get("seed")) is int and result["seed"] == 0, "resolved seed drift")
    _require(result.get("recipe_id") == RECIPE_ID, "resolved recipe ID drift")
    _require(result.get("recipe_digest") == RECIPE_DIGEST, "resolved recipe digest drift")
    return result


def _skip_json_whitespace(data: bytes, offset: int) -> int:
    while offset < len(data) and data[offset] in b" \t\r\n":
        offset += 1
    return offset


def _scan_json_string(data: bytes, offset: int) -> int:
    _require(offset < len(data) and data[offset] == ord('"'), "expected JSON string")
    offset += 1
    while offset < len(data):
        character = data[offset]
        if character == ord('"'):
            return offset + 1
        if character == ord("\\"):
            offset += 2
            continue
        _require(character >= 0x20, "control byte in JSON string")
        offset += 1
    raise ValueError("unterminated JSON string")


def _scan_opaque_json_value(data: bytes, offset: int) -> int:
    offset = _skip_json_whitespace(data, offset)
    _require(offset < len(data), "missing JSON value")
    if data[offset] == ord('"'):
        return _scan_json_string(data, offset)
    if data[offset] in b"{[":
        closers = [ord("}") if data[offset] == ord("{") else ord("]")]
        offset += 1
        while offset < len(data) and closers:
            character = data[offset]
            if character == ord('"'):
                offset = _scan_json_string(data, offset)
                continue
            if character in b"{[":
                closers.append(ord("}") if character == ord("{") else ord("]"))
            elif character in b"}]":
                _require(character == closers.pop(), "mismatched JSON container")
            offset += 1
        _require(not closers, "unterminated JSON container")
        return offset
    end = offset
    while end < len(data) and data[end] not in b",}":
        end += 1
    _require(bool(data[offset:end].strip()), "missing JSON scalar")
    return end


def _extract_report_config(report_bytes: bytes, authority: PrelaunchAuthority) -> object:
    _require(type(report_bytes) is bytes, "report bytes type")
    offset = _skip_json_whitespace(report_bytes, 0)
    _require(
        offset < len(report_bytes) and report_bytes[offset] == ord("{"),
        "report must be an object",
    )
    offset += 1
    seen_keys: set[str] = set()
    raw_values: dict[str, bytes] = {}
    while True:
        offset = _skip_json_whitespace(report_bytes, offset)
        _require(offset < len(report_bytes), "unterminated report object")
        if report_bytes[offset] == ord("}"):
            offset += 1
            break
        key_start = offset
        key_end = _scan_json_string(report_bytes, key_start)
        key = load_strict_json_value_bytes(report_bytes[key_start:key_end])
        _require(type(key) is str, "report key type")
        _require(key not in seen_keys, f"duplicate report key: {key}")
        seen_keys.add(key)
        offset = _skip_json_whitespace(report_bytes, key_end)
        _require(
            offset < len(report_bytes) and report_bytes[offset] == ord(":"),
            "report key missing colon",
        )
        value_start = _skip_json_whitespace(report_bytes, offset + 1)
        value_end = _scan_opaque_json_value(report_bytes, value_start)
        if key == "methods":
            _require(
                report_bytes[value_start] == ord("{") and report_bytes[value_end - 1] == ord("}"),
                "report methods must be an opaque object",
            )
        else:
            raw_values[key] = report_bytes[value_start:value_end]
        offset = _skip_json_whitespace(report_bytes, value_end)
        _require(offset < len(report_bytes), "unterminated report object")
        if report_bytes[offset] == ord(","):
            offset += 1
            continue
        _require(report_bytes[offset] == ord("}"), "invalid report member delimiter")
        offset += 1
        break
    _require(
        not report_bytes[_skip_json_whitespace(report_bytes, offset) :],
        "trailing report bytes",
    )
    _require(seen_keys == EXPECTED_REPORT_KEYS, "report keys drift")
    parsed = {key: load_strict_json_value_bytes(raw) for key, raw in raw_values.items()}
    for key in ("name", "dataset_name", "protocol"):
        _require(type(parsed[key]) is str, f"report {key} type drift")
    for key in ("train_examples", "test_examples"):
        _require(type(parsed[key]) is int and parsed[key] >= 0, f"report {key} type drift")
    expected_config = _validate_operating_config(
        load_strict_json_bytes(authority.expected_config_bytes)
    )
    bundle = authority.payload["dataset"]["bundle"]
    expected = {
        "name": REPORT_NAME,
        "dataset_name": expected_config["dataset_name"],
        "protocol": expected_config["protocol"],
        "train_examples": bundle["train"],
        "test_examples": bundle["query"],
    }
    for key, value in expected.items():
        _require(parsed[key] == value, f"report {key} drift")
    return parsed["config"]


def derive_resolved_config(
    report_bytes: bytes,
    checkpoint: CheckpointMetadata,
    authority: PrelaunchAuthority,
) -> bytes:
    config = canonical_json_bytes(
        _validate_operating_config(_extract_report_config(report_bytes, authority))
    )
    _require(config == authority.expected_config_bytes, "report config drift")
    _require(
        hashlib.sha256(config).hexdigest() == checkpoint.training_config_sha256,
        "checkpoint config drift",
    )
    return config


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(child) for key, child in value.items()}
    if type(value) is tuple:
        return [_plain_json(child) for child in value]
    if type(value) is list:
        return [_plain_json(child) for child in value]
    return value


def _capture_rows(capture: CapturedAuthority) -> list[dict[str, object]]:
    return [
        {"sample_index": sample_index, "example_id": example_id, "label": label}
        for sample_index, example_id, label in capture.rows
    ]


def _validate_capture_authority(
    capture: CapturedAuthority, authority: PrelaunchAuthority
) -> list[dict[str, object]]:
    _require(type(capture.config_bytes) is bytes, "capture config type drift")
    for field, value in (
        ("recipe ID", capture.recipe_id),
        ("recipe digest", capture.recipe_digest),
        ("protocol", capture.protocol),
        ("protocol name", capture.protocol_name),
        ("membership", capture.resolved_membership_sha256),
    ):
        _require(type(value) is str and bool(value), f"capture {field} type drift")
    for field, value in (
        ("train count", capture.train_count),
        ("query count", capture.query_count),
        ("gallery count", capture.gallery_count),
        ("resolved training steps", capture.resolved_train_steps),
        ("steps per epoch", capture.steps_per_epoch),
        ("total epochs", capture.total_epochs),
    ):
        _require(type(value) is int and value >= 0, f"capture {field} type drift")
    _require(type(capture.rows) is tuple, "capture rows type drift")
    for index, row in enumerate(capture.rows):
        _require(
            type(row) is tuple
            and len(row) == 3
            and type(row[0]) is int
            and row[0] == index
            and type(row[1]) is str
            and bool(row[1])
            and type(row[2]) is int,
            "capture row type drift",
        )
    expected_dataset = authority.payload["dataset"]
    expected_bundle = expected_dataset["bundle"]
    expected_optimization = expected_dataset["optimization_authority"]
    expected_execution = authority.payload["execution"]
    _require(capture.config_bytes == authority.expected_config_bytes, "config drift")
    _require(capture.recipe_id == RECIPE_ID, "recipe ID drift")
    _require(capture.recipe_digest == RECIPE_DIGEST, "recipe digest drift")
    _require(
        capture.recipe_id == expected_execution["recipe_id"],
        "authority recipe ID drift",
    )
    _require(
        capture.recipe_digest == expected_execution["recipe_digest"],
        "authority recipe digest drift",
    )
    _require(capture.train_count == expected_bundle["train"], "train count drift")
    _require(capture.query_count == expected_bundle["query"], "query count drift")
    _require(capture.gallery_count == expected_bundle["gallery"], "gallery count drift")
    _require(capture.protocol == expected_bundle["protocol"], "protocol drift")
    _require(capture.protocol_name == expected_bundle["protocol_name"], "protocol name drift")
    rows = _capture_rows(capture)
    _require(len(rows) == expected_optimization["row_count"], "row count drift")
    _require(
        len({row["label"] for row in rows}) == expected_optimization["identity_count"],
        "identity count drift",
    )
    _require(
        _ordered_hash(rows) == expected_optimization["ordered_row_sha256"],
        "ordered row drift",
    )
    _require(
        capture.resolved_membership_sha256 == expected_optimization["resolved_membership_sha256"],
        "resolved membership drift",
    )
    _require(capture.resolved_train_steps == authority.expected_train_steps, "schedule drift")
    _require(capture.steps_per_epoch == authority.steps_per_epoch, "schedule drift")
    _require(capture.total_epochs == authority.total_epochs, "schedule drift")
    return rows


def derive_train_manifest(capture: CapturedAuthority, authority: PrelaunchAuthority) -> bytes:
    rows = _validate_capture_authority(capture, authority)
    dataset = authority.payload["dataset"]
    files = sorted(
        (_plain_json(value) for value in authority.payload["source"]["files"]),
        key=lambda value: value["path"].encode("utf-8"),
    )
    payload = {
        "schema_version": "pass201-train-manifest-v1",
        "algorithm_id": "pass201-inshop-benchmark-row-suffix-v2",
        "source_commit": authority.source_commit,
        "dataset_authority": {
            "root": dataset["root"],
            "partition_sha256": dataset["partition"]["sha256"],
            "resolved_image_root": dataset["resolved_image_root"],
            "image_tree_sha256": dataset["image_tree"]["root_sha256"],
            "bundle": _plain_json(dataset["bundle"]),
            "selection_policy": dataset["selection_policy"],
        },
        "rows": rows,
        "derivation": {
            "call_graph": list(TRAIN_MANIFEST_CALL_GRAPH),
            "source_files": files,
            "resolved_config_sha256": authority.expected_config_sha256,
            "row_count": len(rows),
            "identity_count": len({row["label"] for row in rows}),
            "ordered_row_sha256": _ordered_hash(rows),
            "resolved_membership_count": len(rows),
            "resolved_membership_sha256": capture.resolved_membership_sha256,
        },
    }
    validate_train_manifest(payload, authority)
    return canonical_json_bytes(payload)


def validate_completed_epoch(
    capture: CapturedAuthority,
    checkpoint_step: int,
    optimization_count: int,
    batch_size: int,
) -> int:
    _require(type(checkpoint_step) is int and checkpoint_step > 0, "invalid checkpoint step")
    _require(type(optimization_count) is int and optimization_count > 0, "invalid row count")
    _require(type(batch_size) is int and batch_size > 0, "invalid batch size")
    _require(
        type(capture.resolved_train_steps) is int and capture.resolved_train_steps > 0,
        "invalid resolved training steps",
    )
    _require(
        type(capture.steps_per_epoch) is int and capture.steps_per_epoch > 0,
        "invalid steps per epoch",
    )
    _require(
        type(capture.total_epochs) is int and capture.total_epochs > 0,
        "invalid total epochs",
    )
    _require(
        capture.steps_per_epoch == max(1, optimization_count // batch_size),
        "drop-last schedule drift",
    )
    _require(checkpoint_step == capture.resolved_train_steps, "training step drift")
    _require(checkpoint_step % capture.steps_per_epoch == 0, "partial epoch")
    completed = checkpoint_step // capture.steps_per_epoch
    _require(completed == capture.total_epochs, "completed epoch drift")
    return completed


def encode_sidecar_frame(frame: SidecarFrame) -> bytes:
    _require(type(frame.pid) is int and frame.pid > 0, "invalid child PID")
    _require(type(frame.config_bytes) is bytes, "invalid config bytes")
    _require(type(frame.manifest_bytes) is bytes, "invalid manifest bytes")
    config_hash = hashlib.sha256(frame.config_bytes).hexdigest()
    manifest_hash = hashlib.sha256(frame.manifest_bytes).hexdigest()
    _require(frame.config_sha256 == config_hash, "config frame hash drift")
    _require(frame.manifest_sha256 == manifest_hash, "manifest frame hash drift")
    return b"".join(
        (
            SIDECAR_FRAME_MAGIC,
            struct.pack(">Q", frame.pid),
            struct.pack(">Q", len(frame.config_bytes)),
            frame.config_bytes,
            bytes.fromhex(frame.config_sha256),
            struct.pack(">Q", len(frame.manifest_bytes)),
            frame.manifest_bytes,
            bytes.fromhex(frame.manifest_sha256),
        )
    )


def decode_sidecar_frame(data: bytes) -> SidecarFrame:
    _require(type(data) is bytes and data.startswith(SIDECAR_FRAME_MAGIC), "sidecar frame magic")
    offset = len(SIDECAR_FRAME_MAGIC)

    def take(count: int) -> bytes:
        nonlocal offset
        _require(count >= 0 and count <= len(data) - offset, "truncated sidecar frame")
        result = data[offset : offset + count]
        offset += count
        return result

    pid = struct.unpack(">Q", take(8))[0]
    config_size = struct.unpack(">Q", take(8))[0]
    config = take(config_size)
    config_hash = take(32).hex()
    manifest_size = struct.unpack(">Q", take(8))[0]
    manifest = take(manifest_size)
    manifest_hash = take(32).hex()
    _require(offset == len(data), "trailing sidecar frame bytes")
    frame = SidecarFrame(pid, config, manifest, config_hash, manifest_hash)
    _require(encode_sidecar_frame(frame) == data, "noncanonical sidecar frame")
    return frame


def validate_sidecar_identity(first: SidecarFrame, second: SidecarFrame) -> tuple[bytes, bytes]:
    encode_sidecar_frame(first)
    encode_sidecar_frame(second)
    _require(first.pid != second.pid, "sidecar child PIDs must be distinct")
    _require(first.config_bytes == second.config_bytes, "sidecar config identity drift")
    _require(first.manifest_bytes == second.manifest_bytes, "sidecar manifest identity drift")
    _require(first.config_sha256 == second.config_sha256, "sidecar config hash drift")
    _require(first.manifest_sha256 == second.manifest_sha256, "sidecar manifest hash drift")
    return first.config_bytes, first.manifest_bytes


def _read_immutable_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode), "sidecar input is not regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    _require(
        _stat_identity(before) == _stat_identity(after),
        "sidecar input changed during read",
    )
    data = b"".join(chunks)
    _require(len(data) == before.st_size, "sidecar input size drift")
    return data


def _exact_existing_path(path: Path, *, directory: bool) -> Path:
    candidate = Path(os.path.abspath(path))
    try:
        mode = os.lstat(candidate).st_mode
    except OSError as exc:
        raise ValueError(f"sidecar path does not exist: {candidate}") from exc
    if stat.S_ISLNK(mode):
        raise ValueError(f"sidecar path must not be a symlink: {candidate}")
    expected_kind = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
    _require(expected_kind, f"invalid sidecar path type: {candidate}")
    resolved = candidate.resolve(strict=True)
    _require(resolved == candidate, f"sidecar path contains a symlink: {candidate}")
    return candidate


def _exact_regular_path(path: Path) -> Path:
    return _exact_existing_path(path, directory=False)


def _validate_bound_environment(authority: PrelaunchAuthority) -> None:
    expected = dict(authority.payload["execution"]["environment"])
    _require(dict(os.environ) == expected, "sidecar environment drift")


def _run_private_child(
    authority: PrelaunchAuthority,
    script_name: str,
    command: Sequence[str],
    request: bytes,
) -> bytes:
    execution = authority.payload["execution"]
    interpreter = str(execution["python"]["path"])
    script = authority.checkout_root / "scripts" / script_name
    try:
        result = subprocess.run(
            [interpreter, str(script), *command],
            cwd=authority.checkout_root,
            env=dict(execution["environment"]),
            input=request,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"{command[0]} failed to start") from exc
    _require(result.returncode == 0, f"{command[0]} failed")
    _require(not result.stderr, f"{command[0]} emitted stderr")
    return result.stdout


def _run_restricted_checkpoint_child(
    authority: PrelaunchAuthority,
    command: Sequence[str],
    request: bytes,
) -> bytes:
    execution = authority.payload["execution"]
    interpreter = str(execution["python"]["path"])
    script = authority.checkout_root / "scripts" / "pass201_pa_source_v2_contract.py"
    try:
        result = subprocess.run(
            [interpreter, "-I", "-S", str(script), *command],
            cwd=authority.checkout_root,
            env=dict(execution["environment"]),
            input=request,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ValueError(f"{command[0]} failed to start") from exc
    _require(result.returncode == 0, f"{command[0]} failed")
    _require(not result.stderr, f"{command[0]} emitted stderr")
    return result.stdout


def _run_capture_child(authority: PrelaunchAuthority) -> CapturedAuthority:
    request = encode_capture_request(
        tuple(authority.payload["execution"]["argv"]),
        Path(authority.payload["dataset"]["root"]),
    )
    response = _run_private_child(
        authority, "run_pass201_pa_source_v2.py", ("capture-authority-child",), request
    )
    return decode_capture_response(response)


def _run_metadata_child(
    authority: PrelaunchAuthority, checkpoint_path: Path
) -> BoundCheckpointMetadata:
    response = _run_restricted_checkpoint_child(
        authority,
        ("restricted-metadata-child", "--checkpoint", str(checkpoint_path)),
        encode_checkpoint_metadata_request(authority),
    )
    return decode_checkpoint_metadata_response(response, authority, checkpoint_path)


def _run_binding_child(
    authority: PrelaunchAuthority,
    checkpoint_path: Path,
    expected: ExternalFileBinding,
) -> ExternalFileBinding:
    response = _run_restricted_checkpoint_child(
        authority,
        ("restricted-binding-child", "--checkpoint", str(checkpoint_path)),
        encode_checkpoint_binding_request(expected),
    )
    return decode_checkpoint_binding_response(response, expected, checkpoint_path)


def derive_sidecars_from_files(
    manifest_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> SidecarFrame:
    manifest_path = _exact_regular_path(manifest_path)
    report_path = _exact_regular_path(report_path)
    checkpoint_path = _exact_regular_path(checkpoint_path)
    output_dir = _exact_existing_path(output_dir, directory=True)
    manifest_bytes = _read_immutable_regular(manifest_path)
    authority = validate_prelaunch(load_strict_json_bytes(manifest_bytes))
    _validate_bound_environment(authority)
    expected_manifest = (
        authority.checkout_root / authority.payload["authorization"]["manifest_path"]
    ).resolve(strict=True)
    expected_output_dir = (
        authority.checkout_root / authority.payload["outputs"]["run_directory"]
    ).resolve(strict=True)
    _require(manifest_path == expected_manifest, "alternative manifest path")
    _require(output_dir == expected_output_dir, "alternative output directory")
    expected_report = (
        authority.checkout_root / authority.payload["outputs"]["report"]["path"]
    ).resolve(strict=True)
    expected_checkpoint = (
        authority.checkout_root / authority.payload["outputs"]["checkpoint"]["path"]
    ).resolve(strict=True)
    _require(
        report_path == expected_report,
        "alternative report path",
    )
    _require(
        checkpoint_path == expected_checkpoint,
        "alternative checkpoint path",
    )
    report_bytes = _read_immutable_regular(report_path)
    capture = _run_capture_child(authority)
    _validate_capture_authority(capture, authority)
    bound_checkpoint = _run_metadata_child(authority, checkpoint_path)
    checkpoint = bound_checkpoint.metadata
    expected_config = load_strict_json_bytes(authority.expected_config_bytes)
    _require(expected_config.get("drop_last_train_batch") is True, "drop-last config drift")
    batch_size = expected_config.get("batch_size")
    _require(type(batch_size) is int, "batch size config drift")
    validate_completed_epoch(capture, checkpoint.training_step, len(capture.rows), batch_size)
    config = derive_resolved_config(report_bytes, checkpoint, authority)
    manifest = derive_train_manifest(capture, authority)
    frame = SidecarFrame(
        os.getpid(),
        config,
        manifest,
        hashlib.sha256(config).hexdigest(),
        hashlib.sha256(manifest).hexdigest(),
    )
    _require(_read_immutable_regular(manifest_path) == manifest_bytes, "manifest input drift")
    _require(_read_immutable_regular(report_path) == report_bytes, "report input drift")
    _run_binding_child(authority, checkpoint_path, bound_checkpoint.binding)
    return frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_pass201_pa_source_v2.py")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-authority")
    freeze.add_argument("--frozen-absence-checked-utc", required=True)
    freeze.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    sidecars = commands.add_parser("derive-sidecars")
    sidecars.add_argument("--manifest", required=True, type=Path)
    sidecars.add_argument("--report", required=True, type=Path)
    sidecars.add_argument("--checkpoint", required=True, type=Path)
    sidecars.add_argument("--output-dir", required=True, type=Path)
    commands.add_parser("capture-authority-child")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "freeze-authority":
        checkout = Path.cwd().resolve(strict=True)
        output = args.output if args.output.is_absolute() else checkout / args.output
        frozen = freeze_authority(
            FreezeArgs(
                checkout_root=checkout,
                dataset_root=DATASET_ROOT,
                python_path=Path(sys.executable).absolute(),
                frozen_absence_checked_utc=args.frozen_absence_checked_utc,
                output_path=output.absolute(),
            )
        )
        publish_new_file(output.absolute(), frozen, mode=0o644)
        return 0
    if args.command == "run":
        manifest = args.manifest if args.manifest.is_absolute() else Path.cwd() / args.manifest
        run_authorized_source(manifest.absolute())
        return 0
    if args.command == "capture-authority-child":
        sys.stdout.buffer.write(_capture_child_output())
        sys.stdout.buffer.flush()
        return 0
    _require(args.command == "derive-sidecars", "command drift")
    frame = derive_sidecars_from_files(
        args.manifest,
        args.report,
        args.checkpoint,
        args.output_dir,
    )
    sys.stdout.buffer.write(encode_sidecar_frame(frame))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
