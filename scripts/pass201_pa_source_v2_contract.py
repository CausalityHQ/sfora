"""Strict, descriptor-safe authority contracts for the Pass201 source-v2 run."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

SHA256 = frozenset("0123456789abcdef")
ENVIRONMENT_KEYS = frozenset(
    (
        "HOME",
        "PATH",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "LC_ALL",
        "LANG",
        "TZ",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
    )
)


@dataclass(frozen=True)
class ExternalFileBinding:
    path: Path
    mode: int
    device: int
    inode: int
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class RepoBlob:
    path: PurePosixPath
    git_mode: Literal["100644", "100755"]
    byte_count: int
    sha256: str
    git_blob: str


@dataclass(frozen=True)
class MerkleBinding:
    root: Path
    algorithm: Literal["pass201-length-framed-merkle-v1"]
    count: int
    byte_count: int
    root_sha256: str


@dataclass(frozen=True)
class ImportTreeBinding:
    algorithm: Literal["pass201-import-tree-v1"]
    regular_count: int
    symlink_count: int
    byte_count: int
    root_sha256: str


@dataclass(frozen=True)
class ExternalFileImportTarget:
    link_relative_path: PurePosixPath
    target_text: str
    resolved_path: Path
    kind: Literal["file"]
    file: ExternalFileBinding


@dataclass(frozen=True)
class ImportDirectoryBinding:
    root: Path
    tree: ImportTreeBinding
    external_symlink_targets: tuple[ExternalFileImportTarget | ExternalDirectoryImportTarget, ...]


@dataclass(frozen=True)
class ExternalDirectoryImportTarget:
    link_relative_path: PurePosixPath
    target_text: str
    resolved_path: Path
    kind: Literal["directory"]
    directory: ImportDirectoryBinding


@dataclass(frozen=True)
class NonexistentImportRoot:
    entry: str
    status: Literal["nonexistent"] = "nonexistent"


@dataclass(frozen=True)
class ZipImportRoot:
    entry: str
    file: ExternalFileBinding
    status: Literal["zip"] = "zip"


@dataclass(frozen=True)
class DirectoryImportRoot:
    entry: str
    directory: ImportDirectoryBinding
    status: Literal["directory"] = "directory"


ImportRootBinding: TypeAlias = (  # noqa: UP040 - required public annotation form
    NonexistentImportRoot | ZipImportRoot | DirectoryImportRoot
)


@dataclass(frozen=True)
class PrelaunchAuthority:
    payload: Mapping[str, Any]
    source_commit: str
    checkout_root: Path
    expected_config_bytes: bytes
    expected_config_sha256: str
    expected_train_steps: int
    steps_per_epoch: int
    total_epochs: int


@dataclass(frozen=True)
class OutputEvidence:
    path: PurePosixPath
    file_type: Literal["regular"]
    mode: int
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class CompleteReceipt:
    payload: Mapping[str, Any]
    authorization_commit: str
    output_evidence: Mapping[str, OutputEvidence]


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_json_native(value: Any, where: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{where}: non-finite number")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_json_native(child, f"{where}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise ValueError(f"{where}: object key is not a string")
            _validate_json_native(child, f"{where}.{key}")
        return
    raise ValueError(f"{where}: not an exact JSON-native value")


def canonical_json_bytes(value: Any) -> bytes:
    _validate_json_native(value)
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def load_strict_json_bytes(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON") from exc
    _validate_json_native(value)
    return _dict(value, "$")


def _dict(value: object, where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{where}: expected object")
    return value  # type: ignore[return-value]


def _keys(value: object, expected: set[str], where: str) -> dict[str, Any]:
    result = _dict(value, where)
    if set(result) != expected:
        raise ValueError(f"{where}: keys must be exactly {sorted(expected)}")
    return result


def _str(value: object, where: str, literal: str | None = None) -> str:
    if type(value) is not str or not value or (literal is not None and value != literal):
        raise ValueError(f"{where}: invalid string")
    return value


def _int(value: object, where: str, *, minimum: int = 0, literal: int | None = None) -> int:
    if type(value) is not int or value < minimum or (literal is not None and value != literal):
        raise ValueError(f"{where}: invalid integer")
    return value


def _true(value: object, where: str) -> None:
    if value is not True:
        raise ValueError(f"{where}: must be true")


def _literal(value: object, literal: object, where: str) -> None:
    if type(value) is not type(literal) or value != literal:
        raise ValueError(f"{where}: must equal {literal!r}")


def _hash(value: object, where: str, length: int = 64) -> str:
    result = _str(value, where)
    if len(result) != length or any(char not in SHA256 for char in result):
        raise ValueError(f"{where}: invalid lowercase hash")
    return result


def _repo_path(value: object, where: str) -> str:
    result = _str(value, where)
    path = PurePosixPath(result)
    if (
        path.is_absolute()
        or result != path.as_posix()
        or any(p in ("", ".", "..") for p in path.parts)
    ):
        raise ValueError(f"{where}: invalid repository path")
    return result


def _abs_path(value: object, where: str) -> str:
    result = _str(value, where)
    path = PurePosixPath(result)
    if (
        not path.is_absolute()
        or result != path.as_posix()
        or any(p in ("", ".", "..") for p in path.parts[1:])
    ):
        raise ValueError(f"{where}: invalid absolute path")
    return result


def _list(value: object, where: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{where}: expected array")
    return value  # type: ignore[return-value]


def _file(value: object, where: str) -> None:
    obj = _keys(value, {"path", "git_mode", "bytes", "sha256", "git_blob"}, where)
    _repo_path(obj["path"], f"{where}.path")
    if obj["git_mode"] not in ("100644", "100755"):
        raise ValueError(f"{where}.git_mode")
    _int(obj["bytes"], f"{where}.bytes")
    _hash(obj["sha256"], f"{where}.sha256")
    _hash(obj["git_blob"], f"{where}.git_blob", 40)


def _external(value: object, where: str) -> None:
    obj = _keys(value, {"path", "mode", "device", "inode", "bytes", "sha256"}, where)
    _abs_path(obj["path"], f"{where}.path")
    for key in ("mode", "device", "inode", "bytes"):
        _int(obj[key], f"{where}.{key}")
    _hash(obj["sha256"], f"{where}.sha256")


def _merkle(value: object, where: str) -> None:
    obj = _keys(value, {"root", "algorithm", "count", "bytes", "root_sha256"}, where)
    (_abs_path if str(obj["root"]).startswith("/") else _repo_path)(obj["root"], f"{where}.root")
    _literal(obj["algorithm"], "pass201-length-framed-merkle-v1", f"{where}.algorithm")
    _int(obj["count"], f"{where}.count")
    _int(obj["bytes"], f"{where}.bytes")
    _hash(obj["root_sha256"], f"{where}.root_sha256")


def _import_root(value: object, where: str) -> None:
    obj = _dict(value, where)
    status = obj.get("status")
    if status == "nonexistent":
        obj = _keys(obj, {"entry", "status"}, where)
        _str(obj["entry"], f"{where}.entry")
    elif status == "zip":
        obj = _keys(obj, {"entry", "status", "file"}, where)
        _str(obj["entry"], f"{where}.entry")
        _external(obj["file"], f"{where}.file")
    elif status == "directory":
        obj = _keys(obj, {"entry", "status", "directory"}, where)
        _str(obj["entry"], f"{where}.entry")
        _import_directory(obj["directory"], f"{where}.directory")
    else:
        raise ValueError(f"{where}.status")


def _import_directory(value: object, where: str) -> None:
    obj = _keys(value, {"root", "tree", "external_symlink_targets"}, where)
    _abs_path(obj["root"], f"{where}.root")
    tree = _keys(
        obj["tree"],
        {"algorithm", "regular_count", "symlink_count", "bytes", "root_sha256"},
        f"{where}.tree",
    )
    _literal(tree["algorithm"], "pass201-import-tree-v1", f"{where}.tree.algorithm")
    for key in ("regular_count", "symlink_count", "bytes"):
        _int(tree[key], f"{where}.tree.{key}")
    _hash(tree["root_sha256"], f"{where}.tree.root_sha256")
    for index, target in enumerate(
        _list(obj["external_symlink_targets"], f"{where}.external_symlink_targets")
    ):
        base = {"link_relative_path", "target_text", "resolved_path", "kind"}
        target_obj = _dict(target, f"{where}.external_symlink_targets[{index}]")
        kind = target_obj.get("kind")
        nested = "file" if kind == "file" else "directory" if kind == "directory" else None
        if nested is None:
            raise ValueError(f"{where}.external_symlink_targets[{index}].kind")
        target_obj = _keys(
            target_obj, base | {nested}, f"{where}.external_symlink_targets[{index}]"
        )
        _repo_path(target_obj["link_relative_path"], f"{where}.link_relative_path")
        _str(target_obj["target_text"], f"{where}.target_text")
        _abs_path(target_obj["resolved_path"], f"{where}.resolved_path")
        (_external if kind == "file" else _import_directory)(
            target_obj[nested], f"{where}.{nested}"
        )


def validate_prelaunch(payload: object) -> PrelaunchAuthority:
    _validate_json_native(payload)
    top = _keys(
        payload,
        {
            "schema_version",
            "status",
            "purpose",
            "source_commit",
            "authorization",
            "controller",
            "source",
            "execution",
            "dataset",
            "outputs",
            "sidecars",
            "postconditions",
        },
        "prelaunch",
    )
    _literal(top["schema_version"], "pass201-pa-source-v2-prelaunch-v1", "schema_version")
    _literal(top["status"], "frozen", "status")
    _str(top["purpose"], "purpose")
    source_commit = _hash(top["source_commit"], "source_commit", 40)
    auth = _keys(
        top["authorization"],
        {
            "manifest_path",
            "required_parent_commit",
            "required_diff_paths",
            "required_diff_status",
            "required_diff_modes",
            "clean_policy",
            "frozen_absence_checked_utc",
            "frozen_absence",
        },
        "authorization",
    )
    _literal(
        auth["manifest_path"],
        "docs/pass201_pa_source_v2_prelaunch.json",
        "authorization.manifest_path",
    )
    parent = _hash(auth["required_parent_commit"], "authorization.required_parent_commit", 40)
    if parent != source_commit:
        raise ValueError("authorization.required_parent_commit must equal source_commit")
    for key, expected in (
        ("required_diff_paths", ["docs/pass201_pa_source_v2_prelaunch.json"]),
        ("required_diff_status", ["A"]),
        ("required_diff_modes", ["100644"]),
    ):
        _literal(auth[key], expected, f"authorization.{key}")
    _literal(auth["clean_policy"], "empty-porcelain-v1-z", "authorization.clean_policy")
    _str(auth["frozen_absence_checked_utc"], "authorization.frozen_absence_checked_utc")
    absence = _keys(
        auth["frozen_absence"],
        {
            "run_directory",
            "report",
            "checkpoint",
            "log",
            "resolved_config",
            "train_manifest",
            "receipt",
        },
        "authorization.frozen_absence",
    )
    for key, value in absence.items():
        _literal(value, "ENOENT", f"authorization.frozen_absence.{key}")
    _file(top["controller"], "controller")
    source = _keys(
        top["source"],
        {"files", "python_tree", "pyproject", "lockfile", "equivalence_test_id"},
        "source",
    )
    files = _list(source["files"], "source.files")
    if not files:
        raise ValueError("source.files: empty")
    for index, value in enumerate(files):
        _file(value, f"source.files[{index}]")
    _merkle(source["python_tree"], "source.python_tree")
    _file(source["pyproject"], "source.pyproject")
    _file(source["lockfile"], "source.lockfile")
    _str(source["equivalence_test_id"], "source.equivalence_test_id")
    execution = _keys(
        top["execution"],
        {
            "checkout_root",
            "cwd",
            "python",
            "python_realpath",
            "python_version",
            "git",
            "python_packages",
            "python_import_roots",
            "environment",
            "environment_policy",
            "argv",
            "objective",
            "seed",
            "expected_config_json",
            "expected_config_sha256",
            "recipe_id",
            "recipe_digest",
            "schedule",
            "pretrained_checkpoint",
        },
        "execution",
    )
    checkout = _abs_path(execution["checkout_root"], "execution.checkout_root")
    cwd = _abs_path(execution["cwd"], "execution.cwd")
    if cwd != checkout:
        raise ValueError("execution.cwd must equal checkout_root")
    _external(execution["python"], "execution.python")
    _abs_path(execution["python_realpath"], "execution.python_realpath")
    _str(execution["python_version"], "execution.python_version")
    _external(execution["git"], "execution.git")
    packages = _keys(execution["python_packages"], {"bytes", "sha256"}, "execution.python_packages")
    _int(packages["bytes"], "execution.python_packages.bytes")
    _hash(packages["sha256"], "execution.python_packages.sha256")
    for i, item in enumerate(
        _list(execution["python_import_roots"], "execution.python_import_roots")
    ):
        _import_root(item, f"execution.python_import_roots[{i}]")
    environment = _keys(execution["environment"], set(ENVIRONMENT_KEYS), "execution.environment")
    for key, value in environment.items():
        _str(value, f"execution.environment.{key}")
    _literal(environment["PYTHONNOUSERSITE"], "1", "execution.environment.PYTHONNOUSERSITE")
    _literal(
        environment["PYTHONDONTWRITEBYTECODE"],
        "1",
        "execution.environment.PYTHONDONTWRITEBYTECODE",
    )
    _literal(environment["PYTHONPATH"], f"{checkout}/src", "execution.environment.PYTHONPATH")
    python_directory = str(PurePosixPath(execution["python"]["path"]).parent)
    _literal(
        environment["PATH"],
        f"{python_directory}:/usr/bin:/bin",
        "execution.environment.PATH",
    )
    _literal(execution["environment_policy"], "replace", "execution.environment_policy")
    argv = _list(execution["argv"], "execution.argv")
    if not argv:
        raise ValueError("execution.argv")
    for i, arg in enumerate(argv):
        _str(arg, f"execution.argv[{i}]")
    _literal(execution["objective"], "proxy_anchor", "execution.objective")
    _int(execution["seed"], "execution.seed", literal=0)
    config_text = _str(execution["expected_config_json"], "execution.expected_config_json")
    config_bytes = config_text.encode("utf-8")
    config_obj = load_strict_json_bytes(config_bytes)
    if canonical_json_bytes(config_obj) != config_bytes:
        raise ValueError("execution.expected_config_json: not canonical")
    config_sha = _hash(execution["expected_config_sha256"], "execution.expected_config_sha256")
    if hashlib.sha256(config_bytes).hexdigest() != config_sha:
        raise ValueError("execution.expected_config_sha256: mismatch")
    _str(execution["recipe_id"], "execution.recipe_id")
    _hash(execution["recipe_digest"], "execution.recipe_digest")
    schedule = _keys(
        execution["schedule"],
        {"resolved_train_steps", "steps_per_epoch", "total_epochs"},
        "execution.schedule",
    )
    steps = _int(
        schedule["resolved_train_steps"], "execution.schedule.resolved_train_steps", minimum=1
    )
    spe = _int(schedule["steps_per_epoch"], "execution.schedule.steps_per_epoch", minimum=1)
    epochs = _int(schedule["total_epochs"], "execution.schedule.total_epochs", minimum=1)
    if steps != spe * epochs:
        raise ValueError("execution.schedule.resolved_train_steps must equal epoch schedule")
    _external(execution["pretrained_checkpoint"], "execution.pretrained_checkpoint")
    _validate_dataset(top["dataset"])
    _validate_outputs(top["outputs"])
    _validate_sidecars(top["sidecars"])
    _validate_postconditions(top["postconditions"])
    frozen = MappingProxyType(copy.deepcopy(top))
    return PrelaunchAuthority(
        frozen, source_commit, Path(checkout), config_bytes, config_sha, steps, spe, epochs
    )


def _validate_dataset(value: object) -> None:
    obj = _keys(
        value,
        {
            "root",
            "partition",
            "partition_lines",
            "bundle",
            "declared_image_root",
            "resolved_image_root",
            "image_root_link",
            "image_tree",
            "image_tree_leaf_base",
            "image_tree_leaf_schema",
            "selection_policy",
            "optimization_authority",
        },
        "dataset",
    )
    _literal(obj["root"], "/home/riomus/datasets/inshop_official_standard", "dataset.root")
    _external(obj["partition"], "dataset.partition")
    _int(obj["partition_lines"], "dataset.partition_lines")
    bundle = _keys(
        obj["bundle"], {"train", "query", "gallery", "protocol", "protocol_name"}, "dataset.bundle"
    )
    for k in ("train", "query", "gallery"):
        _int(bundle[k], f"dataset.bundle.{k}")
    _literal(bundle["protocol"], "query_gallery", "dataset.bundle.protocol")
    _str(bundle["protocol_name"], "dataset.bundle.protocol_name")
    _literal(
        obj["declared_image_root"],
        "/home/riomus/datasets/inshop_official_standard/Img/img",
        "dataset.declared_image_root",
    )
    _literal(
        obj["resolved_image_root"],
        "/home/riomus/datasets/inshop_official_standard/img/img",
        "dataset.resolved_image_root",
    )
    link = _keys(
        obj["image_root_link"], {"path", "target", "lstat_mode"}, "dataset.image_root_link"
    )
    _literal(
        link["path"],
        "/home/riomus/datasets/inshop_official_standard/Img",
        "dataset.image_root_link.path",
    )
    _literal(link["target"], "img", "dataset.image_root_link.target")
    _int(link["lstat_mode"], "dataset.image_root_link.lstat_mode")
    _merkle(obj["image_tree"], "dataset.image_tree")
    if obj["image_tree"]["root"] != obj["resolved_image_root"]:
        raise ValueError("dataset.image_tree.root must equal resolved_image_root")
    _literal(obj["image_tree_leaf_base"], "resolved_image_root", "dataset.image_tree_leaf_base")
    _literal(
        obj["image_tree_leaf_schema"], "relative_path,size,sha256", "dataset.image_tree_leaf_schema"
    )
    _literal(obj["selection_policy"], "full_official_partition", "dataset.selection_policy")
    opt = _keys(
        obj["optimization_authority"],
        {
            "algorithm_id",
            "row_count",
            "identity_count",
            "ordered_row_sha256",
            "resolved_membership_sha256",
        },
        "dataset.optimization_authority",
    )
    _literal(
        opt["algorithm_id"],
        "pass201-production-invocation-capture-v1",
        "dataset.optimization_authority.algorithm_id",
    )
    _int(opt["row_count"], "dataset.optimization_authority.row_count")
    _int(opt["identity_count"], "dataset.optimization_authority.identity_count")
    _hash(opt["ordered_row_sha256"], "dataset.optimization_authority.ordered_row_sha256")
    _hash(
        opt["resolved_membership_sha256"],
        "dataset.optimization_authority.resolved_membership_sha256",
    )


def _validate_outputs(value: object) -> None:
    keys = {
        "run_directory",
        "run_directory_required_absent",
        "report",
        "checkpoint",
        "log",
        "resolved_config",
        "train_manifest",
        "receipt",
    }
    obj = _keys(value, keys, "outputs")
    run_directory = _repo_path(obj["run_directory"], "outputs.run_directory")
    _literal(
        run_directory,
        "reports/generated/pass201_source_v2/run-v2",
        "outputs.run_directory",
    )
    _true(obj["run_directory_required_absent"], "outputs.run_directory_required_absent")
    names = {
        "report": "report.json",
        "checkpoint": "checkpoint.pt",
        "log": "training.log",
        "resolved_config": "resolved_config.json",
        "train_manifest": "train_manifest.json",
        "receipt": "receipt.json",
    }
    for key in keys - {"run_directory", "run_directory_required_absent"}:
        item = _keys(obj[key], {"path", "required_absent"}, f"outputs.{key}")
        _literal(item["path"], f"{run_directory}/{names[key]}", f"outputs.{key}.path")
        _true(item["required_absent"], f"outputs.{key}.required_absent")


def _validate_sidecars(value: object) -> None:
    expected = {
        "config_algorithm": "pass201-resolved-config-v2",
        "manifest_algorithm": "pass201-inshop-benchmark-row-suffix-v2",
        "schedule_algorithm": "pass201-inshop-completed-epoch-v1",
        "config_schema": "canonical-json-object-v1",
        "manifest_schema": "pass201-train-manifest-v1",
    }
    obj = _keys(value, set(expected), "sidecars")
    for key, literal in expected.items():
        _literal(obj[key], literal, f"sidecars.{key}")


def _validate_postconditions(value: object) -> None:
    bools = {
        "require_source_equal",
        "require_partition_equal",
        "require_image_tree_equal",
        "require_two_process_sidecar_identity",
        "require_restricted_checkpoint_metadata",
        "require_complete_receipt",
    }
    obj = _keys(value, bools | {"required_exit_code"}, "postconditions")
    _int(obj["required_exit_code"], "postconditions.required_exit_code", literal=0)
    for key in bools:
        _true(obj[key], f"postconditions.{key}")


def _read_regular_file(path: Path) -> tuple[os.stat_result, bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"cannot safely open regular file: {path}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(fd)

        def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
            )

        if identity(before) != identity(after) or sum(map(len, chunks)) != before.st_size:
            raise ValueError(f"file changed during read: {path}")
        return before, b"".join(chunks), digest.hexdigest()
    finally:
        os.close(fd)


def bind_external_file(path: Path) -> ExternalFileBinding:
    absolute = path.absolute()
    before, data, digest = _read_regular_file(absolute)
    return ExternalFileBinding(
        absolute, before.st_mode, before.st_dev, before.st_ino, len(data), digest
    )


def _frame(data: bytes) -> bytes:
    return len(data).to_bytes(8, "big") + data


def _reduce_merkle(hashes: list[bytes]) -> bytes:
    if not hashes:
        return hashlib.sha256(b"empty\0").digest()
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [
            hashlib.sha256(b"node\0" + hashes[i] + hashes[i + 1]).digest()
            for i in range(0, len(hashes), 2)
        ]
    return hashes[0]


def bind_merkle(root: Path) -> MerkleBinding:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"not a directory: {root}")
    leaves: list[tuple[bytes, int, dict[str, Any]]] = []
    pending = [resolved]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                name_bytes = entry.name.encode("utf-8")
                relative = entry.path[len(str(resolved)) + 1 :]
                relative_bytes = relative.encode("utf-8")
                if entry.is_symlink():
                    raise ValueError(f"symlink rejected: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"unsupported file type: {entry.path}")
                _, data, digest = _read_regular_file(Path(entry.path))
                leaves.append(
                    (
                        relative_bytes,
                        len(data),
                        {"relative_path": relative, "bytes": len(data), "sha256": digest},
                    )
                )
                del name_bytes
    leaves.sort(key=lambda item: item[0])
    if len({item[0] for item in leaves}) != len(leaves):
        raise ValueError("duplicate normalized names")
    hashes = [
        hashlib.sha256(b"leaf\0" + _frame(canonical_json_bytes(item[2]))).digest()
        for item in leaves
    ]
    return MerkleBinding(
        resolved,
        "pass201-length-framed-merkle-v1",
        len(leaves),
        sum(item[1] for item in leaves),
        _reduce_merkle(hashes).hex(),
    )


def _bind_import_directory(root: Path, active_realpaths: frozenset[Path]) -> ImportDirectoryBinding:
    resolved = root.resolve(strict=True)
    if resolved in active_realpaths:
        raise ValueError(f"external symlink cycle at {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"not an import directory: {root}")
    active = active_realpaths | {resolved}
    leaves: list[tuple[bytes, str, int, dict[str, Any]]] = []
    targets: list[ExternalFileImportTarget | ExternalDirectoryImportTarget] = []
    pending = [resolved]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative_text = entry.path[len(str(resolved)) + 1 :]
                relative = PurePosixPath(relative_text)
                relative_bytes = relative_text.encode("utf-8")
                before = os.lstat(entry.path)
                if stat.S_ISLNK(before.st_mode):
                    target_text = os.readlink(entry.path)
                    after = os.lstat(entry.path)
                    if (
                        before.st_dev,
                        before.st_ino,
                        before.st_mode,
                        before.st_mtime_ns,
                        target_text,
                    ) != (
                        after.st_dev,
                        after.st_ino,
                        after.st_mode,
                        after.st_mtime_ns,
                        os.readlink(entry.path),
                    ):
                        raise ValueError(f"symlink changed during read: {entry.path}")
                    target = (Path(entry.path).parent / target_text).resolve(strict=True)
                    try:
                        target.relative_to(resolved)
                        scope = "internal"
                    except ValueError:
                        scope = "external"
                    payload = {
                        "kind": "symlink",
                        "relative_path": relative_text,
                        "target_text": target_text,
                        "resolved_path": target.as_posix(),
                        "resolved_scope": scope,
                    }
                    leaves.append((relative_bytes, "symlink", 0, payload))
                    if scope == "external":
                        target_st = target.stat()
                        if stat.S_ISREG(target_st.st_mode):
                            targets.append(
                                ExternalFileImportTarget(
                                    relative,
                                    target_text,
                                    target,
                                    "file",
                                    bind_external_file(target),
                                )
                            )
                        elif stat.S_ISDIR(target_st.st_mode):
                            targets.append(
                                ExternalDirectoryImportTarget(
                                    relative,
                                    target_text,
                                    target,
                                    "directory",
                                    _bind_import_directory(target, active),
                                )
                            )
                        else:
                            raise ValueError(f"unsupported external symlink target: {target}")
                elif stat.S_ISDIR(before.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(before.st_mode):
                    _, data, digest = _read_regular_file(Path(entry.path))
                    leaves.append(
                        (
                            relative_bytes,
                            "file",
                            len(data),
                            {
                                "kind": "file",
                                "relative_path": relative_text,
                                "bytes": len(data),
                                "sha256": digest,
                            },
                        )
                    )
                else:
                    raise ValueError(f"unsupported import entry: {entry.path}")
    leaves.sort(key=lambda item: (item[0], item[1]))
    if len({item[0] for item in leaves}) != len(leaves):
        raise ValueError("duplicate normalized names")
    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(_frame(canonical_json_bytes(leaf[3])))
    tree = ImportTreeBinding(
        "pass201-import-tree-v1",
        sum(x[1] == "file" for x in leaves),
        sum(x[1] == "symlink" for x in leaves),
        sum(x[2] for x in leaves),
        digest.hexdigest(),
    )
    targets.sort(key=lambda target: target.link_relative_path.as_posix().encode("utf-8"))
    return ImportDirectoryBinding(resolved, tree, tuple(targets))


def bind_import_roots(
    interpreter: Path, env: Mapping[str, str], checkout: Path
) -> tuple[ImportRootBinding, ...]:
    program = "import json,sys;print(json.dumps(sys.path,separators=(',',':'),ensure_ascii=False))"
    result = subprocess.run(
        [str(interpreter), "-c", program],
        cwd=checkout,
        env=dict(env),
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("interpreter emitted invalid sys.path") from exc
    if type(entries) is not list or any(type(entry) is not str for entry in entries):
        raise ValueError("interpreter emitted invalid sys.path")
    checkout_resolved = checkout.resolve(strict=True)
    excluded = {checkout_resolved, (checkout_resolved / "src").resolve(strict=False)}
    bindings: list[ImportRootBinding] = []
    for entry in entries:
        candidate = checkout_resolved if entry == "" else Path(entry)
        if not candidate.is_absolute():
            candidate = checkout_resolved / candidate
        absolute = candidate.absolute()
        resolved = absolute.resolve(strict=False)
        if resolved in excluded:
            continue
        if not absolute.exists():
            bindings.append(NonexistentImportRoot(entry))
            continue
        mode = os.lstat(absolute).st_mode
        if stat.S_ISDIR(mode):
            bindings.append(
                DirectoryImportRoot(entry, _bind_import_directory(absolute, frozenset()))
            )
        elif stat.S_ISREG(mode) and zipfile.is_zipfile(absolute):
            bindings.append(ZipImportRoot(entry, bind_external_file(absolute)))
        else:
            raise ValueError(f"unsupported import root: {entry}")
    return tuple(bindings)


def _git(repo: Path, *args: str, text: bool = False) -> bytes | str:
    git_path = shutil.which("git")
    if git_path is None:
        raise ValueError("Git executable not found")
    try:
        completed = subprocess.run(
            [str(Path(git_path).resolve(strict=True)), *args],
            cwd=repo,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Git command failed: {' '.join(args)}") from exc
    return completed.stdout


def bind_repo_blob(repo: Path, revision: str, path: PurePosixPath) -> RepoBlob:
    normalized = _repo_path(path.as_posix(), "repository blob path")
    _hash(revision, "revision", 40)
    raw = _git(repo, "ls-tree", "-z", revision, "--", normalized)
    assert isinstance(raw, bytes)
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1:
        raise ValueError(f"repository blob does not exist uniquely: {normalized}")
    try:
        metadata, raw_path = records[0].split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split(" ")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid Git tree record") from exc
    if raw_path.decode("utf-8") != normalized or kind != "blob" or mode not in ("100644", "100755"):
        raise ValueError("invalid repository blob")
    data = _git(repo, "cat-file", "blob", oid)
    assert isinstance(data, bytes)
    return RepoBlob(
        PurePosixPath(normalized), mode, len(data), hashlib.sha256(data).hexdigest(), oid
    )  # type: ignore[arg-type]


def validate_authorization_topology(repo: Path, authority: PrelaunchAuthority) -> str:
    repo = repo.resolve(strict=True)
    git_path = shutil.which("git")
    if git_path is None:
        raise ValueError("Git executable not found")
    bound_git = bind_external_file(Path(git_path).resolve(strict=True))
    expected_git = authority.payload["execution"]["git"]
    if (
        str(bound_git.path),
        bound_git.mode,
        bound_git.device,
        bound_git.inode,
        bound_git.byte_count,
        bound_git.sha256,
    ) != (
        expected_git["path"],
        expected_git["mode"],
        expected_git["device"],
        expected_git["inode"],
        expected_git["bytes"],
        expected_git["sha256"],
    ):
        raise ValueError("Git executable binding mismatch")
    head_raw = _git(repo, "rev-parse", "HEAD", text=True)
    assert isinstance(head_raw, str)
    head = head_raw.strip()
    _hash(head, "HEAD", 40)
    symbolic = subprocess.run(
        [str(Path(git_path).resolve(strict=True)), "symbolic-ref", "-q", "HEAD"],
        cwd=repo,
        stdout=subprocess.PIPE,
    )
    if symbolic.returncode not in (0, 1):
        raise ValueError("cannot establish detached HEAD state")
    if symbolic.returncode == 0:
        raise ValueError("HEAD must be detached")
    parents_raw = _git(repo, "rev-list", "--parents", "-n", "1", head, text=True)
    assert isinstance(parents_raw, str)
    if parents_raw.strip().split() != [head, authority.source_commit]:
        raise ValueError("authorization must have exactly source_commit as parent")
    manifest_path = authority.payload["authorization"]["manifest_path"]
    status_raw = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        authority.source_commit,
        head,
        text=True,
    )
    assert isinstance(status_raw, str)
    if status_raw.splitlines() != [f"A\t{manifest_path}"]:
        raise ValueError("authorization must contain one manifest addition")
    blob = bind_repo_blob(repo, head, PurePosixPath(manifest_path))
    if blob.git_mode != "100644":
        raise ValueError("authorization manifest must be mode 100644")
    expected = canonical_json_bytes(dict(authority.payload))
    actual = _git(repo, "show", f"{head}:{manifest_path}")
    assert isinstance(actual, bytes)
    if actual != expected:
        raise ValueError("authorization manifest bytes differ from validated payload")
    status = _git(repo, "status", "--porcelain=v1", "-z")
    assert isinstance(status, bytes)
    if status:
        raise ValueError("worktree is not clean")
    return head


def _output_evidence(value: object, where: str) -> OutputEvidence:
    obj = _keys(value, {"path", "file_type", "mode", "bytes", "sha256"}, where)
    path = _repo_path(obj["path"], f"{where}.path")
    _literal(obj["file_type"], "regular", f"{where}.file_type")
    mode = _int(obj["mode"], f"{where}.mode")
    size = _int(obj["bytes"], f"{where}.bytes")
    digest = _hash(obj["sha256"], f"{where}.sha256")
    return OutputEvidence(PurePosixPath(path), "regular", mode, size, digest)


def validate_complete_receipt(payload: object, authority: PrelaunchAuthority) -> CompleteReceipt:
    _validate_json_native(payload)
    top = _keys(
        payload,
        {
            "schema_version",
            "status",
            "candidate_values_computed",
            "authorization",
            "controller",
            "command",
            "preflight",
            "process",
            "postflight",
            "outputs",
            "checkpoint_metadata",
            "sidecar_derivation",
            "scope",
        },
        "receipt",
    )
    _literal(top["schema_version"], "pass201-pa-source-v2-receipt-v1", "receipt.schema_version")
    _literal(top["status"], "complete", "receipt.status")
    _literal(top["candidate_values_computed"], False, "receipt.candidate_values_computed")
    auth = _keys(
        top["authorization"],
        {
            "authorization_commit",
            "source_commit",
            "manifest_path",
            "manifest_bytes",
            "manifest_sha256",
            "manifest_git_blob",
            "parent_verified",
            "single_addition_verified",
            "detached_head_verified",
            "clean_policy_verified",
        },
        "receipt.authorization",
    )
    authorization_commit = _hash(
        auth["authorization_commit"], "receipt.authorization.authorization_commit", 40
    )
    _literal(auth["source_commit"], authority.source_commit, "receipt.authorization.source_commit")
    _literal(
        auth["manifest_path"],
        authority.payload["authorization"]["manifest_path"],
        "receipt.authorization.manifest_path",
    )
    _int(auth["manifest_bytes"], "receipt.authorization.manifest_bytes")
    _hash(auth["manifest_sha256"], "receipt.authorization.manifest_sha256")
    _hash(auth["manifest_git_blob"], "receipt.authorization.manifest_git_blob", 40)
    for key in (
        "parent_verified",
        "single_addition_verified",
        "detached_head_verified",
        "clean_policy_verified",
    ):
        _true(auth[key], f"receipt.authorization.{key}")
    controller = _keys(
        top["controller"],
        {"file", "python", "python_packages", "source_tree"},
        "receipt.controller",
    )
    _file(controller["file"], "receipt.controller.file")
    _external(controller["python"], "receipt.controller.python")
    packages = _keys(
        controller["python_packages"], {"bytes", "sha256"}, "receipt.controller.python_packages"
    )
    _int(packages["bytes"], "receipt.controller.python_packages.bytes")
    _hash(packages["sha256"], "receipt.controller.python_packages.sha256")
    _merkle(controller["source_tree"], "receipt.controller.source_tree")
    command = _keys(top["command"], {"cwd", "environment", "argv"}, "receipt.command")
    _abs_path(command["cwd"], "receipt.command.cwd")
    env = _keys(command["environment"], set(ENVIRONMENT_KEYS), "receipt.command.environment")
    for key, value in env.items():
        _str(value, f"receipt.command.environment.{key}")
    argv = _list(command["argv"], "receipt.command.argv")
    for i, value in enumerate(argv):
        _str(value, f"receipt.command.argv[{i}]")
    if command != {
        "cwd": authority.payload["execution"]["cwd"],
        "environment": authority.payload["execution"]["environment"],
        "argv": authority.payload["execution"]["argv"],
    }:
        raise ValueError("receipt.command does not equal authority")
    _validate_flights(top["preflight"], top["postflight"])
    process = _keys(
        top["process"], {"pid", "started_utc", "ended_utc", "exit_code"}, "receipt.process"
    )
    _int(process["pid"], "receipt.process.pid", minimum=1)
    _str(process["started_utc"], "receipt.process.started_utc")
    _str(process["ended_utc"], "receipt.process.ended_utc")
    _int(process["exit_code"], "receipt.process.exit_code", literal=0)
    outputs_obj = _keys(
        top["outputs"],
        {"report", "checkpoint", "log", "resolved_config", "train_manifest"},
        "receipt.outputs",
    )
    output_bindings = {
        key: _output_evidence(value, f"receipt.outputs.{key}") for key, value in outputs_obj.items()
    }
    _validate_checkpoint_metadata(top["checkpoint_metadata"], authority)
    _validate_sidecar_derivation(top["sidecar_derivation"], authority)
    _validate_receipt_authority_bindings(
        top,
        authority,
        output_bindings,
    )
    scope = _keys(
        top["scope"],
        {
            "ordinary_source_uses_official_query_gallery",
            "uses_pass201_operator_data",
            "pass201_candidate_paths_read",
            "authorized_action",
        },
        "receipt.scope",
    )
    _true(
        scope["ordinary_source_uses_official_query_gallery"],
        "receipt.scope.ordinary_source_uses_official_query_gallery",
    )
    _literal(scope["uses_pass201_operator_data"], False, "receipt.scope.uses_pass201_operator_data")
    _literal(
        scope["pass201_candidate_paths_read"], False, "receipt.scope.pass201_candidate_paths_read"
    )
    _literal(scope["authorized_action"], "source_binding_only", "receipt.scope.authorized_action")
    return CompleteReceipt(
        MappingProxyType(copy.deepcopy(top)),
        authorization_commit,
        MappingProxyType(output_bindings),
    )


def _validate_receipt_authority_bindings(
    receipt: dict[str, Any],
    authority: PrelaunchAuthority,
    outputs: dict[str, OutputEvidence],
) -> None:
    expected = authority.payload
    controller = receipt["controller"]
    for receipt_key, expected_value in (
        ("file", expected["controller"]),
        ("python", expected["execution"]["python"]),
        ("python_packages", expected["execution"]["python_packages"]),
        ("source_tree", expected["source"]["python_tree"]),
    ):
        if controller[receipt_key] != expected_value:
            raise ValueError(f"receipt.controller.{receipt_key} does not equal authority")
    for phase in ("preflight", "postflight"):
        flight = receipt[phase]
        for receipt_key, expected_value in (
            ("source_tree", expected["source"]["python_tree"]),
            ("partition", expected["dataset"]["partition"]),
            ("image_tree", expected["dataset"]["image_tree"]),
            ("pretrained_checkpoint", expected["execution"]["pretrained_checkpoint"]),
        ):
            if flight[receipt_key] != expected_value:
                raise ValueError(f"receipt.{phase}.{receipt_key} does not equal authority")
    for key, evidence in outputs.items():
        if evidence.path.as_posix() != expected["outputs"][key]["path"]:
            raise ValueError(f"receipt.outputs.{key}.path does not equal authority")
    sidecar = receipt["sidecar_derivation"]
    if sidecar["source_files"] != expected["source"]["files"]:
        raise ValueError("receipt.sidecar_derivation.source_files does not equal authority")
    hashes = sidecar["input_hashes"]
    hash_expectations = {
        "manifest": receipt["authorization"]["manifest_sha256"],
        "source_tree": expected["source"]["python_tree"]["root_sha256"],
        "partition": expected["dataset"]["partition"]["sha256"],
        "image_tree": expected["dataset"]["image_tree"]["root_sha256"],
        "pretrained_checkpoint": expected["execution"]["pretrained_checkpoint"]["sha256"],
        "report": outputs["report"].sha256,
        "checkpoint": outputs["checkpoint"].sha256,
        "expected_config": authority.expected_config_sha256,
    }
    if hashes != hash_expectations:
        raise ValueError("receipt.sidecar_derivation.input_hashes do not bind receipt inputs")
    children = sidecar["child_processes"]
    if children[0]["config_sha256"] != outputs["resolved_config"].sha256:
        raise ValueError("receipt child config hash does not bind resolved_config")
    if children[0]["manifest_sha256"] != outputs["train_manifest"].sha256:
        raise ValueError("receipt child manifest hash does not bind train_manifest")
    optimization = expected["dataset"]["optimization_authority"]
    for receipt_key, authority_key in (
        ("row_count", "row_count"),
        ("identity_count", "identity_count"),
        ("ordered_row_sha256", "ordered_row_sha256"),
        ("resolved_membership_sha256", "resolved_membership_sha256"),
    ):
        if sidecar[receipt_key] != optimization[authority_key]:
            raise ValueError(f"receipt.sidecar_derivation.{receipt_key} does not equal authority")


def _validate_flights(pre_value: object, post_value: object) -> None:
    pre = _keys(
        pre_value,
        {
            "started_utc",
            "run_directory_absent",
            "source_tree",
            "partition",
            "image_tree",
            "pretrained_checkpoint",
            "outputs_absent",
        },
        "receipt.preflight",
    )
    _str(pre["started_utc"], "receipt.preflight.started_utc")
    _true(pre["run_directory_absent"], "receipt.preflight.run_directory_absent")
    for key, fn in (
        ("source_tree", _merkle),
        ("partition", _external),
        ("image_tree", _merkle),
        ("pretrained_checkpoint", _external),
    ):
        fn(pre[key], f"receipt.preflight.{key}")
    absent = _keys(
        pre["outputs_absent"],
        {"report", "checkpoint", "log", "resolved_config", "train_manifest", "receipt"},
        "receipt.preflight.outputs_absent",
    )
    for key in absent:
        _true(absent[key], f"receipt.preflight.outputs_absent.{key}")
    post = _keys(
        post_value,
        {
            "ended_utc",
            "source_tree",
            "partition",
            "image_tree",
            "pretrained_checkpoint",
            "source_equal",
            "partition_equal",
            "image_tree_equal",
            "pretrained_checkpoint_equal",
        },
        "receipt.postflight",
    )
    _str(post["ended_utc"], "receipt.postflight.ended_utc")
    for key, fn in (
        ("source_tree", _merkle),
        ("partition", _external),
        ("image_tree", _merkle),
        ("pretrained_checkpoint", _external),
    ):
        fn(post[key], f"receipt.postflight.{key}")
    for key in (
        "source_equal",
        "partition_equal",
        "image_tree_equal",
        "pretrained_checkpoint_equal",
    ):
        _true(post[key], f"receipt.postflight.{key}")


def _validate_checkpoint_metadata(value: object, authority: PrelaunchAuthority) -> None:
    obj = _keys(
        value,
        {
            "literal_top_keys",
            "artifact_selection",
            "evaluation_model_source",
            "arch",
            "training_step",
            "training_config_sha256",
            "state_dict_storage_materialized",
        },
        "receipt.checkpoint_metadata",
    )
    _literal(
        obj["literal_top_keys"],
        [
            "arch",
            "artifact_selection",
            "evaluation_model_source",
            "state_dict",
            "training_config",
            "training_step",
        ],
        "receipt.checkpoint_metadata.literal_top_keys",
    )
    _literal(
        obj["artifact_selection"],
        "final_training_state",
        "receipt.checkpoint_metadata.artifact_selection",
    )
    _literal(
        obj["evaluation_model_source"],
        "student",
        "receipt.checkpoint_metadata.evaluation_model_source",
    )
    arch_expected = {
        "backbone_name": "bn_inception",
        "pretrained_weights": "bn_inception_52deb4733",
        "head_pooling": "avg_max",
        "embedding_dimensions": 512,
        "embedding_head_init": "kaiming_normal",
        "embedding_layer_norm": False,
    }
    _literal(obj["arch"], arch_expected, "receipt.checkpoint_metadata.arch")
    _int(
        obj["training_step"],
        "receipt.checkpoint_metadata.training_step",
        literal=authority.expected_train_steps,
    )
    _literal(
        obj["training_config_sha256"],
        authority.expected_config_sha256,
        "receipt.checkpoint_metadata.training_config_sha256",
    )
    _literal(
        obj["state_dict_storage_materialized"],
        False,
        "receipt.checkpoint_metadata.state_dict_storage_materialized",
    )


def _validate_sidecar_derivation(value: object, authority: PrelaunchAuthority) -> None:
    keys = {
        "config_algorithm",
        "manifest_algorithm",
        "schedule_algorithm",
        "source_files",
        "input_hashes",
        "child_processes",
        "row_count",
        "identity_count",
        "ordered_row_sha256",
        "resolved_membership_count",
        "resolved_membership_sha256",
        "membership_covered_by_preflight",
        "membership_covered_by_postflight",
    }
    obj = _keys(value, keys, "receipt.sidecar_derivation")
    for key in ("config_algorithm", "manifest_algorithm", "schedule_algorithm"):
        _literal(obj[key], authority.payload["sidecars"][key], f"receipt.sidecar_derivation.{key}")
    files = _list(obj["source_files"], "receipt.sidecar_derivation.source_files")
    if not files:
        raise ValueError("receipt.sidecar_derivation.source_files")
    for i, value in enumerate(files):
        _file(value, f"receipt.sidecar_derivation.source_files[{i}]")
    paths = [value["path"] for value in files]
    if paths != sorted(paths, key=lambda p: p.encode("utf-8")) or len(paths) != len(set(paths)):
        raise ValueError("receipt.sidecar_derivation.source_files order")
    hashes = _keys(
        obj["input_hashes"],
        {
            "manifest",
            "source_tree",
            "partition",
            "image_tree",
            "pretrained_checkpoint",
            "report",
            "checkpoint",
            "expected_config",
        },
        "receipt.sidecar_derivation.input_hashes",
    )
    for key in hashes:
        _hash(hashes[key], f"receipt.sidecar_derivation.input_hashes.{key}")
    children = _list(obj["child_processes"], "receipt.sidecar_derivation.child_processes")
    if len(children) != 2:
        raise ValueError("receipt.sidecar_derivation.child_processes must have exactly two entries")
    parsed = []
    for index, value in enumerate(children):
        child = _keys(
            value,
            {"ordinal", "pid", "config_sha256", "manifest_sha256"},
            f"receipt.sidecar_derivation.child_processes[{index}]",
        )
        _int(child["ordinal"], f"child_processes[{index}].ordinal", literal=index + 1)
        pid = _int(child["pid"], f"child_processes[{index}].pid", minimum=1)
        config = _hash(child["config_sha256"], f"child_processes[{index}].config_sha256")
        manifest = _hash(child["manifest_sha256"], f"child_processes[{index}].manifest_sha256")
        parsed.append((pid, config, manifest))
    if parsed[0][0] == parsed[1][0]:
        raise ValueError("child_processes PIDs must be distinct")
    if parsed[0][1:] != parsed[1][1:]:
        raise ValueError("child_processes hashes must be identical")
    for key in ("row_count", "identity_count", "resolved_membership_count"):
        _int(obj[key], f"receipt.sidecar_derivation.{key}")
    _hash(obj["ordered_row_sha256"], "receipt.sidecar_derivation.ordered_row_sha256")
    _hash(
        obj["resolved_membership_sha256"], "receipt.sidecar_derivation.resolved_membership_sha256"
    )
    _true(
        obj["membership_covered_by_preflight"],
        "receipt.sidecar_derivation.membership_covered_by_preflight",
    )
    _true(
        obj["membership_covered_by_postflight"],
        "receipt.sidecar_derivation.membership_covered_by_postflight",
    )
