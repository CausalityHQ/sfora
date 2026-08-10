#!/usr/bin/env python3
"""Outcome-blind verifier for the immutable Pass 200 RSTA scientific artifact."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
import stat
import struct
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType


class ArtifactInvalid(ValueError):
    """The authenticated artifact failed its registered validation."""


class StructuralFailure(RuntimeError):
    """The verifier provenance, runtime, or process contract failed."""


LEGACY_HANDOFF_COMMIT = "c04574e2bb751c3229bce673408577cfedc00a88"
LEGACY_SOURCE_COMMIT = "15234a529a181c39c1c8b6477ad7eb7823fd0798"
LEGACY_MANIFEST_PATH = "docs/pass200_rsta_receipt_stage_a_manifest.json"
LEGACY_MANIFEST_SHA256 = "9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe"
LEGACY_DIAGNOSTIC_PATH = "scripts/diagnose_pass200_rsta_stage_a.py"
LEGACY_DIAGNOSTIC_SHA256 = "85958a940c5a4c9f0ae27f3342e436a8a37e49d94fe9515b22db0340d597ef6e"
ARTIFACT_PATH = (
    "reports/generated/pass200_rsta_receipt/c04574e2bb751c3229bce673408577cfedc00a88-stage-a.json"
)
ARTIFACT_SHA256 = "e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae"
RECOVERY_AMENDMENT_PATH = (
    "docs/pass200_rsta_scientific_artifact_roundtrip_recovery_amendment_2026-08-10.md"
)
RECOVERY_AMENDMENT_SHA256 = "6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591"
RECOVERY_AMENDMENT_COMMIT = "043121f8a414b91d7fb2e3d6a1635a6bd585676a"
RECOVERY_PLAN_COMMIT = "a3b7ad2bd1edd8cc749854f5d563449ed378a3e8"
VERIFIER_PATH = "scripts/verify_pass200_rsta_scientific_artifact.py"
CHILD_TIMEOUT = 600
CHILD_OUTPUT_LIMIT = 64
VALID_TOKEN = b"RSTA_LEGACY_VALID\n"
INVALID_TOKEN = b"RSTA_LEGACY_INVALID\n"
STRUCTURAL_TOKEN = b"RSTA_LEGACY_STRUCTURAL\n"

LEGACY_SOURCE_ORDER = (
    "scripts/diagnose_pass159_cotangent_stage_a.py",
    "scripts/diagnose_pass200_rsta_stage_a.py",
    "scripts/rsta_normwise_adjoint.py",
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
ROUNDTRIP_SOURCE_ORDER = (*LEGACY_SOURCE_ORDER[:3], VERIFIER_PATH, *LEGACY_SOURCE_ORDER[3:])
LEGACY_SCIENTIFIC_MANIFEST_ORDER = (
    "path",
    "sha256",
    "base_preregistration",
    "amendment",
    "deterministic_pool_amendment",
    "zero_jacobian_classifier_amendment",
    "binding_receipt",
    "historical",
    "artifact_schema",
    "source",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str and len(value) == length and all(c in "0123456789abcdef" for c in value)
    )


def _require_lower_hex(value: object, *, length: int) -> str:
    if not _lower_hex(value, length):
        raise StructuralFailure("registered hexadecimal identity differs")
    return value


def strict_json_object(data: bytes, *, name: str) -> dict[str, object]:
    """Parse one concrete finite JSON object without normalization."""

    def reject_constant(value: str) -> object:
        raise ArtifactInvalid(f"nonfinite JSON constant in {name}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactInvalid(f"duplicate JSON key in {name}")
            result[key] = value
        return result

    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactInvalid(f"invalid strict JSON: {name}") from error
    if type(value) is not dict:
        raise ArtifactInvalid(f"strict JSON root must be an object: {name}")

    def validate(item: object) -> None:
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ArtifactInvalid("JSON object key type differs")
            for child in item.values():
                validate(child)
        elif type(item) is list:
            for child in item:
                validate(child)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ArtifactInvalid("JSON number is nonfinite")
        elif item is None or type(item) in (bool, int, str):
            return
        else:
            raise ArtifactInvalid("JSON concrete scalar type differs")

    validate(value)
    return value


def exact_ordered_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return list(left) == list(right) and all(
            exact_ordered_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            exact_ordered_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is float:
        return math.isfinite(left) and struct.pack(">d", left) == struct.pack(">d", right)
    if left is None:
        return True
    if type(left) in (bool, int, str):
        return left == right
    return False


def adapt_legacy_support_keys(
    raw: dict[str, object],
) -> tuple[dict[str, object], tuple[tuple[str, ...], ...]]:
    if type(raw) is not dict:
        raise ArtifactInvalid("artifact root must be a concrete object")
    try:
        primary = raw["panel_binding"]["primary"]
        labels = primary["eligible_labels"]
        support = primary["support_ids_by_label"]
    except (KeyError, TypeError) as error:
        raise ArtifactInvalid("canonical support mapping is missing") from error
    if (
        type(labels) is not list
        or any(type(label) is not int or label < 0 for label in labels)
        or len(set(labels)) != len(labels)
        or type(support) is not dict
        or list(support) != [str(label) for label in labels]
    ):
        raise ArtifactInvalid("canonical support mapping differs")
    adapted = copy.deepcopy(raw)
    copied_support = adapted["panel_binding"]["primary"]["support_ids_by_label"]
    converted = {label: value for label, value in zip(labels, copied_support.values(), strict=True)}
    adapted["panel_binding"]["primary"]["support_ids_by_label"] = converted
    ledger = tuple(
        ("panel_binding", "primary", "support_ids_by_label", key, "str", "int") for key in support
    )
    restored = copy.deepcopy(adapted)
    restored["panel_binding"]["primary"]["support_ids_by_label"] = {
        str(key): value for key, value in converted.items()
    }
    if not exact_ordered_equal(restored, raw):
        raise ArtifactInvalid("legacy adapter changed unregistered content")
    return adapted, ledger


def legacy_scientific_payload_arguments(adapted: dict[str, object]) -> dict[str, object]:
    return {
        "manifest_audit": adapted["manifest"],
        "execution_audit": adapted["execution_audit"],
        "environment": adapted["environment"],
        "seed_audits": adapted["seed_audits"],
        "primary_rows": adapted["rows"]["primary"],
        "alternate_rows": adapted["rows"]["alternate"],
        "integrity": adapted["integrity"],
        "aggregation": adapted["aggregation"],
        "bootstrap": adapted["bootstrap"],
        "panel_binding": adapted["panel_binding"],
    }


def validate_legacy_roundtrip(
    raw_bytes: bytes,
    legacy_module: ModuleType,
    *,
    expected_numpy_version: str | None = None,
) -> None:
    raw = strict_json_object(raw_bytes, name="immutable scientific artifact")
    forbidden_later = {
        "adjoint_integrity_amendment",
        "normwise_adjoint_calibration_protocol",
        "normwise_adjoint_calibration_result",
        "normwise_adjoint_amendment",
        "normwise_adjoint_sign_control_amendment",
    }
    if type(raw.get("manifest")) is not dict or forbidden_later.intersection(raw["manifest"]):
        raise ArtifactInvalid("legacy manifest projection differs")
    if expected_numpy_version is not None:
        environment = raw.get("environment")
        persisted = environment.get("numpy_version") if type(environment) is dict else None
        runtime_numpy = sys.modules.get("numpy")
        if (
            type(expected_numpy_version) is not str
            or not expected_numpy_version
            or type(persisted) is not str
            or persisted != expected_numpy_version
            or getattr(legacy_module, "np", None) is not runtime_numpy
            or type(getattr(runtime_numpy, "__version__", None)) is not str
            or runtime_numpy.__version__ != expected_numpy_version
        ):
            raise ArtifactInvalid("persisted NumPy runtime differs")
    adapted, _ledger = adapt_legacy_support_keys(raw)
    arguments = legacy_scientific_payload_arguments(adapted)
    callable_value = getattr(legacy_module, "scientific_payload", None)
    if not callable(callable_value):
        raise StructuralFailure("legacy scientific payload callable differs")
    try:
        recomputed = callable_value(**arguments)
    except ValueError as error:
        raise ArtifactInvalid("legacy scientific payload rejected artifact") from error
    if not exact_ordered_equal(recomputed, raw):
        raise ArtifactInvalid("legacy roundtrip exact equality differs")
    try:
        encoded = (
            json.dumps(recomputed, indent=2, sort_keys=False, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactInvalid("legacy roundtrip serialization differs") from error
    if encoded != raw_bytes:
        raise ArtifactInvalid("legacy roundtrip bytes differ")


def legacy_manifest_projection(repository: Path, manifest: dict[str, object]) -> dict[str, object]:
    del repository
    try:
        projection = {
            "path": LEGACY_MANIFEST_PATH,
            "sha256": LEGACY_MANIFEST_SHA256,
            "base_preregistration": manifest["base_preregistration"],
            "amendment": manifest["amendment"],
            "deterministic_pool_amendment": manifest["deterministic_pool_amendment"],
            "zero_jacobian_classifier_amendment": manifest["zero_jacobian_classifier_amendment"],
            "binding_receipt": manifest["binding_receipt"],
            "historical": manifest["historical"],
            "artifact_schema": manifest["artifact_schema"],
            "source": manifest["current_scientific_source"],
        }
    except KeyError as error:
        raise StructuralFailure("legacy manifest authority is incomplete") from error
    if tuple(projection) != LEGACY_SCIENTIFIC_MANIFEST_ORDER:
        raise StructuralFailure("legacy manifest projection order differs")
    return projection


def _git(repository: Path, *arguments: str, text: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=False,
        text=text,
    )
    if result.returncode != 0:
        raise StructuralFailure("Git provenance command failed")
    return result.stdout


def _git_blob(repository: Path, revision: str, path: str) -> bytes:
    value = _git(repository, "cat-file", "blob", f"{revision}:{path}")
    if not isinstance(value, bytes):
        raise StructuralFailure("Git blob output type differs")
    return value


def authenticate_legacy_provenance(repository: Path) -> dict[str, object]:
    parent_line = str(
        _git(repository, "rev-list", "--parents", "-n", "1", LEGACY_HANDOFF_COMMIT, text=True)
    ).strip()
    if parent_line.split() != [LEGACY_HANDOFF_COMMIT, LEGACY_SOURCE_COMMIT]:
        raise StructuralFailure("legacy handoff parent differs")
    changed = str(
        _git(
            repository,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            LEGACY_HANDOFF_COMMIT,
            text=True,
        )
    ).splitlines()
    if changed != [LEGACY_MANIFEST_PATH]:
        raise StructuralFailure("legacy handoff scope differs")
    manifest_bytes = _git_blob(repository, LEGACY_HANDOFF_COMMIT, LEGACY_MANIFEST_PATH)
    if _sha256(manifest_bytes) != LEGACY_MANIFEST_SHA256:
        raise StructuralFailure("legacy manifest digest differs")
    try:
        manifest = strict_json_object(manifest_bytes, name="legacy handoff manifest")
    except ArtifactInvalid as error:
        raise StructuralFailure("legacy manifest schema differs") from error
    current = manifest.get("current_scientific_source")
    if type(current) is not dict or current.get("git_revision") != LEGACY_SOURCE_COMMIT:
        raise StructuralFailure("legacy source commit differs")
    files = current.get("files")
    if type(files) is not dict or tuple(files) != LEGACY_SOURCE_ORDER:
        raise StructuralFailure("legacy source order differs")
    for path, expected in files.items():
        if not _lower_hex(expected, 64):
            raise StructuralFailure("legacy source digest syntax differs")
        source_blob = _git_blob(repository, LEGACY_SOURCE_COMMIT, path)
        handoff_blob = _git_blob(repository, LEGACY_HANDOFF_COMMIT, path)
        if _sha256(source_blob) != expected or handoff_blob != source_blob:
            raise StructuralFailure("legacy source Git blob differs")
    diagnostic = _git_blob(repository, LEGACY_SOURCE_COMMIT, LEGACY_DIAGNOSTIC_PATH)
    if (
        _sha256(diagnostic) != LEGACY_DIAGNOSTIC_SHA256
        or _git_blob(repository, LEGACY_HANDOFF_COMMIT, LEGACY_DIAGNOSTIC_PATH) != diagnostic
    ):
        raise StructuralFailure("legacy diagnostic digest differs")
    return {
        "handoff_commit": LEGACY_HANDOFF_COMMIT,
        "source_commit": LEGACY_SOURCE_COMMIT,
        "manifest_path": LEGACY_MANIFEST_PATH,
        "manifest_sha256": LEGACY_MANIFEST_SHA256,
        "diagnostic_path": LEGACY_DIAGNOSTIC_PATH,
        "diagnostic_sha256": LEGACY_DIAGNOSTIC_SHA256,
    }


def authenticate_runtime(repository: Path) -> dict[str, str]:
    registered = (repository / ".venv/bin/python").absolute()
    executing = Path(sys.executable)
    if executing != registered:
        raise StructuralFailure("registered Python invocation path differs")
    try:
        resolved_registered = registered.resolve(strict=True)
        resolved_executing = executing.resolve(strict=True)
    except OSError as error:
        raise StructuralFailure("registered Python resolution failed") from error
    if (
        resolved_registered != resolved_executing
        or not resolved_registered.is_file()
        or not os.access(resolved_registered, os.X_OK)
        or tuple(sys.version_info[:3]) != (3, 12, 3)
    ):
        raise StructuralFailure("registered Python runtime differs")
    import numpy

    if (
        sys.modules.get("numpy") is not numpy
        or type(numpy.__version__) is not str
        or not numpy.__version__
    ):
        raise StructuralFailure("registered NumPy runtime differs")
    return {
        "python_executable": ".venv/bin/python",
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy_version": numpy.__version__,
    }


def authenticate_verifier_provenance(repository: Path, manifest_path: Path) -> dict[str, object]:
    expected_file = (repository / VERIFIER_PATH).resolve()
    observed_file = Path(str(globals().get("__file__", ""))).resolve()
    if observed_file != expected_file or not expected_file.is_file() or expected_file.is_symlink():
        raise StructuralFailure("verifier executing path differs")
    head = str(_git(repository, "rev-parse", "HEAD", text=True)).strip()
    detached = subprocess.run(
        ["git", "-C", str(repository), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
        check=False,
    )
    if detached.returncode != 1 or detached.stdout or detached.stderr:
        raise StructuralFailure("verifier checkout is not clean detached HEAD")
    parent_line = (
        str(_git(repository, "rev-list", "--parents", "-n", "1", head, text=True)).strip().split()
    )
    if len(parent_line) != 2 or parent_line[0] != head:
        raise StructuralFailure("verifier handoff parent differs")
    source = parent_line[1]
    source_parent = str(
        _git(repository, "rev-list", "--parents", "-n", "1", source, text=True)
    ).strip()
    if source_parent.split() != [source, RECOVERY_PLAN_COMMIT]:
        raise StructuralFailure("verifier source parent is not the exact recovery plan")
    plan_parent = str(
        _git(
            repository,
            "rev-list",
            "--parents",
            "-n",
            "1",
            RECOVERY_PLAN_COMMIT,
            text=True,
        )
    ).strip()
    if plan_parent.split() != [RECOVERY_PLAN_COMMIT, RECOVERY_AMENDMENT_COMMIT]:
        raise StructuralFailure("recovery plan parent is not the exact amendment")
    source_paths = str(
        _git(
            repository,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            source,
            text=True,
        )
    ).splitlines()
    if set(source_paths) != {
        "scripts/diagnose_pass200_rsta_stage_a.py",
        "scripts/verify_pass200_rsta_scientific_artifact.py",
        "tests/test_diagnose_pass200_rsta_stage_a.py",
        "tests/test_verify_pass200_rsta_scientific_artifact.py",
    }:
        raise StructuralFailure("verifier source commit scope differs")
    if str(_git(repository, "status", "--porcelain", "--untracked-files=all", text=True)):
        raise StructuralFailure("verifier checkout is dirty")
    changed = str(
        _git(repository, "diff-tree", "--no-commit-id", "--name-only", "-r", head, text=True)
    ).splitlines()
    if changed != [LEGACY_MANIFEST_PATH]:
        raise StructuralFailure("verifier handoff scope differs")
    literal_manifest = (repository / LEGACY_MANIFEST_PATH).resolve()
    if manifest_path.resolve() != literal_manifest or literal_manifest.is_symlink():
        raise StructuralFailure("verifier manifest path differs")
    manifest_bytes = literal_manifest.read_bytes()
    if manifest_bytes != _git_blob(repository, head, LEGACY_MANIFEST_PATH):
        raise StructuralFailure("verifier manifest worktree differs")
    try:
        manifest = strict_json_object(manifest_bytes, name="verifier handoff manifest")
    except ArtifactInvalid as error:
        raise StructuralFailure("verifier handoff manifest schema differs") from error
    current = manifest.get("current_scientific_source")
    files = current.get("files") if type(current) is dict else None
    if (
        current.get("git_revision") != source
        or type(files) is not dict
        or tuple(files) != ROUNDTRIP_SOURCE_ORDER
    ):
        raise StructuralFailure("verifier source domain differs")
    for path, expected in files.items():
        worktree = repository / path
        if worktree.is_symlink() or not worktree.is_file():
            raise StructuralFailure("verifier source path differs")
        blob = _git_blob(repository, source, path)
        if _sha256(blob) != expected or _sha256(worktree.read_bytes()) != expected:
            raise StructuralFailure("verifier source digest differs")
    authority = manifest.get("scientific_artifact_roundtrip_recovery_amendment")
    expected_authority = {
        "path": RECOVERY_AMENDMENT_PATH,
        "sha256": RECOVERY_AMENDMENT_SHA256,
        "commit": RECOVERY_AMENDMENT_COMMIT,
    }
    if (
        type(authority) is not dict
        or list(authority) != ["path", "sha256", "commit"]
        or authority != expected_authority
    ):
        raise StructuralFailure("recovery amendment authority differs")
    if (
        _sha256(_git_blob(repository, RECOVERY_AMENDMENT_COMMIT, RECOVERY_AMENDMENT_PATH))
        != RECOVERY_AMENDMENT_SHA256
    ):
        raise StructuralFailure("recovery amendment Git blob differs")
    amendment_path = repository / RECOVERY_AMENDMENT_PATH
    if (
        amendment_path.is_symlink()
        or not amendment_path.is_file()
        or _sha256(amendment_path.read_bytes()) != RECOVERY_AMENDMENT_SHA256
    ):
        raise StructuralFailure("recovery amendment worktree differs")
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            RECOVERY_AMENDMENT_COMMIT,
            source,
        ],
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise StructuralFailure("recovery amendment ancestry differs")
    verifier_digest = files[VERIFIER_PATH]
    if _sha256(expected_file.read_bytes()) != verifier_digest:
        raise StructuralFailure("executing verifier digest differs")
    return {
        "source_commit": source,
        "handoff_commit": head,
        "manifest_path": LEGACY_MANIFEST_PATH,
        "manifest_sha256": _sha256(manifest_bytes),
        "verifier_path": VERIFIER_PATH,
        "verifier_sha256": verifier_digest,
        "amendment": authority,
    }


def _load_legacy_module(checkout: Path) -> ModuleType:
    path = (checkout / LEGACY_DIAGNOSTIC_PATH).resolve()
    if (
        path != checkout.resolve() / LEGACY_DIAGNOSTIC_PATH
        or path.is_symlink()
        or _sha256(path.read_bytes()) != LEGACY_DIAGNOSTIC_SHA256
    ):
        raise StructuralFailure("legacy diagnostic worktree differs")
    module_name = "_pass200_rsta_authenticated_legacy_producer"
    if module_name in sys.modules:
        raise StructuralFailure("legacy module identity is preexisting")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise StructuralFailure("legacy diagnostic loader differs")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if Path(module.__file__).resolve() != path or not callable(module.scientific_payload):
            raise StructuralFailure("legacy diagnostic callable differs")
        return module
    finally:
        sys.modules.pop(module_name, None)
        sys.path[:] = old_path


def _create_legacy_checkout(repository: Path) -> tempfile.TemporaryDirectory[str]:
    temporary = tempfile.TemporaryDirectory(prefix="pass200-rsta-legacy-")
    checkout = Path(temporary.name) / "repo"
    _git(repository, "clone", "--no-hardlinks", "--no-checkout", str(repository), str(checkout))
    _git(checkout, "checkout", "--detach", LEGACY_HANDOFF_COMMIT)
    if str(_git(checkout, "rev-parse", "HEAD", text=True)).strip() != LEGACY_HANDOFF_COMMIT or str(
        _git(checkout, "status", "--porcelain", "--untracked-files=all", text=True)
    ):
        temporary.cleanup()
        raise StructuralFailure("legacy isolated checkout differs")
    temporary.checkout = checkout
    return temporary


def run_isolated_legacy_child(
    repository: Path,
    artifact_fd: int,
    *,
    verifier_source_commit: str,
    verifier_handoff_commit: str,
    python_executable: Path,
    expected_numpy_version: str,
) -> tuple[int, int]:
    _require_lower_hex(verifier_source_commit, length=40)
    _require_lower_hex(verifier_handoff_commit, length=40)
    created = _create_legacy_checkout(repository)
    if isinstance(created, Path):
        checkout = created
        cleanup = None
    else:
        checkout = created.checkout
        cleanup = created
    command = [
        str(python_executable),
        "-I",
        "-B",
        str((repository / VERIFIER_PATH).resolve()),
        "--legacy-child",
        "--live-repository",
        str(repository.resolve()),
        "--old-checkout",
        str(checkout.resolve()),
        "--artifact-fd",
        str(artifact_fd),
        "--verifier-source-commit",
        verifier_source_commit,
        "--verifier-handoff-commit",
        verifier_handoff_commit,
        "--expected-numpy-version",
        expected_numpy_version,
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PATH": os.defpath,
        "LC_ALL": "C",
    }
    try:
        process = subprocess.Popen(
            command,
            cwd=checkout,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(artifact_fd,),
            start_new_session=True,
            env=environment,
        )
        try:
            stdout, stderr = process.communicate(timeout=CHILD_TIMEOUT)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise StructuralFailure("legacy child timed out") from error
        if len(stdout) > CHILD_OUTPUT_LIMIT or len(stderr) > CHILD_OUTPUT_LIMIT or stderr:
            raise StructuralFailure("legacy child output contract differs")
        expected = {0: VALID_TOKEN, 1: INVALID_TOKEN, 2: STRUCTURAL_TOKEN}.get(process.returncode)
        if expected is None or stdout != expected or process.returncode == 2:
            raise StructuralFailure("legacy child token contract differs")
        return process.pid, process.returncode
    finally:
        if cleanup is not None:
            cleanup.cleanup()


def receipt_path(repository: Path, handoff_commit: str) -> Path:
    _require_lower_hex(handoff_commit, length=40)
    return (
        repository
        / "reports/generated/pass200_rsta_receipt"
        / (f"{handoff_commit}-scientific-artifact-roundtrip-validation.json")
    )


_RECEIPT_ORDER = (
    "schema_version",
    "validation",
    "mode",
    "attempt",
    "status",
    "outcome_disclosed",
    "artifact",
    "legacy_provenance",
    "verifier_provenance",
    "process",
)
_FORBIDDEN_RECEIPT_NAMES = {
    "verdict",
    "decisive_clause",
    "candidate",
    "candidate_value",
    "field",
    "row",
    "score",
    "metric",
    "aggregate",
    "bootstrap",
    "criterion",
    "exclusion",
    "excerpt",
}


def validate_roundtrip_receipt(value: dict[str, object]) -> None:
    if type(value) is not dict or tuple(value) != _RECEIPT_ORDER:
        raise StructuralFailure("roundtrip receipt top-level schema differs")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["validation"] != "pass200-rsta-scientific-artifact-roundtrip"
        or value["mode"] != "offline_immutable_artifact"
        or type(value["attempt"]) is not int
        or value["attempt"] != 1
        or type(value["status"]) is not str
        or value["status"] not in {"VALID", "INVALID"}
        or value["outcome_disclosed"] is not False
    ):
        raise StructuralFailure("roundtrip receipt scalar contract differs")
    schemas = {
        "artifact": ("path", "sha256", "producer_pid", "producer_exit_code", "immutable"),
        "legacy_provenance": (
            "handoff_commit",
            "source_commit",
            "manifest_path",
            "manifest_sha256",
            "diagnostic_path",
            "diagnostic_sha256",
        ),
        "verifier_provenance": (
            "source_commit",
            "handoff_commit",
            "manifest_path",
            "manifest_sha256",
            "verifier_path",
            "verifier_sha256",
            "amendment",
        ),
        "process": (
            "parent_pid",
            "child_pid",
            "child_exit_code",
            "python_executable",
            "python_version",
            "numpy_version",
            "isolated",
            "child_head_commit",
            "cuda_visible_devices",
        ),
    }
    for name, order in schemas.items():
        if type(value[name]) is not dict or tuple(value[name]) != order:
            raise StructuralFailure(f"roundtrip receipt {name} schema differs")
    artifact = value["artifact"]
    if not exact_ordered_equal(
        artifact,
        {
            "path": ARTIFACT_PATH,
            "sha256": ARTIFACT_SHA256,
            "producer_pid": 1002393,
            "producer_exit_code": 0,
            "immutable": True,
        },
    ):
        raise StructuralFailure("roundtrip receipt artifact authority differs")
    legacy = value["legacy_provenance"]
    expected_legacy = {
        "handoff_commit": LEGACY_HANDOFF_COMMIT,
        "source_commit": LEGACY_SOURCE_COMMIT,
        "manifest_path": LEGACY_MANIFEST_PATH,
        "manifest_sha256": LEGACY_MANIFEST_SHA256,
        "diagnostic_path": LEGACY_DIAGNOSTIC_PATH,
        "diagnostic_sha256": LEGACY_DIAGNOSTIC_SHA256,
    }
    if not exact_ordered_equal(legacy, expected_legacy):
        raise StructuralFailure("roundtrip receipt legacy provenance differs")
    verifier = value["verifier_provenance"]
    amendment = verifier["amendment"]
    if (
        not _lower_hex(verifier["source_commit"], 40)
        or not _lower_hex(verifier["handoff_commit"], 40)
        or verifier["manifest_path"] != LEGACY_MANIFEST_PATH
        or not _lower_hex(verifier["manifest_sha256"], 64)
        or verifier["verifier_path"] != VERIFIER_PATH
        or not _lower_hex(verifier["verifier_sha256"], 64)
        or type(amendment) is not dict
        or list(amendment) != ["path", "sha256", "commit"]
        or amendment
        != {
            "path": RECOVERY_AMENDMENT_PATH,
            "sha256": RECOVERY_AMENDMENT_SHA256,
            "commit": RECOVERY_AMENDMENT_COMMIT,
        }
    ):
        raise StructuralFailure("roundtrip receipt verifier provenance differs")
    process = value["process"]
    expected_exit = 0 if value["status"] == "VALID" else 1
    if (
        type(process["parent_pid"]) is not int
        or process["parent_pid"] <= 0
        or type(process["child_pid"]) is not int
        or process["child_pid"] <= 0
        or type(process["child_exit_code"]) is not int
        or process["child_exit_code"] != expected_exit
        or process["python_executable"] != ".venv/bin/python"
        or process["python_version"] != "3.12.3"
        or type(process["numpy_version"]) is not str
        or not process["numpy_version"]
        or process["isolated"] is not True
        or process["child_head_commit"] != LEGACY_HANDOFF_COMMIT
        or process["cuda_visible_devices"] != ""
    ):
        raise StructuralFailure("roundtrip receipt process contract differs")

    def reject_names(item: object) -> None:
        if type(item) is dict:
            if any(key.lower() in _FORBIDDEN_RECEIPT_NAMES for key in item):
                raise StructuralFailure("roundtrip receipt discloses scientific content")
            for child in item.values():
                reject_names(child)
        elif type(item) is list:
            for child in item:
                reject_names(child)

    reject_names(value)


def write_validation_receipt_atomic(path: Path, value: dict[str, object]) -> None:
    validate_roundtrip_receipt(value)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise StructuralFailure("roundtrip receipt destination is unavailable")
    encoded = (json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise StructuralFailure("roundtrip receipt temporary path is unavailable")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if path.exists() and path.read_bytes() != encoded:
            raise
        raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    reloaded = strict_json_object(path.read_bytes(), name="roundtrip validation receipt")
    validate_roundtrip_receipt(reloaded)
    if not exact_ordered_equal(reloaded, value) or path.read_bytes() != encoded:
        raise StructuralFailure("roundtrip receipt publication differs")


def _read_artifact_fd(descriptor: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ArtifactInvalid("artifact descriptor is not regular")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise ArtifactInvalid("artifact descriptor identity changed")
    return b"".join(chunks)


def _legacy_child(arguments: argparse.Namespace) -> int:
    try:
        live = Path(arguments.live_repository).resolve()
        old = Path(arguments.old_checkout).resolve()
        runtime = authenticate_runtime(live)
        verifier = authenticate_verifier_provenance(live, live / LEGACY_MANIFEST_PATH)
        if (
            verifier["source_commit"] != arguments.verifier_source_commit
            or verifier["handoff_commit"] != arguments.verifier_handoff_commit
            or runtime["numpy_version"] != arguments.expected_numpy_version
        ):
            raise StructuralFailure("child supplied provenance differs")
        if str(_git(old, "rev-parse", "HEAD", text=True)).strip() != LEGACY_HANDOFF_COMMIT or str(
            _git(old, "status", "--porcelain", "--untracked-files=all", text=True)
        ):
            raise StructuralFailure("child old checkout differs")
        authenticate_legacy_provenance(old)
        raw_bytes = _read_artifact_fd(arguments.artifact_fd)
        if _sha256(raw_bytes) != ARTIFACT_SHA256:
            raise ArtifactInvalid("artifact digest differs")
        raw = strict_json_object(raw_bytes, name="immutable scientific artifact")
        old_manifest = strict_json_object(
            (old / LEGACY_MANIFEST_PATH).read_bytes(), name="legacy manifest"
        )
        if not exact_ordered_equal(
            raw.get("manifest"), legacy_manifest_projection(old, old_manifest)
        ):
            raise ArtifactInvalid("artifact legacy manifest projection differs")
        module = _load_legacy_module(old)
        runtime_numpy = sys.modules.get("numpy")
        if (
            getattr(module, "np", None) is not runtime_numpy
            or type(getattr(runtime_numpy, "__version__", None)) is not str
            or runtime_numpy.__version__ != runtime["numpy_version"]
        ):
            raise StructuralFailure("legacy child NumPy module differs")
        environment = raw.get("environment")
        persisted_numpy = environment.get("numpy_version") if type(environment) is dict else None
        if (
            type(persisted_numpy) is not str
            or not persisted_numpy
            or persisted_numpy != runtime["numpy_version"]
        ):
            raise ArtifactInvalid("artifact persisted NumPy version differs")
        validate_legacy_roundtrip(
            raw_bytes, module, expected_numpy_version=runtime["numpy_version"]
        )
    except ArtifactInvalid:
        os.write(1, INVALID_TOKEN)
        return 1
    except Exception:
        os.write(1, STRUCTURAL_TOKEN)
        return 2
    os.write(1, VALID_TOKEN)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest")
    parser.add_argument("--artifact")
    parser.add_argument("--output")
    parser.add_argument("--validate-immutable-artifact-once", action="store_true")
    parser.add_argument("--legacy-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--live-repository", help=argparse.SUPPRESS)
    parser.add_argument("--old-checkout", help=argparse.SUPPRESS)
    parser.add_argument("--artifact-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--verifier-source-commit", help=argparse.SUPPRESS)
    parser.add_argument("--verifier-handoff-commit", help=argparse.SUPPRESS)
    parser.add_argument("--expected-numpy-version", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.legacy_child:
        public = (
            arguments.manifest,
            arguments.artifact,
            arguments.output,
            arguments.validate_immutable_artifact_once,
        )
        private = (
            arguments.live_repository,
            arguments.old_checkout,
            arguments.artifact_fd,
            arguments.verifier_source_commit,
            arguments.verifier_handoff_commit,
            arguments.expected_numpy_version,
        )
        if any(value is not None and value is not False for value in public) or any(
            value is None for value in private
        ):
            return 2
        return _legacy_child(arguments)
    private = (
        arguments.live_repository,
        arguments.old_checkout,
        arguments.artifact_fd,
        arguments.verifier_source_commit,
        arguments.verifier_handoff_commit,
        arguments.expected_numpy_version,
    )
    if (
        any(value is not None for value in private)
        or not arguments.validate_immutable_artifact_once
    ):
        return 2
    repository = Path.cwd().resolve()
    try:
        runtime = authenticate_runtime(repository)
        manifest_path = (repository / LEGACY_MANIFEST_PATH).resolve()
        artifact_path = (repository / ARTIFACT_PATH).absolute()
        verifier = authenticate_verifier_provenance(repository, manifest_path)
        expected_output = receipt_path(repository, verifier["handoff_commit"])
        if (
            arguments.manifest != LEGACY_MANIFEST_PATH
            or arguments.artifact != ARTIFACT_PATH
            or Path(arguments.output).absolute() != expected_output.absolute()
            or artifact_path.resolve().is_relative_to(repository) is False
            or artifact_path.is_symlink()
            or expected_output.exists()
            or expected_output.is_symlink()
            or not expected_output.parent.is_dir()
            or expected_output.parent.is_symlink()
            or expected_output.parent.resolve()
            != (repository / "reports/generated/pass200_rsta_receipt").resolve()
        ):
            raise StructuralFailure("public verifier path contract differs")
        temporary = expected_output.with_name(f".{expected_output.name}.{os.getpid()}.tmp")
        if temporary.exists() or temporary.is_symlink():
            raise StructuralFailure("public verifier temporary path differs")
        legacy = authenticate_legacy_provenance(repository)
        descriptor = os.open(artifact_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise ArtifactInvalid("artifact path is not regular")
            child_pid, child_exit = run_isolated_legacy_child(
                repository,
                descriptor,
                verifier_source_commit=verifier["source_commit"],
                verifier_handoff_commit=verifier["handoff_commit"],
                python_executable=Path(sys.executable),
                expected_numpy_version=runtime["numpy_version"],
            )
        finally:
            os.close(descriptor)
        status = "VALID" if child_exit == 0 else "INVALID"
        receipt = {
            "schema_version": 1,
            "validation": "pass200-rsta-scientific-artifact-roundtrip",
            "mode": "offline_immutable_artifact",
            "attempt": 1,
            "status": status,
            "outcome_disclosed": False,
            "artifact": {
                "path": ARTIFACT_PATH,
                "sha256": ARTIFACT_SHA256,
                "producer_pid": 1002393,
                "producer_exit_code": 0,
                "immutable": True,
            },
            "legacy_provenance": legacy,
            "verifier_provenance": verifier,
            "process": {
                "parent_pid": os.getpid(),
                "child_pid": child_pid,
                "child_exit_code": child_exit,
                "python_executable": runtime["python_executable"],
                "python_version": runtime["python_version"],
                "numpy_version": runtime["numpy_version"],
                "isolated": True,
                "child_head_commit": LEGACY_HANDOFF_COMMIT,
                "cuda_visible_devices": "",
            },
        }
        write_validation_receipt_atomic(expected_output, receipt)
        return child_exit
    except Exception:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
