"""Finite JSON origin correspondence, not a serving loader or artifact schema.

PRECONDITION: the caller has authenticated the original full/fixed payload and
environment. ``package_roots`` is exactly its package -> root projection and
``source_files`` is its environment.files path -> SHA256 table. Original paths
are inert evidence: this module never opens or resolves them. Their canonical
filesystem identity, including absence of symlinks, belongs to that prior
authentication, not to lexical projection here.

Only encoder_identity.runtime.modules[i].file (kind="model") or the separate
processor.origin.file (kind="processor") becomes a SOURCE, exactly
{package, path, sha256}. All other JSON members, types and ordering survive.
JSON means exact dict/list/str/int/float/bool/None, string keys and finite floats;
there is no coercion, tensor traversal, pin generation or recursive path masking.

Installed roots must be independently caller-bound to the intended packages
before calling materialize_identity, with the same package keys as the original
roots. Fresh selected-source hashes alone do not authenticate a distribution,
version, complete environment, or native permissions. Materialization produces
EXPECTED metadata, never a live observation. The caller must independently
observe and compare live identities and maintain source integrity through use.
No native imports, construction, conversion or numerical qualification are authorized.
"""

import hashlib
import json
import math
import os
import re
import stat
from pathlib import PurePosixPath
from typing import cast

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]


def _clone_json(value: object) -> JSONValue:
    kind = type(value)
    if kind is dict:
        if any(type(key) is not str for key in cast(dict[object, object], value)):
            raise ValueError("JSON object keys must be strings")
        return {
            cast(str, key): _clone_json(item)
            for key, item in cast(dict[object, object], value).items()
        }
    if kind is list:
        return [_clone_json(item) for item in cast(list[object], value)]
    if (
        value is None
        or kind in (str, int, bool)
        or kind is float
        and math.isfinite(cast(float, value))
    ):
        return cast(JSONValue, value)
    raise ValueError("finite JSON metadata required")


def _absolute_path(value: object) -> PurePosixPath:
    if (
        type(value) is not str
        or not value.startswith("/")
        or "\\" in value
        or "\0" in value
        or any(part in ("", ".", "..") for part in value.split("/")[1:])
    ):
        raise ValueError("canonical absolute POSIX path required")
    return PurePosixPath(value)


def _roots(roots: object) -> dict[str, PurePosixPath]:
    if type(roots) is not dict or not roots:
        raise ValueError("authenticated package roots required")
    result: dict[str, PurePosixPath] = {}
    for package, root in roots.items():
        if type(package) is not str or not package.isidentifier():
            raise ValueError("top-level package name required")
        path = _absolute_path(root)
        if any(
            path.is_relative_to(other) or other.is_relative_to(path) for other in result.values()
        ):
            raise ValueError("ambiguous package roots")
        result[package] = path
    return result


def _origins(metadata: object, kind: object) -> list[JSONObject]:
    if type(metadata) is not dict or type(kind) is not str:
        raise ValueError("model or processor JSON identity required")
    if kind == "model":
        runtime = metadata.get("runtime")
        if type(runtime) is not dict or type(runtime.get("modules")) is not list:
            raise ValueError("model runtime.modules list required")
        origins = runtime["modules"]
    elif kind == "processor":
        origins = [metadata.get("origin")]
    else:
        raise ValueError("kind must be model or processor")
    if not origins or any(
        type(row) is not dict or not {"class", "file"} <= row.keys() for row in origins
    ):
        raise ValueError(kind + " class/file origin occurrences required")
    return cast(list[JSONObject], origins)


def project_identity(
    original: object, *, kind: object, package_roots: object, source_files: object
) -> JSONObject:
    """Copy authenticated model/processor JSON, replacing only explicit file leaves.

    Each SOURCE has string package/path/sha256. Package is the unchanged class's
    top-level name, path is a nonempty canonical package-relative POSIX suffix,
    and sha256 is the exact lowercase 64-hex member of authenticated source_files.
    Out-of-package or ambiguous originals reject; no fallback roots are inferred.
    """
    projected = _clone_json(original)
    roots = _roots(package_roots)
    if type(source_files) is not dict:
        raise ValueError("authenticated source file table required")
    for row in _origins(projected, kind):
        name = row["class"]
        if type(name) is not str or "." not in name or not name.rpartition(".")[2]:
            raise ValueError("qualified class required")
        package = name.split(".")[0]
        if package not in roots:
            raise ValueError("unknown class package")
        path = _absolute_path(row["file"])
        root = roots[package]
        if path == root or not path.is_relative_to(root):
            raise ValueError("original source is outside its class package root")
        sha = source_files.get(row["file"])
        if type(sha) is not str or re.fullmatch("[0-9a-f]{64}", sha) is None:
            raise ValueError("missing or invalid authenticated source hash")
        row["file"] = {"package": package, "path": str(path.relative_to(root)), "sha256": sha}
    return cast(JSONObject, projected)


def restore_identity(
    projected: object,
    original: object,
    *,
    kind: object,
    package_roots: object,
    source_files: object,
) -> JSONObject:
    """Invert explicit occurrences and reject any other typed metadata change.

    Returns a fresh original-shaped JSON dict. Neither input is changed. JSON
    serialization compares exact types (including bool/int/float), object order,
    list order and signed floating zero, after strict JSON validation.
    """
    expected = project_identity(
        original, kind=kind, package_roots=package_roots, source_files=source_files
    )
    restored = _clone_json(projected)
    rows = _origins(restored, kind)
    expected_rows = _origins(expected, kind)
    original_rows = _origins(original, kind)
    if len(rows) != len(expected_rows):
        raise ValueError("origin occurrence count differs")
    for row, reference, old in zip(rows, expected_rows, original_rows, strict=True):
        if row["file"] != reference["file"]:
            raise ValueError("projected SOURCE correspondence differs")
        row["file"] = old["file"]
    if json.dumps(restored, allow_nan=False) != json.dumps(original, allow_nan=False):
        raise ValueError("original typed metadata correspondence differs")
    return cast(JSONObject, restored)


def _installed_sha(path: PurePosixPath) -> str:
    # Walk open directory descriptors so no parent or leaf symlink is followed.
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        source = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        failure: BaseException | None = None
        try:
            with os.fdopen(source, "rb", closefd=False) as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("installed source must be a regular nonsymlink file")
                digest = hashlib.sha256()
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                return digest.hexdigest()
        except BaseException as primary:
            failure = primary
            raise
        finally:
            # The stream never owns the fd, including partial-construction failures.
            try:
                os.close(source)
            except BaseException as cleanup:
                if failure is None:
                    raise
                failure.add_note("source descriptor close failed: " + repr(cleanup))
    except OSError as error:
        raise ValueError("installed source must exist beneath nonsymlink directories") from error
    finally:
        os.close(directory)


def materialize_identity(
    projected: object,
    original: object,
    *,
    kind: object,
    package_roots: object,
    source_files: object,
    installed_roots: object,
) -> JSONObject:
    """Return expected installed metadata after inverse checks and fresh hashing.

    Independently bound installed_roots are canonical absolute POSIX directories.
    Every selected source is opened afresh on each call, through nonsymlink
    directories, and must be regular with the exact original SHA. No source or
    native module is imported. Duplicate occurrences share one read within this
    call only. Failures raise ValueError; inputs remain untouched.
    """
    expected = _clone_json(projected)
    restore_identity(
        expected, original, kind=kind, package_roots=package_roots, source_files=source_files
    )
    roots = _roots(installed_roots)
    if roots.keys() != cast(dict[str, str], package_roots).keys():
        raise ValueError("installed package root keys differ")
    checked: set[PurePosixPath] = set()
    for row in _origins(expected, kind):
        source = cast(dict[str, str], row["file"])
        path = roots[source["package"]] / source["path"]
        if path not in checked:
            if _installed_sha(path) != source["sha256"]:
                raise ValueError("installed source hash differs: " + str(path))
            checked.add(path)
        row["file"] = str(path)
    return cast(JSONObject, expected)
