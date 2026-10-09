"""Fresh, finite installed-file correspondence; never execution permission.

Callers supply independently trusted hashes of accepted original bundle/audit
BYTES. Only selected wheel files move to the explicit target root. Exact system
anchors remain unread and require a separate pre-native verification gate.

This verifies neither import/pyc/pth resolution, interpreter/ISA, global target
owner uniqueness, mapped native images nor integrity after descriptors close.
It imports no model/native library and caches no digest across invocations.
"""

from __future__ import annotations

import base64
import csv
import email.parser
import hashlib
import io
import json
import math
import os
import re
import stat
from pathlib import PurePosixPath
from typing import cast

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]
_METADATA_LIMIT = 16 * 1024 * 1024
_PACKAGES = {"PIL", "numpy", "safetensors", "torch", "torchvision", "transformers"}
_BUNDLE_KEYS = {
    "base_vision_sha256",
    "code",
    "encoder_identity",
    "endpoint_state_sha256",
    "environment",
    "files",
    "schema",
    "scope",
    "vision_sha256",
}
_AUDIT_KEYS = {
    "schema",
    "profiles",
    "records",
    "selected_members",
    "selected_native_members",
    "current_python_sources_sha_pass",
    "native_file_bytes_hashed",
    "native_modules_imported",
    "native_qualification",
    "product_go",
    "unique_owner_pass",
    "original_record_sha_and_metadata_row_pass",
}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _object(value: object, keys: set[str] | None = None) -> JSONObject:
    _require(
        type(value) is dict and all(type(k) is str for k in value), "exact JSON object required"
    )
    result = cast(JSONObject, value)
    if keys is not None:
        _require(result.keys() == keys, "exact object keys differ")
    return result


def _string(value: object) -> str:
    _require(type(value) is str and bool(value), "nonempty builtin string required")
    return cast(str, value)


def _sha(value: object) -> str:
    result = _string(value)
    _require(re.fullmatch("[0-9a-f]{64}", result) is not None, "exact SHA256 required")
    return result


def _size(value: object) -> int:
    _require(type(value) is int and value >= 0, "nonnegative builtin byte count required")
    return cast(int, value)


def _path(value: object, *, absolute: bool) -> PurePosixPath:
    s = _string(value)
    _require(
        "\\" not in s and "\0" not in s and s.startswith("/") is absolute,
        "canonical POSIX path required",
    )
    parts = s[1:].split("/") if absolute else s.split("/")
    _require(all(p not in {"", ".", ".."} for p in parts), "canonical path components required")
    return PurePosixPath(s)


def _pairs(pairs: list[tuple[str, JSONValue]]) -> JSONObject:
    result: JSONObject = {}
    for k, v in pairs:
        _require(k not in result, "duplicate JSON key")
        result[k] = v
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant: " + value)


def _float(value: str) -> float:
    result = float(value)
    _require(math.isfinite(result), "nonfinite JSON number")
    return result


def _authenticated(raw: bytes, expected: str) -> JSONObject:
    _require(type(raw) is bytes, "immutable bytes required")
    _require(hashlib.sha256(raw).hexdigest() == _sha(expected), "trusted input SHA differs")
    try:
        return _object(
            json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("strict JSON required") from error


def _hash_table(value: object) -> dict[str, str]:
    return {str(_path(k, absolute=True)): _sha(v) for k, v in _object(value).items()}


def _exact(left: JSONObject, right: JSONObject) -> bool:
    return json.dumps(
        left, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ) == json.dumps(right, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _prepare(
    original_bundle: bytes,
    original_ownership_audit: bytes,
    trusted_bundle_sha256: str,
    trusted_ownership_audit_sha256: str,
    site_packages: str,
) -> tuple[JSONObject, dict[str, JSONObject], dict[str, JSONObject]]:
    # Authenticate BOTH before parsing either or performing any target I/O.
    _require(
        type(original_bundle) is bytes and type(original_ownership_audit) is bytes,
        "immutable bytes required",
    )
    _require(
        hashlib.sha256(original_bundle).hexdigest() == _sha(trusted_bundle_sha256),
        "trusted bundle SHA differs",
    )
    _require(
        hashlib.sha256(original_ownership_audit).hexdigest()
        == _sha(trusted_ownership_audit_sha256),
        "trusted ownership SHA differs",
    )
    bundle = _object(_authenticated(original_bundle, trusted_bundle_sha256), _BUNDLE_KEYS)
    audit = _object(
        _authenticated(original_ownership_audit, trusted_ownership_audit_sha256), _AUDIT_KEYS
    )
    _require(
        bundle["schema"] == "siglip2-connected-mlp-bundle-v1", "original bundle schema differs"
    )
    _require(
        audit["schema"] == "connected-original-complete-record-ownership-source-audit-v1",
        "original ownership schema differs",
    )
    for key in [
        "records",
        "selected_members",
        "selected_native_members",
        "current_python_sources_sha_pass",
    ]:
        _size(audit[key])
    for key in _AUDIT_KEYS - {
        "schema",
        "profiles",
        "records",
        "selected_members",
        "selected_native_members",
        "current_python_sources_sha_pass",
    }:
        _require(type(audit[key]) is bool, "exact audit flag type required")
    env = _object(
        bundle["environment"], {"files", "native_files", "packages", "vision_constructor"}
    )
    files, natives = _hash_table(env["files"]), _hash_table(env["native_files"])
    _require(
        natives.keys() <= files.keys() and all(files[p] == h for p, h in natives.items()),
        "native membership/hash differs",
    )
    packages = _object(env["packages"], _PACKAGES)
    roots = []
    for name, value in packages.items():
        package = _object(value, {"root", "origin", "version"})
        root = _path(package["root"], absolute=True)
        _require(
            root.name == name
            and str(_path(package["origin"], absolute=True)) == str(root / "__init__.py"),
            "package root/origin differs",
        )
        _require(cast(str, package["origin"]) in files, "package origin not selected")
        _string(package["version"])
        roots.append(root.parent)
    _require(len(set(roots)) == 1, "one original site root required")
    original = roots[0]
    target = _path(site_packages, absolute=True)
    _require(
        not target.is_relative_to(original) and not original.is_relative_to(target),
        "target overlaps original root",
    )
    selected = {p for p in files if PurePosixPath(p).is_relative_to(original)}
    anchors = files.keys() - selected
    _require(anchors <= natives.keys(), "external non-native file forbidden")
    for p in anchors:
        anchor = PurePosixPath(p)
        _require(
            not anchor.is_relative_to(target) and not target.is_relative_to(anchor.parent),
            "target overlaps system anchor",
        )
    profiles: dict[str, JSONObject] = {}
    members: dict[str, JSONObject] = {}
    owner_versions: dict[str, str] = {}
    for record, value in _object(audit["profiles"]).items():
        rp = _path(record, absolute=True)
        _require(
            rp.parent.parent == original
            and rp.name == "RECORD"
            and rp.parent.name.endswith(".dist-info"),
            "exact original RECORD anchor required",
        )
        profile = _object(
            value, {"metadata", "name", "record_sha256", "selected_members", "version"}
        )
        record_sha = _sha(profile["record_sha256"])
        _string(profile["name"])
        version = _string(profile["version"])
        metadata = _object(profile["metadata"], {"path", "sha256", "bytes"})
        _require(
            _path(metadata["path"], absolute=True) == rp.parent / "METADATA",
            "metadata path differs",
        )
        _sha(metadata["sha256"])
        _size(metadata["bytes"])
        rows = profile["selected_members"]
        _require(type(rows) is list and bool(rows), "nonempty selected rows required")
        for value in cast(list[JSONValue], rows):
            row = _object(
                value,
                {"file", "record_relative_path", "sha256", "bytes", "record", "record_sha256"},
            )
            rel = _path(row["record_relative_path"], absolute=False)
            p = str(_path(row["file"], absolute=True))
            _require(
                p == str(original / rel) and p in selected and p not in members,
                "selected ownership omission/duplicate/addition/path differs",
            )
            _require(_sha(row["sha256"]) == files[p], "original member SHA differs")
            _size(row["bytes"])
            _require(
                row["record"] == record and _sha(row["record_sha256"]) == record_sha,
                "member RECORD association differs",
            )
            members[p] = row
            owner_versions[p] = version
        profiles[str(rp.relative_to(original))] = profile
    _require(
        members.keys() == selected,
        "complete selected set required; omitted wheel file is not an anchor",
    )
    for value in packages.values():
        package = _object(value)
        _require(
            owner_versions[cast(str, package["origin"])] == package["version"],
            "package owner version differs",
        )
    constructor = str(_path(env["vision_constructor"], absolute=True))
    _require(constructor in selected, "constructor must be selected")

    def move(p: str) -> str:
        path = PurePosixPath(p)
        return str(target / path.relative_to(original)) if path.is_relative_to(original) else p

    def unmove(p: str) -> str:
        path = _path(p, absolute=True)
        return str(original / path.relative_to(target)) if path.is_relative_to(target) else p

    mapping = {p: move(p) for p in files}
    _require(len(set(mapping.values())) == len(files), "destination collision")
    expected: JSONObject = {}
    restored: JSONObject = {}
    for key, value in env.items():
        if key in {"files", "native_files"}:
            table = _hash_table(value)
            mapped: JSONObject = {mapping[p]: h for p, h in table.items()}
            expected[key] = mapped
            restored[key] = {unmove(p): h for p, h in mapped.items()}
        elif key == "vision_constructor":
            expected[key] = move(constructor)
            restored[key] = unmove(_string(expected[key]))
        else:
            mapped_packages: JSONObject = {}
            restored_packages: JSONObject = {}
            for name, original_package in packages.items():
                package = _object(original_package)
                mapped_packages[name] = {
                    k: move(cast(str, v)) if k in {"root", "origin"} else v
                    for k, v in package.items()
                }
                # Invert the installed paths independently, preserving types/order.
                restored_packages[name] = {
                    k: str(original / _path(v, absolute=True).relative_to(target))
                    if k in {"root", "origin"}
                    else v
                    for k, v in _object(mapped_packages[name]).items()
                }
            expected[key] = mapped_packages
            restored[key] = restored_packages
    _require(_exact(restored, env), "full typed/order inverse differs")
    descriptors: JSONObject = {}
    for record, profile in sorted(profiles.items()):
        metadata = _object(profile["metadata"])
        descriptors[record] = {
            "record_sha256": profile["record_sha256"],
            "name": profile["name"],
            "version": profile["version"],
            "metadata": {
                "path": str(_path(metadata["path"], absolute=True).relative_to(original)),
                "sha256": metadata["sha256"],
                "bytes": metadata["bytes"],
            },
        }
    result: JSONObject = {
        "schema": "connected-installed-environment-correspondence-v1",
        "original_bundle_sha256": trusted_bundle_sha256,
        "original_ownership_audit_sha256": trusted_ownership_audit_sha256,
        "site_packages": site_packages,
        "distributions": descriptors,
        "expected_environment": expected,
    }
    return result, profiles, members


def _read_file(
    path: PurePosixPath, expected: str, size: int | None, *, capture: bool = False
) -> bytes:
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    failure: BaseException | None = None
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            parent, directory = directory, child
            os.close(parent)
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        try:
            with os.fdopen(fd, "rb", closefd=False) as stream:
                before = os.fstat(stream.fileno())
                _require(stat.S_ISREG(before.st_mode), "regular installed file required")
                if size is not None:
                    _require(before.st_size == size, "installed byte count differs")
                if capture:
                    _require(before.st_size <= _METADATA_LIMIT, "metadata exceeds byte bound")
                digest = hashlib.sha256()
                count = 0
                chunks = []
                while block := stream.read(1024 * 1024):
                    digest.update(block)
                    count += len(block)
                    if capture:
                        _require(count <= _METADATA_LIMIT, "metadata exceeds byte bound")
                        chunks.append(block)
                after = os.fstat(stream.fileno())
                _require(
                    (
                        before.st_dev,
                        before.st_ino,
                        before.st_size,
                        before.st_mtime_ns,
                        before.st_ctime_ns,
                    )
                    == (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ),
                    "installed file changed while reading",
                )
                _require(
                    count == before.st_size
                    and (size is None or count == size)
                    and digest.hexdigest() == expected,
                    "fresh installed SHA/size differs",
                )
                return b"".join(chunks) if capture else b""
        except BaseException as primary:
            failure = primary
            raise
        finally:
            try:
                os.close(fd)
            except BaseException as cleanup:
                if failure is None:
                    raise
                failure.add_note("source descriptor close failed: " + repr(cleanup))
    except OSError as error:
        failure = ValueError("nonsymlink installed path required")
        raise failure from error
    except BaseException as primary:
        failure = primary
        raise
    finally:
        try:
            os.close(directory)
        except BaseException as cleanup:
            if failure is None:
                raise
            failure.add_note("directory descriptor close failed: " + repr(cleanup))


def _record_row(rel: str, sha: str, size: int) -> list[str]:
    return [
        rel,
        "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(sha)).decode("ascii").rstrip("="),
        str(size),
    ]


def verify_installed_environment(
    original_bundle: bytes,
    original_ownership_audit: bytes,
    *,
    trusted_bundle_sha256: str,
    trusted_ownership_audit_sha256: str,
    site_packages: str,
) -> JSONObject:
    """Return expected metadata after fresh selected wheel verification.

    System anchors are preserved UNREAD; this is not runtime/native admission.
    The target must be distinct from the inert original site root. Unselected
    RECORD paths are only text, never filesystem operations. No digest persists.
    """
    result, profiles, members = _prepare(
        original_bundle,
        original_ownership_audit,
        trusted_bundle_sha256,
        trusted_ownership_audit_sha256,
        site_packages,
    )
    target = _path(site_packages, absolute=True)
    for record, profile in sorted(profiles.items()):
        raw = _read_file(target / record, _sha(profile["record_sha256"]), None, capture=True)
        metadata = _object(profile["metadata"])
        metadata_rel = str(PurePosixPath(record).parent / "METADATA")
        required = {
            metadata_rel: _record_row(
                metadata_rel, _sha(metadata["sha256"]), _size(metadata["bytes"])
            )
        }
        for value in cast(list[JSONValue], profile["selected_members"]):
            row = _object(value)
            rel = _string(row["record_relative_path"])
            selected_row = _record_row(rel, _sha(row["sha256"]), _size(row["bytes"]))
            _require(
                rel not in required or required[rel] == selected_row, "selected metadata conflict"
            )
            required[rel] = selected_row
        seen: set[str] = set()
        try:
            for csv_row in csv.reader(io.StringIO(raw.decode("utf-8")), strict=True):
                _require(len(csv_row) == 3, "exact RECORD row shape required")
                if csv_row[0] in required:
                    _require(
                        csv_row[0] not in seen and csv_row == required[csv_row[0]],
                        "exact selected RECORD row differs",
                    )
                    seen.add(csv_row[0])
        except (UnicodeError, csv.Error) as error:
            raise ValueError("strict RECORD text required") from error
        _require(seen == required.keys(), "selected RECORD row missing")
        raw_metadata = _read_file(
            target / metadata_rel, _sha(metadata["sha256"]), _size(metadata["bytes"]), capture=True
        )
        parsed = email.parser.BytesParser().parsebytes(raw_metadata)
        _require(
            parsed.get_all("Name", []) == [profile["name"]]
            and parsed.get_all("Version", []) == [profile["version"]],
            "exact metadata Name/Version differs",
        )
    for row in members.values():
        _read_file(
            target / _string(row["record_relative_path"]), _sha(row["sha256"]), _size(row["bytes"])
        )
    return result
