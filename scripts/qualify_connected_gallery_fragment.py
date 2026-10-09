#!/usr/bin/env python3
"""Authenticate and publish an opaque source fragment; never serving acceptance.

Invoke canonical Python -I -S -B canonical driver --authority canonicalFILE
--authority-sha256 SHA. Only the root supplies authority and derived references.
A parent must collect normal exit 0 AND verify the single stdout report/files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
import re
import resource
import signal
import stat
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, cast

LIMIT = 16 * 1024 * 1024
SECONDS = 120
ADDRESS_SPACE = 1073741824
SOURCES = {
    "package_init": "sfora",
    "provenance": "sfora.connected_gallery_provenance",
    "conversion": "sfora.connected_gallery_conversion",
    "publication": "sfora.atomic_publication",
}
FILE_ROLES = {"driver", "test", *SOURCES}
INPUT_ROLES = {"receipt", "bundle", "ownership_audit", "combined_wire"}
MEMBERS = {
    "origin.json",
    "origin-export.json",
    "origin-owners.json",
    "gallery.bin",
    "gallery-ids.json",
    "gallery-provenance.json",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def obj(value: object, keys: set[str] | None = None) -> dict[str, Any]:
    require(type(value) is dict, "builtin object required")
    result = cast(dict[str, Any], value)
    require(keys is None or result.keys() == keys, "exact object keys differ")
    return result


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def nonfinite(value: str) -> NoReturn:
    raise ValueError("nonfinite JSON value: " + value)


def finite(value: str) -> float:
    number = float(value)
    require(math.isfinite(number), "nonfinite JSON number")
    return number


def parse(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite)


def sha(value: object) -> str:
    require(
        type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
        "strict lowercase SHA256 required",
    )
    return cast(str, value)


def path(value: object, *, new: bool = False) -> Path:
    require(type(value) is str and bool(value), "canonical absolute path required")
    result = Path(cast(str, value))
    require(
        result.is_absolute() and str(result) == value and "\x00" not in str(result),
        "canonical absolute path required",
    )
    require(result.resolve(strict=not new) == result, "symlink or noncanonical path")
    return result


def identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def stamp(info: os.stat_result) -> tuple[int, ...]:
    return (
        *identity(info),
        info.st_size,
        info.st_mode,
        info.st_nlink,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def read_fd(fd: int, *, capture: bool) -> tuple[bytes, str, os.stat_result]:
    before = os.fstat(fd)
    require(stat.S_ISREG(before.st_mode), "regular file required")
    require(not capture or before.st_size <= LIMIT, "file exceeds bounded read")
    hasher = hashlib.sha256()
    chunks: list[bytes] = []
    offset = 0
    while offset < before.st_size:
        chunk = os.pread(fd, min(1024 * 1024, before.st_size - offset), offset)
        require(bool(chunk), "file truncated during read")
        hasher.update(chunk)
        if capture:
            chunks.append(chunk)
        offset += len(chunk)
    require(
        not os.pread(fd, 1, offset) and stamp(before) == stamp(os.fstat(fd)),
        "file changed during read",
    )
    return b"".join(chunks), hasher.hexdigest(), before


class Captured:
    def __init__(self, descriptor: object, *, capture: bool = True) -> None:
        row = obj(descriptor, {"path", "sha256"})
        self.path = path(row["path"])
        self.sha256 = sha(row["sha256"])
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            self.raw, observed, self.info = read_fd(self.fd, capture=capture)
            require(observed == self.sha256, "trusted SHA differs: " + str(self.path))
            self.check_path()
        except BaseException:
            os.close(self.fd)
            raise

    def check_path(self) -> None:
        require(
            path(str(self.path)) == self.path and stamp(self.path.lstat()) == stamp(self.info),
            "file path ownership changed: " + str(self.path),
        )

    def fresh(self) -> None:
        self.check_path()
        _, observed, current = read_fd(self.fd, capture=False)
        require(
            observed == self.sha256 and stamp(current) == stamp(self.info),
            "fresh SHA or identity differs: " + str(self.path),
        )
        self.check_path()

    def facts(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.info.st_size,
            "device": self.info.st_dev,
            "inode": self.info.st_ino,
        }

    def close(self) -> None:
        os.close(self.fd)


class SourceModules(importlib.abc.MetaPathFinder):
    """Own exactly four modules; no filesystem/cache loader for package code."""

    def __init__(self) -> None:
        self.owned: dict[str, types.ModuleType] = {}

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == "sfora" or fullname.startswith("sfora."):
            raise ImportError("uncaptured package import: " + fullname)

    def load(self, sources: dict[str, Captured]) -> None:
        require(
            not any(n == "sfora" or n.startswith("sfora.") for n in sys.modules),
            "foreign package registry entry",
        )
        sys.meta_path.insert(0, self)
        for role, name in SOURCES.items():
            self.check()
            source = sources[role]
            module = types.ModuleType(name)
            module.__file__ = str(source.path)
            module.__package__ = "sfora"
            module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=name == "sfora")
            if name == "sfora":
                module.__path__ = []
            require(name not in sys.modules, "foreign package registry entry")
            self.owned[name] = module
            sys.modules[name] = module
            exec(
                compile(source.raw, str(source.path), "exec", dont_inherit=True, optimize=0),
                module.__dict__,
            )
            if name != "sfora":
                setattr(self.owned["sfora"], name.rsplit(".", 1)[1], module)
            self.check()

    def check(self) -> None:
        present = {n for n in sys.modules if n == "sfora" or n.startswith("sfora.")}
        require(
            present == self.owned.keys()
            and all(sys.modules.get(n) is m for n, m in self.owned.items()),
            "package registry ownership changed",
        )

    def close(self) -> None:
        for name, module in self.owned.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
        sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not self]


def authority(raw: bytes) -> dict[str, Any]:
    result = obj(
        parse(raw),
        {"schema", "files", "inputs", "python", "output", "resource_policy", "reference"},
    )
    require(
        result["schema"] == "connected-gallery-fragment-authority-v1", "authority schema differs"
    )
    files = obj(result["files"], FILE_ROLES)
    inputs = obj(result["inputs"], INPUT_ROLES)
    for row in [*files.values(), *inputs.values(), result["python"]]:
        descriptor = obj(row, {"path", "sha256"})
        path(descriptor["path"])
        sha(descriptor["sha256"])
    require(files["driver"]["path"] == str(path(__file__)), "driver path differs")
    require(
        result["python"]["path"] == str(Path(sys.executable).resolve()), "interpreter path differs"
    )
    output = path(result["output"], new=True)
    require(not os.path.lexists(output), "output already exists")
    policy = obj(result["resource_policy"], {"seconds", "address_space_bytes"})
    require(
        type(policy["seconds"]) is int
        and policy["seconds"] == SECONDS
        and type(policy["address_space_bytes"]) is int
        and policy["address_space_bytes"] == ADDRESS_SPACE,
        "resource policy differs",
    )
    reference = obj(result["reference"], {"gallery_count", "wire_sha256", "ordered_ids_sha256"})
    require(
        type(reference["gallery_count"]) is int and reference["gallery_count"] > 0,
        "positive builtin gallery count required",
    )
    sha(reference["wire_sha256"])
    sha(reference["ordered_ids_sha256"])
    return result


def verify_members(
    members: object, inputs: dict[str, Captured], reference: dict[str, Any]
) -> dict[str, Any]:
    converted = obj(members, MEMBERS)
    require(all(type(v) is bytes for v in converted.values()), "exact member bytes required")
    for name, role in (
        ("origin.json", "bundle"),
        ("origin-export.json", "receipt"),
        ("origin-owners.json", "ownership_audit"),
    ):
        require(converted[name] == inputs[role].raw, "original bytes changed")
    receipt = obj(parse(inputs["receipt"].raw))
    bundle = obj(parse(inputs["bundle"].raw))
    gallery_batches = [b for b in receipt["images"] if b["role"] == "gallery"]
    rows = [row for batch in gallery_batches for row in batch["rows"]]
    batch_sizes = [len(b["rows"]) for b in gallery_batches]
    require(
        bool(rows)
        and batch_sizes == receipt["batch_sizes"]["gallery"]
        and all(n == 32 for n in batch_sizes[:-1])
        and 1 <= batch_sizes[-1] <= 32,
        "original B32/tail membership differs",
    )
    all_rows = [row for batch in receipt["images"] for row in batch["rows"]]
    count = len(all_rows)
    require(
        sorted(r["panel_ordinal"] for r in all_rows) == list(range(count))
        and all(type(r["panel_ordinal"]) is int for r in all_rows),
        "original panel membership differs",
    )
    require(inputs["combined_wire"].info.st_size == count * 130, "original wire size differs")
    require(
        digest(canonical(sorted(all_rows, key=lambda r: r["panel_ordinal"])))
        == receipt["ordered_images_sha256"],
        "original row order digest differs",
    )
    require(len(converted["gallery.bin"]) == len(rows) * 130, "gallery byte count differs")
    ordinals = [row["panel_ordinal"] for row in rows]
    require(ordinals == sorted(set(ordinals)), "gallery order differs")
    for index, ordinal in enumerate(ordinals):
        original = os.pread(inputs["combined_wire"].fd, 130, ordinal * 130)
        require(
            len(original) == 130
            and original == converted["gallery.bin"][index * 130 : (index + 1) * 130],
            "selected row differs at gallery index " + str(index),
        )
    ids = [row["relative_path"] for row in rows]
    require(
        all(type(i) is str and bool(i) for i in ids) and len(set(ids)) == len(ids),
        "original gallery IDs differ",
    )
    require(converted["gallery-ids.json"] == canonical(ids), "gallery ID order differs")
    observed = {
        "gallery_count": len(rows),
        "wire_sha256": digest(converted["gallery.bin"]),
        "ordered_ids_sha256": digest(converted["gallery-ids.json"]),
    }
    require(canonical(observed) == canonical(reference), "independent derived reference differs")
    producer = {
        "schema": "connected-gallery-producer-binding-v1",
        "original_receipt_sha256": inputs["receipt"].sha256,
        "original_bundle_sha256": inputs["bundle"].sha256,
        "identity": {
            **receipt["payload_facts"],
            "numerical_flags": receipt["numerical_flags"],
            "bundle_files": bundle["files"],
        },
        "combined_wire": {
            "files": receipt["files"],
            "count": count,
            "dimensions": 128,
            "bytes_per_row": 130,
            "panel_facts": receipt["panel_facts"],
        },
        "ordered_images_sha256": receipt["ordered_images_sha256"],
        "gallery_batches": [
            {
                **{k: v for k, v in b.items() if k != "rows"},
                "panel_ordinals": [r["panel_ordinal"] for r in b["rows"]],
            }
            for b in gallery_batches
        ],
        "gallery_rows": rows,
        "gallery_wire_sha256": None,
    }
    expected = {
        "schema": "connected-gallery-wire-selection-v1",
        "producer": producer,
        "gallery_wire_sha256": observed["wire_sha256"],
        "ordered_ids_sha256": observed["ordered_ids_sha256"],
    }
    require(
        converted["gallery-provenance.json"] == canonical(expected),
        "original producer rows/batches/identity differ",
    )
    return {
        **observed,
        "selected_rows_checked": len(rows),
        "panel_count": count,
        "panel_ordinals": ordinals,
        "batch_sizes": batch_sizes,
        "reference_kind": "externally supplied derived bytes, not producer observations",
    }


def deadline(started: float) -> None:
    if time.monotonic() - started >= SECONDS:
        raise TimeoutError("source fragment deadline exceeded")


class Output:
    def __init__(self, target: Path) -> None:
        self.path = target
        self.parent = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.fd = -1
        self.owned: dict[str, Any] = {}
        self.info: os.stat_result | None = None

    def create(self) -> None:
        require(
            path(str(self.path.parent)) == self.path.parent
            and identity(self.path.parent.stat()) == identity(os.fstat(self.parent)),
            "output parent changed",
        )
        os.mkdir(self.path.name, mode=0o700, dir_fd=self.parent)
        self.info = os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False)
        self.fd = os.open(
            self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.parent
        )
        self.guard()
        os.fsync(self.parent)

    def guard(self) -> None:
        require(self.info is not None, "missing output ownership")
        require(
            path(str(self.path)) == self.path
            and identity(self.path.parent.stat()) == identity(os.fstat(self.parent))
            and identity(self.path.lstat())
            == identity(os.fstat(self.fd))
            == identity(cast(os.stat_result, self.info))
            and stat.S_IMODE(os.fstat(self.fd).st_mode) == 0o700,
            "output directory ownership changed",
        )

    def verify(self, members: dict[str, bytes]) -> dict[str, object]:
        self.guard()
        require(set(os.listdir(self.fd)) == MEMBERS, "output member inventory differs")
        facts: dict[str, object] = {}
        for name, retained in self.owned.items():
            raw, observed, info = read_fd(retained.descriptor, capture=True)
            current = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            require(
                identity(info) == identity(current) == retained.identity
                and stat.S_ISREG(current.st_mode)
                and current.st_nlink == info.st_nlink == 1
                and current.st_size == info.st_size == retained.size == len(members[name])
                and raw == retained.payload == members[name]
                and observed == digest(members[name]),
                "published member bytes or ownership differ: " + name,
            )
            facts[name] = {"path": str(self.path / name), "bytes": len(raw), "sha256": observed}
        self.guard()
        return facts

    def abort(self, attempt: Callable[[Callable[[], Any]], None]) -> None:
        for name, retained in self.owned.items():

            def unlink(name: str = name, retained: Any = retained) -> None:
                try:
                    current = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
                except FileNotFoundError:
                    return
                if identity(current) == retained.identity:
                    os.unlink(name, dir_fd=self.fd)

            attempt(unlink)
        if self.fd >= 0:
            attempt(lambda: os.fsync(self.fd))
        if self.info is not None:

            def remove_directory() -> None:
                try:
                    current = os.stat(self.path.name, dir_fd=self.parent, follow_symlinks=False)
                except FileNotFoundError:
                    return
                if identity(current) == identity(cast(os.stat_result, self.info)):
                    os.rmdir(self.path.name, dir_fd=self.parent)
                    os.fsync(self.parent)

            attempt(remove_directory)


def run(authority_path: str, authority_sha256: str, started: float) -> dict[str, Any]:
    captured: list[Captured] = []
    modules = SourceModules()
    output: Output | None = None
    primary: BaseException | None = None
    report: dict[str, Any] = {}

    def capture(row: object, *, data: bool = True) -> Captured:
        result = Captured(row, capture=data)
        captured.append(result)
        return result

    def attempt(action: Callable[[], Any]) -> None:
        nonlocal primary
        try:
            action()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note("cleanup: " + repr(error))

    try:
        deadline(started)
        auth = capture({"path": authority_path, "sha256": authority_sha256})
        spec = authority(auth.raw)
        sources = {role: capture(row) for role, row in spec["files"].items()}
        interpreter = capture(spec["python"], data=False)
        inputs = {role: capture(row) for role, row in spec["inputs"].items()}
        deadline(started)
        # Every pin is checked before the first byte of package source executes.
        modules.load(sources)
        converter = modules.owned[SOURCES["conversion"]]
        members: dict[str, bytes] = converter.extract_gallery_members(
            inputs["receipt"].raw,
            inputs["bundle"].raw,
            inputs["ownership_audit"].raw,
            inputs["combined_wire"].raw,
            trusted_receipt_sha256=inputs["receipt"].sha256,
            trusted_bundle_sha256=inputs["bundle"].sha256,
            trusted_ownership_audit_sha256=inputs["ownership_audit"].sha256,
        )
        observed = verify_members(members, inputs, spec["reference"])
        modules.check()
        for original in captured:
            original.fresh()
        deadline(started)
        output = Output(path(spec["output"], new=True))
        output.create()
        publisher = modules.owned[SOURCES["publication"]]
        for name, raw in members.items():
            deadline(started)
            output.guard()

            def validate(payload: bytes, expected: bytes = raw) -> None:
                require(payload == expected, "publication validator bytes differ")

            output.owned[name] = publisher.publish_bytes_noreplace(
                output.path / name, raw, validator=validate
            )
            output.guard()
        facts = output.verify(members)
        deadline(started)
        report = {
            "schema": "connected-gallery-fragment-result-v1",
            "procedure": "source-owned procedure execution",
            "authority": auth.facts(),
            "sources": {role: file.facts() for role, file in sources.items()},
            "inputs": {role: file.facts() for role, file in inputs.items()},
            "python": {**interpreter.facts(), "version": sys.version},
            "output": str(output.path),
            "members": facts,
            "observed": observed,
            "resource_policy": spec["resource_policy"],
            "normal_terminal_required": True,
            "native_verified": False,
            "serving_accepted": False,
            "quality_speed_goal_met": False,
        }
    except BaseException as error:
        primary = error
    finally:
        for original in captured:
            attempt(original.fresh)
        if modules.owned:
            attempt(modules.check)
        if primary is None and output is not None:
            attempt(lambda: output.verify(members))
        attempt(lambda: deadline(started))
        attempt(modules.close)
        for original in captured:
            attempt(original.close)
        if output is not None:
            already_failed = primary is not None
            if already_failed:
                output.abort(attempt)
            for retained in output.owned.values():
                attempt(retained.close)
            if not already_failed and primary is not None:
                output.abort(attempt)
            if output.fd >= 0:
                attempt(lambda: os.close(output.fd))
            attempt(lambda: os.close(output.parent))
    if primary is not None:
        raise primary
    deadline(started)
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return report


def expired(signum: int, frame: types.FrameType | None) -> NoReturn:
    raise TimeoutError("source fragment deadline exceeded")


def main() -> int:
    started = time.monotonic()
    try:
        require(
            sys.platform == "linux" and (3, 12) <= sys.version_info[:2] <= (3, 14),
            "supported Linux Python 3.12/3.13/3.14 required",
        )
        require(
            sys.flags.isolated == 1
            and sys.flags.no_site == 1
            and sys.flags.dont_write_bytecode == 1
            and sys.flags.optimize == 0
            and not sys._xoptions
            and not sys.warnoptions,
            "canonical -I -S -B flags required",
        )
        require(
            sys.orig_argv[:5] == [str(Path(sys.executable).resolve()), "-I", "-S", "-B", __file__],
            "canonical interpreter/driver invocation required",
        )
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        bound = min(n for n in (soft, hard, ADDRESS_SPACE) if n != resource.RLIM_INFINITY)
        resource.setrlimit(resource.RLIMIT_AS, (bound, bound))
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, SECONDS - (time.monotonic() - started)))
        parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
        parser.add_argument("--authority", required=True)
        parser.add_argument("--authority-sha256", required=True)
        args = parser.parse_args()
        require(
            sys.argv[1:]
            == ["--authority", args.authority, "--authority-sha256", args.authority_sha256],
            "exact canonical CLI required",
        )
        report = run(args.authority, args.authority_sha256, started)
        deadline(started)
        print(canonical(report).decode(), flush=True)
        return 0
    except Exception as error:
        print("fragment gate failed: " + str(error), file=sys.stderr)
        for note in getattr(error, "__notes__", ()):
            print(note, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
