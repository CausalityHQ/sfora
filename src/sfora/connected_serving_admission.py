"""Fresh filesystem admission of one serving artifact directory; nothing more.

Exactly ten prescribed names, descriptor-relative beneath a nonsymlink directory chain, each
a regular single-link file. Pins for the manifest and the six fragment files come from the
caller, never from serving.json. Payloads are streamed against the bound manifest's declared
size/SHA and are never captured or deserialized; original paths are never opened.

Each file is freshly read in sequence and the sequential final identities are rechecked; this
is NOT an atomic cross-file or directory snapshot (the OS gives no such guarantee here) and
does not protect against later mutation. Any later use needs fresh per-call guards. It does
not authenticate typed payload contents, installed sources, import resolution, native
execution, serving parity, quality or speed, and it caches nothing, so every call rereads all
ten files.

Errors: validation failures raise ValueError; OSError is converted to ValueError (cause kept);
any other BaseException propagates unchanged. A descriptor close failure is never swallowed:
it raises when nothing else failed, and is recorded as a note on the primary otherwise.
Memory: payload streaming is O(1 MiB). Captured metadata (serving.json and the six fragment
files, including opaque origin-owners.json and gallery.bin) is capped at 64 MiB EACH by this
module, so the worst case is 7 x 64 MiB plus a chunk join and the binder's JSON expansion.
"""

import hashlib
import os
import stat
from types import TracebackType
from typing import Self

from .connected_gallery_provenance import JSONObject, _count, _object, _path, _require, _sha
from .connected_serving_artifact import _FRAGMENT, _PAYLOAD, bind_serving_manifest

_BLOCK = 1024 * 1024
_METADATA_LIMIT = 64 * 1024 * 1024
_METADATA = ("serving.json", *_FRAGMENT)
_NAMES = (*_METADATA, *_PAYLOAD)
_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_NOCTTY | os.O_CLOEXEC


class _Owned:
    """Sole close site: each descriptor is popped before its one close attempt."""

    def __init__(self) -> None:
        self.fds: list[int] = []

    def open(self, path: str, flags: int, dir_fd: int | None = None) -> int:
        fd = os.open(path, flags, dir_fd=dir_fd)
        self.fds.append(fd)
        return fd

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        primary: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        failure: BaseException | None = None
        while self.fds:
            try:
                os.close(self.fds.pop())
            except BaseException as cleanup:  # attempt every close exactly once
                if primary is not None:
                    primary.add_note("descriptor close failed: " + repr(cleanup))
                elif failure is None:
                    failure = cleanup
                else:
                    failure.add_note("descriptor close failed: " + repr(cleanup))
        if failure is not None:
            raise failure


def _key(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        info.st_mode,
    )


def _walk(owned: _Owned, parts: list[str]) -> tuple[int, list[tuple[int, int]]]:
    fd = owned.open("/", _DIRECTORY)
    info = os.fstat(fd)
    chain = [(info.st_dev, info.st_ino)]
    for part in parts:
        fd = owned.open(part, _DIRECTORY, fd)
        info = os.fstat(fd)
        chain.append((info.st_dev, info.st_ino))
    return fd, chain


def _members(dirfd: int, message: str) -> None:
    _require(set(os.listdir(dirfd)) == set(_NAMES), message)


def _read(
    dirfd: int, name: str, expected: tuple[int, str] | None
) -> tuple[tuple[int, ...], str, bytes]:
    """Stream one file afresh; capture it only when no expectation is given (metadata)."""
    pre = os.lstat(name, dir_fd=dirfd)
    _require(
        stat.S_ISREG(pre.st_mode) and pre.st_nlink == 1,
        f"{name}: regular single-link file required",
    )
    if expected is None:
        _require(pre.st_size <= _METADATA_LIMIT, f"{name} exceeds metadata bound")
    else:
        _require(pre.st_size == expected[0], f"{name} byte count differs")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    count = 0
    with _Owned() as leaf:
        fd = leaf.open(name, _FILE, dirfd)
        # The stream never owns the fd, including after a partial construction failure.
        with os.fdopen(fd, "rb", closefd=False) as stream:
            before = os.fstat(fd)
            _require(_key(before) == _key(pre), f"{name} changed before read")
            while block := stream.read(_BLOCK):
                count += len(block)
                _require(count <= before.st_size, f"{name} grew while reading")
                digest.update(block)
                if expected is None:
                    chunks.append(block)
            after = os.fstat(fd)
            end = os.lstat(name, dir_fd=dirfd)
    key = _key(pre)
    _require(
        count == pre.st_size and _key(after) == key and _key(end) == key,
        f"{name} changed while reading",
    )
    if expected is not None:
        _require(digest.hexdigest() == expected[1], f"{name} SHA differs")
    return key, digest.hexdigest(), b"".join(chunks)


def _fact(root: str, name: str, key: tuple[int, ...], digest: str) -> JSONObject:
    return {
        "path": f"{root}/{name}",
        "bytes": key[2],
        "sha256": digest,
        "device": key[0],
        "inode": key[1],
        "mtime_ns": key[3],
        "ctime_ns": key[4],
    }


def _admit(
    owned: _Owned, root: str, parts: list[str], serving_pin: str, pins: dict[str, str]
) -> JSONObject:
    dirfd, chain = _walk(owned, parts)
    _members(dirfd, "exact ten prescribed names required")
    keys: dict[str, tuple[int, ...]] = {}
    facts: JSONObject = {}
    captured: dict[str, bytes] = {}
    for name in _METADATA:
        keys[name], digest, captured[name] = _read(dirfd, name, None)
        facts[name] = _fact(root, name, keys[name], digest)
    serving = captured.pop("serving.json")
    bound = bind_serving_manifest(
        serving, captured, trusted_serving_sha256=serving_pin, trusted_fragment_sha256=pins
    )
    declared = _object(_object(bound["manifest"])["files"])
    for name in _PAYLOAD:
        file = _object(declared[name])
        keys[name], digest, _ = _read(dirfd, name, (_count(file["bytes"]), _sha(file["sha256"])))
        facts[name] = _fact(root, name, keys[name], digest)
    _members(dirfd, "directory membership changed during admission")
    for name in _NAMES:
        _require(
            _key(os.lstat(name, dir_fd=dirfd)) == keys[name],
            f"{name} replaced or changed during admission",
        )
    _require(_walk(owned, parts)[1] == chain, "directory chain replaced during admission")
    return {
        "manifest": bound["manifest"],
        "producer": bound["producer"],
        "directory": {"path": root, "device": chain[-1][0], "inode": chain[-1][1]},
        "files": facts,
    }


def admit_serving_artifact(
    directory: str,
    *,
    trusted_serving_sha256: str,
    trusted_fragment_sha256: dict[str, str],
) -> JSONObject:
    """Return detached binder metadata plus fresh per-file byte facts and guards.

    ``directory`` must be a canonical absolute POSIX path whose entries are exactly the ten
    prescribed names, checked before any member is read and again before returning. Pins are
    validated before any filesystem access. The result holds
    ``manifest``/``producer`` (from bind_serving_manifest), ``directory`` (path, device,
    inode) and ``files`` (path, bytes, sha256, device, inode, mtime_ns, ctime_ns per name).
    """
    root = _path(directory, absolute=True)
    serving_pin = _sha(trusted_serving_sha256)
    _require(
        type(trusted_fragment_sha256) is dict and trusted_fragment_sha256.keys() == set(_FRAGMENT),
        "exact fragment pin names required",
    )
    pins = {name: _sha(pin) for name, pin in trusted_fragment_sha256.items()}
    try:
        with _Owned() as owned:
            return _admit(owned, root, root[1:].split("/"), serving_pin, pins)
    except OSError as error:
        raise ValueError(f"serving artifact admission I/O failed: {error}") from error
