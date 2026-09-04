"""One crash-durable, race-safe protocol for immutable file publication."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

Validator = Callable[[bytes], None]
LargeValidator = Callable[[int, int], None]
Writer = Callable[[int], None]


@dataclass
class PublishedFile:
    """The exact bytes and retained inode identity accepted by publication."""

    payload: bytes
    identity: tuple[int, int]
    size: int
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> PublishedFile:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


@dataclass
class PublishedLargeFile:
    """A retained immutable file identity without a materialized byte payload."""

    identity: tuple[int, int]
    size: int
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def __enter__(self) -> PublishedLargeFile:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


class BudgetedPublisher:
    """Reload one externally rooted publication budget at every write boundary."""

    def __init__(
        self,
        *,
        campaign_root: Path,
        budget_path: Path,
        budget_sha256: str,
        exact_budget: Mapping[str, object],
        statvfs: Callable[[Path], object] = os.statvfs,
        physical_admission: bool = True,
    ) -> None:
        self.root = campaign_root.resolve()
        self.budget_path = budget_path.absolute()
        self.budget_sha256 = budget_sha256
        self.exact_budget = dict(exact_budget)
        self.statvfs = statvfs
        self.physical_admission = physical_admission

    def _row(self, name: str, destination: Path) -> Mapping[str, object]:
        descriptor = os.open(self.budget_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if not os.path.isfile(f"/proc/self/fd/{descriptor}"):
                raise ValueError("publication budget file differs")
            payload = _pread_all(descriptor)
            current = self.budget_path.lstat()
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("publication budget ownership differs")
        finally:
            os.close(descriptor)
        if hashlib.sha256(payload).hexdigest() != self.budget_sha256:
            raise ValueError("publication budget bytes differ")
        persisted = json.loads(payload)
        if persisted != self.exact_budget:
            raise ValueError("exact publication budget differs")
        rows = persisted.get("publications") if type(persisted) is dict else None
        try:
            relative = destination.resolve().relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("publication destination path differs") from error
        matching = [
            row
            for row in rows or []
            if type(row) is dict
            and row.get("name") == name
            and row.get("path") == relative
        ]
        if len(matching) != 1:
            raise ValueError("publication budget row differs")
        row = matching[0]
        registered = {
            candidate["path"]: candidate
            for candidate in rows or []
            if type(candidate) is dict and type(candidate.get("path")) is str
        }
        if len(registered) != len(rows or []):
            raise ValueError("publication budget path inventory differs")
        existing: set[str] = set()
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError("publication root contains a symlink")
            relative_path = path.relative_to(self.root).as_posix()
            if path.is_dir() or path == self.budget_path:
                if relative_path in registered:
                    existing.add(relative_path)
                continue
            if relative_path not in registered:
                raise ValueError("publication root contains an unregistered path")
            existing.add(relative_path)
            registered_row = registered[relative_path]
            registered_bound = (
                registered_row["persistent_bytes"]
                or registered_row["temporary_bytes"]
            )
            if path.is_file() and path.stat().st_size > registered_bound:
                raise OSError(errno.EFBIG, "persisted publication exceeds budget")
        remaining_rows = [
            candidate
            for candidate in rows or []
            if candidate["path"] not in existing
        ]
        required_bytes = sum(
            candidate["persistent_bytes"] + candidate["temporary_bytes"]
            for candidate in remaining_rows
        )
        required_inodes = sum(
            candidate["persistent_inodes"] + candidate["temporary_inodes"]
            for candidate in remaining_rows
        )
        if self.physical_admission:
            statistics = self.statvfs(self.root)
            if (
                statistics.f_bavail * statistics.f_frsize < required_bytes
                or statistics.f_favail < required_inodes
            ):
                raise OSError(errno.ENOSPC, "publication capacity is insufficient")
        return row

    def publish_bytes(
        self,
        *,
        name: str,
        destination: Path,
        payload: bytes,
        validator: Validator,
    ) -> PublishedFile:
        self.validate_payload(name=name, destination=destination, payload=payload)
        return publish_bytes_noreplace(destination, payload, validator=validator)

    def validate_payload(
        self, *, name: str, destination: Path, payload: bytes
    ) -> Mapping[str, object]:
        return self.validate_size(
            name=name, destination=destination, size=len(payload)
        )

    def validate_size(
        self, *, name: str, destination: Path, size: int
    ) -> Mapping[str, object]:
        if type(size) is not int or size < 0:
            raise ValueError("publication payload size differs")
        row = self._row(name, destination)
        bound = row["persistent_bytes"] or row["temporary_bytes"]
        if size > bound:
            raise OSError(errno.EFBIG, "publication payload bytes exceed budget")
        return row

    def publish_writer(
        self,
        *,
        name: str,
        destination: Path,
        writer: Writer,
        validator: Validator,
    ) -> PublishedFile:
        row = self._row(name, destination)
        bound = row["persistent_bytes"] or row["temporary_bytes"]

        def bounded(payload: bytes) -> None:
            if len(payload) > bound:
                raise OSError(errno.EFBIG, "publication payload bytes exceed budget")
            validator(payload)

        return publish_writer_noreplace(destination, writer, validator=bounded)


def _link_fd_noreplace(descriptor: int, directory: int, name: str) -> None:
    linkat = getattr(ctypes.CDLL(None, use_errno=True), "linkat", None)
    if linkat is None:
        raise RuntimeError("linkat is required for immutable publication")
    destination = os.fsencode(name)
    if not linkat(descriptor, b"", directory, destination, 0x1000):
        return
    error = ctypes.get_errno()
    if error in {errno.EPERM, errno.ENOENT}:
        source = os.fsencode(f"/proc/self/fd/{descriptor}")
        if not linkat(-100, source, directory, destination, 0x400):
            return
        error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(name)
    raise OSError(error, os.strerror(error), name)


def _pread_all(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(size - offset, 1024 * 1024), offset)
        if not chunk:
            raise RuntimeError("immutable publication read was truncated")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _descriptors_equal(left: int, right: int, size: int) -> bool:
    """Compare two exact file images with bounded resident memory."""

    if type(size) is not int or size < 0:
        raise ValueError("immutable publication size differs")
    if os.fstat(left).st_size != size or os.fstat(right).st_size != size:
        return False
    offset = 0
    while offset < size:
        length = min(size - offset, 1024 * 1024)
        left_chunk = os.pread(left, length, offset)
        right_chunk = os.pread(right, length, offset)
        if len(left_chunk) != length or len(right_chunk) != length:
            raise RuntimeError("immutable publication read was truncated")
        if left_chunk != right_chunk:
            return False
        offset += length
    return True


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise RuntimeError("immutable publication write made no progress")
        offset += written


def publish_writer_noreplace(
    path: Path,
    writer: Writer,
    *,
    validator: Validator,
) -> PublishedFile:
    """Publish one absent file and return its distinctly reopened exact bytes."""

    if not isinstance(path, Path) or path.name in {"", ".", ".."}:
        raise ValueError("immutable publication path differs")
    parent = path.parent
    parent_info = parent.lstat()
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("immutable publication parent differs")
    if os.path.lexists(path):
        raise FileExistsError(path)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    descriptor: int | None = None
    published = False
    completed = False
    owned: tuple[int, int] | None = None
    retained_descriptor: int | None = None
    try:
        if (os.fstat(directory).st_dev, os.fstat(directory).st_ino) != (
            parent_info.st_dev,
            parent_info.st_ino,
        ):
            raise RuntimeError("immutable publication parent ownership differs")
        descriptor = os.open(parent, os.O_RDWR | os.O_TMPFILE, 0o600)
        writer(descriptor)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        payload = _pread_all(descriptor)
        validator(payload)
        try:
            _link_fd_noreplace(descriptor, directory, path.name)
        except Exception:
            try:
                linked_info = path.lstat()
            except FileNotFoundError:
                pass
            else:
                published = (linked_info.st_dev, linked_info.st_ino) == owned
            raise
        published = True
        os.fsync(directory)
        reopened = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            reopened_info = os.fstat(reopened)
            reopened_payload = _pread_all(reopened)
            retained_descriptor = os.dup(reopened)
        finally:
            os.close(reopened)
        final_info = path.lstat()
        if (
            (reopened_info.st_dev, reopened_info.st_ino) != owned
            or (final_info.st_dev, final_info.st_ino) != owned
            or reopened_payload != payload
        ):
            raise RuntimeError("immutable publication inode ownership differs")
        # A second directory barrier makes the completed ownership decision
        # durable independently of the link-creation barrier.
        os.fsync(directory)
        final_info = path.lstat()
        if (final_info.st_dev, final_info.st_ino) != owned:
            raise RuntimeError("immutable publication final ownership differs")
        completed = True
        return PublishedFile(
            payload=reopened_payload,
            identity=owned,
            size=len(reopened_payload),
            descriptor=retained_descriptor,
        )
    finally:
        if published and not completed and owned is not None and os.path.lexists(path):
            info = path.lstat()
            if (info.st_dev, info.st_ino) == owned:
                path.unlink()
                os.fsync(directory)
        if descriptor is not None:
            os.close(descriptor)
        if not completed and retained_descriptor is not None:
            os.close(retained_descriptor)
        os.close(directory)


def publish_large_writer_noreplace(
    path: Path,
    writer: Writer,
    *,
    validator: LargeValidator,
) -> PublishedLargeFile:
    """Publish one absent large file without materializing its encoded bytes."""

    if not isinstance(path, Path) or path.name in {"", ".", ".."}:
        raise ValueError("immutable publication path differs")
    parent = path.parent
    parent_info = parent.lstat()
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("immutable publication parent differs")
    if os.path.lexists(path):
        raise FileExistsError(path)
    directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    descriptor: int | None = None
    published = False
    completed = False
    owned: tuple[int, int] | None = None
    retained_descriptor: int | None = None
    try:
        if (os.fstat(directory).st_dev, os.fstat(directory).st_ino) != (
            parent_info.st_dev,
            parent_info.st_ino,
        ):
            raise RuntimeError("immutable publication parent ownership differs")
        descriptor = os.open(parent, os.O_RDWR | os.O_TMPFILE, 0o600)
        writer(descriptor)
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        read_descriptor = os.open(
            f"/proc/self/fd/{descriptor}", os.O_RDONLY | os.O_CLOEXEC
        )
        try:
            validator(read_descriptor, info.st_size)
        finally:
            os.close(read_descriptor)
        try:
            _link_fd_noreplace(descriptor, directory, path.name)
        except Exception:
            try:
                linked_info = path.lstat()
            except FileNotFoundError:
                pass
            else:
                published = (linked_info.st_dev, linked_info.st_ino) == owned
            raise
        published = True
        os.fsync(directory)
        reopened = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            reopened_info = os.fstat(reopened)
            same_bytes = _descriptors_equal(descriptor, reopened, info.st_size)
            retained_descriptor = os.dup(reopened)
        finally:
            os.close(reopened)
        final_info = path.lstat()
        if (
            (reopened_info.st_dev, reopened_info.st_ino) != owned
            or (final_info.st_dev, final_info.st_ino) != owned
            or not same_bytes
        ):
            raise RuntimeError("immutable publication inode ownership differs")
        os.fsync(directory)
        final_info = path.lstat()
        if (final_info.st_dev, final_info.st_ino) != owned:
            raise RuntimeError("immutable publication final ownership differs")
        completed = True
        return PublishedLargeFile(
            identity=owned,
            size=info.st_size,
            descriptor=retained_descriptor,
        )
    finally:
        if published and not completed and owned is not None and os.path.lexists(path):
            final = path.lstat()
            if (final.st_dev, final.st_ino) == owned:
                path.unlink()
                os.fsync(directory)
        if descriptor is not None:
            os.close(descriptor)
        if not completed and retained_descriptor is not None:
            os.close(retained_descriptor)
        os.close(directory)


def publish_bytes_noreplace(
    path: Path, payload: bytes, *, validator: Validator
) -> PublishedFile:
    if type(payload) is not bytes:
        raise TypeError("immutable publication payload must be bytes")
    return publish_writer_noreplace(
        path, lambda descriptor: _write_all(descriptor, payload), validator=validator
    )
