from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import json
import os
import traceback
from collections.abc import Callable
from pathlib import Path, PosixPath
from typing import Any

import pytest


def test_large_writer_publication_validates_and_compares_without_materializing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    destination = tmp_path / "checkpoint.pt"
    payload = b"registered-checkpoint" * 257
    validations: list[tuple[int, bytes]] = []

    monkeypatch.setattr(
        publication,
        "_pread_all",
        lambda _descriptor: (_ for _ in ()).throw(
            AssertionError("large publication materialized its payload")
        ),
    )

    def writer(descriptor: int) -> None:
        publication._write_all(descriptor, payload)

    def validator(descriptor: int, size: int) -> None:
        validations.append((size, os.pread(descriptor, size, 0)))

    with publication.publish_large_writer_noreplace(
        destination, writer, validator=validator
    ) as published:
        assert validations == [(len(payload), payload)]
        assert published.size == len(payload)
        assert published.identity == (
            destination.lstat().st_dev,
            destination.lstat().st_ino,
        )
        assert os.pread(published.descriptor, len(payload), 0) == payload


def test_review7_retained_publisher_loops_writes_and_fsyncs_directory_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short write or crash cannot weaken the one shared publication protocol."""

    publication = importlib.import_module("sfora.atomic_publication")
    original_write = publication.os.write
    original_fsync = publication.os.fsync
    directory_syncs = 0

    def short_write(descriptor: int, payload: bytes) -> int:
        return original_write(descriptor, payload[: max(1, len(payload) // 3)])

    def count_sync(descriptor: int) -> None:
        nonlocal directory_syncs
        if os.path.isdir(f"/proc/self/fd/{descriptor}"):
            directory_syncs += 1
        original_fsync(descriptor)

    monkeypatch.setattr(publication.os, "write", short_write)
    monkeypatch.setattr(publication.os, "fsync", count_sync)
    destination = tmp_path / "authority.json"
    payload = b'{"registered":true}\n'
    publication.publish_bytes_noreplace(
        destination, payload, validator=lambda persisted: persisted == payload or None
    )
    assert destination.read_bytes() == payload
    assert directory_syncs == 2


def test_review7_retained_publisher_preserves_racer_owned_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    destination = tmp_path / "authority.json"
    original_link = publication._link_fd_noreplace

    def race(descriptor: int, directory: int, name: str) -> None:
        destination.write_bytes(b"racer")
        original_link(descriptor, directory, name)

    monkeypatch.setattr(publication, "_link_fd_noreplace", race)
    with pytest.raises(FileExistsError):
        publication.publish_bytes_noreplace(
            destination,
            b"owned\n",
            validator=lambda persisted: persisted == b"owned\n" or None,
        )
    assert destination.read_bytes() == b"racer"


def test_review8_publication_requires_semantics_and_returns_retained_identity(
    tmp_path: Path,
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    destination = tmp_path / "authority.json"
    with pytest.raises(TypeError, match="validator|semantic"):
        publication.publish_bytes_noreplace(destination, b'{"registered":true}\n')
    assert not destination.exists()

    published = publication.publish_bytes_noreplace(
        destination,
        b'{"registered":true}\n',
        validator=lambda payload: (
            None
            if payload == b'{"registered":true}\n'
            else (_ for _ in ()).throw(ValueError("semantic mismatch"))
        ),
    )
    assert published.payload == destination.read_bytes()
    info = destination.lstat()
    assert published.identity == (info.st_dev, info.st_ino)


def test_review9_published_file_context_closes_retained_descriptor(tmp_path: Path) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    with publication.publish_bytes_noreplace(
        tmp_path / "authority.json",
        b'{"registered":true}\n',
        validator=lambda _payload: None,
    ) as published:
        descriptor = published.descriptor
        os.fstat(descriptor)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_review10_budgeted_publisher_reloads_exact_row_and_checks_actual_bytes(
    tmp_path: Path,
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    root = tmp_path / "campaign"
    (root / "preflight").mkdir(parents=True)
    destination = root / "stage/terminal.json"
    destination.parent.mkdir()
    budget = {
        "schema": "unicom-fepf-publication-budget-v1",
        "publications": [{
            "name": "stage:terminal",
            "path": "stage/terminal.json",
            "persistent_bytes": 4,
            "temporary_bytes": 4,
            "persistent_inodes": 1,
            "temporary_inodes": 1,
        }],
    }
    payload = (json.dumps(budget, indent=2) + "\n").encode()
    budget_path = root / "preflight/publication-budget.json"
    budget_path.write_bytes(payload)
    publisher = publication.BudgetedPublisher(
        campaign_root=root,
        budget_path=budget_path,
        budget_sha256=hashlib.sha256(payload).hexdigest(),
        exact_budget=budget,
    )
    with pytest.raises(OSError, match="bytes|budget"):
        publisher.publish_bytes(
            name="stage:terminal",
            destination=destination,
            payload=b"12345",
            validator=lambda _payload: None,
        )
    assert not destination.exists()


def test_review11_semantic_validator_runs_once_with_postlink_identity_checks(
    tmp_path: Path,
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    calls: list[bytes] = []
    payload = b'{"registered":true}\n'
    destination = tmp_path / "authority.json"

    published = publication.publish_bytes_noreplace(
        destination,
        payload,
        validator=lambda candidate: calls.append(candidate),
    )
    try:
        assert calls == [payload]
        assert published.payload == destination.read_bytes()
        assert published.identity == (
            destination.lstat().st_dev,
            destination.lstat().st_ino,
        )
    finally:
        published.close()


def test_review14_linkat_empty_path_capability_has_safe_noreplace_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    real_library = ctypes.CDLL(None, use_errno=True)
    real_linkat = real_library.linkat
    observed: list[tuple[bytes, int]] = []

    class RestrictedLibc:
        @staticmethod
        def linkat(
            source_directory: int,
            source_name: bytes,
            destination_directory: int,
            destination_name: bytes,
            flags: int,
        ) -> int:
            observed.append((source_name, flags))
            if source_name == b"":
                ctypes.set_errno(errno.EPERM)
                return -1
            return real_linkat(
                source_directory, source_name,
                destination_directory, destination_name, flags,
            )

    monkeypatch.setattr(
        publication.ctypes, "CDLL", lambda *_args, **_kwargs: RestrictedLibc()
    )
    destination = tmp_path / "authority.json"
    payload = b'{"registered":true}\n'
    with publication.publish_bytes_noreplace(
        destination, payload,
        validator=lambda candidate: candidate == payload or None,
    ) as published:
        assert published.payload == payload
        assert published.identity == (
            destination.lstat().st_dev, destination.lstat().st_ino
        )
    assert observed[0] == (b"", 0x1000)
    assert observed[1][0].startswith(b"/proc/self/fd/")
    assert observed[1][1] == 0x400

    destination.write_bytes(b"racer")
    with pytest.raises(FileExistsError):
        publication.publish_bytes_noreplace(
            destination, payload, validator=lambda _candidate: None
        )
    assert destination.read_bytes() == b"racer"


def test_review12_each_write_admits_whole_remaining_campaign_inventory(
    tmp_path: Path,
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    root = tmp_path / "campaign"
    (root / "preflight").mkdir(parents=True)
    stage = root / "stage"
    stage.mkdir()
    rows = [
        {
            "name": f"stage:file-{index}", "path": f"stage/file-{index}.json",
            "persistent_bytes": 4, "temporary_bytes": 4,
            "persistent_inodes": 1, "temporary_inodes": 1,
        }
        for index in range(3)
    ]
    budget = {
        "schema": "unicom-fepf-publication-budget-v1", "publications": rows
    }
    payload = (json.dumps(budget, indent=2) + "\n").encode()
    budget_path = root / "preflight/publication-budget.json"
    budget_path.write_bytes(payload)
    capacities = iter((24, 12))

    def capacity(_root: Path):
        available = next(capacities)
        return type("Capacity", (), {
            "f_bavail": available, "f_frsize": 1, "f_favail": available,
        })()

    publisher = publication.BudgetedPublisher(
        campaign_root=root, budget_path=budget_path,
        budget_sha256=hashlib.sha256(payload).hexdigest(), exact_budget=budget,
        statvfs=capacity,
    )
    first = publisher.publish_bytes(
        name="stage:file-0", destination=stage / "file-0.json", payload=b"1234",
        validator=lambda candidate: candidate == b"1234" or None,
    )
    first.close()
    with pytest.raises(OSError, match="capacity|space"):
        publisher.publish_bytes(
            name="stage:file-1", destination=stage / "file-1.json", payload=b"1234",
            validator=lambda candidate: candidate == b"1234" or None,
        )


# --- cleanup hygiene: genuine O_TMPFILE/linkat publication on tiny temp files -------------------

PAYLOAD = b"opaque-tiny-bytes"
APIS = ("small", "large")


def _fd_table() -> dict[int, str]:
    """Every open descriptor of this process; the listing's own descriptor is closed by then."""

    table: dict[int, str] = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            table[int(name)] = os.readlink(f"/proc/self/fd/{name}")
        except FileNotFoundError:
            continue
    return table


def _publish(
    publication: Any,
    api: str,
    destination: Path,
    writer: Callable[[int], None] | None = None,
) -> Any:
    def write(descriptor: int) -> None:
        publication._write_all(descriptor, PAYLOAD)

    if api == "small":
        return publication.publish_writer_noreplace(
            destination, writer or write, validator=lambda _payload: None
        )
    return publication.publish_large_writer_noreplace(
        destination, writer or write, validator=lambda _descriptor, _size: None
    )


def _raising(error: BaseException) -> Callable[[], None]:
    def hook() -> None:
        raise error

    return hook


class _CleanupInterrupt(BaseException):
    """Stand-in for KeyboardInterrupt/SystemExit: a BaseException that is not an Exception."""


class _UnlinkDenied(PosixPath):
    def unlink(self, missing_ok: bool = False) -> None:
        raise PermissionError(errno.EACCES, "injected unlink denial", str(self))


class _UnlinkInterrupted(PosixPath):
    def unlink(self, missing_ok: bool = False) -> None:
        raise _CleanupInterrupt("injected unlink interrupt")


def _close_eio(role: str) -> BaseException:
    return OSError(errno.EIO, f"injected {role} close failure")


def _close_interrupt(role: str) -> BaseException:
    return _CleanupInterrupt(f"injected {role} close interrupt")


class _FdSpy:
    """Run every real descriptor call, track who owns what, and inject failures around them.

    A failing close is raised after the real close: Linux releases the descriptor even when
    close reports an error, so the publisher must neither retry it nor close it again.
    """

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        directory_fsync_hooks: dict[int, Callable[[], None]] | None = None,
        failing_closes: tuple[str, ...] = (),
        close_failure: Callable[[str], BaseException] = _close_eio,
    ) -> None:
        self.roles: dict[int, str] = {}
        self.stray_closes: list[int] = []
        self.injected: list[BaseException] = []
        self.directory_fsyncs = 0
        hooks = directory_fsync_hooks or {}
        real_open, real_dup, real_close, real_fsync = os.open, os.dup, os.close, os.fsync

        def traced_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            descriptor = real_open(path, flags, *args, **kwargs)
            if flags & os.O_TMPFILE == os.O_TMPFILE:
                self.roles[descriptor] = "tmpfile"
            elif flags & os.O_DIRECTORY:
                self.roles[descriptor] = "directory"
            else:
                self.roles[descriptor] = "scratch"
            return descriptor

        def traced_dup(descriptor: int) -> int:
            duplicate = real_dup(descriptor)
            self.roles[duplicate] = "retained"
            return duplicate

        def traced_fsync(descriptor: int) -> None:
            if self.roles.get(descriptor) == "directory":
                self.directory_fsyncs += 1
                if self.directory_fsyncs in hooks:
                    hooks[self.directory_fsyncs]()
            real_fsync(descriptor)

        def traced_close(descriptor: int) -> None:
            role = self.roles.pop(descriptor, None)
            if role is None:
                self.stray_closes.append(descriptor)
            real_close(descriptor)
            if role in failing_closes:
                error = close_failure(role)
                self.injected.append(error)
                raise error

        monkeypatch.setattr(os, "open", traced_open)
        monkeypatch.setattr(os, "dup", traced_dup)
        monkeypatch.setattr(os, "fsync", traced_fsync)
        monkeypatch.setattr(os, "close", traced_close)

    def assert_every_descriptor_released_once(self, error: BaseException | None = None) -> None:
        if error is not None:
            # A raised error keeps the publisher frames alive; dropping their locals runs any
            # finalizer that would otherwise close a descriptor number a second time, late.
            traceback.clear_frames(error.__traceback__)
        assert self.stray_closes == []
        assert self.roles == {}


@pytest.fixture(scope="module")
def warmed_ctypes(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Python 3.14 libffi keeps a descriptor after first use; open it before any baseline."""

    publication = importlib.import_module("sfora.atomic_publication")
    scratch = tmp_path_factory.mktemp("warm")
    for api in APIS:
        _publish(publication, api, scratch / f"{api}.bin").close()


@pytest.fixture
def fd_baseline(warmed_ctypes: None) -> dict[int, str]:
    return _fd_table()


@pytest.mark.parametrize("api", APIS)
def test_postlink_primary_survives_unlink_failure_and_closes_every_owned_descriptor(
    api: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fd_baseline: dict[int, str]
) -> None:
    """The original falsifier: post-link ValueError, then cleanup unlink PermissionError."""

    publication = importlib.import_module("sfora.atomic_publication")
    primary = ValueError("post-link directory barrier failed")
    spy = _FdSpy(monkeypatch, directory_fsync_hooks={1: _raising(primary)})
    destination = _UnlinkDenied(tmp_path / "authority.bin")

    with pytest.raises(ValueError) as caught:
        _publish(publication, api, destination)

    assert caught.value is primary
    assert len(primary.__notes__) == 1
    assert "PermissionError" in primary.__notes__[0]
    assert destination.read_bytes() == PAYLOAD
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("primary_barrier", [1, 2])
def test_cleanup_barrier_failure_is_noted_on_the_exact_primary(
    api: str,
    primary_barrier: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fd_baseline: dict[int, str],
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    primary = ValueError("directory barrier failed")
    barrier = OSError(errno.EIO, "cleanup barrier failed")
    spy = _FdSpy(
        monkeypatch,
        directory_fsync_hooks={
            primary_barrier: _raising(primary),
            primary_barrier + 1: _raising(barrier),
        },
    )
    destination = tmp_path / "authority.bin"

    with pytest.raises(ValueError) as caught:
        _publish(publication, api, destination)

    assert caught.value is primary
    assert primary.__notes__ == [f"immutable publication cleanup also failed: {barrier!r}"]
    assert not destination.exists()
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    "failing_closes",
    [
        ("tmpfile",),
        ("directory",),
        ("retained",),
        ("tmpfile", "directory", "retained"),
    ],
)
def test_every_owned_close_is_attempted_once_despite_earlier_close_failures(
    api: str,
    failing_closes: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fd_baseline: dict[int, str],
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    primary = ValueError("final directory barrier failed")
    spy = _FdSpy(
        monkeypatch, directory_fsync_hooks={2: _raising(primary)}, failing_closes=failing_closes
    )
    destination = tmp_path / "authority.bin"

    with pytest.raises(ValueError) as caught:
        _publish(publication, api, destination)

    assert caught.value is primary
    assert len(spy.injected) == len(failing_closes)
    assert primary.__notes__ == [
        f"immutable publication cleanup also failed: {error!r}" for error in spy.injected
    ]
    assert not destination.exists()
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("failing_closes", [("tmpfile",), ("directory",), ("tmpfile", "directory")])
def test_cleanup_failure_without_primary_raises_first_after_closing_retained_descriptor(
    api: str,
    failing_closes: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fd_baseline: dict[int, str],
) -> None:
    """Residual: publication had completed, so the linked inode stays (diagnostic-only).

    The call still raises the first cleanup failure; no result or descriptor is transferred.
    """

    publication = importlib.import_module("sfora.atomic_publication")
    spy = _FdSpy(monkeypatch, failing_closes=failing_closes)
    destination = tmp_path / "authority.bin"

    with pytest.raises(OSError, match="injected") as caught:
        _publish(publication, api, destination)

    assert len(spy.injected) == len(failing_closes)
    assert caught.value is spy.injected[0]
    assert getattr(caught.value, "__notes__", []) == [
        f"immutable publication cleanup also failed: {error!r}" for error in spy.injected[1:]
    ]
    assert destination.read_bytes() == PAYLOAD
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
def test_foreign_replacement_is_preserved_after_postlink_failure(
    api: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fd_baseline: dict[int, str]
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    primary = ValueError("directory barrier failed")
    destination = tmp_path / "authority.bin"

    def replace_then_fail() -> None:
        destination.unlink()
        destination.write_bytes(b"racer")
        raise primary

    spy = _FdSpy(monkeypatch, directory_fsync_hooks={1: replace_then_fail})

    with pytest.raises(ValueError) as caught:
        _publish(publication, api, destination)

    assert caught.value is primary
    assert not hasattr(primary, "__notes__")
    assert destination.read_bytes() == b"racer"
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
def test_happy_path_and_collisions_keep_descriptor_ownership(
    api: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fd_baseline: dict[int, str]
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    spy = _FdSpy(monkeypatch)
    destination = tmp_path / "authority.bin"

    published = _publish(publication, api, destination)
    assert spy.directory_fsyncs == 2
    assert published.identity == (destination.lstat().st_dev, destination.lstat().st_ino)
    assert published.size == len(PAYLOAD)
    assert _fd_table() == {**fd_baseline, published.descriptor: str(destination)}
    published.close()
    spy.assert_every_descriptor_released_once()
    assert _fd_table() == fd_baseline
    assert destination.read_bytes() == PAYLOAD

    with pytest.raises(FileExistsError):
        _publish(publication, api, destination)
    assert spy.roles == {}
    assert _fd_table() == fd_baseline

    raced = tmp_path / "raced.bin"
    syncs = spy.directory_fsyncs

    def race(_descriptor: int) -> None:
        raced.write_bytes(b"racer")

    with pytest.raises(FileExistsError):
        _publish(publication, api, raced, writer=race)
    assert raced.read_bytes() == b"racer"
    assert spy.directory_fsyncs == syncs
    spy.assert_every_descriptor_released_once()
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
def test_prelink_writer_failure_leaves_no_descriptor_or_file(
    api: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fd_baseline: dict[int, str]
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    spy = _FdSpy(monkeypatch)
    failure = RuntimeError("writer failed")

    def writer(_descriptor: int) -> None:
        raise failure

    with pytest.raises(RuntimeError) as caught:
        _publish(publication, api, tmp_path / "authority.bin", writer=writer)

    assert caught.value is failure
    assert list(tmp_path.iterdir()) == []
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
def test_cleanup_failure_is_raised_even_while_the_caller_handles_another_exception(
    api: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fd_baseline: dict[int, str]
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    spy = _FdSpy(monkeypatch, failing_closes=("directory",))
    handled = KeyError("the caller is handling this")

    try:
        raise handled
    except KeyError:
        with pytest.raises(OSError, match="injected directory") as caught:
            _publish(publication, api, tmp_path / "authority.bin")

    assert caught.value is spy.injected[0]
    assert not hasattr(handled, "__notes__")
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize(
    ("step", "notes", "destination_survives"),
    [("unlink", 1, True), ("barrier", 1, False), ("closes", 3, False)],
)
def test_baseexception_cleanup_failure_neither_skips_cleanup_nor_masks_the_primary(
    api: str,
    step: str,
    notes: int,
    destination_survives: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fd_baseline: dict[int, str],
) -> None:
    publication = importlib.import_module("sfora.atomic_publication")
    primary = ValueError("final directory barrier failed")
    hooks = {2: _raising(primary)}
    path = tmp_path / "authority.bin"
    destination = _UnlinkInterrupted(path) if step == "unlink" else path
    if step == "barrier":
        hooks[3] = _raising(_CleanupInterrupt("injected cleanup barrier interrupt"))
    spy = _FdSpy(
        monkeypatch,
        directory_fsync_hooks=hooks,
        failing_closes=("tmpfile", "directory", "retained") if step == "closes" else (),
        close_failure=_close_interrupt,
    )

    with pytest.raises(ValueError) as caught:
        _publish(publication, api, destination)

    assert caught.value is primary
    assert len(primary.__notes__) == notes
    assert all("_CleanupInterrupt" in note for note in primary.__notes__)
    assert path.exists() is destination_survives
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("failing_closes", [("tmpfile",), ("directory",), ("tmpfile", "directory")])
def test_baseexception_cleanup_failure_without_primary_is_raised_after_all_attempts(
    api: str,
    failing_closes: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fd_baseline: dict[int, str],
) -> None:
    """Residual: the completed publication's linked inode stays (diagnostic-only), no transfer."""

    publication = importlib.import_module("sfora.atomic_publication")
    spy = _FdSpy(monkeypatch, failing_closes=failing_closes, close_failure=_close_interrupt)
    destination = tmp_path / "authority.bin"

    with pytest.raises(_CleanupInterrupt) as caught:
        _publish(publication, api, destination)

    assert len(spy.injected) == len(failing_closes)
    assert caught.value is spy.injected[0]
    assert getattr(caught.value, "__notes__", []) == [
        f"immutable publication cleanup also failed: {error!r}" for error in spy.injected[1:]
    ]
    assert destination.read_bytes() == PAYLOAD
    spy.assert_every_descriptor_released_once(caught.value)
    assert _fd_table() == fd_baseline
