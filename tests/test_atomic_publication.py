from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import json
import os
from pathlib import Path

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
