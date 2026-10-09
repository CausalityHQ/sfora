#!/usr/bin/env python3
"""Tiny synthetic opaque bytes only; these pins are never producer evidence."""

import ast
import hashlib
import importlib.abc
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/qualify_connected_gallery_fragment.py"
PYTHON = str(Path(sys.executable).resolve())


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load(path: Path, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


@contextmanager
def observed_calls() -> Iterator[list[tuple[str, str, object]]]:
    calls: list[tuple[str, str, object]] = []

    def profile(frame: types.FrameType, event: str, arg: object) -> None:
        if event in {"call", "return"} and frame.f_code.co_name in {
            "extract_gallery_members",
            "publish_bytes_noreplace",
            "<module>",
        }:
            calls.append((event, frame.f_code.co_filename + ":" + frame.f_code.co_name, arg))

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".")[0] in {
            "torch",
            "numpy",
            "PIL",
            "transformers",
            "safetensors",
            "torchvision",
        }:
            raise AssertionError("forbidden native import: " + fullname)


def parent_accepts(result: subprocess.CompletedProcess[str]) -> bool:
    """Tiny parent model: survivors or an old success report cannot authorize exit 1."""
    if result.returncode != 0 or len(result.stdout.splitlines()) != 1:
        return False
    report = json.loads(result.stdout)
    if report.get("schema") != "connected-gallery-fragment-result-v1":
        return False
    for facts in report["members"].values():
        raw = Path(facts["path"]).read_bytes()
        if len(raw) != facts["bytes"] or sha(raw) != facts["sha256"]:
            return False
    return True


class FragmentTests(unittest.TestCase):
    def setUp(self) -> None:
        # Python 3.14's ctypes owns a persistent libffi FD; include it in the baseline.
        importlib.import_module("ctypes")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        sentinel = NoNative()
        sys.meta_path.insert(0, sentinel)
        self.addCleanup(sys.meta_path.remove, sentinel)

    def fixture(self, count: int = 33) -> None:
        fixture = load(ROOT / "scripts/test_connected_gallery_provenance.py", "fixture")
        receipt, bundle = fixture.fixture(count)
        # Ordinal-distinct opaque rows: no packed numeric interpretation.
        self.wire = b"".join(bytes([i]) * 130 for i in range(count * 2))
        self.gallery = b"".join(bytes([i]) * 130 for i in range(1, count * 2, 2))
        self.ids = [f"Img/é{i}.jpg" for i in range(1, count * 2, 2)]
        receipt["files"]["control-179061.packed.bin"] = sha(self.wire)
        self.receipt = receipt
        self.inputs = {
            "receipt": encoded(receipt),
            "bundle": encoded(bundle),
            "ownership_audit": b"synthetic opaque owners, not an installation verdict\n",
            "combined_wire": self.wire,
        }
        self.authority: dict[str, Any] = {
            "schema": "connected-gallery-fragment-authority-v1",
            "files": {},
            "inputs": {},
            "python": {"path": PYTHON, "sha256": sha(Path(PYTHON).read_bytes())},
            "output": str(self.root / "output"),
            "resource_policy": {"seconds": 120, "address_space_bytes": 1073741824},
            "reference": {
                "gallery_count": count,
                "wire_sha256": sha(self.gallery),
                "ordered_ids_sha256": sha(encoded(self.ids)),
            },
        }
        for role, source in {
            "driver": DRIVER,
            "test": Path(__file__).resolve(),
            "package_init": ROOT / "src/sfora/__init__.py",
            "provenance": ROOT / "src/sfora/connected_gallery_provenance.py",
            "conversion": ROOT / "src/sfora/connected_gallery_conversion.py",
            "publication": ROOT / "src/sfora/atomic_publication.py",
        }.items():
            path = self.root / source.name
            path.write_bytes(source.read_bytes())
            self.authority["files"][role] = {"path": str(path), "sha256": sha(path.read_bytes())}
        for role, raw in self.inputs.items():
            path = self.root / (role + ".input")
            path.write_bytes(raw)
            self.authority["inputs"][role] = {"path": str(path), "sha256": sha(raw)}
        self.output = Path(self.authority["output"])
        self.auth_path = self.root / "authority.json"

    def save(self) -> str:
        raw = encoded(self.authority)
        self.auth_path.write_bytes(raw)
        return sha(raw)

    def api(self) -> Any:
        return load(Path(self.authority["files"]["driver"]["path"]), "fragment_gate_under_test")

    def invoke(self, api: Any) -> dict[str, Any]:
        return dict(api.run(str(self.auth_path), self.save(), time.monotonic()))

    def assert_clean(self, before: set[str]) -> None:
        current = set(os.listdir("/proc/self/fd"))
        self.assertEqual(
            current,
            before,
            {
                fd: os.readlink("/proc/self/fd/" + fd)
                for fd in current - before
                if os.path.exists("/proc/self/fd/" + fd)
            },
        )
        self.assertFalse(any(n == "sfora" or n.startswith("sfora.") for n in sys.modules))

    def repin_input(self, role: str, raw: bytes) -> None:
        row = self.authority["inputs"][role]
        Path(row["path"]).write_bytes(raw)
        row["sha256"] = sha(raw)

    def cli(
        self, pin: str | None = None, flags: tuple[str, ...] = ()
    ) -> subprocess.CompletedProcess[str]:
        pin = self.save() if pin is None else pin
        return subprocess.run(
            [
                PYTHON,
                "-I",
                "-S",
                "-B",
                *flags,
                self.authority["files"]["driver"]["path"],
                "--authority",
                str(self.auth_path),
                "--authority-sha256",
                pin,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_cli_genuine_interleaved_bytes_and_terminal(self) -> None:
        self.assertTrue(DRIVER.is_file(), "missing actual-byte fragment gate driver")
        self.fixture()
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema"], "connected-gallery-fragment-result-v1")
        self.assertEqual(report["observed"]["gallery_count"], 33)
        self.assertEqual(report["observed"]["batch_sizes"], [32, 1])
        self.assertEqual((self.output / "gallery.bin").read_bytes(), self.gallery)
        self.assertEqual((self.output / "gallery-ids.json").read_bytes(), encoded(self.ids))
        for name, role in (
            ("origin.json", "bundle"),
            ("origin-export.json", "receipt"),
            ("origin-owners.json", "ownership_audit"),
        ):
            self.assertEqual((self.output / name).read_bytes(), self.inputs[role])
        self.assertEqual(set(p.name for p in self.output.iterdir()), set(report["members"]))
        for name, facts in report["members"].items():
            path = self.output / name
            raw = path.read_bytes()
            self.assertEqual(facts, {"path": str(path), "bytes": len(raw), "sha256": sha(raw)})
            self.assertEqual(path.stat().st_nlink, 1)
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        self.assertTrue(report["normal_terminal_required"])
        for key in ("native_verified", "serving_accepted", "quality_speed_goal_met"):
            self.assertIs(report[key], False)
        provenance = json.loads((self.output / "gallery-provenance.json").read_bytes())
        self.assertIsNone(provenance["producer"]["gallery_wire_sha256"])
        self.assertEqual(
            provenance["producer"]["gallery_rows"],
            [r for b in self.receipt["images"] if b["role"] == "gallery" for r in b["rows"]],
        )

    def test_all_pins_precede_package_execution_and_input_parsing(self) -> None:
        self.fixture(1)
        api = self.api()
        for section, roles in (
            ("files", tuple(self.authority["files"])),
            ("inputs", tuple(self.authority["inputs"])),
            ("python", (None,)),
        ):
            for role in roles:
                row = self.authority[section] if role is None else self.authority[section][role]
                old = row["sha256"]
                row["sha256"] = "0" * 64
                before = set(os.listdir("/proc/self/fd"))
                with self.subTest(section=section, role=role), observed_calls() as calls:
                    with self.assertRaisesRegex(ValueError, "trusted SHA"):
                        self.invoke(api)
                    self.assertFalse(
                        any(
                            "/" + Path(self.authority["files"][r]["path"]).name + ":" in name
                            for _, name, _ in calls
                            for r in api.SOURCES
                        )
                    )
                    self.assertFalse(self.output.exists())
                    self.assert_clean(before)
                row["sha256"] = old
        with self.assertRaisesRegex(ValueError, "trusted SHA"):
            api.run(str(self.auth_path), "0" * 64, time.monotonic())

    def test_strict_authority_schema_and_paths_before_publication(self) -> None:
        self.fixture(1)
        api = self.api()
        original = encoded(self.authority)
        for raw in (
            original[:-1] + b',"schema":"duplicate"}',
            original.replace(b'"gallery_count":1', b'"gallery_count":true'),
            original.replace(b'"seconds":120', b'"seconds":120.0'),
            original.replace(b'"seconds":120', b'"seconds":true'),
            original.replace(b'"seconds":120', b'"seconds":NaN'),
            original.replace(b'"seconds":120', b'"seconds":1e999'),
            original[:-1] + b',"unknown":0}',
            b"[]",
        ):
            with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                self.auth_path.write_bytes(raw)
                api.run(str(self.auth_path), sha(raw), time.monotonic())
            self.assertFalse(self.output.exists())
        for bad in ("UPPER" * 13, "a" * 63, "A" * 64, 123, True):
            self.authority["reference"]["wire_sha256"] = bad
            with self.subTest(sha=bad), self.assertRaises(ValueError):
                self.invoke(api)

    def test_symlink_input_parent_and_nonregular_source_rejected(self) -> None:
        self.fixture(1)
        api = self.api()
        row = self.authority["inputs"]["ownership_audit"]
        original = Path(row["path"])
        alias = self.root / "alias"
        alias.symlink_to(original)
        for value in (
            str(alias),
            str(self.root / ".." / self.root.name / original.name),
            "relative",
        ):
            row["path"] = value
            with self.assertRaises(ValueError):
                self.invoke(api)
        alias.unlink()
        alias.symlink_to(self.root, target_is_directory=True)
        row["path"] = str(alias / original.name)
        with self.assertRaises(ValueError):
            self.invoke(api)
        alias.unlink()
        row["path"] = str(original)
        original.unlink()
        os.mkfifo(original)
        with self.assertRaisesRegex(ValueError, "regular file"):
            self.invoke(api)
        self.assertFalse(self.output.exists())

    def test_external_references_and_existing_targets_rejected(self) -> None:
        self.fixture(1)
        api = self.api()
        for key in self.authority["reference"]:
            previous = self.authority["reference"][key]
            self.authority["reference"][key] = 2 if key == "gallery_count" else "0" * 64
            with (
                self.subTest(reference=key),
                self.assertRaisesRegex(ValueError, "derived reference"),
            ):
                self.invoke(api)
            self.assertFalse(self.output.exists())
            self.authority["reference"][key] = previous
        for kind in ("file", "directory", "symlink"):
            if kind == "file":
                self.output.write_bytes(b"foreign")
            elif kind == "directory":
                self.output.mkdir()
            else:
                self.output.symlink_to(self.root / "absent")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.invoke(api)
            if kind == "file":
                self.assertEqual(self.output.read_bytes(), b"foreign")
            if kind == "directory":
                self.output.rmdir()
            else:
                self.output.unlink()

    def test_foreign_registry_preserved_and_source_caches_ignored(self) -> None:
        self.fixture(1)
        api = self.api()
        foreign = types.ModuleType("foreign")
        for name, value in (
            ("sfora", foreign),
            ("sfora.connected_gallery_provenance", None),
            ("sfora.unknown", foreign),
        ):
            cast(dict[str, Any], sys.modules)[name] = value
            try:
                with self.assertRaisesRegex(ValueError, "foreign package registry"):
                    self.invoke(api)
                self.assertIs(sys.modules[name], value)
                self.assertFalse(self.output.exists())
            finally:
                del sys.modules[name]
        # Would explode if a filesystem or bytecode loader were used.
        (self.root / "__pycache__").mkdir()
        for role in api.SOURCES:
            p = Path(self.authority["files"][role]["path"])
            p.with_suffix(".pyc").write_bytes(b"invalid foreign cache")
            (p.parent / "__pycache__" / (p.stem + ".cpython-312.pyc")).write_bytes(b"foreign")
        before = set(os.listdir("/proc/self/fd"))
        with observed_calls() as calls:
            self.invoke(api)
        self.assertEqual(
            sum(
                event == "call" and name.endswith(":extract_gallery_members")
                for event, name, _ in calls
            ),
            1,
        )
        retained = [
            cast(Any, value)
            for event, name, value in calls
            if event == "return" and name.endswith(":publish_bytes_noreplace")
        ]
        self.assertEqual(len(retained), 6)
        self.assertTrue(all(p.descriptor == -1 for p in retained))
        self.assert_clean(before)

    def test_independent_oracle_rejects_suffix_rows_ids_and_producer_mutants(self) -> None:
        self.fixture()
        api = self.api()
        captured = {role: api.Captured(row) for role, row in self.authority["inputs"].items()}
        sources = {role: api.Captured(row) for role, row in self.authority["files"].items()}
        modules = api.SourceModules()
        try:
            modules.load(sources)
            converter = modules.owned[api.SOURCES["conversion"]]
            members = converter.extract_gallery_members(
                *[
                    captured[r].raw
                    for r in ("receipt", "bundle", "ownership_audit", "combined_wire")
                ],
                trusted_receipt_sha256=captured["receipt"].sha256,
                trusted_bundle_sha256=captured["bundle"].sha256,
                trusted_ownership_audit_sha256=captured["ownership_audit"].sha256,
            )
            for key, replacement in (
                ("gallery.bin", self.wire[-len(self.gallery) :]),
                ("gallery.bin", self.gallery[130:] + self.gallery[:130]),
                ("gallery.bin", self.gallery[:-1] + b"x"),
                ("gallery-ids.json", encoded(list(reversed(self.ids)))),
                ("origin.json", b"{}"),
            ):
                mutant = {**members, key: replacement}
                with self.subTest(key=key), self.assertRaises(ValueError):
                    api.verify_members(mutant, captured, self.authority["reference"])
            for field in ("gallery_rows", "gallery_batches", "gallery_wire_sha256", "identity"):
                provenance = json.loads(members["gallery-provenance.json"])
                producer = provenance["producer"]
                if field in {"gallery_rows", "gallery_batches"}:
                    producer[field] = list(reversed(producer[field]))
                elif field == "identity":
                    producer[field]["numerical_flags"]["threads"] = True
                else:
                    producer[field] = sha(self.gallery)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    api.verify_members(
                        {**members, "gallery-provenance.json": encoded(provenance)},
                        captured,
                        self.authority["reference"],
                    )
            self.assertFalse(self.output.exists())
        finally:
            modules.close()
            for file in [*captured.values(), *sources.values()]:
                file.close()

    def test_failed_converter_and_publisher_still_fresh_check_every_capture(self) -> None:
        self.fixture(1)
        api = self.api()
        interpreter = self.root / "python-copy"
        shutil.copyfile(PYTHON, interpreter)
        self.authority["python"]["path"] = str(interpreter)
        fresh = api.Captured.fresh
        paths = {
            str(self.auth_path),
            str(interpreter),
            *(r["path"] for r in self.authority["files"].values()),
            *(r["path"] for r in self.authority["inputs"].values()),
        }
        changes = [
            Path(self.authority["files"]["provenance"]["path"]),
            Path(self.authority["inputs"]["ownership_audit"]["path"]),
            interpreter,
        ]
        originals = [p.read_bytes() for p in changes]
        original_receipt = self.inputs["receipt"]
        real_write = os.write
        for failure in ("converter", "publisher"):
            checked: list[str] = []
            writes = 0

            def drift() -> None:
                for p in changes:
                    with p.open("ab") as stream:
                        stream.write(b"late mutation")

            def check(file: Any, checked: list[str] = checked) -> None:
                checked.append(str(file.path))
                fresh(file)

            def write(fd: int, raw: bytes) -> int:
                nonlocal writes
                writes += 1
                if writes == 2:
                    drift()
                    raise OSError("injected publisher write failure")
                return real_write(fd, raw)

            def profile(frame: types.FrameType, event: str, arg: object) -> None:
                if event == "call" and frame.f_code.co_name == "extract_gallery_members":
                    drift()

            if failure == "converter":
                invalid = json.loads(original_receipt)
                invalid["schema"] = "invalid-synthetic-schema"
                self.repin_input("receipt", encoded(invalid))
            before = set(os.listdir("/proc/self/fd"))
            try:
                with (
                    patch.object(sys, "executable", str(interpreter)),
                    patch.object(api.Captured, "fresh", check),
                    patch("os.write", side_effect=write),
                ):
                    if failure == "converter":
                        sys.setprofile(profile)
                    with self.assertRaises((ValueError, OSError)) as caught:
                        self.invoke(api)
                self.assertIn(
                    "original export role/schema"
                    if failure == "converter"
                    else "injected publisher",
                    str(caught.exception),
                )
                self.assertEqual(set(checked), paths)
                notes = " ".join(getattr(caught.exception, "__notes__", []))
                for p in changes:
                    self.assertIn(str(p), notes)
                self.assertFalse(self.output.exists())
                self.assert_clean(before)
            finally:
                sys.setprofile(None)
                for p, raw in zip(changes, originals, strict=True):
                    p.write_bytes(raw)
                self.repin_input("receipt", original_receipt)

    def test_cli_rejects_optimized_or_extra_flags_and_bad_authority_hash(self) -> None:
        self.fixture(1)
        pin = self.save()
        for extra in (("-O",), ("-X", "dev")):
            result = self.cli(pin, extra)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertFalse(self.output.exists())
        result = self.cli("0" * 64)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_corruption_foreign_members_and_directory_replacement(self) -> None:
        self.fixture(1)
        api = self.api()
        real_fsync = os.fsync
        foreign = self.root / "foreign"
        foreign.write_bytes(b"foreign data must survive")
        for fault in ("bytes", "hardlink", "symlink", "extra", "directory"):
            fired = False

            def fsync(fd: int, fault: str = fault) -> None:
                nonlocal fired
                real_fsync(fd)
                if fired or not (self.output / "gallery-provenance.json").exists():
                    return
                fired = True
                victim = self.output / "origin.json"
                if fault == "bytes":
                    victim.write_bytes(b"corrupt")
                elif fault == "hardlink":
                    os.link(victim, self.root / "alias")
                elif fault == "symlink":
                    victim.unlink()
                    victim.symlink_to(foreign)
                elif fault == "extra":
                    (self.output / "foreign-extra").write_bytes(b"foreign")
                else:
                    self.output.rename(self.root / "moved-owned")
                    self.output.mkdir(mode=0o700)
                    (self.output / "foreign-extra").write_bytes(b"foreign")

            before = set(os.listdir("/proc/self/fd"))
            with (
                self.subTest(fault=fault),
                patch("os.fsync", side_effect=fsync),
                observed_calls() as calls,
                self.assertRaises((ValueError, OSError)),
            ):
                self.invoke(api)
            self.assertTrue(fired)
            self.assert_clean(before)
            self.assertEqual(foreign.read_bytes(), b"foreign data must survive")
            retained = [
                cast(Any, p)
                for e, n, p in calls
                if e == "return" and n.endswith(":publish_bytes_noreplace") and p is not None
            ]
            self.assertTrue(all(p.descriptor == -1 for p in retained))
            if fault in {"bytes", "hardlink"}:
                self.assertFalse(self.output.exists())
            elif fault == "symlink":
                self.assertTrue((self.output / "origin.json").is_symlink())
                self.assertEqual(list(self.output.iterdir()), [self.output / "origin.json"])
                (self.output / "origin.json").unlink()
                self.output.rmdir()
            else:
                self.assertEqual(list(self.output.iterdir()), [self.output / "foreign-extra"])
                self.assertEqual((self.output / "foreign-extra").read_bytes(), b"foreign")
                (self.output / "foreign-extra").unlink()
                self.output.rmdir()
            if fault == "hardlink":
                self.assertEqual((self.root / "alias").read_bytes(), self.inputs["bundle"])
                (self.root / "alias").unlink()
            if fault == "directory":
                # The failing publisher never returned this inode; preserve it as diagnostic.
                self.assertEqual(
                    [p.name for p in (self.root / "moved-owned").iterdir()],
                    ["gallery-provenance.json"],
                )

    def test_cleanup_attempts_all_files_and_closes_descriptors_preserving_primary(self) -> None:
        self.fixture(1)
        api = self.api()
        real_write, real_unlink = os.write, os.unlink
        writes = 0
        attempted: list[str] = []

        def write(fd: int, raw: bytes) -> int:
            nonlocal writes
            writes += 1
            if writes == 4:
                raise OSError("primary write failure")
            return real_write(fd, raw)

        def unlink(name: Any, *, dir_fd: int | None = None) -> None:
            attempted.append(str(name))
            if name == "origin.json":
                raise PermissionError("cleanup failure")
            real_unlink(name, dir_fd=dir_fd)

        before = set(os.listdir("/proc/self/fd"))
        with (
            patch("os.write", side_effect=write),
            patch("os.unlink", side_effect=unlink),
            self.assertRaisesRegex(OSError, "primary write failure") as caught,
        ):
            self.invoke(api)
        self.assertEqual(
            set(attempted), {"origin.json", "origin-export.json", "origin-owners.json"}
        )
        self.assertIn("cleanup failure", " ".join(getattr(caught.exception, "__notes__", [])))
        self.assertEqual(list(self.output.iterdir()), [self.output / "origin.json"])
        self.assert_clean(before)

    def test_registry_drift_preserves_foreign_entries_even_during_partial_load(self) -> None:
        self.fixture(1)
        api = self.api()
        key = "sfora.connected_gallery_provenance"
        foreign = types.ModuleType("replacement")
        for stage in ("load", "publication"):
            fired = False

            def profile(
                frame: types.FrameType, event: str, arg: object, stage: str = stage
            ) -> None:
                nonlocal fired
                target = (
                    event == "call"
                    and frame.f_code.co_name == "<module>"
                    and frame.f_code.co_filename == self.authority["files"]["publication"]["path"]
                    if stage == "load"
                    else event == "return" and frame.f_code.co_name == "publish_bytes_noreplace"
                )
                if target and not fired:
                    fired = True
                    sys.modules[key] = foreign
                    sys.modules["sfora.foreign"] = foreign

            before = set(os.listdir("/proc/self/fd"))
            sys.setprofile(profile)
            try:
                with self.assertRaisesRegex(ValueError, "registry ownership changed"):
                    self.invoke(api)
                self.assertTrue(fired)
                self.assertIs(sys.modules[key], foreign)
                self.assertIs(sys.modules["sfora.foreign"], foreign)
                self.assertFalse(self.output.exists())
            finally:
                sys.setprofile(None)
                sys.modules.pop(key, None)
                sys.modules.pop("sfora.foreign", None)
            self.assert_clean(before)

    def test_deadline_after_publication_cleans_and_never_returns_report(self) -> None:
        self.fixture(1)
        api = self.api()
        real_time = time.monotonic
        started = real_time()

        def clock() -> float:
            return (
                started + 121 if (self.output / "gallery-provenance.json").exists() else real_time()
            )

        before = set(os.listdir("/proc/self/fd"))
        with (
            patch.object(api.time, "monotonic", side_effect=clock),
            self.assertRaisesRegex(TimeoutError, "deadline"),
        ):
            self.invoke(api)
        self.assertFalse(self.output.exists())
        self.assert_clean(before)

    def test_nonzero_terminal_and_surviving_bytes_are_denied_by_parent(self) -> None:
        self.fixture(1)
        result = self.cli()
        self.assertTrue(parent_accepts(result), result.stderr)
        result.returncode = 1
        self.assertFalse(parent_accepts(result))
        result.returncode = 0
        (self.output / "gallery.bin").write_bytes(b"changed")
        self.assertFalse(parent_accepts(result))

    def test_bounded_source_read(self) -> None:
        self.fixture(1)
        api = self.api()
        source = Path(self.authority["files"]["test"]["path"])
        with source.open("wb") as stream:
            stream.truncate(16 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(ValueError, "bounded read"):
            self.invoke(api)
        self.assertFalse(self.output.exists())

    def test_report_measurements_include_fresh_exit_checks(self) -> None:
        self.fixture(1)
        api = self.api()
        real_fresh = api.Captured.fresh
        now, rss = 100.0, 1024
        reads = 0

        def fresh(file: Any) -> None:
            nonlocal reads, now, rss
            real_fresh(file)
            if file.path == self.auth_path:
                reads += 1
                if reads == 2:
                    now += 5.0
                    rss = 2048

        with (
            patch.object(api.Captured, "fresh", fresh),
            patch.object(api.time, "monotonic", side_effect=lambda: now),
            patch.object(
                api.resource,
                "getrusage",
                side_effect=lambda _: types.SimpleNamespace(ru_maxrss=rss),
            ),
        ):
            report = self.invoke(api)
        self.assertEqual(reads, 2)
        self.assertEqual(report["elapsed_seconds"], 5.0)
        self.assertEqual(report["peak_rss_bytes"], 2048 * 1024)

    def test_successful_publication_then_source_input_interpreter_drift_is_failure(self) -> None:
        self.fixture(1)
        api = self.api()
        interpreter = self.root / "python-copy"
        shutil.copyfile(PYTHON, interpreter)
        self.authority["python"]["path"] = str(interpreter)
        changes = [
            Path(self.authority["files"]["test"]["path"]),
            Path(self.authority["inputs"]["combined_wire"]["path"]),
            interpreter,
        ]
        real_fsync = os.fsync
        fired = False

        def fsync(fd: int) -> None:
            nonlocal fired
            real_fsync(fd)
            if not fired and (self.output / "gallery-provenance.json").exists():
                fired = True
                for p in changes:
                    with p.open("ab") as stream:
                        stream.write(b"late drift")

        before = set(os.listdir("/proc/self/fd"))
        with (
            patch.object(sys, "executable", str(interpreter)),
            patch("os.fsync", side_effect=fsync),
            self.assertRaisesRegex(ValueError, "ownership changed") as caught,
        ):
            self.invoke(api)
        self.assertTrue(fired)
        errors = str(caught.exception) + " ".join(getattr(caught.exception, "__notes__", []))
        for p in changes:
            self.assertIn(str(p), errors)
        self.assertFalse(self.output.exists())
        self.assert_clean(before)

    def test_alarm_between_cleanup_attempts_preserves_primary_and_releases_owners(self) -> None:
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_trace = sys.gettrace()
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        for failure in (False, True):
            self.fixture(1)
            api = self.api()
            tree = ast.parse(Path(api.__file__).read_bytes())
            run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
            close_line = next(
                n.lineno
                for n in ast.walk(run)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "attempt"
                and len(n.args) == 1
                and isinstance(n.args[0], ast.Attribute)
                and isinstance(n.args[0].value, ast.Name)
                and n.args[0].value.id == "modules"
                and n.args[0].attr == "close"
            )
            if failure:
                invalid = json.loads(self.inputs["receipt"])
                invalid["schema"] = "invalid-synthetic-schema"
                self.repin_input("receipt", encoded(invalid))
            captured: list[Any] = []
            modules: Any = None
            output: Any = None
            fd_owners: dict[int, tuple[int, int]] = {}
            fired = False

            def trace(
                frame: types.FrameType,
                event: str,
                arg: object,
                api: Any = api,
                close_line: int = close_line,
                fd_owners: dict[int, tuple[int, int]] = fd_owners,
            ) -> Any:
                nonlocal fired, captured, modules, output
                if (
                    frame.f_code is api.run.__code__
                    and event == "line"
                    and frame.f_lineno == close_line
                    and not fired
                ):
                    fired = True
                    captured = list(frame.f_locals["captured"])
                    modules = frame.f_locals["modules"]
                    output = frame.f_locals["output"]
                    descriptors = [item.fd for item in captured]
                    if output is not None:
                        descriptors += [output.parent, output.fd]
                        descriptors += [item.descriptor for item in output.owned.values()]
                    for fd in descriptors:
                        info = os.fstat(fd)
                        fd_owners[fd] = (info.st_dev, info.st_ino)
                    signal.raise_signal(signal.SIGALRM)
                return trace

            before = set(os.listdir("/proc/self/fd"))
            signal.signal(signal.SIGALRM, api.expired)
            sys.settrace(trace)
            try:
                expected = ValueError if failure else TimeoutError
                with self.subTest(original_failure=failure), self.assertRaises(expected) as caught:
                    self.invoke(api)
                sys.settrace(previous_trace)
                self.assertTrue(fired)
                if failure:
                    self.assertIn("original export role/schema", str(caught.exception))
                    self.assertIn(
                        "deadline exceeded", " ".join(getattr(caught.exception, "__notes__", []))
                    )
                self.assertFalse(self.output.exists())
                self.assert_clean(before)
                self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), previous_mask)
            finally:
                sys.settrace(previous_trace)
                signal.signal(signal.SIGALRM, previous_handler)
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                if modules is not None:
                    modules.close()
                # RED cleanup touches only identities captured from this actual run.
                for fd, owner in fd_owners.items():
                    try:
                        info = os.fstat(fd)
                    except OSError:
                        continue
                    if (info.st_dev, info.st_ino) == owner:
                        os.close(fd)
                if output is not None:
                    for item in output.owned.values():
                        item.descriptor = -1
                if self.output.exists():
                    shutil.rmtree(self.output)

    def test_foreign_file_or_symlink_created_before_publish_is_preserved(self) -> None:
        self.fixture(1)
        api = self.api()
        real_fsync = os.fsync
        foreign = self.root / "foreign-target"
        foreign.write_bytes(b"foreign")
        for kind in ("file", "symlink"):
            fired = False

            def fsync(fd: int, kind: str = kind) -> None:
                nonlocal fired
                real_fsync(fd)
                if not fired and self.output.is_dir():
                    fired = True
                    if kind == "file":
                        (self.output / "origin.json").write_bytes(b"foreign")
                    else:
                        (self.output / "origin.json").symlink_to(foreign)

            before = set(os.listdir("/proc/self/fd"))
            with patch("os.fsync", side_effect=fsync), self.assertRaises(FileExistsError):
                self.invoke(api)
            self.assertTrue(fired)
            self.assertEqual((self.output / "origin.json").read_bytes(), b"foreign")
            self.assertEqual(foreign.read_bytes(), b"foreign")
            self.assert_clean(before)
            (self.output / "origin.json").unlink()
            self.output.rmdir()


if __name__ == "__main__":
    unittest.main()
