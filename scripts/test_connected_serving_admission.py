#!/usr/bin/env python3
"""Synthetic filesystem admission only; typed payload, installed and native admission untested."""

import contextlib
import errno
import json
import os
import shutil
import signal
import stat
import sys
import tempfile
import tracemalloc
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any
from unittest.mock import patch

import test_connected_serving_artifact as artifact_tests
from test_connected_gallery_provenance import encoded, sha

ROOT = Path(__file__).resolve().parents[1]
FRAGMENT = (
    "origin.json",
    "origin-export.json",
    "origin-owners.json",
    "gallery.bin",
    "gallery-ids.json",
    "gallery-provenance.json",
)
# The genuine fixture's original bundle declares exactly these payload SHAs.
PAYLOAD = {
    "vision.pt": b"synthetic metadata, never native proof",
    "endpoint.pt": b"synthetic metadata, never native proof",
    "processor.json": b"file",
}
NAMES = ("serving.json", *FRAGMENT, *PAYLOAD)
DEFAULT: Any = object()  # distinguishes "use the valid fixture value" from an explicit None


class Trace:
    def __init__(self) -> None:
        self.opened: list[tuple[str, int, int | None, int]] = []
        self.meta: dict[int, tuple[str, int]] = {}
        self.live: set[int] = set()
        self.errors: list[str] = []


def _alarm(signum: int, frame: FrameType | None) -> None:
    raise AssertionError("test exceeded its alarm (blocking open?)")


class AdmissionTests(unittest.TestCase):
    api: Any
    binder: Any

    @classmethod
    def setUpClass(cls) -> None:
        artifact_tests.ServingTests.setUpClass()
        sentinel = artifact_tests.NoNativeImports()
        sys.meta_path.insert(0, sentinel)
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from sfora import connected_serving_admission, connected_serving_artifact

            cls.api = connected_serving_admission
            cls.binder = connected_serving_artifact
        finally:
            sys.path.pop(0)
            sys.meta_path.remove(sentinel)
        assert not artifact_tests.NATIVE.intersection(sys.modules)

    def setUp(self) -> None:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(20)
        self.addCleanup(signal.alarm, 0)
        builder = artifact_tests.ServingTests("setUp")
        builder.setUp()
        self.manifest: dict[str, Any] = builder.manifest
        self.fragment: dict[str, bytes] = builder.fragment
        self.pins: dict[str, str] = builder.pins
        for name, raw in PAYLOAD.items():
            self.assertEqual(sha(raw), self.manifest["files"][name]["sha256"])
            self.manifest["files"][name]["bytes"] = len(raw)
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="admission-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.directory = self.root + "/artifact/serving"
        os.makedirs(self.directory)
        self.write_all()

    # -- helpers ---------------------------------------------------------

    def path(self, name: str) -> Path:
        return Path(self.directory, name)

    def write_all(self) -> None:
        self.serving = encoded(self.manifest)
        self.serving_pin = sha(self.serving)
        contents = {"serving.json": self.serving, **self.fragment, **PAYLOAD}
        for name in NAMES:
            path = self.path(name)
            path.unlink(missing_ok=True)
            path.write_bytes(contents[name])
            path.chmod(0o644)
        for extra in ("alias", "swap", "outside"):
            Path(self.root, extra).unlink(missing_ok=True)

    def admit(
        self,
        directory: Any = DEFAULT,
        serving_pin: Any = DEFAULT,
        pins: Any = DEFAULT,
    ) -> Any:
        return self.api.admit_serving_artifact(
            self.directory if directory is DEFAULT else directory,
            trusted_serving_sha256=self.serving_pin if serving_pin is DEFAULT else serving_pin,
            trusted_fragment_sha256=self.pins if pins is DEFAULT else pins,
        )

    @contextmanager
    def trace(self, fail_close: Callable[[str, int], bool] | None = None) -> Iterator[Trace]:
        trace = Trace()
        real_open, real_close = os.open, os.close

        def traced_open(
            path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
        ) -> int:
            fd = real_open(path, flags, mode, dir_fd=dir_fd)
            if fd in trace.live:
                trace.errors.append(f"fd {fd} reopened while live")
            trace.live.add(fd)
            trace.meta[fd] = (path, flags)
            trace.opened.append((path, flags, dir_fd, fd))
            return fd

        def traced_close(fd: int) -> None:
            if fd not in trace.live:
                trace.errors.append(f"close of non-live fd {fd}")
            trace.live.discard(fd)
            real_close(fd)
            if fail_close is not None and fail_close(*trace.meta.get(fd, ("", 0))):
                raise OSError(errno.EIO, "injected close failure")

        with patch.object(os, "open", traced_open), patch.object(os, "close", traced_close):
            yield trace

    @contextmanager
    def during_first_read(self, name: str, mutate: Callable[[Path], None]) -> Iterator[None]:
        inode = self.path(name).stat().st_ino
        target = self.path(name)
        real = os.fdopen
        fired: list[bool] = []

        class Hook:
            def __init__(self, stream: Any) -> None:
                self.stream = stream

            def __enter__(self) -> "Hook":
                self.stream.__enter__()
                return self

            def __exit__(self, *exc: Any) -> Any:
                return self.stream.__exit__(*exc)

            def read(self, size: int) -> bytes:
                data: bytes = self.stream.read(size)
                if not fired:
                    fired.append(True)
                    mutate(target)
                return data

        def hooked(fd: int, *args: Any, **kwargs: Any) -> Any:
            stream = real(fd, *args, **kwargs)
            return Hook(stream) if os.fstat(fd).st_ino == inode else stream

        with patch.object(os, "fdopen", hooked):
            yield
        self.assertTrue(fired, "mutation hook never fired")

    def mutations(self) -> dict[str, Callable[[Path], None]]:
        def append(p: Path) -> None:
            with p.open("ab") as stream:
                stream.write(b"x")

        def rewrite(p: Path) -> None:
            info = p.stat()
            with p.open("r+b") as stream:
                stream.write(bytes(b ^ 1 for b in p.read_bytes()))
            os.utime(p, ns=(info.st_atime_ns, info.st_mtime_ns + 10**9))

        def replace(p: Path) -> None:
            swap = Path(self.root, "swap")
            swap.write_bytes(p.read_bytes())
            os.replace(swap, p)

        return {
            "append": append,
            "truncate": lambda p: os.truncate(p, 0),
            "rewrite-same-size": rewrite,
            "rename-over": replace,
            "hardlink": lambda p: os.link(p, Path(self.root, "alias")),
            "chmod": lambda p: p.chmod(0o600),
            "unlink": lambda p: p.unlink(),
        }

    def assert_clean(self, trace: Trace) -> None:
        self.assertEqual(trace.errors, [])
        self.assertEqual(trace.live, set())

    # -- contract --------------------------------------------------------

    def test_admits_fixture_with_exact_facts(self) -> None:
        result = self.admit()
        self.assertEqual(list(result), ["manifest", "producer", "directory", "files"])
        direct = self.binder.bind_serving_manifest(
            self.serving,
            self.fragment,
            trusted_serving_sha256=self.serving_pin,
            trusted_fragment_sha256=self.pins,
        )
        self.assertEqual(result["manifest"], direct["manifest"])
        self.assertEqual(result["producer"], direct["producer"])
        info = os.stat(self.directory)
        self.assertEqual(
            result["directory"],
            {"path": self.directory, "device": info.st_dev, "inode": info.st_ino},
        )
        self.assertEqual(list(result["files"]), list(NAMES))
        for name, fact in result["files"].items():
            raw, st = self.path(name).read_bytes(), self.path(name).stat()
            self.assertEqual(
                fact,
                {
                    "path": f"{self.directory}/{name}",
                    "bytes": len(raw),
                    "sha256": sha(raw),
                    "device": st.st_dev,
                    "inode": st.st_ino,
                    "mtime_ns": st.st_mtime_ns,
                    "ctime_ns": st.st_ctime_ns,
                },
            )
        json.dumps(result)  # plain JSON data, no retained bytes
        result["manifest"]["files"].clear()
        result["files"].clear()
        self.assertEqual(self.admit()["manifest"], direct["manifest"])

    def test_arguments_rejected_before_any_io(self) -> None:
        directories: list[Any] = [
            "relative/dir",
            "",
            "/",
            self.directory + "/",
            self.directory + "/../serving",
            "/a//b",
            "/a/./b",
            "/a\\b",
            "/a\0b",
            self.directory.encode(),
            Path(self.directory),
            None,
        ]
        bad_pins: list[Any] = [
            list(self.pins),
            {k: v for k, v in self.pins.items() if k != "gallery.bin"},
            {**self.pins, "extra.json": "0" * 64},
            {**self.pins, "gallery.bin": "A" * 64},
            {**self.pins, "gallery.bin": "0" * 63},
            {**self.pins, "gallery.bin": 5},
            None,
        ]
        with (
            patch.object(os, "open", side_effect=AssertionError("I/O before validation")),
            patch.object(os, "lstat", side_effect=AssertionError("I/O before validation")),
        ):
            for directory in directories:
                with self.subTest(directory=directory), self.assertRaises(ValueError):
                    self.admit(directory=directory)
            for pins in bad_pins:
                with self.subTest(pins=pins), self.assertRaises(ValueError):
                    self.admit(pins=pins)
            for serving_pin in ("A" * 64, "0" * 63, b"0" * 64, None):
                with self.subTest(serving_pin=serving_pin), self.assertRaises(ValueError):
                    self.admit(serving_pin=serving_pin)

    def test_pins_authenticate_before_any_json(self) -> None:
        with patch.object(self.binder, "_parse", side_effect=AssertionError("parsed early")):
            with self.assertRaisesRegex(ValueError, "^trusted serving SHA differs$"):
                self.admit(serving_pin="0" * 64)
            for name in FRAGMENT:
                with (
                    self.subTest(pin=name),
                    self.assertRaisesRegex(ValueError, f"^trusted {name} SHA differs$"),
                ):
                    self.admit(pins={**self.pins, name: "f" * 64})
            # Malformed bytes under the ORIGINAL pins fail on the pin, never the parser.
            for name in ("serving.json", *FRAGMENT):
                with self.subTest(file=name):
                    self.path(name).write_bytes(b"\xff{not json")
                    with self.assertRaisesRegex(ValueError, "SHA differs"):
                        self.admit()
                    self.write_all()

    def test_manifest_files_grant_no_authority(self) -> None:
        # A coherent forgery (bigger payload + matching manifest) still dies on the pin.
        forged = json.loads(self.serving)
        forged["files"]["vision.pt"]["bytes"] += 1
        self.path("serving.json").write_bytes(encoded(forged))
        self.path("vision.pt").write_bytes(PAYLOAD["vision.pt"] + b"\0")
        with self.assertRaisesRegex(ValueError, "trusted serving SHA differs"):
            self.admit()
        self.write_all()
        # A manifest naming an extra file is not extra authority either.
        self.manifest["files"]["extra.bin"] = {"path": "extra.bin", "bytes": 0, "sha256": sha(b"")}
        self.write_all()
        with self.assertRaisesRegex(ValueError, "exact object keys differ"):
            self.admit()

    # -- payload bytes ---------------------------------------------------

    def test_payload_substitution_passes_binder_but_not_admission(self) -> None:
        bound = self.binder.bind_serving_manifest(
            self.serving,
            self.fragment,
            trusted_serving_sha256=self.serving_pin,
            trusted_fragment_sha256=self.pins,
        )
        self.assertIn("manifest", bound)  # metadata alone is accepted; only bytes can reject
        for name, raw in PAYLOAD.items():
            variants = {
                "same-size": (raw[:-1] + bytes([raw[-1] ^ 1]), f"{name} SHA differs"),
                "shorter": (raw[:-1], f"{name} byte count differs"),
                "longer": (raw + b"\0", f"{name} byte count differs"),
                "empty": (b"", f"{name} byte count differs"),
            }
            for label, (changed, message) in variants.items():
                with self.subTest(name=name, variant=label):
                    self.path(name).write_bytes(changed)
                    with self.assertRaisesRegex(ValueError, message):
                        self.admit()
                    self.write_all()
        self.assertEqual(self.admit()["files"]["vision.pt"]["sha256"], sha(PAYLOAD["vision.pt"]))

    def test_declared_size_gates_before_streaming(self) -> None:
        self.manifest["files"]["vision.pt"]["bytes"] = 999  # SHA still right, size wrong
        self.write_all()
        with (
            self.trace() as trace,
            self.assertRaisesRegex(ValueError, "vision.pt byte count differs"),
        ):
            self.admit()
        self.assertNotIn("vision.pt", [row[0] for row in trace.opened])
        self.assert_clean(trace)

    def test_directory_must_be_exactly_the_ten_names(self) -> None:
        for name in NAMES:
            with self.subTest(missing=name):
                os.rename(self.path(name), Path(self.root, "moved"))
                with self.assertRaisesRegex(ValueError, "exact ten prescribed names required"):
                    self.admit()
                os.rename(Path(self.root, "moved"), self.path(name))

        def write(p: Path) -> None:
            p.write_bytes(b"x")

        extras: dict[str, Callable[[Path], None]] = {
            "extra.bin": write,
            ".hidden": write,
            "vision.pt.orig": write,
            "original": os.mkdir,
            "link": lambda p: p.symlink_to("vision.pt"),
        }
        for name, make in extras.items():
            with self.subTest(extra=name):
                make(self.path(name))
                with (
                    self.trace() as trace,
                    self.assertRaisesRegex(ValueError, "exact ten prescribed names required"),
                ):
                    self.admit()
                # Membership is decided before any member is opened or read.
                self.assertEqual([row[0] for row in trace.opened if row[0] in NAMES], [])
                self.assert_clean(trace)
                if self.path(name).is_dir() and not self.path(name).is_symlink():
                    os.rmdir(self.path(name))
                else:
                    self.path(name).unlink()
        self.path("original").mkdir()
        self.path("original/vision.pt").write_bytes(b"nested payload impostor")
        with self.assertRaisesRegex(ValueError, "exact ten prescribed names required"):
            self.admit()
        shutil.rmtree(self.path("original"))
        self.assertEqual(list(self.admit()["files"]), list(NAMES))

    def test_membership_change_after_first_listing_rejected(self) -> None:
        real = self.api._read
        late = self.path("late-extra")
        for label, make in (
            ("file", lambda: late.write_bytes(b"late")),
            ("directory", lambda: late.mkdir()),
        ):
            with self.subTest(extra=label):

                def spy(dirfd: int, name: str, expected: Any, *, make: Any = make) -> Any:
                    out = real(dirfd, name, expected)
                    if name == "processor.json":
                        make()
                    return out

                with (
                    patch.object(self.api, "_read", spy),
                    self.assertRaisesRegex(ValueError, "membership changed during admission"),
                ):
                    self.admit()
                if late.is_dir():
                    late.rmdir()
                else:
                    late.unlink()

    # -- path and file kinds ---------------------------------------------

    def test_symlinks_rejected(self) -> None:
        outside = Path(self.root, "outside")
        for name in NAMES:
            with self.subTest(leaf=name):
                raw = self.path(name).read_bytes()
                outside.write_bytes(raw)
                self.path(name).unlink()
                self.path(name).symlink_to(outside)
                with self.assertRaises(ValueError):
                    self.admit()
                self.path(name).unlink()
                self.path(name).symlink_to(self.root + "/dangling")
                with self.assertRaises(ValueError):
                    self.admit()
                self.write_all()
        os.symlink(self.root + "/artifact", self.root + "/link")
        with self.subTest(parent="link"), self.assertRaises(ValueError):
            self.admit(directory=self.root + "/link/serving")
        os.symlink(self.directory, self.root + "/final")
        with self.subTest(parent="final"), self.assertRaises(ValueError):
            self.admit(directory=self.root + "/final")
        os.rename(self.root + "/artifact", self.root + "/moved")
        os.symlink(self.root + "/moved", self.root + "/artifact")
        with self.subTest(parent="swapped"), self.assertRaises(ValueError):
            self.admit()

    def test_hardlinks_rejected(self) -> None:
        for name in NAMES:
            with self.subTest(name=name):
                # An in-directory alias is an extra name (exact-ten); this is the outside link.
                os.link(self.path(name), Path(self.root, "alias"))
                with self.assertRaisesRegex(ValueError, "single-link"):
                    self.admit()
                self.write_all()

    def test_special_files_rejected_without_blocking(self) -> None:
        makers: dict[str, Callable[[Path], None]] = {
            "fifo": os.mkfifo,
            "socket": lambda p: os.mknod(p, stat.S_IFSOCK | 0o600),
            "directory": os.mkdir,
        }
        for name in ("serving.json", "gallery.bin", "vision.pt"):
            for kind, make in makers.items():
                with self.subTest(name=name, kind=kind):
                    self.path(name).unlink()
                    make(self.path(name))
                    with self.assertRaises(ValueError):
                        self.admit()
                    if kind == "directory":
                        os.rmdir(self.path(name))
                    self.write_all()

    def test_opens_are_directory_relative_nofollow_and_nonblocking(self) -> None:
        components = {"artifact", "serving"}
        with self.trace() as trace:
            self.admit()
        self.assert_clean(trace)
        leaves = [row for row in trace.opened if row[0] in NAMES]
        self.assertEqual(sorted(row[0] for row in leaves), sorted(NAMES))
        for path, flags, dir_fd, _ in trace.opened:
            with self.subTest(path=path):
                self.assertFalse(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
                if path == "/":
                    self.assertTrue(flags & os.O_DIRECTORY and dir_fd is None)
                elif path in NAMES:
                    self.assertTrue(flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK)
                    self.assertIsNotNone(dir_fd)
                elif path in components or path in self.root.split("/"):
                    self.assertTrue(flags & os.O_DIRECTORY and flags & os.O_NOFOLLOW)
                    self.assertIsNotNone(dir_fd)
                else:
                    self.fail("unexpected open: " + path)

    # -- change during admission -----------------------------------------

    def test_change_during_read_rejected(self) -> None:
        for name in NAMES:
            for label, mutate in self.mutations().items():
                with self.subTest(name=name, mutation=label):
                    with (
                        self.during_first_read(name, mutate),
                        self.assertRaises(ValueError),
                    ):
                        self.admit()
                    self.write_all()

    def test_replacement_after_read_caught_by_final_pass(self) -> None:
        real = self.api._read
        for victim in NAMES:
            for label, mutate in self.mutations().items():
                with self.subTest(victim=victim, mutation=label):

                    def spy(
                        dirfd: int,
                        name: str,
                        expected: Any,
                        *,
                        victim: str = victim,
                        mutate: Callable[[Path], None] = mutate,
                    ) -> Any:
                        out = real(dirfd, name, expected)
                        if name == "processor.json":
                            mutate(self.path(victim))
                        return out

                    with (
                        patch.object(self.api, "_read", spy),
                        self.assertRaisesRegex(ValueError, "during admission"),
                    ):
                        self.admit()
                    self.write_all()

    def test_directory_chain_replacement_rejected(self) -> None:
        real = self.api._read
        for moved in (self.directory, self.root + "/artifact"):
            with self.subTest(moved=moved):

                def spy(dirfd: int, name: str, expected: Any, *, moved: str = moved) -> Any:
                    out = real(dirfd, name, expected)
                    if name == "processor.json":
                        os.rename(moved, moved + ".old")
                        shutil.copytree(moved + ".old", moved)
                    return out

                with (
                    patch.object(self.api, "_read", spy),
                    self.assertRaisesRegex(ValueError, "directory chain replaced"),
                ):
                    self.admit()
                shutil.rmtree(moved)
                os.rename(moved + ".old", moved)

    def test_every_invocation_rereads_everything(self) -> None:
        for _ in range(2):
            with self.trace() as trace:
                self.admit()
            self.assertEqual(
                sorted(row[0] for row in trace.opened if row[0] in NAMES), sorted(NAMES)
            )
        raw = self.path("vision.pt").read_bytes()
        self.path("vision.pt").write_bytes(raw[:-1] + b"?")
        with self.assertRaisesRegex(ValueError, "vision.pt SHA differs"):
            self.admit()
        self.write_all()
        self.assertTrue(self.admit())
        changed = self.fragment["gallery.bin"][:-1] + b"?"
        self.path("gallery.bin").write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "trusted gallery.bin SHA differs"):
            self.admit()

    # -- descriptors, memory, isolation ----------------------------------

    def test_failures_leak_no_descriptors_and_close_each_once(self) -> None:
        def missing() -> None:
            os.rename(self.path("gallery.bin"), self.path("moved"))

        def symlink() -> None:
            self.path("vision.pt").unlink()
            self.path("vision.pt").symlink_to(self.path("endpoint.pt"))

        def mismatch() -> None:
            self.path("processor.json").write_bytes(b"fine")

        def hardlink() -> None:
            os.link(self.path("origin.json"), Path(self.root, "alias"))

        count = len(os.listdir("/proc/self/fd"))
        cases: list[tuple[str, Callable[[], None]]] = [
            ("ok", lambda: None),
            ("missing", missing),
            ("symlink", symlink),
            ("mismatch", mismatch),
            ("hardlink", hardlink),
        ]
        for label, setup in cases:
            with self.subTest(case=label):
                setup()
                with self.trace() as trace, contextlib.suppress(ValueError):
                    self.admit()
                self.assert_clean(trace)
                self.assertEqual(len(os.listdir("/proc/self/fd")), count)
                self.write_all()
                self.path("moved").unlink(missing_ok=True)

    def test_fdopen_failure_keeps_primary_and_closes_once(self) -> None:
        for primary in (MemoryError("primary"), OSError(errno.EIO, "primary")):
            expected = MemoryError if isinstance(primary, MemoryError) else ValueError
            with self.subTest(primary=repr(primary)), self.trace() as trace:
                with (
                    patch.object(os, "fdopen", side_effect=primary),
                    self.assertRaises(expected) as caught,
                ):
                    self.admit()
                self.assertIs(
                    caught.exception if expected is MemoryError else caught.exception.__cause__,
                    primary,
                )
            self.assert_clean(trace)

    def test_close_failure_is_rejected_not_hidden(self) -> None:
        kinds: dict[str, Callable[[str, int], bool]] = {
            "leaf": lambda path, flags: path in NAMES,
            "directory": lambda path, flags: bool(flags & os.O_DIRECTORY),
        }
        for kind, fail in kinds.items():
            with self.subTest(kind=kind), self.trace(fail) as trace:
                with self.assertRaises(ValueError) as caught:
                    self.admit()
                cause = caught.exception.__cause__
                self.assertIsInstance(cause, OSError)
                self.assertIn("injected close failure", str(cause))
            self.assert_clean(trace)

    def test_close_failure_never_replaces_primary(self) -> None:
        self.path("processor.json").write_bytes(b"fine")  # primary: payload mismatch
        # Directory descriptors close only after the primary exists; a leaf close failing
        # earlier (no primary yet) is correctly raised instead.
        with (
            self.trace(lambda path, flags: bool(flags & os.O_DIRECTORY)) as trace,
            self.assertRaisesRegex(ValueError, "processor.json SHA differs") as caught,
        ):
            self.admit()
        self.assert_clean(trace)
        notes = getattr(caught.exception, "__notes__", [])
        self.assertTrue(any("descriptor close failed" in note for note in notes), notes)
        self.write_all()
        with (
            self.trace(lambda path, flags: True) as trace,
            patch.object(os, "fdopen", side_effect=MemoryError("primary")),
            self.assertRaises(MemoryError) as memory,
        ):
            self.admit()
        self.assert_clean(trace)
        self.assertGreaterEqual(len(getattr(memory.exception, "__notes__", [])), 2)

    def test_payload_stream_is_bounded_and_uncaptured(self) -> None:
        data = bytes(range(256)) * (96 * 1024)  # 24 MiB, built before tracing starts
        Path(self.root, "big.bin").write_bytes(data)
        expected = (len(data), sha(data))
        dirfd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            tracemalloc.start()
            _, digest, raw = self.api._read(dirfd, "big.bin", expected)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
            os.close(dirfd)
        self.assertEqual((digest, raw), (expected[1], b""))
        self.assertLess(peak, 4 * 1024 * 1024)

    def test_metadata_bound_is_explicit(self) -> None:
        self.assertEqual(self.api._METADATA_LIMIT, 64 * 1024 * 1024)
        with (
            patch.object(self.api, "_METADATA_LIMIT", 10),
            self.assertRaisesRegex(ValueError, "exceeds metadata bound"),
        ):
            self.admit()

    def test_only_directory_relative_access_and_no_native_imports(self) -> None:
        self.admit()  # warm lazy imports outside the audited window
        seen: list[tuple[str, tuple[Any, ...]]] = []
        active = False

        def audit(event: str, args: tuple[Any, ...]) -> None:
            if active and (event == "open" or event.startswith(("subprocess.", "ctypes."))):
                seen.append((event, args))

        sys.addaudithook(audit)
        sentinel = artifact_tests.NoNativeImports()
        sys.meta_path.insert(0, sentinel)
        try:
            active = True
            self.admit()
        finally:
            active = False
            sys.meta_path.remove(sentinel)
        allowed = {"/", *self.root.split("/"), "artifact", "serving", *NAMES}
        self.assertEqual([event for event, _ in seen if event != "open"], [])
        # io.open(fd, closefd=False) audits an integer fd, which names no path.
        paths = {args[0] for _, args in seen if isinstance(args[0], str)}
        self.assertEqual(sorted(paths - allowed), [])
        self.assertTrue(paths >= set(NAMES))
        self.assertFalse(artifact_tests.NATIVE.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
