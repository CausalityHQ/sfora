"""Real tiny-file descriptor handoff checks; no model or native imports."""

import hashlib
import importlib.util
import os
import stat
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class DirectoryHandoffTests(unittest.TestCase):
    def test_parent_close_failure_keeps_foreign_fd_and_releases_child(self) -> None:
        for filename, function, extra in (
            ("connected_artifact_identity.py", "_installed_sha", ()),
            ("connected_installed_environment.py", "_read_file", ("0" * 64, None)),
        ):
            with self.subTest(filename=filename):
                self.check_handoff(filename, function, extra)

    def test_read_error_survives_both_descriptor_close_errors(self) -> None:
        for filename, function, extra in (
            ("connected_artifact_identity.py", "_installed_sha", ()),
            ("connected_installed_environment.py", "_read_file", ("0" * 64, None)),
        ):
            with self.subTest(filename=filename):
                self.check_read_cleanup(filename, function, extra)

    def check_read_cleanup(self, filename, function, extra) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = importlib.util.spec_from_file_location(
                "_cleanup_under_test", ROOT / "src/sfora" / filename
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            leaf = Path(tmp, "opaque")
            leaf.write_bytes(b"opaque")
            primary = MemoryError("original read failure")
            closing_read = False
            closed = []
            proxy = SimpleNamespace(
                **{name: getattr(os, name) for name in dir(os) if not name.startswith("__")}
            )

            def opening_stream(*args, **kwargs):
                nonlocal closing_read
                closing_read = True
                raise primary

            def closing(fd):
                os.close(fd)
                if closing_read:
                    closed.append(fd)
                    raise OSError("close failed after releasing descriptor")

            proxy.fdopen, proxy.close = opening_stream, closing
            with patch.object(module, "os", proxy), self.assertRaises(MemoryError) as caught:
                getattr(module, function)(PurePosixPath(leaf), *extra)
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(closed), 2)
            self.assertEqual(len(primary.__notes__), 2)
            for fd in closed:
                with self.assertRaises(OSError):
                    os.fstat(fd)

    def test_cleanup_only_failure_is_not_swallowed_inside_callers_except(self) -> None:
        for filename, function, extra in (
            ("connected_artifact_identity.py", "_installed_sha", ()),
            ("connected_installed_environment.py", "_read_file", ("0" * 64, None)),
        ):
            with self.subTest(filename=filename):
                self.check_cleanup_only(filename, function, extra)

    def check_cleanup_only(self, filename, function, extra) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = importlib.util.spec_from_file_location(
                "_cleanup_only", ROOT / "src/sfora" / filename
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            leaf = Path(tmp, "opaque")
            leaf.write_bytes(b"opaque")
            args = () if not extra else (hashlib.sha256(b"opaque").hexdigest(), None)
            cleanup = RuntimeError("directory close failed")
            proxy = SimpleNamespace(
                **{name: getattr(os, name) for name in dir(os) if not name.startswith("__")}
            )

            close_read = False

            def closing(fd):
                directory = stat.S_ISDIR(os.fstat(fd).st_mode)
                os.close(fd)
                if directory and close_read:
                    raise cleanup

            original_fdopen = proxy.fdopen

            def opening_stream(*args, **kwargs):
                nonlocal close_read
                close_read = True
                return original_fdopen(*args, **kwargs)

            proxy.fdopen, proxy.close = opening_stream, closing
            caller = ValueError("already handled by caller")
            try:
                raise caller
            except ValueError:
                with patch.object(module, "os", proxy), self.assertRaises(RuntimeError) as caught:
                    getattr(module, function)(PurePosixPath(leaf), *args)
            self.assertIs(caught.exception, cleanup)
            self.assertFalse(hasattr(caller, "__notes__"))

    def check_handoff(self, filename: str, function: str, extra: tuple[object, ...]) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = importlib.util.spec_from_file_location(
                "_handoff_under_test", ROOT / "src/sfora" / filename
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            leaf = Path(tmp, "opaque")
            leaf.write_bytes(b"tiny opaque bytes")
            owned: list[tuple[int, os.stat_result]] = []
            foreign: list[tuple[int, os.stat_result]] = []
            closed: list[int] = []
            primary = RuntimeError("parent close failed after release")
            proxy = SimpleNamespace(
                **{name: getattr(os, name) for name in dir(os) if not name.startswith("__")}
            )

            def opening(
                path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
            ) -> int:
                fd = os.open(path, flags, mode, dir_fd=dir_fd)
                owned.append((fd, os.fstat(fd)))
                return fd

            def closing(fd: int) -> None:
                os.close(fd)
                closed.append(fd)
                if len(closed) == 1:
                    replacement = os.open(leaf, os.O_RDONLY)
                    foreign.append((replacement, os.fstat(replacement)))
                    self.assertEqual(replacement, fd)
                    raise primary

            def same(fd: int, before: os.stat_result) -> bool:
                try:
                    after = os.fstat(fd)
                except OSError:
                    return False
                return (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)

            proxy.open, proxy.close = opening, closing
            try:
                with (
                    patch.object(module, "os", proxy),
                    self.assertRaises(RuntimeError) as caught,
                ):
                    getattr(module, function)(PurePosixPath(leaf), *extra)
                self.assertIs(caught.exception, primary)
                self.assertEqual(len(foreign), 1)
                self.assertTrue(same(*foreign[0]), "foreign reused descriptor was closed")
                self.assertFalse(any(same(*row) for row in owned), "child directory leaked")
            finally:
                for row in owned + foreign:
                    if same(*row):
                        os.close(row[0])


if __name__ == "__main__":
    unittest.main()
