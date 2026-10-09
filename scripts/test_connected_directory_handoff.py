"""Real tiny-file descriptor handoff checks; no model or native imports."""

import importlib.util
import os
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
