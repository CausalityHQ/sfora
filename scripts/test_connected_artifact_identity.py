#!/usr/bin/env python3
"""Stdlib correspondence checks; no native, tensor or serving qualification."""

import ast
import copy
import errno
import hashlib
import importlib.abc
import importlib.util
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/sfora/connected_artifact_identity.py"
EVIDENCE = ROOT / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
OLD_ROOT = "/original-evidence-not-readable/json"


class NoNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "sfora",
            "torch",
            "numpy",
            "PIL",
            "transformers",
            "safetensors",
            "torchvision",
        }:
            raise AssertionError("native import sentinel: " + fullname)


def no_old_reads(event, args):
    if event == "open" and isinstance(args[0], str) and args[0].startswith(OLD_ROOT):
        raise AssertionError("old evidence path was opened")


sys.addaudithook(no_old_reads)
sys.meta_path.insert(0, NoNativeImports())

ORIGINAL_SOURCE_SHA256 = "3886a1e0c21824824f296f810dcdb89af25a7365b857f7a5afff6c91ece4abc3"
ORIGINAL_HELPER = """\
def _installed_sha(path: PurePosixPath) -> str:
    # Walk open directory descriptors so no parent or leaf symlink is followed.
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        source = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(source, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("installed source must be a regular nonsymlink file")
            digest = hashlib.sha256()
            while block := stream.read(1024 * 1024):
                digest.update(block)
            return digest.hexdigest()
    except OSError as error:
        raise ValueError("installed source must exist beneath nonsymlink directories") from error
    finally:
        os.close(directory)
"""


def restore_original_helper(data):
    """Finite inverse of the descriptor-ownership repair: swap back the one helper."""
    helpers = [
        node
        for node in ast.parse(data).body
        if isinstance(node, ast.FunctionDef) and node.name == "_installed_sha"
    ]
    assert len(helpers) == 1
    lines = data.splitlines(keepends=True)
    return (
        b"".join(lines[: helpers[0].lineno - 1])
        + ORIGINAL_HELPER.encode()
        + b"".join(lines[helpers[0].end_lineno :])
    )


class InverseSource(type(Path())):
    """Whole-source reads see the original helper, so the AST pin of the other code holds."""

    def read_bytes(self):
        return restore_original_helper(super().read_bytes())


RAW_SOURCE = SOURCE
SOURCE = InverseSource(RAW_SOURCE)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def open_fds():
    return set(os.listdir("/proc/self/fd"))


def close_leaks(seen):
    """Close, as test hygiene only, a descriptor still open on the exact file it was."""
    for fd, info in seen:
        if still_open(fd, info):
            os.close(fd)


def still_open(fd, info):
    try:
        now = os.fstat(fd)
    except OSError:
        return False
    return (now.st_dev, now.st_ino) == (info.st_dev, info.st_ino)


class StopRun(BaseException):
    pass


class StopCleanup(BaseException):
    pass


class IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert SOURCE.is_file(), "missing correspondence implementation"
        spec = importlib.util.spec_from_file_location("artifact_identity_under_test", SOURCE)
        cls.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.api)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.installed = Path(self.temp.name).resolve() / "installed/json"
        self.installed.mkdir(parents=True)
        self.roots = {"json": OLD_ROOT}
        self.files = {}
        rows = []
        # Genuine stdlib source bytes exercise relocation without ML dependencies.
        for name, klass in (("", json.JSONDecoder), ("encoder", json.JSONEncoder)):
            source = Path(inspect.getsourcefile(klass))
            content = source.read_bytes()
            old = OLD_ROOT + "/" + source.name
            self.files[old] = hashlib.sha256(content).hexdigest()
            (self.installed / source.name).write_bytes(content)
            rows.append(
                {
                    "name": name,
                    "class": klass.__module__ + "." + klass.__qualname__,
                    "file": old,
                    "training": False,
                    "attributes": {"enabled": True, "count": 1, "scale": 1.0},
                }
            )
        self.model = {
            "inventory": {"weight": {"shape": [1, 2], "dtype": "float32"}},
            "runtime": {
                "config": {"provenance": rows[0]["file"], "nested": {"file": rows[1]["file"]}},
                "attn_implementation": "sdpa",
                "modules": rows,
            },
            "nonpersistent": [],
        }
        self.processor = {
            "origin": {"class": rows[0]["class"], "file": rows[0]["file"]},
            "config": {"file": rows[0]["file"], "enabled": True, "factor": 1.0},
        }

    def project(self, original=None, kind="model", **overrides):
        return self.api.project_identity(
            self.model if original is None else original,
            kind=kind,
            package_roots=overrides.get("roots", self.roots),
            source_files=overrides.get("files", self.files),
        )

    def restore(self, projected, original=None, kind="model"):
        return self.api.restore_identity(
            projected,
            self.model if original is None else original,
            kind=kind,
            package_roots=self.roots,
            source_files=self.files,
        )

    def materialize(self, projected, original=None, kind="model", roots=None):
        return self.api.materialize_identity(
            projected,
            self.model if original is None else original,
            kind=kind,
            package_roots=self.roots,
            source_files=self.files,
            installed_roots={"json": str(self.installed)} if roots is None else roots,
        )

    def test_typing_erasure_restores_original_whole_source_ast(self):
        original = subprocess.check_output(
            [
                "git",
                "show",
                "f81dd383de1bd67920e0cd0a3e570a51de8ad0fc:src/sfora/connected_artifact_identity.py",
            ],
            cwd=ROOT,
        )
        self.assertEqual(
            hashlib.sha256(original).hexdigest(),
            "296742a2f8c06ed66f07c1bfe500e87f9aed0161ded7f2c05f88e84bd92aeded",
        )

        class EraseTyping(ast.NodeTransformer):
            def __init__(self):
                self.casts = 0
                self.aliases = []
                self.locals = 0
                self.functions = 0
                self.imports = 0

            def visit_ImportFrom(self, node):
                if node.module == "typing":
                    if [(item.name, item.asname) for item in node.names] != [("cast", None)]:
                        raise AssertionError("unexpected typing import")
                    self.imports += 1
                    return None
                return node

            def visit_TypeAlias(self, node):
                self.aliases.append(node.name.id)
                return None

            def visit_FunctionDef(self, node):
                self.functions += 1
                node.returns = None
                return self.generic_visit(node)

            def visit_arg(self, node):
                node.annotation = None
                return node

            def visit_AnnAssign(self, node):
                self.locals += 1
                if node.value is None:
                    raise AssertionError("unexpected unassigned annotation")
                return ast.Assign(
                    targets=[node.target],
                    value=self.visit(node.value),
                    type_comment=None,
                )

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "cast":
                    if len(node.args) != 2 or node.keywords:
                        raise AssertionError("unexpected cast form")
                    self.casts += 1
                    return self.visit(node.args[1])
                return self.generic_visit(node)

        inverse = EraseTyping()
        restored = inverse.visit(ast.parse(SOURCE.read_bytes()))
        self.assertEqual(inverse.casts, 12)
        self.assertEqual(inverse.aliases, ["JSONValue", "JSONObject"])
        self.assertEqual(inverse.locals, 2)
        self.assertEqual(inverse.functions, 8)
        self.assertEqual(inverse.imports, 1)
        self.assertEqual(
            ast.dump(restored, include_attributes=False),
            ast.dump(ast.parse(original), include_attributes=False),
        )

    def test_projection_inverse_preserve_every_other_typed_member_and_input(self):
        before = copy.deepcopy((self.model, self.processor, self.roots, self.files))
        projected = self.project()
        expected = copy.deepcopy(self.model)
        for row, filename in zip(
            expected["runtime"]["modules"], ("decoder.py", "encoder.py"), strict=True
        ):
            row["file"] = {
                "package": "json",
                "path": filename,
                "sha256": self.files[OLD_ROOT + "/" + filename],
            }
        self.assertEqual(json.dumps(projected), json.dumps(expected))
        self.assertEqual(json.dumps(self.restore(projected)), json.dumps(self.model))
        restored = self.restore(projected)
        restored["inventory"]["weight"]["shape"][0] = 9
        processor = self.project(self.processor, "processor")
        self.assertEqual(processor["origin"]["file"], expected["runtime"]["modules"][0]["file"])
        self.assertEqual(processor["config"], self.processor["config"])
        self.assertEqual(self.restore(processor, self.processor, "processor"), self.processor)
        projected["runtime"]["modules"][0]["attributes"]["count"] = 99
        processor["config"]["enabled"] = False
        self.assertEqual(before, (self.model, self.processor, self.roots, self.files))

    def test_committed_genuine_model_rows_have_exact_source_correspondence(self):
        # Four finite rows include repeated source occurrences and two real packages.
        receipt = EVIDENCE / "connected-mlp-evaluation-full-export-control-179061-v2/receipt.json"
        content = receipt.read_bytes()
        self.assertEqual(
            hashlib.sha256(content).hexdigest(),
            "db63db8270f2bf9a75448863e7ff701b7894b0c6aaf45c748730b6ea52393407",
        )
        data = json.loads(content)
        original = data["payload_facts"]["identity"]["encoder_identity"]
        original["runtime"]["modules"] = original["runtime"]["modules"][:4]
        roots = {key: value["root"] for key, value in data["origins"]["packages"].items()}
        projected = self.project(original, roots=roots, files=data["origins"]["files"])
        rows = projected["runtime"]["modules"]
        self.assertEqual(
            rows[0]["file"],
            {
                "package": "transformers",
                "path": "models/siglip/modeling_siglip.py",
                "sha256": "274eafdb9bff99607f15eb4c4fb6fc47256b07358807653baa48539d13376b31",
            },
        )
        self.assertEqual(rows[1]["file"], rows[0]["file"])
        self.assertEqual(
            rows[2]["file"],
            {
                "package": "torch",
                "path": "nn/modules/conv.py",
                "sha256": "fcd11330712354ae22f6e928bf72994fb1c8f6ed30b28d58be3c64acc99a44a4",
            },
        )
        restored = self.api.restore_identity(
            projected,
            original,
            kind="model",
            package_roots=roots,
            source_files=data["origins"]["files"],
        )
        self.assertEqual(json.dumps(restored), json.dumps(original))

    def test_equal_installed_bytes_relocate_only_explicit_origins(self):
        projected = self.project()
        before = copy.deepcopy(projected)
        result = self.materialize(projected)
        expected = copy.deepcopy(self.model)
        for row, filename in zip(
            expected["runtime"]["modules"], ("decoder.py", "encoder.py"), strict=True
        ):
            row["file"] = str(self.installed / filename)
        self.assertEqual(json.dumps(result), json.dumps(expected))
        self.assertEqual(projected, before)
        processor = self.project(self.processor, "processor")
        result = self.materialize(processor, self.processor, "processor")
        self.assertEqual(result["origin"]["file"], str(self.installed / "decoder.py"))
        self.assertEqual(result["config"], self.processor["config"])
        with self.assertRaisesRegex(ValueError, "model"):
            self.project(self.processor)
        with self.assertRaisesRegex(ValueError, "processor"):
            self.project(self.model, "processor")

    def test_same_class_path_version_changed_bytes_reject_before_native_and_recheck(self):
        self.model["runtime"]["config"]["version"] = "unchanged"
        projected = self.project()
        self.materialize(projected)
        path = self.installed / "decoder.py"
        original_bytes = path.read_bytes()
        original_stat = path.stat()
        path.write_bytes(original_bytes + b"\n# altered installed source\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.materialize(projected)
            __import__("torch")
        path.write_bytes(original_bytes)
        self.materialize(projected)
        path.write_bytes(original_bytes.replace(b"import", b"IMPORT", 1))
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "hash"):
            self.materialize(projected)

    def test_wrong_original_hash_cannot_authorize_installed_bytes(self):
        self.files[OLD_ROOT + "/decoder.py"] = "0" * 64
        projected = self.project()
        with self.assertRaisesRegex(ValueError, "hash"):
            self.materialize(projected)

    def test_original_paths_must_be_canonical_component_contained(self):
        for path in (
            OLD_ROOT + "-other/decoder.py",
            OLD_ROOT + "/../decoder.py",
            OLD_ROOT + "/./decoder.py",
            OLD_ROOT + "//decoder.py",
            OLD_ROOT + "/decoder.py/",
            OLD_ROOT + "\\decoder.py",
            "relative/decoder.py",
            OLD_ROOT,
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                original = copy.deepcopy(self.model)
                original["runtime"]["modules"][0]["file"] = path
                self.project(original, files={**self.files, path: "0" * 64})

    def test_wrong_unknown_ambiguous_roots_or_missing_hash_reject(self):
        for roots in (
            {"json": OLD_ROOT + "-other"},
            {},
            {"unknown": OLD_ROOT},
            {"json": OLD_ROOT, "duplicate": OLD_ROOT},
            {"json": OLD_ROOT, "parent": "/original-evidence-not-readable"},
            {"json": OLD_ROOT + "/.."},
            {"json": OLD_ROOT + "\\alias"},
        ):
            with self.subTest(roots=roots), self.assertRaises(ValueError):
                self.project(roots=roots)
        for files in (
            {},
            {**self.files, OLD_ROOT + "/decoder.py": True},
            {**self.files, OLD_ROOT + "/decoder.py": "F" * 64},
        ):
            with self.subTest(files=files), self.assertRaises(ValueError):
                self.project(files=files)

    def test_inverse_rejects_non_origin_values_types_and_module_order(self):
        for mutation in (
            "config",
            "attribute",
            "bool",
            "float",
            "order",
            "class",
            "extra",
            "key_order",
        ):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                projected = self.project()
                runtime = projected["runtime"]
                if mutation == "config":
                    runtime["config"]["provenance"] = "different"
                elif mutation == "attribute":
                    runtime["modules"][0]["attributes"]["count"] = 2
                elif mutation == "bool":
                    runtime["modules"][0]["attributes"]["enabled"] = 1
                elif mutation == "float":
                    runtime["modules"][0]["attributes"]["scale"] = 1
                elif mutation == "order":
                    runtime["modules"].reverse()
                elif mutation == "class":
                    runtime["modules"][0]["class"] = "json.decoder.Other"
                elif mutation == "key_order":
                    runtime["config"] = dict(reversed(list(runtime["config"].items())))
                else:
                    projected["extra"] = None
                self.restore(projected)
        projected = self.project(self.processor, "processor")
        projected["config"]["enabled"] = 1
        with self.assertRaises(ValueError):
            self.materialize(projected, self.processor, "processor")

    def test_inverse_rejects_changed_source_shape_path_package_and_hash(self):
        sources = (
            {"package": "json", "path": "../decoder.py", "sha256": "0" * 64},
            {"package": "json", "path": "decoder.py", "sha256": "0" * 64},
            {"package": "other", "path": "decoder.py", "sha256": "0" * 64},
            {"package": "json", "path": "decoder.py"},
            OLD_ROOT + "/decoder.py",
        )
        for source in sources:
            with self.subTest(source=source), self.assertRaises(ValueError):
                projected = self.project()
                projected["runtime"]["modules"][0]["file"] = source
                self.restore(projected)
        projected = self.project()
        projected["runtime"]["modules"][0]["file"]["extra"] = "forbidden"
        with self.assertRaises(ValueError):
            self.restore(projected)

    def test_json_only_no_coercion_or_nonfinite_values(self):
        for value in ((1, 2), {1: "value"}, float("nan"), float("inf"), b"bytes", Path("x")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                original = copy.deepcopy(self.model)
                original["extra"] = value
                self.project(original)
        for kind in ("endpoint", True):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.project(kind=kind)
        for original in (
            {},
            {"runtime": {}},
            {"runtime": {"modules": []}},
            {"runtime": {"modules": [{}]}},
        ):
            with self.subTest(original=original), self.assertRaises(ValueError):
                self.project(original)

    def test_installed_source_symlink_directory_fifo_or_missing_rejects(self):
        projected = self.project()
        source = self.installed / "decoder.py"
        content = source.read_bytes()
        target = self.installed / "same_bytes.py"
        target.write_bytes(content)
        source.unlink()
        source.symlink_to(target)
        with self.assertRaises(ValueError):
            self.materialize(projected)
        source.unlink()
        source.mkdir()
        with self.assertRaises(ValueError):
            self.materialize(projected)
        source.rmdir()
        os.mkfifo(source)
        with self.assertRaises(ValueError):
            self.materialize(projected)
        source.unlink()
        with self.assertRaises(ValueError):
            self.materialize(projected)

    def test_installed_roots_are_independent_canonical_unambiguous_nonsymlinks(self):
        projected = self.project()
        alias = self.installed.parent / "alias"
        alias.symlink_to(self.installed, target_is_directory=True)
        parent_alias = self.installed.parent.parent / "parent_alias"
        parent_alias.symlink_to(self.installed.parent, target_is_directory=True)
        for roots in (
            {},
            {"json": str(alias)},
            {"json": str(parent_alias / "json")},
            {"json": str(self.installed) + "/."},
            {"json": str(self.installed), "extra": str(self.installed)},
            {"json": "relative"},
        ):
            with self.subTest(roots=roots), self.assertRaises(ValueError):
                self.materialize(projected, roots=roots)


class InstalledSourceOwnershipTests(unittest.TestCase):
    """Genuine helper over tiny opaque temp files; only os.fdopen/os.close are observed."""

    CONTENT = b"\x00opaque\xff" * 7
    TRANSLATED = "installed source must exist beneath nonsymlink directories"

    @classmethod
    def setUpClass(cls):
        cls.api = load_module("artifact_identity_ownership_under_test", RAW_SOURCE)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name).resolve()
        self.source = self.dir / "opaque.bin"
        self.source.write_bytes(self.CONTENT)
        self.digest = hashlib.sha256(self.CONTENT).hexdigest()

    def variant(self, name, data):
        path = self.dir / (name + ".py")
        path.write_bytes(data)
        return load_module("artifact_identity_" + name, path)

    def sha(self, path=None, module=None):
        return (module or self.api)._installed_sha(PurePosixPath(path or self.source))

    def watch(self, build=None):
        """Observe os.fdopen/os.close for this test; closes counted are those after fdopen."""
        seen, closed = [], []
        real_fdopen, real_close = os.fdopen, os.close

        def fdopen(fd, mode="rb", **kwargs):
            seen.append((fd, os.fstat(fd)))
            closed.clear()
            return (build or real_fdopen)(fd, mode, **kwargs)

        def close(fd):
            closed.append(fd)
            real_close(fd)

        self.addCleanup(close_leaks, seen)
        self.enterContext(mock.patch.object(os, "fdopen", fdopen))
        self.enterContext(mock.patch.object(os, "close", close))
        return seen, closed

    def fail_before_adoption(self, errors, expected, module=None):
        """Fail os.fdopen on the live, unadopted source; report raised errors and leaks."""
        seen = []
        queue = iter(errors)

        def failing(fd, *args, **kwargs):
            seen.append((fd, os.fstat(fd)))
            raise next(queue)

        raised = []
        before = open_fds()
        try:
            with mock.patch.object(os, "fdopen", failing):
                for _ in errors:
                    with self.assertRaises(expected) as caught:
                        self.sha(module=module)
                    raised.append(caught.exception)
            leaked = open_fds() != before
        finally:
            close_leaks(seen)
        self.assertEqual([info.st_size for _, info in seen], [len(self.CONTENT)] * len(errors))
        return raised, leaked

    def test_fdopen_failure_before_adoption_closes_the_source_descriptor(self):
        os_error = OSError(errno.EMFILE, "injected")
        for error, expected in (
            (os_error, ValueError),
            (MemoryError("injected"), MemoryError),
            (StopRun("injected"), StopRun),
        ):
            with self.subTest(error=type(error).__name__):
                (raised,), leaked = self.fail_before_adoption([error], expected)
                self.assertFalse(leaked)
                if error is os_error:
                    self.assertEqual(str(raised), self.TRANSLATED)
                    self.assertIs(raised.__cause__, error)
                else:
                    self.assertIs(raised, error)
                    self.assertIsNone(raised.__cause__)
                self.assertFalse(hasattr(error, "__notes__"))

    def test_repeated_failed_adoption_leaves_no_descriptor(self):
        errors = [OSError(errno.EMFILE, "injected") for _ in range(32)]
        raised, leaked = self.fail_before_adoption(errors, ValueError)
        self.assertFalse(leaked)
        self.assertTrue(all(r.__cause__ is e for r, e in zip(raised, errors, strict=True)))
        self.assertEqual(self.sha(), self.digest)

    def test_successful_digest_closes_source_once_and_leaves_no_descriptor(self):
        seen, closed = self.watch()
        before = open_fds()
        self.assertEqual(self.sha(), self.digest)
        self.assertEqual(open_fds(), before)
        self.assertLessEqual(closed.count(seen[0][0]), 1)

    def test_read_failure_after_adoption_is_not_closed_twice(self):
        class Unreadable(io.BufferedReader):
            def read(self, size=-1):
                raise failure

        def build(fd, mode, **kwargs):
            return Unreadable(io.FileIO(fd, mode, closefd=kwargs.get("closefd", True)))

        seen, closed = self.watch(build)
        for failure, expected in (
            (OSError(errno.EIO, "injected read"), ValueError),
            (RuntimeError("injected read"), RuntimeError),
        ):
            with self.subTest(failure=type(failure).__name__):
                seen.clear()
                before = open_fds()
                with self.assertRaises(expected) as caught:
                    self.sha()
                self.assertEqual(open_fds(), before)
                if expected is ValueError:
                    self.assertEqual(str(caught.exception), self.TRANSLATED)
                    self.assertIs(caught.exception.__cause__, failure)
                else:
                    self.assertIs(caught.exception, failure)
                self.assertEqual(len(seen), 1)
                self.assertFalse(still_open(*seen[0]))
                self.assertLessEqual(closed.count(seen[0][0]), 1)

    def test_rejections_keep_their_error_and_leave_no_descriptor(self):
        link = self.dir / "link.bin"
        link.symlink_to(self.source)
        real = self.dir / "real"
        real.mkdir()
        (real / "leaf.bin").write_bytes(self.CONTENT)
        alias = self.dir / "alias"
        alias.symlink_to(real, target_is_directory=True)
        directory = self.dir / "directory.bin"
        directory.mkdir()
        fifo = self.dir / "fifo.bin"
        os.mkfifo(fifo)
        self.watch()
        for path, message in (
            (link, self.TRANSLATED),
            (alias / "leaf.bin", self.TRANSLATED),
            (self.dir / "missing.bin", self.TRANSLATED),
            (directory, self.TRANSLATED),
            (fifo, "installed source must be a regular nonsymlink file"),
        ):
            with self.subTest(path=path.name):
                before = open_fds()
                with self.assertRaises(ValueError) as caught:
                    self.sha(path)
                self.assertEqual(str(caught.exception), message)
                self.assertEqual(open_fds(), before)
                # The unadopted source close succeeded, so it was neither leaked nor doubled.
                self.assertEqual(getattr(caught.exception.__cause__, "__notes__", []), [])

    def test_finite_inverse_restores_the_original_helper_bytes(self):
        fixed = hashlib.sha256(Path.read_bytes(RAW_SOURCE)).hexdigest()
        restored = restore_original_helper(Path.read_bytes(RAW_SOURCE))
        self.assertNotEqual(fixed, ORIGINAL_SOURCE_SHA256)
        self.assertEqual(hashlib.sha256(restored).hexdigest(), ORIGINAL_SOURCE_SHA256)
        again = hashlib.sha256(restore_original_helper(restored)).hexdigest()
        self.assertEqual(again, ORIGINAL_SOURCE_SHA256)

    def test_original_helper_leaks_and_repaired_helper_does_not(self):
        original = self.variant("original", restore_original_helper(Path.read_bytes(RAW_SOURCE)))
        errors = [OSError(errno.EMFILE, "injected")]
        self.assertTrue(self.fail_before_adoption(errors, ValueError, original)[1])
        errors = [OSError(errno.EMFILE, "injected")]
        self.assertFalse(self.fail_before_adoption(errors, ValueError)[1])
        # A real directory leaf makes the genuine fdopen itself fail before adoption.
        directory = self.dir / "directory.bin"
        directory.mkdir()
        seen, _ = self.watch()
        for module, leaked in ((original, True), (self.api, False)):
            before = open_fds()
            with self.assertRaises(ValueError):
                self.sha(directory, module)
            changed = open_fds() != before
            close_leaks(seen)
            seen.clear()
            self.assertEqual(changed, leaked)

    def test_close_failure_after_failed_adoption_keeps_the_primary(self):
        real_close = os.close
        state = {}

        def failing(fd, *args, **kwargs):
            state["fd"] = fd
            if state["released"]:
                real_close(fd)  # as if io.open had already closed the raw stream
            raise state["error"]

        def close(fd):
            real_close(fd)
            if fd == state.get("fd") and not state["released"]:
                raise state["cleanup"]

        eio = OSError(errno.EIO, "injected close")
        for released, error, expected, cleanup in (
            (False, OSError(errno.EMFILE, "injected"), ValueError, eio),
            (True, OSError(errno.EMFILE, "injected"), ValueError, None),
            (False, MemoryError("injected"), MemoryError, eio),
            (True, StopRun("injected"), StopRun, None),
            (False, OSError(errno.EMFILE, "injected"), ValueError, StopCleanup("injected close")),
            (False, StopRun("injected"), StopRun, StopCleanup("injected close")),
        ):
            with self.subTest(
                released=released, error=type(error).__name__, cleanup=type(cleanup).__name__
            ):
                state.clear()
                state.update(released=released, error=error, cleanup=cleanup)
                before = open_fds()
                with (
                    mock.patch.object(os, "fdopen", failing),
                    mock.patch.object(os, "close", close),
                    self.assertRaises(expected) as caught,
                ):
                    self.sha()
                self.assertEqual(open_fds(), before)
                raised = caught.exception
                if expected is ValueError:
                    self.assertEqual(str(raised), self.TRANSLATED)
                    raised = raised.__cause__
                self.assertIs(raised, error)
                (note,) = error.__notes__
                self.assertIn("Bad file descriptor" if released else "injected close", note)

    def test_partial_stream_failure_never_closes_a_reused_foreign_descriptor(self):
        foreign = self.dir / "foreign.bin"
        foreign.write_bytes(b"foreign owner")
        state = {}
        failure = MemoryError("buffer construction failed")
        real_close = os.close

        def failing(fd, mode="rb", **kwargs):
            # FileIO owns the fd by default; failure building the buffer releases it.
            raw = io.FileIO(fd, mode, closefd=kwargs.get("closefd", True))
            raw.close()
            replacement = os.open(foreign, os.O_RDONLY)
            state.update(fd=replacement, info=os.fstat(replacement))
            if kwargs.get("closefd", True):
                self.assertEqual(replacement, fd)
            else:
                self.assertNotEqual(replacement, fd)
            raise failure

        before = open_fds()
        try:
            with mock.patch.object(os, "fdopen", failing), self.assertRaises(MemoryError) as caught:
                self.sha()
            self.assertIs(caught.exception, failure)
            self.assertTrue(still_open(state["fd"], state["info"]), "foreign owner was closed")
        finally:
            if state and still_open(state["fd"], state["info"]):
                real_close(state["fd"])
        self.assertEqual(open_fds(), before)


if __name__ == "__main__":
    unittest.main()
