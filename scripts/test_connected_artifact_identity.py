#!/usr/bin/env python3
"""Stdlib correspondence checks; no native, tensor or serving qualification."""

import copy
import hashlib
import importlib.abc
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
