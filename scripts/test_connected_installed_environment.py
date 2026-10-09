"""Stdlib-only installed-environment correspondence falsifiers; no ML imports."""

import base64
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "src/sfora/connected_installed_environment.py"
spec = importlib.util.spec_from_file_location("_installed_environment_test", MODULE)
assert spec and spec.loader
helper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helper
spec.loader.exec_module(helper)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(raw):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")


class EnvironmentTests(unittest.TestCase):
    def test_rejected_directory_leaves_close_descriptors(self):
        for relative in (
            "torch-1.0.dist-info/RECORD",
            "torch-1.0.dist-info/METADATA",
            "torch/__init__.py",
        ):
            with self.subTest(relative=relative):
                path = self.target / relative
                original = path.read_bytes()
                path.unlink()
                path.mkdir()
                try:
                    before = len(os.listdir("/proc/self/fd"))
                    for _ in range(3):
                        with self.assertRaises(ValueError):
                            self.verify()
                    self.assertEqual(len(os.listdir("/proc/self/fd")), before)
                finally:
                    path.rmdir()
                    path.write_bytes(original)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = Path(self.tmp.name) / "installed"
        self.target.mkdir()
        self.old = "/inert-original/sfora/site-packages"
        self.anchor = "/inert-system/sfora/liboriginal.so"
        self.members = {}
        self.profiles = {}
        self.packages = {}
        for package in ["PIL", "numpy", "safetensors", "torch", "torchvision", "transformers"]:
            dist = ("pillow" if package == "PIL" else package) + "-1.0.dist-info"
            metadata = f"Name: {dist.split('-1.0')[0]}\nVersion: 1.0\n".encode()
            members = {package + "/__init__.py": b'raise RuntimeError("must never import")\n'}
            if package == "transformers":
                members.update(
                    {
                        "transformers/model.py": b"# constructor source\n",
                        "transformers/native.so": b"not a native library",
                    }
                )
            rows = []
            selected = []
            record = self.old + "/" + dist + "/RECORD"
            for rel, raw in members.items():
                p = self.target / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(raw)
                full = self.old + "/" + rel
                self.members[full] = digest(raw)
                rows.append(f"{rel},{encoded(raw)},{len(raw)}\n")
                selected.append(
                    {
                        "file": full,
                        "record_relative_path": rel,
                        "sha256": digest(raw),
                        "bytes": len(raw),
                        "record": record,
                        "record_sha256": "",
                    }
                )
            rows.append(f"{dist}/METADATA,{encoded(metadata)},{len(metadata)}\n")
            rows.append("../../../never-open-this,sha256=irrelevant,99\n")
            record_bytes = "".join(rows).encode()
            d = self.target / dist
            d.mkdir()
            (d / "METADATA").write_bytes(metadata)
            (d / "RECORD").write_bytes(record_bytes)
            for member in selected:
                member["record_sha256"] = digest(record_bytes)
            self.profiles[record] = {
                "record_sha256": digest(record_bytes),
                "name": dist.split("-1.0")[0],
                "version": "1.0",
                "metadata": {
                    "path": self.old + "/" + dist + "/METADATA",
                    "sha256": digest(metadata),
                    "bytes": len(metadata),
                },
                "selected_members": selected,
            }
            self.packages[package] = {
                "root": self.old + "/" + package,
                "origin": self.old + "/" + package + "/__init__.py",
                "version": "1.0",
            }
        self.environment = {
            "files": {**self.members, self.anchor: "1" * 64},
            "native_files": {
                self.old + "/transformers/native.so": self.members[
                    self.old + "/transformers/native.so"
                ],
                self.anchor: "1" * 64,
            },
            "packages": self.packages,
            "vision_constructor": self.old + "/transformers/model.py",
        }
        self.bundle = {
            "schema": "siglip2-connected-mlp-bundle-v1",
            "environment": self.environment,
            "code": {},
            "files": {},
            "encoder_identity": {},
            "base_vision_sha256": "2" * 64,
            "vision_sha256": "2" * 64,
            "endpoint_state_sha256": "2" * 64,
            "scope": {},
        }
        self.audit = {
            "schema": "connected-original-complete-record-ownership-source-audit-v1",
            "profiles": self.profiles,
            "records": 196,
            "selected_members": 8,
            "selected_native_members": 1,
            "current_python_sources_sha_pass": 7,
            "native_file_bytes_hashed": False,
            "native_modules_imported": False,
            "native_qualification": False,
            "product_go": False,
            "unique_owner_pass": True,
            "original_record_sha_and_metadata_row_pass": True,
        }

    def verify(self, bundle=None, audit=None, root=None, bundle_pin=None, audit_pin=None):
        b = json.dumps(self.bundle if bundle is None else bundle).encode()
        a = json.dumps(self.audit if audit is None else audit).encode()
        return helper.verify_installed_environment(
            b,
            a,
            trusted_bundle_sha256=digest(b) if bundle_pin is None else bundle_pin,
            trusted_ownership_audit_sha256=digest(a) if audit_pin is None else audit_pin,
            site_packages=str(self.target) if root is None else root,
        )

    def test_valid_complete_inverse_and_no_original_or_unselected_reads(self):
        original = json.dumps(self.bundle)
        opened = []
        read = helper._read_file

        def spy(path, *args, **kwargs):
            opened.append(str(path))
            self.assertTrue(str(path).startswith(str(self.target) + "/"))
            return read(path, *args, **kwargs)

        modules = set(sys.modules)
        with patch.object(helper, "_read_file", side_effect=spy):
            result = self.verify()
        env = result["expected_environment"]
        self.assertEqual(env["files"][self.anchor], "1" * 64)
        self.assertEqual(len(env["files"]), len(self.environment["files"]))
        self.assertEqual(env["native_files"][self.anchor], "1" * 64)
        self.assertEqual(env["vision_constructor"], str(self.target / "transformers/model.py"))
        self.assertEqual(len(opened), 6 + 6 + 8)
        self.assertEqual(modules, set(sys.modules))
        self.assertEqual(original, json.dumps(self.bundle))
        self.assertEqual(
            set(result),
            {
                "schema",
                "original_bundle_sha256",
                "original_ownership_audit_sha256",
                "site_packages",
                "distributions",
                "expected_environment",
            },
        )

    def test_trusted_pins_before_parsing_or_target_reads(self):
        with patch.object(
            helper, "_read_file", side_effect=AssertionError("premature target read")
        ):
            for kwargs in [{"bundle_pin": "0" * 64}, {"audit_pin": "0" * 64}]:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.verify(**kwargs)
            with self.assertRaises(ValueError):
                helper.verify_installed_environment(
                    b"not JSON",
                    b"not JSON",
                    trusted_bundle_sha256="0" * 64,
                    trusted_ownership_audit_sha256="0" * 64,
                    site_packages=str(self.target),
                )

    def test_omission_duplicate_addition_hash_and_path_forgery_before_reads(self):
        for mutant in [
            "omit",
            "duplicate",
            "add",
            "hash",
            "relative",
            "record",
            "size_bool",
            "metadata",
        ]:
            audit = json.loads(json.dumps(self.audit))
            p = next(iter(audit["profiles"].values()))
            m = p["selected_members"][0]
            if mutant == "omit":
                p["selected_members"].pop()
            elif mutant == "duplicate":
                p["selected_members"].append(dict(m))
            elif mutant == "add":
                p["selected_members"].append(
                    {**m, "file": self.old + "/extra.py", "record_relative_path": "extra.py"}
                )
            elif mutant == "hash":
                m["sha256"] = "0" * 64
            elif mutant == "relative":
                m["record_relative_path"] = "../outside.py"
            elif mutant == "record":
                m["record"] = self.old + "/foreign.dist-info/RECORD"
            elif mutant == "size_bool":
                m["bytes"] = True
            elif mutant == "metadata":
                p["metadata"]["path"] = self.old + "/foreign/METADATA"
            with (
                self.subTest(mutant=mutant),
                patch.object(
                    helper, "_read_file", side_effect=AssertionError("premature target read")
                ),
                self.assertRaises(ValueError),
            ):
                self.verify(audit=audit)

    def test_fresh_same_size_restored_mtime_source_and_native_mutation(self):
        self.verify()
        for rel in ["torch/__init__.py", "transformers/native.so"]:
            p = self.target / rel
            raw, stamp = p.read_bytes(), p.stat()
            p.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            os.utime(p, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            with self.subTest(rel=rel), self.assertRaises(ValueError):
                self.verify()
            p.write_bytes(raw)
        self.verify()

    def test_record_and_metadata_drift(self):
        for rel in ["torch-1.0.dist-info/RECORD", "torch-1.0.dist-info/METADATA"]:
            p = self.target / rel
            raw = p.read_bytes()
            p.write_bytes(raw + b"\n")
            with self.subTest(rel=rel), self.assertRaises(ValueError):
                self.verify()
            p.write_bytes(raw)

    def test_exact_selected_rows_even_in_authenticated_synthetic_audit(self):
        for mutation in ["duplicate", "padded", "size", "metadata_duplicate"]:
            audit = json.loads(json.dumps(self.audit))
            record, profile = next(iter(audit["profiles"].items()))
            p = self.target / Path(record).relative_to(self.old)
            raw = p.read_bytes()
            rows = raw.decode().splitlines()
            if mutation == "duplicate":
                rows.append(rows[0])
            elif mutation == "padded":
                fields = rows[0].split(",")
                fields[1] += "="
                rows[0] = ",".join(fields)
            elif mutation == "size":
                fields = rows[0].split(",")
                fields[2] = "0" + fields[2]
                rows[0] = ",".join(fields)
            else:
                rows.append(next(row for row in rows if "/METADATA," in row))
            new = ("\n".join(rows) + "\n").encode()
            p.write_bytes(new)
            profile["record_sha256"] = digest(new)
            for m in profile["selected_members"]:
                m["record_sha256"] = digest(new)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.verify(audit=audit)
            p.write_bytes(raw)

    def test_alias_fifo_missing_and_root_overlap(self):
        p = self.target / "torch/__init__.py"
        raw = p.read_bytes()
        p.unlink()
        p.symlink_to(self.target / "numpy/__init__.py")
        with self.assertRaises(ValueError):
            self.verify()
        p.unlink()
        os.mkfifo(p)
        with self.assertRaises(ValueError):
            self.verify()
        p.unlink()
        with self.assertRaises(ValueError):
            self.verify()
        p.write_bytes(raw)
        for root in [
            self.old,
            self.old + "/child",
            "/inert-original",
            str(self.target) + "/..",
            str(self.target) + "/",
            "relative",
            "/inert-system/sfora",
        ]:
            with self.subTest(root=root), self.assertRaises(ValueError):
                self.verify(root=root)
        moved = self.target / "torch-real"
        (self.target / "torch").rename(moved)
        (self.target / "torch").symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.verify()

    def test_strict_json_and_package_version(self):
        b = json.dumps(self.bundle).encode()
        for bad in [
            b.replace(b'"schema":', b'"schema":"duplicate","schema":', 1),
            b.replace(b'"code": {}', b'"code": NaN'),
        ]:
            with self.assertRaises(ValueError):
                helper.verify_installed_environment(
                    bad,
                    json.dumps(self.audit).encode(),
                    trusted_bundle_sha256=digest(bad),
                    trusted_ownership_audit_sha256=digest(json.dumps(self.audit).encode()),
                    site_packages=str(self.target),
                )
        changed = json.loads(json.dumps(self.bundle))
        changed["environment"]["packages"]["torch"]["version"] = "2.0"
        with self.assertRaises(ValueError):
            self.verify(bundle=changed)

    def test_audit_pass_flags_are_not_authority(self):
        changed = json.loads(json.dumps(self.audit))
        changed["unique_owner_pass"] = False
        changed["original_record_sha_and_metadata_row_pass"] = False
        self.assertEqual(
            self.verify(audit=changed)["expected_environment"],
            self.verify()["expected_environment"],
        )

    def test_real_committed_correspondence_without_installed_file_reads(self):
        root = MODULE.parents[2]
        evidence = (
            root
            / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
            / "connected-original-environment-audit-v1"
        )
        bundle = (evidence / "original-bundle.json").read_bytes()
        audit = (evidence / "complete-record-owners.json").read_bytes()
        with patch.object(
            helper, "_read_file", side_effect=AssertionError("real installed read forbidden")
        ):
            result, profiles, members = helper._prepare(
                bundle,
                audit,
                "e12429efd5cf1a6bd43d4bbf1c55fc163548fffb415813e5c3315658c144f153",
                "8e38d2e503a44cdc1f2fe3cf4bf08c2a86dfb1f86603363aef5dcb262812f36f",
                "/inert-relocated-production/site-packages",
            )
        env = result["expected_environment"]
        self.assertEqual(
            (len(profiles), len(members), len(env["files"]), len(env["native_files"])),
            (32, 1754, 1767, 262),
        )
        original = json.loads(bundle)["environment"]
        anchors = {p: h for p, h in original["files"].items() if p not in members}
        self.assertEqual(len(anchors), 13)
        self.assertTrue(all(env["files"][p] == h for p, h in anchors.items()))


if __name__ == "__main__":
    unittest.main()
