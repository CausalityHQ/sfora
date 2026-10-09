#!/usr/bin/env python3
"""Synthetic manifest binding only; payload/native/serving admission is untested."""

import copy
import importlib.abc
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from test_connected_gallery_provenance import encoded, fixture, sha

ROOT = Path(__file__).resolve().parents[1]
NATIVE = {
    "torch",
    "numpy",
    "PIL",
    "transformers",
    "safetensors",
    "torchvision",
    "sfora_native",
    "ctypes",
    "cffi",
}


class NoNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".")[0] in NATIVE or fullname.startswith("sfora._native"):
            raise AssertionError("native import sentinel: " + fullname)


class ServingTests(unittest.TestCase):
    api: Any
    provenance: Any
    converter: Any

    @classmethod
    def setUpClass(cls) -> None:
        sentinel = NoNativeImports()
        assert not NATIVE.intersection(sys.modules)
        sys.meta_path.insert(0, sentinel)
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from sfora import (
                connected_gallery_conversion,
                connected_gallery_provenance,
                connected_serving_artifact,
            )

            cls.api = connected_serving_artifact
            cls.provenance = connected_gallery_provenance
            cls.converter = connected_gallery_conversion
        finally:
            sys.path.pop(0)
            sys.meta_path.remove(sentinel)
        assert not NATIVE.intersection(sys.modules)

    def setUp(self) -> None:
        self.prepare(33)

    def prepare(self, count: int) -> None:
        receipt, bundle = fixture(count)
        wire = b"".join(i.to_bytes(2, "little") * 65 for i in range(count * 2))
        receipt["files"]["control-179061.packed.bin"] = sha(wire)
        raw = json.dumps(receipt, ensure_ascii=False, indent=2).encode() + b"\n"
        original = encoded(bundle)
        owners = b"opaque owners, not JSON: \xff\x00\n"
        self.fragment: dict[str, bytes] = self.converter.extract_gallery_members(
            raw,
            original,
            owners,
            wire,
            trusted_receipt_sha256=sha(raw),
            trusted_bundle_sha256=sha(original),
            trusted_ownership_audit_sha256=sha(owners),
        )
        self.pins = {name: sha(value) for name, value in self.fragment.items()}
        self.producer = self.provenance.bind_gallery_provenance(
            raw,
            original,
            trusted_receipt_sha256=self.pins["origin-export.json"],
            trusted_bundle_sha256=self.pins["origin.json"],
        )
        self.manifest: dict[str, Any] = {
            "schema": "siglip2-connected-mlp-serving-v2",
            "files": {
                **{
                    name: {"path": name, "bytes": len(value), "sha256": sha(value)}
                    for name, value in self.fragment.items()
                },
                **{
                    name: {"path": name, "bytes": 123, "sha256": digest}
                    for name, digest in bundle["files"].items()
                },
            },
            "origin": {
                "bundle_sha256": self.pins["origin.json"],
                "endpoint_state_sha256": bundle["endpoint_state_sha256"],
                "fixed_sha256": self.producer["identity"]["fixed_sha256"],
            },
            "gallery": {
                "count": count,
                "dimensions": 128,
                "bytes_per_row": 130,
                "wire_sha256": self.pins["gallery.bin"],
                "ordered_ids_sha256": self.pins["gallery-ids.json"],
                "provenance_sha256": self.pins["gallery-provenance.json"],
                "encoder_binding_sha256": self.association(self.producer),
            },
            "request": {
                "device": "cuda",
                "batch_min": 1,
                "batch_max": 32,
                "dimensions": 128,
                "k": 10,
                "ties": "ordinal-ascending",
            },
        }

    def association(self, producer: dict[str, Any]) -> str:
        return sha(
            encoded({key: producer[key] for key in ("identity", "gallery_batches", "gallery_rows")})
        )

    def bind(self, raw: bytes | None = None) -> Any:
        serving = encoded(self.manifest) if raw is None else raw
        return self.api.bind_serving_manifest(
            serving,
            self.fragment,
            trusted_serving_sha256=sha(serving),
            trusted_fragment_sha256=self.pins,
        )

    def repin(self, name: str, raw: bytes) -> None:
        # Only synthetic evidence is repinned; original receipt/bundle pins stay fixed.
        assert name not in {"origin.json", "origin-export.json"}
        self.fragment[name] = raw
        self.pins[name] = sha(raw)
        self.manifest["files"][name] = {"path": name, "bytes": len(raw), "sha256": sha(raw)}
        field = {
            "gallery-provenance.json": "provenance_sha256",
            "gallery-ids.json": "ordered_ids_sha256",
            "gallery.bin": "wire_sha256",
        }.get(name)
        if field is not None:
            self.manifest["gallery"][field] = sha(raw)

    def test_genuine_converter_and_detached_nonmutating_result(self) -> None:
        before = copy.deepcopy((self.fragment, self.pins, self.manifest))
        result = self.bind()
        self.assertEqual(result, {"manifest": self.manifest, "producer": self.producer})
        self.assertIsNone(result["producer"]["gallery_wire_sha256"])
        self.assertEqual(result["producer"]["gallery_batches"][1]["panel_ordinals"], [65])
        result["producer"]["gallery_rows"][0]["relative_path"] = "changed"
        result["manifest"]["request"]["k"] = 99
        self.assertEqual(before, (self.fragment, self.pins, self.manifest))
        self.assertEqual(self.bind(), {"manifest": self.manifest, "producer": self.producer})

    def test_repinned_provenance_cannot_replace_original_producer(self) -> None:
        original_pins = {name: self.pins[name] for name in ("origin.json", "origin-export.json")}
        original = self.fragment["gallery-provenance.json"]
        for mutation in ("encoder", "batch", "rows", "null"):
            with self.subTest(mutation=mutation):
                selection = json.loads(original)
                producer = selection["producer"]
                if mutation == "encoder":
                    producer["identity"]["identity"]["encoder_identity"]["runtime"] = {
                        "forged": True
                    }
                elif mutation == "batch":
                    producer["gallery_batches"][0]["panel_ordinals"][0] = 0
                elif mutation == "rows":
                    producer["gallery_rows"].reverse()
                else:
                    producer["gallery_wire_sha256"] = selection["gallery_wire_sha256"]
                self.repin("gallery-provenance.json", encoded(selection))
                self.assertEqual(original_pins, {name: self.pins[name] for name in original_pins})
                for digest in (self.association(self.producer), self.association(producer)):
                    self.manifest["gallery"]["encoder_binding_sha256"] = digest
                    with self.assertRaises(ValueError):
                        self.bind()

    def test_all_pins_precede_every_json_parse(self) -> None:
        bad_json = b"{not JSON"
        with (
            patch.object(self.api, "_parse", side_effect=AssertionError("early manifest parse")),
            patch.object(
                self.provenance, "_parse", side_effect=AssertionError("early origin parse")
            ),
        ):
            for name in self.fragment:
                with self.subTest(name=name):
                    pins = {**self.pins, name: "0" * 64}
                    with self.assertRaisesRegex(ValueError, "trusted " + name + " SHA differs"):
                        self.api.bind_serving_manifest(
                            bad_json,
                            self.fragment,
                            trusted_serving_sha256=sha(bad_json),
                            trusted_fragment_sha256=pins,
                        )
            with self.assertRaisesRegex(ValueError, "trusted serving SHA differs"):
                self.api.bind_serving_manifest(
                    bad_json,
                    self.fragment,
                    trusted_serving_sha256="0" * 64,
                    trusted_fragment_sha256=self.pins,
                )
        # A correct serving pin must not allow a malformed receipt to hide a wrong bundle pin.
        self.fragment["origin-export.json"] = bad_json
        self.pins["origin-export.json"] = sha(bad_json)
        self.pins["origin.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "trusted origin.json SHA differs"):
            self.bind()

    def test_builtin_types_and_exact_input_sets(self) -> None:
        class ByteSubclass(bytes):
            pass

        class DictSubclass(dict[str, Any]):
            pass

        class StringSubclass(str):
            pass

        serving = encoded(self.manifest)
        arguments: dict[str, Any] = {
            "serving": serving,
            "fragment": self.fragment,
            "trusted_serving_sha256": sha(serving),
            "trusted_fragment_sha256": self.pins,
        }
        changes: list[tuple[str, object]] = [
            ("serving", bytearray(serving)),
            ("serving", ByteSubclass(serving)),
            ("fragment", DictSubclass(self.fragment)),
            ("trusted_fragment_sha256", DictSubclass(self.pins)),
            ("fragment", {**self.fragment, "origin.json": memoryview(b"{}")}),
            ("fragment", {**self.fragment, "gallery.bin": ByteSubclass(b"wire")}),
            ("fragment", {**self.fragment, "extra.json": b"{}"}),
            ("trusted_fragment_sha256", {**self.pins, "extra.json": "0" * 64}),
            ("fragment", {k: v for k, v in self.fragment.items() if k != "origin-owners.json"}),
            ("trusted_fragment_sha256", {k: v for k, v in self.pins.items() if k != "gallery.bin"}),
            ("fragment", {StringSubclass(k): v for k, v in self.fragment.items()}),
            ("trusted_fragment_sha256", {StringSubclass(k): v for k, v in self.pins.items()}),
            ("trusted_fragment_sha256", {**self.pins, "gallery.bin": StringSubclass("0" * 64)}),
        ]
        changes.extend(
            ("trusted_serving_sha256", value)
            for value in (True, 1, None, "A" * 64, "g" * 64, "0" * 63, StringSubclass(sha(serving)))
        )
        for index, (key, value) in enumerate(changes):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.api.bind_serving_manifest(**{**arguments, key: value})

    def test_snapshot_keeps_authenticated_bytes_and_pins_together(self) -> None:
        original_sha256 = self.api.hashlib.sha256
        last_input = self.fragment["gallery-provenance.json"]

        def change_caller_after_hash(raw: bytes) -> Any:
            digest = original_sha256(raw)
            if raw is last_input:
                self.fragment["origin-export.json"] = b"changed by caller after authentication"
                self.pins["origin-export.json"] = sha(self.fragment["origin-export.json"])
            return digest

        with patch.object(self.api.hashlib, "sha256", side_effect=change_caller_after_hash):
            self.assertEqual(self.bind()["producer"], self.producer)

    def test_manifest_exact_keys_and_file_records(self) -> None:
        original = copy.deepcopy(self.manifest)
        for path in (
            (),
            ("files",),
            ("files", "gallery.bin"),
            ("origin",),
            ("gallery",),
            ("request",),
        ):
            for extra in (False, True):
                with self.subTest(path=path, extra=extra):
                    self.manifest = copy.deepcopy(original)
                    target = self.manifest
                    for key in path:
                        target = target[key]
                    if extra:
                        target["unexpected"] = None
                    else:
                        target.pop(next(iter(target)))
                    with self.assertRaises(ValueError):
                        self.bind()

    def test_payload_sha_and_original_identity_are_bound(self) -> None:
        original = copy.deepcopy(self.manifest)
        for name in ("vision.pt", "endpoint.pt", "processor.json"):
            with self.subTest(payload=name):
                self.manifest = copy.deepcopy(original)
                self.manifest["files"][name]["sha256"] = sha(b"substitute payload")
                with self.assertRaisesRegex(ValueError, "original payload SHA differs"):
                    self.bind()
        self.manifest = copy.deepcopy(original)
        self.manifest["files"]["processor.json"]["sha256"] = original["files"]["vision.pt"][
            "sha256"
        ]
        with self.assertRaisesRegex(ValueError, "original payload SHA differs"):
            self.bind()
        for key in original["origin"]:
            with self.subTest(origin=key):
                self.manifest = copy.deepcopy(original)
                self.manifest["origin"][key] = "0" * 64
                with self.assertRaisesRegex(ValueError, "original identity differs"):
                    self.bind()
        self.manifest = copy.deepcopy(original)
        for name, size in (("vision.pt", 0), ("endpoint.pt", 2**80), ("processor.json", 1)):
            self.manifest["files"][name]["bytes"] = size
        self.assertEqual(self.bind()["manifest"], self.manifest)  # Expected sizes only.

    def test_fragment_file_facts_must_match_actual_bytes(self) -> None:
        original = copy.deepcopy(self.manifest)
        for name in self.fragment:
            for key, value in (("bytes", 0), ("sha256", "0" * 64)):
                with self.subTest(name=name, key=key):
                    self.manifest = copy.deepcopy(original)
                    self.manifest["files"][name][key] = value
                    with self.assertRaises(ValueError):
                        self.bind()
        self.manifest = copy.deepcopy(original)
        for name, other in (
            ("origin.json", "origin-export.json"),
            ("origin-export.json", "origin.json"),
        ):
            for key in ("sha256", "bytes"):
                self.manifest["files"][name][key] = original["files"][other][key]
        with self.assertRaisesRegex(ValueError, "fragment size differs"):
            self.bind()

    def test_exact_types_values_and_basenames(self) -> None:
        original = copy.deepcopy(self.manifest)
        changes = [
            (("schema",), "siglip2-connected-mlp-serving-v1"),
            (("gallery", "count"), True),
            (("gallery", "count"), 33.0),
            (("gallery", "dimensions"), 64),
            (("gallery", "bytes_per_row"), 128),
            (("gallery", "encoder_binding_sha256"), self.producer["identity"]["fixed_sha256"]),
            (("request", "batch_min"), True),
            (("request", "batch_max"), 32.0),
            (("request", "device"), "cpu"),
            (("request", "k"), 11),
            (("request", "dimensions"), 64),
            (("request", "ties"), "unspecified"),
        ]
        changes.extend((("files", "vision.pt", "bytes"), value) for value in (True, -1, 1.0))
        changes.extend(
            (("files", "vision.pt", "sha256"), value)
            for value in (None, "A" * 64, "g" * 64, "0" * 63)
        )
        changes.extend(
            (("files", "vision.pt", "path"), value)
            for value in (
                "/vision.pt",
                "../vision.pt",
                "a/vision.pt",
                "./vision.pt",
                "a\\vision.pt",
                "vision.pt\0",
                "",
                "endpoint.pt",
            )
        )
        for path, value in changes:
            with self.subTest(path=path, value=value):
                self.manifest = copy.deepcopy(original)
                target = self.manifest
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(ValueError):
                    self.bind()

    def test_strict_json_and_utf8(self) -> None:
        raw = encoded(self.manifest)
        for malformed in (
            b'{"schema":"duplicate",' + raw[1:],
            raw.replace(b'"count":33', b'"count":1e999'),
            raw.replace(b'"count":33', b'"count":NaN'),
            raw.replace(b'"count":33', b'"count":Infinity'),
            raw.replace(b'"count":33', b'"count":-Infinity'),
            raw.replace(b"cuda", b"\xff"),
            b"[]",
            raw + b"{}",
            raw.decode().encode("utf-16"),
            raw.decode().encode("utf-32-le"),
        ):
            with self.subTest(raw=malformed[:60]), self.assertRaises(ValueError):
                self.bind(malformed)
        for name in (
            "origin.json",
            "origin-export.json",
            "gallery-provenance.json",
            "gallery-ids.json",
        ):
            original = self.fragment[name]
            original_pin = self.pins[name]
            for malformed in (b"\xff", b'{"a":1,"a":2}', b'{"a":1e999}'):
                with self.subTest(name=name, raw=malformed):
                    self.fragment[name] = malformed
                    self.pins[name] = sha(malformed)
                    with self.assertRaises(ValueError):
                        self.bind()
            self.fragment[name], self.pins[name] = original, original_pin

    def test_canonical_selection_and_ids(self) -> None:
        original = self.fragment["gallery-provenance.json"]
        selection = json.loads(original)
        for value in (
            encoded({k: v for k, v in selection.items() if k != "producer"}),
            encoded({**selection, "extra": None}),
            encoded({**selection, "schema": "wrong"}),
            original + b"\n",
            b'{"schema":"duplicate",' + original[1:],
        ):
            with self.subTest(selection=value[:40]):
                self.repin("gallery-provenance.json", value)
                with self.assertRaises(ValueError):
                    self.bind()
        self.repin("gallery-provenance.json", original)
        ids = json.loads(self.fragment["gallery-ids.json"])
        for changed in (
            encoded(ids[::-1]),
            encoded([ids[0], *ids[:-1]]),
            encoded(ids[:-1]),
            encoded(ids) + b"\n",
        ):
            with self.subTest(ids=changed[:40]):
                self.repin("gallery-ids.json", changed)
                with self.assertRaises(ValueError):
                    self.bind()

    def test_wire_reorder_duplicate_and_length(self) -> None:
        original = self.fragment["gallery.bin"]
        for changed in (
            original[130:] + original[:130],
            original[:130] + original[:-130],
            original[:-1],
            original + b"\0",
        ):
            with self.subTest(length=len(changed)):
                self.fragment["gallery.bin"] = changed
                with self.assertRaisesRegex(ValueError, "trusted gallery.bin SHA differs"):
                    self.bind()
                self.repin("gallery.bin", changed)
                with self.assertRaises(ValueError):
                    self.bind()
                self.repin("gallery.bin", original)

    def test_minimum_gallery_size(self) -> None:
        self.prepare(9)
        with self.assertRaisesRegex(ValueError, "at least ten gallery rows required"):
            self.bind()
        self.prepare(10)
        self.assertEqual(self.bind()["manifest"]["gallery"]["count"], 10)

    def test_json_size_bound_before_parse(self) -> None:
        # A length sentinel exercises the bound without allocating a large fixture.
        oversized = b"synthetic oversized JSON"

        def size(value: Any) -> int:
            return 64 * 1024 * 1024 + 1 if value is oversized else len(value)

        with (
            patch.object(self.api, "len", side_effect=size, create=True),
            patch.object(self.api, "_parse", side_effect=AssertionError("early parse")),
        ):
            with self.assertRaisesRegex(ValueError, "JSON exceeds 64 MiB"):
                self.bind(oversized)
            for name in (
                "origin.json",
                "origin-export.json",
                "gallery-ids.json",
                "gallery-provenance.json",
            ):
                with self.subTest(name=name):
                    original, pin = self.fragment[name], self.pins[name]
                    self.fragment[name] = oversized
                    self.pins[name] = sha(oversized)
                    with self.assertRaisesRegex(ValueError, "^JSON exceeds 64 MiB$"):
                        self.bind()
                    self.fragment[name], self.pins[name] = original, pin

    def test_no_filesystem_or_native_access(self) -> None:
        active = False

        def audit(event: str, args: tuple[object, ...]) -> None:
            if active and (event == "open" or event.startswith(("os.", "subprocess.", "ctypes."))):
                raise AssertionError("filesystem/native audit: " + event)

        sys.addaudithook(audit)
        sentinel = NoNativeImports()
        sys.meta_path.insert(0, sentinel)
        try:
            active = True
            with (
                self.assertRaisesRegex(AssertionError, "filesystem/native audit: open"),
                open("/unread/audit-sentinel"),
            ):
                pass
            result = self.bind()
        finally:
            active = False
            sys.meta_path.remove(sentinel)
        self.assertEqual(result["producer"], self.producer)
        self.assertFalse(NATIVE.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
