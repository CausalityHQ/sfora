#!/usr/bin/env python3
"""Synthetic source-only conversion checks; actual c226 wire/native UNRUN."""

import importlib.abc
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from test_connected_gallery_provenance import encoded, fixture, sha

ROOT = Path(__file__).resolve().parents[1]
NATIVE = {"torch", "numpy", "PIL", "transformers", "safetensors", "torchvision", "sfora_native"}


class NoNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".")[0] in NATIVE:
            raise AssertionError("native import sentinel: " + fullname)


def synthetic(count: int = 33) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    receipt, bundle = fixture(count)
    wire = b"".join(i.to_bytes(2, "little") * 65 for i in range(2 * count))
    receipt["files"]["control-179061.packed.bin"] = sha(wire)
    return receipt, bundle, wire


def ordered_digest(receipt: dict[str, Any]) -> None:
    rows = [row for batch in receipt["images"] for row in batch["rows"]]
    receipt["ordered_images_sha256"] = sha(encoded(sorted(rows, key=lambda r: r["panel_ordinal"])))


class ConversionTests(unittest.TestCase):
    api: Any
    binder: Any

    @classmethod
    def setUpClass(cls) -> None:
        sentinel = NoNativeImports()
        sys.meta_path.insert(0, sentinel)
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from sfora import connected_gallery_conversion, connected_gallery_provenance

            cls.api = connected_gallery_conversion
            cls.binder = connected_gallery_provenance
        finally:
            sys.path.pop(0)
            sys.meta_path.remove(sentinel)
        assert not NATIVE.intersection(sys.modules)

    def extract(
        self,
        receipt: dict[str, Any],
        bundle: dict[str, Any],
        wire: bytes,
        owners: bytes = b"opaque original owner evidence\n",
    ) -> dict[str, bytes]:
        # These are fresh SYNTHETIC pins, never the actual accepted wire pin.
        raw = json.dumps(receipt, ensure_ascii=False, indent=2).encode() + b"\n"
        manifest = encoded(bundle)
        result: dict[str, bytes] = self.api.extract_gallery_members(
            raw,
            manifest,
            owners,
            wire,
            trusted_receipt_sha256=sha(raw),
            trusted_bundle_sha256=sha(manifest),
            trusted_ownership_audit_sha256=sha(owners),
        )
        return result

    def assert_slices(self, wire: bytes, selected: bytes, ordinals: list[int]) -> None:
        self.assertEqual(len(selected), len(ordinals) * 130)
        for j, ordinal in enumerate(ordinals):
            # Independent byte-position oracle checks all 130 bytes of every row.
            self.assertEqual(
                selected[j * 130 : (j + 1) * 130],
                bytes(wire[ordinal * 130 + k] for k in range(130)),
            )

    def test_all_slices_and_verbatim_members_preserve_producer(self) -> None:
        for count in (1, 32, 33, 65):
            with self.subTest(count=count):
                receipt, bundle, wire = synthetic(count)
                before = encoded([receipt, bundle])
                owners = b"not parsed or normalized \xff\n"
                result = self.extract(receipt, bundle, wire, owners)
                raw = json.dumps(receipt, ensure_ascii=False, indent=2).encode() + b"\n"
                manifest = encoded(bundle)
                producer = self.binder.bind_gallery_provenance(
                    raw,
                    manifest,
                    trusted_receipt_sha256=sha(raw),
                    trusted_bundle_sha256=sha(manifest),
                )
                self.assertEqual(
                    set(result),
                    {
                        "origin.json",
                        "origin-export.json",
                        "origin-owners.json",
                        "gallery.bin",
                        "gallery-ids.json",
                        "gallery-provenance.json",
                    },
                )
                self.assertEqual(result["origin.json"], manifest)
                self.assertEqual(result["origin-export.json"], raw)
                self.assertEqual(result["origin-owners.json"], owners)
                self.assertEqual(encoded([receipt, bundle]), before)
                rows = [r for b in receipt["images"] if b["role"] == "gallery" for r in b["rows"]]
                self.assert_slices(wire, result["gallery.bin"], list(range(1, count * 2, 2)))
                self.assertEqual(
                    json.loads(result["gallery-ids.json"]), [r["relative_path"] for r in rows]
                )
                self.assertEqual(
                    result["gallery-ids.json"], encoded([r["relative_path"] for r in rows])
                )
                expected = {
                    "schema": "connected-gallery-wire-selection-v1",
                    "producer": producer,
                    "gallery_wire_sha256": sha(result["gallery.bin"]),
                    "ordered_ids_sha256": sha(result["gallery-ids.json"]),
                }
                self.assertEqual(result["gallery-provenance.json"], encoded(expected))
                self.assertEqual(producer["gallery_rows"], rows)
                expected_batches = [
                    {
                        **{k: v for k, v in batch.items() if k != "rows"},
                        "panel_ordinals": [r["panel_ordinal"] for r in batch["rows"]],
                    }
                    for batch in receipt["images"]
                    if batch["role"] == "gallery"
                ]
                self.assertEqual(producer["gallery_batches"], expected_batches)
                self.assertIsNone(producer["gallery_wire_sha256"])
                self.assertEqual(self.extract(receipt, bundle, wire, owners), result)

    def test_slice_oracle_rejects_suffix_reorder_and_query_mutants(self) -> None:
        receipt, bundle, wire = synthetic()
        good = self.extract(receipt, bundle, wire)["gallery.bin"]
        mutants = (
            wire[-len(good) :],
            good[130:] + good[:130],
            wire[:130] + good[130:],
        )
        for mutant in mutants:
            with self.subTest(sha=sha(mutant)):
                self.assertEqual(len(mutant), len(good))
                self.assertEqual(len(sha(mutant)), 64)
                with self.assertRaises(AssertionError):
                    self.assert_slices(wire, mutant, list(range(1, 66, 2)))

    def test_all_original_pins_precede_json_parsing(self) -> None:
        originals = [b"invalid receipt JSON", b"invalid bundle JSON", b"opaque owners"]
        keys = ["trusted_receipt_sha256", "trusted_bundle_sha256", "trusted_ownership_audit_sha256"]
        for side in range(3):
            pins = dict(zip(keys, map(sha, originals), strict=True))
            pins[keys[side]] = "0" * 64
            with self.subTest(side=side), patch.object(self.binder.json, "loads") as loads:
                with self.assertRaisesRegex(ValueError, "trusted .*SHA"):
                    self.api.extract_gallery_members(*originals, b"wire", **pins)
                loads.assert_not_called()
        receipt, bundle, wire = synthetic()
        raw, manifest, owners = encoded(receipt), encoded(bundle), b"accepted owners"
        with self.assertRaisesRegex(ValueError, "ownership.*SHA"):
            self.api.extract_gallery_members(
                raw,
                manifest,
                owners + b"substitution",
                wire,
                trusted_receipt_sha256=sha(raw),
                trusted_bundle_sha256=sha(manifest),
                trusted_ownership_audit_sha256=sha(owners),
            )

    def test_wire_hash_and_length_reject_mutations(self) -> None:
        receipt, bundle, wire = synthetic()
        for mutant in (bytes([wire[0] ^ 1]) + wire[1:], wire[:-1], wire + b"\0"):
            with self.subTest(length=len(mutant)), self.assertRaises(ValueError):
                self.extract(receipt, bundle, mutant)
        for mutant in (wire[:-1], wire + b"\0"):
            # Synthetic repinning isolates byte-count checks from the SHA check.
            receipt["files"]["control-179061.packed.bin"] = sha(mutant)
            with self.subTest(repinned_length=len(mutant)), self.assertRaises(ValueError):
                self.extract(receipt, bundle, mutant)

    def test_fresh_synthetic_pins_do_not_hide_invalid_rows_counts_or_ids(self) -> None:
        for mutation in (
            "duplicate_ids",
            "empty_id",
            "bool_count",
            "bool_ordinal",
            "row_order",
            "batch_order",
        ):
            receipt, bundle, wire = synthetic()
            first, tail = receipt["images"][2:]
            if mutation == "duplicate_ids":
                tail["rows"][0]["relative_path"] = first["rows"][0]["relative_path"]
                ordered_digest(receipt)
            elif mutation == "empty_id":
                first["rows"][0]["relative_path"] = ""
                ordered_digest(receipt)
            elif mutation == "bool_count":
                receipt["panel_facts"]["raw"]["shape"][0] = True
            elif mutation == "bool_ordinal":
                first["rows"][0]["panel_ordinal"] = True
            elif mutation == "row_order":
                first["rows"][0], first["rows"][1] = first["rows"][1], first["rows"][0]
            else:
                receipt["images"][2:] = [tail, first]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.extract(receipt, bundle, wire)

    def test_builtin_bytes_only_and_no_path_opens_or_native_imports(self) -> None:
        class BytesSubclass(bytes):
            pass

        for side in range(4):
            for value in (bytearray(b"x"), memoryview(b"x"), BytesSubclass(b"x"), "x"):
                inputs: list[Any] = [b"x"] * 4
                inputs[side] = value
                with (
                    self.subTest(side=side, type=type(value)),
                    self.assertRaisesRegex(ValueError, "immutable"),
                ):
                    self.api.extract_gallery_members(
                        *inputs,
                        trusted_receipt_sha256=sha(b"x"),
                        trusted_bundle_sha256=sha(b"x"),
                        trusted_ownership_audit_sha256=sha(b"x"),
                    )
        receipt, bundle, wire = synthetic()
        sentinel = NoNativeImports()
        sys.meta_path.insert(0, sentinel)
        try:
            with (
                patch("builtins.open", side_effect=AssertionError("open called")),
                patch("io.open", side_effect=AssertionError("io.open called")),
                patch("os.open", side_effect=AssertionError("os.open called")),
            ):
                self.extract(receipt, bundle, wire)
        finally:
            sys.meta_path.remove(sentinel)
        self.assertFalse(NATIVE.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
