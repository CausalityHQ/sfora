#!/usr/bin/env python3
"""Source-only metadata checks; no payload, native or serving qualification."""

import copy
import hashlib
import importlib.abc
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/sfora/connected_gallery_provenance.py"
EVIDENCE = ROOT / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
RECEIPT_SHA = "db63db8270f2bf9a75448863e7ff701b7894b0c6aaf45c748730b6ea52393407"
BUNDLE_SHA = "e12429efd5cf1a6bd43d4bbf1c55fc163548fffb415813e5c3315658c144f153"
SCOPE = {
    "arm": "control",
    "manifest_sha256": "55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726",
    "arm_sha256": "1f3ad34bbd20a3b375ccb4908f9a3da05b63b514395cb553e1c81925789b2280",
}
ORACLE = {
    "C_exact_zero": False,
    "residual_nonzero_witness": True,
    "omitted_C_mutant_rejected": True,
    "wrong_mu_mutant_rejected": True,
}
MEMBERS = [
    "config",
    "buffers",
    "processor",
    "head",
    "A",
    "means",
    "C",
    "mu_train",
    "mu_train_provenance",
    "scope",
    "common_statistics",
    "base_vision",
    "encoder",
    "encoder_identity",
    "arm",
]


def encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixture(role_count: int = 33) -> tuple[dict[str, Any], dict[str, Any]]:
    h = sha(b"synthetic metadata, never native proof")
    bundle_path = "/unread/original/control-179061-bundle/bundle.json"
    encoder = {"buffers_sha256": h, "inventory": {}, "runtime": {"modules": []}}
    bundle: dict[str, Any] = {
        "schema": "siglip2-connected-mlp-bundle-v1",
        "base_vision_sha256": h,
        "vision_sha256": h,
        "endpoint_state_sha256": h,
        "encoder_identity": encoder,
        "scope": SCOPE,
        "files": {"endpoint.pt": h, "vision.pt": h, "processor.json": sha(b"file")},
        "code": {"train_siglip2_connected_mlp.py": h, "test_siglip2_connected_mlp.py": h},
        "environment": {},
    }
    descriptor = {"path": bundle_path, "sha256": sha(encoded(bundle))}
    source = {"warm_source": {"source_checkpoint": {"path": "/unread/base.pt", "sha256": h}}}
    flags = {"threads": 1, "autocast_cuda": False}
    identity = {
        "arm": "control",
        "seed": 179061,
        "scope": SCOPE,
        "source": source,
        "encoder_identity": encoder,
        "numerical_flags": flags,
        "base_vision": {"sha256": h, "checkpoint": source["warm_source"]["source_checkpoint"]},
    }
    endpoint = {
        "arm": "control",
        "seed": 179061,
        "bundle": descriptor,
        "inference_state_sha256": h,
        "terminal_state_sha256": h,
        "checkpoint": {"path": "/unread/endpoint.pt", "sha256": h},
        "launch": {"path": "/unread/launch.json", "sha256": h},
        "terminal": {
            "receipt": {"path": "/unread/receipt.json", "sha256": h},
            "log": {"path": "/unread/original.log", "sha256": h},
        },
    }
    facts = {
        "identity": identity,
        "members": dict.fromkeys(MEMBERS, h),
        "bundle": descriptor,
        "vision_sha256": h,
        "base_vision_sha256": h,
        "fixed_sha256": h,
        "processor_config_sha256": sha(b"config"),
        "inference_state_sha256": h,
        "terminal_state_sha256": h,
    }
    images = []
    all_rows: list[dict[str, Any]] = []
    # Interleaved panel roles distinguish panel order from producer batch order.
    for role, offset in (("query", 0), ("gallery", 1)):
        rows = [
            {
                "image_sha256": h,
                "original_row": i + 100,
                "panel_ordinal": i,
                "path": f"/unread/images/é{i}.jpg",
                "relative_path": f"Img/é{i}.jpg",
                "product": "id_1",
                "role": role,
                "target": 0,
                "train_row": i + 200,
            }
            for i in range(offset, role_count * 2, 2)
        ]
        all_rows.extend(rows)
        for start in range(0, role_count, 32):
            images.append(
                {
                    "role": role,
                    "rows": rows[start : start + 32],
                    "rgb_sha256": h,
                    "pixels_sha256": h,
                    "outputs_sha256": h,
                    "residual_oracle": ORACLE if start == 0 or start + 32 >= role_count else None,
                }
            )
    guards = {bundle_path: descriptor["sha256"], "/unread/base.pt": h}
    guards.update(
        dict.fromkeys(
            (
                "/unread/endpoint.pt",
                "/unread/launch.json",
                "/unread/receipt.json",
                "/unread/original.log",
                "/unread/evaluate_siglip2_connected_mlp.py",
                "/unread/test_connected_mlp_evaluation.py",
            ),
            h,
        )
    )
    guards["/unread/scope.json"] = SCOPE["manifest_sha256"]
    guards.update(
        {
            bundle_path.rsplit("/", 1)[0] + "/" + k: v
            for k, v in {**bundle["files"], **bundle["code"]}.items()
        }
    )
    receipt = {
        "schema": "siglip2-connected-mlp-evaluation-v1",
        "phase": "export",
        "stage": "full",
        "panel": "selection",
        "arm": "control",
        "seed": 179061,
        "binding": {
            "endpoints": [endpoint],
            "scope": {"path": "/unread/scope.json", "sha256": SCOPE["manifest_sha256"]},
            "stage": "full",
            "panel": "selection",
            "training": {"code": bundle["code"]},
        },
        "payload_facts": facts,
        "inference_state_sha256": h,
        "source": source,
        "numerical_flags": flags,
        "input_guards": guards,
        "source_code": {
            "evaluate_siglip2_connected_mlp.py": h,
            "test_connected_mlp_evaluation.py": h,
        },
        "invocation": {"argv": ["/unread/evaluate_siglip2_connected_mlp.py"]},
        "batch_sizes": {
            role: [min(32, role_count - i) for i in range(0, role_count, 32)]
            for role in ("query", "gallery")
        },
        "images": images,
        "ordered_images_sha256": sha(
            encoded(sorted(all_rows, key=lambda row: row["panel_ordinal"]))
        ),
        "files": {"control-179061" + ext: h for ext in (".raw.npy", ".unit.npy", ".packed.bin")},
        "panel_facts": {
            name: {
                "shape": [role_count * 2] if name == "inverse_norms" else [role_count * 2, 128],
                "dtype": dtype,
                "sha256": h,
            }
            for name, dtype in (
                ("raw", "torch.float32"),
                ("unit", "torch.float32"),
                ("codes", "torch.int8"),
                ("inverse_norms", "torch.float16"),
            )
        },
    }
    receipt["launch"] = {"endpoints": [endpoint]}
    return json.loads(encoded(receipt)), json.loads(encoded(bundle))


class NoNativeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
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


class GalleryTests(unittest.TestCase):
    api: Any

    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("gallery_provenance_under_test", SOURCE)
        assert spec is not None and spec.loader is not None
        cls.api = importlib.util.module_from_spec(spec)
        sentinel = NoNativeImports()
        sys.meta_path.insert(0, sentinel)
        try:
            spec.loader.exec_module(cls.api)
        finally:
            sys.meta_path.remove(sentinel)

    def bind(self, receipt: dict[str, Any], bundle: dict[str, Any]) -> Any:
        raw, manifest = encoded(receipt), encoded(bundle)
        return self.api.bind_gallery_provenance(
            raw, manifest, trusted_receipt_sha256=sha(raw), trusted_bundle_sha256=sha(manifest)
        )

    def test_tiny_association_preserves_full_rows_and_distinct_hashes(self) -> None:
        r, b = fixture()
        before = encoded([r, b])
        result = self.bind(r, b)
        self.assertEqual(
            set(result),
            {
                "schema",
                "original_receipt_sha256",
                "original_bundle_sha256",
                "identity",
                "combined_wire",
                "ordered_images_sha256",
                "gallery_batches",
                "gallery_rows",
                "gallery_wire_sha256",
            },
        )
        self.assertEqual(result["schema"], "connected-gallery-producer-binding-v1")
        self.assertIsNone(result["gallery_wire_sha256"])
        self.assertEqual(
            result["combined_wire"],
            {
                "files": r["files"],
                "count": 66,
                "dimensions": 128,
                "bytes_per_row": 130,
                "panel_facts": r["panel_facts"],
            },
        )
        self.assertEqual(result["gallery_rows"], r["images"][2]["rows"] + r["images"][3]["rows"])
        self.assertEqual(result["gallery_batches"][0]["panel_ordinals"], list(range(1, 64, 2)))
        self.assertEqual(
            result["identity"],
            {
                **r["payload_facts"],
                "numerical_flags": r["numerical_flags"],
                "bundle_files": b["files"],
            },
        )
        self.assertEqual(
            len(
                {
                    result["identity"]["processor_config_sha256"],
                    result["identity"]["members"]["processor"],
                    result["identity"]["bundle_files"]["processor.json"],
                }
            ),
            3,
        )
        result["gallery_rows"][0]["product"] = "changed result"
        self.assertEqual(encoded([r, b]), before)

    def test_both_pins_precede_either_parse(self) -> None:
        for receipt, bundle, receipt_pin, bundle_pin in (
            (b"not JSON", b"{}", sha(b"not JSON"), "0" * 64),
            (b"{}", b"not JSON", "0" * 64, sha(b"not JSON")),
        ):
            with self.subTest(receipt=receipt), patch.object(self.api.json, "loads") as loads:
                with self.assertRaisesRegex(ValueError, "trusted .*SHA"):
                    self.api.bind_gallery_provenance(
                        receipt,
                        bundle,
                        trusted_receipt_sha256=receipt_pin,
                        trusted_bundle_sha256=bundle_pin,
                    )
                loads.assert_not_called()

    def test_actual_committed_metadata_matches_audit_without_original_reads(self) -> None:
        raw = (
            EVIDENCE / "connected-mlp-evaluation-full-export-control-179061-v2/receipt.json"
        ).read_bytes()
        bundle = (
            EVIDENCE / "connected-mlp-train-control-179061-v1/bundle-manifest.json"
        ).read_bytes()
        audit = json.loads(
            (
                EVIDENCE / "connected-artifact-relocation-plan-v1/gallery-source-audit.json"
            ).read_bytes()
        )
        self.assertEqual(sha(raw), RECEIPT_SHA)
        self.assertEqual(sha(bundle), BUNDLE_SHA)
        sentinel = NoNativeImports()
        sys.meta_path.insert(0, sentinel)
        try:
            with (
                patch("builtins.open", side_effect=AssertionError("open called")),
                patch("io.open", side_effect=AssertionError("io.open called")),
                patch("os.open", side_effect=AssertionError("os.open called")),
            ):
                result = self.api.bind_gallery_provenance(
                    raw,
                    bundle,
                    trusted_receipt_sha256=RECEIPT_SHA,
                    trusted_bundle_sha256=BUNDLE_SHA,
                )
        finally:
            sys.meta_path.remove(sentinel)
        r = json.loads(raw)
        self.assertEqual(result["original_receipt_sha256"], audit["receipt"]["sha256"])
        self.assertEqual(result["original_bundle_sha256"], audit["bundle"]["sha256"])
        self.assertEqual(result["ordered_images_sha256"], audit["ordered_images_sha256"])
        self.assertEqual(result["combined_wire"]["files"], audit["combined_wire_files"])
        self.assertEqual(result["combined_wire"]["panel_facts"], audit["panel_facts"])
        self.assertEqual(result["combined_wire"]["count"], 3449)
        self.assertEqual(len(result["gallery_rows"]), 1715)
        self.assertEqual(
            [row["panel_ordinal"] for row in result["gallery_rows"]],
            audit["roles"]["gallery"]["panel_ordinals"],
        )
        self.assertEqual(
            [row["relative_path"] for row in result["gallery_rows"]],
            audit["roles"]["gallery"]["ids_exact_relative_paths"],
        )
        self.assertEqual(
            [len(batch["panel_ordinals"]) for batch in result["gallery_batches"]],
            audit["roles"]["gallery"]["batch_sizes"],
        )
        self.assertEqual(
            result["gallery_rows"],
            [row for batch in r["images"] if batch["role"] == "gallery" for row in batch["rows"]],
        )
        self.assertIsNone(result["gallery_wire_sha256"])
        for original, observed in zip(r["images"][55:], result["gallery_batches"], strict=True):
            self.assertEqual(
                {k: v for k, v in original.items() if k != "rows"},
                {k: v for k, v in observed.items() if k != "panel_ordinals"},
            )
        for mutation in ("cross_batch", "drop", "role"):
            mutant = copy.deepcopy(r)
            first, last = mutant["images"][55], mutant["images"][-1]
            if mutation == "cross_batch":
                first["rows"][0], last["rows"][0] = last["rows"][0], first["rows"][0]
            elif mutation == "drop":
                last["rows"].pop()
                mutant["batch_sizes"]["gallery"][-1] -= 1
            else:
                first["rows"][0]["role"] = "query"
            with self.subTest(actual_mutation=mutation), self.assertRaises(ValueError):
                self.bind(mutant, json.loads(bundle))

    def test_freshly_pinned_row_and_batch_mutants(self) -> None:
        for mutation in (
            "cross_batch",
            "drop",
            "drop_complete",
            "duplicate",
            "role",
            "order",
            "oversize",
            "tail",
            "bool_count",
            "bool_ordinal",
            "bool_shape",
            "shape",
            "dtype",
            "row_keys",
            "image_sha",
            "relative_path",
            "absolute_path",
            "digest",
            "witness",
            "oracle",
            "oracle_bool",
        ):
            r, b = fixture()
            batches = r["images"]
            with self.subTest(mutation=mutation):
                if mutation == "cross_batch":
                    batches[2]["rows"][0], batches[3]["rows"][0] = (
                        batches[3]["rows"][0],
                        batches[2]["rows"][0],
                    )
                elif mutation == "drop":
                    batches[3]["rows"].pop()
                    r["batch_sizes"]["gallery"][-1] = 0
                elif mutation == "drop_complete":
                    batches[2]["rows"].pop()
                    batches[2]["rows"].extend(batches[3]["rows"])
                    batches.pop()
                    r["batch_sizes"]["gallery"] = [32]
                elif mutation == "duplicate":
                    batches[3]["rows"][0] = copy.deepcopy(batches[2]["rows"][0])
                elif mutation == "role":
                    batches[2]["rows"][0]["role"] = "query"
                elif mutation == "order":
                    batches[0], batches[2] = batches[2], batches[0]
                elif mutation == "oversize":
                    batches[2]["rows"].extend(batches[3]["rows"])
                    batches.pop()
                    r["batch_sizes"]["gallery"] = [33]
                elif mutation == "tail":
                    batches[2], batches[3] = batches[3], batches[2]
                    r["batch_sizes"]["gallery"] = [1, 32]
                elif mutation == "bool_count":
                    r["batch_sizes"]["gallery"][-1] = True
                elif mutation == "bool_ordinal":
                    batches[2]["rows"][0]["panel_ordinal"] = True
                elif mutation == "bool_shape":
                    r["panel_facts"]["codes"]["shape"][0] = True
                elif mutation == "shape":
                    r["panel_facts"]["codes"]["shape"][1] = 130
                elif mutation == "dtype":
                    r["panel_facts"]["inverse_norms"]["dtype"] = "torch.float32"
                elif mutation == "row_keys":
                    del batches[2]["rows"][0]["product"]
                elif mutation == "image_sha":
                    batches[2]["rows"][0]["image_sha256"] = "invalid"
                elif mutation == "relative_path":
                    batches[2]["rows"][0]["relative_path"] = "Img/../bad.jpg"
                elif mutation == "absolute_path":
                    batches[2]["rows"][0]["path"] = "relative.jpg"
                elif mutation == "digest":
                    batches[2]["rows"][0]["product"] = "reassociated"
                elif mutation == "witness":
                    del batches[2]["rgb_sha256"]
                elif mutation == "oracle":
                    batches[3]["residual_oracle"] = None
                elif mutation == "oracle_bool":
                    batches[2]["residual_oracle"]["omitted_C_mutant_rejected"] = 1
                with self.assertRaises(ValueError):
                    self.bind(r, b)

    def test_freshly_pinned_identity_mutants(self) -> None:
        for mutation, error in (
            ("endpoint", "endpoint state"),
            ("vision", "vision/base"),
            ("base", "vision/base"),
            ("encoder", "encoder identity"),
            ("scope", "bundle scope"),
            ("file", "file/source guard"),
            ("code", "training source"),
            ("source", "source differs"),
            ("flags", "numerical flags"),
            ("bool_flag", "numerical flags"),
            ("wrong_endpoint", "exactly one"),
            ("duplicate_endpoint", "exactly one"),
            ("terminal", "terminal state"),
            ("binding", "bound bundle"),
            ("member", "exact object keys"),
            ("buffers", "encoder buffers"),
        ):
            r, b = fixture()
            with self.subTest(mutation=mutation):
                if mutation in {"endpoint", "vision", "base"}:
                    key = {
                        "endpoint": "endpoint_state_sha256",
                        "vision": "vision_sha256",
                        "base": "base_vision_sha256",
                    }[mutation]
                    b[key] = "0" * 64
                elif mutation == "encoder":
                    b["encoder_identity"]["runtime"]["modules"] = ["changed"]
                elif mutation == "scope":
                    b["scope"]["arm"] = "candidate"
                elif mutation == "file":
                    b["files"]["processor.json"] = "0" * 64
                elif mutation == "code":
                    b["code"]["train_siglip2_connected_mlp.py"] = "0" * 64
                elif mutation == "source":
                    r["source"]["warm_source"] = {}
                elif mutation == "flags":
                    r["numerical_flags"]["threads"] = 2
                elif mutation == "bool_flag":
                    r["numerical_flags"]["threads"] = True
                elif mutation == "wrong_endpoint":
                    r["binding"]["endpoints"][0]["seed"] = 179069
                elif mutation == "duplicate_endpoint":
                    r["binding"]["endpoints"].append(copy.deepcopy(r["binding"]["endpoints"][0]))
                elif mutation == "terminal":
                    r["payload_facts"]["terminal_state_sha256"] = "0" * 64
                elif mutation == "binding":
                    r["binding"]["endpoints"][0]["bundle"]["path"] = "/wrong/bundle.json"
                elif mutation == "member":
                    del r["payload_facts"]["members"]["encoder"]
                elif mutation == "buffers":
                    r["payload_facts"]["members"]["buffers"] = "0" * 64
                # Re-pin bundle references too, to reach the actual association predicate.
                digest = sha(encoded(b))
                r["payload_facts"]["bundle"]["sha256"] = digest
                for endpoint in r["binding"]["endpoints"]:
                    endpoint["bundle"]["sha256"] = digest
                r["launch"]["endpoints"] = copy.deepcopy(r["binding"]["endpoints"])
                r["input_guards"][r["payload_facts"]["bundle"]["path"]] = digest
                with self.assertRaisesRegex(ValueError, error):
                    self.bind(r, b)

    def test_endpoint_and_evaluator_guards_reject_reassociated_originals(self) -> None:
        for mutation in ("checkpoint", "launch", "receipt", "log", "source_code", "bound_endpoint"):
            r, b = fixture()
            with self.subTest(mutation=mutation):
                endpoint = r["binding"]["endpoints"][0]
                if mutation in {"checkpoint", "launch"}:
                    endpoint[mutation]["sha256"] = "0" * 64
                elif mutation in {"receipt", "log"}:
                    endpoint["terminal"][mutation]["sha256"] = "0" * 64
                elif mutation == "source_code":
                    r["source_code"]["evaluate_siglip2_connected_mlp.py"] = "0" * 64
                else:
                    r["launch"]["endpoints"][0]["terminal_state_sha256"] = "0" * 64
                if mutation != "bound_endpoint":
                    r["launch"]["endpoints"] = copy.deepcopy(r["binding"]["endpoints"])
                with self.assertRaisesRegex(ValueError, "guard|bound endpoint"):
                    self.bind(r, b)

    def test_first_last_and_interior_oracles(self) -> None:
        for count in (1, 32, 65):
            r, b = fixture(count)
            result = self.bind(r, b)
            self.assertEqual(len(result["gallery_rows"]), count)
        r, b = fixture(65)
        for index, oracle in ((0, None), (2, None), (1, ORACLE), (4, ORACLE), (5, None)):
            mutant = copy.deepcopy(r)
            mutant["images"][index]["residual_oracle"] = oracle
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "oracle"):
                self.bind(mutant, b)

    def test_strict_json_and_immutable_inputs(self) -> None:
        for raw in (
            b'{"x":1,"x":2}',
            b'{"nested":{"x":1,"x":2}}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":-Infinity}',
            b'{"x":1e999}',
            b"[]",
            b"\xff",
        ):
            for side in (0, 1):
                pair = [b"{}", b"{}"]
                pair[side] = raw
                with self.subTest(raw=raw, side=side), self.assertRaises(ValueError):
                    self.api.bind_gallery_provenance(
                        *pair,
                        trusted_receipt_sha256=sha(pair[0]),
                        trusted_bundle_sha256=sha(pair[1]),
                    )

        class BytesSubclass(bytes):
            pass

        for value in (bytearray(b"{}"), memoryview(b"{}"), BytesSubclass(b"{}"), "{}"):
            with self.subTest(value=type(value)), self.assertRaisesRegex(ValueError, "immutable"):
                self.api.bind_gallery_provenance(
                    value,
                    b"{}",
                    trusted_receipt_sha256=sha(b"{}"),
                    trusted_bundle_sha256=sha(b"{}"),
                )

    def test_acceptance_flags_are_not_authority(self) -> None:
        r, b = fixture()
        expected = self.bind(r, b)
        r.update({"pass": False, "engineering_admission_pass": False, "product_go": True})
        observed = self.bind(r, b)
        expected.pop("original_receipt_sha256")
        observed.pop("original_receipt_sha256")
        self.assertEqual(expected, observed)


if __name__ == "__main__":
    unittest.main()
