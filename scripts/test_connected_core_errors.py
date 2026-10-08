#!/usr/bin/env python3
"""Stdlib fixtures and admission tests; never score the real selection wires."""

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().with_name("census_connected_core_errors.py")


def load_script():
    spec = importlib.util.spec_from_file_location("core_census", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_mapping():
    names = ["A", "B", "C", "D"]
    targets = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]
    fit = {
        "schema": "native256-frozen-fit-manifest-v1",
        "dataset_root": "/fixture/images",
        "fit_images": 10, "fit_identities": 4,
        "class_names": names, "targets": targets,
        "rows": [{"product": names[t], "relative_path": f"img/{names[t]}/{i}.jpg",
                  "image_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                  "train_row": i} for i, t in enumerate(targets)],
    }
    partition = {
        "schema": "siglip2-identity-mix-partition-v1",
        "global_class_names": names,
        "panels": {
            "selection": {"original_rows": [0, 1, 2, 3, 4, 5],
                          "original_class_ids": [0, 1], "query": [0, 3],
                          "gallery": [1, 2, 4, 5]},
            "train": {"original_rows": [6, 7], "original_class_ids": [2]},
            "validation": {"original_rows": [8, 9], "original_class_ids": [3],
                           "query": [0], "gallery": [1]},
        },
    }
    guards = {str(Path(fit["dataset_root"]) / r["relative_path"]): r["image_sha256"]
              for r in fit["rows"]}
    return fit, partition, {"input_guards": guards}


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = load_script()

    def test_decode_signed_codes_and_little_endian_half(self):
        wire = bytes([128, 255, 0, 127] + [1] * 124) + b"\x00\x38"
        row, = self.c.decode_wire(wire, 1)
        self.assertEqual(row[0][:4], (-128, -1, 0, 127))
        self.assertEqual(row[1], 0.5)
        self.assertEqual(self.c.packed_score(row, row), 8159.5)

    def test_two_ordered_float32_multiplications(self):
        q = ((127,) * 128, 0.0004973411560058594)
        g = ((127,) * 128, 0.000514984130859375)
        self.assertEqual(self.c.packed_score(q, g), 0.5287686586380005)
        self.assertNotEqual(self.c.packed_score(q, g), 0.5287685990333557)

    def test_wire_rejects_shape_zero_codes_and_invalid_norms(self):
        for wire in [b"", b"x" * 129, b"x" * 131,
                     bytes(128) + struct.pack("<e", 1),
                     bytes([1] * 128) + struct.pack("<e", 0),
                     bytes([1] * 128) + struct.pack("<e", -1),
                     bytes([1] * 128) + struct.pack("<e", math.inf),
                     bytes([1] * 128) + struct.pack("<e", math.nan)]:
            with self.subTest(wire=wire[-2:]), self.assertRaises(ValueError):
                self.c.decode_wire(wire, 1)

    def test_stable_gallery_order_and_ap_at_r(self):
        rows = [((1,) * 128, 0.125)] * 4
        value = self.c.describe_query(rows, ["A", "B", "A", "A"], 0, [1, 3, 2], 2)
        self.assertEqual(value["top_impostor_gallery_index"], 0)
        self.assertEqual(value["top_impostor_panel_ordinal"], 1)
        self.assertEqual(value["best_positive_gallery_index"], 1)
        self.assertEqual(value["best_positive_panel_ordinal"], 3)
        self.assertEqual(value["best_positive_rank"], 2)
        self.assertEqual(value["positive_count"], 2)
        self.assertEqual(value["per_query_r1"], 0)
        self.assertEqual(value["per_query_ap"], 0.25)
        self.assertEqual(value["positive_minus_impostor_margin"], 0.0)

    def test_ap_precision_is_binary32_and_only_r_positions_count(self):
        rows = [((1,) * 128, 1), ((1,) * 128, 4), ((1,) * 128, 3),
                ((1,) * 128, 2), ((1,) * 128, 1), ((1,) * 128, 0.5)]
        value = self.c.describe_query(rows, ["A", "B", "B", "A", "A", "A"],
                                      0, [1, 2, 3, 4, 5], 3)
        self.assertEqual(value["best_positive_rank"], 3)
        self.assertEqual(value["per_query_ap"], 0.1111111119389534)
        self.assertEqual(value["positive_minus_impostor_margin"], -256.0)

    def test_replay_rejects_even_one_scalar_ulp(self):
        value = {"per_query_r1": 0, "per_query_ap": 0.25}
        expected = {"per_query_r1": [0], "per_query_ap": [0.25]}
        self.c.replay_query(value, expected, 0)
        for key, changed in [("per_query_r1", 1), ("per_query_ap", 0.2500000298023224)]:
            mutant = dict(value, **{key: changed})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.c.replay_query(mutant, expected, 0)

    def test_core_uses_all_four_endpoints_in_original_query_order(self):
        quality = {seed: {arm: {"per_query_r1": hits, "per_query_ap": [0.0] * 4}
                         for arm, hits in arms.items()}
                   for seed, arms in {
                       "179061": {"control": [0, 0, 0, 1], "candidate": [0, 1, 0, 1]},
                       "179069": {"control": [0, 0, 0, 1], "candidate": [0, 0, 0, 1]},
                   }.items()}
        self.assertEqual(self.c.core_queries(quality, 4), [0, 2])
        for mutation in ["missing", "short", "boolean", "nonfinite", "extra"]:
            mutant = copy.deepcopy(quality)
            if mutation == "missing": del mutant["179069"]["candidate"]
            if mutation == "short": mutant["179061"]["control"]["per_query_ap"].pop()
            if mutation == "boolean": mutant["179061"]["control"]["per_query_r1"][0] = False
            if mutation == "nonfinite": mutant["179061"]["control"]["per_query_ap"][0] = math.nan
            if mutation == "extra": mutant["179070"] = mutant["179069"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.c.core_queries(mutant, 4)


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c = load_script()

    def test_exact_complete_fit_and_role_mapping(self):
        fit, partition, receipt = fixture_mapping()
        rows = self.c.map_rows(fit, partition, receipt)
        self.assertEqual(rows[0], {"panel_ordinal": 0, "original_row": 0, "role": "query",
                                  "path": "/fixture/images/img/A/0.jpg",
                                  "guarded_path": "/fixture/images/img/A/0.jpg",
                                  "relative_path": "img/A/0.jpg", "product": "A",
                                  "image_sha256": fit["rows"][0]["image_sha256"],
                                  "train_row": 0, "target": 0})
        self.assertEqual([r["role"] for r in rows], ["query", "gallery", "gallery",
                                                    "query", "gallery", "gallery"])

    def test_manifest_spelling_and_unique_guarded_image_hash_join(self):
        fit, partition, receipt = fixture_mapping()
        root = "/home/riomus/datasets/inshop_official_standard"
        receipt["input_guards"] = {p.replace("/fixture/images", root): v
                                   for p, v in receipt["input_guards"].items()}
        fit["dataset_root"] = root
        fit["rows"][0]["relative_path"] = "Img/A/0.jpg"
        rows = self.c.map_rows(fit, partition, receipt)
        self.assertEqual(rows[0]["path"], root + "/Img/A/0.jpg")
        self.assertEqual(rows[0]["guarded_path"], root + "/img/A/0.jpg")
        receipt["input_guards"][root + "/alias/A/0.jpg"] = fit["rows"][0]["image_sha256"]
        with self.assertRaises(ValueError): self.c.map_rows(fit, partition, receipt)

    def test_unproven_alias_rejected(self):
        fit, partition, receipt = fixture_mapping()
        fit["rows"][0]["relative_path"] = "Alias/A/0.jpg"
        with self.assertRaises(ValueError): self.c.map_rows(fit, partition, receipt)

    def test_rejects_mapping_order_role_inventory_labels_and_paths(self):
        for mutation in ["duplicate", "reverse", "overlap", "missing_role", "negative",
                         "class", "target", "product", "escape", "hash", "train_row",
                         "partition_classes", "missing_fit", "cross_panel", "no_positive"]:
            fit, partition, receipt = fixture_mapping()
            panel = partition["panels"]["selection"]
            if mutation == "duplicate": panel["original_rows"][1] = 0
            if mutation == "reverse": panel["original_rows"].reverse()
            if mutation == "overlap": panel["gallery"][0] = 0
            if mutation == "missing_role": panel["gallery"].pop()
            if mutation == "negative": panel["query"][0] = -1
            if mutation == "class": panel["original_class_ids"] = [0]
            if mutation == "target": fit["targets"][0] = 100
            if mutation == "product": fit["rows"][0]["product"] = "B"
            if mutation == "escape": fit["rows"][0]["relative_path"] = "../outside.jpg"
            if mutation == "hash": receipt["input_guards"]["/fixture/images/img/A/0.jpg"] = "0" * 64
            if mutation == "train_row": fit["rows"][0]["train_row"] = 1
            if mutation == "partition_classes": partition["global_class_names"].reverse()
            if mutation == "missing_fit": fit["rows"].pop()
            if mutation == "cross_panel": partition["panels"]["train"]["original_rows"][0] = 0
            if mutation == "no_positive": panel["query"] = [0, 1, 2, 3]; panel["gallery"] = [4, 5]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.c.map_rows(fit, partition, receipt)


class FileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c = load_script()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.path = self.root / "input.json"
        self.path.write_bytes(b'{"value":1}\n')
        self.pin = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_hash_admission_and_fresh_exit_reject_mutated_bytes(self):
        guards = {}
        self.assertEqual(self.c.read_bound(str(self.path), self.pin, guards), b'{"value":1}\n')
        self.c.rehash(guards)
        self.path.write_bytes(b'{"value":2}\n')
        with self.assertRaises(ValueError): self.c.rehash(guards)
        with self.assertRaises(ValueError): self.c.read_bound(str(self.path), self.pin, {})

    def test_exit_rejects_input_replaced_by_symlink_with_identical_bytes(self):
        guards = {}; self.c.read_bound(str(self.path), self.pin, guards)
        copy_path = self.root / "copy.json"; copy_path.write_bytes(self.path.read_bytes())
        self.path.unlink(); self.path.symlink_to(copy_path)
        output = self.root / "output.json"
        with self.assertRaises(ValueError): self.c.publish(str(output), {}, guards)
        self.assertFalse(output.exists())

    def test_input_changes_during_output_write_are_rejected(self):
        guards = {}; self.c.read_bound(str(self.path), self.pin, guards)
        output = self.root / "output.json"; real_fsync = self.c.os.fsync
        def mutate_after_write(fd):
            real_fsync(fd)
            self.path.write_bytes(b'changed during output serialization')
        with patch.object(self.c.os, "fsync", side_effect=mutate_after_write):
            with self.assertRaises(ValueError): self.c.publish(str(output), {}, guards)
        self.assertFalse(output.exists())
        self.assertEqual([p.name for p in self.root.iterdir()], ["input.json"])

    def test_destination_created_during_write_is_never_overwritten(self):
        output = self.root / "output.json"; real_fsync = self.c.os.fsync
        def create_after_write(fd):
            real_fsync(fd)
            output.write_bytes(b'competing original output')
        with patch.object(self.c.os, "fsync", side_effect=create_after_write):
            with self.assertRaises(FileExistsError): self.c.publish(str(output), {}, {})
        self.assertEqual(output.read_bytes(), b'competing original output')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["input.json", "output.json"])

    def test_rejects_symlink_ancestors_relative_noncanonical_and_nonregular(self):
        link = self.root / "link"; link.symlink_to(self.path)
        folder = self.root / "folder"; folder.symlink_to(self.root, target_is_directory=True)
        fifo = self.root / "fifo"; os.mkfifo(fifo)
        for path in [str(link), str(folder / self.path.name), "input.json",
                     str(self.root) + "/./input.json", str(self.root), str(fifo)]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.c.read_bound(path, self.pin, {})

    def test_exclusive_publication_and_exit_rehash(self):
        guards = {}; self.c.read_bound(str(self.path), self.pin, guards)
        output = self.root / "output.json"
        self.c.publish(str(output), {"accepted": True}, guards)
        self.assertEqual(json.loads(output.read_text()), {"accepted": True})
        with self.assertRaises(FileExistsError): self.c.publish(str(output), {}, guards)
        self.assertEqual(json.loads(output.read_text()), {"accepted": True})
        output.unlink(); output.symlink_to(self.path)
        with self.assertRaises(FileExistsError): self.c.publish(str(output), {}, guards)
        output.unlink(); self.path.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.c.publish(str(output), {}, guards)
        self.assertFalse(output.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["input.json"])

    def test_write_failure_cleans_temp_and_never_publishes(self):
        output = self.root / "output.json"
        with patch.object(self.c.os, "fsync", side_effect=OSError("fixture write failure")):
            with self.assertRaises(OSError): self.c.publish(str(output), {}, {})
        self.assertFalse(output.exists())
        self.assertEqual([p.name for p in self.root.iterdir()], ["input.json"])

    def test_input_json_rejects_duplicate_keys_and_nonfinite(self):
        for value in [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}']:
            with self.subTest(value=value), self.assertRaises(ValueError): self.c.json_value(value)

    def test_cli_bad_hash_exits_nonzero_without_output(self):
        output = self.root / "never.json"
        run = subprocess.run([sys.executable, str(SCRIPT), "--inputs", str(self.path),
                              "--inputs-sha256", "0" * 64, "--output", str(output)],
                             capture_output=True, text=True, timeout=5)
        self.assertEqual(run.returncode, 1)
        self.assertIn("sha256", run.stderr)
        self.assertFalse(output.exists())


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c = load_script()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        fit, partition, receipt = fixture_mapping()
        quality = {seed: {arm: {"per_query_r1": [0, 1], "per_query_ap": [0.25, 1.0]}
                          for arm in ["control", "candidate"]}
                   for seed in ["179061", "179069"]}
        receipt.update({"panel": "selection", "query_images": 2, "gallery_images": 4,
                        "products": 2, "decision": "KILL", "quality_pass": False,
                        "product_go": False, "selection_previously_exposed": True,
                        "persisted_wire_scoring_replay_exact": True, "exit_rehash_pass": True,
                        "quality": quality})
        receipt["binding"] = {"endpoints": [{"seed": int(s), "arm": a}
                                            for s in quality for a in quality[s]],
                              "reference": {"root": "/fixture/scorer-reference"}}
        files = {}
        for seed in quality:
            for arm in quality[seed]:
                name = f"{arm}-{seed}.packed.bin"
                wire = b"".join(struct.pack("<128be", *([code] * 128), 0.125)
                                for code in [1, 1, 1, 2, 2, -1])
                path = self.root / name; path.write_bytes(wire)
                original = f"/home/riomus/runs/sfora-connected-mlp-evaluation-full-export-{arm}-{seed}-v2/{name}"
                pin = hashlib.sha256(path.read_bytes()).hexdigest()
                receipt["input_guards"][original] = pin
                files[name] = {"path": str(path), "original_path": original,
                               "sha256": pin, "bytes": 780}
        fp = self.root / "fit.json"; fp.write_text(json.dumps(fit))
        fitpin = hashlib.sha256(fp.read_bytes()).hexdigest()
        files["fit.json"] = {"path": str(fp), "original_path": self.c.ORIGINAL_FIT,
                             "sha256": fitpin, "bytes": fp.stat().st_size}
        receipt["input_guards"][self.c.ORIGINAL_FIT] = fitpin
        pp = self.root / "partition.json"
        partition["original_fit"] = {"path": self.c.ORIGINAL_FIT, "sha256": fitpin}
        pp.write_text(json.dumps(partition)); pp_pin = hashlib.sha256(pp.read_bytes()).hexdigest()
        receipt["input_guards"][self.c.ORIGINAL_PARTITION] = pp_pin
        for name, pin in self.c.SOURCES.items():
            receipt["input_guards"][self.c.SOURCE_ORIGINAL[name]] = pin
        rp = self.root / "receipt.json"; rp.write_text(json.dumps(receipt))
        self.rp = rp; self.rpin = hashlib.sha256(rp.read_bytes()).hexdigest()
        self.record = {"schema": "sfora-existing-core-census-inputs-v1",
                       "accepted_receipt": {"path": str(rp), "sha256": self.rpin},
                       "partition": {"path": str(pp), "original_path": self.c.ORIGINAL_PARTITION,
                                     "sha256": pp_pin}, "files": files,
                       "images_read": 0, "model_executions": 0,
                       "scientific_gate_changed": False, "original_fetch_exit": 1,
                       "original_fetch_session": 13199, "transfer_and_verify_seconds": None,
                       "metadata_correction": "fixture mirrors corrected parent fetch"}
        self.ip = self.root / "inputs.json"

    def admit(self):
        self.ip.write_text(json.dumps(self.record))
        pin = hashlib.sha256(self.ip.read_bytes()).hexdigest()
        with patch.object(self.c, "ORIGINAL_RECEIPT_SHA", self.rpin), \
             patch.object(self.c, "POPULATION", (6, 2, 4, 2)), \
             patch.object(self.c, "CORE_COUNT", 1):
            return self.c.admit_inputs(str(self.ip), pin)

    def test_complete_admission_and_fixture_only_census(self):
        state = self.admit()
        self.assertEqual(len(state["guards"]), 11)
        self.assertEqual(state["core"], [0])
        value = self.c.build_census(state)
        self.assertEqual(value["core_query_indices"], [0])
        self.assertEqual(value["per_product_core_counts"], [{"product": "A", "core_queries": 1,
                                                          "panel_queries": 1, "panel_gallery": 2}])
        row, = value["queries"]
        self.assertEqual(row["query"]["path"], "/fixture/images/img/A/0.jpg")
        self.assertEqual(row["endpoints"]["control-179061"]["top_impostor"]["product"], "B")

    def test_rejects_fetch_schema_roles_hashes_shape_and_unbound_original(self):
        for mutation in ["schema", "extra", "missing_wire", "extra_wire", "wrong_original",
                         "wrong_hash", "wrong_bytes", "receipt_pin", "partition_pin",
                         "images", "models", "gate"]:
            original = copy.deepcopy(self.record)
            if mutation == "schema": self.record["schema"] = "other"
            if mutation == "extra": self.record["fallback"] = True
            if mutation == "missing_wire": del self.record["files"]["candidate-179069.packed.bin"]
            if mutation == "extra_wire": self.record["files"]["concat.packed.bin"] = {}
            if mutation == "wrong_original": self.record["files"]["control-179061.packed.bin"]["original_path"] = "/unbound"
            if mutation == "wrong_hash": self.record["files"]["control-179061.packed.bin"]["sha256"] = "0" * 64
            if mutation == "wrong_bytes": self.record["files"]["control-179061.packed.bin"]["bytes"] = 779
            if mutation == "receipt_pin": self.record["accepted_receipt"]["sha256"] = "0" * 64
            if mutation == "partition_pin": self.record["partition"]["sha256"] = "0" * 64
            if mutation == "images": self.record["images_read"] = 1
            if mutation == "models": self.record["model_executions"] = 1
            if mutation == "gate": self.record["scientific_gate_changed"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.admit()
            self.record = original

    def test_fixture_replay_failure_never_publishes(self):
        state = self.admit()
        state["receipt"]["quality"]["179061"]["control"]["per_query_ap"][0] = 0.2500000298023224
        output = self.root / "output.json"
        with self.assertRaises(ValueError):
            self.c.publish(str(output), self.c.build_census(state), state["guards"])
        self.assertFalse(output.exists())

    def test_changed_core_selection_is_rejected_before_scoring(self):
        state = self.admit()
        state["core"].append(1)
        with patch.object(self.c, "describe_query", side_effect=AssertionError("must not score extra query")):
            with self.assertRaises(ValueError): self.c.build_census(state)

    def test_current_complete_real_input_provenance_without_scoring(self):
        path = Path("/tmp/sfora-connected-core-census-inputs-v1/fetch-receipt.json")
        if not path.exists(): self.skipTest("parent staged inputs absent; no new fetch")
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        state = self.c.admit_inputs(str(path), pin)
        self.assertEqual(len(state["guards"]), 11)
        self.assertEqual(len(state["rows"]), 3449)
        self.assertEqual(len(state["core"]), 44)
        self.assertEqual(set(state["wires"]), {"control-179061", "candidate-179061",
                                               "control-179069", "candidate-179069"})
        self.c.rehash(state["guards"])


if __name__ == "__main__":
    unittest.main()
