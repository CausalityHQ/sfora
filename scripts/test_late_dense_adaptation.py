#!/usr/bin/env python3
"""Stdlib-only authority/counter/file negatives; never import the native driver."""
if not __debug__:
    raise SystemExit("Checks require assertions")

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import late_dense_boundary as late

PATH = Path(__file__).with_name("train_late_dense_adaptation.py")
TREE = ast.parse(PATH.read_text(), filename=str(PATH))
FUNCTIONS = {"counters", "diagnostic", "validate_cpu", "validate_mechanics", "validate_resume",
             "atomic_write", "read_authority"}
NAMESPACE = {"late": late, "os": os, "json": json, "METHOD": "late-dense-adaptation-v1",
             "pair": SimpleNamespace(sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest())}
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS],
                        type_ignores=[]), str(PATH), "exec"), NAMESPACE)


def cpu_receipt():
    receipt = {"pass": True, "execution_sha256": "cpu112", "code": {"original.py": "original"},
        "source_checkpoint_sha256": late.SOURCE_SHA, "optimizer_members": {"12": 208, "10": 240},
        "optimizer_updates": 0, "quality_read": False}
    receipt.update({k: True for k in ("matched_control_candidate_initial_state", "fit_held_identity_and_path_disjoint",
        "native400_whole_head_packed_reload_exact", "changed_driver_rejected", "RNG_preserved",
        "missing_duplicate_optimizer_and_frozen_buffer_mutation_rejected")})
    return receipt


def mechanics_receipt():
    receipt = {"pass": True, "intervention": NAMESPACE["METHOD"], "execution_sha256": "gpu114",
        "cpu_authority_sha256": "cpu", "cpu_log_sha256": "cpu-log", "cpu_execution_sha256": "cpu112",
        "phase": "mechanics", "boundary": 10, "seed": 179032, "source_checkpoint_sha256": late.SOURCE_SHA,
        "updates": 17, "completed_step": 17, "quality_read": False, "checkpoint_sha256": None,
        "chunk100_admission_seconds": 269, "total_seconds": 119, "peak_cuda_allocated_bytes": 9_000_000_000,
        "steps": [{"step": i, "optimizer_counter": i, "augmentation_step": 1000 + i} for i in range(1, 18)]}
    receipt.update({k: True for k in ("training_state_discarded", "native_17_equals_serialized8_plus9_exact",
        "strict400_reload_whole_head_packed_exact", "new32_weights_changed", "frozen_named_state_buffers_rng_preserved")})
    return receipt


class BoundedDriver(unittest.TestCase):
    def test_original_cpu_append_only_authority(self):
        receipt = cpu_receipt()
        code = {**receipt["code"], "train_late_dense_adaptation.py": "driver", "test_late_dense_adaptation.py": "check"}
        validate = NAMESPACE["validate_cpu"]
        validate(receipt, "cpu112", code)  # GPU114 differs intentionally from CPU112.
        for name, value in (("execution_sha256", "gpu114"), ("source_checkpoint_sha256", "F5"),
                            ("optimizer_updates", 1), ("quality_read", True), ("RNG_preserved", False),
                            ("optimizer_members", {"12": 208, "10": 208})):
            bad = copy.deepcopy(receipt); bad[name] = value
            with self.subTest(name=name), self.assertRaises(AssertionError):
                validate(bad, "cpu112", code)
        for change in ("original", "extra", "missing", "removed"):
            bad = code.copy()
            if change == "original":
                bad["original.py"] = "changed"
            elif change == "extra":
                bad["unrequested.py"] = "extra"
            else:
                del bad["test_late_dense_adaptation.py" if change == "missing" else "original.py"]
            with self.subTest(change=change), self.assertRaises((AssertionError, KeyError)):
                validate(receipt, "cpu112", bad)

    def test_mechanics_authority_and_counter_offset(self):
        validate = lambda r: NAMESPACE["validate_mechanics"](r, "gpu114", "cpu", "cpu-log", "cpu112")
        receipt = mechanics_receipt(); validate(receipt)
        for name, value in (("seed", 179041), ("boundary", 12), ("phase", "train"),
                            ("execution_sha256", "old"), ("cpu_authority_sha256", "other"),
                            ("cpu_execution_sha256", "gpu114"), ("cpu_log_sha256", "other"),
                            ("source_checkpoint_sha256", "F5"), ("updates", 100),
                            ("checkpoint_sha256", "retained"), ("training_state_discarded", False),
                            ("chunk100_admission_seconds", 270), ("total_seconds", 120),
                            ("peak_cuda_allocated_bytes", 10_000_000_000), ("quality_read", True)):
            bad = copy.deepcopy(receipt); bad[name] = value
            with self.subTest(name=name), self.assertRaises(AssertionError):
                validate(bad)
        for name in ("step", "optimizer_counter", "augmentation_step"):
            bad = copy.deepcopy(receipt); bad["steps"][8][name] = 8
            with self.subTest(name=name), self.assertRaises(AssertionError):
                validate(bad)
        for i in range(1, 101):
            self.assertEqual(NAMESPACE["counters"](i), (i, 1000 + i))
        for value in (0, 101, -1, True, 1.0):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                NAMESPACE["counters"](value)

    def test_complete_resume_and_foreign_identity_rejected(self):
        base = {"schema": "late-dense-complete-resume-v1", "parameter_names": [str(i) for i in range(240)],
                "boundary": 10, "seed": 179032, "initial_state_sha256": "initial", "execution_sha256": "gpu114"}
        groups = [{"params": list(range(237)), "lr": 1e-5}, {"params": [237, 238], "lr": 1e-4},
                  {"params": [239], "lr": 1e-4}]
        saved = {k: {} for k in late.RESUME_KEYS}
        saved.update(identity={**base, "global_step": 8}, vision={str(i): None for i in range(400)},
                     optimizer={"param_groups": groups, "state": {i: {"step": 8, "exp_avg": None, "exp_avg_sq": None}
                                                                  for i in range(240)}},
                     scaler={"scale": 128}, cuda_rng=["state"])
        validate = lambda r: NAMESPACE["validate_resume"](r, base, 8, groups)
        validate(saved)
        for key in late.RESUME_KEYS:
            bad = copy.deepcopy(saved); del bad[key]
            with self.subTest(missing=key), self.assertRaises(AssertionError):
                validate(bad)
        for key in ("global_step", "boundary", "seed", "initial_state_sha256", "execution_sha256"):
            bad = copy.deepcopy(saved); bad["identity"][key] = "foreign"
            with self.subTest(identity=key), self.assertRaises(AssertionError):
                validate(bad)
        for case in ("missing_member", "wrong_counter", "fractional_counter", "group_order", "missing_moment", "no_scaler", "no_rng"):
            bad = copy.deepcopy(saved)
            if case == "missing_member": del bad["optimizer"]["state"][0]
            elif case == "wrong_counter": bad["optimizer"]["state"][0]["step"] = 1008
            elif case == "fractional_counter": bad["optimizer"]["state"][0]["step"] = 8.5
            elif case == "group_order": bad["optimizer"]["param_groups"][0]["params"].reverse()
            elif case == "missing_moment": del bad["optimizer"]["state"][0]["exp_avg"]
            elif case == "no_scaler": bad["scaler"] = None
            else: bad["cuda_rng"] = []
            with self.subTest(case=case), self.assertRaises(AssertionError): validate(bad)

    def test_real_file_hash_log_and_exclusive_atomic_publication(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary); path = root / "proof.json"
            write = NAMESPACE["atomic_write"]
            write(path, lambda s: s.write(b'{"pass": true}'))
            sha = NAMESPACE["pair"].sha(path)
            self.assertEqual(NAMESPACE["read_authority"](path, sha), {"pass": True})
            with self.assertRaises(AssertionError): NAMESPACE["read_authority"](path, "fake")
            with self.assertRaises(AssertionError): write(path, lambda s: s.write(b"overwritten"))
            self.assertEqual(NAMESPACE["pair"].sha(path), sha)
            log = root / "cpu.log"
            good = "Finished with result: success\ncode=exited/status=0\nMemory swap peak: 0B\n"
            log.write_text(good)
            NAMESPACE["read_authority"](log, NAMESPACE["pair"].sha(log), log=True)
            for text in (good.replace("status=0", "status=1"), good.replace("0B", "1B")):
                log.write_text(text)
                with self.assertRaises(AssertionError):
                    NAMESPACE["read_authority"](log, NAMESPACE["pair"].sha(log), log=True)
            race = root / "race"
            def concurrent_writer(s):
                s.write(b"new"); race.write_bytes(b"winner")
            with self.assertRaises(FileExistsError): write(race, concurrent_writer)
            self.assertEqual(race.read_bytes(), b"winner")
            self.assertFalse(race.with_name("race.part").exists())

    def test_native_ast_contract(self):
        calls = {ast.unparse(n.func) for n in ast.walk(TREE) if isinstance(n, ast.Call)}
        self.assertIn("old.payload", calls)
        self.assertFalse({"old.save", "old.restore", "old.identity", "old.coverage.frozen_digest"} & calls)
        keys = {n.value for n in ast.walk(next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "identity"))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertTrue({"initial_state_sha256", "source_checkpoint_sha256", "seed", "boundary", "runtime",
                         "buffers_sha256", "schedule_sha256", "parameter_names", "optimizer_groups",
                         "frozen_names", "frozen_sha256", "cpu_authority_sha256", "execution_sha256"} <= keys)


if __name__ == "__main__":
    unittest.main()
