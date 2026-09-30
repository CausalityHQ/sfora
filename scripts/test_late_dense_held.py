#!/usr/bin/env python3
"""Stdlib negative authority/statistical checks; never import native code."""
if not __debug__:
    raise SystemExit("Checks require assertions")

import ast
import copy
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
import unittest

import late_dense_boundary as late

ROOT = Path(__file__).parent
EXPORT = ast.parse((ROOT / "export_late_dense_adaptation.py").read_text())
SCORE = ast.parse((ROOT / "score_late_dense_adaptation.py").read_text())
NAMESPACE = {"math": math, "statistics": statistics, "Path": Path, "late": late,
             "training": SimpleNamespace(METHOD="late-dense-adaptation-v1")}
for node in EXPORT.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
        "TRAIN_CODE", "CPU_CODE", "CPU_SHA", "MECHANICS_SHA", "MECHANICS_LOG_SHA", "ADDED", "SEEDS", "ORDER", "PRECISION"}:
        NAMESPACE[node.targets[0].id] = ast.literal_eval(node.value)
NAMESPACE["export"] = SimpleNamespace(SEEDS=NAMESPACE["SEEDS"], PRECISION=NAMESPACE["PRECISION"])
NAMESPACE["METRICS"] = ("per_query_r1", "per_query_ap")
for tree, names in ((EXPORT, {"validate_closure", "validate_unit", "validate_endpoint", "paired_cost", "frozen_saved"}),
                    (SCORE, {"averaged_deltas", "quality_gate", "validate_wire"})):
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]),
                 "stdlib-authority-check", "exec"), NAMESPACE)


def example(seed=179032, arm="candidate"):
    boundary = 10 if arm == "candidate" else 12
    spec = {"seed": seed, "arm": arm, "checkpoint_sha256": "fresh", "terminal_state_fingerprint": "complete",
            "unit": f"late-{seed}-{arm}", "invocation_id": "original", "service_seconds": 200,
            "host_max_rss_kib": 7 * 1024 * 1024, "host_swap_kib": 0}
    authority = {"cpu": {"receipt_sha256": NAMESPACE["CPU_SHA"], "log_sha256": "cpu-log"},
                 "mechanics": {"receipt_sha256": NAMESPACE["MECHANICS_SHA"], "log_sha256": NAMESPACE["MECHANICS_LOG_SHA"]}}
    base = {"schema": "late-dense-complete-resume-v1", "intervention": NAMESPACE["training"].METHOD,
        "arm": "half", "width": 128, "total_updates": 100, "seed": seed, "boundary": boundary,
        "augmentation_offset": 1000, "execution_sha256": NAMESPACE["TRAIN_CODE"],
        "source_checkpoint_sha256": late.SOURCE_SHA, "source_execution_sha256": late.SOURCE_CODE,
        "source_schedule_sha256": late.SOURCE_SCHEDULE, "cpu_authority_sha256": NAMESPACE["CPU_SHA"],
        "cpu_execution_sha256": NAMESPACE["CPU_CODE"], "cpu_log_sha256": "cpu-log", "initial_state_sha256": "initial",
        "schedule_sha256": f"schedule-{seed}", "precision": "native_float32_fp16_autocast", "batch": 64, "micro": 16,
        "parameter_names": [str(i) for i in range(240 if boundary == 10 else 208)],
        "target_positive_sha256": "order", "buffers_sha256": "buffers", "runtime": "runtime"}
    value = {"pass": True, "phase": "train", "updates": 100, "completed_step": 100, "quality_read": False,
        "training_state_discarded": False, "frozen_named_state_buffers_rng_preserved": True,
        "mechanics_sha256": NAMESPACE["MECHANICS_SHA"], "mechanics_log_sha256": NAMESPACE["MECHANICS_LOG_SHA"],
        "new32_weights_changed": boundary == 10, "new_parameter_changes": {str(i): boundary == 10 for i in range(32)},
        "resume_identity": base, "checkpoint_sha256": "fresh", "terminal_state_fingerprint": "complete",
        "steps": [{"step": i, "optimizer_counter": i, "augmentation_step": 1000 + i, "rgb_sha256": f"rgb-{i}",
                   "pixels_sha256": f"pixels-{i}", "rank_active_before": True, "seconds": 1.0} for i in range(1, 101)],
        "training_wall_seconds": 100., "median_step_3_end_seconds": 1., "unit_invocation_id": "original",
        "unit_cgroup": f"/system.slice/{spec['unit']}.service", "unit_memory_max_bytes": 8 * 1024**3,
        "unit_memory_swap_max_bytes": 0, "host_max_rss_kib": spec["host_max_rss_kib"], "host_swap_kib": 0,
        "total_seconds": 190., "peak_cuda_allocated_bytes": 7_000_000_000}
    value.update({k: base[k] for k in ("intervention", "seed", "boundary", "source_checkpoint_sha256", "execution_sha256",
                 "cpu_authority_sha256", "cpu_execution_sha256", "cpu_log_sha256", "initial_state_sha256", "schedule_sha256")})
    return value, spec, {"initial_state_sha256": "initial"}, authority


class HeldGate(unittest.TestCase):
    def test_exact_append_only_closure(self):
        previous = {str(i): str(i) for i in range(114)}
        code = {**previous, **{n: "new" for n in NAMESPACE["ADDED"]}}
        NAMESPACE["validate_closure"](code, previous)
        for mutation in ("prefix", "extra", "missing"):
            bad = code.copy()
            if mutation == "prefix": bad["0"] = "changed"
            elif mutation == "extra": bad["other.py"] = "new"
            else: del bad["export_late_dense_adaptation.py"]
            with self.subTest(mutation=mutation), self.assertRaises((AssertionError, KeyError)):
                NAMESPACE["validate_closure"](bad, previous)

    def test_fresh_terminal_resource_and_identity_negatives(self):
        value, spec, cpu, authority = example()
        validate = lambda v: NAMESPACE["validate_endpoint"](v, spec, cpu, authority)
        validate(value)
        for key, changed in (("phase", "mechanics"), ("updates", 17), ("completed_step", 99),
            ("seed", 179041), ("boundary", 12), ("checkpoint_sha256", "archived"),
            ("terminal_state_fingerprint", "partial"), ("source_checkpoint_sha256", "F5"),
            ("execution_sha256", "old114"), ("cpu_authority_sha256", "oldCPU"),
            ("mechanics_sha256", "oldGPU"), ("new32_weights_changed", False),
            ("quality_read", True), ("host_max_rss_kib", 9 * 1024 * 1024), ("host_swap_kib", 1),
            ("total_seconds", 300), ("peak_cuda_allocated_bytes", 10_000_000_000), ("training_wall_seconds", float("nan"))):
            bad = copy.deepcopy(value); bad[key] = changed
            with self.subTest(key=key), self.assertRaises(AssertionError): validate(bad)
        for key in ("schema", "source_schedule_sha256", "precision", "augmentation_offset", "batch", "boundary", "seed"):
            bad = copy.deepcopy(value); bad["resume_identity"][key] = "foreign"
            with self.subTest(identity=key), self.assertRaises(AssertionError): validate(bad)
        bad = copy.deepcopy(value); bad["steps"][9]["augmentation_step"] = 10
        with self.assertRaises(AssertionError): validate(bad)
        for key, changed in (("service_seconds", 300), ("host_swap_kib", 1), ("invocation_id", "foreign")):
            bad = spec.copy(); bad[key] = changed
            with self.subTest(unit=key), self.assertRaises(AssertionError):
                NAMESPACE["validate_endpoint"](value, bad, cpu, authority)

    def test_same_pair_and_fresh_cost_denominator(self):
        endpoints = {(seed, arm): example(seed, arm)[0] for seed, arm in NAMESPACE["ORDER"]}
        self.assertEqual(NAMESPACE["paired_cost"](endpoints)[179032]["training_wall_ratio"], 1)
        for key in ("rgb_sha256", "pixels_sha256", "rank_active_before"):
            bad = copy.deepcopy(endpoints); bad[179032, "candidate"]["steps"][49][key] = "changed"
            with self.subTest(key=key), self.assertRaises(AssertionError): NAMESPACE["paired_cost"](bad)
        for key, changed in (("training_wall_seconds", 150.01), ("median_step_3_end_seconds", 1.5001), ("schedule_sha256", "foreign")):
            bad = copy.deepcopy(endpoints); bad[179041, "candidate"][key] = changed
            with self.subTest(cost=key), self.assertRaises(AssertionError): NAMESPACE["paired_cost"](bad)
        bad = endpoints.copy(); del bad[179032, "control"]
        with self.assertRaises(AssertionError): NAMESPACE["paired_cost"](bad)
        saved = {"vision": {"embeddings.weight": 1, "encoder.layers.10.weight": 2, "encoder.layers.12.weight": 3},
                 "buffers": {"embeddings.position_ids": 4}}
        self.assertEqual(set(NAMESPACE["frozen_saved"](saved, 10)), {"embeddings.weight", "embeddings.position_ids"})
        self.assertIn("encoder.layers.10.weight", NAMESPACE["frozen_saved"](saved, 12))

    def test_equal_seed_average_both_metric_bounds_and_seed_floors(self):
        quality = {s: {a: {m: [0.5 + (0.004 if a == "candidate" else 0)] * 6354 for m in NAMESPACE["METRICS"]}
                       for a in ("control", "candidate")} for s in NAMESPACE["SEEDS"]}
        deltas, average = NAMESPACE["averaged_deltas"](quality)
        intervals = {m: {"mean_delta": statistics.mean(average[m]), "product_lower95": .001, "product_upper95": .008,
                         "query_lower95": .001, "query_upper95": .008} for m in NAMESPACE["METRICS"]}
        self.assertEqual(NAMESPACE["quality_gate"](deltas, intervals), (True, True))
        low = copy.deepcopy(deltas)
        for seed in NAMESPACE["SEEDS"]:
            low[seed]["per_query_r1"] = [.001] * 6354
        low_intervals = copy.deepcopy(intervals); low_intervals["per_query_r1"]["mean_delta"] = .001
        self.assertEqual(NAMESPACE["quality_gate"](low, low_intervals), (True, False))
        zero_ap = copy.deepcopy(deltas)
        zero_ap[179041]["per_query_ap"] = [0.] * 6354
        zero_ap[179032]["per_query_ap"] = [.008] * 6354
        self.assertEqual(NAMESPACE["quality_gate"](zero_ap, intervals), (True, True))
        for metric in NAMESPACE["METRICS"]:
            bad = copy.deepcopy(intervals); bad[metric]["product_lower95"] = 0
            self.assertEqual(NAMESPACE["quality_gate"](deltas, bad), (True, False))
        quality[179041]["candidate"]["per_query_r1"] = [0.5] * 6354
        deltas, average = NAMESPACE["averaged_deltas"](quality)
        intervals["per_query_r1"]["mean_delta"] = statistics.mean(average["per_query_r1"])
        self.assertEqual(NAMESPACE["quality_gate"](deltas, intervals), (False, False))
        self.assertAlmostEqual(average["per_query_r1"][0], .002)
        quality[179041]["candidate"]["per_query_ap"] = [0.499] * 6354
        deltas, average = NAMESPACE["averaged_deltas"](quality)
        intervals["per_query_ap"]["mean_delta"] = statistics.mean(average["per_query_ap"])
        self.assertEqual(NAMESPACE["quality_gate"](deltas, intervals), (False, False))
        quality[179032]["candidate"]["per_query_ap"][0] = float("nan")
        with self.assertRaises(AssertionError): NAMESPACE["averaged_deltas"](quality)

    def test_wire_precision_complete_source_binding(self):
        training, endpoint, _, _ = example()
        endpoint["receipt_sha256"] = "original-training"
        spec, code = {"execution_sha256": "held117"}, {"export.py": "frozen"}
        frozen = {"held_manifest": ["synthetic"], "query": [0], "gallery": [1]}
        receipt = {**frozen, "pass": True, "authority_sha256": "parent-frozen", "execution_sha256": "held117",
            "source_code": code, "seed": 179032, "arm": "candidate", "training_receipt_sha256": "original-training",
            "checkpoint_sha256": "fresh", "terminal_state_fingerprint": "complete", "boundary": 10,
            "batch": 32, "width": 128, "precision": NAMESPACE["PRECISION"], "optimizer_updates": 0,
            "quality_read": False, "official_read": False, "claim_eligible": False,
            "public_serving_qualified": False, "public_latency_measured": False,
            "full_held_independent_whole_head_packed_exact": True, "source_head_rng_flags_preserved": True,
            "files": {n: "digest" for n in ("held.npy", "held.codes.npy", "held.inverse.npy")}}
        validate = lambda r: NAMESPACE["validate_wire"](r, endpoint, training, spec, code, "parent-frozen", frozen)
        validate(receipt)
        for key, changed in (("checkpoint_sha256", "archived"), ("arm", "control"), ("precision", "fp32_autocast"),
                             ("batch", 16), ("width", 256), ("authority_sha256", "unreviewed"),
                             ("full_held_independent_whole_head_packed_exact", False), ("query", [1]), ("quality_read", True)):
            bad = copy.deepcopy(receipt); bad[key] = changed
            with self.subTest(wire=key), self.assertRaises(AssertionError): validate(bad)

    def test_native_reload_precision_and_shared_draw_contract(self):
        calls = {ast.unparse(n.func) for n in ast.walk(EXPORT) if isinstance(n, ast.Call)}
        self.assertTrue({"qualification.fresh", "training.identity", "training.verify", "training.validate_resume",
                         "old.fingerprint", "model.half", "old.previous.training.packed_equal"} <= calls)
        self.assertFalse({"teacher.qualified.fp16", "training.restore"} & calls)
        self.assertNotIn("torch.cuda.reset_peak_memory_stats", calls)
        helper = ast.parse((ROOT / "score_inshop_crop_view_pair.py").read_text())
        bootstrap = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == "bootstrap_lower")
        source = ast.unparse(bootstrap)
        self.assertIn("np.random.default_rng(179019)", source)
        self.assertIn("np.empty(5000)", source)
        self.assertIn("np.unique(labels, return_inverse=True)", source)
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == "export.pair.bootstrap_lower" for n in ast.walk(SCORE)), 2)
        self.assertNotIn("export_large_dense_pilot", (ROOT / "score_late_dense_adaptation.py").read_text())


if __name__ == "__main__":
    unittest.main()
