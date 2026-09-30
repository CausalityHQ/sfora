#!/usr/bin/env python3
"""Stdlib negative authority/statistical checks; never import native code."""
if not __debug__:
    raise SystemExit("Checks require assertions")

import ast
import copy
import collections
import hashlib
import sys
import json
import re
import math
import statistics
from pathlib import Path
from types import SimpleNamespace
import unittest

import late_dense_boundary as late
import train_image_queue_adaptation as driver

ROOT = Path(__file__).parent
EXPORT = ast.parse((ROOT / "export_image_queue_adaptation.py").read_text())
SCORE = ast.parse((ROOT / "score_image_queue_adaptation.py").read_text())
NAMESPACE = {"math": math, "statistics": statistics, "Path": Path, "late": late,
             "training": SimpleNamespace(METHOD=driver.METHOD), "driver": driver,
             "re": re, "json": json, "collections": collections}
for node in EXPORT.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {
        "TRAIN_CODE", "CPU_CODE", "CPU_SHA", "CPU_LOG_SHA", "ADDED", "SEEDS", "ORDER", "PRECISION"}:
        NAMESPACE[node.targets[0].id] = ast.literal_eval(node.value)
NAMESPACE["export"] = SimpleNamespace(SEEDS=NAMESPACE["SEEDS"], PRECISION=NAMESPACE["PRECISION"], driver=driver)
NAMESPACE["METRICS"] = ("per_query_r1", "per_query_ap")
for tree, names in ((EXPORT, {"validate_closure", "validate_unit", "validate_endpoint", "paired_cost", "frozen_saved", "validate_startup", "validate_image_ids", "validate_logged_steps", "validate_distinct"}),
                    (SCORE, {"averaged_deltas", "quality_gate", "validate_wire"})):
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]),
                 "stdlib-authority-check", "exec"), NAMESPACE)


def example(seed=179032, arm="queue"):
    boundary = 12
    spec = {"seed": seed, "arm": arm, "checkpoint_sha256": "fresh", "terminal_state_fingerprint": "complete",
            "unit": f"queue-{seed}-{arm}", "invocation_id": "original", "service_seconds": 200,
            "host_max_rss_kib": 7 * 1024 * 1024, "host_swap_kib": 0}
    authority = {"cpu": {"receipt_sha256": NAMESPACE["CPU_SHA"], "log_sha256": NAMESPACE["CPU_LOG_SHA"]},
                 "startup": {"receipt_sha256": "startup"},
                 "mechanics": {"receipt_sha256": "mechanics", "log_sha256": "mechanics-log"}}
    base = {"schema": "image-queue-complete-resume-v1", "sampling_arm": arm, "class_sequence_sha256": f"classes-{seed}", "intervention": NAMESPACE["training"].METHOD,
        "arm": "half", "width": 128, "total_updates": 100, "seed": seed, "boundary": boundary,
        "augmentation_offset": 1000, "execution_sha256": NAMESPACE["TRAIN_CODE"],
        "source_checkpoint_sha256": late.SOURCE_SHA, "source_execution_sha256": late.SOURCE_CODE,
        "source_schedule_sha256": late.SOURCE_SCHEDULE, "cpu_authority_sha256": NAMESPACE["CPU_SHA"],
        "cpu_execution_sha256": NAMESPACE["CPU_CODE"], "cpu_log_sha256": NAMESPACE["CPU_LOG_SHA"], "initial_state_sha256": "initial",
        "schedule_sha256": f"schedule-{seed}-{arm}", "precision": "native_float32_fp16_autocast", "batch": 64, "micro": 16,
        "parameter_names": [f"encoder.layers.12.{i}" for i in range(205)] + ["compact_head.weight", "compact_head.bias", "classifier"],
        "target_positive_sha256": "order", "buffers_sha256": "buffers", "runtime": "runtime",
        "frozen_names": ["frozen"], "frozen_sha256": "frozen",
        "model_roles": [(f"embeddings.{i}", False) for i in range(195)] + [(f"encoder.layers.12.{i}", True) for i in range(205)]}
    base["optimizer_groups"] = [{"parameter_names": n, "options": {"lr": lr, "weight_decay": .05}}
        for n, lr in ((base["parameter_names"][:-3], 1e-5), (base["parameter_names"][-3:-1], 1e-4), (["classifier"], 1e-4))]
    value = {"pass": True, "phase": "train", "updates": 100, "completed_step": 100, "quality_read": False,
        "training_state_discarded": False, "frozen_named_state_buffers_rng_preserved": True,
        "schema": "image-queue-train-v1", "arm": arm, "claim_eligible": False, "startup_authority_sha256": "startup",
        "mechanics_sha256": "mechanics", "mechanics_log_sha256": "mechanics-log",
        "resume_identity": base, "checkpoint_sha256": "fresh", "terminal_state_fingerprint": "complete",
        "steps": [{"step": i, "optimizer_counter": i, "augmentation_step": 1000 + i, "rgb_sha256": f"rgb-{arm}-{i}",
                   "pixels_sha256": f"pixels-{arm}-{i}", "image_ids": list(range(64)), "rank_active_before": True, "seconds": 1.0} for i in range(1, 101)],
        "training_wall_seconds": 100., "median_step_3_end_seconds": 1., "unit_invocation_id": "original",
        "unit_cgroup": f"/system.slice/{spec['unit']}.service", "unit_memory_max_bytes": 8 * 1024**3,
        "unit_memory_swap_max_bytes": 0, "host_max_rss_kib": spec["host_max_rss_kib"], "host_swap_kib": 0,
        "total_seconds": 190., "peak_cuda_allocated_bytes": 7_000_000_000}
    value.update({k: base[k] for k in ("intervention", "seed", "boundary", "source_checkpoint_sha256", "execution_sha256",
                 "cpu_authority_sha256", "cpu_execution_sha256", "cpu_log_sha256", "initial_state_sha256", "schedule_sha256")})
    return value, spec, {"initial_state_sha256": "initial", "schedules": {str(seed): {arm + "_schedule_sha256": base["schedule_sha256"], "class_sequence_sha256": base["class_sequence_sha256"]}}}, authority


class HeldGate(unittest.TestCase):
    def test_exact_append_only_closure(self):
        manifest = ROOT.parent / "docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/image-queue-train-execution-v4.json"
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), NAMESPACE["TRAIN_CODE"])
        previous = json.loads(manifest.read_text())
        self.assertEqual(len(previous), 120)
        code = {**previous, **{n: "new" for n in NAMESPACE["ADDED"]}}
        NAMESPACE["validate_closure"](code, previous)
        cpu_manifest = manifest.with_name("image-queue-cpu-execution-v2.json")
        foreign_prefix = json.loads(cpu_manifest.read_text())
        with self.assertRaises(AssertionError):
            NAMESPACE["validate_closure"]({**foreign_prefix, **{n: "new" for n in NAMESPACE["ADDED"]}}, foreign_prefix)
        for mutation in ("prefix", "extra", "missing"):
            bad = code.copy()
            if mutation == "prefix": bad[next(iter(previous))] = "changed"
            elif mutation == "extra": bad["other.py"] = "new"
            else: del bad["export_image_queue_adaptation.py"]
            with self.subTest(mutation=mutation), self.assertRaises((AssertionError, KeyError)):
                NAMESPACE["validate_closure"](bad, previous)

    def test_fresh_terminal_resource_and_identity_negatives(self):
        value, spec, cpu, authority = example()
        validate = lambda v: NAMESPACE["validate_endpoint"](v, spec, cpu, authority)
        validate(value)
        for key, changed in (("schema", "late-dense-train-v1"), ("intervention", "late-dense-adaptation-v1"), ("phase", "mechanics"), ("updates", 17), ("completed_step", 99),
            ("seed", 179041), ("boundary", 10), ("checkpoint_sha256", "archived"),
            ("terminal_state_fingerprint", "partial"), ("source_checkpoint_sha256", "F5"),
            ("execution_sha256", "old114"), ("cpu_authority_sha256", "oldCPU"),
            ("mechanics_sha256", "oldGPU"), ("startup_authority_sha256", "old-startup"), ("arm", "candidate"),
            ("quality_read", True), ("host_max_rss_kib", 9 * 1024 * 1024), ("host_swap_kib", 1),
            ("total_seconds", 300), ("peak_cuda_allocated_bytes", 10_000_000_000), ("training_wall_seconds", float("nan"))):
            bad = copy.deepcopy(value); bad[key] = changed
            with self.subTest(key=key), self.assertRaises(AssertionError): validate(bad)
        for key in ("schema", "source_schedule_sha256", "precision", "augmentation_offset", "batch", "boundary", "seed", "sampling_arm", "class_sequence_sha256"):
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
        for key in ("rank_active_before",):
            bad = copy.deepcopy(endpoints); bad[179032, "queue"]["steps"][49][key] = "changed"
            with self.subTest(key=key), self.assertRaises(AssertionError): NAMESPACE["paired_cost"](bad)
        for key, changed in (("training_wall_seconds", 150.01), ("median_step_3_end_seconds", 1.5001)):
            bad = copy.deepcopy(endpoints); bad[179041, "queue"][key] = changed
            with self.subTest(cost=key), self.assertRaises(AssertionError): NAMESPACE["paired_cost"](bad)
        bad = endpoints.copy(); del bad[179032, "control"]
        with self.assertRaises(AssertionError): NAMESPACE["paired_cost"](bad)
        saved = {"vision": {"embeddings.weight": 1, "encoder.layers.10.weight": 2, "encoder.layers.12.weight": 3},
                 "buffers": {"embeddings.position_ids": 4}}
        self.assertIn("encoder.layers.10.weight", NAMESPACE["frozen_saved"](saved, 12))

    def test_equal_seed_average_both_metric_bounds_and_seed_floors(self):
        quality = {s: {a: {m: [0.5 + (0.004 if a == "queue" else 0)] * 6354 for m in NAMESPACE["METRICS"]}
                       for a in ("control", "queue")} for s in NAMESPACE["SEEDS"]}
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
        query_negative = copy.deepcopy(intervals)
        for metric in NAMESPACE["METRICS"]:
            query_negative[metric]["query_lower95"] = -.001
        self.assertEqual(NAMESPACE["quality_gate"](deltas, query_negative), (True, True))
        quality[179041]["queue"]["per_query_r1"] = [0.5] * 6354
        deltas, average = NAMESPACE["averaged_deltas"](quality)
        intervals["per_query_r1"]["mean_delta"] = statistics.mean(average["per_query_r1"])
        self.assertEqual(NAMESPACE["quality_gate"](deltas, intervals), (False, False))
        self.assertAlmostEqual(average["per_query_r1"][0], .002)
        quality[179041]["queue"]["per_query_ap"] = [0.499] * 6354
        deltas, average = NAMESPACE["averaged_deltas"](quality)
        intervals["per_query_ap"]["mean_delta"] = statistics.mean(average["per_query_ap"])
        self.assertEqual(NAMESPACE["quality_gate"](deltas, intervals), (False, False))
        quality[179032]["queue"]["per_query_ap"][0] = float("nan")
        with self.assertRaises(AssertionError): NAMESPACE["averaged_deltas"](quality)

    def test_wire_precision_complete_source_binding(self):
        training, endpoint, _, _ = example()
        endpoint["receipt_sha256"] = "original-training"
        spec, code = {"execution_sha256": "held123"}, {"export.py": "frozen"}
        frozen = {"held_manifest": ["synthetic"], "query": [0], "gallery": [1]}
        receipt = {**frozen, "pass": True, "authority_sha256": "parent-frozen", "execution_sha256": "held123",
            "source_code": code, "seed": 179032, "arm": "queue", "intervention": driver.METHOD, "training_receipt_sha256": "original-training",
            "checkpoint_sha256": "fresh", "terminal_state_fingerprint": "complete", "boundary": 12,
            "batch": 32, "width": 128, "precision": NAMESPACE["PRECISION"], "optimizer_updates": 0,
            "quality_read": False, "official_read": False, "claim_eligible": False,
            "public_serving_qualified": False, "public_latency_measured": False,
            "full_held_independent_whole_head_packed_exact": True, "source_head_rng_flags_preserved": True,
            "files": {n: "digest" for n in ("held.npy", "held.codes.npy", "held.inverse.npy")}}
        validate = lambda r: NAMESPACE["validate_wire"](r, endpoint, training, spec, code, "parent-frozen", frozen)
        validate(receipt)
        for key, changed in (("intervention", "late-dense-adaptation-v1"), ("checkpoint_sha256", "archived"), ("arm", "control"), ("precision", "fp32_autocast"),
                             ("batch", 16), ("width", 256), ("authority_sha256", "unreviewed"),
                             ("full_held_independent_whole_head_packed_exact", False), ("query", [1]), ("quality_read", True)):
            bad = copy.deepcopy(receipt); bad[key] = changed
            with self.subTest(wire=key), self.assertRaises(AssertionError): validate(bad)

    def test_all100_image_ids_rank_flags_and_independent_log_diagnostics(self):
        value = example()[0]
        target = list(range(64)) * 2
        batches = [list(range(64)) for _ in range(100)]
        NAMESPACE["validate_image_ids"](value, batches, target)
        log = "\n".join(json.dumps(r) for r in value["steps"])
        NAMESPACE["validate_logged_steps"](log, value["steps"])
        for position in (0, 16, 99):
            for key, changed in (("image_ids", list(range(64, 128))), ("rank_active_before", False)):
                bad = copy.deepcopy(value); bad["steps"][position][key] = changed
                with self.subTest(position=position, key=key), self.assertRaises(AssertionError):
                    NAMESPACE["validate_image_ids"](bad, batches, target)
            for key in ("rgb_sha256", "pixels_sha256", "optimizer_counter"):
                bad = copy.deepcopy(value); bad["steps"][position][key] = "foreign"
                with self.subTest(position=position, key=key), self.assertRaises(AssertionError):
                    NAMESPACE["validate_logged_steps"](log, bad["steps"])
        for invalid in (-1, 13283, True):
            value, spec, cpu, authority = example()
            value["steps"][-1]["image_ids"][0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                NAMESPACE["validate_endpoint"](value, spec, cpu, authority)

    def test_parameter_roles_optimizer_members_and_class_schedule_identity(self):
        value, spec, cpu, authority = example()
        for mutation in ("duplicate", "missing", "frozen_role", "rate", "groups", "arm"):
            bad = copy.deepcopy(value); base = bad["resume_identity"]
            if mutation == "duplicate": base["parameter_names"][-1] = base["parameter_names"][0]
            elif mutation == "missing": base["parameter_names"].pop()
            elif mutation == "frozen_role": base["model_roles"][0] = (base["model_roles"][0][0], True)
            elif mutation == "rate": base["optimizer_groups"][0]["options"]["lr"] = 1e-4
            elif mutation == "groups": base["optimizer_groups"][0]["parameter_names"].reverse()
            else: base["arm"] = "queue"
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                NAMESPACE["validate_endpoint"](bad, spec, cpu, authority)
        endpoints = {(seed, arm): example(seed, arm)[0] for seed, arm in NAMESPACE["ORDER"]}
        for key in ("class_sequence_sha256", "model_roles", "frozen_sha256"):
            bad = copy.deepcopy(endpoints); bad[179032, "queue"]["resume_identity"][key] = "foreign"
            with self.subTest(key=key), self.assertRaises(AssertionError): NAMESPACE["paired_cost"](bad)
        for key in ("training_wall_seconds", "median_step_3_end_seconds"):
            for invalid in (0., -1., float("nan"), float("inf")):
                bad = copy.deepcopy(endpoints); bad[179032, "control"][key] = invalid
                with self.subTest(key=key, invalid=invalid), self.assertRaises(AssertionError): NAMESPACE["paired_cost"](bad)

    def test_actual_unit_final_rss_and_original_log_bindings(self):
        evidence = ROOT.parent / "docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1"
        for name, seconds, final_rss in (("image-queue-cpu-v2", 37.327, 5156656),
                                         ("image-queue-startup-v1", 12.993, 998348)):
            filename = "image-queue-cpu-proof-v2.json" if "cpu" in name else name + ".json"
            value = json.loads((evidence / filename).read_text())
            log = (evidence / (name + ".log")).read_text()
            spec = {"unit": Path(value["unit_cgroup"]).name.removesuffix(".service"),
                "invocation_id": value["unit_invocation_id"], "service_seconds": seconds,
                "host_max_rss_kib": final_rss, "host_swap_kib": 0}
            validate = NAMESPACE["validate_unit"]
            validate(spec, 120, value, False, log)
            for key, changed in (("service_seconds", 120), ("service_seconds", seconds + .1),
                                 ("host_max_rss_kib", final_rss - 1), ("host_max_rss_kib", 8388609),
                                 ("host_swap_kib", 1), ("invocation_id", "foreign")):
                with self.subTest(name=name, key=key), self.assertRaises(AssertionError):
                    validate({**spec, key: changed}, 120, value, False, log)
            for altered in (log.replace("code=exited/status=0", "code=exited/status=1"),
                            log.replace(value["unit_invocation_id"], "foreign"),
                            log.replace("flock -n /home/riomus/.sfora-siglip2-gpu.lock", "missing-lock")):
                with self.assertRaises(AssertionError): validate(spec, 120, value, False, altered)

    def test_startup_identity_and_duplicate_original_units(self):
        value, _, _, authority = example()
        startup = {**value, "phase": "startup"}
        NAMESPACE["validate_startup"](startup, authority)
        for key, changed in (("schema", "late-dense-train-v1"), ("intervention", "late-dense-adaptation-v1"),
                             ("phase", "train"), ("arm", "candidate"), ("seed", 179043), ("boundary", 10),
                             ("execution_sha256", "foreign"), ("cpu_execution_sha256", "foreign")):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                NAMESPACE["validate_startup"]({**startup, key: changed}, authority)
        units = [{"unit": str(i), "invocation_id": str(i), "receipt": f"/original/{i}/receipt.json", "run": f"/original/{i}"} for i in range(4)]
        NAMESPACE["validate_distinct"](units)
        for key in ("invocation_id", "receipt", "run"):
            bad = copy.deepcopy(units); bad[-1][key] = bad[0][key]
            with self.subTest(key=key), self.assertRaises(AssertionError): NAMESPACE["validate_distinct"](bad)

    def test_complete_mmap_released_before_fit_forward(self):
        main = next(n for n in EXPORT.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        source = ast.unparse(main)
        self.assertLess(source.index("del saved"), source.index("features(m, h, pixels)"))
        self.assertLess(source.index("model_pair(control, saved, state)"), source.index("del saved"))
        self.assertLess(source.index("checkpoint(endpoint, value)"), source.index("del saved"))

    def test_native_reload_precision_and_shared_draw_contract(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)
        calls = {ast.unparse(n.func) for n in ast.walk(EXPORT) if isinstance(n, ast.Call)}
        self.assertTrue({"qualification.fresh", "training.identity", "training.verify", "training.validate_resume", "driver.validate_cpu", "driver.validate_mechanics", "driver.validate_log", "queue_cpu.closure", "qualification.startup",
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
        self.assertNotIn("export_large_dense_pilot", (ROOT / "score_image_queue_adaptation.py").read_text())


if __name__ == "__main__":
    unittest.main()
