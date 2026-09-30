#!/usr/bin/env python3
"""Stdlib authority negatives and mmap ownership checks; no Torch/NumPy import."""
if not __debug__:
    raise SystemExit("Checks require assertions")
import ast
import copy
import collections
import gc
import hashlib
import json
import math
import re
import statistics
import sys
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import late_dense_boundary as late
import train_role_matched_adaptation as driver

ROOT = Path(__file__).parent
REFERENCE = ROOT if (ROOT.parent / "docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/role-matched-control-179041-v1.json").exists() else Path("/home/rb/worktrees/sfora-positive-causality/scripts")
EVIDENCE = REFERENCE.parent / "docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1"
EXPORT = ast.parse((ROOT / "export_role_matched_adaptation.py").read_text())
SCORE = ast.parse((ROOT / "score_role_matched_adaptation.py").read_text())
NS = {"math":math,"statistics":statistics,"Path":Path,"late":late,"driver":driver,
      "re":re,"json":json,"collections":collections,"copy":copy,"gc":gc,"SimpleNamespace":SimpleNamespace}
for node in EXPORT.body:
    if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {
        "TRAIN_CODE","CPU_CODE","CPU_SHA","CPU_LOG_SHA","STARTUP_SHA","STARTUP_LOG_SHA","MECHANICS_SHA","MECHANICS_LOG_SHA","ADDED","SEEDS","ORDER","PRECISION"}:
        NS[node.targets[0].id] = ast.literal_eval(node.value)
NS["export"] = SimpleNamespace(SEEDS=NS["SEEDS"],PRECISION=NS["PRECISION"],driver=driver)
NS["METRICS"] = ("per_query_r1","per_query_ap")
for tree,names in ((EXPORT,{"validate_closure","validate_unit","validate_endpoint","paired_cost","frozen_saved",
                           "validate_image_ids","validate_logged_steps","validate_distinct","independent_model"}),
                   (SCORE,{"averaged_deltas","quality_gate","validate_wire"})):
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),
                 "actual-held-guards","exec"),NS)


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def endpoint():
    value = read("role-matched-control-179032-v1.json")
    log = (EVIDENCE / "role-matched-control-179032-v1.log").read_text()
    runtime = re.findall(r"Service runtime: (?:(\d+)min )?([\d.]+)s",log)[0]
    spec = {"seed":value["seed"],"arm":value["arm"],"checkpoint_sha256":value["checkpoint_sha256"],
        "terminal_state_fingerprint":value["terminal_state_fingerprint"],
        "unit":Path(value["unit_cgroup"]).name.removesuffix(".service"),"invocation_id":value["unit_invocation_id"],
        "service_seconds":int(runtime[0] or 0)*60+float(runtime[1]),
        "host_max_rss_kib":int(re.findall(r"Maximum resident set size \(kbytes\): (\d+)",log)[0]),"host_swap_kib":0}
    authority = {key:{"receipt_sha256":NS[key.upper()+"_SHA"],"log_sha256":NS[key.upper()+"_LOG_SHA"]}
                 for key in ("cpu","startup","mechanics")}
    return value,spec,read("role-matched-cpu-v3.json"),authority,read("role-matched-startup-v1.json"),log


class HeldChecks(unittest.TestCase):
    def test_actual_train120_cpu118v3_pins_and_append_only_123(self):
        files = {"TRAIN_CODE":"role-matched-train-execution-v1.json","CPU_CODE":"role-matched-cpu-execution-v3.json",
            "CPU_SHA":"role-matched-cpu-v3.json","CPU_LOG_SHA":"role-matched-cpu-v3.log",
            "STARTUP_SHA":"role-matched-startup-v1.json","STARTUP_LOG_SHA":"role-matched-startup-v1.log",
            "MECHANICS_SHA":"role-matched-mechanics-v1.json","MECHANICS_LOG_SHA":"role-matched-mechanics-v1.log"}
        for key,name in files.items():
            self.assertEqual(hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest(),NS[key])
        previous = read(files["TRAIN_CODE"])
        code = {**previous,**{n:"added" for n in NS["ADDED"]}}
        NS["validate_closure"](code,previous)
        for kind in ("prefix","missing","extra","cpu118"):
            bad,old = code.copy(),previous
            if kind == "prefix": bad[next(iter(previous))] = "changed"
            elif kind == "missing": del bad["export_role_matched_adaptation.py"]
            elif kind == "extra": bad["extra.py"] = "new"
            else: old = dict(list(previous.items())[:118])
            with self.subTest(kind=kind),self.assertRaises(AssertionError): NS["validate_closure"](bad,old)

    def test_fresh_control_endpoint_and_complete_identity_negatives(self):
        value,spec,cpu,authority,startup,_ = endpoint()
        validate = lambda v:NS["validate_endpoint"](v,spec,cpu,authority,startup)
        validate(value)
        for key,change in (("schema","late-dense-train-v1"),("intervention","foreign"),("phase","mechanics"),
            ("boundary",10),("arm","role_matched"),("updates",17),("completed_step",99),("seed",179041),
            ("checkpoint_sha256","archived"),("terminal_state_fingerprint","partial"),("source_checkpoint_sha256","foreign"),
            ("execution_sha256","v3"),("cpu_authority_sha256","old"),("startup_authority_sha256","v3"),
            ("startup_log_sha256","old"),("mechanics_sha256","v3"),("quality_read",True),("training_state_discarded",True),
            ("total_seconds",300),("host_max_rss_kib",8388609),("host_swap_kib",1),("peak_cuda_allocated_bytes",10_000_000_000)):
            bad = copy.deepcopy(value); bad[key] = change
            with self.subTest(key=key),self.assertRaises(AssertionError): validate(bad)
        for key in ("schema","arm","objective","objective_sha256","role_positive_sha256","module_sha256",
                    "mask_rule","rank_coefficient","rank_temperature","parameter_names","model_roles","optimizer_groups","runtime",
                    "schedule_sha256","input_authorities","head_roles","classifier_role","precision"):
            bad = copy.deepcopy(value); bad["resume_identity"][key] = "foreign"
            with self.subTest(identity=key),self.assertRaises(AssertionError): validate(bad)
        for invalid in (-1,13283,True):
            bad = copy.deepcopy(value); bad["steps"][-1]["image_ids"][0] = invalid
            with self.assertRaises(AssertionError): validate(bad)
        for key in ("ce","rank","loss","preclip_norm","seconds"):
            bad = copy.deepcopy(value); bad["steps"][0][key] = float("nan")
            with self.subTest(key=key),self.assertRaises(AssertionError): validate(bad)
        bad = copy.deepcopy(value); bad["optimizer_members"] = 209
        with self.assertRaises(AssertionError): validate(bad)

    def test_actual_original_log_final_rss_and_both_locks(self):
        value,spec,_,_,_,log = endpoint()
        NS["validate_unit"](spec,300,value,True,log)
        NS["validate_logged_steps"](log,value["steps"])
        for key,changed in (("service_seconds",300),("service_seconds",spec["service_seconds"]+.1),
                            ("host_max_rss_kib",spec["host_max_rss_kib"]-1),("host_swap_kib",1),("invocation_id","foreign")):
            with self.subTest(key=key),self.assertRaises(AssertionError):
                NS["validate_unit"]({**spec,key:changed},300,value,True,log)
        for token in ("code=exited/status=0","Finished with result: success",value["unit_invocation_id"],
                      "flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock","flock -n /home/riomus/.sfora-siglip2-gpu.lock"):
            with self.subTest(token=token),self.assertRaises(AssertionError):
                NS["validate_unit"](spec,300,value,True,log.replace(token,"foreign"))
        bad = copy.deepcopy(value["steps"]); bad[-1]["pixels_sha256"] = "foreign"
        with self.assertRaises(AssertionError): NS["validate_logged_steps"](log,bad)

    def test_all_four_actual_endpoints_and_nonparameter_mask(self):
        startup = read("role-matched-startup-v1.json")
        cpu = read("role-matched-cpu-v3.json")
        inventory = read("role-matched-collected-training-units-v1.json")
        endpoints = {}
        for spec in inventory["endpoints"]:
            stem = f"role-matched-{spec['arm']}-{spec['seed']}-v1"
            value, log = read(stem+".json"), (EVIDENCE/(stem+".log")).read_text()
            NS["validate_endpoint"](value,spec,cpu,inventory,startup)
            NS["validate_unit"](spec,300,value,True,log)
            NS["validate_logged_steps"](log,value["steps"])
            self.assertEqual(len(value["resume_identity"]["parameter_names"]),208)
            self.assertEqual(len(value["resume_identity"]["optimizer_groups"]),3)
            endpoints[spec["seed"],spec["arm"]] = {**value,"service_seconds":spec["service_seconds"]}
        costs = NS["paired_cost"](endpoints)
        self.assertTrue(all(v["training_wall_ratio"] <= 1.5 and v["median_step_ratio"] <= 1.5 for v in costs.values()))
        mechanics = read("role-matched-mechanics-v1.json")
        self.assertEqual(endpoints[179032,"role_matched"]["resume_identity"],mechanics["resume_identity"])
        self.assertEqual([driver.diagnostic(v) for v in endpoints[179032,"role_matched"]["steps"][:17]],
                         [driver.diagnostic(v) for v in mechanics["steps"]])
        for name in ("cpu","startup","mechanics"):
            spec = inventory[name]
            value = read("role-matched-"+name+("-v3.json" if name == "cpu" else "-v1.json"))
            log = (EVIDENCE/("role-matched-"+name+("-v3.log" if name == "cpu" else "-v1.log"))).read_text()
            NS["validate_unit"](spec,120,value,name == "mechanics",log)

    def test_matched_images_and_fresh_whole_service_cost(self):
        control,spec,_,_,_,_ = endpoint(); endpoints = {}
        for seed,arm in NS["ORDER"]:
            value = copy.deepcopy(control); value["service_seconds"] = spec["service_seconds"]
            endpoints[seed,arm] = value
        self.assertEqual(NS["paired_cost"](endpoints)[179032]["whole_service_wall_ratio"],1.)
        for key in ("image_ids","rgb_sha256","pixels_sha256","rank_active_before"):
            bad = copy.deepcopy(endpoints); bad[179032,"role_matched"]["steps"][-1][key] = "foreign"
            with self.subTest(key=key),self.assertRaises(AssertionError): NS["paired_cost"](bad)
        for key in ("service_seconds","training_wall_seconds","median_step_3_end_seconds"):
            for factor in (1.50001,0.,float("nan")):
                bad = copy.deepcopy(endpoints); bad[179041,"role_matched"][key] *= factor
                with self.subTest(key=key,factor=factor),self.assertRaises(AssertionError): NS["paired_cost"](bad)
        units = [{"invocation_id":str(i),"receipt":f"/tmp/{i}/receipt.json","run":f"/tmp/{i}"} for i in range(4)]
        NS["validate_distinct"](units)
        for key in ("invocation_id","receipt","run"):
            bad = copy.deepcopy(units); bad[-1][key] = bad[0][key]
            with self.assertRaises(AssertionError): NS["validate_distinct"](bad)

    def test_joint_equal_seed_floors_and_product_bounds(self):
        quality = {s:{a:{m:[.5+(.004 if a == "role_matched" else 0)]*6354 for m in NS["METRICS"]}
                      for a in ("control","role_matched")} for s in NS["SEEDS"]}
        deltas,average = NS["averaged_deltas"](quality)
        intervals = {m:{"mean_delta":statistics.mean(average[m]),"product_lower95":.001,"product_upper95":.008,
                        "query_lower95":-.001,"query_upper95":.008} for m in NS["METRICS"]}
        self.assertEqual(NS["quality_gate"](deltas,intervals),(True,True))
        for metric in NS["METRICS"]:
            bad = copy.deepcopy(intervals); bad[metric]["product_lower95"] = 0.
            self.assertEqual(NS["quality_gate"](deltas,bad),(True,False))
        for metric in NS["METRICS"]:
            low = copy.deepcopy(deltas); low_intervals = copy.deepcopy(intervals)
            for seed in NS["SEEDS"]: low[seed][metric] = [.001]*6354
            low_intervals[metric]["mean_delta"] = .001
            self.assertEqual(NS["quality_gate"](low,low_intervals),(True,False))
        low_ap = copy.deepcopy(deltas); low_ap[179041]["per_query_ap"] = [-.001]*6354
        low_intervals = copy.deepcopy(intervals)
        low_intervals["per_query_ap"]["mean_delta"] = statistics.mean(statistics.mean(low_ap[s]["per_query_ap"]) for s in NS["SEEDS"])
        self.assertEqual(NS["quality_gate"](low_ap,low_intervals),(False,False))
        quality[179041]["role_matched"]["per_query_r1"] = [.5]*6354
        deltas,average = NS["averaged_deltas"](quality)
        intervals["per_query_r1"]["mean_delta"] = statistics.mean(average["per_query_r1"])
        self.assertEqual(NS["quality_gate"](deltas,intervals),(False,False))
        quality[179032]["role_matched"]["per_query_ap"][0] = float("nan")
        with self.assertRaises(AssertionError): NS["averaged_deltas"](quality)

    def test_role_wire_requires_actual_objective_mask_and_updated_reload(self):
        trained,endpoint_spec,_,_,_,_ = endpoint()
        endpoint_spec["receipt_sha256"] = "fresh-training"
        code = {"role_matched_bank_rank.py":"actual-module"}
        spec = {"execution_sha256":"held123"}
        frozen = {"held_manifest":["synthetic"],"query":[0],"gallery":[1]}
        wire = {**frozen,"module_sha256":"actual-module","schema":"role-matched-held-export-v1","role_positive_sha256":driver.MASK,"intervention":driver.METHOD,
            "pass":True,"authority_sha256":"frozen","execution_sha256":"held123","source_code":code,
            "seed":179032,"arm":"control","training_receipt_sha256":"fresh-training",
            "checkpoint_sha256":trained["checkpoint_sha256"],"terminal_state_fingerprint":trained["terminal_state_fingerprint"],
            "boundary":12,"batch":32,"width":128,"precision":NS["PRECISION"],
            "full_held_independent_whole_head_packed_exact":True,"source_head_rng_flags_preserved":True,
            "optimizer_updates":0,"quality_read":False,"official_read":False,"claim_eligible":False,
            "public_serving_qualified":False,"public_latency_measured":False,
            "files":{n:"digest" for n in ("held.npy","held.codes.npy","held.inverse.npy")}}
        validate = lambda v:NS["validate_wire"](v,endpoint_spec,trained,spec,code,"frozen",frozen)
        validate(wire)
        for key,changed in (("module_sha256","foreign"),("role_positive_sha256",""),("arm","role_matched"),
            ("full_held_independent_whole_head_packed_exact",False),("checkpoint_sha256","historical"),
            ("query",[1]),("width",256),("precision","fp32_autocast"),("quality_read",True)):
            with self.subTest(key=key),self.assertRaises(AssertionError): validate({**wire,key:changed})

    def test_reference_constructor_and_mmap_release_before_live_fingerprints(self):
        events,refs = [],[]
        class Mapped(dict): pass
        class Model:
            def __init__(self,config): events.append("constructor")
            def float(self): return self
            def eval(self): return self
            def requires_grad_(self,flag): return self
        def load(*args,**kwargs):
            mapped = Mapped(); refs.append(weakref.ref(mapped)); events.append("mmap")
            return mapped
        def fingerprint(*args):
            self.assertIsNone(refs[0]()); events.append("live-fingerprint")
            return {"owned":True}
        namespace = {**NS,"torch":SimpleNamespace(random=SimpleNamespace(fork_rng=lambda **kw:nullcontext()),
                        device=lambda d:nullcontext(),load=load),"nn":SimpleNamespace(Linear=lambda *a:Model({})),
                     "copy_inference":lambda *a:{"owned":True},"inference_fingerprints":fingerprint}
        fn = next(n for n in EXPORT.body if isinstance(n,ast.FunctionDef) and n.name == "independent_model")
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"actual-independent-reload","exec"),namespace)
        for device in ("cpu","cuda"):
            refs.clear(); events.clear()
            namespace["independent_model"](Model,{}, {"run":"/not-read"},{"owned":True},device)
            self.assertLess(events.index("constructor"),events.index("mmap"))
            self.assertEqual(events[-1],"live-fingerprint")

    def test_updated_model_reuses_fresh_and_releases_complete_state(self):
        events, refs = [], []
        class Owned(dict): pass
        class Module:
            def eval(self): return self
            def requires_grad_(self,flag): return self
        def fresh(*args):
            events.append("fresh")
            state = Owned(model=Module(),head=Module(),processor=object())
            refs.append(weakref.ref(state))
            return state,driver.INITIAL
        def checkpoint(*args):
            events.append("checkpoint")
            saved = Owned(); refs.append(weakref.ref(saved))
            return saved
        def verify(*args):
            if "checkpoint" in events:
                self.assertIsNone(refs[1]())
            events.append("verify")
        def fingerprints(*args):
            self.assertTrue(all(ref() is None for ref in refs))
            events.append("live-fingerprint")
            return {"owned":True}
        base = {"runtime":{"buffer_devices":{"embeddings.position_ids":"cpu"}},"precision":"cpu_float32",
                "numerical_flags":{},"schedule_sha256":"schedule","class_sequence_sha256":"classes",
                "initial_rng_sha256":"initial"}
        recorded = {**base,"precision":"native_float32_fp16_autocast","runtime":late.source_runtime(base["runtime"])}
        def identity(state,args,initial,chosen,code,**kw):
            self.assertEqual(code,{"train":"120"})
            return base
        namespace = {**NS,"torch":SimpleNamespace(random=SimpleNamespace(fork_rng=lambda **kw:nullcontext())),
            "qualification":SimpleNamespace(fresh=fresh),"driver":SimpleNamespace(INITIAL=driver.INITIAL,
                attach_roles=lambda *a:events.append("sidecar"),identity=identity,verify=verify),
            "arguments":lambda *a:None,"checkpoint":checkpoint,"copy_inference":lambda *a:{"owned":True},
            "canonical":lambda v:v,"inference_fingerprints":fingerprints}
        fn = next(n for n in EXPORT.body if isinstance(n,ast.FunctionDef) and n.name == "updated_model")
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"actual-updated-reload","exec"),namespace)
        namespace["updated_model"](None,None,{"arms":{"half":{"rows":[]}}},
            {"seed":179032,"arm":"role_matched"},{"resume_identity":recorded,"code":{"train":"120"}},
            {},{"held":"123"})
        self.assertEqual(events.count("fresh"),1)
        self.assertEqual(events[-1],"live-fingerprint")

    def test_syntax_and_fixed_native_contract(self):
        for name in NS["ADDED"]: compile((ROOT/name).read_text(),name,"exec")
        import export_role_matched_adaptation
        import score_role_matched_adaptation
        self.assertEqual(export_role_matched_adaptation.ORDER,NS["ORDER"])
        self.assertEqual(score_role_matched_adaptation.METRICS,NS["METRICS"])
        self.assertNotIn("torch",sys.modules); self.assertNotIn("numpy",sys.modules)
        calls = {ast.unparse(n.func) for n in ast.walk(EXPORT) if isinstance(n,ast.Call)}
        self.assertTrue({"driver.validate_resume","driver.validate_mechanics","driver.validate_startup","driver.closure",
                         "driver.attach_roles","driver.identity","driver.verify","old.previous.training.packed_equal"} <= calls)
        self.assertEqual(sum(isinstance(n,ast.Call) and ast.unparse(n.func) == "torch.cuda.reset_peak_memory_stats" for n in ast.walk(EXPORT)),1)
        helper = ast.parse((REFERENCE/"score_inshop_crop_view_pair.py").read_text())
        bootstrap = next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name == "bootstrap_lower")
        text = ast.unparse(bootstrap)
        self.assertIn("np.random.default_rng(179019)",text); self.assertIn("np.empty(5000)",text)
        self.assertIn("np.unique(labels, return_inverse=True)",text)
        main = next(n for n in EXPORT.body if isinstance(n,ast.FunctionDef) and n.name == "main")
        text = ast.unparse(main)
        self.assertLess(text.index("torch.cuda.reset_peak_memory_stats()"),text.index("authority(root,"))
        self.assertLess(text.index("torch.cuda.reset_peak_memory_stats()"),text.index("updated_model("))
        self.assertLess(text.index("del model, head"),text.index("independent_model("))
        self.assertLess(text.index("parity(features(model, head, fit)"),text.index("for role in ('query', 'gallery')"))
        self.assertIn("range(0, len(frozen[role]), 32)",text)
        self.assertIn("torch.empty((12599, 128), dtype=torch.float32)",text)
        checkpoint = next(n for n in EXPORT.body if isinstance(n,ast.FunctionDef) and n.name == "checkpoint")
        text = ast.unparse(checkpoint)
        self.assertLess(text.index("del template, members, named"),text.index("old.fingerprint(saved)"))
        self.assertIn("set(saved) == driver.RESUME_KEYS",text)
        self.assertNotIn("torch",sys.modules); self.assertNotIn("numpy",sys.modules)


if __name__ == "__main__":
    unittest.main()
