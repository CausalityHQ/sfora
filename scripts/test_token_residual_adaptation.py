"""Stdlib admission negatives; no Torch import or model execution."""
import copy
import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("train_token_residual_adaptation.py")
EVIDENCE = PATH.parents[1] / "docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1"


class Tensor:
    """Layout/value stand-in for pure saved-state guards; no simulated training."""
    def __init__(self, shape=(), dtype="float32", value=0.):
        self.shape, self.dtype, self.value = shape, dtype, value

    def __float__(self):
        return float(self.value)

    def numel(self):
        return math.prod(self.shape)

    def __repr__(self):
        return repr((self.shape,self.dtype,self.value))


def validator_namespace():
    torch = SimpleNamespace(Tensor=Tensor,float32="float32",
        isfinite=lambda t:SimpleNamespace(all=lambda:math.isfinite(t.value)),
        count_nonzero=lambda t:int(t.value != 0),equal=lambda a,b:repr(a) == repr(b))
    namespace = {"torch":torch,"math":math,"RESUME_KEYS":frozenset(),
        "old":SimpleNamespace(fingerprint=lambda v:hashlib.sha256(repr(v).encode()).hexdigest())}
    tree = ast.parse(PATH.with_name("token_residual_readout.py").read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "validate_weight")
    function.body = [n for n in function.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(PATH),"exec"),namespace)
    namespace["residual"] = SimpleNamespace(validate_weight=namespace["validate_weight"])
    tree = ast.parse(PATH.read_text())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("tensor_layout","validate_resume")]
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(PATH),"exec"),namespace)
    return namespace


def complete_state(driver, arm="candidate", step=8, cpu=True):
    count = 208 + (arm == "candidate")
    members = [(str(i),Tensor((2,))) for i in range(208)]
    if arm == "candidate": members.append(("residual",Tensor((128,4096))))
    groups = [{"params":list(range(205)),"lr":1e-5,"weight_decay":.05},
              {"params":[205,206],"lr":1e-4,"weight_decay":.05},
              {"params":[207],"lr":1e-4,"weight_decay":.05}]
    if arm == "candidate": groups.append({"params":[208],"lr":1e-4,"weight_decay":.05})
    base = {"parameter_names":[n for n,_ in members],"residual_arm":arm,"module_sha256":"module",
        "precision":"cpu_float32" if cpu else "native_float32_fp16_autocast",
        "frozen_names":["v0","embeddings.position_ids"]}
    saved = {"identity":{**base,"global_step":step},"vision":{f"v{i}":Tensor((2,)) for i in range(400)},
        "buffers":{"embeddings.position_ids":Tensor((1,256),"int64")},
        "head":{"weight":Tensor((128,1024)),"bias":Tensor((128,))},"residual":Tensor((128,4096)),
        "classifier":Tensor((2004,128)),"bank":Tensor((13283,128)),
        "optimizer":{"param_groups":groups,"state":{i:{"step":Tensor(value=step),"exp_avg":Tensor(p.shape),
             "exp_avg_sq":Tensor(p.shape)} for i,(_,p) in enumerate(members)} if step else {}},
        "scaler":None if cpu else {"scale":128.,"growth_factor":2.,"backoff_factor":.5,"growth_interval":2000,"_growth_tracker":step},
        "cpu_rng":Tensor((32,),"uint8"),"cuda_rng":[] if cpu else [Tensor((32,),"uint8")]}
    assert len(members) == count and set(saved) == driver.RESUME_KEYS
    fingerprint = validator_namespace()["old"].fingerprint
    base["buffers_sha256"] = fingerprint(saved["buffers"])
    named = {**saved["vision"],**saved["buffers"]}
    base["frozen_sha256"] = fingerprint({n:named[n] for n in base["frozen_names"]})
    saved["identity"] = {**base,"global_step":step}
    return saved,base,groups,members


class Admission(unittest.TestCase):
    def tearDown(self):
        self.assertNotIn("torch",sys.modules,"stdlib checks imported Torch")

    def setUp(self):
        self.assertTrue(PATH.exists(), "W-aware trainer is missing")
        import train_token_residual_adaptation as driver
        self.driver = driver
        self.cpu = json.loads((EVIDENCE / "token-residual-cpu-proof-v1.json").read_text())
        self.code = {**self.cpu["code"], **dict.fromkeys(driver.ADDED, "new")}

    def test_cpu_authority_rejects_foreign_arm_shape_source_and_closure(self):
        d = self.driver
        d.validate_cpu(self.cpu, self.cpu["execution_sha256"], self.code, self.cpu["code"])
        for key, value in (("optimizer_members", {"control":209,"candidate":209}),
                           ("boundary",10), ("token_shape",[2,257,1024]), ("token_dtype","torch.float16"),
                           ("module_sha256","foreign"), ("source_checkpoint_sha256","foreign"),
                           ("nonzero_W_live_token_witness",False), ("quality_read",True),
                           ("initial_state_sha256","foreign"), ("optimizer_updates",1)):
            bad = {**self.cpu, key:value}
            with self.subTest(key=key), self.assertRaises(AssertionError):
                d.validate_cpu(bad, self.cpu["execution_sha256"], self.code, self.cpu["code"])
        for case in ("changed", "missing", "extra"):
            bad = dict(self.code)
            if case == "changed": bad["token_residual_readout.py"] = "foreign"
            elif case == "missing": bad.pop(next(iter(d.ADDED)))
            else: bad["extra.py"] = "foreign"
            with self.subTest(case=case), self.assertRaises((AssertionError, KeyError)):
                d.validate_cpu(self.cpu, self.cpu["execution_sha256"], bad, self.cpu["code"])

    def test_resource_and_original_log_caps(self):
        d = self.driver
        text = (EVIDENCE / "token-residual-cpu-v1.log").read_text()
        d.resources(self.cpu,120,False); d.validate_log(text,self.cpu,120)
        for key,value in (("total_seconds",120), ("total_seconds",float("nan")),
                          ("host_swap_kib",1), ("peak_cuda_allocated_bytes",1),
                          ("unit_memory_max_bytes",16*1024**3), ("host_max_rss_kib",8388609)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                d.resources({**self.cpu,key:value},120,False)
        for bad in (text.replace(self.cpu["unit_invocation_id"],"foreign"),
                    text.replace("status=0","status=1"),
                    text.replace("Service runtime: 34.302s","Service runtime: 120s"),
                    text.replace("flock -n /home/riomus/.sfora-siglip2-gpu.lock","missing")):
            with self.assertRaises(AssertionError): d.validate_log(bad,self.cpu,120)

    def test_real_closure_hashes_and_changed_module(self):
        d = self.driver
        import qualify_token_residual_cpu as cpu
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = {f"file{i}.py":"" for i in range(117)}
            previous["token_residual_readout.py"] = ""
            code = {**previous,**dict.fromkeys(d.ADDED,"")}
            for name in code:
                (root / name).write_text(name)
                code[name] = d.sha(root / name)
                if name in previous: previous[name] = code[name]
            manifest = root / "token-residual-train-execution.json"
            manifest.write_text(json.dumps(code))
            digest = d.sha(manifest)
            with patch.object(cpu,"closure",return_value=previous):
                self.assertEqual(d.closure(root,digest,d.CPU_EXECUTION),(code,previous))
                (root / "token_residual_readout.py").write_text("changed")
                with self.assertRaises(AssertionError): d.closure(root,digest,d.CPU_EXECUTION)

    def test_complete_resume_with_nonpersistent_buffer_and_every_missing_key(self):
        ns = validator_namespace(); ns["RESUME_KEYS"] = self.driver.RESUME_KEYS
        for arm in ("control","candidate"):
            for step in (0,8):
                saved,base,groups,members = complete_state(self.driver,arm,step)
                validate = lambda v:ns["validate_resume"](v,base,step,groups,saved,members)
                validate(saved)
                for key in self.driver.RESUME_KEYS:
                    bad = copy.deepcopy(saved); del bad[key]
                    with self.subTest(arm=arm,step=step,key=key), self.assertRaises(AssertionError): validate(bad)
                bad = copy.deepcopy(saved); bad["buffers"]["embeddings.position_ids"].value = 1
                with self.assertRaises(AssertionError): validate(bad)

    def test_W_identity_moment_shape_dtype_order_and_finite_negatives(self):
        ns = validator_namespace(); ns["RESUME_KEYS"] = self.driver.RESUME_KEYS
        saved,base,groups,members = complete_state(self.driver)
        validate = lambda v:ns["validate_resume"](v,base,8,groups,saved,members)
        validate(saved)
        for key in ("residual_arm","module_sha256","precision","global_step"):
            bad = copy.deepcopy(saved); bad["identity"][key] = "foreign"
            with self.subTest(key=key), self.assertRaises(AssertionError): validate(bad)
        for key in ("residual","classifier","bank"):
            for bad_tensor in (Tensor((1,)),Tensor(saved[key].shape,"float64"),Tensor(saved[key].shape,value=float("nan"))):
                bad = copy.deepcopy(saved); bad[key] = bad_tensor
                with self.subTest(key=key,tensor=bad_tensor), self.assertRaises(AssertionError): validate(bad)
        for case in ("member","moment","fraction","order","options","moment_shape","moment_dtype","nonfinite"):
            bad = copy.deepcopy(saved); optimizer = bad["optimizer"]
            if case == "member": del optimizer["state"][208]
            elif case == "moment": del optimizer["state"][208]["exp_avg"]
            elif case == "fraction": optimizer["state"][208]["step"].value = 8.5
            elif case == "order": optimizer["param_groups"][0]["params"].reverse()
            elif case == "options": optimizer["param_groups"][-1]["lr"] = 1e-3
            elif case == "moment_shape": optimizer["state"][208]["exp_avg"].shape = (1,)
            elif case == "moment_dtype": optimizer["state"][208]["exp_avg"].dtype = "float64"
            else: optimizer["state"][208]["exp_avg_sq"].value = float("inf")
            with self.subTest(case=case), self.assertRaises(AssertionError): validate(bad)
        saved,base,groups,members = complete_state(self.driver,"control")
        bad = copy.deepcopy(saved); bad["residual"].value = .1
        with self.assertRaises(AssertionError): ns["validate_resume"](bad,base,8,groups,saved,members)

    def test_cuda_saved_state_requires_complete_rng_and_scaler(self):
        ns = validator_namespace(); ns["RESUME_KEYS"] = self.driver.RESUME_KEYS
        saved,base,groups,members = complete_state(self.driver,cpu=False)
        validate = lambda v:ns["validate_resume"](v,base,8,groups,saved,members)
        validate(saved)
        for case in ("rng","rng_dtype","scale","growth"):
            bad = copy.deepcopy(saved)
            if case == "rng": bad["cuda_rng"] = []
            elif case == "rng_dtype": bad["cuda_rng"][0].dtype = "float32"
            elif case == "scale": bad["scaler"]["scale"] = float("nan")
            else: del bad["scaler"]["growth_factor"]
            with self.subTest(case=case), self.assertRaises(AssertionError): validate(bad)

    def test_real_atomic_no_clobber_and_file_authority(self):
        tree = ast.parse(PATH.with_name("train_late_dense_adaptation.py").read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "atomic_write")
        ns = {"os":os}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(PATH),"exec"),ns)
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            ns["atomic_write"](path,lambda s:s.write(b'{"pass":true}'))
            digest = self.driver.sha(path)
            self.assertEqual(self.driver.read_authority(path,digest),{"pass":True})
            with self.assertRaises(AssertionError): self.driver.read_authority(path,"foreign")
            with self.assertRaises(AssertionError): ns["atomic_write"](path,lambda s:s.write(b"bad"))
            self.assertEqual(self.driver.sha(path),digest)
            race = path.with_name("race")
            def writer(s): s.write(b"loser"); race.write_bytes(b"winner")
            with self.assertRaises(FileExistsError): ns["atomic_write"](race,writer)
            self.assertEqual(race.read_bytes(),b"winner")
            self.assertFalse(race.with_name("race.part").exists())

    def test_startup_and_mechanics_admission_authority(self):
        d = self.driver
        args = SimpleNamespace(execution_sha256="train",cpu_sha256=d.CPU_PROOF,cpu_log_sha256=d.CPU_LOG,
            cpu_execution_sha256=d.CPU_EXECUTION,startup_sha256="startup",startup_log_sha256="startup-log")
        startup = {**self.cpu,"schema":"token-residual-train-v1","intervention":d.METHOD,"phase":"startup",
            "code":self.code,"execution_sha256":"train","cpu_authority_sha256":d.CPU_PROOF,"cpu_log_sha256":d.CPU_LOG,
            "cpu_execution_sha256":d.CPU_EXECUTION,"both_arms_complete_state_exact":True,"resume_negatives_rejected":True,
            "optimizer_roles_options_exact":True,"changed_module_rejected":True,
            "schedules":{str(k):{"schedule_sha256":str(k),"class_sequence_sha256":"classes",
                "input_authority_sha256":v} for k,v in d.INPUT_AUTHORITIES.items()}}
        startup["resume_identities"] = {arm:{"schema":"token-residual-complete-resume-v1","intervention":d.METHOD,
            "residual_arm":arm,"residual_role":arm == "candidate","precision":"cpu_float32","boundary":12,"width":128,
            "module_sha256":self.code["token_residual_readout.py"],"residual_shape":[128,4096],"residual_dtype":"torch.float32",
            "grid":[16,16],"quadrant_order":["NW","NE","SW","SE"],"pooling":"fixed_8x8_mean_fp32",
            "unit_normalization":"concatenated_quadrants_l2_before_W","execution_sha256":"train",
            "source_checkpoint_sha256":d.SOURCE,"initial_state_sha256":d.INITIAL,
            "cpu_authority_sha256":d.CPU_PROOF,"cpu_log_sha256":d.CPU_LOG,"cpu_execution_sha256":d.CPU_EXECUTION,
            "parameter_names":[str(i) for i in range(208+(arm == "candidate"))],
            "optimizer_groups":[{} for _ in range(3+(arm == "candidate"))]} for arm in ("control","candidate")}
        d.validate_startup(startup,args,self.cpu)
        for key,value in (("phase","mechanics"),("resume_negatives_rejected",False),
                          ("cpu_execution_sha256","foreign"),("optimizer_updates",1)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                d.validate_startup({**startup,key:value},args,self.cpu)
        mechanics = {**startup,"phase":"mechanics","arm":"candidate","seed":179032,
            "startup_authority_sha256":"startup","startup_log_sha256":"startup-log","updates":17,"completed_step":17,
            "checkpoint_sha256":None,"training_state_discarded":True,"native_17_equals_serialized8_plus9_exact":True,
            "strict400_head_W_whole_packed_exact":True,"frozen_named_state_buffers_rng_preserved":True,
            "chunk100_admission_seconds":269.,"schedule_sha256":"179032","peak_cuda_allocated_bytes":6_000_000_000,
            "resume_identity":{"residual_arm":"candidate","intervention":d.METHOD,"schedule_sha256":"179032",
                "class_sequence_sha256":"classes","module_sha256":self.code["token_residual_readout.py"],
                "parameter_names":[str(i) for i in range(209)]},
            "steps":[{"step":i,"optimizer_counter":i,"augmentation_step":1000+i,"image_ids":list(range(64)),
                      "W_gradient_norm":1.,"token_shape":[16,256,1024],"token_dtype":"torch.float16",
                      "pooled_shape":[16,1024],"pooled_dtype":"torch.float16","encoder_calls":4,
                      "zero_W_raw_packed_same_pass_exact":i == 1}
                     for i in range(1,18)]}
        d.validate_mechanics(mechanics,args,self.cpu,startup)
        for key,value in (("arm","control"),("seed",179041),("completed_step",8),
                          ("startup_log_sha256","foreign"),("strict400_head_W_whole_packed_exact",False),
                          ("chunk100_admission_seconds",float("nan")),("chunk100_admission_seconds",270.)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                d.validate_mechanics({**mechanics,key:value},args,self.cpu,startup)
        for key,value in (("image_ids",[True]+list(range(1,64))),("W_gradient_norm",0.),("token_shape",[16,257,1024])):
            bad = copy.deepcopy(mechanics); bad["steps"][0][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError): d.validate_mechanics(bad,args,self.cpu,startup)

    def test_strict_tensor_sections_fail_before_restore_copies(self):
        ns = validator_namespace(); ns["RESUME_KEYS"] = self.driver.RESUME_KEYS
        saved,base,groups,members = complete_state(self.driver)
        for key in ("vision","head","buffers"):
            for case in ("keys","shape","dtype","finite"):
                bad = copy.deepcopy(saved); name = next(iter(bad[key]))
                if case == "keys": del bad[key][name]
                elif case == "shape": bad[key][name].shape = (1,)
                elif case == "dtype": bad[key][name].dtype = "float64"
                else: bad[key][name].value = float("nan")
                with self.subTest(key=key,case=case), self.assertRaises(AssertionError):
                    ns["validate_resume"](bad,base,8,groups,saved,members)
        tree = ast.parse(PATH.read_text())
        restore = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "restore")
        calls = [ast.unparse(n.func) for statement in restore.body for n in ast.walk(statement) if isinstance(n,ast.Call)]
        load = next(i for i,c in enumerate(calls) if c.endswith(".load_state_dict"))
        self.assertLess(calls.index("validate_resume"),load)


if __name__ == "__main__":
    unittest.main()
