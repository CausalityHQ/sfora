#!/usr/bin/env python3
"""Frozen token-residual endpoints: fit-only CPU proof and native FP16 held export.

Use a NEW root: token-residual-held-execution.json is the exact immutable TRAIN120
prefix plus these three scripts (123 files). Historical startup still uses
ORIGINAL114; CPU118 is validated against TRAIN120, never held123.
CLI: --phase cpu|export --seed 179032|179041 --arm control|candidate
--authority-sha256 SHA --output NEW_DIR. Export additionally requires
--cpu-proof PATH --cpu-sha256 SHA --cpu-log PATH --cpu-log-sha256 SHA.

Parent pins token-residual-held-authority.json, schema token-residual-held-authority-v1:
execution_sha256, training_root (the immutable TRAIN120 directory), cpu,
startup, mechanics, endpoints. Every UNIT_SPEC contains
receipt, receipt_sha256, log, log_sha256, unit, invocation_id, service_seconds,
host_max_rss_kib (FINAL native timer RSS, not systemd MemoryPeak), host_swap_kib.
Endpoints add seed, arm (control|candidate), run, checkpoint_sha256,
terminal_state_fingerprint, ordered control032,candidate032,candidate041,control041.
Parent supplies actual future hashes/units/resources; no predicted authorities.
Export CLI also pins its own fit-only CPU proof and original log by SHA256.
"""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import time
UNIT_STARTED = time.perf_counter()

import argparse
import copy
import collections
import gc
import json
import math
import os
import resource
import re
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_late_dense_adaptation as training
import train_token_residual_adaptation as driver
import qualify_token_residual_cpu as cpu_module
import token_residual_readout as residual
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

qualification, late, old, pair = training.qualification, training.late, training.old, training.pair
sha = pair.sha
# The driver normally binds these only in its main(); reuse its actual guards.
driver.torch, driver.native, driver.qualified = torch, training, qualification
driver.old, driver.pair, driver.residual, driver.cpu_module = old, pair, residual, cpu_module
ORIGINAL114 = "c9cec7b71d0c2ba90de2eb0d5c9810f111d4e7feb88f10e63055fbabf8a6d8b3"
TRAIN_CODE = "77e843bfe59bc94e89c48e147ef707598e3febcb9fac8cef960efd12289d1b0d"
CPU_CODE = "96377b69e3c6fc1730bdffa674884977150ec4a4c35febab154aa42151ead408"
CPU_SHA = "67c7c2452d211f4755770c703147d970a588cddace63d6e1550ed2728ecec06b"
CPU_LOG_SHA = "cff9c3de737c1b486501abda47c591486f0d449f2268441611df3955029cb6aa"
ADDED = {"export_token_residual_adaptation.py", "score_token_residual_adaptation.py", "test_token_residual_held.py"}
STARTUP_SHA = "b12ab73e5a3ed8aeee6769580b07860e3a7c7f1348d4d6229c69206f95d2d7be"
STARTUP_LOG_SHA = "bfe87a842f064325ceacf7cb5155d17300832d5f68233ca7949b32abe9be92a8"
MECHANICS_SHA = "62f190c27b71629bd25bcee04bebb4ac28657f8f00301007e91b2c9ea7ee7f2d"
MECHANICS_LOG_SHA = "14e4baadc9235079fe4af1005e638e28cb5f5403141927df7b15bc4b48d60f4b"
SEEDS = (179032, 179041)
ORDER = ((179032, "control"), (179032, "candidate"), (179041, "candidate"), (179041, "control"))
PRECISION = "private_native_fp16"


def rss_checkpoint(stage):
    status = Path("/proc/self/status").read_text().splitlines()
    current = int(next(v for v in status if v.startswith("VmRSS:")).split()[1])
    print(json.dumps({"stage": stage, "rss_kib": current,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds": time.perf_counter() - UNIT_STARTED}), flush=True)


def canonical(value):
    return json.loads(json.dumps(value))


def validate_closure(code, previous):
    assert len(previous) == 120 and len(code) == 123
    assert set(code) - set(previous) == ADDED
    assert all(code[n] == h for n, h in previous.items()), "qualified prefix changed"


def validate_unit(spec, cap, receipt=None, gpu=True, log=None):
    assert spec["unit"] and spec["invocation_id"]
    assert math.isfinite(spec["service_seconds"]) and 0 < spec["service_seconds"] < cap
    assert 0 < spec["host_max_rss_kib"] <= 8 * 1024 * 1024 and spec["host_swap_kib"] == 0
    if receipt is not None:
        driver.resources(receipt, cap, gpu)
        assert receipt["unit_invocation_id"] == spec["invocation_id"]
        assert Path(receipt["unit_cgroup"]).name == spec["unit"] + ".service"
        assert receipt["host_max_rss_kib"] <= spec["host_max_rss_kib"]
    if log is not None:
        assert receipt is not None
        driver.validate_log(log, receipt, cap)
        runtime = re.findall(r"Service runtime: (?:(\d+)min )?([\d.]+)s", log)
        seconds = int(runtime[0][0] or 0) * 60 + float(runtime[0][1])
        assert seconds == spec["service_seconds"]
        rss = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", log)
        assert int(rss[0]) == spec["host_max_rss_kib"]


def read_unit(spec, cap, gpu):
    value = training.read_authority(Path(spec["receipt"]), spec["receipt_sha256"])
    log = training.read_authority(Path(spec["log"]), spec["log_sha256"], log=True)
    validate_unit(spec, cap, value, gpu, log)
    return value


def arguments(authority, seed=179032, arm="candidate"):
    return SimpleNamespace(seed=seed, arm=arm, execution_sha256=TRAIN_CODE,
        cpu_sha256=CPU_SHA, cpu_log_sha256=CPU_LOG_SHA, cpu_execution_sha256=CPU_CODE,
        startup_sha256=authority["startup"]["receipt_sha256"],
        startup_log_sha256=authority["startup"]["log_sha256"])


def validate_endpoint(value, spec, cpu, authority, startup):
    seed, arm = spec["seed"], spec["arm"]
    assert seed in SEEDS and arm in ("control", "candidate")
    assert value["schema"] == "token-residual-train-v1" and value["pass"] is True
    assert value["intervention"] == driver.METHOD and value["phase"] == "train" and value["arm"] == arm
    assert value["seed"] == seed and value["boundary"] == 12
    assert value["updates"] == value["completed_step"] == 100
    assert value["quality_read"] is False and value["claim_eligible"] is False and value["training_state_discarded"] is False
    assert value["frozen_named_state_buffers_rng_preserved"] is True
    for key, expected in (("source_checkpoint_sha256", driver.SOURCE), ("execution_sha256", TRAIN_CODE),
        ("cpu_authority_sha256", CPU_SHA), ("cpu_execution_sha256", CPU_CODE), ("cpu_log_sha256", CPU_LOG_SHA),
        ("initial_state_sha256", driver.INITIAL), ("checkpoint_sha256", spec["checkpoint_sha256"]),
        ("terminal_state_fingerprint", spec["terminal_state_fingerprint"]),
        ("startup_authority_sha256", STARTUP_SHA), ("startup_log_sha256", STARTUP_LOG_SHA),
        ("mechanics_sha256", MECHANICS_SHA), ("mechanics_log_sha256", MECHANICS_LOG_SHA)):
        assert value[key] == expected, key
    assert value["code"] == startup["code"]
    base = value["resume_identity"]
    reference = startup["resume_identities"][arm]
    schedule = startup["schedules"][str(seed)]
    expected = {**reference, "seed": seed, "schedule_sha256": schedule["schedule_sha256"],
        "class_sequence_sha256": schedule["class_sequence_sha256"], "precision": "native_float32_fp16_autocast",
        "runtime": late.source_runtime(reference["runtime"]), "initial_rng_sha256": base["initial_rng_sha256"],
        "startup_authority_sha256": STARTUP_SHA, "startup_log_sha256": STARTUP_LOG_SHA,
        "numerical_flags": {**reference["numerical_flags"], "deterministic":True, "cudnn_tf32":False}}
    assert base == expected, "foreign complete residual identity"
    assert base["initial_rng_sha256"] and base["residual_arm"] == arm and base["residual_role"] is (arm == "candidate")
    assert value["schedule_sha256"] == schedule["schedule_sha256"]
    assert len(base["parameter_names"]) == len(set(base["parameter_names"])) == 208 + (arm == "candidate")
    assert base["parameter_names"][:208] == startup["resume_identities"]["control"]["parameter_names"]
    if arm == "candidate": assert base["parameter_names"][-1] == "residual"
    assert len(value["steps"]) == 100
    for step, row in enumerate(value["steps"], 1):
        assert row["step"] == row["optimizer_counter"] == step and row["augmentation_step"] == 1000 + step
        assert row["rgb_sha256"] and row["pixels_sha256"] and type(row["rank_active_before"]) is bool
        assert len(row["image_ids"]) == len(set(row["image_ids"])) == 64
        assert all(type(i) is int and 0 <= i < 13283 for i in row["image_ids"])
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
        assert math.isfinite(row["W_norm"]) and math.isfinite(row["residual_global_raw_norm_ratio"])
        if arm == "control":
            assert row["W_norm"] == row["residual_global_raw_norm_ratio"] == 0
        else:
            assert row["W_norm"] > 0 and row["W_gradient_norm"] > 0 and math.isfinite(row["W_gradient_norm"])
            assert row["token_shape"] == [16,256,1024] and row["pooled_shape"] == [16,1024]
            assert row["token_dtype"] in ("torch.float16", "torch.float32") and row["pooled_dtype"] in ("torch.float16", "torch.float32")
            assert row["encoder_calls"] == 4 and row["zero_W_raw_packed_same_pass_exact"] is (step == 1)
    assert math.isfinite(value["training_wall_seconds"]) and value["training_wall_seconds"] > 0
    assert value["median_step_3_end_seconds"] == statistics.median(r["seconds"] for r in value["steps"][2:])
    validate_unit(spec, 300, value)


def validate_image_ids(value, batches, target):
    assert [r["image_ids"] for r in value["steps"]] == batches[:len(value["steps"])]
    counts = collections.Counter(target)
    assert all(len({target[i] for i in r["image_ids"]}) == 64 for r in value["steps"])
    assert all(r["rank_active_before"] == all(counts[target[i]] > 1 for i in r["image_ids"]) for r in value["steps"])


def paired_cost(endpoints):
    assert set(endpoints) == set(ORDER)
    costs = {}
    for seed in SEEDS:
        control, candidate = (endpoints[seed, arm] for arm in ("control", "candidate"))
        for k in ("schedule_sha256", "initial_state_sha256", "source_checkpoint_sha256"):
            assert candidate[k] == control[k], "same-pair source/schedule differs"
        for k in ("class_sequence_sha256", "target_positive_sha256", "buffers_sha256", "runtime",
                  "model_roles", "frozen_names", "frozen_sha256", "initial_rng_sha256", "numerical_flags"):
            assert candidate["resume_identity"][k] == control["resume_identity"][k]
        assert candidate["resume_identity"]["parameter_names"][:208] == control["resume_identity"]["parameter_names"]
        assert candidate["resume_identity"]["optimizer_groups"][:3] == control["resume_identity"]["optimizer_groups"]
        assert all(a[k] == b[k] for a, b in zip(control["steps"], candidate["steps"], strict=True)
                   for k in ("step", "optimizer_counter", "augmentation_step", "image_ids", "rgb_sha256", "pixels_sha256", "rank_active_before"))
        for value in (control, candidate):
            assert all(math.isfinite(value[k]) and value[k] > 0 for k in ("service_seconds", "training_wall_seconds", "median_step_3_end_seconds"))
        service = candidate["service_seconds"] / control["service_seconds"]
        assert math.isfinite(service) and 0 < service <= 1.50
        wall = candidate["training_wall_seconds"] / control["training_wall_seconds"]
        median = candidate["median_step_3_end_seconds"] / control["median_step_3_end_seconds"]
        assert math.isfinite(wall) and math.isfinite(median) and 0 < wall <= 1.50 and 0 < median <= 1.50
        costs[seed] = {"whole_service_wall_ratio": service, "training_wall_ratio": wall, "median_step_ratio": median,
                       "control_wall_seconds": control["training_wall_seconds"],
                       "candidate_wall_seconds": candidate["training_wall_seconds"]}
    return costs


def frozen_saved(saved, boundary):
    roots = ("embeddings.",) + tuple(f"encoder.layers.{i}." for i in range(boundary))
    return {n: v for group in (saved["vision"], saved["buffers"]) for n, v in group.items() if n.startswith(roots)}


def checkpoint(spec, value, state=None):
    path = Path(spec["run"]) / "resume.pt"
    assert sha(path) == spec["checkpoint_sha256"] == value["checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert set(saved) == driver.RESUME_KEYS, "complete W resume required"
    assert canonical(saved["identity"]) == {**value["resume_identity"], "global_step": 100}
    base = {k: v for k, v in saved["identity"].items() if k != "global_step"}
    assert set(saved["vision"]) == {n for n,_ in base["model_roles"]}
    assert set(saved["buffers"]) == set(base["runtime"]["buffer_devices"])
    assert set(saved["scaler"]) == {"scale","growth_factor","backoff_factor","growth_interval","_growth_tracker"}
    assert saved["scaler"]["growth_factor"] == 2. and saved["scaler"]["backoff_factor"] == .5 and saved["scaler"]["growth_interval"] == 2000
    groups = saved["optimizer"]["param_groups"]
    offset = 0
    for group, role in zip(groups, base["optimizer_groups"], strict=True):
        assert group["params"] == list(range(offset, offset + len(role["parameter_names"])))
        assert canonical({k: v for k, v in group.items() if k != "params"}) == canonical(role["options"])
        offset += len(role["parameter_names"])
    assert offset == len(base["parameter_names"]) == 208 + (spec["arm"] == "candidate")
    assert saved["classifier"].shape == (2004,128) and saved["bank"].shape == (13283,128)
    assert set(saved["head"]) == {"weight", "bias"}
    assert saved["head"]["weight"].shape == (128,1024) and saved["head"]["bias"].shape == (128,)
    assert all(t.dtype == torch.float32 for key in ("vision","head") for t in saved[key].values() if t.is_floating_point())
    assert saved["classifier"].dtype == saved["bank"].dtype == torch.float32
    assert saved["cpu_rng"].dtype == torch.uint8 and saved["cpu_rng"].ndim == 1
    assert len(saved["cuda_rng"]) == 1 and saved["cuda_rng"][0].dtype == torch.uint8 and saved["cuda_rng"][0].ndim == 1
    named = {**saved["vision"], **saved["head"], "compact_head.weight": saved["head"]["weight"],
             "compact_head.bias": saved["head"]["bias"], "classifier": saved["classifier"], "residual": saved["residual"]}
    members = [(n,named[n]) for n in base["parameter_names"]]
    template = saved
    if state is not None:
        template = driver.payload(state, base)
        # The CPU fresh constructor has no scaler/CUDA RNG. Validate the trained
        # CUDA layout separately, retaining only the template key/shape contract.
        template = {**template, "scaler": saved["scaler"], "cuda_rng": saved["cuda_rng"]}
        assert groups == state["optimizer"].state_dict()["param_groups"]
        members = state["params"]
    driver.validate_resume(saved, base, 100, groups, template, members)
    assert bool(torch.count_nonzero(saved["residual"])) is (spec["arm"] == "candidate")
    assert old.fingerprint(saved) == spec["terminal_state_fingerprint"] == value["terminal_state_fingerprint"]
    return saved


def validate_logged_steps(log, steps):
    rows = []
    for line in log.splitlines():
        if line.startswith("{"):
            value = json.loads(line)
            if "image_ids" in value:
                rows.append(value)
    assert rows == steps, "original log image/augmentation diagnostics differ"


def authority(root, expected):
    spec = training.read_authority(root / "token-residual-held-authority.json", expected)
    assert spec["schema"] == "token-residual-held-authority-v1"
    train_root = Path(spec["training_root"]).resolve()
    assert Path(spec["training_root"]).is_absolute() and not root.resolve().is_relative_to(train_root)
    code = training.read_authority(root / "token-residual-held-execution.json", spec["execution_sha256"])
    previous, cpu_code = driver.closure(root, TRAIN_CODE, CPU_CODE)
    validate_closure(code, previous)
    assert previous == training.read_authority(train_root / "token-residual-train-execution.json", TRAIN_CODE)
    assert all(sha(root / n) == h and sha(train_root / n) == h for n,h in previous.items())
    assert all(sha(root / n) == h for n,h in code.items())
    for key, digest, log_digest in (("cpu",CPU_SHA,CPU_LOG_SHA), ("startup",STARTUP_SHA,STARTUP_LOG_SHA),
                                   ("mechanics",MECHANICS_SHA,MECHANICS_LOG_SHA)):
        assert spec[key]["receipt_sha256"] == digest and spec[key]["log_sha256"] == log_digest
    cpu = read_unit(spec["cpu"], 120, False)
    driver.validate_cpu(cpu, CPU_CODE, previous, cpu_code)
    startup = read_unit(spec["startup"], 120, False)
    options = arguments(spec)
    driver.validate_startup(startup, options, cpu)
    assert startup["code"] == previous
    mechanics = read_unit(spec["mechanics"], 120, True)
    driver.validate_mechanics(mechanics, options, cpu, startup)
    assert [training.diagnostic(r) for r in mechanics["resumed_steps"]] == [training.diagnostic(r) for r in mechanics["steps"][8:]]
    selected = qualification.archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, "helpers", lambda r, _: helpers(r, code)):
        control, source, prior, proof, frozen, _ = qualification.startup(root, ORIGINAL114)
    pair.executing_authority(root, code)
    assert len(frozen["query"]) == 6354 and len(frozen["gallery"]) == 6245
    assert set(frozen["query"]).isdisjoint(frozen["gallery"])
    assert sorted(frozen["query"] + frozen["gallery"]) == list(range(12599))
    assert len({r["product"] for r in frozen["held_manifest"]}) == 1993
    values = old.initializers(proof, "half")
    target = values["target"].tolist()
    assert len(target) == 13283 and len(set(target)) == 2004
    names = sorted({r["product"] for r in frozen["fit_manifest"]})
    assert [names[i] for i in target] == [r["product"] for r in frozen["fit_manifest"]]
    positive = pair.smoke.member_bank_positive_ordinals(np.asarray(target), allow_singletons=True)
    target_sha = old.fingerprint({"target":values["target"], "positive":positive})
    del values
    schedules = {}
    for seed in SEEDS:
        batches = old.coverage.schedule(target, seed=seed).tolist()
        expected_schedule = startup["schedules"][str(seed)]
        assert pair.smoke.digest({"batches":torch.tensor(batches)}) == expected_schedule["schedule_sha256"]
        import hashlib
        assert hashlib.sha256(json.dumps([[target[i] for i in b] for b in batches],separators=(",",":")).encode()).hexdigest() == expected_schedule["class_sequence_sha256"]
        schedules[seed] = batches
    validate_image_ids(mechanics, schedules[179032], target)
    assert sha(qualification.RESUME) == late.SOURCE_SHA
    original = torch.load(qualification.RESUME, map_location="cpu", weights_only=True, mmap=True)
    source_identity = canonical(original["identity"])
    source_frozen_names = sorted(frozen_saved(original, 12))
    source_frozen = old.fingerprint(frozen_saved(original, 12))
    del original; gc.collect()
    assert [(e["seed"],e["arm"]) for e in spec["endpoints"]] == list(ORDER)
    endpoints = {}
    for endpoint in spec["endpoints"]:
        assert Path(endpoint["receipt"]) == Path(endpoint["run"]) / "receipt.json"
        value = read_unit(endpoint, 300, True)
        validate_endpoint(value, endpoint, cpu, spec, startup)
        log = training.read_authority(Path(endpoint["log"]), endpoint["log_sha256"], log=True)
        validate_logged_steps(log, value["steps"])
        base = value["resume_identity"]
        assert base["target_positive_sha256"] == target_sha and base["frozen_sha256"] == source_frozen
        assert base["frozen_names"] == source_frozen_names
        assert all(base[k] == source_identity[k] for k in ("model_roles","buffers_sha256","runtime"))
        assert base["parameter_names"][:208] == source_identity["parameter_names"]
        validate_image_ids(value, schedules[endpoint["seed"]], target)
        if (endpoint["seed"],endpoint["arm"]) == (179032,"candidate"):
            assert base == mechanics["resume_identity"]
            assert [training.diagnostic(r) for r in value["steps"][:17]] == [training.diagnostic(r) for r in mechanics["steps"]]
        saved = checkpoint(endpoint, value)
        del saved; gc.collect()
        rss_checkpoint(f"endpoint-{endpoint['seed']}-{endpoint['arm']}")
        value["service_seconds"] = endpoint["service_seconds"]
        endpoints[endpoint["seed"],endpoint["arm"]] = value
    validate_distinct(spec["endpoints"] + [spec[k] for k in ("cpu","startup","mechanics")])
    return spec,code,endpoints,paired_cost(endpoints),control,source,prior,proof,frozen


def validate_distinct(units):
    assert len({e["invocation_id"] for e in units}) == len(units)
    assert len({str(Path(e["receipt"]).resolve()) for e in units}) == len(units)
    runs = [str(Path(e["run"]).resolve()) for e in units if "run" in e]
    assert len(set(runs)) == len(runs)


def copy_inference(model, head, weight, saved):
    assert len(saved["vision"]) == 400
    model.load_state_dict(saved["vision"], strict=True)
    head.load_state_dict(saved["head"], strict=True)
    buffers = dict(model.named_buffers())
    assert buffers.keys() == saved["buffers"].keys()
    with torch.no_grad():
        weight.copy_(saved["residual"])
        for n in buffers: buffers[n].copy_(saved["buffers"][n])
    return {key:old.fingerprint(saved[key]) for key in ("vision","buffers","head","residual")}


def inference_fingerprints(model, head, weight):
    return {"vision":old.fingerprint(model.state_dict()), "buffers":old.fingerprint(dict(model.named_buffers())),
        "head":old.fingerprint(head.state_dict()), "residual":old.fingerprint(weight.detach())}


def updated_model(control, source, proof, endpoint, value, spec, code):
    with torch.random.fork_rng(devices=[]):
        state, initial = qualification.fresh(control, source, proof, 12, "cpu")
        residual.attach(state, endpoint["arm"])
    assert initial == driver.INITIAL
    options = arguments(spec, endpoint["seed"], endpoint["arm"])
    recorded = value["resume_identity"]
    base = driver.identity(state, options, initial, recorded["schedule_sha256"], recorded["class_sequence_sha256"], code,
                           initial_rng_sha256=recorded["initial_rng_sha256"])
    assert {**canonical(base), "precision":recorded["precision"], "runtime":canonical(late.source_runtime(base["runtime"]))} == recorded
    driver.verify(state, base)
    saved = checkpoint(endpoint, value, state)
    expected = copy_inference(state["model"],state["head"],state["residual"],saved)
    del saved; gc.collect()
    # Only inference tensors leave this scope; no saved optimizer scalar/view survives.
    driver.verify(state, base)
    model, head, weight, processor = (state[k] for k in ("model","head","residual","processor"))
    del state; gc.collect()
    assert inference_fingerprints(model,head,weight) == expected
    for module in (model,head): module.eval().requires_grad_(False)
    weight.requires_grad_(False)
    residual.validate_weight(weight,endpoint["arm"])
    assert bool(torch.count_nonzero(weight)) is (endpoint["arm"] == "candidate")
    return model,head,weight,processor,expected


def independent_model(model_type, config, endpoint, expected, device):
    # One host model at a time; CUDA reference construction stays inside its RNG fork.
    with torch.random.fork_rng(devices=[0] if device == "cuda" else []):
        with torch.device(device):
            model = model_type(copy.deepcopy(config)).float().eval()
            head = nn.Linear(1024,128).eval()
            weight = residual.new_weight(device,False)
    saved = torch.load(Path(endpoint["run"]) / "resume.pt",map_location="cpu",weights_only=True,mmap=True)
    assert copy_inference(model,head,weight,saved) == expected
    del saved; gc.collect()
    assert inference_fingerprints(model,head,weight) == expected
    model.requires_grad_(False); head.requires_grad_(False)
    return model,head,weight


def features(model, head, weight, pixels, independent=False):
    output = model(pixel_values=pixels)
    if independent:
        grid = output.last_hidden_state.float().reshape(len(pixels),16,16,1024)
        quadrants = torch.cat([grid[:,r:r+8,c:c+8].mean(dim=(1,2)) for r,c in ((0,0),(0,8),(8,0),(8,8))],dim=1)
        assert torch.isfinite(quadrants).all() and (quadrants.norm(dim=1) > 0).all()
        raw = pair.smoke.compact_head_features(output.pooler_output.float(),head) + F.linear(F.normalize(quadrants,dim=1),weight)
    else:
        raw = residual.raw_features(output.pooler_output.float(),output.last_hidden_state,head,weight)
    assert torch.isfinite(raw).all()
    return raw, F.normalize(raw,dim=1), output.pooler_output, output.last_hidden_state


def parity(a, b):
    assert all(torch.equal(x,y) for x,y in zip(a,b,strict=True)), "independent raw/head/W/token reload differs"
    old.previous.training.packed_equal(a[1],b[1])
    return a[1]


def resource_unit(cap, gpu, started=None):
    group = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip().lstrip("/")
    assert "memory" in (group / "cgroup.controllers").read_text().split()
    assert int((group / "memory.max").read_text()) == 8 * 1024**3 and int((group / "memory.swap.max").read_text()) == 0
    assert str(os.getpid()) in (group / "cgroup.procs").read_text().split()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    swap = int(next(v for v in Path("/proc/self/status").read_text().splitlines() if v.startswith("VmSwap:")).split()[1])
    total = time.perf_counter() - (UNIT_STARTED if started is None else started)
    assert total < cap and 0 < rss <= 8 * 1024 * 1024 and swap == 0
    peak = torch.cuda.max_memory_allocated() if gpu else 0
    assert peak < 10_000_000_000
    return {"unit_invocation_id": os.environ["INVOCATION_ID"], "unit_cgroup": str(group),
        "unit_memory_max_bytes": 8 * 1024**3, "unit_memory_swap_max_bytes": 0,
        "host_max_rss_kib": rss, "host_swap_kib": swap, "total_seconds": total,
        "peak_cuda_allocated_bytes": peak}


def publish(path, value):
    training.atomic_write(path, lambda s: s.write((json.dumps(value, indent=2) + "\n").encode()))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--authority-sha256",required=True)
    p.add_argument("--phase",choices=("cpu","export"),required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--arm",choices=("control","candidate"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cpu-proof",type=Path)
    p.add_argument("--cpu-sha256")
    p.add_argument("--cpu-log",type=Path)
    p.add_argument("--cpu-log-sha256")
    args = p.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    gpu = args.phase == "export"
    assert torch.cuda.is_available() == gpu
    if gpu:
        assert torch.cuda.device_count() == 1 and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert all((args.cpu_proof,args.cpu_sha256,args.cpu_log,args.cpu_log_sha256))
    else:
        assert not any((args.cpu_proof,args.cpu_sha256,args.cpu_log,args.cpu_log_sha256))
    resource_unit(300 if gpu else 120,gpu)
    torch.set_num_threads(8); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parent
    spec,code,endpoints,_,control,source,prior,proof,frozen = authority(root,args.authority_sha256)
    assert not args.output.resolve().is_relative_to(Path(spec["training_root"]).resolve())
    assert not args.output.resolve().is_relative_to(root)
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    endpoint = next(e for e in spec["endpoints"] if (e["seed"],e["arm"]) == (args.seed,args.arm))
    value = endpoints[args.seed,args.arm]
    binding = {"intervention":driver.METHOD,"authority_sha256":args.authority_sha256,"execution_sha256":spec["execution_sha256"],
        "seed":args.seed,"arm":args.arm,"boundary":12,"module_sha256":code["token_residual_readout.py"],
        "training_receipt_sha256":endpoint["receipt_sha256"],"checkpoint_sha256":endpoint["checkpoint_sha256"],
        "terminal_state_fingerprint":endpoint["terminal_state_fingerprint"]}
    cpu_rng = torch.random.get_rng_state().clone()
    model,head,weight,processor,expected = updated_model(control,source,proof,endpoint,value,spec,code)
    assert canonical(late.source_runtime(old.coverage.trained.base.runtime_identity(model))) == value["resume_identity"]["runtime"]
    assert canonical(old.coverage.trained.native.environment(model,processor)) == source["environment"]
    model_type,config = type(model),copy.deepcopy(model.config)
    whole = pair.smoke.digest(old.coverage.trained.base.whole_state(model))
    head_sha = pair.smoke.digest(head.state_dict()); W_sha = old.fingerprint(weight.detach())
    with patch.object(pair,"SEED",args.seed):
        images,_ = pair.augmented_images(control.dataset_root,frozen["fit_manifest"],tuple(value["steps"][0]["image_ids"][:2]),1001)
    pixels = pair.pixels(processor,images,"large")
    pixel_sha = pair.smoke.digest({"pixels":pixels})
    with torch.inference_mode():
        original_fit = features(model,head,weight,pixels)
    model.half()
    f16whole = pair.smoke.digest(old.coverage.trained.base.whole_state(model))
    rss_checkpoint("updated-model-mmap-released")
    if gpu:
        cpu = training.read_authority(args.cpu_proof,args.cpu_sha256)
        driver.validate_log(training.read_authority(args.cpu_log,args.cpu_log_sha256,log=True),cpu,120)
        driver.resources(cpu,120,False)
        assert all(cpu[k] == v for k,v in binding.items())
        assert cpu["pass"] is True and cpu["source_code"] == code and cpu["quality_read"] is False
        assert cpu["strict400_whole_head_W_fit_packed_reload_exact"] and cpu["frozen_roles_complete_state_exact"] and cpu["cpu_rng_preserved"]
        assert cpu["optimizer_updates"] == 0
        assert all(cpu[k] == v for k,v in (("whole_sha256",whole),("f16_whole_sha256",f16whole),("head_sha256",head_sha),("W_sha256",W_sha),("fit_pixels_sha256",pixel_sha)))
        model.cuda(); head.cuda(); weight = weight.detach().cuda()
    else:
        # Retain only fit outputs; release the first host model before constructing
        # the independent config clone and opening the trained mmap again.
        del model,head,weight; gc.collect()
    clone,clone_head,clone_W = independent_model(model_type,config,endpoint,expected,"cuda" if gpu else "cpu")
    assert pair.smoke.digest(old.coverage.trained.base.whole_state(clone)) == whole
    assert pair.smoke.digest(clone_head.state_dict()) == head_sha and old.fingerprint(clone_W.detach()) == W_sha
    assert canonical(late.source_runtime(old.coverage.trained.base.runtime_identity(clone))) == value["resume_identity"]["runtime"]
    assert canonical(old.coverage.trained.native.environment(clone,processor)) == source["environment"]
    if not gpu:
        with torch.inference_mode(): parity(original_fit,features(clone,clone_head,clone_W,pixels,True))
    del original_fit
    clone.half()
    assert pair.smoke.digest(old.coverage.trained.base.whole_state(clone)) == f16whole
    rss_checkpoint("independent-reload-mmap-released")
    if not gpu:
        assert torch.equal(cpu_rng,torch.random.get_rng_state())
        assert all(sha(root / n) == h for n,h in code.items())
        assert sha(Path(endpoint["run"]) / "resume.pt") == binding["checkpoint_sha256"]
        args.output.mkdir()
        publish(args.output / "proof.json",{**binding,**resource_unit(120,False),"pass":True,
            "source_code":code,"whole_sha256":whole,"f16_whole_sha256":f16whole,"head_sha256":head_sha,"W_sha256":W_sha,
            "fit_pixels_sha256":pixel_sha,"strict400_whole_head_W_fit_packed_reload_exact":True,
            "frozen_roles_complete_state_exact":True,"cpu_rng_preserved":True,"optimizer_updates":0,"quality_read":False})
        return
    assert all(p.dtype == torch.float16 for m in (model,clone) for p in m.parameters())
    assert all(p.dtype == torch.float32 for h in (head,clone_head) for p in h.parameters())
    assert weight.dtype == clone_W.dtype == torch.float32
    rng = old.fingerprint({"cpu":torch.random.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()})
    values = torch.empty((12599,128),dtype=torch.float32)
    with torch.inference_mode():
        fit = pixels.half().cuda()
        parity(features(model,head,weight,fit),features(clone,clone_head,clone_W,fit,True))
        # No held read precedes the complete authority and updated fit reload.
        for role in ("query","gallery"):
            for start in range(0,len(frozen[role]),32):
                indices = frozen[role][start:start+32]
                rows = [frozen["held_manifest"][i] for i in indices]
                images,_ = pair.augmented_images(control.dataset_root,rows,tuple(range(len(rows))),None)
                pixels = pair.pixels(processor,images,"large").half().cuda()
                values[indices] = parity(features(model,head,weight,pixels),features(clone,clone_head,clone_W,pixels,True)).cpu()
    torch.cuda.synchronize()
    assert rng == old.fingerprint({"cpu":torch.random.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()})
    for m,h,w in ((model,head,weight),(clone,clone_head,clone_W)):
        assert pair.smoke.digest(old.coverage.trained.base.whole_state(m)) == f16whole
        assert pair.smoke.digest(h.state_dict()) == head_sha and old.fingerprint(w.detach()) == W_sha
        assert all(p.grad is None for p in (*m.parameters(),*h.parameters(),w))
        assert canonical(old.coverage.trained.native.environment(m,processor)) == source["environment"]
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    assert all(sha(root / n) == h for n,h in code.items())
    assert sha(Path(endpoint["run"]) / "resume.pt") == binding["checkpoint_sha256"]
    packed = pack_int8_unit_embeddings(values)
    args.output.mkdir()
    hashes = {}
    for name,tensor in (("held.npy",values),("held.codes.npy",packed.codes),("held.inverse.npy",packed.inverse_norms)):
        path = args.output / name
        training.atomic_write(path,lambda stream,tensor=tensor:np.save(stream,tensor.numpy(),allow_pickle=False))
        hashes[name] = sha(path)
    publish(args.output / "receipt.json",{**binding,**resource_unit(300,True),"pass":True,
        "source_code":code,"cpu_authority_sha256":args.cpu_sha256,"cpu_proof":str(args.cpu_proof),"cpu_log_sha256":args.cpu_log_sha256,
        "whole_sha256":whole,"f16_whole_sha256":f16whole,"head_sha256":head_sha,"W_sha256":W_sha,
        "files":hashes,"held_manifest":frozen["held_manifest"],"query":frozen["query"],"gallery":frozen["gallery"],
        "full_held_independent_whole_head_W_packed_exact":True,"source_head_rng_flags_preserved":True,
        "batch":32,"width":128,"precision":PRECISION,"optimizer_updates":0,"quality_read":False,"official_read":False,
        "claim_eligible":False,"public_serving_qualified":False,"public_latency_measured":False})


if __name__ == "__main__":
    main()
