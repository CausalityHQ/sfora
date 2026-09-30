#!/usr/bin/env python3
"""Frozen image-queue endpoints: fit-only CPU proof and native FP16 held export.

Use a NEW root: image-queue-held-execution.json is the exact immutable TRAIN120
prefix plus these three scripts (123 files). Historical startup still uses
ORIGINAL114; CPU118 is validated against TRAIN120, never held123.

Parent pins image-queue-held-authority.json, schema image-queue-held-authority-v1:
execution_sha256, cpu, startup, mechanics, endpoints. Every UNIT_SPEC contains
receipt, receipt_sha256, log, log_sha256, unit, invocation_id, service_seconds,
host_max_rss_kib (FINAL native timer RSS, not systemd MemoryPeak), host_swap_kib.
Endpoints add seed, arm (control|queue), run, checkpoint_sha256,
terminal_state_fingerprint, ordered control032,queue032,queue041,control041.
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
import train_image_queue_adaptation as driver
import qualify_image_queue_adaptation_cpu as queue_cpu
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

qualification, late, old, pair = training.qualification, training.late, training.old, training.pair
sha = pair.sha
TRAIN_ROOT = Path("/home/riomus/runs/sfora-image-queue-source-v4")
ORIGINAL114 = "c9cec7b71d0c2ba90de2eb0d5c9810f111d4e7feb88f10e63055fbabf8a6d8b3"
TRAIN_CODE = "e8f27358a48b1c2543ec5dbcb85fbbca4d6b7968022bfca5c484d38291f5b4a6"
CPU_CODE = "fdf48c2d8d1bf90143fecc68afce2bed2a0e468ddc6dd65148429165a9879188"
CPU_SHA = "5d490958903ef5a1b0be1e06af2514566fc4464929f17827a159c83c4f7a1668"
CPU_LOG_SHA = "8092cb815b2661a3065f188b6fd59e27dacc564365336c526f0cea5c5f1f22ca"
ADDED = {"export_image_queue_adaptation.py", "score_image_queue_adaptation.py", "test_image_queue_held.py"}
SEEDS = (179032, 179041)
ORDER = ((179032, "control"), (179032, "queue"), (179041, "queue"), (179041, "control"))
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


def validate_startup(value, authority):
    assert value["schema"] == "image-queue-train-v1" and value["pass"]
    assert value["intervention"] == driver.METHOD and value["phase"] == "startup"
    assert value["arm"] in ("control", "queue") and value["seed"] in SEEDS and value["boundary"] == 12
    assert value["execution_sha256"] == TRAIN_CODE and value["cpu_execution_sha256"] == CPU_CODE
    assert value["cpu_authority_sha256"] == authority["cpu"]["receipt_sha256"] == CPU_SHA
    assert value["cpu_log_sha256"] == authority["cpu"]["log_sha256"] == CPU_LOG_SHA
    assert value["source_checkpoint_sha256"] == late.SOURCE_SHA
    assert value["quality_read"] is False and value["claim_eligible"] is False


def validate_endpoint(value, spec, cpu, authority):
    seed, arm = spec["seed"], spec["arm"]
    assert seed in SEEDS and arm in ("control", "queue")
    assert value["schema"] == "image-queue-train-v1" and value["pass"]
    assert value["intervention"] == driver.METHOD and value["phase"] == "train" and value["arm"] == arm
    assert value["seed"] == seed and value["boundary"] == 12
    assert value["updates"] == value["completed_step"] == 100
    assert value["quality_read"] is False and value["claim_eligible"] is False and value["training_state_discarded"] is False
    assert value["frozen_named_state_buffers_rng_preserved"]
    assert value["source_checkpoint_sha256"] == late.SOURCE_SHA and value["execution_sha256"] == TRAIN_CODE
    assert value["cpu_authority_sha256"] == authority["cpu"]["receipt_sha256"] == CPU_SHA
    assert value["cpu_execution_sha256"] == CPU_CODE and value["cpu_log_sha256"] == authority["cpu"]["log_sha256"] == CPU_LOG_SHA
    assert value["startup_authority_sha256"] == authority["startup"]["receipt_sha256"]
    assert value["mechanics_sha256"] == authority["mechanics"]["receipt_sha256"]
    assert value["mechanics_log_sha256"] == authority["mechanics"]["log_sha256"]
    assert value["initial_state_sha256"] == cpu["initial_state_sha256"]
    assert value["checkpoint_sha256"] == spec["checkpoint_sha256"]
    assert value["terminal_state_fingerprint"] == spec["terminal_state_fingerprint"]
    schedule = cpu["schedules"][str(seed)]
    assert value["schedule_sha256"] == schedule[arm + "_schedule_sha256"]
    base = value["resume_identity"]
    expected = {"schema": "image-queue-complete-resume-v1", "intervention": driver.METHOD,
        "arm": "half", "sampling_arm": arm, "width": 128, "total_updates": 100, "seed": seed, "boundary": 12,
        "augmentation_offset": 1000, "execution_sha256": TRAIN_CODE,
        "source_checkpoint_sha256": late.SOURCE_SHA, "source_execution_sha256": late.SOURCE_CODE,
        "source_schedule_sha256": late.SOURCE_SCHEDULE, "cpu_authority_sha256": CPU_SHA,
        "cpu_execution_sha256": CPU_CODE, "cpu_log_sha256": CPU_LOG_SHA,
        "initial_state_sha256": cpu["initial_state_sha256"], "schedule_sha256": value["schedule_sha256"],
        "class_sequence_sha256": schedule["class_sequence_sha256"],
        "precision": "native_float32_fp16_autocast", "batch": 64, "micro": 16}
    assert all(base[k] == v for k, v in expected.items()), "foreign image-queue resume identity"
    assert len(base["parameter_names"]) == len(set(base["parameter_names"])) == 208
    roots = ("embeddings.",) + tuple(f"encoder.layers.{i}." for i in range(12))
    roles = base["model_roles"]
    assert len(roles) == len({n for n, _ in roles}) == 400
    assert all(type(flag) is bool and flag == (not n.startswith(roots)) for n, flag in roles)
    assert base["parameter_names"] == [n for n, flag in roles if flag] + ["compact_head.weight", "compact_head.bias", "classifier"]
    groups = base["optimizer_groups"]
    assert [g["parameter_names"] for g in groups] == [base["parameter_names"][:-3], base["parameter_names"][-3:-1], ["classifier"]]
    assert [g["options"]["lr"] for g in groups] == [1e-5, 1e-4, 1e-4]
    assert all(g["options"]["weight_decay"] == .05 for g in groups)
    assert len(value["steps"]) == 100
    for step, row in enumerate(value["steps"], 1):
        assert row["step"] == row["optimizer_counter"] == step and row["augmentation_step"] == 1000 + step
        assert row["rgb_sha256"] and row["pixels_sha256"] and type(row["rank_active_before"]) is bool
        assert len(row["image_ids"]) == len(set(row["image_ids"])) == 64
        assert all(type(i) is int and 0 <= i < 13283 for i in row["image_ids"])
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
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
        control, queue = (endpoints[seed, arm] for arm in ("control", "queue"))
        for k in ("initial_state_sha256", "source_checkpoint_sha256"):
            assert queue[k] == control[k], "same-pair source differs"
        for k in ("class_sequence_sha256", "target_positive_sha256", "buffers_sha256", "runtime",
                  "parameter_names", "model_roles", "optimizer_groups", "frozen_names", "frozen_sha256"):
            assert queue["resume_identity"][k] == control["resume_identity"][k]
        assert all(a[k] == b[k] for a, b in zip(control["steps"], queue["steps"], strict=True)
                   for k in ("step", "optimizer_counter", "augmentation_step", "rank_active_before"))
        # RGB/pixels/image rows intentionally differ: each is bound to its own log and schedule.
        for value in (control, queue):
            assert all(math.isfinite(value[k]) and value[k] > 0 for k in ("training_wall_seconds", "median_step_3_end_seconds"))
        wall = queue["training_wall_seconds"] / control["training_wall_seconds"]
        median = queue["median_step_3_end_seconds"] / control["median_step_3_end_seconds"]
        assert math.isfinite(wall) and math.isfinite(median) and 0 < wall <= 1.50 and 0 < median <= 1.50
        costs[seed] = {"training_wall_ratio": wall, "median_step_ratio": median,
                       "control_wall_seconds": control["training_wall_seconds"],
                       "queue_wall_seconds": queue["training_wall_seconds"]}
    return costs


def frozen_saved(saved, boundary):
    roots = ("embeddings.",) + tuple(f"encoder.layers.{i}." for i in range(boundary))
    return {n: v for group in (saved["vision"], saved["buffers"]) for n, v in group.items() if n.startswith(roots)}


def checkpoint(spec, value):
    path = Path(spec["run"]) / "resume.pt"
    assert sha(path) == spec["checkpoint_sha256"] == value["checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert set(saved) == late.RESUME_KEYS, "complete image-queue resume required"
    assert canonical(saved["identity"]) == {**value["resume_identity"], "global_step": 100}
    # JSON receipts stringify config mapping keys and turn role tuples into lists.
    # Authenticate their canonical form, retain native types for the strict guard.
    base = {k: v for k, v in saved["identity"].items() if k != "global_step"}
    training.validate_resume(saved, base, 100, saved["optimizer"]["param_groups"])
    assert old.fingerprint(saved) == spec["terminal_state_fingerprint"] == value["terminal_state_fingerprint"]
    groups = saved["optimizer"]["param_groups"]
    assert len(groups) == len(base["optimizer_groups"]) == 3
    offset = 0
    for group, role in zip(groups, base["optimizer_groups"], strict=True):
        names = role["parameter_names"]
        assert group["params"] == list(range(offset, offset + len(names)))
        assert {k: v for k, v in group.items() if k != "params"} == role["options"]
        offset += len(names)
    assert offset == 208
    assert saved["cpu_rng"].dtype == torch.uint8 and saved["cpu_rng"].ndim == 1
    assert saved["cuda_rng"][0].dtype == torch.uint8 and saved["cuda_rng"][0].ndim == 1
    assert saved["classifier"].shape == (2004, 128) and saved["bank"].shape == (13283, 128)
    assert old.fingerprint(saved["buffers"]) == base["buffers_sha256"]
    assert old.fingerprint(frozen_saved(saved, base["boundary"])) == base["frozen_sha256"]
    assert set(saved["head"]) == {"weight", "bias"}
    assert saved["head"]["weight"].shape == (128, 1024) and saved["head"]["bias"].shape == (128,)
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
    spec = training.read_authority(root / "image-queue-held-authority.json", expected)
    assert spec["schema"] == "image-queue-held-authority-v1"
    assert not root.resolve().is_relative_to(TRAIN_ROOT.resolve()), "immutable training source cannot be extended in place"
    code = training.read_authority(root / "image-queue-held-execution.json", spec["execution_sha256"])
    previous = training.read_authority(root / "image-queue-train-execution.json", TRAIN_CODE)
    validate_closure(code, previous)
    assert previous == training.read_authority(TRAIN_ROOT / "image-queue-train-execution.json", TRAIN_CODE)
    assert all(sha(root / n) == h and sha(TRAIN_ROOT / n) == h for n, h in previous.items())
    assert all(sha(root / n) == h for n, h in code.items())
    cpu_spec = spec["cpu"]
    assert cpu_spec["receipt_sha256"] == CPU_SHA and cpu_spec["log_sha256"] == CPU_LOG_SHA
    cpu = read_unit(cpu_spec, 120, False)
    cpu_code = queue_cpu.closure(root, CPU_CODE)
    driver.validate_cpu(cpu, CPU_CODE, previous, cpu_code)
    assert cpu["source_checkpoint_sha256"] == late.SOURCE_SHA
    startup = read_unit(spec["startup"], 120, False)
    validate_startup(startup, spec)
    mechanics = read_unit(spec["mechanics"], 120, True)
    arguments = SimpleNamespace(execution_sha256=TRAIN_CODE, cpu_sha256=CPU_SHA,
        cpu_log_sha256=CPU_LOG_SHA, cpu_execution_sha256=CPU_CODE, startup_sha256=spec["startup"]["receipt_sha256"])
    driver.validate_mechanics(mechanics, arguments, cpu)
    assert [training.diagnostic(r) for r in mechanics["resumed_steps"]] == [training.diagnostic(r) for r in mechanics["steps"][8:]]
    selected = qualification.archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, "helpers", lambda r, _: helpers(r, code)):
        control, source, prior, proof, frozen, _ = qualification.startup(root, ORIGINAL114)
    pair.executing_authority(root, code)
    values = old.initializers(proof, "half")
    target = values["target"].tolist()
    assert len(target) == 13283 and len(set(target)) == 2004
    names = sorted({r["product"] for r in frozen["fit_manifest"]})
    assert [names[i] for i in target] == [r["product"] for r in frozen["fit_manifest"]]
    positive = pair.smoke.member_bank_positive_ordinals(np.asarray(target), allow_singletons=True)
    target_sha = old.fingerprint({"target": values["target"], "positive": positive})
    del values
    schedules = {}
    for seed in SEEDS:
        original = old.coverage.schedule(target, seed=seed).tolist()
        queue = queue_cpu.schedules(target, original, seed)
        for arm, batches in (("control", original), ("queue", queue)):
            expected_schedule = cpu["schedules"][str(seed)]
            assert old.fingerprint({"batches": torch.tensor(batches)}) == expected_schedule[arm + "_schedule_sha256"]
            assert queue_cpu.sha_bytes([[target[i] for i in b] for b in batches]) == expected_schedule["class_sequence_sha256"]
            schedules[seed, arm] = batches
    validate_image_ids(mechanics, schedules[179032, "queue"], target)
    # Pin common roles/runtime and the frozen12 prefix to the actual TRAIN1000 source.
    assert sha(qualification.RESUME) == late.SOURCE_SHA
    original = torch.load(qualification.RESUME, map_location="cpu", weights_only=True, mmap=True)
    source_identity = canonical(original["identity"])
    source_frozen_names = sorted(frozen_saved(original, 12))
    source_frozen = old.fingerprint(frozen_saved(original, 12))
    del original; gc.collect()
    assert [(e["seed"], e["arm"]) for e in spec["endpoints"]] == list(ORDER)
    endpoints = {}
    for endpoint in spec["endpoints"]:
        assert Path(endpoint["receipt"]) == Path(endpoint["run"]) / "receipt.json"
        value = read_unit(endpoint, 300, True)
        validate_endpoint(value, endpoint, cpu, spec)
        log = training.read_authority(Path(endpoint["log"]), endpoint["log_sha256"], log=True)
        validate_logged_steps(log, value["steps"])
        base = value["resume_identity"]
        assert base["target_positive_sha256"] == target_sha and base["frozen_sha256"] == source_frozen
        assert base["frozen_names"] == source_frozen_names
        assert all(base[k] == source_identity[k] for k in ("parameter_names", "model_roles", "buffers_sha256", "runtime"))
        validate_image_ids(value, schedules[endpoint["seed"], endpoint["arm"]], target)
        if (endpoint["seed"], endpoint["arm"]) == (179032, "queue"):
            assert base == mechanics["resume_identity"]
            assert [training.diagnostic(r) for r in value["steps"][:17]] == [training.diagnostic(r) for r in mechanics["steps"]]
        saved = checkpoint(endpoint, value)
        del saved; gc.collect()
        rss_checkpoint(f"endpoint-{endpoint['seed']}-{endpoint['arm']}")
        endpoints[endpoint["seed"], endpoint["arm"]] = value
    validate_distinct(spec["endpoints"] + [spec[k] for k in ("cpu", "startup", "mechanics")])
    costs = paired_cost(endpoints)
    rss_checkpoint("source-authority")
    return spec, code, endpoints, costs, control, source, prior, proof, frozen


def validate_distinct(units):
    assert len({e["invocation_id"] for e in units}) == len(units)
    assert len({str(Path(e["receipt"]).resolve()) for e in units}) == len(units)
    runs = [str(Path(e["run"]).resolve()) for e in units if "run" in e]
    assert len(set(runs)) == len(runs)


def model_pair(control, saved, state=None):
    """Actual source loader and independent config constructor; complete buffers."""
    with torch.random.fork_rng(devices=[]):
        if state is None:
            model, processor = pair.smoke.load_arm(control, "large")
        else:
            model, processor = state["model"], state["processor"]
        clone = type(model)(copy.deepcopy(model.config)).float()
        heads = [nn.Linear(1024, 128) if state is None else state["head"], nn.Linear(1024, 128)]
    assert len(saved["vision"]) == 400
    for vision, head in zip((model, clone), heads, strict=True):
        vision.load_state_dict(saved["vision"], strict=True)
        head.load_state_dict(saved["head"], strict=True)
        buffers = dict(vision.named_buffers())
        assert buffers.keys() == saved["buffers"].keys()
        with torch.no_grad():
            for n, v in buffers.items():
                v.copy_(saved["buffers"][n])
        vision.eval().requires_grad_(False); head.eval().requires_grad_(False)
        assert old.fingerprint(vision.state_dict()) == old.fingerprint(saved["vision"])
        assert old.fingerprint(head.state_dict()) == old.fingerprint(saved["head"])
        assert old.fingerprint(buffers) == old.fingerprint(saved["buffers"])
    return (model, clone), heads, processor


def features(model, head, pixels):
    return F.normalize(pair.smoke.compact_head_features(model(pixel_values=pixels).pooler_output, head), dim=1)


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
    p.add_argument("--authority-sha256", required=True)
    p.add_argument("--phase", choices=("cpu", "export"), required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p.add_argument("--arm", choices=("control", "queue"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-proof", type=Path)
    p.add_argument("--cpu-sha256")
    p.add_argument("--cpu-log", type=Path)
    p.add_argument("--cpu-log-sha256")
    args = p.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    assert not args.output.resolve().is_relative_to(TRAIN_ROOT.resolve())
    gpu = args.phase == "export"
    assert torch.cuda.is_available() == gpu
    if gpu:
        assert torch.cuda.device_count() == 1 and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    else:
        assert not any((args.cpu_proof, args.cpu_sha256, args.cpu_log, args.cpu_log_sha256))
    resource_unit(300 if gpu else 120, gpu)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parent
    spec, code, endpoints, _, control, source, prior, proof, frozen = authority(root, args.authority_sha256)
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    rss_checkpoint("all-authority")
    endpoint = next(e for e in spec["endpoints"] if (e["seed"], e["arm"]) == (args.seed, args.arm))
    value = endpoints[args.seed, args.arm]
    binding = {"intervention": driver.METHOD, "authority_sha256": args.authority_sha256, "execution_sha256": spec["execution_sha256"],
        "seed": args.seed, "arm": args.arm, "boundary": value["boundary"],
        "training_receipt_sha256": endpoint["receipt_sha256"], "checkpoint_sha256": endpoint["checkpoint_sha256"],
        "terminal_state_fingerprint": endpoint["terminal_state_fingerprint"]}
    cpu_rng = torch.random.get_rng_state().clone()
    if not gpu:
        with torch.random.fork_rng(devices=[]):
            state, initial = qualification.fresh(control, source, proof, value["boundary"], "cpu")
        rss_checkpoint("fresh-initializer")
        target = state["target"].cpu().tolist()
        batches = old.coverage.schedule(target, seed=args.seed).tolist()
        if args.arm == "queue":
            batches = queue_cpu.schedules(target, batches, args.seed)
        schedule_sha = old.fingerprint({"batches": torch.tensor(batches)})
        arguments = SimpleNamespace(seed=args.seed, boundary=value["boundary"], execution_sha256=TRAIN_CODE,
            cpu_sha256=CPU_SHA, cpu_log_sha256=spec["cpu"]["log_sha256"], cpu_execution_sha256=CPU_CODE)
        base = training.identity(state, arguments, initial, schedule_sha)
        base.update(schema="image-queue-complete-resume-v1", intervention=driver.METHOD, sampling_arm=args.arm,
            class_sequence_sha256=queue_cpu.sha_bytes([[target[i] for i in b] for b in batches]))
        expected = {**base, "runtime": late.source_runtime(base["runtime"])}
        assert canonical(expected) == value["resume_identity"]
        # Do not keep a complete updated mmap resident during source construction.
        saved = checkpoint(endpoint, value)
        rss_checkpoint("selected-complete-state")
        training.validate_resume(saved, expected, 100, state["optimizer"].state_dict()["param_groups"])
        state["model"].load_state_dict(saved["vision"], strict=True)
        state["head"].load_state_dict(saved["head"], strict=True)
        training.verify(state, base)
        assert old.fingerprint(frozen_saved(saved, value["boundary"])) == base["frozen_sha256"]
        models, heads, processor = model_pair(control, saved, state)
        del state; gc.collect()
    else:
        saved = checkpoint(endpoint, value)
        models, heads, processor = model_pair(control, saved)
    del saved; gc.collect()
    rss_checkpoint("model-pair")
    assert all(canonical(late.source_runtime(old.coverage.trained.base.runtime_identity(m))) == value["resume_identity"]["runtime"]
               for m in models)
    assert all(canonical(old.coverage.trained.native.environment(m, processor)) == source["environment"] for m in models)
    whole = pair.smoke.digest(old.coverage.trained.base.whole_state(models[0]))
    head_sha = pair.smoke.digest(heads[0].state_dict())
    assert all(pair.smoke.digest(old.coverage.trained.base.whole_state(m)) == whole for m in models)
    images, _ = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], (0, 1), None)
    pixels = pair.pixels(processor, images, "large")
    pixel_sha = pair.smoke.digest({"pixels": pixels})
    if gpu:
        cpu = training.read_authority(args.cpu_proof, args.cpu_sha256)
        driver.validate_log(training.read_authority(args.cpu_log, args.cpu_log_sha256, log=True), cpu, 120)
        driver.resources(cpu, 120, False)
        assert all(cpu[k] == v for k, v in binding.items())
        assert cpu["pass"] and cpu["source_code"] == code and cpu["quality_read"] is False
        assert cpu["strict400_whole_head_fit_packed_reload_exact"] and cpu["frozen_roles_complete_state_exact"]
        assert cpu["cpu_rng_preserved"]
        assert cpu["whole_sha256"] == whole and cpu["head_sha256"] == head_sha and cpu["fit_pixels_sha256"] == pixel_sha
        assert cpu["optimizer_updates"] == 0
    else:
        with torch.inference_mode():
            a, b = (features(m, h, pixels) for m, h in zip(models, heads, strict=True))
            assert torch.equal(a, b); old.previous.training.packed_equal(a, b)
    rss_checkpoint("fit-forward")
    for model in models:
        model.half()
    f16whole = pair.smoke.digest(old.coverage.trained.base.whole_state(models[0]))
    assert all(pair.smoke.digest(old.coverage.trained.base.whole_state(m)) == f16whole for m in models)
    rss_checkpoint("half-models")
    if not gpu:
        assert torch.equal(cpu_rng, torch.random.get_rng_state())
        assert all(sha(root / n) == h for n, h in code.items())
        assert sha(Path(endpoint["run"]) / "resume.pt") == binding["checkpoint_sha256"]
        args.output.mkdir()
        publish(args.output / "proof.json", {**binding, **resource_unit(120, False), "pass": True,
            "source_code": code, "whole_sha256": whole, "f16_whole_sha256": f16whole, "head_sha256": head_sha,
            "fit_pixels_sha256": pixel_sha, "strict400_whole_head_fit_packed_reload_exact": True,
            "frozen_roles_complete_state_exact": True, "cpu_rng_preserved": True,
            "optimizer_updates": 0, "quality_read": False})
        return
    assert cpu["f16_whole_sha256"] == f16whole
    for model, head in zip(models, heads, strict=True):
        model.cuda(); head.cuda()
        assert all(p.dtype == torch.float16 for p in model.parameters())
        assert all(p.dtype == torch.float32 for p in head.parameters())
    rng = old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    values = torch.empty((12599, 128), dtype=torch.float32)
    with torch.inference_mode():
        fit = pixels.half().cuda()
        a, b = (features(m, h, fit) for m, h in zip(models, heads, strict=True))
        assert torch.equal(a, b); old.previous.training.packed_equal(a, b)
        # First held image read follows frozen authority and native fit reload.
        # Preserve archived native query/gallery B32 grouping, including tails.
        for role in ("query", "gallery"):
            for start in range(0, len(frozen[role]), 32):
                indices = frozen[role][start:start + 32]
                rows = [frozen["held_manifest"][i] for i in indices]
                images, _ = pair.augmented_images(control.dataset_root, rows, tuple(range(len(rows))), None)
                pixels = pair.pixels(processor, images, "large").half().cuda()
                a, b = (features(m, h, pixels) for m, h in zip(models, heads, strict=True))
                assert a.shape == (len(rows), 128) and torch.isfinite(a).all() and torch.equal(a, b)
                old.previous.training.packed_equal(a, b)
                values[indices] = a.cpu()
    torch.cuda.synchronize()
    assert rng == old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    assert all(pair.smoke.digest(old.coverage.trained.base.whole_state(m)) == f16whole for m in models)
    assert all(pair.smoke.digest(h.state_dict()) == head_sha for h in heads)
    assert all(p.grad is None for m in (*models, *heads) for p in m.parameters())
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    assert all(canonical(old.coverage.trained.native.environment(m, processor)) == source["environment"] for m in models)
    assert all(sha(root / n) == h for n, h in code.items())
    assert sha(Path(endpoint["run"]) / "resume.pt") == binding["checkpoint_sha256"]
    packed = pack_int8_unit_embeddings(values)
    args.output.mkdir()
    hashes = {}
    for name, tensor in (("held.npy", values), ("held.codes.npy", packed.codes), ("held.inverse.npy", packed.inverse_norms)):
        path = args.output / name
        training.atomic_write(path, lambda s, tensor=tensor: np.save(s, tensor.numpy(), allow_pickle=False))
        hashes[name] = sha(path)
    publish(args.output / "receipt.json", {**binding, **resource_unit(300, True), "pass": True,
        "source_code": code, "cpu_authority_sha256": args.cpu_sha256, "cpu_proof": str(args.cpu_proof),
        "cpu_log_sha256": args.cpu_log_sha256,
        "whole_sha256": whole, "f16_whole_sha256": f16whole, "head_sha256": head_sha,
        "files": hashes, "held_manifest": frozen["held_manifest"], "query": frozen["query"], "gallery": frozen["gallery"],
        "full_held_independent_whole_head_packed_exact": True, "source_head_rng_flags_preserved": True,
        "batch": 32, "width": 128, "precision": PRECISION, "optimizer_updates": 0,
        "quality_read": False, "official_read": False, "claim_eligible": False,
        "public_serving_qualified": False, "public_latency_measured": False})


if __name__ == "__main__":
    main()
