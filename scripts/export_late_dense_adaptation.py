#!/usr/bin/env python3
"""Parent-frozen fresh late100 endpoints: fit-only CPU proof, native FP16 held.

Freeze late-dense-held-execution.json as source-v4's exact 114 members plus
export_late_dense_adaptation.py, score_late_dense_adaptation.py and
test_late_dense_held.py. Never extend source-v4 in place. Parent reviews and
pins late-dense-held-authority.json before invoking this script.

Authority fields: schema="late-dense-held-authority-v1", execution_sha256,
cpu, mechanics, endpoints. Each entry has receipt/receipt_sha256,
log/log_sha256, unit/invocation_id, service_seconds, host_max_rss_kib,
host_swap_kib. Endpoints also have seed, arm, run, checkpoint_sha256,
terminal_state_fingerprint; order control032,candidate032,candidate041,control041.
All hashes of future endpoints are supplied by the parent, never guessed here.
"""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import time
UNIT_STARTED = time.perf_counter()

import argparse
import copy
import gc
import json
import math
import os
import resource
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_late_dense_adaptation as training
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

qualification, late, old, pair = training.qualification, training.late, training.old, training.pair
sha = pair.sha
TRAIN_ROOT = Path("/home/riomus/runs/sfora-late-dense-source-v4")
TRAIN_CODE = "c9cec7b71d0c2ba90de2eb0d5c9810f111d4e7feb88f10e63055fbabf8a6d8b3"
CPU_CODE = "e5aac1bd966873100bf41fe53866c91942dd18e535739d6782c930972bc6c434"
CPU_SHA = "cb9d97b57bd2ed4e914fc97944091bb1fa7bead12274ea3f3eee0046f268de0e"
MECHANICS_SHA = "1423ee7cfdb65893268e91c7a8ae54baac257a84ef6dd127803d4aa577b5ba3a"
MECHANICS_LOG_SHA = "2919a90acef1f508317becf511388dcc62f29104fd2869b952fc96385ba2b6d4"
ADDED = {"export_late_dense_adaptation.py", "score_late_dense_adaptation.py", "test_late_dense_held.py"}
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
    assert len(previous) == 114 and len(code) == 117
    assert set(code) - set(previous) == ADDED
    assert all(code[n] == h for n, h in previous.items()), "qualified prefix changed"


def validate_unit(spec, cap, receipt=None):
    assert spec["unit"] and spec["invocation_id"]
    assert math.isfinite(spec["service_seconds"]) and 0 < spec["service_seconds"] < cap
    assert 0 < spec["host_max_rss_kib"] <= 8 * 1024 * 1024 and spec["host_swap_kib"] == 0
    if receipt is not None:
        assert receipt["unit_invocation_id"] == spec["invocation_id"]
        assert Path(receipt["unit_cgroup"]).name == spec["unit"] + ".service"
        assert receipt["unit_memory_max_bytes"] == 8 * 1024**3 and receipt["unit_memory_swap_max_bytes"] == 0
        assert receipt["host_max_rss_kib"] == spec["host_max_rss_kib"] and receipt["host_swap_kib"] == 0
        assert math.isfinite(receipt["total_seconds"]) and 0 < receipt["total_seconds"] < cap
        assert 0 < receipt["peak_cuda_allocated_bytes"] < 10_000_000_000


def validate_endpoint(value, spec, cpu, authority):
    seed, arm = spec["seed"], spec["arm"]
    assert seed in SEEDS and arm in ("control", "candidate")
    boundary = 12 if arm == "control" else 10
    assert value["pass"] and value["intervention"] == training.METHOD and value["phase"] == "train"
    assert value["seed"] == seed and value["boundary"] == boundary
    assert value["updates"] == value["completed_step"] == 100
    assert value["quality_read"] is False and value["training_state_discarded"] is False
    assert value["frozen_named_state_buffers_rng_preserved"]
    assert value["source_checkpoint_sha256"] == late.SOURCE_SHA and value["execution_sha256"] == TRAIN_CODE
    assert value["cpu_authority_sha256"] == authority["cpu"]["receipt_sha256"] == CPU_SHA
    assert value["cpu_execution_sha256"] == CPU_CODE and value["cpu_log_sha256"] == authority["cpu"]["log_sha256"]
    assert value["mechanics_sha256"] == authority["mechanics"]["receipt_sha256"] == MECHANICS_SHA
    assert value["mechanics_log_sha256"] == authority["mechanics"]["log_sha256"] == MECHANICS_LOG_SHA
    assert value["initial_state_sha256"] == cpu["initial_state_sha256"]
    assert value["checkpoint_sha256"] == spec["checkpoint_sha256"]
    assert value["terminal_state_fingerprint"] == spec["terminal_state_fingerprint"]
    assert value["new32_weights_changed"] == (boundary == 10)
    assert len(value["new_parameter_changes"]) == 32
    assert all(v == (boundary == 10) for v in value["new_parameter_changes"].values())
    base = value["resume_identity"]
    expected = {"schema": "late-dense-complete-resume-v1", "intervention": training.METHOD,
        "arm": "half", "width": 128, "total_updates": 100, "seed": seed, "boundary": boundary,
        "augmentation_offset": 1000, "execution_sha256": TRAIN_CODE,
        "source_checkpoint_sha256": late.SOURCE_SHA, "source_execution_sha256": late.SOURCE_CODE,
        "source_schedule_sha256": late.SOURCE_SCHEDULE, "cpu_authority_sha256": CPU_SHA,
        "cpu_execution_sha256": CPU_CODE, "cpu_log_sha256": authority["cpu"]["log_sha256"],
        "initial_state_sha256": cpu["initial_state_sha256"], "schedule_sha256": value["schedule_sha256"],
        "precision": "native_float32_fp16_autocast", "batch": 64, "micro": 16}
    assert all(base[k] == v for k, v in expected.items()), "foreign late resume identity"
    assert len(base["parameter_names"]) == (208 if boundary == 12 else 240)
    assert len(base["parameter_names"]) == len(set(base["parameter_names"]))
    assert len(value["steps"]) == 100
    for step, row in enumerate(value["steps"], 1):
        assert row["step"] == row["optimizer_counter"] == step and row["augmentation_step"] == 1000 + step
        assert row["rgb_sha256"] and row["pixels_sha256"]
        assert math.isfinite(row["seconds"]) and row["seconds"] > 0
    assert math.isfinite(value["training_wall_seconds"]) and value["training_wall_seconds"] > 0
    assert value["median_step_3_end_seconds"] == statistics.median(r["seconds"] for r in value["steps"][2:])
    validate_unit(spec, 300, value)


def paired_cost(endpoints):
    assert set(endpoints) == set(ORDER)
    costs = {}
    for seed in SEEDS:
        control, candidate = (endpoints[seed, arm] for arm in ("control", "candidate"))
        for k in ("schedule_sha256", "initial_state_sha256", "source_checkpoint_sha256"):
            assert candidate[k] == control[k], "same-pair source/schedule differs"
        for k in ("target_positive_sha256", "buffers_sha256", "runtime"):
            assert candidate["resume_identity"][k] == control["resume_identity"][k]
        assert all(a[k] == b[k] for a, b in zip(control["steps"], candidate["steps"], strict=True)
                   for k in ("step", "optimizer_counter", "augmentation_step", "rgb_sha256", "pixels_sha256", "rank_active_before"))
        wall = candidate["training_wall_seconds"] / control["training_wall_seconds"]
        median = candidate["median_step_3_end_seconds"] / control["median_step_3_end_seconds"]
        assert math.isfinite(wall) and math.isfinite(median) and 0 < wall <= 1.50 and 0 < median <= 1.50
        costs[seed] = {"training_wall_ratio": wall, "median_step_ratio": median,
                       "control_wall_seconds": control["training_wall_seconds"],
                       "candidate_wall_seconds": candidate["training_wall_seconds"]}
    return costs


def frozen_saved(saved, boundary):
    roots = ("embeddings.",) + tuple(f"encoder.layers.{i}." for i in range(boundary))
    return {n: v for group in (saved["vision"], saved["buffers"]) for n, v in group.items() if n.startswith(roots)}


def checkpoint(spec, value):
    path = Path(spec["run"]) / "resume.pt"
    assert sha(path) == spec["checkpoint_sha256"] == value["checkpoint_sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert set(saved) == late.RESUME_KEYS, "complete late resume required"
    assert canonical(saved["identity"]) == {**value["resume_identity"], "global_step": 100}
    # JSON receipts stringify config mapping keys and turn role tuples into lists.
    # Authenticate their canonical form, retain native types for the strict guard.
    base = {k: v for k, v in saved["identity"].items() if k != "global_step"}
    training.validate_resume(saved, base, 100, saved["optimizer"]["param_groups"])
    assert old.fingerprint(saved) == spec["terminal_state_fingerprint"] == value["terminal_state_fingerprint"]
    assert saved["classifier"].shape == (2004, 128) and saved["bank"].shape == (13283, 128)
    assert old.fingerprint(saved["buffers"]) == base["buffers_sha256"]
    assert old.fingerprint(frozen_saved(saved, base["boundary"])) == base["frozen_sha256"]
    return saved


def authority(root, expected):
    spec = training.read_authority(root / "late-dense-held-authority.json", expected)
    assert spec["schema"] == "late-dense-held-authority-v1"
    assert root.resolve() != TRAIN_ROOT.resolve(), "immutable training source cannot be extended in place"
    code = training.read_authority(root / "late-dense-held-execution.json", spec["execution_sha256"])
    previous = training.read_authority(root / "late-dense-execution.json", TRAIN_CODE)
    validate_closure(code, previous)
    assert previous == training.read_authority(TRAIN_ROOT / "late-dense-execution.json", TRAIN_CODE)
    assert all(sha(root / n) == h and sha(TRAIN_ROOT / n) == h for n, h in previous.items())
    assert all(sha(root / n) == h for n, h in code.items())
    cpu_spec, mechanics_spec = spec["cpu"], spec["mechanics"]
    assert cpu_spec["receipt_sha256"] == CPU_SHA
    assert mechanics_spec["receipt_sha256"] == MECHANICS_SHA and mechanics_spec["log_sha256"] == MECHANICS_LOG_SHA
    cpu = training.read_authority(Path(cpu_spec["receipt"]), CPU_SHA)
    training.validate_cpu(cpu, CPU_CODE, previous)
    assert len(cpu["code"]) == 112
    assert training.read_authority(qualification.SOURCE_ROOT / "late-dense-execution.json", CPU_CODE) == cpu["code"]
    validate_unit(cpu_spec, 120)
    mechanics = training.read_authority(Path(mechanics_spec["receipt"]), MECHANICS_SHA)
    training.validate_mechanics(mechanics, TRAIN_CODE, CPU_SHA, cpu_spec["log_sha256"], CPU_CODE)
    assert mechanics["initial_state_sha256"] == cpu["initial_state_sha256"]
    validate_unit(mechanics_spec, 120, mechanics)
    for unit in (cpu_spec, mechanics_spec):
        training.read_authority(Path(unit["log"]), unit["log_sha256"], log=True)
    assert [(e["seed"], e["arm"]) for e in spec["endpoints"]] == list(ORDER)
    endpoints = {}
    for endpoint in spec["endpoints"]:
        assert Path(endpoint["receipt"]) == Path(endpoint["run"]) / "receipt.json"
        value = training.read_authority(Path(endpoint["receipt"]), endpoint["receipt_sha256"])
        training.read_authority(Path(endpoint["log"]), endpoint["log_sha256"], log=True)
        validate_endpoint(value, endpoint, cpu, spec)
        saved = checkpoint(endpoint, value)
        del saved
        rss_checkpoint(f"endpoint-{endpoint['seed']}-{endpoint['arm']}")
        endpoints[endpoint["seed"], endpoint["arm"]] = value
    assert len({e["run"] for e in spec["endpoints"]}) == 4
    assert len({e["invocation_id"] for e in spec["endpoints"]}) == 4
    costs = paired_cost(endpoints)
    selected = qualification.archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, "helpers", lambda r, _: helpers(r, code)):
        control, source, prior, proof, frozen, _ = qualification.startup(root, TRAIN_CODE)
    rss_checkpoint("source-authority")
    pair.executing_authority(root, code)
    return spec, code, endpoints, costs, control, source, prior, proof, frozen


def model_pair(control, saved, state=None):
    """Actual source loader and independent config constructor; complete buffers."""
    with torch.random.fork_rng(devices=[]):
        if state is None:
            model, processor = pair.smoke.load_arm(control, "large")
        else:
            model, processor = state["model"], state["processor"]
        clone = type(model)(copy.deepcopy(model.config)).float()
        heads = [nn.Linear(1024, 128) if state is None else state["head"], nn.Linear(1024, 128)]
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
    p.add_argument("--arm", choices=("control", "candidate"), required=True)
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
    binding = {"authority_sha256": args.authority_sha256, "execution_sha256": spec["execution_sha256"],
        "seed": args.seed, "arm": args.arm, "boundary": value["boundary"],
        "training_receipt_sha256": endpoint["receipt_sha256"], "checkpoint_sha256": endpoint["checkpoint_sha256"],
        "terminal_state_fingerprint": endpoint["terminal_state_fingerprint"]}
    cpu_rng = torch.random.get_rng_state().clone()
    if not gpu:
        with torch.random.fork_rng(devices=[]):
            state, initial = qualification.fresh(control, source, proof, value["boundary"], "cpu")
        rss_checkpoint("fresh-initializer")
        batches = old.coverage.schedule(state["target"].cpu().numpy(), seed=args.seed)
        schedule_sha = pair.smoke.digest({"batches": torch.from_numpy(batches)})
        arguments = SimpleNamespace(seed=args.seed, boundary=value["boundary"], execution_sha256=TRAIN_CODE,
            cpu_sha256=CPU_SHA, cpu_log_sha256=spec["cpu"]["log_sha256"], cpu_execution_sha256=CPU_CODE)
        base = training.identity(state, arguments, initial, schedule_sha)
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
        training.read_authority(args.cpu_log, args.cpu_log_sha256, log=True)
        assert all(cpu[k] == v for k, v in binding.items())
        assert cpu["pass"] and cpu["source_code"] == code and cpu["quality_read"] is False
        assert cpu["strict400_whole_head_fit_packed_reload_exact"] and cpu["frozen_roles_complete_state_exact"]
        assert cpu["cpu_rng_preserved"]
        assert cpu["whole_sha256"] == whole and cpu["head_sha256"] == head_sha and cpu["fit_pixels_sha256"] == pixel_sha
        assert cpu["optimizer_updates"] == 0
        assert math.isfinite(cpu["total_seconds"]) and 0 < cpu["total_seconds"] < 120
        assert 0 < cpu["host_max_rss_kib"] <= 8 * 1024 * 1024 and cpu["host_swap_kib"] == 0
        assert cpu["unit_invocation_id"] and cpu["unit_memory_max_bytes"] == 8 * 1024**3
        assert cpu["unit_memory_swap_max_bytes"] == 0 and cpu["peak_cuda_allocated_bytes"] == 0
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
    del saved; gc.collect()
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
