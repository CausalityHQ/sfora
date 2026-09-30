#!/usr/bin/env python3
"""One discarded candidate17 mechanics unit or one fresh late-stage TRAIN100 arm."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import time

UNIT_STARTED = time.perf_counter()

import argparse
import copy
import gc
import json
import os
import resource
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import qualify_late_dense_adaptation_cpu as qualification
import late_dense_boundary as late
import train_pe_teacher_retained256 as objective

old = qualification.old
pair = old.pair
METHOD = "late-dense-adaptation-v1"


def counters(step):
    assert type(step) is int and 1 <= step <= 100
    return step, 1000 + step


def diagnostic(row):
    return {k: v for k, v in row.items() if k != "seconds"}


def validate_cpu(receipt, execution, code):
    assert receipt["pass"] and receipt["execution_sha256"] == execution
    original = receipt["code"]
    assert set(code) - set(original) == {"train_late_dense_adaptation.py", "test_late_dense_adaptation.py"}
    assert all(code[n] == h for n, h in original.items()), "qualified CPU closure changed"
    assert receipt["source_checkpoint_sha256"] == late.SOURCE_SHA
    assert all(receipt[k] for k in (
        "matched_control_candidate_initial_state", "fit_held_identity_and_path_disjoint",
        "native400_whole_head_packed_reload_exact", "changed_driver_rejected", "RNG_preserved",
        "missing_duplicate_optimizer_and_frozen_buffer_mutation_rejected"))
    assert receipt["optimizer_members"] == {"12": 208, "10": 240}
    assert receipt["optimizer_updates"] == 0 and receipt["quality_read"] is False


def validate_mechanics(receipt, execution, cpu_sha, cpu_log_sha, cpu_execution):
    assert receipt["pass"] and receipt["intervention"] == METHOD
    assert receipt["execution_sha256"] == execution and receipt["cpu_authority_sha256"] == cpu_sha
    assert receipt["cpu_log_sha256"] == cpu_log_sha
    assert receipt["cpu_execution_sha256"] == cpu_execution
    assert receipt["phase"] == "mechanics" and receipt["boundary"] == 10 and receipt["seed"] == 179032
    assert receipt["source_checkpoint_sha256"] == late.SOURCE_SHA
    assert receipt["updates"] == receipt["completed_step"] == 17
    assert all(receipt[k] for k in ("training_state_discarded", "native_17_equals_serialized8_plus9_exact",
        "strict400_reload_whole_head_packed_exact", "new32_weights_changed",
        "frozen_named_state_buffers_rng_preserved"))
    assert receipt["quality_read"] is False and receipt["checkpoint_sha256"] is None
    assert receipt["chunk100_admission_seconds"] <= 269 and receipt["total_seconds"] < 120
    assert receipt["peak_cuda_allocated_bytes"] < 10_000_000_000
    assert 0 < receipt["host_max_rss_kib"] <= 8 * 1024 * 1024
    assert receipt["host_swap_kib"] == 0
    assert receipt["unit_memory_max_bytes"] == 8 * 1024**3
    assert receipt["unit_memory_swap_max_bytes"] == 0 and receipt["unit_invocation_id"]
    assert len(receipt["steps"]) == 17
    for step, row in enumerate(receipt["steps"], 1):
        counter, augmentation = counters(step)
        assert row["step"] == row["optimizer_counter"] == counter
        assert row["augmentation_step"] == augmentation


def validate_resume(saved, base, expected_step, groups):
    assert set(saved) == late.RESUME_KEYS, "complete adaptation state required"
    assert type(expected_step) is int and 0 < expected_step <= 100
    assert saved["identity"] == {**base, "global_step": expected_step}, "foreign resume identity/counter"
    assert len(saved["vision"]) == 400
    optimizer = saved["optimizer"]
    assert optimizer["param_groups"] == groups, "saved optimizer order/options differ"
    ids = [p for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(base["parameter_names"])
    assert set(optimizer["state"]) == set(ids), "missing optimizer members"
    assert all(set(v) == {"step", "exp_avg", "exp_avg_sq"} and float(v["step"]) == expected_step
               for v in optimizer["state"].values()), "optimizer counter/state differs"
    assert saved["scaler"] is not None and len(saved["cuda_rng"]) == 1


def atomic_write(path, writer):
    """Publish completed bytes with an exclusive hard link; never replace a destination."""
    temporary = path.with_name(path.name + ".part")
    assert not path.exists() and not path.is_symlink()
    with temporary.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def read_authority(path, expected, log=False):
    assert path and expected and pair.sha(path) == expected, "authority file digest differs"
    value = path.read_text()
    if log:
        assert all(s in value for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
        return value
    return json.loads(value)


def identity(state, args, initial, schedule_sha):
    names = {id(p): n for n, p in state["params"]}
    return {"schema": "late-dense-complete-resume-v1", "intervention": METHOD,
        "arm": "half", "width": 128, "total_updates": 100, "seed": args.seed,
        "boundary": args.boundary, "augmentation_offset": 1000,
        "execution_sha256": args.execution_sha256, "cpu_authority_sha256": args.cpu_sha256,
        "cpu_log_sha256": args.cpu_log_sha256, "cpu_execution_sha256": args.cpu_execution_sha256,
        "source_checkpoint_sha256": late.SOURCE_SHA,
        "source_execution_sha256": late.SOURCE_CODE, "source_schedule_sha256": late.SOURCE_SCHEDULE,
        "initial_state_sha256": initial, "schedule_sha256": schedule_sha,
        "target_positive_sha256": old.fingerprint({k: state[k] for k in ("target", "positive")}),
        "parameter_names": [n for n, _ in state["params"]],
        "model_roles": [(n, p.requires_grad) for n, p in state["model"].named_parameters()],
        "runtime": old.coverage.trained.base.runtime_identity(state["model"]),
        "buffers_sha256": old.fingerprint(dict(state["model"].named_buffers())),
        "frozen_names": sorted(late.frozen_state(state["model"], args.boundary)),
        "frozen_sha256": old.fingerprint(late.frozen_state(state["model"], args.boundary)),
        "optimizer_groups": [{"parameter_names": [names[id(p)] for p in g["params"]],
                              "options": {k: v for k, v in g.items() if k != "params"}}
                             for g in state["optimizer"].param_groups],
        "precision": "native_float32_fp16_autocast", "batch": 64, "micro": 16}


def verify(state, base):
    model = state["model"]
    roots = ("embeddings.",) + tuple(f"encoder.layers.{i}." for i in range(base["boundary"]))
    assert all(p.requires_grad == (not n.startswith(roots)) for n, p in model.named_parameters())
    expected = old.coverage.parameters(model, state["head"], state["classifier"])
    assert [(n, id(p)) for n, p in state["params"]] == [(n, id(p)) for n, p in expected]
    assert [n for n, _ in expected] == base["parameter_names"]
    assert len(expected) == (240 if base["boundary"] == 10 else 208)
    ids = [id(p) for g in state["optimizer"].param_groups for p in g["params"]]
    assert ids == [id(p) for _, p in expected] and len(ids) == len(set(ids))
    names = {id(p): n for n, p in expected}
    assert [{"parameter_names": [names[id(p)] for p in g["params"]],
             "options": {k: v for k, v in g.items() if k != "params"}}
            for g in state["optimizer"].param_groups] == base["optimizer_groups"]
    frozen = late.frozen_state(model, base["boundary"])
    assert sorted(frozen) == base["frozen_names"] and old.fingerprint(frozen) == base["frozen_sha256"]
    assert old.fingerprint(dict(model.named_buffers())) == base["buffers_sha256"]
    assert old.coverage.trained.base.runtime_identity(model) == base["runtime"]
    assert [(n, p.requires_grad) for n, p in model.named_parameters()] == base["model_roles"]
    assert old.fingerprint({k: state[k] for k in ("target", "positive")}) == base["target_positive_sha256"]
    assert all(p.dtype == torch.float32 for p in (*model.parameters(), *state["head"].parameters(), state["classifier"]))


def save(state, base, path):
    state["optimizer"].zero_grad(set_to_none=True)
    verify(state, base)
    value = old.payload(state, base)
    validate_resume(value, base, state["counter"], state["optimizer"].state_dict()["param_groups"])
    fingerprint = old.fingerprint(value)
    atomic_write(path, lambda stream: torch.save(value, stream))
    disk = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    assert old.fingerprint(disk) == fingerprint, "serialized complete state differs"
    return pair.sha(path), fingerprint


def restore(state, base, path, sha, fingerprint, step):
    verify(state, base)
    assert pair.sha(path) == sha, "resume checkpoint digest differs"
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    validate_resume(saved, base, step, state["optimizer"].state_dict()["param_groups"])
    assert old.fingerprint(saved) == fingerprint, "resume complete state digest differs"
    assert saved["buffers"].keys() == dict(state["model"].named_buffers()).keys()
    assert old.fingerprint(saved["buffers"]) == base["buffers_sha256"]
    assert saved["classifier"].shape == state["classifier"].shape == (2004, 128)
    assert saved["bank"].shape == state["bank"].shape == (13283, 128)
    state["model"].load_state_dict(saved["vision"], strict=True)
    state["head"].load_state_dict(saved["head"], strict=True)
    with torch.no_grad():
        state["classifier"].copy_(saved["classifier"])
        state["bank"].copy_(saved["bank"])
        for n, value in state["model"].named_buffers():
            value.copy_(saved["buffers"][n])
    state["optimizer"].load_state_dict(saved["optimizer"])
    state["scaler"].load_state_dict(saved["scaler"])
    torch.random.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    state["counter"] = step
    verify(state, base)
    assert old.fingerprint(old.payload(state, base)) == fingerprint, "restored complete state differs"


def new_weights(model):
    return {n: p for n, p in model.named_parameters()
            if n.startswith(("encoder.layers.10.", "encoder.layers.11."))}


def strict_reload(state, path, pixels, base):
    disk = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state["model"].eval(); state["head"].eval()
    with torch.random.fork_rng(devices=[0]):
        model = type(state["model"])(copy.deepcopy(state["model"].config)).float().eval()
        head = nn.Linear(1024, 128).eval()
        model.load_state_dict(disk["vision"], strict=True)
        head.load_state_dict(disk["head"], strict=True)
        buffers = dict(model.named_buffers())
        assert buffers.keys() == disk["buffers"].keys()
        with torch.no_grad():
            for n, value in buffers.items():
                value.copy_(disk["buffers"][n])
        assert len(disk["vision"]) == 400
        assert old.fingerprint(model.state_dict()) == old.fingerprint(state["model"].state_dict())
        assert old.fingerprint(head.state_dict()) == old.fingerprint(state["head"].state_dict())
        assert old.fingerprint(buffers) == base["buffers_sha256"]
        assert old.fingerprint(late.frozen_state(model, base["boundary"])) == base["frozen_sha256"]
        model.cuda(); head.cuda()
        with torch.no_grad():
            a, b = (old.previous.training.fp16(m, pixels) for m in (state["model"], model))
            assert torch.equal(a, b)
            va, vb = (F.normalize(pair.smoke.compact_head_features(v, h), dim=1)
                      for v, h in ((a, state["head"]), (b, head)))
            assert torch.equal(va, vb)
            old.previous.training.packed_equal(va, vb)


def main():
    started = UNIT_STARTED
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--phase", choices=("mechanics", "train"), required=True)
    p.add_argument("--boundary", type=int, choices=(10, 12), required=True)
    p.add_argument("--seed", type=int, choices=(179032, 179041), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execution-sha256", required=True)
    p.add_argument("--cpu-execution-sha256", required=True)
    for name in ("cpu", "mechanics"):
        p.add_argument(f"--{name}-proof", type=Path, required=name == "cpu")
        p.add_argument(f"--{name}-sha256", required=name == "cpu")
        p.add_argument(f"--{name}-log", type=Path, required=name == "cpu")
        p.add_argument(f"--{name}-log-sha256", required=name == "cpu")
    args = p.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip().lstrip("/")
    assert "memory" in (cgroup / "cgroup.controllers").read_text().split()
    assert int((cgroup / "memory.max").read_text()) == 8 * 1024**3
    assert int((cgroup / "memory.swap.max").read_text()) == 0
    assert str(os.getpid()) in (cgroup / "cgroup.procs").read_text().split()
    invocation = os.environ["INVOCATION_ID"]
    if args.phase == "mechanics":
        assert (args.boundary, args.seed) == (10, 179032)
        assert not any((args.mechanics_proof, args.mechanics_sha256, args.mechanics_log, args.mechanics_log_sha256))
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    control, source, prior, proof, _, code = qualification.startup(root, args.execution_sha256)
    assert code[Path(__file__).name] == pair.sha(Path(__file__))
    pair.executing_authority(root, code)
    cpu = read_authority(args.cpu_proof, args.cpu_sha256)
    validate_cpu(cpu, args.cpu_execution_sha256, code)
    cpu_code = read_authority(qualification.SOURCE_ROOT / "late-dense-execution.json", args.cpu_execution_sha256)
    assert cpu_code == cpu["code"]
    read_authority(args.cpu_log, args.cpu_log_sha256, log=True)
    mechanics = None
    if args.phase == "train":
        mechanics = read_authority(args.mechanics_proof, args.mechanics_sha256)
        validate_mechanics(mechanics, args.execution_sha256, args.cpu_sha256, args.cpu_log_sha256, args.cpu_execution_sha256)
        assert mechanics["initial_state_sha256"] == cpu["initial_state_sha256"]
        read_authority(args.mechanics_log, args.mechanics_log_sha256, log=True)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    flags = old.coverage.teacher.qualified.numerical_flags()
    assert flags == prior["numerical_flags"]
    torch.cuda.reset_peak_memory_stats()
    state, initial = qualification.fresh(control, source, proof, args.boundary, "cuda")
    assert initial == cpu["initial_state_sha256"]
    assert state["counter"] == 0 and not state["optimizer"].state and state["scaler"].get_scale() == 128
    target = state["target"].cpu().numpy()
    batches = old.coverage.schedule(target, seed=args.seed)
    assert batches.shape == (100, 64)
    schedule_sha = pair.smoke.digest({"batches": torch.from_numpy(batches)})
    base = identity(state, args, initial, schedule_sha)
    verify(state, base)
    if mechanics and args.seed == 179032:
        assert schedule_sha == mechanics["schedule_sha256"]
    before = {n: v.detach().cpu().clone() for n, v in new_weights(state["model"]).items()}
    assert len(before) == 32
    rng = old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    counts = np.bincount(target)
    original_terms = old.coverage.terms

    def update(s, step):
        counter, augmentation = counters(step)
        assert s["counter"] == counter - 1
        torch.cuda.synchronize(); tick = time.perf_counter()
        ids = tuple(map(int, batches[step - 1]))
        with patch.object(pair, "SEED", args.seed):
            images, rgb = pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], ids, augmentation)
        pixels = pair.pixels(s["processor"], images, "large")
        assert pixels.shape == (64, 3, 256, 256)
        pixel_sha = pair.smoke.digest({"pixels": pixels})
        active = bool((counts[target[list(ids)]] > 1).all())
        audit = {}
        clipping = torch.nn.utils.clip_grad_norm_

        def preclip(parameters, *a, **kw):
            assert [id(v) for v in parameters] == [id(v) for _, v in s["params"]]
            if args.boundary == 10:
                audit["new_preclip_gradient_norms"] = {n: float(v.grad.double().norm())
                    for n, v in new_weights(s["model"]).items()}
                assert all(np.isfinite(v) for v in audit["new_preclip_gradient_norms"].values())
                assert all(sum(v for n, v in audit["new_preclip_gradient_norms"].items()
                               if n.startswith(f"encoder.layers.{i}.")) > 0 for i in (10, 11))
                assert all(audit["new_preclip_gradient_norms"][f"encoder.layers.{i}.{name}"] > 0
                           for i in (10, 11) for name in
                           ("self_attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"))
            return clipping(parameters, *a, **kw)

        with patch.object(old.coverage, "terms", objective.terms), patch.object(torch.nn.utils, "clip_grad_norm_", preclip):
            row = old.step(s, pixels, ids, active, micro=16)
        assert s["counter"] == counter and old.coverage.terms is original_terms
        row.update(optimizer_counter=counter, augmentation_step=augmentation,
                   rgb_sha256=rgb, pixels_sha256=pixel_sha, rank_active_before=active, **audit)
        if mechanics and (args.boundary, args.seed) == (10, 179032) and step <= 17:
            assert diagnostic(row) == diagnostic(mechanics["steps"][step - 1]), "candidate mechanics replay differs"
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        torch.cuda.synchronize(); row["seconds"] = time.perf_counter() - tick
        print(json.dumps(row), flush=True)
        return row

    checkpoint_sha = None
    serialization_seconds = 0.
    if args.phase == "mechanics":
        with TemporaryDirectory(prefix="discard-late-dense-") as temporary:
            path = Path(temporary) / "step8.pt"
            rows = []
            for step in range(1, 18):
                rows.append(update(state, step))
                if step == 8:
                    tick = time.perf_counter()
                    saved_sha, saved_fingerprint = save(state, base, path)
                    serialization_seconds += time.perf_counter() - tick
            terminal = old.fingerprint(old.payload(state, base))
            del state; gc.collect(); torch.cuda.empty_cache()
            state, restored_initial = qualification.fresh(control, source, proof, args.boundary, "cuda")
            assert restored_initial == initial and identity(state, args, initial, schedule_sha) == base
            restore(state, base, path, saved_sha, saved_fingerprint, 8)
            resumed = [update(state, step) for step in range(9, 18)]
            assert [diagnostic(r) for r in resumed] == [diagnostic(r) for r in rows[8:]]
            assert old.fingerprint(old.payload(state, base)) == terminal
            path17 = Path(temporary) / "step17.pt"
            tick = time.perf_counter()
            _, disk_fingerprint = save(state, base, path17)
            serialization_seconds += time.perf_counter() - tick
            assert disk_fingerprint == terminal
            verify(state, base)
            del state["optimizer"]; gc.collect(); torch.cuda.empty_cache()
            with patch.object(pair, "SEED", args.seed):
                images, _ = pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], tuple(map(int, batches[0, :2])), 1001)
            calibration = pair.pixels(state["processor"], images, "large").cuda()
            strict_reload(state, path17, calibration, base)
        facts = {"training_state_discarded": True, "native_17_equals_serialized8_plus9_exact": True,
            "strict400_reload_whole_head_packed_exact": True, "terminal_state_fingerprint": terminal,
            "resumed_steps": resumed, "serialization_seconds": serialization_seconds}
    else:
        tick = time.perf_counter()
        rows = [update(state, step) for step in range(1, 101)]
        wall = time.perf_counter() - tick
        verify(state, base)
        args.output.mkdir(exist_ok=False)
        tick = time.perf_counter()
        checkpoint_sha, terminal = save(state, base, args.output / "resume.pt")
        serialization_seconds = time.perf_counter() - tick
        facts = {"training_state_discarded": False, "terminal_state_fingerprint": terminal,
                 "training_wall_seconds": wall, "serialization_seconds": serialization_seconds}
    after = new_weights(state["model"])
    changed = {n: not torch.equal(before[n], v.detach().cpu()) for n, v in after.items()}
    assert all(changed.values()) if args.boundary == 10 else not any(changed.values())
    frozen_state = late.frozen_state(state["model"], args.boundary)
    assert sorted(frozen_state) == base["frozen_names"] and old.fingerprint(frozen_state) == base["frozen_sha256"]
    assert old.fingerprint(dict(state["model"].named_buffers())) == base["buffers_sha256"]
    assert old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}) == rng
    assert old.coverage.teacher.qualified.numerical_flags() == flags and old.coverage.terms is original_terms
    assert json.loads(json.dumps(old.coverage.trained.native.environment(state["model"], state["processor"]))) == source["environment"]
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert pair.sha(qualification.RESUME) == late.SOURCE_SHA
    total = time.perf_counter() - started
    assert total < (120 if args.phase == "mechanics" else 300)
    peak = torch.cuda.max_memory_allocated()
    assert peak < 10_000_000_000
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert 0 < rss <= 8 * 1024 * 1024
    status = Path("/proc/self/status").read_text().splitlines()
    swap = int(next(v for v in status if v.startswith("VmSwap:")).split()[1])
    assert swap == 0
    if args.phase == "mechanics":
        overhead = time.perf_counter() - started - sum(r["seconds"] for r in rows + resumed)
        median = statistics.median(r["seconds"] for r in rows[2:])
        admission = 100 * max(median, statistics.mean(r["seconds"] for r in rows)) + max(30., overhead)
        assert admission <= 269
        facts.update(chunk100_admission_seconds=admission, measured_unit_overhead_seconds=overhead)
        args.output.mkdir(exist_ok=False)
    receipt = {"pass": True, "intervention": METHOD, "phase": args.phase,
        "boundary": args.boundary, "seed": args.seed, "updates": len(rows), "completed_step": len(rows),
        "unit_invocation_id": invocation, "unit_cgroup": str(cgroup),
        "unit_memory_max_bytes": 8 * 1024**3, "unit_memory_swap_max_bytes": 0,
        "execution_sha256": args.execution_sha256, "cpu_authority_sha256": args.cpu_sha256,
        "cpu_log_sha256": args.cpu_log_sha256, "cpu_execution_sha256": args.cpu_execution_sha256,
        "mechanics_sha256": args.mechanics_sha256,
        "mechanics_log_sha256": args.mechanics_log_sha256, "source_checkpoint_sha256": late.SOURCE_SHA,
        "initial_state_sha256": initial, "schedule_sha256": schedule_sha, "resume_identity": base,
        "checkpoint_sha256": checkpoint_sha, "steps": rows, "new32_weights_changed": all(changed.values()),
        "new_parameter_changes": changed, "frozen_named_state_buffers_rng_preserved": True,
        "median_step_3_end_seconds": statistics.median(r["seconds"] for r in rows[2:]),
        "peak_cuda_allocated_bytes": peak, "host_max_rss_kib": rss,
        "host_swap_kib": swap, "total_seconds": total, "quality_read": False, "claim_eligible": False, **facts}
    atomic_write(args.output / "receipt.json", lambda stream: stream.write((json.dumps(receipt, indent=2) + "\n").encode()))
    print("PASS discarded late17 mechanics" if args.phase == "mechanics" else "PASS fresh late100 arm; no quality read", flush=True)


if __name__ == "__main__":
    main()
