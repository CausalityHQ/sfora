#!/usr/bin/env python3
"""Fixed boundary12 image-queue startup, discarded17 mechanics, or fresh100."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import resource
import statistics
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch

STARTED = time.perf_counter()
METHOD = "image-queue-adaptation-v1"
ADDED = {"train_image_queue_adaptation.py", "test_image_queue_adaptation.py"}


def validate_cpu(value, execution, code, previous):
    assert value["schema"] == "image-queue-cpu-v1" and value["pass"]
    assert value["execution_sha256"] == execution and value["code"] == previous
    assert len(previous) == 118 and len(code) == 120 and set(code) - set(previous) == ADDED
    assert all(code[n] == h for n, h in previous.items())
    assert value["boundary"] == 12 and value["optimizer_members"] == 208
    assert all(value[k] for k in ("matched_control_candidate_initial_state", "native400_whole_head_packed_reload_exact",
        "optimizer_and_frozen_buffer_negatives_rejected", "changed_driver_rejected", "RNG_preserved"))
    assert value["optimizer_updates"] == 0 and value["quality_read"] is False
    assert {s: v["unique_images"] for s, v in value["schedules"].items()} == {"179032": 6330, "179041": 6331}
    resources(value, 120, False)


def resources(value, cap, gpu):
    assert value["unit_invocation_id"] and Path(value["unit_cgroup"]).name.endswith(".service")
    assert value["unit_memory_max_bytes"] == 8 * 1024**3 and value["unit_memory_swap_max_bytes"] == 0
    assert math.isfinite(value["total_seconds"]) and 0 < value["total_seconds"] < cap
    assert 0 < value["host_max_rss_kib"] <= 8 * 1024 * 1024 and value["host_swap_kib"] == 0
    peak = value["peak_cuda_allocated_bytes"]
    assert 0 < peak < 10_000_000_000 if gpu else peak == 0


def validate_log(text, value, cap):
    assert all(s in text for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
    unit = Path(value["unit_cgroup"]).name
    assert f"Running as unit: {unit}; invocation ID: {value['unit_invocation_id']}" in text
    matches = re.findall(r"Service runtime: (?:(\d+)min )?([\d.]+)s", text)
    assert len(matches) == 1 and 0 < int(matches[0][0] or 0) * 60 + float(matches[0][1]) < cap
    rss = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", text)
    # The receipt samples RSS before publication and interpreter shutdown.
    # The original process timer supplies the final whole-process peak.
    assert rss and 0 < value["host_max_rss_kib"] <= int(rss[0]) <= 8 * 1024 * 1024
    assert "flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock" in text
    assert "flock -n /home/riomus/.sfora-siglip2-gpu.lock" in text


def validate_mechanics(value, args, cpu):
    assert value["schema"] == "image-queue-train-v1" and value["pass"] and value["intervention"] == METHOD
    assert value["phase"] == "mechanics" and value["arm"] == "queue" and value["seed"] == 179032
    assert value["execution_sha256"] == args.execution_sha256 and value["cpu_authority_sha256"] == args.cpu_sha256
    assert value["cpu_execution_sha256"] == args.cpu_execution_sha256
    assert value["startup_authority_sha256"] == args.startup_sha256
    assert value["cpu_log_sha256"] == args.cpu_log_sha256
    assert value["initial_state_sha256"] == cpu["initial_state_sha256"]
    assert value["source_checkpoint_sha256"] == cpu["source_checkpoint_sha256"]
    assert value["updates"] == value["completed_step"] == 17 and value["boundary"] == 12 and value["checkpoint_sha256"] is None
    assert value["training_state_discarded"] and value["native_17_equals_serialized8_plus9_exact"]
    assert value["strict400_reload_whole_head_packed_exact"] and value["frozen_named_state_buffers_rng_preserved"]
    projection = value["chunk100_admission_seconds"]
    assert value["quality_read"] is False and math.isfinite(projection) and 0 < projection <= 269
    assert value["schedule_sha256"] == cpu["schedules"]["179032"]["queue_schedule_sha256"]
    assert value["resume_identity"]["sampling_arm"] == "queue"
    assert value["resume_identity"]["schedule_sha256"] == value["schedule_sha256"]
    assert value["resume_identity"]["class_sequence_sha256"] == cpu["schedules"]["179032"]["class_sequence_sha256"]
    assert len(value["steps"]) == 17
    for step, row in enumerate(value["steps"], 1):
        assert row["step"] == row["optimizer_counter"] == step and row["augmentation_step"] == 1000 + step
        assert len(row["image_ids"]) == 64 and len(set(row["image_ids"])) == 64
        assert all(type(i) is int and 0 <= i < 13283 for i in row["image_ids"])
    resources(value, 120, True)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--phase", choices=("startup", "mechanics", "train"), required=True)
    p.add_argument("--arm", choices=("control", "queue"), required=True)
    p.add_argument("--seed", type=int, choices=(179032, 179041), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execution-sha256", required=True)
    p.add_argument("--cpu-execution-sha256", required=True)
    for name in ("cpu", "startup", "mechanics"):
        for suffix in ("proof", "log"):
            p.add_argument(f"--{name}-{suffix}", type=Path, required=name == "cpu")
        p.add_argument(f"--{name}-sha256", required=name == "cpu")
        p.add_argument(f"--{name}-log-sha256", required=name == "cpu")
    args = p.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    root = Path(__file__).resolve().parent
    import qualify_image_queue_adaptation_cpu as qualified_cpu
    manifest = root / "image-queue-train-execution.json"
    assert qualified_cpu.sha(manifest) == args.execution_sha256
    code = json.loads(manifest.read_text())
    assert all(qualified_cpu.sha(root / n) == h for n, h in code.items())
    previous = qualified_cpu.closure(root, args.cpu_execution_sha256)
    cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip().lstrip("/")
    assert int((cgroup / "memory.max").read_text()) == 8 * 1024**3 and int((cgroup / "memory.swap.max").read_text()) == 0
    assert str(os.getpid()) in (cgroup / "cgroup.procs").read_text().split()
    invocation = os.environ["INVOCATION_ID"]
    import numpy as np
    import torch
    import train_late_dense_adaptation as native
    qualified, old, pair = native.qualification, native.old, native.pair
    cpu = native.read_authority(args.cpu_proof, args.cpu_sha256)
    validate_cpu(cpu, args.cpu_execution_sha256, code, previous)
    assert cpu["source_checkpoint_sha256"] == qualified.late.SOURCE_SHA
    validate_log(native.read_authority(args.cpu_log, args.cpu_log_sha256, log=True), cpu, 120)
    selected = qualified.archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, "helpers", lambda r, _: helpers(r, code)):
        control, source, prior, proof, _, _ = qualified.startup(root, qualified_cpu.ORIGINAL)
    pair.executing_authority(root, code)
    torch.set_num_threads(8); torch.manual_seed(args.seed)
    facts = {"schema": "image-queue-train-v1", "pass": True, "intervention": METHOD,
        "phase": args.phase, "arm": args.arm, "seed": args.seed, "boundary": 12,
        "execution_sha256": args.execution_sha256, "cpu_authority_sha256": args.cpu_sha256,
        "cpu_log_sha256": args.cpu_log_sha256, "cpu_execution_sha256": args.cpu_execution_sha256,
        "source_checkpoint_sha256": qualified.late.SOURCE_SHA, "quality_read": False, "claim_eligible": False}
    mechanics = None
    if args.phase == "startup":
        assert not torch.cuda.is_available()
    else:
        startup = native.read_authority(args.startup_proof, args.startup_sha256)
        assert startup["phase"] == "startup" and startup["execution_sha256"] == args.execution_sha256
        assert startup["pass"] and startup["cpu_authority_sha256"] == args.cpu_sha256
        assert startup["cpu_log_sha256"] == args.cpu_log_sha256
        resources(startup, 120, False)
        validate_log(native.read_authority(args.startup_log, args.startup_log_sha256, log=True), startup, 120)
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        if args.phase == "mechanics":
            assert (args.arm, args.seed) == ("queue", 179032)
            assert not any((args.mechanics_proof, args.mechanics_sha256, args.mechanics_log, args.mechanics_log_sha256))
        else:
            mechanics = native.read_authority(args.mechanics_proof, args.mechanics_sha256)
            validate_mechanics(mechanics, args, cpu)
            validate_log(native.read_authority(args.mechanics_log, args.mechanics_log_sha256, log=True), mechanics, 120)
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        flags = old.coverage.teacher.qualified.numerical_flags(); assert flags == prior["numerical_flags"]
        torch.cuda.reset_peak_memory_stats()
        state, initial = qualified.fresh(control, source, proof, 12, "cuda")
        assert initial == cpu["initial_state_sha256"] and state["counter"] == 0 and not state["optimizer"].state
        target = state["target"].cpu().tolist()
        original = old.coverage.schedule(target, seed=args.seed).tolist()
        batches = original if args.arm == "control" else qualified_cpu.schedules(target, original, args.seed)
        schedule_sha = old.fingerprint({"batches": torch.tensor(batches)})
        expected = cpu["schedules"][str(args.seed)]
        assert schedule_sha == expected[args.arm + "_schedule_sha256"]
        assert qualified_cpu.sha_bytes([[target[i] for i in b] for b in batches]) == expected["class_sequence_sha256"]
        if mechanics:
            queue = batches if args.arm == "queue" else qualified_cpu.schedules(target, original, args.seed)
            # Mechanics is seed032; validate it against that authenticated schedule for either TRAIN seed.
            if args.seed != 179032:
                original032 = old.coverage.schedule(target, seed=179032).tolist()
                queue = qualified_cpu.schedules(target, original032, 179032)
            assert [r["image_ids"] for r in mechanics["steps"]] == queue[:17]
        identity_args = SimpleNamespace(**vars(args), boundary=12)
        base = native.identity(state, identity_args, initial, schedule_sha)
        base.update(schema="image-queue-complete-resume-v1", intervention=METHOD, sampling_arm=args.arm,
                    class_sequence_sha256=expected["class_sequence_sha256"])
        native.verify(state, base)
        rng = old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
        counts = np.bincount(target)

        def update(s, step):
            counter, augmentation = native.counters(step)
            assert s["counter"] == counter - 1
            torch.cuda.synchronize(); tick = time.perf_counter()
            ids = tuple(batches[step - 1])
            with patch.object(pair, "SEED", args.seed):
                images, rgb = pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], ids, augmentation)
            pixels = pair.pixels(s["processor"], images, "large")
            assert pixels.shape == (64, 3, 256, 256)
            active = bool((counts[np.asarray(target)[list(ids)]] > 1).all())
            with patch.object(old.coverage, "terms", native.objective.terms):
                row = old.step(s, pixels, ids, active, micro=16)
            assert s["counter"] == counter
            row.update(optimizer_counter=counter, augmentation_step=augmentation, image_ids=list(ids),
                rgb_sha256=rgb, pixels_sha256=pair.smoke.digest({"pixels": pixels}), rank_active_before=active)
            if mechanics and args.arm == "queue" and args.seed == 179032 and step <= 17:
                assert native.diagnostic(row) == native.diagnostic(mechanics["steps"][step - 1])
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            torch.cuda.synchronize(); row["seconds"] = time.perf_counter() - tick
            print(json.dumps(row), flush=True)
            return row

        checkpoint_sha = None
        if args.phase == "mechanics":
            with TemporaryDirectory(dir=root, prefix="discard-image-queue-") as temporary:
                path = Path(temporary) / "step8.pt"
                rows = []
                for step in range(1, 18):
                    rows.append(update(state, step))
                    if step == 8: saved_sha, saved_fingerprint = native.save(state, base, path)
                terminal = old.fingerprint(old.payload(state, base))
                del state; gc.collect(); torch.cuda.empty_cache()
                state, again = qualified.fresh(control, source, proof, 12, "cuda")
                assert again == initial
                native.restore(state, base, path, saved_sha, saved_fingerprint, 8)
                resumed = [update(state, step) for step in range(9, 18)]
                assert [native.diagnostic(r) for r in resumed] == [native.diagnostic(r) for r in rows[8:]]
                assert old.fingerprint(old.payload(state, base)) == terminal
                path17 = Path(temporary) / "step17.pt"
                _, fingerprint = native.save(state, base, path17); assert fingerprint == terminal
                del state["optimizer"]; gc.collect(); torch.cuda.empty_cache()
                with patch.object(pair, "SEED", args.seed):
                    images, _ = pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], tuple(batches[0][:2]), 1001)
                native.strict_reload(state, path17, pair.pixels(state["processor"], images, "large").cuda(), base)
            overhead = time.perf_counter() - STARTED - sum(r["seconds"] for r in rows + resumed)
            admission = 100 * max(statistics.median(r["seconds"] for r in rows[2:]), statistics.mean(r["seconds"] for r in rows)) + max(30., overhead)
            assert admission <= 269
            facts.update(training_state_discarded=True, native_17_equals_serialized8_plus9_exact=True,
                strict400_reload_whole_head_packed_exact=True, resumed_steps=resumed,
                chunk100_admission_seconds=admission, measured_unit_overhead_seconds=overhead)
        else:
            tick = time.perf_counter(); rows = [update(state, step) for step in range(1, 101)]
            facts["training_wall_seconds"] = time.perf_counter() - tick
            args.output.mkdir(exist_ok=False)
            tick = time.perf_counter(); checkpoint_sha, terminal = native.save(state, base, args.output / "resume.pt")
            facts.update(training_state_discarded=False, serialization_seconds=time.perf_counter() - tick)
        assert old.fingerprint(qualified.late.frozen_state(state["model"], 12)) == base["frozen_sha256"]
        assert old.fingerprint(dict(state["model"].named_buffers())) == base["buffers_sha256"]
        assert old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}) == rng
        assert old.coverage.teacher.qualified.numerical_flags() == flags
        assert json.loads(json.dumps(old.coverage.trained.native.environment(state["model"], state["processor"]))) == source["environment"]
        assert pair.sha(qualified.RESUME) == qualified.late.SOURCE_SHA
        facts.update(initial_state_sha256=initial, schedule_sha256=schedule_sha, resume_identity=base,
            updates=len(rows), completed_step=len(rows), steps=rows, checkpoint_sha256=checkpoint_sha,
            terminal_state_fingerprint=terminal, frozen_named_state_buffers_rng_preserved=True,
            median_step_3_end_seconds=statistics.median(r["seconds"] for r in rows[2:]),
            startup_authority_sha256=args.startup_sha256, mechanics_sha256=args.mechanics_sha256,
            mechanics_log_sha256=args.mechanics_log_sha256)
    assert all(qualified_cpu.sha(root / n) == h for n, h in code.items())
    elapsed = time.perf_counter() - STARTED
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    swap = int(next(s for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmSwap:")).split()[1])
    facts.update(unit_invocation_id=invocation, unit_cgroup=str(cgroup), unit_memory_max_bytes=8 * 1024**3,
        unit_memory_swap_max_bytes=0, total_seconds=elapsed, host_max_rss_kib=rss, host_swap_kib=swap,
        peak_cuda_allocated_bytes=0 if args.phase == "startup" else torch.cuda.max_memory_allocated())
    resources(facts, 300 if args.phase == "train" else 120, args.phase != "startup")
    if args.phase != "train": args.output.mkdir(exist_ok=False)
    native.atomic_write(args.output / "receipt.json", lambda stream: stream.write((json.dumps(facts, indent=2) + "\n").encode()))


if __name__ == "__main__":
    main()
