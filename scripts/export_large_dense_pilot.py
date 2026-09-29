#!/usr/bin/env python3
"""Qualify a pilot checkpoint, then export complete independently replayed held wires."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import train_large_dense_pilot as training
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

CONTROLS = training.pilot.CONTROLS
sha = training.pilot.mechanics.sha

old = training.old
teacher = old.coverage.teacher
TRAIN_CODE = "916e3679574613465da283a920d3526b5add38eac80843849f3fa1869446e3bf"
TRAIN_ROOT = Path("/home/riomus/runs/sfora-dense-pilot-source-v1")


def authority(root, expected, seed, arm):
    path = root / "dense-pilot-export-execution.json"
    assert sha(path) == expected
    code = json.loads(path.read_text())
    previous = json.loads((root / "dense-pilot-execution.json").read_text())
    assert set(code) == set(previous) | {"export_large_dense_pilot.py"}
    assert all(code[n] == h for n, h in previous.items())
    assert all(training.pilot.mechanics.sha(root / n) == h for n, h in code.items()), "export source differs"
    assert len(previous) == 100 and len(code) == 101
    assert sha(TRAIN_ROOT / "dense-pilot-execution.json") == TRAIN_CODE
    assert previous == json.loads((TRAIN_ROOT / "dense-pilot-execution.json").read_text())
    helpers = old.previous.selected.helpers
    with patch.object(old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
        (control, _, prior, native, _, initial_reference, _), _, _ = training.startup(root, TRAIN_CODE, seed)
        _, frozen, _ = teacher.startup(root, old.coverage.trained.TEACHER_CODE_SHA)
    assert frozen["fit_manifest"] == native["arms"]["half"]["rows"]
    assert len(frozen["held_manifest"]) == 12599 and len(frozen["query"]) == 6354 and len(frozen["gallery"]) == 6245
    assert {r["product"] for r in frozen["held_manifest"]}.isdisjoint(r["product"] for r in frozen["fit_manifest"])
    # Close both fixed jobs before any export or quality read; pass is integrity only.
    candidates = {}
    for declared_seed in CONTROLS:
        run = Path(f"/home/riomus/runs/sfora-dense-pilot-{declared_seed}-v1")
        value = json.loads((run / "receipt.json").read_text())
        record = json.loads((run / "training.json").read_text())
        assert all(value[k] == v for k, v in record.items())
        assert value["pass"] and value["cost_pass"] and value["checkpoint_saved"]
        assert value["seed"] == declared_seed and value["updates"] == 100 and not value["quality_read"]
        assert value["execution_sha256"] == TRAIN_CODE and value["source_code"] == previous
        assert all(value[k] for k in ("strict400_reload_whole_head_packed_exact",
            "native400_whole_head_packed_disk_reload_exact", "new32_weights_changed",
            "frozen_parameters_and_nonpersistent_buffers_unchanged", "rng_preserved"))
        assert value["source_checkpoint_sha256"] == teacher.TEACHER_SHA
        assert value["cpu_proofs"] == json.loads(json.dumps(training.mechanics.CPU_PROOFS))
        assert value["control_receipt_sha256"] == CONTROLS[declared_seed][1]
        assert value["training_wall_ratio"] <= 1.50 and value["median_step_ratio"] <= 1.50
        assert value["peak_cuda_allocated_bytes"] < 10_000_000_000
        assert sha(run / "native.pt") == value["checkpoint_sha256"]
        if declared_seed == seed:
            reference = initial_reference
        else:
            with patch.object(old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
                (_, _, _, _, _, reference, _), _, _ = training.startup(root, TRAIN_CODE, declared_seed)
        assert value["initial_state_sha256"] == reference["initial_state_sha256"]
        assert value["training_wall_ratio"] == value["training_wall_seconds"] / reference["training_wall_seconds"]
        assert value["median_step_seconds"] == statistics.median(r["seconds"] for r in value["steps"][2:])
        assert value["median_step_ratio"] == value["median_step_seconds"] / statistics.median(r["seconds"] for r in reference["steps"][2:])
        assert [r["step"] for r in value["steps"]] == list(range(1, 101))
        assert all(row[k] == ref[k] for row, ref in zip(value["steps"], reference["steps"], strict=True)
                   for k in ("rgb_sha256", "pixels_sha256", "rank_active_before"))
        log = (TRAIN_ROOT / f"sfora-dense-pilot-train-{declared_seed}-v1.log").read_text()
        assert all(s in log for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
        candidates[declared_seed] = (run / "native.pt", value, sha(run / "receipt.json"))
    assert arm == "candidate"
    checkpoint, value, receipt_sha = candidates[seed]
    return control, frozen, prior, code, checkpoint, value, receipt_sha


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--execution-sha256", required=True)
    p.add_argument("--seed", type=int, choices=tuple(CONTROLS), required=True)
    p.add_argument("--arm", choices=("candidate",), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qualify-cpu", action="store_true")
    p.add_argument("--cpu-proof", type=Path)
    p.add_argument("--cpu-sha256")
    args = p.parse_args()
    assert args.output == Path(f"/home/riomus/runs/sfora-dense-pilot-{'source' if args.qualify_cpu else 'wires'}-{args.seed}-candidate-v2")
    assert not args.output.exists()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    startup_started = time.perf_counter()
    control, frozen, prior, code, checkpoint, value, receipt_sha = authority(root, args.execution_sha256, args.seed, args.arm)
    print(json.dumps({"authority_seconds": time.perf_counter() - startup_started}), flush=True)
    binding = {"seed": args.seed, "arm": args.arm, "training_receipt_sha256": receipt_sha,
               "training_checkpoint_sha256": value["checkpoint_sha256"], "completed_updates": 100}
    if args.qualify_cpu:
        assert not torch.cuda.is_available()
        args.output.mkdir()
        checkpoint_sha = sha(checkpoint)
        def startup(r, execution):
            assert r == root and execution == args.execution_sha256
            assert all(sha(root / n) == h for n, h in code.items())
            return control, frozen, code
        with ExitStack() as scope, TemporaryDirectory(prefix="discard-dense-source-cpu-", dir=root) as tmp:
            scope.enter_context(patch.object(teacher, "TEACHER", checkpoint))
            scope.enter_context(patch.object(teacher, "TEACHER_SHA", checkpoint_sha))
            scope.enter_context(patch.object(teacher, "startup", startup))
            scope.enter_context(patch.object(sys, "argv", ["dense-source-cpu", "--execution-sha256", args.execution_sha256,
                "--output", str(Path(tmp) / "source"), "--qualify-cpu"]))
            teacher.main()
            proof = json.loads((Path(tmp) / "source/preflight.json").read_text())
        assert proof["teacher_checkpoint_sha256"] == checkpoint_sha
        assert proof["read_only"] and proof["optimizer_updates"] == 0 and not proof["quality_read"]
        assert proof["teacher_whole_sha256"] == value["updated_whole_sha256"]
        assert proof["teacher_head_sha256"] == value["updated_head_sha256"]
        original_sha = sha
        with patch.object(training.pilot.mechanics, "sha", lambda f: "changed" if Path(f).resolve() == Path(__file__).resolve() else original_sha(f)):
            try:
                authority(root, args.execution_sha256, args.seed, args.arm)
            except AssertionError:
                pass
            else:
                raise AssertionError("changed export driver accepted")
        old.pair.smoke.save(args.output / "proof.json", {**proof, **binding, "pass": True,
            "native_path": str(checkpoint), "execution_sha256": args.execution_sha256, "changed_driver_rejected": True})
        return
    assert torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert args.cpu_proof and args.cpu_sha256 and sha(args.cpu_proof) == args.cpu_sha256
    proof = json.loads(args.cpu_proof.read_text())
    assert proof["pass"] and proof["code"] == code and proof["execution_sha256"] == args.execution_sha256
    assert all(proof[k] == v for k, v in binding.items()) and proof["changed_driver_rejected"]
    assert proof["strict400_native_head_reload_and_direct_whole_calibration_exact"]
    assert proof["prefix_data_mutation_rejected_at_exit"] and proof["cpu_cuda_rng_unchanged"]
    assert Path(proof["native_path"]) == checkpoint
    assert proof["teacher_checkpoint_sha256"] == value["checkpoint_sha256"]
    assert proof["teacher_whole_sha256"] == value["updated_whole_sha256"]
    assert proof["teacher_head_sha256"] == value["updated_head_sha256"]
    assert proof["read_only"] and proof["optimizer_updates"] == 0 and not proof["quality_read"]
    assert proof["fit_manifest"] == frozen["fit_manifest"]
    assert sha(checkpoint) == proof["teacher_checkpoint_sha256"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assert teacher.qualified.numerical_flags() == prior["numerical_flags"]
    torch.cuda.reset_peak_memory_stats()
    with patch.object(teacher, "TEACHER", checkpoint):
        loaded, live, heads, processor = teacher.teacher_pair(control, torch.device("cuda"))
    def unchanged():
        assert all(old.pair.smoke.digest(teacher.base.whole_state(m)) == proof["teacher_whole_sha256"] for m in (loaded, live))
        assert all(old.pair.smoke.digest(h.state_dict()) == proof["teacher_head_sha256"] for h in heads)
        assert all(p.dtype == torch.float32 for m in (loaded, live, *heads) for p in m.parameters())
        assert all(p.grad is None for m in (loaded, live, *heads) for p in m.parameters())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for model in (loaded, live) for m in model.modules())
        assert sha(checkpoint) == proof["teacher_checkpoint_sha256"]
        assert all(sha(root / n) == h for n, h in code.items())
        assert teacher.qualified.numerical_flags() == prior["numerical_flags"]
        assert all(json.loads(json.dumps(teacher.native.environment(m, processor))) == proof["environment"] for m in (loaded, live))
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
    unchanged()
    images, _ = old.pair.augmented_images(control.dataset_root, frozen["fit_manifest"], (0, 1), None)
    pixels = old.pair.pixels(processor, images, "large")
    assert old.pair.smoke.digest({"pixels": pixels}) == proof["first_two_fit_pixels_sha256"]
    with torch.no_grad():
        x = pixels.cuda()
        direct = teacher.qualified.fp16(live, x)
        assert torch.equal(direct, teacher.qualified.fp16(loaded, x))
        full = live(pixel_values=x).pooler_output.float()
        calibration = {"pooled": F.cosine_similarity(direct, full).tolist(),
            "compact": F.cosine_similarity(old.pair.smoke.compact_head_features(direct, heads[1]),
                                           old.pair.smoke.compact_head_features(full, heads[1])).tolist()}
        assert all(min(v) >= .999 for v in calibration.values())
    rng = old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    chunks, reference, durations = [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, 12599, 32):
            tick = time.perf_counter()
            rows = frozen["held_manifest"][start:start + 32]
            images, _ = old.pair.augmented_images(control.dataset_root, rows, tuple(range(len(rows))), None)
            pixels = old.pair.pixels(processor, images, "large").cuda()
            a = teacher.qualified.fp16(loaded, pixels)
            b = teacher.qualified.fp16(live, pixels)
            assert torch.equal(a, b)
            va, vb = (F.normalize(old.pair.smoke.compact_head_features(v, h).float(), dim=1) for v, h in zip((a, b), heads, strict=True))
            assert torch.equal(va, vb) and torch.isfinite(va).all()
            old.previous.training.packed_equal(va, vb)
            assert va.shape == (len(rows), 128)
            chunks.append(va.cpu().numpy())
            reference.append(vb.cpu().numpy())
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - tick)
            if len(durations) % 64 == 0 or start + 32 >= 12599:
                print(json.dumps({"full_whole_held_images_verified": min(start + 32, 12599)}), flush=True)
    print(json.dumps({"full_held_forward_seconds": time.perf_counter() - started}), flush=True)
    unchanged()
    assert sha(teacher.TEACHER) == teacher.TEACHER_SHA
    assert all(old.pair.sha(control.dataset_root / r["relative_path"]) == r["image_sha256"]
               for r in frozen["fit_manifest"] + frozen["held_manifest"])
    assert time.perf_counter() - started < 300
    assert rng == old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    values, other = np.concatenate(chunks), np.concatenate(reference)
    assert values.shape == (12599, 128) and np.array_equal(values, other)
    assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
    a, b = (pack_int8_unit_embeddings(torch.from_numpy(v)) for v in (values, other))
    assert np.array_equal(a.codes, b.codes) and np.array_equal(a.inverse_norms, b.inverse_norms)
    args.output.mkdir()
    np.save(args.output / "held.npy", values, allow_pickle=False)
    np.save(args.output / "reference-held.npy", other, allow_pickle=False)
    old.pair.smoke.save(args.output / "receipt.json", {**binding, "pass": True, "execution_sha256": args.execution_sha256,
        "cpu_authority_sha256": args.cpu_sha256, "checkpoint_sha256": proof["teacher_checkpoint_sha256"],
        "held_sha256": sha(args.output / "held.npy"), "reference_held_sha256": sha(args.output / "reference-held.npy"),
        "held_manifest": frozen["held_manifest"], "query": frozen["query"], "gallery": frozen["gallery"],
        "full_held_independent_whole_encoder_head_packed_exact": True, "source_code": code,
        "fp16_fp32_fit_calibration": calibration,
        "source_head_rng_flags_preserved": True, "held_images": 12599, "optimizer_updates": 0,
        "export_wall_seconds": time.perf_counter() - started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "quality_read": False, "official_read": False, "claim_eligible": False})


if __name__ == "__main__":
    main()
