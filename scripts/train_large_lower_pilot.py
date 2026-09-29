#!/usr/bin/env python3
"""One frozen 100-update lower-adapter candidate, matched to an archived control."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import argparse
import copy
import hashlib
import json
import os
import statistics
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import train_large_lower_mechanics as mechanics

old = mechanics.old
MECHANICS = Path("/home/riomus/runs/sfora-large-lower-adapter-mechanics-v4")
MECHANICS_SHA = "2a9e339d92bb09c7ae6938f4a4a145d2ac100f325a9deb9f81dd5868aeab0558"
MECHANICS_CODE = "e5a295eff71618288569fae74b6a4d75a158bb107d5751d71715f208a88470ed"
CONTROLS = {
    179032: ("sfora-teacher-retained256-128-100-v1", "9c258702991b99463f08ec315c3a01f17332d28cc42e527baede11ef5e868590", "resume.pt", "model"),
    179041: ("sfora-native-valid-anchor-179041-treatment-100-v1", "771613484f49fb22d816f965f41ace03480ac1291745b6afb446ca50d977fc75", "native.pt", "vision"),
}


def initial_fingerprint(state, seed):
    vision = OrderedDict((k.replace(".parametrizations.weight.original", ".weight"), v)
                         for k, v in state["model"].state_dict().items()
                         if ".parametrizations.weight.0." not in k)
    assert len(vision) == 400
    # Archived controls use the framed1056 format; mechanics retains legacya377.
    digest = hashlib.sha256()
    def frame(value):
        encoded = value.encode()
        digest.update(str(len(encoded)).encode() + b":" + encoded)
    def visit(value):
        frame(type(value).__name__)
        if isinstance(value, torch.Tensor):
            frame(old.pair.smoke.digest({"tensor": value}))
        elif isinstance(value, Mapping):
            frame(str(len(value)))
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            frame(str(len(value)))
            for item in value:
                visit(item)
        else:
            frame(repr(value))
    visit({CONTROLS[seed][3]: vision, "head": state["head"].state_dict(),
           "classifier": state["classifier"], "bank": state["bank"]})
    return digest.hexdigest()


def startup(root, expected, seed):
    assert mechanics.sha(root / "lower-pilot-execution.json") == expected
    code = json.loads((root / "lower-pilot-execution.json").read_text())
    previous = json.loads((root / "large-lower-gpu-execution.json").read_text())
    assert set(code) == set(previous) | {"train_large_lower_pilot.py"}
    assert all(code[k] == v for k, v in previous.items())
    assert all(mechanics.sha(root / k) == v for k, v in code.items()), "pilot source differs"
    helpers = old.previous.selected.helpers
    with patch.object(old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
        control, native, prior, proof, _ = mechanics.startup(root, MECHANICS_CODE)
    assert mechanics.sha(MECHANICS / "lower-mechanics-proof-v4.json") == MECHANICS_SHA
    qualified = json.loads((MECHANICS / "lower-mechanics-proof-v4.json").read_text())
    assert qualified["pass"] and qualified["merged_native_strict400_exact"]
    assert qualified["recovered_rank_updates"] == 7 and len(qualified["steps"]) == 17
    assert not qualified["checkpoint_saved"] and not qualified["quality_read"]
    assert qualified["median_step_3_to_17_seconds"] * 100 < 269
    name, receipt_sha, checkpoint, _ = CONTROLS[seed]
    run = Path("/home/riomus/runs") / name
    assert mechanics.sha(run / "receipt.json") == receipt_sha
    archived = json.loads((run / "receipt.json").read_text())
    assert archived["pass"] and archived["seed"] == seed and not archived["quality_read"]
    assert archived.get("updates", archived.get("completed_step")) == 100
    assert not archived["training_state_discarded"]
    assert archived["source_checkpoint_sha256"] == old.coverage.teacher.TEACHER_SHA
    assert mechanics.sha(run / checkpoint) == archived["checkpoint_sha256"]
    assert [r["step"] for r in archived["steps"]] == list(range(1, 101))
    assert archived["peak_cuda_allocated_bytes"] < 10_000_000_000
    return control, native, prior, proof, code, archived, qualified


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=tuple(CONTROLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--cpu-proof", type=Path)
    parser.add_argument("--cpu-sha256")
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    control, native, prior, proof, code, archived, qualified = startup(root, args.execution_sha256, args.seed)
    if args.startup_only:
        assert not torch.cuda.is_available()
        state = old.fresh(control, native, proof, "half", "cpu")
        assert initial_fingerprint(state, args.seed) == archived["initial_state_sha256"]
        original_sha = mechanics.sha
        with patch.object(mechanics, "sha", lambda p: "changed" if Path(p).resolve() == Path(__file__).resolve() else original_sha(p)):
            try:
                startup(root, args.execution_sha256, args.seed)
            except AssertionError:
                pass
            else:
                raise AssertionError("changed pilot driver accepted")
        old.pair.smoke.save(args.output, {"pass": True, "seed": args.seed,
            "execution_sha256": args.execution_sha256, "initial_state_sha256": archived["initial_state_sha256"],
            "changed_driver_rejected": True, "optimizer_updates": 0, "quality_read": False})
        return
    assert torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert args.cpu_proof and args.cpu_sha256 and mechanics.sha(args.cpu_proof) == args.cpu_sha256
    admitted = json.loads(args.cpu_proof.read_text())
    assert admitted["pass"] and admitted["seed"] == args.seed and admitted["execution_sha256"] == args.execution_sha256
    assert admitted["initial_state_sha256"] == archived["initial_state_sha256"]
    assert admitted["changed_driver_rejected"] and admitted["optimizer_updates"] == 0 and not admitted["quality_read"]
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    args.output.mkdir()
    state = mechanics.fresh(control, native, proof)
    assert initial_fingerprint(state, args.seed) == archived["initial_state_sha256"]
    originals = {n: p.detach().cpu().clone() for n, p in state["model"].named_parameters() if not p.requires_grad}
    rng = old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    target = state["target"].cpu().numpy()
    batches = old.coverage.schedule(target, seed=args.seed)
    assert batches.shape == (100, 64)
    counts = np.bincount(target)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step, batch in enumerate(batches, 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        ids = tuple(map(int, batch))
        with patch.object(old.pair, "SEED", args.seed):
            images, rgb = old.pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], ids, step)
        pixels = old.pair.pixels(state["processor"], images, "large")
        pixel_sha = old.pair.smoke.digest({"pixels": pixels})
        reference = archived["steps"][step - 1]
        assert rgb == reference["rgb_sha256"] and pixel_sha == reference["pixels_sha256"]
        active = bool((counts[target[batch]] > 1).all())
        assert active == reference["rank_active_before"]
        with patch.object(old.coverage, "terms", mechanics.lane.terms):
            row = old.step(state, pixels, ids, active)
        if args.seed == 179032 and step <= 17:
            assert all(row[k] == qualified["steps"][step - 1][k] for k in row), "candidate mechanics replay differs"
        lower = [float(module.parametrizations.weight[0].B.grad.norm()) for module, _ in state["sites"]]
        assert all(v > 0 and np.isfinite(v) for v in lower)
        assert state["counter"] == step and torch.cuda.max_memory_allocated() < 10_000_000_000
        torch.cuda.synchronize()
        row.update(seconds=time.perf_counter() - tick, rgb_sha256=rgb, pixels_sha256=pixel_sha,
                   rank_active_before=active, rank_recovered=not active, lower_factor_b_min_gradient=min(lower))
        rows.append(row)
        print(json.dumps(row), flush=True)
    wall = time.perf_counter() - started
    median = statistics.median(r["seconds"] for r in rows[2:])
    baseline_median = statistics.median(r["seconds"] for r in archived["steps"][2:])
    training = {"seed": args.seed, "updates": 100, "steps": rows, "training_wall_seconds": wall,
        "median_step_seconds": median, "images_per_second": 6400 / wall,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "quality_read": False,
        "initial_state_sha256": archived["initial_state_sha256"], "control_receipt_sha256": CONTROLS[args.seed][1],
        "archival_control_cost": True, "training_wall_ratio": wall / archived["training_wall_seconds"],
        "median_step_ratio": median / baseline_median, "recovered_rank_updates": sum(r["rank_recovered"] for r in rows)}
    old.pair.smoke.save(args.output / "training.json", training)
    assert training["training_wall_ratio"] <= 1.50 and training["median_step_ratio"] <= 1.50
    assert rng == old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    assert all(torch.equal(p.detach().cpu(), originals[n]) for n, p in state["model"].named_parameters() if not p.requires_grad)
    calibration = pixels[:2].cuda()
    model, head = state["model"].eval(), state["head"].eval()
    with torch.no_grad():
        before = old.previous.training.fp16(model, calibration)
    mechanics.merge(state["sites"])
    assert len(model.state_dict()) == 400
    with torch.random.fork_rng(devices=[0]):
        saved = {"vision": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                 "classifier": state["classifier"].detach().cpu(), "bank": state["bank"].cpu(),
                 "classes": proof["arms"]["half"]["classes"]}
        torch.save(saved, args.output / "native.pt")
        del saved, state, originals
        torch.cuda.empty_cache()
        disk = torch.load(args.output / "native.pt", map_location="cpu", weights_only=True, mmap=True)
        strict = type(model)(copy.deepcopy(model.config)).float().eval()
        strict.load_state_dict(disk["vision"], strict=True)
        strict.cuda()
        strict_head = nn.Linear(1024, 128).cuda().eval()
        strict_head.load_state_dict(disk["head"], strict=True)
        with torch.no_grad():
            after = old.previous.training.fp16(strict, calibration)
            assert torch.equal(before, after)
            a = F.normalize(old.pair.smoke.compact_head_features(before, head), dim=1)
            b = F.normalize(old.pair.smoke.compact_head_features(after, strict_head), dim=1)
            assert torch.equal(a, b)
            old.previous.training.packed_equal(a, b)
        whole_sha = old.pair.smoke.digest(old.coverage.trained.base.whole_state(model))
        head_sha = old.pair.smoke.digest(head.state_dict())
        assert whole_sha == old.pair.smoke.digest(old.coverage.trained.base.whole_state(strict))
        assert head_sha == old.pair.smoke.digest(strict_head.state_dict())
    assert rng == old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    assert all(mechanics.sha(root / n) == h for n, h in code.items())
    assert mechanics.sha(old.coverage.teacher.TEACHER) == old.coverage.teacher.TEACHER_SHA
    old.initializers(proof, "half")
    assert torch.cuda.max_memory_allocated() < 10_000_000_000
    old.pair.smoke.save(args.output / "receipt.json", {**training, "pass": True,
        "execution_sha256": args.execution_sha256, "mechanics_sha256": MECHANICS_SHA,
        "cpu_admission_sha256": args.cpu_sha256,
        "source_checkpoint_sha256": old.coverage.teacher.TEACHER_SHA,
        "checkpoint_sha256": mechanics.sha(args.output / "native.pt"), "updated_whole_sha256": whole_sha,
        "updated_head_sha256": head_sha, "strict400_reload_whole_head_packed_exact": True,
        "source_and_training_rng_preserved": True, "original_frozen_weights_preserved_before_merge": True,
        "training_state_discarded": False, "claim_eligible": False})
    print("PASS fixed100 lower-adapter candidate and merged native parity; no quality read")


if __name__ == "__main__":
    main()
