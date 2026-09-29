#!/usr/bin/env python3
"""One discarded dense10 mechanics gate; unchanged native update and objective."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import qualify_large_dense_boundary_cpu as qualification

dense, pilot = qualification.dense, qualification.pilot
old = pilot.old
CPU_ROOT = Path("/home/riomus/runs/sfora-dense-boundary-source-v5")
CPU_CODE = "065f942bc0fd7824cd05b14a9acb9a10fae4ba6ed4a904974376a99ad09fe605"
CPU_PROOFS = {179032: ("cpu-179032-v3.json", "a50617f269782766f9567c5d9d4ec12fd373d400e7891e802e2987a80ef26848"),
              179041: ("cpu-179041-v1.json", "70e8aae99862e937a6a8ac8e2fba9d4843fba4e6fbea2bc9a0260f90b01fe7ed")}


def startup(root, expected):
    sha = pilot.mechanics.sha
    assert sha(root / "dense-mechanics-execution.json") == expected
    code = json.loads((root / "dense-mechanics-execution.json").read_text())
    previous = json.loads((root / "dense-boundary-execution.json").read_text())
    assert set(code) == set(previous) | {"train_large_dense_mechanics.py"}
    assert all(code[n] == h for n, h in previous.items()) and all(sha(root / n) == h for n, h in code.items())
    assert sha(CPU_ROOT / "dense-boundary-execution.json") == CPU_CODE
    assert previous == json.loads((CPU_ROOT / "dense-boundary-execution.json").read_text())
    for seed, (name, digest) in CPU_PROOFS.items():
        assert sha(CPU_ROOT / name) == digest
        cpu = json.loads((CPU_ROOT / name).read_text())
        assert cpu["pass"] and cpu["seed"] == seed and cpu["execution_sha256"] == CPU_CODE
        assert cpu["optimizer_members"] == 240 and cpu["native400_whole_head_packed_reload_exact"]
        assert cpu["frozen_nonpersistent_buffer_mutation_rejected"] and not cpu["quality_read"] and cpu["optimizer_updates"] == 0
        log = (CPU_ROOT / name.replace(".json", ".log")).read_text()
        assert all(s in log for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
    helpers = old.previous.selected.helpers
    with patch.object(old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
        values, _ = qualification.startup(root, CPU_CODE, 179032)
    return values, code


def main():
    assert __debug__
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(179032)
    (control, native, prior, proof, _, archived, _), code = startup(root, args.execution_sha256)
    if args.startup_only:
        assert not torch.cuda.is_available()
        original_sha = pilot.mechanics.sha
        with patch.object(pilot.mechanics, "sha", lambda p: "changed" if Path(p).resolve() == Path(__file__).resolve() else original_sha(p)):
            qualification.rejects(lambda: startup(root, args.execution_sha256))
        old.pair.smoke.save(args.output, {"pass": True, "execution_sha256": args.execution_sha256,
            "changed_driver_rejected": True, "optimizer_updates": 0, "quality_read": False})
        return
    assert torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    torch.cuda.reset_peak_memory_stats()
    state = dense.fresh(control, native, proof, "cuda")
    assert pilot.initial_fingerprint(state, 179032) == archived["initial_state_sha256"]
    frozen = {n: p.detach().cpu().clone() for n, p in dense.frozen_state(state["model"]).items()}
    new_before = {n: p.detach().cpu().clone() for n, p in state["model"].named_parameters()
                  if n.startswith(("encoder.layers.10.", "encoder.layers.11."))}
    assert len(new_before) == 32
    rng = old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    target = state["target"].cpu().numpy()
    batches = old.coverage.schedule(target, seed=179032)[:17]
    counts = np.bincount(target)
    rows = []
    started = time.perf_counter()
    for step, batch in enumerate(batches, 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        ids = tuple(map(int, batch))
        with patch.object(old.pair, "SEED", 179032):
            images, rgb = old.pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], ids, step)
        pixels = old.pair.pixels(state["processor"], images, "large")
        pixel_sha = old.pair.smoke.digest({"pixels": pixels})
        reference = archived["steps"][step - 1]
        assert rgb == reference["rgb_sha256"] and pixel_sha == reference["pixels_sha256"]
        active = bool((counts[target[batch]] > 1).all())
        assert active == reference["rank_active_before"]
        gradient_audit = {}
        clipping = torch.nn.utils.clip_grad_norm_
        def audit(parameters, *a, **kw):
            assert [id(p) for p in parameters] == [id(p) for _, p in state["params"]]
            gradient_audit["block_preclip_norms"] = {str(i): sum(float(p.grad.double().norm()) for p in state["model"].encoder.layers[i].parameters()) for i in range(10, 24)}
            gradient_audit["head_proxy_preclip_norm"] = sum(float(p.grad.double().norm()) for p in (*state["head"].parameters(), state["classifier"]))
            gradient_audit["new_path_preclip_norms"] = {f"{i}.{name}": float(dict(state["model"].encoder.layers[i].named_parameters())[name].grad.double().norm())
                for i in (10, 11) for name in ("self_attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight")}
            assert all(v > 0 and np.isfinite(v) for v in (*gradient_audit["block_preclip_norms"].values(), *gradient_audit["new_path_preclip_norms"].values(), gradient_audit["head_proxy_preclip_norm"]))
            return clipping(parameters, *a, **kw)
        with patch.object(old.coverage, "terms", pilot.mechanics.lane.terms), patch.object(torch.nn.utils, "clip_grad_norm_", audit):
            row = old.step(state, pixels, ids, active)
        if step == 1:
            assert all(row[k] == reference[k] for k in ("ce", "rank", "loss", "scale", "gradient_norms"))
            assert row["preclip_norm"] > reference["preclip_norm"]
        dense.verify_optimizer(state)
        assert state["counter"] == step and torch.cuda.max_memory_allocated() < 10_000_000_000
        torch.cuda.synchronize()
        row.update(gradient_audit, seconds=time.perf_counter() - tick, rgb_sha256=rgb,
            pixels_sha256=pixel_sha, rank_active_before=active, rank_recovered=not active)
        rows.append(row)
        print(json.dumps(row), flush=True)
    wall = time.perf_counter() - started
    median = statistics.median(r["seconds"] for r in rows[2:])
    training = {"updates": 17, "steps": rows, "training_wall_seconds": wall, "median_step_seconds": median,
        "images_per_second": 1088 / wall, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "quality_read": False, "checkpoint_saved": False}
    old.pair.smoke.save(args.output.with_suffix(".training.json"), training)
    assert rng == old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    dense.verify_frozen(state["model"], frozen)
    assert all(not torch.equal(p.detach().cpu(), new_before[n]) for n, p in state["model"].named_parameters() if n in new_before)
    calibration = pixels[:2].cuda()
    model, head = state["model"].eval(), state["head"].eval()
    with torch.no_grad():
        pooled = old.previous.training.fp16(model, calibration)
        compact = F.normalize(old.pair.smoke.compact_head_features(pooled, head), dim=1)
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "native.pt"
        torch.save({"vision": model.state_dict(), "head": head.state_dict()}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert len(saved["vision"]) == 400
        state = None
        torch.cuda.empty_cache()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            strict, reload_head, _, _ = old.coverage.load_native(control, native)
        strict.load_state_dict(saved["vision"], strict=True)
        reload_head.load_state_dict(saved["head"], strict=True)
        dense.configure(strict)
        dense.verify_frozen(strict, frozen)
        strict.cuda().eval()
        reload_head.cuda().eval()
        with torch.no_grad():
            after = old.previous.training.fp16(strict, calibration)
            other = F.normalize(old.pair.smoke.compact_head_features(after, reload_head), dim=1)
        assert torch.equal(pooled, after) and torch.equal(compact, other)
        old.previous.training.packed_equal(compact, other)
    assert all(pilot.mechanics.sha(root / n) == h for n, h in code.items())
    assert rng == old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    peak = torch.cuda.max_memory_allocated()
    assert peak < 10_000_000_000
    projected_ratio = median * 100 / archived["training_wall_seconds"]
    median_ratio = median / statistics.median(r["seconds"] for r in archived["steps"][2:])
    admitted = projected_ratio <= 1.50 and median_ratio <= 1.50 and median * 100 < 269
    old.pair.smoke.save(args.output, {**training, "pass": True, "pilot_admitted": admitted,
        "execution_sha256": args.execution_sha256, "source_code": code, "cpu_proofs": CPU_PROOFS,
        "native400_whole_head_packed_disk_reload_exact": True, "new32_weights_changed": True,
        "frozen_parameters_and_nonpersistent_buffers_unchanged": True, "rng_preserved": True,
        "peak_cuda_allocated_bytes": peak, "recovered_rank_updates": sum(r["rank_recovered"] for r in rows),
        "projected100_training_wall_ratio": projected_ratio, "median_historical_ratio": median_ratio,
        "cost_projection_is_estimate": True})
    print("PASS discarded17 dense mechanics; pilot_admitted=" + str(admitted), flush=True)


if __name__ == "__main__":
    main()
