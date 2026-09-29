#!/usr/bin/env python3
"""Fixed dense10 100-update pilot; matched to authenticated archived controls."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import train_large_dense_mechanics as mechanics

qualification = mechanics.qualification

dense, pilot = qualification.dense, qualification.pilot
old = pilot.old
MECHANICS_ROOT = Path("/home/riomus/runs/sfora-dense-mechanics-source-v2")
MECHANICS_CODE = "5525632bd09d728f2385205841c75de6ba2337a6f375a88bf7b1ded9777e3453"
MECHANICS_SHA = "eb572f3b5b224a690e3cb28bef46b6b04ddd757621972f46b511f16b4f535a05"


def startup(root, expected, seed):
    sha = pilot.mechanics.sha
    assert sha(root / "dense-pilot-execution.json") == expected
    code = json.loads((root / "dense-pilot-execution.json").read_text())
    previous = json.loads((root / "dense-mechanics-execution.json").read_text())
    assert set(code) == set(previous) | {"train_large_dense_pilot.py"}
    assert all(code[n] == h for n, h in previous.items()) and all(sha(root / n) == h for n, h in code.items())
    assert sha(MECHANICS_ROOT / "mechanics-v2.json") == MECHANICS_SHA
    qualified = json.loads((MECHANICS_ROOT / "mechanics-v2.json").read_text())
    assert qualified["pass"] and qualified["pilot_admitted"] and qualified["updates"] == 17
    assert qualified["execution_sha256"] == MECHANICS_CODE and qualified["source_code"] == previous
    assert qualified["native400_whole_head_packed_disk_reload_exact"] and qualified["new32_weights_changed"]
    assert qualified["frozen_parameters_and_nonpersistent_buffers_unchanged"] and qualified["rng_preserved"]
    assert not qualified["quality_read"] and not qualified["checkpoint_saved"]
    log = (MECHANICS_ROOT / "mechanics-v2.log").read_text()
    assert all(s in log for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
    helpers = old.previous.selected.helpers
    with patch.object(old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
        mechanics.startup(root, MECHANICS_CODE)
        values, _ = qualification.startup(root, mechanics.CPU_CODE, seed)
    return values, code, qualified


def main():
    assert __debug__
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--seed", type=int, choices=tuple(pilot.CONTROLS), required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    (control, native, prior, proof, _, archived, _), code, qualified = startup(root, args.execution_sha256, args.seed)
    if args.startup_only:
        assert not torch.cuda.is_available()
        state = dense.fresh(control, native, proof, "cpu")
        assert pilot.initial_fingerprint(state, args.seed) == archived["initial_state_sha256"]
        dense.verify_optimizer(state)
        original_sha = pilot.mechanics.sha
        with patch.object(pilot.mechanics, "sha", lambda p: "changed" if Path(p).resolve() == Path(__file__).resolve() else original_sha(p)):
            qualification.rejects(lambda: startup(root, args.execution_sha256, args.seed))
        old.pair.smoke.save(args.output, {"pass": True, "execution_sha256": args.execution_sha256,
            "seed": args.seed, "initial_state_sha256": archived["initial_state_sha256"],
            "changed_driver_rejected": True, "optimizer_updates": 0, "quality_read": False})
        return
    assert torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    args.output.mkdir()
    torch.cuda.reset_peak_memory_stats()
    state = dense.fresh(control, native, proof, "cuda")
    assert pilot.initial_fingerprint(state, args.seed) == archived["initial_state_sha256"]
    frozen = {n: p.detach().cpu().clone() for n, p in dense.frozen_state(state["model"]).items()}
    new_before = {n: p.detach().cpu().clone() for n, p in state["model"].named_parameters()
                  if n.startswith(("encoder.layers.10.", "encoder.layers.11."))}
    assert len(new_before) == 32
    rng = old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    target = state["target"].cpu().numpy()
    batches = old.coverage.schedule(target, seed=args.seed)
    assert batches.shape == (100, 64)
    counts = np.bincount(target)
    rows = []
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
        if args.seed == 179032 and step <= 17:
            expected = qualified["steps"][step - 1]
            assert all(row[k] == expected[k] for k in row), "dense mechanics replay differs"
            assert all(v == expected[k] for k, v in gradient_audit.items())
        dense.verify_optimizer(state)
        assert state["counter"] == step and torch.cuda.max_memory_allocated() < 10_000_000_000
        torch.cuda.synchronize()
        row.update(gradient_audit, seconds=time.perf_counter() - tick, rgb_sha256=rgb,
            pixels_sha256=pixel_sha, rank_active_before=active, rank_recovered=not active)
        rows.append(row)
        print(json.dumps(row), flush=True)
    wall = time.perf_counter() - started
    median = statistics.median(r["seconds"] for r in rows[2:])
    training = {"seed": args.seed, "updates": 100, "steps": rows, "training_wall_seconds": wall, "median_step_seconds": median,
        "images_per_second": 6400 / wall, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "quality_read": False, "initial_state_sha256": archived["initial_state_sha256"],
        "training_wall_ratio": wall / archived["training_wall_seconds"],
        "median_step_ratio": median / statistics.median(r["seconds"] for r in archived["steps"][2:])}
    old.pair.smoke.save(args.output / "training.json", training)
    assert rng == old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    dense.verify_frozen(state["model"], frozen)
    assert all(not torch.equal(p.detach().cpu(), new_before[n]) for n, p in state["model"].named_parameters() if n in new_before)
    calibration = pixels[:2].cuda()
    model, head = state["model"].eval(), state["head"].eval()
    with torch.no_grad():
        pooled = old.previous.training.fp16(model, calibration)
        compact = F.normalize(old.pair.smoke.compact_head_features(pooled, head), dim=1)
    path = args.output / "native.pt"
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
    admitted = training["training_wall_ratio"] <= 1.50 and training["median_step_ratio"] <= 1.50
    old.pair.smoke.save(args.output / "receipt.json", {**training, "pass": True, "cost_pass": admitted,
        "execution_sha256": args.execution_sha256, "source_code": code, "cpu_proofs": mechanics.CPU_PROOFS,
        "source_checkpoint_sha256": old.coverage.teacher.TEACHER_SHA,
        "checkpoint_sha256": pilot.mechanics.sha(path),
        "updated_whole_sha256": old.pair.smoke.digest(old.coverage.trained.base.whole_state(model)),
        "updated_head_sha256": old.pair.smoke.digest(head.state_dict()),
        "strict400_reload_whole_head_packed_exact": True,
        "checkpoint_saved": True, "control_receipt_sha256": pilot.CONTROLS[args.seed][1],
        "native400_whole_head_packed_disk_reload_exact": True, "new32_weights_changed": True,
        "frozen_parameters_and_nonpersistent_buffers_unchanged": True, "rng_preserved": True,
        "peak_cuda_allocated_bytes": peak, "recovered_rank_updates": sum(r["rank_recovered"] for r in rows),
        "archival_control_cost": True, "cost_projection_is_estimate": False})
    print("PASS fixed100 dense native integrity; cost_pass=" + str(admitted), flush=True)


if __name__ == "__main__":
    main()
