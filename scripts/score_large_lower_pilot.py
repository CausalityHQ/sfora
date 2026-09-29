#!/usr/bin/env python3
"""Independent CPU packed replay and the frozen two-seed TRAIN pilot decision."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import export_large_lower_pilot as export

EXPORT_CODE = "d2550a6e7cd59916e8d6e9e9a5b9f70dbe60efbbf37aaab3a076b529ca7badb5"


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--execution-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    sha = export.training.mechanics.sha
    code_path = root / "lower-pilot-score-execution.json"
    assert sha(code_path) == args.execution_sha256
    code = json.loads(code_path.read_text())
    previous = json.loads((root / "lower-pilot-export-execution.json").read_text())
    assert set(code) == set(previous) | {"score_large_lower_pilot.py"}
    assert all(code[n] == h for n, h in previous.items()) and all(sha(root / n) == h for n, h in code.items())
    pair = export.old.pair
    helpers = export.old.previous.selected.helpers
    qualities, inputs, deltas, cost = {}, [], {}, {}
    common = None
    for seed in export.training.CONTROLS:
        qualities[seed] = {}
        for arm in ("control", "candidate"):
            with patch.object(export.old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
                _, frozen, _, _, _, training, training_sha = export.authority(root, EXPORT_CODE, seed, arm)
            run = Path(f"/home/riomus/runs/sfora-lower-pilot-wires-{seed}-{arm}-v1")
            receipt = json.loads((run / "receipt.json").read_text())
            assert receipt["pass"] and receipt["seed"] == seed and receipt["arm"] == arm
            assert receipt["training_receipt_sha256"] == training_sha and receipt["training_checkpoint_sha256"] == training["checkpoint_sha256"]
            assert receipt["execution_sha256"] == EXPORT_CODE and receipt["source_code"] == previous
            assert receipt["full_held_independent_whole_encoder_head_packed_exact"] and receipt["source_head_rng_flags_preserved"]
            assert receipt["held_images"] == 12599 and receipt["optimizer_updates"] == 0 and not receipt["quality_read"]
            assert receipt["held_manifest"] == frozen["held_manifest"] and receipt["query"] == frozen["query"] and receipt["gallery"] == frozen["gallery"]
            log = (Path("/home/riomus/runs/sfora-large-lower-pilot-export-source-v1") / f"lower-pilot-wires-{seed}-{arm}-v1.log").read_text()
            assert all(s in log for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
            arrays = []
            for name, key in (("held.npy", "held_sha256"), ("reference-held.npy", "reference_held_sha256")):
                assert sha(run / name) == receipt[key]
                values = np.load(run / name, allow_pickle=False)
                assert values.dtype == np.float32 and values.shape == (12599, 128) and np.isfinite(values).all()
                assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
                arrays.append(values)
            assert np.array_equal(*arrays)
            labels = tuple(r["product"] for r in frozen["held_manifest"])
            layout = (labels, frozen["query"], frozen["gallery"])
            assert common is None or common == layout
            common = layout
            quality = pair.packed_quality(arrays[0], labels, frozen["query"], frozen["gallery"], device=torch.device("cpu"))
            qualities[seed][arm] = quality
            inputs.append({"seed": seed, "arm": arm, "receipt_sha256": sha(run / "receipt.json"),
                           "held_sha256": receipt["held_sha256"], "checkpoint_sha256": receipt["checkpoint_sha256"]})
            if arm == "candidate":
                cost[seed] = {k: training[k] for k in ("training_wall_seconds", "median_step_seconds", "images_per_second", "peak_cuda_allocated_bytes", "training_wall_ratio", "median_step_ratio")}
        deltas[seed] = {name: np.asarray(qualities[seed]["candidate"][name]) - np.asarray(qualities[seed]["control"][name])
                        for name in ("per_query_r1", "per_query_ap")}
    groups = np.asarray(common[0])[common[1]]
    intervals = {}
    for name in ("per_query_r1", "per_query_ap"):
        delta = np.mean([deltas[seed][name] for seed in export.training.CONTROLS], axis=0)
        intervals[name] = {"mean_delta": float(delta.mean())}
        for kind, labels in (("product", groups), ("query", np.arange(len(delta)))):
            intervals[name][kind + "_lower95"] = pair.bootstrap_lower(delta, labels)
            intervals[name][kind + "_upper95"] = -pair.bootstrap_lower(-delta, labels)
    each_seed = all(deltas[seed]["per_query_r1"].mean() > 0 and deltas[seed]["per_query_ap"].mean() >= 0 for seed in deltas)
    quality_go = each_seed and all(v["mean_delta"] >= .002 and v["product_lower95"] > 0 for v in intervals.values())
    cost_go = all(v["training_wall_ratio"] <= 1.50 and v["median_step_ratio"] <= 1.50 for v in cost.values())
    assert all(sha(root / n) == h for n, h in code.items())
    result = {"pass": True, "decision": "GO" if quality_go and cost_go else "KILL",
        "dataset": "DeepFashion In-Shop", "split": "previously observed TRAIN-held",
        "fit_images": 13283, "fit_products": 2004, "query_images": 6354, "gallery_images": 6245,
        "held_products": 1993, "updates_per_seed": 100, "seeds": list(export.training.CONTROLS),
        "execution_sha256": args.execution_sha256, "inputs": inputs, "quality": qualities,
        "paired_seed_average_intervals": intervals, "cost": cost, "each_seed_quality_pass": each_seed,
        "quality_pass": bool(quality_go), "cost_pass": cost_go, "bootstrap_draws": 5000, "bootstrap_seed": 179019,
        "interval_scope": "conditional on the two frozen seeds; paired product/query resampling",
        "cost_denominator": "archival matched controls, not contemporaneous overhead",
        "metric_units": "R@1 and mAP@R fractions; multiply deltas by100 for percentage points",
        "quality_read": "In-Shop TRAIN-held only", "official_read": False, "claim_eligible": False,
        "public_latency_measured": False, "global_production_goal_met": False}
    pair.smoke.save(args.output, result)
    print(json.dumps({"decision": result["decision"], "intervals": intervals, "each_seed_quality_pass": each_seed, "cost_pass": cost_go}), flush=True)


if __name__ == "__main__":
    main()
