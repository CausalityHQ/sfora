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

import export_large_dense_pilot as export

EXPORT_ROOT = Path("/home/riomus/runs/sfora-dense-pilot-export-source-v1")
CONTROL_EXPORT_CODE = "d2550a6e7cd59916e8d6e9e9a5b9f70dbe60efbbf37aaab3a076b529ca7badb5"


def authority(root, expected):
    sha = export.sha
    code_path = root / "dense-pilot-score-execution.json"
    assert sha(code_path) == expected
    code = json.loads(code_path.read_text())
    previous = json.loads((root / "dense-pilot-export-execution.json").read_text())
    assert set(code) == set(previous) | {"score_large_dense_pilot.py"}
    assert all(code[n] == h for n, h in previous.items()) and all(sha(root / n) == h for n, h in code.items())
    assert len(previous) == 101 and len(code) == 102
    return code, previous


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--execution-sha256", required=True)
    p.add_argument("--export-execution-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    sha = export.sha
    code, previous = authority(root, args.execution_sha256)
    pair = export.old.pair
    helpers = export.old.previous.selected.helpers
    assert sha(root / "dense-pilot-export-execution.json") == args.export_execution_sha256
    assert sha(EXPORT_ROOT / "dense-pilot-export-execution.json") == args.export_execution_sha256
    assert previous == json.loads((EXPORT_ROOT / "dense-pilot-export-execution.json").read_text())
    assert all(sha(EXPORT_ROOT / n) == h for n, h in previous.items())
    control_root = Path("/home/riomus/runs/sfora-large-lower-pilot-export-source-v1")
    assert sha(control_root / "lower-pilot-export-execution.json") == CONTROL_EXPORT_CODE
    control_code = json.loads((control_root / "lower-pilot-export-execution.json").read_text())
    assert len(control_code) == 96 and all(sha(control_root / n) == h for n, h in control_code.items())
    assert all(previous[n] == h for n, h in control_code.items() if n in previous)
    pinned = {
        179032: ("6077003088cac1865c9a9731aa88058cf47f713f27c2fc7009ea8fc2af61a0cd",
                 "34635dc46718e4470ed0f63641e80cd2377f1a7d8b92e09889c85f38759c7a26",
                 "947ad6c51de596be664883c44c3801a7c94c5f91492ba26ea69afa81ba33d08d"),
        179041: ("8d6c5972d7e42f6360cbbcaa8e61682b7caf1d29ae875fb0db3e213c923a708a",
                 "9c7d66215adf93c617c0db05f7f71dac4f50e26697c90c6aa7e7ea0f659d73ae",
                 "9572012b36bf445d001cc9829ce143980cfe05e27e37ab2ddcaf4b4387f7aadc"),
    }
    qualities, inputs, deltas, cost, wires = {}, [], {}, {}, {}
    common = None
    # Authenticate every training job and all four complete wires before scoring.
    for seed in export.CONTROLS:
        with patch.object(export.old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
            _, frozen, _, _, _, training, training_sha = export.authority(root, args.export_execution_sha256, seed, "candidate")
            (_, _, _, _, _, archived, _), _, _ = export.training.startup(root, export.TRAIN_CODE, seed)
        qualities[seed] = {}
        for arm in ("control", "candidate"):
            run = Path(f"/home/riomus/runs/sfora-{'lower' if arm == 'control' else 'dense'}-pilot-wires-{seed}-{arm}-v1")
            receipt_sha = sha(run / "receipt.json")
            receipt = json.loads((run / "receipt.json").read_text())
            assert receipt["pass"] and receipt["seed"] == seed and receipt["arm"] == arm
            if arm == "control":
                assert (receipt_sha, receipt["held_sha256"], receipt["checkpoint_sha256"]) == pinned[seed]
                assert receipt["training_receipt_sha256"] == export.CONTROLS[seed][1]
                assert receipt["training_checkpoint_sha256"] == archived["checkpoint_sha256"]
                assert receipt["execution_sha256"] == CONTROL_EXPORT_CODE and receipt["source_code"] == control_code
                log_path = control_root / f"lower-pilot-wires-{seed}-control-v1.log"
            else:
                assert receipt["training_receipt_sha256"] == training_sha
                assert receipt["training_checkpoint_sha256"] == training["checkpoint_sha256"]
                assert receipt["checkpoint_sha256"] == training["checkpoint_sha256"]
                assert receipt["execution_sha256"] == args.export_execution_sha256 and receipt["source_code"] == previous
                log_path = EXPORT_ROOT / f"sfora-dense-pilot-export-{seed}-candidate-v1.log"
                cpu_path = Path(f"/home/riomus/runs/sfora-dense-pilot-source-{seed}-candidate-v1/proof.json")
                assert sha(cpu_path) == receipt["cpu_authority_sha256"]
                cpu = json.loads(cpu_path.read_text())
                assert cpu["pass"] and cpu["changed_driver_rejected"] and cpu["code"] == previous
                assert cpu["training_receipt_sha256"] == training_sha and cpu["training_checkpoint_sha256"] == training["checkpoint_sha256"]
                assert cpu["teacher_checkpoint_sha256"] == training["checkpoint_sha256"]
                assert cpu["teacher_whole_sha256"] == training["updated_whole_sha256"]
                assert cpu["teacher_head_sha256"] == training["updated_head_sha256"]
                assert cpu["strict400_native_head_reload_and_direct_whole_calibration_exact"]
                assert cpu["prefix_data_mutation_rejected_at_exit"] and cpu["cpu_cuda_rng_unchanged"]
                assert cpu["read_only"] and cpu["optimizer_updates"] == 0 and not cpu["quality_read"]
            assert receipt["full_held_independent_whole_encoder_head_packed_exact"] and receipt["source_head_rng_flags_preserved"]
            assert receipt["held_images"] == 12599 and receipt["optimizer_updates"] == 0 and not receipt["quality_read"]
            assert not receipt["official_read"] and not receipt["claim_eligible"]
            assert receipt["peak_cuda_allocated_bytes"] < 10_000_000_000
            assert receipt["held_manifest"] == frozen["held_manifest"] and receipt["query"] == frozen["query"] and receipt["gallery"] == frozen["gallery"]
            log = log_path.read_text()
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
            wires[seed, arm] = arrays[0]
            inputs.append({"seed": seed, "arm": arm, "receipt_sha256": receipt_sha,
                           "held_sha256": receipt["held_sha256"], "checkpoint_sha256": receipt["checkpoint_sha256"]})
        cost[seed] = {k: training[k] for k in ("training_wall_seconds", "median_step_seconds", "images_per_second", "peak_cuda_allocated_bytes", "training_wall_ratio", "median_step_ratio")}
    original_sha = export.training.pilot.mechanics.sha
    with patch.object(export.training.pilot.mechanics, "sha", lambda f: "changed" if Path(f).resolve() == Path(__file__).resolve() else original_sha(f)):
        with patch.object(export, "sha", export.training.pilot.mechanics.sha):
            try:
                authority(root, args.execution_sha256)
            except AssertionError:
                pass
            else:
                raise AssertionError("changed score driver accepted")
    for seed in export.CONTROLS:
        for arm in ("control", "candidate"):
            qualities[seed][arm] = pair.packed_quality(wires[seed, arm], common[0], common[1], common[2], device=torch.device("cpu"))
        deltas[seed] = {name: np.asarray(qualities[seed]["candidate"][name]) - np.asarray(qualities[seed]["control"][name])
                        for name in ("per_query_r1", "per_query_ap")}
    groups = np.asarray(common[0])[common[1]]
    intervals = {}
    for name in ("per_query_r1", "per_query_ap"):
        delta = np.mean([deltas[seed][name] for seed in export.CONTROLS], axis=0)
        intervals[name] = {"mean_delta": float(delta.mean())}
        for kind, labels in (("product", groups), ("query", np.arange(len(delta)))):
            intervals[name][kind + "_lower95"] = pair.bootstrap_lower(delta, labels)
            intervals[name][kind + "_upper95"] = -pair.bootstrap_lower(-delta, labels)
    each_seed = all(deltas[seed]["per_query_r1"].mean() > 0 and deltas[seed]["per_query_ap"].mean() >= 0 for seed in deltas)
    quality_go = each_seed and all(v["mean_delta"] >= .002 and v["product_lower95"] > 0 for v in intervals.values())
    cost_go = all(v["training_wall_ratio"] <= 1.50 and v["median_step_ratio"] <= 1.50 for v in cost.values())
    assert all(sha(root / n) == h for n, h in code.items())
    result = {"pass": True, "decision": "GO" if quality_go and cost_go else "KILL",
        "procedure": "fixed dense10 100-update two-seed TRAIN pilot only", "changed_driver_rejected": True,
        "dataset": "DeepFashion In-Shop", "split": "previously observed TRAIN-held",
        "fit_images": 13283, "fit_products": 2004, "query_images": 6354, "gallery_images": 6245,
        "held_products": 1993, "updates_per_seed": 100, "seeds": list(export.CONTROLS),
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
