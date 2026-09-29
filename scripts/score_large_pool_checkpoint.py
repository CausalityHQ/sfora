#!/usr/bin/env python3
"""CPU packed replay and the frozen pooling-only checkpoint quality decision."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import export_large_pool_checkpoint as export


def authority(root, expected, export_expected):
    path = root / "large-pool-score-execution.json"
    assert export.sha(path) == expected
    code = json.loads(path.read_text())
    previous_path = root / "large-pool-export-execution.json"
    assert export.sha(previous_path) == export_expected
    previous = json.loads(previous_path.read_text())
    assert len(code) == 73 and len(previous) == 72
    assert set(code) == set(previous) | {"score_large_pool_checkpoint.py"}
    assert all(code[n] == h for n, h in previous.items())
    export.checked_map(root, code)
    assert Path(__file__).resolve() == root / "score_large_pool_checkpoint.py"
    assert export.sha(export.EXPORT_ROOT / previous_path.name) == export_expected
    assert previous == json.loads((export.EXPORT_ROOT / previous_path.name).read_text())
    export.checked_map(export.EXPORT_ROOT, previous)
    export.closure(root, code)
    return code, previous


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--export-execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    assert not args.output.exists() and not args.output.is_symlink()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    assert root == export.SCORE_ROOT
    pair, sha = export.pair, export.sha
    code, previous = authority(root, args.execution_sha256, args.export_execution_sha256)
    control, frozen, _, _, training, gpu, cpu, inputs, proofs = export.authority(root, args.export_execution_sha256, code)
    run = export.WIRES
    receipt_path = run / "receipt.json"
    receipt_sha = sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["pass"] and receipt["execution_sha256"] == args.export_execution_sha256
    assert receipt["source_code"] == receipt["code"] == previous
    assert receipt["checkpoint_sha256"] == export.CHECKPOINT_SHA and receipt["training_sha256"] == export.TRAIN_SHA
    assert receipt["training_execution_sha256"] == export.TRAIN_CODE
    assert receipt["historical_live_reload_batches"] == 373 and not receipt["lost_original_trained_live_instance_used"]
    assert receipt["original_pilot_status"] == "terminal timeout at373/394; training100 cost PASS"
    assert all(receipt[k] for k in ("two_independent_checkpoint_copies_whole_head_normalized_packed_exact",
        "source_state_rng_flags_data_preserved", "all_named_buffers_including_nonpersistent_preserved",
        "fp32_parameters_fp16_fresh_separate_scopes_heads_outside"))
    assert receipt["held_images"] == 12599 and receipt["batches"] == 394 and receipt["batch_size"] == 32
    assert receipt["optimizer_updates"] == 0 and not receipt["quality_read"]
    assert not any(receipt[k] for k in ("official_read", "claim_eligible", "public_latency_measured", "full_production_goal_met"))
    assert receipt["peak_cuda_allocated_bytes"] < 10_000_000_000
    assert receipt["max_rss_kib"] * 1024 < 8 * 1024**3 and receipt["process_swap_count"] == 0
    assert 0 < receipt["export_wall_seconds"] < 300
    assert receipt["held_manifest"] == frozen["held_manifest"]
    assert receipt["query"] == frozen["query"] and receipt["gallery"] == frozen["gallery"]
    assert receipt["environment"] == gpu["environment"]
    assert receipt["state"]["frozen_sha256"] == [training["frozen_sha256"]] * 2
    assert receipt["state"]["head_sha256"] == [training["final_group_sha256"]["compact_head"]] * 2
    assert receipt["state"]["whole_sha256"][0] == receipt["state"]["whole_sha256"][1]
    assert all(min(v) >= .999 for v in receipt["fp16_fp32_fit_calibration"].values())
    assert sha(export.CPU_PROOF) == receipt["cpu_authority_sha256"]
    proof = json.loads(export.CPU_PROOF.read_text())
    assert proof["pass"] and proof["code"] == previous and proof["execution_sha256"] == args.export_execution_sha256
    assert all(proof[k] for k in ("changed_driver_rejected", "read_only",
        "strict400_native_head_reload_and_direct_whole_calibration_exact",
        "all_named_buffers_including_nonpersistent_preserved", "source_state_rng_flags_data_preserved", "cpu_cuda_rng_unchanged"))
    assert proof["held_images"] == proof["optimizer_updates"] == 0 and not proof["quality_read"]
    assert 0 < proof["seconds"] < 120 and proof["max_rss_kib"] * 1024 < 8 * 1024**3
    assert proof["first_augmented_pixels_sha256"] == inputs["pixels_sha256"][0]
    for key in ("checkpoint_sha256", "training_sha256", "training_execution_sha256", "state", "environment", "fit_manifest", "numerical_flags"):
        assert proof[key] == receipt[key]
    assert proof["fit_manifest"] == frozen["fit_manifest"]
    assert proof["numerical_flags"] == {"deterministic": True, "warn_only": False,
        "matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False,
        "autocast_cuda": False, "autocast_cpu": False}
    assert receipt["proof_hashes"] == {**proof["proof_hashes"], str(export.CPU_PROOF): receipt["cpu_authority_sha256"]}
    # The scorer clone has different path names but identical frozen manifest hashes.
    expected_proofs = {str(export.EXPORT_ROOT / Path(p).name) if Path(p).parent == root else p: h for p, h in proofs.items()}
    assert proof["proof_hashes"] == expected_proofs
    assert all(sha(Path(p)) == h for p, h in receipt["proof_hashes"].items())
    log_path = export.EXPORT_ROOT / "sfora-large-pool-checkpoint-export-v3.log"
    log_sha = sha(log_path)
    log = log_path.read_text()
    assert all(s in log for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
    assert "Finished with result: timeout" not in log
    arrays = []
    for name, key in (("held.npy", "held_sha256"), ("reference-held.npy", "reference_held_sha256")):
        assert sha(run / name) == receipt[key]
        values = np.load(run / name, allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (12599, 128) and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
        arrays.append(values)
    assert np.array_equal(*arrays)
    export.packed_equal(*(torch.from_numpy(v) for v in arrays))
    original_sha = export.sha
    with patch.object(export, "sha", lambda p: "changed" if Path(p).resolve() == Path(__file__).resolve() else original_sha(p)):
        try:
            authority(root, args.execution_sha256, args.export_execution_sha256)
        except AssertionError:
            pass
        else:
            raise AssertionError("changed score driver accepted")
    assert all(sha(control.dataset_root / r["relative_path"]) == r["image_sha256"] for r in frozen["fit_manifest"] + frozen["held_manifest"])
    labels = tuple(r["product"] for r in frozen["held_manifest"])
    dense_prior = json.loads((export.native.INIT / "receipt.json").read_text())["arms"]["pe"]
    # All source/checkpoint/proof/log/array/layout gates precede the first quality call.
    quality, reference = (pair.packed_quality(v, labels, frozen["query"], frozen["gallery"], device=torch.device("cpu")) for v in arrays)
    assert all(np.array_equal(np.asarray(quality[k]), np.asarray(reference[k])) for k in quality)
    products = np.asarray(labels)[frozen["query"]]
    intervals = {}
    for name in ("per_query_r1", "per_query_ap"):
        delta = np.asarray(quality[name]) - np.asarray(dense_prior["quality"][name])
        assert delta.shape == (6354,) and np.isfinite(delta).all()
        intervals[name] = {"mean_delta": float(delta.mean()),
            "product_lower95": pair.bootstrap_lower(delta, products),
            "product_upper95": -pair.bootstrap_lower(-delta, products),
            "query_lower95": pair.bootstrap_lower(delta, np.arange(len(delta))),
            "query_upper95": -pair.bootstrap_lower(-delta, np.arange(len(delta)))}
    quality_go = bool(quality["recall_at_1"] >= .951720176 and quality["map_at_r"] >= .776237120
        and all(v["product_lower95"] > 0 for v in intervals.values()))
    authority(root, args.execution_sha256, args.export_execution_sha256)
    assert sha(receipt_path) == receipt_sha and sha(log_path) == log_sha
    assert all(sha(Path(p)) == h for p, h in receipt["proof_hashes"].items())
    assert sha(run / "held.npy") == receipt["held_sha256"] and sha(run / "reference-held.npy") == receipt["reference_held_sha256"]
    export.closure(root, code)
    pair.smoke.save(args.output, {"pass": True, "decision": "GO" if quality_go else "KILL",
        "procedure": "fixed pooling-only100 checkpoint quality only; evaluation integrity is distinct from decision",
        "quality_pass": quality_go, "checkpoint_retained_on_kill": True, "changed_driver_rejected": True,
        "execution_sha256": args.execution_sha256, "export_execution_sha256": args.export_execution_sha256,
        "export_receipt_sha256": receipt_sha, "export_log_sha256": log_sha,
        "checkpoint_sha256": export.CHECKPOINT_SHA, "training_sha256": export.TRAIN_SHA,
        "cpu_authority_sha256": receipt["cpu_authority_sha256"], "source_code": code,
        "held_sha256": receipt["held_sha256"], "reference_held_sha256": receipt["reference_held_sha256"],
        "dataset": "DeepFashion In-Shop", "split": "previously observed TRAIN-held",
        "fit_images": 13283, "query_images": 6354, "gallery_images": 6245, "held_products": 1993,
        "quality": quality, "reference_quality": reference, "paired_dense_pe_intervals": intervals,
        "floors": {"recall_at_1": .951720176, "map_at_r": .776237120, "both_product_lower95_strictly_positive": True},
        "bootstrap_draws": 5000, "bootstrap_seed": 179019,
        "historical_nonconcurrent_controls_percent": {"Large": [95.6720176, 78.6237120], "DensePE": [95.0739692, 76.3915922]},
        "paired_reference": "original frozen DensePE, not a concurrent same-backbone control",
        "metric_units": "R@1 and mAP@R fractions; delta times100 is percentage points",
        "original_pilot_status": receipt["original_pilot_status"], "lost_original_trained_live_instance_used": False,
        "training_cost": receipt["training_cost"], "quality_read": "In-Shop TRAIN-held only",
        "official_read": False, "claim_eligible": False, "public_latency_measured": False,
        "full_production_goal_met": False, "global_production_goal_met": False})
    print(json.dumps({"decision": "GO" if quality_go else "KILL", "intervals": intervals}), flush=True)


if __name__ == "__main__":
    main()
