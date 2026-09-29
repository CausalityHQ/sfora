#!/usr/bin/env python3
"""CPU replay of a separately authorized fresh validation; never rescues the pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import validate_pe_large_pool_checkpoint as driver


def load_vectors(output, receipt):
    arrays = []
    for name, key in (("loaded-held.npy", "loaded_held_sha256"), ("reference-held.npy", "reference_held_sha256")):
        path = output / name
        assert driver.native.pair.sha(path) == receipt[key], "saved-vector hash differs"
        values = np.load(path, allow_pickle=False)
        assert values.shape == (12599, 128) and values.dtype == np.float32
        assert np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
        arrays.append(values)
    assert np.array_equal(*arrays), "independent saved vectors differ"
    return arrays


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--auditor-sha256", required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    pair = driver.native.pair
    assert pair.sha(Path(__file__)) == args.auditor_sha256
    root = Path(__file__).resolve().parent
    _, frozen, checkpoint, _, _, dense, code = driver.startup(root, args.execution_sha256)
    receipt_path = args.output / "receipt.json"
    assert pair.sha(receipt_path) == args.receipt_sha256
    r = json.loads(receipt_path.read_text())
    assert r["execution_sha256"] == args.execution_sha256
    assert r["checkpoint_sha256"] == pair.sha(checkpoint)
    assert r["original_pilot_status"] == "terminal whole-job timeout; not rescued"
    assert r["quality_read"] == "TRAIN-held fresh extra diagnostic only"
    assert r["optimizer_updates"] == 0
    assert r["official_read"] is False and r["claim_eligible"] is False
    assert r["lost_original_trained_live_instance_used"] is False
    assert 0 < r["peak_cuda_allocated_bytes"] < 10_000_000_000
    assert all(r[k] is True for k in ("all394_fresh_whole_independent_checkpoint_copy_batches_exact", "state_runtime_frozen_complement_preserved", "packed_per_query_exact", "cpu_cuda_rng_unchanged"))
    values, independent = load_vectors(args.output, r)
    labels = tuple(row["product"] for row in frozen["held_manifest"])
    quality = pair.packed_quality(values, labels, frozen["query"], frozen["gallery"], device=torch.device("cpu"))
    reference = pair.packed_quality(independent, labels, frozen["query"], frozen["gallery"], device=torch.device("cpu"))
    assert all(np.array_equal(np.asarray(v), np.asarray(reference[k])) for k, v in quality.items())
    assert all(np.max(np.abs(np.asarray(v) - np.asarray(r["quality"][k]))) < 1e-6 for k, v in quality.items())
    products = np.asarray(labels)[frozen["query"]]
    intervals = {}
    for k in ("per_query_r1", "per_query_ap"):
        delta = np.asarray(quality[k]) - np.asarray(dense["quality"][k])
        intervals[k] = {"mean_delta": float(np.mean(delta))}
        for kind, groups in (("product", products), ("query", np.arange(len(delta)))):
            intervals[k][kind + "_lower95"] = pair.bootstrap_lower(delta, groups)
            intervals[k][kind + "_upper95"] = -pair.bootstrap_lower(-delta, groups)
        assert all(abs(v - r["paired_dense_pe_intervals"][k][name]) < 1e-6 for name, v in intervals[k].items())
    floors = bool(quality["recall_at_1"] >= 0.951720176 and quality["map_at_r"] >= 0.776237120 and all(v["product_lower95"] > 0 for v in intervals.values()))
    assert floors == r["diagnostic_floors_pass"]
    assert pair.sha(receipt_path) == args.receipt_sha256
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert pair.sha(Path(__file__)) == args.auditor_sha256
    pair.smoke.save(args.output / "cpu-audit.json", {"pass": True, "diagnostic_floors_pass": floors, "original_pilot_status": r["original_pilot_status"], "quality": quality, "paired_dense_pe_intervals": intervals, "receipt_sha256": args.receipt_sha256, "execution_sha256": args.execution_sha256, "auditor_sha256": args.auditor_sha256, "claim_eligible": False, "quality_read": "TRAIN-held saved-vector CPU replay only"})
    print("PASS authenticated fresh saved vectors, CPU packed per-query scores, product/query intervals and diagnostic decision; original pilot not rescued")


if __name__ == "__main__":
    main()
