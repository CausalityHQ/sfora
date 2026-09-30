#!/usr/bin/env python3
"""Authenticate four fresh native wires, replay packed CPU ranks, paired image-queue gate.

--wires-authority-sha256 pins image-queue-held-wires.json: schema equals
image-queue-held-wires-v1, authority_sha256, wires in the training endpoint order.
Each wire has seed, arm, run and receipt/log/resource fields documented by
the exporter, plus cpu with those same fields for its fit-only proof.json.
Freeze this collection only after original CPU/export units have been collected.
"""
if not __debug__:
    raise SystemExit("Scoring requires assertions")

import time
UNIT_STARTED = time.perf_counter()

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

import export_image_queue_adaptation as export

METRICS = ("per_query_r1", "per_query_ap")


def averaged_deltas(quality):
    assert set(quality) == set(export.SEEDS)
    deltas = {}
    for seed in export.SEEDS:
        assert set(quality[seed]) == {"control", "queue"}
        deltas[seed] = {}
        for metric in METRICS:
            a, b = (quality[seed][arm][metric] for arm in ("control", "queue"))
            assert len(a) == len(b) == 6354
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in (*a, *b))
            deltas[seed][metric] = [y - x for x, y in zip(a, b, strict=True)]
    average = {m: [(a + b) / 2 for a, b in zip(deltas[179032][m], deltas[179041][m], strict=True)] for m in METRICS}
    return deltas, average


def quality_gate(deltas, intervals):
    assert set(deltas) == set(export.SEEDS) and set(intervals) == set(METRICS)
    assert all(math.isclose(intervals[m]["mean_delta"], statistics.mean(statistics.mean(deltas[s][m]) for s in export.SEEDS),
                            rel_tol=0, abs_tol=1e-12) for m in METRICS)
    each_seed = all(statistics.mean(deltas[s]["per_query_r1"]) > 0
                    and statistics.mean(deltas[s]["per_query_ap"]) >= 0 for s in export.SEEDS)
    bounds = all(all(math.isfinite(v[k]) for k in ("mean_delta", "product_lower95", "product_upper95", "query_lower95", "query_upper95"))
                 and v["mean_delta"] >= .002 and v["product_lower95"] > 0 for v in intervals.values())
    return each_seed, bool(each_seed and bounds)


def validate_wire(receipt, endpoint, training, spec, code, authority_sha, frozen):
    assert receipt["intervention"] == export.driver.METHOD
    assert receipt["pass"] and receipt["authority_sha256"] == authority_sha
    assert receipt["execution_sha256"] == spec["execution_sha256"] and receipt["source_code"] == code
    assert (receipt["seed"], receipt["arm"]) == (endpoint["seed"], endpoint["arm"])
    assert receipt["training_receipt_sha256"] == endpoint["receipt_sha256"]
    assert receipt["checkpoint_sha256"] == endpoint["checkpoint_sha256"] == training["checkpoint_sha256"]
    assert receipt["terminal_state_fingerprint"] == endpoint["terminal_state_fingerprint"]
    assert receipt["boundary"] == training["boundary"]
    assert receipt["batch"] == 32 and receipt["width"] == 128 and receipt["precision"] == export.PRECISION
    assert receipt["full_held_independent_whole_head_packed_exact"] and receipt["source_head_rng_flags_preserved"]
    assert receipt["optimizer_updates"] == 0 and receipt["quality_read"] is False
    assert receipt["official_read"] is False and receipt["claim_eligible"] is False
    assert receipt["public_serving_qualified"] is False and receipt["public_latency_measured"] is False
    assert all(receipt[k] == frozen[k] for k in ("held_manifest", "query", "gallery"))
    assert set(receipt["files"]) == {"held.npy", "held.codes.npy", "held.inverse.npy"}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--authority-sha256", required=True)
    p.add_argument("--wires-authority-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists() and not args.output.is_symlink()
    assert not args.output.resolve().is_relative_to(export.TRAIN_ROOT.resolve())
    export.resource_unit(120, False, UNIT_STARTED)
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    spec, code, endpoints, costs, _, _, _, _, frozen = export.authority(root, args.authority_sha256)
    collection = export.training.read_authority(root / "image-queue-held-wires.json", args.wires_authority_sha256)
    assert collection["schema"] == "image-queue-held-wires-v1" and collection["authority_sha256"] == args.authority_sha256
    assert [(w["seed"], w["arm"]) for w in collection["wires"]] == list(export.ORDER)
    export.validate_distinct(spec["endpoints"] + [spec[k] for k in ("cpu", "startup", "mechanics")]
        + collection["wires"] + [w["cpu"] for w in collection["wires"]])
    arrays, inputs = {}, []
    # Close every checkpoint, receipt, original log and saved wire before scoring.
    for wire, endpoint in zip(collection["wires"], spec["endpoints"], strict=True):
        run = Path(wire["run"])
        assert Path(wire["receipt"]) == run / "receipt.json"
        receipt = export.read_unit(wire, 300, True)
        validate_wire(receipt, endpoint, endpoints[wire["seed"], wire["arm"]], spec, code, args.authority_sha256, frozen)
        cpu_spec = wire["cpu"]
        assert Path(cpu_spec["receipt"]) == Path(receipt["cpu_proof"])
        assert cpu_spec["receipt_sha256"] == receipt["cpu_authority_sha256"]
        assert cpu_spec["log_sha256"] == receipt["cpu_log_sha256"]
        cpu = export.read_unit(cpu_spec, 120, False)
        assert cpu["pass"] and cpu["source_code"] == code and cpu["optimizer_updates"] == 0 and cpu["quality_read"] is False
        assert cpu["strict400_whole_head_fit_packed_reload_exact"] and cpu["frozen_roles_complete_state_exact"]
        assert cpu["cpu_rng_preserved"]
        assert all(cpu[k] == receipt[k] for k in ("intervention", "authority_sha256", "execution_sha256", "seed", "arm", "boundary",
            "training_receipt_sha256", "checkpoint_sha256", "terminal_state_fingerprint", "whole_sha256", "f16_whole_sha256", "head_sha256"))
        for name, expected in receipt["files"].items():
            assert export.sha(run / name) == expected
        values = np.load(run / "held.npy", allow_pickle=False)
        codes = np.load(run / "held.codes.npy", allow_pickle=False)
        inverse = np.load(run / "held.inverse.npy", allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (12599, 128) and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
        assert codes.dtype == np.int8 and codes.shape == values.shape
        assert inverse.dtype == np.float16 and inverse.shape == (12599,)
        packed = export.pack_int8_unit_embeddings(torch.from_numpy(values))
        assert np.array_equal(codes, packed.codes.numpy()) and np.array_equal(inverse, packed.inverse_norms.numpy())
        arrays[wire["seed"], wire["arm"]] = values
        inputs.append({"seed": wire["seed"], "arm": wire["arm"], "receipt_sha256": wire["receipt_sha256"],
                       "checkpoint_sha256": receipt["checkpoint_sha256"], "files": receipt["files"]})
    labels = tuple(r["product"] for r in frozen["held_manifest"])
    quality = {s: {a: export.pair.packed_quality(arrays[s, a], labels, frozen["query"], frozen["gallery"], device=torch.device("cpu"))
                   for a in ("control", "queue")} for s in export.SEEDS}
    deltas, average = averaged_deltas(quality)
    intervals = {}
    for metric in METRICS:
        delta = np.asarray(average[metric])
        intervals[metric] = {"mean_delta": float(delta.mean())}
        for kind, groups in (("product", np.asarray(labels)[frozen["query"]]), ("query", np.arange(6354))):
            # Existing helper resets RNG179019 for each call: same 5000 draws,
            # sorted groups, both metrics and both interval endpoints per scheme.
            intervals[metric][kind + "_lower95"] = export.pair.bootstrap_lower(delta, groups)
            intervals[metric][kind + "_upper95"] = -export.pair.bootstrap_lower(-delta, groups)
    each_seed, quality_go = quality_gate(deltas, intervals)
    assert all(export.sha(root / n) == h for n, h in code.items())
    export.publish(args.output, {"pass": True, "decision": "GO" if quality_go else "KILL",
        "authority_sha256": args.authority_sha256, "wires_authority_sha256": args.wires_authority_sha256,
        "execution_sha256": spec["execution_sha256"], "inputs": inputs, "quality": quality,
        "paired_seed_average_intervals": intervals, "each_seed_quality_pass": each_seed,
        "quality_pass": quality_go, "cost_pass": True, "cost": costs,
        "bootstrap_draws": 5000, "bootstrap_seed": 179019,
        "query_images": 6354, "gallery_images": 6245, "fit_images": 13283, "fit_products": 2004, "held_products": 1993,
        "updates_per_endpoint": 100, "continuation_input_seeds": list(export.SEEDS),
        "intervention": export.driver.METHOD,
        "negative_scope": "a negative closes this fixed recipe only",
        "interval_scope": "equal-seed per-query deltas; paired product/query resampling conditional on shared trained source",
        "independent_pretraining_seeds": False, "intermediate_checkpoint_selection": False,
        "cost_denominator": "fresh contemporaneous matched image-queue100 controls for each seed",
        "metric_units": "fractions; multiply deltas by100 for percentage points",
        "quality_read": "previously observed In-Shop TRAIN-held only", "official_read": False,
        "claim_eligible": False, "public_serving_qualified": False, "public_latency_measured": False,
        "global_production_goal_met": False, **export.resource_unit(120, False, UNIT_STARTED)})


if __name__ == "__main__":
    main()
