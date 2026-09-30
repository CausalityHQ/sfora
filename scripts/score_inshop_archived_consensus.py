#!/usr/bin/env python3
"""Fixed archived control/dense/queue packed-score consensus on TRAIN-held only.

Authority archived-consensus-v1 pins the original 117/123 manifests, two terminal
decisions, six original export/CPU unit specs and training checkpoint_path fields.
held_manifest_sha256 pins canonical JSON of held_manifest/query/gallery together.
The execution mapping contains only this scorer and its stdlib test, relative to
this scripts directory. Numerical helpers come from the original 123 PYTHONPATH.
No original authority constructor, model startup, training or export is invoked.
"""
if not __debug__:
    raise SystemExit("Scoring requires assertions")

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

SEEDS = (179032, 179041)
ARMS = ("control", "dense", "queue")
ORDER = tuple((seed, arm) for seed in SEEDS for arm in ARMS)
METRICS = ("per_query_r1", "per_query_ap")
FILES = {"held.npy", "held.codes.npy", "held.inverse.npy"}
EXECUTION = {"score_inshop_archived_consensus.py", "test_inshop_archived_consensus.py"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path, expected):
    assert Path(path).is_absolute() and sha(path) == expected, f"SHA differs: {path}"
    return json.loads(Path(path).read_text())


def check_code(root, code):
    root = Path(root).resolve()
    for name, expected in code.items():
        path = root / name
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert path.resolve().is_relative_to(root) and sha(path) == expected, f"Source differs: {path}"


def validate_inventory(spec):
    assert spec["schema"] == "archived-consensus-v1"
    assert set(spec["execution"]) == EXECUTION
    assert len(spec["source_manifests"]) == len(spec["decisions"]) == 2
    assert [(e["seed"], e["arm"]) for e in spec["endpoints"]] == list(ORDER)
    units = spec["endpoints"] + [e["cpu"] for e in spec["endpoints"]]
    for field in ("receipt", "log", "unit", "invocation_id"):
        values = [str(Path(u[field]).resolve()) if field in ("receipt", "log") else u[field] for u in units]
        assert len(set(values)) == len(units), f"Duplicate {field}"
    for unit in units:
        assert all(Path(unit[k]).is_absolute() for k in ("receipt", "log"))
        assert unit["unit"] and not unit["unit"].endswith(".service") and unit["invocation_id"]
    for e in spec["endpoints"]:
        assert all(Path(e[k]).is_absolute() for k in ("run", "receipt", "log", "checkpoint_path"))
        assert Path(e["receipt"]) == Path(e["run"]) / "receipt.json"
        assert e["unit"] and not e["unit"].endswith(".service")
        assert Path(e["checkpoint_path"]).name == "resume.pt"


def validate_wire(receipt, endpoint, decision, code, frozen):
    alias = "candidate" if endpoint["arm"] == "dense" else endpoint["arm"]
    assert decision["pass"] is True and decision["decision"] in ("GO", "KILL")
    assert receipt["pass"] is True
    assert (receipt["seed"], receipt["arm"]) == (endpoint["seed"], alias)
    assert receipt["authority_sha256"] == decision["authority_sha256"]
    assert receipt["execution_sha256"] == decision["execution_sha256"]
    assert receipt["source_code"] == code
    inputs = [v for v in decision["inputs"] if (v["seed"], v["arm"]) == (endpoint["seed"], alias)]
    assert len(inputs) == 1
    original = inputs[0]
    assert endpoint["receipt_sha256"] == original["receipt_sha256"]
    assert receipt["checkpoint_sha256"] == endpoint["checkpoint_sha256"] == original["checkpoint_sha256"]
    assert set(receipt["files"]) == FILES and receipt["files"] == original["files"]
    assert receipt["batch"] == 32 and receipt["width"] == 128 and receipt["precision"] == "private_native_fp16"
    assert receipt["full_held_independent_whole_head_packed_exact"] is True
    assert receipt["source_head_rng_flags_preserved"] is True and receipt["optimizer_updates"] == 0
    assert all(receipt[k] is False for k in ("quality_read", "official_read", "claim_eligible",
                                           "public_serving_qualified", "public_latency_measured"))
    assert all(receipt[k] == frozen[k] for k in ("held_manifest", "query", "gallery"))


def validate_cpu(cpu, receipt, endpoint, code):
    spec = endpoint["cpu"]
    assert spec["receipt"] == receipt["cpu_proof"]
    assert spec["receipt_sha256"] == receipt["cpu_authority_sha256"]
    assert spec["log_sha256"] == receipt["cpu_log_sha256"]
    assert cpu["pass"] is True and cpu["source_code"] == code
    assert cpu["optimizer_updates"] == 0 and cpu["quality_read"] is False
    assert all(cpu[k] is True for k in ("strict400_whole_head_fit_packed_reload_exact",
                                       "frozen_roles_complete_state_exact", "cpu_rng_preserved"))
    assert all(cpu[k] == receipt[k] for k in ("authority_sha256", "execution_sha256", "seed", "arm", "boundary",
        "training_receipt_sha256", "checkpoint_sha256", "terminal_state_fingerprint", "whole_sha256",
        "f16_whole_sha256", "head_sha256"))


def validate_roles(frozen, expected):
    assert hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == expected
    rows, query, gallery = (frozen[k] for k in ("held_manifest", "query", "gallery"))
    assert len(rows) == 12599 and len(query) == 6354 and len(gallery) == 6245
    assert all(type(i) is int for i in query + gallery)
    assert sorted(query + gallery) == list(range(12599))
    assert len({r["relative_path"] for r in rows}) == 12599
    assert len({r["product"] for r in rows}) == 1993
    assert {rows[i]["product"] for i in query} == {rows[i]["product"] for i in gallery}


def reduce_rankings(orders, query_labels, gallery_labels):
    """mAP@R truncates each row at its positive count, not the padded width."""
    counts = Counter(gallery_labels)
    assert len(orders) == len(query_labels) and orders
    hits, aps = [], []
    width = max(counts[label] for label in query_labels)
    for order, label in zip(orders, query_labels, strict=True):
        relevant = counts[label]
        assert relevant > 0 and len(order) == width and len(set(order)) == width
        assert all(type(i) is int and 0 <= i < len(gallery_labels) for i in order)
        hits.append(int(gallery_labels[order[0]] == label))
        matched, total = 0, 0.
        for rank, index in enumerate(order[:relevant], 1):
            if gallery_labels[index] == label:
                matched += 1
                total += matched / rank
        aps.append(total / relevant)
    return {"recall_at_1": statistics.mean(hits), "map_at_r": statistics.mean(aps),
            "per_query_r1": hits, "per_query_ap": aps}


def validate_replay(actual, original):
    maximum = {}
    for metric in METRICS:
        a, b = actual[metric], original[metric]
        assert len(a) == len(b) and a
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in (*a, *b))
        maximum[metric] = max(abs(x - y) for x, y in zip(a, b, strict=True))
        assert maximum[metric] <= (0 if metric == "per_query_r1" else 1e-6), f"Replay differs: {metric}"
    return maximum


def quality_gate(deltas, intervals):
    assert set(deltas) == set(SEEDS) and set(intervals) == set(METRICS)
    assert all(set(deltas[s]) == set(METRICS) for s in SEEDS)
    assert all(math.isclose(intervals[m]["mean_delta"], statistics.mean(statistics.mean(deltas[s][m]) for s in SEEDS),
                            rel_tol=0, abs_tol=1e-12) for m in METRICS)
    each_seed = all(statistics.mean(deltas[s]["per_query_r1"]) > 0
                    and statistics.mean(deltas[s]["per_query_ap"]) >= 0 for s in SEEDS)
    bounds = all(all(math.isfinite(v[k]) for k in ("mean_delta", "product_lower95", "product_upper95",
                    "query_lower95", "query_upper95")) and v["mean_delta"] >= .002 and v["product_lower95"] > 0
                 for v in intervals.values())
    return each_seed, bool(each_seed and bounds)


def publish(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(payload)


def packed_quality(pairs, query, gallery, qlabels, glabels, torch):
    assert len(pairs) in (1, 3)
    width = max(Counter(glabels)[label] for label in qlabels)
    orders = []
    with torch.inference_mode():
        for start in range(0, len(query), 128):
            rows = query[start:start + 128]
            combined = None
            for code, inverse in pairs:
                scores = (code[rows] @ code[gallery].T) * inverse[rows, None] * inverse[None, gallery]
                combined = scores if combined is None else combined.add_(scores)
            if len(pairs) == 3:
                combined.div_(3.)
            orders.extend(torch.argsort(combined, dim=1, descending=True, stable=True)[:, :width].tolist())
    return reduce_rankings(orders, qlabels, glabels)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"), "CUDA must be hidden"
    spec = read_json(args.authority, args.authority_sha256)
    validate_inventory(spec)
    own_root = Path(__file__).resolve().parent
    check_code(own_root, spec["execution"])
    codes, roots = [], []
    for unit, size in zip(spec["source_manifests"], (117, 123), strict=True):
        code = read_json(unit["path"], unit["sha256"])
        assert len(code) == size
        root = Path(unit["path"]).resolve().parent
        assert not own_root.is_relative_to(root), "Scorer requires a separate source root"
        assert not args.output.resolve().is_relative_to(root)
        check_code(root, code)
        codes.append(code); roots.append(root)
    assert all(codes[0][n] == codes[1][n] for n in codes[0].keys() & codes[1].keys())
    decisions = [read_json(d["path"], d["sha256"]) for d in spec["decisions"]]
    for d, manifest in zip(decisions, spec["source_manifests"], strict=True):
        assert d["pass"] is True and d["decision"] in ("GO", "KILL")
        assert d["execution_sha256"] == manifest["sha256"]
        assert (d["query_images"], d["gallery_images"], d["held_products"]) == (6354, 6245, 1993)
        assert (d["fit_images"], d["fit_products"], d["updates_per_endpoint"]) == (13283, 2004, 100)
        assert d["quality_read"] == "previously observed In-Shop TRAIN-held only"
        assert all(d[k] is False for k in ("official_read", "claim_eligible", "public_serving_qualified",
                                           "public_latency_measured", "global_production_goal_met"))

    # Only now import the hash-checked original resource/ranking/packing helpers.
    import numpy as np
    import torch
    import export_image_queue_adaptation as export
    assert Path(export.__file__).resolve() == roots[1] / "export_image_queue_adaptation.py"
    assert Path(export.pair.__file__).resolve().is_relative_to(roots[1])
    packing_module = sys.modules[export.pack_int8_unit_embeddings.__module__]
    assert Path(packing_module.__file__).resolve() == roots[1] / "src/sfora/joint_relational_compaction.py"
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    export.resource_unit(120, False, started)
    receipts, inputs = {}, []
    frozen = None
    for endpoint in spec["endpoints"]:
        index = int(endpoint["arm"] == "queue")
        receipt = export.read_unit(endpoint, 300, True)
        if frozen is None:
            frozen = {k: receipt[k] for k in ("held_manifest", "query", "gallery")}
            validate_roles(frozen, spec["held_manifest_sha256"])
        validate_wire(receipt, endpoint, decisions[index], codes[index], frozen)
        cpu = export.read_unit(endpoint["cpu"], 120, False)
        validate_cpu(cpu, receipt, endpoint, codes[index])
        checkpoint = Path(endpoint["checkpoint_path"])
        assert not args.output.resolve().is_relative_to(checkpoint.parent.resolve())
        assert not args.output.resolve().is_relative_to(Path(endpoint["run"]).resolve())
        assert sha(checkpoint) == receipt["checkpoint_sha256"]
        for name, expected in receipt["files"].items():
            assert sha(Path(endpoint["run"]) / name) == expected
        key = endpoint["seed"], endpoint["arm"]
        receipts[key] = receipt
        inputs.append({**endpoint, "files": receipt["files"]})
    # No packed quality is read until every original boundary is authenticated.
    labels = tuple(r["product"] for r in frozen["held_manifest"])
    classes = {label: i for i, label in enumerate(sorted(set(labels)))}
    integer_labels = [classes[label] for label in labels]
    query, gallery = frozen["query"], frozen["gallery"]
    qlabels, glabels = ([integer_labels[i] for i in rows] for rows in (query, gallery))
    arrays = {}
    for endpoint in spec["endpoints"]:
        run = Path(endpoint["run"])
        values = np.load(run / "held.npy", allow_pickle=False)
        codes_array = np.load(run / "held.codes.npy", allow_pickle=False)
        inverse = np.load(run / "held.inverse.npy", allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (12599, 128) and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
        assert codes_array.dtype == np.int8 and codes_array.shape == values.shape
        assert inverse.dtype == np.float16 and inverse.shape == (12599,) and np.isfinite(inverse).all()
        packed = export.pack_int8_unit_embeddings(torch.from_numpy(values))
        assert np.array_equal(codes_array, packed.codes.numpy())
        assert np.array_equal(inverse, packed.inverse_norms.numpy())
        arrays[endpoint["seed"], endpoint["arm"]] = (torch.from_numpy(codes_array).float(), torch.from_numpy(inverse).float())
    quality, replays, seconds = {}, {}, {}
    # All six historical replays must pass before evaluating a mixture.
    for seed in SEEDS:
        quality[seed], replays[seed], seconds[seed] = {}, {}, {}
        for arm in ARMS:
            tick = time.perf_counter()
            quality[seed][arm] = packed_quality([arrays[seed, arm]], query, gallery, qlabels, glabels, torch)
            seconds[seed][arm] = time.perf_counter() - tick
            index = int(arm == "queue")
            alias = "candidate" if arm == "dense" else arm
            replays[seed][arm] = validate_replay(quality[seed][arm], decisions[index]["quality"][str(seed)][alias])
    for seed in SEEDS:
        tick = time.perf_counter()
        quality[seed]["consensus"] = packed_quality([arrays[seed, a] for a in ARMS], query, gallery, qlabels, glabels, torch)
        seconds[seed]["consensus"] = time.perf_counter() - tick
    deltas = {s: {m: [b - a for a, b in zip(quality[s]["control"][m], quality[s]["consensus"][m], strict=True)]
                  for m in METRICS} for s in SEEDS}
    average = {m: [(a + b) / 2 for a, b in zip(*(deltas[s][m] for s in SEEDS), strict=True)] for m in METRICS}
    intervals = {}
    for metric in METRICS:
        delta = np.asarray(average[metric])
        intervals[metric] = {"mean_delta": float(delta.mean())}
        for kind, groups in (("product", np.asarray(qlabels)), ("query", np.arange(6354))):
            # Original helper resets RNG179019: both metrics/bounds share 5000 draws.
            intervals[metric][kind + "_lower95"] = export.pair.bootstrap_lower(delta, groups)
            intervals[metric][kind + "_upper95"] = -export.pair.bootstrap_lower(-delta, groups)
    each_seed, quality_go = quality_gate(deltas, intervals)
    diagnostics = {}
    for seed in SEEDS:
        control, consensus = (quality[seed][a]["per_query_r1"] for a in ("control", "consensus"))
        oracle = [int(any(row)) for row in zip(*(quality[seed][a]["per_query_r1"] for a in ARMS), strict=True)]
        diagnostics[seed] = {"correct_to_wrong": sum(a == 1 and b == 0 for a, b in zip(control, consensus)),
            "wrong_to_correct": sum(a == 0 and b == 1 for a, b in zip(control, consensus)),
            "oracle_union_r1": statistics.mean(oracle), "oracle_union_per_query_r1": oracle,
            "persistent_miss_query_ordinals": [i for i, hit in enumerate(oracle) if not hit],
            "oracle_realizable_quality": False}
    check_code(own_root, spec["execution"])
    for root, code in zip(roots, codes, strict=True):
        check_code(root, code)
    publish(args.output, {"pass": True, "decision": "GO" if quality_go else "KILL", "quality_pass": quality_go,
        "each_seed_quality_pass": each_seed, "authority_sha256": args.authority_sha256,
        "execution": spec["execution"], "source_manifests": spec["source_manifests"], "decisions": spec["decisions"],
        "inputs": inputs, "quality": quality, "replay_max_abs_delta": replays, "per_seed_deltas": deltas,
        "cpu_quality_seconds": seconds, "archived_training_costs": [d["cost"] for d in decisions],
        "cpu_cost_scope": "component replay and three-score mixture reduction; includes ranking, no encoder/public latency",
        "paired_seed_average_deltas": average, "paired_seed_average_intervals": intervals, "diagnostics": diagnostics,
        "arithmetic": "CPU float32 packed codes@codes.T * query inverse * gallery inverse; control+dense+queue in order /3",
        "query_images": 6354, "gallery_images": 6245, "held_products": 1993, "endpoint_count": 6,
        "endpoints_per_seed": 3, "score_weights": [1/3, 1/3, 1/3], "query_batch": 128,
        "bootstrap_draws": 5000, "bootstrap_seed": 179019, "continuation_input_seeds": list(SEEDS),
        "metric_units": "fractions", "interval_scope": "paired equal-seed deltas conditional on shared TRAIN1000 initialization",
        "quality_read": "previously observed In-Shop TRAIN-held only", "diagnostic_multi_model_consensus": True,
        "independent_pretraining_seeds": False, "single_model_promotion": False, "cost_quality_claim": False,
        "official_read": False, "claim_eligible": False, "public_serving_qualified": False,
        "public_latency_measured": False, "global_production_goal_met": False,
        **export.resource_unit(120, False, started)})


if __name__ == "__main__":
    main()
