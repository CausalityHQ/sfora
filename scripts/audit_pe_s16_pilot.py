#!/usr/bin/env python3
"""CPU saved-checkpoint/packed-quality replay for the native100 S16 full-encoder pilot."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_inshop_pe_pair as pair
import pe_s16_training as l14
import pe_s16_training as pool


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parent
    assert pair.sha(args.output / "receipt.json") == args.receipt_sha256
    r = json.loads((args.output / "receipt.json").read_text())
    source, frozen, prior = l14.control(root)
    t = json.loads((args.output / "training.json").read_text())
    attempt = json.loads((args.output / "attempt.json").read_text())
    cpu = Path("/home/riomus/runs/sfora-pe-s16-gpu-v2/preflight.json")
    assert pair.sha(cpu) == attempt["preflight_sha256"] == t["preflight_sha256"]
    execution = root / "s16-training-execution.json"
    assert pair.sha(execution) == attempt["execution_sha256"] == r["execution_sha256"]
    code = json.loads(execution.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert t["caller_cpu_rng_unchanged"] and t["one_pending_cpu_batch"]
    assert len(t["worker_input_seconds"]) == 100
    assert t["training_wall_seconds_including_fill_drain"] >= sum(t["step_seconds"])
    assert r["frozen_complement_unchanged"] and r["full_held_head_reload_exact"]
    assert (
        r["full_held_independent_whole_encoder_exact"]
        and r["full_live_loaded_native_state_exact"]
    )
    qualified = json.loads(cpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in qualified["code"].items())
    assert t["attributed_sdpa_autograd_nodes"] == qualified["sdpa_autograd_nodes"]
    assert len(qualified["sdpa_autograd_nodes"]) == 13
    mechanics = Path("/home/riomus/runs/sfora-pe-s16-mechanics-v1")
    assert (
        not (mechanics / "pe.pt").exists()
        and not (mechanics / "mechanics-fit.npy").exists()
    )
    assert pair.sha(mechanics / "receipt.json") == attempt["mechanics_receipt_sha256"]
    m = json.loads((mechanics / "receipt.json").read_text())
    mt = json.loads((mechanics / "training.json").read_text())
    assert m["advance"] and m["fit_only_export_path_exact"]
    assert m["execution_sha256"] == r["execution_sha256"]
    assert t["losses"][:17] == mt["losses"] and t["scales"][:17] == mt["scales"]
    assert len(t["step_seconds"]) == len(t["losses"]) == len(t["scales"]) == 100
    assert all(np.isfinite(x) for x in t["losses"] + t["step_seconds"])
    assert t["scales"] == [128] * 100 and t["rgb_sha256"] == prior["rgb_sha256"]
    assert (
        np.median(t["step_seconds"][2:]) == r["median_step_3_100_seconds"] <= 0.71769696
    )
    assert r["peak_cuda_allocated_bytes"] < 10_000_000_000
    assert (
        r["native_gpu_output_exact"]
        and r["updated_gpu_strict_reload_exact"]
        and r["packed_per_query_parity"]
    )
    assert r["updated_loaded_minimum_cosine"] >= 0.999999
    assert all(len(d["data_gradient_norms"]) == 166 for d in t["diagnostics"])
    assert [d["step"] for d in t["diagnostics"]] == [1, 100]
    vision, _ = l14.load()
    pool.freeze(vision)
    assert pair.smoke.digest(pool.frozen_state(vision)) == t["frozen_sha256"]
    checkpoint = args.output / "pe.pt"
    assert pair.sha(checkpoint) == r["checkpoint_sha256"]
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    vision.load_state_dict(saved["vision"], strict=True)
    vision.rope.rope.load_state_dict(saved["rope"], strict=True)
    assert torch.equal(vision.rope.freq, saved["rope_freq"])
    assert pair.smoke.digest(pool.frozen_state(vision)) == t["frozen_sha256"]
    assert (
        pool.runtime_identity(vision)["native_config"]
        == t["runtime_identity"]["native_config"]
    )
    assert pool.runtime_identity(vision)["pool_heads"] == 8
    for name, values in (
        ("full_native_encoder", dict(vision.named_parameters())),
        ("compact_head", saved["head"]),
        ("classifier", {"classifier": saved["classifier"]}),
    ):
        assert pair.smoke.digest(values) == t["final_group_sha256"][name]
        assert t["final_group_sha256"][name] != t["initial_group_sha256"][name]
    assert all(
        torch.isfinite(v).all()
        for group in ("vision", "head", "rope")
        for v in saved[group].values()
    )
    assert saved["bank"].shape == (13283, 128) and saved["classifier"].shape == (
        2004,
        128,
    )
    assert (
        torch.isfinite(saved["bank"]).all()
        and torch.isfinite(saved["classifier"]).all()
    )
    assert torch.allclose(
        saved["bank"].norm(dim=1), torch.ones(13283), atol=1e-5, rtol=0
    )
    path = args.output / "pe.held.npy"
    assert pair.sha(path) == r["held_sha256"]
    values = np.load(path, allow_pickle=False)
    assert (
        values.shape == (12599, 128)
        and values.dtype == np.float32
        and np.isfinite(values).all()
    )
    assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
    live_path = args.output / "pe.live-held.npy"
    assert pair.sha(live_path) == r["live_held_sha256"]
    live = np.load(live_path, allow_pickle=False)
    assert np.array_equal(values, live) and values.shape == live.shape
    labels = tuple(m["product"] for m in frozen["held_manifest"])
    quality = pair.packed_quality(
        values, labels, frozen["query"], frozen["gallery"], device=torch.device("cpu")
    )
    live_quality = pair.packed_quality(
        live, labels, frozen["query"], frozen["gallery"], device=torch.device("cpu")
    )
    assert all(
        np.max(np.abs(np.asarray(v) - np.asarray(live_quality[k]))) < 1e-6
        for k, v in quality.items()
    )
    assert all(
        np.max(np.abs(np.asarray(v) - np.asarray(r["quality"][k]))) < 1e-6
        for k, v in quality.items()
    )
    products = np.asarray(labels)[frozen["query"]]
    intervals = {}
    for k in ("per_query_r1", "per_query_ap"):
        delta = np.asarray(quality[k]) - np.asarray(prior["quality"][k])
        bounds = [
            pair.bootstrap_lower(delta, products),
            -pair.bootstrap_lower(-delta, products),
        ]
        assert (
            np.max(
                np.abs(
                    np.asarray(bounds)
                    - [
                        r["paired_dense_pe_intervals"][k]["product_lower95"],
                        r["paired_dense_pe_intervals"][k]["product_upper95"],
                    ]
                )
            )
            < 1e-6
        )
        intervals[k] = {
            "product95": bounds,
            "query95": [
                pair.bootstrap_lower(delta, np.arange(len(delta))),
                -pair.bootstrap_lower(-delta, np.arange(len(delta))),
            ],
        }
    for key, interval in intervals.items():
        assert (
            np.max(
                np.abs(
                    np.asarray(interval["query95"])
                    - [
                        r["paired_dense_pe_intervals"][key]["query_lower95"],
                        r["paired_dense_pe_intervals"][key]["query_upper95"],
                    ]
                )
            )
            < 1e-6
        )
    advance = bool(
        quality["recall_at_1"] >= 0.951720176
        and quality["map_at_r"] >= 0.776237120
        and all(v["product95"][0] > 0 for v in intervals.values())
    )
    assert advance == r["advance"]
    pair.smoke.save(
        args.output / "cpu-audit.json",
        {
            "pass": True,
            "advance": advance,
            "receipt_sha256": args.receipt_sha256,
            "paired_intervals": intervals,
            "quality_read": "TRAIN-held saved-score replay only",
            "auditor_sha256": pair.sha(Path(__file__)),
        },
    )
    print(
        "PASS actual frozen checkpoint, bank, 100 RGB/scaler records, packed held scores, product/query uncertainty and decision replay",
        flush=True,
    )


if __name__ == "__main__":
    main()
