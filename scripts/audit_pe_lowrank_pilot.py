#!/usr/bin/env python3
"""CPU saved-checkpoint/packed-quality replay for the fixed100 PE pilot."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import train_inshop_pe_pair as pair


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
    source = SimpleNamespace(
        root=root,
        output=Path("/home/riomus/runs/sfora-pe-augmented-100-v2"),
        preflight_sha256="41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293",
        cache=Path("/home/riomus/runs/sfora-pe-fullfit-cache-v3"),
        mechanics_dir=Path("/home/riomus/runs/sfora-pe-fp16-smoke-v1"),
        dataset_root=Path("/home/riomus/datasets/inshop_official_standard"),
        large_snapshot=Path(
            "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"
        ),
    )
    frozen = pair.check_startup(source)
    t = json.loads((args.output / "training.json").read_text())
    attempt = json.loads((args.output / "attempt.json").read_text())
    assert all(pair.sha(root / n) == h for n, h in attempt["code"].items())
    prior = json.loads((source.output / "receipt.json").read_text())["arms"]["pe"]
    assert len(t["step_seconds"]) == len(t["losses"]) == len(t["scales"]) == 100
    assert all(np.isfinite(x) for x in t["losses"] + t["step_seconds"])
    assert t["scales"] == [128] * 100 and t["rgb_sha256"] == prior["rgb_sha256"]
    assert (
        np.median(t["step_seconds"][2:]) == r["median_step_3_100_seconds"] <= 0.71769696
    )
    assert r["peak_cuda_allocated_bytes"] < 10_000_000_000
    assert (
        r["updated_gpu_merge_exact"]
        and r["updated_gpu_strict_reload_exact"]
        and r["packed_per_query_parity"]
    )
    assert r["updated_loaded_minimum_cosine"] >= 0.999999
    vision, _ = pair.smoke.load_arm(source, "pe")
    inventory = pair.smoke.freeze_prefix(vision, "pe")
    vision.rope.update_grid(torch.device("cpu"), 14, 14)
    assert (
        pair.smoke.digest(pair.smoke.frozen_state(vision, "pe", inventory))
        == t["frozen_sha256"]
    )
    checkpoint = args.output / "pe.pt"
    assert pair.sha(checkpoint) == r["checkpoint_sha256"]
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    vision.load_state_dict(saved["vision"], strict=True)
    vision.rope.rope.load_state_dict(saved["rope"], strict=True)
    assert torch.equal(vision.rope.freq, saved["rope_freq"])
    assert (
        pair.smoke.digest(pair.smoke.frozen_state(vision, "pe", inventory))
        == t["frozen_sha256"]
    )
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
    labels = tuple(m["product"] for m in frozen["held_manifest"])
    quality = pair.packed_quality(
        values, labels, frozen["query"], frozen["gallery"], device=torch.device("cpu")
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
