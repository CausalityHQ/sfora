#!/usr/bin/env python3
"""Predeclared CPU secondary read of the four saved raw encoder states."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

import probe_inshop_pe_training_smoke as smoke
from compare_inshop_sop_warmstart_100 import packed_quality
from diagnose_inshop_pe_learning_attribution import (
    PRIOR_PREFLIGHT,
    PRIOR_SHA,
    golden_parity,
)
from score_inshop_crop_view_pair import bootstrap_lower


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("attribution-dir", "training-dir", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--receipt-sha256", required=True)
    args = p.parse_args()
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    started = time.perf_counter()
    assert smoke.sha(args.attribution_dir / "receipt.json") == args.receipt_sha256
    receipt = json.loads((args.attribution_dir / "receipt.json").read_text())
    assert receipt["prior_stop_preserved"] and not receipt["claim_eligible"]
    assert set(receipt["golden_cosine_min"]) == {"large", "pe"}
    assert min(receipt["golden_cosine_min"].values()) >= 0.999999
    assert smoke.sha(args.training_dir / "preflight.json") == PRIOR_PREFLIGHT
    assert smoke.sha(args.training_dir / "receipt.json") == PRIOR_SHA
    prior = json.loads((args.training_dir / "receipt.json").read_text())
    frozen = json.loads((args.training_dir / "preflight.json").read_text())
    labels = tuple(r["product"] for r in frozen["held_manifest"])
    products = np.asarray([labels[i] for i in frozen["query"]])
    args.output.mkdir(exist_ok=False)
    quality, hashes = {}, {}
    for arm in ("large", "pe"):
        for state in ("0", "1"):
            key = arm + "." + state
            path = args.attribution_dir / (arm + ".vision" + state + ".npy")
            hashes[key] = smoke.sha(path)
            assert hashes[key] == receipt["reports"][key]["features_sha256"]
            values = np.load(path, allow_pickle=False)
            assert values.shape == (12599, 1280) and values.dtype == np.float32
            assert np.isfinite(values).all()
            raw = values[:, :1024].copy()
            assert np.allclose(np.linalg.norm(raw, axis=1), 1, atol=1e-5, rtol=0)
            if state == "1":
                assert (
                    smoke.sha(args.training_dir / (arm + ".held.npy"))
                    == prior["arms"][arm]["held_sha256"]
                )
                old = np.load(
                    args.training_dir / (arm + ".held.npy"), allow_pickle=False
                )
                new = values[:, 1152:1280].copy()
                q = packed_quality(
                    new,
                    labels,
                    frozen["query"],
                    frozen["gallery"],
                    device=torch.device("cpu"),
                )
                golden_parity(old, new, receipt["quality"][arm + ".11"], q)
            quality[key] = packed_quality(
                raw,
                labels,
                frozen["query"],
                frozen["gallery"],
                device=torch.device("cpu"),
            )
            smoke.save(args.output / (key + ".json"), quality[key])
    contrasts = {}
    for metric, field in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        arrays = {k: np.asarray(v[field]) for k, v in quality.items()}
        deltas = {
            arm + ".change": arrays[arm + ".1"] - arrays[arm + ".0"]
            for arm in ("large", "pe")
        }
        deltas.update(
            {
                "pe_minus_large.state" + s: arrays["pe." + s] - arrays["large." + s]
                for s in ("0", "1")
            }
        )
        deltas["pe_minus_large.change"] = deltas["pe.change"] - deltas["large.change"]
        contrasts[metric] = {
            k: {
                "delta_pp": 100 * float(v.mean()),
                "product_bootstrap_95_pp": [
                    100 * bootstrap_lower(v, products),
                    -100 * bootstrap_lower(-v, products),
                ],
                "query_bootstrap_95_pp": [
                    100 * bootstrap_lower(v, np.arange(len(v))),
                    -100 * bootstrap_lower(-v, np.arange(len(v))),
                ],
            }
            for k, v in deltas.items()
        }
    smoke.save(
        args.output / "receipt.json",
        {
            "quality": quality,
            "contrasts": contrasts,
            "source_receipt_sha256": args.receipt_sha256,
            "matrix_sha256": hashes,
            "executed_script_sha256": smoke.sha(Path(__file__)),
            "whole_wall_seconds": time.perf_counter() - started,
            "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "quality_read": "TRAIN-held secondary raw encoder diagnostic",
            "claim_eligible": False,
            "prior_stop_preserved": True,
        },
    )
    print(
        json.dumps(
            {
                k: {m: v[m] for m in ("recall_at_1", "map_at_r")}
                for k, v in quality.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
