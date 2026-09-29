#!/usr/bin/env python3
"""CPU replay of saved S16 TRAIN-fit matrix, references and authority."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

import export_pe_s16_fit as fit
from pe_core_authority import sha


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    preflight = args.output / "preflight.json"
    receipt = args.output / "receipt.json"
    assert (
        sha(preflight) == args.preflight_sha256 and sha(receipt) == args.receipt_sha256
    )
    frozen = json.loads(preflight.read_text())
    r = json.loads(receipt.read_text())
    fit.source_authority()
    assert all(sha(fit.ROOT / name) == value for name, value in frozen["code"].items())
    assert {n: importlib.metadata.version(n) for n in fit.VERSIONS} == frozen[
        "versions"
    ]
    assert frozen["manifest"] == json.loads(fit.FIT.read_text())["manifest"]
    assert r["pass"] and r["source_receipt_sha256"] == fit.SOURCE_SHA256
    assert r["preflight_sha256"] == args.preflight_sha256
    features = args.output / "s16.fit.npy"
    references = args.output / "references.npz"
    assert (
        sha(features) == r["features_sha256"]
        and sha(references) == r["references_sha256"]
    )
    error = fit.check_features(features, 13283)
    assert abs(error - r["max_norm_error_float64"]) < 1e-12
    values = np.load(features, mmap_mode="r", allow_pickle=False)[:4].astype(np.float64)
    with np.load(references, allow_pickle=False) as refs:
        cosines = {}
        for name in ("fp32", "fp16"):
            reference = refs[name].astype(np.float64)
            assert reference.shape == values.shape and np.isfinite(reference).all()
            cosine = (values * reference).sum(1) / (
                np.linalg.norm(values, axis=1) * np.linalg.norm(reference, axis=1)
            )
            assert cosine.min() >= 0.999
            assert np.max(np.abs(cosine - r["cache_reference_cosines"][name])) < 1e-6
            cosines[name] = cosine.tolist()
        a, b = refs["fp32"].astype(np.float64), refs["fp16"].astype(np.float64)
        cosine = (a * b).sum(1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        assert (
            cosine.min() >= 0.999
            and np.max(np.abs(cosine - r["fp16_fp32_cosine"])) < 1e-6
        )
    assert (
        r["source_state_unchanged"] and r["held_images"] == r["optimizer_updates"] == 0
    )
    assert not r["quality_read"] and not r["public_latency_measured"]
    assert 0 < r["export_seconds"] < r["whole_seconds"] < 300
    assert abs(13283 / r["export_seconds"] - r["export_images_per_second"]) < 1e-9
    assert r["peak_cuda_allocated_bytes"] < 10_000_000_000
    fit.smoke.save(
        args.output / "cpu-audit.json",
        {
            "pass": True,
            "script_sha256": sha(Path(__file__)),
            "receipt_sha256": args.receipt_sha256,
            "features_sha256": sha(features),
            "references_sha256": sha(references),
            "cache_reference_cosines_float64": cosines,
            "max_norm_error_float64": error,
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "cuda_used": False,
        },
    )
    print(
        "PASS saved full-fit S16 matrix/reference parity, source/code/environment/resource receipt replay"
    )


if __name__ == "__main__":
    main()
