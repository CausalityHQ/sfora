#!/usr/bin/env python3
"""Replay frozen fit-cache authority, startup rejection and saved descriptors on CPU."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from preflight_inshop_siglip2_unseen_gallery import split
from sfora.unicom_inshop import parse_inshop_partition

PREFLIGHT_SHA = "4e6c886e87ec8abca45455c5790e35e252cf60694f98d043a1c55d5d21aea3ff"
PILOT_SHA = "e8ddc97a1bd9d5c77562f35bf3b9f5d143d1b6f70b2fd5140d0a5652b9e62735"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("root", "cache", "dataset-root", "large-snapshot", "mechanics-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--startup-only", action="store_true")
    args = p.parse_args()
    frozen = json.loads((args.cache / "preflight.json").read_text())
    assert sha(args.cache / "preflight.json") == PREFLIGHT_SHA
    assert sha(args.root / "pilot.features.npz") == PILOT_SHA
    rows = tuple(
        r for r in parse_inshop_partition(args.dataset_root) if r.split == "train"
    )
    fit, held = split(tuple(r.label for r in rows))
    assert tuple(m["train_row"] for m in frozen["manifest"]) == fit
    assert set(rows[i].label for i in fit).isdisjoint(rows[i].label for i in held)
    for i, m in zip(fit, frozen["manifest"], strict=True):
        assert m["product"] == rows[i].label
        assert m["relative_path"] == str(
            rows[i].image_path.relative_to(args.dataset_root)
        )
    if args.startup_only:
        command = [sys.executable, str(args.root / "export_inshop_pe_fit_features.py")]
        for name in ("root", "dataset_root", "large_snapshot", "mechanics_dir"):
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        command += ["--preflight-sha256", PREFLIGHT_SHA]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "preflight.json"
            frozen["manifest"][0]["product"] = "tampered-product"
            altered.write_text(json.dumps(frozen))
            bad = subprocess.run(
                command + ["--output", directory],
                env=env,
                capture_output=True,
                text=True,
                timeout=40,
            )
            assert bad.returncode != 0 and "smoke.sha(args.output" in bad.stderr, (
                bad.stderr
            )
            assert "torch.cuda.is_available()" not in bad.stderr
        good = subprocess.run(
            command + ["--output", str(args.cache)],
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert good.returncode != 0 and "torch.cuda.is_available()" in good.stderr, (
            good.stderr
        )
        assert not list(args.cache.glob("*.npy"))
        print(
            "PASS altered manifest rejected before CUDA; original passes metadata to CUDA guard"
        )
        return
    receipt = json.loads((args.cache / "receipt.json").read_text())
    assert receipt["preflight_sha256"] == PREFLIGHT_SHA
    assert not receipt["quality_read"] and not receipt["claim_eligible"]
    assert receipt["whole_wall_seconds"] < 480
    prototype = json.loads((args.root / "cpu-preflight-v2.json").read_text())[
        "image_manifest"
    ]
    assert [frozen["manifest"][i] for i in frozen["anchors"]] == prototype
    with np.load(args.root / "pilot.features.npz", allow_pickle=False) as pilot:
        for arm in ("large", "pe"):
            report = receipt["arms"][arm]
            path = args.cache / (arm + ".fit.npy")
            assert sha(path) == report["features_sha256"]
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            assert values.shape == (13283, 1024) and values.dtype == np.float32
            assert np.isfinite(values).all()
            norms = np.linalg.norm(values.astype(np.float64), axis=1)
            assert np.max(np.abs(norms - 1)) < 1e-5
            a = values[frozen["anchors"]].astype(np.float64)
            b = pilot[arm].astype(np.float64)
            cosine = (a * b).sum(1) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            )
            assert cosine.min() >= 0.999
            assert np.max(np.abs(cosine - report["anchor_cosines"])) < 1e-5
            assert min(report["fp16_fp32_cosine"]) >= 0.999
            assert report["peak_cuda_allocated_bytes"] < 10_000_000_000
            assert report["post_cleanup_allocated_bytes"] < 8 * 1024**2
            assert not report["quality_read"]
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "min_anchor_cosine_float64": float(cosine.min()),
                        "max_unit_error_float64": float(np.max(np.abs(norms - 1))),
                    }
                )
            )
    assert all(sha(args.root / name) == value for name, value in frozen["code"].items())
    print("PASS complete saved fit matrices and authority replay; no quality read")


if __name__ == "__main__":
    main()
