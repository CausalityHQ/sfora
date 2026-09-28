#!/usr/bin/env python3
"""Acquire paired original FP16 source features on TRAIN-fit identities only."""

import argparse
import gc
import importlib.metadata
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

import probe_inshop_pe_training_smoke as smoke
from export_sop_siglip2_train import export_features
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from sfora.unicom_inshop import parse_inshop_partition

MECHANICS_SHA = "44fd422a70078767a1cadafa36bb8911deb68ef3759d34b091856d67483ae969"
FIT_SHA = "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be"


def source_authority(args):
    assert smoke.sha(args.mechanics_dir / "receipt.json") == MECHANICS_SHA
    r = json.loads((args.mechanics_dir / "receipt.json").read_text())
    assert r["advance"] and not r["quality_read"]
    prior = json.loads((args.mechanics_dir / "preflight.json").read_text())
    smoke.check_startup(
        SimpleNamespace(
            **{**vars(args), "output": args.mechanics_dir, "vision_precision": "fp16"}
        ),
        prior,
    )
    assert (
        smoke.sha(args.dataset_root / "Eval/list_eval_partition.txt") == PARTITION_SHA
    )
    source = json.loads((args.root / "cpu-preflight-v2.json").read_text())
    assert source["torch"] == torch.__version__
    assert all(
        importlib.metadata.version(n) == v for n, v in source["versions"].items()
    )


def validate_membership(labels, fit, held):
    assert len(labels) == 25882 and len(set(labels)) == 3997
    assert len(fit) == 13283 and len(held) == 12599 and digest_rows(fit) == FIT_SHA
    assert set(labels[i] for i in fit).isdisjoint(labels[i] for i in held)
    counts = Counter(labels[i] for i in fit)
    assert len(counts) == 2004 and sum(n == 1 for n in counts.values()) == 12


def preflight(args):
    source_authority(args)
    rows = tuple(
        r for r in parse_inshop_partition(args.dataset_root) if r.split == "train"
    )
    labels = tuple(r.label for r in rows)
    fit, held = split(labels)
    validate_membership(labels, fit, held)
    try:
        validate_membership(labels, fit[:-1], held)
    except AssertionError:
        pass
    else:
        raise AssertionError("malformed fit membership accepted")
    manifest = [
        {
            "train_row": i,
            "product": labels[i],
            "relative_path": str(rows[i].image_path.relative_to(args.dataset_root)),
            "image_sha256": smoke.sha(rows[i].image_path),
        }
        for i in fit
    ]
    prototype = json.loads((args.root / "cpu-preflight-v2.json").read_text())[
        "image_manifest"
    ]
    lookup = {r["train_row"]: i for i, r in enumerate(manifest)}
    anchors = [lookup[r["train_row"]] for r in prototype]
    assert all(manifest[i] == old for i, old in zip(anchors, prototype, strict=True))
    # Import native PE before freezing lazy import authority; no model is created.
    from einops import _torch_specific

    from core.vision_encoder.pe import VisionTransformer

    assert callable(VisionTransformer.from_config)
    assert Path(_torch_specific.__file__).is_file()
    args.output.mkdir(exist_ok=False)
    smoke.save(
        args.output / "preflight.json",
        {
            "manifest": manifest,
            "anchors": anchors,
            "code": smoke.authority(),
            "mechanics_sha256": MECHANICS_SHA,
            "fit_sha256": FIT_SHA,
            "partition_sha256": PARTITION_SHA,
            "quality_read": False,
        },
    )
    print("PASS full fit inventory/hash/mapping and malformed membership rejection")


def acquire(args, arm, frozen):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    vision, processor = smoke.load_arm(args, arm)
    vision = vision.cuda().eval()

    def pixels(batch):
        images = []
        for row in batch:
            path = args.dataset_root / row["relative_path"]
            assert smoke.sha(path) == row["image_sha256"]
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return (
            torch.stack([processor(x) for x in images])
            if arm == "pe"
            else processor(images=images, return_tensors="pt")["pixel_values"]
        ).cuda()

    with torch.inference_mode():
        x = pixels(frozen["manifest"][:4])
        fp32 = smoke.encode(vision, x, arm).float()
        with torch.autocast("cuda", dtype=torch.float16):
            fp16 = smoke.encode(vision, x, arm).float()
        cosine = F.cosine_similarity(fp32, fp16).tolist()
        print(json.dumps({"arm": arm, "fp16_fp32_cosine": cosine}), flush=True)
        assert min(cosine) >= 0.999
    del x, fp32, fp16

    @torch.inference_mode()
    def encode(batch):
        with torch.autocast("cuda", dtype=torch.float16):
            raw = smoke.encode(vision, pixels(batch), arm).float()
        assert torch.isfinite(raw).all() and (raw.norm(dim=1) > 0).all()
        return F.normalize(raw, dim=1).cpu().numpy()

    out = args.output / (arm + ".fit.npy")
    export_started = time.perf_counter()
    export_features(frozen["manifest"], encode, out, width=1024, batch_size=32)
    export_seconds = time.perf_counter() - export_started
    values = np.load(out, mmap_mode="r", allow_pickle=False)
    assert values.shape == (13283, 1024) and np.isfinite(values).all()
    assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
    cached = np.load(args.root / "pilot.features.npz", allow_pickle=False)[arm]
    anchor_cosines = F.cosine_similarity(
        torch.from_numpy(values[frozen["anchors"]].copy()), torch.from_numpy(cached)
    ).tolist()
    report = {
        "arm": arm,
        "features_sha256": smoke.sha(out),
        "shape": list(values.shape),
        "fp16_fp32_cosine": cosine,
        "anchor_cosines": anchor_cosines,
        "export_seconds": export_seconds,
        "export_images_per_second": len(values) / export_seconds,
        "arm_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "anchors_pass": min(anchor_cosines) >= 0.999,
        "quality_read": False,
    }
    smoke.save(args.output / (arm + ".json"), report)
    assert (
        report["anchors_pass"] and report["peak_cuda_allocated_bytes"] < 10_000_000_000
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("root", "output", "dataset-root", "large-snapshot", "mechanics-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.preflight_only:
        preflight(args)
        return
    started = time.perf_counter()
    source_authority(args)
    frozen = json.loads((args.output / "preflight.json").read_text())
    assert (
        args.preflight_sha256
        and smoke.sha(args.output / "preflight.json") == args.preflight_sha256
    )
    assert digest_rows(tuple(r["train_row"] for r in frozen["manifest"])) == FIT_SHA
    assert all(smoke.sha(args.root / n) == h for n, h in frozen["code"].items())
    assert torch.cuda.is_available() and not (args.output / "receipt.json").exists()
    reports = {}
    for arm in ("large", "pe"):
        reports[arm] = acquire(args, arm, frozen)
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        reports[arm]["post_cleanup_allocated_bytes"] = torch.cuda.memory_allocated()
        smoke.save(
            args.output / (arm + ".cleanup.json"),
            {"allocated_cuda_bytes": reports[arm]["post_cleanup_allocated_bytes"]},
        )
        assert reports[arm]["post_cleanup_allocated_bytes"] < 8 * 1024**2
    assert time.perf_counter() - started < 480
    current = smoke.authority()
    assert all(current.get(n, h) == h for n, h in frozen["code"].items())
    assert set(current) <= set(frozen["code"])
    smoke.save(
        args.output / "receipt.json",
        {
            "arms": reports,
            "whole_wall_seconds": time.perf_counter() - started,
            "preflight_sha256": smoke.sha(args.output / "preflight.json"),
            "quality_read": False,
            "claim_eligible": False,
        },
    )
    print("PASS paired complete fit-only source acquisition", flush=True)


if __name__ == "__main__":
    main()
