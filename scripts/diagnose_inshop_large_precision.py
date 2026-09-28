#!/usr/bin/env python3
"""Forward-only numerical falsifier; never optimizes or computes retrieval quality."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

import probe_inshop_pe_training_smoke as smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("root", "output", "dataset-root", "large-snapshot"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cpu-check", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frozen = json.loads((args.output / "preflight.json").read_text())
    smoke.check_startup(args, frozen)
    rows = frozen["updated_rows"][:4]
    assert rows == [0, 1, 2, 5]
    manifest = json.loads((args.root / "cpu-preflight-v2.json").read_text())["image_manifest"]
    assert all(
        smoke.sha(args.dataset_root / manifest[i]["relative_path"]) == manifest[i]["image_sha256"]
        for i in rows
    )
    cached = torch.from_numpy(
        np.load(args.root / "pilot.features.npz", allow_pickle=False)["large"][rows]
    )
    assert cached.shape == (4, 1024) and torch.isfinite(cached).all()
    assert torch.allclose(cached.norm(dim=1), torch.ones(4), atol=1e-5, rtol=0)
    if args.cpu_check:
        print("PASS original executable/source/cache/image authority; four-row unit features")
        return
    started = time.perf_counter()
    assert torch.cuda.is_available()
    destination = args.output / "large-precision-diagnostic.json"
    assert not destination.exists()
    torch.cuda.reset_peak_memory_stats()
    vision, processor = smoke.load_arm(args, "large")
    pixels = []
    for i in rows:
        with Image.open(args.dataset_root / manifest[i]["relative_path"]) as image:
            pixels.append(
                processor(images=[image.convert("RGB")], return_tensors="pt")["pixel_values"][0]
            )
    pixels = torch.stack(pixels).cuda()
    vision = vision.cuda().eval()
    initial_sha = smoke.digest(vision.state_dict())
    features = {"cached_fp16": cached.cuda()}
    with torch.no_grad():
        features["fp32_eval"] = smoke.encode(vision, pixels, "large").float()
        for name, dtype in [("fp16_eval", torch.float16), ("bf16_eval", torch.bfloat16)]:
            with torch.autocast("cuda", dtype=dtype):
                features[name] = smoke.encode(vision, pixels, "large").float()
        vision.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features["bf16_train"] = smoke.encode(vision, pixels, "large").float()
    assert all(torch.isfinite(x).all() and (x.norm(dim=1) > 0).all() for x in features.values())
    pairs = [
        ("fp32_eval", "fp16_eval"),
        ("fp32_eval", "bf16_eval"),
        ("cached_fp16", "fp16_eval"),
        ("cached_fp16", "bf16_train"),
        ("bf16_eval", "bf16_train"),
    ]
    cosines = {
        a + "_vs_" + b: F.cosine_similarity(features[a], features[b]).tolist() for a, b in pairs
    }
    torch.cuda.synchronize()
    assert smoke.digest(vision.state_dict()) == initial_sha
    feature_path = args.output / "large-precision-diagnostic.npz"
    with feature_path.open("xb") as stream:
        np.savez(stream, **{name: x.cpu().numpy() for name, x in features.items()})
    result = {
        "rows": rows,
        "cosines": cosines,
        "bf16_train_eval_max_abs": float(
            (features["bf16_train"] - features["bf16_eval"]).abs().max()
        ),
        "weights_unchanged_sha256": initial_sha,
        "features_sha256": smoke.sha(feature_path),
        "diagnostic_code_sha256": smoke.sha(__file__),
        "preflight_sha256": smoke.sha(args.output / "preflight.json"),
        "whole_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_updates": 0,
        "quality_read": False,
        "claim_eligible": False,
    }
    assert (
        result["whole_wall_seconds"] < 60 and result["peak_cuda_allocated_bytes"] < 10_000_000_000
    )
    smoke.save(destination, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
