#!/usr/bin/env python3
"""TRAIN-held efficiency screen: promoted SOP So400m checkpoint encoded at lower test resolution.

Pre-registered (committed before any reduced-resolution number exists):
  model    : So400m pair checkpoint, seed 179024 (receipt 05e87eaf...), unchanged weights; position
             embeddings bilinearly interpolated (HF interpolate_pos_encoding) for r < 384.
  split    : SOP official TRAIN, product-disjoint seed 179019 fit 0.9; held queries vs full 59,551 gallery,
             packed int8 128-D, native top-10 exact (trainer evaluate_packed).
  arms     : r in (384, 336, 280, 224) -> 729/576/400/256 patch tokens. r=384 must reproduce the receipt's
             per-query R@1 exactly, else INVALID.
  cost     : GPU-only encode img/s at batch 64 fp16 autocast on a fixed tensor (decode excluded) per r,
             and Large/16@256 on the same bench.
  decision : smallest r whose held R@1 paired product-bootstrap lower95 vs the Large pair control
             (receipt d7f7383f per-query) is > 0 becomes the efficiency candidate for ONE official read.
             If none below 384, the lane is closed. mAP@R reported.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import train_sop_siglip2_substrate_pair as pair

RESOLUTIONS = (384, 336, 280, 224)
RECEIPTS = {"large": "d7f7383f9c8320e9d4ca5143880a637ed9bf4bc63ef61a10767c04e8305c8494",
            "so400": "05e87eaf5aeb5f80231fb0d3edd2240fe41d599416e1b2e6d64297a0e037f91a"}


class Interpolated(nn.Module):
    def __init__(self, vision: nn.Module) -> None:
        super().__init__()
        self.vision = vision

    def forward(self, **kwargs):
        return self.vision(**kwargs, interpolate_pos_encoding=True)


def load(snapshot: Path, checkpoint: Path, width: int):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, backend="torchvision")
    full = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16)
    vision = full.vision_model.float().cuda().eval()
    del full
    head = nn.Linear(width, 128).cuda().eval()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert ckpt["seed"] == 179024 and ckpt["updates"] == 1000
    vision.load_state_dict(ckpt["vision"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    return processor, vision, head


@torch.inference_mode()
def gpu_bench(vision: nn.Module, r: int, interpolate: bool) -> float:
    x = torch.randn(64, 3, r, r, device="cuda")
    for i in range(23):
        if i == 3:
            torch.cuda.synchronize()
            t = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            vision(pixel_values=x, interpolate_pos_encoding=interpolate)
    torch.cuda.synchronize()
    return 20 * 64 / (time.perf_counter() - t)


def product_interval(d: np.ndarray, products: np.ndarray) -> list[float]:
    _, inv = np.unique(products, return_inverse=True)
    s, n, rng, k = np.bincount(inv, d), np.bincount(inv), np.random.default_rng(179019), inv.max() + 1
    draws = []
    for _ in range(5000):
        c = np.bincount(rng.integers(0, k, k), minlength=k)
        draws.append((c * s).sum() / (c * n).sum())
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("pair-root", "so400-snapshot", "large-snapshot", "archive", "dataset-root",
                 "native-library", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    runs = {s: a.pair_root / f"sfora-sop-pair-{s}-179024-1000-v3" for s in RECEIPTS}
    receipts = {}
    for s, d in runs.items():
        assert pair.sha256(d / "receipt.json") == RECEIPTS[s]
        receipts[s] = json.loads((d / "receipt.json").read_text())
        assert pair.sha256(d / "checkpoint.pt") == receipts[s]["checkpoint_sha256"]
    with np.load(a.archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"])
    paths = pair.paths_from_archive(a.dataset_root, relatives)
    part = pair.deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=pair.FIT_SEED)
    fit = np.asarray(part.fit_row_indexes, dtype=np.int64)
    held = np.asarray(part.validation_row_indexes, dtype=np.int64)
    assert (len(fit), len(held)) == (53_700, 5_851)
    large_r1 = np.asarray(receipts["large"]["quality"]["per_query_r1"], float)
    result = {"schema": "sop-so400m-eval-resolution-screen-v1", "official_read": False,
              "large_control": {"recall_at_1": receipts["large"]["quality"]["recall_at_1"],
                                "map_at_r": receipts["large"]["quality"]["map_at_r"]}, "arms": {}}

    processor, vision, _ = load(a.large_snapshot, runs["large"] / "checkpoint.pt", 1024)
    result["large_control"]["gpu_img_per_s"] = gpu_bench(vision, 256, False)
    del vision
    torch.cuda.empty_cache()

    processor, vision, head = load(a.so400_snapshot, runs["so400"] / "checkpoint.pt", 1152)
    for r in RESOLUTIONS:
        processor.size = {"height": r, "width": r}
        model = vision if r == 384 else Interpolated(vision)
        t = time.perf_counter()
        values = pair.export_all(model, head, paths, tuple(map(int, labels)), processor, workers=4, batch_size=64)
        export_seconds = time.perf_counter() - t
        q = pair.evaluate_packed(values, labels, fit, held, a.native_library)
        r1 = np.asarray(q["per_query_r1"], float)
        if r == 384 and not np.array_equal(r1, np.asarray(receipts["so400"]["quality"]["per_query_r1"], float)):
            result["decision"] = "INVALID: r=384 does not reproduce receipt per-query R@1"
            break
        arm = {"tokens": (r // 14) ** 2, "recall_at_1": q["recall_at_1"], "map_at_r": q["map_at_r"],
               "dR1_vs_large": float((r1 - large_r1).mean()),
               "dR1_vs_large_product_95": product_interval(r1 - large_r1, labels[held]),
               "export_img_per_s_with_decode": len(paths) / export_seconds,
               "gpu_img_per_s": gpu_bench(vision, r, r != 384)}
        result["arms"][str(r)] = arm
        print(json.dumps({r: arm}), flush=True)
    else:
        passing = [r for r in RESOLUTIONS if result["arms"][str(r)]["dR1_vs_large_product_95"][0] > 0]
        best = min(passing) if passing else None
        result["decision"] = f"CANDIDATE r={best}" if best and best < 384 else "CLOSE"
    a.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"]}), flush=True)


if __name__ == "__main__":
    main()
