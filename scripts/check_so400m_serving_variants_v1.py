#!/usr/bin/env python3
"""Loop P gate (v1 arms; --fp8 adds arm D, pre-registered 2026-10-24 after bench_so400m_fp8_linear_v1 1.56x): end-to-end So400m serving variants on the real 5,851 SOP TRAIN-held queries.

Arms (all encode the 59,551 TRAIN images, pack int8, search held vs full gallery, self excluded):
  A ref_workers : reference arithmetic (fp32 weights, fp16 autocast), preprocessing moved into 8
                  DataLoader workers. Same ops => must be bitwise equal to train_embeddings.npy.
  B fp16_native : vision weights cast to fp16 (no autocast), head fp32, 8 workers.
  C fp16_compile: B + torch.compile(mode='max-autotune-no-cudagraphs'), 8 workers.
  D fp8_compile : C with every vision nn.Linear outside nn.MultiheadAttention as dynamic FP8
                  (bench_so400m_fp8_linear_v1.FP8Linear); only with --fp8. Same keep rule as B/C.
Pre-registered keep rule (committed before any number exists):
  A keeps iff bitwise equal and img/s > P1 parity (72.4 @280 / 41.8 @384).
  B/C keep iff img/s beats the best already-kept arm by > 5% AND net held correct-at-1 change
  vs the receipt >= -3 of 5,851 (-0.05 pp). Flip counts and mean |cos drift| are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.so400m_compact_serving import BATCH_SIZE, So400mCompactEncoder, So400mCompactIndex

WORKERS = 8
P1_IMG_S = {280: 72.4, 384: 41.8}


class Pixels(Dataset):  # type: ignore[misc]
    def __init__(self, paths: list[Path], processor) -> None:
        self.paths, self.processor = paths, processor

    def __len__(self) -> int:
        return len(self.paths) // BATCH_SIZE + bool(len(self.paths) % BATCH_SIZE)

    def __getitem__(self, index: int) -> torch.Tensor:  # one whole batch, so processor batching matches
        images = []
        for path in self.paths[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return self.processor(images=images, return_tensors="pt")["pixel_values"]


def run_arm(name: str, enc: So400mCompactEncoder, paths: list[Path]) -> tuple[torch.Tensor, float]:
    vision = enc.vision
    native = name != "ref_workers"
    if native:
        vision = vision.half()
    if name == "fp8_compile":
        from bench_so400m_fp8_linear_v1 import swap
        swap(vision)
    if name in ("fp16_compile", "fp8_compile"):
        vision = torch.compile(vision, mode="max-autotune-no-cudagraphs")
    loader = DataLoader(Pixels(paths, enc.processor), batch_size=None, num_workers=WORKERS,
                        prefetch_factor=4, pin_memory=True, persistent_workers=False)

    @torch.inference_mode()
    def encode(pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.to(enc.device, non_blocking=True)
        if native:
            pooled = vision(pixel_values=pixels.half(), interpolate_pos_encoding=enc.interpolate).pooler_output
        else:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pooled = vision(pixel_values=pixels, interpolate_pos_encoding=enc.interpolate).pooler_output
        with torch.autocast(device_type="cuda", enabled=False):
            return F.normalize(enc.head(F.normalize(pooled.float(), dim=1)), dim=1).cpu()

    if name in ("fp16_compile", "fp8_compile"):  # compile/autotune outside the timed region
        warm = Pixels(paths[:BATCH_SIZE], enc.processor)[0]
        encode(warm)
    torch.cuda.synchronize()
    started = time.perf_counter()
    values = torch.cat([encode(batch) for batch in loader])
    seconds = time.perf_counter() - started
    return values, seconds  # arms run in order ref -> fp16 -> fp16+compile, so the in-place .half() is never undone


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resolution", type=int, choices=(280, 384), required=True)
    p.add_argument("--model-snapshot", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--best-img-s", type=float, help="already-kept best img/s for this resolution (fp8-only run)")
    a = p.parse_args()
    if a.output.exists() or not torch.cuda.is_available():
        raise SystemExit("output exists or CUDA unavailable")
    receipt = json.loads((a.run_dir / "receipt.json").read_text())
    expected = np.asarray(receipt["quality"]["per_query_r1"], dtype=np.float64)
    output_dim = int(receipt["quality"].get("output_width", 128))
    with np.load(a.archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
    held = np.asarray(deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=179019)
                      .validation_row_indexes, dtype=np.int64)
    assert held.shape == expected.shape == (5_851,)
    paths = [a.dataset_root / r for r in relatives]
    reference = torch.from_numpy(np.load(a.run_dir / "train_embeddings.npy", allow_pickle=False))
    enc = So400mCompactEncoder.from_checkpoint(a.model_snapshot, a.run_dir / "checkpoint.pt",
                                               resolution=a.resolution, output_dim=output_dim,
                                               device=torch.device("cuda"))
    result = {"schema": "sfora-so400m-serving-variants-v1", "run_dir": str(a.run_dir),
              "resolution": a.resolution, "output_dim": output_dim, "workers": WORKERS,
              "receipt_r1": float(expected.mean()), "torch": torch.__version__, "arms": {}}
    best = P1_IMG_S[a.resolution]
    arms = ("ref_workers", "fp16_native", "fp16_compile")
    if a.fp8:
        arms, best = ("fp8_compile",), a.best_img_s
    for name in arms:
        values, seconds = run_arm(name, enc, paths)
        packed = pack_int8_unit_embeddings(values)
        index = So400mCompactIndex(packed, device=enc.device)
        queries = type(packed)(packed.codes[held].contiguous(), packed.inverse_norms[held].contiguous())
        ordinals, _ = index.search(queries, 1, exclude=held)
        observed = (labels[ordinals[:, 0]] == labels[held]).astype(np.float64)
        img_s = len(paths) / seconds
        net = int(observed.sum() - expected.sum())
        arm = {"images_per_second": img_s, "seconds": seconds, "observed_r1": float(observed.mean()),
               "net_correct_vs_receipt": net, "fixed": int(((observed == 1) & (expected == 0)).sum()),
               "broken": int(((observed == 0) & (expected == 1)).sum()),
               "bitwise_equal": bool(torch.equal(values.view(torch.int32), reference.view(torch.int32))),
               "mean_abs_cos_drift": float((1 - (values * reference).sum(1)).abs().mean()),
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}
        if name == "ref_workers":
            arm["keep"] = arm["bitwise_equal"] and img_s > best
        else:
            arm["keep"] = img_s > best * 1.05 and net >= -3
        if arm["keep"]:
            best = img_s
        result["arms"][name] = arm
        print(json.dumps({name: arm}), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
