#!/usr/bin/env python3
"""End-to-end SOP 2-seed shared-trunk encode: JPEG decode + preprocess in 8 DataLoader workers -> GPU encode.
Arms: fp16 eager, fp16 compiled, fp8 compiled (compile = each encoder layer + pooling head compiled in place, max-autotune-no-cudagraphs).
Reports img/s over N TRAIN images (after a warmup pass) and max cos drift vs fp16 eager. Not a quality read.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble
import train_sop_siglip2_substrate_pair as pair


class Rows(torch.utils.data.Dataset):
    def __init__(self, paths, proc): self.paths, self.proc = paths, proc
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.proc(images=[Image.open(self.paths[i]).convert("RGB")], return_tensors="pt")["pixel_values"][0]


def main():
    p = argparse.ArgumentParser()
    for n in ("snapshot", "ckpt-a", "ckpt-b", "archive", "dataset-root"): p.add_argument(f"--{n}", type=Path, required=True)
    p.add_argument("--n", type=int, default=2048); a = p.parse_args(); dev = torch.device("cuda")
    with np.load(a.archive, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str)[: a.n]
    paths = pair.paths_from_archive(a.dataset_root, rel); out = {"n": a.n}; ref = None
    for arm, fp8, comp in (("fp16_eager", False, False), ("fp16_compile", False, True), ("fp8_compile", True, True)):
        enc = [So400mCompactEncoder.from_checkpoint(a.snapshot, c, resolution=384, output_dim=512, device=dev,
                                                    fp16_native=True, fp8=fp8) for c in (a.ckpt_a, a.ckpt_b)]
        ens = SharedTrunkEnsemble(enc, compile=comp); fn = ens.encode_pixels  # segment compile (module)
        dl = lambda: torch.utils.data.DataLoader(Rows(paths, enc[0].processor), batch_size=64, num_workers=8, pin_memory=True)
        t0 = time.perf_counter()
        for px in dl(): fn(px.to(dev, non_blocking=True))
        torch.cuda.synchronize(); warm = time.perf_counter() - t0; rows = []
        t0 = time.perf_counter()
        for px in dl(): rows.append(fn(px.to(dev, non_blocking=True)).float().cpu())
        torch.cuda.synchronize(); dt = time.perf_counter() - t0; r = torch.cat(rows)
        ref = r if ref is None else ref
        out[arm] = {"img_s": a.n / dt, "warmup_s": warm, "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "max_cos_drift_vs_fp16_eager": float((1 - (r * ref).sum(1)).abs().max())}
        print(arm, out[arm], flush=True); del enc, ens, fn; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
