#!/usr/bin/env python3
"""Parity + throughput: shared-trunk 2-seed SOP So400m@384/512 ensemble vs two separate encoders (same precision).

Pass iff max |cos drift| of ensemble rows < 1e-3 vs separate-encoder concat and img/s improves. Images: first N SOP
TRAIN-held rows (decode included in both arms via a shared prefetched PIL list; encode-only timing).
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble
import train_sop_siglip2_substrate_pair as pair


def main():
    p = argparse.ArgumentParser()
    for n in ("snapshot", "ckpt-a", "ckpt-b", "archive", "dataset-root"): p.add_argument(f"--{n}", type=Path, required=True)
    p.add_argument("--n", type=int, default=1024); p.add_argument("--fp8", action="store_true")
    a = p.parse_args(); dev = torch.device("cuda")
    with np.load(a.archive, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str)[: a.n]
    imgs = [Image.open(x).convert("RGB") for x in pair.paths_from_archive(a.dataset_root, rel)]
    enc = [So400mCompactEncoder.from_checkpoint(a.snapshot, c, resolution=384, output_dim=512, device=dev,
                                                fp16_native=True, fp8=a.fp8) for c in (a.ckpt_a, a.ckpt_b)]
    ens = SharedTrunkEnsemble(enc)
    sep_fn = lambda xs: torch.cat([e.encode_images(xs) for e in enc], 1) / math.sqrt(2)
    out = {"n": a.n, "fp8": a.fp8}
    for name, fn in (("separate", sep_fn), ("shared", ens.encode_images)):
        fn(imgs[:64]); torch.cuda.synchronize(); ts = []
        for _ in range(3):
            t = time.perf_counter(); r = fn(imgs); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        out[name] = {"img_s": a.n / min(ts), "seconds": ts}; out[name + "_rows"] = r
    s, t = out.pop("separate_rows"), out.pop("shared_rows")
    out["max_abs_cos_drift"] = float((1 - (s * t).sum(1) / s.norm(dim=1) / t.norm(dim=1)).abs().max())
    out["speedup"] = out["shared"]["img_s"] / out["separate"]["img_s"]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
