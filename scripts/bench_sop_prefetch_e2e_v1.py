#!/usr/bin/env python3
"""Exact e2e lever for the served SOP FP8 per-layer ensemble: overlap host->device copy of batch i+1 with compute of
batch i (side CUDA stream, pinned memory). Arithmetic unchanged; pre-registered keep iff the min of 3 prefetch repeats
beats the max of 3 baseline repeats AND outputs are bitwise equal. 2,048 TRAIN images, 8 workers, warm."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sop_shared_trunk_e2e_v1 as b
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble

snap, ca, cb, arch, root = map(Path, sys.argv[1:6]); dev = torch.device("cuda")
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str)[:2048]
enc = [So400mCompactEncoder.from_checkpoint(snap, c, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True) for c in (ca, cb)]
ens = SharedTrunkEnsemble(enc, compile=True); paths = b.pair.paths_from_archive(root, rel)
dl = lambda: torch.utils.data.DataLoader(b.Rows(paths, enc[0].processor), batch_size=64, num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=False)

def base():
    return [ens.encode_pixels(px.to(dev, non_blocking=True)) for px in dl()]

def prefetch():
    s = torch.cuda.Stream(); out = []; it = iter(dl()); nxt = None
    with torch.cuda.stream(s): nxt = next(it).to(dev, non_blocking=True)
    for px in it:
        torch.cuda.current_stream().wait_stream(s); cur = nxt; cur.record_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s): nxt = px.to(dev, non_blocking=True)
        out.append(ens.encode_pixels(cur))
    torch.cuda.current_stream().wait_stream(s); nxt.record_stream(torch.cuda.current_stream()); out.append(ens.encode_pixels(nxt))
    return out

def t(fn):
    torch.cuda.synchronize(); a = time.perf_counter(); r = fn(); torch.cuda.synchronize(); return 2048 / (time.perf_counter() - a), torch.cat(r).cpu()
base(); res = {"base": [], "prefetch": []}
for _ in range(3):
    for k, fn in (("base", base), ("prefetch", prefetch)):
        v, rows = t(fn); res[k].append(v); res[k + "_rows"] = rows
res["bitwise_equal"] = bool(torch.equal(res.pop("base_rows"), res.pop("prefetch_rows")))
res["decision"] = "KEEP" if min(res["prefetch"]) > max(res["base"]) and res["bitwise_equal"] else "KILL"
print("RESULT", json.dumps(res))
