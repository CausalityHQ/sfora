#!/usr/bin/env python3
"""Pareto candidate (speed): SOP single-model (seed 179025, official 93.990) served with the fused FP8 encoder (mx).

Parity check of an already-read model (not a selection): full SOP TEST (60,502, symmetric, self excluded), 8 decode
workers, packed int8 512-D, score_symmetric. Reference = per-query top-1 of the official read's saved fp32
test_embeddings.npy, packed and scored identically. Pre-registered KEEP (Pareto speed point) iff net correct vs the read
>= -13/60,502 (same rate as the In-Shop gate -3/14,218). Reports e2e img/s and fixed/broken.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sop_shared_trunk_e2e_v1 as b
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_evaluation import score_symmetric
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_fused_fp8 import FusedSharedTrunkEnsemble

snap, ckpt, ref_dir, arch, root = map(Path, sys.argv[1:6]); dev = torch.device("cuda")
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["test_relative_paths"]).astype(str); lab = np.asarray(z["test_labels"], dtype=np.int64)
enc = So400mCompactEncoder.from_checkpoint(snap, ckpt, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True)
ens = FusedSharedTrunkEnsemble([enc], mx_mlp=True)
dl = torch.utils.data.DataLoader(b.Rows(b.pair.paths_from_archive(root, rel), enc.processor), batch_size=64, num_workers=8, pin_memory=True)
t = time.perf_counter()
e = torch.cat([ens.encode_pixels(px.to(dev, non_blocking=True)).float().cpu() for px in dl]); torch.cuda.synchronize(); secs = time.perf_counter() - t
L = torch.from_numpy(lab).cuda()
def score(x):
    pk = pack_int8_unit_embeddings(x); return score_symmetric(pk.codes.float().cuda(), L, inverse_norms=pk.inverse_norms.cuda())
q, r = score(e), score(torch.from_numpy(np.load(ref_dir / "test_embeddings.npy")))
o, x = np.asarray(q["per_query_r1"], float), np.asarray(r["per_query_r1"], float)
out = {"serving_r1": q["recall_at_1"], "serving_map": q["map_at_r"], "read_r1": r["recall_at_1"], "net": int(o.sum() - x.sum()),
       "fixed": int(((o == 1) & (x == 0)).sum()), "broken": int(((o == 0) & (x == 1)).sum()),
       "e2e_img_s_incl_warmup": len(lab) / secs, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
out["decision"] = "KEEP" if out["net"] >= -13 else "FAIL"; print("RESULT", json.dumps(out), flush=True)
