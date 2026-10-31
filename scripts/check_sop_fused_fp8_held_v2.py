#!/usr/bin/env python3
"""v2: + arm `cutlass` (src/sfora/so400m_cutlass_fp8.py, lib bec8c18d), pre-registered same gate >= 95.166, run once.
Held quality gate + e2e throughput for the fused FP8 shared-trunk SOP ensemble (src/sfora/so400m_fused_fp8.py).

Pre-registered order (Loop P keep rule: gain beyond noise AND quality gate): arm `mx` (fc1+GELU->MXFP8, 1.171x GPU-only)
first; arm `plain` (1.147x) only if `mx` fails. Gate: held R@1 (5,851 vs 59,551 TRAIN, renormalised member mean, packed
int8 512-D, score_packed_full_gallery) >= 95.166. e2e img/s = full TRAIN encode with 8 decode workers (incl. warmup).
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sop_shared_trunk_e2e_v1 as b
import train_sop_so400m_384_dim512_pilot as dim512
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_fused_fp8 import FusedSharedTrunkEnsemble
from sfora.so400m_cutlass_fp8 import CutlassSharedTrunkEnsemble

snap, ca, cb, arch, root = map(Path, sys.argv[1:6]); arm = sys.argv[6]; dev = torch.device("cuda")
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str); lab = np.asarray(z["train_labels"], dtype=np.int64)
enc = [So400mCompactEncoder.from_checkpoint(snap, c, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True) for c in (ca, cb)]
ens = CutlassSharedTrunkEnsemble(enc) if arm == "cutlass" else FusedSharedTrunkEnsemble(enc, mx_mlp=arm == "mx")
dl = torch.utils.data.DataLoader(b.Rows(b.pair.paths_from_archive(root, rel), enc[0].processor), batch_size=64, num_workers=8, pin_memory=True)
t = time.perf_counter()
e = torch.cat([ens.encode_pixels(px.to(dev, non_blocking=True)).float().cpu() for px in dl])
torch.cuda.synchronize(); secs = time.perf_counter() - t
e = torch.nn.functional.normalize(e[:, :512] + e[:, 512:], dim=1)
held = np.asarray(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes, dtype=np.int64)
pk = pack_int8_unit_embeddings(e)
q = dim512.score_packed_full_gallery(pk.codes.float(), pk.inverse_norms, torch.from_numpy(lab.copy()), torch.from_numpy(held.copy()), device=dev)
out = {"arm": arm, "r1": q["recall_at_1"], "map": q["map_at_r"], "gate": 0.95166, "e2e_img_s_incl_warmup": len(lab) / secs,
       "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
out["decision"] = "PASS" if out["r1"] >= out["gate"] else "FAIL"; print("RESULT", json.dumps(out), flush=True)
