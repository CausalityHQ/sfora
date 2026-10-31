#!/usr/bin/env python3
"""ONE official SOP TEST read of the served configuration: FP8-compiled shared-trunk 2-seed So400m@384/512 ensemble.
Scored exactly as the fp32 incumbent (score_sop_official_seed_ensemble_v1.py: concat/sqrt(2), packed int8 1,026 B/row,
score_symmetric). Pre-registered: the served path is ACCEPTED iff official R@1 >= 94.129 - 0.10 (serving drift budget);
this is not a selection read (no alternatives compared)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sop_shared_trunk_e2e_v1 as b
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_evaluation import score_symmetric
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble

snap, ca, cb, arch, root = map(Path, sys.argv[1:6]); dev = torch.device("cuda")
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["test_relative_paths"]).astype(str); lab = np.asarray(z["test_labels"], dtype=np.int64)
enc = [So400mCompactEncoder.from_checkpoint(snap, c, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True) for c in (ca, cb)]
ens = SharedTrunkEnsemble(enc, compile=True)
dl = torch.utils.data.DataLoader(b.Rows(b.pair.paths_from_archive(root, rel), enc[0].processor), batch_size=64, num_workers=8, pin_memory=True)
e = torch.cat([ens.encode_pixels(px.to(dev, non_blocking=True)).float().cpu() for px in dl])
pk = pack_int8_unit_embeddings(e)
q = score_symmetric(pk.codes.float().cuda(), torch.from_numpy(lab).cuda(), inverse_norms=pk.inverse_norms.cuda())
out = {"n": len(lab), "r1": q["recall_at_1"], "map": q["map_at_r"], "gate": 0.94129 - 0.001}
out["decision"] = "ACCEPT" if out["r1"] >= out["gate"] else "REJECT"; print("RESULT", json.dumps(out))
