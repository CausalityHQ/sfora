#!/usr/bin/env python3
"""Quality gate (pre-registered): FP8-compiled shared-trunk SOP 2-seed ensemble, held R@1 (5,851 held vs 59,551 TRAIN
gallery, packed int8 1,024-D, score_packed_full_gallery). PASS iff R@1 >= 95.266 - 0.10 (fp32 incumbent screen value)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sop_shared_trunk_e2e_v1 as b
import train_sop_so400m_384_dim512_pilot as dim512
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.so400m_compact_serving import So400mCompactEncoder
from sfora.so400m_shared_trunk_ensemble import SharedTrunkEnsemble

snap, ca, cb, arch, root = map(Path, sys.argv[1:6]); dev = torch.device("cuda")
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str); lab = np.asarray(z["train_labels"], dtype=np.int64)
enc = [So400mCompactEncoder.from_checkpoint(snap, c, resolution=384, output_dim=512, device=dev, fp16_native=True, fp8=True) for c in (ca, cb)]
ens = SharedTrunkEnsemble(enc, compile=True)  # segment compile (module)
dl = torch.utils.data.DataLoader(b.Rows(b.pair.paths_from_archive(root, rel), enc[0].processor), batch_size=64, num_workers=8, pin_memory=True)
e = torch.cat([ens.encode_pixels(px.to(dev, non_blocking=True)).float().cpu() for px in dl])
held = np.asarray(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes, dtype=np.int64)
e = torch.nn.functional.normalize(e[:, :512] + e[:, 512:], dim=1)  # renormalised mean, as the 95.266 screen
pk = pack_int8_unit_embeddings(e)
q = dim512.score_packed_full_gallery(pk.codes.float(), pk.inverse_norms, torch.from_numpy(lab.copy()), torch.from_numpy(held.copy()), device=dev)
out = {"r1": q["recall_at_1"], "map": q["map_at_r"], "gate": 0.95266 - 0.001}
out["decision"] = "PASS" if out["r1"] >= out["gate"] else "FAIL"; print("RESULT", json.dumps(out))
