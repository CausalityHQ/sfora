#!/usr/bin/env python3
"""Quality screen (proposals v3 #6): WiSE-FT of the SOP So400m@384/512 seed-179024 vision tower toward pretrained.

theta(a) = a * finetuned + (1 - a) * pretrained for every vision tensor (frozen ones are identical); trained linear head
kept. Arms a in {1.0, 0.85, 0.7}, all exported by THIS script with the same encoder path (fp16 native, no FP8) and
scored identically: 5,851 held vs 59,551 TRAIN gallery, packed int8 512-D, score_packed_full_gallery.
Pre-registered: KEEP iff the best a < 1 beats a = 1 on held R@1 with paired product-bootstrap lower95 > 0
(then confirm on seed 179025 before any ensemble/official use). Else KILL.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_sop_shared_trunk_e2e_v1 as b
import train_sop_so400m_384_dim512_pilot as dim512
from screen_inshop_pilot_pair_v1 import paired
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.so400m_compact_serving import So400mCompactEncoder

snap, ckpt, arch, root = map(Path, sys.argv[1:5]); dev = torch.device("cuda")
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str); lab = np.asarray(z["train_labels"], dtype=np.int64)
held = np.asarray(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes, dtype=np.int64)
enc = So400mCompactEncoder.from_checkpoint(snap, ckpt, resolution=384, output_dim=512, device=dev, fp16_native=False)
ft = {k: v.detach().clone() for k, v in enc.vision.state_dict().items()}
from transformers import AutoModel
pre = AutoModel.from_pretrained(snap, local_files_only=True, use_safetensors=True, dtype=torch.float32).vision_model.state_dict()
paths = b.pair.paths_from_archive(root, rel); out = {}; per = {}
for a in (1.0, 0.85, 0.7):
    enc.vision.load_state_dict({k: a * ft[k] + (1 - a) * pre[k].to(ft[k].device, ft[k].dtype) if ft[k].is_floating_point() else ft[k] for k in ft})
    dl = torch.utils.data.DataLoader(b.Rows(paths, enc.processor), batch_size=64, num_workers=8, pin_memory=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        e = []
        for px in dl:
            pooled = enc.vision(pixel_values=px.to(dev, non_blocking=True)).pooler_output
            with torch.autocast("cuda", enabled=False):
                e.append(torch.nn.functional.normalize(enc.head(torch.nn.functional.normalize(pooled.float(), dim=1)), dim=1).cpu())
    pk = pack_int8_unit_embeddings(torch.cat(e))
    q = dim512.score_packed_full_gallery(pk.codes.float(), pk.inverse_norms, torch.from_numpy(lab.copy()), torch.from_numpy(held.copy()), device=dev)
    out[str(a)] = [q["recall_at_1"], q["map_at_r"]]; per[a] = q; print("ARM", a, out[str(a)], flush=True)
best = max((0.85, 0.7), key=lambda a: per[a]["recall_at_1"])
out["best"] = best
out["r1"] = paired(np.asarray(per[best]["per_query_r1"]), np.asarray(per[1.0]["per_query_r1"]), lab[held])
out["map"] = paired(np.asarray(per[best]["per_query_ap"]), np.asarray(per[1.0]["per_query_ap"]), lab[held])
out["decision"] = "KEEP" if out["r1"]["lower95"] > 0 else "KILL"; print("RESULT", json.dumps(out), flush=True)
