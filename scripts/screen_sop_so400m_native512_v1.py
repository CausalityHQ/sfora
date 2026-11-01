#!/usr/bin/env python3
"""Quality screen (proposals v3 #1, stage 0): pretrained SigLIP2 So400m native-512 (patch16, 1024 tokens) vs native 384.

Zero training. SOP TRAIN rows; held = deterministic_class_partition(seed 179019, fit 0.9); fit = rest. Both towers at
their native resolution (no position interpolation), pooled features scored with score_corpus packed_pca128
(full_train_gallery and holdout_only R@1). Pre-registered: ADVANCE to one fine-tuned 512 pilot iff
Δ full_train_gallery R@1 (512 − 384) >= +1.0 pp AND holdout_only also higher. Else CLOSE.
Context (not a gate): SOP long side median 400 px, 0.9 % >= 512 px (2,000-image sample), so 512 input is upsampled.
"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_siglip2_substrate_pair as pair
from score_sop_pretrained_substrate import score_corpus
from screen_inshop_so400m_resolution_v1 import export_arm
from sfora.representation_ceiling import deterministic_class_partition

s384, s512, arch, root, outdir = map(Path, sys.argv[1:6]); outdir.mkdir(parents=True)
with np.load(arch, allow_pickle=False) as z: rel = np.asarray(z["train_relative_paths"]).astype(str); lab = np.asarray(z["train_labels"], dtype=np.int64)
rows = [SimpleNamespace(image_path=p) for p in pair.paths_from_archive(root, rel)]
held = np.asarray(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes, dtype=np.int64)
fit = np.setdiff1d(np.arange(len(lab)), held); res = {}
for name, snap in (("n384", s384), ("n512", s512)):
    f = outdir / f"{name}.npy"; res[name + "_export"] = export_arm(rows, snap, None, f, 32)
    sc = score_corpus(name, np.load(f), lab, fit, held)["score"]["packed_pca128"]
    res[name] = {"full": sc["full_train_gallery"]["recall_at_1"], "holdout": sc["holdout_only"]["recall_at_1"]}
    print("ARM", name, res[name], flush=True)
d = (res["n512"]["full"] - res["n384"]["full"]) * 100
res["delta_pp"] = d; res["decision"] = "ADVANCE" if d >= 1.0 and res["n512"]["holdout"] > res["n384"]["holdout"] else "CLOSE"
(outdir / "result.json").write_text(json.dumps(res, indent=1)); print("RESULT", json.dumps({"delta_pp": d, "decision": res["decision"], **{k: res[k] for k in ("n384", "n512")}}), flush=True)
