#!/usr/bin/env python3
"""Pre-kill E1 for 'uncertainty-conditioned pair scoring' (proposals v2 #1), ORACLE role score.

r(i) = -(mean cosine of i to the other images of its own product)  [label oracle: most favourable role score].
Held queries (seed 179019, fit 0.9) vs full TRAIN gallery, self excluded, cached global_512 (held R@1 94.924).
E1 = errors with positive in top-32 where EVERY gallery item ranked above the best positive F* has r > r(F*)
(so a monotone role prior could reorder them). Pre-registered: KILL if E1 per 2k queries < 10.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "src")
from sfora.representation_ceiling import deterministic_class_partition

e = torch.from_numpy(np.load(sys.argv[1])).float()
with np.load(sys.argv[2], allow_pickle=False) as z: lab = np.asarray(z["train_labels"], dtype=np.int64)
held = np.asarray(deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019).validation_row_indexes)
L = torch.from_numpy(lab)
# oracle role
u, inv = np.unique(lab, return_inverse=True); inv = torch.from_numpy(inv)
S = torch.zeros(len(u), e.shape[1]).index_add_(0, inv, e); n = torch.bincount(inv).float()
own = ((e * S[inv]).sum(1) - (e * e).sum(1)) / (n[inv] - 1).clamp(min=1)
r = -own
err = e1 = in32 = 0
for c in torch.split(torch.from_numpy(held), 512):
    s = e[c] @ e.T; s[torch.arange(len(c)), c] = -9
    top = s.topk(32, dim=1).indices
    for qi, q in enumerate(c.tolist()):
        t = top[qi]; pos = (L[t] == L[q]).nonzero()
        if len(pos) and pos[0].item() == 0: continue
        err += 1
        if not len(pos): continue
        in32 += 1; k = pos[0].item()
        if bool((r[t[:k]] > r[t[k]]).all()): e1 += 1
out = {"held": len(held), "errors": err, "pos_in_top32": in32, "E1": e1, "E1_per_2k": e1 * 2000 / len(held)}
out["decision"] = "KILL" if out["E1_per_2k"] < 10 else "PASS"
print(json.dumps(out))
