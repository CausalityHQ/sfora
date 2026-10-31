#!/usr/bin/env python3
"""Redesign v2 candidate 1: role-conditioned re-scoring of the top-32 (learned role, cross-fitted). Pre-registered.

r_hat: ridge (lambda chosen by 5-fold CV on FIT) from [global_512, context_1152] to oracle role
r = -(mean cos to own product's other images), trained on FIT-split images only.
Score: s'(q,g) = s - b1*r_hat(g) - b2*r_hat(q)*r_hat(g) on the top-32; (b1,b2) grid-searched on product-half A of held
queries, applied to half B, then swapped (2-fold cross-fit). Paired product bootstrap (5000, seed 179019).
Rule G: KILL if dR@1 < +0.3 pp, or lower95 <= 0, or dmAP@R(top-32 proxy) < -0.3.
"""
import json, sys
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from sfora.representation_ceiling import deterministic_class_partition

e = torch.from_numpy(np.load(sys.argv[1])).float(); ctx = torch.from_numpy(np.load(sys.argv[2])).float()
with np.load(sys.argv[3], allow_pickle=False) as z: lab = np.asarray(z["train_labels"], dtype=np.int64)
part = deterministic_class_partition(tuple(map(int, lab)), fit_fraction=0.9, seed=179019)
held = np.asarray(part.validation_row_indexes); fit = np.setdiff1d(np.arange(len(lab)), held)
L = torch.from_numpy(lab); u, inv = np.unique(lab, return_inverse=True); inv = torch.from_numpy(inv)
S = torch.zeros(len(u), e.shape[1]).index_add_(0, inv, e); n = torch.bincount(inv).float()
r = -(((e * S[inv]).sum(1) - 1) / (n[inv] - 1).clamp(min=1))
X = torch.cat([e, torch.nn.functional.normalize(ctx, dim=1), torch.ones(len(e), 1)], 1).double()
def ridge(idx, lam): A = X[idx]; return torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], dtype=A.dtype), A.T @ r[idx].double())
rng = np.random.default_rng(179019); folds = np.array_split(rng.permutation(fit), 5); best = None
for lam in (1e-1, 1, 10, 100):
    err = sum(float(((X[f] @ ridge(np.setdiff1d(fit, f), lam) - r[f].double()) ** 2).mean()) for f in folds)
    best = min(best or (err, lam), (err, lam))
w = ridge(fit, best[1]); rh = (X @ w).float()
auc_corr = float(np.corrcoef(rh[held].numpy(), r[held].numpy())[0, 1])
tops, sims = [], []
for c in torch.split(torch.from_numpy(held), 512):
    s = e[c] @ e.T; s[torch.arange(len(c)), c] = -9; v, t = s.topk(32, dim=1); tops.append(t); sims.append(v)
T, V = torch.cat(tops), torch.cat(sims); H = torch.from_numpy(held)
rel = (L[T] == L[H][:, None]).float()
def metrics(b1, b2, rows):
    sc = V[rows] - b1 * rh[T[rows]] - b2 * rh[H[rows]][:, None] * rh[T[rows]]
    o = sc.argsort(1, descending=True); rr = rel[rows].gather(1, o)
    r1 = rr[:, 0]; cs = rr.cumsum(1); ap = (cs / torch.arange(1, 33) * rr).sum(1) / rr.sum(1).clamp(min=1)
    return r1.numpy(), ap.numpy()
hp = lab[held]; prods = np.unique(hp); rng2 = np.random.default_rng(179019); A = set(rng2.permutation(prods)[: len(prods) // 2])
half = np.array([p in A for p in hp]); grid = [(b1, b2) for b1 in np.linspace(0, 0.3, 13) for b2 in np.linspace(-0.6, 0.6, 13)]
new_r1, new_ap = np.zeros(len(held)), np.zeros(len(held)); chosen = {}
for name, tr in (("A", half), ("B", ~half)):
    idx = np.nonzero(tr)[0]; b = max(grid, key=lambda g: metrics(*g, idx)[0].mean() + 1e-3 * metrics(*g, idx)[1].mean())
    chosen[name] = b; te = np.nonzero(~tr)[0]; a1, a2 = metrics(*b, te); new_r1[te], new_ap[te] = a1, a2
b1, b2 = metrics(0, 0, np.arange(len(held)))
def paired(d):
    pu, pinv = np.unique(hp, return_inverse=True); Sd = np.bincount(pinv, d); N = np.bincount(pinv); rg = np.random.default_rng(179019); bs = []
    for _ in range(5000): c = np.bincount(rg.integers(0, len(pu), len(pu)), minlength=len(pu)); bs.append((c * Sd).sum() / (c * N).sum())
    return {"delta_pp": 100 * float(d.mean()), "lower95": 100 * float(np.percentile(bs, 2.5)), "upper95": 100 * float(np.percentile(bs, 97.5))}
out = {"ridge_lambda": best[1], "corr_rhat_r_held": auc_corr, "chosen": {k: list(map(float, v)) for k, v in chosen.items()},
       "base_r1": float(b1.mean()), "new_r1": float(new_r1.mean()), "fixed": int(((new_r1 == 1) & (b1 == 0)).sum()),
       "broken": int(((new_r1 == 0) & (b1 == 1)).sum()), "r1": paired(new_r1 - b1), "map32": paired(new_ap - b2)}
o = out; out["decision"] = "PASS" if o["r1"]["delta_pp"] >= 0.3 and o["r1"]["lower95"] > 0 and o["map32"]["delta_pp"] >= -0.3 else "KILL"
print(json.dumps(out))
