#!/usr/bin/env python3
"""CPU-only CMR coalition-marginal signature gate."""
from __future__ import annotations
import argparse, hashlib
import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("pack"); args = ap.parse_args()
    d = np.load(args.pack); z = d["embeddings"].astype(np.float32); y = d["labels"]
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    rng = np.random.default_rng(170)
    auc_num = auc_den = 0.0; gains = []
    all_support, all_selected, eval_queries, eval_labels = [], [], [], []
    for c in np.unique(y):
        ix = np.flatnonzero(y == c)
        if len(ix) < 6: continue
        # deterministic per-class split; signatures are fit on support and
        # evaluated on disjoint queries, never on official test identities.
        h = np.array([int(hashlib.sha256(str(int(i)).encode()).hexdigest()[:8],16) for i in ix])
        sup = ix[np.argsort(h)[:len(ix)//2]]; qry = ix[np.argsort(h)[len(ix)//2:]]
        if len(sup) < 4 or len(qry) < 4: continue
        qfit, qeval = qry[:len(qry)//2], qry[len(qry)//2:]
        # phi[i,q] by 64 Monte-Carlo coalitions of support images.
        sims = z[qfit] @ z[sup].T; n = len(sup); phi = np.zeros((n,len(qfit)), np.float32)
        for r in range(64):
            perm = rng.permutation(n); base = np.full(len(qfit), -1.0, np.float32)
            for j in perm:
                new = np.maximum(base, sims[:,j]); phi[j] += new-base; base = new
        phi /= 64.0
        # pair score: agreement of marginal profiles; target: mutual utility
        # above the median nearest-support utility for held-out queries.
        pnorm = phi / np.maximum(np.linalg.norm(phi,axis=1,keepdims=True),1e-8)
        agree = pnorm @ pnorm.T
        util = phi.mean(1); target = (util[:,None] + util[None,:]) / 2
        tri = np.triu_indices(n,1); s = agree[tri]; t = target[tri]
        order = np.argsort(s); ranks = np.empty_like(order); ranks[order] = np.arange(len(order))
        pos = t >= np.median(t); auc_num += float((ranks[pos].mean()-ranks[~pos].mean())/len(ranks)) * pos.sum()* (~pos).sum(); auc_den += pos.sum()* (~pos).sum()
        # retrieval using all support vs top half by utility (a utility check).
        keep = sup[np.argsort(util)[len(util)//2:]]
        all_support.extend(sup.tolist()); all_selected.extend(keep.tolist())
        eval_queries.extend(qeval.tolist()); eval_labels.extend([int(c)] * len(qeval))
    # Cross-fitted held-out retrieval utility: select support by qfit, evaluate
    # nearest-neighbour R@1 on disjoint qeval against all classes.
    qz = z[np.asarray(eval_queries)];
    def r1(gix):
        gz = z[np.asarray(gix)]; gl = y[np.asarray(gix)]
        return float(np.mean(gl[np.argmax(qz @ gz.T, axis=1)] == np.asarray(eval_labels)))
    r_all = r1(all_support); r_sel = r1(all_selected); gains.append(r_sel-r_all)
    auc = 0.5 + (auc_num/auc_den if auc_den else 0.0)
    print(f"CMR_CPU_AUC={auc:.6f} support_to_query_r1_all={r_all:.6f} selected={r_sel:.6f} delta={np.mean(gains):+.6f} classes={len(np.unique(y))}")
    raise SystemExit(0 if auc >= 0.60 and np.mean(gains) >= 0.005 else 2)
if __name__ == "__main__": main()
