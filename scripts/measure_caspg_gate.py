#!/usr/bin/env python3
"""CPU-only conformal ambiguity-set pair gate diagnostic (Pass 175)."""
from __future__ import annotations
import argparse, hashlib
import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("--classes", type=int, default=800); args = ap.parse_args()
    d = np.load(args.pack); z = d["embeddings"].astype(np.float32); y = d["labels"].astype(np.int64)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
    classes = np.unique(y)[:args.classes]; keep = np.isin(y, classes); z, y = z[keep], y[keep]
    rng = np.random.default_rng(175); protos=[]; cl=[]; supports=[]; queries=[]
    for c in classes:
        ix=np.flatnonzero(y==c); h=np.array([int(hashlib.sha256(str(int(i)).encode()).hexdigest()[:8],16) for i in ix]); ix=ix[np.argsort(h)]
        if len(ix)<6: continue
        cut=max(3,len(ix)//2); supports.append(ix[:cut]); queries.append(ix[cut:]); protos.append(z[ix[:cut]].mean(0)); cl.append(c)
    cl=np.asarray(cl); P=np.asarray(protos); P/=np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-8)
    # calibration on support images: conformal nonconformity is 1-cosine to class prototype.
    cal=[]; all_sup=[]; all_q=[]
    for c,s,q in zip(cl,supports,queries):
        cal.extend((1-(z[s]@P[cl==c][0])).tolist()); all_sup.extend(s.tolist()); all_q.extend(q.tolist())
    alpha=.10; qthr=float(np.quantile(cal,1-alpha,method='higher'))
    sets=[]
    for i in range(len(z)):
        score=1-z[i]@P.T; sets.append(set(np.flatnonzero(score<=qthr).tolist()))
    # gate same-class pairs by target-excluded conformal-set Jaccard.
    pos=[]; neg=[]
    for c,s,q in zip(cl,supports,queries):
        for a in q[:min(8,len(q))]:
            for b in q[:min(8,len(q))]:
                if a<b:
                    A=sets[a]-{int(np.flatnonzero(cl==c)[0])}; B=sets[b]-{int(np.flatnonzero(cl==c)[0])}; u=len(A|B); pos.append(len(A&B)/u if u else 1.)
        other=np.flatnonzero(y!=c)
        for b in other[:min(8,len(other))]:
            A=sets[q[0]]-{int(np.flatnonzero(cl==c)[0])}; B=sets[b]-{int(np.flatnonzero(cl==y[b])[0]) if np.any(cl==y[b]) else -1}; u=len(A|B); neg.append(len(A&B)/u if u else 1.)
    scores=np.r_[pos,neg]; labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; order=np.argsort(scores); ranks=np.empty_like(order); ranks[order]=np.arange(len(order)); auc=float((ranks[labels==1].mean()-ranks[labels==0].mean())/len(scores)+.5)
    # selected support retrieval: retain support images whose set overlaps query set.
    gains=[]
    for c,s,q in zip(cl,supports,queries):
        for qi in q:
            A=sets[qi]-{int(np.flatnonzero(cl==c)[0])}; sim=z[qi]@z[np.asarray(all_sup)].T
            selected=[j for j in s if len(A & (sets[j]-{int(np.flatnonzero(cl==c)[0])}))>0]
            if selected:
                gains.append(float(y[s[np.argmax(z[qi]@z[s].T)]]==c)-float(y[np.asarray(all_sup)[np.argmax(sim)]]==c))
    rdelta=float(np.mean(gains)) if gains else 0.
    density=float(np.mean([len(s) for s in sets])); print(f"CASPG_CPU_AUC={auc:.6f} set_size={density:.2f} threshold={qthr:.6f} pairs={len(scores)} selected_delta={rdelta:+.6f} classes={len(cl)}")
    raise SystemExit(0 if auc>.70 and rdelta>=.005 else 2)
if __name__=='__main__': main()
