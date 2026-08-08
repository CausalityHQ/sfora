#!/usr/bin/env python3
"""CPU post-hoc falsifier for CAMW; no model training or GPU use."""
from __future__ import annotations
import argparse, hashlib
import numpy as np

def norm(x): return x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('train'); ap.add_argument('gallery'); ap.add_argument('query'); ap.add_argument('--classes',type=int,default=500); args=ap.parse_args()
    tr=np.load(args.train); ga=np.load(args.gallery); qu=np.load(args.query)
    X=tr['embeddings'].astype(np.float64); y=tr['labels']; G=ga['embeddings'].astype(np.float64); gy=ga['labels']; Qy=qu['labels']; U=qu['embeddings'].astype(np.float64)
    classes=np.unique(y)[:args.classes]; mask=np.isin(y,classes); X=X[mask]; y=y[mask]
    mu=X.mean(0); Xc=X-mu; pooled=(Xc.T@Xc)/max(len(X)-1,1)
    # Deterministic pooled PCA is the cheapest common-axis approximation.
    ev,V=np.linalg.eigh(pooled); order=np.argsort(ev)[::-1][:64]; V=V[:,order]; lam=np.maximum(ev[order],1e-8)
    covs=[]; means=[]; held=[]; fit=[]
    for c in classes:
        ix=np.flatnonzero(y==c)
        if len(ix)<6: continue
        h=np.array([int(hashlib.sha256(str(int(i)).encode()).hexdigest()[:8],16) for i in ix]); ix=ix[np.argsort(h)]; k=len(ix)//2
        a,b=ix[:k],ix[k:]; fit.append(a); held.append(b); means.append(X[a].mean(0)); covs.append(np.cov(X[a],rowvar=False,bias=False))
    # projected between/within statistics; robust class-quantile denominator.
    means=np.asarray(means); B=np.cov(means,rowvar=False,bias=False); bj=np.diag(V.T@B@V)
    v=[]; off_fit=[]; off_hold=[]
    for c,a,b in zip(np.unique(y)[:len(covs)],fit,held):
        ca=np.cov(X[a],rowvar=False,bias=False); cb=np.cov(X[b],rowvar=False,bias=False)
        v.append(np.diag(V.T@ca@V)); off_fit.append(np.linalg.norm((V.T@ca@V)-np.diag(np.diag(V.T@ca@V)))**2/np.linalg.norm(ca)**2); off_hold.append(np.linalg.norm((V.T@cb@V)-np.diag(np.diag(V.T@cb@V)))**2/np.linalg.norm(cb)**2)
    v=np.asarray(v); m=np.clip((bj+1e-6)/(np.quantile(v,.9,axis=0)+1e-6),.25,4.0)
    W=V*np.sqrt(m)[None,:]
    # Embed using top 64 weighted axes, retain complement untouched.
    def trans(Z):
        z=Z-mu; p=z@V; return norm(p*np.sqrt(m))
    tg=trans(G); tq=trans(U); base=norm(G) ; bq=norm(U)
    rbase=float(np.mean(gy[np.argmax(bq@base.T,axis=1)]==Qy)); rcam=float(np.mean(gy[np.argmax(tq@tg.T,axis=1)]==Qy))
    # split-half axis stability from diagonal variance ratios.
    va=np.asarray([np.diag(V.T@np.cov(X[a],rowvar=False,bias=False)@V) for a in fit]); vb=np.asarray([np.diag(V.T@np.cov(X[b],rowvar=False,bias=False)@V) for b in held]); ra=np.mean(va,axis=0); rb=np.mean(vb,axis=0); rho=float(np.corrcoef(ra,rb)[0,1])
    off_red=1-float(np.mean(off_hold)/max(np.mean(off_fit),1e-12)); print(f'CAMW_CPU baseline={rbase:.6f} camw={rcam:.6f} delta={(rcam-rbase):+.6f} off_reduction={off_red:+.6f} axis_rho={rho:.6f} classes={len(covs)}')
    raise SystemExit(0 if off_red>=.20 and rho>=.60 and rcam>rbase and rcam-rbase>=.005 else 2)
if __name__=='__main__': main()
