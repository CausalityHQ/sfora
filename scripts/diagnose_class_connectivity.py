#!/usr/bin/env python3
"""CPU diagnostic for class-manifold connectivity on a train embedding pack."""
from __future__ import annotations
import argparse, hashlib
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('pack'); args=ap.parse_args()
    d=np.load(args.pack); z=d['embeddings'].astype(np.float32); y=d['labels']; z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    conn=[]; acc=[]; nn=[]
    for c in np.unique(y):
        ix=np.flatnonzero(y==c)
        if len(ix)<8: continue
        h=np.array([int(hashlib.sha256(str(int(i)).encode()).hexdigest()[:8],16) for i in ix])
        ix=ix[np.argsort(h)]; sup,qry=ix[:len(ix)//2],ix[len(ix)//2:]
        S=z[sup]@z[sup].T; np.fill_diagonal(S,-np.inf); k=min(4,len(sup)-1)
        A=np.zeros_like(S); nbr=np.argpartition(-S,kth=k-1,axis=1)[:,:k]; rows=np.arange(len(sup))[:,None]; A[rows,nbr]=np.maximum(0,S[rows,nbr]); A=np.maximum(A,A.T)
        deg=A.sum(1); inv=np.zeros_like(deg); inv[deg>0]=1/np.sqrt(deg[deg>0]); L=np.eye(len(sup))-inv[:,None]*A*inv[None,:]
        ev=np.linalg.eigvalsh(L); lam=float(ev[1]) if len(ev)>1 else 0.0
        # Cross-fitted utility against all support embeddings; per-class query R@1.
        qsim=z[qry]@z[np.concatenate([sup,np.flatnonzero(y!=c)])].T
        gal=np.concatenate([sup,np.flatnonzero(y!=c)]); pred=y[gal[np.argmax(qsim,axis=1)]]
        acc.append(float(np.mean(pred==c))); conn.append(lam); nn.append(float(np.mean(np.max(S,axis=1))))
    conn=np.asarray(conn); acc=np.asarray(acc); nn=np.asarray(nn)
    def corr(a,b): return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else float('nan')
    print(f"CONNECTIVITY_CLASSES={len(conn)} corr_lambda_acc={corr(conn,acc):+.6f} corr_nn_acc={corr(nn,acc):+.6f} acc_mean={acc.mean():.6f}")
    raise SystemExit(0 if corr(conn,acc)>=0.20 and abs(corr(conn,acc))>abs(corr(nn,acc))+0.05 else 2)
if __name__=='__main__': main()
