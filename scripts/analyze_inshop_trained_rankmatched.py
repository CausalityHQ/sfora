#!/usr/bin/env python3
"""Apply the same rank-matched foreign statistic to corrected final embeddings."""
import argparse, json
from pathlib import Path
import numpy as np
def kth(q, qlab, pool, plab, k, chunk=256):
    best=[]
    q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12); pool=pool/(np.linalg.norm(pool,axis=1,keepdims=True)+1e-12)
    for s in range(0,len(q),chunk):
        sim=q[s:s+chunk]@pool.T; sim[qlab[s:s+chunk,None]==plab[None,:]]=-np.inf
        best.append(np.partition(sim,-k,axis=1)[:,-k])
    return float(np.mean(np.concatenate(best)))
def load(path):
    x=np.load(path); return x['embeddings'] if 'embeddings' in x else x['features']
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--seeds',nargs='+',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    rows=[]
    for sd in a.seeds:
        d=a.root/sd/'reports/emb' if (a.root/sd/'reports/emb').exists() else a.root/'reports/emb'
        tr,qu,ga=[load(d/f'inshop_corrected_pa_seed{sd}_{x}_final.npz') for x in ('train','query','gallery')]
        labels=np.load(d/f'inshop_corrected_pa_seed{sd}_train_final.npz')['labels']; ql=np.load(d/f'inshop_corrected_pa_seed{sd}_query_final.npz')['labels']; gl=np.load(d/f'inshop_corrected_pa_seed{sd}_gallery_final.npz')['labels']
        rows.append({'seed':sd,'train_rank2_foreign_cosine':kth(tr,labels,tr,labels,2),'query_rank1_gallery_foreign_cosine':kth(qu,ql,ga,gl,1)})
    json.dump(rows,open(a.output,'w'),indent=2); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
