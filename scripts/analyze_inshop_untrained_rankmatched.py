#!/usr/bin/env python3
"""Rank-matched foreign cosine diagnostic for cached avg+max features."""
import argparse, json
import numpy as np
def main():
    p=argparse.ArgumentParser(); p.add_argument('--cache', required=True); p.add_argument('--output', required=True); p.add_argument('--chunk',type=int,default=256); a=p.parse_args()
    import torch
    z=np.load(a.cache); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def kth(q, qlab, pool, plab, k):
        q=torch.from_numpy(q).to(device); pool=torch.from_numpy(pool).to(device)
        q=q/(q.norm(dim=1,keepdim=True)+1e-12); pool=pool/(pool.norm(dim=1,keepdim=True)+1e-12)
        out=[]
        for s in range(0,len(q),a.chunk):
            sim=q[s:s+a.chunk]@pool.T
            same=torch.from_numpy(qlab[s:s+a.chunk,None]==plab[None,:]).to(device)
            sim[same]=-float('inf')
            out.append(torch.topk(sim,k,dim=1).values[:,-1].cpu().numpy())
        return float(np.mean(np.concatenate(out)))
    train=z['train']; tl=z['train_labels']; query=z['query']; ql=z['query_labels']; gallery=z['gallery']; gl=z['gallery_labels']
    result={'train_rank2_foreign_cosine':kth(train,tl,train,tl,2), 'query_rank1_gallery_foreign_cosine':kth(query,ql,gallery,gl,1), 'train_rows':len(train),'query_rows':len(query),'gallery_rows':len(gallery),'device':str(device),'pool_matching':'train rank2 vs query/gallery rank1'}
    open(a.output,'w').write(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
