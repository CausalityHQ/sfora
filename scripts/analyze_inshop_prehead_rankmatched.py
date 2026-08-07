#!/usr/bin/env python3
"""Rank-matched train-vs-unseen excess for trained pre-head exports."""
import argparse,json
import numpy as np
def kth(q,ql,p,pl,k,chunk=256):
 q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12); p=p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-12); out=[]
 for s in range(0,len(q),chunk):
  z=q[s:s+chunk]@p.T; z[ql[s:s+chunk,None]==pl[None,:]]=-np.inf; out.append(np.partition(z,-k,axis=1)[:,-k])
 return float(np.mean(np.concatenate(out)))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--directory',required=True); p.add_argument('--output',required=True); p.add_argument('--seeds',nargs='+',required=True); a=p.parse_args(); rows=[]
 for s in a.seeds:
  z=np.load(f'{a.directory}/inshop_trained_prehead_seed{s}.npz'); tr,tl,qu,ql,ga,gl=[z[k] for k in ('train','train_labels','query','query_labels','gallery','gallery_labels')]
  rows.append({'seed':s,'train_rank2':kth(tr,tl,tr,tl,2),'query_gallery_rank1':kth(qu,ql,ga,gl,1)})
 json.dump(rows,open(a.output,'w'),indent=2); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
