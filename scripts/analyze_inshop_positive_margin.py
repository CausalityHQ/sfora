#!/usr/bin/env python3
"""Compute same-class positive, foreign, and P-F margin statistics."""
import argparse,json
import numpy as np
def stats(q,ql,p,pl,foreign_k,chunk=256):
 q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12); p=p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-12); pos=[]; forg=[]
 for s in range(0,len(q),chunk):
  z=q[s:s+chunk]@p.T; same=ql[s:s+chunk,None]==pl[None,:]; zf=z.copy(); zf[same]=-np.inf
  zp=z.copy(); zp[~same]=-np.inf
  # self is not a positive candidate for train->train.
  if len(q)==len(p) and np.array_equal(ql,pl):
   rows=np.arange(s,min(s+chunk,len(q))); zp[np.arange(len(rows)),rows]=-np.inf
  pos.append(zp.max(axis=1)); forg.append(np.partition(zf,-foreign_k,axis=1)[:,-foreign_k])
 pos=np.concatenate(pos); forg=np.concatenate(forg); return {'positive':float(pos.mean()),'foreign':float(forg.mean()),'margin':float((pos-forg).mean()),'margin_fail_rate':float(np.mean(pos<=forg))}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--directory',required=True); p.add_argument('--prefix',required=True); p.add_argument('--seeds',nargs='+',required=True); p.add_argument('--output',required=True); a=p.parse_args(); rows=[]
 for s in a.seeds:
  z=np.load(f'{a.directory}/{a.prefix}{s}.npz'); tr,tl,qu,ql,ga,gl=[z[k] for k in ('train','train_labels','query','query_labels','gallery','gallery_labels')]
  seen=stats(tr,tl,tr,tl,2); unseen=stats(qu,ql,ga,gl,1); rows.append({'seed':s,'seen':seen,'unseen':unseen,'delta_positive':unseen['positive']-seen['positive'],'delta_foreign':unseen['foreign']-seen['foreign'],'delta_margin':unseen['margin']-seen['margin']})
 json.dump(rows,open(a.output,'w'),indent=2); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
