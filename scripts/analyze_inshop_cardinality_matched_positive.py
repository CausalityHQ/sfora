#!/usr/bin/env python3
import argparse,json
import numpy as np
def load(path):
 z=np.load(path); return z['train'],z['train_labels'],z['query'],z['query_labels'],z['gallery'],z['gallery_labels']
def matched_mean(a,al,b,bl,k,rng,reps=20):
 vals=[]
 for _ in range(reps):
  row=[]
  for i,y in enumerate(al):
   cand=np.flatnonzero(bl==y)
   if a is b: cand=cand[cand!=i]
   if len(cand)==0: continue
   take=rng.choice(cand,size=min(k,len(cand)),replace=False)
   q=a[i]/(np.linalg.norm(a[i])+1e-12); p=b[take]; p=p/(np.linalg.norm(p,axis=1,keepdims=True)+1e-12)
   row.append(float(np.max(p@q)))
  vals.append(float(np.mean(row)))
 return float(np.mean(vals)),float(np.std(vals,ddof=1))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--directory',required=True); p.add_argument('--prefix',required=True); p.add_argument('--seeds',nargs='+',required=True); p.add_argument('--output',required=True); p.add_argument('--k',type=int,default=3); a=p.parse_args(); rows=[]
 for j,s in enumerate(a.seeds):
  tr,tl,qu,ql,ga,gl=load(f'{a.directory}/{a.prefix}{s}.npz'); rng=np.random.default_rng(20260807+int(j))
  seen,seen_sd=matched_mean(tr,tl,tr,tl,a.k,rng); unseen,unseen_sd=matched_mean(qu,ql,ga,gl,a.k,rng)
  rows.append({'seed':s,'k':a.k,'seen_positive':seen,'unseen_positive':unseen,'gap':unseen-seen,'seen_resample_sd':seen_sd,'unseen_resample_sd':unseen_sd})
 json.dump(rows,open(a.output,'w'),indent=2); print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
