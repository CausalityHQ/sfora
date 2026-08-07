#!/usr/bin/env python3
"""CPU-only PEBH exchangeability diagnostic on frozen In-Shop embeddings."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

def fold(label: int) -> int:
    return int.from_bytes(hashlib.sha256(str(int(label)).encode()).digest()[:4], "little") % 5

class Head(torch.nn.Module):
    def __init__(self, d: int, w: int, self_only: bool):
        super().__init__(); self.self_only = self_only
        self.a = torch.nn.Linear(d, w, bias=False)
        self.b = torch.nn.Linear(d, w, bias=False)
        self.c = torch.nn.Linear(w, d, bias=False)
    def forward(self, x, y=None):
        if y is None: y = x
        return F.normalize(self.c(self.a(x) * self.b(y)), dim=-1)

def sample_pairs(x, y, n, rng):
    by = {}
    for i, lab in enumerate(y.tolist()): by.setdefault(int(lab), []).append(i)
    labs = [k for k,v in by.items() if len(v) >= 2]
    ia=[]; ib=[]; ic=[]
    for _ in range(n):
        lab = labs[int(rng.integers(len(labs)))]
        p = rng.choice(by[lab], 2, replace=False); ia.append(p[0]); ib.append(p[1]); ic.append(lab)
    return np.asarray(ia), np.asarray(ib), np.asarray(ic)

def evaluate(head, x, y, rng, n=3000):
    head.eval();
    with torch.no_grad(): z = head(torch.from_numpy(x)).numpy()
    ids = rng.choice(len(x), min(n, len(x)), replace=False)
    sims = z[ids] @ z.T
    same = (y[None,:] == y[ids,None]); same[np.arange(len(ids)), ids] = False
    foreign = ~same; foreign[np.arange(len(ids)), ids] = False
    pos = np.where(same, sims, -2).max(1); neg = np.where(foreign, sims, -2).max(1)
    nn = sims.copy(); nn[np.arange(len(ids)), ids] = -2
    pred = y[nn.argmax(1)]
    return {"positive": float(pos.mean()), "foreign": float(neg.mean()), "loo_r1": float((pred==y[ids]).mean())}

def run(path: Path, seed: int, width: int, epochs: int):
    q=np.load(path); x=q["embeddings"].astype("float32"); y=q["labels"].astype("int64")
    train=np.array([fold(v) < 4 for v in y]); rng=np.random.default_rng(seed)
    xt,yt=x[train],y[train]; xv,yv=x[~train],y[~train]
    out={"seed":seed,"train":len(xt),"test":len(xv)}
    for name,self_only in (("self",True),("pebh",False)):
        torch.manual_seed(seed+17*(name=="pebh")); h=Head(x.shape[1],width,self_only); opt=torch.optim.AdamW(h.parameters(),lr=2e-3,weight_decay=1e-4)
        tx=torch.from_numpy(xt)
        for _ in range(epochs):
            ia,ib,_=sample_pairs(xt,yt,2048,rng); xa=tx[ia]; xb=tx[ib]
            za=h(xa); zij=h(xa, xb) if not self_only else za
            # Same-class exchange must be close; random foreign target is the negative.
            ni=rng.integers(0,len(xt),size=len(ia)); zn=h(tx[ni])
            loss=(F.softplus(0.2-(za*zij).sum(1)+(za*zn).sum(1))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        out[name]=evaluate(h,xv,yv,rng)
    out["delta_positive"]=out["pebh"]["positive"]-out["self"]["positive"]
    out["delta_foreign"]=out["pebh"]["foreign"]-out["self"]["foreign"]
    out["delta_loo_r1"]=out["pebh"]["loo_r1"]-out["self"]["loo_r1"]
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--packs",nargs="+",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--width",type=int,default=128); ap.add_argument("--epochs",type=int,default=20); a=ap.parse_args()
    rows=[run(p,s,a.width,a.epochs) for s,p in enumerate(a.packs)]
    a.out.write_text(json.dumps({"rows":rows},indent=2)+"\n")
    print(json.dumps({"rows":rows},indent=2))
if __name__ == "__main__": main()
