#!/usr/bin/env python3
"""GPU-assisted Gate-1 diagnostic for persistent within-class topology.

For each training identity, compute connected-component counts of its cosine
graph over fixed global within-class distance thresholds.  The resulting
area/persistence statistics are then compared with leave-one-out nearest-
neighbour correctness.  This is measurement only: no threshold is fitted to
the evaluation split and no training relation is changed.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np


def _components(adj: np.ndarray) -> int:
    n = adj.shape[0]
    seen = np.zeros(n, dtype=bool)
    count = 0
    for root in range(n):
        if seen[root]:
            continue
        count += 1
        stack = [root]
        seen[root] = True
        while stack:
            i = stack.pop()
            nxt = np.flatnonzero(adj[i] & ~seen)
            if len(nxt):
                seen[nxt] = True
                stack.extend(nxt.tolist())
    return count


def measure(emb: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> dict:
    x = emb.astype(np.float32, copy=False)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    # GPU is optional; the diagnostic remains reproducible on CPU.
    try:
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xt = torch.from_numpy(x).to(dev)
        sim_all = (xt @ xt.T).cpu().numpy()
        device = str(dev)
    except Exception:
        sim_all = x @ x.T
        device = "numpy"
    loo = np.empty(len(x), dtype=bool)
    np.fill_diagonal(sim_all, -np.inf)
    # Nearest same-class versus nearest foreign; this is a train-only outcome.
    for start in range(0, len(x), 1024):
        stop = min(start + 1024, len(x))
        s = sim_all[start:stop]
        same = labels[start:stop, None] == labels[None, :]
        same[np.arange(stop - start), np.arange(start, stop)] = False
        same_best = np.where(same, s, -np.inf).max(axis=1)
        foreign_best = np.where(~same, s, -np.inf).max(axis=1)
        loo[start:stop] = same_best > foreign_best
    rows = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 3:
            continue
        sub = sim_all[np.ix_(idx, idx)]
        vals = []
        for t in thresholds:
            vals.append(_components(sub >= t) )
        vals = np.asarray(vals, dtype=np.float64)
        # Excess components above one, integrated over the registered grid.
        persistence = float(np.trapezoid(vals - 1.0, thresholds))
        rows.append({"label": int(label), "n": int(len(idx)),
                     "components": vals.tolist(), "persistence": persistence,
                     "loo_top1": float(loo[idx].mean())})
    p = np.asarray([r["persistence"] for r in rows])
    y = np.asarray([r["loo_top1"] for r in rows])
    corr = float(np.corrcoef(p, y)[0, 1]) if len(p) > 1 and p.std() > 0 and y.std() > 0 else None
    return {"device": device, "thresholds": thresholds.tolist(),
            "classes": len(rows), "rows": rows,
            "persistence_mean": float(p.mean()), "persistence_std": float(p.std()),
            "persistence_loo_correlation": corr,
            "loo_top1_mean": float(y.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    z = np.load(args.pack)
    # Fixed thresholds: no test-derived tuning.  Cosine thresholds span the
    # observed local-graph regime and are recorded in the artifact.
    thresholds = np.linspace(0.20, 0.95, 16, dtype=np.float32)
    result = measure(z["embeddings"], z["labels"], thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
