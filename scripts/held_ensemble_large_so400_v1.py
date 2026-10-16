#!/usr/bin/env python3
"""TRAIN-held screen: equal-weight Large+So400m embedding ensemble vs the better single arm.

Pre-registered (committed before any ensemble number was computed):
  inputs  : v9h2 held exports (TRAIN100 pilots, seeds 179032/179041), split preflight 41b5fe09.
  ensemble: e = [l2norm(large), l2norm(so400)] / sqrt(2)  (score = mean of the two cosines).
  sanity  : FP32 per-query R@1 hits of each single arm must equal the recorded h2 per_query_r1
            exactly and mAP@R within 2e-3 (h2 scored int8-packed codes); otherwise INVALID.
  PASS iff  seed-averaged per-query dR@1 (ensemble - So400m) has paired product-bootstrap
            lower95 > 0 (5000 draws, seed 179019) AND dR@1 > 0 at each seed.
  PASS => ONE official read of the full-scale Large+So400m ensemble from the saved v1 official
          features; FAIL => ensemble lane closed. mAP@R is reported, never gating.
"""
import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path('/home/riomus/runs')
SPLIT = RUNS / 'sfora-pe-augmented-100-v2/preflight.json'
H2 = RUNS / 'sfora-native256-held-score-v9h2.json'
SEEDS = ('179032', '179041')


def quality(emb, labels, query, gallery):
    q, g = emb[query], emb[gallery]
    ql, gl = labels[query], labels[gallery]
    r1, ap = np.empty(len(query), bool), np.empty(len(query))
    for start in range(0, len(query), 512):
        sims = q[start:start + 512] @ g.T
        order = np.argsort(-sims, axis=1, kind='stable')
        for i, row in enumerate(order):
            rel = gl[row] == ql[start + i]
            r = int(rel.sum())
            r1[start + i] = rel[0]
            hits = rel[:r]
            ap[start + i] = (np.cumsum(hits) / np.arange(1, r + 1))[hits].sum() / r
    return r1, ap


def bootstrap(delta, groups, draws=5000, seed=179019):
    uniq, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, delta), np.bincount(inv)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(draws, len(uniq)))
    stats = sums[pick].sum(1) / counts[pick].sum(1)
    return float(np.quantile(stats, .025)), float(np.quantile(stats, .975))


def main():
    split, h2 = json.loads(SPLIT.read_text()), json.loads(H2.read_text())
    labels = np.array([r['product'] for r in split['held_manifest']])
    query, gallery = np.array(split['query']), np.array(split['gallery'])
    out, deltas = {'schema': 'held-ensemble-large-so400-v1', 'official_read': False}, []
    for seed in SEEDS:
        emb = {arm: np.load(RUNS / f'sfora-native256-held-export-{arm}-{seed}-v9h2/held.npy') for arm in ('large', 'so400')}
        emb['ensemble'] = np.concatenate([emb['large'], emb['so400']], 1) / np.sqrt(2)
        res = {arm: quality(e.astype(np.float32), labels, query, gallery) for arm, e in emb.items()}
        for arm, role in (('large', 'control'), ('so400', 'candidate')):
            ref = h2['quality'][seed][role]
            ok = (np.array_equal(res[arm][0], np.array(ref['per_query_r1'], bool)) and
                  abs(res[arm][1].mean() - ref['map_at_r']) < 2e-3)
            if not ok:
                print(json.dumps({'decision': 'INVALID', 'seed': seed, 'arm': arm,
                                  'r1': float(res[arm][0].mean()), 'h2_r1': ref['recall_at_1'],
                                  'map': float(res[arm][1].mean()), 'h2_map': ref['map_at_r']}))
                sys.exit(1)
        out[seed] = {arm: {'recall_at_1': float(r.mean()), 'map_at_r': float(a.mean())} for arm, (r, a) in res.items()}
        d = res['ensemble'][0].astype(float) - res['so400'][0]
        out[seed]['dR1_vs_so400'] = float(d.mean())
        deltas.append(d)
    avg = np.mean(deltas, 0)
    lo, hi = bootstrap(avg, labels[query])
    out['dR1_vs_so400'] = {'mean': float(avg.mean()), 'product_lower95': lo, 'product_upper95': hi}
    out['decision'] = 'PASS' if lo > 0 and all(out[s]['dR1_vs_so400'] > 0 for s in SEEDS) else 'FAIL'
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
