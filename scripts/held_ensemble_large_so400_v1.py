#!/usr/bin/env python3
"""TRAIN-held screen: equal-weight Large+So400m embedding ensemble vs the better single arm.

Pre-registered (committed before any ensemble number was computed):
  inputs  : v9h2 held exports (TRAIN100 pilots, seeds 179032/179041), split preflight 41b5fe09.
  ensemble: e = [l2norm(large), l2norm(so400)] / sqrt(2)  (score = mean of the two cosines).
  sanity  : each single arm's per-query R@1 and AP must equal the recorded h2 lists exactly;
            otherwise INVALID. Amendment 1 (after attempt 1 tripped this gate on FP32 scoring,
            before any ensemble number existed): score with the unchanged h2 reference functions
            packed_quality / bootstrap_lower and packing module 4ca0de1b; rule otherwise unchanged.
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


HELD = RUNS / 'sfora-native256-held-v9h2'
PINS = {'reference_compare_inshop_sop_warmstart_100.py': ('8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250', 'packed_quality'),
        'reference_score_inshop_crop_view_pair.py': ('16e27ccaa7325b9ef7efdb3512cb95791a682afdd5ae59bd1f0847d63b87f5ed', 'bootstrap_lower')}
PACKING = (RUNS / 'sfora-native256-initialized-cpu-source-v3/joint_relational_compaction.py',
           '4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67')


def reference():
    import ast, hashlib, importlib.util, torch
    def checked(path, pin):
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == pin, f'{path} differs'
        return data
    checked(*PACKING)
    spec = importlib.util.spec_from_file_location('_packing', PACKING[0])
    packing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packing)
    ns = {'np': np, 'torch': torch, 'pack_int8_unit_embeddings': packing.pack_int8_unit_embeddings}
    for name, (pin, fn) in PINS.items():
        src = checked(HELD / name, pin).decode()
        node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == fn]
        assert len(node) == 1
        exec(compile('from __future__ import annotations\n' + ast.get_source_segment(src, node[0]), name, 'exec'), ns)
    return ns['packed_quality'], ns['bootstrap_lower'], torch.device('cpu')


def main():
    split, h2 = json.loads(SPLIT.read_text()), json.loads(H2.read_text())
    labels = np.array([r['product'] for r in split['held_manifest']])
    query, gallery = np.array(split['query']), np.array(split['gallery'])
    out, deltas = {'schema': 'held-ensemble-large-so400-v1', 'official_read': False}, []
    packed_quality, bootstrap_lower, cpu = reference()
    labels_t = tuple(labels.tolist())
    for seed in SEEDS:
        emb = {arm: np.load(RUNS / f'sfora-native256-held-export-{arm}-{seed}-v9h2/held.npy') for arm in ('large', 'so400')}
        emb['ensemble'] = np.concatenate([emb['large'], emb['so400']], 1) / np.sqrt(2)
        res = {arm: packed_quality(np.ascontiguousarray(e, np.float32), labels_t, query.tolist(), gallery.tolist(), device=cpu)
               for arm, e in emb.items()}
        res = {arm: (np.array(r['per_query_r1'], bool), np.array(r['per_query_ap'])) for arm, r in res.items()}
        for arm, role in (('large', 'control'), ('so400', 'candidate')):
            ref = h2['quality'][seed][role]
            ok = (np.array_equal(res[arm][0], np.array(ref['per_query_r1'], bool)) and
                  np.array_equal(res[arm][1], np.array(ref['per_query_ap'])))
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
    lo, hi = bootstrap_lower(avg, labels[query]), -bootstrap_lower(-avg, labels[query])
    out['dR1_vs_so400'] = {'mean': float(avg.mean()), 'product_lower95': lo, 'product_upper95': hi}
    out['decision'] = 'PASS' if lo > 0 and all(out[s]['dR1_vs_so400'] > 0 for s in SEEDS) else 'FAIL'
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
