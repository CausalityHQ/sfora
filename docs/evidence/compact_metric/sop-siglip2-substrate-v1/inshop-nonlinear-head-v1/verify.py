"""Replay saved descriptors and first-seed gate without training."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from preflight_inshop_siglip2_unseen_gallery import split
from train_sop_siglip2_compact import score_packed_full_gallery
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

root, partition, cache = map(Path, sys.argv[1:])
report = json.loads((root / 'seed17.json').read_text())
manifest = json.loads((root / 'preflight.json').read_text())
assert report['manifest'] == manifest and report['claim_eligible'] is False
for path, expected in ((partition, manifest['partition_sha256']),
                       (cache, manifest['cache_sha256']),
                       (root / 'seed17.weights.npz', report['weights_sha256'])):
    with path.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == expected
labels = tuple(line.split()[1] for line in partition.read_text().splitlines()[2:]
               if line.split()[-1] == 'train')
fit, _ = split(labels)
products = sorted({labels[i] for i in fit},
                  key=lambda p: (hashlib.sha256(b'sfora-nonlinear-v1\0' + p.encode()).digest(), p))
train_names = set(products[:1002])
rows = [i for i in fit if labels[i] not in train_names]
codec = {name: i for i, name in enumerate(sorted({labels[i] for i in rows}))}
target = torch.tensor([codec[labels[i]] for i in rows])
queries = torch.nonzero(torch.bincount(target)[target] > 1).flatten()
torch.set_num_threads(8)
values = torch.from_numpy(np.load(cache, mmap_mode='r')[rows].copy())
unit = F.normalize(values, dim=1)
weights = np.load(root / 'seed17.weights.npz', allow_pickle=False)
with torch.no_grad():
    for arm in ('control', 'nonlinear'):
        state = {key[len(arm)+1:]: torch.from_numpy(weights[key])
                 for key in weights.files if key.startswith(arm + '.')}
        primary = F.linear(unit, state['primary.weight'], state['primary.bias'])
        z = F.linear(unit - state['center'], state['down.weight'])
        activation = F.gelu(z) if arm == 'nonlinear' else .5 * z
        raw = primary + F.linear(activation, state['up.weight'])
        packed = pack_int8_unit_embeddings(F.normalize(raw, dim=1))
        scored = score_packed_full_gallery(packed.codes.float(), packed.inverse_norms,
                                          target, queries, device=torch.device('cpu'))
        assert scored == report['arms'][arm]['quality'], arm
for quality in [report['base']] + [a['quality'] for a in report['arms'].values()]:
    for mean, raw in (('recall_at_1','per_query_r1'),('map_at_r','per_query_ap')):
        assert len(quality[raw]) == manifest['eligible_queries']
        assert abs(sum(quality[raw])/len(quality[raw]) - quality[mean]) < 1e-12
n, c, b = report['arms']['nonlinear']['quality'], report['arms']['control']['quality'], report['base']
gate = (n['recall_at_1'] > c['recall_at_1'] and n['recall_at_1'] > b['recall_at_1']
        and n['map_at_r'] - c['map_at_r'] >= .002 and n['map_at_r'] >= b['map_at_r'])
assert gate is False and report['advance_seed17'] is False
assert min(report['curvature'].values()) >= .1
assert all(len(a['losses']) == 1000 and a['rank_updates'] == 909 for a in report['arms'].values())
print('PASS saved-weight packed per-query replay, hashes, means, first gate and curvature qualification')
