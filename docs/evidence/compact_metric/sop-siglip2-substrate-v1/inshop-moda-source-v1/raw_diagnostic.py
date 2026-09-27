import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from preflight_inshop_siglip2_unseen_gallery import split
from probe_sop_relational_linear import score_symmetric
from sfora.unicom_inshop import parse_inshop_partition

run = Path('/home/riomus/runs/sfora-inshop-moda-source-v1/result')
receipt = json.loads((run / 'receipt.json').read_text())
feature = run / 'train_features.npy'
assert hashlib.sha256(feature.read_bytes()).hexdigest() == receipt['features_sha256']
rows = tuple(row for row in parse_inshop_partition(Path('/home/riomus/datasets/inshop_official_standard')) if row.split == 'train')
labels = tuple(row.label for row in rows)
fit, held = split(labels)
assert receipt['fit_rows_sha256'] and len(fit) == 13283 and len(held) == 12599
values = F.normalize(torch.from_numpy(np.asarray(np.load(feature, mmap_mode='r')[list(held)]).copy()).float(), dim=1)
names = tuple(labels[i] for i in held)
mapping = {name: i+1 for i, name in enumerate(sorted(set(names)))}
ids = tuple(mapping[name] for name in names)
score = score_symmetric(values, ids, candidate_width=max(Counter(ids).values())-1, device=torch.device('cuda'))
print(json.dumps({'raw_float768_r1': score['r1'], 'raw_float768_map_at_r': score['map_at_r'], 'packed128_r1': receipt['quality']['recall_at_1'], 'packed128_map_at_r': receipt['quality']['map_at_r']}))
