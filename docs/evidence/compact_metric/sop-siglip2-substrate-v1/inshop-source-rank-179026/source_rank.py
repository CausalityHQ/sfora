import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from preflight_inshop_siglip2_unseen_gallery import split
from sfora.unicom_inshop import parse_inshop_partition

root = Path('/home/riomus/datasets/In-shop Clothes Retrieval Benchmark')
cache = Path('/home/riomus/runs/sfora-inshop-siglip2-train-features-v1/train_features.npy')
prior = Path('/home/riomus/runs/sfora-inshop-expected-gallery-179026-v1/receipt.json')
out = Path('/home/riomus/runs/sfora-inshop-source-rank-179026-v1/receipt.json')
assert not out.exists()
assert hashlib.sha256(cache.read_bytes()).hexdigest() == 'f232584491bf4ed1cf75daa7fcc03f218e7db5df797f4d57dd2bf034ac110885'
assert hashlib.sha256(prior.read_bytes()).hexdigest() == 'd48e2c382fcfb7aa4807b00cc8b2a76fe098aebfa77334dbc2fd8402d36b46a4'
train = tuple(row for row in parse_inshop_partition(root) if row.split == 'train')
_, held = split(tuple(row.label for row in train))
labels = [train[i].label for i in held]
paths = [train[i].image_path for i in held]
grouped = defaultdict(list)
for i, label in enumerate(labels):
    grouped[label].append(i)
query, gallery = [], []
for label in sorted(grouped):
    rows = sorted(grouped[label], key=lambda i: hashlib.sha256(str(paths[i].relative_to(root)).encode()).digest())
    count = max(1, min(len(rows) - 1, round(len(rows) / 2)))
    gallery.extend(rows[:count])
    query.extend(rows[count:])
query.sort(); gallery.sort()
assert len(query) == 6354 and len(gallery) == 6245
assert hashlib.sha256(np.asarray(query, dtype='<i4').tobytes()).hexdigest() == '89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68'
assert hashlib.sha256(np.asarray(gallery, dtype='<i4').tobytes()).hexdigest() == 'e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3'
old = json.loads(prior.read_text())
assert len(old['actual_proxy_per_query_r1']) == len(query)
source = np.load(cache, mmap_mode='r', allow_pickle=False)
torch.backends.cuda.matmul.allow_tf32 = False
values = F.normalize(torch.from_numpy(np.asarray(source[list(held)]).copy()).float(), dim=1).cuda()
gallery_values = values[gallery]
ids = {name: i for i, name in enumerate(sorted(grouped))}
label_ids = torch.tensor([ids[label] for label in labels], device='cuda')
gallery_labels = label_ids[gallery]
hits, margins = [], []
for rows in torch.tensor(query, device='cuda').split(128):
    score = values[rows] @ gallery_values.T
    best_index = score.argmax(dim=1)
    hits.extend((label_ids[rows] == gallery_labels[best_index]).cpu().tolist())
    positive = label_ids[rows, None] == gallery_labels[None, :]
    best_p = score.masked_fill(~positive, -torch.inf).max(dim=1).values
    best_n = score.masked_fill(positive, -torch.inf).max(dim=1).values
    margins.extend((best_p - best_n).cpu().tolist())
packed = old['actual_proxy_per_query_r1']
assert len(hits) == len(packed)
report = {
    'schema': 'sfora-inshop-pretrained-source-asymmetric-train-v1',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
    'prior_receipt_sha256': hashlib.sha256(prior.read_bytes()).hexdigest(),
    'query_rows': len(query), 'gallery_rows': len(gallery),
    'pretrained_source_r1': sum(hits)/len(hits),
    'trained_packed_r1': sum(packed)/len(packed),
    'source_recovers_packed_misses': sum(a and not b for a,b in zip(hits,packed,strict=True)),
    'source_loses_packed_hits': sum(b and not a for a,b in zip(hits,packed,strict=True)),
    'pretrained_source_per_query_r1': hits,
    'pretrained_source_per_query_margin': margins,
}
out.write_text(json.dumps(report, sort_keys=True)+'\n')
print(json.dumps({key: report[key] for key in ('pretrained_source_r1','trained_packed_r1','source_recovers_packed_misses','source_loses_packed_hits')},sort_keys=True))
